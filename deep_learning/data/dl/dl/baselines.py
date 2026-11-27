"""Simple, model-aware output-token prediction baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deep_learning.dl.metrics import evaluate_predictions
from deep_learning.dl.features import load_embedding_cache


def fit_mean_baseline(train: pd.DataFrame) -> dict[str, float]:
    """Fit independent output-length means for each historical model."""
    if train.empty:
        raise ValueError("training rows are empty")
    return train.groupby("model_id")["output_tokens"].mean().astype(float).to_dict()


def predict_mean_baseline(model: dict[str, float], rows: pd.DataFrame) -> np.ndarray:
    """Predict in original token units; fail if a model has no training data."""
    predictions = rows["model_id"].map(model)
    if predictions.isna().any():
        missing = sorted(set(rows.loc[predictions.isna(), "model_id"]))
        raise ValueError(f"unknown model in baseline: {missing}")
    return predictions.to_numpy(dtype=np.float64)


def build_xgb_features(
    text_vectors: np.ndarray,
    model_indices: np.ndarray,
    text_lengths: np.ndarray,
    num_models: int,
) -> np.ndarray:
    """Give XGBoost the same frozen text vectors and model identity as DL."""
    if not (len(text_vectors) == len(model_indices) == len(text_lengths)):
        raise ValueError("feature arrays must have the same row count")
    indices = np.asarray(model_indices, dtype=np.int64)
    if (indices < 0).any() or (indices >= num_models).any():
        raise ValueError("model index is outside the catalog")
    one_hot = np.eye(num_models, dtype=np.float32)[indices]
    lengths = np.log1p(np.asarray(text_lengths, dtype=np.float32)).reshape(-1, 1)
    return np.concatenate((np.asarray(text_vectors, dtype=np.float32), lengths, one_hot), axis=1)


def run_baselines(data_dir: str | Path, embedding_dir: str | Path, output_dir: str | Path) -> dict:
    """Train mean and XGBoost baselines on the saved split; do not call LLMs."""
    import xgboost as xgb

    data_dir, embedding_dir, output_dir = Path(data_dir), Path(embedding_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(data_dir / "records.parquet")
    catalog = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
    model_ids = [item["model_id"] for item in catalog]
    mapping = {name: i for i, name in enumerate(model_ids)}
    cache_metadata = json.loads((embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    embeddings, row_index = load_embedding_cache(
        rows["prompt"].tolist(), embedding_dir, cache_metadata["encoder_id"],
        encoder_revision=cache_metadata["encoder_revision"])
    model_indices = rows["model_id"].map(mapping)
    if model_indices.isna().any():
        raise ValueError("processed records contain models outside catalog")
    features = build_xgb_features(
        embeddings[row_index], model_indices.to_numpy(dtype=np.int64),
        rows["prompt"].str.len().to_numpy(dtype=np.float32), len(model_ids),
    )
    train_mask = rows["split"].eq("train").to_numpy()
    test_mask = rows["split"].eq("test").to_numpy()
    train, test = rows.loc[train_mask], rows.loc[test_mask]
    mean_model = fit_mean_baseline(train)
    mean_predictions = predict_mean_baseline(mean_model, test)
    xgb_model = xgb.XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        n_jobs=8, random_state=42,
    )
    xgb_model.fit(features[train_mask], np.log1p(train["output_tokens"].to_numpy(dtype=np.float32)))
    predicted_log = xgb_model.predict(features[test_mask])
    xgb_predictions = np.maximum(1, np.rint(np.expm1(np.clip(predicted_log, 0, np.log1p(1_000_000)))))
    result = {
        "mean": evaluate_predictions(test, mean_predictions, catalog=catalog),
        "xgboost": evaluate_predictions(test, xgb_predictions, catalog=catalog),
    }
    (output_dir / "baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "mean_by_model.json").write_text(json.dumps(mean_model, indent=2), encoding="utf-8")
    xgb_model.save_model(str(output_dir / "xgboost.json"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_baselines(args.data_dir, args.embedding_dir, args.output_dir)
    print(json.dumps({name: report["overall"] for name, report in result.items()}, indent=2))
