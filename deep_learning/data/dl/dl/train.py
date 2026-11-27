"""Train the offline model-specific output-token predictor when requested."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_learning.dl.features import DEFAULT_ENCODER, DEFAULT_ENCODER_REVISION, FEATURE_STRATEGY, TEXT_LIMIT, build_embedding_cache
from deep_learning.dl.metrics import evaluate_predictions
from deep_learning.dl.model import MAX_OUTPUT_TOKENS, ModelAwareRegressor
from deep_learning.dl.data_pipeline import _question_key


def prepare_training_arrays(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    row_index: np.ndarray,
    model_ids: list[str],
) -> dict[str, np.ndarray]:
    """Align cached text vectors, model heads, targets, and saved split labels."""
    if len(rows) != len(row_index):
        raise ValueError("row index length does not match records")
    mapping = {name: index for index, name in enumerate(model_ids)}
    unknown = sorted(set(rows["model_id"]) - set(mapping))
    if unknown:
        raise ValueError(f"records contain models outside catalog: {unknown}")
    model_indices = rows["model_id"].map(mapping).to_numpy(dtype=np.int64)
    output_tokens = rows["output_tokens"].to_numpy(dtype=np.float32)
    if (output_tokens <= 0).any():
        raise ValueError("output token labels must be positive")
    return {
        "text_vectors": np.asarray(embeddings[row_index], dtype=np.float32),
        "model_indices": model_indices,
        "text_lengths": rows["prompt"].str.len().to_numpy(dtype=np.float32),
        "targets_log": np.log1p(output_tokens),
        "train_mask": rows["split"].eq("train").to_numpy(),
        "valid_mask": rows["split"].eq("valid").to_numpy(),
        "test_mask": rows["split"].eq("test").to_numpy(),
    }


def _loader(arrays: dict[str, np.ndarray], mask: np.ndarray, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(np.array(arrays["text_vectors"][mask], copy=True)),
        torch.from_numpy(np.array(arrays["model_indices"][mask], copy=True)),
        torch.from_numpy(np.array(arrays["text_lengths"][mask], copy=True)),
        torch.from_numpy(np.array(arrays["targets_log"][mask], copy=True)),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def _predict_original(model: ModelAwareRegressor, loader: DataLoader, device: str, max_log: float) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for vectors, model_indices, lengths, _targets in loader:
            log_output = model(vectors.to(device), model_indices.to(device), lengths.to(device))
            original = torch.expm1(torch.clamp(log_output, min=0, max=max_log)).cpu().numpy()
            original = np.maximum(1, np.rint(original))
            chunks.append(original)
    return np.concatenate(chunks).astype(np.float64)


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def validate_question_splits(rows: pd.DataFrame) -> None:
    """Use the same normalized question identity as the split assignment."""
    keys = rows["original_query"].map(_question_key)
    if (rows.groupby(keys)["split"].nunique() > 1).any():
        raise ValueError("one original question appears in multiple splits")


def train_predictor(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    encoder_id: str = DEFAULT_ENCODER,
    encoder_revision: str = DEFAULT_ENCODER_REVISION,
    seed: int = 42,
    epochs: int = 12,
    batch_size: int = 1024,
    patience: int = 3,
    device: str = "auto",
) -> dict:
    """Train once on saved rows; choose checkpoint on validation and then test."""
    from sentence_transformers import SentenceTransformer

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(data_dir / "records.parquet")
    catalog = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
    model_ids = [item["model_id"] for item in catalog]
    if not model_ids or not {"train", "valid", "test"}.issubset(set(rows["split"])):
        raise ValueError("catalog and train/valid/test rows are required")
    validate_question_splits(rows)
    train_ids = set(rows.loc[rows["split"] == "train", "model_id"])
    if train_ids != set(model_ids):
        raise ValueError(f"models missing training rows: {sorted(set(model_ids) - train_ids)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    encoder = SentenceTransformer(encoder_id, device=device, revision=encoder_revision)
    embeddings, row_index = build_embedding_cache(
        rows["prompt"].tolist(), encoder, output_dir / "embedding_cache",
        batch_size=128, encoder_id=encoder_id, encoder_revision=encoder_revision,
    )
    arrays = prepare_training_arrays(rows, embeddings, row_index, model_ids)
    train_mask, valid_mask, test_mask = (arrays[f"{part}_mask"] for part in ("train", "valid", "test"))
    pin_memory = device == "cuda"
    train_loader = _loader(arrays, train_mask, batch_size, True, pin_memory)
    valid_loader = _loader(arrays, valid_mask, batch_size, False, pin_memory)
    test_loader = _loader(arrays, test_mask, batch_size, False, pin_memory)
    model = ModelAwareRegressor(embeddings.shape[1], len(model_ids)).to(device)
    with torch.no_grad():
        for index in range(len(model_ids)):
            labels = arrays["targets_log"][train_mask & (arrays["model_indices"] == index)]
            model.model_head.weight[index, -1] = float(labels.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=0.5)
    max_log = math.log1p(MAX_OUTPUT_TOKENS)
    best_mae, best_epoch, best_state, stale_epochs = math.inf, 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for vectors, indices, lengths, targets in train_loader:
            vectors, indices, lengths, targets = (tensor.to(device) for tensor in (vectors, indices, lengths, targets))
            optimizer.zero_grad(set_to_none=True)
            predicted_log = model(vectors, indices, lengths)
            loss = loss_function(predicted_log, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        validation = _predict_original(model, valid_loader, device, max_log)
        validation_mae = float(np.mean(np.abs(validation - rows.loc[valid_mask, "output_tokens"].to_numpy())))
        history.append({"epoch": epoch, "train_log_huber": total_loss / batches, "validation_mae": validation_mae})
        print(json.dumps(history[-1]), flush=True)
        if validation_mae < best_mae:
            best_mae, best_epoch = validation_mae, epoch
            best_state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "model.pt")
    configuration = {
        "encoder_id": encoder_id, "encoder_revision": encoder_revision,
        "embedding_dim": int(embeddings.shape[1]),
        "model_ids": model_ids, "hidden_dim": model.hidden_dim,
        "text_limit": TEXT_LIMIT, "feature_strategy": FEATURE_STRATEGY,
        "seed": seed, "batch_size": batch_size, "max_epochs": epochs,
        "best_epoch": best_epoch, "best_validation_mae": best_mae,
        "torch_version": _version("torch"), "sentence_transformers_version": _version("sentence-transformers"),
    }
    (output_dir / "model_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    test_rows = rows.loc[test_mask, ["source", "dataset", "original_query", "model_id", "output_tokens"]].reset_index(drop=True)
    test_predictions = _predict_original(model, test_loader, device, max_log)
    report = evaluate_predictions(test_rows, test_predictions, catalog=catalog)
    report["training_history"] = history
    (output_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    test_rows.assign(predicted_output_tokens=test_predictions).to_parquet(output_dir / "test_predictions.parquet", index=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-id", default=DEFAULT_ENCODER)
    parser.add_argument("--encoder-revision", default=DEFAULT_ENCODER_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = train_predictor(
        args.data_dir, args.output_dir, encoder_id=args.encoder_id, encoder_revision=args.encoder_revision, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, device=args.device,
    )
    print(json.dumps(result["overall"], indent=2))
