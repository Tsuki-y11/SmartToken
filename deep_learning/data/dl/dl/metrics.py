"""Token-error reports in the original (not log) unit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

from deep_learning.dl.data_pipeline import _question_key


def _summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - actual
    return {
        "count": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def evaluate_predictions(rows: pd.DataFrame, predictions: list[float] | np.ndarray, *, catalog: list[dict] | None = None) -> dict:
    """Report overall and model/source errors using held-out observations."""
    actual = rows["output_tokens"].to_numpy(dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("actual and predicted arrays must have the same nonzero length")
    if not np.isfinite(predicted).all():
        raise ValueError("predictions must be finite")
    report = {"overall": _summary(actual, predicted), "by_source": {}, "by_model": {},
              "by_source_model": {}, "by_pool": {}, "matched_pairs": {}}
    for column, key in (("source", "by_source"), ("model_id", "by_model")):
        for value, indices in rows.groupby(column, sort=True).indices.items():
            report[key][str(value)] = _summary(actual[indices], predicted[indices])
    for (source, model), indices in rows.groupby(["source", "model_id"], sort=True).indices.items():
        report["by_source_model"][f"{source}/{model}"] = _summary(actual[indices], predicted[indices])
    if catalog is not None:
        model_pools = {item["model_id"]: set(item["pools"]) for item in catalog}
        pools = []
        for source, model in zip(rows["source"], rows["model_id"]):
            eligible = model_pools.get(model, set())
            pool = "xroute" if source == "xroutebench" and "xroute" in eligible else next(
                (name for name in ("llmrouter_small", "llmrouter_flagship") if name in eligible), None)
            if pool is None or (source == "xroutebench" and pool != "xroute"):
                raise ValueError(f"source/model is absent from catalog: {source}/{model}")
            pools.append(pool)
        for pool, indices in pd.Series(pools).groupby(pd.Series(pools), sort=True).indices.items():
            report["by_pool"][pool] = _summary(actual[indices], predicted[indices])
    required = {"source", "dataset", "original_query", "model_id"}
    if required.issubset(rows.columns):
        errors = rows.loc[:, list(required)].copy().reset_index(drop=True)
        errors["question_key"] = rows["original_query"].map(_question_key).to_numpy()
        errors["absolute_error"] = np.abs(predicted - actual)
        grouped = errors.groupby(["source", "dataset", "question_key", "model_id"], sort=False)["absolute_error"].mean().reset_index()
        for (source, dataset), subset in grouped.groupby(["source", "dataset"], sort=True):
            matrix = subset.pivot(index="question_key", columns="model_id", values="absolute_error")
            for a, b in combinations(sorted(matrix.columns), 2):
                shared = matrix[[a, b]].dropna()
                if len(shared):
                    report["matched_pairs"][f"{source}/{dataset}/{a}/{b}"] = {
                        "shared_questions": int(len(shared)),
                        "mae_a": float(shared[a].mean()), "mae_b": float(shared[b].mean()),
                    }
    return report
