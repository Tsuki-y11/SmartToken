"""Validation-only coverage diagnostics; never fit, tune, promote, or execute."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib

from .coverage import (_check_external_partitions, _reviewed_rows,
                       load_research_rows)
from .data import LABELS
from .proposal import validate_scores


def evaluate_label_validation(model, rows):
    """Report label decisions/abstention using the unchanged 0.08 margin."""
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("coverage evaluation accepts validation rows only")
    thresholds = validate_scores(model.thresholds)
    margin = .08
    predictions = []
    for row in rows:
        scores = validate_scores(model.score_labels(row["text"], row["context"]))
        dimensions = [d for d in LABELS if scores[d] >= thresholds[d]]
        distances = {d: abs(scores[d] - thresholds[d]) for d in LABELS}
        uncertain = [d for d in LABELS if distances[d] <= margin]
        predictions.append({"sample_id": row["sample_id"], "group_id": row["group_id"], "source": row["source"],
            "classification_input_sha256": hashlib.sha256((row["context"] + "\nUSER:\n" + row["text"]).encode("utf-8")).hexdigest(),
            "context_supplied": bool(row["context"]), "reference_dimensions": list(row["dimensions"]),
            "scores": scores, "thresholds": dict(thresholds), "distances": distances,
            "predicted_dimensions": dimensions, "uncertain_dimensions": uncertain})
    per_dimension = {}
    for label in LABELS:
        tp = sum(label in r["reference_dimensions"] and label in r["predicted_dimensions"] for r in predictions)
        fp = sum(label not in r["reference_dimensions"] and label in r["predicted_dimensions"] for r in predictions)
        fn = sum(label in r["reference_dimensions"] and label not in r["predicted_dimensions"] for r in predictions)
        tn = len(rows) - tp - fp - fn
        below = [r["scores"][label] for r in predictions if r["scores"][label] < thresholds[label]]
        above = [r["scores"][label] for r in predictions if r["scores"][label] >= thresholds[label]]
        lower, upper = max(below) if below else None, min(above) if above else None
        per_dimension[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "positive_support": tp + fn, "negative_support": tn + fp,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "threshold": thresholds[label],
            "uncertain_count": sum(label in r["uncertain_dimensions"] for r in predictions),
            "largest_score_below_threshold": lower,
            "smallest_score_at_or_above_threshold": upper,
            "decision_equivalent_gap_midpoint": (lower + upper) / 2 if lower is not None and upper is not None else None}
    uncertain_count = sum(bool(r["uncertain_dimensions"]) for r in predictions)
    return {"sample_count": len(rows), "uncertainty_margin": margin,
            "uncertain_request_count": uncertain_count, "label_only_clear_rate": 1 - uncertain_count / len(rows),
            "exact_match": sum(set(r["reference_dimensions"]) == set(r["predicted_dimensions"]) for r in predictions) / len(rows),
            "macro_positive_f1": sum(r["f1"] for r in per_dimension.values()) / len(LABELS),
            "per_dimension": per_dimension, "rows": predictions,
            "scope": "label-only validation diagnostics; split and semantic gates are not planner readiness",
            "thresholds_changed": False}


def compare_coverage(base_artifact, candidate_artifact, dataset_dir, *, encoder_path=None, reviewed_records=None):
    """Re-audit an existing candidate, then compare on original Validation only."""
    from .models import TaskModels
    rows, provenance = load_research_rows(dataset_dir)
    base = TaskModels.load(base_artifact, encoder_path=encoder_path)
    candidate = TaskModels.load(candidate_artifact, encoder_path=encoder_path)
    trained = candidate.metadata.get("coverage_training") or {}
    if trained.get("base_model_sha256") != base.metadata["model_sha256"]:
        raise ValueError("candidate does not reference this base artifact")
    if trained.get("files") != provenance["files"]:
        raise ValueError("evaluation data differs from the candidate's recorded source files")
    extra = []
    if reviewed_records is not None:
        extra, reviewed_hash = _reviewed_rows(reviewed_records)
        if trained.get("reviewed_seed_sha256") is not None and trained["reviewed_seed_sha256"] != reviewed_hash:
            raise ValueError("reviewed records differ from the candidate's recorded seed")
    partition_audit = _check_external_partitions(
        rows + extra, base.metadata, provenance["heldout_fingerprints"], reviewed_records)
    validation = [row for row in rows if row["split"] == "validation"]
    base_report = evaluate_label_validation(base, validation)
    # Both models have verified the same encoder bytes; reuse the loaded object.
    if base._features.encoder_sha256 != candidate._features.encoder_sha256:
        raise ValueError("base and candidate encoder fingerprints differ")
    candidate._features._encoder = base._features._encoder
    candidate_report = evaluate_label_validation(candidate, validation)
    return {"evaluation": "RESEARCH_VALIDATION_ONLY_NOT_INDEPENDENT_BLIND", "created_at": datetime.now(timezone.utc).isoformat(),
            "human_gold": False, "test_labels_opened": False, "fit_performed": False, "thresholds_changed": False,
            "promotion_status": "NOT_DECIDED_BY_EVALUATOR", "base_partition_reaudit": partition_audit,
            "source_files": provenance["files"], "base_model_sha256": base.metadata["model_sha256"],
            "candidate_model_sha256": candidate.metadata["model_sha256"],
            "candidate_threshold_policy": trained.get("threshold_policy", "legacy_unrecorded"),
            "candidate_threshold_placement": {d: candidate.metadata["calibration"][d].get("threshold_placement") for d in LABELS},
            "preprocessing": "context_newline_USER_newline_text_v1; supplied CSV context is preserved",
            "warning": "Candidate calibration and thresholds use this Validation split; metrics are descriptive, not independent accuracy evidence",
            "base": base_report, "candidate": candidate_report}
