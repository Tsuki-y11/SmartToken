"""Validate capability labels and estimate output tokens per task and model.

V1/v2 carry labels as metadata; v3 also consumes labels and their provenance.
Imports of ML runtimes are deferred until an artifact is explicitly loaded.
"""

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence


DIMENSIONS = frozenset({
    "agents", "coding", "reasoning", "knowledge", "math", "multilingual",
    "multimodal", "instruction_following",
})
SCHEMA_VERSION = "1.0"


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _alias(row: dict, primary: str, legacy: str) -> object:
    if primary in row and legacy in row and row[primary] != row[legacy]:
        raise ValueError(f"conflicting {primary} and {legacy}")
    return row.get(primary, row.get(legacy))


def validate_request(payload: object, *, require_label_source: bool = False) -> dict:
    """Normalize aliases and validate the entire batch without invoking a model."""
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    request_id = _nonempty_string(payload.get("request_id"), "request_id")
    rows = payload.get("subtasks")
    if rows == [] and isinstance(payload.get("direct_task"), dict):
        rows = [payload["direct_task"]]
    if not isinstance(rows, list) or not rows:
        raise ValueError("subtasks must be a nonempty list")
    normalized, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"subtask {index} must be an object")
        subtask_id = _nonempty_string(_alias(row, "subtask_id", "step_id"), "subtask_id")
        if subtask_id in seen:
            raise ValueError(f"duplicate subtask_id: {subtask_id}")
        seen.add(subtask_id)
        dimension = row.get("dimension")
        label_metadata = {}
        if "dimension" in row and (not isinstance(dimension, str) or dimension not in DIMENSIONS):
            raise ValueError(f"subtask {subtask_id}: dimension must be one of {sorted(DIMENSIONS)}")
        if "dimensions" in row:
            labels = row["dimensions"]
            if not isinstance(labels, list):
                raise ValueError(f"subtask {subtask_id}: dimensions must be a list")
            if any(not isinstance(label, str) or label not in DIMENSIONS for label in labels):
                raise ValueError(f"subtask {subtask_id}: unknown or non-string dimensions label")
            if len(set(labels)) != len(labels):
                raise ValueError(f"subtask {subtask_id}: duplicate dimensions labels")
            if "dimension" in row and labels != [dimension]:
                raise ValueError(f"subtask {subtask_id}: conflicting dimension and dimensions")
            label_metadata["dimensions"] = list(labels)
        elif "dimension" in row:
            label_metadata["dimension"] = dimension
        has_labels = "dimension" in row or "dimensions" in row
        if "label_source" in row:
            source = row["label_source"]
            allowed = ("human", "ml", "dataset_proxy") if has_labels else ("missing",)
            if source not in allowed:
                raise ValueError(f"subtask {subtask_id}: label_source must be one of {allowed}")
            label_metadata["label_source"] = source
        elif not has_labels:
            label_metadata["label_source"] = "missing"
        elif require_label_source:
            raise ValueError(f"subtask {subtask_id}: v3 requires label_source human, ml or dataset_proxy for present labels")
        mask = row.get("dimension_mask")
        if "dimension_mask" in row:
            if not has_labels:
                if mask is not None:
                    raise ValueError(f"subtask {subtask_id}: missing labels require dimension_mask None")
            else:
                if not isinstance(mask, list):
                    raise ValueError(f"subtask {subtask_id}: dimension_mask must be a list")
                if any(not isinstance(label, str) or label not in DIMENSIONS for label in mask):
                    raise ValueError(f"subtask {subtask_id}: unknown or non-string dimension_mask label")
                if len(set(mask)) != len(mask):
                    raise ValueError(f"subtask {subtask_id}: duplicate dimension_mask labels")
                positives = label_metadata.get("dimensions", [dimension] if "dimension" in row else [])
                if not set(positives).issubset(mask):
                    raise ValueError(f"subtask {subtask_id}: dimensions must be a subset of dimension_mask")
            label_metadata["dimension_mask"] = list(mask) if mask is not None else None
        if label_metadata.get("label_source") == "dataset_proxy" and mask is None:
            raise ValueError(f"subtask {subtask_id}: dataset_proxy requires an explicit dimension_mask")
        prompt = _alias(row, "prompt", "prompt_template")
        if "instruction" in row:
            if ("prompt" in row or "prompt_template" in row) and prompt != row["instruction"]:
                raise ValueError(f"subtask {subtask_id}: conflicting prompt and instruction")
            prompt = row["instruction"]
        dependencies, references = row.get("depends_on", []), row.get("input_refs", [])
        if not isinstance(dependencies, list) or not isinstance(references, list):
            raise ValueError(f"subtask {subtask_id}: depends_on and input_refs must be lists")
        has_upstream = bool(dependencies) or any(isinstance(ref, str) and ref.endswith(".output") for ref in references)
        if has_upstream and "rendered_prompt" not in row:
            raise ValueError(f"subtask {subtask_id}: upstream context requires a complete rendered_prompt")
        if "rendered_prompt" in row:
            prompt = row["rendered_prompt"]
        prompt = _nonempty_string(prompt, f"subtask {subtask_id} prompt")
        if "{{" in prompt or "}}" in prompt:
            raise ValueError(f"subtask {subtask_id}: unresolved placeholder; supply a complete rendered_prompt")
        normalized.append({"subtask_id": subtask_id, "prompt": prompt, **label_metadata})
    return {"request_id": request_id, "subtasks": normalized}


def _model_selection(model_ids: Sequence[str] | None, available: Sequence[str]) -> list[str]:
    if isinstance(model_ids, (str, bytes)):
        raise ValueError("model IDs must be a sequence of IDs")
    requested = list(available if model_ids is None else model_ids)
    if not requested or any(not isinstance(name, str) or not name.strip() for name in requested):
        raise ValueError("model IDs must be nonempty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate model IDs are not allowed")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown model IDs: {unknown}")
    return requested


class SubtaskPredictor:
    """Adapter around a predictor with model_ids and predict(text, model_ids)."""

    def __init__(self, predictor: object, predictor_version: str | None = None):
        self.predictor = predictor
        self.predictor_version = _nonempty_string(
            predictor_version or getattr(predictor, "predictor_version", "unknown"), "predictor_version")

    def predict(self, payload: object, model_ids: Sequence[str] | None = None) -> dict:
        uses_labels = self.predictor_version == "model_aware_v3"
        request = validate_request(payload, require_label_source=uses_labels)
        requested = _model_selection(model_ids, self.predictor.model_ids)
        predictions = []
        for row in request["subtasks"]:
            labels = row.get("dimensions", [row["dimension"]] if "dimension" in row else None)
            if uses_labels:
                estimates = self.predictor.predict(row["prompt"], requested,
                                                   dimensions=labels, label_source=row.get("label_source"),
                                                   dimension_mask=row.get("dimension_mask"))
            else:
                estimates = self.predictor.predict(row["prompt"], requested)
            if not isinstance(estimates, dict) or set(estimates) != set(requested):
                raise ValueError("predictor must return exactly the requested model IDs")
            for model_id in requested:
                estimate = estimates[model_id]
                if isinstance(estimate, bool) or not isinstance(estimate, Integral) or estimate <= 0:
                    raise ValueError("output token estimates must be positive finite integers")
                metadata = {"dimension": row["dimension"]} if "dimension" in row else {
                    "dimensions": list(labels) if labels is not None else None}
                if "label_source" in row:
                    metadata["label_source"] = row["label_source"]
                if "dimension_mask" in row:
                    mask = row["dimension_mask"]
                    metadata["dimension_mask"] = list(mask) if mask is not None else None
                predictions.append({"subtask_id": row["subtask_id"], **metadata,
                                    "model_id": model_id, "output_tokens_est": int(estimate)})
        return {"schema_version": SCHEMA_VERSION, "request_id": request["request_id"],
                "predictor_version": self.predictor_version, "predictions": predictions}

    @classmethod
    def load(cls, artifact_dir: str | Path, device: str = "cpu", **loader_kwargs) -> "SubtaskPredictor":
        artifact_dir = Path(artifact_dir)
        metadata = json.loads((artifact_dir / "model_config.json").read_text(encoding="utf-8"))
        architecture = metadata.get("architecture", "model_aware_v1")
        if architecture == "model_aware_v3":
            from deep_learning.dl.predictor_v3 import TokenPredictorV3
            predictor = TokenPredictorV3.load(artifact_dir, device=device, **loader_kwargs)
        elif architecture == "model_aware_v2":
            from deep_learning.dl.predictor_v2 import TokenPredictorV2
            predictor = TokenPredictorV2.load(artifact_dir, device=device, **loader_kwargs)
        elif architecture == "model_aware_v1":
            from deep_learning.dl.predictor import TokenPredictor
            predictor = TokenPredictor.load(artifact_dir, device=device, **loader_kwargs)
        else:
            raise ValueError(f"unknown predictor architecture: {architecture}")
        return cls(predictor, predictor_version=architecture)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write JSON to this path (default: stdout)")
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.tasks.read_text(encoding="utf-8-sig"))
        request = validate_request(payload)
        metadata = json.loads((args.artifact_dir / "model_config.json").read_text(encoding="utf-8"))
        requested = _model_selection(args.model_ids, metadata.get("model_ids", []))
        if metadata.get("architecture") == "model_aware_v3":
            request = validate_request(request, require_label_source=True)
        adapter = SubtaskPredictor.load(args.artifact_dir, device=args.device)
        response = adapter.predict(request, requested)
        output = json.dumps(response, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
