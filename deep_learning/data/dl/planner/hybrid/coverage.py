"""Explicit AI-labelled research refit; reviewed planning gates stay frozen.

This is separate from the human-only ``train_models`` entrypoint. Private TEST
labels are never read. Label-free TEST inputs are fingerprinted for exclusion.
No candidate is promoted automatically and no task is executed here.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile

from .data import (LABELS, SUPPORTED_POLICY_VERSION, normalize_text,
                   normalized_prompt_sha256, normalized_task_context_sha256,
                   validate_records)


POLICY_SHA256 = "aebcea2a8b73c6404bd652d9ca2e556c43dfbafd4ed1b07081d6b292ad809a57"
LABEL_CREATOR = "ChatGPT independent v0.5.1 annotation review; NOT human double-adjudicated Gold"
RESEARCH_STATUS = "READY_FOR_RESEARCH_TRAINING_NOT_INDEPENDENT_BLIND_BENCHMARK"
THRESHOLD_POLICIES = ("original", "equivalent-midpoint")


def _place_research_threshold(scores, threshold, policy):
    """Optionally center a threshold inside its decision-equivalent score gap.

    Scores must come from the same calibration feature batch used by _fit_head.
    No truth labels, diagnostic prompts or Test inputs influence this placement.
    """
    if policy not in THRESHOLD_POLICIES:
        raise ValueError("unknown research threshold policy")
    values = [float(value) for value in scores]
    original = float(threshold)
    if not values or not all(math.isfinite(value) and 0 <= value <= 1 for value in [original, *values]):
        raise ValueError("research threshold placement requires finite probability scores")
    below = [value for value in values if value < original]
    above = [value for value in values if value >= original]
    lower, upper = max(below) if below else None, min(above) if above else None
    chosen, status = original, "kept_original_policy"
    if policy == "equivalent-midpoint":
        status = "kept_no_two_sided_gap"
        if lower is not None and upper is not None:
            midpoint = lower + (upper - lower) / 2
            status = "kept_no_representable_midpoint"
            if lower < midpoint < upper:
                chosen, status = midpoint, "equivalent_midpoint"
    identical = all((value >= original) == (value >= chosen) for value in values)
    if not identical:
        raise ValueError("research threshold placement changed calibration decisions")
    return chosen, {"policy": policy, "status": status, "original_threshold": original,
        "selected_threshold": chosen, "largest_score_below_threshold": lower,
        "smallest_score_at_or_above_threshold": upper,
        "gap_width": upper - lower if lower is not None and upper is not None else None,
        "calibration_batch_size": len(values), "identical_validation_decisions": identical,
        "score_source": "same_validation_feature_batch_as_fit_head"}


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_fingerprint(path):
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate CSV columns")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError("research CSV must contain complete rows")
    return fields, rows


def _isolation(rows):
    """Check text alone as well as text/context so contexts cannot hide overlap."""
    seen = {name: {} for name in ("id", "group", "text", "input")}
    for row in rows:
        split = row["split"]
        keys = {"id": row["sample_id"], "group": row["group_id"],
                "text": normalized_prompt_sha256(row["text"]),
                "input": normalized_task_context_sha256(row["text"], row["context"])}
        for kind, key in keys.items():
            previous = seen[kind].setdefault(key, split)
            if previous != split:
                raise ValueError(f"{kind} leakage/overlap across {previous} and {split}")


def load_research_rows(directory):
    """Load fixed v0.5.1 Train/Validation files, preserving AI provenance.

    The manifest attests provenance; it cannot certify independent human review.
    Test inputs participate only in identity/group/text exclusions, never in fit,
    calibration, vocabulary construction, or model selection.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    if (manifest.get("status") != RESEARCH_STATUS or manifest.get("label_creator") != LABEL_CREATOR
            or manifest.get("policy_sha256") != POLICY_SHA256):
        raise ValueError("unsupported AI research provenance/policy; never claim human approval")
    rows, heldout, files = [], [], {}
    for split, filename in (("train", "train.csv"), ("validation", "validation.csv"), ("test", "test_inputs.csv")):
        path = directory / filename
        fields, raw = _csv(path)
        required = {"sample_id", "text", "context", "group_id", "source"}
        if not required.issubset(fields) or (split != "test" and not set(LABELS).issubset(fields)):
            raise ValueError(f"{filename} missing required input or label columns")
        if set(fields) & {"reviewer", "review_source", "review_status", "human_gold"}:
            raise ValueError("AI research CSV cannot carry human review provenance claims")
        if split == "test" and set(fields) & set(LABELS):
            raise ValueError("test inputs must be label-free; private test labels are forbidden")
        if type(manifest.get("counts", {}).get(split.upper())) is not int or len(raw) != manifest["counts"][split.upper()]:
            raise ValueError(f"{filename} row count differs from manifest")
        ids = set()
        for item in raw:
            if any(not isinstance(item.get(key), str) or not item[key].strip()
                   for key in ("sample_id", "text", "group_id", "source")) or not isinstance(item.get("context"), str):
                raise ValueError("research input fields must be explicit strings")
            if "split" in item and item["split"] != split:
                raise ValueError("CSV split does not match its fixed train/validation/test role")
            if item["sample_id"] in ids:
                raise ValueError("duplicate sample_id in research CSV")
            ids.add(item["sample_id"])
            row = {key: item[key] for key in required}
            row.update(split=split, label_source="ai_research")
            if split != "test":
                if any(item[label] not in {"0", "1"} for label in LABELS):
                    raise ValueError("every research label must be an explicit binary 0/1")
                row["dimensions"] = [label for label in LABELS if item[label] == "1"]
                rows.append(row)
            else:
                heldout.append(row)
        files[filename] = _file_fingerprint(path)
    _isolation(rows + heldout)
    files["manifest.json"] = _file_fingerprint(directory / "manifest.json")
    return rows, {"kind": "v051_ai_label_only_research", "human_gold": False,
                  "label_creator": manifest["label_creator"], "label_policy_version": SUPPORTED_POLICY_VERSION,
                  "policy_sha256": POLICY_SHA256, "test_records_consumed": 0,
                  "test_input_rows_audited": len(heldout), "files": files,
                  "historical_independence": manifest.get("full_historical_pool_isolation", "UNVERIFIED"),
                  "known_limits": manifest.get("known_limits", []),
                  "heldout_fingerprints": [{"sample_id": r["sample_id"], "group_id": r["group_id"],
                      "text_sha256": normalized_prompt_sha256(r["text"]),
                      "input_sha256": normalized_task_context_sha256(r["text"], r["context"])} for r in heldout]}


def _reviewed_rows(records):
    checked = validate_records(records)
    rows = []
    for record in checked:
        tasks = [{"instruction": record["original_prompt"], "context": "", "dimensions": record["direct_dimensions"]}, *record["tasks"]]
        for index, task in enumerate(tasks):
            rows.append({"sample_id": f"{record['request_id']}:label:{index}", "group_id": record["group_id"],
                         "text": task["instruction"], "context": task["context"],
                         "dimensions": task["dimensions"], "split": record["split"],
                         "source": "reviewed_seed", "label_source": "human_reviewed_seed"})
    return rows, hashlib.sha256(_json_bytes(checked)).hexdigest()


def _check_external_partitions(rows, base_metadata, heldout, reviewed_records=None):
    base_sources = base_metadata.get("data", {}).get("sources", [])
    if not base_sources:
        raise ValueError("retained base source provenance is required")
    # Frozen gates also consumed the base Train/Validation records. Exclude
    # their known fingerprints independently of the newly fitted label rows.
    for source in base_sources:
        for test in heldout:
            if (source.get("request_id") == test["sample_id"] or source.get("group_id") == test["group_id"]
                    or source.get("normalized_prompt_sha256") == test["text_sha256"]
                    or test["input_sha256"] in source.get("normalized_task_context_sha256", [])):
                raise ValueError("retained base source overlaps label-free test inputs")
    base_rows = []
    if reviewed_records is not None:
        checked = validate_records(reviewed_records)
        records_by_id = {r["request_id"]: r for r in checked}
        original = []
        for source in base_sources:
            record = records_by_id.get(source.get("request_id"))
            if record is None or hashlib.sha256(_json_bytes(record)).hexdigest() != source.get("record_sha256"):
                raise ValueError("reviewed data must include every unchanged original base record")
            original.append(record)
        base_rows, _ = _reviewed_rows(original)
    elif any(not source.get("normalized_task_context_sha256")
             or len(set(source["normalized_task_context_sha256"])) > 1 for source in base_sources):
        # A pair hash does not reveal a subtask's text by itself. A new context
        # must not hide a text-only overlap with Test or an opposite partition.
        raise ValueError("raw original reviewed data is required to audit retained base subtask text")
    _isolation(rows + base_rows)
    for row in base_rows:
        if any(normalized_prompt_sha256(row["text"]) == test["text_sha256"] for test in heldout):
            raise ValueError("retained base subtask text overlaps label-free test inputs")
    for row in rows:
        prompt = normalized_prompt_sha256(row["text"])
        pair = normalized_task_context_sha256(row["text"], row["context"])
        for source in base_sources:
            if source.get("split") != row["split"] and (
                row["group_id"] == source.get("group_id") or prompt == source.get("normalized_prompt_sha256")
                or pair in source.get("normalized_task_context_sha256", [])):
                raise ValueError("research/seed row overlaps the base artifact's opposite split")
        for source in heldout:
            if (row["sample_id"] == source["sample_id"] or row["group_id"] == source["group_id"]
                    or prompt == source["text_sha256"] or pair == source["input_sha256"]):
                raise ValueError("training/validation row overlaps label-free test inputs")
    return {"retained_base_sources_checked": len(base_sources),
            "raw_original_records_verified": len(base_sources) if reviewed_records is not None else 0,
            "test_inputs_audited": len(heldout), "status": "NO_KNOWN_PARTITION_OVERLAP"}


def refit_label_heads(base_artifact, dataset_dir, directory, *, encoder_path=None,
                      reviewed_records=None, acknowledge_ai_research=False, threshold_policy="original"):
    """Refit labels only; emit a new TaskModels candidate with original gates."""
    if acknowledge_ai_research is not True:
        raise ValueError("explicit acknowledgement of AI research labels is required")
    if threshold_policy not in THRESHOLD_POLICIES:
        raise ValueError("unknown research threshold policy")
    from .models import (ARTIFACT_VERSION, FEATURE_SETTINGS, TaskModels, _Features,
                         _artifact_metadata, _fit_head, _label_text, _sha256_file)
    import joblib
    import numpy as np
    import sklearn

    output = Path(directory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("candidate output directory must be absent or empty; source artifacts are immutable")
    research, provenance = load_research_rows(dataset_dir)
    base = TaskModels.load(base_artifact, encoder_path=encoder_path)
    if base.metadata.get("coverage_training"):
        raise ValueError("coverage refit requires the original reviewed base artifact")
    rows = list(research)
    reviewed_sha256 = None
    if reviewed_records is not None:
        extra, reviewed_sha256 = _reviewed_rows(reviewed_records)
        # The two origins must be disjoint, even when their split agrees.
        research_groups = {r["group_id"] for r in research}
        research_texts = {normalized_prompt_sha256(r["text"]) for r in research}
        if any(r["group_id"] in research_groups or normalized_prompt_sha256(r["text"]) in research_texts for r in extra):
            raise ValueError("reviewed seed and AI research data overlap")
        rows.extend(extra)
    _isolation(rows)
    base_partition_audit = _check_external_partitions(
        rows, base.metadata, provenance["heldout_fingerprints"], reviewed_records)
    unique = {}
    for row in rows:
        key = (row["split"], normalized_task_context_sha256(row["text"], row["context"]))
        if key in unique and set(unique[key]["dimensions"]) != set(row["dimensions"]):
            raise ValueError("conflicting labels for duplicate classifier input")
        unique.setdefault(key, row)
    examples = list(unique.values())
    train = [r for r in examples if r["split"] == "train"]
    valid = [r for r in examples if r["split"] == "validation"]
    for label in LABELS:
        positives = sum(label in r["dimensions"] for r in valid)
        if positives < 2 or len(valid) - positives < 2:
            raise ValueError(f"validation {label} needs at least two positive and negative distinct inputs")
    features = _Features(base._features.encoder_path)
    if features.encoder_sha256 != base._features.encoder_sha256:
        raise ValueError("research classifier encoder must match the original LLC encoder")
    features._encoder = base._features._encoder
    train_text = [_label_text(r["text"], r["context"]) for r in train]
    valid_text = [_label_text(r["text"], r["context"]) for r in valid]
    features.fit(train_text)
    train_x, valid_x = features.transform(train_text), features.transform(valid_text)
    heads, thresholds, calibration = {}, {}, {}
    for label in LABELS:
        heads[label], thresholds[label], calibration[label] = _fit_head(
            label, train_x, np.asarray([int(label in r["dimensions"]) for r in train]),
            valid_x, np.asarray([int(label in r["dimensions"]) for r in valid]))
        thresholds[label], placement = _place_research_threshold(
            heads[label].scores(valid_x), thresholds[label], threshold_policy)
        calibration[label]["threshold_placement"] = placement
    gate_state = pickle.dumps((base._features, base._gates), protocol=4)
    metadata = copy.deepcopy(base.metadata)
    for key in ("model_sha256", "metadata_sha256", "manifest_sha256"):
        metadata.pop(key, None)
    metadata.update(model_id="hybrid_label_coverage_research_v1", research_only=True,
                    created_at=datetime.now(timezone.utc).isoformat(), thresholds=thresholds)
    metadata["calibration"].update(calibration)
    metadata["label_features"] = {"type": "char_tfidf_local_embeddings" if features.encoder_path else "char_tfidf",
        "settings": json.loads(_json_bytes(FEATURE_SETTINGS)), "fit_split": "train",
        "vocabulary_size": len(features.vectorizer.vocabulary_),
        "label_preprocessing": "context_newline_USER_newline_text_v1",
        "encoder_path": features.encoder_path, "encoder_sha256": features.encoder_sha256}
    metadata["training"] = {"label_heads": 8, "estimator": "independent_LogisticRegression",
        "fit_split": "train", "test_records_consumed": 0, "random_state": 0,
        "gate_training": "unchanged_original_reviewed_artifact"}
    metadata["versions"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__}
    metadata["coverage_training"] = {**provenance, "base_model_sha256": base.metadata["model_sha256"],
        "threshold_policy": threshold_policy,
        "base_partition_audit": base_partition_audit,
        "base_metadata_sha256": base.metadata["metadata_sha256"],
        "base_manifest_sha256": base.metadata["manifest_sha256"],
        "gate_state_sha256": hashlib.sha256(gate_state).hexdigest(), "gates_refitted": False,
        "reviewed_seed_sha256": reviewed_sha256, "deduplicated_label_inputs": len(rows) - len(examples),
        "label_examples": {"train": len(train), "validation": len(valid)},
        "research_rows": {s: sum(r["split"] == s for r in research) for s in ("train", "validation")},
        "promotion_status": "CANDIDATE_NOT_PROMOTED", "uncertainty_margin_changed": False,
        "warning": "AI-labelled research candidate; calibration and thresholds share validation; no independent accuracy claim"}
    sources = copy.deepcopy(base.metadata["data"]["sources"])
    sources.extend({"request_id": r["sample_id"], "group_id": r["group_id"], "split": r["split"],
        "record_sha256": hashlib.sha256(_json_bytes(r)).hexdigest(),
        "prompt_sha256": hashlib.sha256(r["text"].encode("utf-8")).hexdigest(),
        "normalized_prompt_sha256": normalized_prompt_sha256(r["text"]),
        "normalized_task_context_sha256": [normalized_task_context_sha256(r["text"], r["context"])],
        "label_source": r["label_source"]} for r in examples)
    metadata["data"] = {"sha256": hashlib.sha256(_json_bytes(examples)).hexdigest(),
        "sources": sources, "leakage_checks": ["sample_id", "group", "normalized_text", "normalized_text_context", "base_split", "test_inputs"],
        "independence_caution": "Historical full-pool isolation is unverified; TEST labels were never opened",
        "splits": {s: {"records": sum(r["split"] == s for r in sources),
                        "group_ids": sorted({r["group_id"] for r in sources if r["split"] == s})}
                   for s in ("train", "validation")}}
    # Recheck the exact source bytes before writing a candidate.
    for filename, expected in provenance["files"].items():
        if _file_fingerprint(Path(dataset_dir) / filename) != expected:
            raise ValueError("research source changed during training")
    for filename, key in (("models.joblib", "model_sha256"), ("metadata.json", "metadata_sha256"), ("manifest.json", "manifest_sha256")):
        if _sha256_file(Path(base_artifact) / filename) != base.metadata[key]:
            raise ValueError("base artifact changed during training")
    bundle = {"artifact_version": ARTIFACT_VERSION, "features": base._features,
              "label_features": features, "heads": heads, "gates": base._gates}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coverage-models-", dir=output.parent) as temporary:
        staging = Path(temporary)
        joblib.dump(bundle, staging / "models.joblib", compress=3)
        (staging / "metadata.json").write_bytes(_json_bytes(metadata))
        manifest = {"artifact_version": ARTIFACT_VERSION, "files": {
            name: _sha256_file(staging / name) for name in ("models.joblib", "metadata.json")}}
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        output.mkdir(exist_ok=True)
        for name in ("models.joblib", "metadata.json", "manifest.json"):
            os.replace(staging / name, output / name)
    return _artifact_metadata(output, metadata, manifest)
