"""Reproducible inference runs and separately supplied, human-reviewed Gold."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from deep_learning.planner.src.task_decomposer.schema import DIMENSIONS
from deep_learning.planner.src.task_decomposer.validator import validate_plan_for_request, validate_run_request
from .planner import summarize_usage


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _labels(value):
    if (not isinstance(value, list) or any(not isinstance(d, str) for d in value)
            or len(set(value)) != len(value) or not set(value) <= set(DIMENSIONS)):
        raise ValueError("complete dimension lists are required; unknown is not zero")
    return set(value)


def multilabel_metrics(truth, predicted):
    if not truth or len(truth) != len(predicted):
        raise ValueError("nonempty equally sized label arrays are required")
    gold, pred = list(map(_labels, truth)), list(map(_labels, predicted))
    per = {}
    for label in DIMENSIONS:
        tp = sum(label in g and label in p for g, p in zip(gold, pred))
        fp = sum(label not in g and label in p for g, p in zip(gold, pred))
        fn = sum(label in g and label not in p for g, p in zip(gold, pred))
        tn = len(gold) - tp - fp - fn
        per[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "positive_support": tp + fn, "negative_support": tn + fp,
                      "precision": tp / (tp + fp) if tp + fp else 0.0,
                      "recall": tp / (tp + fn) if tp + fn else 0.0,
                      "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    tp, fp, fn = (sum(v[k] for v in per.values()) for k in ("tp", "fp", "fn"))
    return {"sample_count": len(gold), "per_dimension": per,
            "macro_f1": sum(v["f1"] for v in per.values()) / len(DIMENSIONS),
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "exact_match": sum(g == p for g, p in zip(gold, pred)) / len(gold),
            "zero_positive_labels": [d for d in DIMENSIONS if per[d]["positive_support"] == 0],
            "zero_division_policy": 0}


def inference_request(row):
    allowed = {"schema_version", "request_id", "original_prompt", "constraints", "generation", "group_id", "source"}
    if not isinstance(row, dict) or set(row) - allowed:
        raise ValueError("inference inputs must contain only request data, never Gold or review labels")
    result = {"schema_version": row.get("schema_version", "1.0"), "request_id": row.get("request_id"),
              "original_prompt": row.get("original_prompt"),
              "constraints": row.get("constraints", {"budget_usd": 0.0, "quality_priority": "balanced",
                                                      "max_subtasks": 4, "allow_split": True}),
              "generation": row.get("generation", {"temperature": .2, "max_output_tokens": 1024})}
    validate_run_request(result)
    return result


def run_experiment(inputs, planners, output_dir):
    requests = [inference_request(row) for row in inputs]
    if not requests or len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("experiment needs nonempty, uniquely identified inputs")
    if not planners or not set(planners) <= {"baseline", "ml", "llm", "hybrid"}:
        raise ValueError("invalid experiment modes")
    if any(planner.mode != mode for mode, planner in planners.items()):
        raise ValueError("planner mode and experiment mode must match")
    target = Path(output_dir)
    if target.exists():
        raise ValueError("experiment output directory must be new; existing runs are never overwritten")
    encoded = json.dumps(requests, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest = {"schema_version": "1.0", "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_sha256": hashlib.sha256(encoded).hexdigest(), "input_count": len(requests),
                "request_ids": [r["request_id"] for r in requests], "modes": list(planners),
                "gold_used_during_inference": False, "record_count": 0,
                "configuration": {m: {"planner": asdict(p.config), "model": getattr(p.models, "metadata", {}),
                                      "provider": getattr(p.llm, "public_config", {})}
                                  for m, p in planners.items()}}
    manifest["configuration_sha256"] = hashlib.sha256(
        json.dumps(manifest["configuration"], ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    target.mkdir(parents=True)
    write_json(target / "manifest.json", manifest)
    failures = 0
    with (target / "results.jsonl").open("w", encoding="utf-8") as output:
        for req in requests:
            for mode, planner in planners.items():
                try:
                    result = planner.analyze(req)
                    if result["plan"] is not None:
                        validate_plan_for_request(result["plan"], req)
                except Exception as exc:
                    # No provider body or arbitrary exception message is published.
                    result = {"plan": None, "audit": {"status": "failed", "error_type": type(exc).__name__,
                                                       "llm_calls": [], "usage_accounting_complete": False}}
                failures += result["plan"] is None
                record = {"request_id": req["request_id"], "mode": mode, **result}
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                manifest["record_count"] += 1
    manifest.update(status="complete_with_failures" if failures else "complete", failure_count=failures,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    manifest["results_sha256"] = hashlib.sha256((target / "results.jsonl").read_bytes()).hexdigest()
    write_json(target / "manifest.json", manifest)
    return manifest


def _reviewed(row):
    if (not isinstance(row, dict) or row.get("review_status") != "approved"
            or row.get("review_source") != "human" or not isinstance(row.get("reviewer"), str)
            or not row["reviewer"].strip()):
        raise ValueError("Gold/alignment/semantic reviews require approved human review provenance")


def _check_known_training_overlap(gold, metadata):
    from .data import normalized_prompt_sha256, normalized_task_context_sha256
    if metadata.get("label_policy_version", gold["label_policy_version"]) != gold["label_policy_version"]:
        raise ValueError("model and Gold label policy mismatch")
    data = metadata.get("data", {})
    groups = {g for split in data.get("splits", {}).values() for g in split.get("group_ids", [])}
    if gold["group_id"] in groups:
        raise ValueError("test group overlaps training/validation provenance")
    raw_hash = hashlib.sha256(gold["original_prompt"].encode()).hexdigest()
    normalized_hash = normalized_prompt_sha256(gold["original_prompt"])
    task_hashes = {normalized_task_context_sha256(gold["original_prompt"], "")}
    task_hashes.update(normalized_task_context_sha256(t["instruction"], t.get("context", ""))
                       for t in gold.get("tasks", []) if isinstance(t.get("instruction"), str))
    for source in data.get("sources", []):
        if (source.get("request_id") == gold["request_id"] or source.get("prompt_sha256") == raw_hash
                or source.get("normalized_prompt_sha256") == normalized_hash
                or task_hashes.intersection(source.get("normalized_task_context_sha256", []))):
            raise ValueError("test request/prompt/task overlaps training/validation provenance")


def score_results(results, gold, *, alignments=None, reviews=None, expected_modes=None):
    if not gold:
        raise ValueError("empty Gold")
    gold_map, policies = {}, set()
    for row in gold:
        _reviewed(row)
        if row.get("split") != "test" or row.get("label_policy_version") != "0.5.1":
            raise ValueError("scoring requires held-out test Gold with supported policy 0.5.1")
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid.strip() or rid in gold_map:
            raise ValueError("invalid/duplicate Gold request_id")
        if not isinstance(row.get("original_prompt"), str) or not row["original_prompt"].strip():
            raise ValueError("Gold original_prompt is required")
        if not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
            raise ValueError("Gold source group_id is required")
        _labels(row.get("direct_dimensions"))
        policies.add(row["label_policy_version"])
        tasks = row.get("tasks", [])
        if not isinstance(tasks, list):
            raise ValueError("Gold tasks must be a list")
        seen = set()
        for task in tasks:
            tid = task.get("task_id")
            if not isinstance(tid, str) or not tid.strip() or tid in seen:
                raise ValueError("invalid/duplicate Gold task_id")
            seen.add(tid)
            _labels(task.get("dimensions"))
        gold_map[rid] = row
    if len(policies) != 1:
        raise ValueError("mixed annotation policies")
    modes = list(expected_modes) if expected_modes is not None else sorted({r.get("mode") for r in results})
    if not modes or not set(modes) <= {"baseline", "ml", "llm", "hybrid"} or len(set(modes)) != len(modes):
        raise ValueError("missing/invalid expected modes")
    keyed = {}
    for row in results:
        key = (row.get("mode"), row.get("request_id"))
        if key in keyed or key[0] not in modes or key[1] not in gold_map:
            raise ValueError("duplicate or unexpected prediction IDs/modes")
        _check_known_training_overlap(gold_map[key[1]], row.get("audit", {}).get("model_metadata", {}))
        if row.get("plan") is not None:
            if row["plan"].get("request_id") != key[1] or row["plan"]["direct_task"]["instruction"] != gold_map[key[1]]["original_prompt"]:
                raise ValueError("prediction and Gold inputs do not match")
            _labels(row["plan"]["direct_task"].get("dimensions"))
        keyed[key] = row
    if set(keyed) != {(mode, rid) for mode in modes for rid in gold_map}:
        raise ValueError("missing prediction records; failures must be recorded explicitly")
    aligned = {mode: ([], []) for mode in modes}
    used_gold, used_pred = set(), set()
    for row in alignments or []:
        _reviewed(row)
        key = (row.get("mode"), row.get("request_id"))
        if key not in keyed or keyed[key].get("plan") is None:
            raise ValueError("alignment references a missing/failed plan")
        gid, pid = row.get("gold_task_id"), row.get("predicted_task_id")
        if (*key, gid) in used_gold or (*key, pid) in used_pred:
            raise ValueError("only unique human-confirmed one-to-one task alignments are supported")
        truth = {t["task_id"]: t["dimensions"] for t in gold_map[key[1]].get("tasks", [])}
        plan = keyed[key]["plan"]
        predicted = {t["subtask_id"]: t["dimensions"] for t in (plan["subtasks"] or [plan["direct_task"]])}
        if gid not in truth or pid not in predicted:
            raise ValueError("unknown task ID in human alignment")
        aligned[key[0]][0].append(truth[gid])
        aligned[key[0]][1].append(predicted[pid])
        used_gold.add((*key, gid))
        used_pred.add((*key, pid))
    review_fields = ("coverage_ok", "inputs_ok", "dependencies_ok", "constraint_scope_ok", "executable")
    review_map = {}
    for row in reviews or []:
        _reviewed(row)
        key = (row.get("mode"), row.get("request_id"))
        if key not in keyed or key in review_map or keyed[key].get("plan") is None:
            raise ValueError("invalid/duplicate semantic review")
        if any(not isinstance(row.get(f), bool) for f in review_fields):
            raise ValueError("semantic review must explicitly score all five criteria")
        review_map[key] = row
    summary = {"research_only": True, "label_policy_version": next(iter(policies)), "modes": {},
               "independent_gold_provenance_verified": False,
               "warning": "Review fields and group IDs cannot independently prove historical train/test isolation. F1 is conditional on returned plans; report failure/ready rates alongside it. No semantic score is inferred from JSON validity."}
    for mode in modes:
        rows = [keyed[(mode, rid)] for rid in gold_map]
        success = [r for r in rows if r.get("plan") is not None]
        truth = [gold_map[r["request_id"]]["direct_dimensions"] for r in success]
        pred = [r["plan"]["direct_task"]["dimensions"] for r in success]
        calls = [c for r in rows for c in r.get("audit", {}).get("llm_calls", [])]
        usage = summarize_usage(calls)
        if any(r.get("audit", {}).get("usage_accounting_complete") is False for r in rows):
            usage.update(total_input_tokens=None, total_output_tokens=None, has_unknown_usage=True)
        checked = [v for (m, _), v in review_map.items() if m == mode]
        a_gold, a_pred = aligned[mode]
        latency = [r.get("audit", {}).get("latency_ms") for r in rows]
        known_latency = [v for v in latency if type(v) in (int, float) and math.isfinite(v) and v >= 0]
        summary["modes"][mode] = {
            "request_count": len(rows), "returned_plan_count": len(success),
            "failure_count": len(rows) - len(success),
            "ready_rate": sum(r.get("audit", {}).get("status") == "ok" and r.get("plan") is not None for r in rows) / len(rows),
            "request_label_metrics": multilabel_metrics(truth, pred) if success else None,
            "failure_inclusive_exact_match": sum(set(a) == set(b) for a, b in zip(truth, pred)) / len(rows),
            "aligned_subtask_label_metrics": multilabel_metrics(a_gold, a_pred) if a_gold else None,
            "aligned_subtask_count": len(a_gold),
            "gold_subtask_count": sum(len(g.get("tasks", [])) for g in gold_map.values()),
            "semantic_review_count": len(checked),
            "semantic_rates": {f: sum(v[f] for v in checked) / len(checked) for f in review_fields} if checked else None,
            "planning_call_count": len(calls), "planning_usage": usage,
            "mean_planning_latency_ms": sum(known_latency) / len(rows) if len(known_latency) == len(rows) else None,
            "latency_observed_count": len(known_latency),
        }
    return summary
