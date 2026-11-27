"""Bounded ML/LLM orchestration; all research diagnostics remain a sidecar."""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import asdict, dataclass

from deep_learning.planner.src.task_decomposer.decomposer import TaskDecomposer
from deep_learning.planner.src.task_decomposer.schema import DIMENSIONS
from deep_learning.planner.src.task_decomposer.validator import validate_plan_for_request, validate_run_request
from .proposal import candidate_clauses, excluded_references, step_references, validate_proposal, validate_scores


@dataclass(frozen=True)
class PlannerConfig:
    uncertainty_margin: float = .08
    max_llm_calls: int = 2

    def __post_init__(self):
        if (isinstance(self.max_llm_calls, bool) or not isinstance(self.max_llm_calls, int)
                or not 0 <= self.max_llm_calls <= 2):
            raise ValueError("max_llm_calls must be an integer from 0 through 2")
        if (isinstance(self.uncertainty_margin, bool) or not isinstance(self.uncertainty_margin, (int, float))
                or not math.isfinite(self.uncertainty_margin) or not 0 <= self.uncertainty_margin <= .5):
            raise ValueError("uncertainty_margin must be finite and between 0 and .5")


def _token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def summarize_usage(calls):
    known_in = sum(_token_count(c.get("usage", {}).get("input_tokens")) or 0 for c in calls)
    known_out = sum(_token_count(c.get("usage", {}).get("output_tokens")) or 0 for c in calls)
    unknown_in = any(_token_count(c.get("usage", {}).get("input_tokens")) is None for c in calls)
    unknown_out = any(_token_count(c.get("usage", {}).get("output_tokens")) is None for c in calls)
    return {"known_input_tokens": known_in, "known_output_tokens": known_out,
            "total_input_tokens": None if unknown_in else known_in,
            "total_output_tokens": None if unknown_out else known_out,
            "has_unknown_usage": unknown_in or unknown_out,
            "scope": "planning_only_excludes_task_execution"}


class HybridPlanner:
    def __init__(self, models, llm=None, mode="hybrid", config=None):
        if mode not in {"baseline", "ml", "llm", "hybrid"}:
            raise ValueError("unknown planner mode")
        if models is None and mode != "llm":
            raise ValueError("this mode requires a classifier")
        self.models, self.llm, self.mode = models, llm, mode
        self.config = config or PlannerConfig()

    def _task(self, base, scores, thresholds):
        dimensions = [d for d in DIMENSIONS if scores[d] >= thresholds[d]]
        return {**base, "dimensions": dimensions,
                "dimension_confidence": {d: scores[d] for d in dimensions}}

    def _score(self, task_id, text, context, audit):
        scores = validate_scores(self.models.score_labels(text, context))
        audit["scores"][task_id] = scores
        audit["classification_inputs"][task_id] = {"text": text, "context": context}
        return scores

    def _uncertain(self, scores):
        return any(abs(scores[d] - self.models.thresholds[d]) <= self.config.uncertainty_margin for d in DIMENSIONS)

    def _uncertainty_details(self, scores):
        return [{"dimension": d, "score": scores[d], "threshold": self.models.thresholds[d],
                 "distance": abs(scores[d] - self.models.thresholds[d]),
                 "margin": self.config.uncertainty_margin,
                 "selected": scores[d] >= self.models.thresholds[d]}
                for d in DIMENSIONS
                if abs(scores[d] - self.models.thresholds[d]) <= self.config.uncertainty_margin]

    def _call(self, purpose, payload, audit):
        if self.llm is None or len(audit["llm_calls"]) >= self.config.max_llm_calls:
            return None
        record = {"attempt": len(audit["llm_calls"]) + 1, "purpose": purpose,
                  "status": "failed", "usage": {"input_tokens": None, "output_tokens": None}}
        audit["llm_calls"].append(record)
        started = time.perf_counter()
        try:
            response = self.llm.complete(purpose, payload)
            usage = response.get("usage") or {}
            record.update(status="received", usage={k: _token_count(usage.get(k)) for k in ("input_tokens", "output_tokens")},
                          model=response.get("model"), prompt_sha256=response.get("prompt_sha256"))
            return response["data"]
        except Exception as exc:
            # Provider exceptions may contain credentials/raw bodies; only a
            # sanitized typed code is recorded, never str(exc).
            code = getattr(exc, "code", type(exc).__name__)
            record["error_code"] = code if re.fullmatch(r"[A-Za-z0-9_]{1,80}", str(code)) else "PROVIDER_ERROR"
            usage = getattr(exc, "usage", None) or {}
            record["usage"] = {k: _token_count(usage.get(k)) for k in ("input_tokens", "output_tokens")}
            return None
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def _initial_proposal(self, request, audit):
        prompt = request["original_prompt"]
        direct = {"tasks": [{"instruction": prompt, "source_quotes": [prompt], "depends_on": [],
                              "input_refs": ["original_prompt"], "constraints": []}],
                  "final_constraints": [], "missing_information": []}
        constraints = request["constraints"]
        if not constraints["allow_split"] or constraints["max_subtasks"] < 2:
            return direct, []
        chunks = candidate_clauses(prompt)
        probability = self.models.score_split(prompt)
        if probability is None:
            audit["gate_source"] = "heuristic_no_trained_split_artifact"
            if self.mode == "ml":
                raise ValueError("ML mode requires a trained split/boundary artifact")
            actions = re.findall(r"读取|计算|分析|解释|翻译|总结|编写|生成|搜索|查找|绘制", prompt)
            actions += re.findall(r"\b(?:read|calculate|compute|analy[sz]e|explain|translate|summari[sz]e|write|generate|search|find|plot)\b", prompt, re.I)
            reasons = ["heuristic_split_requires_llm"] if len(chunks) > 1 or len(actions) > 1 else []
            return direct, reasons
        if not isinstance(probability, (float, int)) or isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("invalid learned split score")
        audit["gate_source"] = "learned"
        audit["split_score"] = float(probability)
        reasons = []
        if abs(probability - self.models.split_threshold) <= self.config.uncertainty_margin:
            reasons.append("uncertain_split")
        if probability < self.models.split_threshold:
            return direct, reasons
        if len(chunks) < 2:
            return direct, reasons + ["split_without_candidate_boundary"]
        groups, current = [], [chunks[0]]
        for left, right in zip(chunks, chunks[1:]):
            boundary = self.models.score_boundary(left, right)
            if boundary is None:
                raise ValueError("ML mode requires a trained boundary artifact")
            if isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or not math.isfinite(boundary) or not 0 <= boundary <= 1:
                raise ValueError("invalid learned boundary score")
            if abs(boundary - self.models.boundary_threshold) <= self.config.uncertainty_margin:
                reasons.append("uncertain_boundary")
            if boundary >= self.models.boundary_threshold:
                groups.append(current)
                current = [right]
            else:
                current.append(right)
        groups.append(current)
        if not 2 <= len(groups) <= constraints["max_subtasks"]:
            return direct, reasons + ["no_admissible_split"]
        parallel = bool(re.search(r"同时|分别|meanwhile", prompt, re.I)) and not re.search(r"首先|然后|最后|\bthen\b|\bfinally\b", prompt, re.I)
        tasks = []
        for index, quotes in enumerate(groups, start=1):
            instruction = "；".join(quotes)
            refs = step_references(instruction) - excluded_references(instruction)
            refs = {r for r in refs if int(r[1:]) < index}
            deps = sorted(refs, key=lambda v: int(v[1:])) if refs else ([] if index == 1 or parallel else [f"T{index - 1}"])
            excluded = excluded_references(instruction)
            deps = [d for d in deps if d not in excluded]
            tasks.append({"instruction": instruction, "source_quotes": quotes, "depends_on": deps,
                          "input_refs": ["original_prompt"] + [f"{d}.output" for d in deps], "constraints": []})
        if re.search(r"最后|最终|只根据|不使用|不要|第[一二三四\d]步|\bonly\b|\bfinally\b|JSON|中文|英文", prompt, re.I):
            reasons.append("scope_or_reference_requires_review")
        return {"tasks": tasks, "final_constraints": [], "missing_information": []}, list(dict.fromkeys(reasons))

    def _materialize(self, normalized, direct, audit):
        if len(normalized["tasks"]) == 1:
            return []
        tasks = []
        for base in normalized["tasks"]:
            task_id = base["subtask_id"]
            context = "\n".join(normalized["source_quotes"][task_id])
            scores = self._score(task_id, base["instruction"], context, audit)
            tasks.append(self._task(base, scores, self.models.thresholds))
        return tasks

    def analyze(self, request):
        validate_run_request(request)
        started = time.perf_counter()
        prompt = request["original_prompt"]
        audit = {"mode": self.mode, "research_only": True, "status": "ok", "reasons": [],
                 "scores": {}, "classification_inputs": {}, "llm_calls": [],
                 "confidence_source": "ml_model_score", "semantic_quality_verified": False,
                 "config": asdict(self.config), "model_metadata": getattr(self.models, "metadata", {}),
                 "provider_config": getattr(self.llm, "public_config", {}),
                 "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}

        def finish(plan):
            if plan is not None:
                validate_plan_for_request(plan, request)
            if self.models is not None:
                relevant = ([task["subtask_id"] for task in plan.get("subtasks", [])]
                            if plan else []) or ["DIRECT"]
                audit["uncertain_dimensions"] = {
                    task_id: self._uncertainty_details(audit["scores"][task_id])
                    for task_id in relevant if task_id in audit["scores"]}
            audit["usage"] = summarize_usage(audit["llm_calls"])
            audit["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            audit["reasons"] = list(dict.fromkeys(audit["reasons"]))
            return {"plan": plan, "audit": audit}

        direct_base = {"subtask_id": "DIRECT", "instruction": prompt, "depends_on": [],
                       "input_refs": ["original_prompt"], "output_type": "text"}
        plan = {"schema_version": "1.0", "request_id": request["request_id"], "subtasks": [], "direct_task": None}
        if self.mode == "llm":
            return self._llm_only(request, plan, direct_base, audit, finish)
        validate_scores(self.models.thresholds)
        if self.mode == "baseline":
            records = {}
            owner = self

            class ClassifierAdapter:
                def classify(self, text, context=""):
                    scores = validate_scores(owner.models.score_labels(text, context))
                    records[text] = scores
                    value = owner._task({}, scores, owner.models.thresholds)
                    return value

            plan = TaskDecomposer(ClassifierAdapter()).analyze(request)
            for task in [plan["direct_task"], *plan["subtasks"]]:
                audit["scores"][task["subtask_id"]] = records[task["instruction"]]
            audit["gate_source"] = "unchanged_rule_baseline"
            return finish(plan)

        scores = self._score("DIRECT", prompt, "", audit)
        direct = self._task(direct_base, scores, self.models.thresholds)
        plan["direct_task"] = direct
        raw, reasons = self._initial_proposal(request, audit)
        validation_errors = []
        try:
            normalized = validate_proposal(raw, prompt, request["constraints"]["max_subtasks"])
        except ValueError as exc:
            reasons.append("invalid_initial_proposal")
            validation_errors = [str(exc)]
            normalized = {"tasks": [direct_base], "source_quotes": {"DIRECT": [prompt]},
                          "missing_information": [], "final_constraints": [],
                          "semantic_completeness_verified": False}
        tasks = self._materialize(normalized, direct, audit)
        relevant = [t["subtask_id"] for t in tasks] or ["DIRECT"]
        if any(self._uncertain(audit["scores"][tid]) for tid in relevant):
            reasons.append("uncertain_labels")
        if self.mode == "ml":
            plan["subtasks"] = tasks
            audit["status"] = "needs_review" if reasons else "ok"
            audit["reasons"] = reasons
            audit["proposal_checks"] = normalized
            return finish(plan)
        if not reasons:
            plan["subtasks"] = tasks
            audit["proposal_checks"] = normalized
            return finish(plan)
        audit["reasons"].extend(reasons)
        if self.llm is None:
            audit["status"] = "needs_review"
            audit["reasons"].append("llm_unavailable")
            return finish(plan)

        max_tasks = request["constraints"]["max_subtasks"] if request["constraints"]["allow_split"] else 1
        accepted = None
        while len(audit["llm_calls"]) < self.config.max_llm_calls:
            purpose = "decompose" if not audit["llm_calls"] else "review"
            payload = {"original_prompt": prompt, "max_subtasks": max_tasks,
                       "allow_split": request["constraints"]["allow_split"],
                       "prior_proposal": raw, "review_reasons": audit["reasons"],
                       "validation_errors": validation_errors}
            candidate = self._call(purpose, payload, audit)
            if candidate is None:
                audit["reasons"].append("provider_failed")
                continue
            try:
                normalized = validate_proposal(candidate, prompt, max_tasks)
            except ValueError as exc:
                audit["llm_calls"][-1]["status"] = "invalid_response"
                audit["reasons"].append("invalid_semantic_proposal")
                raw, validation_errors = candidate, [str(exc)]
                continue
            raw, accepted = candidate, normalized
            if normalized["missing_information"]:
                audit["status"] = "needs_input"
                audit["missing_information"] = normalized["missing_information"]
                audit["proposal_checks"] = normalized
                return finish(plan)
            try:
                tasks = self._materialize(normalized, direct, audit)
            except Exception as exc:
                audit["status"] = "failed"
                audit["reasons"].append("classification_failed_after_provider")
                audit["error_type"] = type(exc).__name__
                return finish(None)
            relevant = [t["subtask_id"] for t in tasks] or ["DIRECT"]
            uncertain = any(self._uncertain(audit["scores"][tid]) for tid in relevant)
            if not uncertain:
                break
            audit["reasons"].append("uncertain_labels_after_semantic_parse")
        if accepted is None:
            audit["status"] = "needs_review"
            audit["reasons"].append("no_valid_semantic_proposal")
        else:
            plan["subtasks"] = tasks
            audit["proposal_checks"] = accepted
            if uncertain:
                audit["status"] = "needs_review"
        return finish(plan)

    def _llm_only(self, request, plan, direct_base, audit, finish):
        audit["confidence_source"] = "llm_self_report_uncalibrated"
        audit["gate_source"] = "llm"
        audit["status"] = "failed"
        if self.llm is None:
            audit["reasons"].append("llm_unavailable")
            return finish(None)
        limit = request["constraints"]["max_subtasks"] if request["constraints"]["allow_split"] else 1
        raw = self._call("decompose", {"original_prompt": request["original_prompt"], "max_subtasks": limit,
                                     "allow_split": request["constraints"]["allow_split"]}, audit)
        try:
            normalized = validate_proposal(raw, request["original_prompt"], limit)
        except ValueError:
            audit["reasons"].append("invalid_semantic_proposal")
            return finish(None)
        if normalized["missing_information"]:
            audit["status"] = "needs_input"
            audit["missing_information"] = normalized["missing_information"]
            return finish(None)
        tasks = normalized["tasks"] if len(normalized["tasks"]) > 1 else []
        bases = [direct_base, *tasks]
        expected = {t["subtask_id"] for t in bases}
        payload = {"tasks": [{"task_id": t["subtask_id"], "instruction": t["instruction"],
                              "context": "\n".join(normalized["source_quotes"].get(t["subtask_id"], []))} for t in bases]}
        response = self._call("classify", payload, audit)
        try:
            rows = response["predictions"]
            if not isinstance(rows, list) or len(rows) != len(expected):
                raise ValueError("classification count mismatch")
            predictions = {r["task_id"]: validate_scores(r["scores"]) for r in rows}
            if set(predictions) != expected:
                raise ValueError("classification IDs mismatch")
        except (TypeError, KeyError, ValueError):
            audit["reasons"].append("invalid_llm_classification")
            return finish(None)
        thresholds = {d: .5 for d in DIMENSIONS}
        plan["direct_task"] = self._task(direct_base, predictions["DIRECT"], thresholds)
        plan["subtasks"] = [self._task(t, predictions[t["subtask_id"]], thresholds) for t in tasks]
        audit.update(status="ok", scores=predictions, proposal_checks=normalized)
        return finish(plan)
