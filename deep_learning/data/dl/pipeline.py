"""Prompt -> learned task plan -> fitted per-model output-token predictions."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def normalize_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    if payload.get("input_refs") or payload.get("material") or payload.get("attachments"):
        raise ValueError("MATERIAL_NOT_SUPPORTED: include required text explicitly in original_prompt")
    prompt = payload.get("original_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("original_prompt must be nonempty")
    request_id = payload.get("request_id", str(uuid.uuid4()))
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be nonempty")
    return {"schema_version": payload.get("schema_version", "1.0"),
            "request_id": request_id, "original_prompt": prompt,
            "constraints": payload.get("constraints", {"budget_usd": 0.0,
                "quality_priority": "balanced", "max_subtasks": 4, "allow_split": True}),
            "generation": payload.get("generation", {"temperature": .2, "max_output_tokens": 1024})}


def render_tasks(plan, original_prompt, completed_outputs=None):
    """Defer dependent tasks until real upstream text is explicitly provided."""
    outputs = {} if completed_outputs is None else completed_outputs
    if not isinstance(outputs, dict):
        raise ValueError("completed_outputs must be an object of task ID -> actual text")
    tasks = ([plan["direct_task"]] if plan.get("direct_task") else []) + list(plan.get("subtasks", []))
    ids = {t["subtask_id"] for t in tasks}
    if set(outputs) - ids or any(not isinstance(v, str) or not v.strip() for v in outputs.values()):
        raise ValueError("completed_outputs contains an unknown task ID or empty/non-string text")
    rendered, pending = [], []
    for task in tasks:
        task_id = task["subtask_id"]
        refs = task.get("input_refs", [])
        required = {ref[:-7] for ref in refs if isinstance(ref, str) and ref.endswith(".output")}
        template_refs = set(re.findall(r"\{\{([A-Za-z0-9_-]+)\.output\}\}", task["instruction"]))
        required |= template_refs
        if required - set(task.get("depends_on", [])):
            raise ValueError("task refers to an output not declared in depends_on")
        missing = sorted((required | set(task.get("depends_on", []))) - set(outputs))
        if missing:
            pending.append({"subtask_id": task_id, "reason": "upstream_output_required",
                            "missing_outputs": missing})
            continue
        text = task["instruction"]
        for upstream in sorted(required):
            placeholder = "{{" + upstream + ".output}}"
            if placeholder in text:
                text = text.replace(placeholder, outputs[upstream])
            else:
                text += f"\n\n{upstream} 实际输出：\n{outputs[upstream]}"
        if "original_prompt" in refs and text.strip() != original_prompt.strip():
            text = "原始用户请求：\n" + original_prompt + "\n\n当前子任务：\n" + text
        if "{{" in text or "}}" in text:
            raise ValueError("unresolved placeholder; supply complete upstream text")
        rendered.append({"subtask_id": task_id, "prompt": text,
                         "dimensions": list(task["dimensions"]), "label_source": "ml"})
    return rendered, pending


class PredictionPipeline:
    def __init__(self, planner, predictor):
        self.planner, self.predictor = planner, predictor

    @property
    def model_ids(self):
        return list(self.predictor.model_ids)

    @classmethod
    def load(cls, *, planner_dir=None, predictor_dir=None, encoder_path=None, mode="ml", device="cpu"):
        from deep_learning.planner.hybrid.models import TaskModels
        from deep_learning.planner.hybrid.planner import HybridPlanner
        from deep_learning.dl.predictor_v3 import TokenPredictorV3
        if mode not in {"ml", "hybrid"}:
            raise ValueError("delivery pipeline mode must be ml or hybrid")
        encoder_path = Path(encoder_path or os.environ.get("SMARTTOKEN_ENCODER_PATH", ROOT / "artifacts/encoder"))
        models = TaskModels.load(planner_dir or ROOT / "artifacts/planner", encoder_path=encoder_path)
        encoder = models.local_encoder()
        encoder.to(device)
        predictor = TokenPredictorV3.load(predictor_dir or ROOT / "artifacts/token_predictor/labels",
                                           device=device, encoder=encoder)
        llm = None
        if mode == "hybrid":
            from deep_learning.planner.hybrid.llm import LLMClient
            llm = LLMClient.from_env()
        return cls(HybridPlanner(models, llm, mode=mode), predictor)

    def select_models(self, model_ids=None):
        if isinstance(model_ids, (str, bytes)):
            raise ValueError("model_ids must be a list")
        selected = self.model_ids if model_ids is None else list(model_ids)
        if (not selected or any(not isinstance(m, str) or m not in self.model_ids for m in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("model_ids must be nonempty, unique trained model IDs")
        return selected

    def analyze(self, payload):
        return self.planner.analyze(normalize_request(payload))

    def predict(self, payload, *, model_ids=None, completed_outputs=None):
        if completed_outputs is not None:
            raise ValueError("use repredict(prior_result, completed_outputs=...) to retain the original task plan")
        request = normalize_request(payload)
        selected = self.select_models(model_ids)
        analysis = self.planner.analyze(request)
        return self._estimate(request, analysis, selected, None)

    def repredict(self, prior_result, *, completed_outputs, model_ids=None):
        """Re-estimate the saved plan without running the planner a second time."""
        from deep_learning.planner.src.task_decomposer.validator import validate_plan_for_request
        if not isinstance(prior_result, dict):
            raise ValueError("prior_result must be a saved prediction result")
        request = normalize_request(prior_result.get("run_request"))
        plan, audit = prior_result.get("task_plan"), prior_result.get("planning_audit")
        validate_plan_for_request(plan, request)
        if not isinstance(audit, dict) or audit.get("status") not in {"ok", "needs_review", "needs_input", "failed"}:
            raise ValueError("prior_result has no valid planning audit")
        selected = self.select_models(model_ids)
        return self._estimate(request, {"plan": copy.deepcopy(plan), "audit": copy.deepcopy(audit)}, selected, completed_outputs)

    def _estimate(self, request, analysis, selected, completed_outputs):
        plan, audit = analysis["plan"], analysis["audit"]
        if plan is None:
            return {"schema_version": "1.0", "request_id": request["request_id"],
                    "status": audit["status"], "task_plan": None, "planning_audit": audit,
                    "prediction_matrix": {"schema_version": "1.0", "request_id": request["request_id"],
                        "predictor_version": self.predictor.predictor_version, "predictions": []},
                    "pending_predictions": [], "ready_for_optimization": False, "execution_performed": False}
        rendered, pending = render_tasks(plan, request["original_prompt"], completed_outputs)
        rows = []
        for task in rendered:
            estimates = self.predictor.predict(task["prompt"], selected,
                                               dimensions=task["dimensions"], label_source="ml")
            if not isinstance(estimates, dict) or set(estimates) != set(selected):
                raise ValueError("predictor did not return every requested model")
            for model_id in selected:
                estimate = estimates[model_id]
                if type(estimate) is not int or estimate <= 0:
                    raise ValueError("predictor returned a non-positive or non-integer estimate")
                rows.append({"subtask_id": task["subtask_id"], "model_id": model_id,
                             "dimensions": task["dimensions"], "label_source": "ml",
                             "output_tokens_est": estimate,
                             "phase": "with_actual_context" if completed_outputs and task["subtask_id"] != "DIRECT" else "pre_execution"})
        status = audit["status"] if audit["status"] != "ok" else ("awaiting_context" if pending else "ok")
        return {"schema_version": "1.0", "request_id": request["request_id"], "status": status,
                "run_request": request, "task_plan": copy.deepcopy(plan), "planning_audit": audit,
                "prediction_matrix": {"schema_version": "1.0", "request_id": request["request_id"],
                    "predictor_version": self.predictor.predictor_version, "predictions": rows},
                "pending_predictions": pending, "rendered_tasks": rendered,
                "ready_for_optimization": status == "ok", "execution_performed": False,
                "scope": "output token estimates only; candidate pricing, optimization and execution are separate"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--input", type=Path)
    source.add_argument("--previous-result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--completed-outputs", type=Path)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--mode", choices=("ml", "hybrid"), default="ml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        payload = None if args.previous_result else ({"original_prompt": args.prompt} if args.prompt else json.loads(args.input.read_text(encoding="utf-8-sig")))
        outputs = json.loads(args.completed_outputs.read_text(encoding="utf-8-sig")) if args.completed_outputs else None
        if args.completed_outputs and not args.previous_result:
            raise ValueError("--completed-outputs requires --previous-result to freeze the original plan")
        if args.previous_result and outputs is None:
            raise ValueError("--previous-result requires --completed-outputs")
        if args.output and args.output.exists():
            raise ValueError("output already exists; choose a new output path")
        pipeline = PredictionPipeline.load(encoder_path=args.encoder_path, mode=args.mode, device=args.device)
        if args.previous_result:
            previous = json.loads(args.previous_result.read_text(encoding="utf-8-sig"))
            result = deep_learning.pipeline.repredict(previous, model_ids=args.model_ids, completed_outputs=outputs)
        else:
            result = deep_learning.pipeline.predict(payload, model_ids=args.model_ids)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if result["ready_for_optimization"] else 2
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
