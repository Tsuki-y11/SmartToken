"""Adapter between SmartToken TaskPlan v1.7 and the LLC task_plans contract."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from .validator import validate_plan, validate_run_request


class AdapterError(ValueError):
    """Raised when an LLC request or TaskPlan cannot be adapted safely."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError("INVALID_REQUEST", f"{field} must be a non-empty string")
    return value


def llc_request_to_run_request(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an LLC `/plans` request to a v1.7 RunRequest.

    `input_refs` with content is intentionally rejected in v0.1. Accepting a
    filename without content would silently drop user material, and v1.7 does
    not yet define how to carry arbitrary material alongside the prompt.
    """
    if not isinstance(payload, dict):
        raise AdapterError("INVALID_REQUEST", "request body must be a JSON object")

    prompt_id = _require_nonempty_str(payload.get("prompt_id"), "prompt_id")
    original_prompt = _require_nonempty_str(
        payload.get("original_prompt"), "original_prompt"
    )

    input_refs = payload.get("input_refs", [])
    if input_refs:
        raise AdapterError(
            "MATERIAL_NOT_SUPPORTED",
            "input_refs content is not supported in Task Decomposer v0.1",
        )

    defaults = {
        "constraints": {"budget_usd": 1.0, "quality_priority": "balanced",
                        "max_subtasks": 4, "allow_split": True},
        "generation": {"temperature": 0.2, "max_output_tokens": 1024},
    }
    request: Dict[str, Any] = {
        "schema_version": "1.0",
        "request_id": prompt_id,
        "original_prompt": original_prompt,
    }
    for field in ("constraints", "generation"):
        provided = payload.get(field)
        if provided is None:
            request[field] = dict(defaults[field])
        elif not isinstance(provided, dict):
            raise AdapterError("INVALID_REQUEST", f"{field} must be an object")
        else:
            request[field] = {**defaults[field], **provided}
    validate_run_request(request)
    return request


def _dimensions(task: Mapping[str, Any]) -> List[str]:
    dimensions = task.get("dimensions")
    if not isinstance(dimensions, list):
        raise AdapterError(
            "INVALID_TASK",
            "dimensions must be a list; [] is allowed for general-text tasks.",
        )
    return list(dimensions)


def _prompt_template(task: Mapping[str, Any], original_prompt: str) -> str:
    instruction = task.get("instruction", "")
    if not isinstance(instruction, str) or not instruction.strip():
        raise AdapterError("INVALID_TASK", "instruction must be a non-empty string")

    dependencies = task.get("depends_on", [])
    if not isinstance(dependencies, list):
        raise AdapterError("INVALID_TASK", "depends_on must be an array")

    missing_placeholders: List[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, str):
            continue
        placeholder = "{{" + dependency + ".output}}"
        if placeholder not in instruction:
            missing_placeholders.append(placeholder)

    if missing_placeholders:
        # The LLC executor replaces `{{Tn.output}}`; the v1.7 instruction may
        # say "T1 result" without the executable placeholder syntax.
        suffix = "\n\n上游结果：\n" + "\n".join(missing_placeholders)
        instruction = instruction + suffix

    # input_refs is executable data, not metadata. Materialize the original
    # request for split tasks so the executor receives the facts being acted on.
    # Direct tasks already equal the original prompt and must not be duplicated.
    input_refs = task.get("input_refs", [])
    if "original_prompt" in input_refs and instruction.strip() != original_prompt.strip():
        instruction = (
            "原始用户请求：\n" + original_prompt +
            "\n\n当前子任务：\n" + instruction
        )
    return instruction


def _context_from(task: Mapping[str, Any]) -> List[Dict[str, str]]:
    input_refs = task.get("input_refs", [])
    if not isinstance(input_refs, list):
        raise AdapterError("INVALID_TASK", "input_refs must be an array")

    contexts: List[Dict[str, str]] = []
    for ref in input_refs:
        if not isinstance(ref, str):
            continue
        match = re.fullmatch(r"(T\d+)\.output", ref)
        if match:
            contexts.append(
                {
                    "step_id": match.group(1),
                    "mode": "full_output",
                }
            )
    return contexts


def _dimension_confidence(task: Mapping[str, Any]) -> Dict[str, float]:
    """Preserve LY's real per-dimension confidence in the LLC contract."""
    confidence = task.get("dimension_confidence", {})
    if not isinstance(confidence, dict):
        raise AdapterError(
            "INVALID_TASK",
            "dimension_confidence must be an object",
        )
    return dict(confidence)


def task_plan_to_task_plans(
    plan: Mapping[str, Any],
    original_prompt: str,
) -> Dict[str, Any]:
    """Convert a validated v1.7 TaskPlan to the LLC `task_plans.json` format."""
    original_prompt = _require_nonempty_str(original_prompt, "original_prompt")
    validate_plan(plan)

    request_id = plan.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise AdapterError("INVALID_TASK_PLAN", "TaskPlan.request_id must be non-empty")

    v1_subtasks = plan.get("subtasks", [])
    if not isinstance(v1_subtasks, list):
        raise AdapterError("INVALID_TASK_PLAN", "TaskPlan.subtasks must be an array")

    source_tasks: List[Mapping[str, Any]]
    if v1_subtasks:
        source_tasks = v1_subtasks
    else:
        direct_task = plan.get("direct_task")
        if not isinstance(direct_task, dict):
            raise AdapterError(
                "INVALID_TASK_PLAN",
                "direct_task is required when subtasks is empty",
            )
        source_tasks = [direct_task]

    legacy_subtasks: List[Dict[str, Any]] = []
    for index, task in enumerate(source_tasks, start=1):
        task_id = task.get("subtask_id", f"T{index}")
        if task_id == "DIRECT":
            task_id = f"T{index}"
        legacy_subtasks.append(
            {
                "step_id": task_id,
                "order": index,
                "prompt_template": _prompt_template(task, original_prompt),
                "dimensions": _dimensions(task),
                "dimension_confidence": _dimension_confidence(task),
                "depends_on": list(task.get("depends_on", [])),
                "context_from": _context_from(task),
            }
        )

    return {
        "schema_version": "1.0",
        "plans": [
            {
                "prompt_id": request_id,
                "original_prompt": original_prompt,
                "planning_method": "rules_ml",
                "subtasks": legacy_subtasks,
            }
        ],
    }
