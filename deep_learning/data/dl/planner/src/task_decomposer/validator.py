"""Strict validators for the frozen RunRequest/TaskPlan JSON V1.0 contract.

The validator intentionally does not infer user intent. It checks only the
contract-level invariants that every downstream module may rely on.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .schema import (
    DIMENSIONS,
    DIMENSION_SET,
    GENERATION_KEYS,
    OUTPUT_TYPES,
    QUALITY_PRIORITIES,
    RUN_REQUEST_KEYS,
    SCHEMA_VERSION,
    CONSTRAINT_KEYS,
    TASK_KEYS,
    TASK_PLAN_KEYS,
)


class TaskPlanValidationError(ValueError):
    """Raised when a request or plan violates the frozen V1.0 contract."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        if not self.errors:
            self.errors = ["unknown validation error"]
        super().__init__("; ".join(self.errors))


def _is_mapping(value: Any, field: str, errors: List[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return False
    return True


def _check_exact_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    field: str,
    errors: List[str],
) -> None:
    actual = set(value)
    missing = sorted(allowed - actual)
    extra = sorted(actual - allowed)
    if missing:
        errors.append(f"{field} is missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{field} has unexpected fields: {', '.join(extra)}")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_run_request(request: Mapping[str, Any]) -> None:
    """Validate a complete RunRequest and raise on the first invalid state."""

    errors: List[str] = []
    if not _is_mapping(request, "RunRequest", errors):
        raise TaskPlanValidationError(errors)

    _check_exact_keys(request, RUN_REQUEST_KEYS, "RunRequest", errors)

    if request.get("schema_version") != SCHEMA_VERSION:
        errors.append("RunRequest.schema_version must be '1.0'")

    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        errors.append("RunRequest.request_id must be a non-empty string")

    prompt = request.get("original_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("RunRequest.original_prompt must be a non-empty string")

    constraints = request.get("constraints")
    if _is_mapping(constraints, "RunRequest.constraints", errors):
        assert constraints is not None
        _check_exact_keys(constraints, CONSTRAINT_KEYS, "RunRequest.constraints", errors)

        budget = constraints.get("budget_usd")
        if (
            not isinstance(budget, (int, float))
            or isinstance(budget, bool)
            or not math.isfinite(float(budget))
            or float(budget) < 0.0
        ):
            errors.append("constraints.budget_usd must be a finite non-negative number")

        if constraints.get("quality_priority") not in QUALITY_PRIORITIES:
            errors.append(
                "constraints.quality_priority must be one of: "
                + ", ".join(sorted(QUALITY_PRIORITIES))
            )

        max_subtasks = constraints.get("max_subtasks")
        if not _is_integer(max_subtasks) or not 1 <= max_subtasks <= 4:
            errors.append("constraints.max_subtasks must be an integer from 1 through 4")

        if not isinstance(constraints.get("allow_split"), bool):
            errors.append("constraints.allow_split must be a boolean")

    generation = request.get("generation")
    if _is_mapping(generation, "RunRequest.generation", errors):
        assert generation is not None
        _check_exact_keys(generation, GENERATION_KEYS, "RunRequest.generation", errors)

        temperature = generation.get("temperature")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or not 0.0 <= float(temperature) <= 2.0
        ):
            errors.append("generation.temperature must be a finite number from 0 through 2")

        max_tokens = generation.get("max_output_tokens")
        if not _is_integer(max_tokens) or max_tokens < 1:
            errors.append("generation.max_output_tokens must be a positive integer")

    if errors:
        raise TaskPlanValidationError(errors)


def _validate_dimensions(
    task: Mapping[str, Any],
    field: str,
    errors: List[str],
    confidence_required: bool,
) -> None:
    dimensions = task.get("dimensions")
    if not isinstance(dimensions, list):
        errors.append(f"{field}.dimensions must be an array")
        return

    if any(not isinstance(item, str) for item in dimensions):
        errors.append(f"{field}.dimensions must contain only strings")
        return

    if len(set(dimensions)) != len(dimensions):
        errors.append(f"{field}.dimensions must not contain duplicates")

    invalid_dimensions = sorted(set(dimensions) - DIMENSION_SET)
    if invalid_dimensions:
        errors.append(
            f"{field}.dimensions has invalid values: {', '.join(invalid_dimensions)}"
        )

    confidence = task.get("dimension_confidence")
    if confidence_required and confidence is None:
        errors.append(f"{field}.dimension_confidence is required")
        return
    if confidence is None:
        return
    if not _is_mapping(confidence, f"{field}.dimension_confidence", errors):
        return

    expected = set(dimensions) & DIMENSION_SET
    actual = set(confidence)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(
                f"{field}.dimension_confidence is missing: {', '.join(missing)}"
            )
        if extra:
            errors.append(
                f"{field}.dimension_confidence has unexpected keys: {', '.join(extra)}"
            )

    for dimension, value in confidence.items():
        if not _is_probability(value):
            errors.append(
                f"{field}.dimension_confidence.{dimension} must be a number from 0 through 1"
            )


def _validate_task(
    task: Mapping[str, Any],
    field: str,
    errors: List[str],
    valid_task_ids: set[str],
    confidence_required: bool,
) -> None:
    if not _is_mapping(task, field, errors):
        return

    _check_exact_keys(task, TASK_KEYS, field, errors)

    task_id = task.get("subtask_id")
    if not isinstance(task_id, str) or not task_id.strip():
        errors.append(f"{field}.subtask_id must be a non-empty string")
    elif task_id not in valid_task_ids:
        errors.append(f"{field}.subtask_id is not allowed: {task_id}")

    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append(f"{field}.instruction must be a non-empty string")

    _validate_dimensions(task, field, errors, confidence_required)

    if task.get("output_type") not in OUTPUT_TYPES:
        errors.append(f"{field}.output_type must be 'text'")

    depends_on = task.get("depends_on")
    if not isinstance(depends_on, list):
        errors.append(f"{field}.depends_on must be an array")
    else:
        if any(not isinstance(item, str) for item in depends_on):
            errors.append(f"{field}.depends_on must contain only strings")
        else:
            if len(set(depends_on)) != len(depends_on):
                errors.append(f"{field}.depends_on must not contain duplicates")
            if task_id in depends_on:
                errors.append(f"{field}.depends_on must not reference itself")
            unknown = set(depends_on) - valid_task_ids
            if unknown:
                errors.append(
                    f"{field}.depends_on references unknown tasks: {', '.join(sorted(unknown))}"
                )

    input_refs = task.get("input_refs")
    if not isinstance(input_refs, list):
        errors.append(f"{field}.input_refs must be an array")
    else:
        if any(not isinstance(item, str) or not item.strip() for item in input_refs):
            errors.append(f"{field}.input_refs must contain only non-empty strings")
        else:
            if len(set(input_refs)) != len(input_refs):
                errors.append(f"{field}.input_refs must not contain duplicates")
            for ref in input_refs:
                if ref == "original_prompt":
                    continue
                match = re.fullmatch(r"(T\d+)\.output", ref)
                if match is None:
                    errors.append(f"{field}.input_refs has invalid reference: {ref}")
                    continue
                upstream_id = match.group(1)
                if upstream_id not in valid_task_ids:
                    errors.append(f"{field}.input_refs references unknown task: {upstream_id}")
                elif upstream_id not in depends_on:
                    errors.append(
                        f"{field}.input_refs references {upstream_id}.output "
                        "but that task is not in depends_on"
                    )


def _has_cycle(subtasks: Sequence[Mapping[str, Any]]) -> bool:
    graph = {
        task["subtask_id"]: set(task.get("depends_on", []))
        for task in subtasks
        if isinstance(task.get("subtask_id"), str)
        and isinstance(task.get("depends_on"), list)
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dependency in graph.get(node, set()):
            if dependency in graph and visit(dependency):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in list(graph))


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Validate a TaskPlan independently of its originating RunRequest."""

    errors: List[str] = []
    if not _is_mapping(plan, "TaskPlan", errors):
        raise TaskPlanValidationError(errors)

    _check_exact_keys(plan, TASK_PLAN_KEYS, "TaskPlan", errors)

    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append("TaskPlan.schema_version must be '1.0'")

    request_id = plan.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        errors.append("TaskPlan.request_id must be a non-empty string")

    subtasks = plan.get("subtasks")
    if not isinstance(subtasks, list):
        errors.append("TaskPlan.subtasks must be an array")
        subtasks = []
    else:
        if len(subtasks) == 1:
            errors.append("TaskPlan.subtasks must be empty for Direct or contain 2-4 tasks")
        if len(subtasks) > 4:
            errors.append("TaskPlan.subtasks must contain at most 4 tasks")

    expected_ids = {f"T{index}" for index in range(1, len(subtasks) + 1)}
    actual_ids: List[str] = []
    for index, task in enumerate(subtasks, start=1):
        if not _is_mapping(task, f"TaskPlan.subtasks[{index}]", errors):
            continue
        task_id = task.get("subtask_id")
        if isinstance(task_id, str):
            actual_ids.append(task_id)
        _validate_task(
            task,
            f"TaskPlan.subtasks[{index}]",
            errors,
            expected_ids,
            confidence_required=True,
        )

    if len(set(actual_ids)) != len(actual_ids):
        errors.append("TaskPlan.subtasks has duplicate subtask_id values")
    if set(actual_ids) != expected_ids:
        errors.append("TaskPlan.subtasks IDs must be T1, T2, ... without gaps")

    direct_task = plan.get("direct_task")
    if not _is_mapping(direct_task, "TaskPlan.direct_task", errors):
        direct_task = {}
    else:
        direct_id = direct_task.get("subtask_id")
        if direct_id != "DIRECT":
            errors.append("TaskPlan.direct_task.subtask_id must be 'DIRECT'")
        # The canonical interface example omits confidence for the direct task,
        # so it is optional here. Our generated plans always provide it.
        _validate_task(
            direct_task,
            "TaskPlan.direct_task",
            errors,
            {"DIRECT"} | expected_ids,
            confidence_required=False,
        )
        if direct_task.get("depends_on") != []:
            errors.append("TaskPlan.direct_task.depends_on must be empty")
        if direct_task.get("input_refs") != ["original_prompt"]:
            errors.append("TaskPlan.direct_task.input_refs must be ['original_prompt']")

    if isinstance(subtasks, list) and _has_cycle(subtasks):
        errors.append("TaskPlan.subtasks dependency graph must be acyclic")

    if errors:
        raise TaskPlanValidationError(errors)


def validate_plan_for_request(plan: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    """Validate TaskPlan semantics that depend on the originating request."""

    validate_run_request(request)
    validate_plan(plan)

    errors: List[str] = []
    if plan.get("request_id") != request.get("request_id"):
        errors.append("TaskPlan.request_id must match RunRequest.request_id")

    direct_task = plan.get("direct_task")
    if isinstance(direct_task, dict):
        if direct_task.get("instruction") != request.get("original_prompt"):
            errors.append("TaskPlan.direct_task.instruction must preserve original_prompt")

    constraints = request.get("constraints")
    subtasks = plan.get("subtasks")
    if isinstance(constraints, dict) and isinstance(subtasks, list):
        allow_split = constraints.get("allow_split")
        max_subtasks = constraints.get("max_subtasks")
        if allow_split is False and subtasks:
            errors.append("allow_split=false requires TaskPlan.subtasks to be empty")
        if isinstance(max_subtasks, int) and not isinstance(max_subtasks, bool):
            if len(subtasks) > max_subtasks:
                errors.append("TaskPlan.subtasks exceeds constraints.max_subtasks")

    if errors:
        raise TaskPlanValidationError(errors)
