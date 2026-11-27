"""Deterministic Task Decomposer implementation for SmartToken."""

from .validator import TaskPlanValidationError, validate_plan, validate_plan_for_request, validate_run_request


def __getattr__(name):
    if name in {"TaskDecomposer", "analyze"}:
        from . import decomposer
        return getattr(decomposer, name)
    raise AttributeError(name)

__all__ = [
    "TaskDecomposer",
    "analyze",
    "TaskPlanValidationError",
    "validate_plan",
    "validate_plan_for_request",
    "validate_run_request",
]

