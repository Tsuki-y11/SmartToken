"""Frozen JSON V1.0 constants for RunRequest and TaskPlan."""

from __future__ import annotations

from typing import Any, Dict, List

SCHEMA_VERSION = "1.0"

# The order is fixed by the team interface document. A task may select any
# subset; these dimensions are never treated as subtasks.
DIMENSIONS = [
    "agents",
    "coding",
    "reasoning",
    "knowledge",
    "math",
    "multilingual",
    "multimodal",
    "instruction_following",
]
DIMENSION_SET = frozenset(DIMENSIONS)

QUALITY_PRIORITIES = frozenset({"cost_first", "balanced", "quality_first"})
OUTPUT_TYPES = frozenset({"text"})

RunRequest = Dict[str, Any]
Task = Dict[str, Any]
TaskPlan = Dict[str, Any]
Dimensions = List[str]

RUN_REQUEST_KEYS = frozenset(
    {"schema_version", "request_id", "original_prompt", "constraints", "generation"}
)
CONSTRAINT_KEYS = frozenset(
    {"budget_usd", "quality_priority", "max_subtasks", "allow_split"}
)
GENERATION_KEYS = frozenset({"temperature", "max_output_tokens"})
TASK_PLAN_KEYS = frozenset({"schema_version", "request_id", "subtasks", "direct_task"})
TASK_KEYS = frozenset(
    {
        "subtask_id",
        "instruction",
        "dimensions",
        "dimension_confidence",
        "depends_on",
        "input_refs",
        "output_type",
    }
)
