"""Reviewed Gold schema and deterministic split-leakage checks.

Approval fields are attestations supplied by the data owner, not a substitute
for a human review. Exact normalization cannot establish historical independence.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import unicodedata


LABELS = (
    "agents", "coding", "reasoning", "knowledge", "math", "multilingual",
    "multimodal", "instruction_following",
)
SUPPORTED_POLICY_VERSION = "0.5.1"


def normalize_text(text: str) -> str:
    """Normalize text only for leakage fingerprints, never for source quoting."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_prompt_sha256(text: str) -> str:
    """SHA-256 of UTF-8 NFKC/casefold/whitespace-normalized prompt text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def normalized_task_context_sha256(instruction: str, context: str = "") -> str:
    """Hash the canonical JSON pair of normalized instruction and context."""
    encoded = json.dumps([normalize_text(instruction), normalize_text(context)],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string(record: Mapping[str, Any], field: str, location: str, *, empty=False) -> str:
    value = record.get(field)
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{location}.{field} must be a {'possibly empty ' if empty else 'nonempty '}string")
    return value


def _dimensions(record: Mapping[str, Any], field: str, location: str) -> None:
    labels = record.get(field)
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise ValueError(f"{location}.{field} must be an explicit list of dimensions")
    if len(set(labels)) != len(labels) or not set(labels).issubset(LABELS):
        raise ValueError(f"{location}.{field} contains duplicate or unknown dimensions")


def validate_records(records: Iterable[Mapping[str, Any]], allow_test: bool = False) -> list[dict]:
    """Return independent copies of complete reviewed records or raise ValueError.

    Test records can be validated for evaluation only via ``allow_test=True``;
    training always calls this with the default. A group may repeat within one
    split. Request IDs must be globally unique. Empty dimension lists are valid.
    """
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of record objects")
    try:
        validated = copy.deepcopy(list(records))
    except TypeError as exc:
        raise ValueError("records must be an iterable of record objects") from exc
    if not validated:
        raise ValueError("records must not be empty")
    request_ids: set[str] = set()
    policies: set[str] = set()
    seen: dict[str, dict[Any, str]] = {kind: {} for kind in ("group", "prompt", "task", "boundary")}

    def isolate(kind: str, key: Any, split: str) -> None:
        previous = seen[kind].setdefault(key, split)
        if previous != split:
            raise ValueError(f"{kind} leakage across splits: {previous} and {split}")

    for index, record in enumerate(validated):
        location = f"record[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{location} must be an object")
        for field in ("request_id", "group_id", "split", "review_status", "review_source",
                      "reviewer", "label_policy_version", "original_prompt"):
            _string(record, field, location)
        request_id = record["request_id"]
        if request_id in request_ids:
            raise ValueError(f"duplicate request_id: {request_id}")
        request_ids.add(request_id)
        split = record["split"]
        if split not in ({"train", "validation", "test"} if allow_test else {"train", "validation"}):
            raise ValueError(f"{location}.split {split!r} is not allowed; test records cannot enter training")
        if record["review_status"] != "approved":
            raise ValueError(f"{location}.review_status must be approved")
        if record["review_source"] != "human":
            raise ValueError(f"{location}.review_source must be human")
        if record["label_policy_version"] != SUPPORTED_POLICY_VERSION:
            raise ValueError(f"unsupported label policy version; expected {SUPPORTED_POLICY_VERSION}")
        policies.add(record["label_policy_version"])
        if len(policies) != 1:
            raise ValueError("inconsistent label policy versions")
        if type(record.get("should_split")) is not bool:
            raise ValueError(f"{location}.should_split must be a boolean")
        _dimensions(record, "direct_dimensions", location)
        tasks = record.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"{location}.tasks must be a nonempty list")
        isolate("group", record["group_id"], split)
        isolate("prompt", normalize_text(record["original_prompt"]), split)
        # Compare direct requests and task inputs in the same namespace as well.
        isolate("task", (normalize_text(record["original_prompt"]), ""), split)
        for task_index, task in enumerate(tasks):
            task_location = f"{location}.tasks[{task_index}]"
            if not isinstance(task, dict):
                raise ValueError(f"{task_location} must be an object")
            instruction = _string(task, "instruction", task_location)
            context = _string(task, "context", task_location, empty=True)
            _dimensions(task, "dimensions", task_location)
            isolate("task", (normalize_text(instruction), normalize_text(context)), split)
        boundaries = record.get("boundaries")
        if not isinstance(boundaries, list):
            raise ValueError(f"{location}.boundaries must be an explicit list")
        for boundary_index, boundary in enumerate(boundaries):
            boundary_location = f"{location}.boundaries[{boundary_index}]"
            if not isinstance(boundary, dict):
                raise ValueError(f"{boundary_location} must be an object")
            left = _string(boundary, "left", boundary_location)
            right = _string(boundary, "right", boundary_location)
            if type(boundary.get("is_boundary")) is not bool:
                raise ValueError(f"{boundary_location}.is_boundary must be a boolean")
            prompt = record["original_prompt"]
            left_position = prompt.find(left)
            if left_position < 0 or prompt.find(right, left_position + len(left)) < 0:
                raise ValueError(f"{boundary_location} must contain exact source quotations in left-to-right order")
            isolate("boundary", (normalize_text(left), normalize_text(right)), split)
    return validated


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSON objects, reporting physical line numbers for malformed input."""
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected object on line {line_number}")
            records.append(record)
    return records
