"""DAG construction for split candidate instructions."""

from __future__ import annotations

from typing import List, Sequence
import re


_ORDINALS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _explicit_dependencies(instruction: str, current: int) -> List[str]:
    """Find explicit references such as ``第一步`` or ``first step``."""
    found: List[int] = []
    for value in re.findall(r"第([一二三四五六七八九十]|\d+)步", instruction):
        number = int(value) if value.isdigit() else _ORDINALS[value]
        if 1 <= number < current and number not in found:
            found.append(number)
    for value in re.findall(
        r"\b(first|second|third|fourth|fifth)\s+step\b", instruction, flags=re.I
    ):
        number = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}[value.lower()]
        if number < current and number not in found:
            found.append(number)
    return [f"T{number}" for number in found]


def build_dag(instructions: Sequence[str], parallel: bool) -> List[dict[str, object]]:
    """Build sequential or parallel task relations.

    Sequential tasks form a chain so every downstream task can use the previous
    result. Parallel tasks have no dependencies and all read original_prompt.
    """

    count = len(instructions)
    tasks: List[dict[str, object]] = []
    for index, instruction in enumerate(instructions, start=1):
        task_id = f"T{index}"
        explicit = _explicit_dependencies(instruction, index)
        if parallel:
            dependencies = []
        else:
            dependencies = [] if index == 1 else [f"T{index - 1}"]
        for dependency in explicit:
            if dependency not in dependencies:
                dependencies.append(dependency)
        input_refs = ["original_prompt"] + [f"{dependency}.output" for dependency in dependencies]

        tasks.append(
            {
                "subtask_id": task_id,
                "instruction": instruction,
                "depends_on": dependencies,
                "input_refs": input_refs,
            }
        )
    return tasks
