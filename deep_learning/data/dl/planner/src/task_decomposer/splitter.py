"""Conservative rule-based split candidate generation.

The splitter is deliberately cue-driven rather than generative. It only
proposes boundaries at explicit sequencing or parallel markers, preserves all
segments, and lets the decomposer fall back to Direct when in doubt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class SplitResult:
    instructions: List[str]
    parallel: bool
    global_constraints: List[str]


def _looks_chinese(text: str) -> bool:
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_words = len(re.findall(r"[A-Za-z]+", text))
    return cjk >= latin_words


def extract_global_constraints(prompt: str) -> List[str]:
    """Extract explicit constraints without turning them into subtasks."""

    constraints: List[str] = []
    patterns = [
        r"(?:用|使用|以|请用)[中英法日德西]文",
        r"(?:answer|respond|write|explain|summarize)[^.;\n]{0,20}?in (?:Chinese|English|French|Japanese|German|Spanish)",
        r"(?:JSON|Markdown|LaTeX|CSV)",
        r"(?:Markdown|LaTeX|CSV)\s*(?:格式|表格|输出)?",
        r"(?:表格|列表|代码块|分点|要点)",
        r"(?:不超过|最多|至少|恰好)[^,，。;；\n]{0,12}(?:字|词|条|个|段)",
        r"\b(?:no more than|at least|exactly)\s+\d+\s+(?:words|characters|items|bullets)\b",
        r"\b\d+\s+(?:words|characters|items|bullets)\b",
        r"(?:[一二三四五六七八九十两]|\d+)\s*条(?:建议|理由|示例|要点)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value and value not in constraints:
                constraints.append(value)
    return constraints


_CHINESE_CUE = (
    r"(?:首先|然后|接着|其次|最后|再|同时|分别|另外|此外|并且|第[一二三四五六七八九十\d]+步)"
)
_ENGLISH_CUE = (
    r"(?:first|second|third|fourth|then|next|after that|afterwards|"
    r"finally|lastly|meanwhile|at the same time|also|additionally)"
)
_BOUNDARY = r"(?:^|[，。；;!？?\n])\s*"


def _constraints_for_each_task(
    prompt: str,
    segments: Sequence[str],
    constraints: Sequence[str],
) -> List[str]:
    """Return only requirements that are safe to copy to every segment.

    A requirement occurring inside one segment is scoped to that segment. A
    trailing request such as ``请用中文回答`` is global because it explicitly
    addresses the answer, while ``然后翻译成中文`` is local to its step.
    """
    global_markers = re.compile(
        r"(?:全局[^。；;]*|所有(?:步骤|任务|结果)[^。；;]*|"
        r"每(?:一步|个步骤|个任务)[^。；;]*|整个(?:任务|回答)[^。；;]*|"
        r"(?:最后|最终)(?:回答|输出)[^。；;]*|"
        r"请(?:用|使用|以)[^。；;]*(?:回答|输出)[^。；;]*|"
        r"(?:answer|respond|write|return|output)[^.;\n]{0,40}"
        r"\b(?:in|as)\s+(?:Chinese|English|French|Japanese|German|Spanish|JSON|Markdown)\b)",
        flags=re.IGNORECASE,
    )
    global_matches = [match.group(0) for match in global_markers.finditer(prompt)]
    copy: List[str] = []
    for constraint in constraints:
        in_segment = any(constraint in segment for segment in segments)
        explicitly_global = any(constraint in marker for marker in global_matches)
        if explicitly_global or not in_segment:
            copy.append(constraint)
    return copy


def _split_segments(prompt: str) -> List[str]:
    cue_pattern = re.compile(
        rf"{_BOUNDARY}(?:{_CHINESE_CUE}|{_ENGLISH_CUE}(?![A-Za-z]))",
        flags=re.IGNORECASE,
    )

    matches = list(cue_pattern.finditer(prompt))
    if not matches:
        return []

    segments: List[str] = []
    first_start = matches[0].start()
    if first_start > 0:
        first_segment = prompt[:first_start].strip()
        first_segment = first_segment.strip("，。；;!？? \t\r\n")
        if first_segment:
            segments.append(first_segment)

    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        segment = prompt[match.end():next_start].strip()
        segment = segment.strip("，。；;!？? \t\r\n")
        if segment:
            segments.append(segment)

    return segments


def _merge_to_limit(
    instructions: Sequence[str],
    limit: int,
    chinese: bool,
) -> List[str]:
    if len(instructions) <= limit:
        return list(instructions)

    merged = list(instructions)
    separator = "；" if chinese else "; "
    while len(merged) > limit:
        merged[-2] = separator.join([merged[-2], merged[-1]])
        merged.pop()
    return merged


def generate_split_candidates(prompt: str, max_subtasks: int) -> SplitResult:
    """Generate candidate instructions, or return an empty result for Direct."""

    if not isinstance(prompt, str) or not prompt.strip():
        return SplitResult([], False, [])
    if not isinstance(max_subtasks, int) or isinstance(max_subtasks, bool) or max_subtasks < 2:
        return SplitResult([], False, [])

    chinese = _looks_chinese(prompt)
    segments = _split_segments(prompt)
    if len(segments) < 2:
        return SplitResult([], False, [])

    parallel_markers = re.search(r"(?:同时|分别|另外|此外|meanwhile|at the same time)", prompt, re.I)
    sequential_markers = re.search(r"(?:首先|然后|接着|其次|最后|再|第[一二三四五六七八九十\d]+步|then|next|after that|finally|lastly)", prompt, re.I)
    parallel = bool(parallel_markers and not sequential_markers)

    merged = _merge_to_limit(segments, max_subtasks, chinese)
    if len(merged) < 2:
        return SplitResult([], False, [])

    constraints = extract_global_constraints(prompt)
    task_constraints = _constraints_for_each_task(prompt, merged, constraints)
    if task_constraints:
        prefix = "全局约束：" if chinese else "Global constraints: "
        suffix = prefix + ("；" if chinese else "; ").join(task_constraints)
        if chinese:
            suffix = "（" + suffix + "）"
        else:
            suffix = " (" + suffix + ")"
        merged = [instruction + suffix for instruction in merged]

    return SplitResult(merged, parallel, task_constraints)
