"""Main Task Decomposer entrypoint for LY."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Mapping, Protocol

from .classifier import ClassificationProtocol
from .dag_builder import build_dag
from .schema import SCHEMA_VERSION
from .splitter import SplitResult, generate_split_candidates, extract_global_constraints
from .validator import TaskPlanValidationError, validate_plan_for_request, validate_run_request


class SplitterProtocol(Protocol):
    def generate_split_candidates(self, prompt: str, max_subtasks: int) -> SplitResult:
        """Return split candidates, or an empty result for Direct."""


class TaskDecomposer:
    """Convert a valid RunRequest into a valid TaskPlan.

    Budget and quality preferences are intentionally ignored here. They are
    optimizer inputs; LY must not decide Direct/Split based on cost.
    """

    def __init__(
        self,
        classifier: ClassificationProtocol,
        splitter: SplitterProtocol | None = None,
        *,
        enable_explicit_mixed: bool = False,
        enable_explicit_parallel_objects: bool = False,
    ) -> None:
        self.classifier = classifier
        self.splitter = splitter
        self.enable_explicit_mixed = enable_explicit_mixed
        self.enable_explicit_parallel_objects = enable_explicit_parallel_objects
        self.last_plan_strategy = "original"
        self.last_fallback_reason: str | None = None

    # Opt-in, conservative syntax only. These changes do not alter classifier
    # weights, thresholds or the raw request -> classifier input contract.
    _PARALLEL_OBJECT_RE = re.compile(
        r"^\s*(?:同时|分别)?\s*(?P<action>总结|分析|翻译|提取|检查|解释)"
        r"(?P<first>[^，。；;和与]+?)(?:和|与)(?P<second>[^，。；;]+?)"
        r"，分别(?P<tail>输出|给出|列出|生成)[^。！!?？]+[。！!?？]?\s*$"
    )
    _MIXED_RE = re.compile(
        r"^\s*(?:先|首先|第一步)\s*(?P<first>.+?)\s*[,，;；]\s*"
        r"(?:然后|接着|其次|第二步)\s*(?:同时|分别)\s*(?P<middle>.+?)\s*[,，;；]\s*"
        r"(?:最后|最终|第三步)\s*(?P<last>.+?)\s*[。.!！]?\s*$", re.S
    )
    _MIXED_ACTIONS = ("翻译", "计算", "分析", "提取", "识别", "总结", "检查", "生成", "写", "绘制", "整理", "查询", "统计", "校验", "转写", "概括")

    def _explicit_parallel_instructions(self, prompt: str) -> list[str] | None:
        match = self._PARALLEL_OBJECT_RE.fullmatch(prompt)
        if not match:
            return None
        first, second = match.group("first").strip(), match.group("second").strip()
        if not first or not second or first == second:
            return None
        tail = re.search(r"，分别(?P<tail>.+?)[。！!?？]?\s*$", prompt)
        if not tail:
            return None
        action = match.group("action")
        return [f"{action}{first}，{tail.group('tail').strip()}", f"{action}{second}，{tail.group('tail').strip()}"]

    def _explicit_mixed_instructions(self, prompt: str) -> list[str] | None:
        match = self._MIXED_RE.fullmatch(prompt)
        if not match:
            return None
        branches = re.split(r"\s*(?:和|与|以及|并且)\s*", match.group("middle"), maxsplit=1)
        if len(branches) != 2 or not all(any(a in b for a in self._MIXED_ACTIONS) for b in branches):
            return None
        chunks = [match.group("first").strip(), branches[0].strip(), branches[1].strip(), match.group("last").strip()]
        if not all(chunks):
            return None
        constraints = extract_global_constraints(prompt)
        if constraints:
            chunks[-1] += "（全局约束：" + "；".join(constraints) + "）"
        return chunks

    def _explicit_plan(self, request: Mapping[str, Any], chunks: list[str],
                       kind: str) -> Dict[str, Any]:
        if kind == "parallel":
            deps = [[], []]
        else:
            deps = [[], ["T1"], ["T1"], ["T2", "T3"]]
        subtasks=[]
        for index, (instruction, parents) in enumerate(zip(chunks, deps), start=1):
            task={"subtask_id": f"T{index}", "instruction": instruction,
                  "depends_on": parents.copy(),
                  "input_refs": ["original_prompt"] + [f"{p}.output" for p in parents]}
            subtasks.append(self._classify_task(task))
        plan=self._direct_plan(request)
        plan["subtasks"]=subtasks
        validate_plan_for_request(plan, request)
        return plan

    def _classify_task(
        self,
        base_task: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self.classifier.classify(base_task["instruction"])
        dimensions = result.get("dimensions", [])
        confidence = result.get("dimension_confidence", {})
        return {
            **base_task,
            "dimensions": list(dimensions),
            "dimension_confidence": dict(confidence),
            "output_type": "text",
        }

    def _direct_plan(
        self,
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        direct_task = self._classify_task(
            {
                "subtask_id": "DIRECT",
                "instruction": request["original_prompt"],
                "depends_on": [],
                "input_refs": ["original_prompt"],
            }
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request["request_id"],
            "subtasks": [],
            "direct_task": direct_task,
        }

    def analyze(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        self.last_fallback_reason = None
        self.last_plan_strategy = "original"
        validate_run_request(request)

        constraints = request["constraints"]
        # Do not enable new grammar by default: older V5 TaskPlans retain
        # their original behavior unless the integration explicitly opts in.
        if constraints["allow_split"] and constraints["max_subtasks"] >= 2:
            if self.enable_explicit_parallel_objects:
                chunks = self._explicit_parallel_instructions(request["original_prompt"])
                if chunks:
                    try:
                        result = self._explicit_plan(request, chunks, "parallel")
                    except TaskPlanValidationError:
                        self.last_fallback_reason = "parallel_candidate_invalid"
                    else:
                        self.last_plan_strategy = "explicit_parallel_objects"
                        return result
            if self.enable_explicit_mixed and constraints["max_subtasks"] >= 4:
                chunks = self._explicit_mixed_instructions(request["original_prompt"])
                if chunks:
                    try:
                        result = self._explicit_plan(request, chunks, "mixed")
                    except TaskPlanValidationError:
                        self.last_fallback_reason = "mixed_candidate_invalid"
                    else:
                        self.last_plan_strategy = "explicit_mixed_fork_join"
                        return result
        if not constraints["allow_split"] or constraints["max_subtasks"] < 2:
            self.last_fallback_reason = "split_disabled"
            plan = self._direct_plan(request)
            validate_plan_for_request(plan, request)
            return plan

        if self.splitter is None:
            candidates = generate_split_candidates(
                request["original_prompt"],
                constraints["max_subtasks"],
            )
        else:
            candidates = self.splitter.generate_split_candidates(
                request["original_prompt"],
                constraints["max_subtasks"],
            )
        if len(candidates.instructions) < 2:
            self.last_fallback_reason = "no_reliable_boundary"
            plan = self._direct_plan(request)
            validate_plan_for_request(plan, request)
            return plan

        relation_tasks = build_dag(candidates.instructions, candidates.parallel)
        subtasks: List[Dict[str, Any]] = []
        for task in relation_tasks:
            subtasks.append(self._classify_task(task))

        direct_task = self._classify_task(
            {
                "subtask_id": "DIRECT",
                "instruction": request["original_prompt"],
                "depends_on": [],
                "input_refs": ["original_prompt"],
            }
        )
        split_plan = {
            "schema_version": SCHEMA_VERSION,
            "request_id": request["request_id"],
            "subtasks": subtasks,
            "direct_task": direct_task,
        }

        try:
            validate_plan_for_request(split_plan, request)
        except TaskPlanValidationError:
            # Invalid candidate plans are never emitted. Conservative Direct
            # fallback is the only safe user-visible behavior in v0.1.
            self.last_fallback_reason = "candidate_failed_validation"
            plan = self._direct_plan(request)
            validate_plan_for_request(plan, request)
            return plan

        return copy.deepcopy(split_plan)


def analyze(request: Mapping[str, Any], classifier: ClassificationProtocol) -> Dict[str, Any]:
    """Convenience function matching the team's analyze(request) signature."""

    return TaskDecomposer(classifier).analyze(request)
