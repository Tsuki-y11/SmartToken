"""ML-backed splitter using Data B should_split and boundary models."""

from __future__ import annotations
from submission_compat import load_trusted_joblib

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import joblib

from .splitter import (
    SplitResult,
    _constraints_for_each_task,
    _looks_chinese,
    extract_global_constraints,
)


@dataclass(frozen=True)
class _BoundaryMatch:
    start: int
    end: int
    cue: str
    candidate_text: str


class MlSplitter:
    """Use trained Data B models to decide split and boundary validity."""

    def __init__(
        self,
        should_split_model: str | Path,
        boundary_model: str | Path,
    ) -> None:
        self.should_split_bundle = load_trusted_joblib(should_split_model)
        self.boundary_bundle = load_trusted_joblib(boundary_model)
        if not isinstance(self.should_split_bundle, dict):
            raise ValueError("should_split model bundle must be a dictionary")
        if not isinstance(self.boundary_bundle, dict):
            raise ValueError("boundary model bundle must be a dictionary")

    def _boundary_matches(self, prompt: str) -> List[_BoundaryMatch]:
        # Keep this cue set aligned with the rule splitter and Data B builder.
        cue_pattern = re.compile(
            r"首先|然后|接着|其次|最后|再|同时|分别|另外|此外|并且|"
            r"第[一二三四五六七八九十\d]+步|first|second|third|fourth|then|next|"
            r"after that|finally|lastly|meanwhile|at the same time|also|additionally",
            re.IGNORECASE,
        )
        matches: List[_BoundaryMatch] = []
        for match in cue_pattern.finditer(prompt):
            start = max(0, match.start() - 40)
            end = min(len(prompt), match.end() + 40)
            matches.append(
                _BoundaryMatch(
                    start=match.start(),
                    end=match.end(),
                    cue=match.group(0),
                    candidate_text=prompt[start:end],
                )
            )
        return matches

    def _should_split(self, prompt: str) -> bool:
        vectorizer = self.should_split_bundle["vectorizer"]
        model = self.should_split_bundle["model"]
        threshold = float(self.should_split_bundle["threshold"])
        probability = model.predict_proba(vectorizer.transform([prompt]))[0, 1]
        return bool(probability >= threshold)

    def _should_split_probability(self, prompt: str) -> float:
        model = self.should_split_bundle["model"]
        if self.should_split_bundle.get("feature") == "structural":
            return float(model.predict_proba([prompt])[0, 1])
        vectorizer = self.should_split_bundle["vectorizer"]
        return float(model.predict_proba(vectorizer.transform([prompt]))[0, 1])

    def _selected_boundaries(self, prompt: str) -> List[_BoundaryMatch]:
        vectorizer = self.boundary_bundle["vectorizer"]
        model = self.boundary_bundle["model"]
        threshold = float(self.boundary_bundle["threshold"])
        candidates = self._boundary_matches(prompt)
        if not candidates:
            return []
        texts = [candidate.candidate_text for candidate in candidates]
        probabilities = model.predict_proba(vectorizer.transform(texts))[:, 1]
        return [
            candidate
            for candidate, probability in zip(candidates, probabilities)
            if probability >= threshold
        ]

    @staticmethod
    def _instructions_from_boundaries(
        prompt: str,
        boundaries: Sequence[_BoundaryMatch],
    ) -> List[str]:
        instructions: List[str] = []
        cursor = 0
        for boundary in boundaries:
            instruction = prompt[cursor : boundary.start].strip()
            instruction = instruction.strip("，。；;!？? \t\r\n")
            if instruction:
                instructions.append(instruction)
            cursor = boundary.end
        tail = prompt[cursor:].strip().strip("，。；;!？? \t\r\n")
        if tail:
            instructions.append(tail)
        return instructions

    def generate_split_candidates(self, prompt: str, max_subtasks: int) -> SplitResult:
        if not isinstance(prompt, str) or not prompt.strip():
            return SplitResult([], False, [])
        if not isinstance(max_subtasks, int) or isinstance(max_subtasks, bool) or max_subtasks < 2:
            return SplitResult([], False, [])
        split_probability = self._should_split_probability(prompt)
        split_threshold = float(self.should_split_bundle["threshold"])
        selected = self._selected_boundaries(prompt)

        # Keep the primary decision with should_split. When it is only
        # marginally below the threshold, use boundary evidence from the
        # second ML model rather than silently falling back to a rule.
        if split_probability < split_threshold:
            uncertainty_margin = 0.10
            in_uncertainty_band = (
                split_threshold - uncertainty_margin
                <= split_probability
                < split_threshold
            )
            if not (in_uncertainty_band and selected):
                return SplitResult([], False, [])

        if not selected:
            return SplitResult([], False, [])

        instructions = self._instructions_from_boundaries(prompt, selected)
        if len(instructions) < 2:
            return SplitResult([], False, [])

        parallel_cues = {"同时", "分别", "meanwhile", "at the same time"}
        parallel = all(boundary.cue.lower() in parallel_cues for boundary in selected)

        if len(instructions) > max_subtasks:
            chinese = _looks_chinese(prompt)
            separator = "；" if chinese else "; "
            while len(instructions) > max_subtasks:
                instructions[-2] = separator.join([instructions[-2], instructions[-1]])
                instructions.pop()

        constraints = extract_global_constraints(prompt)
        task_constraints = _constraints_for_each_task(prompt, instructions, constraints)
        if task_constraints:
            chinese = _looks_chinese(prompt)
            prefix = "全局约束：" if chinese else "Global constraints: "
            suffix = prefix + ("；" if chinese else "; ").join(task_constraints)
            if chinese:
                suffix = "（" + suffix + "）"
            else:
                suffix = " (" + suffix + ")"
            instructions = [instruction + suffix for instruction in instructions]

        return SplitResult(instructions, parallel, constraints)
