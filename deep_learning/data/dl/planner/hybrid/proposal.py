"""Conservative structural validation, separate from semantic quality evaluation."""
from __future__ import annotations

import math
import re

from deep_learning.planner.src.task_decomposer.schema import DIMENSIONS


def candidate_clauses(text: str) -> list[str]:
    """Broad candidates: learned boundaries decide which cuts to retain.

    Preserve inline/fenced code, quoted text, bracketed expressions and numeric
    punctuation. This is not a semantic sentence parser; abbreviations remain
    candidates and must be judged by the trained boundary model.
    """
    chunks, start, index = [], 0, 0
    quote, fence, brackets = None, None, []
    pairs = {"(": ")", "[": "]", "{": "}", "（": "）", "【": "】"}
    while index < len(text):
        char = text[index]
        if fence:
            if text.startswith(fence, index):
                index += len(fence)
                fence = None
                continue
        elif quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
        elif text.startswith("```", index):
            fence = "```"
            index += 3
            continue
        elif char == "`":
            fence = "`"
        elif char in ('"', "“", "‘") or (char == "'" and (index == 0 or not text[index - 1].isalnum())):
            quote = {"“": "”", "‘": "’"}.get(char, char)
        elif char in pairs:
            brackets.append(pairs[char])
        elif brackets and char == brackets[-1]:
            brackets.pop()
        elif not brackets and char in ",，;；。.!！?？\n":
            numeric = (char in ".," and 0 < index < len(text) - 1
                       and text[index - 1].isdigit() and text[index + 1].isdigit())
            if not numeric:
                part = text[start:index].strip()
                if part:
                    chunks.append(part)
                start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks


_ORDINALS = {"一": 1, "二": 2, "三": 3, "四": 4, "first": 1, "second": 2, "third": 3, "fourth": 4}


def step_references(text: str) -> set[str]:
    values = re.findall(r"第([一二三四]|\d+)步", text)
    values += re.findall(r"\b(first|second|third|fourth)\s+step\b", text, flags=re.I)
    refs = {f"T{int(v) if v.isdigit() else _ORDINALS[v.lower()]}" for v in values}
    refs.update(re.findall(r"\b(T[1-4])(?:\.output)?\b", text))
    return refs


def excluded_references(text: str) -> set[str]:
    snippets = re.findall(r"(?:不要|不得|禁止|不)(?:使用|读取|参考|依据|包含|传入)[^。；;，,\n]{0,25}", text)
    snippets += re.findall(r"\b(?:do not|don't|never|without)\s+(?:use|using|read|reading|include|including|reference)[^.;,\n]{0,45}", text, flags=re.I)
    return set().union(*(step_references(s) for s in snippets)) if snippets else set()


def _required_data_references(text):
    required = set()
    for clause in re.split(r"[，,。；;\n]", text):
        targets = re.findall(r"\bT\d+\.output\b|\b(?:first|second|third|fourth)\s+step(?:['’]s)?\s+(?:outputs?|results?)\b|第[一二三四\d]+步(?:的)?(?:结果|输出)", clause, re.I)
        for match in re.finditer(r"读取|使用|引用|翻译|解释|总结|比较|根据|依据|\b(?:read|use|translate|explain|summari[sz]e|compare|based on)\b", clause, re.I):
            target = re.split(r"\b(?:after|before|once|when|then)\b|完成后|之后|之前", clause[match.end():], maxsplit=1, flags=re.I)[0]
            targets.append(target)
        for target in targets:
            required.update(step_references(target) - excluded_references(clause))
    return required


def _check_input_scope(task, quotes, *, split):
    text = task["instruction"] + "\n" + "\n".join(quotes)
    inputs = {ref.removesuffix(".output") for ref in task["input_refs"] if ref.endswith(".output")}
    if excluded_references(text) & inputs:
        raise ValueError("excluded_input: a prohibited step output is included")
    clauses = re.split(r"[，,。；;\n]", text)
    for clause in clauses:
        only = re.search(
            r"(?:只|仅)(?:能|允许)?(?:使用|读取|参考|依据|根据|翻译)?[^。；;，,\n]{0,12}?第([一二三四]|\d+)步|"
            r"\b(?:only|solely|exclusively)\s+(?:(?:use|using|read|reading|from|the|output|of|result|results)\s+)*"
            r"(?:(first|second|third|fourth)\s+step|(T[1-4])(?:\.output)?)", clause, re.I)
        if only:
            allowed = step_references(only.group(0))
            suffix = clause[only.end():]
            while True:
                conjunction = re.match(r"\s*(?:and|or|以及|和|与|及|、)\s*(?:the\s+)?(?:T[1-4](?:\.output)?|(?:first|second|third|fourth)\s+step|第[一二三四\d]+步)", suffix, re.I)
                if not conjunction:
                    break
                allowed.update(step_references(conjunction.group(0)))
                suffix = suffix[conjunction.end():]
            allowed -= excluded_references(clause)
            if inputs - allowed:
                raise ValueError("excluded_input: explicit-only step reference forbids other outputs")
    if split and _required_data_references(text) - inputs:
        raise ValueError("missing_input: an explicitly requested step result is not provided")


def _strings(value, name, *, nonempty=False):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(v, str) or not v.strip() for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must be a unique list of nonempty strings")
    return list(value)


def validate_scores(scores) -> dict[str, float]:
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise ValueError("scores must contain all eight dimensions")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not 0 <= v <= 1 or not math.isfinite(v) for v in scores.values()):
        raise ValueError("scores must be finite numbers between zero and one")
    return {label: float(scores[label]) for label in DIMENSIONS}


def validate_proposal(raw: dict, prompt: str, max_tasks: int = 4) -> dict:
    """Return scoped task skeletons; authentic quotes are not semantic proof."""
    if not isinstance(raw, dict) or set(raw) != {"tasks", "final_constraints", "missing_information"}:
        raise ValueError("invalid proposal fields")
    tasks = raw["tasks"]
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= max_tasks:
        raise ValueError("proposal task count is out of bounds")
    final = _strings(raw["final_constraints"], "final_constraints")
    missing = _strings(raw["missing_information"], "missing_information")
    if any(c not in prompt for c in final):
        raise ValueError("final constraint is not quoted from original request")
    result, quotes_by_id, covered = [], {}, set()
    for index, task in enumerate(tasks, start=1):
        fields = {"instruction", "source_quotes", "depends_on", "input_refs", "constraints"}
        if not isinstance(task, dict) or set(task) != fields:
            raise ValueError("invalid task proposal fields")
        instruction = task["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("empty proposed instruction")
        quotes = _strings(task["source_quotes"], "source_quotes", nonempty=True)
        constraints = _strings(task["constraints"], "constraints")
        for quote in quotes + constraints:
            if quote not in prompt:
                raise ValueError("source quote does not occur in original request")
            offset = 0
            while (offset := prompt.find(quote, offset)) >= 0:
                covered.update(range(offset, offset + len(quote)))
                offset += len(quote)
        deps = _strings(task["depends_on"], "depends_on")
        refs = _strings(task["input_refs"], "input_refs", nonempty=True)
        prior = {f"T{n}" for n in range(1, index)}
        if not set(deps) <= prior:
            raise ValueError("dependencies must refer to preceding tasks")
        for ref in refs:
            if ref != "original_prompt" and ref not in {f"{d}.output" for d in deps}:
                raise ValueError("input reference must name original_prompt or a dependency output")
        referenced = set(re.findall(r"\{\{(T\d+)\.output\}\}", instruction))
        if not {f"{ref}.output" for ref in referenced} <= set(refs):
            raise ValueError("instruction template refers to an unprovided input")
        for requirement in constraints:
            if requirement not in instruction:
                instruction += "\n任务约束：" + requirement
        task_id = f"T{index}"
        quotes_by_id[task_id] = quotes
        result.append({"subtask_id": task_id, "instruction": instruction,
                       "depends_on": deps, "input_refs": refs, "output_type": "text"})
    used = {dep for task in result for dep in task["depends_on"]}
    sinks = [task for task in result if task["subtask_id"] not in used]
    if final and len(sinks) != 1:
        raise ValueError("final-only constraints require a unique final sink or explicit merge")
    if final:
        for requirement in final:
            if requirement not in sinks[0]["instruction"]:
                sinks[0]["instruction"] += "\n最终输出约束：" + requirement
            start = prompt.find(requirement)
            covered.update(range(start, start + len(requirement)))
    for task in result:
        _check_input_scope(task, quotes_by_id[task["subtask_id"]], split=len(result) > 1)
    meaningful = {i for i, c in enumerate(prompt) if c.isalnum()}
    return {"tasks": result, "source_quotes": quotes_by_id, "missing_information": missing,
            "final_constraints": final,
            "source_character_coverage": len(covered & meaningful) / len(meaningful) if meaningful else 1.0,
            "semantic_completeness_verified": False}
