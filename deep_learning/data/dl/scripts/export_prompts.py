#!/usr/bin/env python3
"""Export current shipped prompts deterministically; Python stdlib only, no LLM.

Run from any directory. --check is read-only and fails on missing/stale exports.
Source hashes normalize CRLF to LF so Git checkout settings do not cause drift.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "faf6b6538b610d3e42266b9115eea31aaedc0661"
SOURCE_PATHS = (
    "planner/hybrid/prompts.py", "planner/hybrid/deep_learning.llm.py", "planner/hybrid/planner.py",
    "llm/server/prompts.js", "llm/server/index.js", "llm/server/local-executor.js",
    "optimization/recommendation_factory.py", "optimization/src/calibration.py",
    "prompts/examples/payloads.json", "scripts/export_prompts.py",
)


def read_text(relative):
    return (ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def node_literals(source):
    """Accept only the current explicit string exports; never eval JavaScript.

    Fail closed if templates gain interpolation, escapes, or computed values.
    Such a source change needs a deliberate exporter update, not a guessed value.
    """
    source = re.sub(r"^\s*//[^\n]*(?:\n|$)", "", source, flags=re.MULTILINE)
    pattern = re.compile(
        r"\s*export\s+const\s+([A-Z_]+)\s*=\s*"
        r"('(?:[^'\\\r\n])*'|\"(?:[^\"\\\r\n])*\"|`(?:[^`\\])*`)\s*;"
    )
    result = {}
    position = 0
    while source[position:].strip():
        match = pattern.match(source, position)
        if match is None:
            raise ValueError("Node prompt source no longer consists of supported explicit literals")
        name, literal = match.groups()
        if name in result or "${" in literal:
            raise ValueError("Duplicate or interpolated Node prompt export")
        result[name] = literal[1:-1]
        position = match.end()
    expected = {"CHAT_SYSTEM_PROMPT", "REPORT_SYSTEM_PROMPT", "EXECUTION_SYSTEM_PROMPT"}
    if set(result) != expected:
        raise ValueError("Node prompt exports do not match the three supported templates")
    return result


def excerpt(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("Template assembly source anchors changed; update exporter explicitly")
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish].rstrip() + "\n"


def forecast_export(payloads):
    source = read_text("optimization/recommendation_factory.py")
    tree = ast.parse(source)
    selected = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SYSTEM_INSTRUCTION" for target in node.targets
        )
    ) or (isinstance(node, ast.FunctionDef) and node.name == "forecast_text")]
    if len(selected) != 2:
        raise ValueError("Forecast template declarations changed")
    # Execute only the shipped constant/function AST, not factory imports or model loading.
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "forecast_template", "exec"), namespace)
    calibration = ast.parse(read_text("optimization/src/calibration.py"))
    versions = [ast.literal_eval(node.value) for node in calibration.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "TEMPLATE_VERSION" for target in node.targets)]
    if len(versions) != 1:
        raise ValueError("Forecast template version declaration changed")
    return {
        "purpose": "DL token forecast input only; proposed template, never an executed LLM request",
        "template_version": versions[0],
        "system_instruction": namespace["SYSTEM_INSTRUCTION"],
        "exact_source_excerpt": "\n\n".join(ast.get_source_segment(source, node) for node in selected) + "\n",
        "examples": {name: {"illustrative_only": True, "input": payload,
                             "text": namespace["forecast_text"](**payload)} for name, payload in payloads.items()},
    }


def build_exports():
    spec = importlib.util.spec_from_file_location("smarttoken_prompt_export", ROOT / "planner/hybrid/prompts.py")
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    payloads = json.loads(read_text("prompts/examples/payloads.json"))
    exports = {}
    message_hashes = {}
    for purpose in ("decompose", "review", "classify"):
        messages = prompts.build_messages(purpose, payloads[purpose])
        exports[f"{purpose}.messages.json"] = json_text(messages)
        exports[f"{purpose}.schema.json"] = json_text(prompts.output_schema(purpose, payloads[purpose]))
        message_hashes[purpose] = sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    server = read_text("llm/server/index.js")
    executor = read_text("llm/server/local-executor.js")
    exports["node_templates.json"] = json_text({
        "purpose": "Optional Node chat/report/execution code; not used by normal plan-only ML/DL service",
        "system_prompts": node_literals(read_text("llm/server/prompts.js")),
        "exact_assembly_source": {
            "chat": excerpt(server, "async function chatRoute(req,res) {", "function extractJSON(content) {"),
            "report_request": excerpt(server, "async function reportRoute(req,res) {", "  const result=extractJSON(raw);"),
            "execution_render": excerpt(executor, "function renderPrompt(step,outputs) {", "export function createLocalExecutor"),
            "execution_messages": excerpt(executor, "        const prompt=renderPrompt(step,outputs);", "        if(typeof result?.content"),
        },
    })
    exports["recommendation_forecast.json"] = json_text(forecast_export(payloads["forecast"]))
    manifest = {
        "format_version": "1.0", "prompt_version": prompts.PROMPT_VERSION,
        "notice": payloads["_notice"],
        "provenance": {
            "upstream_commit": UPSTREAM_COMMIT,
            "basis": "Current shipped source is canonical; upstream commit records origin, not a historical experiment revision",
            "path_root": "deep_learning/",
            "relocated_sources": {"optimization/recommendation_factory.py": "optimization/recommendation_factory.py", "optimization/src/calibration.py": "optimization/src/calibration.py"},
            "relocation_note": "Keys are shipped llc-relative paths; values are original repository-relative paths",
            "hash_encoding": "sha256 of UTF-8 text normalized to LF; no timestamp or machine-specific path",
        },
        "sources": {path: {"sha256_utf8_lf": sha256(read_text(path))} for path in SOURCE_PATHS},
        "message_sha256": message_hashes,
        "message_hash_encoding": "Same canonical JSON serialization as LLMClient.complete prompt_sha256",
        "exports": {name: {"sha256_utf8_lf": sha256(value)} for name, value in exports.items()},
    }
    exports["manifest.json"] = json_text(manifest)
    return exports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prompts/exports")
    parser.add_argument("--check", action="store_true", help="fail if exports are absent or stale; never write")
    args = parser.parse_args(argv)
    try:
        exports = build_exports()
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 2
    if args.check:
        mismatches = []
        for name, expected in exports.items():
            path = args.output_dir / name
            if not path.is_file():
                mismatches.append(f"Missing: {name}")
            elif path.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
                mismatches.append(f"Stale: {name}")
        if mismatches:
            print("\n".join(mismatches))
            return 1
        print(f"Prompt exports match current sources ({len(exports)} files).")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in exports.items():
        (args.output_dir / name).write_bytes(value.encode("utf-8"))
    print(f"Exported {len(exports)} prompt files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
