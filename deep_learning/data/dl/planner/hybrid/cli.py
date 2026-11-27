"""Run with python -m deep_learning.planner.hybrid.cli from the SmartToken application root."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from .evaluation import inference_request, read_jsonl, run_experiment, score_results, write_json
from .planner import HybridPlanner, PlannerConfig


def build_planner(args, mode=None):
    mode = mode or args.mode
    models = None
    if mode != "llm":
        from .models import TaskModels, FrozenModels
        if args.artifact and mode != "baseline":
            encoder_path = getattr(args, "encoder_path", None)
            options = {"encoder_path": encoder_path} if encoder_path is not None else {}
            models = TaskModels.load(args.artifact, **options)
        else:
            models = FrozenModels(backend=args.backend)
            # Force startup errors before opening a server/experiment output.
            _ = models.thresholds
    if mode == "ml" and not args.artifact:
        raise ValueError("ml mode requires --artifact with trained split/boundary models")
    llm = None
    if mode in {"llm", "hybrid"} and not args.offline:
        from .llm import LLMClient
        llm = LLMClient.from_env()
    if mode == "llm" and args.offline:
        raise ValueError("llm mode cannot run offline")
    return HybridPlanner(models, llm, mode, PlannerConfig(args.uncertainty_margin, args.max_llm_calls))


def _common(parser):
    parser.add_argument("--backend", choices=["v5", "candidate_c"], default="v5")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--encoder-path", type=Path, help="relocated local encoder; verified against artifact hashes")
    parser.add_argument("--offline", action="store_true", help="never instantiate a provider client")
    parser.add_argument("--uncertainty-margin", type=float, default=.08)
    parser.add_argument("--max-llm-calls", type=int, default=2)


def parser():
    root = argparse.ArgumentParser(description="Opt-in hybrid ML/LLM planning experiments")
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="analyze one JSON request")
    _common(plan)
    plan.add_argument("--mode", choices=["baseline", "ml", "llm", "hybrid"], default="hybrid")
    plan.add_argument("--input", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    run = commands.add_parser("run", help="compare modes on label-free JSONL inputs")
    _common(run)
    run.add_argument("--modes", nargs="+", choices=["baseline", "ml", "llm", "hybrid"], default=["baseline", "llm", "hybrid"])
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--outdir", type=Path, required=True)
    train = commands.add_parser("train", help="train reviewed train/validation data; test Gold is forbidden")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--outdir", type=Path, required=True)
    train.add_argument("--encoder-path", type=Path)
    coverage = commands.add_parser("train-coverage", help="refit research label heads only; preserve reviewed gates")
    coverage.add_argument("--base-artifact", type=Path, required=True)
    coverage.add_argument("--dataset-dir", type=Path, required=True)
    coverage.add_argument("--reviewed-data", type=Path)
    coverage.add_argument("--outdir", type=Path, required=True)
    coverage.add_argument("--encoder-path", type=Path)
    coverage.add_argument("--acknowledge-ai-research", action="store_true")
    coverage.add_argument("--threshold-policy", choices=["original", "equivalent-midpoint"], default="original")
    comparison = commands.add_parser("compare-coverage", help="re-audit candidate and compare label-only Validation diagnostics")
    comparison.add_argument("--base-artifact", type=Path, required=True)
    comparison.add_argument("--candidate-artifact", type=Path, required=True)
    comparison.add_argument("--dataset-dir", type=Path, required=True)
    comparison.add_argument("--reviewed-data", type=Path)
    comparison.add_argument("--encoder-path", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="score a completed run against separate human Gold")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--alignments", type=Path)
    score.add_argument("--reviews", type=Path)
    score.add_argument("--output", type=Path)
    serve = commands.add_parser("serve", help="serve research endpoints; HYBRID_SERVICE_API_KEY required")
    _common(serve)
    serve.add_argument("--mode", choices=["baseline", "ml", "llm", "hybrid"], default="hybrid")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8103)
    preflight = commands.add_parser("preflight", help="check prerequisites without downloads or provider calls")
    preflight.add_argument("--artifact", type=Path)
    preflight.add_argument("--encoder-path", type=Path, help="relocated local encoder; verified against artifact hashes")
    preflight.add_argument("--offline", action="store_true", help="check local prerequisites without requiring a provider")
    return root


def preflight(artifact=None, *, encoder_path=None, offline=False):
    packages = {name: importlib.util.find_spec(name) is not None for name in
                ("numpy", "sklearn", "joblib", "torch", "sentence_transformers", "sentencepiece")}
    provider, code = False, None
    if not offline:
        try:
            from .llm import LLMClient
            LLMClient.from_env()
            provider = True
        except Exception as exc:
            code = getattr(exc, "code", type(exc).__name__)
    artifact_ok = False
    if artifact:
        from .models import TaskModels
        options = {"encoder_path": encoder_path} if encoder_path is not None else {}
        TaskModels.load(artifact, **options)
        artifact_ok = True
    result = {"packages_discoverable": packages, "provider_configured": provider,
              "provider_required": not offline,
              "provider_configuration_error": code, "artifact_verified": artifact_ok,
              "real_inference_tested": False, "provider_reachable_tested": False,
              "note": "Discovery is not runtime verification. Run real plan and reviewed experiments on the server."}
    return result


def _emit(data, path=None):
    if path:
        if path.exists():
            raise ValueError("output already exists; choose a fresh path")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, data)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.artifact, encoder_path=args.encoder_path, offline=args.offline)
            _emit(result)
            provider_ok = not result["provider_required"] or result["provider_configured"]
            return 0 if provider_ok and (result["artifact_verified"] or all(result["packages_discoverable"].values())) else 2
        if args.command == "train":
            from .models import train_models
            metadata = train_models(read_jsonl(args.data), args.outdir, encoder_path=args.encoder_path)
            _emit(metadata)
            return 0
        if args.command == "train-coverage":
            from .coverage import refit_label_heads
            metadata = refit_label_heads(args.base_artifact, args.dataset_dir, args.outdir,
                encoder_path=args.encoder_path,
                reviewed_records=read_jsonl(args.reviewed_data) if args.reviewed_data else None,
                acknowledge_ai_research=args.acknowledge_ai_research,
                threshold_policy=args.threshold_policy)
            _emit(metadata)
            return 0
        if args.command == "compare-coverage":
            from .coverage_evaluation import compare_coverage
            if args.output.exists():
                raise ValueError("output already exists; choose a fresh path")
            result = compare_coverage(args.base_artifact, args.candidate_artifact, args.dataset_dir,
                encoder_path=args.encoder_path,
                reviewed_records=read_jsonl(args.reviewed_data) if args.reviewed_data else None)
            _emit(result, args.output)
            return 0
        if args.command == "score":
            manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") not in {"complete", "complete_with_failures"}:
                raise ValueError("cannot score an incomplete run")
            if hashlib.sha256((args.run_dir / "results.jsonl").read_bytes()).hexdigest() != manifest.get("results_sha256"):
                raise ValueError("run results hash does not match its manifest")
            result = score_results(read_jsonl(args.run_dir / "results.jsonl"), read_jsonl(args.gold),
                                   alignments=read_jsonl(args.alignments) if args.alignments else None,
                                   reviews=read_jsonl(args.reviews) if args.reviews else None,
                                   expected_modes=manifest["modes"])
            _emit(result, args.output)
            return 0
        if args.command == "run":
            if len(set(args.modes)) != len(args.modes):
                raise ValueError("duplicate experiment modes")
            inputs = read_jsonl(args.inputs)
            # Reject accidental Gold before expensive model loading or any call.
            for row in inputs:
                inference_request(row)
            planners = {mode: build_planner(args, mode) for mode in args.modes}
            result = run_experiment(inputs, planners, args.outdir)
            _emit(result)
            return 0 if result["failure_count"] == 0 else 2
        if args.command == "plan":
            req = inference_request(json.loads(args.input.read_text(encoding="utf-8-sig")))
            if args.output and args.output.exists():
                raise ValueError("output already exists; choose a fresh path")
            result = build_planner(args).analyze(req)
            _emit(result, args.output)
            return 0 if result["audit"]["status"] == "ok" else 2
        if args.command == "serve":
            from .service import make_server
            key = os.environ.get("HYBRID_SERVICE_API_KEY", "")
            if not key:
                raise ValueError("HYBRID_SERVICE_API_KEY is required")
            server = make_server(build_planner(args), key, args.host, args.port)
            print(f"Hybrid research service listening on {args.host}:{args.port}", flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
            return 0
    except Exception as exc:
        error = {"error": getattr(exc, "code", type(exc).__name__), "command": args.command}
        # Local validation errors are actionable; provider errors are code-only.
        if isinstance(exc, (ValueError, FileNotFoundError, ModuleNotFoundError)):
            error["detail"] = str(exc)[:500]
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
