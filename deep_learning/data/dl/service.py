"""Local authenticated prediction HTTP API and existing frontend adapters."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from deep_learning.pipeline import PredictionPipeline, normalize_request, render_tasks


def dispatch(method, path, authorization, body, pipeline, api_key):
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "models": len(deep_learning.pipeline.model_ids), "real_models_loaded": True}
    if not api_key or not hmac.compare_digest(authorization, "Bearer " + api_key):
        return 401, {"error": "UNAUTHORIZED"}
    try:
        if method == "GET" and path == "/models":
            return 200, {"source": "trained_token_predictor", "models": [
                {"model_id": m, "callable": False} for m in deep_learning.pipeline.model_ids]}
        if method != "POST":
            return 404, {"error": "NOT_FOUND"}
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        if path == "/predict":
            result = deep_learning.pipeline.predict(body, model_ids=body.get("model_ids"),
                                      completed_outputs=body.get("completed_outputs"))
            return 200, result
        if path == "/repredict":
            return 200, deep_learning.pipeline.repredict(body.get("previous_result"),
                completed_outputs=body.get("completed_outputs"), model_ids=body.get("model_ids"))
        if path == "/plans":
            from deep_learning.planner.src.task_decomposer.adapter import llc_request_to_run_request, task_plan_to_task_plans
            request = llc_request_to_run_request(body)
            result = deep_learning.pipeline.analyze(request)
            if result["audit"]["status"] != "ok":
                return 422, {"error": "REVIEW_REQUIRED", "planning_audit": result["audit"]}
            bundle = task_plan_to_task_plans(result["plan"], request["original_prompt"])
            bundle["planning_audit"] = result["audit"]
            return 200, bundle
        if path == "/token-predictions":
            from deep_learning.dl.subtasks import SubtaskPredictor
            from deep_learning.dl.dimensions import normalize_annotation
            bundle = body.get("task_plans")
            if not isinstance(bundle, dict) or bundle.get("schema_version") != "1.0":
                raise ValueError("task_plans must have schema_version 1.0")
            plans = bundle.get("plans")
            if not isinstance(plans, list) or len(plans) != 1 or not isinstance(plans[0], dict):
                raise ValueError("exactly one frontend plan is supported per prediction request")
            source = plans[0]
            if source.get("input_refs") or source.get("material") or source.get("attachments"):
                raise ValueError("MATERIAL_NOT_SUPPORTED: put the actual material explicitly in prompt_template")
            if not isinstance(source.get("prompt_id"), str) or not source["prompt_id"].strip() or not isinstance(source.get("original_prompt"), str) or not source["original_prompt"].strip():
                raise ValueError("prompt_id and original_prompt must be nonempty")
            steps = source.get("subtasks")
            if not isinstance(steps, list) or not steps:
                raise ValueError("subtasks must be nonempty")
            tasks = []
            prior = set()
            for index, row in enumerate(steps, 1):
                if not isinstance(row, dict) or row.get("order") != index:
                    raise ValueError("steps must be objects in continuous order")
                deps, contexts = row.get("depends_on"), row.get("context_from")
                if not isinstance(deps, list) or not isinstance(contexts, list):
                    raise ValueError("depends_on and context_from must be lists")
                if row.get("step_id") != f"T{index}" or any(not isinstance(d, str) or d not in prior for d in deps) or len(set(deps)) != len(deps):
                    raise ValueError("step IDs must be T1, T2, ... with unique backward dependencies")
                normalize_annotation(row.get("dimensions"), "ml")
                if not isinstance(row.get("prompt_template"), str) or not row["prompt_template"].strip():
                    raise ValueError("prompt_template must be nonempty")
                if any(not isinstance(c, dict) or c.get("mode") != "full_output" or c.get("step_id") not in deps for c in contexts):
                    raise ValueError("invalid context_from dependency")
                tasks.append({"subtask_id": row.get("step_id"), "instruction": row.get("prompt_template"),
                    "dimensions": row.get("dimensions"), "dimension_confidence": row.get("dimension_confidence", {}),
                    "depends_on": deps, "input_refs": [c["step_id"] + ".output" for c in contexts], "output_type": "text"})
                prior.add(row["step_id"])
            native = {"schema_version": "1.0", "request_id": source.get("prompt_id"),
                      "direct_task": None, "subtasks": tasks}
            rendered, pending = render_tasks(native, source.get("original_prompt", ""), body.get("completed_outputs"))
            estimates = []
            selected = deep_learning.pipeline.select_models(body.get("model_ids"))
            if rendered:
                result = SubtaskPredictor(deep_learning.pipeline.predictor).predict(
                    {"request_id": source["prompt_id"], "subtasks": rendered}, selected)
                estimates = [{"step_id": r["subtask_id"], "model_id": r["model_id"],
                    "predicted_output_tokens": r["output_tokens_est"], "estimated_input_tokens": None,
                    "phase": "post_dependency" if body.get("completed_outputs") else "pre_execution",
                    "is_provisional": False} for r in result["predictions"]]
            return 200, {"schema_version": "frontend-1.0", "prompt_id": source["prompt_id"],
                         "source": "model_aware_v3", "estimates": estimates, "pending_predictions": pending,
                         "notes": "部分依赖步骤等待实际上游输出：" + ", ".join(r["subtask_id"] for r in pending) if pending else "所有列出的预测均来自已加载的 DL v3 权重；输入 Token 未估算。"}
        return 404, {"error": "NOT_FOUND"}
    except (ValueError, KeyError, TypeError) as exc:
        code = getattr(exc, "code", "INVALID_REQUEST")
        return 400, {"error": code, "detail": str(exc)[:500]}


def make_server(pipeline, api_key, host="127.0.0.1", port=8100):
    if not api_key:
        raise ValueError("SMARTTOKEN_API_KEY is required")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            path = urlsplit(self.path).path
            body = None
            if self.command == "POST":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 262144:
                        self.send_json(413, {"error": "BODY_SIZE_LIMIT"})
                        return
                    body = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeDecodeError):
                    self.send_json(400, {"error": "INVALID_JSON"})
                    return
            try:
                with lock:
                    status, result = dispatch(self.command, path, self.headers.get("Authorization", ""), body, pipeline, api_key)
            except Exception:
                self.send_json(500, {"error": "INFERENCE_FAILED"})
                return
            self.send_json(status, result)

        def send_json(self, status, result):
            data = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        do_GET = handle_request
        do_POST = handle_request

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--mode", choices=("ml", "hybrid"), default="ml")
    parser.add_argument("--encoder-path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    api_key = os.environ.get("SMARTTOKEN_API_KEY", "")
    if not api_key:
        parser.error("set SMARTTOKEN_API_KEY before starting")
    pipeline = PredictionPipeline.load(encoder_path=args.encoder_path, mode=args.mode, device=args.device)
    server = make_server(pipeline, api_key, args.host, args.port)
    print(f"Prediction service ready: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
