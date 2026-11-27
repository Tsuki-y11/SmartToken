"""Opt-in localhost research service; uncertain plans cannot enter execution endpoints."""
from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deep_learning.planner.src.task_decomposer.adapter import AdapterError, llc_request_to_run_request, task_plan_to_task_plans
from deep_learning.planner.src.task_decomposer.validator import TaskPlanValidationError

MAX_BODY_BYTES = 262144


def dispatch(method, path, authorization, body, planner, api_key):
    if not api_key:
        raise ValueError("HYBRID_SERVICE_API_KEY is required")
    if not hmac.compare_digest(authorization.encode("utf-8"), ("Bearer " + api_key).encode("utf-8")):
        return 401, {"error": "UNAUTHORIZED"}
    if method == "GET" and path == "/health":
        return 200, {"status": "configured", "mode": planner.mode, "research_only": True,
                     "real_inference_verified_by_health": False}
    if method != "POST" or path not in {"/analyze", "/task-plan", "/plans"}:
        return 404, {"error": "NOT_FOUND"}
    if len(body) > MAX_BODY_BYTES:
        return 413, {"error": "REQUEST_TOO_LARGE"}
    try:
        payload = json.loads(body)
        request = llc_request_to_run_request(payload) if path == "/plans" else payload
        result = planner.analyze(request)
        if path == "/analyze":
            return (503 if result["plan"] is None and result["audit"]["status"] == "failed" else 200), result
        if result["plan"] is None or result["audit"]["status"] != "ok":
            return 422, {"error": "INPUT_REQUIRED" if result["audit"]["status"] == "needs_input" else "REVIEW_REQUIRED",
                         "request_id": request["request_id"], "audit": result["audit"]}
        if path == "/plans":
            return 200, task_plan_to_task_plans(result["plan"], request["original_prompt"])
        return 200, result["plan"]
    except AdapterError as exc:
        return 422 if exc.code != "INVALID_REQUEST" else 400, {"error": exc.code}
    except (ValueError, UnicodeDecodeError, TaskPlanValidationError):
        return 400, {"error": "INVALID_REQUEST"}


def make_server(planner, api_key, host="127.0.0.1", port=8103):
    if not api_key:
        raise ValueError("HYBRID_SERVICE_API_KEY is required")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, format, *args):
            return  # Never log credentials, prompts or provider responses.

        def _send(self, status, data):
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _handle(self):
            if not lock.acquire(blocking=False):
                self._send(503, {"error": "BUSY"})
                return
            try:
                length = self.headers.get("Content-Length", "0")
                if self.headers.get("Transfer-Encoding") or not length.isascii() or not length.isdecimal():
                    self._send(400, {"error": "INVALID_CONTENT_LENGTH"})
                    return
                size = int(length)
                if size > MAX_BODY_BYTES:
                    self._send(413, {"error": "REQUEST_TOO_LARGE"})
                    return
                body = self.rfile.read(size) if size else b""
                if len(body) != size:
                    self._send(400, {"error": "INCOMPLETE_BODY"})
                    return
                status, data = dispatch(self.command, self.path, self.headers.get("Authorization", ""), body, planner, api_key)
                self._send(status, data)
            except (TimeoutError, ConnectionError):
                self.close_connection = True
            except Exception:
                self._send(500, {"error": "INTERNAL_ERROR"})
            finally:
                lock.release()

        do_GET = _handle
        do_POST = _handle

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
