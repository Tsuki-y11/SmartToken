"""Bounded OpenAI-compatible chat transport for hybrid planner research.

No provider is contacted during construction, no retries occur here, and no model
is downloaded. Only the external HTTP transport is injectable for offline tests.
The planner independently validates source evidence, dependencies and scope.
"""

import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import time
from urllib import error, parse, request

from .prompts import LABELS, PURPOSES, build_messages, output_schema


_ERROR_CODES = frozenset({
    "config_missing", "config_invalid", "invalid_purpose", "invalid_payload",
    "request_too_large", "timeout", "transport_error", "http_error",
    "response_too_large", "invalid_response", "invalid_json", "invalid_output",
})


def _token_count(value):
    return value if type(value) is int and value >= 0 else None


def _normalized_usage(value):
    source = value if isinstance(value, dict) else {}
    return {
        "input_tokens": _token_count(source.get("input_tokens")),
        "output_tokens": _token_count(source.get("output_tokens")),
    }


class LLMError(RuntimeError):
    """A safe error code and observed accounting, without provider text/secrets."""

    def __init__(self, code, usage=None, latency_ms=None):
        self.code = code if isinstance(code, str) and code in _ERROR_CODES else "llm_error"
        self.usage = _normalized_usage(usage)
        self.latency_ms = (
            float(latency_ms)
            if type(latency_ms) in (int, float) and math.isfinite(latency_ms) and latency_ms >= 0
            else None
        )
        super().__init__(self.code)


class _NoRedirect(request.HTTPRedirectHandler):
    """Do not forward credentials or permit transport downgrades via redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_transport(req, *, timeout):
    return request.build_opener(_NoRedirect()).open(req, timeout=timeout)


def _safe_close(response):
    if response is not None:
        try:
            response.close()
        except Exception:
            # Closing must neither mask the normalized result nor expose a body.
            pass


def _validate_endpoint(api_url):
    if any(ord(c) <= 32 or ord(c) == 127 for c in api_url) or "\\" in api_url:
        raise ValueError
    url = parse.urlsplit(api_url)
    if (
        url.scheme not in {"http", "https"} or not url.hostname or
        url.username is not None or url.password is not None or url.query or url.fragment
    ):
        raise ValueError
    # Accessing port validates malformed and out-of-range port numbers.
    if url.port is not None and not 1 <= url.port <= 65535:
        raise ValueError
    loopback = url.hostname.lower() == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            loopback = False
    if url.scheme == "http" and not loopback:
        raise ValueError
    return loopback


def _json_safe(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(type(key) is str and _json_safe(item) for key, item in value.items())
    return False


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError


def _parse_content(content):
    text = content.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.DOTALL | re.IGNORECASE)
        if match is None:
            raise ValueError
        text = match.group(1)
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError
    return value


def _strings(value, *, nonempty=False):
    return (
        isinstance(value, list) and (not nonempty or bool(value)) and
        all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _valid_output(purpose, data, payload=None):
    if purpose == "classify":
        if set(data) != {"predictions"} or not isinstance(data["predictions"], list) or not data["predictions"]:
            return False
        seen = set()
        for prediction in data["predictions"]:
            if not isinstance(prediction, dict) or set(prediction) != {"task_id", "scores"}:
                return False
            task_id = prediction["task_id"]
            if not isinstance(task_id, str) or not re.fullmatch(r"DIRECT|T[1-9][0-9]*", task_id) or task_id in seen:
                return False
            seen.add(task_id)
            scores = prediction["scores"]
            if not isinstance(scores, dict) or set(scores) != set(LABELS):
                return False
            if any(type(score) not in (int, float) or not 0 <= score <= 1 or not math.isfinite(score) for score in scores.values()):
                return False
        requested = (payload or {}).get("tasks")
        if requested is not None:
            if not isinstance(requested, list) or any(not isinstance(t, dict) for t in requested):
                return False
            expected = [task.get("task_id") for task in requested]
            if any(not isinstance(tid, str) for tid in expected) or len(set(expected)) != len(expected):
                return False
            if seen != set(expected):
                return False
        return True
    if set(data) != {"tasks", "final_constraints", "missing_information"}:
        return False
    if not isinstance(data["tasks"], list) or not data["tasks"]:
        return False
    if not _strings(data["final_constraints"]) or not _strings(data["missing_information"]):
        return False
    for task in data["tasks"]:
        if not isinstance(task, dict) or set(task) != {"instruction", "source_quotes", "depends_on", "input_refs", "constraints"}:
            return False
        if not isinstance(task["instruction"], str) or not task["instruction"].strip():
            return False
        if not _strings(task["source_quotes"], nonempty=True):
            return False
        if any(not _strings(task[key]) for key in ("depends_on", "input_refs", "constraints")):
            return False
    return True


class LLMClient:
    """One provider attempt per complete(); callers own request-level budgets.

    ``api_url`` is the complete chat-completions endpoint. API key configuration
    is required for remote endpoints and optional for verified loopback hosts.
    ``timeout`` is the urllib/socket timeout in seconds, not a workflow deadline.
    ``transport`` accepts a Request and keyword timeout and returns a readable,
    closable HTTP response; production defaults to urllib with redirects disabled.
    """

    def __init__(self, api_url, api_key, model, *, transport=None, timeout=30.0,
                 max_request_bytes=262144, max_response_bytes=1048576,
                 max_tokens=2048, reasoning_effort=None, response_format="json_object"):
        values = (api_url, model)
        if any(value is None or (isinstance(value, str) and not value.strip()) for value in values):
            raise LLMError("config_missing")
        try:
            if any(not isinstance(value, str) or any(ord(c) < 32 or ord(c) == 127 for c in value) for value in values):
                raise ValueError
            loopback = _validate_endpoint(api_url)
            if api_key is None or (isinstance(api_key, str) and not api_key.strip()):
                if not loopback:
                    raise LLMError("config_missing")
                api_key = ""
            if not isinstance(api_key, str) or any(ord(c) < 32 or ord(c) == 127 for c in api_key):
                raise ValueError
            api_key.encode("ascii")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
                raise ValueError
            if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
                raise ValueError
            if not isinstance(response_format, str) or response_format not in {"json_object", "json_schema"}:
                raise ValueError
            if reasoning_effort is not None and (
                not isinstance(reasoning_effort, str) or
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", reasoning_effort) is None
            ):
                raise ValueError
            if any(type(bound) is not int or bound <= 0 for bound in (max_request_bytes, max_response_bytes)):
                raise ValueError
            if transport is not None and not callable(transport):
                raise ValueError
        except (TypeError, ValueError, UnicodeError, OverflowError):
            raise LLMError("config_invalid") from None
        self.api_url = api_url
        self._api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self._transport = _http_transport if transport is None else transport

    @property
    def public_config(self):
        """Return reproducible generation settings without endpoint or credentials."""
        return {
            "model": self.model, "timeout_seconds": self.timeout,
            "max_tokens": self.max_tokens, "reasoning_effort": self.reasoning_effort,
            "response_format": self.response_format,
        }

    @classmethod
    def from_env(cls, *, environ=None, transport=None):
        """Load prefixed settings, falling back individually to legacy LLM_*.

        An explicitly present but empty prefixed setting is invalid/missing, not
        silently replaced with another credential or endpoint.
        """
        env = os.environ if environ is None else environ
        def setting(name, default=None):
            return env.get("HYBRID_" + name, env.get(name, default))
        try:
            timeout_text = setting("LLM_TIMEOUT_SECONDS", "30")
            tokens_text = setting("LLM_MAX_TOKENS", "2048")
            if not isinstance(timeout_text, str) or not isinstance(tokens_text, str):
                raise ValueError
            if re.fullmatch(r"[0-9]+", tokens_text) is None:
                raise ValueError
            timeout = float(timeout_text)
            max_tokens = int(tokens_text)
        except (TypeError, ValueError, OverflowError):
            raise LLMError("config_invalid") from None
        reasoning_effort = env.get("HYBRID_LLM_REASONING_EFFORT")
        if reasoning_effort == "":
            reasoning_effort = None
        return cls(
            setting("LLM_API_URL"), setting("LLM_API_KEY"), setting("LLM_MODEL"),
            transport=transport, timeout=timeout, max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            response_format=env.get("HYBRID_LLM_RESPONSE_FORMAT", "json_object"),
        )

    def complete(self, purpose, payload):
        """Return a validated structured object plus truthful attempt accounting."""
        if not isinstance(purpose, str) or purpose not in PURPOSES:
            raise LLMError("invalid_purpose")
        try:
            if not isinstance(payload, dict) or not _json_safe(payload):
                raise ValueError
            messages = build_messages(purpose, payload)
            prompt_bytes = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            request_data = {
                "model": self.model, "messages": messages, "stream": False,
                "temperature": 0, "response_format": {"type": "json_object"},
                "max_tokens": self.max_tokens,
            }
            if self.response_format == "json_schema":
                request_data["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "smarttoken_" + purpose, "strict": True,
                    "schema": output_schema(purpose, payload),
                }}
            if self.reasoning_effort is not None:
                request_data["reasoning_effort"] = self.reasoning_effort
            body = json.dumps(request_data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
            raise LLMError("invalid_payload") from None
        if len(body) > self.max_request_bytes:
            raise LLMError("request_too_large")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        req = request.Request(self.api_url, data=body, method="POST", headers=headers)
        started = time.monotonic()
        usage = {"input_tokens": None, "output_tokens": None}
        response = None
        try:
            response = self._transport(req, timeout=self.timeout)
            status = getattr(response, "status", None)
            if status is not None and (type(status) is not int or not 200 <= status < 300):
                raise LLMError("http_error")
            raw = response.read(self.max_response_bytes + 1)
            if not isinstance(raw, bytes):
                raise LLMError("invalid_response")
            if len(raw) > self.max_response_bytes:
                raise LLMError("response_too_large")
            try:
                # Invalid provider usage constants must become unknown, while the
                # structured assistant JSON is parsed strictly below.
                document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError, RecursionError):
                raise LLMError("invalid_response") from None
            if not isinstance(document, dict):
                raise LLMError("invalid_response")
            remote_usage = document.get("usage")
            if isinstance(remote_usage, dict):
                usage = _normalized_usage({
                    "input_tokens": remote_usage.get("prompt_tokens"),
                    "output_tokens": remote_usage.get("completion_tokens"),
                })
            choices = document.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise LLMError("invalid_response")
            choice = choices[0]
            message = choice.get("message")
            if (
                choice.get("finish_reason") not in (None, "stop") or
                not isinstance(message, dict) or message.get("refusal") or
                message.get("tool_calls") or not isinstance(message.get("content"), str)
            ):
                raise LLMError("invalid_response")
            try:
                data = _parse_content(message["content"])
            except (ValueError, UnicodeError, RecursionError):
                raise LLMError("invalid_json") from None
            if not _valid_output(purpose, data, payload):
                raise LLMError("invalid_output")
            return {
                "data": data, "usage": usage,
                "latency_ms": (time.monotonic() - started) * 1000,
                "model": self.model, "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            }
        except LLMError as failure:
            raise LLMError(failure.code, usage=usage, latency_ms=(time.monotonic() - started) * 1000) from None
        except error.HTTPError as failure:
            _safe_close(failure)
            raise LLMError("http_error", usage=usage, latency_ms=(time.monotonic() - started) * 1000) from None
        except (TimeoutError, socket.timeout):
            raise LLMError("timeout", usage=usage, latency_ms=(time.monotonic() - started) * 1000) from None
        except error.URLError as failure:
            code = "timeout" if isinstance(failure.reason, (TimeoutError, socket.timeout)) else "transport_error"
            raise LLMError(code, usage=usage, latency_ms=(time.monotonic() - started) * 1000) from None
        except Exception:
            raise LLMError("transport_error", usage=usage, latency_ms=(time.monotonic() - started) * 1000) from None
        finally:
            _safe_close(response)
