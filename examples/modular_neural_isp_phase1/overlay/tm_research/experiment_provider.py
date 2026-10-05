"""Bounded streaming Chat Completions with durable request and failure evidence.

No retry or repair of generated JSON occurs here. For text, token booking uses
the larger of the configured reservation and UTF-8 request bytes plus 512 units
per message for templates. Image calls can use an explicit provider-calibrated
reservation; otherwise they also use the byte booking. These are conservative
accounting units, not measured tokens or proof of a hard tokenizer/context cap.
Providers must enforce requested max_tokens (including reasoning) for output bounds.
"""
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .experiment_costs import _atomic_json, _usage


class ProviderError(ValueError):
    """A model request failed, truncated, or violated its response contract."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _strict_object(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number: " + value)
        return number

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


def _bounded_chunks(response, raw_wire, max_bytes, deadline):
    """read1 returns after one socket read, so drip-fed lines cannot reset a deadline."""
    while True:
        if time.monotonic() >= deadline:
            raise ProviderError("Model request walltime exceeded")
        chunk = response.read1(min(4096, max_bytes - len(raw_wire) + 1))
        if not chunk:
            return
        raw_wire.extend(chunk)
        if len(raw_wire) > max_bytes:
            raise ProviderError("Model response exceeds byte limit")
        yield chunk


def _stream_lines(response, raw_wire, max_bytes, deadline):
    pending = bytearray()
    for chunk in _bounded_chunks(response, raw_wire, max_bytes, deadline):
        pending.extend(chunk)
        while b"\n" in pending:
            position = pending.index(b"\n") + 1
            line = bytes(pending[:position])
            del pending[:position]
            yield line
    if pending:
        yield bytes(pending)


class ChatClient:
    def __init__(self, config, ledger):
        self.config = dict(config)
        self.ledger = ledger
        base = self.config.get("base_url", "")
        url = urlsplit(base)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) endpoint without embedded credentials")
        self.endpoint = base.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.model = self.config.get("model")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model identity is required")
        self.timeout = self.config.get("timeout_seconds", 120)
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.max_response_bytes = self.config.get("max_response_bytes", 8 * 1024 * 1024)
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if not self.config.get("log_dir"):
            raise ValueError("log_dir is required for model evidence")
        self.log_dir = Path(self.config["log_dir"]).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if "api_key" in self.config:
            raise ValueError("Credentials must come from api_key_env")
        self.key_env = self.config.get("api_key_env")
        self.secret = os.environ.get(self.key_env) if self.key_env else None
        if self.key_env and not self.secret:
            raise ValueError("Configured API-key environment variable is unset")
        self.secret_pattern = None
        if self.secret:
            parts = []
            for character in self.secret:
                unicode_escape = "\\u%04x" % ord(character) if ord(character) <= 0xffff else json.dumps(character, ensure_ascii=True)[1:-1]
                parts.append("(?:" + re.escape(character) + "|(?i:" + re.escape(unicode_escape) + "))")
            self.secret_pattern = re.compile("".join(parts))

    def _safe(self, value):
        """Preserve wire evidence except an exact configured credential, if echoed."""
        clone = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

        def redact(item):
            if isinstance(item, str):
                return self.secret_pattern.sub("[REDACTED]", item) if self.secret_pattern else item
            if isinstance(item, list):
                return [redact(child) for child in item]
            if isinstance(item, dict):
                return {redact(key): redact(child) for key, child in item.items()}
            return item

        return redact(clone)

    def complete(self, messages, *, request_id, role, max_output_tokens=4096):
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request_id):
            raise ValueError("request_id must be a safe unique file identifier")
        if not isinstance(messages, list) or not messages or any(not isinstance(message, dict) for message in messages):
            raise ValueError("messages must be a nonempty list of objects")
        request_body = {"model": self.model, "messages": messages, "stream": True,
                        "stream_options": {"include_usage": True}, "max_tokens": max_output_tokens}
        for name in ("temperature", "top_p", "seed"):
            if name in self.config:
                request_body[name] = self.config[name]
        encoded = json.dumps(request_body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        input_booking = self.config.get("input_token_reservation")
        basis = "configured_conservative_reservation"
        byte_booking = len(encoded) + 512 * len(messages)
        has_image = any(isinstance(message.get("content"), list) and any(
            isinstance(block, dict) and block.get("type") == "image_url" for block in message["content"])
            for message in messages)
        if input_booking is None:
            input_booking = byte_booking
            basis = "utf8_bytes_plus_template_allowance"
        elif isinstance(input_booking, bool) or not isinstance(input_booking, int) or input_booking < 0:
            raise ValueError("input_token_reservation must be a nonnegative integer or null")
        elif not has_image:
            input_booking = max(input_booking, byte_booking)
            basis = "configured_and_utf8_conservative_max"
        request_path = self.log_dir / (request_id + ".request.json")
        response_path = self.log_dir / (request_id + ".response.json")
        if request_path.exists() or response_path.exists():
            raise ProviderError("Request evidence identity already exists")
        self.ledger.reserve(request_id, input_booking, max_output_tokens, role)
        raw_wire, raw_text, usage = bytearray(), "", None
        started = time.time_ns()
        deadline = time.monotonic() + self.timeout
        settled = False
        try:
            _atomic_json(request_path, self._safe({"request_id": request_id, "role": role,
                "model": self.model, "endpoint": self.endpoint, "started_ns": started,
                "input_booking_basis": basis, "reserved_input_tokens": input_booking, "body": request_body}))
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            if self.secret:
                headers["Authorization"] = "Bearer " + self.secret
            request = Request(self.endpoint, data=encoded, headers=headers, method="POST")
            normal_stop, done = False, False
            opener = build_opener(_NoRedirect())
            try:
                response = opener.open(request, timeout=self.timeout)
            except HTTPError as exc:
                for _ in _bounded_chunks(exc, raw_wire, self.max_response_bytes, deadline):
                    pass
                raise ProviderError("HTTP status " + str(exc.code)) from exc
            with response:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    for _ in _bounded_chunks(response, raw_wire, self.max_response_bytes, deadline):
                        pass
                    raise ProviderError("Expected streaming text/event-stream response")
                data_lines = []
                for line in _stream_lines(response, raw_wire, self.max_response_bytes, deadline):
                    decoded = line.decode("utf-8").rstrip("\r\n")
                    if decoded.startswith("data:"):
                        data_lines.append(decoded[5:].lstrip(" "))
                    if decoded or not data_lines:
                        continue
                    data = "\n".join(data_lines)
                    data_lines = []
                    if data == "[DONE]":
                        done = True
                        break
                    event = _strict_object(data)
                    if event.get("error") is not None:
                        raise ProviderError("Provider returned a streaming error")
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    choices = event.get("choices", [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise ProviderError("Expected exactly one generation choice")
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("index", 0) != 0:
                        raise ProviderError("Unexpected generation choice index")
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict) or delta.get("tool_calls") or delta.get("function_call"):
                        raise ProviderError("Expected visible JSON text, not tool calls")
                    content = delta.get("content")
                    if content is not None:
                        if normal_stop or not isinstance(content, str):
                            raise ProviderError("Invalid or post-finish content")
                        raw_text += content
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        if finish != "stop" or normal_stop:
                            raise ProviderError("Generation did not finish with a single normal stop")
                        normal_stop = True
            if not normal_stop or not done:
                raise ProviderError("Truncated stream: normal stop and [DONE] are required")
            payload = _strict_object(raw_text)
            measured = _usage(usage)
            overrun = measured is not None and (measured["input"] > input_booking or measured["output"] > max_output_tokens)
            overrun_error = "Provider usage exceeded the configured reservation" if overrun else None
            record = self.ledger.settle(request_id, self._safe(usage), error=overrun_error, input_booking_basis=basis, model=self.model,
                                       request_ref=str(request_path), response_ref=str(response_path))
            settled = True
            if record["reservation_overrun"]:
                raise ProviderError(overrun_error)
            result = {"payload": payload, "raw_text": raw_text, "usage": usage,
                      "request_id": request_id, "model": self.model, "role": role}
            _atomic_json(response_path, self._safe({**result, "raw_stream": raw_wire.decode("utf-8", errors="replace"),
                                                  "error": None, "finished_ns": time.time_ns()}))
            return result
        except Exception as exc:
            error = str(exc)
            if self.secret:
                error = error.replace(self.secret, "[REDACTED]")
            if not settled:
                try:
                    self.ledger.settle(request_id, self._safe(usage), error=error, input_booking_basis=basis, model=self.model,
                                       request_ref=str(request_path), response_ref=str(response_path))
                except ValueError:
                    self.ledger.settle(request_id, None, error=error + "; unusable provider usage",
                                       input_booking_basis=basis, model=self.model,
                                       request_ref=str(request_path), response_ref=str(response_path))
            _atomic_json(response_path, self._safe({"request_id": request_id, "role": role, "model": self.model,
                "raw_text": raw_text, "raw_stream": raw_wire.decode("utf-8", errors="replace"),
                "usage": usage, "error": error, "finished_ns": time.time_ns()}))
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(error) from exc
