"""Process-safe model reservations; provider usage and conservative charges differ.

Token reservations are booking units, not measurements. Missing provider usage
keeps actual amounts null and consumes the full reservation, including failures.
Completion tokens already include reasoning tokens in Chat Completions usage.
"""
from contextlib import contextmanager
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time


class LedgerQuotaError(ValueError):
    """A request would exceed a frozen model budget."""


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _money(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return float(value)


def _limits(value):
    if not isinstance(value, dict):
        raise ValueError("Model limits must be an object")
    result = {name: _integer(value.get(name), name, 1) for name in
              ("max_calls", "max_input_tokens_per_call", "max_output_tokens_per_call")}
    for name in ("max_total_input_tokens", "max_total_output_tokens"):
        result[name] = None if value.get(name) is None else _integer(value[name], name)
    total = value.get("max_total_tokens")
    if total is None:
        if result["max_total_input_tokens"] is None or result["max_total_output_tokens"] is None:
            raise ValueError("max_total_tokens or both separate total limits are required")
        total = result["max_total_input_tokens"] + result["max_total_output_tokens"]
    result["max_total_tokens"] = _integer(total, "max_total_tokens", 1)
    for name in ("max_cost_usd", "input_usd_per_million", "output_usd_per_million"):
        result[name] = None if value.get(name) is None else _money(value[name], name)
    if result["max_cost_usd"] is not None and any(result[name] is None for name in
            ("input_usd_per_million", "output_usd_per_million")):
        raise ValueError("A monetary cap requires both token prices for conservative booking")
    return result


def _usage(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Provider usage must be an object or null")
    prompt, completion = value.get("prompt_tokens"), value.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    prompt = _integer(prompt, "prompt_tokens")
    completion = _integer(completion, "completion_tokens")
    if value.get("total_tokens") is not None and _integer(value["total_tokens"], "total_tokens") != prompt + completion:
        raise ValueError("Provider total_tokens disagrees with prompt + completion")
    details = value.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        raise ValueError("completion_tokens_details must be an object")
    reasoning = details.get("reasoning_tokens")
    if reasoning is not None:
        reasoning = _integer(reasoning, "reasoning_tokens")
        if reasoning > completion:
            raise ValueError("Reasoning tokens must be included in completion_tokens")
    return {"input": prompt, "output": completion, "reasoning": reasoning}


def _atomic_json(path, value):
    """Replace a ledger while its independent lock remains held."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CostLedger:
    def __init__(self, path, limits, *, scope_id=None, scope_limits=None):
        self.path = Path(path).resolve()
        self.limits = _limits(limits)
        self.scope_id = scope_id
        self.scope_limits = None
        if (scope_id is None) != (scope_limits is None):
            raise ValueError("Scope identity and scope limits must be supplied together")
        if scope_id is not None:
            self._validate_scope_id(scope_id)
            if not isinstance(scope_limits, dict):
                raise ValueError("scope_limits must be an object")
            prices = ("input_usd_per_million", "output_usd_per_million")
            scoped = {**scope_limits, **{field: scope_limits.get(field, self.limits[field]) for field in prices}}
            self.scope_limits = _limits(scoped)
            if any(self.scope_limits[field] != self.limits[field] for field in prices):
                raise ValueError("Scope token prices must match global accounting rates")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as state:
            if state is None:
                state = {"schema_version": 1, "limits": self.limits, "requests": {}, "scoped_limits": {}}
                if scope_id is not None:
                    state["scoped_limits"][scope_id] = self.scope_limits
                _atomic_json(self.path, state)
            elif scope_id is not None and scope_id not in state.get("scoped_limits", {}):
                state.setdefault("scoped_limits", {})[scope_id] = self.scope_limits
                _atomic_json(self.path, state)

    @staticmethod
    def _validate_scope_id(scope_id):
        if not isinstance(scope_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", scope_id):
            raise ValueError("scope_id must be a bounded safe campaign identity")

    @contextmanager
    def _locked(self):
        with self.path.with_name(self.path.name + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text()) if self.path.exists() else None
                if state is not None and (state.get("schema_version") != 1 or
                        state.get("limits") != self.limits or not isinstance(state.get("requests"), dict) or
                        not isinstance(state.get("scoped_limits", {}), dict)):
                    raise ValueError("Ledger identity, schema, or frozen limits mismatch")
                if state is not None and self.scope_id in state.get("scoped_limits", {}) and \
                        state["scoped_limits"][self.scope_id] != self.scope_limits:
                    raise ValueError("Frozen scope limits mismatch")
                yield state
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _cost(self, input_tokens, output_tokens):
        a, b = self.limits["input_usd_per_million"], self.limits["output_usd_per_million"]
        return None if a is None or b is None else (input_tokens * a + output_tokens * b) / 1000000

    @staticmethod
    def _summary(state, scope_id=None):
        requests = [record for record in state["requests"].values()
                    if scope_id is None or record.get("scope_id") == scope_id]
        result = {"calls": len(requests), "requests": copy.deepcopy(requests)}
        for kind in ("input_tokens", "output_tokens", "tokens", "cost_usd"):
            for prefix in ("charged", "actual"):
                key = prefix + "_" + kind
                values = [item[key] for item in requests]
                result[key] = None if any(value is None for value in values) else sum(values)
        return result

    @staticmethod
    def _check_quota(summary, limits, input_tokens, output_tokens, reserved_cost, label=""):
        if summary["calls"] >= limits["max_calls"]:
            raise LedgerQuotaError(label + "Model call budget exhausted")
        for kind, count in (("input", input_tokens), ("output", output_tokens)):
            if count > limits["max_" + kind + "_tokens_per_call"]:
                raise LedgerQuotaError(label + kind + " per-call reservation exceeds limit")
            cap = limits["max_total_" + kind + "_tokens"]
            if cap is not None and summary["charged_" + kind + "_tokens"] + count > cap:
                raise LedgerQuotaError(label + kind + " total reservation exceeds limit")
        if summary["charged_tokens"] + input_tokens + output_tokens > limits["max_total_tokens"]:
            raise LedgerQuotaError(label + "Total token reservation exceeds limit")
        if limits["max_cost_usd"] is not None and summary["charged_cost_usd"] + reserved_cost > limits["max_cost_usd"]:
            raise LedgerQuotaError(label + "Monetary reservation exceeds limit")

    def reserve(self, request_id, input_tokens, output_tokens, role):
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id is required")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role is required")
        input_tokens = _integer(input_tokens, "input_tokens")
        output_tokens = _integer(output_tokens, "output_tokens", 1)
        with self._locked() as state:
            if request_id in state["requests"]:
                raise ValueError("Request identity already reserved; no automatic retry")
            summary = self._summary(state)
            reserved_cost = self._cost(input_tokens, output_tokens)
            self._check_quota(summary, self.limits, input_tokens, output_tokens, reserved_cost)
            if self.scope_id is not None:
                if self.scope_id not in state.get("scoped_limits", {}):
                    raise ValueError("Frozen scope limits missing from ledger")
                self._check_quota(self._summary(state, self.scope_id), self.scope_limits,
                                  input_tokens, output_tokens, reserved_cost, "scope " + self.scope_id + ": ")
            record = {"request_id": request_id, "scope_id": self.scope_id, "role": role, "status": "reserved", "reserved_ns": time.time_ns(),
                      "reserved_input_tokens": input_tokens, "reserved_output_tokens": output_tokens,
                      "reserved_cost_usd": reserved_cost, "charged_input_tokens": input_tokens,
                      "charged_output_tokens": output_tokens, "charged_tokens": input_tokens + output_tokens,
                      "charged_cost_usd": reserved_cost, "actual_input_tokens": None,
                      "actual_output_tokens": None, "actual_tokens": None, "actual_cost_usd": None,
                      "reasoning_tokens": None, "usage": None, "error": None, "reservation_overrun": False}
            state["requests"][request_id] = record
            _atomic_json(self.path, state)
            return copy.deepcopy(record)

    def settle(self, request_id, usage, error=None, **metadata):
        measured = _usage(usage)
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be text or null")
        # Provider logs never pass credentials, and arbitrary metadata cannot overwrite accounting.
        json.dumps(metadata, allow_nan=False)
        with self._locked() as state:
            if request_id not in state["requests"]:
                raise ValueError("Settle requires an existing reservation")
            record = state["requests"][request_id]
            if self.scope_id is not None and record.get("scope_id") != self.scope_id:
                raise ValueError("Cannot settle a different scope request")
            if set(metadata) & (set(record) | {"scope_id", "scoped_limits", "limits"}):
                raise ValueError("Metadata cannot overwrite accounting fields or scope identity")
            if record["status"] == "settled":
                if record["usage"] == usage and record["error"] == error:
                    return copy.deepcopy(record)
                raise ValueError("Settled request cannot be rewritten")
            record.update(metadata)
            record.update(status="settled", settled_ns=time.time_ns(), usage=copy.deepcopy(usage), error=error)
            if measured is not None:
                a, b = measured["input"], measured["output"]
                actual_cost = self._cost(a, b)
                record.update(actual_input_tokens=a, actual_output_tokens=b, actual_tokens=a + b,
                              actual_cost_usd=actual_cost, reasoning_tokens=measured["reasoning"])
                record.update(charged_input_tokens=a, charged_output_tokens=b, charged_tokens=a + b,
                              charged_cost_usd=actual_cost,
                              reservation_overrun=a > record["reserved_input_tokens"] or b > record["reserved_output_tokens"])
            _atomic_json(self.path, state)
            return copy.deepcopy(record)

    def summary(self, *, scope_id=None):
        if scope_id is not None:
            self._validate_scope_id(scope_id)
        with self._locked() as state:
            if scope_id is not None and scope_id not in state.get("scoped_limits", {}):
                raise ValueError("Unknown scope identity")
            result = self._summary(state, scope_id)
            if scope_id is not None:
                result.update(scope_id=scope_id, limits=copy.deepcopy(state["scoped_limits"][scope_id]))
            return result
