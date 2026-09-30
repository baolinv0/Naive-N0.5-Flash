"""Small OpenAI streaming Chat Completions bridge for Naive's chat template.

Run ``python -m tm_research.naive_adapter --model-id /path/to/weights``.
The protocol helpers and injected-generator server need only the standard library.
"""

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from queue import Empty, Queue
import re
from threading import Thread
import time
from uuid import uuid4


TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNCTION = re.compile(r"\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*", re.S)
PARAMETER = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str
    starts_in_think: bool = False


def prepare_messages(messages):
    """Convert OpenAI tool-call argument JSON to the mapping Naive's template iterates."""
    prepared = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant":
            item["content"] = item.get("content") or ""
            if item.get("tool_calls"):
                calls = []
                for call in item["tool_calls"]:
                    function = dict(call["function"])
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool call arguments must be a JSON object")
                    function["arguments"] = arguments
                    calls.append({**call, "function": function})
                item["tool_calls"] = calls
        prepared.append(item)
    return prepared


def _validate(value, schema, path):
    """Validate the declared JSON types and required fields used by tool schemas."""
    kinds = schema.get("type", [])
    if isinstance(kinds, str):
        kinds = [kinds]
    matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool)
                  and (not isinstance(value, float) or math.isfinite(value)),
        "null": value is None,
    }
    if kinds and not any(matches.get(kind, False) for kind in kinds):
        raise ValueError(f"{path} must have type {' or '.join(kinds)}")
    if isinstance(value, dict):
        missing = set(schema.get("required", ())) - value.keys()
        if missing:
            raise ValueError(f"{path} missing required fields: {', '.join(sorted(missing))}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"{path} has undeclared field: {key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


def parse_assistant_text(raw, tools=(), *, starts_in_think=False, allow_tool_calls=True):
    """Return visible text and (name, argument mapping) pairs from a Naive reply."""
    if starts_in_think:
        raw = "<think>" + raw
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    # An unfinished reasoning span can contain apparent tool calls as private text.
    raw = raw.split("<think>", 1)[0]
    raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    schemas = {
        tool["function"]["name"]: tool["function"].get("parameters", {})
        for tool in tools or () if "function" in tool
    }
    calls = []
    for match in TOOL_CALL.finditer(raw):
        if not allow_tool_calls:
            continue
        function = FUNCTION.fullmatch(match.group(1))
        if function is None:
            raise ValueError("malformed tool function block")
        name = function.group(1).strip()
        if name not in schemas:
            raise ValueError(f"undeclared tool: {name}")
        schema = schemas[name]
        body = function.group(2)
        arguments = {}
        position = 0
        for parameter in PARAMETER.finditer(body):
            if body[position:parameter.start()].strip():
                raise ValueError(f"malformed parameters for {name}")
            key, value = parameter.group(1).strip(), parameter.group(2)
            if key in arguments:
                raise ValueError(f"duplicate parameter: {name}.{key}")
            kind = schema.get("properties", {}).get(key, {}).get("type")
            if kind and kind != "string":
                try:
                    value = json.loads(value)
                except ValueError:
                    if not isinstance(kind, list) or "string" not in kind:
                        raise ValueError(f"invalid JSON for {name}.{key}") from None
            arguments[key] = value
            position = parameter.end()
        if body[position:].strip():
            raise ValueError(f"malformed parameters for {name}")
        _validate(arguments, schema, name)
        calls.append((name, arguments))
    content = TOOL_CALL.sub("", raw)
    # Drop a trailing unfinished tool block, including a partially generated tag.
    content = content.split("<tool_call", 1)[0]
    for length in range(1, len("<tool_call")):
        if content.endswith("<tool_call"[:length]):
            content = content[:-length]
            break
    return content.strip(), calls


def make_server(host, port, generate, model_name="Naive-N0.5-Flash", heartbeat_interval=10):
    """Build a server; generate yields text fragments or a final GenerationResult."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if request.get("stream") is not True:
                    raise ValueError("stream: true is required")
                request["messages"] = prepare_messages(request["messages"])
            except (ValueError, KeyError, TypeError) as error:
                body = json.dumps({"error": {"message": str(error)}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            identifier = "chatcmpl-" + uuid4().hex
            created = int(time.time())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(delta, finish_reason=None):
                chunk = {"id": identifier, "object": "chat.completion.chunk", "created": created,
                         "model": model_name, "choices": [{"index": 0, "delta": delta,
                         "finish_reason": finish_reason}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()

            send({"role": "assistant"})
            pieces = Queue()
            end = object()

            def produce():
                try:
                    for piece in generate(request):
                        pieces.put(piece)
                except Exception as error:
                    pieces.put(error)
                finally:
                    pieces.put(end)

            Thread(target=produce, daemon=True).start()
            raw_parts = []
            finish_reason = "stop"
            starts_in_think = False
            while True:
                try:
                    piece = pieces.get(timeout=heartbeat_interval)
                except Empty:
                    self.wfile.write(b": generating\n\n")
                    self.wfile.flush()
                    continue
                if piece is end:
                    break
                if isinstance(piece, Exception):
                    self.wfile.write(b"data: " + json.dumps({"error": {"message": str(piece)}}).encode() + b"\n\n")
                    self.wfile.flush()
                    return
                if isinstance(piece, GenerationResult):
                    raw_parts.append(piece.text)
                    finish_reason = piece.finish_reason
                    starts_in_think = piece.starts_in_think
                else:
                    raw_parts.append(piece)
            # Parse the complete reply before content deltas so tool XML is not shown.
            try:
                content, calls = parse_assistant_text(
                    "".join(raw_parts), request.get("tools", []),
                    starts_in_think=starts_in_think, allow_tool_calls=finish_reason != "length",
                )
            except ValueError as error:
                self.wfile.write(b"data: " + json.dumps({"error": {"message": str(error)}}).encode() + b"\n\n")
                self.wfile.flush()
                return
            if content:
                send({"content": content})
            for index, (name, arguments) in enumerate(calls):
                send({"tool_calls": [{"index": index, "id": "call_" + uuid4().hex,
                                      "type": "function", "function": {"name": name,
                                      "arguments": json.dumps(arguments, ensure_ascii=False)}}]})
            send({}, "tool_calls" if calls else finish_reason)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return ThreadingHTTPServer((host, port), Handler)


def transformers_generator(model_id):
    """Load weights only for a real server, then return its generation callback."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", device_map="auto"
    )

    def generate(request):
        inputs = tokenizer.apply_chat_template(
            request["messages"], tools=request.get("tools", []), add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        temperature = request.get("temperature", 1.0)
        generation = {"max_new_tokens": request.get("max_completion_tokens", request.get("max_tokens", 2048))}
        if temperature == 0:
            generation["do_sample"] = False
        else:
            generation.update(do_sample=True, temperature=temperature, top_p=request.get("top_p", 0.95))
        output = model.generate(**inputs, **generation)
        generated_ids = output[0, inputs["input_ids"].shape[1]:]
        eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos_ids is None:
            eos_ids = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(eos_ids, (list, tuple)):
            eos_ids = [eos_ids]
        ended_on_eos = len(generated_ids) > 0 and int(generated_ids[-1]) in eos_ids
        finish_reason = "length" if len(generated_ids) >= generation["max_new_tokens"] and not ended_on_eos else "stop"
        prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        prompt = prompt.rsplit("<|im_start|>assistant", 1)[-1]
        starts_in_think = prompt.rfind("<think>") > prompt.rfind("</think>")
        # Preserve Naive's end token until parsing, rather than stripping it early.
        yield GenerationResult(tokenizer.decode(generated_ids, skip_special_tokens=False),
                               finish_reason, starts_in_think)

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="NaiveAI/Naive-N0.5-Flash-FP8")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.host, args.port, transformers_generator(args.model_id), args.model_id)
    print(f"Naive Chat Completions at http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
