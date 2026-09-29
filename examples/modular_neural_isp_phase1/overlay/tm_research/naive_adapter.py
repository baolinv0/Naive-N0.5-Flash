"""Small OpenAI streaming Chat Completions bridge for Naive's chat template.

Run ``python -m tm_research.naive_adapter --model-id /path/to/weights``.
The protocol helpers and injected-generator server need only the standard library.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from queue import Empty, Queue
import re
from threading import Thread
import time
from uuid import uuid4


TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
PARAMETER = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)


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


def parse_assistant_text(raw, tools=()):
    """Return visible text and (name, argument mapping) pairs from a Naive reply."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    properties = {
        tool["function"]["name"]: tool["function"].get("parameters", {}).get("properties", {})
        for tool in tools or () if "function" in tool
    }
    calls = []
    for match in TOOL_CALL.finditer(raw):
        name = match.group(1).strip()
        arguments = {}
        for parameter in PARAMETER.finditer(match.group(2)):
            key, value = parameter.group(1).strip(), parameter.group(2)
            kind = properties.get(name, {}).get(key, {}).get("type")
            if kind in ("object", "array", "integer", "number", "boolean"):
                value = json.loads(value)
            arguments[key] = value
        calls.append((name, arguments))
    return TOOL_CALL.sub("", raw).strip(), calls


def make_server(host, port, generate, model_name="Naive-N0.5-Flash", heartbeat_interval=10):
    """Build an HTTP server; generate(request) yields raw decoded model fragments."""

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
                raw_parts.append(piece)
            # Parse the complete reply before content deltas so tool XML is not shown.
            content, calls = parse_assistant_text("".join(raw_parts), request.get("tools", []))
            if content:
                send({"content": content})
            for index, (name, arguments) in enumerate(calls):
                send({"tool_calls": [{"index": index, "id": "call_" + uuid4().hex,
                                      "type": "function", "function": {"name": name,
                                      "arguments": json.dumps(arguments, ensure_ascii=False)}}]})
            send({}, "tool_calls" if calls else "stop")
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
        # Preserve Naive's end token until parsing, rather than stripping it early.
        yield tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)

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
