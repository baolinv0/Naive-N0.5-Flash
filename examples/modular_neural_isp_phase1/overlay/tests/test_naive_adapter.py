import json
import sys
import threading
import types
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from tm_research.naive_adapter import make_server, parse_assistant_text, prepare_messages, transformers_generator


class NaiveProtocolTests(unittest.TestCase):
    def test_parses_multiline_tool_arguments_and_plain_answer(self):
        tools = [{"type": "function", "function": {"name": "run_baseline", "parameters": {
            "type": "object", "properties": {"config": {"type": "object"}, "note": {"type": "string"}}
        }}}]
        content, calls = parse_assistant_text(
            '<think>private</think>Starting.\n<tool_call>\n<function=run_baseline>\n'
            '<parameter=config>{"epochs": 2}</parameter>\n'
            '<parameter=note>first line\nsecond line</parameter>\n</function>\n</tool_call><|im_end|>',
            tools,
        )
        self.assertEqual(content, "Starting.")
        self.assertEqual(calls, [("run_baseline", {"config": {"epochs": 2}, "note": "first line\nsecond line"})])
        self.assertEqual(parse_assistant_text("Done.<|im_end|>"), ("Done.", []))

    def test_thinking_continuation_ends_before_visible_answer(self):
        # A generation prompt can leave the assistant inside its thinking span.
        self.assertEqual(parse_assistant_text("planning steps\n</think>Answer.<|im_end|>"),
                         ("Answer.", []))

    def test_headers_and_heartbeat_arrive_during_generation(self):
        release = threading.Event()

        def generate(_request):
            release.wait(timeout=2)
            yield "Ready.<|im_end|>"

        server = make_server("127.0.0.1", 0, generate, heartbeat_interval=0.05)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            body = b'{"stream":true,"messages":[{"role":"user","content":"Hello"}]}'
            with urlopen(Request(url, body, {"Content-Type": "application/json"}), timeout=2) as response:
                self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                self.assertIn(b'"role": "assistant"', response.readline())
                response.readline()
                self.assertIn(b": generating", response.readline())
                release.set()
                self.assertIn(b"[DONE]", response.read())
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_history_converts_openai_json_arguments_for_official_template(self):
        messages = [
            {"role": "user", "content": "Run"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "run_baseline", "arguments": '{"config":{"epochs":2}}'
            }}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"run_id":"exp_001"}'},
        ]
        normalized = prepare_messages(messages)
        self.assertEqual(normalized[1]["tool_calls"][0]["function"]["arguments"], {"config": {"epochs": 2}})
        self.assertEqual(normalized[1]["content"], "")
        self.assertEqual(normalized[2]["role"], "tool")
        self.assertEqual(normalized[2]["content"], '{"run_id":"exp_001"}')

    def test_zero_temperature_uses_greedy_generation(self):
        generated_with = []

        class Inputs(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return Inputs(input_ids=types.SimpleNamespace(shape=(1, 2)))

            def decode(self, _tokens, skip_special_tokens):
                self.assertFalse(skip_special_tokens)
                return "Ready.<|im_end|>"

            def assertFalse(self, value):
                if value:
                    raise AssertionError("special tokens were stripped before parsing")

        class Output:
            def __getitem__(self, _index):
                return [1]

        class Model:
            device = "cpu"

            def generate(self, **kwargs):
                generated_with.append(kwargs)
                return Output()

        fake = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Tokenizer()),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Model()),
        )
        with patch.dict(sys.modules, {"transformers": fake}):
            generate = transformers_generator("local")
            self.assertEqual(list(generate({"messages": [{"role": "user", "content": "Hi"}],
                                            "temperature": 0, "top_p": 0.7})), ["Ready.<|im_end|>"])
        self.assertIs(generated_with[0]["do_sample"], False)
        self.assertNotIn("temperature", generated_with[0])
        self.assertNotIn("top_p", generated_with[0])

    def test_streamed_http_tool_loop_with_injected_generation(self):
        seen = []

        def generate(request):
            seen.append(request)
            if request["messages"][-1]["role"] == "tool":
                yield "Run exp_001 finished.<|im_end|>"
            else:
                yield "<tool_call>\n<function=run_baseline>\n"
                yield '<parameter=config>{"epochs":2}</parameter>\n</function>\n</tool_call><|im_end|>'

        server = make_server("127.0.0.1", 0, generate, "Naive-N0.5-Flash")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

        def post(messages):
            body = json.dumps({
                "model": "Naive-N0.5-Flash", "stream": True, "messages": messages,
                "tools": [{"type": "function", "function": {
                    "name": "run_baseline",
                    "parameters": {"type": "object", "properties": {"config": {"type": "object"}}},
                }}],
            }).encode()
            with urlopen(Request(url, body, {"Content-Type": "application/json"}), timeout=5) as response:
                self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                lines = response.read().decode().splitlines()
            self.assertEqual([line for line in lines if line][-1], "data: [DONE]")
            return [json.loads(line[6:]) for line in lines if line.startswith("data: {")]

        try:
            first = post([{"role": "user", "content": "Run two epochs"}])
            tool_call = next(chunk["choices"][0]["delta"]["tool_calls"][0] for chunk in first
                             if chunk["choices"][0]["delta"].get("tool_calls"))
            self.assertEqual(json.loads(tool_call["function"]["arguments"]), {"config": {"epochs": 2}})
            self.assertEqual(first[-1]["choices"][0]["finish_reason"], "tool_calls")
            second = post([{"role": "user", "content": "Run two epochs"},
                           {"role": "assistant", "tool_calls": [tool_call], "content": None},
                           {"role": "tool", "tool_call_id": tool_call["id"], "content": '{"run_id":"exp_001"}'}])
            self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in second),
                             "Run exp_001 finished.")
            self.assertEqual(second[-1]["choices"][0]["finish_reason"], "stop")
            self.assertEqual(seen[-1]["messages"][1]["tool_calls"][0]["function"]["arguments"],
                             {"config": {"epochs": 2}})
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
