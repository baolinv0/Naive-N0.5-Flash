import json
import sys
import threading
import types
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from tm_research.naive_adapter import (
    GenerationResult, make_server, parse_assistant_text, prepare_messages, transformers_generator,
)


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

    def test_unfinished_reasoning_never_emits_apparent_tools(self):
        tool = '<tool_call><function=run><parameter=count>1</parameter></function></tool_call>'
        self.assertEqual(parse_assistant_text("<think>still planning " + tool), ("", []))
        self.assertEqual(parse_assistant_text("still planning " + tool, starts_in_think=True), ("", []))
        self.assertEqual(parse_assistant_text("Visible.<think>still planning " + tool), ("Visible.", []))
        self.assertEqual(parse_assistant_text("planning</think>Answer.", starts_in_think=True),
                         ("Answer.", []))

    def test_validates_declared_required_and_nested_types(self):
        tools = [{"type": "function", "function": {"name": "run", "parameters": {
            "type": "object", "required": ["config"], "additionalProperties": False,
            "properties": {
                "config": {"type": "object", "required": ["epochs"],
                           "properties": {"epochs": {"type": "integer"}}},
                "enabled": {"type": "boolean"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
        }}}]
        valid_config = '<parameter=config>{"epochs":2}</parameter>'
        invalid_parameters = [
            "", '<parameter=config>1</parameter>', '<parameter=config>[]</parameter>',
            '<parameter=config>{}</parameter>', '<parameter=config>{"epochs":true}</parameter>',
            '<parameter=config>{"epochs":2.5}</parameter>',
            valid_config + '<parameter=enabled>1</parameter>',
            valid_config + '<parameter=labels>["ok", 1]</parameter>',
            valid_config + '<parameter=extra>unexpected</parameter>',
            '<parameter=config>{"epochs":2}',
            valid_config + valid_config,
        ]
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                parse_assistant_text(f"<tool_call><function=run>{parameters}</function></tool_call>", tools)
        with self.assertRaisesRegex(ValueError, "undeclared tool"):
            parse_assistant_text("<tool_call><function=unknown></function></tool_call>", tools)
        self.assertEqual(parse_assistant_text(
            f"<tool_call><function=run>{valid_config}<parameter=enabled>false</parameter>"
            '<parameter=labels>["first"]</parameter></function></tool_call>', tools,
        ), ("", [("run", {"config": {"epochs": 2}, "enabled": False, "labels": ["first"]})]))

    def test_truncated_tool_block_is_hidden(self):
        truncated_blocks = [
            "<tool_ca",
            '<tool_call><function=run><parameter=config>{"epochs":',
            '<tool_call><function=run><parameter=config>{"epochs":2}</parameter></function></tool_ca',
        ]
        for block in truncated_blocks:
            with self.subTest(block=block):
                self.assertEqual(parse_assistant_text("Starting.\n" + block), ("Starting.", []))

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

        class InputIds:
            shape = (1, 2)

            def __getitem__(self, _index):
                return [10, 11]

        class Inputs(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return Inputs(input_ids=InputIds())

            def decode(self, _tokens, skip_special_tokens):
                self.assertFalse(skip_special_tokens)
                if _tokens == [10, 11]:
                    return "<|im_start|>assistant\n"
                return "Ready.<|im_end|>"

            def assertFalse(self, value):
                if value:
                    raise AssertionError("special tokens were stripped before parsing")

        class Output:
            def __getitem__(self, _index):
                return [1]

        class Model:
            device = "cpu"
            generation_config = types.SimpleNamespace(eos_token_id=1)

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
                                            "temperature": 0, "top_p": 0.7})),
                             [GenerationResult("Ready.<|im_end|>", "stop")])
        self.assertIs(generated_with[0]["do_sample"], False)
        self.assertNotIn("temperature", generated_with[0])
        self.assertNotIn("top_p", generated_with[0])

    def test_real_generator_reports_token_limit_and_eos_at_limit(self):
        class InputIds:
            shape = (1, 2)

            def __getitem__(self, _index):
                return [10, 11]

        class Inputs(dict):
            def to(self, _device):
                return self

        class Output:
            def __init__(self, tokens):
                self.tokens = tokens

            def __getitem__(self, _index):
                return self.tokens

        class Tokenizer:
            eos_token_id = 9

            def apply_chat_template(self, *_args, **_kwargs):
                return Inputs(input_ids=InputIds())

            def decode(self, tokens, **_kwargs):
                return "<|im_start|>assistant\n<think>" if tokens == [10, 11] else "planning"

        class Model:
            device = "cpu"
            generation_config = types.SimpleNamespace(eos_token_id=[8, 9])
            tokens = [1, 2]

            def generate(self, **_kwargs):
                return Output(self.tokens)

        model = Model()
        fake = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Tokenizer()),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model),
        )
        with patch.dict(sys.modules, {"transformers": fake}):
            generate = transformers_generator("local")
            request = {"messages": [], "max_completion_tokens": 2}
            self.assertEqual(list(generate(request)), [GenerationResult("planning", "length", True)])
            model.tokens = [1, 8]
            self.assertEqual(list(generate(request)), [GenerationResult("planning", "stop", True)])
            model.tokens = [1]
            self.assertEqual(list(generate(request)), [GenerationResult("planning", "stop", True)])

    def test_http_preserves_length_and_suppresses_truncated_or_invalid_calls(self):
        call = '<tool_call><function=run><parameter=config>{"epochs":2}</parameter></function></tool_call>'
        replies = {
            "text_limit": GenerationResult("Partial answer", "length"),
            "complete_tool_at_limit": GenerationResult(call, "length"),
            "partial_tool": GenerationResult("Starting. " + call[:-3], "length"),
            "reasoning_limit": GenerationResult("planning " + call, "length", True),
            "bad_type": GenerationResult(call.replace('{"epochs":2}', "1"), "stop"),
        }

        def generate(request):
            yield replies[request["messages"][-1]["content"]]

        server = make_server("127.0.0.1", 0, generate)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            for case in replies:
                with self.subTest(case=case):
                    body = json.dumps({
                        "stream": True, "messages": [{"role": "user", "content": case}],
                        "tools": [{"type": "function", "function": {"name": "run", "parameters": {
                            "type": "object", "required": ["config"],
                            "properties": {"config": {"type": "object"}},
                        }}}],
                    }).encode()
                    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
                    with urlopen(Request(url, body, {"Content-Type": "application/json"}), timeout=5) as response:
                        lines = response.read().decode().splitlines()
                    chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
                    self.assertFalse(any(c.get("choices", [{}])[0].get("delta", {}).get("tool_calls") for c in chunks))
                    if case == "bad_type":
                        self.assertIn("must have type object", chunks[-1]["error"]["message"])
                    else:
                        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
                        self.assertEqual(lines[-2], "data: [DONE]")
                        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
                        self.assertEqual(content, {"text_limit": "Partial answer", "partial_tool": "Starting."}.get(case, ""))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

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
