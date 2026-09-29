# Naive adapter review fixes

Changed `tm_research/naive_adapter.py`, `tests/test_naive_adapter.py`, and `docs/naive-setup.md`.

- Hide unfinished reasoning, including continuations from an open reasoning prompt, before parsing apparent tool calls.
- Hide truncated tool blocks. Validate complete function/parameter blocks, declared tool names, required fields, JSON types, nested properties/items, and `additionalProperties: false`. Reject invalid calls before any tool-call delta is emitted.
- Carry termination metadata from Transformers to the HTTP response. A generated-token budget exhaustion reports `length` and suppresses tool execution; configured EOS termination remains `stop`, with `tool_calls` only for valid completed calls. EOS at the exact budget is classified as EOS termination.
- Preserve `temperature: 0` as greedy generation without sampling parameters.
- Retain plain string callbacks for existing injected-generator tests; `GenerationResult` supplies explicit completion/reasoning metadata.

Validation run:

```sh
/workspace/scratch/4578f81f7dfe/isp-venv/bin/python -m unittest discover -s tests -p test_naive_adapter.py -v
```

Result: 11 tests passed. Cases cover the HTTP tool/result loop, SSE heartbeat, schema failures, reasoning truncation, truncated tool parameters and tags, HTTP token-limit propagation, EOS at the exact budget, and zero-temperature behavior.

The tests use injected output and a mock Transformers model. They do not claim a live Naive/ARIS run or real Naive weight compatibility. The validator supports the schema fields listed above; it does not implement the full JSON Schema specification. No model, training, runner, or CLI changes were made for this adapter task.
