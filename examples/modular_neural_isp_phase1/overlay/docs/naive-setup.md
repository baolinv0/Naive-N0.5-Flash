# ARIS-Code and Naive-N0.5-Flash

The optional adapter in `tm_research/naive_adapter.py` exposes streamed OpenAI Chat Completions at `http://127.0.0.1:8000/v1/chat/completions`. It translates Naive's [official chat template](https://huggingface.co/NaiveAI/Naive-N0.5-Flash/blob/main/chat_template.jinja) tool blocks into `tool_calls`, and converts prior OpenAI JSON argument strings into mappings before `apply_chat_template`. The route requires `stream: true`, as used by ARIS-Code.

If your Naive serving runtime **already** offers a compatible streaming `/v1/chat/completions` endpoint with structured tool calls and tool-result history, use that service directly. Set `executor_base_url` to its `/v1` base and `executor_model` to its served model name. The local adapter is only needed when starting with raw Transformers inference.

For the local path, obtain the [official FP8 model](https://huggingface.co/NaiveAI/Naive-N0.5-Flash-FP8) and an environment with its supported Transformers stack and FP8-capable NVIDIA GPUs. The [model card](https://huggingface.co/NaiveAI/Naive-N0.5-Flash/blob/main/README.md) recommends `transformers[torch,kernels]>=5.17.0` and says the weights occupy about 315 GB, plus inference memory. Install the model's dependencies separately from the ISP training environment. From this repository root, start:

```sh
python -m tm_research.naive_adapter --model-id /path/to/Naive-N0.5-Flash-FP8 --host 127.0.0.1 --port 8000
```

The default model ID is `NaiveAI/Naive-N0.5-Flash-FP8` if the weights are available through Hugging Face. The server loads weights at startup. It sends SSE headers promptly and emits heartbeat comments during inference. It buffers one model answer to distinguish text from tool syntax, then emits SSE deltas, a `finish_reason` (`stop`, `tool_calls`, or `length`), and `[DONE]`; it does not stream individual generated tokens as they arrive. The Transformers callback checks generated token count and the configured EOS IDs, so a token limit remains `length`, including when the last complete text resembles a tool call. An EOS at the token limit remains `stop` (or `tool_calls` for a valid call). Setting `temperature: 0` selects greedy generation.

Unfinished reasoning and tool blocks are hidden. The adapter also detects when the generated prompt already opened a reasoning span. Replies that reach the token limit emit no executable tool calls. Complete calls must name a declared tool, include its required arguments, and match the declared JSON types; nested object properties and array items are checked, as is `additionalProperties: false`. Invalid complete calls return an SSE error without emitting any calls. This is a small validator for these schema fields, not a full JSON Schema implementation.

For local protocol tests, `make_server` accepts a callback yielding text fragments. These callbacks default to normal completion; to report a real token limit or a prompt that starts inside reasoning, yield `GenerationResult(text, finish_reason, starts_in_think)` from `tm_research.naive_adapter`. The built-in Transformers callback supplies this metadata.

Merge the fields from [`configs/aris.example.json`](../configs/aris.example.json) into ARIS-Code's `~/.config/aris/config.json`, adjusting URL and model. The example key is a local placeholder; use the credential required by your chosen service. ARIS-Code's OpenAI-compatible executor uses `executor_provider`, `executor_api_key`, `executor_base_url`, and `executor_model`. Keep any separate reviewer or tool settings you already use. Configure the ARIS process with access to this repository and the same Python environment as the runner. Set real paths and a fixed budget in `configs/baseline.example.yaml`, then invoke the task-specific controller workflow:

```bash
aris "Read docs/aris-campaign-task.md and execute it. Config: configs/baseline.example.yaml. Campaign: runs/campaign. Max trials: 5. No-gain limit: 3. Use the configured Naive executor."
```

The task calls `campaign init`, reads `campaign next`, writes one feedback-informed proposal, calls `campaign submit`, and repeats after reading the actual DEV result until a stopping budget is reached. It then calls `campaign finalize`. TEST is available only through the separate `final-test` command after that choice is frozen. See [`docs/tm_research_task.md`](tm_research_task.md) for the exact CLI and proposal schema. Use `--wait` for hosts that terminate background children on tool return; otherwise a detached worker plus bounded `campaign wait`/`status` calls allows the ARIS session to resume collection later.

You can check the protocol endpoint independently with a small request:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"NaiveAI/Naive-N0.5-Flash-FP8","stream":true,"messages":[{"role":"user","content":"Say hello"}]}'
```

The local test `python -m unittest discover -s tests -p test_naive_adapter.py -v` uses injected generators and a mock Transformers model to check parsing, required/type validation, truncated tool parameters and tags, incomplete reasoning, token-limit/EOS classification, greedy generation, JSON tool history, SSE tool calls, a tool response, and a final answer. It cannot establish that real Naive weights or a live ARIS installation work. End-to-end acceptance still requires a real Naive service driving the configured ISP command and reading its result.
