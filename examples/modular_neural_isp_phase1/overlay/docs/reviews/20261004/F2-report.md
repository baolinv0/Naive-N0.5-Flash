# F2 implementation report — 2026-10-04

Implemented the smallest supported CLI/adapter capture → conversion → registration path. No agent framework, weight hashing platform, real Naive inference, GPU run, or real TEST access was added/performed.

## Files owned

- `overlay/tm_research/workflow.py`: capture ordinary CLI calls, retain inputs/results, derive ordered events, register manifest under campaign lock, validate preserved raw records and rederive index.
- `overlay/tm_research/cli.py`: `capture-workflow`, `register-workflow`; also coordinated F3 `campaign final-checkpoint [--confirmation-dir]` dispatch.
- `overlay/tm_research/naive_adapter.py`: optional `--workflow-dir` / `--host-version`; startup record only after actual loader returns and HTTP bind succeeds; raw request/generation capture, excluding authentication headers and unrelated request-body fields.
- `overlay/tests/test_workflow.py`: normal CLI/HTTP fixture path, partial evidence, reordered/mismatched/mutated raw, attempted event-index-as-raw, binding, repeated registration, failed load/bind, credential-header exclusion.
- W-success portion of `overlay/tests/test_research.py`: migrated from manually authored event index and campaign provenance to the capture/register path.
- `overlay/docs/workflow-evidence.md`: supported live commands, exact host JSON message/response contract, unsupported native ARIS export formats and integrity/engineering boundaries.

F1 owner added the call to this API at the beginning of `_w_claim`'s `try` block; the existing W run/decision/feedback/metric and ordered-event checks remain in place:

```python
validate_workflow_evidence(state, directory, provenance) -> list[dict]
```

The API raises `ValueError` for absent, malformed, altered or unsupported provenance. Registered manifest preserves raw source text plus source locations, and report-time validation verifies raw prefixes/identity and rederives the index. Append-only later service records cannot silently supply events to an older registered evidence scope. No arbitrary accepted-event JSONL import exists.

A consumption event requires exact captured returned feedback in a later actual adapter model request, the model's actual JSON `{proposal, decision}` response, and a later matching actual CLI submit. Submitted proposal and decision are checked against the real campaign trial/snapshot. A feedback pathname alone fails. Completion is derived only from terminal valid trial records actually returned by a captured command; no score/gain is rewritten.

## Validation chronology and exact commands

Working directory unless specified: `/workspace/scratch/6c78134e8d18/Naive-N0.5-Flash`.

1. Before implementation, initial default-Python and `/usr/bin/python3 -m pytest -q tests/test_workflow.py` attempts could not run because those interpreters lacked pytest. **These are environment failures, not meaningful regression RED evidence.** Root restored `.venv/bin/python` afterward.
2. After writing initial code, copied original `HEAD:examples/modular_neural_isp_phase1/overlay/tm_research/*.py` into `/workspace/scratch/6c78134e8d18/review-fixes-20261004/f2-red/tm_research`. A first attempt with the repository pytest config mistakenly imported new overlay modules and ran 15 new tests: **14 failed, 1 passed** due to production worker exit 125. The actual issue was `os.getpid()` returning namespace PID 2 while `/proc/self/stat` described a different PID, yielding `worker_identity=null`; production process guards correctly refused unverified ownership. This invocation is not claimed as baseline RED.
3. Isolated original-source import RED:

```bash
PYTHONPATH=/workspace/scratch/6c78134e8d18/review-fixes-20261004/f2-red .venv/bin/python -m pytest -q -c /dev/null /workspace/scratch/6c78134e8d18/review-fixes-20261004/f2-red/test_workflow.py
```

Result: **1 collection error**, `ModuleNotFoundError: tm_research.workflow`.

4. Behavioral CLI feature RED on original source:

```bash
PYTHONPATH=/workspace/scratch/6c78134e8d18/review-fixes-20261004/f2-red .venv/bin/python -m pytest -q -c /dev/null /workspace/scratch/6c78134e8d18/review-fixes-20261004/f2-red/test_cli_feature.py
```

Result: **2 failed**, parser rejects unknown `capture-workflow` and `register-workflow` subcommands with `SystemExit: 2`.

5. Same CLI tests on changed source (importlib mode prevents the copied baseline test directory from shadowing PYTHONPATH):

```bash
PYTHONPATH=/workspace/scratch/6c78134e8d18/Naive-N0.5-Flash/examples/modular_neural_isp_phase1/overlay .venv/bin/python -m pytest -q -c /dev/null --import-mode=importlib /workspace/scratch/6c78134e8d18/review-fixes-20261004/f2-red/test_cli_feature.py
```

Result: **2 passed in 0.02s**. An earlier attempt without `--import-mode=importlib` still imported the copied baseline and correctly remained RED; it was corrected rather than counted as a changed-source result.

6. Root-approved engineering test helper substitutes synchronous process supervision and invokes actual tiny train/eval subprocesses through `subprocess.run`, leaving production runner safeguards unchanged. Initial focused implementation test:

```bash
.venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py -x
```

Result: **15 passed in 10.68s**.

7. Adapter + migrated W tests:

```bash
.venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_naive_adapter.py examples/modular_neural_isp_phase1/overlay/tests/test_research.py -k 'naive_adapter or w_claim or recognized_w'
```

Result: **13 passed, 15 deselected, 19 subtests passed in 2.44s**.

8. After expanded ordering/tamper/partial/startup tests, final targeted command:

```bash
.venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py examples/modular_neural_isp_phase1/overlay/tests/test_naive_adapter.py examples/modular_neural_isp_phase1/overlay/tests/test_research.py -k 'workflow or naive_adapter or w_claim or recognized_w'
```

Result: **35 passed, 15 deselected, 19 subtests passed in 17.18s**.

`git diff --check` also passed. Root owns full-suite execution and reporting.

## Limits

- The accepted-branch fixture explicitly mocks the successful model-load boundary with `ENGINEERING_TEST_DOUBLE`. Its adapter HTTP, CLI, conversion, registration and W validator calls are real, but model/train/eval are substitutes. This is engineering validation, not live Naive/ARIS evidence.
- The separate injected-generator case stays `engineering_fixture` and W pending. Failed loader/bind produces no service startup evidence.
- The synchronous test launcher excludes production watchdog/supervision validation. Existing production runner tests were not weakened; the host PID-namespace failure remains a distinct environment limitation.
- Supported host contract is exact parsed CLI-result JSON in a model tool/user message and a JSON `{proposal, decision}` visible response. Unknown ARIS exports, terminal transcripts, markdown-wrapped answers, or arbitrary shell-write tool calls remain unsupported/pending.
- Evidence integrity assumes a trusted operator/filesystem. Raw snapshot comparison catches changes; it is not cryptographic attestation against an operator fabricating all original sources and manifest.
- Weights identity records actual model load argument/classes/available revision, not a digest of all weight bytes.

Final fixture cleanup refinement: supervision monkeypatches are installed only after the test HTTP server binds, so a failed bind cannot leave unrelated tests patched. Rechecked with:

```bash
.venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py -k 'normal_cli_capture or failed_service_bind or failed_model_load'
```

Result: **3 passed, 19 deselected in 0.89s**.

## Independent-review corrections — 2026-10-04

Read `F2-review.md` in full and addressed both Important findings and the Minor finding within the existing owned files.

- Interrupted registration: an identical manifest no longer returns before ensuring the derived event index and campaign association exist and match. Retry restores missing/changed derived index and saves the association if absent/different. A normal identical retry does not rewrite campaign state. A fault-injection regression raises `OSError` during the association save after manifest persistence, removes the derived index, retries, and verifies full recovery and stable repeated registration without editing campaign JSON.
- Actual loaded weights: `make_server` keeps the served alias (`model_id`) distinct and derives `weights_ref` from the recorded loader `model_path`. Live evidence requires nonempty model path, model class and tokenizer class; optional revision is typed. Service startup and converter both validate the load metadata and `weights_ref` binding. Arbitrary truthy metadata cannot enable live capture; unrelated metadata fields are not retained. Engineering fixtures now simulate a coherent model/tokenizer/path identity. Regressions cover alias versus actual path, inconsistent declarations, malformed loader metadata and exclusion of an extra credential-like field.
- Typed CLI errors: malformed raw argument lists are rejected explicitly, and registration normalizes malformed JSON/container/index/type failures to `ValueError`, which the CLI returns as JSON error/exit 1. Parameterized CLI tests cover empty/null/string argument values and an invalid top-level raw record container, without producing any registration manifest or association.
- Documentation now includes operator-filled `--control configs/control.local.json` for pilot initialization, synchronized clocks for cross-machine capture, interrupted-registration retry, and explicit display-alias/actual-load-path semantics.

RED was run before changing production implementation:

```bash
.venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py -k 'retry_repairs or actual_loaded_weights or different_from_actual_load or malformed_loaded_identity or reports_malformed_raw' > /workspace/scratch/6c78134e8d18/review-fixes-20261004/f2-review-red.log 2>&1
```

Result: **9 failed, 1 passed, 22 deselected in 4.03s**. All requested bugs reproduced; the string-form malformed argv case already failed through the typed error path, accounting for the passing case.

After implementation, the identical pytest selection without output redirection returned **10 passed, 22 deselected in 3.94s**.

Final combined F2 validation after the metadata-exclusion assertion and documentation corrections:

```bash
.venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py examples/modular_neural_isp_phase1/overlay/tests/test_naive_adapter.py examples/modular_neural_isp_phase1/overlay/tests/test_research.py -k 'workflow or naive_adapter or w_claim or recognized_w'
```

Result: **45 passed, 15 deselected, 19 subtests passed in 20.88s**. `git diff --check` passed. These remain the explicitly labelled engineering-only fixtures described above; no live model/GPU/real TEST or production supervisor success is claimed.
