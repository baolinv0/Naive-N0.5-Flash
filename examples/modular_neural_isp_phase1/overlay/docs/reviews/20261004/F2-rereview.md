# F2 scoped re-review — 2026-10-04

Verdict: **approved within the correction scope**. The two Important findings and one Minor finding from `F2-review.md` are resolved. No new Critical, Important, or Minor finding was identified in these corrections.

Scope was limited to the previous three findings and breakage introduced by their fixes: current `workflow.py`, adapter startup integration, the corresponding new workflow tests and documentation, and the appended F2 implementation report. This is not a new broad review or a full-suite result.

## Resolution checks

1. **Interrupted registration recovery:** registration no longer returns early for an identical manifest. It rederives the index, restores missing/changed index content, and saves a missing or differing campaign association before returning. A normal identical retry avoids rewriting campaign state. The fault-injection regression verifies manifest persistence followed by failed association save, missing index, successful retry, accepted engineering W, and another stable retry. No manual campaign-state edit is used.
2. **Actual weights identity:** adapter startup derives `weights_ref` from validated loader `model_path`, keeping `model_id` as a legitimate display alias. Live startup and conversion require model path/model class/tokenizer class and validate the declared weights against the load path. The actual `transformers_generator` obtains those fields after the loader returns. Arbitrary truthy metadata is rejected, and unrelated metadata is removed. The regression validates a distinct alias/load path and rejection when both service/startup declarations disagree with the retained load argument.
3. **Malformed raw evidence errors:** registration normalizes `KeyError`, `TypeError`, `IndexError`, and `AttributeError` into `ValueError`; malformed argv shapes have an explicit check. JSON decoding failures already derive from `ValueError`. The existing CLI error boundary returns JSON and exit 1. Tests cover empty/null/string argv and an invalid top-level record container, with no manifest or campaign association created.

The revised documentation accurately describes retry recovery, alias/load-path semantics and the engineering-only validation boundary. The correction preserves strict raw-source checks; it repairs derived state without relaxing acceptance based on source declarations.

## Independently executed narrow validation

Working directory: `/workspace/scratch/6c78134e8d18/Naive-N0.5-Flash`.

```bash
.venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py -k 'retry_repairs or actual_loaded_weights or different_from_actual_load or malformed_loaded_identity or reports_malformed_raw'
```

Result: **10 passed, 22 deselected in 3.89s**.

No production sources were edited, no full suite was rerun, and no agents or commits were created. The acceptance fixture still uses an explicitly labelled simulated model-load boundary, tiny train/eval substitutes and synchronous test-only supervision. This review establishes no real Naive/ARIS quality, GPU/real TEST result, or production-runner/watchdog verification.
