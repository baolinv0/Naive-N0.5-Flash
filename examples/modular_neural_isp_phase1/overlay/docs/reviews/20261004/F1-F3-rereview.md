# F1/F3 scoped final re-review

Date: 2026-10-04. Scope: the three findings in `F1-F3-review.md`, their corrections, and directly introduced regressions. Reviewed stable current source and the owners' appended reports. No production edits or full-suite rerun.

**Verdict: approve within this scoped review. All three original Important findings are addressed. No open Critical, Important, or Minor findings from this re-review.**

| Original finding | Verdict | Verified correction |
| --- | --- | --- |
| Continuation could approve a different recipe from queued work | Addressed | `_persist_decision` compares the continuation's `requested_change` with the immutable queued `proposal.recipe` before clearing pending intent. Both launch-intent and prepared-active regression cases reject mismatch without modifying the pending request and permit a matching continuation through the recovered-start authorization. |
| A newer scope-change stop was accepted but ignored | Addressed | A new slow-review action differing from an existing terminal action is explicitly rejected before snapshot, JSONL, or campaign persistence. The regression verifies byte-identical state/log, no rejected snapshot, retained terminal intent, and explicit freeze compatibility. Old proposal replay still cannot clear a newer stop. This implements the explicit-rejection option offered in the initial review. |
| Confirmation/native config could jointly diverge from frozen parent | Addressed | Expected scientific config is derived from validated parent configuration with the permitted winner recipe, predeclared seed and runs directory. DEV count normalization is bound to the frozen native parent result. Both manifest and native scientific configs are checked, and selected resolved config is retained in immutable reference provenance. Initial adoption and frozen-reference reuse both reject the jointly changed epoch count. |

## Verification

Executed from `examples/modular_neural_isp_phase1/overlay`:

```sh
../../../.venv/bin/python -m pytest -q \
  tests/test_slow_decisions.py::test_pending_continuation_must_approve_the_immutable_queued_patch \
  tests/test_slow_decisions.py::test_conflicting_terminal_review_is_rejected_atomically_before_explicit_freeze \
  tests/test_slow_decisions.py::test_replaying_original_propose_cannot_clear_newer_pending_stop \
  tests/test_final_checkpoint.py::test_confirmation_and_actual_run_cannot_rewrite_parent_training_conditions \
  tests/test_final_checkpoint.py::test_predeclared_checkpoint_is_frozen_reported_and_actually_evaluated \
  tests/test_final_checkpoint.py::test_legacy_final_test_passes_predeclared_checkpoint_to_command \
  tests/test_final_checkpoint.py::test_confirmation_reporting_does_not_adopt_under_parent_lock
```

Result: **9 passed in 0.14s**. `git diff --check` passed.

The adjacent positive cases verify controlled and legacy evaluator checkpoint identity, repeated evaluation reuse, and read-only parent reporting under an already-held parent lock. Source inspection confirms native runner result state already persists `expected_count` before successful DEV completion, so the new count binding does not require an invented field or a TEST scan.

These are persisted native fixtures and bounded evaluation/launch-boundary tests. No live model, training process, real TEST data, or production runner workaround was used. Full-suite status and the existing PID-namespace integration limitation remain owned by the root review and are not represented as resolved here.
