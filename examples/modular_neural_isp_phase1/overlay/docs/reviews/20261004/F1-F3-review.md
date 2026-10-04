# F1/F3 independent scoped review — initial findings

Review date: 2026-10-04. Reviewed the F1/F3 requirements in `examples/modular_neural_isp_phase1/review/Code_Fast_Slow_Review_CN.md`, the supplied F1/F3 implementation reports, the review diff, and current source plus new fixture tests in `overlay/tests/test_slow_decisions.py` and `test_final_checkpoint.py`.

**Initial disposition: changes required.** Three Important findings, no Critical findings. This is the initial complete review; it does not approve subsequent edits. The queued-patch guard was being fixed while the review ran and is explicitly pending final verification.

All probes were bounded local fixture operations, using `PYTHONPATH=.:tests ../../../.venv/bin/python` from `examples/modular_neural_isp_phase1/overlay`. They exercised persisted campaign decisions and checkpoint resolution. No training worker, live model, TEST data, production evaluator, or full test suite was run. Production source was not edited by this reviewer. `git diff --check` passed at review time.

## Important 1 — continuation approves a different recipe from the queued experiment

Locations: `research.py::_persist_decision`, `_review_start_state`; `campaign.py::_current_start_allowed`, `_resume_launch`.

The reviewed version removes a verified unstarted request from the temporary review state and validates the slow review's `requested_change`, then clears `pending_slow_decision`. Recovery separately authorizes and starts the original immutable queued recipe. There was no equality check between the resumed review and the queued proposal.

Bounded reproduction:

1. Create the persisted `test_slow_decisions.plateau` fixture.
2. Set `launch_intent` to index 2 with request ID `queued-A`, proposal patch `{"learning_rate": 0.00005}`, and its merged recipe; set `launch_block.pending_role="launch_intent"`.
3. Record a `slow_review/diagnose` named `pause`.
4. Record a new `slow_review/propose` with matching current trigger `fixture/exp_001/valid_plateau`, but `requested_change={"optimizer":"adamw"}`.
5. Reload the campaign and call `_current_start_allowed(..., "launch_intent")`.

Observed output before the owner correction:

```json
{"review_requested_change":{"optimizer":"adamw"},"queued_patch":{"learning_rate":0.00005},"pending":null,"queued_start_allowed":true}
```

The review's prediction and falsifier therefore authorize experiment B while recovery executes experiment A. Reject a mismatching continuation while preserving the queued request; an explicitly matching review may resume it. Cover both `launch_intent` and prepared `active` paths.

**Fix in progress:** subsequent source inspection showed a new comparison against the queued `proposal.recipe` in `_persist_decision`, with two new parametrized regression cases. The corrected version has not yet received this reviewer's final verification.

## Important 2 — a newer scope-change stop is archived but does not control execution

Location: `research.py::_persist_decision`, terminal-state guard around the non-propose pending assignment (approximately lines 293–303 after the queued guard addition).

The guard preserves any existing `report_stop`, `request_scope_change`, or `finalize` pending action, even when a genuinely new and stricter slow decision is accepted. In particular, a new scope-change decision cannot supersede an earlier search-only stop. Both decisions appear in the immutable log, but the older action remains effective.

Bounded reproduction:

1. Create a persisted plateau fixture.
2. Record new `slow_review/report_stop`, ID `search_stop`.
3. Record new `slow_review/request_scope_change`, ID `scope_stop`.
4. Explicitly finalize, then call `check_action(control, state, "confirm")`.

Observed output:

```json
{"latest_logged_action":"request_scope_change","pending_action":"report_stop","confirm_allowed":true}
```

This violates the accepted scope-change instruction to wait for new authorization/new campaign: confirmation remains permitted because only the earlier search stop controls the gate. Either reject a superseding incompatible terminal decision clearly, or apply a stricter new terminal restriction while ensuring replay, diagnosis, and proposal cannot weaken an existing stop. Add regression coverage for this order and replay of the old stop.

## Important 3 — selected confirmation config is not bound to frozen parent config

Location: `confirmation.py::_final_checkpoint_candidate`, approximately lines 673–678 and 696–700.

Actual native run config is compared to `confirmation.json.config`, but that confirmation config is not compared with the already validated frozen parent scientific config. The frozen reference provenance stores `plan`, `control`, and `parent_identity`, omitting the confirmation config. Thus agreement between the manifest and native record can hide a difference from the authorized campaign, including on reuse of an existing immutable reference.

Bounded reproduction, with native `get_result` substituted by reading the completed fixture `state.json` exactly as in the new final-checkpoint tests:

1. Create `test_final_checkpoint.selected_confirmation` and resolve/freeze the selected seed reference.
2. Leave parent config, frozen identity, plan, control, task, result, and checkpoint paths unchanged.
3. Change `confirmation.json.config.epochs` from 1 to 2 and change the selected native `config.json.epochs` to 2.
4. Call `resolve_final_checkpoint` again.

Observed output:

```json
{"parent_epochs":1,"confirmation_epochs":2,"native_epochs":2,"same_frozen_reference":true}
```

This is a specific missing scientific-config/authority binding, not a request for a general tamper-proof storage framework. Derive/check the expected fixed scientific config against the validated frozen parent config, allowing only the intended winner recipe, predeclared seed, native runs directory and existing normalization. Include relevant selected config provenance if needed so immutable reuse detects a changed scientific identity. Verify both initial adoption and reuse reject this mismatch without selecting search weights or launching evaluation.

## Positive scope observations and limits

The main changes address the original defects directly: persisted pending actions gate controlled and legacy starts, recovered unstarted work is checked, collection remains possible, and terminal intent cannot be reopened by ordinary propose replay. Final checkpoint selection does not compare scores; missing, failed, duplicate and ambiguous candidates are rejected, explicit registration is required, and reference identity is propagated to evaluation/report/memory paths. Confirmation reporting avoids acquiring or mutating the already-held parent campaign lock. These observations do not supersede the three findings above.

Native subprocess integration remains outside this review because of the documented inner-PID versus host `/proc` mismatch. The bounded fixtures are appropriate for these specific authorization, persistence and selected-artifact assertions. One scoped final re-review is required after the owners declare the three corrections stable.
