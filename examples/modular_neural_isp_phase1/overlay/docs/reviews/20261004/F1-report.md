# F1 implementation and verification — 2026-10-04

Owned source: overlay/tm_research/research.py, overlay/tm_research/control.py. Added overlay/tests/test_slow_decisions.py; parent additionally authorized migration of the two operational-hold recovery cases in overlay/tests/test_fast_slow_integration.py. No campaign.py edits and no commits.

## Behavior

- Slow-review non-propose actions persist a pending_slow_decision with decision ID, action, trigger, and immutable decision reference. Merely archiving a fast decision, wrong trigger, or diagnostic review cannot authorize plateau continuation.
- report_stop blocks new search and calibration starts; collection and later explicit finalize remain possible. An explicitly frozen candidate may enter authorized confirmation/TEST within the existing control.
- request_scope_change blocks all new work until a new authorized campaign. diagnose/wait retain the pending gate. finalize waits for explicit freeze; existing frozen confirmation gates remain effective.
- A new slow_review action propose must acknowledge the current trigger and pass current recipe, evidence, protocol, liveness and budget gates. It may clear nonterminal pending diagnosis, never terminal slow intent or a protocol hold. Explicit operational hold resolution alone remains blocked until this continuation decision is recorded.
- The pure pending_decision_reason(state, action) hook shares these explicit pending restrictions with legacy campaign paths. Legacy campaigns without an explicitly recorded pending decision retain their existing behavior. Campaign worker owns and has integrated _authorize/next/submit/recovery/final-test hooks.
- Launch-intent/prepared-run recovery uses the existing campaign._current_start_allowed hook: its check_action call now respects pending decisions. Already running work can still be collected.
- Explicit replay of archived pre-upgrade non-propose slow decisions repairs missing kind/pending metadata without automatically migrating history. Old propose replay cannot clear newer pending intent; older diagnoses with later decisions stay archived.
- Immutable IDs remain idempotent. Replaying an old diagnosis does not restore it after a later continuation. Interrupted state-write replay revalidates and reapplies the immutable log record only when campaign metadata was never persisted.
- _w_claim invokes workflow.validate_workflow_evidence inside its live provenance try block before service parsing. Missing/bad registration remains pending.
- Review packets/reports expose the pending decision. Report and exported memory expose state.final_checkpoint_ref, with an explicit unavailable status when absent; checkpoint selection remains the campaign worker’s responsibility.

## RED before production edits

Command (cwd overlay):

```sh
../../../.venv/bin/python -m pytest -q tests/test_slow_decisions.py
```

Raw summary: `18 failed, 3 passed in 1.09s`.

Original RED log: /workspace/scratch/6c78134e8d18/F1-red.log. The four slow-action routes returned propose; terminal action replacements and fast-kind review acknowledgment were allowed; recovered intent starts ignored the slow decisions; report lacked final checkpoint identity. The initial interpreter attempts failed because pytest was unavailable and then the venv interpreter link was broken; parent restored runtime centrally before this RED run.

Additional interrupted-state replay RED: `1 failed in 0.07s`; expected report_stop remained diagnose after immutable-log replay. Raw log: /workspace/scratch/6c78134e8d18/F1-red-replay.log.

Additional shared legacy helper RED: `5 failed in 0.05s` (F1-red-legacy-helper.log). Additional original-metadata replay RED: `2 failed, 1 passed, 31 deselected in 0.06s` (F1-red-upgrade.log).

## Final targeted GREEN

```sh
../../../.venv/bin/python -m pytest -q tests/test_slow_decisions.py tests/test_control.py tests/test_research.py -k 'not q_claim_follows and not q_rejects and not recognized_w and not memory_confirmation'
```

Raw summary: `59 passed, 5 deselected in 0.20s`.

Raw log: /workspace/scratch/6c78134e8d18/F1-green-final.log. Recovery cases now explicitly use persisted completed fixture records and a native launch-boundary surrogate; the record, collection, state reload, gate, and current authorization paths are real. No live Naive, real ISP training, or real TEST access was performed by this worker.

`git diff --check` passed.

## Native subprocess integration limitation

An attempted broader targeted command was:

```sh
../../../.venv/bin/python -m pytest -q tests/test_slow_decisions.py tests/test_control.py tests/test_research.py tests/test_fast_slow_integration.py
```

Raw summary: `27 failed, 53 passed in 9.37s` (F1-green1.log).

Isolated original recovery retry: `5 failed, 1 passed in 1.58s` (F1-green-recovery.log). Failing cases stopped at initial fixture baseline with train.py exit 125, before the intended decision/recovery assertion. Parent independently identified a /proc PID namespace mismatch: inner PIDs resolve unrelated host /proc/PID records, so runner._stage_exec correctly refuses an unverifiable/dead owner. No production runner workaround or guard weakening was introduced. Existing native integration tests are preserved apart from the explicitly required hold-resolution migration; parent owns the full-suite report.

Failure names from the broader attempt, including native Q/W/confirmation setup failures:

```text
FAILED tests/test_slow_decisions.py::test_recovered_unstarted_intent_obeys_slow_decision[report_stop]
FAILED tests/test_slow_decisions.py::test_recovered_unstarted_intent_obeys_slow_decision[request_scope_change]
FAILED tests/test_slow_decisions.py::test_recovered_unstarted_intent_obeys_slow_decision[finalize]
FAILED tests/test_slow_decisions.py::test_recovered_unstarted_intent_obeys_slow_decision[diagnose]
FAILED tests/test_research.py::test_q_claim_follows_external_frozen_confirmation_actual_pairs
FAILED tests/test_research.py::test_q_rejects_unrelated_confirmation_receipt_and_changed_actual_protocol
FAILED tests/test_research.py::test_recognized_w_events_link_actual_runs_and_feedback_before_each_proposal
FAILED tests/test_research.py::test_memory_confirmation_requires_association_to_the_frozen_experience[False]
FAILED tests/test_research.py::test_memory_confirmation_requires_association_to_the_frozen_experience[True]
FAILED tests/test_fast_slow_integration.py::test_trial_references_saved_before_start_and_actual_inheritance
FAILED tests/test_fast_slow_integration.py::test_diagnostic_failure_preserves_valid_baseline_and_missing_csv_legacy
FAILED tests/test_fast_slow_integration.py::test_control_sidecar_latest_ack_explicit_comparison_and_snapshot
FAILED tests/test_fast_slow_integration.py::test_required_diagnostic_missing_is_diagnose_not_invalid[False]
FAILED tests/test_fast_slow_integration.py::test_required_diagnostic_missing_is_diagnose_not_invalid[True]
FAILED tests/test_fast_slow_integration.py::test_actual_usage_blocks_next_job_and_preserves_confirmation_reserve
FAILED tests/test_fast_slow_integration.py::test_plateau_review_independently_recorded_does_not_override_hardstop
FAILED tests/test_fast_slow_integration.py::test_confirmation_cli_executes_frozen_duplicate_arms_without_changing_search
FAILED tests/test_fast_slow_integration.py::test_profile_optin_stays_outside_scientific_config_and_reaches_runner
FAILED tests/test_fast_slow_integration.py::test_control_authorized_final_test_bounded_once_and_accounts_failed_cost
FAILED tests/test_fast_slow_integration.py::test_control_final_test_real_success_and_interrupted_host_result_recovery
FAILED tests/test_fast_slow_integration.py::test_durable_intent_recovers_exact_prepare_return_before_active_save[before_prepare-proposal]
FAILED tests/test_fast_slow_integration.py::test_durable_intent_recovers_exact_prepare_return_before_active_save[after_prepare-proposal]
FAILED tests/test_fast_slow_integration.py::test_optional_feedback_exception_has_native_observation_and_allows_control_proposal
FAILED tests/test_fast_slow_integration.py::test_same_run_feedback_revision_refresh_and_old_ack_rejected[required0]
FAILED tests/test_fast_slow_integration.py::test_recovered_pending_intent_obeys_new_hold_then_resumes_same_request
FAILED tests/test_fast_slow_integration.py::test_recovered_prepared_active_obeys_hold_and_resumes_same_run_after_resolution
FAILED tests/test_fast_slow_integration.py::test_recovered_prepared_active_rechecks_new_actual_resource_usage
```

## Independent-review follow-up: immutable queued-patch binding

Independent F1/F3 review identified that a matching-trigger continuation could request optimizer=adamw while releasing an immutable queued LR=0.00005 proposal. The review action now must approve exactly the queued proposal.recipe patch whenever it releases a blocked launch intent/prepared request; neither the queued request nor its identity is changed, and no baseline proposal is synthesized.

Before production edits, new launch_intent/active variants:

```sh
../../../.venv/bin/python -m pytest -q tests/test_slow_decisions.py::test_pending_continuation_must_approve_the_immutable_queued_patch
```

RED: `2 failed in 0.05s`, both `DID NOT RAISE ValueError`. Raw log F1-red-queued-binding.log.

GREEN: `2 passed in 0.03s` (F1-green-queued-binding-targeted.log). The tests verify rejection preserves pending diagnosis and the same immutable request; matching approval subsequently passes the actual recovered-start authorization with that original identity.

Combined targeted command (explicit node plus file, pytest deduplicated):

```sh
../../../.venv/bin/python -m pytest -q tests/test_slow_decisions.py::test_pending_continuation_must_approve_the_immutable_queued_patch tests/test_slow_decisions.py tests/test_control.py tests/test_research.py -k 'not q_claim_follows and not q_rejects and not recognized_w and not memory_confirmation'
```

Raw summary: `61 passed, 5 deselected in 0.22s` (F1-green-queued-binding.log). `git diff --check` and source/test py_compile pass. No campaign edits, commits, agents, native subprocess permission changes, or real TEST access in this follow-up.

## Independent-review follow-up: conflicting terminal decisions

A new slow_review with a different action now explicitly fails while report_stop, request_scope_change, or finalize is pending. This prevents a successful archival record from falsely implying a control transition that was silently ignored. The immutable decision ID remains idempotent; closure-kind analysis may be archived but does not alter pending control. Explicit finalize still freezes a stopped search, and confirmation then follows its original report_stop/current authorization gates. A new scope requires a separately authorized campaign.

```sh
../../../.venv/bin/python -m pytest -q tests/test_slow_decisions.py::test_conflicting_terminal_review_is_rejected_atomically_before_explicit_freeze
```

RED before source edit: `1 failed in 0.06s` with `DID NOT RAISE ValueError` (F1-red-terminal-conflict.log). GREEN: `1 passed in 0.08s` (F1-green-terminal-conflict.log). The test verifies rejection leaves campaign.json and JSONL byte-identical, creates no conflicting snapshot, and preserves the original control through explicit freeze/confirmation and subsequent closure-only analysis.

Final targeted command:

```sh
../../../.venv/bin/python -m pytest -q tests/test_slow_decisions.py tests/test_control.py tests/test_research.py -k 'not q_claim_follows and not q_rejects and not recognized_w and not memory_confirmation'
```

Raw summary: `62 passed, 5 deselected in 0.18s` (F1-green-final.log). Diff check and compile checks pass. No campaign edits/commits/agents in either review follow-up.
