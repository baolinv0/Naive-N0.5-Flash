# Completed full-suite log triage — F1/F3

Inspected `full-tests.log` and current source only; no test reruns or production edits. Full result: **115 failed, 246 passed, 19 subtests passed, 18 warnings in 156.91s**.

**Finding: no new F1/F3 regression identified in the completed failures.** The failures fall into the existing native process-identity/environment problem and its invalid-baseline consequences. Neither `tests/test_slow_decisions.py` nor `tests/test_final_checkpoint.py` appears in the failure summary.

| Failure group | Count | Evidence and F1/F3 interpretation |
| --- | ---: | --- |
| `test_confirmation.py` | 30 | Each fails in `_initialize_confirmation` at the existing `Frozen winner result is not valid` check. The native search winner is invalid before confirmation/report/final-checkpoint assertions. |
| `test_final_review_regressions.py` | 28 | Same initialization failure, before intended current-parent, report/memory, reservation, or confirmation recovery scenario. |
| `test_research.py` | 4 | Q-claim and memory-confirmation cases fail at the same invalid frozen-winner initialization. |
| `test_campaign.py` | 7 | Initial baseline is invalid, or follow-on submit sees `stopped`. Log lines 38–47 explicitly contain `train.py exited with status 125` in the proposal's baseline evidence. |
| `test_fast_slow_integration.py` | 21 | Invalid initial native baseline produces no best, stopped proposals, failed valid-baseline assertions, or report-stop routing. Recovery and final-test cases cannot reach their intended controlled scenario. |
| `test_runner.py` | 25 | Direct native launch/process identity, timeout, cleanup and evaluator-fixture failures in unchanged runner code; detailed examples below. |
| **Total** | **115** | **62 confirmation-init failures + 28 campaign/integration consequences + 25 runner failures.** |

The grouping was counted directly from all 115 `FAILED` summary entries. There are exactly 62 traceback error lines reporting `Frozen winner result is not valid`, matching the first three groups. `test_confirmation.setup_confirmation` runs real fixture workers, then writes parent trial validity as true without asserting that those workers succeeded. The production initializer subsequently checks the actual winner result and rejects it; this is not the new F3 config/reference validation rejecting a valid candidate.

## F1/F3-specific misleading symptoms

- The plateau integration case fails on its first recipe submission with an invalid baseline/exit 125, before recording its slow review (`full-tests.log` around lines 2416–2446).
- Both operational-hold recovery integrations fail on their initial proposal before creating the blocked/recovered scenario (around lines 2738–2811). This log does not implicate the new explicit continuation or queued-patch checks.
- Controlled final-test integration cases fail in `finalize_campaign` with `There is no valid DEV result to freeze` (around lines 2499–2559), before final checkpoint resolution or evaluation.
- The two inaccessible-feedback cases expect `diagnose` but receive `report_stop` (around lines 2700–2736). `campaign._stop_reason` turns invalid baseline into `invalid_baseline`; `route_next_action` correctly gives that hard stop priority over missing diagnostics. These failures do not show a pending-decision routing regression.

## Runner/environment evidence and limits

The log explicitly records train exit 125 in baseline/empty-DEV/walltime cases, missing artifacts after unsuccessful baselines, unknown liveness preventing a second worker, missing child PID markers, absent published identity/PGID, and orphan-group wait timeouts. Current `runner._stage_exec` returns 125 when its owner identity is not `live`; `_process_identity` and `_group_liveness` inspect `/proc` using Python-reported PIDs/PGIDs. These symptoms match the already diagnosed inner-PID versus host-`/proc` namespace mismatch. `git diff --name-only -- tm_research/runner.py tests/test_runner.py tests/test_confirmation.py` returned no changes.

The cleanup cases also show incorrect live-group observations (`terminal` rather than `unknown`, or cleanup deemed complete while a child remains), and the injected post-Popen error case reports walltime before its intended injected-error assertion. These remain failed native supervision checks in this environment, not successful validation. They do not execute F1/F3 decision/checkpoint code. No separate non-environment F1/F3 issue was exposed by their tracebacks.

The 18 warnings concern third-party deprecations, not F1/F3. This triage does not claim the full suite passed or that the blocked integration scenarios were exercised successfully. The scoped F1/F3 approval remains supported by the previously completed bounded tests; native integration validation is still limited by the runtime environment.
