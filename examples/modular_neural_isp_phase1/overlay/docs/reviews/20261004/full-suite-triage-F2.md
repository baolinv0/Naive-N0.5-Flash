# Full-suite log triage — F2 scope

Reviewed the current `full-tests.log` failure summary, all error-type patterns, F2-related references and representative campaign/confirmation/runner tracebacks. No production edits or test rerun.

**No new F2 regression identified.** The previous scoped F2 approval remains. The log has no failed `test_workflow.py` or `test_naive_adapter.py` cases and no traceback references to `workflow.py` or `naive_adapter.py`. Its four `test_research.py` failures concern Q/confirmation receipt and memory association, not W acceptance.

The actual full result is **115 failed, 246 passed, 18 warnings, 19 subtests passed in 156.91s**. It must not be reported as a passing full suite.

Failure-summary distribution:

| File | Failures |
|---|---:|
| test_campaign.py | 7 |
| test_confirmation.py | 30 |
| test_fast_slow_integration.py | 21 |
| test_final_review_regressions.py | 28 |
| test_research.py | 4 |
| test_runner.py | 25 |

General evidence: the log contains 62 occurrences of `Frozen winner result is not valid`, 12 of `Campaign cannot accept a proposal while stopped`, and explicit `train.py exited with status 125` baseline errors (for example `test_no_gain_stops_and_keeps_baseline`, `test_tiny_raw_gains_stop_without_replacing_effective_best`, and `test_empty_dev_input_is_invalid_even_after_successful_train`). Later missing `artifacts`, absent valid best results, and freeze/confirmation failures are consistent with those upstream failures. The runner tracebacks show unknown liveness, missing/pending process identity, and scientific children not reaching their marker writes. No import/collection failure, missing Python dependency, adapter bind failure, or workflow schema error appears in the failure summary/tracebacks.

**Attribution limit:** this log alone does not prove every failure is exclusively caused by the known PID/proc mismatch. Runner cases also show walltime instead of the injected post-Popen error (`test_runner.py:534`), cleanup returning `True` for a nominally live group (`:591`), and `terminal` instead of `unknown` (`:608`). These remain runner/environment-boundary failures requiring compatible-host or baseline comparison to separate; no F2 code is on their failing paths. No new confirmed non-environment regression was established by this log-only triage.

The passing engineering capture tests use the documented synchronous test-only supervisor helper, so their success cannot resolve those production runner failures or establish live Naive/ARIS performance.
