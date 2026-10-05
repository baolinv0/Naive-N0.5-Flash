# Final fix wave — I1, I2, I3

Date: 2026-10-03. Base runtime: `43345a38e83290c72e5e6c8e6fb084961dd7b243`.
Scope: the three Important findings in `final-review.md`; no other implementation wave, agent, or commit.

## Changes and self-review

### I1 — Current-parent launch gates

A common `_start_allowed` check runs under the current parent campaign lock for the initial preparation/start, prepared-run recovery, and independent-evaluator start after the original-DEV checkpoint freeze (both normal and recovered transitions). It checks frozen parent/control identity, holds, cross-stage activity, current actual costs, task and shared pool limits. Its ledger exclusion is exactly `(resolved confirmation directory, task key)`, so a same-key task in another confirmation remains active. Exclusion removes only the own reservation, retaining completed TRAIN+DEV cost. At checkpoint freeze the remaining reservation is one evaluator job. A blocked frozen task retains its original checkpoint, including freeze timestamp, and resumes without retraining/reselection. Terminal evaluator receipts can still be collected during a parent hold.

Self-review: no change to native search selection, proposal fields, authorization renewal rules, or automatic retries. Parent locking covers the actual evaluator call; no new evaluator begins outside the gate.

### I2 — Known scene overlap and independent claims

Confirmation initialization freezes source references, revisions, scene records, unknown IDs, and explicit overlap evidence from the authorized confirmation profile and available authorized development profile references. Matching known TRAIN/DEV and CONFIRM scene IDs, or explicit cross-split `scene_overlap`, record a conflict. Combined-profile DEV rows are not mapped onto independent inventory rows merely because filenames match.

Confirmation reports, campaign Q, and exported memory disclose the same frozen `scene_independence` evidence. A conflict keeps the native paired scores and means, changes the independent conclusion to `inconclusive`, and leaves a requested confirmed-memory promotion `tentative` with an overlap limitation. Unknown scenes remain unknown; disjoint known scenes report `no_known_overlap`, without asserting that incomplete annotations prove independence.

Self-review: no split mutation, content hashes, held-out image enumeration, filename-based scene inference, or change to descriptive native scores. Profiles changed after initialization cannot erase frozen overlap evidence. Search best and checkpoint-selection protocol remain unchanged.

### I3 — Evaluator release and eventual collection

Both immediate and persisted confirmation collection use the same terminal collector. A receipt is collectible only if runner `evaluation_state.json` is terminal and recorded resources are dead; `recovery_required`, missing state, or unresolved resources retain the attempt. Final TEST uses the same terminal check before recording terminal status/cost. CPU-zero allocation affects GPU accounting only, not liveness or the cross-stage activity block. After the existing watchdog confirms release, the next collection adopts that attempt’s terminal receipt and measured cost without invoking the evaluator again.

Self-review: tests inject the runner’s real `_UnresolvedExecution` path and inspect the actual `recovery_required` state. Recovery uses a genuinely exited owner identity and the existing real watchdog, which writes stopped health, terminal failure, and actual cost. No fake successful evaluator is substituted. The existing isolated final-TEST timeout mock now supplies the runner’s terminal-state/released-resource contract; its assertions are unchanged.

## RED → GREEN evidence

New test file: `examples/modular_neural_isp_phase1/overlay/tests/test_final_review_regressions.py`.

Initial reproduction command, before production edits:

```bash
PYTHONPATH=examples/modular_neural_isp_phase1/overlay .venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_final_review_regressions.py --tb=short
```

Observed RED: **25 failed in 22.01s**. These comprised 12 current-parent start-window cases (prepared recovery, recovered frozen checkpoint, ordinary train→evaluation transition × hold/config/activity/cost), 5 scene-evidence cases (combined, separate, explicit-only, unknown, disjoint), and 8 unresolved evaluator cases (confirmation/final TEST × CPU/positive allocated GPUs × immediate/saved receipt). Failure assertions matched the review: tasks completed despite new gates; scene conflicts promoted or were absent from evidence; unresolved evaluator receipts became failed terminal wrappers.

First same-case GREEN: **25 passed in 21.87s**.

Expanded same command: **28 passed in 24.27s**. The 8 release cases now use actual watchdog recovery, and the additional 3 cases cover terminal collection under hold plus exact own-reservation exclusion/actual training cost for prepared and checkpoint-frozen tasks. They also assert a second confirmation and final-TEST gate remain blocked, the same run ID survives, and no duplicate evaluator executes after recovery.

Covering-suite first pass: **105 passed, 1 failed in 52.01s**. The one failure was an old timeout mock with a result JSON but no runner state. Added `evaluation_state.status=failed` and stopped-health evidence to model the existing runner contract; did not relax production terminal checks or old assertions.

Final covering command:

```bash
PYTHONPATH=examples/modular_neural_isp_phase1/overlay .venv/bin/python -m pytest -q \
  examples/modular_neural_isp_phase1/overlay/tests/test_final_review_regressions.py \
  examples/modular_neural_isp_phase1/overlay/tests/test_confirmation.py \
  examples/modular_neural_isp_phase1/overlay/tests/test_control.py \
  examples/modular_neural_isp_phase1/overlay/tests/test_research.py \
  examples/modular_neural_isp_phase1/overlay/tests/test_campaign.py \
  examples/modular_neural_isp_phase1/overlay/tests/test_fast_slow_integration.py --tb=short
```

Final result: **134 passed in 74.33s (0:01:14)**. All six covering suites passed on the final runtime/test source.

`git diff --check`: passed, no output.

## Evidence boundaries

This wave verifies engineering gates and synthetic fixture behavior. It does not execute live Naive/ARIS, real-camera generalization, or real independent-dataset acceptance. The root’s earlier 238-test/19-subtest full run and real-ISP CPU 4-search/2-original-DEV-confirmation synthetic acceptance remain existing bounded evidence in `fast-slow-verification` documentation/evidence JSON; this report does not relabel or repeat those as new results. Root will run the fresh full suite and one scoped final review after this wave is stable, then commit once.
