# Final scoped re-review

Date: 2026-10-03. Reviewed the stable working-tree fix against `43345a38e83290c72e5e6c8e6fb084961dd7b243`, using `final-fix-report.md`, `final-fix.diff`, current production files, and the complete new `overlay/tests/test_final_review_regressions.py` (28 cases).

**Verdict: APPROVED within the authorized final-fix scope.** I1, I2 and I3 are **ADDRESSED**. Open original findings: **0 Critical, 0 Important, 0 Minor**. New Critical/Important breakage within this fix scope: **0**. No additional implementation wave is requested. This scoped re-review closes the three blocking findings from the preceding whole-branch review; it does not repeat that broad review or claim the root's full-suite run as this reviewer's run.

## I1 — ADDRESSED

`confirmation._start_allowed` is shared by initial task preparation/start, existing prepared-run recovery, and the independent-evaluator start after checkpoint freeze. Each actual start is covered by the current parent campaign lock. The normal TRAIN+DEV→evaluation transition and recovered `checkpoint_frozen` transition both enter the same check. Frozen parent/control identity, current holds, activity and cost limits therefore apply at these launch boundaries.

The exclusion passed into `reconcile_usage`/`check_action` is the exact resolved confirmation directory plus task key. It removes only that task's reservation, retains completed training cost, and leaves a same-key attempt in another confirmation active. At checkpoint freeze, the remaining reservation becomes one evaluator job. A blocked frozen task retains its checkpoint and freeze timestamp; resume neither retrains nor reselects. Terminal independent-evaluator collection remains available under a parent hold.

Fresh regressions passed for prepared recovery, recovered checkpoint freeze and ordinary transition, each with hold/config/activity/new-cost changes; same-identity recovery; exact exclusion; positive-GPU actual training cost; and collection under hold.

## I2 — ADDRESSED

Initialization freezes authorized scene records, source references and revisions, explicit cross-split overlap, and unknown identities. It also compares known TRAIN/DEV scene IDs from authorized development profiles with confirmation records. Explicit DEV records in a combined profile are no longer mapped to an independent sample solely because filenames match.

The same frozen `scene_independence` evidence appears in confirmation reporting, campaign Q and exported memory. A known conflict preserves native paired means and complete-pair facts but sets the independent conclusion/Q to inconclusive and requested confirmed memory to tentative, with the overlap limitation. Profile edits after initialization do not erase the frozen conflict. Unknown stays unknown; known disjoint annotations produce `no_known_overlap` with the stated coverage limitation.

Fresh regressions passed for combined-profile overlap, separate parent/confirmation profiles, explicit-only overlap, unknown and disjoint scenes, report/Q/memory agreement, and frozen evidence after profile mutation. No repartitioning, new image-content checks, or held-out access was introduced.

## I3 — ADDRESSED

Both immediate and recovered confirmation evaluation paths use `_collect_persisted_evaluation`. The shared `evaluator_is_terminal` check requires terminal runner state and dead recorded resources; an invalid receipt alone cannot release activity. Confirmation retains `evaluating_confirmation` plus its task reservation while unresolved. Final TEST retains `recovery_required` and its reservation through the same check. Thus explicit CPU allocation still has zero GPU cost without implying process exit.

Fresh regressions exercised the actual runner `_UnresolvedExecution` receipt for confirmation/final TEST × immediate/persisted receipt × zero/positive GPU allocation. They verified both action gates and a second linked confirmation remain blocked. The existing watchdog then received an actual exited owner identity and produced terminal failure/released-resource evidence with measured cost. Collection adopted the same attempt, released its reservation, and did not invoke a second evaluator. Positive allocation produced positive measured cost; CPU did not bypass liveness.

The adjusted pre-existing final-TEST timeout mock now supplies the runner's terminal state/release contract; its assertions were not weakened.

## Fresh verification

Run from the repository root:

```bash
PYTHONPATH=examples/modular_neural_isp_phase1/overlay .venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_final_review_regressions.py --tb=short
```

Result: **28 passed in 24.88s**. `git diff --check` also exited 0 with no output. No production/test source edits, agents, or commits were made by this reviewer. Only this review report was written. Root owns the separate fresh full-suite run and final integration decision; live Naive/ARIS and real independent-dataset scientific acceptance remain outside these engineering results.
