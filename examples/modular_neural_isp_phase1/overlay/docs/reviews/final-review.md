# Whole-branch final review

Date: 2026-10-03. Base: `91d8577f63231344c8b03c555dbde9511212db8b`. Reviewed supplied `final-review.diff` through `e56f189`, final runtime source unchanged at `43345a38e83290c72e5e6c8e6fb084961dd7b243`. The latter adds only two native-acceptance verification facts. Requirements: `examples/modular_neural_isp_phase1/overlay/docs/plans/aris-fast-slow-research-20261003.md`, including the ledger rulings.

**Verdict: changes required.** The implementation substantially meets the intended architecture, but the complete spec and reliability gate are not yet met because of the three Important issues below. No Critical or Minor findings. This is the complete broad-review finding list; a single fix wave and scoped re-review can address it.

## Important I1 — Confirmation recovery and independent-evaluation starts bypass the current parent gate

Locations: `tm_research/confirmation.py:323-339, 402-406, 416-436, 459-471` (all code paths below are relative to the overlay).

The current parent/control/hold/resource check occurs only when selecting a new pending task (`task is None`). An existing active task with a prepared, never-started run takes lines 463-468 directly. An active `checkpoint_frozen` task takes lines 459-461 directly; `_finish_task` starts independent evaluation at lines 331-339 without a fresh parent check. The ordinary transition from completed TRAIN+DEV to independent evaluation has the same omission. Thus a newly recorded protocol hold, changed frozen parent identity, or new cross-stage activity is ignored at these actual launch boundaries.

Concrete fresh probes used the existing real subprocess `setup_confirmation` fixture, with no production edits:

1. Initialize original-DEV confirmation. Replace `runner.start_prepared_run` with a `KeyboardInterrupt` before its body, call `run_confirmation_next`, restore it, then persist `parent['research_hold'] = {'decision_id': 'new-protocol-hold'}`. Resume with `run_confirmation_next`.

   Observed: `saved_task_status prepared`; parent still contains the hold; `recovered_task_status completed`; `worker_ran True`. Artifacts: `/tmp/final-review-recovery-qd7jgj62`.

2. Initialize independent confirmation. Wrap `confirmation._save` to save normally and then interrupt when the first task is `checkpoint_frozen`. Restore the save function, persist a parent hold, and resume.

   Observed: `saved_task_status checkpoint_frozen`; `task_status_after_hold completed`; `independent_evaluation_valid True`. Artifacts: `/tmp/final-review-frozen-stage-r2w3hlyz`.

This violates spec §§7–8 and the explicit current-start gate ruling already implemented for search recovery. Recheck the current parent under its lock immediately before each actual confirmation start, including recovery and independent evaluation. Exclude only this attempt's own active reservation from the gate; retain all other activity, measured costs, identity checks and holds. A held prepared/frozen task must retain its identity/checkpoint and resume only after legitimate resolution. Collecting a terminal result can remain possible while held.

Scoped verification should cover both reproduced windows, current parent identity/activity as well as a hold, the normal TRAIN+DEV→independent-evaluation transition, and resuming the same task without retraining or reselection after resolution.

## Important I2 — Known cross-split scene overlap is ignored by independent-confirmation claims

Locations: `tm_research/confirmation.py:188-208, 280-297, 586-608`; `tm_research/research.py:391-434`, and the downstream confirmed-memory path at `555-627`.

Independent confirmation loads and preserves the authorized profile, but uses it only to map the independent sample inventory. It does not consume explicit `scene_overlap` evidence or compare known scene IDs against TRAIN/DEV annotations. The report and Q promotion use `scope.kind` and paired scores, so known same-capture samples still produce a supported independent scope without surfacing the conflict. The directory-identity limitation is not a substitute for using known supplied scene evidence.

Fresh real subprocess probe: `setup_confirmation(..., independent=True)`; put an authorized profile in `confirmation_scope.profile_ref` containing `TRAIN/a` and `CONFIRM/a`, both explicitly annotated `scene_id='same-capture'`, plus `scene_overlap=[{'scene_id':'same-capture','sample_ids':['TRAIN/a','CONFIRM/a'],'splits':['TRAIN','CONFIRM']}]`. Initialize and finish all four predeclared tasks; call both confirmation and campaign report.

Observed:

- `launch_allowed True`, and the frozen profile retains that exact overlap.
- Confirmation: `scope.kind='independent'`, `conclusion='supported_in_scope'`, `adaptive_dev_bias=False`.
- Campaign Q: `status='supported_in_scope'`, independent scope, both complete pairs; no overlap limitation.

Artifacts: `/tmp/final-review-overlap-o8qapmbw`. These are fixture scores, not scientific acceptance claims.

Spec §5.1 explicitly requires reporting known same-source cross-split conflicts and pausing claims of independent confirmation. Carry known, source-backed overlap into the frozen confirmation evidence and both report paths; withhold independent confirmation promotion (including memory) when independence is contradicted. Preserve descriptive native scores and the existing split; no automatic repartitioning, content-hash system, or unauthorized held-out enumeration is needed. Unknown scene identities should remain unknown, not be invented or treated as a proven conflict.

Scoped verification should include overlap provided in a combined profile and overlap inferable from authorized parent TRAIN/DEV plus confirmation scene annotations, unknown scenes, and a genuinely disjoint known-scene case.

## Important I3 — Unresolved evaluator release becomes terminal failure and loses the cross-stage activity block

Locations: `tm_research/confirmation.py:345-360, 364-391`; `tm_research/control.py:110-113, 126-149`; analogous final-TEST handling at `tm_research/campaign.py:518-527, 541-548, 563-566`. Runner intentionally distinguishes `recovery_required` at `runner.py:921-930`.

The frozen evaluator can persist `evaluation_result.json` with `valid=False`, `usage.status='unknown'`, `gpu_hours=None`, while `evaluation_state.json` is `recovery_required` because resource release is unconfirmed. Both confirmation collection paths treat the mere receipt as terminal failure, clear `active_task_key`, and omit the task from active reservations. Final TEST likewise marks any invalid receipt terminal. In the shared ledger, explicit zero allocated GPUs converts missing GPU cost to zero; because the wrapper already discarded the liveness state, new CPU work is allowed while an evaluator is explicitly unresolved. Zero GPU cost does not establish process exit.

Fresh probe using the actual frozen-evaluator error path:

1. Initialize independent CPU confirmation via `setup_confirmation`.
2. In the caller only, patch `runner._execute` to raise `runner._UnresolvedExecution('injected unconfirmed evaluator release')`. Training runs in its real subprocess and is unaffected; the actual `evaluate_frozen_checkpoint` persists its recovery receipt.
3. Restore `_execute`. Immediately initialize and run another confirmation directory attached to the same parent.

Observed in one 0.91-second probe:

```text
original_evaluator recovery_required
parent_gate_allows_new_confirm True
second_confirmation_task completed
original_evaluator_still recovery_required
```

The first task was already marked `failed`, `active_task_key=None`, and the parent ledger reported `known=True, active_attempts=[]`. Artifacts: `/tmp/final-review-unknown-cross-stage-5jgq5shi`. The fixture deliberately injects the runner's existing unresolved-release error; it does not assert an actual live orphan was created. A same-directory next is blocked by its private unknown-cost check, but the shared cross-stage gate demonstrably is not.

This violates spec §§8.2–8.3 independently of GPU allocation. Preserve unresolved evaluator activity/reservations until exit is confirmed; inspect evaluator state/health before adopting a receipt as terminal, in confirmation and final TEST. Do not conflate unknown liveness with missing-but-zero CPU GPU cost. Once the existing watchdog/recovery produces a genuinely terminal receipt, collect that same attempt and cost without retry. Otherwise the present cached terminal wrapper can also retain obsolete unknown costs after actual recovery.

Scoped verification should exercise immediate and persisted-receipt collection, CPU and positive-GPU allocations, and a second confirmation/final-TEST cross-stage gate. Confirm eventual terminal recovery releases the original reservation without a duplicate evaluator.

## Review coverage and verification

Reviewed the frozen plan, ledger and prior scoped rulings; full changed-file inventory; campaign/control/research/confirmation integrations; runner start/evaluator and unknown-release paths; evidence reference roles, revision/access validation and required/optional diagnostics; floating evaluator additions; CLI, examples and operational/acceptance documentation.

The intended four-field proposal surface, strict three-key outer JSON, fixed per-search scientific config, legacy min-delta fallback and selection ordering remain intact in the reviewed code. Feedback persists native validity/selection before diagnostics; references distinguish construction, best-before, baseline and explicit comparator; immutable revision/source access and latest-feedback checks are integrated. Confirmation trains against original DEV and freezes checkpoints before independent evaluation; it does not mutate search best. Reports visibly separate W/Q/R/P and synthetic from live evidence. The three findings above are localized cross-component gaps, not a request for a broader platform or extra approval machinery.

Fresh bounded verification run (no complete-suite rerun):

```bash
PYTHONPATH=examples/modular_neural_isp_phase1/overlay .venv/bin/python -m pytest -q examples/modular_neural_isp_phase1/overlay/tests/test_confirmation.py -k 'current_parent_rechecked or recovery_collects or prepare_interruption or inflight_unknown or incompatible_blocked or held_valid'
```

Result: **8 passed, 28 deselected in 4.84s**. These existing regressions do not cover the reproduced gaps. CLI `--help` also exited 0. Working tree was clean before this review report; no production or test source was modified. The recorded full-suite **238 passed / 19 subtests / 75.77s** and native CPU four-search/two-confirmation acceptance were inspected as existing evidence, not rerun or relabeled as this reviewer's fresh full-suite result. They remain accurately bounded to engineering/synthetic original-DEV acceptance; live Naive/ARIS, real-camera generalization and independent-dataset acceptance are not claimed.
