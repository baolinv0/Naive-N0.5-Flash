# F3 final checkpoint consistency

Owned changes: overlay `tm_research/campaign.py`, `tm_research/confirmation.py`, new `tests/test_final_checkpoint.py`. No commits, real TEST data reads, live scientific campaign runs, dependency changes, or edits to research/control. Root owns full-suite verification and docs; other owner wired the CLI.

## Result and API

`campaign.resolve_final_checkpoint(campaign_dir, *, confirmation_dir=None)` validates and freezes a durable `campaign.json.final_checkpoint_ref` without accessing TEST. `campaign final-checkpoint --campaign-dir ... [--confirmation-dir ...]` invokes this API.

- Default retains the search checkpoint when no registered predeclared plan exists. A single registered predeclared plan resolves exactly its winner arm, seed and replicate. Multiple predeclared plans fail ambiguous unless an explicit registered confirmation directory is supplied.
- Receipt association, parent baseline/winner/config/control snapshot, authorization scope, canonical plan, unique completed task, actual native run config/result, native training checkpoint/config artifacts, and checkpoint/config files must agree. Failed, missing, or ambiguous candidates cannot silently fall back to search.
- Frozen reference retains the plan/control/parent snapshots. Reuse validates identity and frozen provenance. Refreezing or retargeting after TEST starts is rejected.
- Both controlled and legacy final evaluation pass the frozen reference's actual weight/config pair. Search `best`, `raw_best`, and `frozen` remain search identities.
- Controlled evaluator recovery rejects a valid result claiming different checkpoint artifacts; terminal reports cannot change run identity.
- Confirmation reports are read-only with respect to the parent campaign: a completed candidate before adoption is marked `final_checkpoint_pending_adoption: true` with no persisted reference. After explicit resolution/final-test the report exposes the same `final_checkpoint_ref`. Parent-lock reporting and memory export do not nest parent locks or mutate the campaign.
- Old retain-search terminal reports can still be reused/recovered without inventing a new frozen reference after TEST starts.

## F1 integration

Agreed with the research/control owner: `control.pending_decision_reason(state, action)` gates `_authorize` in both modes. `next` honors pending decisions in legacy mode; submission and recovered unstarted launches cannot bypass pending state. Legacy final evaluation gates new commands as well. Collection and already terminal evaluation report reuse remain allowed. Pending finalize needs no cleanup: normal freeze permits authorized confirm/TEST while search stays closed.

## Red → green evidence

All meaningful regressions were observed failing before corresponding production edits:

1. Initial native checkpoint fixture suite: **10 failed, 2 passed**, including wrong search checkpoint passed by legacy evaluation and unsafe controlled launch on invalid predeclared candidates; then **12 passed**.
2. Persisted evaluator identity: **1 failed**, accepted mismatching weights; then **13 passed**.
3. Old retain-search report compatibility: **2 failed**, attempted refreeze after TEST; then green.
4. Read-only report under parent lock: **1 failed**, nested-lock RuntimeError; then **16 passed** including memory export parent nonmutation.
5. Legacy pending decision gates: **6 failed**, next/submit/recovered starts or final evaluation allowed; then green.
6. Controlled and legacy terminal report run identity: **2 failed**, incorrect run identity accepted; then green.

Commands run from `examples/modular_neural_isp_phase1/overlay` using `../../../.venv/bin/python`:

- `-m pytest -q tests/test_final_checkpoint.py` → 16 passed before added legacy/run-identity coverage.
- `-m pytest -q tests/test_final_checkpoint.py tests/test_slow_decisions.py tests/test_control.py tests/test_research.py` → **79 passed, 4 failed**. Existing actual-worker research fixtures failed at setup due to runtime PID/proc namespace mismatch (see below).
- Final targeted verification: `-m pytest -q tests/test_final_checkpoint.py tests/test_slow_decisions.py tests/test_control.py` → **71 passed in 0.34s** (24 final-checkpoint/legacy cases plus covering slow/control tests).
- Repo-root `.venv/bin/python -m compileall -q .../campaign.py .../confirmation.py` and `git diff --check` → passed.

The first test attempts could not execute `.venv/bin/python` due to a preexisting symlink loop; root restored the interpreter. A subsequent initial real-worker fixture attempt produced 12 setup failures: inner Python PID and host `/proc` PID differ, so existing process identity supervision exits 125 or remains recovery_required. New tests therefore persist fixture native completed checkpoint records and replace only liveness-result/evaluator boundaries with a bounded in-process surrogate. The assertions exercise production resolution, authorization, persisted identity, actual passed checkpoint/config paths, and native config/artifact mismatch handling. Surrogate evaluator reads only fixture weight files and writes terminal fixture evidence; it does not enumerate or read TEST images.

The four observed covering-test failures were:

- `tests/test_research.py::test_q_claim_follows_external_frozen_confirmation_actual_pairs`
- `tests/test_research.py::test_q_rejects_unrelated_confirmation_receipt_and_changed_actual_protocol`
- `tests/test_research.py::test_memory_confirmation_requires_association_to_the_frozen_experience[False]`
- `tests/test_research.py::test_memory_confirmation_requires_association_to_the_frozen_experience[True]`

Root will run/report the full suite. Existing worker fixtures and supervision logic were preserved.

## Independent review follow-up: fixed science binding

Reviewer probe reproduced: changing both confirmation manifest `config.epochs` and its actual run `config.json.epochs` from 1 to 2 while the parent and frozen reference stayed unchanged was accepted. The new parametrized regression checks both initial selection and reuse of an existing immutable reference; **2 failed** before the production change (`DID NOT RAISE ValueError`).

The candidate now derives fixed scientific configuration from the validated frozen parent configuration, then applies only the declared winner recipe, seed and confirmation runs directory. DEV `expected_count` normalization is bound to the frozen parent's native result rather than trusting an arbitrary confirmation manifest value. Both the confirmation configuration and actual run must match that parent-derived configuration. The reference's provenance now persists the resolved configuration, so reuse also compares its scientific identity.

Final follow-up command from overlay:

`../../../.venv/bin/python -m pytest -q tests/test_final_checkpoint.py tests/test_slow_decisions.py tests/test_control.py`

Result: **76 passed in 0.41s** (26 final-checkpoint/legacy cases plus concurrently expanded covering slow/control tests). Compileall and `git diff --check` passed again. No hashes, TEST scans or runner modifications were added.
