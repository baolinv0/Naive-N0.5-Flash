# Fast feedback, event reviews, frozen confirmation

This opt-in workflow reuses the ARIS host, Naive proposer and serial runner. Deterministic code executes, measures and selects; the proposer explains observations and writes predictions. The four allowed recipe fields and independently reloaded mean per-image RGB DEV PSNR remain unchanged. No TEST inspection belongs in search.

## Prepare a real authorization

Copy `configs/control.example.json`, fill the question, existing authorization reference, source/config identity and actual limits. Null resource fields deliberately fail validation; this template does not grant spending permission. `protocol_id` is the actual evaluator protocol, such as `P512`. `max_search_trials` includes baseline and invalid attempts. Walltime covers TRAIN plus independent DEV. GPU-hours mean allocated GPUs times elapsed walltime. Search preserves the separate confirmation and calibration reserves; missing GPU usage blocks new GPU work. Actual trial, calibration, final-TEST and parent-linked confirmation records are reconciled before a launch; partial cost summaries cannot erase recorded usage. Pending launch requests and cross-stage active jobs retain their reservations and block conflicting starts. Explicit `allocated_gpus: 0` and zero GPU budgets support CPU fixtures with a finite walltime.

Canonical limits are `job_walltime_seconds`, `total_gpu_hours`, `allocated_gpus`, `max_search_trials`, `confirmation_gpu_hours`, and `calibration_gpu_hours`. `slow_policy.consecutive_valid_no_gain` is independent of the existing no-gain stop counter, which includes invalid attempts. `test_permission` is false for the current pilot. `observation_mode: text` is the Naive adapter contract; image paths do not prove visual model access.

From the applied overlay in the official ISP checkout, using its Python environment:

```bash
python -m tm_research.cli control --control configs/control.local.json
python -m tm_research.cli profile --config configs/baseline.local.yaml --rules configs/profile-rules.example.json --output-dir campaigns/pilot/dev_profile
python -m tm_research.cli campaign init --config configs/baseline.local.yaml --campaign-dir campaigns/pilot --control configs/control.local.json --profile campaigns/pilot/dev_profile/profile.json --max-trials 4 --no-gain-limit 3 --min-delta 0.01 --wait
python -m tm_research.cli campaign next --campaign-dir campaigns/pilot
```

Choose and freeze a calibrated `min_delta` before real initialization; 0.01 is an engineering example. The profile scans authorized TRAIN/DEV only, reuses the evaluation transform and records GT output codevalue/gradient thresholds. Scene IDs or labels without actual sources remain unknown. A profile built after reviewing outcomes must declare `retrospective: true`; it is exploratory. ROI measurements are optional, candidate-independent float diagnostics and do not alter the main score.

The configuration remains scientific YAML. Control and observation fields are separate inputs, never extra keys in `normalize_config`. Initialization copies the control snapshot; changing the external file does not enlarge a running campaign. Existing campaigns are resumed without upgrading their thresholds or recreating missing predictions.

## Read evidence, then propose

`next` returns the newest feedback, baseline/best/recent six results, a full historical index and `next_action`. Feedback versions have readable immutable revision IDs. Unchanged rebuilds reuse a version; corrected inputs or rules create a new version and preserve old observations. Each next/submit reads the newest published revision under the campaign lock: historical observations remain citable, but the latest acknowledgment must advance. An unreadable newest artifact is an explicit access failure, never a reason to reuse stale readiness.

```bash
python -m tm_research.cli campaign feedback --campaign-dir campaigns/pilot
python -m tm_research.cli campaign feedback --campaign-dir campaigns/pilot --run exp_002 --section paired_comparison
python -m tm_research.cli campaign feedback --campaign-dir campaigns/pilot --run exp_002 --section cases
python -m tm_research.cli campaign feedback --campaign-dir campaigns/pilot --run exp_002 --section curves
```

Use returned run IDs and references; the sample IDs above are syntax examples. Pairing uses stable sample identity, never CSV row order. Partial coverage reports a matched subset, not a full DEV mean. Group labels overlap and small groups remain descriptive. PNG-8 cases are display evidence; scores and optional regions use float evaluation tensors. Output codevalue statistics are not physical illumination or ISO.

Write the unchanged three-key proposal (`recipe`, `hypothesis`, `based_on`) and a separate decision based on `configs/decision.example.json`. A sidecar must acknowledge the latest terminal run and feedback revision, cite actual observation IDs, include prediction, falsifier and alternative explanation, and match the submitted recipe patch exactly. Citing evidence establishes traceability, not proof of a causal mechanism or an effective research strategy.

```bash
python -m tm_research.cli decision --campaign-dir campaigns/pilot --decision decision.local.json
python -m tm_research.cli campaign submit --campaign-dir campaigns/pilot --proposal proposal.local.json --decision decision.local.json --wait
```

The controller saves the sidecar, resolved scientific config, role references and durable launch request before preparation. Repeated preparation uses that exact request token to recover one run identity; active state is saved before starting its worker. `based_on` is historical evidence. `construction_base_run_id` is the effective best whose recipe receives partial overrides. `comparison_run_id` is the experimental comparator; explicit invalid or incompatible references are rejected. Every trial starts from the fixed initialization, regardless of either reference. To test an old losing recipe, submit its full recipe with the intended change: a partial LR patch still inherits current best. An original→MSE/L1 change combines final-only supervision with changed loss composition and cannot isolate loss form.

Selection persists first. A missing diagnostic artifact cannot erase a valid score or change best. In full control mode required missing diagnostics route to `diagnose`, blocking new proposals until feedback is repaired; legacy missing CSV remains nonblocking. Optional assembly failures persist a minimal native execution/score observation with unavailable chapters and retain readiness when no required chapter is missing. Rebuild using `campaign feedback` after repairing only observation artifacts. Do not change an active scientific protocol to repair an unfavorable result.

## Event review and recovery

```bash
python -m tm_research.cli campaign review-packet --campaign-dir campaigns/pilot
python -m tm_research.cli campaign record-decision --campaign-dir campaigns/pilot --decision review.local.json
python -m tm_research.cli campaign report --campaign-dir campaigns/pilot
python -m tm_research.cli campaign memory --campaign-dir campaigns/pilot
python -m tm_research.cli run inspect --run exp_002 --runs-dir runs
python -m tm_research.cli run stop --run exp_002 --runs-dir runs --reason 'Authorized job cancellation'
python -m tm_research.cli campaign wait --campaign-dir campaigns/pilot --timeout 60
```

Review is event-triggered, not a second resident runtime. `review_required` is separate from the action vocabulary (`wait`, `diagnose`, `propose`, `finalize`, `confirm`, `request_scope_change`, `report_stop`). Recording a slow decision also persists its action in `pending_slow_decision`; a recorded review alone never reopens proposals. Record slow decisions independently even when no training follows.

| Recorded action | Subsequent control state |
|---|---|
| `diagnose` / `wait` | Keep the pending diagnosis/pause; collect existing work, but do not launch another job. |
| `request_scope_change` | Wait for a new authorized campaign; a decision cannot enlarge the current contract. |
| `report_stop` | Stop search persistently. Explicit freezing and already authorized post-freeze confirmation remain possible. |
| `finalize` | Keep proposals blocked until `campaign finalize` explicitly freezes the candidate. |
| `propose` | Resume only through a slow review matching the current trigger, within existing bounds. |

Use the current `next_action.trigger_id` for a continuation review. An ordinary fast proposal, a stale trigger, or replaying an old decision cannot clear a pending diagnosis. A stop or scope-change decision cannot be overwritten to restart search. Protocol suspicion separately establishes `research_hold`; only new diagnostic evidence demonstrating an operational issue with unchanged protocol can resolve it. Resolving a hold with action `diagnose` leaves diagnosis pending; explicitly record a matching continuation review if continuation is justified. Clearing a hold never overrides hard stops, frozen selection or budgets. Prepared-run recovery uses the same current action gate.

If work is already queued, the continuation's `requested_change` must match its immutable proposal recipe patch. A different new slow-review action cannot replace a pending terminal action (`report_stop`, `request_scope_change` or `finalize`): recording fails before writing state, snapshot or decision log. Use a `closure` record for additional analysis without changing that control state; explicit freezing remains a separate action.

A wait timeout bounds the caller, not the worker. Resume the same saved active run. A confirmed never-started prepared run can start once after current authorization, holds, resource/activity and hard-stop checks. A new hold leaves its exact pending request and reservation intact with a reviewable blocked status; proper resolution resumes the same identity. An unknown worker identity blocks another launch and retains its reservation. Stop requests target actual work but remain requests until exit is confirmed. Interrupted jobs preserve partial artifacts and become invalid, with existing baseline and stop behavior. There is no optimizer/RNG resume or automatic retry of identical experiments.

## Confirmation after freezing

```bash
python -m tm_research.cli campaign finalize --campaign-dir campaigns/pilot
python -m tm_research.cli confirmation init --campaign-dir campaigns/pilot --confirmation-dir campaigns/pilot/confirmation --plan configs/confirmation.local.json --control campaigns/pilot/control.json
python -m tm_research.cli confirmation next --confirmation-dir campaigns/pilot/confirmation --wait
python -m tm_research.cli confirmation report --confirmation-dir campaigns/pilot/confirmation
```

Repeat `confirmation next` until its recorded task list is terminal; use its returned task identity and run ID. Confirmation runs frozen baseline/winner recipes directly, keyed by arm+seed+replicate, without search deduplication or updates to search best. A parent association binds frozen campaign/control identity; the parent ledger checks linked confirmation activity and cost together with final TEST before any stage starts. The plan freezes seeds, replication, execution order, GPU budget, useful-delta threshold, failure policy and final-checkpoint rule. Failed or incomplete pairs remain in the report and are never silently removed or automatically retried.

Every actual confirmation start rechecks the locked current parent, including a recovered prepared run and the transition from a frozen checkpoint to independent evaluation. A hold preserves the task/checkpoint for the same authorized continuation after resolution; terminal result collection remains available. The gate excludes only that attempt’s own reservation and retains its completed training cost and all other linked activity.

Original DEV chooses training checkpoints. An explicitly authorized independent split is evaluated only after checkpoint freeze and must match the authorized split identity; TEST cannot be relabeled confirmation. Original-DEV seed replication measures stability within adaptive DEV and does not establish independent generalization. Missing `delta_useful_db` yields descriptive results; the controller does not invent a promotion threshold.

Independent confirmation freezes scene evidence from authorized development and confirmation profiles. Explicit cross-split `scene_overlap` and matching known TRAIN/DEV–confirmation scene IDs are disclosed in confirmation reports, campaign Q claims, and exported memory. A known conflict keeps native paired metrics descriptive and withholds independent confirmation promotion; unknown scenes stay unknown. These checks neither repartition data nor enumerate TEST.

Reports distinguish W (live workflow), Q (recipe evidence), R (research strategy) and P (product evidence), including unavailable costs and unverified visual claims. Synthetic subprocess tests establish engineering behavior only. Live Naive/ARIS transcript, actual GPU/data authorization, real paired seeds and independent confirmation remain pending until collected. No confirmed quality gain, superiority of this strategy, camera generalization or TM-only attribution follows from these implementation tests.

For actual host capture and deterministic W evidence registration, follow [workflow-evidence.md](workflow-evidence.md). Keep original adapter/session and CLI tool-result records. Register through the CLI rather than editing `campaign.json` or writing a success event index. Registration does not itself establish W: service identity, model output, consumed feedback and actual submitted/completed runs must agree and retain their order.


## Authorized final held-out evaluation

The current pilot keeps `test_permission: false`. For an already authorized control campaign with permission true, `final-test --campaign-dir <campaign>` runs one bounded checkpoint evaluation after freezing. Its walltime and actual allocated-GPU usage are recorded, including failure cost; terminal success or failure is returned on repeated calls. A host interrupted after the evaluator saved its terminal result recovers that same result. An invalid receipt alone does not establish exit: the evaluator state must be terminal and recorded resources stopped before collection releases the attempt. `recovery_required` retains the cross-stage activity block and reservation even with zero allocated GPUs. Both confirmation and final TEST adopt the same watchdog-recovered terminal receipt and updated cost without a duplicate evaluation. If liveness remains unknown, another evaluation is refused. Legacy campaigns retain their existing explicit final-test behavior. Never use held-out results to resume recipe search or relabel TEST as an independent confirmation split.

The default `retain_search_checkpoint` retains the frozen search weight. For an authorized `predeclared_seed` plan, resolve the completed declared winner into immutable `final_checkpoint_ref` before final evaluation. Missing/failed declared tasks do not fall back to search weights; multiple candidate confirmation plans require explicit selection. Report, memory and final evaluation use the same reference. Once selected or a final evaluation has started, another seed or checkpoint cannot replace it. Selecting a checkpoint does not grant TEST permission or read TEST.

```bash
python -m tm_research.cli campaign final-checkpoint --campaign-dir campaigns/pilot --confirmation-dir campaigns/pilot/confirmation
```

Reporting is read-only: a completed confirmation candidate is distinct from an adopted final checkpoint until the explicit selection step above. Omitting `--confirmation-dir` resolves only an unambiguous registered predeclared plan, or the default search checkpoint when no such plan exists.
