# ARIS-Code task: adaptive fixed-model ISP recipes

Run this task inside the existing ARIS-Code host using its configured Naive executor and its normal file/Bash tools. Do not build or launch a second generic agent host. This file supplies the task-specific tool workflow; `tm_research.cli` supplies the callable controller. The operator supplies a working config path, campaign directory, maximum trials, no-gain limit and calibrated min delta in dB. Use the repository's Python environment.

Example host invocation, after configuring `configs/aris.example.json` fields in ARIS-Code:

```bash
aris "Read docs/aris-campaign-task.md and execute it. Config: configs/baseline.example.yaml. Campaign: runs/campaign. Max trials: 4. No-gain limit: 3. Min delta: 0.01 dB. Use the configured Naive executor. Do not run TEST."
```

1. Read the config and this task. The model and data protocol are fixed. Allowed research changes are only `loss_family` (original/MSE/L1), Adam/AdamW, learning rate, and optional weight decay. Do not edit model files, data, metrics, seed, initialization, or trial budget.
2. If the campaign does not exist, call `python -m tm_research.cli campaign init --config <config> --campaign-dir <campaign> --max-trials <limit> --no-gain-limit <limit> --min-delta <calibrated-dB> --wait`. Otherwise resume with `campaign status`, preserving its stored threshold. Initialization runs the baseline before any proposal. Use adequate Bash execution time and keep its session active. If your host preserves background workers, you may omit `--wait` and use `campaign wait --timeout 60`; if a wait times out, continue collecting the same run instead of submitting another.
3. Call `python -m tm_research.cli campaign next --campaign-dir <campaign>`. Read the complete returned DEV evidence, latest decision, best recipe and `next_action`. Use `next_action.action` as the sole action selector, following the table below. `proposal_allowed` is only a submission guard: false does not mean the research has ended. After waiting, diagnosis or a recorded review, call `campaign next` again and follow that fresh route.
4. Only when `next_action.action` is `propose` and `proposal_allowed` is true, reason from the actual latest result. Explain which recipe change should improve DEV performance and why. Consider failed hypotheses and invalid runs as evidence. Write one JSON proposal containing `recipe`, a nonempty `hypothesis`, and `based_on` with a recorded `run_id`, its observed `dev_psnr` (null for an invalid run), and a concrete `observation`. Read `docs/tm_research_task.md` for the JSON schema. You can inspect that run's DEV logs and artifacts to understand a failure. Never inspect TEST images or TEST metrics to choose a recipe.
5. Submit only on that `propose` route: call `python -m tm_research.cli campaign submit --campaign-dir <campaign> --proposal <proposal.json> --wait`, adding `--decision <decision.json>` in control mode. This actually trains and independently reloads the model. Do not replace it with a predicted score, mock tool result, or grid lookup. A failed CLI call is not a scientific result; inspect the returned error and correct only the proposal or operational issue it identifies. If the route and guard disagree, inspect the current status and fetch a fresh route; do not submit or infer permission to finalize.
6. Read the stored run result and controller decision. Explicitly distinguish `completed` from `valid`, and `no_gain` from `improved`. `raw_best` records the highest score; `best` is the retained effective candidate. Only a score above `best.dev_psnr + min_delta` is improved. Do not change the frozen threshold or treat a raw-best fluctuation as progress. Return to step 3 and let the new route determine whether to wait, diagnose, propose or close. Never write all proposals in advance.
7. Close search only on `finalize` / `report_stop`, or an explicit user stop. Report the chosen DEV run, recipe, score, protocol, trial count, and stop reason. Freeze once with `python -m tm_research.cli campaign finalize --campaign-dir <campaign>` when there is a valid retained candidate, no unresolved active/pending launch, and closure is permitted; if already frozen, reuse that selection. If the baseline was invalid or the candidate is missing, report the logs and failure without attempting to freeze. A diagnosis, wait or scope-change pause does not enter this step. TEST is a separate final operation: call `python -m tm_research.cli final-test --campaign-dir <campaign>` only when final held-out evaluation was included in the user's task and allowed by the control contract, after freezing the choice. Do not feed its result back into proposal selection.

| `next_action.action` | Required host action |
|---|---|
| `wait` | Collect the same active run with `campaign wait --timeout 60`, then fetch fresh `campaign next`. A timeout does not authorize another launch. If no run is active and a recorded pause remains, preserve it and report the wait reason. |
| `diagnose` | Read the reason and evidence. Repair missing observation artifacts within the existing protocol, or investigate the recorded hold/unknown execution state. If `review_required` is true, read `campaign review-packet --trigger <next_action.trigger_id>`, compare competing explanations and save the outcome through `campaign record-decision --decision <review.json>`. Then fetch fresh `campaign next`; do not finalize because proposals are blocked. |
| `propose` | Follow steps 4–6 only while `proposal_allowed` is true. When resuming a paused review, its justified continuation must already have been recorded against the current trigger; recording it does not itself launch a trial. |
| `finalize` | Follow step 7 to freeze a valid, unfrozen retained candidate; never freeze a nonexistent result or replace an existing frozen selection. |
| `report_stop` | End search and report the reason. Follow step 7 for a valid, unfrozen retained candidate only when closure is permitted; an invalid baseline or an already frozen campaign needs reporting, not another freeze. |
| `request_scope_change` | Pause the affected work and present the required scope change. Preserve the current model, data, protocol and budgets; only a new authorized campaign may enlarge them. |
| `confirm` | Use only the already authorized confirmation plan for the frozen selection. Resume its recorded task IDs through `confirmation next/report`; initialize only if that authorized plan is not already registered. If the plan is complete, report it instead of recreating or rerunning it. Without a plan, report confirmation pending. |

State and trial evidence live in the campaign and run directories. Reuse them when the ARIS session restarts. A real Naive/ARIS research claim requires a real configured service and session transcript showing feedback-informed proposals. Local simulated-proposer tests and assistant-mediated integration do not establish that claim.

## Real Naive pilot: baseline plus three proposals

Run in the ISP checkout with this overlay applied, on one training GPU, serially. Naive inference may use a separate existing service. First have a working `aris` installation, a reachable real Naive executor configured as described in [naive-setup.md](naive-setup.md), and an ISP config with actual data and initialization paths and a small fixed budget. An endpoint greeting alone does not verify tool use or feedback-driven research. Do not silently substitute another proposer model.

Before initialization, calibrate `min_delta` using repeated DEV reloads as described in [tm_research_task.md](tm_research_task.md). The `0.01` in these commands is an example: replace it with the chosen value and save the repeat measurements and rationale. Keep train/dev separated and use `eval_size: 512` for a real-data P512 pilot. If using synthetic or smaller images for integration, label that protocol in the evidence.

On Linux, one way to keep the raw host terminal transcript is:

```bash
python -m tm_research.cli commands --config configs/naive-pilot.yaml
mkdir -p evidence/live-pilot
script -q -e -c 'aris "Read docs/aris-campaign-task.md and execute it. Config: configs/naive-pilot.yaml. Campaign: campaigns/naive-pilot. Max trials: 4. No-gain limit: 3. Min delta: 0.01 dB. Use the configured Naive executor. Preserve proposals and tool results. Do not run TEST."' evidence/live-pilot/aris.typescript
python -m tm_research.cli campaign status --campaign-dir campaigns/naive-pilot > evidence/live-pilot/final-status.json
```

Save the original ARIS session export too if its terminal view omits tool messages. Retain the executor URL and served model/weights identity, ARIS version, inference settings, and startup/service logs that establish which model answered; remove API keys from shared configuration. Keep each `campaign next` response and the proposal written after it, plus `campaign.json` and the four run directories with their actual configs, commands, train/DEV logs and metrics. Three later proposals must each follow the preceding completed result; do not prewrite them.

Pilot acceptance requires four valid, actually executed trials and the transcript showing Naive consuming feedback before each next proposal. If a trial is invalid, report the failure and the pilot as incomplete. No PSNR gain is required for workflow acceptance. Real-data improvement, multiple seeds, and TM attribution remain separate validation. Leave TEST untouched during this pilot.


## Delegation policy for the live pilot

This section governs the steps above. It is an operator/ARIS policy, not a new
controller feature or a security boundary. Legacy mode validates recipes, citations, run validity and DEV selection. The opt-in
`--control` mode additionally checks explicit authorization/resource boundaries and
requires independent decision sidecars. Neither mode judges the scientific meaning
of a hypothesis or prevents arbitrary shell bypass. See the new-mode workflow below.

Rationale: Shao et al., [Human–AI Collaboration at Scale](https://www.alphaxiv.org/abs/2608.human-ai-collaboration-at-scale.pdf),
sections 3–5, distinguish task criticality (reversibility, visibility, impact),
human agency (retained control), and productive/unproductive friction. Their
observational chatbot study excludes Claude Code/Cowork and does not establish
that adding friction causes better research. The rules below are our project
policy derived from those distinctions.

### 1. Authorize a bounded campaign once

Before step 2, retain the operator's existing authorization in
`evidence/live-pilot/control.md`: research question and allowed recipe space;
config and source revision; evaluator/metric, split and fixed protocol;
calibrated min_delta with measurements; max_trials and no_gain_limit;
maximum per-run wall time, total training GPU-hours (calibration and failed
attempts included), and any paid inference cap; TEST permission (false for this
pilot). Record numeric caps or a reference to an already authorized allocation.
Do not invent a spending limit or ask again when existing authorization covers it.
If limits are missing, finish read-only preparation, then request only the missing
allocation before launching compute. Run serially; no extra GPU rental/service.

### 2. Autonomous execution inside that boundary

ARIS may inspect TRAIN/DEV artifacts, wait/resume the same active run, correct
proposal formatting, propose allowed recipes, execute budgeted trials, and
report/finalize the controller-selected DEV result without per-trial approval.
It may revise a local recipe hypothesis after contrary DEV evidence; changing
the research question, model, data, initialization, seed, training budget,
selection objective or allowed search space requires operator approval and a
new campaign. Never rewrite old hypotheses or results.

Before each submit, record in control.md: cited latest result, recipe delta,
predicted observation, and an observation that would weaken the hypothesis.
Keep the existing proposal JSON schema unchanged; these notes are not extra JSON
keys. Prefer one changed factor for interpretable pilot trials. If changing
several allowed fields, state that only the combined recipe can be evaluated,
not the causal contribution of each field. A gain supports recipe selection,
not automatically its proposed mechanism.

### 3. Evidence gates are automatic; protocol changes are human decisions

Use the existing independent DEV reload, finite score/image-count checks, and
strict best + min_delta rule. Invalid is not negative scientific evidence;
no_gain is not a failed execution. Preserve both and the existing stop counters.
Evaluation reload spread calibrates numerical noise, not training-seed variance.

Any suspected evaluator/metric defect, preprocessing/crop/color-space mismatch,
or split contamination pauses new submissions. ARIS may diagnose and prepare a
patch with a small TRAIN/DEV fixture, but must not adopt a new evaluator, metric,
threshold, or protocol in the active campaign. Present the defect, affected runs,
patch and verification evidence to the operator. After approval, use a new
campaign and rerun baseline/comparisons under one protocol; retain the old record
as superseded. Do not repair a disappointing score by changing what is measured.

### 4. Preserve useful friction; bound mechanical recovery

A timeout means inspect/wait for the existing run, not submit another.
Operational repair is allowed within existing caps when new diagnostic evidence
supports it and the scientific protocol remains unchanged. Repeated identical failure,
unknown execution state, or insufficient remaining compute pauses new launches.
Never bypass duplicate-recipe rejection, edit campaign state, or extend trial
limits to obtain four valid runs. Invalid trials still consume the existing
budget; report an incomplete pilot when appropriate.

When a hypothesis is contradicted, log the contradiction and revise the next
in-scope proposal autonomously. When aggregate PSNR conflicts with DEV image
quality (e.g. face tone/texture), log the affected DEV examples, inspect per-image
evidence, and qualify the claim. Keep PSNR selection unchanged unless a new
objective is explicitly authorized. Escalate unresolved protocol/goal ambiguity;
do not escalate every no_gain. Friction is useful only when it changes an
assumption, supplies evidence, or resolves a decision. Repeated rewording or
rerunning without new evidence is a stop signal.

### 5. TEST and spending are separate permissions

Leave TEST untouched in this pilot, including images, labels, metrics and manual
script calls. Outside this pilot, a previously explicit authorization for final
held-out evaluation suffices: record it and freeze the DEV choice before
`final-test`; no redundant approval is needed. Otherwise request it once.
In legacy mode the CLI freeze check alone is not authorization; control mode also checks the frozen TEST permission. Once TEST is observed, do not
use it to select recipes or present that same set as fresh confirmation after
further tuning. Record accidental exposure and ask the operator for a new
held-out evaluation plan.

Before launching, check estimated cost/time against remaining authorized caps;
monitor elapsed usage and stop the job at its authorized runtime limit. Record
the interruption as incomplete/invalid, never as a score. If the host cannot
monitor/enforce the limit, arrange an operator/scheduler limit before launch.
Exceeding caps, additional seeds, larger data/resolution, extra trials or a new
paid service requires specific operator authorization. Preserve partial artifacts.

### 6. Separate reporting from scientific endorsement

ARIS autonomously emits a factual end report: workflow evidence, valid/invalid
trial count, baseline and retained/raw-best DEV scores, protocol, stop reason,
cost, contradictions and unresolved limits. Four valid real trials plus the
Naive feedback transcript establish workflow acceptance only; no gain is needed.
A fixed-seed DEV gain is exploratory. Generalization, TM-specific attribution,
and superiority to non-adaptive search need separate appropriate evidence.
Draft such claims with supporting/contradicting artifacts; the operator decides
whether to endorse or publish them. Do not hide a negative outcome or request
approval merely to report it.

For an escalation, append one compact decision record to control.md:
`trigger | evidence/run paths | proposed change | effect on comparability |
incremental cost | decision needed | operator decision + scope/time`.
Pause the affected action, continue safe read-only preparation, and resume only
within the recorded authorization. End the pilot with the operator reviewing
one rejected hypothesis and one unresolved alternative explanation, so execution
also develops research judgment.


## Opt-in fast/slow workflow

Follow [fast-slow-research.md](fast-slow-research.md) when the operator supplies a control contract. Initialize using `--control` and an optional frozen `--profile`; do not add these policy fields to training YAML. Read `next.feedback`, cite measured observation IDs separately from competing hypotheses, and submit the unchanged proposal with `--decision`. A historical `based_on` or explicit comparator never changes partial-recipe inheritance from effective best or fixed initialization.

The action table in step 3 governs this mode too: `next.next_action.action` selects the action; `proposal_allowed` only guards submission. A `diagnose` route with `review_required: true` requires `campaign review-packet` for its current trigger and a recorded slow outcome before fetching the next route. Save outcomes with `campaign record-decision` even if no run follows. A `diagnose`, `wait` or `request_scope_change` route pauses work without freezing the campaign. Resolving an operational hold requires new evidence and cannot enlarge search limits. Use `campaign feedback`, `review-packet`, `report` and `memory` for reviewable evidence; never invent measurements, scene tags, costs or visual access.

Respect durable `pending_slow_decision`: a logged diagnosis, pause, scope-change request, stop or finalize action is not permission to submit. A justified continuation requires an explicit slow-review `propose` matching the current trigger; an ordinary fast proposal cannot clear that gate. Collect existing jobs while paused. After an operational hold is resolved with `diagnose`, remain paused until the continuation decision is recorded. Stops and requests to change scope do not authorize reopening search.

Capture actual service startup/model exchanges and actual campaign tool results using [workflow-evidence.md](workflow-evidence.md), then register their manifest through the normal CLI. Preserve original records; never synthesize an accepted event log after the fact. Missing/incomplete or unsupported exports keep automatic W pending. Fixtures verify the capture and validator only; they are not a live Naive acceptance.

After freezing, a `confirm` route permits only the already authorized confirmation plan through `confirmation init/next/report`. Resume an existing plan rather than initializing it again; report terminal pairs without retrying them. The route supplies no new data or compute authorization. Original DEV still chooses checkpoints; independent data are evaluated after freeze and TEST is never relabeled. Preserve every failed/incomplete seed pair. Report workflow, recipe, strategy and product claims separately. Current implementation tests do not complete the live pilot described above.
