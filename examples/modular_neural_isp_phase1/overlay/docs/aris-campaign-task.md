# ARIS-Code task: adaptive fixed-model ISP recipes

Run this task inside the existing ARIS-Code host using its configured Naive executor and its normal file/Bash tools. Do not build or launch a second generic agent host. This file supplies the task-specific tool workflow; `tm_research.cli` supplies the callable controller. The operator supplies a working config path, campaign directory, maximum trials, no-gain limit and calibrated min delta in dB. Use the repository's Python environment.

Example host invocation, after configuring `configs/aris.example.json` fields in ARIS-Code:

```bash
aris "Read docs/aris-campaign-task.md and execute it. Config: configs/baseline.example.yaml. Campaign: runs/campaign. Max trials: 4. No-gain limit: 3. Min delta: 0.01 dB. Use the configured Naive executor. Do not run TEST."
```

1. Read the config and this task. The model and data protocol are fixed. Allowed research changes are only `loss_family` (original/MSE/L1), Adam/AdamW, learning rate, and optional weight decay. Do not edit model files, data, metrics, seed, initialization, or trial budget.
2. If the campaign does not exist, call `python -m tm_research.cli campaign init --config <config> --campaign-dir <campaign> --max-trials <limit> --no-gain-limit <limit> --min-delta <calibrated-dB> --wait`. Otherwise resume with `campaign status`, preserving its stored threshold. Initialization runs the baseline before any proposal. Use adequate Bash execution time and keep its session active. If your host preserves background workers, you may omit `--wait` and use `campaign wait --timeout 60`; if a wait times out, continue collecting the same run instead of submitting another.
3. Call `python -m tm_research.cli campaign next --campaign-dir <campaign>`. Read the complete returned DEV evidence, latest decision and best recipe. If `active` is present, wait for it, then call `campaign next` again to obtain fresh evidence before checking `proposal_allowed`. If `proposal_allowed` is false in that fresh response, stop proposing and go to step 7.
4. Reason from the actual latest result. Explain which recipe change should improve DEV performance and why. Consider failed hypotheses and invalid runs as evidence. Write one JSON proposal containing `recipe`, a nonempty `hypothesis`, and `based_on` with a recorded `run_id`, its observed `dev_psnr` (null for an invalid run), and a concrete `observation`. Read `docs/tm_research_task.md` for the JSON schema. You can inspect that run's DEV logs and artifacts to understand a failure. Never inspect TEST images or TEST metrics to choose a recipe.
5. Call `python -m tm_research.cli campaign submit --campaign-dir <campaign> --proposal <proposal.json> --wait`. This actually trains and independently reloads the model. Do not replace it with a predicted score, mock tool result, or grid lookup. A failed CLI call is not a scientific result; inspect the returned error and correct only the proposal or operational issue it identifies.
6. Read the stored run result and controller decision. Explicitly distinguish `completed` from `valid`, and `no_gain` from `improved`. `raw_best` records the highest score; `best` is the retained effective candidate. Only a score above `best.dev_psnr + min_delta` is improved. Do not change the frozen threshold or treat a raw-best fluctuation as progress. Return to step 3 and choose the next recipe using this new evidence. Never write all proposals in advance.
7. Call `python -m tm_research.cli campaign finalize --campaign-dir <campaign>` once no more proposals are allowed (or the user explicitly asks to stop). If the baseline was invalid, report its logs and failure rather than freezing a nonexistent best model. Report the chosen DEV run, recipe, score, protocol, trial count, and stop reason. TEST is a separate final operation: call `python -m tm_research.cli final-test --campaign-dir <campaign>` only when final held-out evaluation was included in the user's task, after freezing the choice. Do not feed its result back into proposal selection.

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
