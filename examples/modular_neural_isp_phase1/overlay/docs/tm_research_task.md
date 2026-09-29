# Fixed-model recipe runner and adaptive campaign

The runner trains the existing photofinishing model, selects its best validation checkpoint, and launches `photofinishing/test.py` in a separate process to reload that checkpoint on **DEV** (the `validation` split). Every ordinary run does this independently, even when no TEST split is configured. `test.py` receives explicit `--eval-size 512` by default; a smaller smoke setting is labeled `P<size>`. Training input size and evaluation size are separate controls.

The comparison score is float64 per-image PSNR with epsilon `1e-12`, averaged over images (`mean_per_image_psnr`, with `mean_psnr` retained as an alias). The runner requires an existing checkpoint, a successful independent reload, finite DEV score, and positive `num_images` equal to the actual number of DEV input PNGs. An optional split `expected_count` additionally checks the dataset count. A process can finish with `status=completed` and `result_status=invalid`; exit success alone does not establish a valid experiment. Campaign decisions use `valid`, `no_gain`, `improved`, or `invalid`, separately from execution status.

Set real paths and budget in `configs/baseline.example.yaml`. Only `loss_family` (`original`, `mse`, `l1`), `optimizer` (`adam`, `adamw`), `learning_rate`, and `weight_decay` may vary across campaign trials. Defaults retain the original recipe: original loss, Adam, learning rate `1e-4`, weight decay `1e-7`. Data, model, initial checkpoint, seed, epochs, batch size, input/evaluation sizes, and validation frequency stay fixed. If no initialization checkpoint is supplied, a fixed seed recreates the initial weights; campaign seed defaults to `1`. Each trial restarts training, rather than continuing from the previous best model.

## Single DEV trial

```bash
python -m tm_research.cli commands --config configs/baseline.example.yaml
python -m tm_research.cli baseline --config configs/baseline.example.yaml
python -m tm_research.cli wait --run exp_001 --runs-dir runs --timeout 60
python -m tm_research.cli report --run exp_001 --runs-dir runs
```

`baseline` returns a run ID with a detached worker. `commands` previews only training and DEV commands. `status`, `wait`, and `report` read persisted results. Python equivalents are `run_baseline(config)`, `get_result(run_id, runs_dir)`, and `wait_for_result(run_id, runs_dir, timeout=...)`. `load_config(path)` resolves YAML-relative paths. Ordinary runs omit TEST paths from their saved configuration and never report TEST scores.

Each run stores `config.json`, exact `commands.json`, `state.json`, worker/train/dev logs, training output under `train/`, and independent DEV output under `dev/`. `results.jsonl` records terminal runner states. The runner does not restore optimizer or scheduler state after a failed process.

## Serial persistent campaign

```bash
python -m tm_research.cli campaign init --config configs/baseline.example.yaml --campaign-dir runs/campaign --max-trials 5 --no-gain-limit 3 --wait
python -m tm_research.cli campaign next --campaign-dir runs/campaign
# An agent reads that real feedback and writes proposal.json.
python -m tm_research.cli campaign submit --campaign-dir runs/campaign --proposal proposal.json --wait
python -m tm_research.cli campaign next --campaign-dir runs/campaign
# Repeat one proposal at a time while proposal_allowed is true.
python -m tm_research.cli campaign status --campaign-dir runs/campaign
python -m tm_research.cli campaign finalize --campaign-dir runs/campaign
# A separate, explicit final evaluation after the DEV choice is frozen:
python -m tm_research.cli final-test --campaign-dir runs/campaign
```

`init` launches the baseline first. `max_trials` includes that baseline; each unsuccessful candidate (invalid or no gain) increments the consecutive `no_gain_limit` counter. Improvement resets that counter. An invalid baseline stops the campaign because there is no valid reference score. `finalize` can also deliberately stop a campaign early and freezes the best valid DEV run. Proposals are rejected afterward. The separate `final-test` command never changes the DEV selection.

An example proposal shape follows. Its reference must name an actual run from `campaign next` and its observed DEV score; the numbers below are illustrative. Rounding within `0.0001` dB is accepted, and comparisons always use the full recorded score. Invalid runs are cited with `dev_psnr: null`. Recipe fields omitted from a proposal inherit the current best recipe.

```json
{
  "recipe": {"loss_family": "mse", "learning_rate": 0.00005},
  "hypothesis": "The preceding DEV result suggests the current update size is too large; reduce it while testing MSE.",
  "based_on": {
    "run_id": "exp_001",
    "dev_psnr": 22.1234,
    "observation": "The baseline established 22.1234 dB; inspect its logged convergence before this change."
  }
}
```

The controller checks the citation and allowed controls, executes the real runner, compares the resulting DEV score, and saves the hypothesis, recipe, evidence, decision, best run and stopping counters in `campaign.json`. `campaign next` includes the prior results and decisions in the next proposer prompt. It never invents a proposal or calls a fixed grid adaptive research. ARIS-Code supplies the actual proposing agent through the task in [aris-campaign-task.md](aris-campaign-task.md).

Without `--wait`, `init` and `submit` return the pending run immediately. Use `campaign wait --campaign-dir runs/campaign --timeout 60` or `status` to collect its result; `next` also collects completed work. This supports restarting the host while a worker continues. Some sandboxes terminate all children when a tool call exits: use `--wait` there and keep the tool execution session alive. A timeout leaves the pending run intact; inspect its logs before retrying. There is no automatic recovery of training interrupted midway. If a worker was killed before it ever began (`queued`, empty worker log, no training output), the internal `_worker --run-dir <run-directory>` can run that queued task in the foreground, followed by `campaign wait` to collect it.

Python campaign functions are `initialize_campaign(config, campaign_dir, max_trials=5, no_gain_limit=3, wait=True)`, `next_proposal(campaign_dir)`, `submit_proposal(campaign_dir, proposal, wait=True)`, `campaign_status(campaign_dir)`, `wait_campaign(campaign_dir, timeout=None)`, `finalize_campaign(campaign_dir)`, and `final_test(campaign_dir)`. These use the same persisted state as the CLI. Re-running a terminal worker does not retrain it, and repeated final-test calls reuse only a previously validated report whose metric file still exists. No hashes or content digests are maintained.

Tests in `test_runner.py` and `test_campaign.py` use substitute tiny train/reload scripts and a clearly labeled simulated proposer. They verify orchestration and feedback decisions, not ISP quality or live Naive/ARIS operation. Real-model CPU integration and live service evidence must be reported separately.
