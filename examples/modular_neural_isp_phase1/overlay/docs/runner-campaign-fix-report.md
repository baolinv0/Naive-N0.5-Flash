# Runner and campaign remediation

Implemented a serial, persistent recipe controller around the existing upstream training/reload scripts. Ordinary runs always independently reload the selected checkpoint on DEV, with explicit `eval_size=512` by default, and do not retain TEST paths in per-run configuration. A run becomes valid only after checkpoint existence, successful independent reload, finite mean per-image PSNR, exact positive expected sample count, and matching `P<size>` protocol checks. Execution status and result validity remain separate.

The recipe surface is `loss_family`, `optimizer`, `learning_rate`, and optional `weight_decay`; defaults remain original/Adam/1e-4/1e-7. Campaign state freezes data, initialization, seed (default 1), and all training/evaluation budgets. Baseline runs first. Each submitted candidate supplies a hypothesis plus a prior run/score/observation citation; ordinary display rounding up to 0.0001 dB is accepted. The controller executes the real runner and stores evidence, improvements/no-gain/invalid decisions and the current best. It enforces serial proposals, total trial and consecutive no-gain budgets. It does not generate proposals; the actual external agent does so after reading `campaign next` feedback.

CLI:

```text
python -m tm_research.cli campaign init --config YAML --campaign-dir DIR --max-trials N --no-gain-limit N [--wait]
python -m tm_research.cli campaign next --campaign-dir DIR
python -m tm_research.cli campaign submit --campaign-dir DIR --proposal JSON [--wait]
python -m tm_research.cli campaign wait --campaign-dir DIR [--timeout SECONDS]
python -m tm_research.cli campaign status --campaign-dir DIR
python -m tm_research.cli campaign finalize --campaign-dir DIR
python -m tm_research.cli final-test --campaign-dir DIR
```

Python equivalents in `tm_research.campaign`: `initialize_campaign(config, campaign_dir, max_trials=5, no_gain_limit=3, wait=True)`, `next_proposal`, `submit_proposal(campaign_dir, proposal, wait=True)`, `wait_campaign(campaign_dir, timeout=None)`, `campaign_status`, `finalize_campaign`, and `final_test`. Old `run_baseline`, `get_result`, and `wait_for_result` interfaces remain available, now for DEV-only trials.

`init`/`submit` without `--wait` return pending work; later status/next/wait calls collect a terminal result exactly once. Use `--wait` in tool sandboxes that kill background children when a command exits. Persistence resumes result collection, not optimizer state. Finalize freezes the DEV choice; a separate final-test command is then allowed and cannot alter the selection. Final-test cache requires a previous valid report and an existing metric file. No hashes or digests are added.

ARIS-Code integration is the executable task workflow in `docs/aris-campaign-task.md`, invoked through `aris "Read docs/aris-campaign-task.md ..."` with the existing configured Naive executor. It calls the same controller CLI, reads actual results and loops one proposal at a time. No parallel generic host or fixed grid is introduced.

Verification: `isp-venv/bin/python -m pytest -q tests/test_runner.py tests/test_campaign.py` initially passed **22 tests** (13.52 s), using substitute subprocess entrypoints. The tests include three result-dependent recipe changes, frozen seed/budgets, mandatory DEV reload, no routine TEST access, final TEST gate, no-gain stopping, malformed/unsupported proposals, stale/false citations, pending-work collection, nonfinite scores, wrong/zero sample count, absent checkpoints and child failures. These tests are simulated-proposer orchestration evidence, not live Naive or real-camera quality evidence. Root integration records actual-model smoke results separately.
