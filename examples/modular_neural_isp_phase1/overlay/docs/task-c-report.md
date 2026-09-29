> 历史记录：本文件描述最初 baseline 阶段。当前功能、协议和范围以 [engineering-contract.md](engineering-contract.md) 和 [review-fix-verification.md](review-fix-verification.md) 为准。

# Task C runner report

Implemented `tm_research.runner` and `tm_research.cli` around the upstream photofinishing train/test scripts, with YAML path resolution, command preview, detached worker, status/wait/report, isolated `exp_NNN` outputs, logs, and metrics artifacts. The wrapper only passes training knobs explicitly given in config. An initialized checkpoint's existing `use_3d_lut` config determines the upstream load flag. The worker consumes upstream `train/metrics.json` for the selected model/config and upstream `test/metrics.json` for test metrics.

Process-level tests use tiny substitute entrypoints, not the actual ISP model. Five checks passed under `isp-venv/bin/python -m pytest -q tests/test_runner.py` (5 passed in 1.87s): YAML-relative paths and omitted defaults; train→test handoff and independent run IDs; failed training retains a log and skips test; CLI wait/report; checkpoint LUT setting. `python -m tm_research.cli commands --config configs/baseline.example.yaml` produced absolute upstream script/data paths and used the configured venv interpreter. No real image dataset, model training, or test PSNR was used in this task's verification. Root owns the full integration run.

A failed run is terminal and a new run ID starts a new process. Weight initialization through `--load` does not resume optimizer, scheduler, or epoch. The test command preserves upstream quarter-resolution preprocessing; the training metrics and test metrics describe separate measurement protocols.
