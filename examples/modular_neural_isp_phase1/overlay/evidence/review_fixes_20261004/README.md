# Review correction verification, 2026-10-04

Source base: `f3ee3784a32cd690eb67cb5f47a37b93db8fab3a`; this records the current uncommitted correction tree later delivered with these files. Python 3.12.14, PyTorch 2.5.1 CPU, NumPy 1.26.4, pytest 9.1.1. No real model or camera-data experiment occurred.

## Root consolidated targeted run

From repository root:

```sh
PYTHONPATH=examples/modular_neural_isp_phase1/overlay \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
 examples/modular_neural_isp_phase1/overlay/tests/test_slow_decisions.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_final_checkpoint.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_control.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_naive_adapter.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_research.py \
 -k 'not q_claim_follows and not q_rejects and not memory_confirmation'
```

Exit 0: **132 passed, 4 deselected, 19 subtests passed in 21.51s**. The four omitted existing native Q/memory integration cases are included in the attempted full run below. [Raw output](targeted-tests.log).

## Current complete-suite attempt

From `examples/modular_neural_isp_phase1/overlay`, using the existing official ISP checkout:

```sh
PYTHONPATH=/workspace/scratch/6c78134e8d18/isp-runtime:/workspace/scratch/6c78134e8d18/isp-runtime/photofinishing \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
timeout 240 ../../../.venv/bin/python -m pytest -q
```

The run completed before timeout, exit **1**: **115 failed, 246 passed, 18 warnings, 19 subtests passed in 156.91s**. [Complete raw output](full-tests.log). Failure groups: campaign 7, confirmation 30, fast/slow native integration 21, previous final-review native regressions 28, research Q/memory native setup 4, runner 25. New slow-decision, final-checkpoint, workflow and adapter files have no failed cases in this run.

[PID environment probe](pid-environment.json) records the mismatched inner PID and host `/proc` view. The unchanged production owner/stage/process-group checks refuse or cannot classify actual child processes in this environment. Native process tests must be rerun in the deployment environment. This is a failed full-suite attempt, not a passing result or proof of production integration.

Independent log triage found 62 invalid-winner setup failures, 28 invalid-baseline downstream failures and 25 unchanged runner-boundary failures. No new F1/F2/F3 regression was identified; the log alone cannot establish that every failure has only an environment cause. See [F1/F3 triage](../../docs/reviews/20261004/full-suite-triage-F1-F3.md) and [F2 triage](../../docs/reviews/20261004/full-suite-triage-F2.md).

## Historical CPU result recollection

[Recollection record](cpu-recollection.json) compares the previously completed synthetic CPU task/run identities and frozen search state under current modules. No new task, training, evaluation or score was added. W remains pending; Q is restricted to original DEV synthetic scope; R/P unavailable. The historical training results remain attributed to their execution versions.

Independent correction-scope review and implementation reports are in [reviews/20261004](../../docs/reviews/20261004/). Engineering fixtures deliberately substitute the model-load and/or execution boundary; they cannot establish live Naive/ARIS W or real-data quality. Production runner code is unchanged by these corrections.

Source compilation and the staged source/document whitespace check passed. Raw pytest logs are retained byte-for-byte, including pytest's trailing spaces; the check excludes `*.log` rather than rewriting evidence.
