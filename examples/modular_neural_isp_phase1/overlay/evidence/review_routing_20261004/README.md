# Routing correction evidence — 2026-10-04

Base remote commit: `2afcafeeacbeff4860de2d9f2108214afca7c87b`, tree `8c2671a0e35d1d00385aa94adae5b3c510f0c629`. The local Git commit has the same base tree; the delivery adds only the prompt/guide correction, tests and these review records.

## RED and focused coverage

Before modifying production source, running the initial `tests/test_next_action_prompt.py` produced **9 failed in 0.07s**. [Raw RED output](routing-red.log). Those cases use persisted campaign state and the real next/action/decision paths, without launching training. The final test file adds a tenth case to preserve the submission guard when a legacy non-ready state still has a propose route.

## Root consolidated run

From repository root:

```sh
PYTHONPATH=examples/modular_neural_isp_phase1/overlay \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
 examples/modular_neural_isp_phase1/overlay/tests/test_next_action_prompt.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_slow_decisions.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_final_checkpoint.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_workflow.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_control.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_naive_adapter.py \
 examples/modular_neural_isp_phase1/overlay/tests/test_research.py \
 -k 'not q_claim_follows and not q_rejects and not memory_confirmation'
```

Exit 0: **142 passed, 4 deselected, 19 subtests passed in 21.89s**. [Raw output](routing-targeted-tests.log). Four existing native Q/memory integration cases remain deselected in this subset. Fixture/model-load/supervision substitutions in the existing core tests retain their documented engineering-only scope.

Independent review: [routing-review.md](../../docs/reviews/20261004/routing-review.md), **47 passed** prompt/slow tests and **2 passed** routing-precedence tests, no blocking findings. Source compilation and source/document whitespace checks passed; original pytest output is retained with its trailing spaces.

No complete-suite rerun, real GPU training, Naive model or TEST access occurred in this correction. [Previous full-suite failure evidence](../review_fixes_20261004/README.md) remains unchanged; deployment native acceptance and live scientific-value evidence remain outstanding. This correction changes Agent instructions, not the evaluator or process supervisor.
