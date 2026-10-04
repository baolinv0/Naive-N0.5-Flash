# Current implementation contract

This review follow-up adds controlled training-recipe research to the original ISP baseline. It supersedes the earlier baseline-only implementation contract; the earlier phase remains documented in verification.md.

## Fixed

Official photofinishing model structure, RAW/metadata color preprocessing, datasets, initialization, seed, training budget, augmentation and evaluation protocol remain fixed within one campaign. No historical user enhancement, style-transfer teacher, or model-code search is included.

## Adjustable

Only loss_family (original/mse/l1), optimizer (adam/adamw), learning_rate and weight_decay may change. Original defaults remain available. Every run stores its effective configuration.

## Metric and execution

PSNR is the mean of per-image RGB PSNR, float64 reduction, range [0,1], minimum MSE 1e-12. The same definition selects best checkpoints, evaluates dev, and compares trials. P512 is the default; smaller engineering fixtures explicitly use P<size>. Legacy quarter/native results have separate protocol names.

Ordinary trials train, reload the selected checkpoint in another process and evaluate dev. Missing/unloadable checkpoint, nonfinite score or incorrect sample count is not a comparable result. Completed execution is distinct from a valid experiment and from a PSNR improvement. The current evaluator shares fixed repository code; it is not a separately isolated evaluation service. Model/code search is outside this contract.

## Research loop

ARIS-Code uses the task-specific campaign CLI to read actual dev evidence, write a hypothesis and recipe with a referenced parent result, run one experiment, and repeat. The controller persists evidence, compares scores and stops at the configured trial/stagnation limit. Trials run serially. Finalize freezes the selected recipe; final-test is separate and never feeds the search prompt.

Prepared runs are persisted before startup and recover by the same run identity; unknown liveness blocks new work. Interrupted or failed attempts remain invalid and are not automatically retried. There is no optimizer/RNG resume or multi-GPU scheduling. Opt-in paired-seed confirmation reuses frozen baseline/winner recipes independently of search; TM-only attribution remains a separate experiment. Synthetic fixtures and simulated proposers must be labeled; neither establishes real-camera gains or live Naive research ability.

Functionality comes first: only checks that directly affect runnable experiments or comparable scores. No content fingerprint system, extreme-case matrix, or mechanical review score.


## Opt-in evidence and authorization

`campaign init --control` freezes an explicit authorization snapshot; `--profile` enables candidate-independent TRAIN/DEV diagnostic mappings. Legacy three-key proposals, scientific configuration and zero-threshold compatibility remain. Control-mode submissions additionally require an independent decision sidecar acknowledging the latest evidence and requested recipe patch. Selection is persisted before separate diagnostic assembly; required missing observations block proposal readiness without invalidating scientific scores.

Resource checks preserve confirmation/calibration reserves, account actual usage of successful and failed attempts and keep active reservations. The runner enforces per-job walltime across TRAIN plus independent DEV. Slow decisions can establish an evidence-resolved research hold, but cannot override limits or frozen state. TEST permission is checked in control mode; legacy final-test keeps its existing explicit finalize contract. These checks coordinate normal CLI use; they are not a sandbox against arbitrary shell commands.

See [fast-slow-research.md](fast-slow-research.md) for actual commands, schema and claim boundaries. This feature provides engineering mechanisms, not evidence of live Naive research, confirmed recipe gain or product quality.
