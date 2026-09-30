# Independent integration review

Verdict: **source changes approved for the scoped workflow; no blocking correctness issue found.**

Reviewed `review_fix_evidence/review.patch`, the remediation plan, the training/evaluation and adapter reports, the current training/evaluation helpers, runner, campaign controller, adapter, CLI, and focused tests. Follow-up review checked the completed `tm_research_task.md`, `aris-campaign-task.md`, and `naive-setup.md` instructions against the CLI, including the documented prior-score rounding tolerance. This review inspected integration and normal user flows; it did not rerun the test suite or execute another model campaign.

- Validation and independent reload share image conversion, clipping, resizing, float64 per-image PSNR, and the finite-score convention. Checkpoint selection uses the arithmetic mean of those image scores. The square evaluation protocol uses the same model forward path as validation; quarter/full inference is explicitly separate.
- Original/MSE/L1 and Adam/AdamW choices reach the trainer, effective recipe metadata is saved, and the original defaults remain intact. A campaign fixes initialization or seeded model construction, data, seed, and training/evaluation budgets while permitting only the four documented recipe fields to vary.
- Ordinary runs select a checkpoint and reload it on DEV. The runner checks checkpoint presence, successful reload, finite PSNR, protocol, and independently counted inputs before accepting the result. Campaign comparisons use the reload score rather than the training report.
- Campaign proposals are serial and persisted with prior-result references, actual results, decisions, and stop conditions. A failed or nonimproving trial cannot replace the best valid result. Finalization freezes that choice; held-out TEST requires a separate command.
- The adapter suppresses unfinished reasoning/tool blocks and token-limited calls, checks declared tools and the documented argument schema subset, and retains EOS versus token-limit termination metadata.
- The completed ARIS-Code/Naive instructions use the result-driven campaign commands, require reading actual DEV feedback before the next proposal, distinguish process completion from valid/improved results, and keep final TEST separate. The prior baseline-only documentation gap is resolved.

The coordinating agent reported a fresh complete suite result of **49 passed**, with **20 dependency warnings and 19 unittest subtests**. That is reported evidence, not an independently repeated run. Real CPU smoke execution and repeated checkpoint reload evidence belong in the integration verification report.

Limits: substitute-script campaign tests establish controller behavior, while synthetic real-model runs establish pipeline execution. Neither establishes camera-quality improvements, live Naive compatibility, or an end-to-end ARIS-Code/Naive campaign. The completed task instructions state these limits and require a real configured service and session evidence for a live research claim.

Documentation follow-up: the ARIS task now explicitly refreshes `campaign next` after waiting for an active run, before checking `proposal_allowed`.
