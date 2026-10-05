# Route/task documentation alignment

Scope: only `overlay/docs/aris-campaign-task.md` and `overlay/docs/fast-slow-research.md`, based on the user's review of `2afcafee`. No code, test/statistic evidence, or previous validation records changed; no commit made.

Changes:

- Replaced early step 3's `proposal_allowed == false -> step 7` fallback with `next_action.action` as the sole action selector. The boolean is only a submission guard.
- Gated proposal writing/submission steps 4–6 on the `propose` route and true guard. An inconsistent route/guard requires current state inspection and a fresh `next`, never submission or inferred finalization.
- Added one action table for all seven existing actions. `wait` preserves/collects the same job or recorded pause; `diagnose` with a required review uses the current-trigger review packet plus `record-decision`, then a fresh route.
- Restricted step 7 to search closure: a valid unfrozen retained candidate and no active/pending launch are prerequisites for freezing. Invalid baseline/missing best and already frozen states are reported without blind re-finalization. Diagnosis, wait and scope-change pauses do not enter closure.
- Kept scope changes outside current model/data/protocol/budget bounds. `confirm` only resumes an already authorized plan bound to the frozen candidate; terminal plans are reported without recreation/retry, and a missing authorized plan stays pending.
- Aligned the later opt-in fast/slow section and confirmation command sequence with the same route contract.

Preserved: four recipe fields, independently reloaded DEV selection and frozen `min_delta`, fixed model/data/seed/initialization/trial budget, separate TEST permission, durable slow-decision/hold recovery, and W/Q/R/P evidence limits. No new agent, runtime, cost framework, or scientific-benefit claim.

Verification: reviewed current `route_next_action`, `finalize_campaign` and CLI argument handling; `git diff --check` passed. Text search confirms the removed `false -> step 7` and `once no more proposals are allowed` directives are absent from both edited documents. Code/prompt consistency regression is owned by the root agent; no execution or live-model acceptance was performed in this documentation subtask.
