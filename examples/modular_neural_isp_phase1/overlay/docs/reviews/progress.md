# SDD ledger — plan: examples/modular_neural_isp_phase1/overlay/docs/plans/aris-fast-slow-research-20261003.md

Base: 91d8577f63231344c8b03c555dbde9511212db8b. Branch: feat/aris-fast-slow-research-20261003.

Ruling: Implement on a new branch in a fresh clone rather than another linked worktree — no existing user checkout is affected — cost if wrong: branch relocation only.
Ruling: Dispatch grouped tasks by file ownership, with campaign/CLI integrations owned by one worker and all commits made centrally — avoids shared index and overlapping edits — cost if wrong: integration rework.

| Task | Internal consistency | Shared interfaces and consumers | Resolution |
|---|---|---|---|
| T1 control | New mode opt-in; legacy unchanged | control → T4/T5/T6/T7 | control dict frozen outside scientific YAML |
| T2a/T2b diagnostics | CSV V1 independent from float ROI V1.1 | profile/measurements → T3; test.py/runner.py flags | profile ref optional; runner forwards without altering metrics |
| T3 evidence | Stable IDs and immutable revisions | evidence → T4/T5/T7/T8 | sole full feedback assembler; invalid/baseline applicability explicit |
| T4 campaign/CLI | Three-key proposals preserved | consumes T1/T3/T5/T6, exposes T7 | one integration owner |
| T5 research | hard stops dominate slow suggestions | control/evidence → route/decision/record | no training in reviewer |
| T6 runner | persistent active before start | prepare/start → T4; evaluate checkpoint → T7 | legacy run_baseline wrapper retained |
| T7 confirmation | frozen recipes, no search mutation | runner/control/evidence → report | separate arm/seed task keys; original DEV checkpoint selection |
| T8 docs/claims/memory | claims follow evidence | research + all APIs → docs/verification | no live claim without real service/data |
| T2 ↔ T3 | same profile schema | sample IDs/tags/fixed masks | shared contract below |
| T1/T3/T5/T6 ↔ T4 | functions produced independently | campaign state + feedback + runner launch | explicit keyword contracts; legacy defaults preserved |
| T6 ↔ T7 | train versus frozen evaluation separate | checkpoint/config/split refs | no confirmation data in training validation |

Progress: environment setup and baseline verification underway.

Baseline: default sandbox 35 passed, 3 HTTP socket permission failures; rerunning with local socket permission. CPU torch installed.

Baseline verification: complete official ISP + exact baseline overlay, CPU PyTorch 2.5.1, 54 passed, 19 subtests; 18 upstream dependency deprecation warnings. Earlier full-suite failures diagnosed as old official train/dataset still loaded; explicit copying corrected test deployment, no production baseline fix.
Ruling: Add a same-host lightweight runner watchdog to enforce job deadline after worker SIGKILL without controller polling — required by T6 actual walltime bound — cost if wrong: extra local monitoring process per run.

T2a/T2b/T3 implemented (commit c1572d1); 35 targeted tests pass. Independent task review underway; not marked complete until reviewed.
Ruling: job_walltime_seconds is inclusive of termination cleanup; begin soft termination before the hard deadline instead of silently adding a budget allowance — aligns execution with the frozen authorized bound — cost if wrong: short timeouts reduce computation time.

T2/T3 review round 1: six Important findings accepted (dataset/profile identity, measurement/mask compatibility, complete expected ROI inventory, incompatible protocol gating, baseline groups applicability, missing detail repair/access). Routed by ownership to diagnostics/evidence implementers. Minor case/dedup/provenance findings included. Native nonbaseline score facts also required when optional CSV is unavailable.

T1/T5/T7/T8 review round 1: eight Important findings accepted, routed to research/control and confirmation owners. Reconcile partial/unknown actual usage and final TEST activity; bind frozen parent controls/holds; reject mismatched or jointly incomplete CSV diagnostics; native blank sample-ID fallback; verified live provenance and frozen Q inventory; external confirmation receipt association. Minor scoped memory/cost/scene items included.
Runner implemented in 55d71ee:34 targeted tests green; independent lifecycle review active. Campaign/CLI/docs owner reports28 targeted tests green; integration task review pending.

T6 runner review round1: R1 monotonic identity publication after exited stage leader and R2 post-launch persistence-failure cleanup accepted; runner owner fixing both with deterministic regression probes.

T2/T3 fix round1 committed798ae04; root fresh52 diagnostics/evidence tests green. Scoped re-review active.
T4 review round1: three Important accepted (durable pre-prepare launch intent, optional diagnostic exception fallback, refresh same-run latest feedback before proposal/ack). Routed to campaign owner; runner owner provides idempotent prepare request_id.
Ruling: compare explicit comparator against effective-best actual resolved configuration, excluding only unused TEST from ordinary search configuration — follows what was actually executed — cost if wrong: comparator rejection/comparison would require correction.
Ruling: use control identity mean_per_image_rgb_psnr for the existing native mean_per_image_psnr computation — names reflect the same RGB score — cost if wrong: metric identity validation requires correction, with no score recalculation intended.
Ruling: keep optional/legacy missing diagnostics nonblocking and gate only declared required evidence — follows the research contract — cost if wrong: a campaign would need a stricter future authorized requirement.

T2/T3 scoped re-review1: six original Important and three Minor ADDRESSED at798ae04; new N1 malformed optional ROI metric-definition escapes feedback assembly. Round2 routed to original evidence implementer, with cross-task minimal optional-failure fallback helper. No native/whole-image facts may disappear on optional ROI error.

T6 fix round1/5: R1/R2 ADDRESSED, no Important open; commits798ae04..c128e96. Rootfresh42tests green; scopedreview4test/partialprepareprobe green. Task T6 complete (b86008e..c128e96, review clean).

Ruling: recheck current hold/control/liveness before a recovered authorized intent starts, excluding only its own intent reservation from the gate — protocol holds dominate actual launches — cost if wrong: queued recovery delays until diagnosis. Campaign owner adding targeted hold/recovery regression.

T1/T5/T7/T8 fix round1 committed828b35c: rootfresh63 control/research/confirmation tests green; scopedCR re-review active. T3 fixround2 committedbe7a39f: rootfresh35 evidence tests green; malformedROI/fallback scopedreview active. Campaign owner38tests green afterhold-recovery ruling; centralizedfull-suite now running.

Centralfullsuite326009c scientifictree:225passed19subtests69.66s;18 upstreamdeprecationwarnings. ActualISP CPUbaselinevalidPSNR21.33483380210185; profile/ROI/feedbackcomplete andproposalready. Scriptedengineeringthreeproposalsrunning (notliveNaive).
T2/T3 fixround3/5: N2ADDRESSED,newImportant0; commits326009c..25f927f. Root43Etests green. T2/T3 complete (9c0b194..25f927f, reviewclean).
T4 scopedreview1: originalI1–3ADDRESSED; newR1 currentholdgate missing active/not_started recovery branch. Round2routedF. T1/T5/T7/T8 scopedreview1: originalI1..8ADDRESSED; newN1 rejectedconfirmationreceiptpoisonsledger routedC, residualmemoryclaimscopeMinor routedS.

Finalcentralfullsuite afterallcurrentfixes:238passed19subtests75.77s;18 upstreamdeprecationwarnings. ActualISP CPU4search+2confirmationjobs completed, allvalid andcostmeasuredCPU0GPUh. OriginalDEVonesyntheticimage/seed7 pair matchesnativeaggregate+CSV/scenes; Wpending,R/Punavailable. NoTestevaluated. Nativefixturegainsnotrealqualityclaims. T4fixround2+CRfixround2 committed; scopedfinaltaskreviewsnext.

T4fixround2/5: R1ADDRESSED,newImportant0; commits5a6a8e0..d15482d. TaskT4complete(55d71ee..d15482d,reviewclean).
CRfixround2/5: N1ADDRESSED,memoryscopeMinorADDRESSED,newImportant0; commits25f927f..5a6a8e0. TasksT1/T5/T7/T8implementationcomplete(c1572d1..5a6a8e0,reviewclean); liveacceptancependingdocumented.
Alltaskreviewgatesclean. Nativeacceptancefacts+tests saved in repo docs/fast-slow-verification.md and evidence/fast_slow_cpu/. Finalwholebranchreviewnext; nodeferredImportant/parkedfindings.

Finalwholebranchreview43345a3:0Critical,3Important,0Minor. ONEfinalfixwaveassignedfreshgpt6astraowner: I1currentparentgate allconfirmation actualstarts; I2knownsceneoverlap report/Q/memoryscope; I3unresolvedevaluatorterminalreceipt losesglobalactivityincCPUandfinalTEST. No publicationyet; native6jobsand238regressionremaincorrectlyboundedtocurrentnativecode. Single scopedfinalre-reviewafterfix.

Final ONE fix wave committed 9d0731a: I1/I2/I3 addressed. Root fresh full suite 266 passed / 19 subtests / 101.83s, exit 0. Scoped final rereview APPROVED, 28 passed / 24.88s, 0 open findings, 0 new Critical/Important. Native terminal recollection on updated modules preserves same 6 jobs/IDs/search best; no retraining or new evaluation. All implementation/review gates closed; live Naive and real-camera scientific acceptance remain explicitly pending.
