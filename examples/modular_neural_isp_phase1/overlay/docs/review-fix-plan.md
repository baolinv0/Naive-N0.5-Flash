# Review remediation plan

Goal: connect fixed-model recipe experiments to result-driven ARIS-Code/Naive research.
Spec: user-provided Code_Review_CN.md, batches A/B/C.
Constraints: functional minimal implementation; no hashes or digest tracking; no model edits, style transfer or historical enhancements; serial trials; no speculative defenses.

- [x] A/B Training: original/MSE/L1, Adam/AdamW, learning rate; retain original defaults. One float64 per-image PSNR definition with epsilon=1e-12. Best uses same metric. Explicit P512 evaluation (P<size> smoke; quarter separate). Independent dev reload and count. Same initialization/seed/budget across trials. Effective recipe saved.
- [x] Runner: expose recipe, dev-only ordinary runs, separate final test, completed/valid/improved distinction. Verify reload, sample count and finite score. Serial campaign; basic cache completion.
- [x] Campaign: prepare/next/submit/status/finalize interfaces, reference earlier result in proposal, restricted recipe overrides, actual dev comparison, trial/stagnation budgets, persistent state. ARIS task skill uses these tools; optional direct compatible-model loop shares controller. No hard-coded grid called adaptive.
- [x] Naive: preserve real termination reason, suppress incomplete reasoning/tool blocks, required/type checks. Test real-world output truncation.
- [x] Integration: real CPU model short original/MSE/L1 runs and reloads; same checkpoint repeat; controller test with real feedback and simulated proposer clearly labeled; independent review; GitHub delivery.

Live Naive/ARIS and real-camera PSNR claims require actual service/data; local tests do not substitute for them.
