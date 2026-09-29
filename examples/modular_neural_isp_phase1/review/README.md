# Code review evidence package

Start with `Code_Review_CN.md`.

`evidence/test_existing.log`: 11 existing adapter/runner tests rerun against connector-fetched fixed-version code.
`evidence/probe_results.json`: numerical and process fault-injection probes, not real model performance.
`evidence/environment.json`: executed environment and limitations.

No repository changes, model weights, data, or third-party source modules are included.

To rerun the diagnostics with the Phase1 overlay applied to an upstream ISP checkout:

```bash
python review_probes.py --overlay-dir /absolute/path/to/modular_neural_isp --out review_probe_results.json
```

The probe imports the existing `tests/test_runner.py` fixture. It creates temporary substitute train/test scripts to exercise process behavior. It does NOT invoke real training, download weights, call a real LLM, or modify the target repository. Torch and the overlay test dependencies must be installed. Results may differ after fixes; this report applies only to the recorded commit.
