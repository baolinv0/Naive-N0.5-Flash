# Original photofinishing baseline runner

This runner executes the repository's `photofinishing/train.py` and, when the `test` split is supplied, `photofinishing/test.py`. It does not change model, preprocessing, augmentation, loss, optimizer, or scheduling. Install the repository dependencies and provide paired input, GT, and metadata directories using the upstream layout before running it. The example at `configs/baseline.example.yaml` needs real data paths.

From the repository root:

```bash
python -m tm_research.cli commands --config configs/baseline.example.yaml
python -m tm_research.cli baseline --config configs/baseline.example.yaml
python -m tm_research.cli status --run exp_001 --runs-dir runs
python -m tm_research.cli wait --run exp_001 --runs-dir runs
python -m tm_research.cli report --run exp_001 --runs-dir runs
```

`baseline` prints a run ID and starts a separate background worker. `wait` blocks until completion and accepts optional `--timeout SECONDS`; status/report are immediate. The Python API is `run_baseline(load_config(path)) -> run_id`, then `get_result(run_id, runs_dir=...) -> dict` (or `wait_for_result(...)`). Relative paths in YAML are resolved against its directory. Relative paths in a direct dict passed to `run_baseline` are resolved against the caller's current directory. Commands passed to the upstream scripts contain absolute file paths; `commands` previews training and testing arguments, showing `<best_checkpoint>` and `<train_config_dir>` placeholders until training selects them. Any omitted control knobs keep the original upstream defaults.

Each `exp_NNN` directory contains the resolved `config.json`, exact `commands.json`, `state.json`, worker/train/test logs, upstream training output under `train/`, and testing output under `test/`. `runs/results.jsonl` appends one terminal state per run. `get_result` includes `status`, log paths, metrics, and artifact paths; a failed child reports its exit code and the relevant log remains available. A failed run starts fresh as a new run ID: `--load` initializes model weights only, and this wrapper cannot restore the optimizer, scheduler, or epoch as a full checkpoint resume.

Training validation `mean_psnr` is an average of per-image PSNR; `original_batch_psnr` identifies the original batch-level aggregate. Test `mean_psnr` is its separate test protocol, which preserves the upstream quarter-resolution default. Compare these values only with their protocol and data split stated. The wrapper passes no `--no-ds` flag and does not modify the official test preprocessing. `init_checkpoint` needs its matching config directory (`init_config_dir`) when the checkpoint config is outside the upstream default location. The runner reads that existing JSON to preserve its `use_3d_lut` setting when loading weights; this does not introduce a new model setting.

The automated runner tests use tiny substitute entrypoint scripts to verify process behavior and artifact handoff. Those fixture outputs do not measure ISP quality or validate the actual model. A real baseline requires the installed dependencies, data, and enough compute for the selected upstream training budget.
