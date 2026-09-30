# Training and evaluation review fixes

The model implementation is unchanged. Training now selects a loss/optimizer recipe while retaining the original defaults: composite photofinishing loss, Adam, learning rate `1e-4`, weight decay `1e-7`, and the existing cosine schedule (`T_max=epochs`, `eta_min=learning_rate/100`).

## Interface

| Entry point | Flag | Default / meaning |
| --- | --- | --- |
| `photofinishing/train.py` | `--loss-family original\|mse\|l1` | `original` |
| `photofinishing/train.py` | `--optimizer adam\|adamw` | `adam` |
| `photofinishing/train.py` | `--learning-rate FLOAT` | `0.0001` |
| `photofinishing/train.py` | `--l2reg FLOAT` or `--weight-decay FLOAT` | `0.0000001` |
| `photofinishing/train.py` | `--in-size SIZE` | Training size, default `512` |
| Both scripts | `--eval-size SIZE` | Square dev/reload size, default `512` |
| `photofinishing/test.py` | `--quarter-resolution` | Explicit separate legacy quarter-resolution inference |
| `photofinishing/test.py` | `--no-ds` | Explicit separate full-resolution inference |

MSE and L1 use unit-weight loss on the final sRGB output only. They do not construct the original loss or VGG, and do not compute linear/color-space targets for unused auxiliary losses. Adam and AdamW retain their respective coupled/decoupled weight-decay semantics. Original loss weights still apply only to the original family.

`--seed` remains available, and `--load` plus `--load-config-dir` permit shared initialization. The runner is responsible for holding initialization, seed and budget fixed across recipe trials. Saved model config and training/evaluation metrics include the effective recipe: loss family, active weights, optimizer, learning rate, weight decay, betas, scheduler, epochs, batch size, train/eval size, augmentation-patch choice, seed, validation frequency and initial-checkpoint path.

## Metric and protocol

`image_psnr_values` computes each image's MSE after converting both tensors to float64, clamps MSE to `1e-12`, and returns `-10*log10(MSE)` for unit-range images. A perfect match therefore scores 120 dB. Reported PSNR is the arithmetic mean over images; batch sizes do not determine image weights. Non-finite image scores cannot become valid aggregate scores.

Checkpoint selection, validation logs and independent reload evaluation all use this definition. The original loss's internal reporting-only PSNR is replaced before reporting. The historical pooled-MSE score no longer chooses the best checkpoint. For candidate A with image MSEs `0.0001, 0.01`, mean per-image PSNR is 30 dB; candidate B with MSEs `0.002, 0.002` scores 26.9897 dB. A wins even though pooled-MSE PSNR would rank B higher.

Both scripts report:

```json
{
  "mean_per_image_psnr": 30.0,
  "mean_psnr": 30.0,
  "num_images": 2,
  "finite": true,
  "protocol": "P512",
  "eval_size": 512,
  "recipe": {}
}
```

`mean_psnr` is a compatibility alias of `mean_per_image_psnr`. Training also reports `best_checkpoint` and `config_dir`; independent evaluation adds `mean_ssim`, `mean_time_seconds`, `per_image_csv` and `images_dir`. Evaluation of older checkpoints lacking recipe metadata reports `recipe: null`.

P512 and P256 share exactly the validation preprocessing: read image pairs by scene stem, convert raw camera RGB using illuminant/CCM, clip linear sRGB, and linearly resize input and target to the selected square size. Dev images are not augmented or split into patches, even when patch extraction is enabled for training. Evaluation size is independent of training size. P256 is a smoke protocol and must not be compared as P512.

For P<size>, independent reload uses `eval()` and `no_grad()` with `training_mode=True`, matching the validation forward path in the unchanged model. `--post-process-ltm` is rejected for this protocol. Quarter/full evaluation uses the model's inference path and reports `protocol: "quarter"` or `"full"`, with `eval_size: null`. The resolution flags are mutually exclusive. Small numerical differences between batch layouts remain possible in the model's float32 forward pass even though score aggregation uses float64.

## Cache and checks

Preprocessing writes a `COMPLETE` marker only after all HDF5 batches have been produced. Missing markers, including caches from interrupted or older preprocessing, cause a rebuild. `--overwrite-temp-folder` also rebuilds. This is a completion check; users must explicitly overwrite a completed cache when changing source data in place.

Executed in the provided Python 3.12 environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_isp_baseline.py tests/test_isp_recipes.py -q --disable-warnings
```

Result: **16 passed**, 20 dependency deprecation warnings, 13.31 seconds. Checks cover scene pairing, repeated HDF5 reads, interrupted-cache rebuild/reuse, float64/epsilon/non-finite handling, the ranking counterexample and actual best-checkpoint save, uneven batch partitions, P256/P512 preprocessing and independent evaluation parity, MSE/L1 gradients and unused-loss isolation, optimizer decay behavior, and original recipe/scheduler defaults. Real-model CPU smoke runs and independently repeated checkpoint reloads are recorded in the integration report; synthetic-data results are pipeline checks, not camera-quality claims.
