"""Small data-pairing and validation helpers for the original photofinishing baseline."""

from pathlib import Path
import math

import numpy as np
import torch

from utils.file_utils import read_json_file
from utils.img_utils import clip, imread, imresize, raw_to_lsrgb

PSNR_MSE_EPSILON = 1e-12


def paired_files(input_dir, gt_dir, metadata_dir):
  """Match PNG inputs, JPG targets and JSON metadata by scene stem."""
  def files(directory, suffixes):
    result = {}
    for path in Path(directory).iterdir():
      if path.suffix.lower() not in suffixes:
        continue
      stem = path.stem
      if stem.endswith('_denoised'):
        stem = stem[:-len('_denoised')]
      if stem in result:
        raise ValueError(f'Duplicate scene stem {stem!r} in {directory}')
      result[stem] = path
    return result

  inputs = files(input_dir, {'.png'})
  targets = files(gt_dir, {'.jpg', '.jpeg'})
  metadata = files(metadata_dir, {'.json'})
  if not inputs or set(inputs) != set(targets) or set(inputs) != set(metadata):
    missing = {label: sorted(set(inputs) - set(group))[:5]
               for label, group in [('ground truth', targets), ('metadata', metadata)]}
    extra = {label: sorted(set(group) - set(inputs))[:5]
             for label, group in [('ground truth', targets), ('metadata', metadata)]}
    raise ValueError(f'Input/ground-truth/metadata scenes do not match: missing={missing}, extra={extra}; '
                     f'input_count={len(inputs)}')
  return [(str(inputs[stem]), str(targets[stem]), str(metadata[stem])) for stem in sorted(inputs)]


def image_psnr_values(output, target):
  """Unit-range PSNR per image; float64 MSE is clamped at 1e-12 (120 dB)."""
  difference = output.detach().to(torch.float64) - target.detach().to(torch.float64)
  mse = difference.square().mean(dim=(1, 2, 3)).clamp_min(PSNR_MSE_EPSILON)
  return (-10 * torch.log10(mse)).cpu().tolist()


def summarize_psnr(values):
  """Summarize image scores without weighting images by batch membership."""
  finite = bool(values) and all(math.isfinite(value) for value in values)
  mean = math.fsum(values) / len(values) if finite else None
  return {'mean_per_image_psnr': mean, 'mean_psnr': mean,
          'num_images': len(values), 'finite': finite}


def best_checkpoint_index(scores):
  """Choose the highest finite mean per-image PSNR; ties retain the earlier epoch."""
  candidates = [(index, score) for index, score in enumerate(scores)
                if score is not None and math.isfinite(score)]
  if not candidates:
    raise ValueError('No finite validation PSNR is available for checkpoint selection')
  return max(candidates, key=lambda item: item[1])[0]


def load_image_pair(input_path, target_path, metadata_path, image_size=None, quarter=False):
  """Shared validation/reload color conversion and linear square resize."""
  raw = imread(input_path)
  target = imread(target_path)
  metadata = read_json_file(metadata_path)
  input_image = clip(raw_to_lsrgb(raw,
                                illum_color=np.array(metadata['cam_illum'], dtype=np.float32),
                                ccm=np.array(metadata['ccm'], dtype=np.float32)))
  if image_size is not None or quarter:
    height, width = ((image_size, image_size) if image_size is not None
                     else (raw.shape[0] // 4, raw.shape[1] // 4))
    input_image = imresize(input_image, height=height, width=width)
    target = imresize(target, height=height, width=width)
  return input_image.astype(np.float32), target.astype(np.float32)
