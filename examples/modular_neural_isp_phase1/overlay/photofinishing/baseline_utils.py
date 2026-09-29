"""Small data-pairing and validation helpers for the original photofinishing baseline."""

from pathlib import Path

import torch


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
  """PSNR for each image in a BCHW float tensor, with the same unit range as training."""
  mse = (output.detach() - target.detach()).square().mean(dim=(1, 2, 3))
  return (-10 * torch.log10(mse)).cpu().tolist()
