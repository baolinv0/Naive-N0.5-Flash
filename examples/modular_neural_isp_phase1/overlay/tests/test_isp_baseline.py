"""Focused checks for pairing, HDF5 reuse, and image-wise validation metrics."""

from collections import OrderedDict
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'photofinishing'))
from baseline_utils import image_psnr_values, paired_files
from dataset import Data
from train import validate


def test_pairing_by_scene_stem_and_missing_file(tmp_path):
  raw, gt, meta = (tmp_path / name for name in ('raw', 'gt', 'data'))
  for folder in (raw, gt, meta):
    folder.mkdir()
  for stem in ('img_0002', 'img_0001'):
    (raw / f'{stem}_denoised.png').touch()
    (gt / f'{stem}.jpg').touch()
    (meta / f'{stem}.json').touch()
  pairs = paired_files(raw, gt, meta)
  assert [Path(p[0]).stem for p in pairs] == ['img_0001_denoised', 'img_0002_denoised']
  (meta / 'img_0002.json').unlink()
  with pytest.raises(ValueError, match='do not match'):
    paired_files(raw, gt, meta)


def test_hdf5_batch_can_be_read_repeatedly(tmp_path):
  path = tmp_path / 'batch.h5'
  with h5py.File(path, 'w') as file:
    file['in_images'] = np.ones((1, 4, 4, 3), dtype=np.float32)
    file['gt_images'] = np.zeros((1, 4, 4, 3), dtype=np.float32)
  data = Data.__new__(Data)
  data._h5_file_paths = [str(path)]
  data._h5_cache = OrderedDict()
  data._max_open_files = 64
  data._shuffle = False
  data._geo_aug = False
  assert torch.equal(data[0]['in_images'], data[0]['in_images'])
  assert data._h5_cache[str(path)].id.valid
  for file in data._h5_cache.values():
    file.close()


def test_image_psnr_averages_images_not_mse_or_batches():
  target = torch.zeros(3, 3, 2, 2)
  output = target.clone()
  output[0] += 0.1
  output[1:] += 0.2
  values = image_psnr_values(output, target)
  assert np.mean(values) == pytest.approx((20 + 2 * 13.9794001) / 3, abs=1e-4)


def test_validation_divides_by_number_of_batches():
  class Model:
    def eval(self):
      pass

    def train(self):
      pass

    def __call__(self, images, training_mode=False):
      return {'output': images, 'gamma_factor': 1, 'processed_lsrgb': images,
              'lsrgb_3d_lut': images, 'processed_cbcr': images[:, :2],
              'cbcr_lut': images[:, :2], 'y_gain': images[:, :1],
              'ltm_y': images[:, :1], 'gtm_y': images[:, :1], 'ltm_params': images[:, :1]}

    def de_gamma(self, images, factor):
      return images

    def rgb_to_ycbcr(self, images):
      return images

  keys = ['total', 'l1', 'ssim', 'delta-e', 'psnr', 'cbcr', 'lut-smoothness',
          'luma-energy-consistency', 'ltm-smoothness', 'tm', 'vgg']
  batches = [
    {'in_images': torch.full((1, 1, 3, 2, 2), 0.1), 'gt_images': torch.zeros(1, 1, 3, 2, 2)},
    {'in_images': torch.full((1, 1, 3, 2, 2), 0.2), 'gt_images': torch.zeros(1, 1, 3, 2, 2)},
  ]
  result = validate(Model(), batches, torch.device('cpu'),
                    lambda **kwargs: (None, dict.fromkeys(keys, 2.0)), None, 0)
  assert result['l1'] == 2.0
  assert result['mean_psnr'] == pytest.approx((20 + 13.9794001) / 2, abs=1e-4)
