"""Focused checks for pairing, HDF5 reuse, and image-wise validation metrics."""

from collections import OrderedDict
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'photofinishing'))
from baseline_utils import image_psnr_values, paired_files, summarize_psnr, best_checkpoint_index
from dataset import Data
from recipe_utils import PixelLoss
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


def test_psnr_float64_epsilon_and_nonfinite_summary():
  target = torch.zeros(2, 3, 2, 2, dtype=torch.float64)
  output = target.clone()
  output[1] = 0.123456789123
  values = image_psnr_values(output, target)
  assert values[0] == 120.0
  assert values[1] == pytest.approx(-20 * np.log10(0.123456789123), abs=1e-12)
  assert summarize_psnr(values)['finite'] is True
  assert summarize_psnr([float('nan')]) == {
    'mean_per_image_psnr': None, 'mean_psnr': None, 'num_images': 1, 'finite': False}


def test_checkpoint_ranking_counterexample():
  target = torch.zeros(2, 3, 2, 2, dtype=torch.float64)
  candidate_a = target.clone()
  candidate_a[0] = 0.01
  candidate_a[1] = 0.1
  candidate_b = torch.full_like(target, np.sqrt(0.002))
  scores = [summarize_psnr(image_psnr_values(candidate, target))['mean_psnr']
            for candidate in (candidate_a, candidate_b)]
  assert scores == pytest.approx([30.0, 26.989700043360187], abs=1e-12)
  pooled_scores = [-10 * torch.log10(candidate.square().mean()).item()
                   for candidate in (candidate_a, candidate_b)]
  assert pooled_scores[0] < pooled_scores[1]
  assert best_checkpoint_index(scores) == 0


class EchoModel:
  def eval(self):
    pass

  def train(self):
    pass

  def __call__(self, images, **kwargs):
    assert kwargs.get('training_mode') is True
    return {'output': images}


def test_validation_score_does_not_depend_on_batch_partition():
  images = torch.tensor([0.01, 0.1, 0.2], dtype=torch.float32).view(3, 1, 1, 1).expand(-1, 3, 2, 2)
  results = []
  for sizes in ([3], [2, 1], [1, 1, 1]):
    batches = [{'in_images': part.unsqueeze(0), 'gt_images': torch.zeros_like(part).unsqueeze(0)}
               for part in images.split(sizes)]
    results.append(validate(EchoModel(), batches, torch.device('cpu'), PixelLoss('mse'), None, 0))
  assert [result['num_images'] for result in results] == [3, 3, 3]
  assert all(result['finite'] for result in results)
  assert all(result['mean_per_image_psnr'] == results[0]['mean_psnr'] for result in results)


def test_incomplete_cache_is_rebuilt_and_completed_cache_is_reused(tmp_path, monkeypatch):
  calls = []

  def create(data):
    calls.append(data._temp_dir)
    data._write_hdf5(0, [np.zeros((4, 4, 3), np.float32)], [np.zeros((4, 4, 3), np.float32)])
    if len(calls) == 1:
      raise RuntimeError('interrupted preprocessing')

  monkeypatch.setattr(Data, '_create_hdf5_files', create)
  kwargs = dict(in_img_dir=str(tmp_path / 'raw'), gt_img_dir=str(tmp_path / 'gt'),
                image_size=4, batch_size=1)
  with pytest.raises(RuntimeError, match='interrupted'):
    Data(**kwargs)
  cache = tmp_path / 'ps_temp_h5_gt_bs_1_sz_4'
  assert not (cache / 'COMPLETE').exists()
  assert len(Data(**kwargs)) == 1
  assert (cache / 'COMPLETE').exists()
  assert len(Data(**kwargs)) == 1
  assert len(calls) == 2


@pytest.mark.parametrize('eval_size', [256, 512])
def test_independent_evaluation_matches_cached_validation(tmp_path, eval_size):
  from PIL import Image
  import torch.utils.data

  raw, gt, meta = (tmp_path / name for name in ('raw', 'gt', 'data'))
  for folder in (raw, gt, meta):
    folder.mkdir()
  for index in range(3):
    image = np.random.default_rng(index).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    Image.fromarray(image).save(raw / f'{index}.png')
    Image.fromarray(np.flip(image, axis=1)).save(gt / f'{index}.jpg')
    (meta / f'{index}.json').write_text(json.dumps({'cam_illum': [0.4, 1, 0.8],
                                                 'ccm': [[1.8, 0, 0], [0, 1, 0], [0, 0, 1.3]]}))
  data = Data(str(raw), str(gt), str(meta), image_size=eval_size, batch_size=2,
              geometric_aug=False, shuffle=False)
  loader = torch.utils.data.DataLoader(data, batch_size=1)
  validation = validate(EchoModel(), loader, torch.device('cpu'), PixelLoss('mse'), None, 0)
  spec = importlib.util.spec_from_file_location('isp_independent_test',
    Path(__file__).resolve().parents[1] / 'photofinishing' / 'test.py')
  evaluation = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(evaluation)
  result_dir = tmp_path / 'evaluation'
  evaluation.test_net(EchoModel(), torch.device('cpu'), str(raw), str(gt), str(meta),
                      False, False, result_dir=str(result_dir), eval_size=eval_size)
  report = json.loads((result_dir / 'metrics.json').read_text())
  assert report['protocol'] == f'P{eval_size}'
  assert report['num_images'] == validation['num_images'] == 3
  assert report['finite'] is True
  assert report['mean_per_image_psnr'] == validation['mean_per_image_psnr']
  for handle in data._h5_cache.values():
    handle.close()


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
