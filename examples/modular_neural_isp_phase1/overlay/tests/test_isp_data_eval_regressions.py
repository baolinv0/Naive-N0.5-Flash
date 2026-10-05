"""Regressions for source-bound preprocessing and completed inference timing."""

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'photofinishing'))
import dataset as dataset_module
from baseline_utils import load_image_pair
from dataset import Data


def make_sources(root, input_value=64, gt_value=128, red_gain=1.0):
  raw, gt, meta = (root / name for name in ('raw', 'gt', 'data'))
  for folder in (raw, gt, meta):
    folder.mkdir(parents=True, exist_ok=True)
  Image.fromarray(np.full((8, 8, 3), input_value, np.uint8)).save(raw / 'scene.png')
  Image.fromarray(np.full((8, 8, 3), gt_value, np.uint8)).save(gt / 'scene.jpg')
  (meta / 'scene.json').write_text(json.dumps({'cam_illum': [1, 1, 1],
    'ccm': [[red_gain, 0, 0], [0, 1, 0], [0, 0, 1]]}))
  return raw, gt, meta


def cached_pair(raw, gt, meta, **kwargs):
  data = Data(str(raw), str(gt), str(meta), image_size=4, batch_size=1,
              geometric_aug=False, **kwargs)
  pair = data[0]
  for handle in data._h5_cache.values():
    handle.close()
  return data, pair


def assert_matches_sources(pair, raw, gt, meta):
  expected_input, expected_gt = load_image_pair(raw / 'scene.png', gt / 'scene.jpg',
                                               meta / 'scene.json', image_size=4)
  np.testing.assert_allclose(pair['in_images'][0].permute(1, 2, 0).numpy(), expected_input)
  np.testing.assert_allclose(pair['gt_images'][0].permute(1, 2, 0).numpy(), expected_gt)


@pytest.mark.parametrize('source', ['raw', 'metadata'])
def test_same_gt_with_different_sources_cannot_reuse_stale_images(tmp_path, source):
  raw, gt, meta = make_sources(tmp_path / 'first')
  other_raw, _, other_meta = make_sources(tmp_path / 'second', input_value=160, red_gain=1.8)
  first_data, first = cached_pair(raw, gt, meta)
  selected_raw = other_raw if source == 'raw' else raw
  selected_meta = other_meta if source == 'metadata' else meta
  second_data, second = cached_pair(selected_raw, gt, selected_meta)
  assert Path(first_data._temp_dir) != Path(second_data._temp_dir)
  assert_matches_sources(second, selected_raw, gt, selected_meta)
  assert not torch.equal(first['in_images'], second['in_images'])
  # Both source identities retain independently reusable cache contents.
  _, first_again = cached_pair(raw, gt, meta)
  assert torch.equal(first['in_images'], first_again['in_images'])


@pytest.mark.parametrize('source', ['raw', 'gt', 'metadata'])
def test_source_changes_in_place_invalidate_completed_cache(tmp_path, source):
  raw, gt, meta = make_sources(tmp_path)
  _, first = cached_pair(raw, gt, meta)
  if source == 'metadata':
    (meta / 'scene.json').write_text(json.dumps({'cam_illum': [1, 1, 1],
      'ccm': [[1.8, 0, 0], [0, 1, 0], [0, 0, 1]]}))
  else:
    folder, extension = (raw, 'png') if source == 'raw' else (gt, 'jpg')
    Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(folder / f'scene.{extension}')
  _, second = cached_pair(raw, gt, meta)
  assert_matches_sources(second, raw, gt, meta)
  key = 'gt_images' if source == 'gt' else 'in_images'
  assert not torch.equal(first[key], second[key])


def test_legacy_complete_cache_is_preserved_and_ignored(tmp_path):
  raw, gt, meta = make_sources(tmp_path)
  legacy = tmp_path / 'ps_temp_h5_gt_bs_1_sz_4'
  legacy.mkdir()
  (legacy / 'COMPLETE').write_text('complete\n')
  sentinel = legacy / 'user-notes.txt'
  sentinel.write_text('retain legacy artifacts')
  with h5py.File(legacy / 'batch_00000.h5', 'w') as file:
    file['in_images'] = np.zeros((1, 4, 4, 3), np.float32)
    file['gt_images'] = np.zeros((1, 4, 4, 3), np.float32)
  data, pair = cached_pair(raw, gt, meta)
  assert_matches_sources(pair, raw, gt, meta)
  assert Path(data._temp_dir) != legacy
  assert sentinel.read_text() == 'retain legacy artifacts'
  assert (legacy / 'batch_00000.h5').exists()


@pytest.mark.parametrize('invalid_manifest', ['missing', 'malformed', 'mismatched'])
def test_complete_marker_alone_cannot_trust_cache_and_preserves_unrelated_files(tmp_path, invalid_manifest):
  raw, gt, meta = make_sources(tmp_path)
  data, _ = cached_pair(raw, gt, meta)
  cache = Path(data._temp_dir)
  manifest = cache / 'manifest.json'
  if invalid_manifest == 'missing':
    manifest.unlink(missing_ok=True)
  else:
    manifest.write_text('{' if invalid_manifest == 'malformed' else '{}')
  sentinel = cache / 'user-notes.txt'
  sentinel.write_text('retain unrelated file')
  with h5py.File(cache / 'batch_00000.h5', 'r+') as file:
    file['in_images'][...] = 0
  _, pair = cached_pair(raw, gt, meta)
  assert_matches_sources(pair, raw, gt, meta)
  assert sentinel.read_text() == 'retain unrelated file'


def test_canonical_sources_reuse_and_explicit_overwrite_rebuilds(tmp_path, monkeypatch):
  raw, gt, meta = make_sources(tmp_path)
  data, first = cached_pair(raw, gt, meta)
  original_load = dataset_module.load_image_pair

  def cannot_preprocess(*args, **kwargs):
    raise RuntimeError('preprocessing called')

  monkeypatch.setattr(dataset_module, 'load_image_pair', cannot_preprocess)
  reused, second = cached_pair(raw / '..' / 'raw', gt / '..' / 'gt', meta / '..' / 'data')
  assert Path(reused._temp_dir) == Path(data._temp_dir)
  assert torch.equal(first['in_images'], second['in_images'])
  with pytest.raises(RuntimeError, match='preprocessing called'):
    cached_pair(raw, gt, meta, overwrite_temp_folder=True)
  monkeypatch.setattr(dataset_module, 'load_image_pair', original_load)
  _, rebuilt = cached_pair(raw, gt, meta)
  assert_matches_sources(rebuilt, raw, gt, meta)


def test_deleting_cache_preserves_sources_and_unrelated_user_artifacts(tmp_path):
  raw, gt, meta = make_sources(tmp_path)
  data, _ = cached_pair(raw, gt, meta)
  cache = Path(data._temp_dir)
  sentinel = cache / 'user.h5'
  sentinel.write_text('retain unrelated file')
  data.delete_cache()
  assert sentinel.read_text() == 'retain unrelated file'
  assert not (cache / 'batch_00000.h5').exists()
  assert not (cache / 'COMPLETE').exists()
  assert (raw / 'scene.png').exists()
  assert (gt / 'scene.jpg').exists()
  assert (meta / 'scene.json').exists()


@pytest.mark.parametrize('device_name', ['cpu', 'cuda:2'])
def test_evaluation_elapsed_time_contains_completed_forward(tmp_path, monkeypatch, device_name):
  raw, gt, meta = make_sources(tmp_path)
  spec = importlib.util.spec_from_file_location('isp_regression_evaluation',
    Path(__file__).resolve().parents[1] / 'photofinishing' / 'test.py')
  evaluation = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(evaluation)
  events, clock, pending = [], [0.0], [0.0]
  device = torch.device(device_name)
  original_to = torch.Tensor.to

  def transfer(tensor, *args, **kwargs):
    requested_device = kwargs.get('device')
    if requested_device is not None and torch.device(requested_device).type == 'cuda':
      # Keep tensors on CPU while exercising the real evaluation control flow.
      pending[0] += 0.5
      return tensor
    return original_to(tensor, *args, **kwargs)

  def synchronize(requested_device):
    assert requested_device == device
    assert device.type == 'cuda', 'CPU evaluation must not synchronize CUDA'
    events.append('synchronize')
    clock[0] += pending[0]
    pending[0] = 0.0

  def timestamp():
    events.append('timestamp')
    return clock[0]

  class TimedEcho:
    def eval(self):
      pass

    def __call__(self, images, **kwargs):
      events.append('forward')
      if device.type == 'cuda':
        pending[0] += 0.25
      else:
        clock[0] += 0.25
      return {'output': images}

  monkeypatch.setattr(torch.Tensor, 'to', transfer)
  monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
  monkeypatch.setattr(evaluation.time, 'time', timestamp)
  result_dir = tmp_path / 'results'
  evaluation.test_net(TimedEcho(), device, str(raw), str(gt), str(meta), False, False,
                      result_dir=str(result_dir), eval_size=8)
  report = json.loads((result_dir / 'metrics.json').read_text())
  assert report['mean_time_seconds'] == pytest.approx(0.25)
  assert events == (['synchronize', 'timestamp', 'forward', 'synchronize', 'timestamp']
                    if device.type == 'cuda' else ['timestamp', 'forward', 'timestamp'])
