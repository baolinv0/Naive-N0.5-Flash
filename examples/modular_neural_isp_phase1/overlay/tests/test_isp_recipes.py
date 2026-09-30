"""Focused recipe and checkpoint selection checks without the expensive model."""

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'photofinishing'))
import loss_utils
import train
from recipe_utils import PixelLoss, effective_recipe, loss_for_batch, make_loss, make_optimizer


@pytest.mark.parametrize(('family', 'expected_loss', 'expected_gradient'),
                         [('mse', 0.0625, 0.5), ('l1', 0.25, 1.0)])
def test_pixel_recipes_skip_original_loss_and_color_targets(monkeypatch, family, expected_loss, expected_gradient):
  def forbidden_original(**kwargs):
    raise AssertionError('MSE/L1 must not instantiate original/VGG losses')

  monkeypatch.setattr(loss_utils, 'PhotofinishingLoss', forbidden_original)
  pixel = torch.tensor([[[[0.25]]]], requires_grad=True)
  compute_loss = make_loss(family, vgg_weight=0.01)
  loss, details = loss_for_batch(object(), {'output': pixel}, torch.zeros_like(pixel), compute_loss)
  loss.backward()
  assert loss.item() == pytest.approx(expected_loss)
  assert pixel.grad.item() == pytest.approx(expected_gradient)
  assert set(details) == {'total', family, 'psnr'}


@pytest.mark.parametrize(('name', 'expected', 'optimizer_class'),
                         [('adam', 1.9, torch.optim.Adam), ('adamw', 1.96, torch.optim.AdamW)])
def test_optimizer_uses_requested_decay_semantics(name, expected, optimizer_class):
  model = torch.nn.Linear(1, 1, bias=False)
  model.weight.data.fill_(2.0)
  optimizer = make_optimizer(model, name, learning_rate=0.1, weight_decay=0.2)
  (model.weight * 0).sum().backward()
  optimizer.step()
  assert type(optimizer) is optimizer_class
  assert model.weight.item() == pytest.approx(expected)
  assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)


def test_default_recipe_retains_original_settings(monkeypatch):
  monkeypatch.setattr(sys, 'argv', ['train.py', '--in-training-dir', 'raw', '--gt-training-dir', 'gt',
                                   '--in-validation-dir', 'val_raw', '--gt-validation-dir', 'val_gt'])
  args = train.get_args()
  recipe = effective_recipe(args)
  assert (args.loss_family, args.optimizer, args.lr, args.l2_r) == ('original', 'adam', 1e-4, 1e-7)
  assert recipe['loss_weights'] == {
    'l1': 2.5, 'ssim': 0.5, 'delta_e': 0.02, 'perceptual': 0.01, 'cbcr': 1.0,
    'lut_smoothness': 0.06, 'tm': 0.5, 'ltm_smoothness': 0.6, 'luma_energy_consistency': 0.2}
  assert recipe['scheduler'] == {'name': 'CosineAnnealingLR', 'T_max': 600, 'eta_min': 1e-6}
  assert recipe['eval_size'] == 512
  args.loss_family = 'mse'
  assert effective_recipe(args)['loss_weights'] == {'mse': 1.0}


def test_training_saves_checkpoint_ranked_by_mean_per_image_psnr(tmp_path, monkeypatch):
  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images, training_mode):
      return {'output': images + self.weight}

  scores = iter([(30.0, 22.9670862188), (26.9897000434, 26.9897000434)])
  model = Model()

  def validation(**kwargs):
    score, old_pooled_score = next(scores)
    model.weight.data.add_(1)
    return {'mean_psnr': score, 'mean_per_image_psnr': score, 'psnr': old_pooled_score,
            'num_images': 2, 'finite': True}

  monkeypatch.setattr(train, 'validate', validation)
  for folder in ('models', 'checkpoints', 'logs', 'config'):
    (tmp_path / folder).mkdir()
  batch = {'in_images': torch.zeros(1, 1, 3, 2, 2), 'gt_images': torch.zeros(1, 1, 3, 2, 2)}
  train.training(model, 2, 0.0, 0.0, torch.device('cpu'), [batch], [], [batch], [0],
                  1, 'ranking', 1, PixelLoss('mse'), None,
                  {'checkpoint_model_name': [], 'val_psnr': []}, str(tmp_path), eval_size=256)
  report = json.loads((tmp_path / 'metrics.json').read_text())
  checkpoint = torch.load(report['best_checkpoint'], weights_only=True)
  assert checkpoint['weight'].item() == 1.0
  assert report['mean_per_image_psnr'] == 30.0
  assert report['num_images'] == 2
  assert report['finite'] is True
  assert report['protocol'] == 'P256'
