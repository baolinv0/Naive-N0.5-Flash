"""Recipe choices for a fixed photofinishing model."""

import torch
from torch.nn import functional as F

from baseline_utils import image_psnr_values, summarize_psnr


class PixelLoss(torch.nn.Module):
  """MSE/L1 recipes operate solely on the final sRGB output."""

  def __init__(self, family):
    super().__init__()
    if family not in ('mse', 'l1'):
      raise ValueError(f'Unknown pixel loss: {family}')
    self.family = family

  def forward(self, out_img, gt_img):
    loss = F.mse_loss(out_img, gt_img) if self.family == 'mse' else F.l1_loss(out_img, gt_img)
    return loss, {'total': loss.item(), self.family: loss.item()}


def make_loss(family, **original_kwargs):
  if family == 'original':
    from loss_utils import PhotofinishingLoss
    return PhotofinishingLoss(**original_kwargs)
  return PixelLoss(family)


def make_optimizer(model, name, learning_rate, weight_decay):
  optimizer_class = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[name]
  return optimizer_class(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                         weight_decay=weight_decay)


def loss_for_batch(model, outputs, target, compute_loss):
  if isinstance(compute_loss, PixelLoss):
    loss, details = compute_loss(out_img=outputs['output'], gt_img=target)
  else:
    target_linear = model.de_gamma(target, outputs['gamma_factor'])
    target_ycbcr = model.rgb_to_ycbcr(target_linear)
    loss, details = compute_loss(
      out_img=outputs['output'], gt_img=target, lsrgb_out_img=outputs['processed_lsrgb'],
      lsrgb_gt_img=target_linear, rgb_lut_out_img=outputs['lsrgb_3d_lut'],
      cbcr_out_img=outputs['processed_cbcr'], cbcr_lut=outputs['cbcr_lut'],
      pre_tm_y=outputs['y_gain'], ltm_y=outputs['ltm_y'], gtm_y=outputs['gtm_y'],
      ltm_map=outputs['ltm_params'], cbcr_gt_img=target_ycbcr[:, 1:, ...],
      y_gt_img=target_ycbcr[:, :1, ...])
  details['psnr'] = summarize_psnr(image_psnr_values(outputs['output'], target))['mean_psnr']
  return loss, details


def effective_recipe(args):
  weights = ({key.removesuffix('_loss_weight'): value for key, value in vars(args).items()
              if key.endswith('_loss_weight')} if args.loss_family == 'original'
             else {args.loss_family: 1.0})
  return {'loss_family': args.loss_family, 'optimizer': args.optimizer,
          'learning_rate': args.lr, 'weight_decay': args.l2_r, 'betas': [0.9, 0.999],
          'loss_weights': weights,
          'scheduler': {'name': 'CosineAnnealingLR', 'T_max': args.epochs, 'eta_min': args.lr / 100},
          'epochs': args.epochs, 'batch_size': args.batch_size, 'in_size': args.in_sz,
          'eval_size': args.eval_size, 'seed': args.seed, 'extract_patches': args.extract_patches,
          'validation_frequency': args.val_frq, 'initial_checkpoint': args.load}
