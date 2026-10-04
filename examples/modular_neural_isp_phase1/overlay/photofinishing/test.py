"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

This file contains the testing script for the photofinishing module.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import time
import csv
import numpy as np

from tabulate import tabulate
from typing import List, Optional
from utils.file_utils import read_json_file, write_json_file

import torch
from photofinishing_model import PhotofinishingModule
from utils.img_utils import imwrite, img_to_tensor, tensor_to_img, get_ssim
from baseline_utils import paired_files, load_image_pair, image_psnr_values, summarize_psnr


def print_line(end: Optional[bool]=False, length: Optional[int]=30):
  """Prints a separator line."""
  line = '-' * length
  if end:
    logging.info(f"{line}\n")
  else:
    logging.info(f"\n{line}")


def test_net(model: PhotofinishingModule, te_device: torch.device, in_te_dir: str,
             gt_te_dir: str, data_te_dir: str, post_process_ltm: bool, no_ds: bool,
             result_dir: Optional[str] = None, eval_size: Optional[int] = 512,
             quarter_resolution: bool = False, recipe: Optional[dict] = None,
             diagnostics_profile: Optional[str] = None,
             diagnostics_root_mapping: Optional[dict] = None) -> str:
  """Tests a given trained model."""

  if data_te_dir is None:
    data_te_dir = os.path.join(os.path.dirname(in_te_dir.rstrip("/\\")), 'data')

  if quarter_resolution or no_ds:
    eval_size = None
  protocol = f'P{eval_size}' if eval_size is not None else ('quarter' if quarter_resolution else 'full')
  if eval_size is not None and post_process_ltm:
    raise ValueError('P<size> evaluation uses the validation forward path without LTM post-processing')
  model.eval()
  pairs = paired_files(in_te_dir, gt_te_dir, data_te_dir)
  if result_dir is not None:
    result_dir = os.path.abspath(result_dir)
    images_dir = os.path.join(result_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

  psnr = np.zeros((len(pairs), 1))
  ssim = np.zeros((len(pairs), 1))
  total_time = 0
  rows = []
  roi_rows = []
  diagnostic_errors = []
  profile = None
  roi_profile = None
  diagnostic_successes = 0
  frozen_diagnostics = None
  evaluation_identity = None
  if diagnostics_profile:
    try:
      from tm_research.diagnostics import (load_sample_masks, measure_fixed_regions, resolve_sample_id,
                                          validate_evaluation_profile, diagnostics_snapshot)
      profile = json.loads(Path(diagnostics_profile).read_text(encoding='utf-8'))
      if profile['eval_size'] != eval_size or quarter_resolution or no_ds:
        raise ValueError('Diagnostic profile evaluation transform does not match')
      mapping = diagnostics_root_mapping
      if isinstance(mapping, (str, Path)):
        mapping = json.loads(Path(mapping).read_text(encoding='utf-8'))
      evaluation_identity = validate_evaluation_profile(profile,
        {'input_dir': in_te_dir, 'gt_dir': gt_te_dir, 'metadata_dir': data_te_dir}, eval_size,
        root_mapping=mapping)
      frozen_diagnostics = diagnostics_snapshot(profile)
      if profile.get('roi_profile_ref'):
        roi_profile = json.loads(Path(profile['roi_profile_ref']).read_text(encoding='utf-8'))
    except Exception as exc:
      diagnostic_errors.append(f'profile: {type(exc).__name__}: {exc}')
      profile = None
  for idx, (in_file, gt_file, data_file) in enumerate(pairs):
    print(f'Processing {idx+1}/{len(pairs)}...')
    lsrgb_img, gt_img = load_image_pair(in_file, gt_file, data_file, image_size=eval_size,
                                       quarter=quarter_resolution)
    lsrgb_img_tensor = img_to_tensor(lsrgb_img).unsqueeze(0).to(device=te_device, dtype=torch.float32)
    start = time.time()
    with torch.no_grad():
      out_img_tensor = model(lsrgb_img_tensor, post_process_ltm=post_process_ltm,
                             training_mode=eval_size is not None)['output']
    end = time.time()
    elapsed = end - start
    total_time += elapsed
    out_img = tensor_to_img(out_img_tensor)
    gt_tensor = img_to_tensor(gt_img).unsqueeze(0).to(device=te_device, dtype=torch.float32)
    psnr[idx] = image_psnr_values(out_img_tensor, gt_tensor)[0]
    ssim[idx] = get_ssim(out_img, gt_img)
    sample_id = ''
    input_relpath = Path(in_file).relative_to(in_te_dir).as_posix()
    # Float measurements happen before PNG-8 export, independently of core scores.
    if profile is not None:
      try:
        sample_id = resolve_sample_id({'input_relpath': input_relpath}, profile)
        if roi_profile is not None:
          masks = load_sample_masks(profile, sample_id, gt_img.shape[:2], profile['profile_revision'])
          metric_config = dict(roi_profile['metric_config'], layout='NCHW', sample_id=sample_id,
                               profile_revision=profile['profile_revision'], profile_ref=profile['profile_ref'])
          roi_rows.extend(measure_fixed_regions(out_img_tensor, gt_tensor, masks, metric_config))
        diagnostic_successes += 1
      except Exception as exc:
        diagnostic_errors.append(f'{input_relpath}: {type(exc).__name__}: {exc}')
    if result_dir is not None:
      image_path = imwrite(out_img, os.path.join(images_dir, os.path.splitext(os.path.basename(in_file))[0]),
                           'PNG-8')
      rows.append({'image': os.path.basename(in_file), 'psnr': float(psnr[idx, 0]),
                   'ssim': float(ssim[idx, 0]), 'time_seconds': elapsed, 'output_image': image_path,
                   'sample_id': sample_id, 'input_relpath': input_relpath})
  mean_time = total_time / len(pairs)
  summary = summarize_psnr(psnr[:, 0].tolist())
  if result_dir is not None:
    csv_path = os.path.join(result_dir, 'per_image.csv')
    with open(csv_path, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=['image', 'psnr', 'ssim', 'time_seconds', 'output_image',
                                            'sample_id', 'input_relpath'])
      writer.writeheader()
      writer.writerows(rows)
    diagnostic_artifacts = {}
    if diagnostics_profile:
      diagnostic_artifacts = {'diagnostics_profile_ref': str(Path(diagnostics_profile).resolve()),
                              'diagnostics_errors': diagnostic_errors,
                              'diagnostics_evaluation_identity': evaluation_identity,
                              'profile_revision': profile.get('profile_revision') if profile else None,
                              'diagnostics_status': ('partial' if diagnostic_errors and diagnostic_successes
                                else 'failed' if diagnostic_errors else 'complete')}
      if roi_profile is not None:
        diagnostic_artifacts['roi_profile_ref'] = profile['roi_profile_ref']
      if frozen_diagnostics is not None:
        try:
          snapshot_path = os.path.join(result_dir, 'diagnostics_snapshot.json')
          with open(snapshot_path, 'w') as f:
            json.dump(frozen_diagnostics, f, indent=2, sort_keys=True, allow_nan=False)
          diagnostic_artifacts['diagnostics_snapshot_ref'] = snapshot_path
        except Exception as exc:
          diagnostic_errors.append(f'diagnostics_snapshot.json: {type(exc).__name__}: {exc}')
          diagnostic_artifacts['diagnostics_status'] = 'failed'
      if roi_rows:
        try:
          roi_path = os.path.join(result_dir, 'roi_metrics.csv')
          with open(roi_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'roi_id', 'metric_name', 'value', 'n_pixels',
              'region_fraction', 'effective_mask_id', 'valid_reason', 'profile_revision', 'profile_ref',
              'metric_definition'])
            writer.writeheader()
            writer.writerows(roi_rows)
          diagnostic_artifacts['roi_metrics_csv'] = roi_path
        except Exception as exc:
          diagnostic_errors.append(f'roi_metrics.csv: {type(exc).__name__}: {exc}')
          diagnostic_artifacts['diagnostics_status'] = 'failed'
    write_json_file({**summary, **diagnostic_artifacts, 'mean_ssim': float(ssim.mean()),
                     'protocol': protocol, 'eval_size': eval_size, 'recipe': recipe,
                     'mean_time_seconds': mean_time,
                     'per_image_csv': csv_path, 'images_dir': images_dir},
                    os.path.join(result_dir, 'metrics.json'))
  if not summary['finite']:
    raise ValueError('Evaluation produced non-finite per-image PSNR')
  return (f"Protocol = {protocol} - PSNR = {summary['mean_psnr']} - SSIM = {ssim.mean()} "
          f"- Count = {summary['num_images']} - Finite = {summary['finite']} - Time = {mean_time}\n")

def get_args():
  parser = argparse.ArgumentParser(description='Test the photofinishing network.')
  parser.add_argument('--model-path', dest='model_path', required=True, help='Path to the trained model.')
  parser.add_argument(
    '--in-testing-dir', dest='in_te_dir', type=str, required=True, help='Testing input image directory.')
  parser.add_argument(
    '--gt-testing-dir', dest='gt_te_dir', type=str, required=True,
    help='Testing ground-truth image directory.')
  parser.add_argument(
    '--data-testing-dir', dest='data_te_dir', default=None, type=str, help='Testing input data directory.')
  parser.add_argument('--post-process-ltm', dest='post_process_ltm', action='store_true',
                      help='Enable multi-scale and refinement of the LTM coeffs to mitigate potential halo artifacts '
                           '(refer to Sec. B.1 of the supp materials).')
  resolution = parser.add_mutually_exclusive_group()
  resolution.add_argument('--eval-size', type=int, default=None,
                          help='Square evaluation size (default 512; standard P512 protocol).')
  resolution.add_argument('--quarter-resolution', action='store_true',
                          help='Use the separate legacy quarter-resolution inference protocol.')
  resolution.add_argument('--no-ds', dest='no_ds', action='store_true',
                          help='Use the separate full-resolution inference protocol.')
  parser.add_argument('--config-dir', dest='config_dir', default='config',
                      help='Directory containing config JSON files.')
  parser.add_argument('--result-dir', dest='result_dir', default='results',
                      help='Directory to save the results report (.txt).')
  parser.add_argument('--diagnostics-profile', default=None,
                      help='Optional frozen TRAIN/DEV profile; float ROI diagnostics never affect PSNR.')
  parser.add_argument('--diagnostics-root-mapping', default=None,
                      help='Optional JSON source-root to relocated-root mapping; source files must remain readable for validation.')
  args = parser.parse_args()
  if not args.no_ds and not args.quarter_resolution:
    args.eval_size = 512 if args.eval_size is None else args.eval_size
    if args.eval_size < 256:
      parser.error('--eval-size must be at least 256')
  return args


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  os.makedirs(args.result_dir, exist_ok=True)
  print(tabulate([(key, value) for key, value in vars(args).items()], headers=['Argument', 'Value'], tablefmt='grid'))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  assert os.path.exists(args.model_path), f"Model file not found at '{args.model_path}'"

  logging.info(f'Using device {device}')
  logging.info(f'Testing of photofinishing module -- model name: {os.path.basename(args.model_path)} ...')
  config_base = os.path.splitext(os.path.basename(args.model_path))[0]
  config = read_json_file(os.path.join(args.config_dir, f'{config_base}.json'))

  net = PhotofinishingModule(device=device, use_3d_lut=config['use_3d_lut'])
  net.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
  net.eval()
  logging.info(f'Model loaded from {args.model_path}')
  net.print_num_of_params()

  results = test_net(model=net, te_device=device, in_te_dir=args.in_te_dir, gt_te_dir=args.gt_te_dir,
                     data_te_dir=args.data_te_dir, post_process_ltm=args.post_process_ltm, no_ds=args.no_ds,
                     result_dir=args.result_dir, eval_size=args.eval_size,
                     quarter_resolution=args.quarter_resolution, recipe=config.get('recipe'),
                     diagnostics_profile=args.diagnostics_profile,
                     diagnostics_root_mapping=args.diagnostics_root_mapping)
  print(results)
  with open(os.path.join(args.result_dir, os.path.splitext(os.path.basename(args.model_path))[0] + '.txt'), 'w') as f:
    f.write(results)
