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

This file defines the data loading pipeline for training the photofinishing module.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from os.path import join, exists, dirname, basename
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Optional, Dict, List
import collections
import h5py
import hashlib
import json
import re

from utils.constants import PHOTOFINISHING_TRAINING_INPUT_SIZE
from utils.img_utils import augment_img, img_to_tensor, extract_non_overlapping_patches
from baseline_utils import paired_files, load_image_pair

class Data(Dataset):
  def __init__(self, in_img_dir: str, gt_img_dir: str, data_dir: Optional[str]=None,
               image_size: Optional[int]=PHOTOFINISHING_TRAINING_INPUT_SIZE, extract_patches: Optional[bool]=False,
               temp_folder: Optional[str]='ps_temp_h5', overwrite_temp_folder: Optional[bool]=False,
               geometric_aug: Optional[bool]=True, batch_size: Optional[int]=8, shuffle: Optional[bool]=False):
    self._in_img_dir = in_img_dir
    self._gt_img_dir = gt_img_dir
    self._data_dir = data_dir
    self._extract_patches = extract_patches
    if self._data_dir is None:
      self._data_dir = join(dirname(self._in_img_dir.rstrip("/\\")), 'data')
    self._in_img_dir = os.path.realpath(self._in_img_dir)
    self._gt_img_dir = os.path.realpath(self._gt_img_dir)
    self._data_dir = os.path.realpath(self._data_dir)
    self._image_size = image_size
    self._geo_aug = geometric_aug
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._max_open_files = 64

    assert self._image_size > 0, 'Invalid image size.'

    if self._extract_patches:
      postfix = '_patches'
    else:
      postfix = ''
    self._pairs = paired_files(self._in_img_dir, self._gt_img_dir, self._data_dir)
    cache_identity = {
      'version': 1,
      'roots': {'input_dir': self._in_img_dir, 'gt_dir': self._gt_img_dir,
                'metadata_dir': self._data_dir},
      'preprocessing': {'image_size': self._image_size, 'batch_size': self._batch_size,
                        'extract_patches': bool(self._extract_patches), 'dtype': 'float32',
                        'color_conversion': 'cam_illum_ccm_clip_v1', 'resize': 'linear_square_v1',
                        'patches': {'num_patches': 0, 'allow_overlap': True,
                                    'add_resized_patch': True} if self._extract_patches else None}}
    cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:16]
    self._temp_dir = os.path.abspath(join(dirname(self._gt_img_dir),
      f'{temp_folder}_{basename(self._gt_img_dir)}_bs_{batch_size}_sz_{self._image_size}{postfix}_v1_{cache_key}'))
    self._check_cache_path()
    self._source_manifest = dict(cache_identity,
      sources=[[self._file_identity(path) for path in pair] for pair in self._pairs])

    completion_marker = join(self._temp_dir, 'COMPLETE')
    re_create = overwrite_temp_folder or not self._cache_matches()
    if re_create:
      if exists(self._temp_dir):
        logging.info('Rebuilding overwritten, incomplete, or mismatched preprocessing cache.')
        self._clear_cache_files()
      os.makedirs(self._temp_dir, exist_ok=True)
      logging.info(f'Preprocessing images with batch_size={batch_size}...')
      self._create_hdf5_files()
      with open(join(self._temp_dir, 'manifest.json'), 'w') as manifest:
        json.dump({'identity': self._source_manifest, 'batches': self._batch_inventory()},
                  manifest, sort_keys=True, indent=2)
      with open(completion_marker, 'w') as marker:
        marker.write('complete\n')
    else:
      logging.info(f'Found complete pre-extracted batches in {self._temp_dir}; skipping reprocessing.')

    self._h5_file_paths = self._batch_paths()
    self._h5_cache: 'collections.OrderedDict[str, h5py.File]' = collections.OrderedDict()

  @staticmethod
  def _file_identity(path):
    """Track ordinary source changes without rereading large image contents."""
    stat = os.stat(path)
    return {'path': os.path.realpath(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

  def _check_cache_path(self):
    """A cache must never resolve to a source directory or its ancestor."""
    if os.path.islink(self._temp_dir):
      raise ValueError('Preprocessing cache directory must not be a symlink')
    cache = os.path.realpath(self._temp_dir)
    for source in (self._in_img_dir, self._gt_img_dir, self._data_dir):
      if os.path.commonpath((cache, source)) == cache:
        raise ValueError('Preprocessing cache overlaps a source directory')

  def _batch_paths(self):
    return sorted(join(self._temp_dir, name) for name in os.listdir(self._temp_dir)
                  if re.fullmatch(r'batch_\d+\.h5', name))

  def _batch_inventory(self):
    return [{'name': basename(path), 'size': os.stat(path).st_size,
             'mtime_ns': os.stat(path).st_mtime_ns} for path in self._batch_paths()]

  def _cache_matches(self):
    marker = join(self._temp_dir, 'COMPLETE')
    manifest_path = join(self._temp_dir, 'manifest.json')
    try:
      if not os.path.isfile(marker) or os.path.islink(marker) or os.path.islink(manifest_path):
        return False
      with open(manifest_path) as manifest_file:
        manifest = json.load(manifest_file)
      batches = self._batch_inventory()
      return (isinstance(manifest, dict) and manifest.get('identity') == self._source_manifest
              and bool(batches) and manifest.get('batches') == batches
              and all(os.path.isfile(path) and not os.path.islink(path) for path in self._batch_paths()))
    except (OSError, ValueError):
      return False

  def _clear_cache_files(self):
    """Remove only filenames reserved by preprocessing, preserving unrelated content."""
    self._check_cache_path()
    for path in self._batch_paths() + [join(self._temp_dir, 'COMPLETE'),
                                       join(self._temp_dir, 'manifest.json')]:
      if os.path.lexists(path):
        os.unlink(path)

  def delete_cache(self):
    """Close readers and delete generated cache artifacts without recursive removal."""
    for handle in self._h5_cache.values():
      handle.close()
    self._h5_cache.clear()
    if exists(self._temp_dir):
      self._clear_cache_files()
      if not os.listdir(self._temp_dir):
        os.rmdir(self._temp_dir)

  def __len__(self):
    return len(self._h5_file_paths)

  def _open_h5_file(self, h5_path: str) -> h5py.File:
    """Opens an HDF5 file with caching to avoid too many open files."""
    if h5_path in self._h5_cache:
      self._h5_cache.move_to_end(h5_path)
      cached = self._h5_cache[h5_path]
      if cached.id.valid:
        return cached
      del self._h5_cache[h5_path]

    if len(self._h5_cache) >= self._max_open_files:
      old_path, old_file = self._h5_cache.popitem(last=False)
      old_file.close()

    f = h5py.File(h5_path, 'r')
    self._h5_cache[h5_path] = f
    return f

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Returns a full batch (of size batch_size) from one HDF5 file."""
    h5_path = self._h5_file_paths[idx]
    f = self._open_h5_file(h5_path)
    in_images = f['in_images'][()]
    gt_images = f['gt_images'][()]
    if self._shuffle:
      indices = np.random.permutation(len(in_images))
      in_images = [in_images[i] for i in indices]
      gt_images = [gt_images[i] for i in indices]
    in_batch = []
    gt_batch = []

    for in_patch, gt_patch in zip(in_images, gt_images):
      if self._geo_aug:
        in_patch, gt_patch = augment_img(in_patch, gt_patch)
      in_batch.append(img_to_tensor(in_patch))
      gt_batch.append(img_to_tensor(gt_patch))
    return {'in_images': torch.stack(in_batch).to(dtype=torch.float32),
            'gt_images': torch.stack(gt_batch).to(dtype=torch.float32)}

  def _create_hdf5_files(self):
    """Creates HDF5 files where each file contains 'batch_size' images."""
    in_images: List[np.ndarray] = []
    gt_images: List[np.ndarray] = []
    file_counter = 0
    for i, (in_img_path, gt_img_path, data_path) in enumerate(self._pairs):
      print(f'Processing {i}/{len(self._pairs)} ...')
      in_img, gt_img = load_image_pair(in_img_path, gt_img_path, data_path,
                                        image_size=None if self._extract_patches else self._image_size)
      if self._extract_patches:
        patches = extract_non_overlapping_patches(img=in_img, gt_img=gt_img, num_patches=0,
                                                  patch_size=self._image_size, allow_overlap=True,
                                                  add_resized_patch=True)
        for in_patch, gt_patch in zip(patches['img'], patches['gt']):
          in_images.append(in_patch.astype(np.float32))
          gt_images.append(gt_patch.astype(np.float32))
          if len(in_images) == self._batch_size:
            self._write_hdf5(file_counter, in_images, gt_images)
            in_images = []
            gt_images = []
            file_counter += 1
      else:
        in_images.append(in_img)
        gt_images.append(gt_img)
        if len(in_images) == self._batch_size:
          self._write_hdf5(file_counter, in_images, gt_images)
          in_images = []
          gt_images = []
          file_counter += 1

    if in_images:
      self._write_hdf5(file_counter, in_images, gt_images)

  def _write_hdf5(self, file_id: int, in_images: List[np.ndarray], gt_images: List[np.ndarray]):
    fname = join(self._temp_dir, f"batch_{file_id:05d}.h5")
    logging.info(f'Writing batch of {len(in_images)} images to {fname}...')
    with h5py.File(fname, 'w') as f:
      f.create_dataset('in_images', data=np.stack(in_images), compression='gzip')
      f.create_dataset('gt_images', data=np.stack(gt_images), compression='gzip')
