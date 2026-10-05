"""Candidate-independent TRAIN/DEV profiles and deterministic float diagnostics.

These measurements describe fixed groups; they never make selection decisions.
"""
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

CODE_WEIGHTS = np.array([.2126, .7152, .0722])
METRICS = ('mse', 'psnr', 'codevalue_bias', 'low_frequency_mse', 'gradient_mse')


def evaluation_transform(size):
    """The actual evaluator/loader contract, shared with profile validation."""
    return {'loader': 'baseline_utils.load_image_pair', 'image_size': size, 'quarter': False,
            'coordinates': 'row_y_column_x', 'color': 'normalized_GT_RGB_codevalues',
            'codevalue_weights': CODE_WEIGHTS.tolist(),
            'threshold_boundaries': 'low < t0; mid t0 <= x < t1; high x >= t1',
            'gradient_operator': 'central_difference_divide_2', 'gradient_boundary': 'reflect'}


def _profile_identity(profile):
    keys = ('profile_ref', 'profile_revision', 'source', 'transform', 'rules', 'split_roots', 'samples')
    if any(key not in profile for key in keys):
        raise ValueError('Frozen profile identity is incomplete')
    return copy.deepcopy({key: profile[key] for key in keys})


def _mask_definition(mask):
    """Exact readable geometry: contiguous [row,start,end-exclusive] spans."""
    spans = []
    for y, row in enumerate(mask):
        changes = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
        spans.extend([[y, int(a), int(b)] for a, b in zip(changes[::2], changes[1::2])])
    return {'shape': list(mask.shape), 'row_spans': spans}


def _effective_mask_id(profile_ref, sample_id, roi_id, support):
    return f'{profile_ref or "unverified_profile"}#{sample_id or "unverified_sample"}/{roi_id}/erode{support}'


def validate_evaluation_profile(profile, details, eval_size, *, root_mapping=None):
    """Check active DEV resources and transform before using frozen identities.

    A relocation is an explicit source-root -> destination-root mapping. Both
    inventories must exist, and relocated files are byte-compared when the source
    remains accessible. Missing source files cannot validate a relocation.
    """
    if profile.get('eval_size') != eval_size or profile.get('transform') != evaluation_transform(eval_size):
        raise ValueError('Diagnostic profile evaluation transform mismatch')
    _profile_identity(profile)
    frozen = profile.get('split_roots', {}).get('DEV')
    if not frozen:
        raise ValueError('Diagnostic profile DEV dataset resource roots missing')
    keys = ('input_dir', 'gt_dir', 'metadata_dir')
    actual = dict(details)
    actual['metadata_dir'] = actual.get('metadata_dir') or str(Path(actual['input_dir']).parent / 'data')
    original = {k: str(Path(frozen[k]).resolve()) for k in keys}
    mappings = root_mapping or {}
    if not isinstance(mappings, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mappings.items()):
        raise ValueError('Dataset root mapping must map source root paths to destination root paths')
    mappings = {str(Path(k).resolve()): str(Path(v).resolve()) for k, v in mappings.items()}
    if set(mappings) - set(original.values()):
        raise ValueError('Dataset root mapping contains an unknown frozen DEV resource root')
    for key in keys:
        expected = mappings.get(original[key], original[key])
        if str(Path(actual[key]).resolve()) != expected:
            raise ValueError(f'Diagnostic profile DEV dataset root mismatch: {key}')
        if not Path(expected).is_dir():
            raise ValueError(f'Diagnostic dataset root unavailable: {key}')
    path_fields = {'input_dir': 'input_relpath', 'gt_dir': 'gt_relpath', 'metadata_dir': 'metadata_relpath'}
    for sample in _dev_samples(profile):
        for key, field in path_fields.items():
            rel = Path(sample[field])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Invalid frozen dataset relative path')
            active_path, source_path = Path(actual[key]) / rel, Path(original[key]) / rel
            if not active_path.is_file():
                raise ValueError('Diagnostic dataset file inventory missing')
            if original[key] in mappings and mappings[original[key]] != original[key]:
                if not source_path.is_file():
                    raise ValueError('Dataset relocation cannot be verified: source file unavailable')
                if source_path.read_bytes() != active_path.read_bytes():
                    raise ValueError('Dataset relocation differs from frozen source file')
    return {'validated': True, 'active_roots': {k: str(Path(actual[k]).resolve()) for k in keys},
            'root_mapping': mappings, 'transform': evaluation_transform(eval_size)}


def diagnostics_snapshot(profile):
    """Persist actual frozen source, rules, geometry, metrics and expected rows."""
    snapshot = _profile_identity(profile)
    snapshot['roi_profile_ref'] = profile.get('roi_profile_ref')
    if profile.get('roi_profile_ref'):
        roi = json.loads(Path(profile['roi_profile_ref']).read_text(encoding='utf-8'))
        if not isinstance(roi, dict):
            raise ValueError('ROI profile must be a JSON object')
        if roi.get('profile_identity') != _profile_identity(profile):
            raise ValueError('ROI frozen profile identity mismatch')
        snapshot.update(metric_config=copy.deepcopy(roi['metric_config']),
                        roi_samples=copy.deepcopy(roi['samples']),
                        roi_inventory=copy.deepcopy(roi['roi_inventory']))
    else:
        snapshot.update(metric_config=None, roi_samples=[], roi_inventory=[])
    return snapshot


def _pair_helpers():
    # The official ISP supplies utils; importing pure comparisons needs no ISP/torch.
    folder = str(Path(__file__).resolve().parents[1] / 'photofinishing')
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from baseline_utils import paired_files, load_image_pair
    return paired_files, load_image_pair


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def _array(image, layout):
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()
    value = np.array(image, dtype=np.float64, copy=True)
    if layout == 'NCHW':
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError('NCHW diagnostics require exactly one image')
        value = value[0].transpose(1, 2, 0)
    elif layout == 'CHW':
        if value.ndim != 3:
            raise ValueError('CHW requires three dimensions')
        value = value.transpose(1, 2, 0)
    elif layout != 'HWC':
        raise ValueError('Explicit layout HWC, CHW or NCHW is required')
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 2:
        raise ValueError('Diagnostics require H x W x 3 images with H,W >= 2')
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite float image')
    return value


def _gradient(value):
    """Central differences /2, reflect boundary, x/y components."""
    p = np.pad(value, ((1, 1), (1, 1)) + ((0, 0),) * (value.ndim - 2), mode='reflect')
    return (p[1:-1, 2:] - p[1:-1, :-2]) / 2, (p[2:, 1:-1] - p[:-2, 1:-1]) / 2


def _features(target, thresholds):
    y = target @ CODE_WEIGHTS
    gx, gy = _gradient(y)
    gradient = np.sqrt(gx * gx + gy * gy)
    return y, gradient, {'mean_codevalue': float(y.mean()),
                         'dark_fraction': float((y < thresholds[0]).mean()),
                         'bright_fraction': float((y >= thresholds[1]).mean()),
                         'mean_gradient': float(gradient.mean())}


def _bins(value, thresholds):
    return np.where(value < thresholds[0], 0, np.where(value < thresholds[1], 1, 2))


def _check_thresholds(rules):
    for key in ('codevalue_thresholds', 'gradient_thresholds'):
        values = rules.get(key)
        if (not isinstance(values, (list, tuple)) or len(values) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 <= values[0] <= values[1]):
            raise ValueError(f'{key} requires two frozen nonnegative ordered values')


def build_dev_profile(config: dict, rules: dict, output_dir: str) -> dict:
    """Freeze shared-transform GT codevalue/gradient rules; never enumerate TEST.

    threshold_source is explicit (required codevalue_thresholds and
    gradient_thresholds) or TRAIN (global TRAIN pixel quantiles). Existing outputs
    are refused: revisions must be saved to a new directory.
    """
    rules = copy.deepcopy(rules)
    source = rules.get('threshold_source', 'explicit')
    if source not in ('TRAIN', 'explicit'):
        raise ValueError('threshold_source must be TRAIN or explicit frozen thresholds')
    revision = rules.get('profile_revision', 'p001')
    if not isinstance(revision, str) or not revision:
        raise ValueError('profile_revision must be a nonempty string')
    splits = rules.get('splits', ['TRAIN', 'DEV'])
    if not splits or any(s not in ('TRAIN', 'DEV') for s in splits) or len(set(splits)) != len(splits):
        raise ValueError('Only unique authorized TRAIN/DEV splits are supported')
    if source == 'TRAIN' and 'TRAIN' not in splits:
        raise ValueError('TRAIN calibration requires the TRAIN split')
    if source == 'explicit':
        _check_thresholds(rules)
    size = config.get('eval_size', 512)
    if type(size) is not int or size < 2:
        raise ValueError('eval_size must be an integer >= 2')
    output = Path(output_dir).resolve()
    if (output / 'profile.json').exists() or (output / 'roi_profile.json').exists():
        raise ValueError('Frozen profile already exists; use a new revision directory')
    pairer, loader = _pair_helpers()
    records, train_y, train_g = [], [], []
    for split in splits:
        details = config['train' if split == 'TRAIN' else 'validation']
        metadata = details.get('metadata_dir') or str(Path(details['input_dir']).parent / 'data')
        for inp, gt, meta in pairer(details['input_dir'], details['gt_dir'], metadata):
            rel = Path(inp).relative_to(details['input_dir']).as_posix()
            sample_id = f'{split}:{rel}'
            annotation = rules.get('annotations', {}).get(sample_id, {})
            metadata_value = json.loads(Path(meta).read_text(encoding='utf-8'))
            # scene_id is an explicit collection relationship, never stem-derived.
            scene = annotation.get('scene_id', metadata_value.get('scene_id', 'unknown')) or 'unknown'
            sample = {'sample_id': sample_id, 'split': split, 'image': Path(inp).name,
                      'input_relpath': rel,
                      'gt_relpath': Path(gt).relative_to(details['gt_dir']).as_posix(),
                      'metadata_relpath': Path(meta).relative_to(metadata).as_posix(),
                      'scene_id': str(scene), 'annotation_source': annotation.get('source', 'GT_codevalue_and_gradient'),
                      'scene_source': annotation.get('source', 'explicit_annotation') if 'scene_id' in annotation else
                        ('metadata.scene_id' if 'scene_id' in metadata_value else 'unknown'),
                      'annotation_version': annotation.get('version', revision),
                      'annotation_tags': annotation.get('tags', [])}
            records.append((sample, (inp, gt, meta)))
            if split == 'TRAIN' and source == 'TRAIN':
                _, target = loader(inp, gt, meta, image_size=size, quarter=False)
                target = _array(target, 'HWC')
                if target.shape[:2] != (size, size):
                    raise ValueError('Shared evaluation transform returned unexpected size')
                y, g, _ = _features(target, (0, 1))
                train_y.append(y.ravel()); train_g.append(g.ravel())
    if len({s['sample_id'] for s, _ in records}) != len(records):
        raise ValueError('Duplicate sample_id in manifest')
    if source == 'TRAIN':
        if not train_y:
            raise ValueError('No TRAIN calibration pixels')
        quantiles = rules.get('calibration_quantiles', [1 / 3, 2 / 3])
        if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError('calibration_quantiles must be two ordered fractions')
        rules['codevalue_thresholds'] = np.quantile(np.concatenate(train_y), quantiles).tolist()
        rules['gradient_thresholds'] = np.quantile(np.concatenate(train_g), quantiles).tolist()
        rules['calibration_quantiles'] = quantiles
        rules['calibration_method'] = 'global_TRAIN_transformed_pixel_quantiles_numpy_linear'
    del train_y, train_g
    _check_thresholds(rules)
    rules['threshold_source'] = source
    rules.setdefault('fraction_high', .25)
    if not 0 <= rules['fraction_high'] <= 1:
        raise ValueError('fraction_high must be between zero and one')
    rules.setdefault('win_tolerance_db', 0.0)
    rules.setdefault('small_group_images', 5)
    rules.setdefault('case_selection', {'persistent_window': 3, 'persistent_fraction': .2,
                                      'max_per_category': 2, 'max_cases': 8,
                                      'ordinary_rule': 'sample_id_order', 'tie_break': 'sample_id'})
    metric_config = {'layout': 'HWC', 'metrics': list(METRICS), 'epsilon': 1e-12,
                     'data_range': 1.0, 'min_pixels': 1, 'lowpass_radius': 1,
                     'lowpass_sigma': 1.0, 'boundary': 'reflect', 'erode_filtered_masks': True,
                     'gradient_operator': 'central_difference_divide_2',
                     'gradient_normalization': 'mean_over_RGB_and_xy'}
    metric_config.update(rules.get('metric_config', {}))
    # Coordinate convention and operators are fixed by this implementation.
    if metric_config['layout'] != 'HWC' or metric_config['boundary'] != 'reflect' or metric_config['gradient_operator'] != 'central_difference_divide_2':
        raise ValueError('Unsupported ROI transform/operator')
    if not metric_config['metrics'] or len(set(metric_config['metrics'])) != len(metric_config['metrics']):
        raise ValueError('ROI metrics must be a nonempty unique metric list')
    for metric in metric_config['metrics']:
        fixed_metric_definition(metric_config, metric)
    output.mkdir(parents=True, exist_ok=True)
    roi_samples, samples = [], []
    names = ('low', 'mid', 'high')
    for index, (sample, paths) in enumerate(records):
        _, target = loader(*paths, image_size=size, quarter=False)
        target = _array(target, 'HWC')
        if target.shape[:2] != (size, size):
            raise ValueError('Shared evaluation transform returned unexpected size')
        y, g, features = _features(target, rules['codevalue_thresholds'])
        tags = [f'codevalue_{names[int(_bins(features["mean_codevalue"], rules["codevalue_thresholds"]))]}',
                f'gradient_{names[int(_bins(features["mean_gradient"], rules["gradient_thresholds"]))]}']
        for name in ('dark', 'bright'):
            if features[f'{name}_fraction'] >= rules['fraction_high']:
                tags.append(f'{name}_fraction_high')
        sample['tags'] = sorted(set(tags + sample.pop('annotation_tags')))
        sample['features'] = features
        samples.append(sample)
        if rules.get('roi', False) and sample['split'] == 'DEV':
            ybin, gbin = _bins(y, rules['codevalue_thresholds']), _bins(g, rules['gradient_thresholds'])
            masks = {'all': np.ones(y.shape, bool)}
            for yi, yn in enumerate(names):
                for gi, gn in enumerate(names):
                    masks[f'codevalue_{yn}__gradient_{gn}'] = (ybin == yi) & (gbin == gi)
            mask_dir = output / 'masks'; mask_dir.mkdir(exist_ok=True)
            path = mask_dir / f'{index:06d}.npz'
            np.savez_compressed(path, **masks)
            roi_samples.append({'sample_id': sample['sample_id'], 'size': list(y.shape),
                                'mask_ref': str(path), 'n_pixels': {k: int(v.sum()) for k, v in masks.items()},
                                'mask_definitions': {k: _mask_definition(v) for k, v in masks.items()}})
    scenes = {}
    for sample in samples:
        if sample['scene_id'] != 'unknown':
            scenes.setdefault(sample['scene_id'], []).append(sample)
    overlap = [{'scene_id': k, 'sample_ids': [s['sample_id'] for s in v],
                'splits': sorted({s['split'] for s in v})} for k, v in sorted(scenes.items())
               if len({s['split'] for s in v}) > 1]
    transform = evaluation_transform(size)
    profile = {'schema_version': 1, 'profile_revision': revision, 'eval_size': size,
               'source': source, 'retrospective': bool(rules.get('retrospective', False)),
               'rules': rules, 'transform': transform, 'samples': samples, 'scene_overlap': overlap,
               'groups_overlap': True, 'profile_ref': str(output / 'profile.json')}
    profile['split_roots'] = {split: dict(config['train' if split == 'TRAIN' else 'validation'])
                              for split in splits}
    for roots in profile['split_roots'].values():
        roots['metadata_dir'] = roots.get('metadata_dir') or str(Path(roots['input_dir']).parent / 'data')
    if roi_samples:
        roi_ref = str(output / 'roi_profile.json')
        profile['roi_profile_ref'] = roi_ref
        inventory = []
        for sample in roi_samples:
            for roi_id in sample['n_pixels']:
                for metric in metric_config['metrics']:
                    support = {'low_frequency_mse': metric_config['lowpass_radius'], 'gradient_mse': 1}.get(metric, 0)
                    support = support if metric_config['erode_filtered_masks'] else 0
                    inventory.append({'sample_id': sample['sample_id'], 'roi_id': roi_id, 'metric_name': metric,
                      'effective_mask_id': _effective_mask_id(profile['profile_ref'], sample['sample_id'], roi_id, support)})
        profile['roi_inventory'] = inventory
        _write_json(roi_ref, {'schema_version': 1, 'profile_revision': revision,
                             'eval_size': size, 'transform': transform, 'mask_source': 'fixed_GT',
                             'metric_config': metric_config, 'aggregation': 'macro_equal_image_weight',
                             'samples': roi_samples, 'roi_inventory': inventory,
                             'profile_identity': _profile_identity(profile)})
    _write_json(output / 'profile.json', profile)
    return profile


def load_sample_masks(profile, sample_id, size, profile_revision=None):
    """Load the same cached masks for every candidate; refuse coordinate mismatch."""
    if profile_revision is not None and profile['profile_revision'] != profile_revision:
        raise ValueError('profile_revision mismatch')
    roi = json.loads(Path(profile['roi_profile_ref']).read_text(encoding='utf-8'))
    if (roi['profile_revision'] != profile['profile_revision'] or roi['transform'] != profile['transform']
            or roi.get('profile_identity') != _profile_identity(profile)):
        raise ValueError('ROI profile revision/transform mismatch')
    entries = [r for r in roi['samples'] if r['sample_id'] == sample_id]
    if len(entries) != 1:
        raise ValueError('Missing or duplicate ROI sample mapping')
    entry = entries[0]
    if list(size) != entry['size']:
        raise ValueError('ROI size mismatch')
    with np.load(entry['mask_ref'], allow_pickle=False) as cached:
        masks = {key: np.array(cached[key], copy=True) for key in cached.files}
    if set(masks) != set(entry.get('mask_definitions', {})):
        raise ValueError('Cached ROI geometry inventory mismatch')
    for key, mask in masks.items():
        if mask.dtype != np.bool_ or mask.shape != tuple(size) or int(mask.sum()) != entry['n_pixels'][key]:
            raise ValueError('Cached ROI mask size/area mismatch')
        if _mask_definition(mask) != entry['mask_definitions'][key]:
            raise ValueError('Cached ROI geometry identity mismatch')
    return masks


def _dev_samples(profile):
    return [s for s in profile.get('samples', []) if s.get('split', 'DEV') in ('DEV', 'validation', 'dev')]


def resolve_sample_id(row, profile):
    """Use explicit IDs, otherwise require unique DEV path/basename identity."""
    if row.get('sample_id'):
        return row['sample_id']
    samples = _dev_samples(profile)
    if row.get('input_relpath'):
        matches = [s for s in samples if s['input_relpath'] == row['input_relpath']]
    else:
        name = row.get('image', '')
        matches = [s for s in samples if s.get('image', Path(s.get('input_relpath', '')).name) == name]
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous sample mapping')
    return matches[0]['sample_id']


def compare_per_image(candidate_rows: list[dict], reference_rows: list[dict], profile: dict) -> dict:
    """Join IDs rather than row order; partial coverage is never a full DEV mean."""
    def index(rows):
        result = {}
        for row in rows:
            sid = resolve_sample_id(row, profile)
            if sid in result:
                raise ValueError(f'Duplicate sample_id {sid}')
            item = dict(row, sample_id=sid)
            for metric in ('psnr', 'ssim'):
                if metric in row and row[metric] not in (None, ''):
                    item[metric] = float(row[metric])
                    if not math.isfinite(item[metric]):
                        raise ValueError(f'Nonfinite {metric} for {sid}')
            if item.get('psnr') in (None, ''):
                raise ValueError(f'Missing PSNR for {sid}')
            result[sid] = item
        return result
    current, reference = index(candidate_rows), index(reference_rows)
    manifest = {s['sample_id']: s for s in _dev_samples(profile)}
    if len(manifest) != len(_dev_samples(profile)):
        raise ValueError('Duplicate sample_id in profile')
    expected = set(manifest) if manifest else set(current) | set(reference)
    matched = sorted(set(current) & set(reference))
    tolerance = float(profile.get('rules', {}).get('win_tolerance_db', 0))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('win_tolerance_db must be finite and nonnegative')
    paired = []
    for sid in matched:
        sample = manifest.get(sid, {})
        row = {'sample_id': sid, 'candidate_psnr': current[sid]['psnr'],
               'reference_psnr': reference[sid]['psnr'],
               'delta_psnr': current[sid]['psnr'] - reference[sid]['psnr'],
               'scene_id': sample.get('scene_id', 'unknown'), 'tags': sample.get('tags', ['unknown'])}
        for key in ('image', 'output_image'):
            if key in current[sid]: row[key] = current[sid][key]
        if all(r[sid].get('ssim') not in (None, '') for r in (current, reference)):
            row.update(candidate_ssim=current[sid]['ssim'], reference_ssim=reference[sid]['ssim'],
                       delta_ssim=current[sid]['ssim'] - reference[sid]['ssim'])
        paired.append(row)
    missing_current, missing_reference = sorted(expected - set(current)), sorted(expected - set(reference))
    extras = sorted((set(current) | set(reference)) - expected)
    complete = not missing_current and not missing_reference and not extras and bool(matched)
    def summary(rows):
        n = len(rows)
        delta = [r['delta_psnr'] for r in rows]
        scenes = {r['scene_id'] for r in rows if r['scene_id'] != 'unknown'}
        return {'n_images': n, 'n_scenes': len(scenes), 'unknown_scenes': sum(r['scene_id'] == 'unknown' for r in rows),
                'mean_candidate_psnr': math.fsum(r['candidate_psnr'] for r in rows) / n if n else None,
                'mean_reference_psnr': math.fsum(r['reference_psnr'] for r in rows) / n if n else None,
                'mean_paired_psnr_delta_db': math.fsum(delta) / n if n else None,
                'wins': sum(v > tolerance for v in delta), 'losses': sum(v < -tolerance for v in delta),
                'ties': sum(abs(v) <= tolerance for v in delta), 'sample_ids': [r['sample_id'] for r in rows],
                'small_sample': n < profile.get('rules', {}).get('small_group_images', 5)}
    tags = sorted({tag for row in paired for tag in row['tags']} |
                  {tag for sample in manifest.values() for tag in sample.get('tags', [])})
    overall = summary(paired); overall['scope'] = 'full' if complete else 'matched_subset'
    groups = {}
    for tag in tags:
        group = summary([r for r in paired if tag in r['tags']])
        expected_group = sum(tag in s.get('tags', []) for s in manifest.values()) if manifest else group['n_images']
        group.update(overlapping=True, n_expected_images=expected_group,
                     scope='full' if group['n_images'] == expected_group else 'matched_subset',
                     status='unavailable' if not group['n_images'] else
                       'complete' if group['n_images'] == expected_group else 'partial')
        groups[tag] = group
    return {'status': 'complete' if complete else ('partial' if matched else 'unavailable'),
            'comparable': bool(matched) and not extras, 'overall': overall, 'groups': groups,
            'paired_rows': paired, 'coverage': {'candidate_count': len(current), 'reference_count': len(reference),
              'matched_count': len(matched), 'expected_count': len(expected), 'missing_candidate': missing_current,
              'missing_reference': missing_reference, 'unknown_sample_ids': extras},
            'unavailable': [] if complete else ['incomplete sample coverage'],
            'profile_revision': profile.get('profile_revision'), 'win_tolerance_db': tolerance}


def _erode(mask, radius):
    if radius == 0: return mask.copy()
    padded = np.pad(mask, radius, mode='constant', constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (radius * 2 + 1,) * 2)
    return windows.all(axis=(-1, -2))


def _lowpass(image, radius, sigma):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-x * x / (2 * sigma * sigma)); kernel /= kernel.sum()
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1), axis=(0, 1))
    return np.einsum('hwcij,i,j->hwc', windows, kernel, kernel)


def fixed_metric_definition(metric_config, metric_name):
    """Canonical actual operator/settings, shared with persisted-row validation."""
    cfg = metric_config
    if metric_name not in METRICS:
        raise ValueError('Unsupported ROI metric')
    epsilon, value_range = float(cfg.get('epsilon', 1e-12)), float(cfg.get('data_range', 1))
    minimum, radius, sigma = cfg.get('min_pixels', 1), cfg.get('lowpass_radius', 1), float(cfg.get('lowpass_sigma', 1))
    if (not all(math.isfinite(v) and v > 0 for v in (epsilon, value_range, sigma))
            or type(minimum) is not int or minimum < 1 or type(radius) is not int or radius < 0):
        raise ValueError('Invalid metric epsilon/range/min_pixels/filter settings')
    if cfg.get('boundary', 'reflect') != 'reflect' or cfg.get('gradient_operator', 'central_difference_divide_2') != 'central_difference_divide_2':
        raise ValueError('Unsupported filter boundary/gradient operator')
    if cfg.get('gradient_normalization', 'mean_over_RGB_and_xy') != 'mean_over_RGB_and_xy':
        raise ValueError('Unsupported gradient normalization')
    if type(cfg.get('erode_filtered_masks', True)) is not bool:
        raise ValueError('erode_filtered_masks must be boolean')
    return {'implementation': 'fixed_regions_float64_v1', 'epsilon': epsilon,
            'data_range': value_range, 'min_pixels': minimum, 'lowpass_radius': radius,
            'lowpass_sigma': sigma, 'lowpass_kernel': 'normalized_gaussian_separable',
            'boundary': 'reflect', 'erode_filtered_masks': cfg.get('erode_filtered_masks', True),
            'gradient_operator': 'central_difference_divide_2',
            'gradient_normalization': 'mean_over_RGB_and_xy',
            'codevalue_weights': CODE_WEIGHTS.tolist(), 'mse_normalization': '3_times_n_pixels',
            'metric_name': metric_name}


def measure_fixed_regions(output, target, masks: dict, metric_config: dict) -> list[dict]:
    """Measure copied float images, with metric-specific eroded mask support.

    MSE divides by 3*n_pixels. Gradient MSE averages RGB and x/y components.
    Gaussian filtering runs on full images with reflect padding before masking.
    """
    cfg = metric_config
    out, gt = _array(output, cfg.get('layout')), _array(target, cfg.get('layout'))
    if out.shape != gt.shape:
        raise ValueError('Output/GT shape mismatch')
    metrics = cfg.get('metrics', METRICS)
    if set(metrics) - set(METRICS):
        raise ValueError('Unsupported ROI metric')
    definition = fixed_metric_definition(cfg, 'mse')
    epsilon, value_range = definition['epsilon'], definition['data_range']
    minimum, radius, sigma = definition['min_pixels'], definition['lowpass_radius'], definition['lowpass_sigma']
    differences = out - gt
    fields = {'mse': np.mean(differences ** 2, axis=2),
              'codevalue_bias': differences @ CODE_WEIGHTS}
    fields['psnr'] = fields['mse']
    if 'low_frequency_mse' in metrics:
        fields['low_frequency_mse'] = np.mean((_lowpass(out, radius, sigma) - _lowpass(gt, radius, sigma)) ** 2, axis=2)
    if 'gradient_mse' in metrics:
        ox, oy = _gradient(out); tx, ty = _gradient(gt)
        fields['gradient_mse'] = np.mean(((ox - tx) ** 2 + (oy - ty) ** 2) / 2, axis=2)
    rows = []
    for roi_id in sorted(masks):
        mask = np.asarray(masks[roi_id])
        if mask.shape != out.shape[:2] or mask.dtype != np.bool_:
            raise ValueError('ROI mask must be boolean with matching image size')
        for metric in metrics:
            support = {'low_frequency_mse': radius, 'gradient_mse': 1}.get(metric, 0)
            support = support if cfg.get('erode_filtered_masks', True) else 0
            effective = _erode(mask, support)
            area = int(effective.sum())
            reason = 'ok' if area >= minimum else ('empty_roi' if not mask.any() else ('eroded_empty_roi' if area == 0 else 'below_min_pixels'))
            value = float(fields[metric][effective].mean()) if reason == 'ok' else None
            if metric == 'psnr' and value is not None:
                value = 10 * math.log10(value_range ** 2 / max(value, epsilon))
            rows.append({'sample_id': cfg.get('sample_id'), 'roi_id': roi_id, 'metric_name': metric,
                         'value': value, 'n_pixels': area, 'region_fraction': area / mask.size,
                         'effective_mask_id': _effective_mask_id(cfg.get('profile_ref'), cfg.get('sample_id'), roi_id, support),
                         'valid_reason': reason, 'profile_revision': cfg.get('profile_revision'),
                         'profile_ref': cfg.get('profile_ref'),
                         'metric_definition': json.dumps(fixed_metric_definition(cfg, metric), sort_keys=True, separators=(',', ':'))})
    return rows
