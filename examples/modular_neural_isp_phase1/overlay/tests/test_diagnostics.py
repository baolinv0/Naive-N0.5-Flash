"""Frozen diagnostics use identity and float measurements, never display pixels."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

OVERLAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OVERLAY))


def diagnostics():
    path = OVERLAY / 'tm_research/diagnostics.py'
    assert path.exists(), 'diagnostics implementation is missing'
    from tm_research import diagnostics as module
    return module


def test_pair_join_ignores_order_and_reports_missing_coverage():
    d = diagnostics()
    profile = {'samples': [{'sample_id': 'a', 'scene_id': 's', 'tags': ['low', 'shared']},
                           {'sample_id': 'b', 'scene_id': 's', 'tags': ['shared']},
                           {'sample_id': 'c', 'scene_id': 'unknown', 'tags': []}]}
    result = d.compare_per_image([{'sample_id': 'b', 'psnr': 12}, {'sample_id': 'a', 'psnr': 11}],
                                [{'sample_id': 'a', 'psnr': 10}, {'sample_id': 'b', 'psnr': 14},
                                 {'sample_id': 'c', 'psnr': 20}], profile)
    assert result['status'] == 'partial'
    assert result['coverage']['missing_candidate'] == ['c']
    assert result['overall']['scope'] == 'matched_subset'
    assert result['overall']['mean_paired_psnr_delta_db'] == -.5
    assert result['groups']['shared']['n_images'] == 2
    assert result['groups']['shared']['n_scenes'] == 1
    assert result['groups']['low']['mean_paired_psnr_delta_db'] == 1
    assert [row['sample_id'] for row in result['paired_rows']] == ['a', 'b']


@pytest.mark.parametrize('rows', [[{'sample_id': 'a', 'psnr': 1}, {'sample_id': 'a', 'psnr': 2}],
                                  [{'sample_id': 'a', 'psnr': float('nan')}],
                                  [{'sample_id': 'a', 'psnr': 1, 'ssim': float('inf')}]])
def test_pair_comparison_rejects_invalid_scores(rows):
    with pytest.raises(ValueError):
        diagnostics().compare_per_image(rows, [{'sample_id': 'a', 'psnr': 1}], {})


def test_roi_rgb_denominator_bias_empty_and_no_tensor_mutation():
    d = diagnostics()
    target = np.zeros((5, 5, 3))
    output = target.copy()
    output[..., 0] = .3
    before = output.copy()
    masks = {'all': np.ones((5, 5), bool), 'empty': np.zeros((5, 5), bool)}
    rows = d.measure_fixed_regions(output, target, masks, {'layout': 'HWC', 'metrics': ['mse', 'psnr', 'codevalue_bias']})
    by_key = {(r['roi_id'], r['metric_name']): r for r in rows}
    assert by_key['all', 'mse']['value'] == pytest.approx(.03)
    assert by_key['all', 'psnr']['value'] == pytest.approx(-10 * np.log10(.03))
    assert by_key['all', 'codevalue_bias']['value'] == pytest.approx(.3 * .2126)
    assert by_key['empty', 'mse']['value'] is None
    assert by_key['empty', 'mse']['valid_reason'] == 'empty_roi'
    np.testing.assert_array_equal(output, before)
    np.testing.assert_array_equal(target, 0)


def test_filtered_metrics_use_full_image_before_erosion_and_own_area():
    d = diagnostics()
    target = np.zeros((7, 7, 3))
    output = np.full_like(target, .1)
    mask = np.zeros((7, 7), bool)
    mask[1:6, 1:6] = True
    rows = d.measure_fixed_regions(output, target, {'center': mask}, {'layout': 'HWC', 'lowpass_radius': 1})
    r = {row['metric_name']: row for row in rows}
    assert r['mse']['n_pixels'] == 25
    assert r['low_frequency_mse']['n_pixels'] == 9
    assert r['gradient_mse']['n_pixels'] == 9
    assert r['low_frequency_mse']['value'] == pytest.approx(.01)
    assert r['gradient_mse']['value'] == pytest.approx(0)
    assert r['mse']['effective_mask_id'] != r['gradient_mse']['effective_mask_id']


def test_layout_shape_and_small_roi_are_explicit():
    d = diagnostics()
    with pytest.raises(ValueError, match='layout'):
        d.measure_fixed_regions(np.zeros((3, 3, 3)), np.zeros((3, 3, 3)), {}, {})
    rows = d.measure_fixed_regions(np.ones((1, 3, 4, 4)) * .2, np.zeros((1, 3, 4, 4)),
                                   {'tiny': np.eye(4, dtype=bool)}, {'layout': 'NCHW', 'min_pixels': 5})
    assert all(r['value'] is None for r in rows)
    assert rows[0]['valid_reason'] == 'below_min_pixels'


def test_profile_uses_shared_preprocessing_and_never_test(tmp_path, monkeypatch):
    d = diagnostics()
    calls = []
    pairs = {}
    for split in ('train', 'validation'):
        details = {key: str(tmp_path / split / key) for key in ('input_dir', 'gt_dir', 'metadata_dir')}
        for folder in details.values():
            Path(folder).mkdir(parents=True)
        pairs[details['input_dir']] = [(str(Path(details['input_dir']) / 'same.png'),
                                       str(Path(details['gt_dir']) / 'same.jpg'),
                                       str(Path(details['metadata_dir']) / 'same.json'))]
        (Path(details['metadata_dir']) / 'same.json').write_text('{"scene_id":"actual-scene"}')
        if split == 'train': train = details
        else: dev = details
    def pairer(a, b, c):
        calls.append(a)
        return pairs[a]
    def loader(a, b, c, image_size=None, quarter=False):
        assert image_size == 8 and not quarter
        v = .1 if '/train/' in a else .9
        return np.full((8, 8, 3), v), np.full((8, 8, 3), v)
    monkeypatch.setattr(d, '_pair_helpers', lambda: (pairer, loader))
    config = {'train': train, 'validation': dev, 'test': {'input_dir': 'DO_NOT_READ'}, 'eval_size': 8}
    rules = {'profile_revision': 'p-test', 'threshold_source': 'TRAIN', 'roi': True}
    built = d.build_dev_profile(config, rules, str(tmp_path / 'profile'))
    p = json.loads(Path(built['profile_ref']).read_text())
    assert len(calls) == 2 and 'DO_NOT_READ' not in calls
    assert [s['sample_id'] for s in p['samples']] == ['TRAIN:same.png', 'DEV:same.png']
    assert p['scene_overlap'][0]['scene_id'] == 'actual-scene'
    assert p['rules']['threshold_source'] == 'TRAIN'
    assert p['rules']['codevalue_thresholds'] == pytest.approx([.1, .1])
    assert p['samples'][1]['features']['dark_fraction'] == 0
    roi = json.loads(Path(built['roi_profile_ref']).read_text())
    assert roi['profile_revision'] == 'p-test'
    masks = d.load_sample_masks(p, 'DEV:same.png', (8, 8), 'p-test')
    assert sum(int(m.sum()) for key, m in masks.items() if key != 'all') == 64
    with pytest.raises(ValueError, match='size'):
        d.load_sample_masks(p, 'DEV:same.png', (7, 8), 'p-test')


def test_ambiguous_csv_basename_is_rejected():
    d = diagnostics()
    profile = {'samples': [{'sample_id': 'a', 'image': 'same.png', 'split': 'DEV'},
                           {'sample_id': 'b', 'image': 'same.png', 'split': 'DEV'}]}
    with pytest.raises(ValueError, match='ambiguous'):
        d.compare_per_image([{'image': 'same.png', 'psnr': 1}], [], profile)


def test_float_eval_profile_preserves_core_psnr_and_isolates_failure(tmp_path, monkeypatch):
    import torch
    sys.path.insert(0, str(OVERLAY / 'photofinishing'))
    spec = importlib.util.spec_from_file_location('diagnostic_eval_test', OVERLAY / 'photofinishing/test.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    class Constant(torch.nn.Module):
        def forward(self, value, **kwargs):
            return {'output': torch.full_like(value, .123456)}
    raw, gt, meta = (tmp_path / n for n in ('raw', 'gt', 'data'))
    for p in (raw, gt, meta): p.mkdir()
    for p in (raw / 'a.png', gt / 'a.jpg', meta / 'a.json'): p.touch()
    monkeypatch.setattr(evaluation, 'load_image_pair', lambda *args, **kwargs: (np.zeros((16, 16, 3), dtype=np.float32), np.zeros((16, 16, 3), dtype=np.float32)))
    (meta / 'a.json').write_text('{}')
    d = diagnostics()
    monkeypatch.setattr(d, '_pair_helpers', lambda: (evaluation.paired_files, evaluation.load_image_pair))
    profile = d.build_dev_profile({'validation': {'input_dir': str(raw), 'gt_dir': str(gt),
                                      'metadata_dir': str(meta)}, 'eval_size': 16},
              {'splits': ['DEV'], 'threshold_source': 'explicit', 'codevalue_thresholds': [.2, .8],
               'gradient_thresholds': [.01, .05], 'roi': True, 'metric_config': {'metrics': ['mse']}},
              str(tmp_path / 'profile'))
    profile_path = Path(profile['profile_ref'])
    def run(folder, diagnostics_profile=None, source_raw=raw):
        evaluation.test_net(Constant(), torch.device('cpu'), str(source_raw), str(gt), str(meta), False, False,
                            result_dir=str(folder), eval_size=16, diagnostics_profile=diagnostics_profile)
        return json.loads((folder / 'metrics.json').read_text())
    plain = run(tmp_path / 'plain')
    observed = run(tmp_path / 'observed', str(profile_path))
    failed = run(tmp_path / 'failed', str(tmp_path / 'missing.json'))
    other_raw = tmp_path / 'wrong-dataset'; other_raw.mkdir(); (other_raw / 'a.png').touch()
    wrong_dataset = run(tmp_path / 'wrong', str(profile_path), source_raw=other_raw)
    for m in (observed, failed, wrong_dataset):
        assert m['mean_psnr'] == plain['mean_psnr']
        assert m['num_images'] == plain['num_images'] == 1
        assert m['finite'] == plain['finite'] is True
    assert observed['diagnostics_status'] == 'complete'
    import csv
    rows = list(csv.DictReader(Path(observed['roi_metrics_csv']).open()))
    assert float(rows[0]['value']) == pytest.approx(float(np.float32(.123456)) ** 2)
    assert rows[0]['sample_id'] == 'DEV:a.png'
    assert failed['diagnostics_status'] == 'failed'
    assert failed['diagnostics_errors']
    assert wrong_dataset['diagnostics_status'] == 'failed'
    assert 'root' in wrong_dataset['diagnostics_errors'][0]


def test_pair_missing_psnr_is_a_clear_invalid_diagnostic():
    with pytest.raises(ValueError, match='PSNR'):
        diagnostics().compare_per_image([{'sample_id': 'a', 'psnr': None}],
                                        [{'sample_id': 'a', 'psnr': 1}], {})


def test_profile_rejects_test_and_unfrozen_rules_before_reading_data(tmp_path):
    d = diagnostics()
    with pytest.raises(ValueError, match='TRAIN/DEV'):
        d.build_dev_profile({}, {'splits': ['TEST']}, str(tmp_path))
    with pytest.raises(ValueError, match='threshold'):
        d.build_dev_profile({}, {'threshold_source': 'DEV'}, str(tmp_path))


def test_roi_different_sized_images_keep_macro_and_micro_distinct():
    d = diagnostics()
    values = []
    for n, error in ((2, .1), (4, .2)):
        row = d.measure_fixed_regions(np.full((n, n, 3), error), np.zeros((n, n, 3)),
                                      {'all': np.ones((n, n), bool)}, {'layout': 'HWC', 'metrics': ['mse', 'psnr']})
        values.append({r['metric_name']: r for r in row})
    macro = np.mean([v['psnr']['value'] for v in values])
    micro_mse = sum(v['mse']['value'] * v['mse']['n_pixels'] for v in values) / sum(v['mse']['n_pixels'] for v in values)
    assert macro == pytest.approx((20 - 20 * np.log10(.2)) / 2)
    assert macro != pytest.approx(-10 * np.log10(micro_mse))


def test_linear_gradient_operator_and_eroded_boundary():
    d = diagnostics()
    ramp = np.broadcast_to(np.arange(5)[None, :, None] * .1, (5, 5, 3)).copy()
    rows = d.measure_fixed_regions(ramp, np.zeros_like(ramp), {'all': np.ones((5, 5), bool)},
                                   {'layout': 'HWC', 'metrics': ['gradient_mse']})
    assert rows[0]['n_pixels'] == 9
    assert rows[0]['value'] == pytest.approx(.1 ** 2 / 2)


def test_fixed_group_with_no_matched_samples_is_retained_as_unavailable():
    p = {'samples': [{'sample_id': 'a', 'tags': ['missing_group']},
                     {'sample_id': 'b', 'tags': ['observed']}]}
    result = diagnostics().compare_per_image([{'sample_id': 'b', 'psnr': 2}],
                                            [{'sample_id': 'a', 'psnr': 1}, {'sample_id': 'b', 'psnr': 1}], p)
    group = result['groups']['missing_group']
    assert group['n_images'] == 0
    assert group['n_expected_images'] == 1
    assert group['mean_paired_psnr_delta_db'] is None
    assert group['status'] == 'unavailable'


def test_profile_requires_explicit_thresholds_before_data_access(tmp_path, monkeypatch):
    d = diagnostics()
    def forbidden():
        pytest.fail('Unfrozen rules must not scan/load any data')
    monkeypatch.setattr(d, '_pair_helpers', forbidden)
    with pytest.raises(ValueError, match='thresholds'):
        d.build_dev_profile({}, {'threshold_source': 'explicit'}, str(tmp_path))


def frozen_profile_fixture(tmp_path, monkeypatch):
    d = diagnostics()
    details = {key: str(tmp_path / key) for key in ('input_dir', 'gt_dir', 'metadata_dir')}
    for folder in details.values(): Path(folder).mkdir(parents=True)
    paths = [str(Path(details[key]) / filename) for key, filename in
             [('input_dir', 'a.png'), ('gt_dir', 'a.jpg'), ('metadata_dir', 'a.json')]]
    for path in paths: Path(path).touch()
    Path(paths[2]).write_text('{}')
    target = np.zeros((7, 7, 3)); target[:, :3] = .1; target[:, 3:] = .9
    monkeypatch.setattr(d, '_pair_helpers', lambda: (lambda *args: [tuple(paths)],
                        lambda *args, **kwargs: (target.copy(), target.copy())))
    config = {'validation': details, 'eval_size': 7}
    p = d.build_dev_profile(config, {'splits': ['DEV'], 'threshold_source': 'explicit',
          'codevalue_thresholds': [.2, .8], 'gradient_thresholds': [.01, .05], 'roi': True},
          str(tmp_path / 'profile'))
    return d, p, config


def test_same_area_mask_replacement_is_rejected(tmp_path, monkeypatch):
    d, p, _ = frozen_profile_fixture(tmp_path, monkeypatch)
    roi = json.loads(Path(p['roi_profile_ref']).read_text())
    entry = roi['samples'][0]
    with np.load(entry['mask_ref']) as data:
        masks = {key: data[key].copy() for key in data.files}
    key = next(key for key, mask in masks.items() if key != 'all' and 0 < mask.sum() < mask.size)
    masks[key] = np.roll(masks[key], 2, axis=1)
    assert masks[key].sum() == entry['n_pixels'][key]
    np.savez_compressed(entry['mask_ref'], **masks)
    with pytest.raises(ValueError, match='identity|geometry'):
        d.load_sample_masks(p, 'DEV:a.png', (7, 7), p['profile_revision'])


def test_wrong_dataset_same_names_and_size_is_rejected(tmp_path, monkeypatch):
    d, p, cfg = frozen_profile_fixture(tmp_path, monkeypatch)
    active = cfg['validation'].copy()
    for key in active:
        folder = tmp_path / 'other' / key
        folder.mkdir(parents=True)
        active[key] = str(folder)
    with pytest.raises(ValueError, match='root|resource|dataset'):
        d.validate_evaluation_profile(p, active, 7)
    moved = {cfg['validation'][key]: active[key] for key in active}
    # Explicit relocation must additionally have matching declared file inventory.
    for key, filename in [('input_dir', 'a.png'), ('gt_dir', 'a.jpg'), ('metadata_dir', 'a.json')]:
        (Path(active[key]) / filename).write_bytes((Path(cfg['validation'][key]) / filename).read_bytes())
    d.validate_evaluation_profile(p, active, 7, root_mapping=moved)
    corrupt = json.loads(json.dumps(p)); corrupt['transform']['loader'] = 'different_loader'
    with pytest.raises(ValueError, match='transform'):
        d.validate_evaluation_profile(corrupt, cfg['validation'], 7)


def test_measurement_rows_carry_exact_operator_definition_and_frozen_identity():
    d = diagnostics()
    target = np.zeros((7, 7, 3)); output = target.copy(); output[3, 3] = 1
    masks = {'all': np.ones((7, 7), bool)}
    cfg = {'layout': 'HWC', 'metrics': ['low_frequency_mse'], 'sample_id': 'DEV:a.png',
           'profile_ref': '/frozen/p1/profile.json', 'profile_revision': 'p001', 'lowpass_sigma': .1}
    a = d.measure_fixed_regions(output, target, masks, cfg)[0]
    b = d.measure_fixed_regions(output, target, masks, dict(cfg, lowpass_sigma=5))[0]
    assert a['metric_definition'] != b['metric_definition']
    assert json.loads(a['metric_definition'])['lowpass_sigma'] == .1
    other = d.measure_fixed_regions(output, target, masks, dict(cfg, profile_ref='/frozen/p2/profile.json'))[0]
    assert a['effective_mask_id'] != other['effective_mask_id']
    sample = d.measure_fixed_regions(output, target, masks, dict(cfg, sample_id='DEV:b.png'))[0]
    assert a['effective_mask_id'] != sample['effective_mask_id']
    with pytest.raises(ValueError, match='normalization'):
        d.measure_fixed_regions(output, target, masks, dict(cfg, gradient_normalization='sum'))


def test_profile_exposes_complete_roi_inventory_and_snapshot(tmp_path, monkeypatch):
    d, p, _ = frozen_profile_fixture(tmp_path, monkeypatch)
    roi = json.loads(Path(p['roi_profile_ref']).read_text())
    assert len(roi['roi_inventory']) == 10 * len(roi['metric_config']['metrics'])
    snapshot = d.diagnostics_snapshot(p)
    assert snapshot['profile_ref'] == p['profile_ref']
    assert snapshot['rules'] == p['rules']
    assert snapshot['roi_inventory'] == roi['roi_inventory']
    assert snapshot['roi_samples'][0]['mask_definitions']
