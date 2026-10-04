"""Evidence invariants: IDs, frozen references and immutable measured facts."""
import csv
import importlib
import json
from pathlib import Path

import pytest


def evidence():
    assert importlib.util.find_spec('tm_research.evidence') is not None, 'T3 evidence module is missing'
    return importlib.import_module('tm_research.evidence')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def csv_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'campaign'
    runs = tmp_path / 'runs'
    config = {'runs_dir': str(runs), 'model': 'isp', 'seed': 1, 'epochs': 3,
              'eval_size': 512, 'train': {'input': '/train'}, 'validation': {'input': '/dev'},
              'init_checkpoint': '/frozen.pth', 'loss_family': 'original', 'optimizer': 'adam',
              'learning_rate': 0.01, 'weight_decay': 0.0}
    trials = []
    for i, scores in enumerate(([20, 21, 22, 23, 24, 25, 26, 27],
                                [19, 22, 22, 23, 24, 25, 26, 27],
                                [18, 23, 22, 23, 24, 25, 26, 27])):
        run_id = f'exp_{i+1:03d}'
        cfg = {**config, 'learning_rate': 0.01 / (i+1)}
        trial = {'run_id': run_id, 'index': i, 'status': 'completed', 'valid': True,
                 'result_status': 'valid' if i == 0 else 'improved', 'dev_psnr': sum(scores)/len(scores),
                 'resolved_config': cfg, 'recipe': {k: cfg[k] for k in
                       ('loss_family', 'optimizer', 'learning_rate', 'weight_decay')},
                 'dev_metrics': {'protocol': 'P512', 'num_images': len(scores)},
                 'proposal': {'based_on': {'run_id': 'exp_001'}}}
        if i:
            trial['references'] = {'baseline_run_id': 'exp_001', 'best_before_run_id': f'exp_{i:03d}',
                'comparison_run_id': 'exp_001', 'construction_base_run_id': f'exp_{i:03d}',
                'based_on_run_id': 'exp_001', 'init_checkpoint_ref': '/frozen.pth'}
        trials.append(trial)
        csv_rows(runs / run_id / 'dev/per_image.csv',
                 [{'sample_id': f's{j}', 'psnr': s, 'ssim': 0.8} for j, s in reversed(list(enumerate(scores)))])
    state = {'campaign_id': 'fixture', 'config': config, 'trials': trials,
             'best': {'run_id': 'exp_003'}, 'min_delta': 0.01}
    dump(root / 'campaign.json', state)
    return root, runs, state


def test_stable_join_and_best_before_is_not_current_best(campaign):
    root, _, _ = campaign
    result = evidence().build_run_feedback(str(root), 'exp_003')
    assert result['best_before_run_id'] == 'exp_002'
    assert result['comparison_run_id'] == 'exp_001'
    assert result['construction_base_run_id'] == 'exp_002'
    assert result['based_on_run_id'] == 'exp_001'
    assert result['init_checkpoint_ref'] == '/frozen.pth'
    comparison = result['comparisons']['comparison']
    assert comparison['overall']['mean_paired_psnr_delta_db'] == 0
    rows = evidence().load_feedback_detail(Path(result['feedback_ref']).parent, 'paired_rows')
    assert next(row for row in rows if row['sample_id'] == 's0' and row['comparison_role'] == 'comparison')['delta_psnr'] == -2
    assert result['groups']['unknown']['n_images'] == 8


def test_revision_idempotent_and_old_observations_immutable(campaign):
    root, runs, state = campaign
    first = evidence().build_run_feedback(root, 'exp_003')
    old_bytes = Path(first['feedback_ref']).read_bytes()
    assert evidence().build_run_feedback(root, 'exp_003') == first
    csv_rows(runs / 'exp_003/dev/per_image.csv', [{'sample_id': 's0', 'psnr': 28, 'ssim': 0.9}])
    state['trials'][-1]['dev_metrics']['num_images'] = 1
    state['trials'][-1]['dev_psnr'] = 28
    dump(root / 'campaign.json', state)
    second = evidence().build_run_feedback(root, 'exp_003')
    assert second['feedback_revision'] == 'r002'
    assert Path(first['feedback_ref']).read_bytes() == old_bytes
    assert second['comparisons']['comparison']['overall']['scope'] == 'matched_subset'
    assert second['observations'][0]['evidence_ref'].startswith('fixture/exp_003/r002/')


def test_scientific_diff_keeps_data_initialization_and_derived_scheduler():
    changes = evidence().diff_scientific_config(
        {'validation': {'input': '/new'}, 'init_checkpoint': '/new.pth', 'learning_rate': 0.1,
         'output_dir': '/new/out', 'logs_dir': '/new/log'},
        {'validation': {'input': '/old'}, 'init_checkpoint': '/old.pth', 'learning_rate': 0.2,
         'output_dir': '/old/out', 'logs_dir': '/old/log'})
    assert changes['validation.input'] == {'from': '/old', 'to': '/new'}
    assert 'init_checkpoint' in changes and 'output_dir' not in changes
    assert changes['scheduler.eta_min']['to'] == 0.001


def test_full_config_incompatibility_and_supervision_limit(campaign):
    root, _, state = campaign
    state['trials'][-1]['resolved_config']['seed'] = 2
    state['trials'][-1]['resolved_config']['loss_family'] = 'mse'
    dump(root / 'campaign.json', state)
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['comparisons']['comparison']['comparable'] is False
    assert 'seed' in result['comparisons']['comparison']['config_delta']
    assert any('supervision' in limit for limit in result['interpretation_limits'])
    assert result['quality']['valid'] is True


def test_missing_legacy_csv_not_score_failure(campaign):
    root, runs, _ = campaign
    (runs / 'exp_003/dev/per_image.csv').unlink()
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['diagnostics_status'] == 'unavailable'
    assert result['quality']['valid'] is True
    assert result['overall']['dev_psnr'] is not None
    assert result['proposal_ready'] is True


def test_required_sections_baseline_and_failed_are_conditional(campaign):
    root, runs, state = campaign
    state['control'] = {'observation_config': {'required_for_proposal':
                        ['sample_alignment', 'config_diff', 'paired_comparison', 'text_summary']}}
    dump(root / 'campaign.json', state)
    baseline = evidence().build_run_feedback(root, 'exp_001')
    assert baseline['comparisons']['comparison']['status'] == 'not_applicable'
    assert baseline['actual_changes_status'] == 'not_applicable'
    assert baseline['proposal_ready'] is True
    (runs / 'exp_003/dev/per_image.csv').unlink()
    assert evidence().build_run_feedback(root, 'exp_003')['proposal_ready'] is False
    state['trials'][-1].update(status='failed', valid=False, dev_psnr=None, error='worker failed')
    dump(root / 'campaign.json', state)
    failed = evidence().build_run_feedback(root, 'exp_003')
    assert failed['proposal_ready'] is True
    assert failed['overall']['dev_psnr'] is None
    assert failed['execution']['error'] == 'worker failed'


def test_cases_deduplicate_and_limited_history(campaign):
    root, _, _ = campaign
    result = evidence().build_run_feedback(root, 'exp_002')
    detail = evidence().load_feedback_detail(Path(result['feedback_ref']).parent, 'cases')
    samples = [card['sample_id'] for card in detail['cases']]
    assert len(samples) == len(set(samples)) <= 8
    assert detail['persistent_poor_status'] == 'evidence_limited'
    assert detail['history_window'] == ['exp_001', 'exp_002']
    assert not any('persistent_poor' in card['selection_reasons'] for card in detail['cases'])


def test_curves_only_existing_numbers_and_detail_access(campaign):
    root, runs, _ = campaign
    dump(runs / 'exp_003/train/logs/exp_003.json',
         {'val_mean_psnr': [20, 22, 21], 'checkpoint_model_name': ['exp_003_1.pth', 'exp_003_2.pth', 'exp_003_3.pth']})
    result = evidence().build_run_feedback(root, 'exp_003')
    curves = evidence().load_feedback_detail(Path(result['feedback_ref']).parent, 'curves')
    assert curves['val_mean_psnr'] == [20, 22, 21]
    assert curves['best_epoch'] == 2
    assert curves['train_validation_gap'] is None
    with pytest.raises(ValueError):
        evidence().load_feedback_detail(Path(result['feedback_ref']).parent, '../campaign')


def test_duplicate_basename_manifest_and_explicit_invalid_reference(campaign, tmp_path):
    root, _, state = campaign
    dump(tmp_path / 'manifest.json', {'samples': [
        {'sample_id': 'a', 'input_relpath': 'a/image.png', 'split': 'DEV'},
        {'sample_id': 'b', 'input_relpath': 'b/image.png', 'split': 'DEV'}]})
    manifest = evidence().load_manifest(tmp_path / 'manifest.json')
    assert manifest['ambiguous_basenames'] == ['image.png']
    with pytest.raises(ValueError, match='comparison'):
        evidence().resolve_trial_references(state, {'based_on': {'run_id': 'exp_001'}},
                                           {'comparison_run_id': 'missing'})


def test_duplicate_ids_and_nonfinite_diagnostics_fail_without_invalidating_score(campaign):
    root, runs, _ = campaign
    csv_rows(runs / 'exp_003/dev/per_image.csv',
             [{'sample_id': 'same', 'psnr': 20}, {'sample_id': 'same', 'psnr': 21}])
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['diagnostics_status'] == 'failed'
    assert result['quality']['valid'] is True
    assert any('duplicate' in reason.lower() for reason in result['unavailable'])


def test_mismatched_csv_score_is_diagnostic_failure_not_fake_measurement(campaign):
    root, _, state = campaign
    state['trials'][-1]['dev_psnr'] = 99
    dump(root / 'campaign.json', state)
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['quality']['valid'] is True
    assert result['overall']['dev_psnr'] == 99
    assert result['comparisons']['comparison']['status'] == 'unavailable'
    assert not any(obs['metric'] == 'mean_paired_psnr_delta_db' for obs in result['observations'])
    assert any('aggregate' in value for value in result['unavailable'])


def test_roi_macro_same_mask_support_and_null_regions(campaign):
    root, runs, state = campaign
    fields = {'sample_id': 's0', 'roi_id': 'low_code', 'metric_name': 'mse', 'value': 0.1,
              'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'mask1',
              'profile_revision': 'p001', 'valid_reason': 'valid'}
    reference = [fields, {**fields, 'sample_id': 's1', 'value': 0.3, 'n_pixels': 30},
                 {**fields, 'sample_id': 's2', 'value': '', 'n_pixels': 0, 'valid_reason': 'empty'},
                 {**fields, 'sample_id': 's3', 'value': 0.5}]
    current = [{**fields, 'value': 0.05}, {**fields, 'sample_id': 's1', 'value': 0.1, 'n_pixels': 30},
               {**fields, 'sample_id': 's2', 'value': '', 'n_pixels': 0, 'valid_reason': 'empty'},
               {**fields, 'sample_id': 's3', 'value': 0.2, 'effective_mask_id': 'different'}]
    csv_rows(runs / 'exp_001/dev/roi_metrics.csv', reference)
    csv_rows(runs / 'exp_003/dev/roi_metrics.csv', current)
    roi_snapshot(root, runs, state)
    result = evidence().build_run_feedback(root, 'exp_003')
    roi = result['comparisons']['comparison']['regions']['low_code/mse']
    assert roi['n_images'] == 2
    assert roi['macro']['delta'] == pytest.approx(-0.125)
    assert roi['micro']['reference'] == pytest.approx(0.25)
    assert roi['micro']['candidate'] == pytest.approx(0.0875)
    assert roi['coverage']['incompatible_support'] == ['s3']
    assert roi['coverage']['unavailable_value'] == ['s2']
    assert any(observation['roi'] == 'low_code' for observation in result['observations'])


def test_baseline_and_failed_have_revision_scoped_factual_observations(campaign):
    root, _, state = campaign
    baseline = evidence().build_run_feedback(root, 'exp_001')
    assert baseline['observations'][0]['metric'] == 'mean_per_image_rgb_psnr'
    assert baseline['observations'][0]['comparison_run_id'] is None
    assert baseline['observations'][0]['value'] == state['trials'][0]['dev_psnr']
    state['trials'][-1].update(status='failed', valid=False, dev_psnr=None, error='worker crashed')
    dump(root / 'campaign.json', state)
    failed = evidence().build_run_feedback(root, 'exp_003')
    assert failed['observations'][0]['metric'] == 'execution_status'
    assert failed['observations'][0]['value'] is None
    assert 'worker crashed' in failed['observations'][0]['finding']


def test_controller_saves_do_not_generate_diagnostic_revisions(campaign):
    root, _, state = campaign
    first = evidence().build_run_feedback(root, 'exp_003')
    state['trials'][-1].update(feedback_ref=first['feedback_ref'], feedback_revision='r001',
                             diagnostics_status=first['diagnostics_status'], proposal_ready=first['proposal_ready'])
    state['best']['run_id'] = 'exp_001'
    dump(root / 'campaign.json', state)
    assert evidence().build_run_feedback(root, 'exp_003')['feedback_revision'] == 'r001'


def test_explicit_reference_checks_frozen_evaluation_protocol(campaign):
    _, _, state = campaign
    state['trials'][0]['dev_metrics']['protocol'] = 'P256'
    with pytest.raises(ValueError, match='protocol'):
        evidence().resolve_trial_references(state, {}, {'comparison_run_id': 'exp_001'})


def test_baseline_missing_frozen_samples_blocks_required_alignment(campaign):
    root, _, state = campaign
    dump(root / 'profile.json', {'samples': [
        {'sample_id': f's{i}', 'split': 'DEV', 'tags': ['measured'], 'scene_id': 'unknown'} for i in range(9)]})
    state['profile_ref'] = 'profile.json'
    state['control'] = {'observation_config': {'required_for_proposal': ['sample_alignment']}}
    dump(root / 'campaign.json', state)
    baseline = evidence().build_run_feedback(root, 'exp_001')
    assert baseline['proposal_ready'] is False
    assert baseline['section_status']['sample_alignment'] is False


def test_legacy_basename_mapping_uses_relative_path_when_names_collide(campaign):
    root, runs, state = campaign
    dump(root / 'profile.json', {'samples': [
        {'sample_id': 's0', 'split': 'DEV', 'input_relpath': 'a/image.png'},
        {'sample_id': 's1', 'split': 'DEV', 'input_relpath': 'b/image.png'}]})
    state['profile_ref'] = 'profile.json'
    state['trials'][0]['dev_metrics']['num_images'] = 2
    state['trials'][0]['dev_psnr'] = 20.5
    dump(root / 'campaign.json', state)
    csv_rows(runs / 'exp_001/dev/per_image.csv',
             [{'image': 'image.png', 'input_relpath': 'b/image.png', 'psnr': 21},
              {'image': 'image.png', 'input_relpath': 'a/image.png', 'psnr': 20}])
    result = evidence().build_run_feedback(root, 'exp_001')
    assert result['section_status']['sample_alignment'] is True
    assert {card['sample_id'] for card in result['cases']['cases']} == {'s0', 's1'}


def test_persistent_poor_uses_three_distinct_recipes_and_preserves_dedup_reason(campaign):
    root, _, _ = campaign
    result = evidence().build_run_feedback(root, 'exp_003')
    cases = result['cases']
    assert cases['persistent_poor_status'] == 'available'
    poor = next(card for card in cases['cases'] if card['sample_id'] == 's0')
    assert 'persistent_poor' in poor['selection_reasons']
    assert 'regression' in poor['selection_reasons']
    assert len([card for card in cases['cases'] if card['sample_id'] == 's0']) == 1


def test_nonfinite_csv_is_not_exposed_as_measurement(campaign):
    root, runs, _ = campaign
    csv_rows(runs / 'exp_003/dev/per_image.csv', [{'sample_id': 's0', 'psnr': 'nan'}])
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['diagnostics_status'] == 'failed'
    assert result['quality']['valid'] is True
    assert not any(obs['metric'] == 'mean_paired_psnr_delta_db' for obs in result['observations'])
    assert result['proposal_ready'] is True


def test_valid_nonbaseline_missing_csv_keeps_native_observation_for_ack(campaign):
    root, runs, state = campaign
    (runs / 'exp_003/dev/per_image.csv').unlink()
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['observations'][0]['metric'] == 'mean_per_image_rgb_psnr'
    assert result['observations'][0]['value'] == state['trials'][-1]['dev_psnr']
    assert result['observations'][0]['comparison_run_id'] is None
    assert not any(obs['metric'] == 'mean_paired_psnr_delta_db' for obs in result['observations'])


def roi_snapshot(root, runs, state, *, sigma=1.0, changed_run=None):
    """Readable fixed measurement inventory shared by both actual run artifacts."""
    profile_ref = root / 'roi_profile_parent.json'
    profile = {'profile_ref': str(profile_ref), 'profile_revision': 'p001',
               'source': 'pretrial_fixture', 'transform': {'eval_size': 512}, 'rules': {},
               'split_roots': {'DEV': {'input_dir': '/dev', 'gt_dir': '/gt', 'metadata_dir': '/meta'}},
               'samples': [{'sample_id': f's{i}', 'split': 'DEV'} for i in range(8)]}
    dump(profile_ref, profile)
    state['profile_ref'] = str(profile_ref)
    dump(root / 'campaign.json', state)
    inventory = [{'sample_id': f's{i}', 'roi_id': 'low_code', 'metric_name': 'mse',
                  'effective_mask_id': 'mask1'} for i in range(8)]
    snapshot = {**profile, 'roi_profile_ref': str(root / 'frozen_roi.json'),
                'metric_config': {'metrics': ['mse'], 'lowpass_sigma': sigma},
                'roi_samples': [{'sample_id': f's{i}', 'mask_definitions': {'low_code': {'row_spans': [[0, 0, 10]]}}} for i in range(8)],
                'roi_inventory': inventory}
    for run in ('exp_001', 'exp_003'):
        value = {**snapshot}
        if changed_run == run:
            value['metric_config'] = {**value['metric_config'], 'lowpass_sigma': 5.0}
        dump(runs / run / 'dev/diagnostics_snapshot.json', value)
    for run in ('exp_001', 'exp_003'):
        path = runs / run / 'dev/roi_metrics.csv'
        if path.is_file():
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                csv_rows(path, [{**row, 'metric_definition': json.dumps(importlib.import_module('tm_research.diagnostics').fixed_metric_definition(snapshot['metric_config'], 'mse'))} for row in rows])
    return snapshot


def test_incompatible_protocol_suspends_pairs_and_required_chapter(campaign):
    root, _, state = campaign
    state['control'] = {'observation_config': {'required_for_proposal': ['paired_comparison']}}
    state['trials'][-1]['dev_metrics']['protocol'] = 'P256'
    dump(root / 'campaign.json', state)
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['quality']['valid'] is True
    assert result['proposal_ready'] is False
    assert result['comparisons']['comparison']['status'] == 'unavailable'
    assert result['comparisons']['comparison'].get('paired_rows', []) == []
    assert not any(obs['comparison_run_id'] is not None for obs in result['observations'])


def test_baseline_groups_are_not_applicable_instead_of_permanent_hold(campaign):
    root, _, state = campaign
    dump(root / 'profile.json', {'samples': [{'sample_id': f's{i}', 'split': 'DEV', 'tags': ['fixed']} for i in range(8)]})
    state.update(profile_ref='profile.json', control={'observation_config': {'required_for_proposal': ['groups']}})
    dump(root / 'campaign.json', state)
    result = evidence().build_run_feedback(root, 'exp_001')
    assert result['proposal_ready'] is True
    assert result['section_applicability']['groups'] == 'not_applicable'


def test_missing_revision_detail_repairs_new_revision_and_missing_read_is_explicit(campaign):
    root, _, _ = campaign
    first = evidence().build_run_feedback(root, 'exp_003')
    path = Path(first['detail_refs']['paired_comparison'])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        evidence().load_feedback_detail(path.parent, 'paired_comparison')
    repaired = evidence().build_run_feedback(root, 'exp_003')
    assert repaired['feedback_revision'] == 'r002'
    assert Path(repaired['detail_refs']['paired_comparison']).is_file()
    assert Path(first['feedback_ref']).is_file()
    assert repaired['repair_of_revision'] == 'r001'
    assert evidence().validate_feedback_access(repaired, repaired['observations'][1]['id'])['available']


def test_roi_different_measurement_definitions_are_unavailable(campaign):
    root, runs, state = campaign
    fields = {'sample_id': 's0', 'roi_id': 'low_code', 'metric_name': 'mse', 'value': 0.1,
              'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'mask1',
              'profile_revision': 'p001', 'valid_reason': 'valid'}
    for run in ('exp_001', 'exp_003'):
        csv_rows(runs / run / 'dev/roi_metrics.csv', [fields])
    roi_snapshot(root, runs, state, changed_run='exp_003')
    result = evidence().build_run_feedback(root, 'exp_003')
    comparison = result['comparisons']['comparison']
    assert comparison['regions_status'] == 'unavailable'
    assert 'measurement' in comparison['regions_reason']
    assert not any(obs['roi'] is not None for obs in result['observations'])


def test_roi_expected_inventory_counts_missing_both_and_empty_files(campaign):
    root, runs, state = campaign
    fields = {'sample_id': 's0', 'roi_id': 'low_code', 'metric_name': 'mse', 'value': 0.1,
              'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'mask1',
              'profile_revision': 'p001', 'valid_reason': 'valid'}
    for run in ('exp_001', 'exp_003'):
        csv_rows(runs / run / 'dev/roi_metrics.csv', [fields])
    roi_snapshot(root, runs, state)
    result = evidence().build_run_feedback(root, 'exp_003')
    region = result['comparisons']['comparison']['regions']['low_code/mse']
    assert region['coverage']['expected_count'] == 8
    assert region['coverage']['missing_both'] == [f's{i}' for i in range(1, 8)]
    assert region['scope'] == 'matched_subset'
    for run in ('exp_001', 'exp_003'):
        path = runs / run / 'dev/roi_metrics.csv'
        path.write_text(path.read_text().splitlines()[0] + '\n')
    empty = evidence().build_run_feedback(root, 'exp_003')['comparisons']['comparison']
    assert empty['regions_status'] == 'unavailable'
    assert empty['regions']['low_code/mse']['macro']['delta'] is None


def test_unverifiable_csv_only_roi_has_no_same_support_claim(campaign):
    root, runs, _ = campaign
    row = {'sample_id': 's0', 'roi_id': 'all', 'metric_name': 'mse', 'value': 0.1,
           'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'p001:all:erode0',
           'profile_revision': 'p001', 'valid_reason': 'ok'}
    for run in ('exp_001', 'exp_003'):
        csv_rows(runs / run / 'dev/roi_metrics.csv', [row])
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['comparisons']['comparison']['regions_status'] == 'unavailable'
    assert not any(obs['roi'] is not None for obs in result['observations'])


def test_recent_distinct_recipe_window_uses_latest_repeat(campaign):
    root, runs, state = campaign
    repeated = json.loads(json.dumps(state['trials'][0]))
    repeated.update(run_id='exp_004', index=3, references={'baseline_run_id': 'exp_001',
                    'comparison_run_id': 'exp_003', 'best_before_run_id': 'exp_003',
                    'construction_base_run_id': 'exp_003'})
    state['trials'].append(repeated)
    dump(root / 'campaign.json', state)
    (runs / 'exp_004/dev').mkdir(parents=True)
    (runs / 'exp_004/dev/per_image.csv').write_text((runs / 'exp_001/dev/per_image.csv').read_text())
    result = evidence().build_run_feedback(root, 'exp_004')
    assert result['cases']['history_window'] == ['exp_002', 'exp_003', 'exp_004']


def test_legacy_postprocessing_marks_profile_retrospective(campaign):
    root, _, state = campaign
    dump(root / 'profile.json', {'samples': [{'sample_id': f's{i}', 'split': 'DEV'} for i in range(8)],
                                 'rules': {'retrospective': False}})
    state['profile_ref'] = 'profile.json'
    dump(root / 'campaign.json', state)
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['provenance']['retrospective'] is True
    assert result['provenance']['profile_available_at_run'] is False
    assert result['observations'][1]['retrospective'] is True


def test_case_region_changes_and_split_root_refs_are_available(campaign):
    root, runs, state = campaign
    fields = {'sample_id': 's0', 'roi_id': 'low_code', 'metric_name': 'mse', 'value': 0.1,
              'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'mask1',
              'profile_revision': 'p001', 'valid_reason': 'valid'}
    csv_rows(runs / 'exp_001/dev/roi_metrics.csv', [fields])
    csv_rows(runs / 'exp_003/dev/roi_metrics.csv', [{**fields, 'value': 0.05}])
    snapshot = roi_snapshot(root, runs, state)
    profile = json.loads(Path(state['profile_ref']).read_text())
    profile['samples'][0].update(input_relpath='input.png', gt_relpath='target.png')
    dump(Path(state['profile_ref']), profile)
    # Keep actual recorded provenance in both snapshots coherent with the fixture profile.
    for run in ('exp_001', 'exp_003'):
        path = runs / run / 'dev/diagnostics_snapshot.json'
        value = json.loads(path.read_text()); value['samples'] = profile['samples']; dump(path, value)
    result = evidence().build_run_feedback(root, 'exp_003')
    case = next(case for case in result['cases']['cases'] if case['sample_id'] == 's0')
    assert case['region_changes'][0]['delta'] == pytest.approx(-0.05)
    assert case['input_ref'] == '/dev/input.png'
    assert case['gt_ref'] == '/gt/target.png'
    details = evidence().load_feedback_detail(Path(result['feedback_ref']).parent, 'details_index')
    assert details['split_roots'] == snapshot['split_roots']


def test_observation_access_validates_pointer_and_remote_mapping(campaign):
    root, _, _ = campaign
    result = evidence().build_run_feedback(root, 'exp_003')
    summary_path = Path(result['feedback_ref'])
    altered = json.loads(summary_path.read_text())
    altered['observations'][1]['source'] = 'paired_comparison.json#/comparison/nonexistent'
    dump(summary_path, altered)
    access = evidence().validate_feedback_access(result, altered['observations'][1]['id'])
    assert access['available'] is False
    assert any('source unavailable' in value for value in access['unavailable'])
    dump(summary_path, result)
    remote = {**result, 'feedback_ref': '/unmounted/root/' + str(summary_path.relative_to(root)),
              'artifact_access': {'artifact_root_mapping': {'/unmounted/root': str(root)}}}
    assert evidence().validate_feedback_access(remote, result['observations'][0]['id'])['available']
    remote['artifact_access']['artifact_root_mapping'] = {'/unmounted/root': str(root / 'missing')}
    assert evidence().validate_feedback_access(remote)['available'] is False


def test_same_area_different_frozen_mask_geometry_is_not_comparable(campaign):
    root, runs, state = campaign
    row = {'sample_id': 's0', 'roi_id': 'low_code', 'metric_name': 'mse', 'value': 0.1,
           'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'mask1',
           'profile_revision': 'p001', 'valid_reason': 'ok'}
    for run in ('exp_001', 'exp_003'):
        csv_rows(runs / run / 'dev/roi_metrics.csv', [row])
    roi_snapshot(root, runs, state)
    path = runs / 'exp_003/dev/diagnostics_snapshot.json'
    snapshot = json.loads(path.read_text())
    snapshot['roi_samples'][0]['mask_definitions']['low_code']['row_spans'] = [[1, 0, 10]]
    dump(path, snapshot)
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['comparisons']['comparison']['regions_status'] == 'unavailable'
    assert not any(obs['roi'] is not None for obs in result['observations'])


def test_roi_row_definition_must_match_frozen_measurement_config(campaign):
    root, runs, state = campaign
    row = {'sample_id': 's0', 'roi_id': 'low_code', 'metric_name': 'mse', 'value': 0.1,
           'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'mask1',
           'profile_revision': 'p001', 'valid_reason': 'ok'}
    for run in ('exp_001', 'exp_003'):
        csv_rows(runs / run / 'dev/roi_metrics.csv', [row])
    roi_snapshot(root, runs, state)
    for run in ('exp_001', 'exp_003'):
        path = runs / run / 'dev/roi_metrics.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        definition = json.loads(rows[0]['metric_definition'])
        definition['data_range'] = 99
        csv_rows(path, [{**rows[0], 'metric_definition': json.dumps(definition)}])
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['comparisons']['comparison']['regions_status'] == 'unavailable'
    assert not any(obs['roi'] is not None for obs in result['observations'])


def test_minimal_fallback_is_native_readable_idempotent_and_optional_ready(campaign):
    root, _, state = campaign
    state['control'] = {'observation_config': {'required_for_proposal': []}}
    dump(root / 'campaign.json', state)
    builder = getattr(evidence(), 'build_fallback_feedback', None)
    assert builder is not None, 'isolated fallback feedback API is missing'
    first = builder(root, 'exp_003', ValueError('optional profile malformed'))
    assert first['proposal_ready'] is True
    assert first['diagnostics_status'] == 'unavailable'
    assert first['assembly_error'] == 'optional profile malformed'
    assert first['quality']['valid'] is True
    assert first['observations'][0]['value'] == state['trials'][-1]['dev_psnr']
    assert first['observations'][0]['metric'] == 'mean_per_image_rgb_psnr'
    assert first['best_before_run_id'] == 'exp_002'
    assert first['comparison_run_id'] == 'exp_001'
    assert first['resolved_config'] == state['trials'][-1]['resolved_config']
    assert evidence().validate_feedback_access(first, 'O1')['available'] is True
    assert builder(root, 'exp_003', 'optional profile malformed') == first
    assert len(list((root / 'feedback/exp_003').glob('r[0-9]*'))) == 1
    repaired = evidence().build_run_feedback(root, 'exp_003')
    assert repaired['feedback_revision'] == 'r002'
    assert Path(first['feedback_ref']).is_file()


def test_fallback_gates_true_required_chapters_and_preserves_old_revision(campaign):
    root, _, state = campaign
    state['control'] = {'observation_config': {'required_for_proposal': ['sample_alignment', 'paired_comparison']}}
    dump(root / 'campaign.json', state)
    original = evidence().build_run_feedback(root, 'exp_003')
    original_bytes = Path(original['feedback_ref']).read_bytes()
    builder = getattr(evidence(), 'build_fallback_feedback', None)
    assert builder is not None, 'isolated fallback feedback API is missing'
    failed = builder(root, 'exp_003', 'diagnostic assembly failed')
    assert failed['feedback_revision'] == 'r002'
    assert failed['proposal_ready'] is False
    assert failed['missing_required'] == ['sample_alignment', 'paired_comparison']
    assert Path(original['feedback_ref']).read_bytes() == original_bytes
    assert evidence().validate_feedback_access(failed)['available'] is True


def test_malformed_optional_roi_definition_preserves_full_image_evidence(campaign):
    root, runs, state = campaign
    fields = {'sample_id': 's0', 'roi_id': 'low_code', 'metric_name': 'mse', 'value': 0.1,
              'n_pixels': 10, 'region_fraction': 0.1, 'effective_mask_id': 'mask1',
              'profile_revision': 'p001', 'valid_reason': 'valid'}
    for run in ('exp_001', 'exp_003'):
        csv_rows(runs / run / 'dev/roi_metrics.csv', [fields])
    roi_snapshot(root, runs, state)
    path = runs / 'exp_003/dev/roi_metrics.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    csv_rows(path, [{**rows[0], 'metric_definition': '{malformed json'}])
    result = evidence().build_run_feedback(root, 'exp_003')
    assert result['comparisons']['comparison']['status'] == 'complete'
    assert result['comparisons']['comparison']['regions_status'] == 'unavailable'
    assert any(obs['metric'] == 'mean_paired_psnr_delta_db' for obs in result['observations'])
    assert not any(obs['roi'] is not None for obs in result['observations'])
    assert result['quality']['valid'] is True


@pytest.mark.parametrize('cache,contents', [
    ('source_snapshot.json', '{truncated'),
    ('source_snapshot.json', '[]'),
    ('source_snapshot.json', None),
    ('latest.json', '{truncated'),
    ('latest.json', '[]'),
    ('latest.json', '{"feedback_revision": 7}'),
    ('latest.json', '{"feedback_revision": "../escape", "run_id": "exp_003"}'),
])
def test_fallback_damaged_revision_cache_is_nonreusable_and_full_recovers(campaign, cache, contents):
    root, _, state = campaign
    first = evidence().build_run_feedback(root, 'exp_003')
    old_summary = Path(first['feedback_ref']).read_bytes()
    cache_path = (Path(first['feedback_ref']).parent / cache if cache == 'source_snapshot.json'
                  else root / 'feedback/exp_003/latest.json')
    if contents is None:
        cache_path.unlink()
    else:
        cache_path.write_text(contents)
    fallback = evidence().build_fallback_feedback(root, 'exp_003', 'cannot parse prior diagnostic source snapshot')
    assert fallback['feedback_revision'] == 'r002'
    assert fallback['proposal_ready'] is True
    assert fallback['quality']['valid'] is True
    assert fallback['observations'][0]['value'] == state['trials'][-1]['dev_psnr']
    assert 'cache' in fallback['repair_reason']
    assert evidence().validate_feedback_access(fallback, 'O1')['available']
    assert Path(first['feedback_ref']).read_bytes() == old_summary
    assert evidence().build_fallback_feedback(root, 'exp_003', 'cannot parse prior diagnostic source snapshot') == fallback
    recovered = evidence().build_run_feedback(root, 'exp_003')
    assert recovered['feedback_revision'] == 'r003'
    assert evidence().validate_feedback_access(recovered)['available']
    assert Path(first['feedback_ref']).read_bytes() == old_summary


def test_fallback_does_not_mask_invalid_authoritative_campaign(campaign):
    root, _, _ = campaign
    (root / 'campaign.json').write_text('{invalid campaign')
    with pytest.raises(json.JSONDecodeError):
        evidence().build_fallback_feedback(root, 'exp_003', 'diagnostic parser failed')
