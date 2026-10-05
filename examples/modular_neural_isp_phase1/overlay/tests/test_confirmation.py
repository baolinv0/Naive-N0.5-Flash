"""Frozen confirmation contract exercised through small real train/reload workers."""
import json
from pathlib import Path

import pytest

from test_runner import config
from test_control import authorized_control
from tm_research import runner


def setup_confirmation(tmp_path, *, independent=False):
    from tm_research import confirmation
    cfg = config(tmp_path)
    evaluator = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    script = evaluator.read_text().replace("'image,psnr\\na,20\\n'", "'image,sample_id,input_relpath,psnr\\na.png,,a.png,'+str(score)+'\\n'")
    evaluator.write_text(script)
    baseline_id = runner.run_baseline(cfg)
    baseline = runner.wait_for_result(baseline_id, cfg['runs_dir'], timeout=15)
    winner_cfg = {**cfg, 'loss_family': 'mse'}
    winner_id = runner.run_baseline(winner_cfg)
    winner = runner.wait_for_result(winner_id, cfg['runs_dir'], timeout=15)
    search = tmp_path / 'search'; search.mkdir()
    state = {'status': 'finalized', 'active': None, 'config': cfg,
             'trials': [{'run_id': baseline_id, 'valid': True, 'recipe': baseline['recipe']},
                        {'run_id': winner_id, 'valid': True, 'recipe': winner['recipe']}],
             'best': {'run_id': winner_id, 'recipe': winner['recipe'], 'dev_psnr': 22.0},
             'frozen': {'run_id': winner_id, 'recipe': winner['recipe']}}
    (search / 'campaign.json').write_text(json.dumps(state))
    scope = {'authorized': True, 'kind': 'original_dev', 'seeds': [3, 9], 'replicates': 1}
    if independent:
        split = {}
        for name in ('input_dir', 'gt_dir'):
            directory = tmp_path / 'confirm_data' / name; directory.mkdir(parents=True)
            split[name] = str(directory)
        (Path(split['input_dir']) / 'a.png').write_text('confirmation')
        scope.update(kind='independent', split_spec=split)
    control = authorized_control()
    control['protocol_id'] = 'P512'
    control['limits'].update(job_walltime_seconds=15, total_gpu_hours=0,
                             allocated_gpus=0, confirmation_gpu_hours=0, calibration_gpu_hours=0)
    control['confirmation_scope'] = scope
    control_path = tmp_path / 'control.json'; control_path.write_text(json.dumps(control))
    plan = {'seeds': [3, 9], 'replicates': 1, 'budget_gpu_hours': 0,
            'delta_useful_db': 1.0, 'scope': {k: v for k, v in scope.items() if k not in ('authorized','seeds','replicates')}}
    destination = tmp_path / 'confirmation'
    return confirmation, search, destination, plan, control_path


def finish(module, destination, count=4):
    for _ in range(count):
        module.run_confirmation_next(destination)
    return module.report_confirmation(destination)


def test_confirmation_allows_recipe_across_seeds(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    state = m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert state['launch_allowed']
    report = finish(m, dest)
    assert [(p['seed'], p['replicate'], p['delta_db']) for p in report['pairs']] == [(3, 1, 2.0), (9, 1, 2.0)]
    assert report['conclusion'] == 'supported_in_scope'
    assert report['adaptive_dev_bias'] is True
    assert report['statistical_significance'] is False
    for task in report['tasks']:
        cfg = json.loads((Path(task['run_dir']) / 'config.json').read_text())
        assert cfg['seed'] in (3, 9)
        assert cfg['validation']['input_dir'] == str(tmp_path / 'validation' / 'input_dir')


def test_confirmation_is_idempotent_and_does_not_mutate_search_best(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    before = json.loads((search / 'campaign.json').read_text())
    original = m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert m.initialize_confirmation(search, dest, plan, control_ref=control) == original
    first = finish(m, dest)
    run_ids = [task['run_id'] for task in first['tasks']]
    m.run_confirmation_next(dest)
    again = m.report_confirmation(dest)
    assert [t['run_id'] for t in again['tasks']] == run_ids
    assert len(list((dest / 'runs').glob('exp_*'))) == 4
    after = json.loads((search / 'campaign.json').read_text())
    after.pop('confirmation_refs', None)
    primary = after.pop('primary_confirmation')
    assert primary['association'] == original['association']
    assert primary['plan'] == original['plan']
    assert after == before


def test_confirmation_never_selects_checkpoint_from_best_seed(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    frozen_id = json.loads((search / 'campaign.json').read_text())['frozen']['run_id']
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert finish(m, dest)['final_checkpoint']['run_id'] == frozen_id
    plan['final_checkpoint_rule'] = {'kind': 'best_confirmation_seed'}
    with pytest.raises(ValueError, match='checkpoint'):
        m.initialize_confirmation(search, tmp_path / 'bad-selection', plan, control_ref=control)


def test_test_split_not_relabelled(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    test = json.loads((search / 'campaign.json').read_text())['config']['test']
    alias = tmp_path / 'innocently_named'; alias.symlink_to(test['input_dir'], target_is_directory=True)
    split = {**test, 'input_dir': str(alias)}
    plan['scope'] = {'kind': 'independent', 'split_spec': split}
    auth = json.loads(control.read_text()); auth['confirmation_scope'].update(kind='independent',split_spec=split)
    control.write_text(json.dumps(auth))
    with pytest.raises(ValueError, match='TEST'):
        m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert not (dest / 'runs').exists()


def test_independent_split_only_after_original_dev_checkpoint_freeze(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    assert report['conclusion'] == 'supported_in_scope' and not report['adaptive_dev_bias']
    for task in report['tasks']:
        assert task['checkpoint']['selection_split'] == 'original_dev'
        cfg = json.loads((Path(task['run_dir']) / 'config.json').read_text())
        assert cfg['validation']['input_dir'] == str(tmp_path / 'validation' / 'input_dir')
        assert task['evaluation']['valid']
        assert (Path(task['run_dir']) / 'confirmation_eval' / 'metrics.json').is_file()


def test_failed_pair_retained_without_retry_or_seed_removal(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    train = tmp_path / 'upstream' / 'photofinishing' / 'train.py'
    source = train.read_text(); train.write_text('import sys\nsys.exit(7)\n')
    m.run_confirmation_next(dest)
    train.write_text(source)
    report = finish(m, dest, count=3)
    assert report['conclusion'] == 'inconclusive'
    assert len(report['pairs']) == 2 and report['pairs'][0]['status'] == 'incomplete'
    assert report['tasks'][0]['status'] == 'failed'
    m.run_confirmation_next(dest)
    assert len(list((dest / 'runs').glob('exp_*'))) == 4


@pytest.mark.parametrize('change', ['seed', 'budget', 'authorization'])
def test_unauthorized_plan_is_reviewable_without_start(tmp_path, change):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    if change == 'seed': plan['seeds'] = [666]
    if change == 'budget': plan['budget_gpu_hours'] = 2
    if change == 'authorization':
        auth = json.loads(control.read_text()); auth['confirmation_scope']['authorized'] = False
        control.write_text(json.dumps(auth))
    state = m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert not state['launch_allowed'] and state['status'] == 'blocked'
    assert m.run_confirmation_next(dest)['status'] == 'blocked'
    assert not (dest / 'runs').exists()


def test_missing_useful_threshold_is_descriptive_only(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    plan.pop('delta_useful_db')
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    assert report['conclusion'] == 'inconclusive' and report['descriptive_only']
    assert report['mean_delta_db'] == 2.0


def test_unknown_positive_gpu_search_cost_blocks_confirmation(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    auth = json.loads(control.read_text())
    auth['limits'].update(allocated_gpus=1, total_gpu_hours=10, confirmation_gpu_hours=1)
    control.write_text(json.dumps(auth)); plan['budget_gpu_hours'] = 1
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    state = m.run_confirmation_next(dest)
    assert state['status'] == 'blocked_parent'
    assert not (dest / 'runs').exists()
    assert m.report_confirmation(dest)['usage']['search_gpu_hours'] is None


def test_inflight_unknown_liveness_preserves_task_and_blocks_new_start(tmp_path, monkeypatch):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    # A host boundary cannot be made cross-host in a local CPU fixture.
    monkeypatch.setattr(runner, 'inspect_run', lambda *a, **k: {'liveness': 'unknown'})
    first = m.run_confirmation_next(dest, wait=False)
    again = m.run_confirmation_next(dest, wait=False)
    assert first['active_task_key'] == again['active_task_key'] == 'baseline:3:1'
    assert first['tasks'][0]['run_id'] == again['tasks'][0]['run_id']
    assert len(list((dest / 'runs').glob('exp_*'))) == 1
    assert all(t['status'] == 'pending' for t in again['tasks'][1:])


def test_predeclared_seed_checkpoint_rule_does_not_use_other_seed(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    plan['final_checkpoint_rule'] = {'kind': 'predeclared_seed', 'seed': 9, 'replicate': 1}
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    selected = report['final_checkpoint']['run_id']
    task = next(t for t in report['tasks'] if t['run_id'] == selected)
    assert (task['arm'], task['seed'], task['replicate']) == ('winner', 9, 1)


def test_crashed_independent_evaluation_never_automatically_retries(tmp_path, monkeypatch):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('controller interrupted at external evaluation boundary')
    monkeypatch.setattr(runner, 'evaluate_frozen_checkpoint', interrupted)
    with pytest.raises(KeyboardInterrupt):
        m.run_confirmation_next(dest)
    state = m.run_confirmation_next(dest)
    assert state['status'] == 'waiting_for_evaluation_recovery'
    assert len(list((dest / 'runs').glob('exp_*'))) == 1
    report = m.report_confirmation(dest)
    assert report['conclusion'] == 'inconclusive' and report['complete_pairs'] == 0


def test_report_scene_difference_uses_frozen_dev_profile(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'profile_revision': 'fixture-r1', 'samples': [
        {'sample_id': 'DEV/a', 'split': 'DEV', 'input_relpath': 'a.png', 'image': 'a.png', 'scene_id': 'scene-1', 'tags': ['dark']}]}))
    auth = json.loads(control.read_text())
    auth['observation_config']['profile_ref'] = str(profile)
    control.write_text(json.dumps(auth))
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    # Parent diagnostics annotations are frozen, not adopted after results.
    profile.write_text('{}')
    report = finish(m, dest)
    assert report['scenes']['status'] == 'available'
    assert report['scenes']['pairs'][0]['scenes'] == [
        {'scene_id': 'scene-1', 'paired_images': 1, 'mean_delta_db': 2.0}]
    assert report['scenes']['unit'] == 'paired per-image differences within each seed; seeds and images are not pooled'


@pytest.mark.parametrize('threshold,winner_loss,expected', [
    (2.0, 'mse', 'inconclusive'), (1.0, 'l1', 'not_supported')])
def test_predeclared_direction_rule_boundaries(tmp_path, threshold, winner_loss, expected):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    # Freeze a valid alternative recipe before confirmation; negative evidence is admissible.
    state = json.loads((search / 'campaign.json').read_text())
    if winner_loss != 'mse':
        cfg = {**state['config'], 'loss_family': winner_loss}
        run_id = runner.run_baseline(cfg)
        result = runner.wait_for_result(run_id, cfg['runs_dir'], timeout=15)
        state['trials'].append({'run_id': run_id, 'valid': True, 'recipe': result['recipe']})
        state['frozen'] = {'run_id': run_id, 'recipe': result['recipe']}
        (search / 'campaign.json').write_text(json.dumps(state))
    plan['delta_useful_db'] = threshold
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    assert report['conclusion'] == expected


def test_parent_protocol_hold_blocks_initialization_without_training(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    parent['research_hold'] = {'decision_id': 'unresolved_protocol'}
    (search / 'campaign.json').write_text(json.dumps(parent))
    state = m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert not state['launch_allowed']
    assert not (dest / 'runs').exists()


@pytest.mark.parametrize('field', ['campaign_id', 'protocol_id', 'source_revision', 'approved_config_ref'])
def test_supplied_control_cannot_change_frozen_parent_identity(tmp_path, field):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    parent['control'] = json.loads(control.read_text())
    (search / 'campaign.json').write_text(json.dumps(parent))
    supplied = json.loads(control.read_text()); supplied[field] = 'different-identity'
    control.write_text(json.dumps(supplied))
    state = m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert not state['launch_allowed']


@pytest.mark.parametrize('change', ['hold', 'config', 'final_test'])
def test_current_parent_rechecked_before_each_launch(tmp_path, change):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    m.run_confirmation_next(dest)
    parent = json.loads((search / 'campaign.json').read_text())
    if change == 'hold': parent['research_hold'] = {'decision_id': 'new_protocol_hold'}
    if change == 'config': parent['config']['epochs'] = 99
    if change == 'final_test': parent['final_test'] = {'status': 'running', 'reserved_gpu_hours': 0}
    (search / 'campaign.json').write_text(json.dumps(parent))
    state = m.run_confirmation_next(dest)
    assert state['status'] == 'blocked_parent'
    assert len(list((dest / 'runs').glob('exp_*'))) == 1


def test_external_confirmation_receipt_persists_frozen_association(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    state = m.initialize_confirmation(search, dest, plan, control_ref=control)
    parent = json.loads((search / 'campaign.json').read_text())
    receipt = parent['confirmation_refs'][0]
    assert receipt['confirmation_dir'] == str(dest)
    assert receipt['parent_campaign'] == str(search)
    assert receipt['frozen_run_id'] == parent['frozen']['run_id']
    assert receipt['baseline_run_id'] == parent['trials'][0]['run_id']
    assert receipt['protocol_id'] == 'P512'
    assert state['association'] == receipt
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert len(json.loads((search / 'campaign.json').read_text())['confirmation_refs']) == 1


def test_native_empty_sample_id_uses_frozen_inventory(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    assert report['pairs'][0]['per_image']['status'] == 'complete'
    assert report['pairs'][0]['per_image']['paired_count'] == 1
    assert report['pairs'][0]['per_image']['deltas'][0]['delta_db'] == 2


@pytest.mark.parametrize('defect', ['missing_both', 'aggregate_mismatch'])
def test_csv_diagnostics_reject_joint_missing_or_incongruent_rows(tmp_path, defect):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    if defect == 'missing_both':
        cfg = json.loads((search / 'campaign.json').read_text())['config']
        (Path(cfg['validation']['input_dir']) / 'b.png').write_text('fixture')
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    if defect == 'aggregate_mismatch':
        task = next(t for t in report['tasks'] if t['arm'] == 'winner')
        Path(task['result']['artifacts']['per_image_csv']).write_text('image,sample_id,input_relpath,psnr\na.png,,a.png,20\n')
        report = m.report_confirmation(dest)
    diagnostics = report['pairs'][0]['per_image']
    assert diagnostics['status'] in ('partial', 'incompatible')
    if defect == 'missing_both':
        assert len(diagnostics['missing_in_both']) == 1
        assert diagnostics['expected_count'] == 2
    else:
        assert 'aggregate' in ' '.join(diagnostics['errors'])
    assert report['scenes']['status'] == 'unavailable'


def test_late_final_test_charge_is_included_before_confirmation_launch(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    for trial in parent['trials']:
        trial['usage'] = {'gpu_hours': 0.0}  # billing boundary: explicit CPU search allocation
    (search / 'campaign.json').write_text(json.dumps(parent))
    auth = json.loads(control.read_text())
    auth['limits'].update(allocated_gpus=1, total_gpu_hours=1, confirmation_gpu_hours=0.5)
    control.write_text(json.dumps(auth)); plan['budget_gpu_hours'] = 0.5
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    parent = json.loads((search / 'campaign.json').read_text())
    parent['final_test'] = {'status': 'completed', 'valid': True, 'usage': {'gpu_hours': 1.0}}
    parent['usage'] = {'search_gpu_hours': 0.0, 'final_test_gpu_hours': 1.0, 'gpu_hours': 1.0}
    (search / 'campaign.json').write_text(json.dumps(parent))
    state = m.run_confirmation_next(dest)
    assert state['status'] == 'blocked_parent'
    assert state['parent_usage']['final_test_gpu_hours'] == 1.0
    assert not (dest / 'runs').exists()


def test_independent_authorized_profile_supplies_scene_annotations(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    profile = tmp_path / 'confirmation-profile.json'
    profile.write_text(json.dumps({'profile_revision': 'independent-r1', 'samples': [
        {'sample_id': 'CONFIRM/a', 'input_relpath': 'a.png', 'image': 'a.png', 'scene_id': 'confirm-scene', 'tags': []}]}))
    auth = json.loads(control.read_text()); auth['confirmation_scope']['profile_ref'] = str(profile)
    control.write_text(json.dumps(auth))
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    assert report['scenes']['status'] == 'available'
    assert report['scenes']['pairs'][0]['scenes'][0]['mean_delta_db'] == 2.0
    assert report['scenes']['profile_revision'] == 'independent-r1'


def test_recovery_collects_persisted_frozen_evaluation_without_retry(tmp_path, monkeypatch):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    evaluate = runner.evaluate_frozen_checkpoint
    def completed_then_interrupted(*args, **kwargs):
        evaluate(*args, **kwargs)
        raise KeyboardInterrupt('controller died after evaluator persisted terminal receipt')
    monkeypatch.setattr(runner, 'evaluate_frozen_checkpoint', completed_then_interrupted)
    with pytest.raises(KeyboardInterrupt):
        m.run_confirmation_next(dest)
    state = m.run_confirmation_next(dest)
    assert state['tasks'][0]['status'] == 'completed'
    assert state['active_task_key'] is None
    assert state['tasks'][0]['usage']['gpu_hours'] == 0
    assert len(list((dest / 'runs').glob('exp_*'))) == 1


def test_confirmation_prepare_interruption_recovers_same_run(tmp_path, monkeypatch):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    prepare = runner.prepare_run
    allocated = []
    def allocated_then_interrupted(config, **kwargs):
        allocated.append(prepare(config, **kwargs))
        raise RuntimeError('controller interrupted after prepare return')
    monkeypatch.setattr(runner, 'prepare_run', allocated_then_interrupted)
    with pytest.raises(RuntimeError, match='after prepare'):
        m.run_confirmation_next(dest)
    monkeypatch.setattr(runner, 'prepare_run', prepare)
    state = m.run_confirmation_next(dest)
    assert state['tasks'][0]['run_id'] == allocated[0]
    assert len(list((dest / 'runs').glob('exp_*'))) == 1


def test_dev_profile_mapping_excludes_train_aliases(tmp_path):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    profile = tmp_path / 'train-dev-profile.json'
    profile.write_text(json.dumps({'profile_revision': 'mixed-r1', 'samples': [
        {'sample_id': 'TRAIN/a', 'split': 'TRAIN', 'input_relpath': 'a.png', 'scene_id': 'training'},
        {'sample_id': 'DEV/a', 'split': 'DEV', 'input_relpath': 'a.png', 'scene_id': 'validation'}]}))
    auth = json.loads(control.read_text()); auth['observation_config']['profile_ref'] = str(profile)
    control.write_text(json.dumps(auth))
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    report = finish(m, dest)
    assert report['scenes']['pairs'][0]['scenes'][0]['scene_id'] == 'validation'


def test_incompatible_blocked_review_plan_does_not_poison_parent_ledger(tmp_path):
    from tm_research.control import reconcile_usage
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    approved = json.loads(control.read_text())
    parent['control'] = approved
    (search / 'campaign.json').write_text(json.dumps(parent))
    wrong = {**approved, 'campaign_id': 'unrelated-campaign'}
    wrong_path = tmp_path / 'wrong-control.json'; wrong_path.write_text(json.dumps(wrong))
    blocked = m.initialize_confirmation(search, tmp_path / 'blocked-review', plan, control_ref=wrong_path)
    assert not blocked['launch_allowed']
    assert (tmp_path / 'blocked-review' / 'confirmation.json').is_file()
    m.initialize_confirmation(search, tmp_path / 'blocked-review', plan, control_ref=wrong_path)
    parent = json.loads((search / 'campaign.json').read_text())
    assert not parent.get('confirmation_refs')
    assert reconcile_usage(approved, parent)['known']
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    started = m.run_confirmation_next(dest)
    assert started['tasks'][0]['status'] == 'completed'
    assert len(json.loads((search / 'campaign.json').read_text())['confirmation_refs']) == 1


def test_held_valid_plan_registers_receipt_before_resumed_prepare(tmp_path, monkeypatch):
    m, search, dest, plan, control = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    parent['control'] = json.loads(control.read_text())
    parent['research_hold'] = {'decision_id': 'diagnose-first'}
    (search / 'campaign.json').write_text(json.dumps(parent))
    held = m.initialize_confirmation(search, dest, plan, control_ref=control)
    assert not held['launch_allowed']
    parent = json.loads((search / 'campaign.json').read_text())
    assert not parent.get('confirmation_refs')
    parent.pop('research_hold')
    (search / 'campaign.json').write_text(json.dumps(parent))
    prepare = runner.prepare_run
    def verify_registered_before_prepare(config, **kwargs):
        fresh = json.loads((search / 'campaign.json').read_text())
        manifest = json.loads((dest / 'confirmation.json').read_text())
        assert fresh['confirmation_refs'] == [manifest['association']]
        assert manifest['launch_allowed'] and manifest['inventory']['expected_count'] == 1
        return prepare(config, **kwargs)
    monkeypatch.setattr(runner, 'prepare_run', verify_registered_before_prepare)
    resumed = m.run_confirmation_next(dest)
    assert resumed['tasks'][0]['status'] == 'completed'
