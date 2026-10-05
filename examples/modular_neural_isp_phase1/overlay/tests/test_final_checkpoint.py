"""Final weight identity uses fixture checkpoints and a bounded CPU evaluator."""
import json
from pathlib import Path

import pytest

from test_runner import config
from test_control import authorized_control
from tm_research import campaign, runner



@pytest.fixture(autouse=True)
def native_result_boundary(monkeypatch):
    # Terminal native fixtures keep process-liveness concerns outside this test.
    def read_native(run_id, runs_dir):
        return json.loads((Path(runs_dir) / run_id / 'state.json').read_text())
    monkeypatch.setattr(runner, 'get_result', read_native)
    monkeypatch.setattr(campaign, 'get_result', read_native)


def native_run(cfg, run_id):
    root = Path(cfg['runs_dir']) / run_id
    models = root / 'train' / 'models'; models.mkdir(parents=True)
    config_dir = root / 'train' / 'config'; config_dir.mkdir()
    checkpoint = models / (run_id + '.pth')
    recipe = {key: cfg[key] for key in runner.RECIPE_DEFAULTS}
    checkpoint.write_text(json.dumps({**recipe, 'seed': cfg['seed']}))
    (config_dir / (checkpoint.stem + '.json')).write_text('{}')
    actual_cfg = dict(cfg); actual_cfg.pop('test', None)
    (root / 'config.json').write_text(json.dumps(actual_cfg))
    train = {'best_checkpoint': str(checkpoint), 'config_dir': str(config_dir)}
    (root / 'train' / 'metrics.json').write_text(json.dumps(train))
    dev = root / 'dev'; dev.mkdir()
    metrics = {'mean_per_image_psnr': 22., 'num_images': 1,
               'protocol': 'P' + str(cfg['eval_size']), 'eval_size': cfg['eval_size']}
    (dev / 'metrics.json').write_text(json.dumps(metrics))
    result = {'run_id': run_id, 'status': 'completed', 'valid': True,
              'recipe': recipe, 'train_metrics': train, 'dev_psnr': 22.,
              'dev_metrics': metrics, 'expected_count': 1,
              'artifacts': {**train, 'dev_metrics': str(dev / 'metrics.json')},
              'usage': {'gpu_hours': 0}}
    (root / 'state.json').write_text(json.dumps(result))
    return result


def setup_confirmation(tmp_path):
    from tm_research import confirmation
    cfg = runner.normalize_config({**runner.BUDGET_DEFAULTS, **runner.RECIPE_DEFAULTS,
                                   **config(tmp_path), 'seed': 1})
    baseline = native_run(cfg, 'exp_001')
    winner = native_run({**cfg, 'loss_family': 'mse'}, 'exp_002')
    search = tmp_path / 'search'; search.mkdir()
    parent = {'status': 'finalized', 'active': None, 'config': cfg,
              'trials': [{'run_id': result['run_id'], 'valid': True,
                          'recipe': result['recipe'], 'usage': {'gpu_hours': 0}}
                         for result in (baseline, winner)],
              'best': {'run_id': 'exp_002', 'recipe': winner['recipe'], 'dev_psnr': 22.},
              'frozen': {'run_id': 'exp_002', 'recipe': winner['recipe']}}
    (search / 'campaign.json').write_text(json.dumps(parent))
    control = authorized_control(); control['protocol_id'] = 'P512'
    control['limits'].update(job_walltime_seconds=15, total_gpu_hours=0,
                             allocated_gpus=0, confirmation_gpu_hours=0, calibration_gpu_hours=0)
    control['confirmation_scope'] = {'authorized': True, 'kind': 'original_dev',
                                    'seeds': [3, 9], 'replicates': 1}
    control_path = tmp_path / 'control.json'; control_path.write_text(json.dumps(control))
    plan = {'seeds': [3, 9], 'replicates': 1, 'budget_gpu_hours': 0,
            'scope': {'kind': 'original_dev'}}
    return confirmation, search, tmp_path / 'confirmation', plan, control_path


def bounded_surrogate(config, checkpoint, split, output, *, execution_limits):
    # No image enumeration or TEST reads; fixture weights prove exact identity.
    assert execution_limits == {'allocated_gpus': 0, 'job_walltime_seconds': 15}
    weight = json.loads(Path(checkpoint['best_checkpoint']).read_text())
    score = {'original': 20., 'mse': 22.}[weight['loss_family']]
    metrics = {'mean_per_image_psnr': score, 'num_images': 1,
               'protocol': 'P512', 'eval_size': 512}
    output = Path(output); output.mkdir(parents=True)
    result = {'valid': True, 'score': score, 'metrics': metrics,
              'artifacts': dict(checkpoint), 'usage': {'gpu_hours': 0}}
    (output / 'metrics.json').write_text(json.dumps(metrics))
    (output / 'evaluation_result.json').write_text(json.dumps(result))
    (output / 'evaluation_state.json').write_text(json.dumps({**result, 'status': 'completed'}))
    (output / 'evaluation_health.json').write_text(json.dumps({'liveness': 'dead'}))
    return result

def selected_confirmation(tmp_path, *, controlled=True):
    module, search, dest, plan, control_path = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    if controlled:
        control = json.loads(control_path.read_text())
        control['test_permission'] = True
        control_path.write_text(json.dumps(control))
        parent['control'] = control
        (search / 'campaign.json').write_text(json.dumps(parent))
    plan['final_checkpoint_rule'] = {'kind': 'predeclared_seed', 'seed': 9, 'replicate': 1}
    first = 'winner:9:1'
    plan['execution_order'] = [first, 'baseline:3:1', 'winner:3:1', 'baseline:9:1']
    module.initialize_confirmation(search, dest, plan, control_ref=control_path)
    state = json.loads((dest / 'confirmation.json').read_text())
    selected = next(t for t in state['tasks'] if t['task_key'] == first)
    run_cfg = {**state['config'], **state['plan']['arms']['winner']['recipe'], 'seed': 9}
    result = native_run(run_cfg, 'exp_003')
    selected.update(status='completed', run_id='exp_003', run_dir=str(dest / 'runs' / 'exp_003'),
                    checkpoint={**result['artifacts'], 'selection_split': 'original_dev'},
                    result=result, score=22., usage={'gpu_hours': 0})
    (dest / 'confirmation.json').write_text(json.dumps(state))
    return module, search, dest, selected


def test_predeclared_checkpoint_is_frozen_reported_and_actually_evaluated(tmp_path, monkeypatch):
    module, search, dest, task = selected_confirmation(tmp_path)
    before = json.loads((search / 'campaign.json').read_text())
    passed = []
    def surrogate(config, checkpoint, split, output, *, execution_limits):
        passed.append(dict(checkpoint))
        return bounded_surrogate(config, checkpoint, split, output,
                                                  execution_limits=execution_limits)
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', surrogate)
    reference = campaign.resolve_final_checkpoint(search)
    report = module.report_confirmation(dest)
    final = campaign.final_test(search)
    assert passed[0]['best_checkpoint'] == task['checkpoint']['best_checkpoint']
    assert passed[0]['config_dir'] == task['checkpoint']['config_dir']
    assert final['valid']
    assert final['run_id'] == task['run_id']
    assert final['final_checkpoint_ref'] == report['final_checkpoint_ref'] == reference
    assert report['final_checkpoint']['best_checkpoint'] == reference['best_checkpoint']
    after = json.loads((search / 'campaign.json').read_text())
    assert after['best'] == before['best'] and after['frozen'] == before['frozen']
    assert campaign.final_test(search) == final
    assert len(passed) == 1


def test_legacy_final_test_passes_predeclared_checkpoint_to_command(tmp_path, monkeypatch):
    module, search, dest, task = selected_confirmation(tmp_path, controlled=False)
    commands = []
    def surrogate(command, *args):
        commands.append(command)
        checkpoint = {'best_checkpoint': command[command.index('--model-path') + 1],
                      'config_dir': command[command.index('--config-dir') + 1]}
        bounded_surrogate({}, checkpoint, {}, command[command.index('--result-dir') + 1],
                          execution_limits={'allocated_gpus': 0, 'job_walltime_seconds': 15})
        return 0
    monkeypatch.setattr(campaign, '_execute', surrogate)
    final = campaign.final_test(search)
    command = commands[0]
    assert command[command.index('--model-path') + 1] == task['checkpoint']['best_checkpoint']
    assert command[command.index('--config-dir') + 1] == task['checkpoint']['config_dir']
    assert final['run_id'] == task['run_id']
    assert final['final_checkpoint_ref'] == module.report_confirmation(dest)['final_checkpoint_ref']


@pytest.mark.parametrize('change', ['failed', 'missing', 'duplicate', 'config', 'checkpoint', 'association', 'control'])
def test_invalid_predeclared_candidate_cannot_fall_back_to_search(tmp_path, monkeypatch, change):
    module, search, dest, task = selected_confirmation(tmp_path)
    path = dest / 'confirmation.json'
    state = json.loads(path.read_text())
    selected = next(t for t in state['tasks'] if t['task_key'] == task['task_key'])
    if change == 'failed': selected['status'] = 'failed'
    if change == 'missing': Path(selected['checkpoint']['best_checkpoint']).unlink()
    if change == 'duplicate': state['tasks'].append(dict(selected))
    if change == 'config':
        config_path = Path(selected['run_dir']) / 'config.json'
        cfg = json.loads(config_path.read_text()); cfg['seed'] = 3
        config_path.write_text(json.dumps(cfg))
    if change == 'checkpoint':
        parent = json.loads((search / 'campaign.json').read_text())
        result = runner.get_result(parent['frozen']['run_id'], parent['config']['runs_dir'])
        selected['checkpoint']['best_checkpoint'] = result['artifacts']['best_checkpoint']
    if change == 'association': state['association']['campaign_id'] = 'other'
    if change == 'control': state['control']['authorization_ref'] = 'other'
    path.write_text(json.dumps(state))
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', lambda *a, **k: pytest.fail('invalid checkpoint must not launch'))
    with pytest.raises(ValueError):
        campaign.final_test(search)
    parent = json.loads((search / 'campaign.json').read_text())
    assert not parent.get('final_test') and not parent.get('final_checkpoint_ref')


def test_later_registered_plan_cannot_override_frozen_primary(tmp_path):
    module, search, dest, task = selected_confirmation(tmp_path)
    path = dest / 'confirmation.json'
    state = json.loads(path.read_text())
    other = tmp_path / 'other_confirmation'; other.mkdir()
    state['association']['confirmation_dir'] = str(other)
    (other / 'confirmation.json').write_text(json.dumps(state))
    parent = json.loads((search / 'campaign.json').read_text())
    parent['confirmation_refs'].append(state['association'])
    (search / 'campaign.json').write_text(json.dumps(parent))
    assert campaign.resolve_final_checkpoint(search)['confirmation_dir'] == str(dest)
    reference = campaign.resolve_final_checkpoint(search, confirmation_dir=dest)
    assert reference['confirmation_dir'] == str(dest)
    with pytest.raises(ValueError):
        campaign.resolve_final_checkpoint(search, confirmation_dir=other)


def test_frozen_reference_cannot_retarget_after_interrupted_test(tmp_path, monkeypatch):
    module, search, dest, task = selected_confirmation(tmp_path)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('host interrupted')
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', interrupted)
    with pytest.raises(KeyboardInterrupt): campaign.final_test(search)
    parent_path = search / 'campaign.json'
    before = json.loads(parent_path.read_text())
    assert before['final_test']['final_checkpoint_ref'] == before['final_checkpoint_ref']
    path = dest / 'confirmation.json'; state = json.loads(path.read_text())
    state['plan']['final_checkpoint_rule']['seed'] = 3
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError): campaign.resolve_final_checkpoint(search)
    with pytest.raises(ValueError): campaign.final_test(search)
    after = json.loads(parent_path.read_text())
    assert after['final_checkpoint_ref'] == before['final_checkpoint_ref']
    assert after['final_test'] == before['final_test']


def test_retain_search_default_uses_search_weight(tmp_path, monkeypatch):
    module, search, dest, plan, control_path = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    control = json.loads(control_path.read_text()); control['test_permission'] = True
    parent['control'] = control; control_path.write_text(json.dumps(control))
    (search / 'campaign.json').write_text(json.dumps(parent))
    module.initialize_confirmation(search, dest, plan, control_ref=control_path)
    passed = []
    def surrogate(config, checkpoint, split, output, *, execution_limits):
        passed.append(dict(checkpoint))
        return bounded_surrogate(config, checkpoint, split, output, execution_limits=execution_limits)
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', surrogate)
    report = campaign.final_test(search)
    expected = runner.get_result(parent['frozen']['run_id'], parent['config']['runs_dir'])['artifacts']
    assert passed[0]['best_checkpoint'] == expected['best_checkpoint']
    assert report['run_id'] == parent['frozen']['run_id']
    assert module.report_confirmation(dest)['final_checkpoint']['best_checkpoint'] == expected['best_checkpoint']


def test_recovered_evaluation_cannot_claim_different_weights(tmp_path, monkeypatch):
    module, search, dest, task = selected_confirmation(tmp_path)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('host interrupted')
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', interrupted)
    with pytest.raises(KeyboardInterrupt): campaign.final_test(search)
    parent = json.loads((search / 'campaign.json').read_text())
    actual = dict(parent['final_checkpoint_ref'])
    actual['best_checkpoint'] = runner.get_result(parent['frozen']['run_id'], parent['config']['runs_dir'])['artifacts']['best_checkpoint']
    bounded_surrogate(parent['config'], actual, {}, search / 'final_test',
                      execution_limits={'allocated_gpus': 0, 'job_walltime_seconds': 15})
    recovered = campaign.final_test(search)
    assert recovered['status'] == 'failed' and not recovered['valid']
    assert 'checkpoint identity' in recovered['error']
    assert recovered['final_checkpoint_ref'] == parent['final_checkpoint_ref']


@pytest.mark.parametrize('controlled', [False, True])
def test_old_retained_search_terminal_report_remains_reusable(tmp_path, monkeypatch, controlled):
    module, search, dest, plan, control_path = setup_confirmation(tmp_path)
    parent = json.loads((search / 'campaign.json').read_text())
    if controlled:
        control = json.loads(control_path.read_text()); control['test_permission'] = True
        parent['control'] = control
        (search / 'campaign.json').write_text(json.dumps(parent))
        monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', bounded_surrogate)
    else:
        def execute(command, *args):
            checkpoint = {'best_checkpoint': command[command.index('--model-path') + 1],
                          'config_dir': command[command.index('--config-dir') + 1]}
            bounded_surrogate({}, checkpoint, {}, command[command.index('--result-dir') + 1],
                              execution_limits={'allocated_gpus': 0, 'job_walltime_seconds': 15})
            return 0
        monkeypatch.setattr(campaign, '_execute', execute)
    report = campaign.final_test(search)
    parent = json.loads((search / 'campaign.json').read_text())
    parent.pop('final_checkpoint_ref'); parent['final_test'].pop('final_checkpoint_ref')
    (search / 'campaign.json').write_text(json.dumps(parent))
    expected = dict(report); expected.pop('final_checkpoint_ref')
    assert campaign.final_test(search) == expected
    assert 'final_checkpoint_ref' not in json.loads((search / 'campaign.json').read_text())


def test_confirmation_reporting_does_not_adopt_under_parent_lock(tmp_path):
    module, search, dest, task = selected_confirmation(tmp_path)
    before = (search / 'campaign.json').read_text()
    with campaign._locked(search):
        report = module.report_confirmation(dest)
    assert (search / 'campaign.json').read_text() == before
    assert report['final_checkpoint_ref'] is None
    assert report['final_checkpoint_pending_adoption'] is True
    assert report['final_checkpoint']['best_checkpoint'] == task['checkpoint']['best_checkpoint']
    from tm_research import research
    research.export_research_memory(search)
    assert (search / 'campaign.json').read_text() == before


@pytest.mark.parametrize('pending_action', ['report_stop', 'finalize', 'diagnose', 'request_scope_change'])
def test_legacy_pending_decision_blocks_next_submit_and_recovered_start(tmp_path, monkeypatch, pending_action):
    from test_slow_decisions import plateau, save
    from test_campaign import proposal
    state = plateau(tmp_path); state.pop('control')
    state['pending_slow_decision'] = {'action': pending_action, 'decision_id': 'durable',
                                    'trigger_id': 'fixture/exp_001/valid_plateau', 'decision_ref': 'decisions/durable.json'}
    save(tmp_path, state)
    event = campaign.next_proposal(tmp_path)
    assert not event['proposal_allowed']
    assert event['next_action']['action'] == pending_action
    monkeypatch.setattr(campaign, '_launch', lambda *a, **k: pytest.fail('pending decision cannot launch'))
    with pytest.raises(ValueError):
        campaign.submit_proposal(tmp_path, proposal(event, {'learning_rate': 0.00005}))
    state['launch_intent'] = {'index': 2, 'proposal': {'recipe': {'learning_rate': 0.00005}}}
    assert not campaign._current_start_allowed(tmp_path, state, 'launch_intent')
    assert state['status'] == 'launch_blocked'


@pytest.mark.parametrize('pending_action', ['diagnose', 'request_scope_change'])
def test_legacy_pending_diagnosis_blocks_final_evaluation(tmp_path, monkeypatch, pending_action):
    module, search, dest, plan, control = setup_confirmation(tmp_path)
    path = search / 'campaign.json'; state = json.loads(path.read_text())
    state['pending_slow_decision'] = {'action': pending_action}
    path.write_text(json.dumps(state))
    monkeypatch.setattr(campaign, '_execute', lambda *a, **k: pytest.fail('pending decision cannot evaluate'))
    with pytest.raises(ValueError): campaign.final_test(search)
    assert not json.loads(path.read_text()).get('final_test')


@pytest.mark.parametrize('controlled', [False, True])
def test_terminal_report_cannot_change_frozen_final_run_identity(tmp_path, monkeypatch, controlled):
    module, search, dest, task = selected_confirmation(tmp_path, controlled=controlled)
    if controlled:
        monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', bounded_surrogate)
    else:
        def execute(command, *args):
            checkpoint = {'best_checkpoint': command[command.index('--model-path') + 1],
                          'config_dir': command[command.index('--config-dir') + 1]}
            bounded_surrogate({}, checkpoint, {}, command[command.index('--result-dir') + 1],
                              execution_limits={'allocated_gpus': 0, 'job_walltime_seconds': 15})
            return 0
        monkeypatch.setattr(campaign, '_execute', execute)
    campaign.final_test(search)
    path = search / 'campaign.json'; parent = json.loads(path.read_text())
    parent['final_test']['run_id'] = parent['frozen']['run_id']
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match='identity'): campaign.final_test(search)


@pytest.mark.parametrize('already_frozen', [False, True])
def test_confirmation_and_actual_run_cannot_rewrite_parent_training_conditions(tmp_path, already_frozen):
    module, search, dest, task = selected_confirmation(tmp_path)
    before = campaign.resolve_final_checkpoint(search) if already_frozen else None
    manifest_path = dest / 'confirmation.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['config']['epochs'] = 2
    manifest_path.write_text(json.dumps(manifest))
    config_path = Path(task['run_dir']) / 'config.json'
    cfg = json.loads(config_path.read_text()); cfg['epochs'] = 2
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='scientific config'):
        campaign.resolve_final_checkpoint(search)
    assert json.loads((search / 'campaign.json').read_text()).get('final_checkpoint_ref') == before
