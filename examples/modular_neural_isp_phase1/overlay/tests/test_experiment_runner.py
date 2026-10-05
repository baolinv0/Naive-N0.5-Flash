"""Outer orchestration uses real campaign APIs with labelled tiny subprocess fixtures."""
import copy
import json
from pathlib import Path

import pytest

from test_control import authorized_control
from test_runner import config


def manifest(tmp_path, strategies=None, *, confirmation=False):
    cfg = config(tmp_path)
    cfg.pop('test')
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps(cfg))
    ctrl = authorized_control()
    ctrl.update(protocol_id='P512', observation_mode='text', approved_config_ref=str(baseline))
    ctrl['observation_config']['required_for_proposal'] = []
    ctrl['limits'].update(allocated_gpus=0, total_gpu_hours=0,
        confirmation_gpu_hours=0, calibration_gpu_hours=0,
        job_walltime_seconds=20, max_search_trials=4)
    ctrl['confirmation_scope'] = ({'authorized': True, 'kind': 'original_dev', 'seeds': [1001, 1002]}
                                if confirmation else {'authorized': False, 'kind': 'none'})
    control = tmp_path / 'control.json'
    control.write_text(json.dumps(ctrl))
    groups = strategies or ['random', 'tpe', 'naive_scalar', 'naive_rich', 'naive_fast_slow']
    data = {'schema_version': 1, 'experiment_id': 'fixture', 'evidence_mode': 'engineering_fixture',
        'authorization_ref': ctrl['authorization_ref'], 'source_revision': ctrl['source_revision'],
        'baseline_config': str(baseline), 'control_template': str(control), 'profile_ref': None,
        'strategies': groups,
        'search_seeds': [101], 'max_trials': 4, 'min_delta': .01, 'campaign_walltime_seconds': 600,
        'experiment_limits': {'total_gpu_hours': 0, 'walltime_seconds': 600},
        'models': {'naive': {'base_url': 'http://127.0.0.1:8000/v1', 'model': 'fixture-naive'}},
        'model_limits': {'max_calls': 32, 'max_input_tokens_per_call': 24000,
            'max_output_tokens_per_call': 4096, 'max_total_tokens': 327680,
            'max_total_input_tokens': 262144, 'max_total_output_tokens': 65536},
        'confirmation': {'seeds_by_block': [[1001, 1002]], 'delta_useful_db': .1,
                         'delta_strategy': .1} if confirmation else None}
    data['campaign_model_limits'] = copy.deepcopy(data['model_limits'])
    for field in ('max_calls', 'max_total_tokens', 'max_total_input_tokens', 'max_total_output_tokens'):
        data['model_limits'][field] *= sum(s.startswith('naive_') for s in groups) or 1
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(data))
    return path


def test_generate_isolated_configs_preserves_science_and_trial_policy(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    path = manifest(tmp_path)
    result = generate_experiment(path, tmp_path / 'outer')
    assert len(result['entries']) == 5
    run_paths = set()
    for entry in result['entries']:
        cfg = json.loads(Path(entry['config_path']).read_text())
        control = json.loads(Path(entry['control_path']).read_text())
        assert cfg['seed'] == 101 and cfg['epochs'] == 1
        assert 'test' not in cfg and control['test_permission'] is False
        assert entry['no_gain_limit'] == 5
        assert control['slow_policy']['consecutive_valid_no_gain'] == (
            2 if entry['strategy'] == 'naive_fast_slow' else 5)
        run_paths.add(cfg['runs_dir'])
    assert len(run_paths) == 5
    assert generate_experiment(path, tmp_path / 'outer') == result


def test_manifest_cannot_expand_control_limits_or_authorization(tmp_path):
    from tm_research.experiment_plan import load_experiment
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    for key, value in [('max_trials', 5), ('authorization_ref', 'invented-authority')]:
        changed = copy.deepcopy(data)
        changed[key] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_experiment(path)


def test_random_steps_execute_native_trials_freeze_once_and_resume(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment, experiment_status
    path = manifest(tmp_path, ['random'])
    root = tmp_path / 'outer'
    index = generate_experiment(path, root)
    for _ in range(10):
        result = step_experiment(root, wait=True)
        if result['status'] == 'search_complete':
            break
    assert experiment_status(root)['status'] == 'search_complete'
    campaign_file = Path(index['entries'][0]['campaign_dir']) / 'campaign.json'
    saved = json.loads(campaign_file.read_text())
    assert len(saved['trials']) == 4 and saved['frozen']
    assert len({t['run_id'] for t in saved['trials']}) == 4
    step_experiment(root, wait=True)
    assert json.loads(campaign_file.read_text()) == saved


class FixtureClient:
    def __init__(self):
        self.messages = []

    def complete(self, messages, *, request_id, role, max_output_tokens=4096):
        from tm_research.experiment_strategies import sampler_payload
        self.messages.append(copy.deepcopy(messages))
        packet = json.loads(messages[1]['content'])
        history = packet['history']
        recipe = {'loss_family': 'l1', 'optimizer': 'adam',
                  'learning_rate': 1e-4 if len(history) == 1 else 3e-4, 'weight_decay': 1e-7}
        if role == 'slow_review':
            packet = copy.deepcopy(packet)
            packet['proposal_allowed'] = True
            packet['next_action'] = {'action': 'propose', 'review_required': False}
            recipe['learning_rate'] = 1e-3
        payload = sampler_payload(packet, recipe, 'random', request_id)
        if role == 'slow_review':
            payload = {'decision': {**payload['decision'], 'kind': 'slow_review',
                'trigger_id': json.loads(messages[1]['content'])['next_action']['trigger_id'], 'action': 'finalize',
                'requested_change': {}}}
        return {'payload': payload, 'raw_text': json.dumps(payload), 'usage': None,
                'request_id': request_id, 'model': 'engineering-fixture', 'role': role}


def test_naive_fast_slow_consumes_actual_result_and_slow_stops(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    path = manifest(tmp_path, ['naive_fast_slow'])
    root = tmp_path / 'outer'
    index = generate_experiment(path, root)
    client = FixtureClient()
    for _ in range(9):
        result = step_experiment(root, wait=True, clients={'naive': client})
        if result['status'] == 'search_complete':
            break
    state = json.loads((Path(index['entries'][0]['campaign_dir']) / 'campaign.json').read_text())
    assert len(state['trials']) == 3 and state['frozen']
    assert any(d['kind'] == 'slow_review' for d in state['research_decisions'])
    packets = [json.loads(messages[1]['content']) for messages in client.messages]
    assert [p['feedback']['overall']['dev_psnr'] for p in packets] == [20, 19, 19]


def test_confirmation_waits_for_all_searches_and_preserves_primary_plan(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import confirm_experiment, step_experiment
    path = manifest(tmp_path, ['random'], confirmation=True)
    root = tmp_path / 'outer'
    index = generate_experiment(path, root)
    with pytest.raises(ValueError, match='frozen|search'):
        confirm_experiment(root, wait=True)
    for _ in range(9):
        if step_experiment(root, wait=True)['status'] == 'search_complete':
            break
    for _ in range(7):
        result = confirm_experiment(root, wait=True)
        if result['status'] == 'confirmation_complete':
            break
    assert result['status'] == 'confirmation_complete'
    state = json.loads((Path(index['entries'][0]['campaign_dir']) / 'campaign.json').read_text())
    assert len(state['confirmation_refs']) == 1
    assert state['primary_confirmation']


class UniqueFixtureClient(FixtureClient):
    """A test-only result reader, not evidence of Naive reasoning quality."""
    def complete(self, messages, *, request_id, role, max_output_tokens=4096):
        if role == 'slow_review':
            return super().complete(messages, request_id=request_id, role=role)
        from tm_research.experiment_strategies import sampler_payload, suggest_recipe
        self.messages.append(copy.deepcopy(messages))
        packet = json.loads(messages[1]['content'])
        recipe = suggest_recipe('random', packet['history'], seed=len(packet['history']) + 200)
        payload = sampler_payload(packet, recipe, 'random', request_id)
        return {'payload': payload, 'raw_text': json.dumps(payload), 'usage': None,
                'request_id': request_id, 'model': 'engineering-fixture', 'role': role}


def test_all_five_groups_share_native_interface_and_scalar_is_projected(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    path = manifest(tmp_path)
    root = tmp_path / 'outer'
    index = generate_experiment(path, root)
    client = UniqueFixtureClient()
    for _ in range(35):
        result = step_experiment(root, wait=True, clients={'naive': client})
        if result['status'] == 'search_complete':
            break
    assert result['status'] == 'search_complete', result
    for entry in index['entries']:
        native = json.loads((Path(entry['campaign_dir']) / 'campaign.json').read_text())
        assert native['frozen'] and 3 <= len(native['trials']) <= 4
        assert len({json.dumps(t['recipe'], sort_keys=True) for t in native['trials']}) == len(native['trials'])
        assert all(t['valid'] for t in native['trials'])
    scalar = [json.loads(m[1]['content']) for m in client.messages
              if 'naive_scalar' in json.loads(m[1]['content'])['campaign_id']]
    assert scalar and all('cases' not in p['feedback'] and 'groups' not in p['feedback'] for p in scalar)
    assert all(p['feedback']['observations'][0]['id'] == 'O1' for p in scalar)


def test_unknown_model_request_is_not_repeated(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path, ['naive_scalar']), root)
    class FailedClient:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            raise OSError('Response outcome unknown')
    client = FailedClient()
    step_experiment(root, wait=True)
    for _ in range(4):
        result = step_experiment(root, wait=True, clients={'naive': client})
    assert result['status'] == 'paused' and client.calls == 1
    assert result['entries'][0]['trial_count'] == 1


def test_campaign_walltime_expiry_preserves_missing_not_fake_freeze(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    root = tmp_path / 'outer'
    index = generate_experiment(manifest(tmp_path, ['random']), root)
    step_experiment(root, wait=True)
    entry = index['entries'][0]
    outer_path = Path(entry['directory']) / 'outer.json'
    outer = json.loads(outer_path.read_text()) if outer_path.exists() else {'attempts': 0, 'pending': None, 'rejections': [], 'auxiliary': {}}
    outer['started_at_epoch'] = 0
    outer_path.write_text(json.dumps(outer))
    result = step_experiment(root, wait=True)
    assert result['status'] == 'paused' and result['entries'][0]['trial_count'] == 1
    assert not json.loads((Path(entry['campaign_dir']) / 'campaign.json').read_text())['frozen']


@pytest.mark.parametrize('boundary', ['launch_intent', 'prepared_active'])
def test_expired_campaign_does_not_resume_unstarted_launch_intent(tmp_path, monkeypatch, boundary):
    from tm_research import campaign
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    root = tmp_path / 'outer'
    index = generate_experiment(manifest(tmp_path, ['random']), root)
    entry = index['entries'][0]
    with monkeypatch.context() as context:
        if boundary == 'launch_intent':
            context.setattr(campaign, '_resume_launch', lambda directory, state: state)
        else:
            context.setattr(campaign, 'start_prepared_run', lambda *args, **kwargs: None)
        step_experiment(root, wait=False)
    native_path = Path(entry['campaign_dir']) / 'campaign.json'
    native = json.loads(native_path.read_text())
    assert native.get('launch_intent') if boundary == 'launch_intent' else native['active']
    outer_path = Path(entry['directory']) / 'outer.json'
    outer = json.loads(outer_path.read_text())
    outer['started_at_epoch'] = 0
    outer_path.write_text(json.dumps(outer))
    result = step_experiment(root, wait=False)
    assert result['status'] == 'paused'
    assert json.loads(native_path.read_text()) == native
    if boundary == 'launch_intent':
        assert not list((Path(entry['directory']) / 'runs').glob('exp_*'))
    else:
        assert not (Path(entry['directory']) / 'runs' / native['active']['run_id'] / 'train').exists()


def test_expired_campaign_can_collect_verified_live_work_without_launch(tmp_path, monkeypatch):
    import os
    from tm_research import campaign, experiment_runner, runner
    from tm_research.experiment_plan import generate_experiment
    root = tmp_path / 'outer'
    index = generate_experiment(manifest(tmp_path, ['random']), root)
    entry = index['entries'][0]
    with monkeypatch.context() as context:
        context.setattr(campaign, 'start_prepared_run', lambda *args, **kwargs: None)
        experiment_runner.step_experiment(root, wait=False)
    outer_path = Path(entry['directory']) / 'outer.json'
    outer = json.loads(outer_path.read_text())
    outer['started_at_epoch'] = 0
    outer_path.write_text(json.dumps(outer))
    native = json.loads((Path(entry['campaign_dir']) / 'campaign.json').read_text())
    liveness = runner._identity_liveness(runner._process_identity(os.getpid()))
    assert liveness == 'live'
    collected = []
    monkeypatch.setattr(runner, 'inspect_run', lambda *args, **kwargs: {'liveness': liveness})
    monkeypatch.setattr(experiment_runner, 'campaign_status', lambda directory: collected.append(directory) or native)
    result = experiment_runner.step_experiment(root, wait=False)
    assert result['status'] == 'paused' and collected == [entry['campaign_dir']]


def test_invalid_model_decisions_are_bounded_without_trial_spending(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    root = tmp_path / 'outer'
    index = generate_experiment(manifest(tmp_path, ['naive_scalar']), root)
    class BadClient:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            return {'payload': {}, 'model': 'fixture'}
    client = BadClient()
    step_experiment(root, wait=True)
    for _ in range(4):
        result = step_experiment(root, wait=True, clients={'naive': client})
    assert client.calls == 2 and result['status'] == 'paused'
    assert result['entries'][0]['trial_count'] == 1
    outer = json.loads((Path(index['entries'][0]['directory']) / 'outer.json').read_text())
    assert len(outer['rejections']) == 2


@pytest.mark.parametrize('bad_proposal', [{}, [], None])
def test_malformed_proposal_structure_is_rejected_and_bounded(tmp_path, bad_proposal):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path, ['naive_scalar']), root)
    class BadClient:
        def complete(self, messages, *, request_id, **kwargs):
            return {'payload': {'proposal': bad_proposal, 'decision': {'decision_id': request_id}}, 'model': 'fixture'}
    step_experiment(root, wait=True)
    for _ in range(4):
        result = step_experiment(root, wait=True, clients={'naive': BadClient()})
    assert result['status'] == 'paused' and result['entries'][0]['trial_count'] == 1


@pytest.mark.parametrize('field,value', [('action', []), ('comparison_run_id', {}), ('reference_run_ids', {})])
def test_malformed_native_decision_types_are_counted_rejections(tmp_path, field, value):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    class BadDecisionClient(UniqueFixtureClient):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            response['payload']['decision'][field] = value
            return response
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path, ['naive_scalar']), root)
    step_experiment(root, wait=True)
    client = BadDecisionClient()
    for _ in range(4):
        result = step_experiment(root, wait=True, clients={'naive': client})
    assert result['status'] == 'paused' and len(client.messages) == 2
    assert result['entries'][0]['trial_count'] == 1


def test_campaigns_interleave_baselines_before_next_proposal(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    root = tmp_path / 'outer'
    index = generate_experiment(manifest(tmp_path), root)
    for _ in range(5):
        step_experiment(root, wait=True, clients={'naive': UniqueFixtureClient()})
    for entry in index['entries']:
        path = Path(entry['campaign_dir']) / 'campaign.json'
        assert path.exists() and len(json.loads(path.read_text())['trials']) == 1


def test_interrupted_baseline_initialization_pauses_instead_of_proposing(tmp_path, monkeypatch):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    from tm_research import campaign
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path, ['random']), root)
    def interrupted(*args, **kwargs):
        raise OSError('crash before baseline launch receipt')
    with monkeypatch.context() as context:
        context.setattr(campaign, '_launch', interrupted)
        with pytest.raises(OSError):
            step_experiment(root, wait=True)
    result = step_experiment(root, wait=True)
    assert result['status'] == 'paused' and result['entries'][0]['trial_count'] == 0


def test_resume_prepared_response_does_not_call_model_again(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    from tm_research.campaign import next_proposal
    from tm_research.experiment_strategies import sampler_payload, suggest_recipe
    root = tmp_path / 'outer'
    index = generate_experiment(manifest(tmp_path, ['naive_scalar']), root)
    entry = index['entries'][0]
    step_experiment(root, wait=True)
    packet = next_proposal(entry['campaign_dir'])
    recipe = suggest_recipe('random', packet['history'], 42)
    payload = sampler_payload(packet, recipe, 'random', 'recovered_request')
    outer_path = Path(entry['directory']) / 'outer.json'
    outer_path.write_text(json.dumps({'attempts': 1, 'rejections': [], 'auxiliary': {}, 'pending': {
        'phase': 'prepared', 'request_id': 'recovered_request', 'role': 'fast_proposal',
        'feedback_ref': packet['feedback']['feedback_ref'], 'payload': payload}}))
    client = UniqueFixtureClient()
    result = step_experiment(root, wait=True, clients={'naive': client})
    assert result['entries'][0]['trial_count'] == 2 and client.messages == []
    # A crash after native commit but before clearing the outer pending marker.
    saved = json.loads(outer_path.read_text())
    saved['pending'] = {'phase': 'prepared', 'request_id': 'recovered_request',
        'role': 'fast_proposal', 'feedback_ref': packet['feedback']['feedback_ref'], 'payload': payload}
    outer_path.write_text(json.dumps(saved))
    result = step_experiment(root, wait=True, clients={'naive': client})
    assert result['entries'][0]['trial_count'] == 2 and client.messages == []


def test_fixture_original_dev_confirmation_cannot_enter_primary_inference(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment, confirm_experiment, collect_analysis_records
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path, ['random'], confirmation=True), root)
    for _ in range(9):
        if step_experiment(root, wait=True)['status'] == 'search_complete':
            break
    for _ in range(7):
        if confirm_experiment(root, wait=True)['status'] == 'confirmation_complete':
            break
    records = collect_analysis_records(root)
    assert records[0]['expected_seeds'] == [1001, 1002]
    assert not records[0]['provenance']['eligible_for_primary_analysis']
    assert all(p['status'] == 'inconclusive' and p['delta_db'] is None for p in records[0]['pairs'])


def test_slow_continuation_launches_exact_reviewed_recipe_without_new_fast_call(tmp_path):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    class ContinueClient(FixtureClient):
        def complete(self, messages, **kwargs):
            result = super().complete(messages, **kwargs)
            if kwargs['role'] == 'slow_review':
                result['payload']['decision'].update(action='propose', requested_change={
                    'loss_family': 'l1', 'optimizer': 'adam', 'learning_rate': 1e-3, 'weight_decay': 1e-7})
            return result
    root = tmp_path / 'outer'
    index = generate_experiment(manifest(tmp_path, ['naive_fast_slow']), root)
    client = ContinueClient()
    for _ in range(10):
        result = step_experiment(root, wait=True, clients={'naive': client})
        if result['status'] == 'search_complete':
            break
    state = json.loads((Path(index['entries'][0]['campaign_dir']) / 'campaign.json').read_text())
    assert result['status'] == 'search_complete' and len(state['trials']) == 4
    assert len(client.messages) == 3
    assert state['trials'][-1]['recipe']['learning_rate'] == 1e-3
    assert state['trials'][-1]['research_decision']['kind'] == 'slow_review'
