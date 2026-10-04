"""Synthetic subprocess integration; these tests do not claim live research gains."""
import json
from pathlib import Path

import pytest

from test_runner import config
from test_campaign import proposal
from tm_research import campaign
from tm_research.cli import main


def test_trial_references_saved_before_start_and_actual_inheritance(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5)
    first = campaign.campaign_status(directory)['trials'][0]['run_id']
    campaign.submit_proposal(directory, proposal(campaign.next_proposal(directory), {'loss_family': 'mse'}))
    seen = []
    real_start = campaign.start_prepared_run
    def start(run_id, runs_dir='runs'):
        active = json.loads((directory / 'campaign.json').read_text())['active']
        assert active['run_id'] == run_id
        assert active['resolved_config']['loss_family'] == 'mse'
        seen.append(active)
        return real_start(run_id, runs_dir)
    monkeypatch.setattr(campaign, 'start_prepared_run', start)
    p = proposal(campaign.next_proposal(directory), {'learning_rate': 0.00005})
    p['based_on'] = {'run_id': first, 'dev_psnr': 20, 'observation': 'Historical baseline evidence.'}
    result = campaign.submit_proposal(directory, p)
    trial = result['trials'][-1]
    assert trial['baseline_run_id'] == trial['based_on_run_id'] == first
    assert trial['best_before_run_id'] == trial['construction_base_run_id'] == 'exp_002'
    assert trial['comparison_run_id'] == 'exp_002'
    assert trial['init_checkpoint_ref'] == cfg.get('init_checkpoint')
    assert len(seen) == 1
    assert len(campaign.campaign_status(directory)['trials']) == 3


def test_diagnostic_failure_preserves_valid_baseline_and_missing_csv_legacy(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(campaign, 'build_run_feedback', lambda *a: (_ for _ in ()).throw(ValueError('diagnostic broke')))
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory)
    assert state['trials'][0]['valid'] and state['best']['dev_psnr'] == 20
    assert state['latest_feedback']['diagnostics_status'] == 'unavailable'
    assert campaign.next_proposal(directory)['proposal_allowed']
    assert len(campaign.campaign_status(directory)['trials']) == 1


def test_confirmation_cli_dispatch(tmp_path, monkeypatch, capsys):
    from tm_research import confirmation
    calls = []
    def initialize(search, target, plan, *, control_ref):
        calls.append((search, target, plan, control_ref)); return {'status': 'ready'}
    monkeypatch.setattr(confirmation, 'initialize_confirmation', initialize)
    plan = tmp_path / 'plan.json'; plan.write_text('{"seeds":[1]}')
    assert main(['confirmation', 'init', '--campaign-dir', str(tmp_path/'search'),
        '--confirmation-dir', str(tmp_path/'confirm'), '--plan', str(plan), '--control', 'control.json']) == 0
    assert calls[0][2] == {'seeds': [1]}
    assert json.loads(capsys.readouterr().out)['status'] == 'ready'


def cpu_control(tmp_path, *, required=None):
    from test_control import authorized_control
    control = authorized_control()
    control.update(protocol_id='P512', primary_metric='mean_per_image_rgb_psnr')
    control['limits'].update(allocated_gpus=0, total_gpu_hours=0, confirmation_gpu_hours=0,
                             calibration_gpu_hours=0, job_walltime_seconds=20, max_search_trials=5)
    control['observation_config']['required_for_proposal'] = required or []
    path = tmp_path / 'control.json'; path.write_text(json.dumps(control))
    return path


def decision_for(evidence, patch, *, comparison=None, name='decision_001'):
    feedback = evidence['feedback']
    run_id = feedback['latest_run_id']
    return {'schema_version': 1, 'decision_id': name, 'kind': 'fast_proposal',
        'feedback_ref': feedback['feedback_ref'], 'latest_seen_run_id': run_id,
        'reference_run_ids': [run_id], 'comparison_run_id': comparison or run_id,
        'observation_refs': [f"{evidence['campaign_id']}/{run_id}/{feedback['feedback_revision']}/{feedback['observations'][0]['id']}"],
        'observation': 'Observed stored fixture evaluation.', 'prediction': 'DEV score may increase.',
        'falsifier': 'DEV does not exceed effective best.', 'alternative_explanation': 'Supervision also changes.',
        'action': 'propose', 'requested_change': patch, 'claim_scope': 'exploratory_dev'}


def test_control_sidecar_latest_ack_explicit_comparison_and_snapshot(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=cpu_control(tmp_path))
    evidence = campaign.next_proposal(directory)
    patch = {'loss_family': 'mse'}
    with pytest.raises(ValueError, match='decision'):
        campaign.submit_proposal(directory, proposal(evidence, patch))
    stale = decision_for(evidence, patch); stale['latest_seen_run_id'] = 'exp_999'
    with pytest.raises(ValueError, match='latest'):
        campaign.submit_proposal(directory, proposal(evidence, patch), decision_ref=stale)
    mismatched = decision_for(evidence, {'optimizer': 'adamw'})
    with pytest.raises(ValueError, match='requested_change'):
        campaign.submit_proposal(directory, proposal(evidence, patch), decision_ref=mismatched)
    first = campaign.submit_proposal(directory, proposal(evidence, patch), decision_ref=decision_for(evidence, patch))
    assert first['trials'][-1]['usage']['gpu_hours'] == 0
    evidence = campaign.next_proposal(directory)
    patch = {'learning_rate': 0.00005}
    d = decision_for(evidence, patch, comparison='exp_001', name='decision_002')
    result = campaign.submit_proposal(directory, proposal(evidence, patch), decision_ref=d)
    trial = result['trials'][-1]
    assert trial['comparison_run_id'] == 'exp_001'
    assert trial['construction_base_run_id'] == 'exp_002'
    assert trial['resolved_config']['loss_family'] == 'mse'
    assert json.loads(Path(trial['decision_ref']).read_text()) == d
    campaign.finalize_campaign(directory)
    with pytest.raises(ValueError, match='TEST'):
        campaign.final_test(directory)


@pytest.mark.parametrize('controlled', [False, True])
def test_required_diagnostic_missing_is_diagnose_not_invalid(tmp_path, controlled):
    cfg = config(tmp_path)
    # Native aggregate is valid while its per-image evidence is absent.
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    script.write_text(script.read_text().replace("(r/'per_image.csv').write_text('image,psnr\\na,20\\n')", "None"))
    directory = tmp_path / 'campaign'
    result = campaign.initialize_campaign(cfg, directory, max_trials=5,
        control_ref=cpu_control(tmp_path, required=['sample_alignment']) if controlled else None)
    assert result['trials'][0]['valid'] and result['best']['dev_psnr'] == 20
    assert not (Path(cfg['runs_dir']) / result['trials'][0]['run_id'] / 'dev' / 'per_image.csv').exists()
    evidence = campaign.next_proposal(directory)
    assert evidence['proposal_allowed'] is (not controlled)
    assert evidence['next_action']['action'] == ('diagnose' if controlled else 'propose')


def test_health_unknown_keeps_same_reservation_and_blocks_submit(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory, wait=False)
    run_id = state['active']['run_id']
    monkeypatch.setattr(campaign, 'inspect_run', lambda *a: {'liveness': 'unknown'})
    monkeypatch.setattr(campaign, 'get_result', lambda *a: {'status': 'running'})
    evidence = campaign.next_proposal(directory)
    assert evidence['active']['run_id'] == run_id and not evidence['proposal_allowed']
    assert not campaign.campaign_status(directory)['trials']
    with pytest.raises(ValueError, match='cannot accept'):
        campaign.submit_proposal(directory, {'recipe': {}, 'hypothesis': 'pending', 'based_on': {}})


def test_prepared_host_interruption_resumes_same_run_before_start(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    real_start = campaign.start_prepared_run
    monkeypatch.setattr(campaign, 'start_prepared_run', lambda *a: (_ for _ in ()).throw(RuntimeError('host interrupted')))
    with pytest.raises(RuntimeError, match='host interrupted'):
        campaign.initialize_campaign(cfg, directory, wait=False)
    saved = json.loads((directory / 'campaign.json').read_text())
    assert saved['active']['run_id'] == 'exp_001'
    monkeypatch.setattr(campaign, 'start_prepared_run', real_start)
    final = campaign.wait_campaign(directory, timeout=15)
    assert [t['run_id'] for t in final['trials']] == ['exp_001']
    assert len(list(Path(cfg['runs_dir']).glob('exp_*'))) == 1


def test_feedback_rebuild_is_revision_idempotent_after_collection(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory)
    first = state['latest_feedback']
    second = campaign.campaign_feedback(directory)
    assert second['feedback_revision'] == first['feedback_revision']
    assert second['feedback_ref'] == first['feedback_ref']
    assert len(campaign.campaign_status(directory)['trials']) == 1


def test_actual_usage_blocks_next_job_and_preserves_confirmation_reserve(tmp_path):
    cfg = config(tmp_path)
    path = cpu_control(tmp_path)
    control = json.loads(path.read_text())
    max_job = 20 / 3600
    control['limits'].update(allocated_gpus=1, confirmation_gpu_hours=.01,
        calibration_gpu_hours=.004, total_gpu_hours=max_job + .014)
    path.write_text(json.dumps(control))
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=path)
    assert state['trials'][0]['usage']['gpu_hours'] > 0
    evidence = campaign.next_proposal(directory)
    assert not evidence['proposal_allowed']
    assert evidence['next_action']['action'] == 'report_stop'
    assert len(state['trials']) == 1 and state['best']['dev_psnr'] == 20


def test_plateau_review_independently_recorded_does_not_override_hardstop(tmp_path):
    from tm_research.research import record_research_decision
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, no_gain_limit=3,
                                 control_ref=cpu_control(tmp_path))
    for i, patch in enumerate(({'loss_family': 'l1'}, {'loss_family': 'l1', 'learning_rate': .0002})):
        e = campaign.next_proposal(directory)
        campaign.submit_proposal(directory, proposal(e, patch), decision_ref=decision_for(e, patch, name=f'fast_{i}'))
    e = campaign.next_proposal(directory)
    assert not e['proposal_allowed'] and e['next_action']['review_required']
    review = decision_for(e, {'learning_rate': .0002}, name='slow_001')
    review.update(kind='slow_review', action='propose', trigger_id=e['next_action']['trigger_id'])
    record_research_decision(directory, review)
    assert campaign.next_proposal(directory)['proposal_allowed']
    record_research_decision(directory, review)
    assert len((directory / 'research_decisions.jsonl').read_text().splitlines()) == 3
    campaign.finalize_campaign(directory)
    assert not campaign.next_proposal(directory)['proposal_allowed']
    with pytest.raises(ValueError, match='cannot accept'):
        campaign.submit_proposal(directory, proposal(e, {'weight_decay': 0}), decision_ref=review)


def test_next_summary_excludes_per_image_tables_and_bounds_observation_preview(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory)
    state['latest_feedback']['comparisons'] = {'comparison': {'status': 'complete',
        'overall': {'n_images': 100, 'sample_ids': [str(i) for i in range(100)]},
        'paired_rows': [{'sample_id': str(i)} for i in range(100)],
        'coverage': {'matched_count': 100}}}
    state['latest_feedback']['observations'] = [{'id': f'O{i}'} for i in range(100)]
    monkeypatch.setattr(campaign, 'campaign_status', lambda *a: state)
    e = campaign.next_proposal(directory)
    comparison = e['feedback']['comparisons']['comparison']
    assert 'paired_rows' not in comparison
    assert 'sample_ids' not in comparison['overall']
    assert comparison['coverage']['matched_count'] == 100
    assert len(e['feedback']['observations']) <= 12
    assert e['feedback']['observations_count'] == 100


def test_confirmation_cli_executes_frozen_duplicate_arms_without_changing_search(tmp_path, capsys):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    control = cpu_control(tmp_path)
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=control)
    before = campaign.finalize_campaign(directory)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'seeds': [1], 'replicates': 1, 'budget_gpu_hours': 0,
        'scope': {'kind': 'original_dev'}, 'delta_useful_db': .1,
        'failure_policy': 'preserve_no_retry'}))
    confirm = directory / 'confirmation'
    assert main(['confirmation', 'init', '--campaign-dir', str(directory),
        '--confirmation-dir', str(confirm), '--plan', str(plan), '--control', str(control)]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized['launch_allowed'], initialized['blocked_reasons']
    for _ in range(2):
        assert main(['confirmation', 'next', '--confirmation-dir', str(confirm), '--wait']) == 0
        capsys.readouterr()
    assert main(['confirmation', 'report', '--confirmation-dir', str(confirm)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['pairs'][0]['delta_db'] == 0
    assert campaign.campaign_status(directory)['best'] == before['best']
    saved = json.loads((confirm / 'confirmation.json').read_text())
    ids = [t['run_id'] for t in saved['tasks']]
    assert len(set(ids)) == 2
    assert main(['confirmation', 'next', '--confirmation-dir', str(confirm), '--wait']) == 0
    capsys.readouterr()
    assert [t['run_id'] for t in json.loads((confirm / 'confirmation.json').read_text())['tasks']] == ids


def test_profile_optin_stays_outside_scientific_config_and_reaches_runner(tmp_path):
    cfg = config(tmp_path)
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'profile_revision': 'p001', 'eval_size': 512,
        'samples': [{'sample_id': 'dev/a', 'split': 'DEV', 'image': 'a',
                     'input_relpath': 'a.png', 'scene_id': 'unknown', 'tags': ['fixed']}]}))
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory, profile_ref=profile)
    run = Path(cfg['runs_dir']) / state['trials'][0]['run_id']
    observed = json.loads((run / 'observation_config.json').read_text())
    commands = json.loads((run / 'commands.json').read_text())
    scientific = json.loads((run / 'config.json').read_text())
    assert observed['profile_ref'] == str(profile)
    assert commands['dev'][commands['dev'].index('--diagnostics-profile') + 1] == str(profile)
    assert 'profile_ref' not in scientific and 'control' not in scientific
    assert state['trials'][0]['valid']


def test_recovery_assembles_feedback_after_scientific_collection_was_saved(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory)
    path = directory / 'campaign.json'
    saved = json.loads(path.read_text())
    saved.pop('latest_feedback')
    for key in ('feedback_ref', 'feedback_revision', 'diagnostics_status', 'proposal_ready'):
        saved['trials'][0].pop(key, None)
    path.write_text(json.dumps(saved))
    recovered = campaign.campaign_status(directory)
    assert recovered['latest_feedback']['latest_run_id'] == state['trials'][0]['run_id']
    assert recovered['best'] == state['best'] and len(recovered['trials']) == 1


def test_control_authorized_final_test_bounded_once_and_accounts_failed_cost(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    path = cpu_control(tmp_path)
    control = json.loads(path.read_text()); control['test_permission'] = True
    path.write_text(json.dumps(control))
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=path)
    campaign.finalize_campaign(directory)
    calls = []
    def evaluate(config, checkpoint, split, output, *, execution_limits):
        calls.append((split, execution_limits))
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        data = {'valid': False, 'error': 'test timeout',
                'usage': {'gpu_hours': 0, 'allocated_gpus': 0, 'elapsed_seconds': 20}}
        (output / 'evaluation_result.json').write_text(json.dumps(data))
        # A timeout is terminal only after the runner confirms process release.
        (output / 'evaluation_state.json').write_text(json.dumps({**data, 'status': 'failed'}))
        (output / 'evaluation_health.json').write_text(json.dumps({'liveness': 'dead'}))
        raise RuntimeError('test timeout')
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', evaluate)
    report = campaign.final_test(directory)
    assert not report['valid'] and report['status'] == 'failed'
    assert calls == [(cfg['test'], {'allocated_gpus': 0, 'job_walltime_seconds': 20})]
    assert report['usage']['elapsed_seconds'] == 20
    assert campaign.campaign_status(directory)['usage']['final_test_gpu_hours'] == 0
    assert campaign.final_test(directory) == report
    assert len(calls) == 1


def test_control_final_test_real_success_and_interrupted_host_result_recovery(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    path = cpu_control(tmp_path)
    control = json.loads(path.read_text()); control['test_permission'] = True
    path.write_text(json.dumps(control))
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=path)
    before = campaign.finalize_campaign(directory)
    report = campaign.final_test(directory)
    assert report['valid'] and report['test_psnr'] == 20
    assert report['usage']['gpu_hours'] == 0
    assert campaign.final_test(directory) == report
    # Simulate host loss after the independent evaluator saved its result but
    # before campaign stored the terminal report and usage.
    state_path = directory / 'campaign.json'
    saved = json.loads(state_path.read_text())
    saved['final_test'] = {'status': 'running', 'valid': False, 'run_id': report['run_id']}
    saved.pop('usage')
    state_path.write_text(json.dumps(saved))
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', lambda *a, **k: pytest.fail('must not evaluate twice'))
    recovered = campaign.final_test(directory)
    assert recovered['valid'] and recovered['test_psnr'] == 20
    assert campaign.campaign_status(directory)['usage']['final_test_gpu_hours'] == 0
    assert campaign.campaign_status(directory)['best'] == before['best']
    (directory / 'final_test' / 'evaluation_result.json').unlink()
    saved['final_test']['status'] = 'running'
    state_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='liveness is unknown'):
        campaign.final_test(directory)


@pytest.mark.parametrize('stage', ['baseline', 'proposal'])
@pytest.mark.parametrize('gap', ['before_prepare', 'after_prepare'])
def test_durable_intent_recovers_exact_prepare_return_before_active_save(tmp_path, monkeypatch, stage, gap):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    if stage == 'proposal':
        campaign.initialize_campaign(cfg, directory)
        p = proposal(campaign.next_proposal(directory), {'loss_family': 'mse'})
    real_prepare = campaign.prepare_run
    prepared = []
    def interrupt(config, **kwargs):
        if gap == 'after_prepare':
            run_id = real_prepare(config, **kwargs)
            prepared.append(run_id)
        raise RuntimeError('host lost before active save')
    monkeypatch.setattr(campaign, 'prepare_run', interrupt)
    with pytest.raises(RuntimeError, match='host lost'):
        if stage == 'baseline':
            campaign.initialize_campaign(cfg, directory, wait=False)
        else:
            campaign.submit_proposal(directory, p, wait=False)
    saved = json.loads((directory / 'campaign.json').read_text())
    assert saved['launch_intent']['request_id']
    assert saved.get('active') is None
    monkeypatch.setattr(campaign, 'prepare_run', real_prepare)
    final = campaign.wait_campaign(directory, timeout=15)
    assert final['trials'][-1]['run_id'] == (prepared[0] if prepared else 'exp_001' if stage == 'baseline' else 'exp_002')
    assert len(list(Path(cfg['runs_dir']).glob('exp_*'))) == (1 if stage == 'baseline' else 2)
    assert final.get('launch_intent') is None


def test_optional_feedback_exception_has_native_observation_and_allows_control_proposal(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    control = cpu_control(tmp_path)
    monkeypatch.setattr(campaign, 'build_run_feedback', lambda *a: (_ for _ in ()).throw(ValueError('optional profile unavailable')))
    state = campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=control)
    evidence = campaign.next_proposal(directory)
    assert evidence['proposal_allowed']
    assert evidence['feedback']['proposal_ready']
    assert evidence['feedback']['diagnostics_status'] == 'unavailable'
    assert evidence['feedback']['observations'][0]['value'] == state['best']['dev_psnr']
    patch = {'loss_family': 'mse'}
    result = campaign.submit_proposal(directory, proposal(evidence, patch), decision_ref=decision_for(evidence, patch))
    assert result['trials'][-1]['valid']


@pytest.mark.parametrize('required', [[], ['sample_alignment']])
def test_same_run_feedback_revision_refresh_and_old_ack_rejected(tmp_path, required):
    from tm_research.evidence import build_run_feedback
    cfg = config(tmp_path)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    script.write_text(script.read_text().replace('image,psnr', 'sample_id,psnr'))
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5,
                                 control_ref=cpu_control(tmp_path, required=required))
    old = campaign.next_proposal(directory)
    assert old['proposal_allowed']
    (Path(cfg['runs_dir']) / 'exp_001' / 'dev' / 'per_image.csv').unlink()
    corrected = build_run_feedback(str(directory), 'exp_001')
    assert corrected['feedback_revision'] != old['feedback']['feedback_revision']
    new = campaign.next_proposal(directory)
    assert new['feedback']['feedback_revision'] == corrected['feedback_revision']
    patch = {'loss_family': 'mse'}
    if required:
        assert not new['proposal_allowed'] and new['next_action']['action'] == 'diagnose'
    else:
        with pytest.raises(ValueError, match='feedback_ref'):
            campaign.submit_proposal(directory, proposal(new, patch), decision_ref=decision_for(old, patch))
        latest_ack_old_observation = decision_for(new, patch)
        latest_ack_old_observation['observation_refs'] = decision_for(old, patch)['observation_refs']
        result = campaign.submit_proposal(directory, proposal(new, patch), decision_ref=latest_ack_old_observation)
        assert result['trials'][-1]['valid']


def test_unreadable_latest_feedback_never_reuses_stale_ready_cache(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=cpu_control(tmp_path))
    first = campaign.next_proposal(directory)
    Path(first['feedback']['feedback_ref']).unlink()
    latest = campaign.next_proposal(directory)
    assert not latest['proposal_allowed']
    assert latest['next_action']['action'] == 'diagnose'
    assert latest['feedback']['diagnostics_status'] == 'unavailable'
    assert latest['feedback']['unavailable']


def test_latest_detail_access_failure_is_diagnosed_then_rebuilt_without_scoring(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    state = campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=cpu_control(tmp_path))
    first = campaign.next_proposal(directory)
    Path(first['feedback']['detail_refs']['details_index']).unlink()
    unavailable = campaign.next_proposal(directory)
    assert not unavailable['proposal_allowed']
    assert unavailable['next_action']['action'] == 'diagnose'
    repaired = campaign.campaign_feedback(directory)
    assert repaired['feedback_revision'] != first['feedback']['feedback_revision']
    assert campaign.next_proposal(directory)['proposal_allowed']
    assert campaign.campaign_status(directory)['best'] == state['best']
    assert len(campaign.campaign_status(directory)['trials']) == 1


def test_recovered_pending_intent_obeys_new_hold_then_resumes_same_request(tmp_path, monkeypatch):
    from tm_research.evidence import build_run_feedback
    from tm_research.research import record_research_decision
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=cpu_control(tmp_path))
    first = campaign.next_proposal(directory)
    patch = {'loss_family': 'mse'}
    real_prepare = campaign.prepare_run
    prepared = []
    def interrupt(config, **kwargs):
        run_id = real_prepare(config, **kwargs); prepared.append(run_id)
        raise RuntimeError('host interrupted after prepare')
    monkeypatch.setattr(campaign, 'prepare_run', interrupt)
    with pytest.raises(RuntimeError, match='host interrupted'):
        campaign.submit_proposal(directory, proposal(first, patch), wait=False,
                                 decision_ref=decision_for(first, patch, name='fast_before_hold'))
    saved = json.loads((directory / 'campaign.json').read_text())
    token = saved['launch_intent']['request_id']
    hold = decision_for(first, {}, name='new_hold')
    hold.update(kind='slow_review', action='diagnose', protocol_suspect=True, reason='Check unchanged protocol')
    record_research_decision(directory, hold)
    monkeypatch.setattr(campaign, 'prepare_run', real_prepare)
    real_start = campaign.start_prepared_run
    monkeypatch.setattr(campaign, 'start_prepared_run', lambda *a: pytest.fail('must not start while held'))
    blocked = campaign.next_proposal(directory)
    assert blocked['next_action']['action'] == 'diagnose' and not blocked['proposal_allowed']
    assert blocked['launch_intent']['request_id'] == token
    assert not (Path(cfg['runs_dir']) / prepared[0] / 'launch.json').exists()
    (Path(cfg['runs_dir']) / 'exp_001' / 'dev' / 'per_image.csv').unlink()
    new_feedback = build_run_feedback(str(directory), 'exp_001')
    resolution = decision_for({**first, 'feedback': new_feedback}, {}, name='resolution')
    resolution.update(kind='resolution', action='diagnose', resolves_decision_id='new_hold',
                      resolution={'operational_issue': True, 'protocol_unchanged': True})
    record_research_decision(directory, resolution)
    monkeypatch.setattr(campaign, 'start_prepared_run', real_start)
    final = campaign.wait_campaign(directory, timeout=15)
    assert final['trials'][-1]['run_id'] == prepared[0]
    assert len(list(Path(cfg['runs_dir']).glob('exp_*'))) == 2
    assert final.get('launch_intent') is None


def test_recovered_prepared_active_obeys_hold_and_resumes_same_run_after_resolution(tmp_path, monkeypatch):
    from tm_research.evidence import build_run_feedback
    from tm_research.research import record_research_decision
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=cpu_control(tmp_path))
    first = campaign.next_proposal(directory)
    patch = {'loss_family': 'mse'}
    real_start = campaign.start_prepared_run
    monkeypatch.setattr(campaign, 'start_prepared_run', lambda *a: (_ for _ in ()).throw(RuntimeError('host lost after active save')))
    with pytest.raises(RuntimeError, match='host lost'):
        campaign.submit_proposal(directory, proposal(first, patch), wait=False,
                                 decision_ref=decision_for(first, patch, name='fast_before_active_hold'))
    saved = json.loads((directory / 'campaign.json').read_text())
    assert saved['active']['run_id'] == 'exp_002' and saved.get('launch_intent') is None
    hold = decision_for(first, {}, name='active_hold')
    hold.update(kind='slow_review', action='diagnose', protocol_suspect=True, reason='Check prepared launch protocol')
    record_research_decision(directory, hold)
    monkeypatch.setattr(campaign, 'start_prepared_run', lambda *a: pytest.fail('prepared active must not start under hold'))
    blocked = campaign.next_proposal(directory)
    assert blocked['active']['run_id'] == 'exp_002'
    assert blocked['next_action']['action'] == 'diagnose'
    assert not blocked['proposal_allowed'] and blocked['launch_block']
    assert not (Path(cfg['runs_dir']) / 'exp_002' / 'launch.json').exists()
    (Path(cfg['runs_dir']) / 'exp_001' / 'dev' / 'per_image.csv').unlink()
    corrected = build_run_feedback(str(directory), 'exp_001')
    resolution = decision_for({**first, 'feedback': corrected}, {}, name='active_resolution')
    resolution.update(kind='resolution', action='diagnose', resolves_decision_id='active_hold',
                      resolution={'operational_issue': True, 'protocol_unchanged': True})
    record_research_decision(directory, resolution)
    monkeypatch.setattr(campaign, 'start_prepared_run', real_start)
    final = campaign.wait_campaign(directory, timeout=15)
    assert final['trials'][-1]['run_id'] == 'exp_002'
    assert len(list(Path(cfg['runs_dir']).glob('exp_*'))) == 2


def test_recovered_prepared_active_rechecks_new_actual_resource_usage(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    control_path = cpu_control(tmp_path)
    control = json.loads(control_path.read_text())
    control['limits'].update(allocated_gpus=1, total_gpu_hours=2)
    control_path.write_text(json.dumps(control))
    directory = tmp_path / 'campaign'
    campaign.initialize_campaign(cfg, directory, max_trials=5, control_ref=control_path)
    evidence = campaign.next_proposal(directory)
    patch = {'loss_family': 'mse'}
    monkeypatch.setattr(campaign, 'start_prepared_run', lambda *a: (_ for _ in ()).throw(RuntimeError('host lost after active save')))
    with pytest.raises(RuntimeError, match='host lost'):
        campaign.submit_proposal(directory, proposal(evidence, patch), wait=False,
                                 decision_ref=decision_for(evidence, patch))
    # Another actual-cost record arrived while the host was absent. It remains
    # charged even though it leaves no room for the prepared search run.
    path = directory / 'campaign.json'
    state = json.loads(path.read_text())
    state['calibration_trials'] = [{'status': 'failed', 'usage': {'gpu_hours': 2}}]
    path.write_text(json.dumps(state))
    monkeypatch.setattr(campaign, 'start_prepared_run', lambda *a: pytest.fail('must recheck actual resource ledger'))
    blocked = campaign.next_proposal(directory)
    assert blocked['active']['run_id'] == 'exp_002'
    assert not blocked['proposal_allowed'] and not blocked['launch_block']['allowed']
    assert blocked['next_action']['action'] == 'report_stop'
    assert len(blocked['history']) == 1
    assert not (Path(cfg['runs_dir']) / 'exp_002' / 'launch.json').exists()
