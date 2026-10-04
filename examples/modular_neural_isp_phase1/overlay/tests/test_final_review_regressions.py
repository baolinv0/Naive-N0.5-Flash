"""Whole-branch review regressions using real prepared runs and evaluator states."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_confirmation import setup_confirmation, finish
from test_research import decision
from tm_research import runner, campaign, research
from tm_research.control import check_action, reconcile_usage


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('window', ['prepared', 'checkpoint_frozen', 'normal_transition'])
@pytest.mark.parametrize('change', ['hold', 'config', 'active', 'cost'])
def test_current_parent_at_every_confirmation_start(tmp_path, monkeypatch, window, change):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=window != 'prepared')
    parent_path = search / 'campaign.json'
    def change_parent():
        parent = read(parent_path)
        if change == 'hold': parent['research_hold'] = {'decision_id': 'new-protocol-hold'}
        if change == 'config': parent['config']['epochs'] = 99
        if change == 'active': parent['final_test'] = {'status': 'running', 'reserved_gpu_hours': 0}
        if change == 'cost': parent['final_test'] = {'status': 'completed', 'usage': {'gpu_hours': 1}}
        write(parent_path, parent)
    original_parent = read(parent_path)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    start, save = runner.start_prepared_run, m._save
    if window == 'prepared':
        def interrupt(*a, **k): raise KeyboardInterrupt()
        monkeypatch.setattr(runner, 'start_prepared_run', interrupt)
    else:
        def freeze(directory, state):
            save(directory, state)
            if state['tasks'][0]['status'] == 'checkpoint_frozen':
                if window == 'normal_transition': change_parent()
                else: raise KeyboardInterrupt()
        monkeypatch.setattr(m, '_save', freeze)
    if window == 'normal_transition':
        state = m.run_confirmation_next(dest)
    else:
        with pytest.raises(KeyboardInterrupt): m.run_confirmation_next(dest)
        monkeypatch.setattr(runner, 'start_prepared_run', start)
        monkeypatch.setattr(m, '_save', save)
        change_parent()
        state = m.run_confirmation_next(dest)
    monkeypatch.setattr(m, '_save', save)
    assert state['status'] == 'blocked_parent'
    task = state['tasks'][0]
    assert task['status'] == ('prepared' if window == 'prepared' else 'checkpoint_frozen')
    assert not (Path(task['run_dir']) / ('launch.json' if window == 'prepared' else 'confirmation_eval/evaluation_state.json')).exists()
    frozen_checkpoint = task.get('checkpoint')
    parent = read(parent_path)
    for key in ('research_hold', 'final_test'): parent.pop(key, None)
    parent['config'] = original_parent['config']
    write(parent_path, parent)
    resumed = m.run_confirmation_next(dest)
    assert resumed['tasks'][0]['status'] == 'completed'
    assert resumed['tasks'][0]['run_id'] == task['run_id']
    if frozen_checkpoint: assert resumed['tasks'][0]['checkpoint'] == frozen_checkpoint
    assert len(list((dest / 'runs').glob('exp_*'))) == 1


@pytest.mark.parametrize('source', ['combined', 'separate', 'explicit_only', 'unknown', 'disjoint'])
def test_scene_independence_frozen_reports_and_memory(tmp_path, source):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    auth = read(control)
    training = {'sample_id': 'TRAIN/a', 'split': 'TRAIN', 'input_relpath': 'a.png', 'scene_id': 'same-capture'}
    confirm = {'sample_id': 'CONFIRM/a', 'split': 'CONFIRM', 'input_relpath': 'a.png',
               'scene_id': 'unknown' if source in ('unknown', 'explicit_only') else 'different' if source == 'disjoint' else 'same-capture'}
    profile = {'samples': [confirm], 'profile_revision': 'authorized-confirm'}
    if source == 'combined': profile['samples'].append(training)
    else:
        parent_profile = tmp_path / 'parent-profile.json'
        write(parent_profile, {'samples': [training], 'profile_revision': 'authorized-parent'})
        auth['observation_config']['profile_ref'] = str(parent_profile)
    if source in ('combined', 'explicit_only'):
        profile['scene_overlap'] = [{'scene_id': 'same-capture', 'sample_ids': ['TRAIN/a', 'CONFIRM/a'], 'splits': ['TRAIN', 'CONFIRM']}]
    profile_path = tmp_path / 'confirm-profile.json'; write(profile_path, profile)
    auth['confirmation_scope']['profile_ref'] = str(profile_path); write(control, auth)
    manifest = m.initialize_confirmation(search, dest, plan, control_ref=control)
    write(profile_path, {})  # authorized evidence is frozen before scores arrive
    report = finish(m, dest)
    conflict = source in ('combined', 'separate', 'explicit_only')
    assert report['conclusion'] == ('inconclusive' if conflict else 'supported_in_scope')
    assert report['mean_delta_db'] == 2 and report['complete_pairs'] == 2
    evidence = report['scene_independence']
    assert bool(evidence['conflicts']) is conflict
    assert evidence['status'] == ('conflict' if conflict else 'unknown' if source == 'unknown' else 'no_known_overlap')
    q = research.build_campaign_report(search)['claims']['Q']
    assert q['status'] == report['conclusion']
    assert q['scene_independence'] == evidence
    receipt = manifest['association']
    d = {**decision(), 'decision_id': 'confirmation', 'memory_status': 'confirmed_in_scope',
         'claim_scope': 'independent_confirmation', 'kind': 'closure', 'action': 'report_stop',
         'requested_change': plan.get('arms', manifest['plan']['arms'])['winner']['recipe'],
         'comparison_run_id': receipt['baseline_run_id'],
         'confirmation_evidence': {'confirmation_dir': str(dest), 'winner_run_id': receipt['frozen_run_id'],
                                  'baseline_run_id': receipt['baseline_run_id'], 'protocol_id': receipt['protocol_id']}}
    (search / 'research_decisions.jsonl').write_text(json.dumps(d))
    memory = research.export_research_memory(search)[0]
    assert memory['status'] == ('tentative' if conflict else 'confirmed_in_scope')
    assert memory['scope']['scene_independence'] == evidence
    if conflict: assert any('overlap' in s for s in memory['scope']['interpretation_limits'])


@pytest.mark.parametrize('allocated', [0, 1])
@pytest.mark.parametrize('stage', ['confirmation', 'final_test'])
@pytest.mark.parametrize('saved_receipt', [False, True])
def test_unresolved_evaluator_keeps_cross_stage_reservation_until_terminal(tmp_path, monkeypatch, allocated, stage, saved_receipt):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    auth = read(control)
    auth['test_permission'] = True
    auth['limits'].update(allocated_gpus=allocated, total_gpu_hours=10, confirmation_gpu_hours=1)
    write(control, auth); plan['budget_gpu_hours'] = 1
    parent = read(search / 'campaign.json')
    parent['control'] = auth
    for t in parent['trials']: t['usage'] = {'gpu_hours': 0}
    write(search / 'campaign.json', parent)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    def unresolved(*a, **k): raise runner._UnresolvedExecution('injected unconfirmed evaluator release')
    monkeypatch.setattr(runner, '_execute', unresolved)
    evaluate = runner.evaluate_frozen_checkpoint
    if saved_receipt:
        def interrupted(*a, **k):
            try: evaluate(*a, **k)
            except runner._UnresolvedExecution: raise KeyboardInterrupt()
        monkeypatch.setattr(runner if stage == 'confirmation' else campaign, 'evaluate_frozen_checkpoint', interrupted)
        with pytest.raises(KeyboardInterrupt):
            m.run_confirmation_next(dest) if stage == 'confirmation' else campaign.final_test(search)
    state = m.run_confirmation_next(dest) if stage == 'confirmation' else campaign.final_test(search)
    if stage == 'confirmation':
        task = state['tasks'][0]
        assert task['status'] == 'evaluating_confirmation' and state['active_task_key'] == task['task_key']
        output = Path(task['run_dir']) / 'confirmation_eval'
    else:
        assert state['status'] == 'recovery_required'
        output = search / 'final_test'
    assert read(output / 'evaluation_state.json')['status'] == 'recovery_required'
    parent = read(search / 'campaign.json')
    ledger = reconcile_usage(auth, parent)
    assert ledger['active_attempts']
    assert ledger['reserved_gpu_hours'] > 0 if allocated else ledger['reserved_gpu_hours'] == 0
    assert not check_action(auth, parent, 'confirm')['allowed']
    assert not check_action(auth, parent, 'final_test')['allowed']
    other = tmp_path / 'other-confirmation'
    m.initialize_confirmation(search, other, plan, control_ref=control)
    assert m.run_confirmation_next(other)['status'] in ('blocked', 'blocked_parent')
    assert not (other / 'runs').exists()
    # Let the real watchdog adopt a genuinely dead owner identity and confirm
    # release. The fixture injected uncertainty, not a live orphan or a retry.
    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.1)'])
    dead_identity = runner._process_identity(owner.pid)
    owner.wait(timeout=3)
    evaluator_state = read(output / 'evaluation_state.json')
    runner._write_json(output / 'evaluation_state.json', {**evaluator_state, 'owner_identity': dead_identity})
    deadline = time.monotonic() + 3
    while read(output / 'evaluation_state.json')['status'] != 'failed' and time.monotonic() < deadline:
        time.sleep(.02)
    assert read(output / 'evaluation_state.json')['status'] == 'failed'
    assert read(output / 'evaluation_health.json')['liveness'] == 'dead'
    receipt = read(output / 'evaluation_result.json')
    assert receipt['usage']['gpu_hours'] is not None
    if allocated: assert receipt['usage']['gpu_hours'] > 0
    def duplicate(*a, **k): pytest.fail('must adopt same attempt without evaluator retry')
    monkeypatch.setattr(runner, 'evaluate_frozen_checkpoint', duplicate)
    monkeypatch.setattr(campaign, 'evaluate_frozen_checkpoint', duplicate)
    recovered = m.run_confirmation_next(dest) if stage == 'confirmation' else campaign.final_test(search)
    if stage == 'confirmation':
        assert recovered['tasks'][0]['status'] == 'failed' and recovered['active_task_key'] is None
        assert recovered['tasks'][0]['run_id'] == task['run_id']
    else: assert recovered['status'] == 'failed'
    ledger = reconcile_usage(auth, read(search / 'campaign.json'))
    assert ledger['known'] and not ledger['active_attempts']
    assert ledger['confirmation_gpu_hours' if stage == 'confirmation' else 'final_test_gpu_hours'] >= receipt['usage']['gpu_hours']
    assert check_action(auth, read(search / 'campaign.json'), 'confirm')['allowed']


def test_terminal_confirmation_collection_is_allowed_under_parent_hold(tmp_path, monkeypatch):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    evaluate = runner.evaluate_frozen_checkpoint
    def completed_then_interrupted(*args, **kwargs):
        evaluate(*args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, 'evaluate_frozen_checkpoint', completed_then_interrupted)
    with pytest.raises(KeyboardInterrupt): m.run_confirmation_next(dest)
    parent = read(search / 'campaign.json'); parent['research_hold'] = {'decision_id': 'new-hold'}
    write(search / 'campaign.json', parent)
    state = m.run_confirmation_next(dest)
    assert state['tasks'][0]['status'] == 'completed' and state['active_task_key'] is None
    assert m.run_confirmation_next(dest)['status'] == 'blocked_parent'


@pytest.mark.parametrize('window', ['prepared', 'checkpoint_frozen'])
def test_confirmation_start_excludes_only_own_reservation_and_keeps_actual_cost(tmp_path, monkeypatch, window):
    m, search, dest, plan, control = setup_confirmation(tmp_path, independent=True)
    auth = read(control); auth['limits'].update(allocated_gpus=1, total_gpu_hours=10, confirmation_gpu_hours=1)
    write(control, auth); plan['budget_gpu_hours'] = 1
    parent = read(search / 'campaign.json')
    for trial in parent['trials']: trial['usage'] = {'gpu_hours': 0}
    write(search / 'campaign.json', parent)
    m.initialize_confirmation(search, dest, plan, control_ref=control)
    other = tmp_path / 'other'
    other_state = m.initialize_confirmation(search, other, plan, control_ref=control)
    start, save = runner.start_prepared_run, m._save
    if window == 'prepared':
        def interrupt(*a, **k): raise KeyboardInterrupt()
        monkeypatch.setattr(runner, 'start_prepared_run', interrupt)
    else:
        def freeze(directory, state):
            save(directory, state)
            if state['tasks'][0]['status'] == 'checkpoint_frozen': raise KeyboardInterrupt()
        monkeypatch.setattr(m, '_save', freeze)
    with pytest.raises(KeyboardInterrupt): m.run_confirmation_next(dest)
    monkeypatch.setattr(runner, 'start_prepared_run', start); monkeypatch.setattr(m, '_save', save)
    # Same task key in another registered directory must not be excluded.
    other_state['tasks'][0].update(status='prepared', reserved_gpu_hours=.01)
    other_state['active_task_key'] = other_state['tasks'][0]['task_key']
    write(other / 'confirmation.json', other_state)
    blocked = m.run_confirmation_next(dest)
    assert blocked['status'] == 'blocked_parent'
    assert blocked['parent_usage']['active_attempts'] == ['confirmation:baseline:3:1']
    assert blocked['parent_usage']['reserved_gpu_hours'] == .01
    if window == 'checkpoint_frozen':
        assert blocked['parent_usage']['confirmation_gpu_hours'] == blocked['tasks'][0]['result']['usage']['gpu_hours'] > 0
    other_state['tasks'][0].update(status='pending'); other_state['active_task_key'] = None
    write(other / 'confirmation.json', other_state)
    resumed = m.run_confirmation_next(dest)
    assert resumed['tasks'][0]['status'] == 'completed'
    assert len(list((dest / 'runs').glob('exp_*'))) == 1
