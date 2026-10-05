"""Durable slow decisions exercise real record/next/submit and recovery gates."""
import json

import pytest

from tm_research import campaign, control, research
from test_campaign import proposal
from test_research import decision, feedback, write_campaign


def plateau(tmp_path):
    current = write_campaign(tmp_path)
    current['trials'][0]['result_status'] = 'no_gain'
    current['trials'].insert(0, {**current['trials'][0], 'run_id': 'exp_000'})
    current.update(active=None, frozen=None, max_trials=5, no_gain_limit=5,
                   no_gain_count=2, raw_best=current['best'], min_delta=0.01)
    for trial in current['trials']:
        trial.update(recipe={'loss_family': 'original', 'optimizer': 'adam',
                             'learning_rate': 0.0001, 'weight_decay': 0.01},
                     status='completed', decision='Measured no gain')
    current['best']['recipe'] = current['trials'][0]['recipe']
    save(tmp_path, current)
    return current


def save(directory, current):
    (directory / 'campaign.json').write_text(json.dumps(current))


def slow(current, action, *, name='slow_001', trigger=None):
    route = research.route_next_action(current, current['control'], feedback())
    return {**decision(), 'decision_id': name, 'kind': 'slow_review', 'action': action,
            'trigger_id': trigger or route['trigger_id'],
            'requested_change': {'learning_rate': 0.00005} if action == 'propose' else {}}


@pytest.mark.parametrize('action', ['report_stop', 'request_scope_change', 'finalize', 'diagnose'])
def test_slow_action_persists_across_next_restart_and_submit(tmp_path, action):
    current = plateau(tmp_path)
    d = slow(current, action)
    research.record_research_decision(tmp_path, d)
    first = campaign.next_proposal(tmp_path)
    assert first['next_action']['action'] == action
    assert not first['proposal_allowed']
    reloaded = json.loads((tmp_path / 'campaign.json').read_text())
    assert reloaded['pending_slow_decision']['decision_id'] == d['decision_id']
    assert campaign.next_proposal(tmp_path)['next_action']['action'] == action
    with pytest.raises(ValueError):
        campaign.submit_proposal(tmp_path, proposal(first, {'learning_rate': 0.00005}),
                                 decision_ref=decision())
    assert len(reloaded['trials']) == 2


@pytest.mark.parametrize('kind,trigger', [('slow_review', 'fixture/exp_000/valid_plateau'),
                                         ('slow_review', 'fixture/exp_001/evidence_conflict'),
                                         ('fast_proposal', 'fixture/exp_001/valid_plateau')])
def test_wrong_stale_or_fast_review_cannot_clear_plateau(tmp_path, kind, trigger):
    current = plateau(tmp_path)
    d = {**slow(current, 'propose', trigger=trigger), 'kind': kind}
    try:
        research.record_research_decision(tmp_path, d)
    except ValueError:
        pass
    event = campaign.next_proposal(tmp_path)
    assert not event['proposal_allowed']
    assert event['next_action']['action'] == 'diagnose'


def test_only_explicit_matching_slow_propose_resumes_pending_diagnosis(tmp_path):
    current = plateau(tmp_path)
    diagnosis = slow(current, 'diagnose')
    research.record_research_decision(tmp_path, diagnosis)
    current = json.loads((tmp_path / 'campaign.json').read_text())
    resume = slow(current, 'propose', name='resume')
    research.record_research_decision(tmp_path, resume)
    assert campaign.next_proposal(tmp_path)['proposal_allowed']
    research.record_research_decision(tmp_path, diagnosis)  # replay cannot restore the old diagnosis
    assert campaign.next_proposal(tmp_path)['proposal_allowed']
    assert len((tmp_path / 'research_decisions.jsonl').read_text().splitlines()) == 2
    with pytest.raises(ValueError, match='immutable'):
        research.record_research_decision(tmp_path, {**resume, 'prediction': 'Changed after the result'})


@pytest.mark.parametrize('action', ['report_stop', 'request_scope_change', 'finalize'])
def test_terminal_pending_decision_cannot_be_replaced_by_propose(tmp_path, action):
    current = plateau(tmp_path)
    research.record_research_decision(tmp_path, slow(current, action))
    current = json.loads((tmp_path / 'campaign.json').read_text())
    with pytest.raises(ValueError):
        research.record_research_decision(tmp_path, slow(current, 'propose', name='resume'))
    assert campaign.next_proposal(tmp_path)['next_action']['action'] == action


@pytest.mark.parametrize('boundary', ['hold', 'frozen', 'hard_stop', 'budget'])
def test_matching_review_cannot_override_other_boundaries(tmp_path, boundary):
    current = plateau(tmp_path)
    d = slow(current, 'propose')
    if boundary == 'hold':
        current['research_hold'] = {'decision_id': 'hold', 'evidence_refs': []}
    elif boundary == 'frozen':
        current['frozen'] = current['best']
    elif boundary == 'hard_stop':
        current['stop_reason'] = 'max_trials'
    else:
        current['trials'][0]['usage']['gpu_hours'] = 10
    save(tmp_path, current)
    with pytest.raises(ValueError):
        research.record_research_decision(tmp_path, d)
    assert not campaign.next_proposal(tmp_path)['proposal_allowed']


@pytest.mark.parametrize('action', ['report_stop', 'request_scope_change', 'finalize', 'diagnose'])
@pytest.mark.parametrize('pending_role', ['launch_intent', 'active'])
def test_recovered_unstarted_request_obeys_slow_decision(tmp_path, monkeypatch, action, pending_role):
    # Completed fixture records are local and deterministic. Only the native
    # launch boundary is substituted; persistence, collection and gates are real.
    current = plateau(tmp_path)
    patch = {'learning_rate': 0.00005}
    pending = {'index': 2, 'request_id': 'pending-native-request',
               'recipe': {**current['best']['recipe'], **patch},
               'proposal': proposal(campaign.next_proposal(tmp_path), patch),
               'run_id': 'exp_002', 'reserved_gpu_hours': 1}
    current[pending_role] = pending
    current['config']['runs_dir'] = str(tmp_path / 'runs')
    save(tmp_path, current)
    research.record_research_decision(tmp_path, slow(current, action))
    monkeypatch.setattr(campaign, 'prepare_run', lambda *a, **kw: pytest.fail('blocked intent must not prepare'))
    monkeypatch.setattr(campaign, 'start_prepared_run', lambda *a, **kw: pytest.fail('blocked request must not start'))
    monkeypatch.setattr(campaign, 'inspect_run', lambda *a: {'liveness': 'not_started'})
    reloaded = campaign.campaign_status(tmp_path)
    assert len(reloaded['trials']) == 2
    assert reloaded[pending_role]['request_id'] == pending['request_id']
    assert reloaded.get('launch_block')
    assert not campaign.next_proposal(tmp_path)['proposal_allowed']


def test_search_stop_allows_explicit_freeze_but_scope_change_blocks_confirmation(tmp_path):
    current = plateau(tmp_path)
    research.record_research_decision(tmp_path, slow(current, 'report_stop'))
    frozen = campaign.finalize_campaign(tmp_path)
    assert control.check_action(frozen['control'], frozen, 'confirm')['allowed']
    assert not control.check_action(frozen['control'], frozen, 'propose')['allowed']
    frozen['pending_slow_decision']['action'] = 'request_scope_change'
    assert not control.check_action(frozen['control'], frozen, 'confirm')['allowed']


def test_report_and_memory_expose_final_checkpoint_identity_without_selection(tmp_path):
    current = plateau(tmp_path)
    research.record_research_decision(tmp_path, slow(current, 'report_stop'))
    assert research.build_campaign_report(tmp_path)['final_checkpoint_ref']['status'] == 'unavailable'
    current = json.loads((tmp_path / 'campaign.json').read_text())
    ref = {'status': 'available', 'run_id': 'confirmation_seed_2', 'checkpoint': {'sha256': 'fixture-sha'}}
    current['final_checkpoint_ref'] = ref
    save(tmp_path, current)
    assert research.build_campaign_report(tmp_path)['final_checkpoint_ref'] == ref
    assert research.export_research_memory(tmp_path)[0]['final_checkpoint_ref'] == ref


def test_idempotent_replay_repairs_a_decision_saved_before_campaign_state(tmp_path, monkeypatch):
    current = plateau(tmp_path)
    d = slow(current, 'report_stop')
    real_save = campaign._save
    monkeypatch.setattr(campaign, '_save', lambda *a: (_ for _ in ()).throw(RuntimeError('interrupted state save')))
    with pytest.raises(RuntimeError, match='interrupted'):
        research.record_research_decision(tmp_path, d)
    monkeypatch.setattr(campaign, '_save', real_save)
    research.record_research_decision(tmp_path, d)
    assert campaign.next_proposal(tmp_path)['next_action']['action'] == 'report_stop'
    assert len((tmp_path / 'research_decisions.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('pending_action', ['report_stop', 'request_scope_change', 'finalize', 'diagnose', 'wait'])
def test_pending_decision_reason_applies_without_inferred_legacy_authority(pending_action):
    current = {'pending_slow_decision': {'action': pending_action, 'decision_id': 'legacy_review'}}
    assert control.pending_decision_reason(current, {'action': 'propose', 'recipe': {}})
    assert control.pending_decision_reason(current, 'finalize') is None
    assert control.pending_decision_reason({}, 'propose') is None
    current['frozen'] = {'run_id': 'fixture'}
    reason = control.pending_decision_reason(current, 'confirm')
    assert bool(reason) == (pending_action in ('request_scope_change', 'diagnose', 'wait'))


@pytest.mark.parametrize('action', ['report_stop', 'diagnose'])
def test_explicit_replay_repairs_original_slow_metadata_without_kind(tmp_path, action):
    current = plateau(tmp_path)
    d = slow(current, action)
    research.record_research_decision(tmp_path, d)
    original = json.loads((tmp_path / 'campaign.json').read_text())
    original.pop('pending_slow_decision')
    original['research_decisions'][0].pop('kind')
    original['handled_slow_triggers'] = [d['trigger_id']]
    save(tmp_path, original)
    research.record_research_decision(tmp_path, d)
    assert campaign.next_proposal(tmp_path)['next_action']['action'] == action
    assert json.loads((tmp_path / 'campaign.json').read_text())['pending_slow_decision']['action'] == action
    assert len((tmp_path / 'research_decisions.jsonl').read_text().splitlines()) == 1


def test_replaying_original_propose_cannot_clear_newer_pending_stop(tmp_path):
    current = plateau(tmp_path)
    old = slow(current, 'propose', name='old_propose')
    research.record_research_decision(tmp_path, old)
    current = json.loads((tmp_path / 'campaign.json').read_text())
    current['research_decisions'][0].pop('kind')
    save(tmp_path, current)
    research.record_research_decision(tmp_path, slow(current, 'report_stop', name='new_stop'))
    research.record_research_decision(tmp_path, old)
    assert campaign.next_proposal(tmp_path)['next_action']['action'] == 'report_stop'


@pytest.mark.parametrize('pending_role', ['launch_intent', 'active'])
def test_pending_continuation_must_approve_the_immutable_queued_patch(tmp_path, monkeypatch, pending_role):
    current = plateau(tmp_path)
    queued_patch = {'learning_rate': 0.00005}
    queued = {'index': 2, 'request_id': 'immutable-request', 'run_id': 'exp_002',
              'recipe': {**current['best']['recipe'], **queued_patch},
              'proposal': proposal(campaign.next_proposal(tmp_path), queued_patch),
              'reserved_gpu_hours': 1}
    current[pending_role] = queued
    current['config']['runs_dir'] = str(tmp_path / 'runs')
    save(tmp_path, current)
    research.record_research_decision(tmp_path, slow(current, 'diagnose'))
    monkeypatch.setattr(campaign, 'inspect_run', lambda *a: {'liveness': 'not_started'})
    blocked = campaign.next_proposal(tmp_path)
    trigger = blocked['next_action']['trigger_id']
    current = json.loads((tmp_path / 'campaign.json').read_text())
    wrong = slow(current, 'propose', name='wrong_continuation', trigger=trigger)
    wrong['requested_change'] = {'optimizer': 'adamw'}
    with pytest.raises(ValueError, match='queued|pending|immutable'):
        research.record_research_decision(tmp_path, wrong)
    still_pending = json.loads((tmp_path / 'campaign.json').read_text())
    assert still_pending['pending_slow_decision']['action'] == 'diagnose'
    assert still_pending[pending_role] == queued
    assert len((tmp_path / 'research_decisions.jsonl').read_text().splitlines()) == 1
    matching = slow(still_pending, 'propose', name='matching_continuation', trigger=trigger)
    matching['requested_change'] = queued_patch
    research.record_research_decision(tmp_path, matching)
    resumed = json.loads((tmp_path / 'campaign.json').read_text())
    assert campaign._current_start_allowed(tmp_path, resumed, pending_role)
    assert resumed[pending_role] == queued
    assert resumed['pending_slow_decision'] is None


def test_conflicting_terminal_review_is_rejected_atomically_before_explicit_freeze(tmp_path):
    current = plateau(tmp_path)
    stop = slow(current, 'report_stop', name='terminal_stop')
    research.record_research_decision(tmp_path, stop)
    current = json.loads((tmp_path / 'campaign.json').read_text())
    before_state = (tmp_path / 'campaign.json').read_bytes()
    before_log = (tmp_path / 'research_decisions.jsonl').read_bytes()
    scope = slow(current, 'request_scope_change', name='conflicting_scope')
    with pytest.raises(ValueError, match='terminal|Terminal|overwrite'):
        research.record_research_decision(tmp_path, scope)
    assert (tmp_path / 'campaign.json').read_bytes() == before_state
    assert (tmp_path / 'research_decisions.jsonl').read_bytes() == before_log
    assert not (tmp_path / 'decisions' / 'conflicting_scope.json').exists()
    frozen = campaign.finalize_campaign(tmp_path)
    assert frozen['pending_slow_decision']['decision_id'] == 'terminal_stop'
    assert control.check_action(frozen['control'], frozen, 'confirm')['allowed']
    archived = {**scope, 'kind': 'closure', 'decision_id': 'archive_scope_analysis'}
    research.record_research_decision(tmp_path, archived)
    after_archive = json.loads((tmp_path / 'campaign.json').read_text())
    assert after_archive['pending_slow_decision']['decision_id'] == 'terminal_stop'
    assert control.check_action(after_archive['control'], after_archive, 'confirm')['allowed']
