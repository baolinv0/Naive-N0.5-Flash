import json
from pathlib import Path

import pytest

from tm_research import research
from test_control import authorized_control


def state():
    return {'campaign_id': 'fixture', 'status': 'ready', 'stop_reason': None,
            'trials': [{'run_id': 'exp_001', 'valid': True, 'result_status': 'valid',
                        'dev_psnr': 20, 'protocol_id': 'fixture-p512', 'usage': {'gpu_hours': 0.1}}],
            'config': {'seed': 1}, 'best': {'run_id': 'exp_001', 'dev_psnr': 20},
            'control': authorized_control()}


def feedback():
    return {'campaign_id': 'fixture', 'run_id': 'exp_001', 'latest_run_id': 'exp_001',
            'feedback_revision': 'r001', 'feedback_ref': 'feedback/exp_001/r001/summary.json',
            'proposal_ready': True, 'observations': [{'id': 'O1', 'finding': 'Fixture measurement',
                                                    'source': 'paired_comparison.json#/finding'}]}


def decision():
    return {'schema_version': 1, 'decision_id': 'decision_001', 'kind': 'fast_proposal',
            'feedback_ref': feedback()['feedback_ref'], 'latest_seen_run_id': 'exp_001',
            'reference_run_ids': ['exp_001'], 'comparison_run_id': 'exp_001',
            'observation_refs': ['fixture/exp_001/r001/O1'], 'observation': 'Fixture measurement',
            'prediction': 'DEV improves', 'falsifier': 'DEV does not improve',
            'alternative_explanation': 'Training stochasticity', 'action': 'propose',
            'requested_change': {'learning_rate': 0.00005}, 'claim_scope': 'exploratory_dev'}


def test_latest_and_revision_observation_identity_are_verified(tmp_path):
    d = decision()
    s = write_campaign(tmp_path)
    actual = research._current_feedback(tmp_path, s)
    assert research.validate_research_decision(d, s, actual) == d
    for key, value in [('latest_seen_run_id', 'exp_000'),
                       ('observation_refs', ['fixture/exp_001/r002/O1']),
                       ('observation_refs', ['fixture/exp_001/r001/O9']),
                       ('observation_refs', ['other/exp_001/r001/O1']),
                       ('comparison_run_id', 'missing'), ('prediction', '')]:
        malformed = {**d, key: value}
        with pytest.raises(ValueError):
            research.validate_research_decision(malformed, s, actual)


def test_invalid_is_not_scientific_plateau_and_hardstop_dominates():
    s = state()
    s['trials'] += [{'run_id': 'exp_002', 'valid': False, 'result_status': 'invalid', 'usage': {'gpu_hours': 0.1}},
                    {'run_id': 'exp_003', 'valid': False, 'result_status': 'invalid', 'usage': {'gpu_hours': 0.1}}]
    s['no_gain_count'] = 2
    f = {**feedback(), 'latest_run_id': 'exp_003', 'run_id': 'exp_003'}
    assert not research.route_next_action(s, s['control'], f)['review_required']
    s['stop_reason'] = 'max_trials'
    assert research.route_next_action(s, s['control'], f)['action'] == 'report_stop'
    s['frozen'] = s['best']
    assert research.route_next_action(s, s['control'], f)['action'] == 'confirm'


def test_diagnose_precedes_plateau_and_review_is_idempotent():
    s = state()
    s['trials'] += [{'run_id': 'exp_002', 'valid': True, 'result_status': 'no_gain', 'usage': {'gpu_hours': 0.1}},
                    {'run_id': 'exp_003', 'valid': True, 'result_status': 'no_gain', 'usage': {'gpu_hours': 0.1}}]
    f = {**feedback(), 'latest_run_id': 'exp_003', 'proposal_ready': False}
    assert research.route_next_action(s, s['control'], f)['action'] == 'diagnose'
    f['proposal_ready'] = True
    first = research.route_next_action(s, s['control'], f)
    assert first['review_required']
    s['handled_slow_triggers'] = [first['trigger_id']]
    assert not research.route_next_action(s, s['control'], f)['review_required']


def write_campaign(tmp_path):
    s = state()
    (tmp_path / 'campaign.json').write_text(json.dumps(s))
    path = tmp_path / feedback()['feedback_ref']
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(feedback()))
    (path.parent / 'paired_comparison.json').write_text(json.dumps({'finding': 'Fixture measurement'}))
    (path.parent.parent / 'latest.json').write_text(json.dumps({'feedback_ref': feedback()['feedback_ref']}))
    return s


def test_record_decision_without_submission_is_idempotent_and_hold_needs_new_evidence(tmp_path):
    write_campaign(tmp_path)
    d = {**decision(), 'kind': 'slow_review', 'action': 'diagnose', 'protocol_suspect': True,
         'reason': 'Evaluate identity mapping', 'requested_change': {}}
    recorded = research.record_research_decision(tmp_path, d)
    research.record_research_decision(tmp_path, d)
    assert len((tmp_path / 'research_decisions.jsonl').read_text().splitlines()) == 1
    s = json.loads((tmp_path / 'campaign.json').read_text())
    assert s['research_hold']['decision_id'] == d['decision_id']
    resolve = {**d, 'decision_id': 'decision_002', 'protocol_suspect': False,
               'resolves_decision_id': d['decision_id'],
               'resolution': {'operational_issue': True, 'protocol_unchanged': True}}
    with pytest.raises(ValueError, match='new'):
        research.record_research_decision(tmp_path, resolve)
    f = {**feedback(), 'feedback_revision': 'r002', 'feedback_ref': 'feedback/exp_001/r002/summary.json'}
    path = tmp_path / f['feedback_ref']
    path.parent.mkdir()
    path.write_text(json.dumps(f))
    (path.parent / 'paired_comparison.json').write_text(json.dumps({'finding': 'New identity evidence'}))
    (path.parent.parent / 'latest.json').write_text(json.dumps({'feedback_ref': f['feedback_ref']}))
    resolve.update(feedback_ref=f['feedback_ref'], observation_refs=['fixture/exp_001/r002/O1'])
    s['stop_reason'] = 'max_trials'
    (tmp_path / 'campaign.json').write_text(json.dumps(s))
    research.record_research_decision(tmp_path, resolve)
    s = json.loads((tmp_path / 'campaign.json').read_text())
    assert not s['research_hold'] and s['stop_reason'] == 'max_trials'


def test_report_claim_levels_do_not_follow_score_or_unverified_images(tmp_path):
    write_campaign(tmp_path)
    report = research.build_campaign_report(tmp_path)
    assert report['claims']['W']['status'] == 'pending'
    assert report['claims']['Q']['status'] == 'exploratory_dev'
    assert report['claims']['R']['status'] == 'unavailable'
    assert report['claims']['P']['status'] == 'unavailable'
    assert report['cost']['inference_cost'] == 'unavailable'
    assert report['visual_claims'] == 'unavailable'
    assert (tmp_path / 'report.md').exists()


def test_memory_retains_contradiction_and_scope(tmp_path):
    write_campaign(tmp_path)
    d = {**decision(), 'action': 'report_stop', 'kind': 'slow_review',
         'memory_status': 'contradicted', 'contradicts_decision_id': 'old_decision'}
    research.record_research_decision(tmp_path, d)
    memory = research.export_research_memory(tmp_path)
    assert memory[0]['status'] == 'contradicted'
    assert memory[0]['contradicts_decision_id'] == 'old_decision'
    assert memory[0]['scope']['protocol_id'] == 'fixture-p512'
    assert memory[0]['alternative_explanation'] == d['alternative_explanation']
    assert research.export_research_memory(tmp_path) == memory
    packet = research.build_review_packet(tmp_path)
    assert packet['alternative_explanations'] and packet['evidence_refs']


def test_explicit_comparison_rejects_fixed_scientific_change(tmp_path):
    s = write_campaign(tmp_path)
    s['trials'][0]['resolved_config'] = {'seed': 8}
    with pytest.raises(ValueError, match='scientific'):
        research.validate_research_decision(decision(), s, research._current_feedback(tmp_path, s))


def test_q_claim_follows_external_frozen_confirmation_actual_pairs(tmp_path):
    from test_confirmation import setup_confirmation, finish
    module, search, destination, plan, control_path = setup_confirmation(tmp_path)
    control_data = json.loads(control_path.read_text())
    control_data['protocol_id'] = 'P512'
    control_path.write_text(json.dumps(control_data))
    module.initialize_confirmation(search, destination, plan, control_ref=control_path)
    finish(module, destination)
    q = research.build_campaign_report(search)['claims']['Q']
    assert q['status'] == 'supported_in_scope' and q['adaptive_dev_bias']
    manifest_path = destination / 'confirmation.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['tasks'][-1]['status'] = 'pending'
    manifest_path.write_text(json.dumps(manifest))
    cached_path = destination / 'report.json'
    cached = json.loads(cached_path.read_text())
    cached['pairs'] = cached['pairs'][:1]
    cached['planned_pairs'] = 1
    cached_path.write_text(json.dumps(cached))
    assert research.build_campaign_report(search)['claims']['Q']['status'] == 'inconclusive'


def test_q_rejects_unrelated_confirmation_receipt_and_changed_actual_protocol(tmp_path):
    from test_confirmation import setup_confirmation, finish
    module, search, destination, plan, control_path = setup_confirmation(tmp_path)
    control_data = json.loads(control_path.read_text()); control_data['protocol_id'] = 'P512'
    control_path.write_text(json.dumps(control_data))
    module.initialize_confirmation(search, destination, plan, control_ref=control_path)
    finish(module, destination)
    manifest = json.loads((destination / 'confirmation.json').read_text())
    task = manifest['tasks'][0]
    metrics_path = Path(task['run_dir']) / 'dev/metrics.json'
    metrics = json.loads(metrics_path.read_text()); metrics['protocol'] = 'wrong'
    metrics_path.write_text(json.dumps(metrics))
    assert research.build_campaign_report(search)['claims']['Q']['status'] == 'inconclusive'
    parent_path = search / 'campaign.json'
    parent = json.loads(parent_path.read_text())
    parent['confirmation_refs'][0]['parent_campaign'] = str(tmp_path / 'unrelated')
    parent_path.write_text(json.dumps(parent))
    assert research.build_campaign_report(search)['claims']['Q']['status'] != 'supported_in_scope'


def test_w_claim_requires_live_provenance_and_real_transcript(tmp_path):
    s = write_campaign(tmp_path)
    s['trials'] = [{**s['trials'][0], 'run_id': f'exp_{i:03d}'} for i in range(1, 5)]
    s['workflow_evidence'] = {'kind': 'live_naive_aris', 'model_identity': 'fixture-model',
        'run_ids': [t['run_id'] for t in s['trials']],
        'transcript_ref': str(tmp_path / 'transcript.jsonl'),
        'service_config_ref': str(tmp_path / 'service.json')}
    (tmp_path / 'campaign.json').write_text(json.dumps(s))
    assert research.build_campaign_report(tmp_path)['claims']['W']['status'] == 'pending'
    (tmp_path / 'transcript.jsonl').write_text('fixture-transcript\n')
    (tmp_path / 'service.json').write_text('{"fixture": true}')
    assert research.build_campaign_report(tmp_path)['claims']['W']['status'] == 'pending'
    s['trials'][-1]['valid'] = False
    (tmp_path / 'campaign.json').write_text(json.dumps(s))
    assert research.build_campaign_report(tmp_path)['claims']['W']['status'] == 'pending'


def test_report_and_memory_do_not_upgrade_an_unsupported_confirmed_claim(tmp_path):
    write_campaign(tmp_path)
    d = {**decision(), 'kind': 'closure', 'action': 'report_stop', 'memory_status': 'confirmed_in_scope'}
    research.record_research_decision(tmp_path, d)
    assert research.export_research_memory(tmp_path)[0]['status'] == 'tentative'


def test_fast_decision_id_cannot_overwrite_recorded_prediction(tmp_path):
    write_campaign(tmp_path)
    research.record_research_decision(tmp_path, decision())
    with pytest.raises(ValueError, match='immutable'):
        research.record_research_decision(tmp_path, {**decision(), 'prediction': 'different future'})
    assert json.loads((tmp_path / 'research_decisions.jsonl').read_text())['prediction'] == decision()['prediction']


def test_observation_acknowledgment_rejects_deleted_actual_source(tmp_path):
    s = write_campaign(tmp_path)
    actual = research._current_feedback(tmp_path, s)
    source = tmp_path / 'feedback/exp_001/r001/paired_comparison.json'
    source.unlink()
    with pytest.raises(ValueError, match='source|artifact|available'):
        research.validate_research_decision(decision(), s, actual)


def test_campaign_cost_reconciles_all_pools_and_rejects_negative_actual(tmp_path):
    s = write_campaign(tmp_path)
    s['usage'] = {'calibration_gpu_hours': 0.2, 'confirmation_gpu_hours': 0.3}
    s['final_test'] = {'status': 'failed', 'usage': {'gpu_hours': 0.4}}
    (tmp_path / 'campaign.json').write_text(json.dumps(s))
    cost = research.build_campaign_report(tmp_path)['cost']
    assert cost['gpu_hours'] == pytest.approx(1.0)
    assert cost['scope'] == 'search+confirmation+calibration+final_test'
    s['trials'][0]['usage']['gpu_hours'] = -1
    (tmp_path / 'campaign.json').write_text(json.dumps(s))
    assert research.build_campaign_report(tmp_path)['cost']['gpu_hours'] == 'unavailable'


def test_recognized_w_events_link_actual_runs_and_feedback_before_each_proposal(tmp_path):
    from test_runner import config
    from test_campaign import proposal
    from tm_research.campaign import initialize_campaign, next_proposal, submit_proposal
    directory = tmp_path / 'live_records'
    current = initialize_campaign(config(tmp_path), directory, max_trials=4)
    model, endpoint = 'Naive-N0.5-Flash', 'http://authorized-host:8000/v1'
    def event(kind, run_id, d=None):
        return {'schema_version': 1, 'event': kind, 'run_id': run_id,
                'campaign_id': current['campaign_id'], 'model_id': model, 'executor_url': endpoint,
                **({'decision_id': d['decision_id'], 'feedback_ref': d['feedback_ref'],
                    'observation_refs': d['observation_refs']} if d else {})}
    events = [event('run_completed', current['trials'][0]['run_id'])]
    for index, patch in enumerate([{'loss_family': 'mse'}, {'optimizer': 'adamw'}, {'learning_rate': 0.00005}], 1):
        evidence = next_proposal(directory)
        f = evidence['feedback']
        previous = current['trials'][-1]['run_id']
        d = {**decision(), 'decision_id': f'live_decision_{index}', 'feedback_ref': f['feedback_ref'],
             'latest_seen_run_id': previous, 'reference_run_ids': [previous], 'comparison_run_id': previous,
             'observation_refs': [f'{current["campaign_id"]}/{previous}/{f["feedback_revision"]}/O1'],
             'requested_change': patch}
        events.append(event('feedback_consumed', previous, d))
        current = submit_proposal(directory, proposal(evidence, patch), decision_ref=d)
        events += [event('proposal_submitted', current['trials'][-1]['run_id'], d),
                   event('run_completed', current['trials'][-1]['run_id'])]
    service = {'schema_version': 1, 'kind': 'naive_adapter_live', 'model_id': model,
               'weights_ref': 'authorized-model-weights', 'executor_url': endpoint,
               'aris_version': 'recorded-version', 'inference_settings': {'temperature': 0},
               'startup_record_ref': str(tmp_path / 'startup.json')}
    (tmp_path / 'startup.json').write_text(json.dumps({'event': 'service_started',
        **{key: service[key] for key in ('model_id', 'executor_url', 'weights_ref')}}))
    (tmp_path / 'service.json').write_text(json.dumps(service))
    transcript = tmp_path / 'transcript.jsonl'
    transcript.write_text('\n'.join(json.dumps(item) for item in events))
    current['workflow_evidence'] = {'kind': 'live_naive_aris', 'model_identity': model,
        'run_ids': [trial['run_id'] for trial in current['trials']],
        'transcript_ref': str(transcript), 'service_config_ref': str(tmp_path / 'service.json')}
    (directory / 'campaign.json').write_text(json.dumps(current))
    assert research.build_campaign_report(directory)['claims']['W']['status'] == 'accepted'
    service['model_id'] = None
    (tmp_path / 'service.json').write_text(json.dumps(service))
    startup = json.loads((tmp_path / 'startup.json').read_text()); startup['model_id'] = None
    (tmp_path / 'startup.json').write_text(json.dumps(startup))
    current['workflow_evidence']['model_identity'] = None
    (directory / 'campaign.json').write_text(json.dumps(current))
    for item in events:
        item['model_id'] = None
    transcript.write_text('\n'.join(json.dumps(item) for item in events))
    assert research.build_campaign_report(directory)['claims']['W']['status'] == 'pending'
    service['model_id'] = startup['model_id'] = current['workflow_evidence']['model_identity'] = model
    (tmp_path / 'service.json').write_text(json.dumps(service))
    (tmp_path / 'startup.json').write_text(json.dumps(startup))
    (directory / 'campaign.json').write_text(json.dumps(current))
    for item in events:
        item['model_id'] = model
    events[1], events[2] = events[2], events[1]
    transcript.write_text('\n'.join(json.dumps(item) for item in events))
    assert research.build_campaign_report(directory)['claims']['W']['status'] == 'pending'


@pytest.mark.parametrize('independent', [False, True])
def test_memory_confirmation_requires_association_to_the_frozen_experience(tmp_path, independent):
    from test_confirmation import setup_confirmation, finish
    module, search, destination, plan, control_path = setup_confirmation(tmp_path, independent=independent)
    control_data = json.loads(control_path.read_text()); control_data['protocol_id'] = 'P512'
    control_path.write_text(json.dumps(control_data))
    manifest = module.initialize_confirmation(search, destination, plan, control_ref=control_path)
    finish(module, destination)
    receipt = manifest['association']
    unrelated = {**decision(), 'decision_id': 'unrelated', 'memory_status': 'confirmed_in_scope'}
    claim_scope = 'independent_confirmation' if independent else 'original_dev_seed_stability'
    associated = {**unrelated, 'decision_id': 'confirmed_winner', 'claim_scope': claim_scope,
                  'kind': 'closure', 'action': 'report_stop', 'requested_change': manifest['plan']['arms']['winner']['recipe'],
                  'comparison_run_id': receipt['baseline_run_id'],
                  'confirmation_evidence': {'confirmation_dir': str(destination),
                       'winner_run_id': receipt['frozen_run_id'], 'baseline_run_id': receipt['baseline_run_id'],
                       'protocol_id': receipt['protocol_id']}}
    (search / 'research_decisions.jsonl').write_text('\n'.join(json.dumps(d) for d in (unrelated, associated)))
    memory = research.export_research_memory(search)
    assert [record['status'] for record in memory] == ['tentative', 'confirmed_in_scope']
    mismatched_claim = {**associated, 'decision_id': 'mismatched_claim',
                        'claim_scope': 'original_dev_seed_stability' if independent else 'independent_confirmation'}
    with (search / 'research_decisions.jsonl').open('a') as stream:
        stream.write('\n' + json.dumps(mismatched_claim))
    mismatched = research.export_research_memory(search)[-1]
    assert mismatched['status'] == 'tentative'
    assert memory[-1]['scope']['confirmation_kind'] == ('independent' if independent else 'original_dev')
    assert memory[-1]['scope']['confirmation_scope'] == manifest['plan']['scope']
    assert memory[-1]['scope']['confirmation_split'] == (manifest['evaluation_split_spec'] if independent else manifest['config']['validation'])
    assert memory[-1]['scope']['adaptive_dev_bias'] is (not independent)
    assert mismatched['scope']['claim_scope'] == mismatched_claim['claim_scope']
    assert mismatched['scope']['confirmation_kind'] == ('independent' if independent else 'original_dev')
    assert mismatched['scope']['interpretation_limits']
    assert json.loads((search / 'research_decisions.jsonl').read_text().splitlines()[-1]) == mismatched_claim
    unrelated_patch = {**associated, 'decision_id': 'unrelated_patch', 'requested_change': {'learning_rate': 0.2}}
    with (search / 'research_decisions.jsonl').open('a') as stream:
        stream.write('\n' + json.dumps(unrelated_patch))
    assert research.export_research_memory(search)[-1]['status'] == 'tentative'
