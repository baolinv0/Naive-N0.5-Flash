"""Scientific authority stays predeclared and native evidence governs reports."""
import json
from pathlib import Path

import pytest

from test_confirmation import finish, setup_confirmation
from test_research import feedback, write_campaign
from tm_research import campaign, research


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value))


def test_primary_plan_is_bound_before_launch(tmp_path):
    module, search, first, plan, control = setup_confirmation(tmp_path)
    module.initialize_confirmation(search, first, plan, control_ref=control)
    parent = read(search / 'campaign.json')
    primary = parent.get('primary_confirmation')
    assert primary and primary['association'] == parent['confirmation_refs'][0]
    assert primary['plan'] == read(first / 'confirmation.json')['plan']
    assert not (first / 'runs').exists()


def test_later_positive_plan_cannot_supersede_negative_primary_evidence(tmp_path):
    module, search, first, plan, control = setup_confirmation(tmp_path)
    module.initialize_confirmation(search, first, {**plan, 'seeds': [3]}, control_ref=control)
    primary = read(search / 'campaign.json').get('primary_confirmation')
    evaluator = tmp_path / 'upstream/photofinishing/test.py'
    source = evaluator.read_text()
    evaluator.write_text(source.replace("if mode=='nonfinite':", "if recipe['loss_family']=='mse': score=18.\nif mode=='nonfinite':"))
    assert finish(module, first, count=2)['conclusion'] == 'not_supported'
    evaluator.write_text(source)
    second = tmp_path / 'confirmation-later'
    module.initialize_confirmation(search, second, {**plan, 'seeds': [9]}, control_ref=control)
    assert finish(module, second, count=2)['conclusion'] == 'supported_in_scope'
    q = research.build_campaign_report(search)['claims']['Q']
    assert q['status'] == 'not_supported'
    assert read(search / 'campaign.json')['primary_confirmation'] == primary
    assert q['confirmation_ref'] == str(first)
    assert [(item['role'], item['status'], item['complete_pairs'], item['planned_pairs'])
            for item in q['confirmation_history']] == [
                ('primary', 'not_supported', 1, 1), ('exploratory', 'supported_in_scope', 1, 1)]


def test_later_checkpoint_rule_cannot_override_primary_retain_rule(tmp_path):
    module, search, first, plan, control = setup_confirmation(tmp_path)
    module.initialize_confirmation(search, first, {**plan, 'seeds': [3]}, control_ref=control)
    later = tmp_path / 'later-seed-checkpoint'
    module.initialize_confirmation(search, later, {**plan, 'seeds': [9],
        'final_checkpoint_rule': {'kind': 'predeclared_seed', 'seed': 9}}, control_ref=control)
    finish(module, later, count=2)
    with pytest.raises(ValueError, match='primary'):
        campaign.resolve_final_checkpoint(search, confirmation_dir=later)
    reference = campaign.resolve_final_checkpoint(search)
    assert reference['kind'] == 'retain_search_checkpoint'
    assert reference['run_id'] == read(search / 'campaign.json')['frozen']['run_id']


def test_legacy_multiple_plans_without_preregistration_fail_closed_even_with_explicit_selection(tmp_path):
    module, search, first, plan, control = setup_confirmation(tmp_path)
    for destination, seed in ((first, 3), (tmp_path / 'later', 9)):
        module.initialize_confirmation(search, destination, {**plan, 'seeds': [seed]}, control_ref=control)
        finish(module, destination, count=2)
    parent = read(search / 'campaign.json')
    parent.pop('primary_confirmation', None)
    save(search / 'campaign.json', parent)
    q = research.build_campaign_report(search)['claims']['Q']
    assert q['status'] == 'inconclusive' and 'ambiguous' in q['reason'].lower()
    assert len(q['confirmation_history']) == 2
    for explicit in (None, first, tmp_path / 'later'):
        with pytest.raises(ValueError, match='ambiguous'):
            campaign.resolve_final_checkpoint(search, confirmation_dir=explicit)


@pytest.mark.parametrize('independent', [False, True])
def test_confirmation_report_revalidates_native_protocol_instead_of_completed_score(tmp_path, independent):
    module, search, destination, plan, control = setup_confirmation(tmp_path, independent=independent)
    module.initialize_confirmation(search, destination, {**plan, 'seeds': [3]}, control_ref=control)
    assert finish(module, destination, count=2)['conclusion'] == 'supported_in_scope'
    task = read(destination / 'confirmation.json')['tasks'][0]
    metrics_path = Path(task['run_dir']) / ('confirmation_eval' if independent else 'dev') / 'metrics.json'
    metrics = read(metrics_path); metrics['protocol'] = 'wrong'
    save(metrics_path, metrics)
    report = module.report_confirmation(destination)
    assert report['conclusion'] == 'inconclusive'
    assert report['complete_pairs'] == 0
    assert report['pairs'][0]['evidence_errors']
    assert research.build_campaign_report(search)['claims']['Q']['status'] == 'inconclusive'


def test_missing_optional_csv_preserves_native_confirmation_claim(tmp_path):
    module, search, destination, plan, control = setup_confirmation(tmp_path)
    module.initialize_confirmation(search, destination, {**plan, 'seeds': [3]}, control_ref=control)
    finish(module, destination, count=2)
    for task in read(destination / 'confirmation.json')['tasks']:
        Path(task['result']['artifacts']['per_image_csv']).unlink()
    report = module.report_confirmation(destination)
    assert report['conclusion'] == 'supported_in_scope'
    assert report['pairs'][0]['per_image']['status'] == 'unavailable'
    assert research.build_campaign_report(search)['claims']['Q']['status'] == 'supported_in_scope'


def test_one_corrupt_native_pair_preserves_valid_pair_coverage_in_both_reports(tmp_path):
    module, search, destination, plan, control = setup_confirmation(tmp_path)
    module.initialize_confirmation(search, destination, plan, control_ref=control)
    finish(module, destination)
    task = read(destination / 'confirmation.json')['tasks'][0]
    metrics_path = Path(task['run_dir']) / 'dev/metrics.json'
    metrics = read(metrics_path); metrics['protocol'] = 'wrong'; save(metrics_path, metrics)
    report = module.report_confirmation(destination)
    q = research.build_campaign_report(search)['claims']['Q']
    assert report['conclusion'] == q['status'] == 'inconclusive'
    assert report['complete_pairs'] == q['complete_pairs'] == 1
    assert report['planned_pairs'] == q['planned_pairs'] == 2
    for pairs in (report['pairs'], q['confirmation_pairs']):
        assert pairs[0]['status'] == 'incomplete' and pairs[0]['evidence_errors']
        assert pairs[1]['status'] == 'complete' and pairs[1]['delta_db'] == 2.


def test_invalid_native_predeclared_checkpoint_remains_reportable_and_cannot_be_selected(tmp_path):
    module, search, destination, plan, control = setup_confirmation(tmp_path)
    plan.update(seeds=[9], final_checkpoint_rule={'kind': 'predeclared_seed', 'seed': 9})
    module.initialize_confirmation(search, destination, plan, control_ref=control)
    finish(module, destination, count=2)
    task = next(task for task in read(destination / 'confirmation.json')['tasks'] if task['arm'] == 'winner')
    metrics_path = Path(task['run_dir']) / 'dev/metrics.json'
    metrics = read(metrics_path); metrics['protocol'] = 'wrong'; save(metrics_path, metrics)
    report = module.report_confirmation(destination)
    assert report['conclusion'] == 'inconclusive'
    assert not report['final_checkpoint']['available']
    assert report['final_checkpoint']['reason']
    assert report['pairs'][0]['evidence_errors']
    with pytest.raises(ValueError):
        campaign.resolve_final_checkpoint(search)


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'reuse_run'])
def test_confirmation_report_rejects_task_inventory_that_q_rejects(tmp_path, change):
    module, search, destination, plan, control = setup_confirmation(tmp_path)
    module.initialize_confirmation(search, destination, plan, control_ref=control)
    finish(module, destination)
    path = destination / 'confirmation.json'; manifest = read(path)
    if change == 'duplicate':
        manifest['tasks'].append(dict(manifest['tasks'][0]))
    elif change == 'missing':
        manifest['tasks'].pop()
    else:
        manifest['tasks'][2].update(run_id=manifest['tasks'][0]['run_id'],
                                  run_dir=manifest['tasks'][0]['run_dir'])
    save(path, manifest)
    assert research.build_campaign_report(search)['claims']['Q']['status'] == 'inconclusive'
    report = module.report_confirmation(destination)
    assert report['conclusion'] == 'inconclusive'
    assert report['evidence_errors']
    assert report['planned_pairs'] == 2


@pytest.mark.parametrize('newer_revision', [False, True])
def test_removed_published_pointer_cannot_reactivate_cached_revision(tmp_path, newer_revision):
    state = write_campaign(tmp_path)
    state['latest_feedback'] = feedback()
    if newer_revision:
        newer = tmp_path / 'feedback/exp_001/r002/summary.json'
        newer.parent.mkdir()
        save(newer, {**feedback(), 'feedback_revision': 'r002', 'feedback_ref': str(newer)})
    (tmp_path / 'feedback/exp_001/latest.json').unlink()
    with pytest.raises(ValueError, match='feedback|Feedback'):
        research._current_feedback(tmp_path, state)


@pytest.mark.parametrize('field,wrong', [('in_size', 256), ('validation_frequency', 5)])
def test_default_canonicalization_preserves_true_native_config_mismatch(tmp_path, field, wrong):
    module, search, destination, plan, control = setup_confirmation(tmp_path)
    plan['final_checkpoint_rule'] = {'kind': 'predeclared_seed', 'seed': 9}
    module.initialize_confirmation(search, destination, plan, control_ref=control)
    report = finish(module, destination)
    selected = next(task for task in report['tasks'] if task['task_key'] == 'winner:9:1')
    cfg_path = Path(selected['run_dir']) / 'config.json'
    cfg = read(cfg_path); cfg[field] = wrong; save(cfg_path, cfg)
    with pytest.raises(ValueError, match='scientific config'):
        campaign.resolve_final_checkpoint(search)


@pytest.mark.parametrize('index', [{}, {'feedback_revision': None}, [], 'unreadable'])
def test_published_feedback_index_cannot_fall_back_to_stale_cached_feedback(tmp_path, index):
    state = write_campaign(tmp_path)
    state['latest_feedback'] = feedback()
    latest = tmp_path / 'feedback/exp_001/latest.json'
    latest.write_text('{' if index == 'unreadable' else json.dumps(index))
    with pytest.raises(ValueError, match='feedback|Feedback'):
        research._current_feedback(tmp_path, state)


@pytest.mark.parametrize('change', ['index_revision', 'index_run', 'summary_revision',
                                  'summary_run', 'summary_latest_run', 'summary_pointer'])
def test_feedback_pointer_rejects_contradictory_revision_run_or_summary_identity(tmp_path, change):
    state = write_campaign(tmp_path)
    summary_path = tmp_path / feedback()['feedback_ref']
    summary = read(summary_path)
    index = {'feedback_ref': feedback()['feedback_ref'], 'feedback_revision': 'r001',
             'run_id': 'exp_001'}
    if change == 'index_revision': index['feedback_revision'] = 'r002'
    if change == 'index_run': index['run_id'] = 'exp_000'
    if change == 'summary_revision': summary['feedback_revision'] = 'r002'
    if change == 'summary_run': summary['run_id'] = 'exp_000'
    if change == 'summary_latest_run': summary['latest_run_id'] = 'exp_000'
    if change == 'summary_pointer': summary['feedback_ref'] = 'feedback/exp_001/r002/summary.json'
    save(summary_path, summary)
    save(tmp_path / 'feedback/exp_001/latest.json', index)
    with pytest.raises(ValueError, match='feedback|Feedback'):
        research._current_feedback(tmp_path, state)
    campaign._refresh_latest_feedback(tmp_path, state)
    assert state['latest_feedback']['proposal_ready'] is False
    assert state['latest_feedback']['unavailable']


def test_feedback_ref_only_legacy_index_keeps_matching_summary_usable(tmp_path):
    state = write_campaign(tmp_path)
    summary = research._current_feedback(tmp_path, state)
    assert summary['feedback_revision'] == 'r001'
    assert summary['run_id'] == 'exp_001'


@pytest.mark.parametrize('field,value', [('frozen', None), ('frozen', []),
                                        ('checkpoint', None), ('checkpoint', [])])
def test_malformed_independent_native_frozen_evidence_is_reported_and_checkpoint_refused(tmp_path, field, value):
    module, search, destination, plan, control = setup_confirmation(tmp_path, independent=True)
    plan.update(seeds=[9], final_checkpoint_rule={'kind': 'predeclared_seed', 'seed': 9})
    module.initialize_confirmation(search, destination, plan, control_ref=control)
    finish(module, destination, count=2)
    task = next(task for task in read(destination / 'confirmation.json')['tasks'] if task['arm'] == 'winner')
    native_path = Path(task['run_dir']) / 'confirmation_eval/evaluation_state.json'
    native = read(native_path)
    if field == 'frozen': native['frozen'] = value
    else: native['frozen']['checkpoint'] = value
    save(native_path, native)
    report = module.report_confirmation(destination)
    q = research.build_campaign_report(search)['claims']['Q']
    assert report['conclusion'] == q['status'] == 'inconclusive'
    assert report['pairs'][0]['evidence_errors']
    assert q['confirmation_pairs'][0]['evidence_errors']
    assert field in ' '.join(report['pairs'][0]['evidence_errors']).lower()
    assert not report['final_checkpoint']['available']
    with pytest.raises(ValueError, match='frozen|checkpoint'):
        campaign.resolve_final_checkpoint(search)


@pytest.mark.parametrize('field,value', [('in_size', 256), ('validation_frequency', 5)])
@pytest.mark.parametrize('rewrite_snapshot', [False, True])
def test_editing_manifest_and_native_configs_cannot_rewrite_frozen_parent_budget(tmp_path, field, value, rewrite_snapshot):
    module, search, destination, plan, control = setup_confirmation(tmp_path)
    module.initialize_confirmation(search, destination, {**plan, 'seeds': [3]}, control_ref=control)
    finish(module, destination, count=2)
    path = destination / 'confirmation.json'; manifest = read(path)
    manifest['config'][field] = value
    if rewrite_snapshot:
        manifest['parent_identity']['config'][field] = value
    for task in manifest['tasks']:
        config_path = Path(task['run_dir']) / 'config.json'
        config = read(config_path); config[field] = value; save(config_path, config)
    save(path, manifest)
    report = module.report_confirmation(destination)
    q = research.build_campaign_report(search)['claims']['Q']
    assert report['conclusion'] == q['status'] == 'inconclusive'
    assert report['complete_pairs'] == 0
    if not rewrite_snapshot:
        assert q['complete_pairs'] == 0
    assert 'frozen parent' in ' '.join(report['pairs'][0]['evidence_errors']).lower()
