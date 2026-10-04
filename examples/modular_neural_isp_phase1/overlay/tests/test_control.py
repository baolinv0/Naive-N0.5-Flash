import json

import pytest

from tm_research import control


def authorized_control():
    return {'schema_version': 1, 'campaign_id': 'fixture', 'parent_campaign_id': None,
            'question': 'Fixture recipe effect', 'authorization_ref': 'fixture-only-authorization',
            'source_revision': 'fixture-source', 'approved_config_ref': 'fixture-config.json',
            'protocol_id': 'fixture-p512', 'primary_metric': 'mean_per_image_rgb_psnr',
            'allowed_recipe_fields': ['loss_family', 'optimizer', 'learning_rate', 'weight_decay'],
            'limits': {'job_walltime_seconds': 3600, 'total_gpu_hours': 10,
                       'allocated_gpus': 1, 'max_search_trials': 5,
                       'confirmation_gpu_hours': 3, 'calibration_gpu_hours': 1},
            'slow_policy': {'consecutive_valid_no_gain': 2},
            'confirmation_scope': {'authorized': True, 'kind': 'original_dev', 'seeds': [1, 2]},
            'test_permission': False, 'observation_mode': 'text',
            'observation_config': {'required_for_proposal': ['summary']}}


def test_missing_authorization_and_resource_values_rejected(tmp_path):
    for key in ['authorization_ref', 'approved_config_ref']:
        data = authorized_control()
        data[key] = None
        path = tmp_path / 'control.json'
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match=key):
            control.load_control(path)
    for value in [None, float('nan'), -1, True]:
        data = authorized_control()
        data['limits']['total_gpu_hours'] = value
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match='total_gpu_hours'):
            control.load_control(path)


def test_search_reserves_confirmation_calibration_and_active_jobs():
    data = authorized_control()
    decision = control.check_action(data, {}, 'propose', {'gpu_hours': 5, 'reserved_gpu_hours': 1})
    assert not decision['allowed']
    assert decision['remaining_budget']['search_gpu_hours'] == 0
    assert control.check_action(data, {}, 'propose', {'gpu_hours': 5})['allowed']
    assert not control.check_action(data, {}, 'propose', {'gpu_hours': 5.01})['allowed']


def test_hard_stop_frozen_test_and_scope_are_not_overridden():
    data = authorized_control()
    for state in [{'stop_reason': 'max_trials'}, {'frozen': {'run_id': 'r'}},
                  {'status': 'running', 'active': {'run_id': 'r'}}, {'research_hold': {'reason': 'suspect'}}]:
        assert not control.check_action(data, state, 'propose', {})['allowed']
    assert not control.check_action(data, {}, 'final_test', {})['allowed']
    assert not control.check_action(data, {}, {'action': 'propose', 'recipe': {'model': 'new'}}, {})['allowed']


def test_resource_accounting_includes_failed_usage_and_keeps_unknown_reservation():
    data = authorized_control()
    state = {'trials': [{'valid': False, 'usage': {'gpu_hours': 5.1}}]}
    assert not control.check_action(data, state, 'propose')['allowed']
    state['trials'][0]['usage'] = {'gpu_hours': None}
    result = control.check_action(data, state, 'propose')
    assert not result['allowed'] and 'unknown' in result['reason']


def test_control_load_is_snapshot_and_zero_gpu_cpu_is_explicit(tmp_path):
    data = authorized_control()
    data['limits'].update(allocated_gpus=0, total_gpu_hours=0,
                          confirmation_gpu_hours=0, calibration_gpu_hours=0)
    data['confirmation_scope'] = {'authorized': False, 'kind': 'none'}
    path = tmp_path / 'control.json'
    path.write_text(json.dumps(data))
    snapshot = control.load_control(path)
    assert snapshot == data
    assert control.check_action(snapshot, {}, 'baseline')['allowed']
    assert not control.check_action(snapshot, {}, 'confirm')['allowed']


def test_load_does_not_assume_image_capability(tmp_path):
    data = authorized_control()
    data['observation_mode'] = 'image'
    path = tmp_path / 'control.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='visual_capability'):
        control.load_control(path)


def test_calibration_cannot_spend_confirmation_reserve():
    data = authorized_control()
    # Search exceeded its pool: confirmation's three hours remain protected.
    assert not control.check_action(data, {}, 'calibrate', {'gpu_hours': 7.5})['allowed']
    data['limits'].update(allocated_gpus=0, total_gpu_hours=0,
                          confirmation_gpu_hours=0, calibration_gpu_hours=0)
    assert not control.check_action(data, {}, 'baseline', {'gpu_hours': 0.01})['allowed']


def test_partial_category_summary_cannot_erase_measured_failed_search():
    data = authorized_control()
    state = {'trials': [{'valid': False, 'usage': {'gpu_hours': 7}}],
             'usage': {'calibration_gpu_hours': 0.1}}
    gate = control.check_action(data, state, 'propose')
    assert not gate['allowed']
    assert gate['remaining_budget']['total_gpu_hours'] == pytest.approx(2.9)
    ledger = control.reconcile_usage(data, state)
    assert ledger['gpu_hours'] == pytest.approx(7.1)


def test_partial_or_complete_summary_never_hides_unknown_gpu_actual():
    data = authorized_control()
    state = {'trials': [{'valid': False, 'usage': {'gpu_hours': None}}]}
    for usage in [{'calibration_gpu_hours': 0.1}, {'gpu_hours': 0}, {'search_gpu_hours': 0}]:
        state['usage'] = usage
        assert not control.check_action(data, state, 'propose')['allowed']


def test_running_final_test_blocks_confirmation_and_reserves_its_gpu_time():
    data = authorized_control()
    state = {'frozen': {'run_id': 'baseline'}, 'trials': [{'usage': {'gpu_hours': 1}}],
             'final_test': {'status': 'running', 'reserved_gpu_hours': 1}}
    gate = control.check_action(data, state, 'confirm')
    assert not gate['allowed']
    assert gate['remaining_budget']['reserved_gpu_hours'] == 1


def test_terminal_final_test_reconciles_total_once_and_nonfinite_cost_blocks():
    data = authorized_control()
    data['limits'].update(total_gpu_hours=8, confirmation_gpu_hours=1, calibration_gpu_hours=0)
    state = {'frozen': {'run_id': 'baseline'}, 'trials': [{'usage': {'gpu_hours': 7}}],
             'final_test': {'status': 'completed', 'usage': {'gpu_hours': 1}},
             'usage': {'search_gpu_hours': 7, 'final_test_gpu_hours': 1, 'gpu_hours': 8}}
    assert not control.check_action(data, state, 'confirm')['allowed']
    assert control.reconcile_usage(data, state)['gpu_hours'] == 8
    for value in [-1, float('nan'), None]:
        state['final_test']['usage']['gpu_hours'] = value
        assert not control.check_action(data, state, 'confirm')['allowed']


def test_external_confirmation_receipt_counts_failed_actual_and_active_reservation(tmp_path):
    data = authorized_control()
    directory = tmp_path / 'external'
    directory.mkdir()
    receipt = {'confirmation_dir': str(directory), 'parent_campaign': str(tmp_path),
               'campaign_id': 'fixture', 'baseline_run_id': 'b', 'frozen_run_id': 'w',
               'protocol_id': data['protocol_id'], 'source_revision': data['source_revision'],
               'approved_config_ref': data['approved_config_ref']}
    manifest = {'association': receipt, 'tasks': [{'status': 'failed', 'usage': {'gpu_hours': 0.5}},
                                               {'status': 'running', 'reserved_gpu_hours': 1}],
                'active_task_key': 'winner:2:1'}
    (directory / 'confirmation.json').write_text(json.dumps(manifest))
    state = {'confirmation_refs': [receipt], 'usage': {'confirmation_gpu_hours': 0.5}}
    ledger = control.reconcile_usage(data, state)
    assert ledger['gpu_hours'] == 0.5 and ledger['reserved_gpu_hours'] == 1
    assert not control.check_action(data, state, 'baseline')['allowed']


def test_prepared_launch_intent_without_active_id_retains_reservation_and_blocks():
    data = authorized_control()
    state = {'frozen': {'run_id': 'winner'}, 'launch_intent': {'request_id': 'pending', 'reserved_gpu_hours': 1}}
    result = control.check_action(data, state, 'confirm')
    assert not result['allowed']
    assert result['remaining_budget']['reserved_gpu_hours'] == 1
