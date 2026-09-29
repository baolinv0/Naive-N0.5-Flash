"""Simulated proposer tests: real subprocess runner, substitute tiny train/eval scripts."""

import json
from pathlib import Path

import pytest

from tm_research.campaign import (campaign_status, final_test, finalize_campaign,
    initialize_campaign, next_proposal, submit_proposal, wait_campaign)
from test_runner import config


def proposal(evidence, overrides, hypothesis='Expected improvement from the cited DEV observation.'):
    previous = evidence['history'][-1]
    return {'recipe': overrides, 'hypothesis': hypothesis,
            'based_on': {'run_id': previous['run_id'], 'dev_psnr': previous['dev_psnr'],
                         'observation': previous['decision']}}


def test_three_result_dependent_changes_fixed_budget_and_frozen_final_test(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    state = initialize_campaign(cfg, directory, max_trials=4, no_gain_limit=3)
    assert state['trials'][0]['result_status'] == 'valid'
    assert state['best']['dev_psnr'] == 20
    # A deliberately simulated proposer changes its decision according to the actual last score.
    observed = []
    for _ in range(3):
        evidence = next_proposal(directory)
        score = evidence['history'][-1]['dev_psnr']
        observed.append(score)
        if score < 22:
            overrides = {'loss_family': 'mse'}
        elif score < 24:
            overrides = {'optimizer': 'adamw'}
        else:
            overrides = {'learning_rate': 0.00005}
        state = submit_proposal(directory, proposal(evidence, overrides))
        assert state['trials'][-1]['result_status'] == 'improved'
    assert observed == [20, 22, 24]
    assert [t['dev_psnr'] for t in state['trials']] == [20, 22, 24, 25]
    assert state['stop_reason'] == 'max_trials'
    assert not next_proposal(directory)['proposal_allowed']
    fixed_keys = ('seed', 'epochs', 'batch_size', 'in_size', 'eval_size', 'num_workers', 'validation_frequency')
    configs = [json.loads((Path(cfg['runs_dir']) / t['run_id'] / 'config.json').read_text()) for t in state['trials']]
    assert all({k: c[k] for k in fixed_keys} == {k: configs[0][k] for k in fixed_keys} for c in configs)
    assert configs[0]['seed'] == 1
    assert all('test' not in c for c in configs)
    assert 'test' not in next_proposal(directory)['fixed']
    assert not (directory / 'final_test').exists()
    with pytest.raises(ValueError, match='Finalize'):
        final_test(directory)
    state = finalize_campaign(directory)
    assert state['frozen']['run_id'] == state['trials'][-1]['run_id']
    with pytest.raises(ValueError, match='cannot accept'):
        submit_proposal(directory, proposal(evidence, {'weight_decay': 0}))
    report = final_test(directory)
    assert report['valid'] and report['test_psnr'] == 25
    assert report['run_id'] == state['frozen']['run_id']
    assert final_test(directory) == report
    assert campaign_status(directory)['best'] == state['best']


def test_no_gain_stops_and_keeps_baseline(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    initialize_campaign(cfg, directory, max_trials=8, no_gain_limit=2)
    state = submit_proposal(directory, proposal(next_proposal(directory), {'loss_family': 'l1'}))
    assert state['trials'][-1]['status'] == 'completed'
    assert state['trials'][-1]['result_status'] == 'no_gain'
    state = submit_proposal(directory, proposal(next_proposal(directory), {'loss_family': 'l1', 'learning_rate': 0.0002}))
    assert state['stop_reason'] == 'no_gain_limit'
    assert state['best']['run_id'] == state['trials'][0]['run_id']
    assert state['no_gain_count'] == 2
    assert not next_proposal(directory)['proposal_allowed']


def test_invalid_baseline_stops_without_best(tmp_path):
    cfg = config(tmp_path, 'nonfinite')
    directory = tmp_path / 'campaign'
    state = initialize_campaign(cfg, directory)
    assert state['stop_reason'] == 'invalid_baseline' and state['best'] is None
    assert not next_proposal(directory)['proposal_allowed']
    with pytest.raises(ValueError, match='no valid'):
        finalize_campaign(directory)


def test_proposal_must_cite_real_score_and_only_change_recipe(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    initialize_campaign(cfg, directory)
    evidence = next_proposal(directory)
    for field in ('epochs', 'seed', 'eval_size', 'init_checkpoint', 'test', 'architecture'):
        bad = proposal(evidence, {'loss_family': 'mse', field: 10})
        with pytest.raises(ValueError, match='Recipe supports only'):
            submit_proposal(directory, bad)
    bad = proposal(evidence, {'loss_family': 'mse'})
    bad['based_on']['dev_psnr'] = 123
    with pytest.raises(ValueError, match='DEV PSNR within'):
        submit_proposal(directory, bad)
    bad = proposal(evidence, {'loss_family': 'mse'})
    bad['based_on']['run_id'] = 'exp_999'
    with pytest.raises(ValueError, match='recorded run'):
        submit_proposal(directory, bad)
    with pytest.raises(ValueError, match='already been evaluated'):
        submit_proposal(directory, proposal(evidence, {}))
    assert len(campaign_status(directory)['trials']) == 1
    rounded = proposal(evidence, {'loss_family': 'mse'})
    rounded['based_on']['dev_psnr'] = 20.00009
    state = submit_proposal(directory, rounded)
    assert state['trials'][-1]['result_status'] == 'improved'
    assert state['best']['dev_psnr'] == 22


def test_pending_work_survives_host_return_and_status_collects_it(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    started = initialize_campaign(cfg, directory, wait=False)
    run_id = started['active']['run_id']
    state = wait_campaign(directory, timeout=15)
    assert state['active'] is None and state['trials'][0]['run_id'] == run_id
    assert len(campaign_status(directory)['trials']) == 1
    started = submit_proposal(directory, proposal(next_proposal(directory), {'loss_family': 'mse'}), wait=False)
    state = wait_campaign(directory, timeout=15)
    assert state['trials'][-1]['run_id'] == started['active']['run_id']
    assert len(state['trials']) == 2
    with pytest.raises(ValueError, match='already exists'):
        initialize_campaign(cfg, directory)


def test_invalid_candidate_does_not_replace_best_and_is_next_prompt_evidence(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    baseline = initialize_campaign(cfg, directory)
    test_script = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    test_script.write_text(test_script.read_text().replace("mode='ok'", "mode='wrong_count'"))
    state = submit_proposal(directory, proposal(next_proposal(directory), {'loss_family': 'mse'}))
    assert state['trials'][-1]['result_status'] == 'invalid'
    assert state['best'] == baseline['best'] and state['no_gain_count'] == 1
    evidence = next_proposal(directory)
    assert evidence['history'][-1]['error'] and evidence['history'][-1]['dev_psnr'] is None
    assert evidence['proposal_allowed']
