"""Simulated proposer tests: real subprocess runner, substitute tiny train/eval scripts."""

import json
from pathlib import Path

import pytest

from tm_research.campaign import (campaign_status, final_test, finalize_campaign,
    initialize_campaign, next_proposal, submit_proposal, wait_campaign)
from tm_research.cli import main
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


def test_tiny_raw_gains_stop_without_replacing_effective_best(tmp_path):
    cfg = config(tmp_path)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    script.write_text(script.read_text().replace("if mode=='nonfinite':",
        "score={'original':25.0,'mse':25.000001,'l1':25.000002}[recipe['loss_family']]\nif mode=='nonfinite':"))
    directory = tmp_path / 'campaign'
    baseline = initialize_campaign(cfg, directory, max_trials=8, no_gain_limit=2)
    state = submit_proposal(directory, proposal(next_proposal(directory), {'loss_family': 'mse'}))
    assert state['trials'][-1]['result_status'] == 'no_gain'
    assert state['no_gain_count'] == 1
    evidence = next_proposal(directory)
    assert evidence['raw_best']['dev_psnr'] == 25.000001
    assert evidence['best'] == baseline['best']
    assert evidence['min_delta'] > 0.000002
    state = submit_proposal(directory, proposal(evidence, {'loss_family': 'l1'}))
    assert state['no_gain_count'] == 2 and state['stop_reason'] == 'no_gain_limit'
    assert state['raw_best']['dev_psnr'] == 25.000002
    assert finalize_campaign(directory)['frozen']['run_id'] == baseline['best']['run_id']


def test_cli_min_delta_uses_last_effective_best_across_status_reads(tmp_path, capsys):
    cfg = config(tmp_path)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    script.write_text(script.read_text().replace("if mode=='nonfinite':",
        "score=25+(score-20)*0.03125\nif mode=='nonfinite':"))
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    directory = tmp_path / 'campaign'
    assert main(['campaign', 'init', '--config', str(config_path), '--campaign-dir', str(directory),
                 '--max-trials', '4', '--min-delta', '0.125', '--wait']) == 0
    baseline = json.loads(capsys.readouterr().out)
    assert baseline['min_delta'] == 0.125
    # 25.0625 and exactly 25.125 are not > baseline + min_delta.
    for overrides in ({'loss_family': 'mse'}, {'loss_family': 'mse', 'optimizer': 'adamw'}):
        state = submit_proposal(directory, proposal(next_proposal(directory), overrides))
        assert state['trials'][-1]['result_status'] == 'no_gain'
        assert state['best'] == baseline['best']
    assert state['raw_best']['dev_psnr'] == 25.125 and state['no_gain_count'] == 2
    overrides = {'loss_family': 'mse', 'optimizer': 'adamw', 'learning_rate': 0.00005}
    state = submit_proposal(directory, proposal(next_proposal(directory), overrides))
    assert state['trials'][-1]['result_status'] == 'improved'
    assert state['best']['dev_psnr'] == state['raw_best']['dev_psnr'] == 25.15625
    assert state['min_delta'] == 0.125 and state['no_gain_count'] == 0


def test_legacy_campaign_keeps_original_zero_threshold(tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / 'campaign'
    initialize_campaign(cfg, directory)
    path = directory / 'campaign.json'
    state = json.loads(path.read_text())
    state.pop('min_delta', None)
    state.pop('raw_best', None)
    path.write_text(json.dumps(state))
    evidence = next_proposal(directory)
    assert evidence['min_delta'] == 0 and evidence['raw_best'] == evidence['best']


@pytest.mark.parametrize('min_delta', [-0.01, float('nan')])
def test_min_delta_must_be_nonnegative_and_finite(tmp_path, min_delta):
    directory = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='min_delta'):
        initialize_campaign(config(tmp_path), directory, min_delta=min_delta)
    assert not directory.exists()


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
    for field in ('epochs', 'seed', 'eval_size', 'init_checkpoint', 'test', 'architecture', 'min_delta'):
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
