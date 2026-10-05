"""Bounded command dispatch and read-only output, without live services."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_experiment_runner import manifest


def cli():
    assert importlib.util.find_spec('tm_research.experiment') is not None, 'experiment CLI is missing'
    return importlib.import_module('tm_research.experiment')


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def test_module_help_lists_bounded_commands_without_services():
    result = subprocess.run([sys.executable, '-m', 'tm_research.experiment', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for command in ('generate', 'status', 'step', 'confirm', 'analyze'):
        assert command in result.stdout


def test_generate_valid_fixture_writes_inputs_without_launching(tmp_path, capsys):
    path = manifest(tmp_path)
    root = tmp_path / 'outer'
    assert cli().main(['generate', '--manifest', str(path), '--output', str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['evidence_mode'] == 'engineering_fixture'
    assert len(result['entries']) == 5
    assert (root / 'experiment.json').is_file()
    assert not list(root.rglob('campaign.json'))
    assert not list(root.rglob('worker.pid'))
    assert not list(root.rglob('*.request.json'))
    saved = snapshot(root)
    assert cli().main(['generate', '--manifest', str(path), '--output', str(root)]) == 0
    capsys.readouterr()
    assert snapshot(root) == saved


def test_validate_has_no_output_directory_side_effect(tmp_path, capsys):
    path = manifest(tmp_path)
    saved = snapshot(tmp_path)
    assert cli().main(['validate', '--manifest', str(path)]) == 0
    assert json.loads(capsys.readouterr().out)['experiment_id'] == 'fixture'
    assert snapshot(tmp_path) == saved


def test_status_prints_controlled_read_only_state(tmp_path, monkeypatch, capsys):
    from tm_research.experiment_plan import generate_experiment
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path), root)
    saved = snapshot(root)
    state = {'status': 'not_started', 'evidence_mode': 'engineering_fixture',
             'costs': {'actual_total_tokens': None, 'reserved_total_tokens': 0}}
    monkeypatch.setitem(sys.modules, 'tm_research.experiment_runner',
                        SimpleNamespace(experiment_status=lambda experiment: state))
    assert cli().main(['status', '--experiment', str(root)]) == 0
    assert json.loads(capsys.readouterr().out) == state
    assert snapshot(root) == saved


def test_status_reads_real_generated_fixture_without_creating_runtime(tmp_path, capsys):
    from tm_research.experiment_plan import generate_experiment
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path), root)
    saved = snapshot(root)
    assert cli().main(['status', '--experiment', str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['evidence_mode'] == 'engineering_fixture'
    assert len(result['entries']) == 5
    assert all(entry['trial_count'] == 0 for entry in result['entries'])
    assert snapshot(root) == saved


@pytest.mark.parametrize('command,api', [('step', 'step_experiment'), ('confirm', 'confirm_experiment')])
@pytest.mark.parametrize('wait', [False, True])
def test_advancement_returns_after_one_runner_action(command, api, wait, tmp_path, monkeypatch, capsys):
    state = {'status': 'paused', 'actions': []}

    def advance(experiment, *, wait=False):
        state['actions'].append({'experiment': str(experiment), 'wait': wait})
        return state

    monkeypatch.setitem(sys.modules, 'tm_research.experiment_runner', SimpleNamespace(**{api: advance}))
    root = tmp_path / 'outer'
    arguments = [command, '--experiment', str(root)] + (['--wait'] if wait else [])
    assert cli().main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {
        'status': 'paused', 'actions': [{'experiment': str(root), 'wait': wait}]}


def test_analyze_exports_fixture_identity_and_registered_threshold(tmp_path, monkeypatch, capsys):
    from tm_research.experiment_plan import generate_experiment
    root = tmp_path / 'outer'
    path = manifest(tmp_path, confirmation=True)
    data = json.loads(path.read_text())
    data['confirmation']['delta_strategy'] = .25
    path.write_text(json.dumps(data))
    index = generate_experiment(path, root)
    records = [{'strategy': entry['strategy'], 'block': entry['block'],
                'expected_seeds': [1001, 1002],
                'provenance': {'strategy_variant': entry['strategy_variant'], 'scope': {'kind': 'independent'}},
                'pairs': [{'seed': seed, 'status': 'valid', 'delta_db': .1}
                          for seed in (1001, 1002)]} for entry in index['entries']]
    monkeypatch.setitem(sys.modules, 'tm_research.experiment_runner',
                        SimpleNamespace(collect_analysis_records=lambda experiment: records))
    saved = snapshot(root)
    out = tmp_path / 'report'
    assert cli().main(['analyze', '--experiment', str(root), '--output', str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    report = json.loads(Path(result['artifacts']['analysis_json']).read_text())
    assert report['delta_strategy'] == .25
    assert report['expected_campaign_count'] == 5
    assert report['experiment_identity'] == index['identity']
    assert report['evidence_mode'] == 'engineering_fixture'
    assert report['source_records'] == records
    assert Path(result['artifacts']['campaigns_csv']).is_file()
    assert Path(result['artifacts']['comparisons_csv']).is_file()
    assert snapshot(root) == saved


def test_analyze_real_unrun_fixture_reports_missing_primary_evidence(tmp_path, capsys):
    from tm_research.experiment_plan import generate_experiment
    root = tmp_path / 'outer'
    generate_experiment(manifest(tmp_path, confirmation=True), root)
    saved = snapshot(root)
    out = tmp_path / 'report'
    assert cli().main(['analyze', '--experiment', str(root), '--output', str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    report = json.loads(Path(result['artifacts']['analysis_json']).read_text())
    assert report['status'] == 'incomplete'
    assert report['primary_family_status'] == 'inconclusive'
    assert report['complete_campaign_count'] == 0
    assert all(record['expected_seeds'] == [1001, 1002] for record in report['source_records'])
    assert all(record['pairs'] == [] for record in report['source_records'])
    assert snapshot(root) == saved


def test_runner_error_is_json_and_nonzero_without_retry(tmp_path, monkeypatch, capsys):
    state = {'attempts': 0}

    def rejected(experiment, *, wait=False):
        state['attempts'] += 1
        raise ValueError('all searches must be frozen')

    monkeypatch.setitem(sys.modules, 'tm_research.experiment_runner',
                        SimpleNamespace(confirm_experiment=rejected))
    assert cli().main(['confirm', '--experiment', str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out) == {'error': 'all searches must be frozen'}
    assert state['attempts'] == 1


def test_example_requires_explicit_resources_and_real_approval_paths():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'experiment.example.json'
    assert path.is_file(), 'reviewable experiment manifest example is missing'
    example = json.loads(path.read_text())
    assert 'PLACEHOLDER' in example['authorization_ref']
    assert 'PLACEHOLDER' in example['source_revision']
    assert all('PLACEHOLDER' in example[field] for field in ('baseline_config', 'control_template', 'profile_ref'))
    assert example['experiment_limits']['total_gpu_hours'] is None
    assert example['experiment_limits']['walltime_seconds'] is None
    assert all(value is None for value in example['model_limits'].values())
    assert example['campaign_model_limits'].keys() == example['model_limits'].keys()
    assert all(value is None for value in example['campaign_model_limits'].values())
    assert example['slow_reviewer'] == 'naive'
    assert example['models']['qwen']['enabled'] is False
    assert len(example['search_seeds']) == 10
    assert len(example['confirmation']['seeds_by_block']) == 10
    assert all(len(seeds) == 3 for seeds in example['confirmation']['seeds_by_block'])
