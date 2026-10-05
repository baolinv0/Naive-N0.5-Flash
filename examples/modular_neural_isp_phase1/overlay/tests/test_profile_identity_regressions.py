"""Launch metadata freezes diagnostic identity independently of scientific config."""
import json
from pathlib import Path

import pytest

from test_runner import config
from tm_research import runner
from tm_research.diagnostics import diagnostics_snapshot, evaluation_transform


def profile_file(tmp_path):
    path = tmp_path / 'profile.json'
    profile = {'profile_ref': str(path), 'profile_revision': 'p001', 'source': 'explicit',
               'eval_size': 512, 'transform': evaluation_transform(512), 'rules': {},
               'split_roots': {'DEV': {'input_dir': '/dev', 'gt_dir': '/gt', 'metadata_dir': '/meta'}},
               'samples': [{'sample_id': 'DEV:a', 'split': 'DEV', 'tags': ['fixed']}]}
    path.write_text(json.dumps(profile))
    return path, profile


def test_prepare_run_freezes_profile_content_without_changing_scientific_config(tmp_path):
    cfg = config(tmp_path)
    path, profile = profile_file(tmp_path)
    observation = {'profile_ref': str(path)}
    run_id = runner.prepare_run(cfg, observation_config=observation)
    run_dir = Path(cfg['runs_dir']) / run_id
    launch = json.loads((run_dir / 'observation_config.json').read_text())
    assert launch.get('profile_snapshot') == diagnostics_snapshot(profile)
    assert json.loads((run_dir / 'state.json').read_text())['observation_config'] == launch
    assert observation == {'profile_ref': str(path)}
    assert 'profile_snapshot' not in json.loads((run_dir / 'config.json').read_text())
    profile['samples'][0]['tags'] = ['changed_after_prepare']
    path.write_text(json.dumps(profile))
    assert json.loads((run_dir / 'observation_config.json').read_text()) == launch


def test_same_request_cannot_replace_prepared_profile_content(tmp_path):
    cfg = config(tmp_path)
    path, profile = profile_file(tmp_path)
    first = runner.prepare_run(cfg, observation_config={'profile_ref': str(path)}, request_id='frozen-profile')
    assert runner.prepare_run(cfg, observation_config={'profile_ref': str(path)}, request_id='frozen-profile') == first
    profile['samples'][0]['tags'] = ['rewritten']
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='different'):
        runner.prepare_run(cfg, observation_config={'profile_ref': str(path)}, request_id='frozen-profile')


def test_missing_optional_profile_is_recorded_without_blocking_prepare(tmp_path):
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg, observation_config={'profile_ref': str(tmp_path / 'missing.json')})
    run_dir = Path(cfg['runs_dir']) / run_id
    launch = json.loads((run_dir / 'observation_config.json').read_text())
    assert launch.get('profile_snapshot') is None
    assert launch.get('profile_snapshot_error')
    assert json.loads((run_dir / 'state.json').read_text())['status'] == 'queued'


@pytest.mark.parametrize('roi_value', [None, []])
def test_malformed_referenced_roi_records_snapshot_error_and_prepares_run(tmp_path, roi_value):
    cfg = config(tmp_path)
    path, profile = profile_file(tmp_path)
    roi_path = tmp_path / 'roi_profile.json'
    roi_path.write_text(json.dumps(roi_value))
    profile['roi_profile_ref'] = str(roi_path)
    path.write_text(json.dumps(profile))
    run_id = runner.prepare_run(cfg, observation_config={'profile_ref': str(path)})
    run_dir = Path(cfg['runs_dir']) / run_id
    launch = json.loads((run_dir / 'observation_config.json').read_text())
    assert launch.get('profile_snapshot') is None
    assert 'ValueError' in launch['profile_snapshot_error']
    assert 'ROI' in launch['profile_snapshot_error']
    assert json.loads((run_dir / 'state.json').read_text())['status'] == 'queued'
    assert '--diagnostics-profile' in json.loads((run_dir / 'commands.json').read_text())['dev']
