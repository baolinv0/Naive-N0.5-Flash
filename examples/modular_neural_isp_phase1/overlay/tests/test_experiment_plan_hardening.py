"""Frozen experiment inputs must remain honest before any costly action."""

import json
from pathlib import Path

import pytest

from test_experiment_runner import manifest
from tm_research.experiment_plan import digest, generate_experiment, load_experiment


def change(path, **updates):
    data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data))
    return data


def test_campaign_model_budget_must_not_starve_equal_groups(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['model_limits']['max_calls'] = 32
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='campaign|allocation'):
        load_experiment(path)


@pytest.mark.parametrize('changed_field', ['epochs', 'validation'])
def test_manifest_cannot_rebind_approved_baseline_to_another_config(tmp_path, changed_field):
    path = manifest(tmp_path, ['random'])
    data = json.loads(path.read_text())
    approved = json.loads(Path(data['baseline_config']).read_text())
    alternate = dict(approved)
    if changed_field == 'epochs':
        alternate['epochs'] = approved['epochs'] + 10
    else:
        alternate['validation'] = {**approved['validation'], 'input_dir': str(tmp_path / 'other-dev')}
    unauthorized = tmp_path / 'unauthorized-baseline.json'
    unauthorized.write_text(json.dumps(alternate))
    data['baseline_config'] = str(unauthorized)
    path.write_text(json.dumps(data))
    output = tmp_path / 'outer'
    with pytest.raises(ValueError, match='approved|baseline|authorization'):
        generate_experiment(path, output)
    assert not output.exists()


def test_approved_baseline_relative_reference_resolves_from_original_control_directory(tmp_path):
    path = manifest(tmp_path, ['random'])
    control = tmp_path / 'control.json'
    change(control, approved_config_ref='baseline.json')
    assert load_experiment(path)['baseline_config'] == str(tmp_path / 'baseline.json')


def test_independent_confirmation_relative_paths_preserve_original_authorized_split(tmp_path):
    from tm_research.confirmation import _normalize_split
    path = manifest(tmp_path, ['random'], confirmation=True)
    control_path = tmp_path / 'control.json'
    control = json.loads(control_path.read_text())
    relative = {'input_dir': 'independent/input', 'gt_dir': 'independent/gt',
                'metadata_dir': 'independent/metadata', 'expected_count': 2}
    control['confirmation_scope'].update(kind='independent', split_spec=relative)
    control_path.write_text(json.dumps(control))
    expected = _normalize_split(relative, control_path.parent)
    output = tmp_path / 'outer'
    index = generate_experiment(path, output)
    entry = index['entries'][0]
    generated_control = json.loads(Path(entry['control_path']).read_text())
    generated_plan = json.loads(Path(entry['confirmation_plan_path']).read_text())
    assert generated_control['confirmation_scope']['split_spec'] == expected
    assert generated_plan['scope']['split_spec'] == expected
    assert all(Path(expected[key]).is_absolute() for key in ('input_dir', 'gt_dir', 'metadata_dir'))
    assert _normalize_split(generated_control['confirmation_scope']['split_spec'], Path(entry['control_path']).parent) == expected


@pytest.mark.parametrize('strategies', [['random'], ['tpe'], ['random', 'tpe']])
def test_non_llm_controls_need_no_naive_or_slow_reviewer_service(tmp_path, strategies):
    path = manifest(tmp_path, strategies)
    change(path, models={})
    data = load_experiment(path)
    assert data['models'] == {}
    result = generate_experiment(path, tmp_path / 'outer')
    assert len(result['entries']) == len(strategies)


def test_scalar_naive_does_not_require_configured_unused_qwen_reviewer(tmp_path):
    path = manifest(tmp_path, ['naive_scalar'])
    change(path, slow_reviewer='qwen')
    assert load_experiment(path)['slow_reviewer'] == 'qwen'


def test_fast_slow_requires_its_actual_reviewer_service(tmp_path):
    path = manifest(tmp_path, ['naive_fast_slow'])
    change(path, slow_reviewer='qwen')
    with pytest.raises(ValueError, match='reviewer'):
        load_experiment(path)


@pytest.mark.parametrize('field,value', [
    ('search_seeds', [[101]]), ('search_seeds', [{'seed': 101}]),
    ('search_seeds', [True]), ('search_seeds', [101, 101]),
    ('strategies', [['random']]), ('strategies', [{'name': 'random'}]),
    ('strategies', ['random', 'random']),
])
def test_bad_seed_and_strategy_types_raise_value_error_before_output(tmp_path, field, value):
    path = manifest(tmp_path)
    change(path, **{field: value})
    output = tmp_path / 'outer'
    with pytest.raises(ValueError):
        generate_experiment(path, output)
    assert not output.exists()


@pytest.mark.parametrize('field,bounds', [('learning_rate', [3e-5, 1e-3]),
                                         ('weight_decay', [0, 1e-5])])
def test_incomplete_authorized_recipe_domain_fails_before_creating_directories(tmp_path, field, bounds):
    path = manifest(tmp_path)
    control = tmp_path / 'control.json'
    change(control, recipe_ranges={field: bounds})
    output = tmp_path / 'outer'
    with pytest.raises(ValueError, match='domain'):
        generate_experiment(path, output)
    assert not output.exists()


def test_profile_is_frozen_locally_with_content_digest_and_index_checksum(tmp_path):
    path = manifest(tmp_path, ['random'], confirmation=True)
    source = tmp_path / 'profile-source.json'
    profile = {'schema_version': 1, 'samples': [{'sample_id': 'scene01', 'split': 'DEV'}]}
    source.write_text(json.dumps(profile))
    change(path, profile_ref=str(source))
    output = tmp_path / 'outer'
    result = generate_experiment(path, output)
    frozen = output / 'profile.json'
    assert result['profile_ref'] == result['manifest']['profile_ref'] == str(frozen)
    assert json.loads(frozen.read_text()) == profile
    assert result['profile_digest'] == digest(profile)
    entry = result['entries'][0]
    plan = json.loads(Path(entry['confirmation_plan_path']).read_text())
    assert entry['confirmation_plan_digest'] == digest(plan)
    assert result['index_digest'] == digest({k: v for k, v in result.items() if k != 'index_digest'})
    source.write_text(json.dumps({'samples': []}))
    assert json.loads(frozen.read_text()) == profile
    assert result['manifest']['profile_ref'] != str(source)


def test_generation_rejects_tampered_existing_index_before_returning_it(tmp_path):
    path = manifest(tmp_path, ['random'])
    output = tmp_path / 'outer'
    generate_experiment(path, output)
    index = output / 'experiment.json'
    saved = json.loads(index.read_text())
    saved['entries'][0]['sampler_seed'] += 1
    index.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='digest|integrity|checksum'):
        generate_experiment(path, output)


def test_approved_control_fallback_profile_is_frozen_before_native_launch(tmp_path):
    path = manifest(tmp_path, ['random'])
    source = tmp_path / 'control-profile.json'
    profile = {'schema_version': 1, 'samples': [{'sample_id': 'scene02'}]}
    source.write_text(json.dumps(profile))
    control_path = tmp_path / 'control.json'
    control = json.loads(control_path.read_text())
    control['observation_config']['profile_ref'] = str(source)
    control_path.write_text(json.dumps(control))
    output = tmp_path / 'outer'
    index = generate_experiment(path, output)
    assert index['profile_ref'] == str(output / 'profile.json')
    frozen_control = json.loads(Path(index['entries'][0]['control_path']).read_text())
    assert frozen_control['observation_config']['profile_ref'] == index['profile_ref']
    assert json.loads(Path(index['profile_ref']).read_text()) == profile


@pytest.mark.parametrize('artifact', ['config_path', 'control_path', 'confirmation_plan_path', 'profile_ref'])
def test_idempotent_generation_rejects_modified_frozen_artifacts(tmp_path, artifact):
    path = manifest(tmp_path, ['random'], confirmation=True)
    profile = tmp_path / 'source-profile.json'
    profile.write_text(json.dumps({'schema_version': 1, 'samples': []}))
    change(path, profile_ref=str(profile))
    output = tmp_path / 'outer'
    saved = generate_experiment(path, output)
    target = Path(saved[artifact] if artifact == 'profile_ref' else saved['entries'][0][artifact])
    original = json.loads(target.read_text())
    original['tampered'] = True
    target.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='digest|integrity|checksum'):
        generate_experiment(path, output)
