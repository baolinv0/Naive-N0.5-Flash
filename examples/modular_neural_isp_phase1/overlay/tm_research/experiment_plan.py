"""Freeze an experiment manifest and allocate independent native campaigns."""
import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path

from .control import load_control
from .runner import BUDGET_DEFAULTS, RECIPE_DEFAULTS, _write_json, load_config


def read_json(path):
    def bad_constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=bad_constant)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _number(value, name, *, positive=False, integer=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or positive and value <= 0 or integer and type(value) is not int):
        raise ValueError(name + ' must be an explicit finite ' + ('positive' if positive else 'nonnegative') + ' number')
    return value


def _seeds(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(name + ' must contain distinct seeds')
    for seed in values:
        _number(seed, name, integer=True)
    if len(set(values)) != len(values):
        raise ValueError(name + ' must contain distinct seeds')
    return values


def _authorized_domain(control):
    from .experiment_space import RECIPE_DOMAINS
    ranges = control.get('recipe_ranges', {})
    for field in ('learning_rate', 'weight_decay'):
        domain = RECIPE_DOMAINS[field]
        lower, upper = min(domain), max(domain)
        old = ranges.get(field, (lower, upper))
        if old[0] > lower or old[1] < upper:
            raise ValueError('Authorization does not cover the full discrete recipe domain')


def _source_control(path):
    """Resolve the approved independent split before relocating its control."""
    from .confirmation import _normalize_split
    control = load_control(path)
    scope = control['confirmation_scope']
    if scope.get('kind') == 'independent':
        try:
            scope['split_spec'] = _normalize_split(scope.get('split_spec'), Path(path).resolve().parent)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid authorized confirmation split_spec: ' + str(exc)) from exc
    return control


def verify_generated_index(index):
    """Check frozen content integrity, not authenticity against malicious rewrites.

    A checksum detects accidental or isolated modification. A party able to
    rewrite the entire index and all checksums remains outside this guarantee;
    host access controls provide that boundary.
    """
    if not isinstance(index, dict) or not isinstance(index.get('index_digest'), str):
        raise ValueError('Generated index checksum is missing')
    actual = digest({key: value for key, value in index.items() if key != 'index_digest'})
    if index['index_digest'] != actual:
        raise ValueError('Generated index digest integrity mismatch')
    def artifact(path, expected, label):
        if path is None:
            if expected is not None:
                raise ValueError(label + ' digest requires a frozen artifact')
            return
        if not isinstance(path, str) or not isinstance(expected, str):
            raise ValueError(label + ' artifact path and digest are required')
        try:
            observed = digest(read_json(path))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(label + ' artifact integrity unavailable: ' + str(exc)) from exc
        if observed != expected:
            raise ValueError(label + ' artifact digest integrity mismatch')
    profile_ref = index.get('profile_ref')
    if (index.get('manifest') or {}).get('profile_ref') != profile_ref:
        raise ValueError('Frozen profile reference integrity mismatch')
    artifact(profile_ref, index.get('profile_digest'), 'Profile')
    entries = index.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Generated index entries integrity is invalid')
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Generated index entry integrity is invalid')
        for name in ('config', 'control', 'confirmation_plan'):
            artifact(entry.get(name + '_path'), entry.get(name + '_digest'), name)


def load_experiment(path):
    from .experiment_space import STRATEGIES, validate_space_recipe
    path = Path(path).resolve()
    data = read_json(path)
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('Experiment schema_version must be 1')
    identifier = data.get('experiment_id')
    if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identifier):
        raise ValueError('experiment_id must be a safe readable identifier')
    for field in ('authorization_ref', 'source_revision'):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(field + ' must be explicit')
    for field in ('baseline_config', 'control_template', 'profile_ref'):
        if data.get(field) is None and field == 'profile_ref':
            continue
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(field + ' requires a path')
        data[field] = str((path.parent / data[field]).resolve())
        if not Path(data[field]).is_file():
            raise ValueError(field + ' is not a readable file')
    ctrl = _source_control(data['control_template'])
    approved = (Path(data['control_template']).parent / ctrl['approved_config_ref']).resolve()
    if Path(data['baseline_config']).resolve() != approved:
        raise ValueError('baseline_config must identify the existing approved_config_ref; a different baseline requires explicit authorization')
    if data.get('profile_ref') is None:
        fallback = ctrl['observation_config'].get('profile_ref')
        if fallback is not None:
            if not isinstance(fallback, str) or not fallback:
                raise ValueError('Authorized fallback profile_ref requires a path')
            data['profile_ref'] = str((Path(data['control_template']).parent / fallback).resolve())
            if not Path(data['profile_ref']).is_file():
                raise ValueError('Authorized fallback profile_ref is not a readable file')
    cfg = {**BUDGET_DEFAULTS, **RECIPE_DEFAULTS, **load_config(data['baseline_config'])}
    cfg.pop('test', None)
    validate_space_recipe({field: cfg[field] for field in RECIPE_DEFAULTS})
    if any(data[field] != ctrl[field] for field in ('authorization_ref', 'source_revision')):
        raise ValueError('Experiment authorization/source must match the control template')
    if ctrl['test_permission'] is not False or ctrl['observation_mode'] != 'text':
        raise ValueError('Search control must use text and test_permission=false')
    if ctrl['protocol_id'] != 'P' + str(cfg['eval_size']):
        raise ValueError('Control protocol differs from the actual evaluator')
    if set(ctrl['allowed_recipe_fields']) != set(RECIPE_DEFAULTS):
        raise ValueError('Five-group search requires authorization for all four recipe fields')
    _authorized_domain(ctrl)
    strategies = data.get('strategies')
    if (not isinstance(strategies, list) or not strategies
            or any(type(s) is not str or s not in STRATEGIES for s in strategies)
            or len(set(strategies)) != len(strategies)):
        raise ValueError('strategies must be distinct supported strategy identifiers')
    seeds = _seeds(data.get('search_seeds'), 'search_seeds')
    k = _number(data.get('max_trials'), 'max_trials', positive=True, integer=True)
    if k > ctrl['limits']['max_search_trials'] or k > 150:
        raise ValueError('max_trials exceeds the authorized limit or recipe domain')
    _number(data.get('min_delta'), 'min_delta')
    limits = data.get('experiment_limits', {})
    if not isinstance(limits, dict):
        raise ValueError('experiment_limits must be an object')
    _number(limits.get('total_gpu_hours'), 'experiment total_gpu_hours')
    _number(limits.get('walltime_seconds'), 'experiment walltime_seconds', positive=True)
    campaign_walltime = _number(data.get('campaign_walltime_seconds'), 'campaign_walltime_seconds', positive=True)
    if campaign_walltime > limits['walltime_seconds']:
        raise ValueError('campaign_walltime_seconds cannot exceed the global walltime limit')
    allocated = len(strategies) * len(seeds) * ctrl['limits']['total_gpu_hours']
    if allocated > limits['total_gpu_hours'] + 1e-12:
        raise ValueError('Global experiment budget cannot fund the generated campaign allocations')
    data.setdefault('evidence_mode', 'live_model_api')
    if data['evidence_mode'] not in ('live_model_api', 'engineering_fixture'):
        raise ValueError('evidence_mode must identify live_model_api or engineering_fixture')
    models = data.setdefault('models', {})
    if not isinstance(models, dict):
        raise ValueError('models must be an object')
    for name in ('naive', 'qwen'):
        model = models.get(name)
        if model is not None:
            if not isinstance(model, dict) or 'api_key' in model:
                raise ValueError('Model credentials must use api_key_env, never stored values')
            if not isinstance(model.get('model'), str) or not model['model'].strip():
                raise ValueError(name + '.model must be explicit')
    if any(s.startswith('naive_') for s in strategies) and not models.get('naive'):
        raise ValueError('Naive groups require an explicit Naive model service')
    data.setdefault('slow_reviewer', 'naive')
    if ('naive_fast_slow' in strategies and
            (data['slow_reviewer'] not in ('naive', 'qwen') or not models.get(data['slow_reviewer']))):
        raise ValueError('slow_reviewer requires a configured naive or qwen model')
    qwen = models.get('qwen') or {}
    if type(qwen.get('enabled', False)) is not bool:
        raise ValueError('qwen.enabled must be boolean')
    model_limits = data.get('model_limits')
    if not isinstance(model_limits, dict):
        raise ValueError('model_limits must be explicit')
    for field in ('max_calls', 'max_input_tokens_per_call', 'max_output_tokens_per_call', 'max_total_tokens'):
        _number(model_limits.get(field), field, positive=True, integer=True)
    for field in ('max_total_input_tokens', 'max_total_output_tokens'):
        if field in model_limits:
            _number(model_limits[field], field, positive=True, integer=True)
    campaign_limits = data.get('campaign_model_limits')
    if not isinstance(campaign_limits, dict):
        raise ValueError('campaign_model_limits must freeze equal per-campaign inference ceilings')
    for field in ('max_calls', 'max_input_tokens_per_call', 'max_output_tokens_per_call',
                  'max_total_tokens', 'max_total_input_tokens', 'max_total_output_tokens'):
        _number(campaign_limits.get(field), 'campaign ' + field, positive=True, integer=True)
        allocation = len(seeds) * sum(s.startswith('naive_') for s in strategies)
        required = campaign_limits[field] * (allocation if field in (
            'max_calls', 'max_total_tokens', 'max_total_input_tokens', 'max_total_output_tokens') else 1)
        if model_limits.get(field, 0) < required:
            raise ValueError('Global model allocation must cover equal campaign ' + field + ' ceilings')
    data.setdefault('schedule_seed', 20261005)
    _number(data['schedule_seed'], 'schedule_seed', integer=True)
    confirmation = data.get('confirmation')
    if confirmation is not None:
        if not isinstance(confirmation, dict) or not ctrl['confirmation_scope'].get('authorized'):
            raise ValueError('Confirmation requires existing explicit authorization')
        blocks = confirmation.get('seeds_by_block')
        if not isinstance(blocks, list) or len(blocks) != len(seeds):
            raise ValueError('seeds_by_block must cover every search block')
        all_seeds = []
        for block in blocks:
            _seeds(block, 'confirmation seeds')
            if not set(block) <= set(ctrl['confirmation_scope']['seeds']):
                raise ValueError('Confirmation seeds exceed authorization')
            all_seeds.extend(block)
        if set(all_seeds) & set(seeds) or len(set(all_seeds)) != len(all_seeds):
            raise ValueError('Confirmation seeds must be new and unique across blocks')
        for field in ('delta_useful_db', 'delta_strategy'):
            _number(confirmation.get(field), field)
    return data


def generate_experiment(manifest_path, output_dir):
    """Generate immutable per-campaign inputs; never launch training or inference."""
    data = load_experiment(manifest_path)
    root = Path(output_dir).resolve()
    cfg = {**BUDGET_DEFAULTS, **RECIPE_DEFAULTS, **load_config(data['baseline_config'])}
    cfg.pop('test', None)
    ctrl = _source_control(data['control_template'])
    inputs = {'manifest': data, 'baseline': cfg, 'control': ctrl,
              'profile': read_json(data['profile_ref']) if data.get('profile_ref') else None}
    identity = digest(inputs)
    index_path = root / 'experiment.json'
    if index_path.exists():
        existing = read_json(index_path)
        verify_generated_index(existing)
        if existing.get('identity') != identity:
            raise ValueError('Experiment already exists with different frozen inputs')
        return existing
    for split in ('train', 'validation'):
        for value in cfg[split].values():
            if isinstance(value, str):
                target = Path(value).resolve()
                if root == target or root in target.parents or target in root.parents:
                    raise ValueError('Experiment output must not overlap dataset directories')
    root.mkdir(parents=True, exist_ok=True)
    if list(root.iterdir()):
        raise ValueError('New experiment output directory must be empty')
    data = copy.deepcopy(data)
    profile_ref = str(root / 'profile.json') if inputs['profile'] is not None else None
    profile_digest = digest(inputs['profile']) if inputs['profile'] is not None else None
    if profile_ref is not None:
        _write_json(profile_ref, inputs['profile'])
    data['profile_ref'] = profile_ref
    entries = []
    for block, seed in enumerate(data['search_seeds']):
        for strategy in data['strategies']:
            directory = root / strategy / f'block_{block:03d}'
            directory.mkdir(parents=True)
            baseline = copy.deepcopy(cfg)
            baseline.update(seed=seed, runs_dir=str(directory / 'runs'))
            control = copy.deepcopy(ctrl)
            control.update(campaign_id=f'{data["experiment_id"]}_{strategy}_{block:03d}',
                           approved_config_ref=str(directory / 'config.json'),
                           test_permission=False)
            control['limits']['max_search_trials'] = data['max_trials']
            control['slow_policy']['consecutive_valid_no_gain'] = (
                2 if strategy == 'naive_fast_slow' else data['max_trials'] + 1)
            if profile_ref is not None:
                control['observation_config']['profile_ref'] = profile_ref
            # Intersect the discrete domain with already-authorized numeric bounds.
            for key, bounds in {'learning_rate': [1e-5, 1e-3], 'weight_decay': [0, 1e-4]}.items():
                control.setdefault('recipe_ranges', {})[key] = bounds
            plan = None
            if data.get('confirmation'):
                confirm = data['confirmation']
                scope = control['confirmation_scope']
                scope['seeds'] = confirm['seeds_by_block'][block]
                order = [f'{arm}:{s}:1' for s in scope['seeds'] for arm in ('baseline', 'winner')]
                random.Random(data['schedule_seed'] + block).shuffle(order)
                plan = {'seeds': scope['seeds'], 'replicates': 1,
                    'budget_gpu_hours': control['limits']['confirmation_gpu_hours'],
                    'delta_useful_db': confirm['delta_useful_db'],
                    'scope': {key: copy.deepcopy(scope[key]) for key in ('kind', 'split_spec') if key in scope},
                    'execution_order': order, 'failure_policy': 'preserve_no_retry',
                    'final_checkpoint_rule': {'kind': 'retain_search_checkpoint'}}
            _write_json(directory / 'config.json', baseline)
            _write_json(directory / 'control.json', control)
            if plan is not None:
                _write_json(directory / 'confirmation_plan.json', plan)
            entries.append({'strategy': strategy, 'block': block, 'search_seed': seed,
                'sampler_seed': data['schedule_seed'] + block * 100 + data['strategies'].index(strategy),
                'config_path': str(directory / 'config.json'), 'control_path': str(directory / 'control.json'),
                'campaign_dir': str(directory / 'campaign'), 'directory': str(directory),
                'confirmation_dir': str(directory / 'confirmation'),
                'confirmation_plan_path': str(directory / 'confirmation_plan.json') if plan else None,
                'confirmation_plan_digest': digest(plan) if plan is not None else None,
                'config_digest': digest(baseline), 'control_digest': digest(control),
                'no_gain_limit': data['max_trials'] + 1,
                'strategy_variant': 'qwen_review' if strategy == 'naive_fast_slow' and data['slow_reviewer'] == 'qwen' else 'primary'})
    random.Random(data['schedule_seed']).shuffle(entries)
    result = {'schema_version': 1, 'experiment_id': data['experiment_id'], 'identity': identity,
              'manifest': data, 'entries': entries, 'evidence_mode': data['evidence_mode'],
              'profile_ref': profile_ref, 'profile_digest': profile_digest}
    result['index_digest'] = digest(result)
    _write_json(index_path, result)
    return result
