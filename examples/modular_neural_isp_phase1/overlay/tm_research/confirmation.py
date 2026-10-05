"""Separate frozen paired confirmation; never edits search decisions or selects seeds."""
from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path

from . import runner


@contextmanager
def _locked(directory, lock_name='.confirmation.lock'):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / lock_name).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another confirmation command is active') from exc
        yield directory


def _read(directory):
    return json.loads((Path(directory) / 'confirmation.json').read_text(encoding='utf-8'))


def _save(directory, state):
    runner._write_json(Path(directory) / 'confirmation.json', state)


def _same_resource(first, second):
    a, b = Path(first).resolve(), Path(second).resolve()
    if a == b or a in b.parents or b in a.parents:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _normalize_split(split, base):
    if not isinstance(split, dict) or set(split) - {'input_dir', 'gt_dir', 'metadata_dir', 'expected_count'}:
        raise ValueError('Confirmation split_spec must contain only data paths and expected_count')
    if not split.get('input_dir') or not split.get('gt_dir'):
        raise ValueError('Confirmation split_spec requires input_dir and gt_dir')
    result = dict(split)
    for name in ('input_dir', 'gt_dir', 'metadata_dir'):
        if result.get(name) is not None:
            result[name] = str((Path(base) / result[name]).resolve())
    if 'expected_count' in result and (type(result['expected_count']) is not int or result['expected_count'] < 1):
        raise ValueError('Confirmation expected_count must be a positive integer')
    return result


def _canonical_plan(plan, search):
    plan = deepcopy(plan)
    allowed = {'seeds', 'replicates', 'budget_gpu_hours', 'delta_useful_db', 'scope',
               'execution_order', 'failure_policy', 'final_checkpoint_rule', 'arms', 'primary_metric'}
    if not isinstance(plan, dict) or set(plan) - allowed:
        raise ValueError('Unsupported confirmation plan fields')
    seeds = plan.get('seeds')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Confirmation seeds must be distinct explicit nonnegative integers')
    plan.setdefault('replicates', 1)
    if type(plan['replicates']) is not int or plan['replicates'] < 1:
        raise ValueError('Confirmation replicates must be a positive integer')
    value = plan.get('budget_gpu_hours')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Confirmation budget_gpu_hours must be explicit finite nonnegative')
    if 'delta_useful_db' in plan:
        value = plan['delta_useful_db']
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('delta_useful_db must be finite nonnegative')
    scope = plan.get('scope')
    if not isinstance(scope, dict) or scope.get('kind') not in ('original_dev', 'independent') or set(scope) - {'kind', 'split_spec'}:
        raise ValueError('Confirmation scope must explicitly be original_dev or independent')
    if scope['kind'] == 'independent':
        scope['split_spec'] = _normalize_split(scope.get('split_spec'), search)
    elif 'split_spec' in scope:
        raise ValueError('original_dev cannot specify another split')
    plan.setdefault('failure_policy', 'preserve_no_retry')
    if plan['failure_policy'] != 'preserve_no_retry':
        raise ValueError('Only preserve_no_retry failure policy is implemented; no automatic retry')
    plan.setdefault('primary_metric', 'mean_per_image_psnr')
    if plan['primary_metric'] != 'mean_per_image_psnr':
        raise ValueError('Confirmation primary metric must be mean_per_image_psnr')
    plan.setdefault('final_checkpoint_rule', {'kind': 'retain_search_checkpoint'})
    rule = plan['final_checkpoint_rule']
    if not isinstance(rule, dict) or rule.get('kind') not in ('retain_search_checkpoint', 'predeclared_seed'):
        raise ValueError('Final checkpoint rule cannot choose best confirmation seed')
    if rule['kind'] == 'predeclared_seed':
        rule.setdefault('replicate', 1)
        if rule.get('seed') not in seeds or type(rule.get('seed')) is not int or type(rule['replicate']) is not int or not 1 <= rule['replicate'] <= plan['replicates']:
            raise ValueError('Final checkpoint must use a predeclared seed and replicate')
        if set(rule) != {'kind', 'seed', 'replicate'}:
            raise ValueError('Unsupported final checkpoint rule fields')
    elif set(rule) != {'kind'}:
        raise ValueError('Unsupported final checkpoint rule fields')
    keys = [f'{arm}:{seed}:{replicate}' for seed in seeds for replicate in range(1, plan['replicates'] + 1) for arm in ('baseline', 'winner')]
    plan.setdefault('execution_order', keys)
    if not isinstance(plan['execution_order'], list) or any(not isinstance(k, str) for k in plan['execution_order']) or sorted(plan['execution_order']) != sorted(keys):
        raise ValueError('execution_order must contain each arm+seed+replicate exactly once')
    return plan


def _authorization(control, plan, cfg):
    scope = control.get('confirmation_scope', {})
    limits = control.get('limits', {})
    reasons = []
    if control.get('protocol_id') != f'P{cfg["eval_size"]}':
        reasons.append('Confirmation control protocol differs from native frozen evaluation protocol')
    if not scope.get('authorized'):
        reasons.append('Confirmation scope is not authorized')
    if scope.get('kind') != plan['scope']['kind']:
        reasons.append('Confirmation scope differs from authorization')
    if any(seed not in scope.get('seeds', []) for seed in plan['seeds']):
        reasons.append('Confirmation seed is not authorized')
    cap = scope.get('replicates', scope.get('max_replicates', 1))
    if type(cap) is not int or plan['replicates'] > cap:
        reasons.append('Confirmation replicates exceed authorization')
    if plan['scope']['kind'] == 'independent':
        authorized = scope.get('split_spec')
        if not authorized or _normalize_split(authorized, Path.cwd()) != plan['scope']['split_spec']:
            reasons.append('Independent confirmation data is not authorized')
    for key in ('job_walltime_seconds', 'allocated_gpus', 'total_gpu_hours', 'confirmation_gpu_hours'):
        value = limits.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'job_walltime_seconds' and value == 0):
            reasons.append(f'Missing usable execution limit {key}')
    if not reasons:
        reservation = limits['job_walltime_seconds'] * limits['allocated_gpus'] / 3600
        if plan['budget_gpu_hours'] > limits['confirmation_gpu_hours']:
            reasons.append('Confirmation budget exceeds authorized confirmation_gpu_hours')
        if reservation * len(plan['execution_order']) * (2 if plan['scope']['kind'] == 'independent' else 1) > plan['budget_gpu_hours'] + 1e-12:
            reasons.append('Confirmation budget cannot reserve all predeclared jobs')
        if plan['budget_gpu_hours'] > limits['total_gpu_hours']:
            reasons.append('Confirmation budget exceeds total_gpu_hours')
    return reasons


def _parent_reasons(parent, control, frozen_identity=None):
    reasons = []
    if parent.get('status') != 'finalized' or not parent.get('frozen'):
        reasons.append('Parent must remain finalized with frozen winner')
    if parent.get('research_hold'):
        reasons.append('Parent has unresolved research/protocol hold')
    if parent.get('liveness') == 'unknown' or parent.get('active_health') == 'unknown':
        reasons.append('Parent worker liveness is unknown')
    if parent.get('active'):
        reasons.append('Parent has active or unresolved worker identity')
    attempt = parent.get('final_test') or {}
    if attempt and attempt.get('status') not in ('completed', 'failed'):
        reasons.append('Parent final TEST evaluation is active or unknown')
    original = parent.get('control')
    if original:
        for key in ('campaign_id', 'protocol_id', 'source_revision', 'approved_config_ref', 'primary_metric'):
            if original.get(key) != control.get(key):
                reasons.append(f'Frozen parent control {key} differs')
        if control != original:
            renewed = (control.get('authorization_ref') != original.get('authorization_ref') and
                       control.get('parent_authorization_ref') == original.get('authorization_ref'))
            if not renewed:
                reasons.append('Confirmation control differs from frozen authorization without compatible renewal provenance')
    if frozen_identity:
        if parent.get('config') != frozen_identity['config'] or parent.get('frozen') != frozen_identity['frozen']:
            reasons.append('Parent scientific config or frozen winner changed')
        first = (parent.get('trials') or [{}])[0]
        if first.get('run_id') != frozen_identity['baseline']['run_id'] or first.get('recipe') != frozen_identity['baseline']['recipe']:
            reasons.append('Parent frozen baseline identity changed')
        if parent.get('control') != frozen_identity['control']:
            reasons.append('Parent frozen control snapshot changed')
    return reasons


def _register_receipt(search, parent, state, directory):
    receipt = state['association']
    receipts = parent.setdefault('confirmation_refs', [])
    same_path = [r for r in receipts if r.get('confirmation_dir') == str(directory)]
    if same_path and any(r != receipt for r in same_path):
        raise ValueError('Existing confirmation receipt association differs')
    if not same_path:
        # Bind authority before the first task can launch. Existing evidence is
        # never retroactively treated as proof of preregistration.
        if not receipts and 'primary_confirmation' not in parent and all(
                task.get('status') == 'pending' and not task.get('run_id')
                and not task.get('result') for task in state['tasks']):
            parent['primary_confirmation'] = {'association': deepcopy(receipt),
                                              'plan': deepcopy(state['plan']),
                                              'registered_at': runner._now()}
        receipts.append(deepcopy(receipt))
        runner._write_json(search / 'campaign.json', parent)


def _primary_receipt(parent):
    """Return frozen authority; old single plans work, old multiple plans do not."""
    receipts = parent.get('confirmation_refs', [])
    if not isinstance(receipts, list):
        raise ValueError('Confirmation receipts must be a list')
    primary = parent.get('primary_confirmation')
    if primary is not None:
        if (not isinstance(primary, dict) or not primary.get('registered_at')
                or not isinstance(primary.get('plan'), dict) or not receipts
                or primary.get('association') != receipts[0]
                or receipts.count(primary['association']) != 1):
            raise ValueError('Frozen primary confirmation association is missing or incompatible')
        return primary['association']
    if len(receipts) > 1:
        raise ValueError('Confirmation authority is ambiguous: multiple legacy plans lack preregistration')
    return receipts[0] if receipts else None


def _validate_primary_plan(parent, receipt, manifest):
    primary = parent.get('primary_confirmation')
    if primary and receipt == primary.get('association') and manifest.get('plan') != primary.get('plan'):
        raise ValueError('Primary confirmation plan differs from its immutable preregistration')


def _effective_config(config):
    """Compare scientific settings using the defaults actually used by workers."""
    return runner.normalize_config({**runner.BUDGET_DEFAULTS, **runner.RECIPE_DEFAULTS, **config})


def _task_inventory(manifest):
    """Validate the entire frozen paired plan before counting any native result."""
    plan = manifest['plan']
    seeds, replicates = plan['seeds'], plan['replicates']
    if (not seeds or len(seeds) != len(set(seeds))
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or type(replicates) is not int or replicates < 1):
        raise ValueError('Invalid frozen seed/replicate inventory')
    expected = {f'{arm}:{seed}:{replicate}' for seed in seeds
                for replicate in range(1, replicates + 1) for arm in ('baseline', 'winner')}
    order, tasks = plan['execution_order'], manifest['tasks']
    by_key = {task['task_key']: task for task in tasks}
    if (set(order) != expected or len(order) != len(expected)
            or set(by_key) != expected or len(tasks) != len(expected)):
        raise ValueError('Task/pair inventory differs from the frozen complete plan')
    completed_ids = [task.get('run_id') for task in tasks if task.get('status') == 'completed']
    if len(set(completed_ids)) != len(completed_ids):
        raise ValueError('One actual run cannot count as multiple confirmation tasks')
    for key, task in by_key.items():
        if key != f'{task["arm"]}:{task["seed"]}:{task["replicate"]}':
            raise ValueError('Task identity differs from frozen task key')
    return by_key


def _scene_independence(profiles):
    """Use only supplied authorized scene metadata; unknown never implies overlap."""
    aliases = {'train': 'TRAIN', 'dev': 'DEV', 'validation': 'DEV',
               'confirm': 'CONFIRM', 'confirmation': 'CONFIRM', 'independent': 'CONFIRM'}
    def split(value):
        return aliases.get(str(value).lower(), str(value))
    records, conflicts, sources = [], [], []
    for entry in profiles:
        profile, role = entry['profile'], entry['role']
        source = {k: entry[k] for k in ('profile_ref', 'role')}
        source['profile_revision'] = profile.get('profile_revision')
        sources.append(source)
        for sample in profile.get('samples', []):
            group = split(sample.get('split', 'CONFIRM' if role == 'confirmation' else 'DEV'))
            if group not in ('TRAIN', 'DEV', 'CONFIRM'):
                continue
            records.append({'sample_id': sample.get('sample_id'), 'split': group,
                            'scene_id': sample.get('scene_id') or 'unknown', 'source': source})
        for overlap in profile.get('scene_overlap', []):
            groups = {split(value) for value in overlap.get('splits', [])}
            if 'CONFIRM' in groups and groups.intersection({'TRAIN', 'DEV'}):
                conflicts.append({'kind': 'explicit_scene_overlap', 'evidence': deepcopy(overlap), 'source': source})
    scenes = {}
    for record in records:
        if record['scene_id'] != 'unknown':
            scenes.setdefault(record['scene_id'], []).append(record)
    for scene, members in sorted(scenes.items()):
        groups = {record['split'] for record in members}
        if 'CONFIRM' in groups and groups.intersection({'TRAIN', 'DEV'}):
            conflicts.append({'kind': 'shared_scene_id', 'scene_id': scene, 'samples': members})
    known = (any(r['split'] in ('TRAIN', 'DEV') for r in records)
             and any(r['split'] == 'CONFIRM' for r in records)
             and all(r['scene_id'] != 'unknown' for r in records))
    return {'status': 'conflict' if conflicts else 'no_known_overlap' if known else 'unknown',
            'conflicts': conflicts, 'sources': sources, 'samples': records,
            'unknown_sample_ids': [r['sample_id'] for r in records if r['scene_id'] == 'unknown'],
            'claim_limit': ('Known TRAIN/DEV–confirmation scene overlap contradicts independent confirmation; scores remain descriptive.'
                            if conflicts else 'Only supplied scene annotations were checked; unknown identities do not establish independence.')}


def _frozen_inventory(details, profile=None, scope_kind='original_dev'):
    names = sorted(p.name for p in Path(details['input_dir']).iterdir()
                   if p.is_file() and p.suffix.lower() == '.png')
    count = runner.expected_count(details)
    samples = []
    allowed_splits = {'DEV', 'dev', 'validation'}
    if scope_kind == 'independent':
        allowed_splits = {'CONFIRM', 'confirm', 'confirmation', 'independent'}
        # A standalone confirmation profile may omit its split. Explicit DEV
        # records in a combined profile describe development data, not this inventory.
    declared = [sample for sample in (profile or {}).get('samples', [])
                if sample.get('split', 'CONFIRM' if scope_kind == 'independent' else 'DEV') in allowed_splits]
    for name in names:
        matches = [s for s in declared if s.get('input_relpath') == name or s.get('image') == name]
        if len(matches) > 1:
            raise ValueError('Ambiguous frozen inventory profile identity')
        sample = deepcopy(matches[0]) if matches else {'sample_id': name, 'scene_id': 'unknown', 'tags': []}
        sample.update(split='DEV', input_relpath=name, image=name)
        samples.append(sample)
    ids = [s['sample_id'] for s in samples]
    if len(set(ids)) != len(ids) or any(not isinstance(sid, str) or not sid for sid in ids):
        raise ValueError('Frozen inventory requires unique stable sample IDs')
    return {'expected_count': count, 'samples': samples, 'profile_revision': (profile or {}).get('profile_revision')}


def initialize_confirmation(search_campaign_dir, confirmation_dir, plan, *, control_ref):
    # Same parent lock as search/final TEST protects association and prelaunch gating.
    with _locked(search_campaign_dir, '.campaign.lock'):
        return _initialize_confirmation(search_campaign_dir, confirmation_dir, plan, control_ref=control_ref)


def _initialize_confirmation(search_campaign_dir, confirmation_dir, plan, *, control_ref):
    """Persist frozen plan first; insufficient authorization yields a reviewable block."""
    search = Path(search_campaign_dir).resolve()
    parent = json.loads((search / 'campaign.json').read_text(encoding='utf-8'))
    if parent.get('status') != 'finalized' or not parent.get('frozen'):
        raise ValueError('Confirmation requires a finalized frozen search campaign')
    plan = _canonical_plan(plan, search)
    cfg = runner.normalize_config(parent['config'])
    trials = parent.get('trials', [])
    baseline = trials[0] if trials else None
    winner = next((t for t in trials if t.get('run_id') == parent['frozen']['run_id']), None)
    if not baseline or not winner or not baseline.get('valid') or not winner.get('valid'):
        raise ValueError('Confirmation requires valid frozen baseline and winner')
    arms = {name: {'run_id': trial['run_id'], 'recipe': runner.validate_recipe(trial['recipe'])}
            for name, trial in (('baseline', baseline), ('winner', winner))}
    if plan.get('arms') is not None and plan['arms'] != arms:
        raise ValueError('Plan arms must match frozen baseline and winner exactly')
    plan['arms'] = arms
    if plan['scope']['kind'] == 'independent':
        details = plan['scope']['split_spec']
        for heldout in ('test', 'validation', 'train'):
            for left in ('input_dir', 'gt_dir', 'metadata_dir'):
                for right in ('input_dir', 'gt_dir', 'metadata_dir'):
                    if details.get(left) and cfg.get(heldout, {}).get(right) and _same_resource(details[left], cfg[heldout][right]):
                        name = 'TEST' if heldout == 'test' else heldout
                        raise ValueError(f'Independent confirmation split reuses {name} resource')
    # Validate control when valid; missing resource fields still produce a readable blocked plan.
    raw_control = json.loads(Path(control_ref).read_text(encoding='utf-8'))
    control = deepcopy(raw_control)
    control_error = None
    try:
        from .control import load_control
        control = load_control(control_ref)
    except ValueError as exc:
        control_error = str(exc)
    if not control_error and control['confirmation_scope'].get('split_spec'):
        control['confirmation_scope']['split_spec'] = _normalize_split(
            control['confirmation_scope']['split_spec'], Path(control_ref).resolve().parent)
    reasons = [control_error] if control_error else _authorization(control, plan, cfg)
    reasons.extend(_parent_reasons(parent, control))
    frozen_result = runner.get_result(winner['run_id'], cfg['runs_dir'])
    if not frozen_result.get('valid'):
        raise ValueError('Frozen winner result is not valid')
    association = {'confirmation_dir': str(Path(confirmation_dir).resolve()),
                   'parent_campaign': str(search), 'frozen_run_id': winner['run_id'],
                   'baseline_run_id': baseline['run_id'],
                   **{k: control.get(k) for k in ('campaign_id', 'protocol_id', 'source_revision', 'approved_config_ref')}}
    identity = {'parent_campaign': str(search), 'plan': plan, 'control': control, 'association': association,
                'control_ref': str(Path(control_ref).resolve())}
    with _locked(confirmation_dir) as directory:
        if (directory / 'confirmation.json').exists():
            existing = _read(directory)
            if any(existing.get(k) != v for k, v in identity.items()):
                raise ValueError('Confirmation already exists with a different frozen plan')
            if existing['launch_allowed']:
                _register_receipt(search, parent, existing, directory)
            return existing
        tasks = []
        for key in plan['execution_order']:
            arm, seed, replicate = key.split(':')
            tasks.append({'task_key': key, 'arm': arm, 'seed': int(seed), 'replicate': int(replicate), 'request_id': 'confirmation_' + key.replace(':', '_'), 'status': 'pending'})
        cfg.pop('test', None)
        cfg['runs_dir'] = str(directory / 'runs')
        profile = None
        profile_ref = control.get('observation_config', {}).get('profile_ref') or parent.get('profile_ref')
        if plan['scope']['kind'] == 'independent':
            profile_ref = control.get('confirmation_scope', {}).get('profile_ref')
        if profile_ref and Path(profile_ref).is_file():
            profile = json.loads(Path(profile_ref).read_text(encoding='utf-8'))
        scene_evidence = None
        if plan['scope']['kind'] == 'independent':
            profiles = []
            if profile is not None:
                profiles.append({'profile_ref': str(Path(profile_ref).resolve()), 'role': 'confirmation', 'profile': deepcopy(profile)})
            parent_refs = [control.get('observation_config', {}).get('profile_ref'), parent.get('profile_ref'),
                           (parent.get('control') or {}).get('observation_config', {}).get('profile_ref')]
            seen = {entry['profile_ref'] for entry in profiles}
            for ref in parent_refs:
                if ref and str(Path(ref).resolve()) not in seen and Path(ref).is_file():
                    resolved = str(Path(ref).resolve()); seen.add(resolved)
                    profiles.append({'profile_ref': resolved, 'role': 'development',
                                     'profile': json.loads(Path(ref).read_text(encoding='utf-8'))})
            scene_evidence = _scene_independence(profiles)
        details = cfg['validation'] if plan['scope']['kind'] == 'original_dev' else plan['scope']['split_spec']
        inventory = _frozen_inventory(details, profile, plan['scope']['kind']) if not reasons else None
        evaluation_split_spec = None
        if inventory:
            cfg['validation']['expected_count'] = runner.expected_count(cfg['validation'])
            if plan['scope']['kind'] == 'independent':
                evaluation_split_spec = {**plan['scope']['split_spec'], 'expected_count': inventory['expected_count']}
        state = {**identity, 'config': cfg, 'created_at': runner._now(), 'status': 'blocked' if reasons else 'ready',
                 'launch_allowed': not reasons, 'control_validated': control_error is None,
                 'blocked_reasons': reasons, 'tasks': tasks,
                 'search_checkpoint': {'run_id': winner['run_id'], **frozen_result['artifacts']},
                 'search_usage_gpu_hours': _search_usage(parent), 'profile': profile, 'inventory': inventory, 'evaluation_split_spec': evaluation_split_spec,
                 'parent_identity': {'config': deepcopy(parent['config']), 'frozen': deepcopy(parent['frozen']),
                                     'baseline': arms['baseline'], 'control': deepcopy(parent.get('control'))},
                 'calibration_gpu_hours': parent.get('usage', {}).get('calibration_gpu_hours'),
                 'scene_independence': scene_evidence, 'active_task_key': None}
        _save(directory, state)
        if state['launch_allowed']:
            _register_receipt(search, parent, state, directory)
        return state


def _search_usage(parent):
    usage = parent.get('usage', {})
    if type(usage.get('search_gpu_hours')) in (int, float):
        return usage['search_gpu_hours']
    values = [t.get('usage', {}).get('gpu_hours') for t in parent.get('trials', [])]
    return sum(values) if values and all(type(v) in (int, float) and math.isfinite(v) for v in values) else None


def _cost(state):
    if any(t['status'] == 'evaluating_confirmation' for t in state['tasks']):
        return None
    values = [t.get('usage', {}).get('gpu_hours') for t in state['tasks'] if t['status'] in ('completed', 'failed')]
    return sum(values) if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values) else None


def _start_allowed(directory, state, task, parent, reservation):
    """Called under the parent lock immediately before each actual launch."""
    from .control import reconcile_usage, check_action
    own = (str(Path(directory).resolve()), task['task_key'])
    limits = state['control']['limits']
    reasons = _parent_reasons(parent, state['control'], state['parent_identity'])
    ledger = reconcile_usage(state['control'], parent, exclude_confirmation=own)
    state['parent_usage'] = ledger
    decision = check_action(state['control'], parent, 'confirm', exclude_confirmation=own)
    if not decision['allowed']:
        reasons.append(decision['reason'])
    remaining = decision.get('remaining_budget')
    if remaining is not None and reservation > remaining['confirmation_gpu_hours'] + 1e-12:
        reasons.append('Full confirmation task reservation cannot fit while preserving other pools')
    if ledger['known'] and ledger['gpu_hours'] + ledger['reserved_gpu_hours'] + reservation > limits['total_gpu_hours'] + 1e-12:
        reasons.append('Reconciled total GPU budget exhausted')
    if ledger['known'] and ledger['confirmation_gpu_hours'] + reservation > limits['confirmation_gpu_hours'] + 1e-12:
        reasons.append('Shared confirmation GPU budget exhausted')
    costs = [t.get('usage', {}).get('gpu_hours') for t in state['tasks'] if t['status'] in ('completed', 'failed')]
    costs += [t['result'].get('usage', {}).get('gpu_hours') for t in state['tasks']
              if t['status'] not in ('completed', 'failed') and t.get('result')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in costs) or sum(costs) + reservation > state['plan']['budget_gpu_hours'] + 1e-12:
        reasons.append('Actual confirmation usage unavailable or budget exhausted')
    if reasons:
        state.update(status='blocked_parent', blocked_reasons=reasons)
        _save(directory, state)
        return False
    state.update(status='running', blocked_reasons=[], launch_allowed=True)
    return True


def _finish_task(directory, state, task, result):
    if task['status'] != 'checkpoint_frozen':
        task.update(result=result, usage=result.get('usage', {}), status='failed')
        if result.get('valid'):
            task['checkpoint'] = {k: result['artifacts'][k] for k in ('best_checkpoint', 'config_dir')}
            task['checkpoint'].update(selection_split='original_dev', frozen_at=runner._now())
            task['status'] = 'checkpoint_frozen'
            task['reserved_gpu_hours'] = (state['control']['limits']['job_walltime_seconds'] *
                                          state['control']['limits']['allocated_gpus'] / 3600
                                          if state['plan']['scope']['kind'] == 'independent' else 0)
            _save(directory, state)
    if result.get('valid') and state['plan']['scope']['kind'] == 'independent':
        output = Path(task['run_dir']) / 'confirmation_eval'
        with _locked(state['parent_campaign'], '.campaign.lock') as search:
            parent = json.loads((search / 'campaign.json').read_text(encoding='utf-8'))
            if not _start_allowed(directory, state, task, parent, task['reserved_gpu_hours']):
                return
            task['status'] = 'evaluating_confirmation'
            _save(directory, state)
            try:
                limits = state['control']['limits']
                runner.evaluate_frozen_checkpoint(state['config'], task['checkpoint'],
                    state['evaluation_split_spec'], str(output),
                    execution_limits={k: limits[k] for k in ('job_walltime_seconds', 'allocated_gpus')})
            except Exception as exc:
                task['error'] = str(exc)
        _collect_persisted_evaluation(directory, state, task)
        return
    if result.get('valid'):
        task.update(score=result['dev_psnr'], status='completed')
    else:
        task['error'] = result.get('error', 'Invalid confirmation run')
    state['active_task_key'] = None
    state['status'] = 'completed' if all(t['status'] in ('completed', 'failed') for t in state['tasks']) else 'ready'
    _save(directory, state)


def _collect_persisted_evaluation(directory, state, task):
    """Adopt only terminal evaluator evidence; never start another evaluation."""
    path = Path(task['run_dir']) / 'confirmation_eval' / 'evaluation_result.json'
    from .control import evaluator_is_terminal
    if not path.is_file() or not evaluator_is_terminal(path.parent):
        state['status'] = 'waiting_for_evaluation_recovery'
        _save(directory, state)
        return
    evaluation = json.loads(path.read_text(encoding='utf-8'))
    task.update(evaluation=evaluation, status='failed')
    try:
        if not evaluation.get('valid'):
            raise ValueError(evaluation.get('error', 'Frozen evaluation invalid'))
        artifacts = evaluation.get('artifacts', {})
        if any(artifacts.get(k) != task['checkpoint'].get(k) for k in ('best_checkpoint', 'config_dir')):
            raise ValueError('Persisted evaluation checkpoint identity differs from frozen checkpoint')
        score = runner.validate_metrics(evaluation['metrics'], state['inventory']['expected_count'], state['config']['eval_size'])
        if not math.isclose(score, evaluation['score'], abs_tol=1e-8, rel_tol=0):
            raise ValueError('Persisted evaluation aggregate differs from validated metrics')
        task.update(score=score, status='completed')
    except (ValueError, TypeError, KeyError) as exc:
        task['error'] = str(exc)
    train_cost = task['result'].get('usage', {}).get('gpu_hours')
    eval_cost = evaluation.get('usage', {}).get('gpu_hours')
    task['usage'] = {'gpu_hours': train_cost + eval_cost
                     if type(train_cost) in (int, float) and type(eval_cost) in (int, float) else None,
                     'training': task['result'].get('usage'), 'evaluation': evaluation.get('usage')}
    state['active_task_key'] = None
    state['status'] = 'completed' if all(t['status'] in ('completed', 'failed') for t in state['tasks']) else 'ready'
    _save(directory, state)


def run_confirmation_next(confirmation_dir, *, wait=True):
    """Advance at most one task; terminal tasks and unknown workers are never retried."""
    with _locked(confirmation_dir) as directory:
        state = _read(directory)
        if not state['launch_allowed']:
            if not state.get('control_validated') or _authorization(state['control'], state['plan'], state['config']):
                return state
        task = next((t for t in state['tasks'] if t['task_key'] == state.get('active_task_key')), None)
        if task is not None and task['status'] == 'evaluating_confirmation':
            _collect_persisted_evaluation(directory, state, task)
            return state
        if task is None:
            task = next((t for t in state['tasks'] if t['status'] == 'pending'), None)
            if task is None:
                return state
            limits = state['control']['limits']
            cost = _cost(state)
            reservation = limits['job_walltime_seconds'] * limits['allocated_gpus'] / 3600 * (2 if state['plan']['scope']['kind'] == 'independent' else 1)
            if cost is None or cost + reservation > state['plan']['budget_gpu_hours'] + 1e-12:
                state.update(status='blocked', launch_allowed=False, blocked_reasons=['Actual confirmation usage unavailable or budget exhausted'])
                _save(directory, state); return state
            with _locked(state['parent_campaign'], '.campaign.lock') as search:
                parent = json.loads((search / 'campaign.json').read_text(encoding='utf-8'))
                if not _start_allowed(directory, state, task, parent, reservation):
                    return state
                if state.get('inventory') is None:
                    scope = state['plan']['scope']
                    details = state['config']['validation'] if scope['kind'] == 'original_dev' else scope['split_spec']
                    state['inventory'] = _frozen_inventory(details, state.get('profile'), scope['kind'])
                    state['config']['validation']['expected_count'] = runner.expected_count(state['config']['validation'])
                    if scope['kind'] == 'independent':
                        state['evaluation_split_spec'] = {**scope['split_spec'], 'expected_count': state['inventory']['expected_count']}
                state['launch_allowed'] = True
                _save(directory, state)
                # Review-only blocked plans never join the executable ledger. Registration
                # occurs only after fresh authorization/identity/resource checks, before prepare.
                _register_receipt(search, parent, state, directory)
                cfg = {**state['config'], **state['plan']['arms'][task['arm']]['recipe'], 'seed': task['seed']}
                run_id = runner.prepare_run(cfg, request_id=task['request_id'], observation_config=state['control'].get('observation_config'),
                    execution_limits={k: limits[k] for k in ('job_walltime_seconds', 'allocated_gpus')})
                task.update(run_id=run_id, run_dir=str(Path(cfg['runs_dir']) / run_id), status='prepared', reserved_gpu_hours=reservation)
                state.update(active_task_key=task['task_key'], status='running', blocked_reasons=[])
                _save(directory, state)
                health = runner.inspect_run(task['run_id'], state['config']['runs_dir'])
                if health.get('liveness') == 'not_started':
                    runner.start_prepared_run(task['run_id'], state['config']['runs_dir'])
                    task['status'] = 'running'; _save(directory, state)
        if task['status'] == 'checkpoint_frozen':
            # Recovery never retrains or reselects a checkpoint.
            _finish_task(directory, state, task, task['result'])
            return state
        health = runner.inspect_run(task['run_id'], state['config']['runs_dir'])
        if health.get('liveness') == 'unknown':
            state['status'] = 'waiting_for_liveness'; _save(directory, state); return state
        if health.get('liveness') in ('not_started', 'dead'):
            with _locked(state['parent_campaign'], '.campaign.lock') as search:
                parent = json.loads((search / 'campaign.json').read_text(encoding='utf-8'))
                if not _start_allowed(directory, state, task, parent, task['reserved_gpu_hours']):
                    return state
                runner.start_prepared_run(task['run_id'], state['config']['runs_dir'])
                task['status'] = 'running'; _save(directory, state)
        result = runner.wait_for_result(task['run_id'], state['config']['runs_dir']) if wait else runner.get_result(task['run_id'], state['config']['runs_dir'])
        if result['status'] in ('completed', 'failed'):
            _finish_task(directory, state, task, result)
        return state


def _per_image(task, inventory):
    """Validate native CSV identity, frozen inventory and aggregate congruence."""
    outcome = {'status': 'unavailable', 'rows': {}, 'errors': [], 'native_count': None}
    if task['status'] != 'completed' or not inventory:
        outcome['errors'].append('Task or frozen inventory unavailable')
        return outcome
    evaluation = task.get('evaluation')
    artifacts = evaluation.get('artifacts', {}) if evaluation else task.get('result', {}).get('artifacts', {})
    metrics = evaluation.get('metrics', {}) if evaluation else task.get('result', {}).get('dev_metrics', {})
    metrics_path = artifacts.get('metrics') if evaluation else artifacts.get('dev_metrics')
    if not metrics_path or not Path(metrics_path).is_file():
        outcome['errors'].append('Native metrics artifact unavailable')
        return outcome
    try:
        native_metrics = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        outcome.update(status='incompatible')
        outcome['errors'].append('Native metrics artifact unreadable: ' + str(exc))
        return outcome
    if native_metrics != metrics:
        outcome.update(status='incompatible')
        outcome['errors'].append('Native metrics artifact differs from recorded task evidence')
        return outcome
    metrics = native_metrics
    outcome['native_count'] = metrics.get('num_images')
    path = artifacts.get('per_image_csv')
    if not path or not Path(path).is_file():
        outcome['errors'].append('Native per-image CSV unavailable')
        return outcome
    from .diagnostics import resolve_sample_id
    expected = {sample['sample_id'] for sample in inventory['samples']}
    try:
        with Path(path).open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                sid = resolve_sample_id(row, inventory)
                score = float(row['psnr'])
                if sid not in expected or sid in outcome['rows'] or not math.isfinite(score):
                    raise ValueError('Unknown, duplicate or nonfinite per-image sample')
                outcome['rows'][sid] = score
        if outcome['native_count'] != inventory['expected_count']:
            raise ValueError('Native num_images differs from frozen expected inventory')
        if len(outcome['rows']) != inventory['expected_count']:
            outcome.update(status='partial')
            outcome['errors'].append('Per-image CSV count differs from native num_images/frozen inventory')
        else:
            outcome['status'] = 'complete'
        native_mean = metrics.get('mean_per_image_psnr', metrics.get('mean_psnr'))
        if type(native_mean) not in (int, float) or not math.isfinite(native_mean):
            raise ValueError('Native mean per-image aggregate unavailable')
        if len(outcome['rows']) == inventory['expected_count']:
            row_mean = math.fsum(outcome['rows'].values()) / len(outcome['rows'])
            if not math.isclose(row_mean, native_mean, abs_tol=1e-6, rel_tol=0):
                raise ValueError('Per-image mean disagrees with native aggregate (tolerance 1e-6 dB)')
    except (ValueError, TypeError, KeyError, OSError) as exc:
        outcome.update(status='incompatible')
        outcome['errors'].append(str(exc))
    return outcome


def _scene_report(pairs, inventory):
    if not inventory or not pairs or any(p['per_image']['status'] != 'complete' for p in pairs):
        return {'status': 'unavailable', 'reason': 'Scene evidence requires complete congruent per-image inventory coverage.'}
    samples = {s['sample_id']: s for s in inventory['samples']}
    annotated = {sid: sample for sid, sample in samples.items() if sample.get('scene_id', 'unknown') != 'unknown'}
    if not annotated:
        return {'status': 'unavailable', 'reason': 'No authorized explicit scene annotations for this scope.'}
    output = []
    for pair in pairs:
        groups = {}
        unmapped = []
        for row in pair['per_image']['deltas']:
            sample = annotated.get(row['sample_id'])
            if sample is None:
                unmapped.append(row['sample_id']); continue
            groups.setdefault(sample['scene_id'], []).append(row['delta_db'])
        output.append({'seed': pair['seed'], 'replicate': pair['replicate'], 'unmapped_sample_ids': unmapped,
                       'expected_count': len(samples), 'annotated_count': len(annotated),
                       'scenes': [{'scene_id': scene, 'paired_images': len(values),
                                   'mean_delta_db': math.fsum(values) / len(values)} for scene, values in sorted(groups.items())]})
    return {'status': 'available', 'profile_revision': inventory.get('profile_revision'), 'pairs': output,
            'unit': 'paired per-image differences within each seed; seeds and images are not pooled'}


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Native evidence must contain an object: ' + str(path))
    return value


def _frozen_confirmation_config(manifest, directory, parent=None):
    """Bind fixed settings to the actual parent and its frozen native DEV count."""
    from .evidence import diff_scientific_config
    parent = parent if parent is not None else _read_object(Path(manifest['parent_campaign']) / 'campaign.json')
    identity = manifest.get('parent_identity', {})
    if (not isinstance(identity, dict) or identity.get('config') != parent.get('config')
            or identity.get('frozen') != parent.get('frozen')):
        raise ValueError('Frozen parent scientific config or winner identity differs from confirmation snapshot')
    base = _effective_config(parent['config'])
    base.pop('test', None)
    base['runs_dir'] = str(Path(directory).resolve() / 'runs')
    parent_result = runner.get_result(parent['frozen']['run_id'], parent['config']['runs_dir'])
    count = parent_result.get('expected_count')
    if (type(count) is not int or count < 1 or
            base['validation'].get('expected_count', count) != count):
        raise ValueError('Confirmation scientific config has no matching frozen parent DEV sample count')
    base['validation']['expected_count'] = count
    if diff_scientific_config(_effective_config(manifest['config']), base):
        raise ValueError('Confirmation scientific config differs from frozen parent')
    return base


def _actual_confirmation_score(manifest, task):
    from .runner import validate_metrics
    from .evidence import diff_scientific_config
    root = Path(manifest['_confirmation_dir']) / 'runs' / task['run_id']
    if Path(task.get('run_dir', '')).resolve() != root.resolve():
        raise ValueError('Task run directory does not match recorded run identity')
    result, config = _read_object(root / 'state.json'), _read_object(root / 'config.json')
    base = _frozen_confirmation_config(manifest, manifest['_confirmation_dir'])
    expected = _effective_config({**base, **manifest['plan']['arms'][task['arm']]['recipe'], 'seed': task['seed']})
    if diff_scientific_config(_effective_config(config), expected):
        raise ValueError('Actual confirmation scientific config differs from frozen task')
    if result.get('run_id') != task['run_id'] or result.get('status') != 'completed' or result.get('valid') is not True:
        raise ValueError('Task has no actual valid completed result')
    metrics = _read_object(root / 'dev/metrics.json')
    expected_count = base['validation']['expected_count']
    if result.get('expected_count') != expected_count:
        raise ValueError('Actual native sample count differs from frozen original DEV inventory')
    score = validate_metrics(metrics, expected_count, config['eval_size'])
    if result.get('dev_metrics') != metrics or not math.isclose(score, result.get('dev_psnr', float('nan')), abs_tol=1e-10, rel_tol=0):
        raise ValueError('Actual native DEV result/metrics disagree')
    if manifest['plan']['scope']['kind'] == 'independent':
        evaluation_root = root / 'confirmation_eval'
        evaluation = _read_object(evaluation_root / 'evaluation_result.json')
        evaluation_state = _read_object(evaluation_root / 'evaluation_state.json')
        frozen = evaluation_state.get('frozen', {})
        if not isinstance(frozen, dict):
            raise ValueError('Independent native evaluation frozen evidence must contain an object')
        split = manifest['evaluation_split_spec']
        actual_checkpoint = frozen.get('checkpoint', {})
        if not isinstance(actual_checkpoint, dict):
            raise ValueError('Independent native evaluation frozen checkpoint must contain an object')
        if (frozen.get('split') != split or {key: actual_checkpoint.get(key) for key in ('best_checkpoint', 'config_dir')} != {key: result['artifacts'][key] for key in ('best_checkpoint', 'config_dir')}
                or actual_checkpoint.get('selection_split', 'original_dev') != 'original_dev'):
            raise ValueError('Independent evaluation does not link frozen checkpoint/split')
        if evaluation.get('valid') is not True:
            raise ValueError('Independent evaluation invalid')
        metrics = _read_object(evaluation_root / 'metrics.json')
        score = validate_metrics(metrics, split['expected_count'], config['eval_size'])
        if evaluation.get('metrics') != metrics or not math.isclose(evaluation.get('score', float('nan')), score, abs_tol=1e-10, rel_tol=0):
            raise ValueError('Independent actual metrics/result disagree')
    if type(task.get('score')) not in (int, float) or not math.isclose(task['score'], score, abs_tol=1e-10, rel_tol=0):
        raise ValueError('Recorded task score differs from actual result')
    return score


def _final_checkpoint_candidate(search, parent, directory, receipt):
    """Validate registered authority and actual weight identity without reading TEST."""
    directory = Path(directory).resolve()
    state = _read(directory)
    plan, control = state['plan'], state['control']
    if (state.get('association') != receipt or receipt.get('confirmation_dir') != str(directory)
            or state.get('parent_campaign') != str(search) or receipt.get('parent_campaign') != str(search)):
        raise ValueError('Final checkpoint confirmation association differs from parent')
    for key in ('campaign_id', 'protocol_id', 'source_revision', 'approved_config_ref'):
        if receipt.get(key) != control.get(key):
            raise ValueError('Final checkpoint confirmation control binding differs: ' + key)
    # TEST may already be underway when checking an immutable reference. Its own
    # active attempt is not a reason to change or invalidate the selected weights.
    identity_parent = dict(parent)
    identity_parent.pop('final_test', None)
    reasons = _parent_reasons(identity_parent, control, state.get('parent_identity'))
    reasons.extend(_authorization(control, plan, state['config']))
    if not state.get('control_validated') or reasons:
        raise ValueError('Final checkpoint confirmation is not authorized: ' + '; '.join(reasons))
    trials = parent.get('trials', [])
    baseline = trials[0] if trials else {}
    winner = next((t for t in trials if t.get('run_id') == parent['frozen']['run_id']), {})
    if (receipt.get('baseline_run_id') != baseline.get('run_id') or
            receipt.get('frozen_run_id') != winner.get('run_id')):
        raise ValueError('Final checkpoint receipt differs from frozen baseline/winner')
    for name, trial in (('baseline', baseline), ('winner', winner)):
        if not trial.get('valid') or plan.get('arms', {}).get(name) != {'run_id': trial.get('run_id'), 'recipe': trial.get('recipe')}:
            raise ValueError('Final checkpoint confirmation arm differs from parent: ' + name)
    rule = plan['final_checkpoint_rule']
    if rule['kind'] != 'predeclared_seed':
        raise ValueError('Final checkpoint candidate must use a predeclared seed')
    if _canonical_plan(plan, search) != plan:
        raise ValueError('Final checkpoint plan differs from canonical frozen plan')
    tasks = [t for t in state['tasks'] if (t.get('arm'), t.get('seed'), t.get('replicate')) ==
             ('winner', rule['seed'], rule['replicate'])]
    if len(tasks) != 1 or tasks[0].get('task_key') != f'winner:{rule["seed"]}:{rule["replicate"]}':
        raise ValueError('Final checkpoint task identity is missing or ambiguous')
    task = tasks[0]
    if task.get('status') != 'completed' or not task.get('run_id'):
        raise ValueError('Predeclared final checkpoint task is not completed')
    _task_inventory(state)
    root = directory / 'runs' / task['run_id']
    if (Path(state['config']['runs_dir']).resolve() != directory / 'runs' or
            Path(task.get('run_dir', '')).resolve() != root):
        raise ValueError('Final checkpoint actual run directory differs from frozen task')
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    # Fixed science comes from the validated parent snapshot, never from a
    # confirmation manifest that could change together with its native run.
    base = _frozen_confirmation_config(state, directory, parent)
    from .evidence import diff_scientific_config
    expected = {**base, **plan['arms']['winner']['recipe'], 'seed': rule['seed']}
    if diff_scientific_config(_effective_config(cfg), _effective_config(expected)):
        raise ValueError('Final checkpoint actual scientific config differs from frozen task')
    _actual_confirmation_score({**state, '_confirmation_dir': str(directory)}, task)
    result = runner.get_result(task['run_id'], str(directory / 'runs'))
    recorded = task.get('result', {})
    if (result.get('run_id') != task['run_id'] or result.get('status') != 'completed'
            or result.get('valid') is not True or recorded.get('run_id') != task['run_id']
            or recorded.get('valid') is not True):
        raise ValueError('Final checkpoint has no actual completed valid result')
    checkpoint = task.get('checkpoint', {})
    train = json.loads((root / 'train' / 'metrics.json').read_text(encoding='utf-8'))
    for key in ('best_checkpoint', 'config_dir'):
        actual = result.get('artifacts', {}).get(key)
        if not actual or any(source.get(key) != actual for source in
                             (checkpoint, train, recorded.get('artifacts', {}))):
            raise ValueError('Final checkpoint artifacts differ from actual selected weights')
    if (checkpoint.get('selection_split', 'original_dev') != 'original_dev' or
            not Path(checkpoint['best_checkpoint']).is_file() or
            not (Path(checkpoint['config_dir']) / (Path(checkpoint['best_checkpoint']).stem + '.json')).is_file()):
        raise ValueError('Final checkpoint weights or native config are missing or invalid')
    return {'kind': rule['kind'], 'run_id': task['run_id'], 'runs_dir': str(directory / 'runs'),
            'best_checkpoint': checkpoint['best_checkpoint'], 'config_dir': checkpoint['config_dir'],
            'confirmation_dir': str(directory), 'task_key': task['task_key'],
            'seed': rule['seed'], 'replicate': rule['replicate'], 'association': deepcopy(receipt),
            'provenance': {**{key: deepcopy(state[key]) for key in ('plan', 'control', 'parent_identity')},
                           'resolved_config': expected}}


def report_confirmation(confirmation_dir):
    """Report fixed direction rule with failures, coverage and explicit claim limits."""
    state = _read(confirmation_dir)
    plan = state['plan']
    native_manifest = {**state, '_confirmation_dir': str(Path(confirmation_dir).resolve())}
    evidence_errors = []
    try:
        by_key = _task_inventory(state)
    except (ValueError, KeyError, TypeError) as exc:
        evidence_errors.append(str(exc))
        by_key = {task.get('task_key'): task for task in state['tasks']}
    pairs = []
    for seed in plan['seeds']:
        for replicate in range(1, plan['replicates'] + 1):
            arms = {arm: by_key.get(f'{arm}:{seed}:{replicate}', {'status': 'missing'})
                    for arm in ('baseline', 'winner')}
            complete = not evidence_errors and all(arms[a]['status'] == 'completed' for a in ('baseline', 'winner'))
            scores, pair_errors = {}, list(evidence_errors)
            if complete:
                for arm in ('baseline', 'winner'):
                    try:
                        scores[arm] = _actual_confirmation_score(native_manifest, arms[arm])
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        pair_errors.append(arm + ': ' + str(exc))
                complete = not pair_errors
            pair = {'seed': seed, 'replicate': replicate, 'status': 'complete' if complete else 'incomplete',
                    'baseline_run_id': arms['baseline'].get('run_id'), 'winner_run_id': arms['winner'].get('run_id'),
                    'delta_db': scores['winner'] - scores['baseline'] if complete else None,
                    'evidence_errors': pair_errors}
            inventory = state.get('inventory')
            left, right = _per_image(arms['baseline'], inventory), _per_image(arms['winner'], inventory)
            expected = {sample['sample_id'] for sample in (inventory or {}).get('samples', [])}
            shared = sorted(set(left['rows']) & set(right['rows']))
            status = 'complete' if left['status'] == right['status'] == 'complete' else (
                'incompatible' if 'incompatible' in (left['status'], right['status']) else
                'partial' if shared else 'unavailable')
            pair['per_image'] = {'status': status, 'paired_count': len(shared), 'expected_count': len(expected),
                'baseline_native_count': left['native_count'], 'winner_native_count': right['native_count'],
                'missing_in_winner': sorted(expected - set(right['rows'])),
                'missing_in_baseline': sorted(expected - set(left['rows'])),
                'missing_in_both': sorted(expected - (set(left['rows']) | set(right['rows']))),
                'errors': left['errors'] + right['errors'], 'metric_tolerance_db': 1e-6,
                'scope': 'full' if status == 'complete' else 'matched_subset',
                'deltas': [{'sample_id': key, 'delta_db': right['rows'][key] - left['rows'][key]} for key in shared]}
            pairs.append(pair)
    values = [p['delta_db'] for p in pairs if p['status'] == 'complete']
    mean = sum(values) / len(values) if values else None
    threshold = plan.get('delta_useful_db')
    conclusion = 'inconclusive'
    if len(values) == len(pairs) and threshold is not None:
        if all(v > 0 for v in values) and mean > threshold:
            conclusion = 'supported_in_scope'
        elif mean <= 0:
            conclusion = 'not_supported'
    if (state.get('scene_independence') or {}).get('conflicts'):
        conclusion = 'inconclusive'
    rule = plan['final_checkpoint_rule']
    final_checkpoint = state['search_checkpoint'] if rule['kind'] == 'retain_search_checkpoint' else next(
        ({'run_id': t.get('run_id'), **t.get('checkpoint', {}), 'available': t['status'] == 'completed'}
         for t in state['tasks'] if t['arm'] == 'winner' and t['seed'] == rule['seed'] and t['replicate'] == rule['replicate']), None)
    from .control import reconcile_usage
    current_parent = json.loads((Path(state['parent_campaign']) / 'campaign.json').read_text(encoding='utf-8'))
    ledger = reconcile_usage(state['control'], current_parent)
    final_checkpoint_ref = current_parent.get('final_checkpoint_ref')
    final_checkpoint_pending_adoption = False
    authority_error = None
    try:
        authority = _primary_receipt(current_parent)
        _validate_primary_plan(current_parent, state.get('association'), state)
        role = 'primary' if authority == state.get('association') else 'exploratory'
    except ValueError as exc:
        authority, role, authority_error = None, 'ambiguous', str(exc)
        conclusion = 'inconclusive'
    if role != 'primary':
        final_checkpoint = {'available': False, 'reason': authority_error or 'Exploratory plan has no final checkpoint authority'}
    if final_checkpoint_ref or (role == 'primary' and rule['kind'] == 'predeclared_seed' and final_checkpoint and final_checkpoint.get('available')):
        from .campaign import _resolve_final_checkpoint
        try:
            candidate = _resolve_final_checkpoint(Path(state['parent_campaign']).resolve(), current_parent, freeze=False)
            final_checkpoint = {**candidate, 'available': True}
            final_checkpoint_pending_adoption = final_checkpoint_ref is None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            conclusion = 'inconclusive'
            final_checkpoint = {'available': False, 'reason': str(exc)}
            evidence_errors.append('Final checkpoint evidence unavailable: ' + str(exc))
    report = {'parent_campaign': state['parent_campaign'], 'association': state.get('association'), 'status': state['status'], 'scope': plan['scope'],
              'conclusion': conclusion, 'scene_independence': state.get('scene_independence'), 'pairs': pairs, 'tasks': state['tasks'], 'mean_delta_db': mean,
              'complete_pairs': len(values), 'planned_pairs': len(pairs), 'delta_useful_db': threshold,
              'descriptive_only': threshold is None, 'statistical_significance': False,
              'adaptive_dev_bias': plan['scope']['kind'] == 'original_dev',
              'adaptive_bias_reason': 'Original DEV was used adaptively in search; repeated seeds do not remove its selection bias.' if plan['scope']['kind'] == 'original_dev' else None,
              'claim_limit': 'Small paired-seed direction rule within frozen scope; no statistical significance or product/generalization claim.',
              'final_checkpoint_rule': rule, 'final_checkpoint': final_checkpoint,
              'confirmation_role': role, 'primary_confirmation': current_parent.get('primary_confirmation'),
              'authority_error': authority_error,
              'evidence_errors': evidence_errors + [error for pair in pairs for error in pair['evidence_errors']],
              'final_checkpoint_ref': final_checkpoint_ref,
              'final_checkpoint_pending_adoption': final_checkpoint_pending_adoption,
              'usage': {'confirmation_gpu_hours': _cost(state), 'search_gpu_hours': ledger['search_gpu_hours'] if ledger['known'] else None,
                        'final_test_gpu_hours': ledger['final_test_gpu_hours'] if ledger['known'] else None,
                        'active_reserved_gpu_hours': sum(t.get('reserved_gpu_hours', 0) for t in state['tasks'] if t['status'] not in ('pending', 'completed', 'failed')),
                        'calibration_gpu_hours': ledger['calibration_gpu_hours'] if ledger['known'] else None},
              'campaign_usage_ledger': ledger,
              'split_identity_limit': 'Directory identity, symlink and ancestry checks only; no held-out file enumeration or content comparison.',
              'scenes': _scene_report(pairs, state.get('inventory')),
              'blocked_reasons': state['blocked_reasons']}
    runner._write_json(Path(confirmation_dir) / 'report.json', report)
    return report
