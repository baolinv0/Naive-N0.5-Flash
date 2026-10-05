"""Explicit research authorization and resource boundaries."""

import copy
import json
import math
from pathlib import Path


RECIPE_FIELDS = {'loss_family', 'optimizer', 'learning_rate', 'weight_decay'}
LIMIT_FIELDS = ('job_walltime_seconds', 'total_gpu_hours', 'allocated_gpus',
                'max_search_trials', 'confirmation_gpu_hours', 'calibration_gpu_hours')


def _number(value, key, *, positive=False, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{key} must be a finite nonnegative number')
    if positive and value <= 0:
        raise ValueError(f'{key} must be positive')
    if integer and type(value) is not int:
        raise ValueError(f'{key} must be an integer')
    return value


def _validate(control):
    if not isinstance(control, dict):
        raise ValueError('control must be an object')
    required = {'schema_version', 'campaign_id', 'parent_campaign_id', 'question',
                'authorization_ref', 'source_revision', 'approved_config_ref', 'protocol_id',
                'primary_metric', 'allowed_recipe_fields', 'limits', 'slow_policy',
                'confirmation_scope', 'test_permission', 'observation_mode', 'observation_config'}
    missing = sorted(required - control.keys())
    if missing:
        raise ValueError('Missing control fields: ' + ', '.join(missing))
    if type(control['schema_version']) is not int or control['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    for key in ('campaign_id', 'question', 'authorization_ref', 'source_revision',
                'approved_config_ref', 'protocol_id', 'primary_metric'):
        if not isinstance(control[key], str) or not control[key].strip():
            raise ValueError(f'{key} requires an explicit nonempty value')
    if '/' in control['campaign_id'] or '\\' in control['campaign_id']:
        raise ValueError('campaign_id cannot contain path separators')
    if control['parent_campaign_id'] is not None and not isinstance(control['parent_campaign_id'], str):
        raise ValueError('parent_campaign_id must be a string or null')
    fields = control['allowed_recipe_fields']
    if not isinstance(fields, list) or not fields or not all(isinstance(k, str) for k in fields) or not set(fields) <= RECIPE_FIELDS:
        raise ValueError('allowed_recipe_fields must be a nonempty subset of the four recipe fields')
    if not isinstance(control['limits'], dict):
        raise ValueError('limits must be an object')
    limits = control['limits']
    for key in LIMIT_FIELDS:
        _number(limits.get(key), key, positive=key in ('job_walltime_seconds', 'max_search_trials'),
                integer=key in ('allocated_gpus', 'max_search_trials'))
    if limits['confirmation_gpu_hours'] + limits['calibration_gpu_hours'] > limits['total_gpu_hours']:
        raise ValueError('confirmation and calibration reserves exceed total_gpu_hours')
    if type(control['test_permission']) is not bool:
        raise ValueError('test_permission must be an explicit boolean')
    for key in ('slow_policy', 'confirmation_scope', 'observation_config'):
        if not isinstance(control[key], dict):
            raise ValueError(f'{key} must be an object')
    threshold = control['slow_policy'].get('consecutive_valid_no_gain', 2)
    _number(threshold, 'consecutive_valid_no_gain', positive=True, integer=True)
    scope = control['confirmation_scope']
    if type(scope.get('authorized')) is not bool:
        raise ValueError('confirmation_scope.authorized must be explicit')
    if scope['authorized']:
        if scope.get('kind') not in ('original_dev', 'independent'):
            raise ValueError('confirmation_scope.kind must be original_dev or independent')
        if not isinstance(scope.get('seeds'), list) or not scope['seeds'] or any(type(s) is not int for s in scope['seeds']):
            raise ValueError('confirmation_scope.seeds must contain explicit integer seeds')
        if scope['kind'] == 'independent' and not isinstance(scope.get('split_spec'), dict):
            raise ValueError('independent confirmation_scope requires split_spec')
    if control['observation_mode'] not in ('text', 'image'):
        raise ValueError('observation_mode must be text or image')
    if control['observation_mode'] == 'image':
        capability = control.get('visual_capability', {})
        if not isinstance(capability, dict) or capability.get('verified') is not True or not capability.get('evidence_ref'):
            raise ValueError('image mode requires verified visual_capability and evidence_ref')
    sections = control['observation_config'].get('required_for_proposal', [])
    if not isinstance(sections, list) or any(not isinstance(s, str) or not s for s in sections):
        raise ValueError('observation_config.required_for_proposal must be a list of section names')
    for key, bounds in control.get('recipe_ranges', {}).items():
        if key not in ('learning_rate', 'weight_decay') or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError('recipe_ranges requires learning_rate/weight_decay [min,max]')
        for value in bounds:
            _number(value, key, positive=key == 'learning_rate')
        if bounds[0] > bounds[1]:
            raise ValueError(f'{key} range is reversed')
    return control


def load_control(path):
    """Load a validated, independent snapshot; absent authorization is never inferred."""
    return copy.deepcopy(_validate(json.loads(Path(path).read_text(encoding='utf-8'))))


def reconcile_usage(control, state, usage=None, *, exclude_confirmation=None):
    """Reconcile actual records and partial summaries without erasing either.

    Persisted terminal records are mandatory evidence, even when a caller supplies
    an aggregate. Category summaries are lower bounds for pools whose detailed
    records may live elsewhere. A total is a lower bound over the category sum,
    never an alternative to measured search/TEST/confirmation costs.
    """
    limits = (control or {}).get('limits', {})
    job = (limits['job_walltime_seconds'] * limits['allocated_gpus'] / 3600
           if type(limits.get('job_walltime_seconds')) in (int, float) and type(limits.get('allocated_gpus')) is int else None)
    keys = ('search_gpu_hours', 'confirmation_gpu_hours', 'calibration_gpu_hours', 'final_test_gpu_hours')
    ledger = {key: 0.0 for key in keys}
    ledger.update(known=True, gpu_hours=0.0, reserved_gpu_hours=0.0, active_attempts=[], reason='Reconciled actual records and summary lower bounds')
    def measured(record, label):
        value = (record.get('usage') or {}).get('gpu_hours')
        if value is None and limits.get('allocated_gpus') == 0:
            return 0.0
        return _number(value, label + '.usage.gpu_hours')
    def active(record, label):
        ledger['active_attempts'].append(label)
        value = record.get('reserved_gpu_hours', record.get('reservation_gpu_hours', job))
        ledger['reserved_gpu_hours'] += _number(value, label + '.reserved_gpu_hours')
    try:
        ledger['search_gpu_hours'] = sum(measured(t, 'trial') for t in state.get('trials', []))
        ledger['calibration_gpu_hours'] = sum(measured(t, 'calibration') for t in state.get('calibration_trials', []))
        if state.get('active'):
            active(state['active'], 'search')
        elif state.get('launch_intent'):
            active(state['launch_intent'], 'search_launch_intent')
        final = state.get('final_test')
        if final:
            if final.get('status') in ('completed', 'failed'):
                ledger['final_test_gpu_hours'] += measured(final, 'final_test')
            else:
                active(final, 'final_test')
        seen = set()
        for receipt in state.get('confirmation_refs', []):
            directory = str(Path(receipt['confirmation_dir']).resolve())
            if directory in seen:
                continue
            seen.add(directory)
            manifest = json.loads((Path(directory) / 'confirmation.json').read_text(encoding='utf-8'))
            if manifest.get('association') != receipt:
                raise ValueError('Confirmation receipt does not match frozen manifest association')
            for key in ('campaign_id', 'protocol_id', 'source_revision', 'approved_config_ref'):
                if receipt.get(key) != (control or manifest.get('control', {})).get(key):
                    raise ValueError('Confirmation receipt identity differs from control: ' + key)
            for task in manifest.get('tasks', []):
                status = task.get('status')
                if status in ('completed', 'failed'):
                    ledger['confirmation_gpu_hours'] += measured(task, 'confirmation')
                elif status != 'pending':
                    # Completed TRAIN+DEV remains an actual cost during the evaluator
                    # reservation, including when its own start gate excludes that reservation.
                    if task.get('result'):
                        ledger['confirmation_gpu_hours'] += measured(task['result'], 'confirmation.training')
                    if exclude_confirmation != (directory, task.get('task_key')):
                        active(task, 'confirmation:' + task.get('task_key', 'unknown'))
        aggregate_lower_bounds = []
        for summary in (state.get('usage'), usage):
            if summary is None:
                continue
            if not isinstance(summary, dict):
                raise ValueError('usage must be an object')
            for key in keys:
                if key in summary:
                    value = summary[key]
                    if value is None and limits.get('allocated_gpus') == 0:
                        value = 0.0
                    ledger[key] = max(ledger[key], _number(value, 'usage.' + key))
            if 'gpu_hours' in summary:
                value = summary['gpu_hours']
                if value is None and limits.get('allocated_gpus') == 0:
                    value = 0.0
                aggregate_lower_bounds.append(_number(value, 'usage.gpu_hours'))
            if 'reserved_gpu_hours' in summary:
                ledger['reserved_gpu_hours'] = max(ledger['reserved_gpu_hours'], _number(summary['reserved_gpu_hours'], 'usage.reserved_gpu_hours'))
        ledger['gpu_hours'] = max([sum(ledger[key] for key in keys)] + aggregate_lower_bounds)
        ledger['unattributed_gpu_hours'] = max(0.0, ledger['gpu_hours'] - sum(ledger[key] for key in keys))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        ledger.update(known=False, gpu_hours=None, reason='GPU usage is unknown or invalid: ' + str(exc))
    return ledger


def _usage(control, state, usage, exclude_confirmation=None):
    limits = control['limits']
    ledger = reconcile_usage(control, state, usage, exclude_confirmation=exclude_confirmation)
    if not ledger['known']:
        return None
    total, reserved = ledger['gpu_hours'], ledger['reserved_gpu_hours']
    remaining_total = max(0.0, limits['total_gpu_hours'] - total - reserved)
    remaining_confirmation = max(0.0, limits['confirmation_gpu_hours'] - ledger['confirmation_gpu_hours'])
    remaining_calibration = max(0.0, limits['calibration_gpu_hours'] - ledger['calibration_gpu_hours'])
    return {'total_gpu_hours': remaining_total,
            'search_gpu_hours': max(0.0, remaining_total - remaining_confirmation - remaining_calibration),
            'confirmation_gpu_hours': min(max(0, remaining_total - remaining_calibration), remaining_confirmation),
            'calibration_gpu_hours': min(max(0, remaining_total - remaining_confirmation), remaining_calibration),
            'reserved_gpu_hours': reserved,
            'next_job_gpu_hours': limits['job_walltime_seconds'] * limits['allocated_gpus'] / 3600,
            'budget_exceeded': total + reserved > limits['total_gpu_hours'] + 1e-12,
            'search_trials': max(0, limits['max_search_trials'] - len(state.get('trials', [])) - bool(state.get('active') or state.get('launch_intent'))),
            'ledger': ledger}


def pending_decision_reason(state, action):
    """Apply an explicitly recorded slow intent without inferring authorization."""
    name = action.get('action') if isinstance(action, dict) else action
    if name not in ('baseline', 'propose', 'search', 'calibrate', 'confirm', 'final_test'):
        return None
    pending_action = (state.get('pending_slow_decision') or {}).get('action')
    if pending_action and pending_action != 'propose':
        search = name in ('baseline', 'propose', 'search', 'calibrate')
        if (search or pending_action in ('diagnose', 'wait', 'request_scope_change')
                or (pending_action == 'finalize' and not state.get('frozen'))):
            return 'Pending slow decision requires ' + pending_action
    return None


def check_action(control, state, action, usage=None, *, exclude_confirmation=None):
    """Return a gate, never authorization or an execution command.

    ``action`` is a name or {action,recipe}; usage is total ``gpu_hours`` or
    category totals, plus ``reserved_gpu_hours``. Missing GPU measurements block
    new work. Explicit zero allocated GPUs remains bounded CPU work.
    """
    _validate(control)
    specification = action if isinstance(action, dict) else {'action': action}
    name = specification.get('action')
    try:
        remaining = _usage(control, state, usage, exclude_confirmation)
    except ValueError as exc:
        return {'allowed': False, 'reason': str(exc), 'requires_human_decision': False, 'remaining_budget': None}
    def result(allowed, reason, human=False):
        return {'allowed': allowed, 'reason': reason, 'requires_human_decision': human, 'remaining_budget': remaining}
    if name in ('wait', 'diagnose', 'report_stop', 'finalize'):
        return result(True, 'Read-only or closure action within existing scope')
    if name in ('request_scope_change', 'continue_search'):
        return result(False, 'A scope change requires a new authorized campaign', True)
    if name not in ('baseline', 'propose', 'search', 'confirm', 'calibrate', 'final_test'):
        return result(False, 'Unknown action')
    if (state.get('active') or state.get('launch_intent') or state.get('liveness') == 'unknown' or state.get('active_health') == 'unknown'
            or (state.get('final_test') and state['final_test'].get('status') not in ('completed', 'failed'))
            or (remaining is not None and remaining['ledger']['active_attempts'])):
        return result(False, 'An active or unknown job prevents another start')
    if state.get('research_hold'):
        return result(False, 'Research hold requires evidence-resolved diagnosis')
    pending_reason = pending_decision_reason(state, name)
    if pending_reason:
        return result(False, pending_reason,
                      (state.get('pending_slow_decision') or {}).get('action') == 'request_scope_change')
    if name in ('baseline', 'propose', 'search'):
        if state.get('frozen') or state.get('stop_reason') or state.get('status') in ('stopped', 'finalized'):
            return result(False, 'Frozen or hard-stopped search cannot continue')
        if state.get('max_trials', 0) > control['limits']['max_search_trials']:
            return result(False, 'max_trials exceeds authorized max_search_trials')
        if state.get('protocol_id', control['protocol_id']) != control['protocol_id']:
            return result(False, 'protocol_id does not match control')
        if state.get('primary_metric', control['primary_metric']) != control['primary_metric']:
            return result(False, 'primary_metric does not match control')
        patch = specification.get('recipe', {})
        if not isinstance(patch, dict) or not set(patch) <= set(control['allowed_recipe_fields']):
            return result(False, 'Recipe change exceeds allowed_recipe_fields', True)
        for key, bounds in control.get('recipe_ranges', {}).items():
            value = patch.get(key, state.get('config', {}).get(key))
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]):
                return result(False, f'{key} exceeds authorized recipe range', True)
    if name == 'final_test' and not control['test_permission']:
        return result(False, 'TEST evaluation is not authorized', True)
    if name in ('confirm', 'final_test') and not state.get('frozen'):
        return result(False, 'Confirmation/TEST requires a frozen candidate')
    if name == 'confirm' and not control['confirmation_scope']['authorized']:
        return result(False, 'Confirmation scope is not authorized', True)
    if remaining is None:
        return result(False, 'GPU usage is unknown; retain reservations and diagnose')
    if remaining['budget_exceeded']:
        return result(False, 'Actual usage/reservations exceed total_gpu_hours')
    if name in ('baseline', 'propose', 'search') and not remaining['search_trials']:
        return result(False, 'Authorized search trial limit reached')
    pool = ('confirmation_gpu_hours' if name == 'confirm' else 'calibration_gpu_hours'
            if name == 'calibrate' else 'total_gpu_hours' if name == 'final_test' else 'search_gpu_hours')
    if remaining[pool] + 1e-12 < remaining['next_job_gpu_hours']:
        return result(False, f'Insufficient {pool} for bounded next job')
    return result(True, 'Action fits frozen authorization and bounded resources')


def evaluator_is_terminal(output):
    """A receipt's validity says nothing about resource exit; runner state does.

    Missing/malformed state and recovery_required retain the reservation even on
    CPU. Inspect recorded resources too, because the watchdog can still be
    settling a terminal receipt. A stale health file cannot override newer state.
    """
    from . import runner
    try:
        state = json.loads((Path(output) / 'evaluation_state.json').read_text(encoding='utf-8'))
        return (state.get('status') in runner.TERMINAL
                and runner._recorded_resource_liveness(Path(output)) == 'dead')
    except (OSError, ValueError, TypeError):
        return False
