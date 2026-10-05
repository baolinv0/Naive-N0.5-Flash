"""Strict scalar observation projection for the feedback ablation."""

import copy
import math
import re

from .experiment_space import STRATEGIES, validate_space_recipe


_STATUSES = {'ready', 'running', 'stopped', 'finalized', 'completed', 'failed',
             'interrupted', 'cancelled', 'prepared', 'pending', 'invalid'}
_RESULTS = {'valid', 'baseline', 'improved', 'no_gain', 'invalid'}
_ACTIONS = {'propose', 'wait', 'diagnose', 'finalize', 'report_stop', 'confirm', 'request_scope_change'}
_NUMERIC = {'remaining_trials', 'min_delta', 'no_gain_count', 'no_gain_limit',
            'consecutive_valid_no_gain', 'seed', 'epochs', 'batch_size', 'in_size',
            'eval_size', 'validation_frequency', 'num_workers', 'job_walltime_seconds',
            'total_gpu_hours', 'search_gpu_hours', 'confirmation_gpu_hours',
            'calibration_gpu_hours', 'reserved_gpu_hours', 'next_job_gpu_hours',
            'search_trials', 'gpu_hours', 'allocated_gpus', 'max_search_trials',
            'final_test_gpu_hours', 'unattributed_gpu_hours'}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _identity(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9_-]+', value))


def _numbers(value):
    if not isinstance(value, dict):
        return {}
    result = {key: item for key, item in value.items() if key in _NUMERIC and _finite(item)}
    for key in ('known', 'budget_exceeded'):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    if isinstance(value.get('ledger'), dict):
        result['ledger'] = _numbers(value['ledger'])
    return result


def _trial(value):
    if not isinstance(value, dict):
        return None
    result = {}
    if _identity(value.get('run_id')):
        result['run_id'] = value['run_id']
    if 'recipe' in value:
        result['recipe'] = validate_space_recipe(value['recipe'])
    if type(value.get('valid')) is bool:
        result['valid'] = value['valid']
    score = value.get('dev_psnr')
    if 'dev_psnr' in value and (score is None or _finite(score)):
        result['dev_psnr'] = score
    if value.get('status') in _STATUSES:
        result['status'] = value['status']
    if value.get('result_status') in _RESULTS:
        result['result_status'] = value['result_status']
    if 'usage' in value:
        result['usage'] = _numbers(value['usage'])
    return result


def aggregate_observation(feedback, *, execution_status=None):
    """Reconstruct a *present* native O1; never invent an absent observation."""
    if not isinstance(feedback, dict):
        raise ValueError('Native feedback with an aggregate/execution O1 observation is required')
    matches = [item for item in feedback.get('observations', []) if isinstance(item, dict)
               and item.get('id', item.get('observation_id')) == 'O1']
    if len(matches) != 1:
        raise ValueError('Exactly one present native O1 observation is required')
    native = matches[0]
    metric = native.get('metric')
    expected = {'mean_per_image_rgb_psnr': 'summary.json#/overall',
                'execution_status': 'summary.json#/execution'}
    if metric not in expected or native.get('source') != expected[metric]:
        raise ValueError('O1 must identify the native aggregate or execution source')
    value = native.get('value')
    if metric == 'mean_per_image_rgb_psnr' and not _finite(value):
        raise ValueError('Native aggregate O1 requires a finite measured value')
    if metric == 'execution_status' and value is not None:
        raise ValueError('Execution O1 cannot contain a fabricated quality score')
    result = {'id': 'O1', 'observation_id': 'O1', 'metric': metric,
              'value': value, 'source': expected[metric]}
    if type(native.get('n_images')) is int and native['n_images'] >= 0:
        result['n_images'] = native['n_images']
    result['finding'] = (f'Recorded native DEV mean per-image RGB PSNR is {value} dB.'
                         if metric == 'mean_per_image_rgb_psnr' else
                         'No valid native DEV quality score is available.' +
                         (f' Execution status: {execution_status}.' if execution_status in _STATUSES else ''))
    return result


def _scalar_feedback(feedback, history):
    if feedback is None:
        return None
    if not isinstance(feedback, dict):
        raise ValueError('feedback must be an object or null')
    result = {}
    for key in ('campaign_id', 'run_id', 'latest_run_id', 'feedback_revision',
                'baseline_run_id', 'best_before_run_id', 'comparison_run_id',
                'construction_base_run_id', 'based_on_run_id'):
        if _identity(feedback.get(key)):
            result[key] = feedback[key]
    # Only the current native summary identity is exposed, not historical
    # details_index, logs, cases, plots, sample paths or diagnosis narratives.
    if isinstance(feedback.get('feedback_ref'), str):
        result['feedback_ref'] = feedback['feedback_ref']
    if type(feedback.get('proposal_ready')) is bool:
        result['proposal_ready'] = feedback['proposal_ready']
    last = history[-1] if history else {}
    execution = feedback.get('execution') or {}
    status = execution.get('status') or last.get('status')
    if status in _STATUSES:
        result['execution'] = {'status': status}
    overall = feedback.get('overall') or {}
    result['overall'] = {}
    score = overall.get('dev_psnr')
    if 'dev_psnr' in overall and (score is None or _finite(score)):
        result['overall']['dev_psnr'] = score
    if type(overall.get('num_images')) is int and overall['num_images'] >= 0:
        result['overall']['num_images'] = overall['num_images']
    if overall.get('primary_metric') == 'mean_per_image_rgb_psnr':
        result['overall']['primary_metric'] = 'mean_per_image_rgb_psnr'
    result['observations'] = [aggregate_observation(feedback, execution_status=status)]
    if 'usage' in feedback:
        result['usage'] = _numbers(feedback['usage'])
    return result


def project_feedback(packet, strategy):
    """Rebuild scalar input from allowlists, or copy the rich input in full."""
    if strategy not in STRATEGIES or not isinstance(packet, dict):
        raise ValueError('Known strategy and feedback packet object are required')
    if strategy in ('naive_rich', 'naive_fast_slow'):
        return copy.deepcopy(packet)
    result = _numbers(packet)
    if _identity(packet.get('campaign_id')):
        result['campaign_id'] = packet['campaign_id']
    if type(packet.get('proposal_allowed')) is bool:
        result['proposal_allowed'] = packet['proposal_allowed']
    if packet.get('status') in _STATUSES:
        result['status'] = packet['status']
    if packet.get('stop_reason') in (None, 'max_trials', 'no_gain_limit', 'user_finalized'):
        result['stop_reason'] = packet.get('stop_reason')
    result['history'] = [_trial(item) for item in packet.get('history', [])]
    index = packet.get('history_index', result['history'])
    if not isinstance(index, list):
        raise ValueError('history_index must be a list of native trial identities')
    result['history_index'] = []
    for entry in index:
        if not isinstance(entry, dict) or not _identity(entry.get('run_id')):
            raise ValueError('history_index entries require a native run_id')
        identifier = {'run_id': entry['run_id']}
        if entry.get('result_status') in _RESULTS:
            identifier['result_status'] = entry['result_status']
        result['history_index'].append(identifier)
    for key in ('best', 'raw_best', 'active', 'launch_intent'):
        if key in packet:
            result[key] = _trial(packet[key])
    result['fixed'] = _numbers(packet.get('fixed'))
    result['remaining_budget'] = _numbers(packet.get('remaining_budget')) if packet.get('remaining_budget') is not None else None
    route = packet.get('next_action') or {}
    result['next_action'] = {}
    if route.get('action') in _ACTIONS:
        result['next_action']['action'] = route['action']
    if type(route.get('review_required')) is bool:
        result['next_action']['review_required'] = route['review_required']
    result['next_action'].update(_numbers(route))
    if isinstance(route.get('trigger_id'), str) and re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', route['trigger_id']):
        result['next_action']['trigger_id'] = route['trigger_id']
    result['feedback'] = _scalar_feedback(packet.get('feedback'), result['history'])
    if result['feedback'] and result['feedback'].get('feedback_ref'):
        result['next_action']['evidence_refs'] = [result['feedback']['feedback_ref']]
    result['prompt'] = ('Follow next_action.action and proposal_allowed. Use only the recorded complete '
        'recipes, native aggregate DEV scores, execution validity and resource bounds. '
        'Return proposal={recipe,hypothesis,based_on={run_id,dev_psnr,observation}} and a '
        'fast_proposal decision sidecar citing the present O1 with the current feedback revision. '
        'Provide a prediction, alternative explanation and falsifier. Each proposal must contain '
        'all four discrete recipe fields. Keep fixed science unchanged. TEST and CONFIRM are inaccessible.')
    return result
