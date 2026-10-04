"""Persistent serial controller; an external ARIS/Naive agent proposes each recipe."""

from contextlib import contextmanager
import fcntl
import json
import math
import re
import uuid
from pathlib import Path

from .runner import (BUDGET_DEFAULTS, RECIPE_DEFAULTS, _execute, _now, _resolve_evaluation,
                     _write_json, evaluation_command, expected_count, get_result,
                     normalize_config, prepare_run, start_prepared_run, inspect_run, validate_metrics, validate_recipe,
                     wait_for_result, evaluate_frozen_checkpoint)


from .evidence import (build_run_feedback, build_fallback_feedback, resolve_trial_references,
                       diff_scientific_config, validate_feedback_access)
from .control import load_control, check_action
from .research import route_next_action, validate_research_decision, _persist_decision, _safe_feedback


# Engineering default only; calibrate on repeated DEV reloads before a live campaign.
DEFAULT_MIN_DELTA = 0.01


@contextmanager
def _locked(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another campaign command is active; use status after it returns') from exc
        yield directory


def _read(directory):
    state = json.loads((Path(directory) / 'campaign.json').read_text(encoding='utf-8'))
    # Preserve the decision rule of campaigns created before min_delta existed.
    state.setdefault('min_delta', 0.0)
    state.setdefault('raw_best', state.get('best'))
    return state


def _save(directory, state):
    state['updated_at'] = _now()
    _write_json(Path(directory) / 'campaign.json', state)


def _stop_reason(state):
    if state['trials'] and not state['trials'][0]['valid']:
        return 'invalid_baseline'
    if len(state['trials']) >= state['max_trials']:
        return 'max_trials'
    if state['no_gain_count'] >= state['no_gain_limit']:
        return 'no_gain_limit'
    return None


def _collect(directory, state):
    if state.get('launch_intent') and not state.get('active'):
        _resume_launch(directory, state)
    active = state.get('active')
    if not active:
        _refresh_latest_feedback(directory, state)
        return state
    health = inspect_run(active['run_id'], state['config']['runs_dir'])
    state['active_health'] = health.get('liveness', 'unknown')
    if state['active_health'] == 'not_started':
        if not _current_start_allowed(directory, state, 'active'):
            _refresh_latest_feedback(directory, state)
            return state
        _save(directory, state)
        start_prepared_run(active['run_id'], state['config']['runs_dir'])
    elif state.get('launch_block'):
        # A hold pauses future starts, not work already verified underway.
        state.pop('launch_block', None)
        state['status'] = 'running'
    result = get_result(active['run_id'], state['config']['runs_dir'])
    if result['status'] not in ('completed', 'failed'):
        _refresh_latest_feedback(directory, state)
        return state
    trial = {**active, **{key: result.get(key) for key in
        ('status', 'valid', 'result_status', 'dev_psnr', 'dev_metrics', 'artifacts', 'logs', 'error', 'reload_succeeded', 'usage')}}
    best = state.get('best')
    if result['valid']:
        candidate = {'run_id': result['run_id'], 'dev_psnr': result['dev_psnr'], 'recipe': active['recipe']}
        raw_best = state.get('raw_best')
        if raw_best is None or result['dev_psnr'] > raw_best['dev_psnr']:
            state['raw_best'] = candidate
        if best is None:
            trial['result_status'] = 'valid'
            decision = 'Baseline establishes the initial DEV score.'
        elif result['dev_psnr'] > best['dev_psnr'] + state['min_delta']:
            trial['result_status'] = 'improved'
            decision = (f'DEV PSNR increased from {best["dev_psnr"]:.8f} to {result["dev_psnr"]:.8f}, '
                        f'exceeding min_delta {state["min_delta"]:.8f} dB; keep this recipe.')
        else:
            trial['result_status'] = 'no_gain'
            decision = (f'DEV PSNR {result["dev_psnr"]:.8f} did not exceed effective best '
                        f'{best["dev_psnr"]:.8f} + min_delta {state["min_delta"]:.8f} dB; '
                        'retain the best recipe. Any higher raw score is still recorded in raw_best.')
        if best is None or trial['result_status'] == 'improved':
            state['best'] = candidate
            state['no_gain_count'] = 0
        else:
            state['no_gain_count'] += 1
    else:
        trial['result_status'] = 'invalid'
        decision = 'Invalid run cannot become best: ' + (result.get('error') or 'DEV validation failed')
        state['no_gain_count'] += 1
    state['consecutive_valid_no_gain'] = (state.get('consecutive_valid_no_gain', 0) + 1
        if trial['valid'] and trial['result_status'] == 'no_gain' else 0)
    trial['decision'] = decision
    state['trials'].append(trial)
    state['active'] = None
    state.pop('active_health', None)
    state['stop_reason'] = _stop_reason(state)
    state['status'] = 'stopped' if state['stop_reason'] else 'ready'
    _save(directory, state)
    _feedback(directory, state, trial['run_id'])
    return state


def _feedback(directory, state, run_id):
    # Scientific validity and selection have already been persisted. Diagnostics
    # may fail independently and must never rewrite the measured DEV result.
    try:
        feedback = build_run_feedback(str(directory), run_id)
    except Exception as exc:
        feedback = build_fallback_feedback(str(directory), run_id, str(exc))
    state['latest_feedback'] = feedback
    trial = next(t for t in state['trials'] if t['run_id'] == run_id)
    trial.update({key: feedback.get(key) for key in
                  ('feedback_ref', 'feedback_revision', 'diagnostics_status', 'proposal_ready')})
    _save(directory, state)
    return feedback


def _refresh_latest_feedback(directory, state):
    if not state.get('trials'):
        return
    trial = state['trials'][-1]
    run_id = trial['run_id']
    pointer = Path(directory) / 'feedback' / run_id / 'latest.json'
    if not pointer.is_file() and not state.get('latest_feedback'):
        _feedback(directory, state, run_id)
        return
    try:
        index = json.loads(pointer.read_text(encoding='utf-8'))
        feedback = _safe_feedback(directory, index['feedback_ref'])
        access = validate_feedback_access(feedback)
        if not access['available']:
            raise ValueError('; '.join(access['unavailable']))
        if (feedback.get('run_id', feedback.get('latest_run_id')) != run_id or
                feedback.get('latest_run_id') != run_id or
                index.get('feedback_revision', feedback.get('feedback_revision')) != feedback.get('feedback_revision') or
                type(feedback.get('proposal_ready')) is not bool):
            raise ValueError('Latest feedback pointer does not match the terminal run and revision')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Published newest evidence is inaccessible. Never use stale readiness or
        # stale observations to acknowledge it; an explicit rebuild can repair it.
        feedback = {'latest_run_id': run_id, 'diagnostics_status': 'unavailable',
                    'proposal_ready': not bool(state.get('control')), 'observations': [],
                    'unavailable': ['latest feedback access failed: ' + str(exc)]}
    if feedback != state.get('latest_feedback'):
        state['latest_feedback'] = feedback
        trial.update({key: feedback.get(key) for key in
                      ('feedback_ref', 'feedback_revision', 'diagnostics_status', 'proposal_ready')})
        _save(directory, state)


def campaign_feedback(campaign_dir, run_id=None):
    with _locked(campaign_dir) as directory:
        state = _collect(directory, _read(directory))
        if not state['trials']:
            raise ValueError('No terminal trial is available for feedback')
        run_id = run_id or state['trials'][-1]['run_id']
        if not any(t['run_id'] == run_id for t in state['trials']):
            raise ValueError('Feedback requires a recorded terminal trial')
        previous = state.get('latest_feedback')
        feedback = _feedback(directory, state, run_id)
        if run_id != state['trials'][-1]['run_id']:
            state['latest_feedback'] = previous
            _save(directory, state)
        return feedback


def _authorize(state, action):
    if state.get('control'):
        result = check_action(state['control'], state, action, None)
        if not result['allowed']:
            raise ValueError('Control disallows action: ' + result['reason'])
        return result


def _launch(directory, state, recipe, proposal, decision=None):
    cfg = {**state['config'], **recipe}
    cfg.pop('test', None)
    refs = resolve_trial_references(state, proposal, decision)
    refs['resolved_config'] = cfg
    base = next((t for t in state['trials'] if t['run_id'] == refs.get('construction_base_run_id')), None)
    base_config = (base or {}).get('resolved_config', state['config'])
    refs['actual_changed_fields'] = diff_scientific_config(cfg, base_config) if base else {}
    if decision:
        name = decision['decision_id']
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('decision_id must be a safe filename')
        snapshot = Path(directory) / 'decisions' / (name + '.json')
        if snapshot.exists() and json.loads(snapshot.read_text()) != decision:
            raise ValueError('A different decision already uses this decision_id')
        _persist_decision(directory, state, decision)
        refs['decision_ref'] = str(snapshot)
        refs['research_decision'] = decision
    observation = dict(state.get('control', {}).get('observation_config', {}))
    if state.get('profile_ref'):
        observation['profile_ref'] = state['profile_ref']
    all_limits = state.get('control', {}).get('limits')
    limits = ({key: all_limits[key] for key in ('allocated_gpus', 'job_walltime_seconds')}
              if all_limits else None)
    intent = {'index': len(state['trials']), 'recipe': recipe, 'proposal': proposal,
              'submitted_at': _now(), **refs,
              'request_id': f'launch_{len(state["trials"]):03d}_{uuid.uuid4().hex}',
              'observation_config': observation or None, 'execution_limits': limits}
    if limits:
        intent['reserved_gpu_hours'] = limits['allocated_gpus'] * limits['job_walltime_seconds'] / 3600
    state['launch_intent'] = intent
    state['status'] = 'running'
    _save(directory, state)
    return _resume_launch(directory, state)


def _current_start_allowed(directory, state, pending_role):
    pending = state[pending_role]
    if state.get('control'):
        # Recheck authority before any actual recovered start, excluding exactly
        # its own unstarted record/reservation. Other jobs and holds still count.
        transient = dict(state)
        transient.pop(pending_role, None)
        action = ('baseline' if pending['index'] == 0 else
                  {'action': 'propose', 'recipe': pending['proposal']['recipe']})
        gate = check_action(state['control'], transient, action, None)
        if not gate['allowed']:
            state['launch_block'] = {**gate, 'pending_role': pending_role}
            state['status'] = 'launch_blocked'
            _save(directory, state)
            return False
    state.pop('launch_block', None)
    state['status'] = 'running'
    return True


def _resume_launch(directory, state):
    intent = state['launch_intent']
    if not _current_start_allowed(directory, state, 'launch_intent'):
        return state
    run_id = prepare_run(intent['resolved_config'], observation_config=intent['observation_config'],
                         execution_limits=intent['execution_limits'], request_id=intent['request_id'])
    state['active'] = {key: value for key, value in intent.items()
                       if key not in ('observation_config', 'execution_limits')}
    state['active']['run_id'] = run_id
    state['launch_intent'] = None
    state['status'] = 'running'
    _save(directory, state)
    start_prepared_run(run_id, state['config']['runs_dir'])
    return state


def initialize_campaign(config, campaign_dir, max_trials=5, no_gain_limit=3, wait=True,
                        *, min_delta=DEFAULT_MIN_DELTA, control_ref=None, profile_ref=None):
    """Freeze data, initialization, seed, budgets and min_delta; launch baseline."""
    for name, value in (('max_trials', max_trials), ('no_gain_limit', no_gain_limit)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(min_delta) not in (int, float) or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError('min_delta must be a finite nonnegative number in dB')
    cfg = normalize_config(config)
    cfg = {**BUDGET_DEFAULTS, **RECIPE_DEFAULTS, **cfg}
    cfg['seed'] = 1 if cfg.get('seed') is None else cfg['seed']
    recipe = validate_recipe({key: cfg[key] for key in RECIPE_DEFAULTS})
    control = load_control(control_ref) if control_ref else None
    if profile_ref:
        profile_ref = str(Path(profile_ref).resolve())
        if not Path(profile_ref).is_file():
            raise ValueError('profile_ref must identify a readable frozen profile')
        json.loads(Path(profile_ref).read_text(encoding='utf-8'))
    with _locked(campaign_dir) as directory:
        if (directory / 'campaign.json').exists():
            raise ValueError('Campaign already exists; use status, wait or next to resume it')
        state = {'created_at': _now(), 'config': cfg, 'max_trials': max_trials,
                 'no_gain_limit': no_gain_limit, 'no_gain_count': 0, 'min_delta': float(min_delta), 'status': 'ready',
                 'stop_reason': None, 'active': None, 'best': None, 'raw_best': None, 'trials': [], 'frozen': None}
        state['campaign_id'] = control['campaign_id'] if control else directory.name
        state['protocol_id'] = 'P' + str(cfg['eval_size'])
        state['primary_metric'] = 'mean_per_image_rgb_psnr'
        state['consecutive_valid_no_gain'] = 0
        if control:
            state['control'] = control
            state['control_ref'] = str(directory / 'control.json')
            _authorize(state, 'baseline')
            _write_json(directory / 'control.json', control)
        if profile_ref:
            state['profile_ref'] = profile_ref
        _save(directory, state)
        _launch(directory, state, recipe, {'hypothesis': 'Establish the fixed-budget baseline.', 'based_on': None})
    return wait_campaign(campaign_dir) if wait else state


def campaign_status(campaign_dir):
    """Refresh terminal pending work; a host restart can resume the same trial."""
    with _locked(campaign_dir) as directory:
        return _collect(directory, _read(directory))


def wait_campaign(campaign_dir, timeout=None):
    state = campaign_status(campaign_dir)
    if state.get('active'):
        wait_for_result(state['active']['run_id'], state['config']['runs_dir'], timeout=timeout)
    return campaign_status(campaign_dir)


def _bounded_feedback(value):
    """Keep measured summaries and explicit preview counts; full tables stay in details."""
    if isinstance(value, list):
        return [_bounded_feedback(item) for item in value[:12]]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ('paired_rows', 'sample_ids', 'val_mean_psnr', 'validation_epochs', 'checkpoint_paths'):
                continue
            if key in ('groups', 'regions') and isinstance(item, dict):
                result[key] = {name: _bounded_feedback(group) for name, group in list(item.items())[:12]}
                result[key + '_count'] = len(item)
            else:
                result[key] = _bounded_feedback(item)
                if isinstance(item, list) and len(item) > 12:
                    result[key + '_count'] = len(item)
        return result
    return value


def _validate_sidecar(directory, state, decision, feedback):
    if not isinstance(decision, dict):
        raise ValueError('Research decision must be an object')
    refs = decision.get('observation_refs', [])
    if not isinstance(refs, list):
        raise ValueError('observation_refs must be a list')
    transient = dict(state)
    summaries = []
    for ref in refs:
        parts = ref.split('/') if isinstance(ref, str) else []
        if (len(parts) != 4 or parts[0] != state.get('campaign_id') or
                not re.fullmatch(r'r[0-9]+', parts[2])):
            raise ValueError('Invalid observation reference identity')
        summaries.append(_safe_feedback(directory, f'feedback/{parts[1]}/{parts[2]}/summary.json'))
    transient['_feedback_summaries'] = summaries
    return validate_research_decision(decision, transient, feedback or {})


def validate_campaign_decision(campaign_dir, decision):
    state = campaign_status(campaign_dir)
    return _validate_sidecar(campaign_dir, state, decision, state.get('latest_feedback'))


def next_proposal(campaign_dir):
    """Return a result-dependent task for the actual external proposer, never a grid."""
    state = campaign_status(campaign_dir)
    fixed = {key: value for key, value in state['config'].items()
             if key not in {*RECIPE_DEFAULTS, 'test'}}
    evidence = [{key: trial.get(key) for key in
                 ('run_id', 'recipe', 'status', 'result_status', 'valid', 'dev_psnr', 'error', 'decision', 'proposal')}
                for trial in state['trials']]
    feedback = state.get('latest_feedback')
    routing_state = dict(state)
    if state.get('launch_block'):
        routing_state.pop(state['launch_block'].get('pending_role', 'launch_intent'), None)
    route = route_next_action(routing_state, state.get('control'), feedback)
    allowed = state['status'] == 'ready' and not state.get('frozen')
    if state.get('control'):
        allowed = allowed and route['action'] == 'propose' and not route.get('review_required')
    if state.get('active_health') == 'unknown':
        allowed = False
    prompt = ('Read the baseline, effective best and recent DEV results below; consult history_index and feedback details for older evidence. Propose exactly one changed recipe. Explain the expected effect, '
              'and cite an observed prior run and its DEV score in based_on (rounding within 0.0001 dB is accepted). Use the latest failure or gain '
              'to choose the next change; do not submit an unevaluated fixed grid. You may change only loss_family '
              '(original|mse|l1), optimizer (adam|adamw), learning_rate, and weight_decay. '
              'Keep the model, data, initialization, seed, epochs, batch size, image sizes and validation frequency fixed. '
              'Selection uses independently reloaded mean per-image DEV PSNR. Never inspect TEST for a proposal. '
              'best is the retained effective candidate; raw_best records the highest observed score. '
              'An improvement must exceed best.dev_psnr + the frozen min_delta; smaller gains are no_gain. '
              'Do not change min_delta during the campaign. '
              'Write JSON with recipe, hypothesis, and based_on={run_id,dev_psnr,observation}, then call campaign submit. '
              'Wait for the actual run and call campaign next again. Stop when proposal_allowed is false.')
    prompt += (' Separate measured observations from untested causal explanations. Cite observation IDs, '
               'give an alternative explanation and a falsifier. based_on cites evidence; partial recipes '
               'inherit effective best, and every run uses the fixed campaign initialization. '
               'An explicit comparison does not change construction inheritance. In control mode write '
               'the independent decision sidecar acknowledging latest feedback, then submit --decision.')
    # Keep baseline, best and recent six results plus the complete historical index.
    keep = {t['run_id'] for t in state['trials'][-6:]}
    if state['trials']:
        keep.add(state['trials'][0]['run_id'])
    if state.get('best'):
        keep.add(state['best']['run_id'])
    evidence = [t for t in evidence if t['run_id'] in keep]
    bounded_feedback = ({key: feedback.get(key) for key in
        ('feedback_ref', 'feedback_revision', 'latest_run_id', 'diagnostics_status', 'proposal_ready',
         'actual_changed_fields', 'baseline_run_id', 'best_before_run_id', 'comparison_run_id',
         'construction_base_run_id', 'based_on_run_id', 'init_checkpoint_ref',
         'overall', 'comparisons', 'groups', 'observations', 'cases', 'curves', 'usage',
         'missing_required', 'section_status', 'interpretation_limits', 'unavailable', 'detail_refs')} if feedback else None)
    bounded_feedback = _bounded_feedback(bounded_feedback)
    budget = (check_action(state['control'], state, 'propose', None).get('remaining_budget')
              if state.get('control') else None)
    return {'proposal_allowed': allowed, 'status': state['status'], 'stop_reason': state['stop_reason'],
            'active': state.get('active'), 'launch_intent': state.get('launch_intent'),
            'launch_block': state.get('launch_block'), 'best': state['best'], 'raw_best': state['raw_best'],
            'min_delta': state['min_delta'], 'fixed': fixed,
            'remaining_trials': max(0, state['max_trials'] - len(state['trials']) - bool(state['active'])),
            'no_gain_count': state['no_gain_count'], 'no_gain_limit': state['no_gain_limit'],
            'history': evidence, 'history_index': [{'run_id': t['run_id'], 'result_status': t['result_status'],
             'feedback_ref': t.get('feedback_ref')} for t in state['trials']],
            'feedback': bounded_feedback, 'next_action': route, 'campaign_id': state.get('campaign_id'),
            'consecutive_valid_no_gain': state.get('consecutive_valid_no_gain', 0),
            'remaining_budget': budget, 'prompt': prompt}


def _validate_proposal(proposal, state):
    if not isinstance(proposal, dict) or set(proposal) != {'recipe', 'hypothesis', 'based_on'}:
        raise ValueError('Proposal must contain recipe, hypothesis and based_on')
    if not isinstance(proposal['hypothesis'], str) or not proposal['hypothesis'].strip():
        raise ValueError('A nonempty hypothesis is required')
    reference = proposal['based_on']
    if not isinstance(reference, dict) or set(reference) != {'run_id', 'dev_psnr', 'observation'}:
        raise ValueError('based_on must contain run_id, dev_psnr and observation')
    if not isinstance(reference['observation'], str) or not reference['observation'].strip():
        raise ValueError('based_on.observation must explain an observed result')
    cited = next((trial for trial in state['trials'] if trial['run_id'] == reference['run_id']), None)
    score = reference['dev_psnr']
    actual = cited.get('dev_psnr') if cited else None
    matches = (score is None and actual is None) or (type(score) in (int, float) and
        type(actual) in (int, float) and math.isclose(score, actual, rel_tol=0, abs_tol=0.0001))
    if cited is None or not matches:
        raise ValueError('based_on must cite a recorded run and its DEV PSNR within 0.0001 dB (null for invalid runs)')
    overrides = proposal['recipe']
    validate_recipe(overrides)
    recipe = validate_recipe({**state['best']['recipe'], **overrides})
    if any(recipe == trial['recipe'] for trial in state['trials']):
        raise ValueError('Recipe has already been evaluated; read its recorded result and propose a change')
    return recipe


def submit_proposal(campaign_dir, proposal, wait=True, *, decision_ref=None):
    with _locked(campaign_dir) as directory:
        state = _collect(directory, _read(directory))
        if state['status'] != 'ready' or state.get('frozen'):
            raise ValueError(f'Campaign cannot accept a proposal while {state["status"]}')
        recipe = _validate_proposal(proposal, state)
        _authorize(state, {'action': 'propose', 'recipe': proposal['recipe']})
        feedback = state.get('latest_feedback')
        if state.get('control'):
            route = route_next_action(state, state['control'], feedback)
            if route['action'] != 'propose' or route.get('review_required'):
                raise ValueError('Campaign requires ' + route['action'] + ': ' + route['reason'])
            if not decision_ref:
                raise ValueError('Control mode requires a decision sidecar')
        decision = None
        if decision_ref:
            decision = (json.loads(Path(decision_ref).read_text(encoding='utf-8'))
                        if not isinstance(decision_ref, dict) else dict(decision_ref))
            decision = _validate_sidecar(directory, state, decision, feedback)
            if decision.get('action') != 'propose' or decision.get('requested_change') != proposal['recipe']:
                raise ValueError('Decision requested_change must match submitted recipe patch and action propose')
        _launch(directory, state, recipe, proposal, decision)
    return wait_campaign(campaign_dir) if wait else state


def finalize_campaign(campaign_dir):
    """Freeze DEV-selected choice. This step does not access TEST."""
    with _locked(campaign_dir) as directory:
        state = _collect(directory, _read(directory))
        if state.get('active') or state.get('launch_intent'):
            raise ValueError('Wait for the active trial or pending launch before freezing the campaign')
        if not state.get('best'):
            raise ValueError('There is no valid DEV result to freeze')
        if not state.get('frozen'):
            state['frozen'] = {**state['best'], 'frozen_at': _now()}
            state['status'] = 'finalized'
            state['stop_reason'] = state['stop_reason'] or 'user_finalized'
            _save(directory, state)
        return state


def _record_final_test_usage(state, report):
    if report.get('usage_recorded'):
        return
    prior = dict(state.get('usage') or {})
    allocated = state['control']['limits']['allocated_gpus']
    values = [(trial.get('usage') or {}).get('gpu_hours') for trial in state['trials']]
    if allocated == 0:
        values = [0 if value is None else value for value in values]
    search = sum(values) if all(type(value) in (int, float) for value in values) else None
    base = prior.get('gpu_hours')
    if 'gpu_hours' not in prior:
        base = (search + prior.get('confirmation_gpu_hours', 0) + prior.get('calibration_gpu_hours', 0)
                if search is not None else None)
    actual = (report.get('usage') or {}).get('gpu_hours')
    if actual is None and allocated == 0:
        actual = 0
    prior.update(search_gpu_hours=search, final_test_gpu_hours=actual,
                 gpu_hours=base + actual if base is not None and actual is not None else None)
    state['usage'] = prior
    report['usage_recorded'] = True


def _finish_controlled_test(directory, state, report, result):
    from .control import evaluator_is_terminal
    if not evaluator_is_terminal(Path(directory) / 'final_test'):
        report.update(status='recovery_required', valid=False, usage=result.get('usage'),
                      error=result.get('error', 'Final TEST resource release is unconfirmed'))
        state['final_test'] = report
        _save(directory, state)
        return report
    report.update(status='completed' if result.get('valid') else 'failed',
                  valid=bool(result.get('valid')), usage=result.get('usage'), finished_at=_now())
    if result.get('valid'):
        report.update(test_psnr=result['score'], test_metrics=result['metrics'])
    else:
        report['error'] = result.get('error', 'Final evaluation failed')
    _record_final_test_usage(state, report)
    state['final_test'] = report
    _save(directory, state)
    return report


def _controlled_final_test(directory, state):
    """One bounded authorized held-out attempt; interruption never implies a retry."""
    if not state['control']['test_permission']:
        raise ValueError('TEST evaluation is not authorized')
    cfg = state['config']
    if not cfg.get('test'):
        raise ValueError('No test split configured')
    output = Path(directory) / 'final_test'
    persisted = output / 'evaluation_result.json'
    previous = state.get('final_test')
    if previous:
        if previous.get('status') in ('completed', 'failed'):
            return previous
        if persisted.is_file():
            result = json.loads(persisted.read_text(encoding='utf-8'))
            if result.get('valid'):
                validate_metrics(result['metrics'], expected_count(cfg['test']), cfg['eval_size'])
            return _finish_controlled_test(directory, state, previous, result)
        raise ValueError('Final TEST evaluation liveness is unknown; inspect its artifacts, do not start another attempt')
    _authorize(state, 'final_test')
    frozen = get_result(state['frozen']['run_id'], cfg['runs_dir'])
    if not frozen.get('valid') or not Path(frozen['artifacts']['best_checkpoint']).is_file():
        raise ValueError('Frozen checkpoint is missing or invalid')
    limits = {key: state['control']['limits'][key] for key in ('allocated_gpus', 'job_walltime_seconds')}
    report = {'run_id': state['frozen']['run_id'], 'status': 'running', 'valid': False,
              'started_at': _now(), 'log': str(output / 'evaluation.log'),
              'reserved_gpu_hours': limits['allocated_gpus'] * limits['job_walltime_seconds'] / 3600}
    state['final_test'] = report
    _save(directory, state)
    try:
        result = evaluate_frozen_checkpoint(cfg, frozen['artifacts'], cfg['test'], str(output),
                                            execution_limits=limits)
    except Exception as exc:
        result = (json.loads(persisted.read_text(encoding='utf-8')) if persisted.is_file() else
                  {'valid': False, 'error': str(exc), 'usage': {'gpu_hours': None}})
    return _finish_controlled_test(directory, state, report, result)


def final_test(campaign_dir):
    """Explicit held-out evaluation, only after finalize froze the chosen run."""
    with _locked(campaign_dir) as directory:
        state = _read(directory)
        if state['status'] != 'finalized' or not state.get('frozen'):
            raise ValueError('Finalize the DEV choice before running final-test')
        if state.get('control'):
            return _controlled_final_test(directory, state)
        cfg = state['config']
        if not cfg.get('test'):
            raise ValueError('No test split configured')
        result = get_result(state['frozen']['run_id'], cfg['runs_dir'])
        checkpoint = Path(result['artifacts']['best_checkpoint'])
        if not result['valid'] or not checkpoint.is_file():
            raise ValueError('Frozen checkpoint is missing or invalid')
        count = expected_count(cfg['test'])
        metrics_path = directory / 'final_test' / 'metrics.json'
        # Reuse only a completed, validated report, never an interrupted output directory.
        if state.get('final_test', {}).get('valid') and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            validate_metrics(metrics, count, cfg['eval_size'])
            return state['final_test']
        command = _resolve_evaluation(evaluation_command(cfg, directory, split='test'), result['train_metrics'])
        _write_json(directory / 'final_test_command.json', command)
        log_path = directory / 'final_test.log'
        report = {'run_id': state['frozen']['run_id'], 'status': 'running', 'valid': False,
                  'command': command, 'log': str(log_path), 'started_at': _now()}
        state['final_test'] = report
        _save(directory, state)
        rc = None
        try:
            rc = _execute(command, log_path, cfg['repo_dir'])
            if rc:
                raise RuntimeError(f'Final test.py exited with status {rc}')
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            score = validate_metrics(metrics, count, cfg['eval_size'])
            report.update(status='completed', valid=True, test_psnr=score, test_metrics=metrics)
        except Exception as exc:
            report.update(status='failed' if rc else 'completed', valid=False, error=str(exc))
        report['finished_at'] = _now()
        _save(directory, state)
        return report
