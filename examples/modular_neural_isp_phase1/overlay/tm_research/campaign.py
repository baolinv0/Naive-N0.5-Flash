"""Persistent serial controller; an external ARIS/Naive agent proposes each recipe."""

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path

from .runner import (BUDGET_DEFAULTS, RECIPE_DEFAULTS, _execute, _now, _resolve_evaluation,
                     _write_json, evaluation_command, expected_count, get_result,
                     normalize_config, run_baseline, validate_metrics, validate_recipe,
                     wait_for_result)


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
    return json.loads((Path(directory) / 'campaign.json').read_text(encoding='utf-8'))


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
    active = state.get('active')
    if not active:
        return state
    result = get_result(active['run_id'], state['config']['runs_dir'])
    if result['status'] not in ('completed', 'failed'):
        return state
    trial = {**active, **{key: result.get(key) for key in
        ('status', 'valid', 'result_status', 'dev_psnr', 'dev_metrics', 'artifacts', 'logs', 'error', 'reload_succeeded')}}
    best = state.get('best')
    if result['valid']:
        if best is None:
            trial['result_status'] = 'valid'
            decision = 'Baseline establishes the initial DEV score.'
        elif result['dev_psnr'] > best['dev_psnr']:
            trial['result_status'] = 'improved'
            decision = f'DEV PSNR increased from {best["dev_psnr"]:.8f} to {result["dev_psnr"]:.8f}; keep this recipe.'
        else:
            trial['result_status'] = 'no_gain'
            decision = f'DEV PSNR {result["dev_psnr"]:.8f} did not exceed best {best["dev_psnr"]:.8f}; retain the best recipe.'
        if best is None or trial['result_status'] == 'improved':
            state['best'] = {'run_id': result['run_id'], 'dev_psnr': result['dev_psnr'], 'recipe': active['recipe']}
            state['no_gain_count'] = 0
        else:
            state['no_gain_count'] += 1
    else:
        trial['result_status'] = 'invalid'
        decision = 'Invalid run cannot become best: ' + (result.get('error') or 'DEV validation failed')
        state['no_gain_count'] += 1
    trial['decision'] = decision
    state['trials'].append(trial)
    state['active'] = None
    state['stop_reason'] = _stop_reason(state)
    state['status'] = 'stopped' if state['stop_reason'] else 'ready'
    _save(directory, state)
    return state


def _launch(directory, state, recipe, proposal):
    cfg = {**state['config'], **recipe}
    cfg.pop('test', None)
    run_id = run_baseline(cfg)
    state['active'] = {'index': len(state['trials']), 'run_id': run_id, 'recipe': recipe,
                       'proposal': proposal, 'submitted_at': _now()}
    state['status'] = 'running'
    _save(directory, state)
    return state


def initialize_campaign(config, campaign_dir, max_trials=5, no_gain_limit=3, wait=True):
    """Freeze data, initialization, seed, budgets; launch baseline before any proposal."""
    for name, value in (('max_trials', max_trials), ('no_gain_limit', no_gain_limit)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    cfg = normalize_config(config)
    cfg = {**BUDGET_DEFAULTS, **RECIPE_DEFAULTS, **cfg}
    cfg['seed'] = 1 if cfg.get('seed') is None else cfg['seed']
    recipe = validate_recipe({key: cfg[key] for key in RECIPE_DEFAULTS})
    with _locked(campaign_dir) as directory:
        if (directory / 'campaign.json').exists():
            raise ValueError('Campaign already exists; use status, wait or next to resume it')
        state = {'created_at': _now(), 'config': cfg, 'max_trials': max_trials,
                 'no_gain_limit': no_gain_limit, 'no_gain_count': 0, 'status': 'ready',
                 'stop_reason': None, 'active': None, 'best': None, 'trials': [], 'frozen': None}
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


def next_proposal(campaign_dir):
    """Return a result-dependent task for the actual external proposer, never a grid."""
    state = campaign_status(campaign_dir)
    fixed = {key: value for key, value in state['config'].items()
             if key not in {*RECIPE_DEFAULTS, 'test'}}
    evidence = [{key: trial.get(key) for key in
                 ('run_id', 'recipe', 'status', 'result_status', 'valid', 'dev_psnr', 'error', 'decision', 'proposal')}
                for trial in state['trials']]
    allowed = state['status'] == 'ready' and not state.get('frozen')
    prompt = ('Read every prior DEV result below and propose exactly one changed recipe. Explain the expected effect, '
              'and cite an observed prior run and its DEV score in based_on (rounding within 0.0001 dB is accepted). Use the latest failure or gain '
              'to choose the next change; do not submit an unevaluated fixed grid. You may change only loss_family '
              '(original|mse|l1), optimizer (adam|adamw), learning_rate, and weight_decay. '
              'Keep the model, data, initialization, seed, epochs, batch size, image sizes and validation frequency fixed. '
              'Selection uses independently reloaded mean per-image DEV PSNR. Never inspect TEST for a proposal. '
              'Write JSON with recipe, hypothesis, and based_on={run_id,dev_psnr,observation}, then call campaign submit. '
              'Wait for the actual run and call campaign next again. Stop when proposal_allowed is false.')
    return {'proposal_allowed': allowed, 'status': state['status'], 'stop_reason': state['stop_reason'],
            'active': state.get('active'), 'best': state['best'], 'fixed': fixed,
            'remaining_trials': max(0, state['max_trials'] - len(state['trials']) - bool(state['active'])),
            'no_gain_count': state['no_gain_count'], 'no_gain_limit': state['no_gain_limit'],
            'history': evidence, 'prompt': prompt}


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


def submit_proposal(campaign_dir, proposal, wait=True):
    with _locked(campaign_dir) as directory:
        state = _collect(directory, _read(directory))
        if state['status'] != 'ready' or state.get('frozen'):
            raise ValueError(f'Campaign cannot accept a proposal while {state["status"]}')
        recipe = _validate_proposal(proposal, state)
        _launch(directory, state, recipe, proposal)
    return wait_campaign(campaign_dir) if wait else state


def finalize_campaign(campaign_dir):
    """Freeze DEV-selected choice. This step does not access TEST."""
    with _locked(campaign_dir) as directory:
        state = _collect(directory, _read(directory))
        if state.get('active'):
            raise ValueError('Wait for the active trial before freezing the campaign')
        if not state.get('best'):
            raise ValueError('There is no valid DEV result to freeze')
        if not state.get('frozen'):
            state['frozen'] = {**state['best'], 'frozen_at': _now()}
            state['status'] = 'finalized'
            state['stop_reason'] = state['stop_reason'] or 'user_finalized'
            _save(directory, state)
        return state


def final_test(campaign_dir):
    """Explicit held-out evaluation, only after finalize froze the chosen run."""
    with _locked(campaign_dir) as directory:
        state = _read(directory)
        if state['status'] != 'finalized' or not state.get('frozen'):
            raise ValueError('Finalize the DEV choice before running final-test')
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
