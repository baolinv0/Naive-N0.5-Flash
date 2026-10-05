"""Bounded outer orchestration; native campaign state remains authoritative.

Each call advances one campaign action. Uncertain HTTP requests are held rather
than repeated, and confirmation is behind an experiment-wide freeze barrier.
"""
import copy
import fcntl
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .campaign import (campaign_status, finalize_campaign, initialize_campaign,
                       next_proposal, submit_proposal, wait_campaign)
from .experiment_plan import read_json, verify_generated_index
from .experiment_projection import project_feedback
from .experiment_space import validate_space_recipe
from .experiment_strategies import sampler_payload, suggest_recipe_with_audit
from .runner import _write_json


@contextmanager
def _locked(root):
    root = Path(root).resolve()
    if not (root / 'experiment.json').is_file():
        raise ValueError('Generate the experiment before advancing it')
    with (root / '.outer.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield root


def _index(root):
    index = read_json(Path(root) / 'experiment.json')
    verify_generated_index(index)
    return index


def _outer(entry):
    path = Path(entry['directory']) / 'outer.json'
    return read_json(path) if path.exists() else {'attempts': 0, 'pending': None, 'rejections': [], 'auxiliary': {}}


def _save(entry, state):
    _write_json(Path(entry['directory']) / 'outer.json', state)


def _native(entry):
    path = Path(entry['campaign_dir']) / 'campaign.json'
    return read_json(path) if path.exists() else None


def _status(root, index):
    entries = []
    for entry in index['entries']:
        native, outer = _native(entry), _outer(entry)
        status = ('paused' if outer.get('pause_reason') else
                  'search_complete' if native and native.get('frozen') else
                  'failed' if outer.get('search_failed') else 'pending' if native is None else 'searching')
        confirmation = Path(entry['confirmation_dir']) / 'confirmation.json'
        confirm = read_json(confirmation) if confirmation.exists() else None
        entries.append({'strategy': entry['strategy'], 'block': entry['block'], 'status': status,
            'native_status': native.get('status') if native else None,
            'trial_count': len(native['trials']) if native else 0,
            'pause_reason': outer.get('pause_reason'),
            'confirmation_status': confirm.get('status') if confirm else None})
    complete = all(e['status'] in ('search_complete', 'failed') for e in entries)
    confirmed = complete and all(e['confirmation_status'] == 'completed' for e in entries)
    status = ('confirmation_complete' if confirmed else 'search_complete' if complete else
              'paused' if any(e['status'] == 'paused' for e in entries) else 'searching')
    result = {'experiment_id': index['experiment_id'], 'evidence_mode': index['evidence_mode'],
              'status': status, 'entries': entries,
              'claim_limit': 'Engineering execution evidence is not a scientific conclusion or ARIS W acceptance.'}
    ledger = Path(root) / 'model_costs.json'
    if ledger.exists():
        from .experiment_costs import CostLedger
        result['model_costs'] = CostLedger(ledger, index['manifest']['model_limits']).summary()
    return result


def experiment_status(root):
    """Read-only status; it never collects or launches native work."""
    root = Path(root).resolve()
    return _status(root, _index(root))


def _clock_allowed(root, index):
    path = Path(root) / 'runtime.json'
    state = read_json(path) if path.exists() else {'started_at_epoch': time.time()}
    if not path.exists():
        _write_json(path, state)
    return time.time() - state['started_at_epoch'] < index['manifest']['experiment_limits']['walltime_seconds']


def _client(root, index, entry, name, clients):
    if clients is not None and name in clients:
        if index['evidence_mode'] != 'engineering_fixture':
            raise ValueError('Injected clients require evidence_mode=engineering_fixture')
        return clients[name]
    from .experiment_costs import CostLedger
    from .experiment_provider import ChatClient
    config = dict(index['manifest']['models'][name])
    config['log_dir'] = str(Path(root) / 'model_requests' / name)
    campaign_id = read_json(entry['control_path'])['campaign_id']
    return ChatClient(config, CostLedger(Path(root) / 'model_costs.json', index['manifest']['model_limits'],
        scope_id=campaign_id, scope_limits=index['manifest']['campaign_model_limits']))


def _messages(packet, role):
    instructions = ('Return exactly a JSON object with proposal and decision. proposal has recipe '
        '(all four discrete fields), hypothesis and based_on={run_id,dev_psnr,observation}. '
        'decision is an independent native decision sidecar: schema_version=1, decision_id, '
        'kind=fast_proposal, feedback_ref, latest_seen_run_id, reference_run_ids, observation_refs '
        '(campaign/run/rN/O1), observation, prediction, falsifier, alternative_explanation, '
        'action=propose, requested_change equal to recipe, claim_scope. Cite actual latest '
        'feedback and real observations. No TEST or CONFIRM access. All non-recipe settings stay fixed. '
        'Discrete loss original|mse|l1, optimizer adam|adamw, learning_rate '
        '1e-5|3e-5|1e-4|3e-4|1e-3, weight_decay 0|1e-7|1e-6|1e-5|1e-4.')
    if role == 'slow_review':
        instructions = ('Return exactly {"decision": native sidecar}. kind=slow_review, '
            'decision_id as supplied, trigger_id=current next_action.trigger_id, all real feedback '
            'references and observation IDs, observation, prediction, falsifier, alternative_explanation, '
            'reference_run_ids and claim_scope. Choose propose, finalize, diagnose, wait or report_stop '
            'only as permitted by the current native review packet. For propose, requested_change is '
            'an allowed complete four-field recipe to approve continuation, otherwise {}. '
            'Never change protocol, model, scope, seed, budget or TEST permission. '
            'A protocol-suspect diagnosis must remain held for human resolution.')
    instructions += (' Required decision fields for BOTH roles: schema_version=1, '
        'decision_id exactly equal to user packet decision_id, kind, feedback_ref, latest_seen_run_id, '
        'reference_run_ids (nonempty real run IDs), observation_refs (real campaign/run/rN/O1 identities), '
        'observation, prediction, falsifier, alternative_explanation, action, requested_change, claim_scope. '
        'Use next_action.trigger_id as trigger_id for slow_review. Only these discrete recipe values: '
        'loss_family original|mse|l1; optimizer adam|adamw; learning_rate '
        '1e-5|3e-5|1e-4|3e-4|1e-3; weight_decay 0|1e-7|1e-6|1e-5|1e-4.')
    return [{'role': 'system', 'content': instructions},
            {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False, allow_nan=False)}]


def _identity_seen(native, decision_id):
    for trial in native.get('trials', []):
        if (trial.get('research_decision') or {}).get('decision_id') == decision_id:
            return True
    for field in ('active', 'launch_intent'):
        value = native.get(field) or {}
        if (value.get('research_decision') or {}).get('decision_id') == decision_id:
            return True
    return False


def _freeze(entry, packet):
    """Record an explicit mechanical closure, not an invented scientific review."""
    from .experiment_projection import aggregate_observation
    from .research import record_research_decision
    feedback = packet['feedback']
    observation = aggregate_observation(feedback)
    run_id = feedback['latest_run_id']
    decision = {'schema_version': 1, 'decision_id': 'outer_closure_' + run_id,
        'kind': 'closure', 'feedback_ref': feedback['feedback_ref'], 'latest_seen_run_id': run_id,
        'reference_run_ids': [run_id],
        'observation_refs': [f'{packet["campaign_id"]}/{run_id}/{feedback["feedback_revision"]}/O1'],
        'observation': observation['finding'], 'prediction': 'No additional search trial will be launched.',
        'falsifier': 'A later search launch would violate this frozen selection.',
        'alternative_explanation': 'A search stop does not establish algorithm superiority.',
        'action': 'finalize', 'requested_change': {},
        'claim_scope': 'Mechanical DEV selection closure only; independent confirmation remains required.'}
    record_research_decision(entry['campaign_dir'], decision)
    finalize_campaign(entry['campaign_dir'])


def _apply_pending(entry, outer, *, wait):
    from .research import record_research_decision
    pending = outer['pending']
    native = campaign_status(entry['campaign_dir'])
    decision_id = pending['request_id']
    if pending['phase'] == 'requesting':
        outer['pause_reason'] = 'Uncertain model request; inspect its persisted request/response, never replay automatically'
        _save(entry, outer)
        return
    payload = pending['payload']
    if _identity_seen(native, decision_id):
        outer['pending'] = None
        _save(entry, outer)
        return
    decision = payload.get('decision') if isinstance(payload, dict) else None
    try:
        if not isinstance(decision, dict) or decision.get('decision_id') != decision_id:
            raise ValueError('Model sidecar must acknowledge the supplied immutable decision_id')
        if pending['role'] == 'fast_proposal':
            if set(payload) != {'proposal', 'decision'}:
                raise ValueError('Fast response requires exactly proposal and decision')
            if not isinstance(payload['proposal'], dict) or set(payload['proposal']) != {'recipe', 'hypothesis', 'based_on'}:
                raise ValueError('Proposal must contain recipe, hypothesis and based_on')
            validate_space_recipe(payload['proposal']['recipe'])
            # An archived fast decision without launch evidence is an uncertain
            # native crash boundary, not permission to create another trial.
            archived = Path(entry['campaign_dir']) / 'decisions' / (decision_id + '.json')
            if archived.exists():
                outer['pause_reason'] = 'Archived proposal has no native launch receipt; inspect before recovery'
                _save(entry, outer)
                return
            submit_proposal(entry['campaign_dir'], payload['proposal'], wait=False, decision_ref=decision)
            if wait:
                wait_campaign(entry['campaign_dir'], timeout=30)
        else:
            if set(payload) != {'decision'} or decision.get('kind') != 'slow_review':
                raise ValueError('Slow response requires exactly a slow-review decision')
            if decision.get('action') == 'propose':
                validate_space_recipe(decision.get('requested_change'))
            record_research_decision(entry['campaign_dir'], decision)
            if decision.get('action') == 'propose':
                # Execute the exact experiment approved by Slow. A fresh Fast
                # request must not silently replace its requested recipe.
                feedback = next_proposal(entry['campaign_dir'])['feedback']
                latest = _native(entry)['trials'][-1]
                proposal = {'recipe': copy.deepcopy(decision['requested_change']),
                    'hypothesis': decision['prediction'], 'based_on': {
                        'run_id': feedback['latest_run_id'], 'dev_psnr': latest['dev_psnr'],
                        'observation': decision['observation']}}
                submit_proposal(entry['campaign_dir'], proposal, wait=False, decision_ref=decision)
                if wait:
                    wait_campaign(entry['campaign_dir'], timeout=30)
        outer['pending'] = None
        outer.pop('pause_reason', None)
    except (ValueError, TypeError, KeyError) as exc:
        native = _native(entry)
        if _identity_seen(native, decision_id):
            outer['pending'] = None
        elif (Path(entry['campaign_dir']) / 'decisions' / (decision_id + '.json')).exists():
            outer['pause_reason'] = 'Uncertain native submission: ' + str(exc)
        else:
            outer['rejections'].append({'request_id': decision_id, 'feedback_ref': pending['feedback_ref'], 'error': str(exc)})
            outer['pending'] = None
            count = sum(r['feedback_ref'] == pending['feedback_ref'] for r in outer['rejections'])
            if count >= 2:
                outer['pause_reason'] = 'Two rejected decisions for the same feedback; human diagnosis required'
    _save(entry, outer)


def _auxiliary(root, index, entry, outer, packet, clients):
    config = index['manifest']['models'].get('qwen') or {}
    if not config.get('enabled') or entry['strategy'] not in ('naive_rich', 'naive_fast_slow'):
        return False
    feedback = packet.get('feedback') or {}
    ref = feedback.get('feedback_ref')
    if not ref or ref in outer['auxiliary']:
        return False
    native = _native(entry)
    run_root = Path(native['config']['runs_dir']) / feedback['latest_run_id']
    image_dir = run_root / 'dev' / 'images'
    images = sorted(p for p in image_dir.iterdir() if p.suffix.casefold() in ('.png', '.jpg', '.jpeg'))[:3] if image_dir.is_dir() else []
    if not images:
        outer['auxiliary'][ref] = {'status': 'unavailable', 'reason': 'No actual DEV PNG outputs; no visual judgment fabricated'}
        _save(entry, outer)
        return True
    from .experiment_vision import build_vision_messages, validate_auxiliary_report
    request_id = f'{native["campaign_id"]}_vision_{outer["attempts"]:04d}'
    outer['attempts'] += 1
    # An unknown request stays marked, and is never automatically repeated.
    outer['auxiliary'][ref] = {'status': 'requesting', 'request_id': request_id}
    _save(entry, outer)
    try:
        messages = build_vision_messages(packet, [str(p) for p in images], run_root)
        provider = _client(root, index, entry, 'qwen', clients)
        response = provider.complete(messages, request_id=request_id,
            role='auxiliary_visual', max_output_tokens=index['manifest']['campaign_model_limits']['max_output_tokens_per_call'])
        if callable(getattr(provider, '_safe', None)):
            response = provider._safe(response)
        report = validate_auxiliary_report(response['payload'])
        outer['auxiliary'][ref] = {'status': 'complete', 'request_id': request_id,
            'model': response['model'], 'report': report,
            'limitations': ['Only predicted DEV images supplied; no GT comparison or numeric scoring authority'],
            'feedback_ref': ref, 'assets': json.loads(messages[1]['content'][0]['text'])['assets']}
    except (ValueError, OSError) as exc:
        outer['auxiliary'][ref] = {'status': 'unavailable', 'request_id': request_id, 'reason': str(exc)}
    _save(entry, outer)
    return True


def step_experiment(root, *, wait=False, clients=None):
    """Advance one authorized search action, preserving native hard gates."""
    with _locked(root) as root:
        index = _index(root)
        if not _clock_allowed(root, index):
            return {**_status(root, index), 'status': 'paused', 'pause_reason': 'Experiment walltime limit reached'}
        runtime = read_json(root / 'runtime.json')
        cursor = runtime.get('cursor', 0) % len(index['entries'])
        ordered = index['entries'][cursor:] + index['entries'][:cursor]
        # Rotate settled campaigns, but collect a recorded active worker first.
        entries = sorted(ordered, key=lambda e: not bool((_native(e) or {}).get('active') or (_native(e) or {}).get('launch_intent')))
        if any(_outer(e).get('pause_reason') for e in entries) and not any(
                (_native(e) or {}).get('active') or (_native(e) or {}).get('launch_intent') for e in entries):
            return _status(root, index)
        for entry in entries:
            outer, native = _outer(entry), _native(entry)
            if outer.get('pause_reason'):
                return _status(root, index)
            if outer.get('search_failed') or native and native.get('frozen'):
                continue
            if 'started_at_epoch' not in outer:
                outer['started_at_epoch'] = datetime.fromisoformat(native['created_at']).timestamp() if native else time.time()
            if time.time() - outer['started_at_epoch'] >= index['manifest']['campaign_walltime_seconds']:
                # campaign_status may START a persisted intent/prepared worker.
                # After the deadline only already-started native work is safe
                # to collect; no recovery may create a new resource charge.
                if native and native.get('active') and not native.get('launch_intent'):
                    from .runner import inspect_run
                    health = inspect_run(native['active']['run_id'], native['config']['runs_dir'])
                    if health['liveness'] in ('live', 'dead', 'terminal'):
                        campaign_status(entry['campaign_dir'])
                outer['pause_reason'] = 'Campaign walltime limit reached; preserve missing evidence without forced closure'
                _save(entry, outer)
                return _status(root, index)
            if native and (native.get('active') or native.get('launch_intent')):
                if wait:
                    wait_campaign(entry['campaign_dir'], timeout=30)
                else:
                    campaign_status(entry['campaign_dir'])
                return _status(root, index)
            _save(entry, outer)
            runtime['cursor'] = (index['entries'].index(entry) + 1) % len(index['entries'])
            _write_json(root / 'runtime.json', runtime)
            if native is None:
                initialize_campaign(read_json(entry['config_path']), entry['campaign_dir'],
                    max_trials=index['manifest']['max_trials'], no_gain_limit=entry['no_gain_limit'],
                    min_delta=index['manifest']['min_delta'], control_ref=entry['control_path'],
                    profile_ref=index.get('profile_ref'), wait=False)
                if wait:
                    wait_campaign(entry['campaign_dir'], timeout=30)
                return _status(root, index)
            if not native.get('trials') and not native.get('active') and not native.get('launch_intent'):
                outer['pause_reason'] = 'Baseline initialization has no native launch receipt; inspect before recovery'
                _save(entry, outer)
                return _status(root, index)
            if outer.get('pending'):
                _apply_pending(entry, outer, wait=wait)
                return _status(root, index)
            packet = next_proposal(entry['campaign_dir'])
            native = _native(entry)
            route = packet['next_action']
            if native.get('active') or native.get('launch_intent'):
                if wait:
                    wait_campaign(entry['campaign_dir'], timeout=30)
                return _status(root, index)
            action = route['action']
            terminal = (not route.get('review_required') or route['trigger_id'].endswith('/hard_stop'))
            if action in ('finalize', 'report_stop', 'confirm') and terminal and not native.get('research_hold'):
                if native.get('best'):
                    _freeze(entry, packet)
                else:
                    outer['search_failed'] = route['reason']
                    _save(entry, outer)
                return _status(root, index)
            role = 'fast_proposal' if action == 'propose' and packet['proposal_allowed'] else 'slow_review'
            if role == 'slow_review' and (not route.get('review_required') or entry['strategy'] != 'naive_fast_slow' or native.get('research_hold')):
                outer['pause_reason'] = 'Native route requires operational/human resolution: ' + route['reason']
                _save(entry, outer)
                return _status(root, index)
            if _auxiliary(root, index, entry, outer, packet, clients):
                return _status(root, index)
            outer['attempts'] += 1
            request_id = f'{native["campaign_id"]}_{role}_{outer["attempts"]:04d}'
            packet['decision_id'] = request_id
            if role == 'fast_proposal' and entry['strategy'] in ('random', 'tpe'):
                try:
                    audit = suggest_recipe_with_audit(entry['strategy'], native['trials'],
                        seed=entry['sampler_seed'] + len(native['trials']))
                except ValueError as exc:
                    outer['sampler_failure'] = {'request_id': request_id,
                        'draws': getattr(exc, 'attempts', []), 'error': str(exc)}
                    outer['pause_reason'] = 'Bounded sampler could not produce an admissible recipe: ' + str(exc)
                    _save(entry, outer)
                    return _status(root, index)
                # Samplers consume every terminal observation, not the bounded prompt history.
                packet['history'] = copy.deepcopy(native['trials'])
                payload = sampler_payload(packet, audit['recipe'], entry['strategy'], request_id)
                payload['decision']['sampler_provenance']['draws'] = audit['draws']
            else:
                projected = project_feedback(packet, entry['strategy'])
                projected['decision_id'] = request_id
                auxiliary = outer['auxiliary'].get(packet['feedback']['feedback_ref'])
                if auxiliary and entry['strategy'] != 'naive_scalar':
                    projected['auxiliary_visual'] = copy.deepcopy(auxiliary)
                if role == 'slow_review':
                    from .research import build_review_packet
                    projected['review_packet'] = build_review_packet(entry['campaign_dir'])
                if outer['rejections']:
                    projected['previous_rejection'] = outer['rejections'][-1]['error']
                outer['pending'] = {'phase': 'requesting', 'request_id': request_id, 'role': role,
                    'feedback_ref': packet['feedback']['feedback_ref']}
                _save(entry, outer)
                name = index['manifest']['slow_reviewer'] if role == 'slow_review' else 'naive'
                try:
                    provider = _client(root, index, entry, name, clients)
                    response = provider.complete(_messages(projected, role),
                        request_id=request_id, role=role,
                        max_output_tokens=index['manifest']['campaign_model_limits']['max_output_tokens_per_call'])
                    if callable(getattr(provider, '_safe', None)):
                        response = provider._safe(response)
                    payload = response['payload']
                    response_dir = Path(entry['directory']) / 'responses'
                    response_dir.mkdir(exist_ok=True)
                    _write_json(response_dir / (request_id + '.json'), response)
                except (ValueError, OSError) as exc:
                    outer['pause_reason'] = 'Model request failed or outcome uncertain; inspect evidence: ' + str(exc)
                    _save(entry, outer)
                    return _status(root, index)
            outer['pending'] = {'phase': 'prepared', 'request_id': request_id, 'role': role,
                'feedback_ref': packet['feedback']['feedback_ref'], 'payload': payload}
            _save(entry, outer)
            _apply_pending(entry, outer, wait=wait)
            return _status(root, index)
        return _status(root, index)


def confirm_experiment(root, *, wait=False):
    """Advance one predeclared paired task only after ALL searches are frozen."""
    from .confirmation import initialize_confirmation, run_confirmation_next
    with _locked(root) as root:
        index = _index(root)
        if not index['manifest'].get('confirmation'):
            raise ValueError('No predeclared confirmation plan')
        if not all((_native(e) or {}).get('frozen') for e in index['entries']):
            raise ValueError('All search selections must be frozen before any CONFIRM access')
        if not _clock_allowed(root, index):
            return {**_status(root, index), 'status': 'paused', 'pause_reason': 'Experiment walltime limit reached'}
        for entry in index['entries']:
            path = Path(entry['confirmation_dir']) / 'confirmation.json'
            if not path.exists():
                initialize_confirmation(entry['campaign_dir'], entry['confirmation_dir'],
                    read_json(entry['confirmation_plan_path']), control_ref=entry['control_path'])
            state = read_json(path)
            if state['status'] == 'completed':
                continue
            outer = _outer(entry)
            if time.time() - outer['started_at_epoch'] >= index['manifest']['campaign_walltime_seconds']:
                outer['pause_reason'] = 'Campaign walltime limit reached before confirmation action'
                _save(entry, outer)
                return _status(root, index)
            state = run_confirmation_next(entry['confirmation_dir'], wait=False)
            if wait and state.get('active_task_key'):
                from .runner import wait_for_result
                task = next(t for t in state['tasks'] if t['task_key'] == state['active_task_key'])
                if task.get('run_id') and task['status'] in ('prepared', 'running'):
                    wait_for_result(task['run_id'], state['config']['runs_dir'], timeout=30)
                    run_confirmation_next(entry['confirmation_dir'], wait=False)
            return _status(root, index)
        return _status(root, index)


def collect_analysis_records(root):
    """Reload native confirmation evidence; never trust cached outer scores."""
    from .confirmation import report_confirmation
    index = _index(Path(root).resolve())
    confirmation = index['manifest'].get('confirmation')
    if not confirmation:
        raise ValueError('Analysis requires predeclared confirmation seeds')
    records = []
    for entry in index['entries']:
        report = None
        if (Path(entry['confirmation_dir']) / 'confirmation.json').is_file():
            report = report_confirmation(entry['confirmation_dir'])
        eligible = bool(report and report['scope']['kind'] == 'independent'
            and report['confirmation_role'] == 'primary' and not report['authority_error']
            and not report['evidence_errors'] and not (report.get('scene_independence') or {}).get('conflicts')
            and index['evidence_mode'] == 'live_model_api' and entry['strategy_variant'] == 'primary')
        native = _native(entry) or {}
        feedback = native.get('latest_feedback') or {}
        eligible = eligible and not native.get('research_hold') and not any(feedback.get(key) for key in (
            'protocol_suspect', 'data_contamination', 'test_exposure'))
        records.append({'strategy': entry['strategy'], 'block': entry['block'],
            'expected_seeds': confirmation['seeds_by_block'][entry['block']],
            'pairs': [{'seed': p['seed'], 'status': 'valid' if eligible and p['status'] == 'complete' else 'inconclusive',
                       'delta_db': p['delta_db'] if eligible and p['status'] == 'complete' else None}
                      for p in report['pairs']] if report else [],
            'provenance': {'confirmation_role': report.get('confirmation_role') if report else None,
                'scope': report.get('scope') if report else None, 'strategy_variant': entry['strategy_variant'],
                'evidence_mode': index['evidence_mode'], 'eligible_for_primary_analysis': eligible,
                'descriptive_mean_delta_db': report.get('mean_delta_db') if report else None}})
    return records
