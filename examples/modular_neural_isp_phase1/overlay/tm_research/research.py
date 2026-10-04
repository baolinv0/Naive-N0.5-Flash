"""Research decisions, event routing and evidence-bounded reporting."""

import copy
import json
import math
import re
from pathlib import Path

from .control import check_action, reconcile_usage


ACTIONS = {'wait', 'diagnose', 'propose', 'finalize', 'confirm', 'request_scope_change', 'report_stop'}


def _campaign_id(state, control=None):
    return state.get('campaign_id') or (control or state.get('control') or {}).get('campaign_id') or 'legacy'


def _valid_no_gain(state):
    count = 0
    for trial in reversed(state.get('trials', [])):
        if trial.get('valid') is True and trial.get('result_status') == 'no_gain':
            count += 1
        else:
            break
    return count


def route_next_action(state, control=None, feedback=None):
    """Pure routing. Persist trigger_id only when a review is actually recorded."""
    feedback = feedback or {}
    last = state.get('trials', [])[-1] if state.get('trials') else {}
    run_id = last.get('run_id', 'initial')
    refs = [feedback['feedback_ref']] if feedback.get('feedback_ref') else []
    handled = set(state.get('handled_slow_triggers', []))
    approved = {d.get('trigger_id') for d in state.get('research_decisions', [])
                if isinstance(d, dict) and d.get('kind') == 'slow_review' and d.get('action') == 'propose'}
    def route(action, reason, event, review=False):
        trigger = f'{_campaign_id(state, control)}/{run_id}/{event}'
        return {'action': action, 'trigger_id': trigger, 'reason': reason,
                'evidence_refs': refs, 'review_required': bool(review and trigger not in handled),
                'consecutive_valid_no_gain': _valid_no_gain(state)}
    if state.get('active') or state.get('launch_intent'):
        unknown = state.get('liveness') == 'unknown' or state.get('active_health') == 'unknown'
        return route('diagnose' if unknown else 'wait',
                     'Existing run must be resolved before new work', 'active_unknown' if unknown else 'active')
    if not control and not state.get('pending_slow_decision'):
        return route('report_stop' if state.get('stop_reason') or state.get('frozen') else 'propose',
                     'Legacy campaign status; no new authorization inferred', 'legacy_status')
    # Freeze and hard stops cannot be undone by missing diagnostics or a review.
    if state.get('frozen') and control:
        gate = check_action(control, state, 'confirm')
        return route('confirm' if gate['allowed'] else 'report_stop',
                     'Frozen candidate: ' + gate['reason'], 'frozen', True)
    if state.get('stop_reason') or state.get('status') in ('stopped', 'finalized'):
        return route('report_stop', 'Search hard stop: ' + str(state.get('stop_reason') or state['status']), 'hard_stop', True)
    if state.get('research_hold') or state.get('liveness') == 'unknown' or state.get('active_health') == 'unknown':
        return route('diagnose', 'Resolve recorded protocol hold or unknown job identity', 'research_hold')
    pending = state.get('pending_slow_decision')
    if pending:
        # Expose the current event identity, even if collection advanced the last
        # terminal run since the diagnosis. A continuation must acknowledge it.
        transient = dict(state)
        transient.pop('pending_slow_decision', None)
        current = route_next_action(transient, control, feedback)
        return {**current, 'action': pending['action'], 'review_required': False,
                'reason': 'Pending slow decision ' + pending['decision_id'] + ': ' + pending['action']}
    if not control:
        return route('propose', 'Legacy campaign status; no new authorization inferred', 'legacy_status')
    if feedback.get('proposal_ready') is False:
        return route('diagnose', 'Rebuild required evidence without retraining', 'required_diagnostics')
    if feedback.get('protocol_suspect') or feedback.get('data_contamination') or feedback.get('test_exposure'):
        return route('diagnose', 'Protocol/data evidence requires diagnosis', 'protocol_diagnosis')
    gate = check_action(control, state, 'propose')
    if not gate['allowed']:
        return route('diagnose' if gate['remaining_budget'] is None else 'report_stop', gate['reason'], 'resource_boundary')
    conflict = feedback.get('evidence_conflict')
    if conflict and isinstance(conflict, dict) and conflict.get('evidence_refs'):
        event = route('diagnose', 'Recorded evidence conflict requires competing explanations', 'evidence_conflict', True)
        if event['trigger_id'] in approved:
            return route('propose', 'Explicit scoped slow review permits another experiment', 'evidence_conflict')
        return event
    threshold = control.get('slow_policy', {}).get('consecutive_valid_no_gain', 2)
    if _valid_no_gain(state) >= threshold:
        event = route('diagnose', 'Consecutive valid no_gain requires scoped slow review', 'valid_plateau', True)
        if event['trigger_id'] in approved:
            return route('propose', 'Explicit scoped slow review permits another experiment', 'valid_plateau')
        return event
    return route('propose', 'Search remains within frozen bounds', 'continue')


def _nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty')


def _observation_catalog(state, feedback):
    summaries = [feedback] + list(state.get('_feedback_summaries', []))
    catalog = set()
    campaign_id = _campaign_id(state)
    for summary in summaries:
        run = summary.get('run_id') or summary.get('latest_run_id')
        revision = summary.get('feedback_revision')
        for observation in summary.get('observations', []):
            identifier = observation.get('id') or observation.get('observation_id')
            if run and revision and identifier:
                catalog.add(f'{campaign_id}/{run}/{revision}/{identifier}')
    return catalog


def validate_research_decision(decision, state, feedback):
    """Validate references and a falsifiable sidecar; does not grade explanations.

    The integrator additionally compares requested_change with proposal.recipe.
    A filesystem-backed caller may provide _feedback_summaries for historical
    observation citations; otherwise only the supplied revision is resolvable.
    """
    if not isinstance(decision, dict):
        raise ValueError('Research decision must be an object')
    required = {'schema_version', 'decision_id', 'kind', 'feedback_ref', 'latest_seen_run_id',
                'reference_run_ids', 'observation_refs', 'observation', 'prediction', 'falsifier',
                'alternative_explanation', 'action', 'requested_change', 'claim_scope'}
    missing = required - decision.keys()
    if missing:
        raise ValueError('Missing decision fields: ' + ', '.join(sorted(missing)))
    if type(decision['schema_version']) is not int or decision['schema_version'] != 1:
        raise ValueError('Decision schema_version must be 1')
    if not isinstance(decision['decision_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', decision['decision_id']):
        raise ValueError('decision_id must be a safe readable identifier')
    if decision['kind'] not in ('fast_proposal', 'slow_review', 'closure', 'resolution'):
        raise ValueError('Unknown research decision kind')
    if decision['action'] not in ACTIONS:
        raise ValueError('Unknown research action')
    for key in ('observation', 'prediction', 'falsifier', 'alternative_explanation', 'claim_scope', 'feedback_ref'):
        _nonempty(decision[key], key)
    trials = state.get('trials', [])
    if not trials or decision['latest_seen_run_id'] != trials[-1]['run_id']:
        raise ValueError('latest_seen_run_id must match latest terminal trial')
    feedback_run = feedback.get('run_id') or feedback.get('latest_run_id')
    if feedback_run != trials[-1]['run_id']:
        raise ValueError('Feedback must describe latest terminal trial')
    expected_ref = feedback.get('feedback_ref')
    if expected_ref and decision['feedback_ref'] != expected_ref:
        # Absolute and relative representations of the same revision may coexist.
        expected_tail = f'feedback/{feedback_run}/{feedback.get("feedback_revision")}/summary.json'
        if not (str(expected_ref).endswith(expected_tail) and decision['feedback_ref'] == expected_tail):
            raise ValueError('feedback_ref does not identify supplied feedback revision')
    known = {t['run_id']: t for t in trials}
    references = decision['reference_run_ids']
    if not isinstance(references, list) or not references or any(not isinstance(r, str) or r not in known for r in references):
        raise ValueError('reference_run_ids must cite existing campaign trials')
    comparator = decision.get('comparison_run_id')
    if comparator is not None:
        trial = known.get(comparator)
        if not trial or trial.get('valid') is not True:
            raise ValueError('comparison_run_id requires an existing valid trial')
        protocol = state.get('control', {}).get('protocol_id', state.get('protocol_id'))
        if trial.get('protocol_id', protocol) != protocol:
            raise ValueError('comparison_run_id protocol differs from campaign')
        actual_protocol = (trial.get('dev_metrics') or {}).get('protocol')
        if actual_protocol is not None and protocol is not None and actual_protocol != protocol:
            raise ValueError('comparison_run_id actual evaluation protocol differs from campaign')
        if trial.get('resolved_config') is not None:
            from .evidence import diff_scientific_config
            campaign_config = {key: value for key, value in state.get('config', {}).items() if key != 'test'}
            delta = diff_scientific_config(campaign_config, trial['resolved_config'])
            # Recipe fields are permitted; every other scientific difference matters.
            changed = {key: value for key, value in delta.items()
                       if key not in ('loss_family', 'optimizer', 'learning_rate', 'weight_decay', 'scheduler.eta_min')}
            if changed:
                raise ValueError('comparison_run_id has incompatible fixed scientific conditions')
    refs = decision['observation_refs']
    catalog = _observation_catalog(state, feedback)
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in catalog for ref in refs):
        raise ValueError('observation_refs must identify real campaign/run/revision/observation evidence')
    from .evidence import validate_feedback_access
    summaries = [feedback] + list(state.get('_feedback_summaries', []))
    for ref in refs:
        _, run_id, revision, observation_id = ref.split('/')
        summary = next(item for item in summaries if (item.get('run_id') or item.get('latest_run_id')) == run_id
                       and item.get('feedback_revision') == revision)
        access = validate_feedback_access(summary, observation_id)
        if not access['available']:
            raise ValueError('Observation source/artifact unavailable: ' + '; '.join(access['unavailable']))
    change = decision['requested_change']
    if not isinstance(change, dict):
        raise ValueError('requested_change must be a recipe object')
    allowed = set(state.get('control', {}).get('allowed_recipe_fields', ['loss_family', 'optimizer', 'learning_rate', 'weight_decay']))
    if not set(change) <= allowed:
        raise ValueError('requested_change exceeds authorized recipe fields')
    if change:
        from .runner import validate_recipe
        validate_recipe(change)
    if decision['action'] == 'propose' and not change:
        raise ValueError('A proposal requires requested_change')
    return copy.deepcopy(decision)


def _load_state(directory):
    return json.loads((Path(directory) / 'campaign.json').read_text(encoding='utf-8'))


def _safe_feedback(directory, ref):
    directory = Path(directory).resolve()
    path = Path(ref)
    path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(directory / 'feedback') or not path.is_file():
        raise ValueError('feedback_ref must resolve to readable campaign feedback')
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['feedback_ref'] = str(path)
    return summary


def _current_feedback(directory, state):
    if not state.get('trials'):
        return {}
    run_id = state['trials'][-1]['run_id']
    latest = Path(directory) / 'feedback' / run_id / 'latest.json'
    if latest.is_file():
        index = json.loads(latest.read_text(encoding='utf-8'))
        if index.get('feedback_ref'):
            return _safe_feedback(directory, index['feedback_ref'])
        revision = index.get('feedback_revision')
        if revision:
            return _safe_feedback(directory, f'feedback/{run_id}/{revision}/summary.json')
    return state.get('latest_feedback') or {}


def _decisions(directory):
    path = Path(directory) / 'research_decisions.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def _review_start_state(state):
    """Exclude only a blocked, verified unstarted request for its review gate."""
    transient = dict(state)
    transient.pop('pending_slow_decision', None)
    blocked = state.get('launch_block') or {}
    role = blocked.get('pending_role')
    if role in ('active', 'launch_intent'):
        transient.pop(role, None)
    elif state.get('launch_intent') and not state.get('active'):
        transient.pop('launch_intent', None)
    return transient


def _persist_decision(directory, state, decision):
    """Caller owns campaign lock and saves state; share persistence with submit."""
    from .runner import _write_json
    directory = Path(directory)
    old = next((d for d in _decisions(directory) if d.get('decision_id') == decision['decision_id']), None)
    if old is not None and old != decision:
        raise ValueError('decision_id already exists with a different immutable decision')
    already_applied = any(d.get('decision_id') == decision['decision_id'] and d.get('kind') == decision['kind']
                          for d in state.get('research_decisions', []))
    pending = state.get('pending_slow_decision') or {}
    if (decision.get('kind') == 'slow_review' and not already_applied
            and pending.get('action') in ('report_stop', 'request_scope_change', 'finalize')
            and decision['action'] != pending['action']):
        raise ValueError('A terminal slow decision cannot be overwritten; use explicit closure or a new authorized campaign')
    if decision.get('kind') == 'slow_review' and decision['action'] == 'propose' and not already_applied:
        transient = _review_start_state(state)
        queued = next((state[role] for role in ('launch_intent', 'active')
                       if state.get(role) and not transient.get(role)), None)
        if pending and queued and decision['requested_change'] != (queued.get('proposal') or {}).get('recipe'):
            raise ValueError('Slow continuation must approve the immutable queued proposal.recipe patch')
        feedback = _current_feedback(directory, state)
        current = route_next_action(transient, state.get('control'), feedback)
        if not decision.get('trigger_id') or decision['trigger_id'] != current['trigger_id']:
            raise ValueError('slow_review propose trigger_id must match the current trigger')
        if current['action'] not in ('propose', 'diagnose'):
            raise ValueError('Current research action does not permit a slow continuation')
        if state.get('control'):
            gate = check_action(state['control'], transient, {'action': 'propose', 'recipe': decision['requested_change']})
            if not gate['allowed']:
                raise ValueError('Control disallows slow continuation: ' + gate['reason'])
        if (state.get('research_hold') or feedback.get('proposal_ready') is False
                or feedback.get('protocol_suspect') or feedback.get('data_contamination') or feedback.get('test_exposure')):
            raise ValueError('Required evidence or protocol diagnosis still blocks continuation')
    snapshot = directory / 'decisions' / (decision['decision_id'] + '.json')
    if snapshot.exists() and json.loads(snapshot.read_text(encoding='utf-8')) != decision:
        raise ValueError('Immutable decision snapshot already exists with different content')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    _write_json(snapshot, decision)
    if old is None:
        with (directory / 'research_decisions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + '\n')
    recorded = next((d for d in state.setdefault('research_decisions', [])
                     if d.get('decision_id') == decision['decision_id']), None)
    if recorded is None:
        state['research_decisions'].append({'decision_id': decision['decision_id'], 'action': decision['action'],
                                           'kind': decision['kind'], 'trigger_id': decision.get('trigger_id'), 'decision_ref': str(snapshot)})
    else:
        recorded.setdefault('kind', decision['kind'])
    if decision.get('kind') == 'slow_review' and not already_applied:
        if decision['action'] == 'propose':
            state['pending_slow_decision'] = None
            if decision['trigger_id'] not in state.setdefault('handled_slow_triggers', []):
                state['handled_slow_triggers'].append(decision['trigger_id'])
        elif (state.get('pending_slow_decision') or {}).get('action') not in ('report_stop', 'request_scope_change', 'finalize'):
            trigger = decision.get('trigger_id') or route_next_action(
                _review_start_state(state), state.get('control'), _current_feedback(directory, state))['trigger_id']
            state['pending_slow_decision'] = {'decision_id': decision['decision_id'],
                                            'action': decision['action'], 'trigger_id': trigger,
                                            'decision_ref': str(snapshot)}
    return copy.deepcopy(decision)


def record_research_decision(campaign_dir, decision):
    """Standalone operation using the same campaign lock; never nest this call.

    Identical decision IDs are idempotent. Protocol holds require a genuinely new
    observation plus explicit unchanged-protocol operational resolution to clear.
    """
    from .campaign import _locked, _save
    with _locked(campaign_dir) as directory:
        state = _load_state(directory)
        old = next((d for d in _decisions(directory) if d.get('decision_id') == decision.get('decision_id')), None)
        if old:
            if old != decision:
                raise ValueError('decision_id already exists with a different immutable decision')
            metadata = state.get('research_decisions', [])
            index = next((i for i, d in enumerate(metadata) if d.get('decision_id') == old['decision_id']), None)
            if index is not None:
                if metadata[index].get('kind') or old.get('kind') != 'slow_review' or old['action'] == 'propose':
                    return old
                # Explicit replay can apply an archived pre-upgrade stop or the
                # latest diagnosis. Never let an old proposal clear newer intent
                # or restore a diagnosis superseded by subsequent decisions.
                if old['action'] in ('diagnose', 'wait') and index != len(metadata) - 1:
                    return old
                _persist_decision(directory, state, old)
                _save(directory, state)
                return old
            # The immutable log may have committed before an interrupted state
            # write. Reapply through the same validation and control path below.
        feedback = _current_feedback(directory, state)
        transient = dict(state)
        # Resolve only explicitly cited historical revisions; never rename observations.
        transient['_feedback_summaries'] = []
        for ref in decision.get('observation_refs', []):
            parts = ref.split('/') if isinstance(ref, str) else []
            if len(parts) != 4 or parts[0] != _campaign_id(state) or not re.fullmatch(r'r\d+', parts[2]):
                raise ValueError('Invalid observation reference identity')
            transient['_feedback_summaries'].append(_safe_feedback(directory, f'feedback/{parts[1]}/{parts[2]}/summary.json'))
        validated = validate_research_decision(decision, transient, feedback)
        hold = state.get('research_hold')
        resolves = validated.get('resolves_decision_id')
        if resolves:
            if not hold or resolves != hold['decision_id']:
                raise ValueError('resolves_decision_id must match active research_hold')
            resolution = validated.get('resolution', {})
            if resolution.get('operational_issue') is not True or resolution.get('protocol_unchanged') is not True:
                raise ValueError('Only an unchanged-protocol operational issue may clear a hold; use a new campaign for protocol changes')
            if not set(validated['observation_refs']) - set(hold.get('evidence_refs', [])):
                raise ValueError('Hold resolution requires new diagnostic evidence')
            state['research_hold'] = None
        if validated.get('protocol_suspect') is True:
            state['research_hold'] = {'decision_id': validated['decision_id'],
                                      'reason': validated.get('reason') or validated['observation'],
                                      'evidence_refs': validated['observation_refs']}
        _persist_decision(directory, state, validated)
        _save(directory, state)
        return validated


def build_review_packet(campaign_dir, trigger=None):
    directory = Path(campaign_dir)
    state = _load_state(directory)
    decisions = _decisions(directory)
    feedback = _current_feedback(directory, state)
    return {'campaign_id': _campaign_id(state),
            'trigger': trigger or route_next_action(state, state.get('control'), feedback),
            'control': state.get('control'), 'feedback': feedback,
            'history': state.get('trials', []), 'best': state.get('best'),
            'frozen': state.get('frozen'), 'research_hold': state.get('research_hold'),
            'pending_slow_decision': state.get('pending_slow_decision'),
            'alternative_explanations': list(dict.fromkeys(d['alternative_explanation'] for d in decisions if d.get('alternative_explanation'))),
            'decisions': decisions,
            'evidence_refs': [str(directory / 'campaign.json')] + ([feedback['feedback_ref']] if feedback.get('feedback_ref') else []),
            'review_questions': ['What facts are measured?', 'Which competing explanations remain?',
                                 'What allowed experiment distinguishes two explanations?',
                                 'What existing evidence can change the decision cheaply?',
                                 'Does the next action fit scope and budget?', 'What uncertainty and stop conditions remain?']}


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Artifact must contain an object: ' + str(path))
    return value


def _confirmation(directory):
    """Load only a registered parent receipt, never an adjacent guessed report."""
    directory = Path(directory).resolve()
    parent = _load_state(directory)
    receipts = parent.get('confirmation_refs', [])
    if not receipts:
        return {}
    if not isinstance(receipts, list):
        return {'_association_error': 'Confirmation receipts must be a list'}
    receipt = receipts[-1]  # Registration order, not a choice by observed score.
    try:
        destination = Path(receipt['confirmation_dir']).resolve()
        manifest = _read_json(destination / 'confirmation.json')
        if manifest.get('association') != receipt or Path(receipt['parent_campaign']).resolve() != directory or manifest.get('parent_campaign') != str(directory):
            raise ValueError('Confirmation receipt/manifest belongs to a different parent')
        trials = parent.get('trials', [])
        baseline = trials[0] if trials else {}
        winner = next((t for t in trials if t.get('run_id') == parent.get('frozen', {}).get('run_id')), {})
        if not baseline.get('valid') or not winner.get('valid') or receipt.get('baseline_run_id') != baseline.get('run_id') or receipt.get('frozen_run_id') != winner.get('run_id'):
            raise ValueError('Receipt does not identify the frozen baseline/winner')
        for name, trial in (('baseline', baseline), ('winner', winner)):
            if manifest.get('plan', {}).get('arms', {}).get(name) != {'run_id': trial['run_id'], 'recipe': trial['recipe']}:
                raise ValueError('Frozen confirmation arm differs from parent: ' + name)
        control = manifest.get('control', {})
        for key in ('campaign_id', 'source_revision', 'approved_config_ref', 'protocol_id'):
            if receipt.get(key) != control.get(key) or (parent.get('control') and parent['control'].get(key) != receipt.get(key)):
                raise ValueError('Frozen confirmation scientific identity differs: ' + key)
        identity = manifest.get('parent_identity', {})
        if identity.get('config') != parent.get('config') or identity.get('frozen') != parent.get('frozen'):
            raise ValueError('Parent scientific config/frozen candidate changed')
        if control.get('protocol_id') != 'P' + str(manifest.get('config', {}).get('eval_size')):
            raise ValueError('Confirmation control protocol differs from native evaluator')
        manifest['_confirmation_dir'] = str(destination)
        manifest['_receipt'] = receipt
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'_association_error': str(exc), '_receipt': receipt}


def _actual_confirmation_score(manifest, task):
    from .runner import normalize_config, validate_metrics
    from .evidence import diff_scientific_config
    root = Path(manifest['_confirmation_dir']) / 'runs' / task['run_id']
    if Path(task.get('run_dir', '')).resolve() != root.resolve():
        raise ValueError('Task run directory does not match recorded run identity')
    result, config = _read_json(root / 'state.json'), _read_json(root / 'config.json')
    expected = normalize_config({**manifest['config'], **manifest['plan']['arms'][task['arm']]['recipe'], 'seed': task['seed']})
    if diff_scientific_config(config, expected):
        raise ValueError('Actual confirmation scientific config differs from frozen task')
    if result.get('run_id') != task['run_id'] or result.get('status') != 'completed' or result.get('valid') is not True:
        raise ValueError('Task has no actual valid completed result')
    metrics = _read_json(root / 'dev/metrics.json')
    expected_count = manifest['config']['validation']['expected_count']
    if result.get('expected_count') != expected_count:
        raise ValueError('Actual native sample count differs from frozen original DEV inventory')
    score = validate_metrics(metrics, expected_count, config['eval_size'])
    if result.get('dev_metrics') != metrics or not math.isclose(score, result.get('dev_psnr', float('nan')), abs_tol=1e-10, rel_tol=0):
        raise ValueError('Actual native DEV result/metrics disagree')
    if manifest['plan']['scope']['kind'] == 'independent':
        evaluation_root = root / 'confirmation_eval'
        evaluation = _read_json(evaluation_root / 'evaluation_result.json')
        evaluation_state = _read_json(evaluation_root / 'evaluation_state.json')
        frozen = evaluation_state.get('frozen', {})
        split = manifest['evaluation_split_spec']
        actual_checkpoint = frozen.get('checkpoint', {})
        if (frozen.get('split') != split or {key: actual_checkpoint.get(key) for key in ('best_checkpoint', 'config_dir')} != {key: result['artifacts'][key] for key in ('best_checkpoint', 'config_dir')}
                or actual_checkpoint.get('selection_split', 'original_dev') != 'original_dev'):
            raise ValueError('Independent evaluation does not link frozen checkpoint/split')
        if evaluation.get('valid') is not True:
            raise ValueError('Independent evaluation invalid')
        metrics = _read_json(evaluation_root / 'metrics.json')
        score = validate_metrics(metrics, split['expected_count'], config['eval_size'])
        if evaluation.get('metrics') != metrics or not math.isclose(evaluation.get('score', float('nan')), score, abs_tol=1e-10, rel_tol=0):
            raise ValueError('Independent actual metrics/result disagree')
    if type(task.get('score')) not in (int, float) or not math.isclose(task['score'], score, abs_tol=1e-10, rel_tol=0):
        raise ValueError('Recorded task score differs from actual result')
    return score


def _q_claim(state, manifest):
    q = {'status': 'exploratory_dev' if state.get('best') else 'unavailable',
         'reason': 'DEV selection is exploratory; confirmation is limited to its frozen protocol'}
    if not manifest:
        return q
    q.update(status='inconclusive', statistical_significance=False)
    if manifest.get('_association_error'):
        q['reason'] = manifest['_association_error']
        return q
    plan = manifest.get('plan', {})
    q.update(scope=plan.get('scope'), adaptive_dev_bias=plan.get('scope', {}).get('kind') == 'original_dev',
             confirmation_ref=manifest['_confirmation_dir'], association=manifest['_receipt'],
             scene_independence=copy.deepcopy(manifest.get('scene_independence')))
    try:
        seeds, replicates = plan['seeds'], plan['replicates']
        if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds) or type(replicates) is not int or replicates < 1:
            raise ValueError('Invalid frozen seed/replicate inventory')
        expected = {f'{arm}:{seed}:{replicate}' for seed in seeds for replicate in range(1, replicates + 1) for arm in ('baseline', 'winner')}
        order = plan['execution_order']
        tasks = manifest['tasks']
        by_key = {task['task_key']: task for task in tasks}
        if set(order) != expected or len(order) != len(expected) or set(by_key) != expected or len(tasks) != len(expected):
            raise ValueError('Task/pair inventory differs from the frozen complete plan')
        completed_ids = [task.get('run_id') for task in tasks if task.get('status') == 'completed']
        if len(set(completed_ids)) != len(completed_ids):
            raise ValueError('One actual run cannot count as multiple confirmation tasks')
        for key, task in by_key.items():
            if key != f'{task["arm"]}:{task["seed"]}:{task["replicate"]}':
                raise ValueError('Task identity differs from frozen task key')
        pairs = []
        for seed in seeds:
            for replicate in range(1, replicates + 1):
                pair = {'seed': seed, 'replicate': replicate, 'status': 'incomplete', 'delta_db': None}
                baseline, winner = (by_key[f'{arm}:{seed}:{replicate}'] for arm in ('baseline', 'winner'))
                if baseline.get('status') == winner.get('status') == 'completed':
                    scores = [_actual_confirmation_score(manifest, task) for task in (baseline, winner)]
                    pair.update(status='complete', delta_db=scores[1]-scores[0], baseline_run_id=baseline['run_id'], winner_run_id=winner['run_id'])
                pairs.append(pair)
        q.update(confirmation_pairs=pairs, planned_pairs=len(seeds)*replicates, complete_pairs=sum(p['status'] == 'complete' for p in pairs))
        threshold = plan.get('delta_useful_db')
        if all(p['status'] == 'complete' for p in pairs) and type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0:
            values = [p['delta_db'] for p in pairs]
            mean = sum(values)/len(values)
            q.update(mean_delta_db=mean, delta_useful_db=threshold)
            q['status'] = 'supported_in_scope' if all(v > 0 for v in values) and mean > threshold else 'not_supported' if mean <= 0 else 'inconclusive'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        q['reason'] = 'Confirmation evidence incomplete or incompatible: ' + str(exc)
    if (manifest.get('scene_independence') or {}).get('conflicts'):
        q.update(status='inconclusive', reason=manifest['scene_independence']['claim_limit'])
    return q


def _w_claim(state, directory):
    """Recognized linked live events; declarations and arbitrary files stay pending."""
    w = {'status': 'pending', 'reason': 'Requires actual ordered baseline and three feedback-consuming live proposal events'}
    provenance = state.get('workflow_evidence', {})
    if not isinstance(provenance, dict) or provenance.get('kind') != 'live_naive_aris' or directory is None:
        return w
    directory = Path(directory).resolve()
    def path(ref):
        value = Path(ref)
        return value if value.is_absolute() else directory / value
    try:
        from .workflow import validate_workflow_evidence
        validate_workflow_evidence(state, directory, provenance)
        service = _read_json(path(provenance['service_config_ref']))
        for field in ('model_id', 'weights_ref', 'executor_url', 'aris_version', 'startup_record_ref'):
            _nonempty(service.get(field), 'live service ' + field)
        if (service.get('schema_version') != 1 or service.get('kind') != 'naive_adapter_live'
                or service.get('model_id') != provenance.get('model_identity') or not service.get('weights_ref')
                or not service.get('executor_url') or not service.get('aris_version') or not isinstance(service.get('inference_settings'), dict)):
            raise ValueError('Unrecognized live model/service provenance')
        startup = _read_json(path(service['startup_record_ref']))
        if startup.get('event') != 'service_started' or startup.get('model_id') != service['model_id'] or startup.get('executor_url') != service['executor_url'] or startup.get('weights_ref') != service['weights_ref']:
            raise ValueError('Service/model identity lacks matching startup record')
        events = [json.loads(line) for line in path(provenance['transcript_ref']).read_text().splitlines() if line.strip()]
        if not all(isinstance(event, dict) for event in events):
            raise ValueError('Live transcript must contain structured event objects')
        trials = state.get('trials', [])
        ids = provenance.get('run_ids')
        if not isinstance(ids, list) or len(ids) != 4 or ids != [t['run_id'] for t in trials[:4]] or len(set(ids)) != 4 or not all(t.get('valid') is True for t in trials[:4]):
            raise ValueError('W requires baseline followed by three actual valid trials')
        from .runner import validate_metrics
        from .evidence import validate_feedback_access
        for trial in trials[:4]:
            root = Path(state['config']['runs_dir']) / trial['run_id']
            actual = _read_json(root / 'state.json')
            metrics = _read_json(root / 'dev/metrics.json')
            config = _read_json(root / 'config.json')
            score = validate_metrics(metrics, actual['expected_count'], config['eval_size'])
            if actual.get('valid') is not True or actual.get('status') != 'completed' or actual.get('run_id') != trial['run_id'] or not math.isclose(score, trial['dev_psnr'], abs_tol=1e-10, rel_tol=0):
                raise ValueError('Transcript run lacks matching actual valid evidence')
        cursor = -1
        for index, trial in enumerate(trials[:4]):
            if index:
                decision_ref = trial.get('decision_ref') or trial.get('research_decision_ref')
                decision = _read_json(path(decision_ref))
                previous = trials[index-1]['run_id']
                if decision.get('kind') != 'fast_proposal' or decision.get('latest_seen_run_id') != previous or decision.get('requested_change') != trial.get('proposal', {}).get('recipe'):
                    raise ValueError('Proposal did not acknowledge preceding terminal feedback')
                summary = _safe_feedback(directory, decision['feedback_ref'])
                access = validate_feedback_access(summary)
                if not access['available']:
                    raise ValueError('Consumed feedback source unavailable')
                expected_events = [('feedback_consumed', previous), ('proposal_submitted', trial['run_id'])]
            else:
                decision = {}
                expected_events = []
            expected_events += [('run_completed', trial['run_id'])]
            for event_name, run in expected_events:
                found = next((i for i in range(cursor+1, len(events)) if events[i].get('event') == event_name and events[i].get('run_id') == run), None)
                if found is None:
                    raise ValueError('Missing ordered live event: ' + event_name)
                event = events[found]
                if event.get('schema_version') != 1 or event.get('campaign_id') != _campaign_id(state) or event.get('model_id') != service['model_id'] or event.get('executor_url') != service['executor_url']:
                    raise ValueError('Live event service/campaign identity differs')
                if event_name != 'run_completed' and (event.get('decision_id') != decision['decision_id'] or event.get('feedback_ref') != decision['feedback_ref'] or event.get('observation_refs') != decision['observation_refs']):
                    raise ValueError('Live feedback/proposal event does not link the recorded decision')
                cursor = found
        w.update(status='accepted', reason='Linked ordered live records establish workflow acceptance only.',
                 model_identity=service['model_id'], evidence_refs=[provenance['transcript_ref'], provenance['service_config_ref']])
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        w['reason'] = 'Live workflow provenance pending: ' + str(exc)
    return w


def _claim_levels(state, confirmation, directory=None):
    return {'W': _w_claim(state, directory), 'Q': _q_claim(state, confirmation),
            'R': {'status': 'unavailable', 'reason': 'Requires a prespecified equal-budget strategy comparator'},
            'P': {'status': 'unavailable', 'reason': 'Requires agreed quality/device/user conditions and independent coverage'}}


def build_campaign_report(campaign_dir):
    """Write a factual report without inferring live, visual, product or cost evidence."""
    directory = Path(campaign_dir)
    state = _load_state(directory)
    trials = state.get('trials', [])
    ledger = reconcile_usage(state.get('control'), state)
    gpu_hours = ledger['gpu_hours'] if ledger['known'] else 'unavailable'
    report = {'campaign_id': _campaign_id(state), 'claims': _claim_levels(state, _confirmation(directory), directory),
              'best': state.get('best'), 'frozen': state.get('frozen'), 'stop_reason': state.get('stop_reason'),
              'pending_slow_decision': copy.deepcopy(state.get('pending_slow_decision')),
              'final_checkpoint_ref': copy.deepcopy(state.get('final_checkpoint_ref') or {'status': 'unavailable'}),
              'trials': [{'run_id': t.get('run_id'), 'valid': t.get('valid'), 'result_status': t.get('result_status'),
                          'dev_psnr': t.get('dev_psnr'), 'error': t.get('error'), 'decision': t.get('decision')}
                         for t in trials],
              'cost': {'gpu_hours': gpu_hours, 'scope': 'search+confirmation+calibration+final_test',
                       'ledger': ledger, 'reserved_gpu_hours': ledger['reserved_gpu_hours'],
                       'inference_cost': 'unavailable', 'inference_tokens': 'unavailable'},
              'human_interventions': state.get('human_interventions', 'unavailable'),
              'valid_experiment_rate': sum(t.get('valid') is True for t in trials) / len(trials) if trials else None,
              'time_to_effective_candidate': 'unavailable', 'visual_claims': 'unavailable',
              'alternative_explanations': [d['alternative_explanation'] for d in _decisions(directory) if d.get('alternative_explanation')],
              'limitations': ['Single search seed and adaptive DEV selection do not establish generalization.',
                             'PNG paths are not evidence that the text-only proposer inspected images.',
                             'Unknown inference cost is unavailable, not zero.',
                             'Group or ROI measurements do not identify internal module causality.']}
    lines = [f'# Campaign {report["campaign_id"]}', '', '## Evidence levels', '']
    for level, claim in report['claims'].items():
        lines.append(f'- {level}: {claim["status"]}. {claim["reason"]}')
    lines += ['', '## Observed trials', '', '| Run | Valid | Result | DEV PSNR |', '|---|---|---|---|']
    lines += [f'| {t["run_id"]} | {t["valid"]} | {t["result_status"]} | {t["dev_psnr"]} |' for t in report['trials']]
    lines += ['', '## Cost and uncertainty', '', f'GPU-hours: {gpu_hours}; inference cost: unavailable.', '']
    lines += ['- ' + item for item in report['limitations']]
    lines += ['', '## Unresolved alternatives', ''] + ['- ' + item for item in report['alternative_explanations']]
    path = directory / 'report.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report['report_ref'] = str(path.resolve())
    return report


def export_research_memory(campaign_dir):
    """Append scoped experiences. Contradictions/supersessions never erase history."""
    from .campaign import _locked
    with _locked(campaign_dir) as directory:
        state = _load_state(directory)
        confirmation = _confirmation(directory)
        control = state.get('control') or confirmation.get('control', {})
        path = directory / 'research_memory.jsonl'
        existing = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
        keys = {(r['campaign_id'], r['decision_id']) for r in existing}
        with path.open('a', encoding='utf-8') as stream:
            for d in _decisions(directory):
                key = (_campaign_id(state), d['decision_id'])
                if key in keys:
                    continue
                status = d.get('memory_status', 'observed')
                if status not in ('observed', 'tentative', 'confirmed_in_scope', 'contradicted', 'superseded'):
                    raise ValueError('Invalid memory_status')
                evidence = d.get('confirmation_evidence', {})
                actual_scope = (confirmation.get('plan', {}).get('scope', {})
                                if evidence.get('confirmation_dir') == confirmation.get('_confirmation_dir') else {})
                scope_kind = actual_scope.get('kind')
                permitted_claims = {'confirmed_in_scope'}
                if scope_kind == 'original_dev':
                    permitted_claims.add('original_dev_seed_stability')
                elif scope_kind == 'independent':
                    permitted_claims.add('independent_confirmation')
                scope_matches = scope_kind in ('original_dev', 'independent') and d['claim_scope'] in permitted_claims
                scope_limits = []
                if scope_kind == 'original_dev':
                    scope_limits.append('Original DEV seed stability retains adaptive DEV selection bias; it does not establish independent confirmation.')
                elif scope_kind == 'independent':
                    scope_limits.append('Confirmation applies only to the frozen authorized split and paired-seed protocol; no broader generalization or statistical significance is established.')
                scene_evidence = confirmation.get('scene_independence') if scope_kind == 'independent' else None
                if (scene_evidence or {}).get('conflicts'):
                    scope_limits.append(scene_evidence['claim_limit'])
                if status == 'confirmed_in_scope':
                    # A declaration cannot promote an unconfirmed exploratory result.
                    q = _claim_levels(state, confirmation, directory)['Q']
                    association = q.get('association', {})
                    matches = (evidence.get('confirmation_dir') == q.get('confirmation_ref')
                               and evidence.get('winner_run_id') == association.get('frozen_run_id')
                               and evidence.get('baseline_run_id') == association.get('baseline_run_id')
                               and evidence.get('protocol_id') == association.get('protocol_id')
                               and d['kind'] in ('closure', 'slow_review')
                               and d.get('comparison_run_id') == association.get('baseline_run_id')
                               and d['requested_change'] == confirmation.get('plan', {}).get('arms', {}).get('winner', {}).get('recipe')
                               and scope_matches)
                    status = 'confirmed_in_scope' if q['status'] == 'supported_in_scope' and matches else 'tentative'
                    if not scope_matches:
                        scope_limits.append('Declared claim_scope does not match a linked frozen confirmation scope; the declaration is retained as tentative.')
                record = {'campaign_id': key[0], 'decision_id': key[1], 'status': status,
                          'final_checkpoint_ref': copy.deepcopy(state.get('final_checkpoint_ref') or {'status': 'unavailable'}),
                          'scope': {'protocol_id': control.get('protocol_id'),
                                    'source_revision': control.get('source_revision'),
                                    'model': state.get('config', {}).get('model'),
                                    'train': state.get('config', {}).get('train'),
                                    'dev': state.get('config', {}).get('validation'),
                                    'claim_scope': d['claim_scope'],
                                    'confirmation_kind': scope_kind,
                                    'confirmation_scope': copy.deepcopy(actual_scope),
                                    'confirmation_split': copy.deepcopy(confirmation.get('evaluation_split_spec')
                                                                       if scope_kind == 'independent' else confirmation.get('config', {}).get('validation')
                                                                       if scope_kind == 'original_dev' else None),
                                    'adaptive_dev_bias': scope_kind == 'original_dev' if scope_kind else None,
                                    'interpretation_limits': scope_limits,
                                    'scene_independence': copy.deepcopy(scene_evidence)},
                          'observation': d['observation'], 'attempt': d['requested_change'],
                          'prediction': d['prediction'], 'falsifier': d['falsifier'],
                          'alternative_explanation': d['alternative_explanation'],
                          'evidence_refs': d['observation_refs'], 'tags': d.get('tags', []),
                          'confirmation_evidence': d.get('confirmation_evidence'),
                          'contradicts_decision_id': d.get('contradicts_decision_id'),
                          'supersedes_decision_id': d.get('supersedes_decision_id')}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                existing.append(record)
                keys.add(key)
        return existing
