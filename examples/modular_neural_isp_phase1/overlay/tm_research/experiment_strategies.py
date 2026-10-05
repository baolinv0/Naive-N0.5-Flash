"""Bounded, auditable non-LLM controls sharing native proposal validation."""

import math
import random
import re

from .experiment_projection import aggregate_observation
from .experiment_space import RECIPE_DOMAINS, recipe_key, recipe_space, validate_space_recipe


class SamplingExhausted(ValueError):
    """Preserve every draw when the preregistered cap is reached."""
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


def _history(history):
    if not isinstance(history, list):
        raise ValueError('history must contain all terminal trials as a list')
    result = []
    seen = set()
    for entry in history:
        if not isinstance(entry, dict) or type(entry.get('valid')) is not bool:
            raise ValueError('Each historical trial requires explicit boolean valid')
        recipe = validate_space_recipe(entry.get('recipe'))
        key = recipe_key(recipe)
        if key in seen:
            raise ValueError('Historical recipe has already been evaluated more than once')
        seen.add(key)
        score = entry.get('dev_psnr')
        if entry['valid'] and (type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('Valid trial requires a finite actual DEV PSNR')
        result.append({'recipe': recipe, 'valid': entry['valid'],
                       'dev_psnr': score if entry['valid'] else None})
    return result, seen


def suggest_recipe_with_audit(strategy, history, seed, max_draws=10):
    """Return a recipe and all draws; duplicates never trigger a fallback."""
    if strategy not in ('random', 'tpe'):
        raise ValueError('Only random and tpe are algorithmic samplers')
    if type(seed) is not int or seed < 0:
        raise ValueError('Sampler seed must be a nonnegative integer')
    if type(max_draws) is not int or max_draws < 1:
        raise ValueError('max_draws must be a positive integer')
    observations, seen = _history(history)
    remaining = [recipe for recipe in recipe_space() if recipe_key(recipe) not in seen]
    if not remaining:
        raise SamplingExhausted('The complete recipe domain is exhausted', [])
    if strategy == 'random':
        recipe = random.Random(seed).choice(remaining)
        return {'recipe': recipe, 'draws': [{'recipe': recipe.copy(), 'key': recipe_key(recipe),
                                            'status': 'selected', 'draw': 1}]}

    import optuna
    from optuna.distributions import CategoricalDistribution
    from optuna.trial import TrialState, create_trial
    if optuna.__version__ != '4.5.0':
        raise ValueError('The preregistered TPE implementation requires Optuna 4.5.0')
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=3,
                                         multivariate=False, constant_liar=False)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    distributions = {key: CategoricalDistribution(values) for key, values in RECIPE_DOMAINS.items()}
    for entry in observations:
        frozen = create_trial(params=entry['recipe'], distributions=distributions,
            value=entry['dev_psnr'] if entry['valid'] else None,
            state=TrialState.COMPLETE if entry['valid'] else TrialState.FAIL)
        study.add_trial(frozen)
    attempts = []
    for index in range(max_draws):
        pending = study.ask()
        recipe = validate_space_recipe({name: pending.suggest_categorical(name, values)
                                        for name, values in RECIPE_DOMAINS.items()})
        key = recipe_key(recipe)
        duplicate = key in seen
        attempts.append({'draw': index + 1, 'recipe': recipe.copy(), 'key': key,
                         'status': 'duplicate' if duplicate else 'selected'})
        if not duplicate:
            return {'recipe': recipe, 'draws': attempts}
        # A rejected draw consumed inference/sampling work but is not assigned a
        # fictitious DEV score and does not replace the actual completed history.
        study.tell(pending, state=TrialState.FAIL)
    raise SamplingExhausted(f'TPE reached its bounded draw limit ({max_draws})', attempts)


def suggest_recipe(strategy, history, seed, max_draws=10):
    return suggest_recipe_with_audit(strategy, history, seed, max_draws)['recipe']


def sampler_payload(packet, recipe, strategy, decision_id):
    """Make honest control proposals citing existing aggregate/execution facts."""
    if strategy not in ('random', 'tpe'):
        raise ValueError('Sampler payload requires random or tpe provenance')
    recipe = validate_space_recipe(recipe)
    if not isinstance(packet, dict) or packet.get('proposal_allowed') is not True:
        raise ValueError('Current native packet must explicitly allow proposal')
    route = packet.get('next_action') or {}
    if route.get('action') != 'propose' or route.get('review_required'):
        raise ValueError('Native route must permit a fast proposal')
    if not isinstance(decision_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', decision_id):
        raise ValueError('decision_id must be a safe identifier')
    history = packet.get('history')
    if not isinstance(history, list) or not history:
        raise ValueError('Recorded terminal trial history is required')
    _, seen = _history(history)
    if recipe_key(recipe) in seen:
        raise ValueError('Recipe has already been evaluated')
    latest = history[-1]
    run_id = latest.get('run_id')
    feedback = packet.get('feedback') or {}
    if not run_id or feedback.get('latest_run_id', feedback.get('run_id')) != run_id:
        raise ValueError('Feedback latest identity must match the latest terminal trial')
    if feedback.get('run_id', run_id) != run_id:
        raise ValueError('Feedback run identity must match the latest terminal trial')
    campaign_id = packet.get('campaign_id')
    revision = feedback.get('feedback_revision')
    if (not isinstance(campaign_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', campaign_id)
            or not isinstance(revision, str) or not re.fullmatch(r'r[0-9]+', revision)
            or not isinstance(feedback.get('feedback_ref'), str) or not feedback['feedback_ref']):
        raise ValueError('Native feedback campaign/revision/reference identity is required')
    if feedback.get('campaign_id', campaign_id) != campaign_id:
        raise ValueError('Feedback campaign identity differs from the native packet')
    observation = aggregate_observation(feedback, execution_status=latest.get('status'))
    if latest['valid']:
        if observation['metric'] != 'mean_per_image_rgb_psnr' or not math.isclose(
                observation['value'], latest['dev_psnr'], rel_tol=0, abs_tol=.0001):
            raise ValueError('Native aggregate O1 must match the actual latest DEV score')
    elif observation['metric'] != 'execution_status':
        raise ValueError('Invalid latest trial requires native execution O1')
    if strategy == 'random':
        hypothesis = ('Random control: this complete recipe was sampled uniformly without replacement '
                      'from the remaining preregistered domain; no result-dependent gain is claimed.')
    else:
        hypothesis = ('TPE control: this complete recipe was selected by Optuna 4.5.0 TPE using all '
                      'recorded valid aggregate DEV scores and unscored FAIL trials. Improvement remains untested.')
    refs = [run_id]
    comparator = None
    best_run = (packet.get('best') or {}).get('run_id')
    best = next((entry for entry in history if entry.get('run_id') == best_run and entry.get('valid') is True), None)
    if best is not None:
        comparator = best_run
        if best_run not in refs:
            refs.append(best_run)
    proposal = {'recipe': recipe, 'hypothesis': hypothesis,
                'based_on': {'run_id': run_id, 'dev_psnr': latest.get('dev_psnr') if latest['valid'] else None,
                             'observation': observation['finding']}}
    decision = {'schema_version': 1, 'decision_id': decision_id, 'kind': 'fast_proposal',
        'feedback_ref': feedback['feedback_ref'], 'latest_seen_run_id': run_id,
        'reference_run_ids': refs, 'comparison_run_id': comparator,
        'observation_refs': [f'{campaign_id}/{run_id}/{revision}/O1'],
        'observation': observation['finding'],
        'prediction': 'The selected recipe will be measured under the fixed native DEV protocol; improvement is uncertain.',
        'falsifier': 'Its valid DEV PSNR does not exceed the retained best by the frozen min_delta, or the run is invalid.',
        'alternative_explanation': 'An observed difference may reflect training stochasticity or adaptive DEV selection.',
        'action': 'propose', 'requested_change': recipe.copy(), 'claim_scope': 'exploratory_dev',
        'sampler_provenance': {'strategy': strategy, 'observation_scope': 'aggregate_execution_only'}}
    return {'proposal': proposal, 'decision': decision}
