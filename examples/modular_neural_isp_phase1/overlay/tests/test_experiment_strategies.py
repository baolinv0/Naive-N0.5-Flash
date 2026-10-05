"""Research strategy boundaries: real scores, no replacement, and no leakage."""

import copy
import json

import pytest

from tm_research.experiment_space import recipe_key, recipe_space, validate_space_recipe
from tm_research.experiment_projection import project_feedback
from tm_research.experiment_strategies import (
    SamplingExhausted, sampler_payload, suggest_recipe, suggest_recipe_with_audit,
)


BASELINE = {'loss_family': 'original', 'optimizer': 'adam',
            'learning_rate': 1e-4, 'weight_decay': 1e-7}


def trial(recipe=BASELINE, *, run_id='exp_001', score=25, valid=True):
    return {'run_id': run_id, 'recipe': copy.deepcopy(recipe), 'valid': valid,
            'dev_psnr': score, 'status': 'completed' if valid else 'failed',
            'result_status': 'baseline' if valid else 'invalid'}


def packet():
    return {'campaign_id': 'fixture', 'proposal_allowed': True, 'status': 'ready',
            'stop_reason': None, 'remaining_trials': 3, 'min_delta': .01,
            'history': [trial()], 'history_index': [{'run_id': 'exp_001',
                'feedback_ref': '/PRIVATE_DIAGNOSTIC/older-summary.json'}],
            'best': trial(), 'raw_best': trial(),
            'fixed': {'seed': 17, 'epochs': 1, 'eval_size': 512,
                      'validation': {'input_dir': '/PRIVATE_DIAGNOSTIC/data'}},
            'remaining_budget': {'search_gpu_hours': 4, 'ledger': {
                'known': True, 'gpu_hours': 1, 'reason': 'PRIVATE_DIAGNOSTIC'}},
            'next_action': {'action': 'propose', 'review_required': False,
                'trigger_id': 'fixture/exp_001/continue', 'reason': 'PRIVATE_DIAGNOSTIC',
                'evidence_refs': ['feedback/exp_001/r001/summary.json']},
            'prompt': 'PRIVATE_DIAGNOSTIC inspect cases and curves',
            'feedback': {'campaign_id': 'fixture', 'run_id': 'exp_001',
                'latest_run_id': 'exp_001', 'feedback_revision': 'r001',
                'feedback_ref': 'feedback/exp_001/r001/summary.json',
                'proposal_ready': True, 'overall': {'dev_psnr': 25,
                    'num_images': 2, 'primary_metric': 'mean_per_image_rgb_psnr',
                    'paired': {'secret': 'PRIVATE_DIAGNOSTIC'}},
                'observations': [{'id': 'O1', 'observation_id': 'O1',
                    'metric': 'mean_per_image_rgb_psnr', 'value': 25, 'n_images': 2,
                    'source': 'summary.json#/overall',
                    'finding': 'PRIVATE_DIAGNOSTIC',
                    'interpretation_limits': ['PRIVATE_DIAGNOSTIC']},
                    {'id': 'O2', 'finding': 'PRIVATE_DIAGNOSTIC'}],
                'cases': {'cases': ['PRIVATE_DIAGNOSTIC']},
                'groups': {'secret': 'PRIVATE_DIAGNOSTIC'},
                'detail_refs': {'curves': '/PRIVATE_DIAGNOSTIC/curves.json'}}}


def test_complete_space_has_exactly_150_unique_recipes_and_baseline():
    recipes = recipe_space()
    assert len(recipes) == len({recipe_key(value) for value in recipes}) == 150
    assert BASELINE in recipes
    assert {r['learning_rate'] for r in recipes} == {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
    assert {r['weight_decay'] for r in recipes} == {0, 1e-7, 1e-6, 1e-5, 1e-4}
    recipes[0]['optimizer'] = 'mutated'
    assert recipe_space()[0]['optimizer'] == 'adam'


@pytest.mark.parametrize('bad', [
    {k: v for k, v in BASELINE.items() if k != 'optimizer'},
    {**BASELINE, 'seed': 1}, {**BASELINE, 'learning_rate': True},
    {**BASELINE, 'learning_rate': float('nan')},
    {**BASELINE, 'weight_decay': float('inf')},
    {**BASELINE, 'learning_rate': '0.0001'},
    {**BASELINE, 'learning_rate': .00011},
    {**BASELINE, 'loss_family': 'ORIGINAL'}, None,
])
def test_space_rejects_partial_coerced_nonfinite_or_out_of_domain_recipes(bad):
    with pytest.raises(ValueError):
        validate_space_recipe(bad)


def test_recipe_numeric_zero_identity_is_canonical():
    assert recipe_key({**BASELINE, 'weight_decay': 0}) == recipe_key({**BASELINE, 'weight_decay': 0.0})


def test_scalar_reconstructs_history_budget_route_and_native_o1_without_diagnostics():
    full = packet()
    full['history'][0].update(proposal={'hypothesis': 'PRIVATE_DIAGNOSTIC'},
                              error='PRIVATE_DIAGNOSTIC', decision='PRIVATE_DIAGNOSTIC')
    scalar = project_feedback(full, 'naive_scalar')
    assert 'PRIVATE_DIAGNOSTIC' not in json.dumps(scalar)
    assert scalar['history'][0]['recipe'] == BASELINE
    assert scalar['history'][0]['dev_psnr'] == 25
    assert scalar['remaining_budget']['search_gpu_hours'] == 4
    assert scalar['next_action']['action'] == 'propose'
    assert scalar['feedback']['feedback_ref'] == full['feedback']['feedback_ref']
    assert scalar['feedback']['observations'][0]['id'] == 'O1'
    assert scalar['feedback']['observations'][0]['value'] == 25
    assert scalar['feedback']['observations'][0]['source'] == 'summary.json#/overall'
    assert len(scalar['feedback']['observations']) == 1
    assert 'paired' not in scalar['feedback']['overall']
    assert 'detail_refs' not in scalar['feedback']
    scalar['history'][0]['recipe']['optimizer'] = 'adamw'
    assert full['history'][0]['recipe']['optimizer'] == 'adam'


def test_scalar_execution_o1_preserves_failure_status_without_error_payload():
    full = packet()
    full['history'] = [trial(score=None, valid=False)]
    full['feedback']['overall']['dev_psnr'] = None
    full['feedback']['observations'][0].update(metric='execution_status', value=None,
                                              source='summary.json#/execution')
    full['feedback']['execution'] = {'status': 'failed', 'error': 'PRIVATE_DIAGNOSTIC',
                                      'logs': {'stderr': '/PRIVATE_DIAGNOSTIC/log'}}
    scalar = project_feedback(full, 'naive_scalar')
    assert 'PRIVATE_DIAGNOSTIC' not in json.dumps(scalar)
    assert scalar['feedback']['observations'][0]['metric'] == 'execution_status'
    assert scalar['feedback']['execution']['status'] == 'failed'
    assert scalar['history'][0]['dev_psnr'] is None


def test_scalar_keeps_complete_native_history_index_after_recent_history_is_bounded():
    full = packet()
    full['history_index'] = [
        {'run_id': 'exp_000', 'result_status': 'no_gain',
         'feedback_ref': '/PRIVATE_DIAGNOSTIC/old-summary.json',
         'decision': 'PRIVATE_DIAGNOSTIC'},
        {'run_id': 'exp_001', 'result_status': 'valid',
         'feedback_ref': '/PRIVATE_DIAGNOSTIC/current-summary.json'},
    ]
    scalar = project_feedback(full, 'naive_scalar')
    assert scalar['history_index'] == [
        {'run_id': 'exp_000', 'result_status': 'no_gain'},
        {'run_id': 'exp_001', 'result_status': 'valid'},
    ]
    assert 'PRIVATE_DIAGNOSTIC' not in json.dumps(scalar)


@pytest.mark.parametrize('strategy', ['naive_rich', 'naive_fast_slow'])
def test_rich_preserves_full_feedback_without_shared_mutation(strategy):
    full = packet()
    rich = project_feedback(full, strategy)
    assert rich == full and rich is not full
    rich['feedback']['observations'][0]['value'] = 1
    assert full['feedback']['observations'][0]['value'] == 25


def test_random_is_deterministic_and_never_reuses_completed_or_failed_recipe():
    failed = recipe_space()[0]
    history = [trial(), trial(failed, run_id='exp_002', valid=False, score=None)]
    one = suggest_recipe('random', history, 55)
    assert one == suggest_recipe('random', history, 55)
    assert one not in [BASELINE, failed]
    for slot in range(148):
        choice = suggest_recipe('random', history, slot)
        assert choice not in [t['recipe'] for t in history]
        history.append(trial(choice, run_id=f'exp_{slot + 3:03d}'))
    with pytest.raises(ValueError, match='exhaust'):
        suggest_recipe('random', history, 1)


def test_random_selects_both_remaining_recipes_across_seeds():
    recipes = recipe_space()
    history = [trial(r, run_id=f'exp_{i:03d}') for i, r in enumerate(recipes[:-2])]
    keys = [recipe_key(suggest_recipe('random', history, seed)) for seed in range(200)]
    count = keys.count(recipe_key(recipes[-1]))
    assert 65 < count < 135


@pytest.mark.parametrize('history', [[trial(score=None)], [trial(score=float('nan'))],
                                     [trial(valid='yes')], [{'recipe': BASELINE}]])
def test_samplers_reject_ambiguous_or_fabricated_history(history):
    with pytest.raises(ValueError):
        suggest_recipe('random', history, 1)


def test_tpe_uses_full_history_and_real_fail_states(monkeypatch):
    import optuna
    from optuna.trial import TrialState
    real_create = optuna.create_study
    studies = []
    def capture(**kwargs):
        study = real_create(**kwargs)
        studies.append(study)
        return study
    monkeypatch.setattr(optuna, 'create_study', capture)
    recipes = recipe_space()[20:28]
    history = [trial(r, run_id=f'exp_{i:03d}', score=20 + i) for i, r in enumerate(recipes)]
    history[3].update(valid=False, status='failed', result_status='invalid', dev_psnr=999)
    selected = suggest_recipe('tpe', history, 8)
    assert selected not in recipes
    frozen = studies[0].trials[:len(history)]
    assert len(frozen) == 8
    assert frozen[3].state == TrialState.FAIL and frozen[3].value is None
    assert frozen[-1].value == 27
    assert all(t.params == h['recipe'] for t, h in zip(frozen, history))
    assert selected == suggest_recipe('tpe', history, 8)


def test_tpe_stops_after_bounded_duplicate_draws_without_random_fallback(monkeypatch):
    import optuna
    class RepeatedTrial:
        def suggest_categorical(self, name, choices):
            return BASELINE[name]
    class RepeatedStudy:
        def __init__(self):
            self.draws = 0
        def add_trial(self, value):
            pass
        def ask(self):
            self.draws += 1
            return RepeatedTrial()
        def tell(self, *args, **kwargs):
            pass
    study = RepeatedStudy()
    monkeypatch.setattr(optuna, 'create_study', lambda **kwargs: study)
    with pytest.raises(SamplingExhausted, match='draw') as error:
        suggest_recipe('tpe', [trial()], 1, max_draws=3)
    assert study.draws == 3
    assert len(error.value.attempts) == 3
    assert all(attempt['status'] == 'duplicate' for attempt in error.value.attempts)


def test_sampler_audit_records_successful_draw_identity():
    output = suggest_recipe_with_audit('random', [trial()], 8)
    assert output['draws'] == [{'draw': 1, 'recipe': output['recipe'],
                               'key': recipe_key(output['recipe']), 'status': 'selected'}]


def test_sampler_payload_cites_real_o1_and_passes_native_proposal_and_sidecar(tmp_path):
    from tm_research.campaign import _validate_proposal
    from tm_research.research import validate_research_decision
    full = packet()
    summary = copy.deepcopy(full['feedback'])
    summary['detail_refs'] = {}
    summary['feedback_ref'] = str(tmp_path / 'summary.json')
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    full['feedback'] = summary
    candidate = {**BASELINE, 'optimizer': 'adamw'}
    payload = sampler_payload(full, candidate, 'random', 'random_proposal_001')
    assert set(payload) == {'proposal', 'decision'}
    assert set(payload['proposal']) == {'recipe', 'hypothesis', 'based_on'}
    assert 'random' in payload['proposal']['hypothesis'].lower()
    assert payload['decision']['observation_refs'] == ['fixture/exp_001/r001/O1']
    state = {'campaign_id': 'fixture', 'trials': full['history'], 'best': full['best'],
             'config': {}, 'control': {'allowed_recipe_fields': list(BASELINE)}}
    assert _validate_proposal(payload['proposal'], state) == candidate
    assert validate_research_decision(payload['decision'], state, summary) == payload['decision']


def test_sampler_payload_refuses_missing_o1_and_stale_feedback():
    full = packet()
    full['feedback']['observations'] = []
    with pytest.raises(ValueError, match='observation|O1'):
        sampler_payload(full, {**BASELINE, 'optimizer': 'adamw'}, 'tpe', 'tpe_001')
    full = packet()
    full['feedback']['latest_run_id'] = 'exp_000'
    with pytest.raises(ValueError, match='latest|identity'):
        sampler_payload(full, {**BASELINE, 'optimizer': 'adamw'}, 'random', 'random_001')


def test_sampler_payload_latest_failure_cites_execution_without_fabricated_score(tmp_path):
    from tm_research.campaign import _validate_proposal
    from tm_research.research import validate_research_decision
    full = packet()
    failed_recipe = {**BASELINE, 'loss_family': 'l1'}
    full['history'].append(trial(failed_recipe, run_id='exp_002', valid=False, score=None))
    summary = copy.deepcopy(full['feedback'])
    summary.update(run_id='exp_002', latest_run_id='exp_002', detail_refs={},
                   execution={'status': 'failed'}, feedback_ref=str(tmp_path / 'summary.json'))
    summary['overall']['dev_psnr'] = None
    summary['observations'][0].update(metric='execution_status', value=None,
                                      source='summary.json#/execution')
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    full['feedback'] = summary
    candidate = {**BASELINE, 'optimizer': 'adamw'}
    payload = sampler_payload(full, candidate, 'random', 'random_after_failure')
    assert payload['proposal']['based_on']['dev_psnr'] is None
    assert payload['decision']['latest_seen_run_id'] == 'exp_002'
    assert payload['decision']['comparison_run_id'] == 'exp_001'
    assert payload['decision']['observation_refs'] == ['fixture/exp_002/r001/O1']
    state = {'campaign_id': 'fixture', 'trials': full['history'], 'best': full['best'],
             'config': {}, 'control': {'allowed_recipe_fields': list(BASELINE)}}
    assert _validate_proposal(payload['proposal'], state) == candidate
    assert validate_research_decision(payload['decision'], state, summary) == payload['decision']


def test_sampler_payload_rejects_cross_campaign_feedback_identity():
    full = packet()
    full['feedback']['campaign_id'] = 'other_campaign'
    with pytest.raises(ValueError, match='campaign|identity'):
        sampler_payload(full, {**BASELINE, 'optimizer': 'adamw'}, 'random', 'random_001')
