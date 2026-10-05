"""End-to-end local HTTP fixtures, explicitly not pretrained-model evidence."""
import base64
import json
from pathlib import Path

from PIL import Image

from test_experiment_providers import server, success_events
from test_experiment_runner import manifest


def test_real_provider_scoped_ledger_to_native_submission(tmp_path):
    from tm_research.campaign import next_proposal
    from tm_research.experiment_costs import CostLedger
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_projection import project_feedback
    from tm_research.experiment_runner import step_experiment
    from tm_research.experiment_strategies import sampler_payload, suggest_recipe
    events = []
    with server(events) as (endpoint, received):
        path = manifest(tmp_path, ['naive_scalar'])
        data = json.loads(path.read_text())
        data['models']['naive']['base_url'] = endpoint
        path.write_text(json.dumps(data))
        root = tmp_path / 'outer'
        index = generate_experiment(path, root)
        entry = index['entries'][0]
        step_experiment(root, wait=True)
        packet = project_feedback(next_proposal(entry['campaign_dir']), 'naive_scalar')
        request_id = packet['campaign_id'] + '_fast_proposal_0001'
        payload = sampler_payload(packet, suggest_recipe('random', packet['history'], 5), 'random', request_id)
        events.extend(success_events(json.dumps(payload), usage=False))
        result = step_experiment(root, wait=True)
    assert result['status'] == 'searching' and result['entries'][0]['trial_count'] == 2
    assert len(received) == 1 and received[0]['body']['model'] == 'fixture-naive'
    user = json.loads(received[0]['body']['messages'][1]['content'])
    assert user['feedback']['overall']['dev_psnr'] == 20
    assert user['decision_id'] == request_id and 'cases' not in user['feedback']
    ledger = CostLedger(root / 'model_costs.json', index['manifest']['model_limits'])
    scope = ledger.summary(scope_id=packet['campaign_id'])
    assert scope['calls'] == 1 and scope['actual_tokens'] is None
    assert scope['charged_tokens'] > 4096
    native = json.loads((Path(entry['campaign_dir']) / 'campaign.json').read_text())
    assert native['trials'][1]['research_decision'] == payload['decision']


def test_qwen_http_receives_actual_dev_image_bytes_only(tmp_path_factory):
    from tm_research.experiment_plan import generate_experiment
    from tm_research.experiment_runner import step_experiment
    # Avoid a path with test_/tests: the image boundary intentionally rejects it.
    directory = tmp_path_factory.mktemp('dev_assets')
    events = success_events(json.dumps({'observations': ['Fixture: a colored square is visible.'],
                                      'limitations': ['No ground truth or real Qwen inference.']}))
    with server(events) as (endpoint, received):
        path = manifest(directory, ['naive_rich'])
        data = json.loads(path.read_text())
        data['models']['qwen'] = {'base_url': endpoint, 'model': 'fixture-qwen-vl', 'enabled': True}
        data['model_limits']['max_output_tokens_per_call'] = 8192
        path.write_text(json.dumps(data))
        root = directory / 'outer'
        index = generate_experiment(path, root)
        entry = index['entries'][0]
        step_experiment(root, wait=True)
        image_dir = Path(entry['directory']) / 'runs' / 'exp_001' / 'dev' / 'images'
        image_dir.mkdir(exist_ok=True)
        asset = image_dir / 'square.png'
        Image.new('RGB', (8, 8), 'purple').save(asset)
        result = step_experiment(root, wait=True)
    assert result['entries'][0]['trial_count'] == 1 and len(received) == 1
    assert received[0]['body']['max_tokens'] == 4096
    message = received[0]['body']['messages'][1]['content']
    assert base64.b64decode(message[1]['image_url']['url'].split(',', 1)[1]) == asset.read_bytes()
    identity = json.loads(message[0]['text'])
    assert identity['feedback_identity']['feedback']['latest_run_id'] == 'exp_001'
    outer = json.loads((Path(entry['directory']) / 'outer.json').read_text())
    report = next(iter(outer['auxiliary'].values()))
    assert report['status'] == 'complete' and report['model'] == 'fixture-qwen-vl'
    assert report['report']['authority'] == 'auxiliary_only'
    assert len(report['assets']) == 1 and report['assets'][0]['path'] == str(asset)
