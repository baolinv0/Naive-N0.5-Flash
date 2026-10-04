"""Engineering fixtures only: no Naive weights, GPU or live ARIS success.

The positive acceptance fixture explicitly simulates the loaded-model boundary;
its HTTP/CLI/runner plumbing is real, but its proposer/train/eval are substitutes.
"""
import json
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from tm_research import research
from tm_research.cli import main
from tm_research.naive_adapter import GenerationResult, make_server
from tm_research.workflow import register_workflow, validate_workflow_evidence


def test_absent_registered_manifest_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='registered'):
        validate_workflow_evidence({}, tmp_path, {})


def captured_campaign(tmp_path, capsys, *, simulate_loaded_model=True, register=True):
    from test_runner import config
    from test_campaign import proposal
    from test_research import decision
    from tm_research import campaign, runner
    import subprocess
    # The managed host exposes /proc from a different PID namespace. This
    # engineering fixture substitutes ONLY process supervision, while executing
    # actual train/eval subprocesses and normal campaign/CLI/adapter code.
    # Production process ownership guards remain unchanged and separately tested.
    def synchronous_start(run_id, runs_dir):
        runner._worker(Path(runs_dir) / run_id)
        return run_id
    def execute(command, log_path, cwd):
        with Path(log_path).open('w') as log:
            return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode
    patcher = pytest.MonkeyPatch()
    service_dir, capture_dir, campaign_dir = (tmp_path / name for name in ('service', 'capture', 'campaign'))
    pending = {}
    def generate(request):
        yield GenerationResult(json.dumps(pending), 'stop')
    if simulate_loaded_model:
        # Deliberate mock of transformers_generator's successful load boundary.
        # This exercises acceptance validation without claiming live model success.
        generate.loaded_identity = {'model_class': 'ENGINEERING_TEST_DOUBLE', 'tokenizer_class': 'ENGINEERING_TOKENIZER_DOUBLE',
                                     'model_path': 'TEST_ONLY'}
    server = make_server('127.0.0.1', 0, generate, workflow_dir=service_dir, host_version='TEST_ONLY')
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    patcher.setattr(campaign, 'start_prepared_run', synchronous_start)
    patcher.setattr(runner, '_execute', execute)
    def capture(action, *args):
        code = main(['capture-workflow', '--capture-dir', str(capture_dir), '--service-dir', str(service_dir),
                     '--', 'campaign', action, '--campaign-dir', str(campaign_dir), *map(str, args)])
        output = capsys.readouterr().out
        assert code == 0, output
        return json.loads(output)
    try:
        cfg = tmp_path / 'config.json'
        cfg.write_text(json.dumps(config(tmp_path)))
        current = capture('init', '--config', cfg, '--max-trials', 4, '--wait')
        for index, patch in enumerate([{'loss_family': 'mse'}, {'optimizer': 'adamw'}, {'learning_rate': 0.00005}], 1):
            evidence = capture('next')
            f = evidence['feedback']
            previous = current['trials'][-1]['run_id']
            d = {**decision(), 'decision_id': f'captured_decision_{index}', 'feedback_ref': f['feedback_ref'],
                 'latest_seen_run_id': previous, 'reference_run_ids': [previous], 'comparison_run_id': previous,
                 'observation_refs': [f'{current["campaign_id"]}/{previous}/{f["feedback_revision"]}/O1'],
                 'requested_change': patch}
            pending.clear()
            pending.update(proposal=proposal(evidence, patch), decision=d)
            request = {'stream': True, 'messages': [{'role': 'user', 'content': json.dumps(evidence)}],
                       'temperature': 0}
            with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                 data=json.dumps(request).encode(), headers={'Content-Type': 'application/json',
                                 'Authorization': 'Bearer MUST_NOT_BE_SAVED'})) as response:
                assert b'[DONE]' in response.read()
            proposal_path, decision_path = tmp_path / 'proposal.json', tmp_path / 'decision.json'
            proposal_path.write_text(json.dumps(pending['proposal']))
            decision_path.write_text(json.dumps(pending['decision']))
            current = capture('submit', '--proposal', proposal_path, '--decision', decision_path, '--wait')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        patcher.undo()
    provenance = None
    if register:
        assert main(['register-workflow', '--campaign-dir', str(campaign_dir), '--capture-dir', str(capture_dir)]) == 0
        provenance = json.loads(capsys.readouterr().out)
    current = json.loads((campaign_dir / 'campaign.json').read_text())
    return campaign_dir, capture_dir, service_dir, current, provenance


def test_normal_cli_capture_registration_and_revalidation(tmp_path, capsys):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys)
    events = validate_workflow_evidence(state, directory, provenance)
    assert [event['event'] for event in events] == ['run_completed'] + [
        'feedback_consumed', 'proposal_submitted', 'run_completed'] * 3
    assert research.build_campaign_report(directory)['claims']['W']['status'] == 'accepted'
    manifest_before = Path(provenance['manifest_ref']).read_text()
    assert register_workflow(directory, capture) == provenance
    assert Path(provenance['manifest_ref']).read_text() == manifest_before
    assert 'MUST_NOT_BE_SAVED' not in (service / 'requests.jsonl').read_text()
    assert not (directory / 'final_test').exists()


def test_injected_fixture_server_cannot_establish_live_acceptance(tmp_path, capsys):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys, simulate_loaded_model=False)
    assert json.loads((service / 'service.json').read_text())['evidence_mode'] == 'engineering_fixture'
    with pytest.raises(ValueError, match='fixture'):
        validate_workflow_evidence(state, directory, provenance)
    assert research.build_campaign_report(directory)['claims']['W']['status'] == 'pending'


@pytest.mark.parametrize('damage', ['deleted_raw', 'mutated_raw', 'event_index', 'startup', 'service', 'model_identity'])
def test_registered_evidence_mutation_stays_pending(tmp_path, capsys, damage):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys)
    path = capture / 'invocations.jsonl'
    if damage == 'deleted_raw':
        path.unlink()
    elif damage == 'mutated_raw':
        path.write_text(path.read_text().replace('cli_result', 'accepted'))
    elif damage == 'event_index':
        path = Path(provenance['transcript_ref'])
        lines = path.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        path.write_text('\n'.join(lines))
    elif damage in ('startup', 'service'):
        path = service / (damage + '.json')
        value = json.loads(path.read_text()); value['session_id'] = 'another-startup'
        path.write_text(json.dumps(value))
    else:
        provenance['model_identity'] = None
        state['workflow_evidence'] = provenance
    with pytest.raises(ValueError):
        validate_workflow_evidence(state, directory, provenance)
    assert research._w_claim(state, directory)['status'] == 'pending'


@pytest.mark.parametrize('damage', ['missing_model', 'reordered', 'wrong_feedback', 'wrong_proposal', 'pathname_only', 'events_as_raw', 'model_before_feedback', 'raw_generation'])
def test_converter_rejects_incomplete_or_unlinked_raw_records(tmp_path, capsys, damage):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys)
    # Remove registration only; exercise converter against malformed source logs.
    Path(provenance['manifest_ref']).unlink()
    requests_path = service / 'requests.jsonl'
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    if damage == 'missing_model':
        requests = requests[1:]
    elif damage == 'reordered':
        path = capture / 'invocations.jsonl'
        records = path.read_text().splitlines(); records[1], records[2] = records[2], records[1]
        path.write_text('\n'.join(records))
    elif damage in ('wrong_feedback', 'pathname_only'):
        requests[0]['request']['messages'][0]['content'] = json.dumps({'feedback_ref': 'path/only'})
    elif damage == 'wrong_proposal':
        requests[0]['response_content'] = '{}'
    elif damage == 'raw_generation':
        requests[0]['raw_generation'] = '{}'
    elif damage == 'model_before_feedback':
        startup = json.loads((service / 'startup.json').read_text())
        requests[0]['started_ns'] = startup['started_ns']
        requests[0]['finished_ns'] = startup['started_ns'] + 1
    elif damage == 'events_as_raw':
        (capture / 'invocations.jsonl').write_text(Path(provenance['transcript_ref']).read_text())
    requests_path.write_text(''.join(json.dumps(item) + '\n' for item in requests))
    with pytest.raises((ValueError, KeyError)):
        register_workflow(directory, capture)


def test_incomplete_capture_registers_but_w_stays_pending(tmp_path, capsys):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys)
    Path(provenance['manifest_ref']).unlink()
    path = capture / 'invocations.jsonl'
    # Genuine baseline CLI record alone supplies no model/submit evidence.
    path.write_text(path.read_text().splitlines()[0] + '\n')
    provenance = register_workflow(directory, capture)
    assert len(validate_workflow_evidence(state, directory, provenance)) == 1
    assert research.build_campaign_report(directory)['claims']['W']['status'] == 'pending'


def test_capture_binding_and_changed_registration_are_rejected(tmp_path, capsys):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys)
    other_service = tmp_path / 'other_service'
    other_service.mkdir()
    other = json.loads((service / 'service.json').read_text())
    other['session_id'] = 'another-real-or-fixture-startup'
    (other_service / 'service.json').write_text(json.dumps(other))
    assert main(['capture-workflow', '--capture-dir', str(capture), '--service-dir', str(other_service),
                 '--', 'campaign', 'status', '--campaign-dir', str(directory)]) == 1
    assert 'different campaign/service startup' in capsys.readouterr().out
    path = capture / 'invocations.jsonl'
    records = path.read_text().splitlines()
    record = json.loads(records[0]); record['stderr'] = 'changed after registration'
    records[0] = json.dumps(record)
    path.write_text('\n'.join(records) + '\n')
    with pytest.raises(ValueError, match='changed, not appended'):
        register_workflow(directory, capture)


def test_raw_append_does_not_invalidate_registered_prefix(tmp_path, capsys):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys)
    with (service / 'requests.jsonl').open('a') as stream:
        stream.write(json.dumps({'unrelated_later_record': True}) + '\n')
    # The registered prefix is the evidence scope; unregistered later records
    # cannot supply missing events or silently alter an existing acceptance.
    assert len(validate_workflow_evidence(state, directory, provenance)) == 10


def test_failed_service_bind_does_not_create_startup_record(tmp_path):
    def generate(request):
        yield GenerationResult('{}', 'stop')
    first = make_server('127.0.0.1', 0, generate)
    try:
        with pytest.raises(OSError):
            make_server('127.0.0.1', first.server_port, generate,
                        workflow_dir=tmp_path / 'failed_service', host_version='TEST_ONLY')
        assert not (tmp_path / 'failed_service/startup.json').exists()
    finally:
        first.server_close()


def test_failed_model_load_does_not_create_startup_record(tmp_path, monkeypatch):
    from tm_research import naive_adapter
    def fail(model_id):
        raise RuntimeError('Engineering model-load failure')
    monkeypatch.setattr(naive_adapter, 'transformers_generator', fail)
    monkeypatch.setattr('sys.argv', ['naive_adapter', '--workflow-dir', str(tmp_path / 'failed_service'),
                                    '--host-version', 'TEST_ONLY'])
    with pytest.raises(RuntimeError, match='model-load failure'):
        naive_adapter.main()
    assert not (tmp_path / 'failed_service/startup.json').exists()


def test_registration_retry_repairs_interrupted_association_and_index(tmp_path, capsys, monkeypatch):
    from tm_research import campaign
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys, register=False)
    original_save = campaign._save
    def interrupted_save(directory, state):
        if state.get('workflow_evidence'):
            raise OSError('Injected interruption after manifest write')
        return original_save(directory, state)
    monkeypatch.setattr(campaign, '_save', interrupted_save)
    with pytest.raises(OSError, match='Injected interruption'):
        register_workflow(directory, capture)
    assert (directory / 'workflow_evidence/manifest.json').exists()
    assert not json.loads((directory / 'campaign.json').read_text()).get('workflow_evidence')
    # A retry must also restore a missing derived index from unchanged raw data.
    (directory / 'workflow_evidence/events.jsonl').unlink()
    monkeypatch.setattr(campaign, '_save', original_save)
    provenance = register_workflow(directory, capture)
    state = json.loads((directory / 'campaign.json').read_text())
    assert state.get('workflow_evidence') == provenance
    assert len(validate_workflow_evidence(state, directory, provenance)) == 10
    assert research.build_campaign_report(directory)['claims']['W']['status'] == 'accepted'
    before = (directory / 'campaign.json').read_text()
    assert register_workflow(directory, capture) == provenance
    assert (directory / 'campaign.json').read_text() == before


def test_service_alias_keeps_actual_loaded_weights_identity(tmp_path):
    from tm_research.workflow import _derive
    def generate(request):
        yield GenerationResult('{}', 'stop')
    generate.loaded_identity = {'model_path': '/actual/engineering-weights',
                               'model_class': 'ENGINEERING_MODEL', 'tokenizer_class': 'ENGINEERING_TOKENIZER',
                               'api_key': 'MUST_NOT_BE_SAVED'}
    server = make_server('127.0.0.1', 0, generate, 'display-model-alias',
                         workflow_dir=tmp_path, host_version='TEST_ONLY')
    try:
        service = json.loads((tmp_path / 'service.json').read_text())
        startup = json.loads((tmp_path / 'startup.json').read_text())
        assert 'MUST_NOT_BE_SAVED' not in (tmp_path / 'startup.json').read_text()
        assert service['model_id'] == 'display-model-alias'
        assert service['weights_ref'] == startup['loaded_identity']['model_path']
        assert _derive({}, tmp_path, service, startup, [], []) == []
    finally:
        server.server_close()


def test_converter_rejects_declared_weights_different_from_actual_load(tmp_path):
    from tm_research.workflow import _derive
    def generate(request):
        yield GenerationResult('{}', 'stop')
    generate.loaded_identity = {'model_path': '/actual/engineering-weights',
                               'model_class': 'ENGINEERING_MODEL', 'tokenizer_class': 'ENGINEERING_TOKENIZER'}
    server = make_server('127.0.0.1', 0, generate, 'display-model-alias',
                         workflow_dir=tmp_path, host_version='TEST_ONLY')
    server.server_close()
    service = json.loads((tmp_path / 'service.json').read_text())
    startup = json.loads((tmp_path / 'startup.json').read_text())
    service['weights_ref'] = startup['weights_ref'] = '/different/claimed-weights'
    with pytest.raises(ValueError, match='load|weights'):
        _derive({}, tmp_path, service, startup, [], [])


@pytest.mark.parametrize('identity', [{'note': 'truthy but no actual load identity'},
                                     {'model_path': '/weights', 'model_class': 'Model'},
                                     {'model_path': '', 'model_class': 'Model', 'tokenizer_class': 'Tokenizer'}])
def test_malformed_loaded_identity_cannot_enable_live_capture(tmp_path, identity):
    def generate(request):
        yield GenerationResult('{}', 'stop')
    generate.loaded_identity = identity
    server = None
    try:
        with pytest.raises(ValueError, match='load|identity'):
            server = make_server('127.0.0.1', 0, generate, workflow_dir=tmp_path, host_version='TEST_ONLY')
        assert not (tmp_path / 'startup.json').exists()
    finally:
        if server:
            server.server_close()


@pytest.mark.parametrize('malformed', [[], None, 'campaign status', {'argv': []}])
def test_registration_cli_reports_malformed_raw_as_json_error(tmp_path, capsys, malformed):
    directory, capture, service, state, provenance = captured_campaign(tmp_path, capsys, register=False)
    path = capture / 'invocations.jsonl'
    lines = path.read_text().splitlines()
    if isinstance(malformed, dict):
        lines[0] = json.dumps([])  # Wrong top-level record container.
    else:
        first = json.loads(lines[0]); first['argv'] = malformed
        lines[0] = json.dumps(first)
    path.write_text('\n'.join(lines) + '\n')
    assert main(['register-workflow', '--campaign-dir', str(directory), '--capture-dir', str(capture)]) == 1
    error = json.loads(capsys.readouterr().out)
    assert isinstance(error.get('error'), str)
    assert not (directory / 'workflow_evidence/manifest.json').exists()
    assert not json.loads((directory / 'campaign.json').read_text()).get('workflow_evidence')
