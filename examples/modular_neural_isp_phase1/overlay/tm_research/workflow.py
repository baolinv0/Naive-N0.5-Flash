"""Small CLI/adapter evidence bridge. Raw records are evidence, never event claims.

This is an audit trail for a trusted operator's filesystem, not an attestation
system. No native ARIS transcript schema is guessed or silently imported.
"""
from contextlib import redirect_stdout, redirect_stderr
import fcntl
import io
import json
from pathlib import Path
import time
from uuid import uuid4


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    from .runner import _write_json
    _write_json(Path(path), value)


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_record(path, record):
    """Serialize concurrent adapter completions/captures without truncating logs."""
    with Path(path).open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()


def validate_loaded_identity(identity):
    """Return only loader-owned identity fields; arbitrary metadata is not live evidence."""
    if not isinstance(identity, dict):
        raise ValueError('Actual loaded identity must be an object')
    fields = ('model_path', 'model_class', 'tokenizer_class')
    for field in fields:
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError('Actual loaded identity requires ' + field)
    if identity.get('revision') is not None and not isinstance(identity['revision'], str):
        raise ValueError('Actual loaded identity revision must be a string or null')
    return {**{field: identity[field] for field in fields}, 'revision': identity.get('revision')}


def start_service_capture(directory, server, *, model_id, weights_ref, host_version,
                          loaded_identity, evidence_mode='live'):
    """Called only after model loading and successful HTTP bind by the adapter."""
    if evidence_mode == 'live':
        loaded_identity = validate_loaded_identity(loaded_identity)
        if weights_ref != loaded_identity['model_path']:
            raise ValueError('Declared weights differ from the actual load argument')
    elif evidence_mode != 'engineering_fixture':
        raise ValueError('Unrecognized service evidence mode')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    startup_path = directory / 'startup.json'
    if startup_path.exists():
        raise ValueError('Use a new service capture directory for each startup')
    session_id = uuid4().hex
    host, port = server.server_address[:2]
    endpoint = f'http://{host}:{port}/v1'
    startup = {'schema_version': 1, 'event': 'service_started', 'producer': 'tm_research.naive_adapter',
               'session_id': session_id, 'model_id': model_id, 'weights_ref': weights_ref,
               'executor_url': endpoint, 'started_ns': time.time_ns(), 'loaded_identity': loaded_identity,
               'evidence_mode': evidence_mode}
    _write(startup_path, startup)
    service = {'schema_version': 1, 'kind': 'naive_adapter_live', 'model_id': model_id,
               'weights_ref': weights_ref, 'executor_url': endpoint, 'aris_version': host_version,
               'inference_settings': {'source': 'raw model request settings'}, 'session_id': session_id,
               'startup_record_ref': str(startup_path), 'requests_ref': str(directory / 'requests.jsonl'),
               'evidence_mode': evidence_mode}
    _write(directory / 'service.json', service)
    (directory / 'requests.jsonl').touch()
    return service


def capture_cli(capture_dir, service_dir, argv):
    """Execute the ordinary CLI and preserve its actual input/output, including failures."""
    from .cli import main
    argv = list(argv)
    if argv[:1] == ['--']:
        argv.pop(0)
    supported = ('init', 'next', 'feedback', 'submit', 'wait', 'status', 'record-decision')
    if len(argv) < 2 or argv[0] != 'campaign' or argv[1] not in supported:
        raise ValueError('Capture supports ordinary campaign init/next/feedback/submit/wait/status/record-decision')
    def option(name):
        if name not in argv:
            return None
        index = argv.index(name) + 1
        if index == len(argv) or argv[index].startswith('--'):
            raise ValueError('Missing value for ' + name)
        return argv[index]
    if not option('--campaign-dir'):
        raise ValueError('Captured campaign command requires --campaign-dir')
    campaign_dir = str(Path(option('--campaign-dir')).resolve())
    directory = Path(capture_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    service_ref = str(Path(service_dir).resolve() / 'service.json')
    service = _read(service_ref)
    metadata = {'schema_version': 1, 'producer': 'tm_research.cli.capture-workflow',
                'campaign_dir': campaign_dir, 'service_config_ref': service_ref,
                'session_id': service['session_id']}
    meta_path = directory / 'capture.json'
    if meta_path.exists() and _read(meta_path) != metadata:
        raise ValueError('Capture directory is bound to a different campaign/service startup')
    _write(meta_path, metadata)
    inputs = {name: _read(option('--' + name)) for name in ('proposal', 'decision') if option('--' + name)}
    out, err = io.StringIO(), io.StringIO()
    started = time.time_ns()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    record = {'schema_version': 1, 'record_type': 'cli_result', 'producer': metadata['producer'],
              'session_id': service['session_id'], 'campaign_dir': campaign_dir,
              'argv': argv, 'inputs': inputs, 'started_ns': started, 'finished_ns': time.time_ns(),
              'stdout': out.getvalue(), 'stderr': err.getvalue(), 'exit_code': code}
    append_record(directory / 'invocations.jsonl', record)
    print(out.getvalue(), end='')
    if err.getvalue():
        import sys
        print(err.getvalue(), end='', file=sys.stderr)
    return code


def _feedback_in_request(request, result):
    # Exact parsed tool message establishes content delivery; a path or a prose
    # assertion of reading feedback cannot satisfy this condition.
    for message in request.get('messages', []):
        if message.get('role') not in ('tool', 'user'):
            continue
        try:
            if json.loads(message.get('content', '')) == result:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _derive(state, directory, service, startup, records, requests):
    if (service.get('schema_version') != 1 or service.get('kind') != 'naive_adapter_live'
            or startup.get('producer') != 'tm_research.naive_adapter'
            or startup.get('event') != 'service_started' or not startup.get('loaded_identity')):
        raise ValueError('Missing adapter load/bind startup evidence')
    if not isinstance(service.get('aris_version'), str) or not service['aris_version'].strip():
        raise ValueError('Missing host version in service evidence')
    if not isinstance(service.get('inference_settings'), dict) or startup.get('schema_version') != 1:
        raise ValueError('Malformed adapter service/startup schema')
    for key in ('model_id', 'weights_ref', 'executor_url', 'session_id', 'evidence_mode'):
        if not isinstance(service.get(key), str) or not service[key].strip() or startup.get(key) != service[key]:
            raise ValueError('Service/startup identity differs: ' + key)
    if service['evidence_mode'] == 'live':
        loaded = validate_loaded_identity(startup['loaded_identity'])
        if service['weights_ref'] != loaded['model_path']:
            raise ValueError('Declared weights differ from the actual load argument')
    elif service['evidence_mode'] != 'engineering_fixture':
        raise ValueError('Unrecognized service evidence mode')
    campaign = str(directory.resolve())
    last_time = startup['started_ns']
    outputs = []
    for record in records:
        if (record.get('schema_version') != 1 or record.get('record_type') != 'cli_result'
                or record.get('producer') != 'tm_research.cli.capture-workflow'
                or record.get('session_id') != service['session_id'] or record.get('campaign_dir') != campaign
                or record['started_ns'] < last_time or record['finished_ns'] < record['started_ns']):
            raise ValueError('Unrecognized, reordered or mismatched raw CLI capture')
        last_time = record['finished_ns']
        argv = record['argv']
        if not isinstance(argv, list) or len(argv) < 4 or not all(isinstance(arg, str) for arg in argv):
            raise ValueError('Captured argv must be a campaign command string list')
        if argv[0] != 'campaign' or str(Path(argv[argv.index('--campaign-dir') + 1]).resolve()) != campaign:
            raise ValueError('Captured invocation targets another campaign')
        if record['exit_code'] == 0:
            outputs.append((record, json.loads(record['stdout'])))
    from .naive_adapter import parse_assistant_text
    for request in requests:
        if (request.get('schema_version') != 1 or request.get('record_type') != 'model_response'
                or request.get('producer') != 'tm_research.naive_adapter'
                or request.get('session_id') != service['session_id']
                or request.get('model_id') != service['model_id']
                or request['started_ns'] < startup['started_ns']
                or request['finished_ns'] < request['started_ns']):
            raise ValueError('Unrecognized or mismatched raw model response')
        content, calls = parse_assistant_text(request['raw_generation'], request['request'].get('tools', []),
            starts_in_think=request.get('starts_in_think', False),
            allow_tool_calls=request.get('finish_reason') != 'length')
        if content != request['response_content'] or json.loads(json.dumps(calls)) != request.get('tool_calls'):
            raise ValueError('Parsed model output differs from its preserved raw generation')
    events, completed = [], set()
    def event(kind, run, decision=None):
        value = {'schema_version': 1, 'event': kind, 'run_id': run,
                 'campaign_id': state['campaign_id'], 'model_id': service['model_id'],
                 'executor_url': service['executor_url']}
        if decision:
            value.update({key: decision[key] for key in ('decision_id', 'feedback_ref', 'observation_refs')})
        events.append(value)
    returned = []
    for record, output in outputs:
        action = record['argv'][1]
        if action in ('next', 'feedback'):
            feedback = output.get('feedback') if action == 'next' else output
            if feedback and feedback.get('feedback_ref'):
                returned.append((record, output, feedback))
        if action == 'submit' and record.get('inputs', {}).get('decision'):
            decision = record['inputs']['decision']
            proposal = record['inputs']['proposal']
            matches = []
            for earlier, raw_output, feedback in returned:
                if (feedback.get('feedback_ref') != decision.get('feedback_ref')
                        or feedback.get('run_id', feedback.get('latest_run_id')) != decision.get('latest_seen_run_id')
                        or decision.get('latest_seen_run_id') not in completed):
                    continue
                for request in requests:
                    if not earlier['finished_ns'] <= request['started_ns'] <= request['finished_ns'] <= record['started_ns']:
                        continue
                    if request.get('finish_reason') != 'stop' or not _feedback_in_request(request['request'], raw_output):
                        continue
                    try:
                        response = json.loads(request['response_content'])
                    except (ValueError, TypeError):
                        continue
                    if response == {'proposal': proposal, 'decision': decision}:
                        matches.append(request)
            if not matches:
                raise ValueError('Submit lacks ordered returned feedback -> model proposal/decision -> actual submit evidence')
            trials = output.get('trials', [])
            submitted = next((trial for trial in trials if trial.get('proposal') == proposal
                              and trial.get('run_id') not in completed), None)
            if submitted is None:
                # Pending background submit may expose its proposal via active.
                submitted = output.get('active')
            if not isinstance(submitted, dict) or not submitted.get('run_id'):
                raise ValueError('Submit output has no identifiable actual submitted run; capture submit --wait')
            run = submitted['run_id']
            actual = next((trial for trial in state['trials'] if trial['run_id'] == run), None)
            if actual is None or actual.get('proposal') != proposal:
                raise ValueError('Submitted proposal differs from campaign trial')
            ref = actual.get('decision_ref') or actual.get('research_decision_ref')
            if not ref or _read(directory / ref) != decision:
                raise ValueError('Submitted decision differs from recorded campaign decision')
            event('feedback_consumed', decision['latest_seen_run_id'], decision)
            event('proposal_submitted', run, decision)
        # Only terminal trial records actually returned by the CLI establish
        # completion. Registration does not synthesize them from campaign state.
        for trial in output.get('trials', []):
            run = trial.get('run_id')
            if run not in completed and trial.get('status') == 'completed' and trial.get('valid') is True:
                event('run_completed', run)
                completed.add(run)
    return events


def register_workflow(campaign_dir, capture_dir):
    """Convert preserved capture and normalize malformed-evidence errors for the CLI."""
    try:
        return _register_workflow(campaign_dir, capture_dir)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('Malformed raw workflow evidence: ' + str(exc)) from exc


def _register_workflow(campaign_dir, capture_dir):
    """Write manifest/index and finish association on retry under the campaign lock."""
    from .campaign import _locked, _read as read_campaign, _save
    with _locked(campaign_dir) as directory:
        state = read_campaign(directory)
        capture = Path(capture_dir).resolve()
        metadata = _read(capture / 'capture.json')
        if metadata.get('producer') != 'tm_research.cli.capture-workflow' or metadata.get('campaign_dir') != str(directory):
            raise ValueError('Expected a CLI capture bound to this campaign')
        service_ref = metadata['service_config_ref']
        service = _read(service_ref)
        if service.get('session_id') != metadata.get('session_id'):
            raise ValueError('Capture service startup binding differs')
        startup_ref = service['startup_record_ref']
        raw_paths = [capture / 'capture.json', capture / 'invocations.jsonl', Path(service_ref),
                     Path(startup_ref), Path(service['requests_ref'])]
        raw = {str(path): path.read_text(encoding='utf-8') for path in raw_paths}
        events = _derive(state, directory, service, _read(startup_ref),
                         _lines(capture / 'invocations.jsonl'), _lines(service['requests_ref']))
        output = directory / 'workflow_evidence'
        output.mkdir(exist_ok=True)
        manifest_ref = output / 'manifest.json'
        transcript_ref = output / 'events.jsonl'
        provenance = {'kind': 'live_naive_aris', 'model_identity': service['model_id'],
                      'run_ids': [trial['run_id'] for trial in state['trials'][:4]],
                      'transcript_ref': str(transcript_ref), 'service_config_ref': service_ref,
                      'manifest_ref': str(manifest_ref)}
        manifest = {'schema_version': 1, 'producer': 'tm_research.workflow.register_workflow',
                    'campaign_id': state['campaign_id'], 'capture_dir': str(capture),
                    'provenance': provenance, 'preserved_raw': raw}
        if manifest_ref.exists():
            previous = _read(manifest_ref)
            appendable = {str(capture / 'invocations.jsonl'), service['requests_ref']}
            old_raw = previous.get('preserved_raw', {})
            if set(old_raw) != set(raw) or any(
                    not raw[path].startswith(content) if path in appendable else raw[path] != content
                    for path, content in old_raw.items()):
                raise ValueError('Previously registered raw evidence was changed, not appended')
        transcript = ''.join(json.dumps(event) + '\n' for event in events)
        if not transcript_ref.exists() or transcript_ref.read_text(encoding='utf-8') != transcript:
            transcript_ref.write_text(transcript, encoding='utf-8')
        if not manifest_ref.exists() or previous != manifest:
            _write(manifest_ref, manifest)
        # A prior attempt may have stopped after writing the manifest. Equality
        # of that file does not establish the independently persisted association.
        if state.get('workflow_evidence') != provenance:
            state['workflow_evidence'] = provenance
            _save(directory, state)
        return provenance


def validate_workflow_evidence(state, directory, provenance):
    """Validate registered raw provenance and rederive the event index; fail closed."""
    try:
        directory = Path(directory).resolve()
        if not isinstance(provenance, dict) or provenance.get('kind') != 'live_naive_aris' or not provenance.get('manifest_ref'):
            raise ValueError('Workflow needs a registered raw capture manifest')
        manifest = _read(provenance['manifest_ref'])
        if (manifest.get('schema_version') != 1 or manifest.get('producer') != 'tm_research.workflow.register_workflow'
                or manifest.get('campaign_id') != state['campaign_id'] or manifest.get('provenance') != provenance):
            raise ValueError('Registered workflow manifest/provenance differs')
        raw = manifest['preserved_raw']
        capture = Path(manifest['capture_dir'])
        metadata = _read(capture / 'capture.json')
        service = _read(provenance['service_config_ref'])
        required = {str(capture / 'capture.json'), str(capture / 'invocations.jsonl'),
                    provenance['service_config_ref'], service['startup_record_ref'], service['requests_ref']}
        appendable = {str(capture / 'invocations.jsonl'), service['requests_ref']}
        if not raw or any(
                not Path(path).read_text(encoding='utf-8').startswith(content) if path in appendable
                else Path(path).read_text(encoding='utf-8') != content for path, content in raw.items()):
            raise ValueError('Raw workflow evidence changed after registration; preserve the original capture')
        if set(raw) != required or metadata.get('session_id') != service['session_id'] or metadata.get('campaign_dir') != str(directory):
            raise ValueError('Manifest raw sources/service binding differs')
        if provenance.get('model_identity') != service.get('model_id'):
            raise ValueError('Registered model identity differs from its service')
        if service.get('evidence_mode') != 'live':
            raise ValueError('Engineering fixture capture cannot establish live W acceptance')
        events = _derive(state, directory, service, _read(service['startup_record_ref']),
                         [json.loads(line) for line in raw[str(capture / 'invocations.jsonl')].splitlines() if line.strip()],
                         [json.loads(line) for line in raw[service['requests_ref']].splitlines() if line.strip()])
        if events != _lines(provenance['transcript_ref']):
            raise ValueError('Structured workflow events differ from preserved raw evidence')
        return events
    except (OSError, KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('Missing or malformed registered raw workflow evidence: ' + str(exc)) from exc
