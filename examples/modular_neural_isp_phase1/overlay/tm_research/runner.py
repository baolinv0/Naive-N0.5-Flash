"""Run fixed-model recipes and independently reload every selected checkpoint on DEV."""

import json
import fcntl
import signal
import socket
import threading
import uuid
from contextlib import contextmanager
import math
import re
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RECIPE_DEFAULTS = {'loss_family': 'original', 'optimizer': 'adam',
                   'learning_rate': 0.0001, 'weight_decay': 0.0000001}
BUDGET_DEFAULTS = {'epochs': 600, 'batch_size': 8, 'in_size': 512,
                   'eval_size': 512, 'validation_frequency': 4, 'num_workers': 12}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _absolute(value, base):
    return str((base / value).resolve())


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) - set(RECIPE_DEFAULTS):
        raise ValueError('Recipe supports only loss_family, optimizer, learning_rate, weight_decay')
    result = {**RECIPE_DEFAULTS, **recipe}
    if result['loss_family'] not in ('original', 'mse', 'l1'):
        raise ValueError('loss_family must be original, mse or l1')
    if result['optimizer'] not in ('adam', 'adamw'):
        raise ValueError('optimizer must be adam or adamw')
    for key in ('learning_rate', 'weight_decay'):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'learning_rate' and value == 0):
            raise ValueError(f'{key} must be finite and ' + ('positive' if key == 'learning_rate' else 'nonnegative'))
    return result


def normalize_config(config, base=None):
    """Resolve paths and check the deliberately small recipe and budget surface."""
    base = Path(base or Path.cwd()).resolve()
    cfg = dict(config)
    allowed = {'repo_dir', 'python', 'runs_dir', 'train', 'validation', 'test',
               'init_checkpoint', 'init_config_dir', 'seed', *BUDGET_DEFAULTS, *RECIPE_DEFAULTS}
    extra = set(cfg) - allowed
    if extra:
        raise ValueError(f'Unsupported config fields: {sorted(extra)}')
    for key in ('repo_dir', 'runs_dir', 'init_checkpoint', 'init_config_dir'):
        if cfg.get(key) is not None:
            cfg[key] = _absolute(cfg[key], base)
    if not cfg.get('repo_dir'):
        raise ValueError('repo_dir is required')
    cfg.setdefault('runs_dir', str(Path(cfg['repo_dir']) / 'runs'))
    interpreter = cfg.get('python', sys.executable)
    if '/' not in interpreter and '\\' not in interpreter:
        interpreter = shutil.which(interpreter) or interpreter
    cfg['python'] = os.path.abspath(os.path.join(base, interpreter))
    for split in ('train', 'validation', 'test'):
        if not cfg.get(split):
            if split == 'test':
                continue
            raise ValueError(f'{split} split is required')
        details = dict(cfg[split])
        if set(details) - {'input_dir', 'gt_dir', 'metadata_dir', 'expected_count'}:
            raise ValueError(f'Unsupported {split} fields')
        for key in ('input_dir', 'gt_dir'):
            if not details.get(key):
                raise ValueError(f'{split}.{key} is required')
        for key in ('input_dir', 'gt_dir', 'metadata_dir'):
            if details.get(key) is not None:
                details[key] = _absolute(details[key], base)
        if 'expected_count' in details and (type(details['expected_count']) is not int or details['expected_count'] < 1):
            raise ValueError(f'{split}.expected_count must be a positive integer')
        cfg[split] = details
    for field in BUDGET_DEFAULTS:
        if field in cfg and (type(cfg[field]) is not int or cfg[field] < (0 if field == 'num_workers' else 1)):
            raise ValueError(f'{field} must be a positive integer' if field != 'num_workers' else 'num_workers must be a nonnegative integer')
    if cfg.get('seed') is not None and (type(cfg['seed']) is not int or cfg['seed'] < 0):
        raise ValueError('seed must be a nonnegative integer')
    validate_recipe({key: cfg[key] for key in RECIPE_DEFAULTS if key in cfg})
    cfg.setdefault('eval_size', 512)
    return cfg


def load_config(path):
    import yaml
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return normalize_config(cfg, path.parent)


def evaluation_command(config, run_dir, split='validation'):
    cfg = normalize_config(config)
    details = cfg[split]
    command = [cfg['python'], str(Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'),
               '--model-path', '<best_checkpoint>', '--config-dir', '<train_config_dir>',
               '--in-testing-dir', details['input_dir'], '--gt-testing-dir', details['gt_dir'],
               '--result-dir', str(Path(run_dir).resolve() / ('dev' if split == 'validation' else 'final_test')),
               '--eval-size', str(cfg['eval_size'])]
    if details.get('metadata_dir'):
        command += ['--data-testing-dir', details['metadata_dir']]
    return command


def commands_for_run(config, run_dir):
    """Return training and DEV reload commands. Ordinary runs never execute TEST."""
    cfg = normalize_config(config)
    run_dir = Path(run_dir).resolve()
    script_dir = Path(cfg['repo_dir']) / 'photofinishing'
    train = [cfg['python'], str(script_dir / 'train.py'),
             '--in-training-dir', cfg['train']['input_dir'], '--gt-training-dir', cfg['train']['gt_dir'],
             '--in-validation-dir', cfg['validation']['input_dir'], '--gt-validation-dir', cfg['validation']['gt_dir'],
             '--output-dir', str(run_dir / 'train'), '--exp-name', run_dir.name]
    if cfg.get('init_checkpoint'):
        config_dir = Path(cfg.get('init_config_dir') or (script_dir / 'config'))
        checkpoint_config = json.loads((config_dir / (Path(cfg['init_checkpoint']).stem + '.json')).read_text(encoding='utf-8'))
        if checkpoint_config['use_3d_lut']:
            train.append('--use-3d-lut')
    for split, label in (('train', 'training'), ('validation', 'validation')):
        if cfg[split].get('metadata_dir'):
            train += [f'--data-{label}-dir', cfg[split]['metadata_dir']]
    for key, flag in (('epochs', '--epochs'), ('batch_size', '--batch-size'),
                      ('in_size', '--in-size'), ('eval_size', '--eval-size'),
                      ('validation_frequency', '--validation-frequency'), ('num_workers', '--num-workers'),
                      ('seed', '--seed'), ('init_checkpoint', '--load'), ('init_config_dir', '--load-config-dir'),
                      ('loss_family', '--loss-family'), ('optimizer', '--optimizer'),
                      ('learning_rate', '--learning-rate'), ('weight_decay', '--weight-decay')):
        if cfg.get(key) is not None:
            train += [flag, str(cfg[key])]
    return train, evaluation_command(cfg, run_dir)


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _state(run_dir, **changes):
    path = Path(run_dir) / 'state.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data.update(changes)
    _write_json(path, data)
    return data


def expected_count(details):
    """Count DEV inputs independently; paired_files in test.py verifies GT/metadata."""
    count = sum(p.is_file() and p.suffix.lower() == '.png' for p in Path(details['input_dir']).iterdir())
    if count < 1:
        raise ValueError('Evaluation requires at least one input image')
    if details.get('expected_count', count) != count:
        raise ValueError(f'Configured expected_count {details["expected_count"]} differs from input count {count}')
    return count


def validate_metrics(metrics, count, eval_size):
    score = metrics.get('mean_per_image_psnr', metrics.get('mean_psnr'))
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError('Independent evaluation must report finite mean per-image PSNR')
    if type(metrics.get('num_images')) is not int or metrics['num_images'] != count or count < 1:
        raise ValueError(f'Independent evaluation sample count must equal {count}')
    if metrics.get('protocol') != f'P{eval_size}' or metrics.get('eval_size') != eval_size:
        raise ValueError(f'Independent evaluation protocol must be P{eval_size}')
    if metrics.get('finite') is False:
        raise ValueError('Independent evaluation reported nonfinite values')
    return float(score)


TERMINAL = ('completed', 'failed', 'interrupted')
_EXECUTION = threading.local()


@contextmanager
def _lock(path, *, blocking=True):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def _process_identity(pid):
    """Linux process start identity; zombies have already released allocations."""
    if not Path('/proc/sys/kernel/random/boot_id').is_file():
        return {'hostname': socket.gethostname(), 'pid': int(pid), 'unverifiable': True}
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'hostname': socket.gethostname(), 'pid': int(pid),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks': fields[19]}
    except FileNotFoundError:
        return None
    except (OSError, IndexError, ValueError):
        return {'hostname': socket.gethostname(), 'pid': int(pid), 'unverifiable': True}


def _identity_liveness(identity):
    if not identity or identity.get('hostname') != socket.gethostname():
        return 'unknown'
    if identity.get('unverifiable') or not identity.get('boot_id') or not identity.get('start_ticks'):
        return 'unknown'
    current = _process_identity(identity.get('pid'))
    if current is None:
        return 'dead'
    if current.get('unverifiable'):
        return 'unknown'
    return 'live' if current == identity else 'dead'


def _validate_limits(limits):
    limits = dict(limits or {})
    if set(limits) - {'allocated_gpus', 'job_walltime_seconds'}:
        raise ValueError('Unsupported execution limits')
    if 'allocated_gpus' in limits and (type(limits['allocated_gpus']) is not int or limits['allocated_gpus'] < 0):
        raise ValueError('allocated_gpus must be a nonnegative integer')
    if 'job_walltime_seconds' in limits:
        value = limits['job_walltime_seconds']
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('job_walltime_seconds must be finite and positive')
    return limits


def prepare_run(config, *, observation_config=None, execution_limits=None, request_id=None):
    """Persist an exact launch request before a controller publishes active state.

    A caller can persist request_id first, then repeat preparation after a crash.
    The durable allocation remains reserved even before its run directory exists.
    """
    cfg = normalize_config(config)
    cfg.pop('test', None)
    limits = _validate_limits(execution_limits)
    observation = dict(observation_config or {})
    profile = observation.get('profile_ref') or observation.get('diagnostics_profile')
    if profile:
        observation['profile_ref'] = str(Path(profile).resolve())
    if request_id is not None and (not isinstance(request_id, str) or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id)):
        raise ValueError('request_id must be a safe nonempty launch request identifier')
    runs_dir = Path(cfg['runs_dir'])
    runs_dir.mkdir(parents=True, exist_ok=True)
    requests = runs_dir / 'prepare_requests'
    with _lock(runs_dir / 'allocation.lock'):
        snapshot = {'config': cfg, 'observation_config': observation, 'execution_limits': limits}
        request_path = None if request_id is None else requests / (request_id + '.json')
        existing = _read_json(request_path) if request_path else None
        if existing:
            if existing['snapshot'] != snapshot:
                raise ValueError('Launch request already belongs to a different config, observation or limits')
            run_id = existing['run_id']
        else:
            reserved = {_read_json(path)['run_id'] for path in requests.glob('*.json')} if requests.exists() else set()
            for number in range(1, 1000000):
                run_id = f'exp_{number:03d}'
                if run_id not in reserved and not (runs_dir / run_id).exists():
                    break
            else:
                raise RuntimeError('No free experiment ID')
            if request_path:
                requests.mkdir(exist_ok=True)
                _write_json(request_path, {'request_id': request_id, 'run_id': run_id,
                                          'snapshot': snapshot, 'created_at': _now()})
        run_dir = runs_dir / run_id
        if (run_dir / 'state.json').exists():
            return run_id
        run_dir.mkdir(exist_ok=True)
        _prepare_run_files(cfg, observation, limits, run_id, run_dir)
        if request_id:
            _write_json(run_dir / 'request.json', {'request_id': request_id})
    return run_id


def _prepare_run_files(cfg, observation, limits, run_id, run_dir):
    train, dev = commands_for_run(cfg, run_dir)
    profile = observation.get('profile_ref')
    if profile:
        dev += ['--diagnostics-profile', profile]
    _write_json(run_dir / 'config.json', cfg)
    _write_json(run_dir / 'commands.json', {'train': train, 'dev': dev})
    _write_json(run_dir / 'execution_limits.json', limits)
    _write_json(run_dir / 'observation_config.json', observation)
    _write_json(run_dir / 'state.json', {'run_id': run_id, 'status': 'queued', 'result_status': 'pending',
        'valid': False, 'reload_succeeded': False, 'created_at': _now(), 'run_dir': str(run_dir),
        'recipe': validate_recipe({k: cfg[k] for k in RECIPE_DEFAULTS if k in cfg}),
        'diagnostics_status': 'pending' if profile else 'unavailable',
        'execution_limits': limits, 'observation_config': observation,
        'logs': {name: str(run_dir / f'{name}.log') for name in ('worker', 'train', 'dev')}})


def _run_path(run_id, runs_dir):
    if not run_id.startswith('exp_') or not run_id[4:].isdigit():
        raise ValueError('Expected an exp_NNN run ID')
    root = Path(runs_dir).resolve() / run_id
    if not (root / 'state.json').is_file():
        raise FileNotFoundError(f'Unknown run {run_id} in {runs_dir}')
    return root


def _recorded_resource_liveness(root):
    process = _read_json(Path(root) / 'process.json', {})
    if process.get('pending'):
        return 'unknown'
    pgid = process.get('pgid')
    if not pgid:
        return 'dead'
    identity = process.get('identity')
    if not identity or _identity_liveness(identity) == 'unknown':
        return 'dead' if _group_liveness(pgid) == 'dead' else 'unknown'
    return _group_liveness(pgid)


def _unknown_release(root, state):
    usage = dict(state.get('usage') or {})
    usage.setdefault('allocated_gpus', _read_json(root / 'execution_limits.json', {}).get('allocated_gpus'))
    usage.update(gpu_hours=None, status='unknown')
    return _state(root, status='recovery_required', result_status='invalid', valid=False,
                  resource_liveness='unknown', usage=usage)


def inspect_run(run_id, runs_dir='runs', *, now=None):
    root = _run_path(run_id, runs_dir)
    result = _read_json(root / 'state.json')
    if result['status'] in TERMINAL:
        liveness = 'terminal' if _recorded_resource_liveness(root) == 'dead' else 'unknown'
    else:
        launch = _read_json(root / 'launch.json')
        identity = result.get('worker_identity') or (launch or {}).get('identity')
        if identity:
            liveness = _identity_liveness(identity)
            if liveness == 'dead' and _recorded_resource_liveness(root) == 'unknown':
                liveness = 'unknown'
        elif launch or result.get('worker_pid') or (root / 'worker.pid').exists():
            liveness = 'unknown'
        elif result['status'] == 'queued' and not (root / 'train').exists():
            liveness = 'not_started'
        else:
            liveness = 'unknown'
    return {**result, 'liveness': liveness}


def start_prepared_run(run_id, runs_dir='runs'):
    root = _run_path(run_id, runs_dir)
    with _lock(root / 'lifecycle.lock'):
        result = inspect_run(run_id, runs_dir)
        if result['liveness'] in ('live', 'terminal', 'dead'):
            if result['liveness'] == 'dead':
                _recover_dead(root)
            return run_id
        if result['liveness'] == 'unknown':
            raise RuntimeError(f'Run {run_id} liveness unknown; refusing another worker')
        cfg = _read_json(root / 'config.json')
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [source_root, env.get('PYTHONPATH', '')]))
        # An interrupted launch with no identity is unknown, never assumed unstarted.
        limits = _read_json(root / 'execution_limits.json', {})
        launched = {'started_at': _now(), 'hostname': socket.gethostname(), 'started_monotonic': time.monotonic()}
        if limits.get('job_walltime_seconds') is not None:
            launched['deadline_monotonic'] = launched['started_monotonic'] + limits['job_walltime_seconds']
        _write_json(root / 'launch.json', launched)
        try:
            with (root / 'worker.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, '-c', 'import sys; from tm_research.runner import _supervise; _supervise(sys.argv[1])', str(root)],
                    cwd=cfg['repo_dir'], env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            _write_json(root / 'launch.json', {'started_at': _now(), 'start_error': str(exc)})
            _recover_dead(root, error=f'Could not start worker: {exc}')
            raise RuntimeError(f'Could not start worker for {run_id}: {exc}') from exc
        _write_json(root / 'launch.json', {**launched, 'identity': _process_identity(child.pid)})
        (root / 'supervisor.pid').write_text(str(child.pid) + '\n', encoding='utf-8')
    return run_id


def _supervise(run_dir):
    """Independent same-host watchdog survives worker crashes and enforces its deadline."""
    root = Path(run_dir).resolve()
    with _lock(root / 'supervisor.lock', blocking=False):
        with _lock(root / 'lifecycle.lock'):
            launch = _read_json(root / 'launch.json', {})
        cfg = _read_json(root / 'config.json')
        previous = _read_json(root / 'state.json')
        if previous['status'] in TERMINAL:
            return
        try:
            worker = subprocess.Popen([sys.executable, '-c',
                'import sys; from tm_research.runner import _worker; _worker(sys.argv[1])', str(root)],
                cwd=cfg['repo_dir'], stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            with _lock(root / 'lifecycle.lock'):
                _recover_dead(root, error=f'Could not start worker: {exc}')
            return
        (root / 'worker.pid').write_text(str(worker.pid) + '\n', encoding='utf-8')
        _write_json(root / 'supervisor.json', {'identity': _process_identity(os.getpid()),
                    'worker_identity': _process_identity(worker.pid), 'started_at': _now()})
        forced_at = None
        while worker.poll() is None:
            deadline = launch.get('deadline_monotonic')
            soft_deadline = None if deadline is None else deadline - min(1., (deadline - launch['started_monotonic']) * .1)
            if soft_deadline is not None and time.monotonic() >= soft_deadline:
                if not (root / 'stop_request.json').exists():
                    _write_json(root / 'stop_request.json', {'reason': 'Whole TRAIN+DEV job walltime exceeded', 'requested_at': _now()})
                forced_at = forced_at or time.monotonic()
            if (root / 'stop_request.json').exists():
                forced_at = forced_at or time.monotonic()
            hard_limit = deadline is not None and time.monotonic() >= deadline
            if hard_limit or forced_at is not None and time.monotonic() - forced_at >= 2.:
                _terminate_group(worker.pid, grace=0. if hard_limit else 1.)
                worker.wait()
                break
            time.sleep(.02)
        # A just-forked stage wrapper publishes its identity before exec. Give
        # that handshake a bounded opportunity to settle after a worker crash.
        settle = time.monotonic() + .2
        while _read_json(root / 'process.json', {}).get('pending') and time.monotonic() < settle:
            time.sleep(.01)
        # The worker is confirmed dead. Converge once, including orphan GPU children.
        with _lock(root / 'lifecycle.lock'):
            reason = _read_json(root / 'stop_request.json', {}).get('reason')
            final = _recover_dead(root, error=reason or 'Worker exited without a terminal result')
        _write_json(root / 'supervisor.json', {'identity': _process_identity(os.getpid()),
                    'worker_returncode': worker.returncode, 'finished_at': _now(),
                    'terminal_status': final.get('status')})


def run_baseline(config, *, observation_config=None, execution_limits=None):
    """Compatibility wrapper: ordinary runs still independently reload DEV."""
    cfg = normalize_config(config)
    run_id = prepare_run(cfg, observation_config=observation_config, execution_limits=execution_limits)
    return start_prepared_run(run_id, cfg['runs_dir'])


class _Interrupted(RuntimeError):
    pass


def _check_execution():
    context = getattr(_EXECUTION, 'context', None)
    if context:
        request = _read_json(context['root'] / 'stop_request.json')
        if request:
            raise _Interrupted('Stop requested: ' + str(request.get('reason', 'requested')))
        if context.get('soft_deadline', context.get('deadline')) is not None and time.monotonic() >= context.get('soft_deadline', context.get('deadline')):
            raise _Interrupted('Whole TRAIN+DEV job walltime exceeded')


def _group_liveness(pgid):
    """Ignore zombies, but never infer exit from an unreadable process table."""
    unknown = False
    try:
        entries = list(Path('/proc').iterdir())
    except OSError:
        return 'unknown'
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                return 'live'
        except FileNotFoundError:
            continue
        except (OSError, IndexError, ValueError):
            unknown = True
    return 'unknown' if unknown else 'dead'


def _terminate_group(pgid, *, grace=1.0):
    context = getattr(_EXECUTION, 'context', None)
    if context and context.get('deadline') is not None:
        grace = min(grace, max(0., context['deadline'] - time.monotonic()))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return _group_liveness(pgid) == 'dead'
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if _group_liveness(pgid) == 'dead':
                    return True
                time.sleep(.01)
    # KILL delivery is asynchronous. Confirm release before clearing ownership;
    # kernel-uninterruptible/unverifiable survivors retain their reservation.
    deadline = time.monotonic() + .2
    while time.monotonic() < deadline:
        if _group_liveness(pgid) == 'dead':
            return True
        time.sleep(.01)
    return _group_liveness(pgid) == 'dead'


def _publish_process_identity(root, identity, pgid):
    """An established stage identity cannot be erased by a later PID lookup."""
    root = Path(root)
    with _lock(root / 'process.lock'):
        previous = _read_json(root / 'process.json', {})
        established = previous.get('identity')
        if established and previous.get('pgid') != pgid:
            raise RuntimeError('Scientific process ownership changed during launch')
        identity = established or identity
        if identity:
            _write_json(root / 'process.json', {'identity': identity, 'pgid': pgid})


class _UnresolvedExecution(RuntimeError):
    pass


def _stage_exec(root, command, owner):
    """Publish a scientific process identity before exec, closing the launch gap."""
    root = Path(root)
    _publish_process_identity(root, _process_identity(os.getpid()), os.getpid())
    if _identity_liveness(owner) != 'live':
        return 125
    os.execvpe(command[0], command, os.environ)


def _execute(command, log_path, cwd):
    """Execute in a separate group so stop/walltime reaches grandchildren."""
    _check_execution()
    with Path(log_path).open('w', encoding='utf-8') as log:
        context = getattr(_EXECUTION, 'context', None)
        actual_command = command
        if context:
            _write_json(context['root'] / 'process.json', {'pending': True, 'identity': None})
            actual_command = [sys.executable, '-c', 'import json,sys; from tm_research.runner import _stage_exec; sys.exit(_stage_exec(sys.argv[1],json.loads(sys.argv[2]),json.loads(sys.argv[3])))',
                              str(context['root']), json.dumps(command), json.dumps(_process_identity(os.getpid()))]
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [source_root, env.get('PYTHONPATH', '')]))
        try:
            child = subprocess.Popen(actual_command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            if context:
                _write_json(context['root'] / 'process.json', {'pending': False, 'identity': None})
            raise
        try:
            if context:
                _publish_process_identity(context['root'], _process_identity(child.pid), child.pid)
            while child.poll() is None:
                _check_execution()
                time.sleep(.02)
            _check_execution()
            return child.returncode
        finally:
            if not _terminate_group(child.pid):
                raise _UnresolvedExecution('Scientific process group release is unconfirmed')
            child.wait()
            if context:
                _write_json(context['root'] / 'process.json', {'identity': None, 'pgid': None, 'returncode': child.returncode})


def _resolve_evaluation(command, train_metrics):
    return [train_metrics['best_checkpoint'] if v == '<best_checkpoint>' else
            train_metrics['config_dir'] if v == '<train_config_dir>' else v for v in command]


def _usage(limits, elapsed):
    allocated = limits.get('allocated_gpus')
    return {'allocated_gpus': allocated, 'elapsed_seconds': max(0., elapsed),
            'gpu_hours': None if allocated is None else allocated * max(0., elapsed) / 3600,
            'status': 'unavailable' if allocated is None else 'measured'}


def _append_result(root, final):
    marker = root / 'result_recorded.json'
    if marker.exists():
        return
    with _lock(root.parent / 'results.lock'):
        path = root.parent / 'results.jsonl'
        # Recover an interrupted append before marker creation without double entries.
        existing = path.read_text().splitlines() if path.exists() else []
        if not any(_json_run_id(line) == final['run_id'] for line in existing):
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(final) + '\n')
        _write_json(marker, {'recorded_at': _now()})


def _json_run_id(line):
    try:
        return json.loads(line).get('run_id')
    except (ValueError, TypeError):
        return None


def _cleanup_recorded_process(root):
    process = _read_json(root / 'process.json', {})
    identity = process.get('identity')
    if process.get('pending'):
        return False
    if not identity:
        return not process.get('pgid') or _group_liveness(process['pgid']) == 'dead'
    launch = _read_json(root / 'launch.json') or _read_json(root / 'evaluation_state.json', {})
    grace = min(1., max(0., launch.get('deadline_monotonic', time.monotonic() + 1.) - time.monotonic()))
    liveness = _identity_liveness(identity)
    if liveness == 'unknown':
        return False
    if liveness == 'live':
        return _terminate_group(process['pgid'], grace=grace)
    elif (_process_identity(identity['pid']) is None
          and identity.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        # The original leader exited but descendants retain its group ID. An ID
        # cannot become a new process group while the old group still exists.
        return _terminate_group(process['pgid'], grace=grace)
    return _group_liveness(process['pgid']) == 'dead'


def _recover_dead(root, error='Worker exited without a terminal result'):
    try:
        with _lock(root / 'worker.lock', blocking=False):
            state = _read_json(root / 'state.json')
            resources_were_live = _recorded_resource_liveness(root) == 'live'
            if not _cleanup_recorded_process(root):
                return _unknown_release(root, state)
            if state['status'] in TERMINAL:
                if resources_were_live:
                    limits = _read_json(root / 'execution_limits.json', {})
                    start_at = state.get('started_at') or _read_json(root / 'launch.json', {}).get('started_at')
                    elapsed = 0. if not start_at else (datetime.now(timezone.utc) - datetime.fromisoformat(start_at)).total_seconds()
                    state = _state(root, finished_at=_now(), usage=_usage(limits, elapsed), resource_liveness='dead')
                _append_result(root, state)
                return state
            limits = _read_json(root / 'execution_limits.json', {})
            elapsed = 0.
            start_at = state.get('started_at') or _read_json(root / 'launch.json', {}).get('started_at')
            if start_at:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(start_at)).total_seconds()
            final = _state(root, status='failed', result_status='invalid', valid=False,
                execution_status='interrupted', error=error, finished_at=_now(), usage=_usage(limits, elapsed))
            _append_result(root, final)
            return final
    except BlockingIOError:
        return _read_json(root / 'state.json')


def request_run_stop(run_id, runs_dir='runs', *, reason='requested'):
    root = _run_path(run_id, runs_dir)
    result = inspect_run(run_id, runs_dir)
    if result['liveness'] == 'terminal':
        return {'run_id': run_id, 'stop_requested': False, 'stopped': True, 'status': result['status']}
    _write_json(root / 'stop_request.json', {'reason': str(reason), 'requested_at': _now()})
    return {'run_id': run_id, 'stop_requested': True, 'stopped': False, 'liveness': result['liveness']}


def _worker(run_dir):
    root = Path(run_dir).resolve()
    try:
        with _lock(root / 'worker.lock', blocking=False):
            # Wait for launcher to finish writing the initial process identity.
            with _lock(root / 'lifecycle.lock'):
                pass
            previous = _read_json(root / 'state.json')
            if previous['status'] in TERMINAL:
                _append_result(root, previous)
                return previous
            return _worker_locked(root)
    except BlockingIOError:
        return _read_json(root / 'state.json')


def _worker_locked(run_dir):
    cfg = _read_json(run_dir / 'config.json')
    commands = _read_json(run_dir / 'commands.json')
    limits = _read_json(run_dir / 'execution_limits.json', {})
    launch = _read_json(run_dir / 'launch.json', {})
    started = launch.get('started_monotonic', time.monotonic())
    context = {'root': run_dir, 'deadline': launch.get('deadline_monotonic', None if 'job_walltime_seconds' not in limits else started + limits['job_walltime_seconds'])}
    if context['deadline'] is not None:
        context['soft_deadline'] = context['deadline'] - min(1., limits['job_walltime_seconds'] * .1)
    _EXECUTION.context = context
    state = _state(run_dir, status='training', started_at=launch.get('started_at', _now()), worker_started_at=_now(), worker_pid=os.getpid(),
                   worker_identity=_process_identity(os.getpid()), hostname=socket.gethostname())
    execution_failed = False
    rc = None
    try:
        _check_execution()
        rc = _execute(commands['train'], state['logs']['train'], cfg['repo_dir'])
        _check_execution()
        if rc:
            execution_failed = True
            raise RuntimeError(f'train.py exited with status {rc}')
        train_metrics = _read_json(run_dir / 'train' / 'metrics.json')
        checkpoint = Path(train_metrics['best_checkpoint'])
        if not checkpoint.is_file():
            raise ValueError(f'Selected checkpoint is missing: {checkpoint}')
        count = expected_count(cfg['validation'])
        dev = _resolve_evaluation(commands['dev'], train_metrics)
        _write_json(run_dir / 'commands.json', {'train': commands['train'], 'dev': dev})
        _state(run_dir, status='evaluating_dev', train_metrics=train_metrics, expected_count=count)
        rc = _execute(dev, state['logs']['dev'], cfg['repo_dir'])
        _check_execution()
        if rc:
            execution_failed = True
            raise RuntimeError(f'Independent DEV test.py reload exited with status {rc}')
        _state(run_dir, reload_succeeded=True)
        metrics_path = run_dir / 'dev' / 'metrics.json'
        dev_metrics = _read_json(metrics_path)
        score = validate_metrics(dev_metrics, count, cfg['eval_size'])
        final = _state(run_dir, status='completed', result_status='valid', valid=True,
            finished_at=_now(), dev_psnr=score, dev_metrics=dev_metrics,
            diagnostics_status=dev_metrics.get('diagnostics_status', 'unavailable'),
            diagnostics_errors=dev_metrics.get('diagnostics_errors', []),
            artifacts={'best_checkpoint': str(checkpoint), 'config_dir': train_metrics['config_dir'],
                       'train_metrics': str(run_dir / 'train' / 'metrics.json'),
                       'dev_metrics': str(metrics_path), 'per_image_csv': dev_metrics.get('per_image_csv'),
                       'roi_metrics_csv': dev_metrics.get('roi_metrics_csv'),
                       'roi_profile_ref': dev_metrics.get('roi_profile_ref')},
            usage=_usage(limits, time.monotonic() - started))
    except Exception as exc:
        interrupted = isinstance(exc, _Interrupted)
        unresolved = isinstance(exc, _UnresolvedExecution)
        usage = _usage(limits, time.monotonic() - started)
        if unresolved:
            usage.update(gpu_hours=None, status='unknown')
        final = _state(run_dir, status='recovery_required' if unresolved else 'failed' if execution_failed or interrupted else 'completed',
            result_status='invalid', valid=False, finished_at=_now(), error=str(exc), returncode=rc,
            execution_status='interrupted' if interrupted else 'finished',
            usage=usage)
    finally:
        _EXECUTION.context = None
    if final['status'] in TERMINAL:
        _append_result(run_dir, final)
    return final


def get_result(run_id, runs_dir='runs'):
    root = _run_path(run_id, runs_dir)
    result = inspect_run(run_id, runs_dir)
    if result['liveness'] == 'dead':
        with _lock(root / 'lifecycle.lock'):
            if inspect_run(run_id, runs_dir)['liveness'] == 'dead':
                _recover_dead(root)
    elif result['status'] in TERMINAL and (result['liveness'] != 'terminal' or not (root / 'result_recorded.json').exists()):
        with _lock(root / 'lifecycle.lock'):
            _recover_dead(root)
    return _read_json(root / 'state.json')


def wait_for_result(run_id, runs_dir='runs', timeout=None, poll_interval=0.2):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        result = get_result(run_id, runs_dir)
        if result['status'] in TERMINAL:
            return result
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f'Run {run_id} is still {result["status"]}')
        time.sleep(poll_interval)


def _evaluation_watchdog(run_dir):
    """Reuse process-group cleanup for a frozen evaluation if its caller dies."""
    root = Path(run_dir).resolve()
    with _lock(root / 'evaluation_watchdog.lock', blocking=False):
        while True:
            state = _read_json(root / 'evaluation_state.json')
            if state['status'] in TERMINAL:
                resources_were_live = _recorded_resource_liveness(root) == 'live'
                if not _cleanup_recorded_process(root):
                    _write_json(root / 'evaluation_health.json', {'liveness': 'unknown'})
                    with _lock(root / 'evaluation.lock'):
                        latest = _read_json(root / 'evaluation_state.json')
                        usage = dict(latest.get('usage') or {})
                        usage.update(gpu_hours=None, status='unknown')
                        result = {'valid': False, 'error': 'Frozen evaluation release is unconfirmed', 'usage': usage}
                        _write_json(root / 'evaluation_result.json', result)
                        _write_json(root / 'evaluation_state.json', {**latest, 'status': 'recovery_required', **result})
                elif resources_were_live:
                    with _lock(root / 'evaluation.lock'):
                        latest = _read_json(root / 'evaluation_state.json')
                        result = _read_json(root / 'evaluation_result.json', {})
                        usage = _usage(latest['execution_limits'], time.monotonic() - latest['started_monotonic'])
                        _write_json(root / 'evaluation_result.json', {**result, 'usage': usage})
                        _write_json(root / 'evaluation_state.json', {**latest, 'finished_at': _now(), 'usage': usage})
                return
            owner = _identity_liveness(state['owner_identity'])
            if owner == 'unknown':
                _write_json(root / 'evaluation_health.json', {'liveness': 'unknown'})
                return
            deadline = state.get('deadline_monotonic')
            now = time.monotonic()
            if deadline is not None and now >= state['soft_deadline_monotonic']:
                if not (root / 'stop_request.json').exists():
                    _write_json(root / 'stop_request.json', {'reason': 'Frozen evaluation job walltime exceeded', 'requested_at': _now()})
            if owner == 'dead' or deadline is not None and now >= deadline:
                settle = min(now + .2, deadline) if deadline is not None else now + .2
                while _read_json(root / 'process.json', {}).get('pending') and time.monotonic() < settle:
                    time.sleep(.01)
                cleaned = _cleanup_recorded_process(root)
                if not cleaned:
                    _write_json(root / 'evaluation_health.json', {'liveness': 'unknown'})
                    return
                stopped = time.monotonic()
                _write_json(root / 'evaluation_health.json', {'execution_stopped_monotonic': stopped, 'liveness': 'dead'})
                if owner == 'dead':
                    with _lock(root / 'evaluation.lock'):
                        latest = _read_json(root / 'evaluation_state.json')
                        if latest['status'] not in TERMINAL:
                            result = {'valid': False, 'error': 'Frozen evaluation caller exited before completion',
                                      'usage': _usage(state['execution_limits'], stopped - state['started_monotonic'])}
                            _write_json(root / 'evaluation_result.json', result)
                            _write_json(root / 'evaluation_state.json', {**latest, 'status': 'failed',
                                        'finished_at': _now(), **result})
                return
            time.sleep(.02)


def _evaluation_usage(root, limits, started):
    stopped = _read_json(root / 'evaluation_health.json', {}).get('execution_stopped_monotonic', time.monotonic())
    return _usage(limits, stopped - started)


def evaluate_frozen_checkpoint(config, checkpoint, split_spec, output_dir, *, execution_limits=None):
    """Evaluate one frozen checkpoint on an explicit split without training.

    The caller owns authorization. A persisted evaluation is never automatically
    retried; successful repeated calls return its existing result.
    """
    cfg = normalize_config(config)
    limits = _validate_limits(execution_limits)
    if not isinstance(checkpoint, dict) or not checkpoint.get('best_checkpoint') or not checkpoint.get('config_dir'):
        raise ValueError('Frozen checkpoint requires best_checkpoint and config_dir')
    if not Path(checkpoint['best_checkpoint']).is_file():
        raise ValueError('Frozen checkpoint is missing')
    if not isinstance(split_spec, dict):
        raise ValueError('Explicit evaluation split is required')
    split_cfg = normalize_config({**cfg, 'test': dict(split_spec)})
    details = split_cfg['test']
    count = expected_count(details)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = evaluation_command(split_cfg, root, split='test')
    command[command.index('--result-dir') + 1] = str(root)
    command = _resolve_evaluation(command, checkpoint)
    frozen = {'config': cfg, 'checkpoint': {key: checkpoint[key] for key in ('best_checkpoint', 'config_dir')},
              'split': details, 'protocol': f'P{cfg["eval_size"]}'}
    with _lock(root / 'evaluation.lock'):
        existing = _read_json(root / 'evaluation_state.json')
        if existing:
            if existing['frozen'] != frozen:
                raise ValueError('Evaluation directory already belongs to another frozen evaluation')
            if existing['status'] == 'completed' and existing.get('valid'):
                if not _cleanup_recorded_process(root):
                    raise RuntimeError('Frozen evaluation release is unconfirmed; refusing reuse')
                return _read_json(root / 'evaluation_result.json')
            raise RuntimeError('Frozen evaluation already recorded; automatic retry is forbidden')
        started = time.monotonic()
        deadline = None if 'job_walltime_seconds' not in limits else started + limits['job_walltime_seconds']
        soft_deadline = None if deadline is None else deadline - min(1., limits['job_walltime_seconds'] * .1)
        state = {'status': 'running', 'owner_identity': _process_identity(os.getpid()),
                 'created_at': _now(), 'started_monotonic': started, 'deadline_monotonic': deadline,
                 'soft_deadline_monotonic': soft_deadline, 'execution_limits': limits, 'frozen': frozen}
        _write_json(root / 'commands.json', {'evaluation': command})
        _write_json(root / 'execution_limits.json', limits)
        _write_json(root / 'evaluation_state.json', state)
    previous_context = getattr(_EXECUTION, 'context', None)
    _EXECUTION.context = {'root': root, 'deadline': deadline, 'soft_deadline': soft_deadline}
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [source_root, env.get('PYTHONPATH', '')]))
    try:
        with (root / 'watchdog.log').open('w', encoding='utf-8') as log:
            watcher = subprocess.Popen([sys.executable, '-c',
                'import sys; from tm_research.runner import _evaluation_watchdog; _evaluation_watchdog(sys.argv[1])', str(root)],
                cwd=cfg['repo_dir'], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        _write_json(root / 'evaluation_watchdog.json', {'identity': _process_identity(watcher.pid)})
        rc = _execute(command, root / 'evaluation.log', cfg['repo_dir'])
        if rc:
            raise RuntimeError(f'Frozen checkpoint evaluation exited with status {rc}')
        metrics = _read_json(root / 'metrics.json')
        score = validate_metrics(metrics, count, cfg['eval_size'])
        _check_execution()
        result = {'valid': True, 'score': score, 'metrics': metrics, 'expected_count': count,
                  'protocol': f'P{cfg["eval_size"]}', 'artifacts': {'metrics': str(root / 'metrics.json'),
                  'per_image_csv': metrics.get('per_image_csv'), **checkpoint},
                  'usage': _evaluation_usage(root, limits, started)}
        with _lock(root / 'evaluation.lock'):
            _write_json(root / 'evaluation_result.json', result)
            _write_json(root / 'evaluation_state.json', {**state, 'status': 'completed', 'finished_at': _now(), **result})
        return result
    except Exception as exc:
        result = {'valid': False, 'error': str(exc), 'usage': _evaluation_usage(root, limits, started)}
        unresolved = isinstance(exc, _UnresolvedExecution)
        if unresolved:
            result['usage'].update(gpu_hours=None, status='unknown')
        with _lock(root / 'evaluation.lock'):
            _write_json(root / 'evaluation_result.json', result)
            _write_json(root / 'evaluation_state.json', {**state, 'status': 'recovery_required' if unresolved else 'failed', 'finished_at': _now(), **result})
        raise
    finally:
        _EXECUTION.context = previous_context
