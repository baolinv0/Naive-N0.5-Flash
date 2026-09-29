"""Process runner for the original photofinishing training and test entrypoints."""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _absolute(value, base):
    return str((base / value).resolve())


def normalize_config(config, base=None):
    """Resolve file paths against a YAML file directory, or cwd for direct dicts."""
    base = Path(base or Path.cwd()).resolve()
    cfg = dict(config)
    for key in ('repo_dir', 'runs_dir', 'init_checkpoint', 'init_config_dir'):
        if cfg.get(key) is not None:
            cfg[key] = _absolute(cfg[key], base)
    if not cfg.get('repo_dir'):
        raise ValueError('repo_dir is required')
    cfg.setdefault('runs_dir', str(Path(cfg['repo_dir']) / 'runs'))
    interpreter = cfg.get('python', sys.executable)
    if '/' not in interpreter and '\\' not in interpreter:
        interpreter = shutil.which(interpreter) or interpreter
    cfg['python'] = os.path.abspath(os.path.join(base, interpreter))  # Keep venv interpreter symlink intact.
    for split in ('train', 'validation', 'test'):
        if split not in cfg or cfg[split] is None:
            if split == 'test':
                continue
            raise ValueError(f'{split} split is required')
        details = dict(cfg[split])
        for key in ('input_dir', 'gt_dir'):
            if not details.get(key):
                raise ValueError(f'{split}.{key} is required')
        for key in ('input_dir', 'gt_dir', 'metadata_dir'):
            if details.get(key) is not None:
                details[key] = _absolute(details[key], base)
        cfg[split] = details
    allowed = {'repo_dir', 'python', 'runs_dir', 'train', 'validation', 'test', 'epochs',
               'batch_size', 'in_size', 'validation_frequency', 'num_workers',
               'init_checkpoint', 'init_config_dir', 'seed'}
    extra = set(cfg) - allowed
    if extra:
        raise ValueError(f'Unsupported config fields: {sorted(extra)}')
    for field in ('epochs', 'batch_size', 'in_size', 'validation_frequency', 'num_workers'):
        if field in cfg and (type(cfg[field]) is not int or cfg[field] < (0 if field == 'num_workers' else 1)):
            raise ValueError(f'{field} must be a positive integer' if field != 'num_workers' else 'num_workers must be a nonnegative integer')
    return cfg


def load_config(path):
    """Read YAML and resolve relative paths relative to that YAML document."""
    import yaml
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return normalize_config(cfg, path.parent)


def commands_for_run(config, run_dir):
    """Return upstream commands; unspecified knobs retain the original CLI defaults."""
    cfg = normalize_config(config)
    run_dir = Path(run_dir).resolve()
    script_dir = Path(cfg['repo_dir']) / 'photofinishing'
    train = [cfg['python'], str(script_dir / 'train.py'),
             '--in-training-dir', cfg['train']['input_dir'],
             '--gt-training-dir', cfg['train']['gt_dir'],
             '--in-validation-dir', cfg['validation']['input_dir'],
             '--gt-validation-dir', cfg['validation']['gt_dir'],
             '--output-dir', str(run_dir / 'train'), '--exp-name', run_dir.name]
    if cfg.get('init_checkpoint'):
        config_dir = Path(cfg.get('init_config_dir') or (script_dir / 'config'))
        config_path = config_dir / (Path(cfg['init_checkpoint']).stem + '.json')
        checkpoint_config = json.loads(config_path.read_text(encoding='utf-8'))
        if checkpoint_config['use_3d_lut']:
            train.append('--use-3d-lut')
    for split, label in (('train', 'training'), ('validation', 'validation')):
        if cfg[split].get('metadata_dir'):
            train += [f'--data-{label}-dir', cfg[split]['metadata_dir']]
    for key, flag in (('epochs', '--epochs'), ('batch_size', '--batch-size'),
                      ('in_size', '--in-size'), ('validation_frequency', '--validation-frequency'),
                      ('num_workers', '--num-workers'), ('seed', '--seed'),
                      ('init_checkpoint', '--load'), ('init_config_dir', '--load-config-dir')):
        if cfg.get(key) is not None:
            train += [flag, str(cfg[key])]
    test = None
    if cfg.get('test'):
        # The checkpoint is selected from train/metrics.json after training succeeds.
        test = [cfg['python'], str(script_dir / 'test.py'), '--model-path', '<best_checkpoint>',
                '--config-dir', '<train_config_dir>',
                '--in-testing-dir', cfg['test']['input_dir'],
                '--gt-testing-dir', cfg['test']['gt_dir'],
                '--result-dir', str(run_dir / 'test')]
        if cfg['test'].get('metadata_dir'):
            test += ['--data-testing-dir', cfg['test']['metadata_dir']]
    return train, test


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _state(run_dir, **changes):
    path = Path(run_dir) / 'state.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data.update(changes)
    _write_json(path, data)
    return data


def run_baseline(config):
    """Start a background worker for one upstream baseline; return its run ID."""
    cfg = normalize_config(config)
    runs_dir = Path(cfg['runs_dir'])
    runs_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1000000):
        run_id = f'exp_{number:03d}'
        run_dir = runs_dir / run_id
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('No free experiment ID')
    train, test = commands_for_run(cfg, run_dir)
    _write_json(run_dir / 'config.json', cfg)
    _write_json(run_dir / 'commands.json', {'train': train, 'test': test})
    _write_json(run_dir / 'state.json', {'run_id': run_id, 'status': 'queued',
                                         'created_at': _now(), 'run_dir': str(run_dir),
                                         'logs': {'worker': str(run_dir / 'worker.log'),
                                                  'train': str(run_dir / 'train.log'),
                                                  'test': str(run_dir / 'test.log')}})
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [source_root, env.get('PYTHONPATH', '')]))
    try:
        with (run_dir / 'worker.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-m', 'tm_research.cli', '_worker',
                                      '--run-dir', str(run_dir)], cwd=cfg['repo_dir'], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
    except OSError as exc:
        _state(run_dir, status='failed', error=f'Could not start worker: {exc}', finished_at=_now())
        raise RuntimeError(f'Could not start worker for {run_id}: {exc}') from exc
    (run_dir / 'worker.pid').write_text(str(child.pid) + '\n', encoding='utf-8')
    return run_id


def _execute(command, log_path, cwd):
    with Path(log_path).open('w', encoding='utf-8') as log:
        return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, check=False).returncode


def _worker(run_dir):
    run_dir = Path(run_dir).resolve()
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    commands = json.loads((run_dir / 'commands.json').read_text(encoding='utf-8'))
    state = _state(run_dir, status='training', started_at=_now(), worker_pid=os.getpid())
    try:
        rc = _execute(commands['train'], state['logs']['train'], cfg['repo_dir'])
        if rc:
            raise RuntimeError(f'train.py exited with status {rc}')
        train_metrics = json.loads((run_dir / 'train' / 'metrics.json').read_text(encoding='utf-8'))
        result = {'train_metrics': train_metrics,
                  'artifacts': {'best_checkpoint': train_metrics['best_checkpoint'],
                                'train_metrics': str(run_dir / 'train' / 'metrics.json')}}
        if commands['test']:
            test = list(commands['test'])
            test[test.index('<best_checkpoint>')] = train_metrics['best_checkpoint']
            test[test.index('<train_config_dir>')] = train_metrics['config_dir']
            _write_json(run_dir / 'commands.json', {'train': commands['train'], 'test': test})
            _state(run_dir, status='testing')
            rc = _execute(test, state['logs']['test'], cfg['repo_dir'])
            if rc:
                raise RuntimeError(f'test.py exited with status {rc}')
            result['test_metrics'] = json.loads((run_dir / 'test' / 'metrics.json').read_text(encoding='utf-8'))
            result['artifacts']['test_metrics'] = str(run_dir / 'test' / 'metrics.json')
            result['artifacts']['per_image_csv'] = result['test_metrics'].get('per_image_csv', str(run_dir / 'test' / 'per_image.csv'))
            if result['test_metrics'].get('images_dir'):
                result['artifacts']['images_dir'] = result['test_metrics']['images_dir']
        final = _state(run_dir, status='completed', finished_at=_now(), **result)
    except Exception as exc:
        final = _state(run_dir, status='failed', finished_at=_now(), error=str(exc),
                       returncode=rc if 'rc' in locals() else None)
    # One append per terminal state. The per-run state is authoritative.
    with (Path(cfg['runs_dir']) / 'results.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(final) + '\n')
    return final


def get_result(run_id, runs_dir='runs'):
    """Return persisted state, logs and metrics for one run."""
    if not run_id.startswith('exp_') or not run_id[4:].isdigit():
        raise ValueError('Expected an exp_NNN run ID')
    path = Path(runs_dir).resolve() / run_id / 'state.json'
    if not path.exists():
        raise FileNotFoundError(f'Unknown run {run_id} in {runs_dir}')
    return json.loads(path.read_text(encoding='utf-8'))


def wait_for_result(run_id, runs_dir='runs', timeout=None, poll_interval=0.2):
    """Block until the background process has completed or failed."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        result = get_result(run_id, runs_dir)
        if result['status'] in ('completed', 'failed'):
            return result
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f'Run {run_id} is still {result["status"]}')
        time.sleep(poll_interval)
