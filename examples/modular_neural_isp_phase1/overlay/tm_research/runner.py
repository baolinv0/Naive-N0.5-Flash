"""Run fixed-model recipes and independently reload every selected checkpoint on DEV."""

import json
import math
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
    temporary = path.with_suffix(path.suffix + '.tmp')
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


def run_baseline(config):
    """Start one recipe worker. DEV reload is mandatory, TEST is reserved for finalize."""
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
    train, dev = commands_for_run(cfg, run_dir)
    # The run has no reason to know held-out TEST paths.
    cfg.pop('test', None)
    _write_json(run_dir / 'config.json', cfg)
    _write_json(run_dir / 'commands.json', {'train': train, 'dev': dev})
    _write_json(run_dir / 'state.json', {'run_id': run_id, 'status': 'queued', 'result_status': 'pending',
        'valid': False, 'reload_succeeded': False, 'created_at': _now(), 'run_dir': str(run_dir),
        'recipe': validate_recipe({k: cfg[k] for k in RECIPE_DEFAULTS if k in cfg}),
        'logs': {name: str(run_dir / f'{name}.log') for name in ('worker', 'train', 'dev')}})
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [source_root, env.get('PYTHONPATH', '')]))
    try:
        with (run_dir / 'worker.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-m', 'tm_research.cli', '_worker', '--run-dir', str(run_dir)],
                cwd=cfg['repo_dir'], env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as exc:
        _state(run_dir, status='failed', result_status='invalid', error=f'Could not start worker: {exc}', finished_at=_now())
        raise RuntimeError(f'Could not start worker for {run_id}: {exc}') from exc
    (run_dir / 'worker.pid').write_text(str(child.pid) + '\n', encoding='utf-8')
    return run_id


def _execute(command, log_path, cwd):
    with Path(log_path).open('w', encoding='utf-8') as log:
        return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, check=False).returncode


def _resolve_evaluation(command, train_metrics):
    return [train_metrics['best_checkpoint'] if v == '<best_checkpoint>' else
            train_metrics['config_dir'] if v == '<train_config_dir>' else v for v in command]


def _worker(run_dir):
    run_dir = Path(run_dir).resolve()
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    commands = json.loads((run_dir / 'commands.json').read_text(encoding='utf-8'))
    previous = json.loads((run_dir / 'state.json').read_text(encoding='utf-8'))
    if previous['status'] in ('completed', 'failed'):
        return previous
    state = _state(run_dir, status='training', started_at=_now(), worker_pid=os.getpid())
    execution_failed = False
    rc = None
    try:
        rc = _execute(commands['train'], state['logs']['train'], cfg['repo_dir'])
        if rc:
            execution_failed = True
            raise RuntimeError(f'train.py exited with status {rc}')
        train_metrics = json.loads((run_dir / 'train' / 'metrics.json').read_text(encoding='utf-8'))
        checkpoint = Path(train_metrics['best_checkpoint'])
        if not checkpoint.is_file():
            raise ValueError(f'Selected checkpoint is missing: {checkpoint}')
        count = expected_count(cfg['validation'])
        dev = _resolve_evaluation(commands['dev'], train_metrics)
        _write_json(run_dir / 'commands.json', {'train': commands['train'], 'dev': dev})
        _state(run_dir, status='evaluating_dev', train_metrics=train_metrics, expected_count=count)
        rc = _execute(dev, state['logs']['dev'], cfg['repo_dir'])
        if rc:
            execution_failed = True
            raise RuntimeError(f'Independent DEV test.py reload exited with status {rc}')
        _state(run_dir, reload_succeeded=True)
        metrics_path = run_dir / 'dev' / 'metrics.json'
        dev_metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        score = validate_metrics(dev_metrics, count, cfg['eval_size'])
        final = _state(run_dir, status='completed', result_status='valid', valid=True,
            finished_at=_now(), dev_psnr=score, dev_metrics=dev_metrics,
            artifacts={'best_checkpoint': str(checkpoint), 'config_dir': train_metrics['config_dir'],
                       'train_metrics': str(run_dir / 'train' / 'metrics.json'),
                       'dev_metrics': str(metrics_path), 'per_image_csv': dev_metrics.get('per_image_csv')})
    except Exception as exc:
        final = _state(run_dir, status='failed' if execution_failed else 'completed',
            result_status='invalid', valid=False, finished_at=_now(), error=str(exc), returncode=rc)
    with (Path(cfg['runs_dir']) / 'results.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(final) + '\n')
    return final


def get_result(run_id, runs_dir='runs'):
    if not run_id.startswith('exp_') or not run_id[4:].isdigit():
        raise ValueError('Expected an exp_NNN run ID')
    path = Path(runs_dir).resolve() / run_id / 'state.json'
    if not path.exists():
        raise FileNotFoundError(f'Unknown run {run_id} in {runs_dir}')
    return json.loads(path.read_text(encoding='utf-8'))


def wait_for_result(run_id, runs_dir='runs', timeout=None, poll_interval=0.2):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        result = get_result(run_id, runs_dir)
        if result['status'] in ('completed', 'failed'):
            return result
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f'Run {run_id} is still {result["status"]}')
        time.sleep(poll_interval)
