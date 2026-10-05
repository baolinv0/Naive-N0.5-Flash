import json
import subprocess
import sys
from pathlib import Path

import pytest

from tm_research.runner import (_worker, commands_for_run, get_result, load_config,
                                normalize_config, run_baseline, wait_for_result)


def fixture_repo(tmp_path, mode='ok'):
    repo = tmp_path / 'upstream'
    scripts = repo / 'photofinishing'
    scripts.mkdir(parents=True)
    (scripts / 'train.py').write_text('''import argparse, json, pathlib, sys
p=argparse.ArgumentParser(); p.add_argument('--output-dir'); p.add_argument('--exp-name')
p.add_argument('--loss-family',default='original'); p.add_argument('--optimizer',default='adam')
p.add_argument('--learning-rate',type=float,default=0.0001); p.add_argument('--weight-decay',type=float,default=0.0000001)
a,_=p.parse_known_args(); mode=%r
r=pathlib.Path(a.output_dir); (r/'models').mkdir(parents=True); (r/'config').mkdir()
if mode=='train_fail': sys.exit(7)
recipe={key:getattr(a,key) for key in ('loss_family','optimizer','learning_rate','weight_decay')}
model=r/'models'/('photofinishing_'+a.exp_name+'-best.pth')
if mode!='missing_checkpoint': model.write_text(json.dumps(recipe))
(r/'config'/(model.stem+'.json')).write_text('{}')
(r/'metrics.json').write_text(json.dumps({'best_checkpoint':str(model),'config_dir':str(r/'config'),'mean_psnr':99.0,'recipe':recipe}))
print('train fixture complete')
''' % mode)
    (scripts / 'test.py').write_text('''import argparse, json, pathlib, sys
p=argparse.ArgumentParser(); p.add_argument('--model-path'); p.add_argument('--config-dir'); p.add_argument('--result-dir')
p.add_argument('--eval-size',type=int); p.add_argument('--in-testing-dir'); a,_=p.parse_known_args(); mode=%r
assert pathlib.Path(a.model_path).is_file(); assert (pathlib.Path(a.config_dir)/(pathlib.Path(a.model_path).stem+'.json')).is_file()
if mode=='reload_fail': sys.exit(8)
recipe=json.loads(pathlib.Path(a.model_path).read_text())
score={'original':20.,'mse':22.,'l1':19.}[recipe['loss_family']]+(2 if recipe['optimizer']=='adamw' else 0)+(1 if recipe['learning_rate']<0.0001 else 0)
if mode=='nonfinite': score=float('nan')
count=len(list(pathlib.Path(a.in_testing_dir).glob('*.png')))
if mode=='wrong_count': count+=1
if mode=='zero_count': count=0
r=pathlib.Path(a.result_dir); r.mkdir(exist_ok=True); (r/'per_image.csv').write_text('image,psnr\\na,20\\n')
(r/'metrics.json').write_text(json.dumps({'mean_psnr':score,'mean_per_image_psnr':score,'num_images':count,'protocol':'P'+str(a.eval_size),'eval_size':a.eval_size,'per_image_csv':str(r/'per_image.csv'),'recipe':recipe}))
print('independent reload fixture complete')
''' % mode)
    return repo


def config(tmp_path, mode='ok'):
    repo = fixture_repo(tmp_path, mode)
    dirs = {}
    for split in ('train', 'validation', 'test'):
        dirs[split] = {}
        for name in ('input_dir', 'gt_dir', 'metadata_dir'):
            directory = tmp_path / split / name
            directory.mkdir(parents=True)
            dirs[split][name] = str(directory)
        (Path(dirs[split]['input_dir']) / 'a.png').write_text('fixture')
    return {'repo_dir': str(repo), 'python': sys.executable, 'runs_dir': str(tmp_path / 'runs'),
            **dirs, 'epochs': 1, 'batch_size': 2, 'num_workers': 0}


def test_config_paths_and_mandatory_dev_protocol(tmp_path):
    fixture_repo(tmp_path)
    interpreter = tmp_path / 'venv-python'
    interpreter.symlink_to(sys.executable)
    path = tmp_path / 'baseline.yaml'
    path.write_text('repo_dir: upstream\npython: venv-python\nruns_dir: runs\ntrain:\n  input_dir: train/in\n  gt_dir: train/gt\nvalidation:\n  input_dir: val/in\n  gt_dir: val/gt\nepochs: 2\n')
    cfg = load_config(path)
    train, dev = commands_for_run(cfg, tmp_path / 'runs' / 'exp_001')
    assert cfg['python'] == str(interpreter)
    assert '--batch-size' not in train
    assert dev[dev.index('--eval-size') + 1] == '512'
    assert dev[dev.index('--in-testing-dir') + 1] == str(tmp_path / 'val' / 'in')
    assert train[train.index('--output-dir') + 1] == str(tmp_path / 'runs' / 'exp_001' / 'train')


def test_background_process_always_independently_reloads_dev_never_test(tmp_path):
    cfg = config(tmp_path)
    first = run_baseline(cfg)
    second = run_baseline(cfg)
    assert (first, second) == ('exp_001', 'exp_002')
    result = wait_for_result(first, cfg['runs_dir'], timeout=15)
    assert result['status'] == 'completed', result
    assert result['valid'] and result['result_status'] == 'valid' and result['reload_succeeded']
    assert result['train_metrics']['mean_psnr'] == 99
    assert result['dev_psnr'] == 20
    assert result['dev_metrics']['num_images'] == result['expected_count'] == 1
    run_dir = Path(result['run_dir'])
    assert 'test' not in json.loads((run_dir / 'commands.json').read_text())
    assert 'test' not in json.loads((run_dir / 'config.json').read_text())
    assert not (run_dir / 'test').exists()
    assert Path(result['artifacts']['per_image_csv']).is_file()
    assert wait_for_result(second, cfg['runs_dir'], timeout=15)['valid']
    # A completed worker invocation is idempotent; incomplete artifacts are never adopted as results.
    before = (Path(cfg['runs_dir']) / 'results.jsonl').read_text()
    assert _worker(run_dir)['dev_psnr'] == 20
    assert (Path(cfg['runs_dir']) / 'results.jsonl').read_text() == before


@pytest.mark.parametrize('mode,status', [('train_fail', 'failed'), ('reload_fail', 'failed'),
    ('missing_checkpoint', 'completed'), ('nonfinite', 'completed'), ('wrong_count', 'completed'), ('zero_count', 'completed')])
def test_invalid_results_never_become_valid(tmp_path, mode, status):
    cfg = config(tmp_path, mode)
    run_id = run_baseline(cfg)
    result = wait_for_result(run_id, cfg['runs_dir'], timeout=15)
    assert result['status'] == status
    assert not result['valid'] and result['result_status'] == 'invalid'
    assert result['error']
    assert not (Path(result['run_dir']) / 'test').exists()


def test_empty_dev_input_is_invalid_even_after_successful_train(tmp_path):
    cfg = config(tmp_path)
    (Path(cfg['validation']['input_dir']) / 'a.png').unlink()
    result = wait_for_result(run_baseline(cfg), cfg['runs_dir'], timeout=15)
    assert not result['valid'] and 'at least one' in result['error']


def test_cli_wait_report_recipe_flags_and_checkpoint_setting(tmp_path):
    cfg = config(tmp_path)
    cfg.update(loss_family='mse', optimizer='adamw', learning_rate=0.002, weight_decay=0.0)
    checkpoint = tmp_path / 'initialized.pth'
    checkpoint.write_text('fixture')
    config_dir = tmp_path / 'model_configs'
    config_dir.mkdir()
    (config_dir / 'initialized.json').write_text('{"use_3d_lut": true}')
    cfg.update(init_checkpoint=str(checkpoint), init_config_dir=str(config_dir))
    train, _ = commands_for_run(cfg, Path(cfg['runs_dir']) / 'exp_001')
    assert '--use-3d-lut' in train
    for flag, value in (('--loss-family', 'mse'), ('--optimizer', 'adamw'), ('--learning-rate', '0.002'), ('--weight-decay', '0.0')):
        assert train[train.index(flag) + 1] == value
    run_id = run_baseline(cfg)
    base = [sys.executable, '-m', 'tm_research.cli']
    wait = subprocess.run(base + ['wait', '--run', run_id, '--runs-dir', cfg['runs_dir'], '--timeout', '15'],
                          capture_output=True, text=True, check=True)
    assert json.loads(wait.stdout)['valid']
    report = subprocess.run(base + ['report', '--run', run_id, '--runs-dir', cfg['runs_dir']],
                            capture_output=True, text=True, check=True)
    summary = json.loads(report.stdout)
    assert summary['dev_psnr'] == 24 and 'test_metrics' not in summary


@pytest.mark.parametrize('changes', [{'loss_family': 'new_loss'}, {'optimizer': 'sgd'},
    {'learning_rate': 0}, {'learning_rate': float('nan')}, {'weight_decay': -1}, {'eval_size': 0}])
def test_reject_unsupported_recipe_or_budget(tmp_path, changes):
    cfg = config(tmp_path)
    with pytest.raises(ValueError):
        normalize_config({**cfg, **changes})


def test_prepare_separates_registration_from_execution_and_start_is_concurrent_idempotent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg, execution_limits={'allocated_gpus': 2, 'job_walltime_seconds': 10})
    root = Path(cfg['runs_dir']) / run_id
    assert runner.inspect_run(run_id, cfg['runs_dir'])['liveness'] == 'not_started'
    assert not (root / 'worker.pid').exists() and not (root / 'train').exists()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: runner.start_prepared_run(run_id, cfg['runs_dir']), range(4))) == [run_id] * 4
    result = runner.wait_for_result(run_id, cfg['runs_dir'], timeout=15)
    assert result['valid']
    assert result['usage']['allocated_gpus'] == 2
    assert result['usage']['gpu_hours'] == pytest.approx(2 * result['usage']['elapsed_seconds'] / 3600)
    assert len((Path(cfg['runs_dir']) / 'results.jsonl').read_text().splitlines()) == 1


def test_liveness_identity_startup_window_and_dead_convergence(tmp_path):
    import os, socket
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg)
    root = Path(cfg['runs_dir']) / run_id
    # Startup may still have queued state while a launcher already owns a live process.
    identity = runner._process_identity(os.getpid())
    runner._write_json(root / 'launch.json', {'identity': identity, 'started_at': runner._now()})
    assert runner.inspect_run(run_id, cfg['runs_dir'])['liveness'] == 'live'
    runner.start_prepared_run(run_id, cfg['runs_dir'])
    assert not (root / 'train').exists()
    identity['hostname'] = socket.gethostname() + '-other'
    runner._write_json(root / 'launch.json', {'identity': identity})
    assert runner.inspect_run(run_id, cfg['runs_dir'])['liveness'] == 'unknown'
    with pytest.raises(RuntimeError, match='unknown'):
        runner.start_prepared_run(run_id, cfg['runs_dir'])
    identity['hostname'] = socket.gethostname()
    identity['start_ticks'] = 'reused-pid'
    runner._write_json(root / 'launch.json', {'identity': identity})
    assert runner.inspect_run(run_id, cfg['runs_dir'])['liveness'] == 'dead'
    result = runner.get_result(run_id, cfg['runs_dir'])
    assert result['status'] == 'failed' and not result['valid']
    assert runner.get_result(run_id, cfg['runs_dir']) == result
    assert len((Path(cfg['runs_dir']) / 'results.jsonl').read_text().splitlines()) == 1


def test_wait_timeout_is_not_job_timeout_and_stop_terminates_child_tree(tmp_path):
    import os, time
    from tm_research import runner
    cfg = config(tmp_path)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'train.py'
    script.write_text("import pathlib, subprocess, sys, time, signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\np=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])\npathlib.Path('child.pid').write_text(str(p.pid))\ntime.sleep(60)\n")
    run_id = runner.run_baseline(cfg, execution_limits={'allocated_gpus': 3, 'job_walltime_seconds': 15})
    deadline = time.monotonic() + 10
    while not (Path(cfg['repo_dir']) / 'child.pid').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    with pytest.raises(TimeoutError):
        runner.wait_for_result(run_id, cfg['runs_dir'], timeout=.02)
    assert runner.inspect_run(run_id, cfg['runs_dir'])['liveness'] == 'live'
    request = runner.request_run_stop(run_id, cfg['runs_dir'], reason='fixture stop')
    assert request['stop_requested'] and not request['stopped']
    result = runner.wait_for_result(run_id, cfg['runs_dir'], timeout=10)
    assert result['execution_status'] == 'interrupted' and not result['valid']
    pid = int((Path(cfg['repo_dir']) / 'child.pid').read_text())
    assert runner._process_identity(pid) is None  # exited or zombie, not a live descendant
    assert result['usage']['gpu_hours'] > 0


def test_job_walltime_bounds_train_and_dev_together_and_failed_work_counts(tmp_path):
    from tm_research import runner
    cfg = config(tmp_path)
    for name in ('train.py', 'test.py'):
        path = Path(cfg['repo_dir']) / 'photofinishing' / name
        path.write_text('import time\ntime.sleep(.35)\n' + path.read_text())
    result = runner.wait_for_result(runner.run_baseline(cfg, execution_limits={
        'allocated_gpus': 1, 'job_walltime_seconds': .55}), cfg['runs_dir'], timeout=10)
    assert result['status'] == 'failed' and result['execution_status'] == 'interrupted'
    assert 'walltime' in result['error']
    assert result['usage']['gpu_hours'] > 0
    assert not result['valid']


def test_frozen_eval_uses_only_explicit_split_and_diagnostics_are_internal(tmp_path):
    from tm_research import runner
    cfg = config(tmp_path)
    baseline = runner.wait_for_result(runner.run_baseline(cfg), cfg['runs_dir'], timeout=15)
    before = json.dumps(cfg, sort_keys=True)
    target = tmp_path / 'frozen'
    result = runner.evaluate_frozen_checkpoint(cfg, baseline['artifacts'], cfg['test'], target)
    assert result['valid'] and result['score'] == 20
    assert json.dumps(cfg, sort_keys=True) == before
    command = json.loads((target / 'commands.json').read_text())['evaluation']
    assert command[command.index('--in-testing-dir') + 1] == cfg['test']['input_dir']
    assert '--output-dir' not in command and not (target / 'train').exists()
    profile = tmp_path / 'profile.json'; profile.write_text('{}')
    prepared = runner.prepare_run(cfg, observation_config={'profile_ref': str(profile)})
    root = Path(cfg['runs_dir']) / prepared
    assert '--diagnostics-profile' in json.loads((root / 'commands.json').read_text())['dev']
    assert 'observation_config' not in json.loads((root / 'config.json').read_text())
    with pytest.raises(ValueError):
        runner.normalize_config({**cfg, 'diagnostics_profile': str(profile)})


@pytest.mark.parametrize('limits', [{'allocated_gpus': -1}, {'allocated_gpus': 1.5}, {'job_walltime_seconds': float('nan')}, {'job_walltime_seconds': 0}])
def test_reject_invalid_internal_limits(tmp_path, limits):
    from tm_research import runner
    with pytest.raises(ValueError):
        runner.prepare_run(config(tmp_path), execution_limits=limits)


def test_dead_worker_cleanup_kills_orphan_group_and_charges_launch_window(tmp_path):
    import os, signal, time
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg, execution_limits={'allocated_gpus': 2})
    root = Path(cfg['runs_dir']) / run_id
    orphan = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        runner._write_json(root / 'process.json', {'identity': runner._process_identity(orphan.pid), 'pgid': orphan.pid})
        dead = runner._process_identity(os.getpid()); dead['start_ticks'] = 'not-this-process'
        runner._write_json(root / 'launch.json', {'identity': dead, 'started_at': runner._now()})
        time.sleep(.03)
        result = runner.get_result(run_id, cfg['runs_dir'])
        orphan.wait(timeout=3)
        assert result['usage']['gpu_hours'] > 0
        assert result['execution_status'] == 'interrupted'
    finally:
        if orphan.poll() is None:
            os.killpg(orphan.pid, signal.SIGKILL); orphan.wait()


def test_terminal_state_recovers_missing_results_append_and_unknown_allocation(tmp_path):
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg)
    root = Path(cfg['runs_dir']) / run_id
    result = runner._worker(root)
    assert result['valid'] and result['usage']['gpu_hours'] is None
    (root / 'result_recorded.json').unlink()
    (Path(cfg['runs_dir']) / 'results.jsonl').unlink()
    assert runner.get_result(run_id, cfg['runs_dir']) == result
    assert len((Path(cfg['runs_dir']) / 'results.jsonl').read_text().splitlines()) == 1
    runner.get_result(run_id, cfg['runs_dir'])
    assert len((Path(cfg['runs_dir']) / 'results.jsonl').read_text().splitlines()) == 1


def test_frozen_checkpoint_rejects_invalid_reload_protocol_and_count(tmp_path):
    from tm_research import runner
    cfg = config(tmp_path)
    result = runner.wait_for_result(runner.run_baseline(cfg), cfg['runs_dir'], timeout=15)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    script.write_text(script.read_text().replace("'protocol':'P'+str(a.eval_size)", "'protocol':'wrong'"))
    with pytest.raises(ValueError, match='protocol'):
        runner.evaluate_frozen_checkpoint(cfg, result['artifacts'], cfg['test'], tmp_path / 'invalid-frozen')


def test_frozen_eval_enforces_and_records_its_own_resource_limit(tmp_path):
    from tm_research import runner
    cfg = config(tmp_path)
    baseline = runner.wait_for_result(runner.run_baseline(cfg), cfg['runs_dir'], timeout=15)
    path = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    path.write_text('import time\ntime.sleep(60)\n' + path.read_text())
    target = tmp_path / 'bounded-frozen'
    with pytest.raises(RuntimeError, match='walltime'):
        runner.evaluate_frozen_checkpoint(cfg, baseline['artifacts'], cfg['test'], target,
            execution_limits={'allocated_gpus': 2, 'job_walltime_seconds': .1})
    result = json.loads((target / 'evaluation_result.json').read_text())
    assert not result['valid'] and result['usage']['gpu_hours'] > 0


def test_supervisor_cleans_child_after_worker_sigkill_without_controller_poll(tmp_path):
    import os, signal, time
    from tm_research import runner
    cfg = config(tmp_path)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'train.py'
    script.write_text("import pathlib,time,signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\npathlib.Path('running.pid').write_text(str(__import__('os').getpid()))\ntime.sleep(60)\n")
    run_id = runner.run_baseline(cfg, execution_limits={'allocated_gpus': 1, 'job_walltime_seconds': 5})
    root = Path(cfg['runs_dir']) / run_id
    deadline = time.monotonic() + 10
    while not (Path(cfg['repo_dir']) / 'running.pid').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    worker = json.loads((root / 'state.json').read_text())['worker_pid']
    child = int((Path(cfg['repo_dir']) / 'running.pid').read_text())
    os.kill(worker, signal.SIGKILL)
    # Only observe the filesystem/process identity; no get_result/inspect cleanup.
    deadline = time.monotonic() + 5
    while runner._process_identity(child) is not None and time.monotonic() < deadline:
        time.sleep(.02)
    try:
        assert runner._process_identity(child) is None
        result = json.loads((root / 'state.json').read_text())
        while result['status'] != 'failed' and time.monotonic() < deadline:
            time.sleep(.01)
            result = json.loads((root / 'state.json').read_text())
        assert result['status'] == 'failed' and result['usage']['gpu_hours'] > 0
    finally:
        if runner._process_identity(child) is not None:
            os.killpg(child, signal.SIGKILL)


def test_dead_group_leader_does_not_hide_live_grandchildren(tmp_path):
    import os, signal, time
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg)
    root = Path(cfg['runs_dir']) / run_id
    child_file = tmp_path / 'grandchild.pid'
    leader = subprocess.Popen([sys.executable, '-c',
        "import pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)", str(child_file)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while (not child_file.exists() or not child_file.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(.02)
        runner._write_json(root / 'process.json', {'identity': runner._process_identity(leader.pid), 'pgid': leader.pid})
        os.kill(leader.pid, signal.SIGKILL); leader.wait()
        dead = runner._process_identity(os.getpid()); dead['start_ticks'] = 'not-current'
        runner._write_json(root / 'launch.json', {'identity': dead})
        runner.get_result(run_id, cfg['runs_dir'])
        grandchild = int(child_file.read_text())
        deadline = time.monotonic() + 2
        while runner._process_identity(grandchild) is not None and time.monotonic() < deadline:
            time.sleep(.02)
        assert runner._process_identity(grandchild) is None
    finally:
        try: os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        leader.wait()


def test_walltime_allocation_ceiling_includes_termination_cleanup(tmp_path):
    from tm_research import runner
    cfg = config(tmp_path)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'train.py'
    script.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n')
    result = runner.wait_for_result(runner.run_baseline(cfg,
        execution_limits={'allocated_gpus': 1, 'job_walltime_seconds': .5}), cfg['runs_dir'], timeout=10)
    assert not result['valid'] and 'walltime' in result['error']
    assert result['usage']['elapsed_seconds'] < .8  # .3s scheduler tolerance, no extra 1s GPU grace


def test_stage_identity_is_persisted_before_scientific_child_exec(tmp_path, monkeypatch):
    import time
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg)
    root = Path(cfg['runs_dir']) / run_id
    original = runner._write_json
    def slow_parent_write(path, data):
        if Path(path).name == 'process.json' and data.get('identity'):
            time.sleep(.2)
        return original(path, data)
    monkeypatch.setattr(runner, '_write_json', slow_parent_write)
    runner._EXECUTION.context = {'root': root, 'deadline': None}
    try:
        rc = runner._execute([sys.executable, '-c',
            "import pathlib,json,sys; p=pathlib.Path(sys.argv[1])/'process.json'; assert p.is_file() and json.loads(p.read_text()).get('identity')", str(root)], root / 'stage.log', cfg['repo_dir'])
        assert rc == 0, (root / 'stage.log').read_text()
    finally:
        runner._EXECUTION.context = None


def test_frozen_evaluation_watchdog_survives_caller_sigkill_and_refuses_retry(tmp_path):
    import os, signal, time
    from tm_research import runner
    cfg = config(tmp_path)
    baseline = runner.wait_for_result(runner.run_baseline(cfg), cfg['runs_dir'], timeout=15)
    script = Path(cfg['repo_dir']) / 'photofinishing' / 'test.py'
    child_file = tmp_path / 'eval-child.pid'
    script.write_text("import pathlib,signal,time,os\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\npathlib.Path(%r).write_text(str(os.getpid()))\ntime.sleep(60)\n" % str(child_file))
    target = tmp_path / 'interrupted-frozen'
    payload = tmp_path / 'eval-input.json'
    payload.write_text(json.dumps({'config': cfg, 'checkpoint': baseline['artifacts'], 'split': cfg['test']}))
    env = os.environ.copy(); env['PYTHONPATH'] = str(Path(runner.__file__).resolve().parent.parent)
    caller = subprocess.Popen([sys.executable, '-c',
        "import json,sys; from tm_research.runner import evaluate_frozen_checkpoint; x=json.load(open(sys.argv[1])); evaluate_frozen_checkpoint(x['config'],x['checkpoint'],x['split'],sys.argv[2],execution_limits={'allocated_gpus':1,'job_walltime_seconds':3})", str(payload), str(target)], env=env)
    deadline = time.monotonic() + 5
    while not child_file.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    try:
        child = int(child_file.read_text())
        os.kill(caller.pid, signal.SIGKILL); caller.wait()
        deadline = time.monotonic() + 5
        while not (target / 'evaluation_result.json').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert runner._process_identity(child) is None
        result = json.loads((target / 'evaluation_result.json').read_text())
        assert not result['valid'] and result['usage']['gpu_hours'] > 0
        assert json.loads((target / 'evaluation_state.json').read_text())['status'] == 'failed'
        with pytest.raises(RuntimeError, match='already'):
            runner.evaluate_frozen_checkpoint(cfg, baseline['artifacts'], cfg['test'], target,
                execution_limits={'allocated_gpus':1,'job_walltime_seconds':3})
    finally:
        if caller.poll() is None:
            caller.kill(); caller.wait()
        if child_file.exists() and runner._process_identity(int(child_file.read_text())) is not None:
            os.killpg(int(child_file.read_text()), signal.SIGKILL)


def test_parent_pause_leader_exit_then_worker_death_preserves_identity_and_cleans_descendant(tmp_path):
    import os, signal, time
    from tm_research import runner
    cfg = config(tmp_path)
    root = Path(cfg['runs_dir']) / runner.prepare_run(cfg,
        execution_limits={'allocated_gpus': 2, 'job_walltime_seconds': 2})
    driver = tmp_path / 'paused-worker.py'
    driver.write_text('''import os,sys,time,signal
from pathlib import Path
from tm_research import runner
root=Path(sys.argv[1]); real=runner.subprocess.Popen
runner._state(root,status='training',started_at=runner._now(),worker_identity=runner._process_identity(os.getpid()))
def pause(*a,**kw):
    p=real(*a,**kw); time.sleep(.3); return p
runner.subprocess.Popen=pause
check=runner._check_execution; checks=0
def stop_after_publication():
    global checks
    checks+=1
    if checks==2: os.kill(os.getpid(),signal.SIGKILL)
    check()
runner._check_execution=stop_after_publication
runner._EXECUTION.context={'root':root,'deadline':None}
runner._execute([sys.executable,'-c',"import pathlib,subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid))",str(root/'descendant.pid')],root/'paused.log',str(root))
''')
    env = os.environ.copy(); env['PYTHONPATH'] = str(Path(runner.__file__).resolve().parent.parent)
    worker = subprocess.Popen([sys.executable, str(driver), str(root)], env=env)
    try:
        worker.wait(timeout=5)
        pid = int((root / 'descendant.pid').read_text())
        record = json.loads((root / 'process.json').read_text())
        result = runner.get_result(root.name, cfg['runs_dir'])
        assert record['identity'] is not None
        assert runner._process_identity(pid) is None
        assert result['status'] == 'failed'
        assert result['usage']['gpu_hours'] > 0
        assert result['usage']['elapsed_seconds'] >= .3
    finally:
        if worker.poll() is None: worker.kill(); worker.wait()
        record = json.loads((root / 'process.json').read_text())
        if record.get('pgid'):
            try: os.killpg(record['pgid'], signal.SIGKILL)
            except ProcessLookupError: pass


@pytest.mark.parametrize('frozen', [False, True])
def test_post_popen_metadata_failure_cleans_scientific_group_and_charges_cleanup(tmp_path, monkeypatch, frozen):
    import os, signal, time
    from tm_research import runner
    cfg = config(tmp_path)
    baseline = None
    if frozen:
        baseline = runner.wait_for_result(runner.run_baseline(cfg), cfg['runs_dir'], timeout=15)
    script = Path(cfg['repo_dir']) / 'photofinishing' / ('test.py' if frozen else 'train.py')
    marker = tmp_path / 'fault-child.pid'
    script.write_text("import pathlib,os,signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\npathlib.Path(%r).write_text(str(os.getpid()))\ntime.sleep(60)\n" % str(marker))
    real_popen = runner.subprocess.Popen
    def scientific_start_before_parent_return(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        if '_stage_exec' in str(args[0]):
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
        return child
    monkeypatch.setattr(runner.subprocess, 'Popen', scientific_start_before_parent_return)
    write = runner._write_json
    failed = False
    def fail_after_scientific_start(path, data):
        nonlocal failed
        if not failed and Path(path).name == 'process.json' and data.get('identity'):
            failed = True
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
            raise OSError('injected post-Popen metadata failure')
        return write(path, data)
    monkeypatch.setattr(runner, '_write_json', fail_after_scientific_start)
    root = tmp_path / 'frozen-fault' if frozen else Path(cfg['runs_dir']) / runner.prepare_run(cfg,
        execution_limits={'allocated_gpus': 2, 'job_walltime_seconds': .5})
    try:
        if frozen:
            with pytest.raises(OSError, match='post-Popen'):
                runner.evaluate_frozen_checkpoint(cfg, baseline['artifacts'], cfg['test'], root,
                    execution_limits={'allocated_gpus': 2, 'job_walltime_seconds': .5})
            result = json.loads((root / 'evaluation_result.json').read_text())
        else:
            result = runner._worker(root)
        pid = int(marker.read_text())
        assert runner._process_identity(pid) is None
        assert not result['valid'] and 'post-Popen' in result['error']
        assert result['usage']['gpu_hours'] == pytest.approx(2 * result['usage']['elapsed_seconds'] / 3600)
        assert .1 < result['usage']['elapsed_seconds'] < .9
    finally:
        if marker.exists() and runner._process_identity(int(marker.read_text())) is not None:
            os.killpg(int(marker.read_text()), signal.SIGKILL)


def test_prepare_request_id_is_durable_across_interrupted_allocation_and_reused_exactly(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from tm_research import runner
    cfg = config(tmp_path)
    real = runner._write_json
    interrupted = False
    def fail_after_request_persisted(path, data):
        nonlocal interrupted
        real(path, data)
        if not interrupted and data.get('request_id') == 'launch-001':
            interrupted = True
            raise OSError('controller interrupted after launch request allocation')
    monkeypatch.setattr(runner, '_write_json', fail_after_request_persisted)
    with pytest.raises(OSError):
        runner.prepare_run(cfg, request_id='launch-001')
    # The durable requested ID remains reserved for the original launch request.
    assert runner.prepare_run(cfg) == 'exp_002'
    with ThreadPoolExecutor(max_workers=3) as pool:
        ids = list(pool.map(lambda _: runner.prepare_run(cfg, request_id='launch-001'), range(3)))
    assert ids == ['exp_001'] * 3
    runner.start_prepared_run('exp_001', cfg['runs_dir'])
    result = runner.wait_for_result('exp_001', cfg['runs_dir'], timeout=15)
    assert runner.prepare_run(cfg, request_id='launch-001') == 'exp_001'
    assert runner.get_result('exp_001', cfg['runs_dir']) == result
    with pytest.raises(ValueError, match='request'):
        runner.prepare_run({**cfg, 'epochs': 2}, request_id='launch-001')


def test_unknown_group_release_retains_nonterminal_state_and_budget_reservation(tmp_path, monkeypatch):
    from tm_research import runner
    cfg = config(tmp_path, mode='train_fail')
    root = Path(cfg['runs_dir']) / runner.prepare_run(cfg,
        execution_limits={'allocated_gpus': 2, 'job_walltime_seconds': .4})
    # /proc is unavailable to confirm release, even after sending KILL.
    monkeypatch.setattr(runner, '_group_liveness', lambda pgid: 'unknown')
    result = runner._worker(root)
    assert result['status'] == 'recovery_required' and not result['valid']
    assert result['usage']['gpu_hours'] is None and result['usage']['status'] == 'unknown'
    assert not (root / 'result_recorded.json').exists()
    assert runner._read_json(root / 'process.json').get('pgid')


def test_null_identity_with_live_group_is_never_confirmed_cleanup(tmp_path):
    import os, signal
    from tm_research import runner
    root = tmp_path / 'ambiguous'; root.mkdir()
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    try:
        runner._write_json(root / 'process.json', {'identity': None, 'pgid': child.pid})
        assert runner._cleanup_recorded_process(root) is False
        assert runner._process_identity(child.pid) is not None
    finally:
        os.killpg(child.pid, signal.SIGKILL); child.wait()


def test_terminal_metadata_does_not_release_unknown_live_group_reservation(tmp_path):
    import os, signal
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg, execution_limits={'allocated_gpus': 1, 'job_walltime_seconds': 1})
    root = Path(cfg['runs_dir']) / run_id
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        runner._write_json(root / 'process.json', {'identity': None, 'pgid': child.pid})
        runner._state(root, status='completed', result_status='invalid', error='old persistence failure',
            usage={'gpu_hours': .001, 'allocated_gpus': 1})
        assert runner.inspect_run(run_id, cfg['runs_dir'])['liveness'] == 'unknown'
        result = runner.get_result(run_id, cfg['runs_dir'])
        assert result['status'] == 'recovery_required' and result['usage']['gpu_hours'] is None
        assert not (root / 'result_recorded.json').exists()
    finally:
        os.killpg(child.pid, signal.SIGKILL); child.wait()


def test_recovery_of_terminal_receipt_with_live_group_includes_late_cleanup_cost(tmp_path):
    import os, signal, time
    from tm_research import runner
    cfg = config(tmp_path)
    run_id = runner.prepare_run(cfg, execution_limits={'allocated_gpus': 2, 'job_walltime_seconds': 2})
    root = Path(cfg['runs_dir']) / run_id
    child = subprocess.Popen([sys.executable, '-c',
        'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'], start_new_session=True)
    try:
        runner._write_json(root / 'process.json', {'identity': runner._process_identity(child.pid), 'pgid': child.pid})
        runner._state(root, status='completed', result_status='invalid', valid=False, started_at=runner._now(),
            usage={'allocated_gpus': 2, 'elapsed_seconds': .001, 'gpu_hours': 2*.001/3600})
        time.sleep(.05)
        result = runner.get_result(run_id, cfg['runs_dir'])
        child.wait(timeout=2)
        assert result['usage']['elapsed_seconds'] > .05
        assert result['usage']['gpu_hours'] == pytest.approx(2 * result['usage']['elapsed_seconds'] / 3600)
        assert runner._process_identity(child.pid) is None
    finally:
        if child.poll() is None: os.killpg(child.pid, signal.SIGKILL); child.wait()
