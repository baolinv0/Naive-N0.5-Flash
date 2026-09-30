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
