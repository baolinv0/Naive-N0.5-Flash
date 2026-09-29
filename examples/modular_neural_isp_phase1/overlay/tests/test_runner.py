import json
import subprocess
import sys
import time
from pathlib import Path


from tm_research.runner import commands_for_run, get_result, load_config, run_baseline, wait_for_result


def fixture_repo(tmp_path, fail=False):
    repo = tmp_path / 'upstream'
    scripts = repo / 'photofinishing'
    scripts.mkdir(parents=True)
    (scripts / 'train.py').write_text('''import argparse, json, pathlib, sys
p=argparse.ArgumentParser(); p.add_argument('--output-dir'); p.add_argument('--exp-name'); a,_=p.parse_known_args()
r=pathlib.Path(a.output_dir); (r/'models').mkdir(parents=True); (r/'config').mkdir()
if %s: sys.exit(7)
model=r/'models'/('photofinishing_'+a.exp_name+'-best.pth'); model.write_text('weight')
(r/'config'/(model.stem+'.json')).write_text('{}')
(r/'metrics.json').write_text(json.dumps({'best_checkpoint':str(model),'config_dir':str(r/'config'),'mean_psnr':22.5,'original_batch_psnr':21.0}))
print('train fixture complete')
''' % fail)
    (scripts / 'test.py').write_text('''import argparse, json, pathlib
p=argparse.ArgumentParser(); p.add_argument('--model-path'); p.add_argument('--config-dir'); p.add_argument('--result-dir'); a,_=p.parse_known_args()
assert pathlib.Path(a.model_path).is_file(); assert (pathlib.Path(a.config_dir)/(pathlib.Path(a.model_path).stem+'.json')).is_file()
r=pathlib.Path(a.result_dir); r.mkdir(); (r/'per_image.csv').write_text('image,psnr\\na,20\\n'); (r/'metrics.json').write_text(json.dumps({'mean_psnr':20.0,'mean_ssim':0.9}))
print('test fixture complete')
''')
    return repo


def config(tmp_path, fail=False):
    repo=fixture_repo(tmp_path, fail)
    dirs={}
    for split in ('train','validation','test'):
        dirs[split]={}
        for name in ('input_dir','gt_dir','metadata_dir'):
            d=tmp_path/split/name; d.mkdir(parents=True)
            dirs[split][name]=str(d)
    return {'repo_dir':str(repo),'python':sys.executable,'runs_dir':str(tmp_path/'runs'),**dirs,'epochs':1,'batch_size':2}


def test_config_paths_relative_to_yaml_and_optional_flags(tmp_path):
    repo=fixture_repo(tmp_path)
    interpreter=tmp_path/'venv-python'
    interpreter.symlink_to(sys.executable)
    path=tmp_path/'baseline.yaml'
    path.write_text('repo_dir: upstream\npython: venv-python'+'\nruns_dir: runs\ntrain:\n  input_dir: train/in\n  gt_dir: train/gt\nvalidation:\n  input_dir: val/in\n  gt_dir: val/gt\nepochs: 2\n')
    cfg=load_config(path)
    train,test=commands_for_run(cfg,tmp_path/'runs'/'exp_001')
    assert cfg['python']==str(interpreter)
    assert cfg['train']['input_dir']==str(tmp_path/'train'/'in')
    assert '--epochs' in train and train[train.index('--epochs')+1]=='2'
    assert '--batch-size' not in train and test is None
    assert train[train.index('--output-dir')+1]==str(tmp_path/'runs'/'exp_001'/'train')
    assert all(Path(train[train.index(flag)+1]).is_absolute() for flag in ('--in-training-dir','--gt-training-dir','--in-validation-dir','--gt-validation-dir'))


def test_background_process_runs_train_then_test_and_saves_result(tmp_path):
    cfg=config(tmp_path)
    first=run_baseline(cfg)
    second=run_baseline(cfg)
    assert (first,second)==('exp_001','exp_002')
    result=wait_for_result(first,cfg['runs_dir'],timeout=15)
    assert result['status']=='completed',result
    assert result['train_metrics']['mean_psnr']==22.5
    assert result['test_metrics']['mean_psnr']==20
    assert Path(result['logs']['train']).read_text().find('train fixture complete')>=0
    assert Path(result['logs']['test']).read_text().find('test fixture complete')>=0
    assert Path(result['artifacts']['per_image_csv']).is_file()
    assert Path(cfg['runs_dir'],'results.jsonl').exists()
    assert wait_for_result(second,cfg['runs_dir'],timeout=15)['status']=='completed'


def test_failed_child_retains_diagnostic_log_and_skips_test(tmp_path):
    cfg=config(tmp_path,fail=True)
    run=run_baseline(cfg)
    result=wait_for_result(run,cfg['runs_dir'],timeout=15)
    assert result['status']=='failed'
    assert result['returncode']==7
    assert 'train.py' in result['error']
    assert Path(result['logs']['train']).is_file()
    assert not Path(cfg['runs_dir'],run,'test').exists()


def test_cli_wait_and_report_read_persisted_run(tmp_path):
    cfg=config(tmp_path)
    run=run_baseline(cfg)
    base=[sys.executable,'-m','tm_research.cli']
    wait=subprocess.run(base+['wait','--run',run,'--runs-dir',cfg['runs_dir'],'--timeout','15'],
                        capture_output=True,text=True,check=True)
    assert json.loads(wait.stdout)['status']=='completed'
    report=subprocess.run(base+['report','--run',run,'--runs-dir',cfg['runs_dir']],
                          capture_output=True,text=True,check=True)
    summary=json.loads(report.stdout)
    assert summary['train_metrics']['mean_psnr']==22.5
    assert summary['test_metrics']['mean_psnr']==20


def test_checkpoint_lut_setting_is_preserved(tmp_path):
    cfg=config(tmp_path)
    checkpoint=tmp_path/'initialized.pth'; checkpoint.write_text('fixture')
    config_dir=tmp_path/'model_configs'; config_dir.mkdir()
    (config_dir/'initialized.json').write_text('{"use_3d_lut": true}')
    cfg['init_checkpoint']=str(checkpoint)
    cfg['init_config_dir']=str(config_dir)
    train,_=commands_for_run(cfg,Path(cfg['runs_dir'])/'exp_001')
    assert '--use-3d-lut' in train
    assert train[train.index('--load-config-dir')+1]==str(config_dir)
