"""Read-only probes against copies of connector-fetched Phase1 source.
These are diagnostic checks, not real Naive inference or real ISP training.
"""
import json
import tempfile
from pathlib import Path
import torch
import sys
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--overlay-dir', required=True, help='Applied overlay repository root, containing tm_research/ and tests/')
parser.add_argument('--out', default='review_probe_results.json')
args = parser.parse_args()
root = Path(args.overlay_dir).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'tests'))
from test_runner import config
from tm_research.runner import commands_for_run, normalize_config, run_baseline, wait_for_result
from tm_research.naive_adapter import parse_assistant_text
from photofinishing.baseline_utils import image_psnr_values

out={}
target=torch.zeros(2,3,2,2,dtype=torch.float64)
a=target.clone();a[0]+=.1;a[1]+=.01
b=target.clone()+.002**.5
scores={}
for name,pred in [('A',a),('B',b)]:
    scores[name]={'image_mean_psnr':sum(image_psnr_values(pred,target))/2,
                  'batch_psnr':(-10*torch.log10((pred-target).square().mean())).item()}
out['checkpoint_ranking']={'scores':scores,'current_selects':max(scores,key=lambda n:scores[n]['batch_psnr']),
                           'planned_metric_selects':max(scores,key=lambda n:scores[n]['image_mean_psnr'])}
with tempfile.TemporaryDirectory() as d:
    cfg=config(Path(d))
    tr,te=commands_for_run(cfg,Path(d)/'runs'/'exp_001')
    out['commands']={'same_repo_train_path':tr[1],'auto_test_present':te is not None,
                     'fixed_test_size_supported':'--in-size' in te or '--eval-size' in te,
                     'learning_rate_flag_present':'--learning-rate' in tr}
    try:
        normalize_config({**cfg,'learning_rate':1e-4})
        out['learning_rate_config']='accepted'
    except Exception as e:
        out['learning_rate_config']=str(e)
    cfg.pop('test')
    script=Path(cfg['repo_dir'])/'photofinishing'/'train.py'
    script.write_text('''import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--output-dir');a,_=p.parse_known_args()
r=pathlib.Path(a.output_dir);r.mkdir(parents=True)
(r/'metrics.json').write_text(json.dumps({'mean_psnr':float('nan'),'best_checkpoint':str(r/'missing.pth'),'config_dir':str(r/'missing-config')}))
''')
    rid=run_baseline(cfg)
    result=wait_for_result(rid,cfg['runs_dir'],timeout=10)
    out['completion_without_checkpoint']={'status':result['status'],
        'checkpoint_exists':Path(result['artifacts']['best_checkpoint']).exists(),
        'mean_psnr_is_finite':bool(torch.isfinite(torch.tensor(result['train_metrics']['mean_psnr'])))}
tools=[{'type':'function','function':{'name':'run_baseline','parameters':{
    'type':'object','properties':{'config':{'type':'object'}},'required':['config']}}}]
call='<tool_call><function=run_baseline><parameter=config>{"epochs":2}</parameter></function></tool_call>'
for name,raw in [('unclosed_think','<think>Planning an experiment '+call),
                 ('missing_required','<tool_call><function=run_baseline></function></tool_call>'),
                 ('wrong_type','<tool_call><function=run_baseline><parameter=config>2</parameter></function></tool_call>'),
                 ('truncated_call','<tool_call><function=run_baseline><parameter=config>{"epochs":')]:
    content,calls=parse_assistant_text(raw,tools)
    out[name]={'content':content,'parsed_calls':calls}
x=torch.zeros(1,3,2,2)
out['metric_exact_match']=str(image_psnr_values(x,x)[0])
y=x.clone();y[0,0,0,0]=float('nan')
out['metric_nan_prediction']=str(image_psnr_values(y,x)[0])
Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2), encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
