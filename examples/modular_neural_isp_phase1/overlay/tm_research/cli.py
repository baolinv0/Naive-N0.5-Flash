"""Command-line controls for a single original photofinishing baseline run."""

import argparse
import json
from pathlib import Path

from .runner import (_worker, commands_for_run, get_result, load_config,
                     run_baseline, wait_for_result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('baseline', 'commands'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
    for name in ('status', 'wait', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--run', required=True)
        p.add_argument('--runs-dir', default='runs')
        if name == 'wait':
            p.add_argument('--timeout', type=float)
    p = sub.add_parser('_worker', help=argparse.SUPPRESS)
    p.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    if args.action == 'baseline':
        cfg = load_config(args.config)
        print(json.dumps({'run_id': run_baseline(cfg), 'runs_dir': cfg['runs_dir']}))
        return 0
    if args.action == 'commands':
        cfg = load_config(args.config)
        preview = Path(cfg['runs_dir']) / 'exp_001'
        train, test = commands_for_run(cfg, preview)
        print(json.dumps({'train': train, 'test': test}, indent=2))
        return 0
    if args.action == '_worker':
        return 0 if _worker(args.run_dir)['status'] == 'completed' else 1
    if args.action == 'wait':
        result = wait_for_result(args.run, args.runs_dir, args.timeout)
    else:
        result = get_result(args.run, args.runs_dir)
    if args.action == 'report':
        result = {key: result.get(key) for key in ('run_id', 'status', 'error', 'train_metrics',
                                                    'test_metrics', 'artifacts', 'logs')}
    print(json.dumps(result, indent=2))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
