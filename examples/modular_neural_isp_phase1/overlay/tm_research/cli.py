"""DEV recipe runner and persistent ARIS/Naive campaign controls."""

import argparse
import json
from pathlib import Path

from .runner import (_worker, commands_for_run, get_result, load_config,
                     run_baseline, wait_for_result)
from .campaign import (DEFAULT_MIN_DELTA, campaign_status, final_test, finalize_campaign, initialize_campaign,
                       next_proposal, submit_proposal, wait_campaign)


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
    p = sub.add_parser('final-test')
    p.add_argument('--campaign-dir', required=True)
    p = sub.add_parser('campaign')
    actions = p.add_subparsers(dest='campaign_action', required=True)
    for name in ('init', 'next', 'submit', 'status', 'wait', 'finalize'):
        command = actions.add_parser(name)
        command.add_argument('--campaign-dir', required=True)
        if name == 'init':
            command.add_argument('--config', required=True)
            command.add_argument('--max-trials', type=int, default=5)
            command.add_argument('--no-gain-limit', type=int, default=3)
            command.add_argument('--min-delta', type=float, default=DEFAULT_MIN_DELTA,
                                 help='Frozen effective DEV gain threshold in dB (default: 0.01; calibrate before live use)')
        if name in ('init', 'submit'):
            command.add_argument('--wait', action='store_true', help='Block through training; otherwise use campaign wait')
        if name == 'submit':
            command.add_argument('--proposal', required=True, help='Path to JSON recipe, hypothesis and prior-result citation')
        if name == 'wait':
            command.add_argument('--timeout', type=float)
    args = parser.parse_args(argv)
    try:
        if args.action == 'baseline':
            cfg = load_config(args.config)
            result = {'run_id': run_baseline(cfg), 'runs_dir': cfg['runs_dir']}
        elif args.action == 'commands':
            cfg = load_config(args.config)
            train, dev = commands_for_run(cfg, Path(cfg['runs_dir']) / 'exp_001')
            result = {'train': train, 'dev': dev}
        elif args.action == '_worker':
            result = _worker(args.run_dir)
        elif args.action == 'final-test':
            result = final_test(args.campaign_dir)
        elif args.action == 'campaign':
            if args.campaign_action == 'init':
                result = initialize_campaign(load_config(args.config), args.campaign_dir,
                                             args.max_trials, args.no_gain_limit, wait=args.wait,
                                             min_delta=args.min_delta)
            elif args.campaign_action == 'submit':
                proposal = json.loads(Path(args.proposal).read_text(encoding='utf-8'))
                result = submit_proposal(args.campaign_dir, proposal, wait=args.wait)
            elif args.campaign_action == 'wait':
                result = wait_campaign(args.campaign_dir, args.timeout)
            else:
                function = {'next': next_proposal, 'status': campaign_status,
                            'finalize': finalize_campaign}[args.campaign_action]
                result = function(args.campaign_dir)
        elif args.action == 'wait':
            result = wait_for_result(args.run, args.runs_dir, args.timeout)
        else:
            result = get_result(args.run, args.runs_dir)
            if args.action == 'report':
                result = {key: result.get(key) for key in ('run_id', 'status', 'result_status', 'valid', 'error',
                    'recipe', 'dev_psnr', 'train_metrics', 'dev_metrics', 'artifacts', 'logs')}
        print(json.dumps(result, indent=2))
        return 1 if result.get('status') == 'failed' or result.get('valid') is False and result.get('status') == 'completed' else 0
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
