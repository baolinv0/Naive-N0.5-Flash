"""DEV runner, bounded research campaigns and frozen confirmation controls."""

import argparse
import json
from pathlib import Path

from .runner import (_worker, commands_for_run, get_result, load_config,
                     run_baseline, wait_for_result, inspect_run, request_run_stop)
from .campaign import (DEFAULT_MIN_DELTA, campaign_status, campaign_feedback, final_test,
                       finalize_campaign, initialize_campaign, next_proposal, submit_proposal,
                       wait_campaign, validate_campaign_decision)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


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
    p = sub.add_parser('run')
    actions = p.add_subparsers(dest='run_action', required=True)
    for name in ('inspect', 'stop'):
        command = actions.add_parser(name)
        command.add_argument('--run', required=True)
        command.add_argument('--runs-dir', default='runs')
        if name == 'stop':
            command.add_argument('--reason', required=True)
    p = sub.add_parser('control')
    p.add_argument('--control', required=True, help='Validate and display a frozen authorization record')
    p = sub.add_parser('decision')
    p.add_argument('--decision', required=True)
    p.add_argument('--campaign-dir', required=True)
    p = sub.add_parser('profile')
    p.add_argument('--config', required=True)
    p.add_argument('--rules', required=True)
    p.add_argument('--output-dir', required=True)
    p = sub.add_parser('confirmation')
    actions = p.add_subparsers(dest='confirmation_action', required=True)
    for name in ('init', 'next', 'report'):
        command = actions.add_parser(name)
        command.add_argument('--confirmation-dir', required=True)
        if name == 'init':
            command.add_argument('--campaign-dir', required=True)
            command.add_argument('--plan', required=True)
            command.add_argument('--control', required=True)
        if name == 'next':
            command.add_argument('--wait', action='store_true')
    p = sub.add_parser('campaign')
    actions = p.add_subparsers(dest='campaign_action', required=True)
    for name in ('init', 'next', 'submit', 'status', 'wait', 'finalize', 'feedback',
                 'review-packet', 'record-decision', 'report', 'memory'):
        command = actions.add_parser(name)
        command.add_argument('--campaign-dir', required=True)
        if name == 'init':
            command.add_argument('--config', required=True)
            command.add_argument('--max-trials', type=int, default=5)
            command.add_argument('--no-gain-limit', type=int, default=3)
            command.add_argument('--min-delta', type=float, default=DEFAULT_MIN_DELTA,
                                 help='Frozen DEV gain threshold in dB; calibrate before live use')
            command.add_argument('--control')
            command.add_argument('--profile')
        if name in ('init', 'submit'):
            command.add_argument('--wait', action='store_true')
        if name == 'submit':
            command.add_argument('--proposal', required=True)
            command.add_argument('--decision')
        if name == 'record-decision':
            command.add_argument('--decision', required=True)
        if name == 'feedback':
            command.add_argument('--run')
            command.add_argument('--section')
            command.add_argument('--key')
        if name == 'review-packet':
            command.add_argument('--trigger')
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
        elif args.action == 'control':
            from .control import load_control
            result = load_control(args.control)
        elif args.action == 'decision':
            result = validate_campaign_decision(args.campaign_dir, _json(args.decision))
        elif args.action == 'profile':
            from .diagnostics import build_dev_profile
            result = build_dev_profile(load_config(args.config), _json(args.rules), args.output_dir)
        elif args.action == 'run':
            result = (inspect_run(args.run, args.runs_dir) if args.run_action == 'inspect' else
                      request_run_stop(args.run, args.runs_dir, reason=args.reason))
        elif args.action == 'confirmation':
            from .confirmation import initialize_confirmation, run_confirmation_next, report_confirmation
            if args.confirmation_action == 'init':
                result = initialize_confirmation(args.campaign_dir, args.confirmation_dir,
                                                 _json(args.plan), control_ref=args.control)
            elif args.confirmation_action == 'next':
                result = run_confirmation_next(args.confirmation_dir, wait=args.wait)
            else:
                result = report_confirmation(args.confirmation_dir)
        elif args.action == 'campaign':
            if args.campaign_action == 'init':
                result = initialize_campaign(load_config(args.config), args.campaign_dir,
                                             args.max_trials, args.no_gain_limit, wait=args.wait,
                                             min_delta=args.min_delta, control_ref=args.control,
                                             profile_ref=args.profile)
            elif args.campaign_action == 'submit':
                result = submit_proposal(args.campaign_dir, _json(args.proposal), wait=args.wait,
                                         decision_ref=args.decision)
            elif args.campaign_action == 'wait':
                result = wait_campaign(args.campaign_dir, args.timeout)
            elif args.campaign_action == 'feedback':
                result = campaign_feedback(args.campaign_dir, args.run)
                if args.section:
                    from .evidence import load_feedback_detail
                    if not result.get('feedback_ref'):
                        raise ValueError('Feedback detail is unavailable')
                    result = load_feedback_detail(str(Path(result['feedback_ref']).parent), args.section, args.key)
            elif args.campaign_action in ('review-packet', 'record-decision', 'report', 'memory'):
                from .research import (build_review_packet, record_research_decision,
                                       build_campaign_report, export_research_memory)
                campaign_status(args.campaign_dir)
                if args.campaign_action == 'review-packet':
                    result = build_review_packet(args.campaign_dir, args.trigger)
                elif args.campaign_action == 'record-decision':
                    # record_research_decision owns its independent campaign lock.
                    result = record_research_decision(args.campaign_dir, _json(args.decision))
                elif args.campaign_action == 'report':
                    result = build_campaign_report(args.campaign_dir)
                else:
                    result = export_research_memory(args.campaign_dir)
            else:
                result = {'next': next_proposal, 'status': campaign_status,
                          'finalize': finalize_campaign}[args.campaign_action](args.campaign_dir)
        elif args.action == 'wait':
            result = wait_for_result(args.run, args.runs_dir, args.timeout)
        else:
            result = get_result(args.run, args.runs_dir)
            if args.action == 'report':
                result = {key: result.get(key) for key in ('run_id', 'status', 'result_status', 'valid', 'error',
                    'recipe', 'dev_psnr', 'train_metrics', 'dev_metrics', 'artifacts', 'logs')}
        print(json.dumps(result, indent=2))
        if isinstance(result, dict):
            return 1 if (result.get('status') == 'failed' or
                         result.get('valid') is False and result.get('status') == 'completed') else 0
        return 0
    except (ValueError, RuntimeError, OSError, TimeoutError, KeyError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
