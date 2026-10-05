"""Generate, inspect, and advance an authorized experiment one bounded step at a time."""
import argparse
import json
from pathlib import Path

from .experiment_plan import generate_experiment, load_experiment, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Validate a manifest without launching work')
    validate.add_argument('--manifest', required=True)
    generate = commands.add_parser('generate', help='Freeze independent campaign inputs without launching work')
    generate.add_argument('--manifest', required=True)
    generate.add_argument('--output', required=True)
    for name in ('status', 'step', 'confirm', 'analyze'):
        command = commands.add_parser(name)
        command.add_argument('--experiment', required=True, help='Generated experiment directory')
        if name in ('step', 'confirm'):
            command.add_argument('--wait', action='store_true', help='Wait for this step; never loop over the experiment')
        if name == 'analyze':
            command.add_argument('--output', required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'validate':
            result = load_experiment(arguments.manifest)
        elif arguments.command == 'generate':
            result = generate_experiment(arguments.manifest, arguments.output)
        elif arguments.command == 'status':
            from .experiment_runner import experiment_status
            result = experiment_status(arguments.experiment)
        elif arguments.command == 'step':
            from .experiment_runner import step_experiment
            result = step_experiment(arguments.experiment, wait=arguments.wait)
        elif arguments.command == 'confirm':
            from .experiment_runner import confirm_experiment
            result = confirm_experiment(arguments.experiment, wait=arguments.wait)
        else:
            from .experiment_analysis import analyze_records, export_report
            from .experiment_runner import collect_analysis_records
            index = read_json(Path(arguments.experiment) / 'experiment.json')
            manifest = index['manifest']
            records = collect_analysis_records(arguments.experiment)
            report = analyze_records(records, strategies=manifest['strategies'],
                blocks=list(range(len(manifest['search_seeds']))),
                delta_strategy=(manifest.get('confirmation') or {}).get('delta_strategy', .1))
            report.update(experiment_id=index['experiment_id'], experiment_identity=index['identity'],
                          evidence_mode=index['evidence_mode'], source_records=records)
            result = {'status': report['status'], 'evidence_mode': report['evidence_mode'],
                      'artifacts': export_report(report, arguments.output)}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if isinstance(result, dict) and result.get('status') == 'failed' else 0
    except (ValueError, RuntimeError, OSError, TimeoutError, KeyError) as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=False, allow_nan=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
