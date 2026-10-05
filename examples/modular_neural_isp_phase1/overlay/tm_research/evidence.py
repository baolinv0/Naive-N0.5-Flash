"""Assemble reproducible TRAIN/DEV evidence without changing score selection.

Revisions retain readable source snapshots, rather than content hashes. The
campaign remains the authority for completed/valid/improved and aggregate score.
"""
import csv
import fcntl
import json
import math
from pathlib import Path
import re

RECIPE_FIELDS = {'loss_family', 'optimizer', 'learning_rate', 'weight_decay'}
IDENTITY_FIELDS = {'runs_dir', 'run_dir', 'run_id', 'output_dir', 'result_dir',
                   'log_dir', 'logs_dir', 'created_at', 'updated_at', 'started_at',
                   'finished_at', 'timestamp', 'exp_name'}
REFERENCE_FIELDS = ('baseline_run_id', 'best_before_run_id', 'comparison_run_id',
                    'construction_base_run_id', 'based_on_run_id', 'init_checkpoint_ref')
DETAIL_FILES = {'summary': 'summary.json', 'paired_rows': 'paired_rows.csv',
                'paired_comparison': 'paired_comparison.json', 'cases': 'cases.json',
                'curves': 'curves.json', 'details_index': 'details_index.json'}


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n', encoding='utf-8')


def _path(root, value):
    value = Path(value)
    return value if value.is_absolute() else Path(root) / value


def _samples(manifest):
    samples = manifest.get('samples', []) if isinstance(manifest, dict) else manifest
    if not isinstance(samples, list):
        raise ValueError('manifest samples must be a list')
    return samples


def load_manifest(path):
    """Load actual sample identities; basename conversion is allowed only uniquely."""
    value = _read(path)
    if value is None:
        raise ValueError(f'Manifest not found: {path}')
    samples = _samples(value)
    ids, names = {}, {}
    for sample in samples:
        sid = sample.get('sample_id')
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError('Manifest sample_id must be nonempty and unique')
        ids[sid] = sample
        name = Path(sample.get('input_relpath', sample.get('image', ''))).name
        if name:
            names.setdefault(name, []).append(sid)
    return {**(value if isinstance(value, dict) else {}), 'samples': samples,
            'by_sample_id': ids,
            'basename_to_sample_id': {name: values[0] for name, values in names.items() if len(values) == 1},
            'ambiguous_basenames': sorted(name for name, values in names.items() if len(values) > 1)}


def _flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        if key in IDENTITY_FIELDS:
            continue
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def diff_scientific_config(resolved, reference):
    """Compare all scientific fields including data and initialization identities."""
    current, prior = _flatten(resolved), _flatten(reference)
    for original, flattened in ((resolved, current), (reference, prior)):
        lr = original.get('learning_rate')
        if type(lr) in (int, float) and math.isfinite(lr):
            flattened.setdefault('scheduler.eta_min', lr / 100)
    return {key: {'from': prior.get(key), 'to': current.get(key)}
            for key in sorted(set(current) | set(prior)) if current.get(key) != prior.get(key)}


def _fixed_changes(changes):
    return {key: item for key, item in changes.items()
            if key not in RECIPE_FIELDS and key != 'scheduler.eta_min'}


def _protocol(state, trial):
    return (trial.get('dev_metrics') or {}).get('protocol') or state.get('control', {}).get('protocol_id')


def resolve_trial_references(state, proposal, decision=None):
    """Resolve launch-time roles; evidence citation never changes recipe ancestry."""
    trials = state.get('trials', [])
    best = state.get('best') or {}
    baseline = next((trial['run_id'] for trial in trials if trial.get('valid')), None)
    comparison = best.get('run_id')
    if decision and decision.get('comparison_run_id') is not None:
        comparison = decision['comparison_run_id']
        target = next((trial for trial in trials if trial['run_id'] == comparison), None)
        if target is None or not target.get('valid'):
            raise ValueError('comparison_run_id must identify an existing valid run')
        cfg = target.get('resolved_config') or {**state.get('config', {}), **target.get('recipe', {})}
        best_trial = next((trial for trial in trials if trial['run_id'] == best.get('run_id')), None)
        frozen_cfg = (best_trial or {}).get('resolved_config') or state.get('config', {})
        fixed = _fixed_changes(diff_scientific_config(cfg, frozen_cfg))
        if fixed:
            raise ValueError('comparison_run_id has incompatible scientific protocol: ' + ', '.join(fixed))
        expected = state.get('control', {}).get('protocol_id') or (_protocol(state, best_trial) if best_trial else None)
        if expected and _protocol(state, target) and _protocol(state, target) != expected:
            raise ValueError('comparison_run_id has incompatible evaluation protocol')
    based_on = (proposal or {}).get('based_on') or {}
    return {'baseline_run_id': baseline, 'best_before_run_id': best.get('run_id'),
            'comparison_run_id': comparison, 'construction_base_run_id': best.get('run_id'),
            'based_on_run_id': based_on.get('run_id') if isinstance(based_on, dict) else None,
            'init_checkpoint_ref': state.get('config', {}).get('init_checkpoint')}


def _historic_references(state, trial):
    index = next(i for i, item in enumerate(state['trials']) if item['run_id'] == trial['run_id'])
    before = state['trials'][:index]
    best = None
    for item in before:
        if item.get('valid') and type(item.get('dev_psnr')) in (int, float):
            if best is None or item['dev_psnr'] > best['dev_psnr'] + state.get('min_delta', 0):
                best = item
    historical = {**state, 'trials': before, 'best': best}
    refs = resolve_trial_references(historical, trial.get('proposal'), None)
    refs.update({key: val for key, val in (trial.get('references') or {}).items() if key in REFERENCE_FIELDS})
    refs.update({key: trial[key] for key in REFERENCE_FIELDS if key in trial})
    if index == 0:
        refs.update({key: None for key in REFERENCE_FIELDS if key != 'init_checkpoint_ref'})
    return refs, index


def _run_dir(root, state, trial):
    if trial.get('run_dir'):
        return _path(root, trial['run_dir'])
    return _path(root, state.get('config', {}).get('runs_dir', 'runs')) / trial['run_id']


def _config(root, state, trial):
    return trial.get('resolved_config') or _read(_run_dir(root, state, trial) / 'config.json') or {
        **state.get('config', {}), **trial.get('recipe', {})}


def _csv(root, state, trial, profile, sources):
    directory = _run_dir(root, state, trial)
    artifacts = trial.get('artifacts') or {}
    metrics = trial.get('dev_metrics') or _read(directory / 'dev/metrics.json', {})
    ref = artifacts.get('per_image_csv') or metrics.get('per_image_csv')
    path = _path(directory, ref) if ref else directory / 'dev/per_image.csv'
    if not path.is_file():
        sources[str(path)] = None
        return None, f'per_image_csv unavailable for {trial["run_id"]}: {path}'
    sources[str(path)] = path.read_text(encoding='utf-8')
    samples = [sample for sample in profile.get('samples', [])
               if str(sample.get('split', 'DEV')).upper() in ('DEV', 'VALIDATION')]
    names = {}
    known = {sample['sample_id'] for sample in samples}
    if len(known) != len(samples):
        raise ValueError('duplicate sample_id in frozen DEV profile')
    for sample in samples:
        name = Path(sample.get('image') or sample.get('input_relpath', '')).name
        names.setdefault(name, []).append(sample['sample_id'])
    rows, seen = [], set()
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            sid = row.get('sample_id')
            if not sid:
                name = row.get('image', '')
                candidates = ([sample['sample_id'] for sample in samples if sample.get('input_relpath') == row['input_relpath']]
                              if row.get('input_relpath') else names.get(Path(name).name, []))
                if samples and len(candidates) != 1:
                    raise ValueError(f'Ambiguous or missing sample mapping: {name}')
                if not samples:
                    raise ValueError('Legacy basename CSV requires a unique manifest/profile mapping')
                sid = candidates[0]
            if sid in seen:
                raise ValueError(f'duplicate sample_id: {sid}')
            if known and sid not in known:
                raise ValueError(f'sample_id outside frozen DEV profile: {sid}')
            seen.add(sid)
            normalized = {**row, 'sample_id': sid}
            for key in ('psnr', 'ssim'):
                if row.get(key) not in (None, ''):
                    value = float(row[key])
                    if not math.isfinite(value):
                        raise ValueError(f'nonfinite {key}: {sid}')
                    normalized[key] = value
            if 'psnr' not in normalized or normalized['psnr'] == '':
                raise ValueError(f'PSNR unavailable: {sid}')
            rows.append(normalized)
    count = metrics.get('num_images')
    score = metrics.get('mean_per_image_psnr', metrics.get('mean_psnr', trial.get('dev_psnr')))
    if count is not None and count != len(rows):
        return None, f'CSV count disagrees with aggregate metrics for {trial["run_id"]}'
    if type(score) in (int, float) and rows and not math.isclose(
            math.fsum(row['psnr'] for row in rows) / len(rows), score, rel_tol=0, abs_tol=1e-6):
        return None, f'CSV mean disagrees with aggregate PSNR for {trial["run_id"]}'
    return rows, None


def _roi_rows(root, state, trial, sources):
    directory = _run_dir(root, state, trial)
    metrics = trial.get('dev_metrics') or _read(directory / 'dev/metrics.json', {})
    ref = (trial.get('artifacts') or {}).get('roi_metrics_csv') or metrics.get('roi_metrics_csv')
    path = _path(directory, ref) if ref else directory / 'dev/roi_metrics.csv'
    sources[str(path)] = path.read_text(encoding='utf-8') if path.is_file() else None
    if not path.is_file():
        return None
    rows, seen = [], set()
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key = (row['sample_id'], row['roi_id'], row['metric_name'])
            if key in seen:
                raise ValueError(f'duplicate ROI metric identity: {key}')
            seen.add(key)
            row['n_pixels'] = int(row['n_pixels'])
            row['region_fraction'] = float(row['region_fraction'])
            raw = row.get('value')
            row['value'] = float(raw) if raw not in (None, '', 'null', 'None') else None
            if row['value'] is not None and not math.isfinite(row['value']):
                raise ValueError(f'nonfinite ROI metric: {key}')
            if row['n_pixels'] < 0 or not math.isfinite(row['region_fraction']):
                raise ValueError(f'invalid ROI support: {key}')
            if not row.get('effective_mask_id') or not row.get('profile_revision'):
                raise ValueError(f'ROI frozen support identity missing: {key}')
            rows.append(row)
    return rows


def _diagnostics_snapshot(root, state, trial, sources):
    directory = _run_dir(root, state, trial)
    metrics = trial.get('dev_metrics') or _read(directory / 'dev/metrics.json', {})
    ref = (trial.get('artifacts') or {}).get('diagnostics_snapshot_ref') or metrics.get('diagnostics_snapshot_ref')
    path = _path(directory, ref) if ref else directory / 'dev/diagnostics_snapshot.json'
    sources[str(path)] = path.read_text(encoding='utf-8') if path.is_file() else None
    if sources[str(path)] is None:
        return None
    snapshot = json.loads(sources[str(path)])
    if not isinstance(snapshot, dict):
        raise ValueError('runtime diagnostics snapshot must be an object')
    return snapshot


def _compare_roi(candidate, reference, current_snapshot=None, prior_snapshot=None, profile_ref=None):
    from .diagnostics import fixed_metric_definition
    required = ('profile_ref', 'profile_revision', 'source', 'transform', 'rules',
                'split_roots', 'samples', 'metric_config', 'roi_samples', 'roi_inventory')
    if candidate is None or reference is None:
        return {}, 'unavailable', 'ROI CSV unavailable'
    if not current_snapshot or not prior_snapshot or any(key not in snap for key in required for snap in (current_snapshot, prior_snapshot)):
        return {}, 'unavailable', 'frozen ROI measurement snapshot unavailable or incomplete'
    if any(current_snapshot[key] != prior_snapshot[key] for key in required + ('roi_profile_ref',)):
        return {}, 'unavailable', 'frozen ROI measurement definition/profile/mask identity mismatch'
    if profile_ref and str(Path(current_snapshot['profile_ref']).resolve()) != str(Path(profile_ref).resolve()):
        return {}, 'unavailable', 'ROI measurement snapshot does not identify the campaign frozen profile'
    inventory = current_snapshot['roi_inventory']
    expected = {(r['sample_id'], r['roi_id'], r['metric_name']): r for r in inventory}
    if not expected or len(expected) != len(inventory):
        return {}, 'unavailable', 'empty or duplicate frozen ROI inventory'
    keyed = lambda rows: {(r['sample_id'], r['roi_id'], r['metric_name']): r for r in rows}
    current, prior = keyed(candidate), keyed(reference)
    groups = sorted({key[1:] for key in expected} | {(r['roi_id'], r['metric_name']) for r in candidate + reference})
    result = {}
    for roi, metric in groups:
        keys = {key for key in set(expected) | set(current) | set(prior) if key[1:] == (roi, metric)}
        valid, incompatible, missing, missing_both, nulls = [], [], [], [], []
        for key in sorted(keys):
            a, b = current.get(key), prior.get(key)
            if a is None or b is None:
                if a is None and b is None:
                    missing_both.append(key[0])
                missing.append(key[0])
                continue
            definition_a = json.loads(a['metric_definition']) if a.get('metric_definition') else None
            definition_b = json.loads(b['metric_definition']) if b.get('metric_definition') else None
            expected_definition = fixed_metric_definition(current_snapshot['metric_config'], metric)
            if (key not in expected or definition_a != expected_definition or definition_a != definition_b or
                    a.get('effective_mask_id') != expected[key].get('effective_mask_id') or
                    a.get('profile_revision') != current_snapshot['profile_revision'] or
                    any(a.get(field) != b.get(field) for field in
                        ('effective_mask_id', 'profile_revision', 'n_pixels', 'region_fraction'))):
                incompatible.append(key[0])
                continue
            if a['value'] is None or b['value'] is None or a['n_pixels'] == 0 or a.get('valid_reason') not in ('valid', 'ok') or b.get('valid_reason') not in ('valid', 'ok'):
                nulls.append(key[0])
                continue
            valid.append({'sample_id': key[0], 'candidate': a['value'], 'reference': b['value'],
                          'delta': a['value'] - b['value'], 'n_pixels': a['n_pixels'],
                          'effective_mask_id': a['effective_mask_id'], 'profile_revision': a['profile_revision']})
        n = len(valid)
        macro = {field: sum(row[field] for row in valid)/n if n else None for field in ('candidate', 'reference', 'delta')}
        # Pool MSE only. Pixel-weighted PSNR/other metrics have no shared definition.
        micro = None
        if metric in ('mse', 'region_mse') and n:
            pixels = sum(row['n_pixels'] for row in valid)
            micro = {field: sum(row[field]*row['n_pixels'] for row in valid)/pixels for field in ('candidate', 'reference', 'delta')}
            micro.update({'n_pixels': pixels, 'definition': 'pixel-weighted pooled MSE on identical valid image/mask support'})
        result[f'{roi}/{metric}'] = {'roi_id': roi, 'metric_name': metric, 'n_images': n,
            'status': 'complete' if n == len(keys) else 'partial' if n else 'unavailable',
            'macro': macro, 'micro': micro, 'paired_rows': valid,
            'scope': 'full' if n == len(keys) else 'matched_subset',
            'coverage': {'expected_count': len(keys), 'matched_count': n, 'missing': missing, 'missing_both': missing_both,
                         'incompatible_support': incompatible, 'unavailable_value': nulls},
            'aggregation': 'equal-weight per-image macro; not pixels as independent samples'}
    status = 'complete' if result and all(item['status'] == 'complete' for item in result.values()) else 'partial' if any(item['n_images'] for item in result.values()) else 'unavailable'
    return result, status, None


def _curves(root, state, trial, sources):
    directory = _run_dir(root, state, trial)
    candidates = [directory / 'train/logs' / f'{trial["run_id"]}.json', directory / 'train/logs' / trial['run_id']]
    value, path = None, None
    for item in candidates:
        sources[str(item)] = item.read_text(encoding='utf-8') if item.is_file() else None
        if item.is_file():
            value, path = _read(item), item
            break
    if not value or not value.get('val_mean_psnr'):
        return {'status': 'unavailable', 'val_mean_psnr': [], 'best_epoch': None,
                'actual_epochs': None, 'train_validation_gap': None, 'source': None,
                'unavailable': ['existing training mean per-image PSNR curve unavailable']}
    scores = value['val_mean_psnr']
    if not all(type(score) in (int, float) and math.isfinite(score) for score in scores):
        return {'status': 'failed', 'source': str(path), 'val_mean_psnr': [], 'train_validation_gap': None,
                'unavailable': ['nonfinite training validation curve']}
    checkpoints = value.get('checkpoint_model_name', [])
    epochs = []
    for checkpoint in checkpoints:
        match = re.search(r'_(\d+)\.pth$', checkpoint)
        epochs.append(int(match.group(1)) if match else None)
    best = max(range(len(scores)), key=lambda i: scores[i])
    metrics_path = directory / 'train/metrics.json'
    sources[str(metrics_path)] = metrics_path.read_text(encoding='utf-8') if metrics_path.is_file() else None
    metrics = _read(metrics_path, {})
    return {'status': 'complete', 'source': str(path), 'val_mean_psnr': scores,
            'validation_epochs': epochs, 'checkpoint_paths': checkpoints,
            'best_epoch': epochs[best] if best < len(epochs) else None,
            'actual_epochs': max((epoch for epoch in epochs if epoch is not None), default=None),
            'best_checkpoint': metrics.get('best_checkpoint'), 'train_validation_gap': None,
            'tail_window': min(3, len(scores)), 'tail_delta_db': scores[-1] - scores[max(0, len(scores)-3)],
            'unavailable': ['training-side identical metric unavailable; gap not calculated']}


def _cases(candidate, comparisons, history, profile, baseline_records=None):
    rules = {'persistent_window': 3, 'persistent_fraction': 0.2, 'max_per_category': 2,
             'max_cases': 8, 'ordinary_rule': 'sample_id_order', 'tie_break': 'sample_id',
             **profile.get('rules', {}).get('case_selection', {})}
    quota = min(2, rules['max_per_category'])
    max_cases = min(8, rules['max_cases'])
    rows = comparisons.get('comparison', {}).get('paired_rows', [])
    pools = {'regression': [row['sample_id'] for row in sorted(rows, key=lambda r: (r['delta_psnr'], r['sample_id'])) if row['delta_psnr'] < 0]}
    window = history[-rules['persistent_window']:]
    low_sets = []
    for _, records in window:
        count = max(1, math.ceil(len(records) * rules['persistent_fraction']))
        low_sets.append({row['sample_id'] for row in sorted(records, key=lambda r: (r['psnr'], r['sample_id']))[:count]})
    enough = len(window) >= rules['persistent_window']
    persistent = set.intersection(*low_sets) if enough and low_sets else set()
    low_current = sorted(candidate or [], key=lambda r: (r['psnr'], r['sample_id']))
    pools['persistent_poor'] = sorted(persistent) if enough else [row['sample_id'] for row in low_current]
    pools['improvement'] = [row['sample_id'] for row in sorted(rows, key=lambda r: (-r['delta_psnr'], r['sample_id'])) if row['delta_psnr'] > 0]
    baseline_rows = baseline_records if baseline_records is not None else next((records for _, records in history), [])
    ordinary = profile.get('rules', {}).get('case_selection', {}).get('ordinary_sample_ids')
    if ordinary is None:
        ordered = sorted(baseline_rows, key=lambda r: (r['psnr'], r['sample_id']))
        # Identity is frozen from baseline, before seeing this candidate.
        middle = ordered[len(ordered)//4:len(ordered)-len(ordered)//4] or ordered
        ordinary = sorted(row['sample_id'] for row in middle)
    pools['ordinary'] = [sid for sid in ordinary if sid in {r['sample_id'] for r in candidate or []}]
    selected, reasons = [], {}
    for category, pool in pools.items():
        added = 0
        for sid in pool:
            reason = 'current_low_score' if category == 'persistent_poor' and not enough else category
            if sid in selected:
                if reason not in reasons[sid]:
                    reasons[sid].append(reason)
                continue
            if added < quota and len(selected) < max_cases:
                selected.append(sid)
                reasons[sid] = [reason]
                added += 1
    sample_map = {sample['sample_id']: sample for sample in profile.get('samples', [])}
    candidate_map = {row['sample_id']: row for row in candidate or []}
    cards = []
    for i, sid in enumerate(selected, 1):
        sample = sample_map.get(sid, {})
        scores = {'candidate': candidate_map[sid]['psnr']}
        for role in ('baseline', 'comparison'):
            row = next((row for row in comparisons.get(role, {}).get('paired_rows', []) if row['sample_id'] == sid), None)
            scores[role] = row['reference_psnr'] if row else None
        regions = []
        for region in comparisons.get('comparison', {}).get('regions', {}).values():
            for change in region['paired_rows']:
                if change['sample_id'] == sid:
                    regions.append({**change, 'roi_id': region['roi_id'], 'metric_name': region['metric_name']})
        roots = profile.get('split_roots', {}).get(str(sample.get('split', 'DEV')).upper(), {})
        input_ref = str(_path(roots['input_dir'], sample['input_relpath'])) if roots.get('input_dir') and sample.get('input_relpath') else sample.get('input_relpath')
        gt_ref = str(_path(roots['gt_dir'], sample['gt_relpath'])) if roots.get('gt_dir') and sample.get('gt_relpath') else sample.get('gt_relpath')
        cards.append({'case_id': f'C{i:02d}', 'sample_id': sid, 'selection_reasons': reasons[sid],
                      'tags': sample.get('tags') or ['unknown'], 'scene_id': sample.get('scene_id') or 'unknown',
                      'scores': scores, 'input_ref': input_ref, 'gt_ref': gt_ref,
                      'output_ref': candidate_map[sid].get('output_image'),
                      'region_changes': regions or None, 'roi': None, 'source': 'paired_rows.csv',
                      'interpretation_limits': ['Exploratory DEV case; not causal module attribution.']})
    return {'cases': cards, 'selection_rules': rules, 'history_window': [rid for rid, _ in window],
            'persistent_poor_status': 'available' if enough else 'evidence_limited',
            'ordinary_status': 'baseline_fixed_identity' if baseline_rows else 'unavailable'}


def load_feedback_detail(feedback_dir, section, key=None):
    """Read revision-local details, with optional group/case lookup."""
    if section not in DETAIL_FILES:
        raise ValueError('Unknown feedback section')
    path = Path(feedback_dir) / DETAIL_FILES[section]
    if not path.is_file():
        raise FileNotFoundError(f'Feedback detail unavailable: {path}')
    if section == 'paired_rows':
        with path.open(newline='', encoding='utf-8') as stream:
            value = list(csv.DictReader(stream))
        for row in value:
            for name in ('candidate_psnr', 'reference_psnr', 'delta_psnr', 'candidate_ssim', 'reference_ssim', 'delta_ssim'):
                if row.get(name) not in ('', None):
                    row[name] = float(row[name])
    else:
        value = _read(path)
    if key is None:
        return value
    if section == 'cases':
        return next((case for case in value['cases'] if case['case_id'] == key or case['sample_id'] == key), None)
    for component in key.split('/'):
        value = value[component]
    return value



def _pointer_escape(value):
    return str(value).replace('~', '~0').replace('/', '~1')


def _mapped_artifact(path, mapping):
    path = Path(path)
    for original, destination in sorted(mapping.items(), key=lambda item: -len(item[0])):
        try:
            relative = path.relative_to(original)
        except ValueError:
            continue
        return Path(destination) / relative
    return path


def validate_feedback_access(feedback, observation_id=None):
    """Check actual readable immutable files and cited facts, including mapped roots."""
    failures = []
    mapping = (feedback.get('artifact_access') or {}).get('artifact_root_mapping', {})
    summary_path = _mapped_artifact(feedback.get('feedback_ref', ''), mapping)
    try:
        summary = _read(summary_path)
        if not isinstance(summary, dict):
            raise ValueError('summary unavailable')
    except (OSError, ValueError) as exc:
        return {'available': False, 'unavailable': [f'summary unavailable: {exc}']}
    for name, reference in summary.get('detail_refs', {}).items():
        path = _mapped_artifact(reference, mapping)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix == '.json':
                _read(path)
            else:
                path.read_text(encoding='utf-8')
        except (OSError, ValueError) as exc:
            failures.append(f'{name} unavailable: {exc}')
    observations = summary.get('observations', [])
    if observation_id is not None:
        observations = [item for item in observations if item.get('id', item.get('observation_id')) == observation_id]
        if not observations:
            failures.append(f'observation {observation_id} unavailable')
    for observation in observations:
        source = observation.get('source')
        try:
            if not source or '#' not in source:
                raise ValueError('observation source missing')
            filename, fragment = source.split('#', 1)
            if filename not in DETAIL_FILES.values():
                raise ValueError('observation source outside revision')
            path = summary_path.parent / filename
            value = _read(path)
            if value is None:
                raise ValueError('source unavailable')
            for part in fragment.lstrip('/').split('/') if fragment else []:
                component = part.replace('~1', '/').replace('~0', '~')
                value = value[int(component)] if isinstance(value, list) else value[component]
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            failures.append(f'observation {observation.get("id")} source unavailable: {exc}')
    return {'available': not failures, 'unavailable': failures}

def build_run_feedback(campaign_dir, run_id):
    """Build or reuse an immutable revision from persisted facts, never train."""
    root = Path(campaign_dir).resolve()
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run_id must be a local run identity')
    base = root / 'feedback' / run_id
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.feedback.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build(root, run_id, base)


def _build(root, run_id, base):
    from .diagnostics import compare_per_image, diagnostics_snapshot
    state = _read(root / 'campaign.json')
    trial = next((trial for trial in state['trials'] if trial['run_id'] == run_id), None)
    if trial is None:
        raise ValueError(f'Unknown recorded run: {run_id}')
    refs, index = _historic_references(state, trial)
    control = state.get('control') or {}
    observation_config = control.get('observation_config') or {}
    profile_ref = state.get('profile_ref') or observation_config.get('profile_ref')
    profile = {}
    sources = {}
    unavailable = []
    diagnostics_failed = False
    current_profile_snapshot = None
    if profile_ref:
        path = _path(root, profile_ref)
        if path.is_dir():
            path /= 'profile.json'
        try:
            sources[str(path)] = path.read_text(encoding='utf-8') if path.is_file() else None
            profile = json.loads(sources[str(path)]) if sources[str(path)] is not None else {}
            if not isinstance(profile, dict) or not isinstance(profile.get('samples', []), list):
                raise ValueError('profile must be an object with a sample list')
            if any(not isinstance(sample, dict) or not isinstance(sample.get('sample_id'), str)
                   or not sample['sample_id'] for sample in profile.get('samples', [])):
                raise ValueError('profile samples require nonempty sample_id values')
            if any(not isinstance(sample.get('tags', []), list)
                   or any(not isinstance(tag, str) or not tag for tag in sample.get('tags', []))
                   for sample in profile.get('samples', [])):
                raise ValueError('profile group tags must be lists of nonempty strings')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            profile = {}
            unavailable.append(f'profile unavailable: {exc}')
        if not profile:
            unavailable.append('profile unavailable')
        try:
            current_profile_snapshot = diagnostics_snapshot(profile)
        except (OSError, ValueError, TypeError, KeyError):
            pass  # Legacy labels can describe retrospective groups, without prelaunch proof.
    elif (root / 'data_manifest.json').is_file():
        profile = load_manifest(root / 'data_manifest.json')
        sources[str(root / 'data_manifest.json')] = (root / 'data_manifest.json').read_text(encoding='utf-8')
    if not profile.get('samples'):
        unavailable.append('profile groups/scenes unavailable: unknown labels')
    declared_group_tags = {tag for sample in profile.get('samples', [])
                           if str(sample.get('split', 'DEV')).upper() in ('DEV', 'VALIDATION')
                           for tag in sample.get('tags', [])
                           if isinstance(tag, str) and tag and tag.lower() != 'unknown'}
    records = {}
    history = []
    history_configs = []
    # Restrict to history at this run, so later trials cannot alter its case window.
    for item in state['trials'][:index+1]:
        if not item.get('valid'):
            continue
        try:
            rows, missing = _csv(root, state, item, profile, sources)
        except (ValueError, TypeError, KeyError) as exc:
            rows, missing = None, f'{item["run_id"]}: {exc}'
            if item['run_id'] == run_id:
                diagnostics_failed = True
        records[item['run_id']] = rows
        if missing:
            unavailable.append(missing)
        if rows is not None:
            cfg = _config(root, state, item)
            recipe = {key: cfg.get(key) for key in RECIPE_FIELDS}
            # Retain the most recent occurrence of each recipe, preserving chronology.
            if recipe in history_configs:
                position = history_configs.index(recipe)
                history_configs.pop(position)
                history.pop(position)
            history_configs.append(recipe)
            history.append((item['run_id'], rows))
    candidate = records.get(run_id)
    if not profile.get('samples') and candidate is not None:
        profile = {**profile, 'samples': [{'sample_id': row['sample_id'], 'split': 'DEV',
                   'scene_id': 'unknown', 'tags': ['unknown']} for row in candidate]}
    resolved = _config(root, state, trial)
    by_id = {item['run_id']: item for item in state['trials']}
    construction = by_id.get(refs['construction_base_run_id'])
    changes = diff_scientific_config(resolved, _config(root, state, construction)) if construction else {}
    roi_records = {}
    diagnostic_snapshots = {}
    for item in state['trials'][:index+1]:
        if item.get('valid'):
            try:
                roi_records[item['run_id']] = _roi_rows(root, state, item, sources)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                roi_records[item['run_id']] = None
                unavailable.append(f'ROI {item["run_id"]}: {exc}')
            try:
                diagnostic_snapshots[item['run_id']] = _diagnostics_snapshot(root, state, item, sources)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                diagnostic_snapshots[item['run_id']] = {}
                unavailable.append(f'diagnostics snapshot {item["run_id"]}: {exc}')
    comparisons = {}
    paired_rows = []
    for role, field in (('baseline', 'baseline_run_id'), ('best_before', 'best_before_run_id'), ('comparison', 'comparison_run_id')):
        target_id = refs.get(field)
        target = by_id.get(target_id)
        if index == 0 or not trial.get('valid'):
            comparisons[role] = {'status': 'not_applicable', 'run_id': target_id, 'comparable': False,
                                 'reason': 'baseline has no prior comparison' if index == 0 else 'invalid run has no valid paired score'}
            continue
        if target is None or not target.get('valid'):
            comparisons[role] = {'status': 'unavailable', 'run_id': target_id, 'comparable': False,
                                 'reason': 'reference unknown or invalid; no substitution'}
            unavailable.append(f'{role} reference unknown or invalid')
            continue
        delta = diff_scientific_config(resolved, _config(root, state, target))
        fixed = _fixed_changes(delta)
        protocol_match = _protocol(state, trial) == _protocol(state, target)
        if candidate is None or records.get(target_id) is None:
            result = {'status': 'unavailable', 'comparable': False, 'reason': 'per-image CSV unavailable'}
        else:
            try:
                result = compare_per_image(candidate, records[target_id], profile)
            except (ValueError, KeyError, TypeError) as exc:
                diagnostics_failed = True
                unavailable.append(f'{role}: {exc}')
                result = {'status': 'failed', 'comparable': False, 'reason': str(exc)}
        try:
            regions, roi_status, roi_reason = _compare_roi(roi_records.get(run_id), roi_records.get(target_id),
                diagnostic_snapshots.get(run_id), diagnostic_snapshots.get(target_id),
                str(_path(root, profile_ref)) if profile_ref else None)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            regions, roi_status, roi_reason = {}, 'unavailable', f'optional ROI comparison failed: {exc}'
            unavailable.append(f'{role}: {roi_reason}')
        result = {**result, 'regions': regions, 'regions_status': roi_status, 'regions_reason': roi_reason,
                  'run_id': target_id, 'config_delta': delta,
                  'fixed_scientific_changes': fixed,
                  'comparable': bool(result.get('comparable')) and not fixed and protocol_match}
        identity_changes = [key for key in fixed if key.startswith('validation.') or key in ('eval_size', 'protocol_id', 'primary_metric', 'data_manifest_ref')]
        if not protocol_match or identity_changes:
            reason = 'evaluation protocol mismatch' if not protocol_match else 'DEV sample/evaluation identity mismatch: ' + ', '.join(identity_changes)
            result.update(status='unavailable', reason=reason, comparable=False,
                          paired_rows=[], overall=None, groups={}, regions={}, regions_status='unavailable', regions_reason=reason)
            unavailable.append(f'{role}: {reason}')
        for row in result.get('paired_rows', []):
            paired_rows.append({**row, 'comparison_role': role, 'comparison_run_id': target_id})
        comparisons[role] = result
    curves = _curves(root, state, trial, sources)
    baseline_records = next((records.get(item['run_id']) for item in state['trials'][:index+1] if item.get('valid')), [])
    cases = _cases(candidate, comparisons, history, profile, baseline_records)
    applicable = trial.get('valid') and index > 0
    expected_ids = {sample['sample_id'] for sample in profile.get('samples', [])
                    if str(sample.get('split', 'DEV')).upper() in ('DEV', 'VALIDATION')}
    candidate_ids = {row['sample_id'] for row in candidate or []}
    aligned = candidate is not None and bool(candidate) and (not expected_ids or expected_ids == candidate_ids)
    if candidate is not None and expected_ids != candidate_ids:
        unavailable.append('candidate sample coverage differs from frozen DEV profile')
    comparison_groups = comparisons['comparison'].get('groups', {})
    groups_available = (bool(declared_group_tags) and comparisons['comparison'].get('status') == 'complete'
                        and all(comparison_groups.get(tag, {}).get('status') == 'complete'
                                and comparison_groups[tag].get('n_images', 0) > 0
                                and comparison_groups[tag].get('mean_paired_psnr_delta_db') is not None
                                for tag in declared_group_tags))
    section_status = {'text_summary': True, 'summary': True, 'execution': True,
                      'config_diff': not applicable or bool(construction),
                      'actual_changes': not applicable or bool(construction),
                      'sample_alignment': not trial.get('valid') or aligned,
                      'paired_comparison': not applicable or comparisons['comparison'].get('status') == 'complete',
                      'per_image_comparison': not applicable or comparisons['comparison'].get('status') == 'complete',
                      'cases': candidate is not None or not trial.get('valid'),
                      'curves': curves['status'] == 'complete',
                      'groups': not applicable or groups_available,
                      'regions': not applicable or comparisons['comparison'].get('regions_status') == 'complete'}
    required = observation_config.get('required_for_proposal', [])
    missing_required = [name for name in required if not section_status.get(name, False)] if trial.get('valid') else []
    if diagnostics_failed:
        status = 'failed'
    elif candidate is None and trial.get('valid'):
        status = 'unavailable'
    elif any(comparisons[role]['status'] in ('partial', 'failed', 'unavailable') for role in comparisons) or unavailable:
        status = 'partial'
    else:
        status = 'complete'
    run_observation_path = _run_dir(root, state, trial) / 'observation_config.json'
    sources[str(run_observation_path)] = run_observation_path.read_text(encoding='utf-8') if run_observation_path.is_file() else None
    launch_observation = trial.get('observation_config') or _read(run_observation_path, {})
    recorded_profile_ref = launch_observation.get('profile_ref')
    launch_profile_snapshot = launch_observation.get('profile_snapshot')
    runtime_profile_snapshot = diagnostic_snapshots.get(run_id)
    pretrial_profile = bool(profile_ref and recorded_profile_ref and current_profile_snapshot
                           and launch_profile_snapshot == current_profile_snapshot
                           and _path(root, recorded_profile_ref).resolve() == path.resolve()
                           and (runtime_profile_snapshot is None or runtime_profile_snapshot == current_profile_snapshot))
    provenance = {'profile_available_at_run': pretrial_profile,
                  'retrospective': bool(profile_ref and not pretrial_profile or profile.get('retrospective') or profile.get('rules', {}).get('retrospective')),
                  'reason': 'matching frozen prelaunch profile snapshot' if pretrial_profile else 'profile content availability at original decision time is not established'}
    inputs = {'assembler_schema_version': 2, 'trial': {key: val for key, val in trial.items() if key not in ('feedback_ref', 'feedback_revision', 'feedback', 'diagnostics_status', 'proposal_ready')},
              'references': refs, 'resolved_config': resolved, 'profile': profile,
              'profile_snapshot': current_profile_snapshot, 'required': required,
              'sources': sources, 'provenance': provenance, 'history': history, 'comparisons': comparisons, 'cases': cases, 'curves': curves}
    inputs = json.loads(json.dumps(inputs, allow_nan=False))
    latest = _read(base / 'latest.json', {})
    repair_of = None
    if latest and _read(base / latest['feedback_revision'] / 'source_snapshot.json') == inputs:
        try:
            prior = _read(base / latest['feedback_revision'] / 'summary.json')
            if prior and validate_feedback_access(prior)['available']:
                return prior
        except (OSError, ValueError, KeyError):
            pass
        repair_of = latest['feedback_revision']
    revisions = [int(path.name[1:]) for path in base.glob('r[0-9][0-9][0-9]*') if path.is_dir() and path.name[1:].isdigit()]
    revision = f'r{max(revisions, default=0)+1:03d}'
    directory = base / revision
    directory.mkdir()
    feedback_ref = str(directory / 'summary.json')
    campaign_id = state.get('campaign_id') or control.get('campaign_id') or root.name
    limits = ['Exploratory DEV history at one fixed seed does not demonstrate independent generalization.',
              'PNG-8 output references are display only; scores come from the recorded float evaluator.',
              'Group and region errors cannot identify a causal internal module.']
    if any(delta.get('loss_family', {}).get('from') == 'original' and delta.get('loss_family', {}).get('to') in ('mse', 'l1')
           or delta.get('loss_family', {}).get('to') == 'original' and delta.get('loss_family', {}).get('from') in ('mse', 'l1')
           for delta in [changes] + [item.get('config_delta', {}) for item in comparisons.values()]):
        limits.append('supervision_change: original uses multiple intermediate outputs and combined losses; mse/l1 supervises final sRGB only. Loss form and supervision location are confounded.')
    observations = []
    if trial.get('status') in ('completed', 'failed', 'interrupted', 'cancelled'):
        oid = 'O1'
        valid = bool(trial.get('valid'))
        observations.append({'id': oid, 'observation_id': oid,
            'evidence_ref': f'{campaign_id}/{run_id}/{revision}/{oid}',
            'metric': 'mean_per_image_rgb_psnr' if valid else 'execution_status',
            'comparison_run_id': None, 'comparison_role': None, 'group': None, 'roi': None,
            'value': trial.get('dev_psnr') if valid else None,
            'n_images': (trial.get('dev_metrics') or {}).get('num_images') if valid else None,
            'finding': f'Recorded native DEV mean per-image RGB PSNR is {trial.get("dev_psnr")} dB.' if valid else
                       f'Trial ended {trial.get("status")}: {trial.get("error") or "invalid result"}. No valid score is available.',
            'source': 'summary.json#/overall' if valid else 'summary.json#/execution',
            'case_ids': [], 'interpretation_limits': limits})
    for role, comparison in comparisons.items():
        for group, stats in [(None, (comparison.get('overall') or {}))] + list(comparison.get('groups', {}).items()):
            if stats.get('mean_paired_psnr_delta_db') is None:
                continue
            oid = f'O{len(observations)+1}'
            observations.append({'id': oid, 'observation_id': oid,
                'evidence_ref': f'{campaign_id}/{run_id}/{revision}/{oid}',
                'metric': 'mean_paired_psnr_delta_db', 'comparison_run_id': comparison['run_id'],
                'comparison_role': role, 'group': group, 'roi': None,
                'value': stats['mean_paired_psnr_delta_db'], 'n_images': stats.get('n_images'),
                'coverage': comparison.get('coverage'), 'scope': stats.get('scope', (comparison.get('overall') or {}).get('scope')),
                'finding': f'Mean paired PSNR change {stats["mean_paired_psnr_delta_db"]:+.8f} dB for {stats.get("n_images")} matched images.',
                'source': f'paired_comparison.json#/{role}/' + ('overall' if group is None else f'groups/{_pointer_escape(group)}'),
                'case_ids': [card['case_id'] for card in cases['cases'] if group is None or group in card['tags']],
                'interpretation_limits': limits + (['Fixed scientific conditions differ; not a pure recipe contrast.'] if not comparison['comparable'] else [])})
    for role, comparison in comparisons.items():
        for key, region in comparison.get('regions', {}).items():
            if region['macro']['delta'] is None:
                continue
            oid = f'O{len(observations)+1}'
            observations.append({'id': oid, 'observation_id': oid,
                'evidence_ref': f'{campaign_id}/{run_id}/{revision}/{oid}',
                'metric': region['metric_name'], 'comparison_run_id': comparison['run_id'],
                'comparison_role': role, 'group': None, 'roi': region['roi_id'],
                'value': region['macro']['delta'], 'n_images': region['n_images'],
                'coverage': region['coverage'], 'aggregation': 'macro',
                'finding': f'Equal-weight per-image {region["metric_name"]} change {region["macro"]["delta"]:+.8f} on {region["n_images"]} identical fixed ROI supports.',
                'source': f'paired_comparison.json#/{role}/regions/{_pointer_escape(key)}',
                'case_ids': [card['case_id'] for card in cases['cases'] if card['sample_id'] in {r['sample_id'] for r in region['paired_rows']}],
                'interpretation_limits': limits + ['ROI support is metric specific; pooled MSE is reported separately as micro.']})
    for observation in observations:
        observation['retrospective'] = provenance['retrospective'] if observation['comparison_run_id'] is not None else False
    detail_refs = {name: str(directory / filename) for name, filename in DETAIL_FILES.items() if name != 'summary'}
    overall = {'dev_psnr': trial.get('dev_psnr') if trial.get('valid') else None,
               'num_images': (trial.get('dev_metrics') or {}).get('num_images'),
               'primary_metric': 'mean_per_image_rgb_psnr',
               'paired': comparisons['comparison'].get('overall')}
    result = {'schema_version': 1, 'campaign_id': campaign_id, 'run_id': run_id, 'latest_run_id': run_id,
              'feedback_ref': feedback_ref, 'feedback_revision': revision, **refs,
              'quality': {'valid': bool(trial.get('valid')), 'result_status': trial.get('result_status')},
              'execution': {'status': trial.get('status'), 'error': trial.get('error'), 'logs': trial.get('logs') or {}},
              'diagnostics_status': status, 'proposal_ready': not missing_required,
              'missing_required': missing_required, 'section_status': section_status, 'section_applicability': {name: 'not_applicable' if not applicable and name in ('paired_comparison', 'per_image_comparison', 'groups', 'regions', 'actual_changes', 'config_diff') else 'applicable' for name in section_status},
              'repair_of_revision': repair_of, 'provenance': provenance,
              'actual_changed_fields': changes, 'actual_changes': changes,
              'actual_changes_status': 'not_applicable' if not applicable else 'available' if construction else 'unknown',
              'overall': overall, 'comparisons': comparisons,
              'groups': comparisons['comparison'].get('groups', {}), 'observations': observations,
              'cases': cases, 'curves': curves, 'usage': trial.get('usage') or {'status': 'unavailable', 'gpu_hours': None},
              'unavailable': sorted(set(unavailable)), 'interpretation_limits': limits,
              'detail_refs': detail_refs, 'observation_mode': control.get('observation_mode', 'text')}
    _write(directory / 'source_snapshot.json', inputs)
    _write(directory / 'paired_comparison.json', comparisons)
    _write(directory / 'cases.json', cases)
    _write(directory / 'curves.json', curves)
    details = {'artifact_root': str(root), 'profile_ref': profile_ref, 'run_dir': str(_run_dir(root, state, trial)),
               'logs': trial.get('logs') or {}, 'artifacts': {key: val for key, val in (trial.get('artifacts') or {}).items()
                     if key not in ('test_metrics', 'final_test')},
               'samples': [{key: sample.get(key) for key in ('sample_id', 'input_relpath', 'gt_relpath', 'metadata_relpath', 'annotation_source')}
                           for sample in profile.get('samples', []) if str(sample.get('split', 'DEV')).upper() in ('TRAIN', 'DEV', 'VALIDATION')],
               'split_roots': profile.get('split_roots', {}),
               'access_status': 'verified_local_revision_files', 'artifact_root_mapping': observation_config.get('artifact_root_mapping', {})}
    _write(directory / 'details_index.json', details)
    fields = ['sample_id', 'comparison_role', 'comparison_run_id', 'candidate_psnr', 'reference_psnr', 'delta_psnr',
              'candidate_ssim', 'reference_ssim', 'delta_ssim', 'scene_id', 'tags']
    with (directory / 'paired_rows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({**row, 'tags': json.dumps(row.get('tags', []))} for row in paired_rows)
    result['artifact_access'] = {'available': True, 'mode': 'local_verified_revision_files', 'artifact_root_mapping': observation_config.get('artifact_root_mapping', {})}
    _write(directory / 'summary.json', result)
    access = validate_feedback_access(result)
    if not access['available']:
        raise ValueError('Generated feedback artifact access failure: ' + '; '.join(access['unavailable']))
    latest_path = base / 'latest.json'
    tmp = base / 'latest.tmp'
    _write(tmp, {'feedback_revision': revision, 'feedback_ref': feedback_ref, 'run_id': run_id})
    tmp.replace(latest_path)
    return result



def build_fallback_feedback(campaign_dir, run_id, error):
    """Publish minimal native facts when optional evidence assembly cannot finish.

    This isolated path reads only the campaign's persisted core JSON. It does not
    parse profiles, CSVs or model artifacts, and never upgrades unavailable pairs.
    Required chapters retain normal baseline/failure applicability.
    """
    root = Path(campaign_dir).resolve()
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run_id must be a local run identity')
    base = root / 'feedback' / run_id
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.feedback.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _read(root / 'campaign.json')
        trial = next((item for item in state['trials'] if item['run_id'] == run_id), None)
        if trial is None:
            raise ValueError(f'Unknown recorded run: {run_id}')
        refs, index = _historic_references(state, trial)
        control = state.get('control') or {}
        observation_config = control.get('observation_config') or {}
        required = observation_config.get('required_for_proposal', [])
        resolved = trial.get('resolved_config') or {**state.get('config', {}), **trial.get('recipe', {})}
        construction = next((item for item in state['trials'] if item['run_id'] == refs['construction_base_run_id']), None)
        base_config = (construction or {}).get('resolved_config') or ({**state.get('config', {}), **(construction or {}).get('recipe', {})} if construction else None)
        changes = diff_scientific_config(resolved, base_config) if base_config is not None else {}
        valid = bool(trial.get('valid'))
        applicable = valid and index > 0
        sections = {'summary': True, 'text_summary': True, 'execution': True,
                    'config_diff': not applicable or construction is not None,
                    'actual_changes': not applicable or construction is not None,
                    'sample_alignment': not valid, 'paired_comparison': not applicable,
                    'per_image_comparison': not applicable, 'groups': not applicable,
                    'regions': not applicable, 'cases': not valid, 'curves': False}
        missing = [section for section in required if not sections.get(section, False)] if valid else []
        inputs = {'assembler_schema_version': 2, 'kind': 'minimal_fallback', 'error': str(error),
                  'trial': {key: value for key, value in trial.items() if key not in
                            ('feedback_ref', 'feedback_revision', 'feedback', 'diagnostics_status', 'proposal_ready')},
                  'references': refs, 'resolved_config': resolved, 'actual_changed_fields': changes,
                  'required': required, 'artifact_root_mapping': observation_config.get('artifact_root_mapping', {})}
        inputs = json.loads(json.dumps(inputs, allow_nan=False))
        latest = None
        cached_inputs = None
        repair_of = None
        repair_reason = None
        latest_path = base / 'latest.json'
        if latest_path.exists() or any(base.glob('r[0-9][0-9][0-9]*')):
            try:
                latest = _read(latest_path)
                if not isinstance(latest, dict):
                    raise ValueError('latest.json must be a readable object')
                cached_revision = latest.get('feedback_revision')
                if (not isinstance(cached_revision, str) or
                        not re.fullmatch(r'r[0-9]{3,}', cached_revision) or
                        latest.get('run_id') != run_id):
                    raise ValueError('latest.json has invalid revision/run identity')
                repair_of = cached_revision
                cached_inputs = _read(base / cached_revision / 'source_snapshot.json')
                if not isinstance(cached_inputs, dict):
                    raise ValueError('source_snapshot.json must be a readable object')
            except (OSError, ValueError, TypeError, KeyError) as exc:
                cached_inputs = None
                repair_reason = f'revision cache metadata unavailable: {exc}'
        if cached_inputs == inputs:
            try:
                prior = _read(base / latest['feedback_revision'] / 'summary.json')
                if prior and validate_feedback_access(prior)['available']:
                    return prior
                repair_reason = 'revision cache artifacts are incomplete or unreadable'
            except (OSError, ValueError, TypeError, KeyError) as exc:
                repair_reason = f'revision cache artifacts unavailable: {exc}'
        elif repair_reason is None:
            repair_of = None
        revisions = [int(path.name[1:]) for path in base.glob('r[0-9][0-9][0-9]*')
                     if path.is_dir() and path.name[1:].isdigit()]
        revision = f'r{max(revisions, default=0)+1:03d}'
        directory = base / revision
        directory.mkdir()
        campaign_id = state.get('campaign_id') or control.get('campaign_id') or root.name
        score = trial.get('dev_psnr') if valid else None
        overall = {'dev_psnr': score, 'num_images': (trial.get('dev_metrics') or {}).get('num_images'),
                   'primary_metric': 'mean_per_image_rgb_psnr', 'paired': None}
        execution = {'status': trial.get('status'), 'error': trial.get('error'), 'logs': trial.get('logs') or {}}
        limits = ['Diagnostic assembly failed; only persisted native execution and aggregate facts are available.',
                  'No paired, group, ROI or causal improvement claim is supported by this revision.']
        observation = {'id': 'O1', 'observation_id': 'O1',
                       'evidence_ref': f'{campaign_id}/{run_id}/{revision}/O1',
                       'metric': 'mean_per_image_rgb_psnr' if valid else 'execution_status',
                       'comparison_run_id': None, 'comparison_role': None, 'group': None, 'roi': None,
                       'value': score, 'n_images': overall['num_images'] if valid else None,
                       'finding': f'Recorded native DEV mean per-image RGB PSNR is {score} dB.' if valid else
                                  f'Trial ended {trial.get("status")}: {trial.get("error") or "invalid result"}; no valid score available.',
                       'source': 'summary.json#/overall' if valid else 'summary.json#/execution',
                       'case_ids': [], 'interpretation_limits': limits, 'retrospective': False}
        comparisons = {role: {'status': 'unavailable' if applicable else 'not_applicable',
                       'run_id': refs[field], 'reason': str(error) if applicable else 'no applicable prior valid comparison',
                       'comparable': False, 'paired_rows': [], 'regions': {}, 'regions_status': 'unavailable'}
                       for role, field in (('baseline', 'baseline_run_id'), ('best_before', 'best_before_run_id'), ('comparison', 'comparison_run_id'))}
        details = {'artifact_root': str(root), 'run_dir': str(_run_dir(root, state, trial)),
                   'logs': execution['logs'], 'samples': [], 'access_status': 'native_facts_only',
                   'unavailable': [str(error)]}
        cases = {'status': 'unavailable', 'cases': [], 'reason': str(error)}
        curves = {'status': 'unavailable', 'val_mean_psnr': [], 'reason': str(error), 'train_validation_gap': None}
        result = {'schema_version': 1, 'campaign_id': campaign_id, 'run_id': run_id, 'latest_run_id': run_id,
                  'feedback_revision': revision, 'feedback_ref': str(directory / 'summary.json'), **refs,
                  'feedback_kind': 'minimal_fallback', 'repair_of_revision': repair_of, 'repair_reason': repair_reason,
                  'quality': {'valid': valid, 'result_status': trial.get('result_status')},
                  'execution': execution, 'overall': overall, 'resolved_config': resolved,
                  'diagnostics_status': 'unavailable', 'assembly_error': str(error), 'proposal_ready': not missing,
                  'missing_required': missing, 'section_status': sections,
                  'section_applicability': {name: 'not_applicable' if not applicable and name in
                    ('paired_comparison', 'per_image_comparison', 'groups', 'regions', 'config_diff', 'actual_changes') else 'applicable' for name in sections},
                  'actual_changed_fields': changes, 'actual_changes': changes,
                  'actual_changes_status': 'not_applicable' if not applicable else 'available' if construction else 'unknown',
                  'comparisons': comparisons, 'groups': {}, 'cases': cases, 'curves': curves,
                  'observations': [observation], 'interpretation_limits': limits, 'unavailable': [str(error)],
                  'usage': trial.get('usage') or {'status': 'unavailable', 'gpu_hours': None},
                  'detail_refs': {name: str(directory / filename) for name, filename in DETAIL_FILES.items() if name != 'summary'},
                  'artifact_access': {'available': True, 'mode': 'local_verified_revision_files',
                                      'artifact_root_mapping': observation_config.get('artifact_root_mapping', {})},
                  'observation_mode': control.get('observation_mode', 'text')}
        for filename, value in (('source_snapshot.json', inputs), ('paired_comparison.json', comparisons),
                                ('cases.json', cases), ('curves.json', curves), ('details_index.json', details)):
            _write(directory / filename, value)
        (directory / 'paired_rows.csv').write_text('sample_id,comparison_role,comparison_run_id,candidate_psnr,reference_psnr,delta_psnr\n', encoding='utf-8')
        _write(directory / 'summary.json', result)
        access = validate_feedback_access(result)
        if not access['available']:
            raise ValueError('Minimal feedback artifact access failure: ' + '; '.join(access['unavailable']))
        tmp = base / 'latest.tmp'
        _write(tmp, {'feedback_revision': revision, 'feedback_ref': result['feedback_ref'], 'run_id': run_id})
        tmp.replace(base / 'latest.json')
        return result
