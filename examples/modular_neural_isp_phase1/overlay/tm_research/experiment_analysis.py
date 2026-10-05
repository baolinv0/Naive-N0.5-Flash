"""Read-only, preregistered statistics for paired search campaigns.

Inference concerns a fixed CONFIRM set under a symmetric location model for
campaign differences. Training seeds are averaged within a campaign; neither
images nor seeds are additional strategy repetitions. Scene generalization
requires separate scene-level records and is deliberately not fabricated here.
"""

from __future__ import annotations

import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np


_STRATEGIES = ("random", "tpe", "naive_scalar", "naive_rich", "naive_fast_slow")
_COMPARISONS = (
    ("naive_fast_slow_minus_tpe", "naive_fast_slow", "tpe"),
    ("naive_rich_minus_naive_scalar", "naive_rich", "naive_scalar"),
    ("naive_fast_slow_minus_naive_rich", "naive_fast_slow", "naive_rich"),
)
_PAIR_STATUSES = {"valid", "failed", "missing", "invalid", "inconclusive", "pending"}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _values(values):
    try:
        result = [_finite(value, "value") for value in values]
    except TypeError as exc:
        raise ValueError("values must be a nonempty numeric sequence") from exc
    if not result:
        raise ValueError("values must be a nonempty numeric sequence")
    return np.asarray(result, dtype=np.float64)


def _mean(values):
    return math.fsum(value / len(values) for value in values)


def _signs(n, seed, draws):
    seed = _integer(seed, "seed")
    draws = _integer(draws, "draws", 1)
    if n <= 20:
        numbers = np.arange(1 << n, dtype=np.uint64)
        signs = np.empty((len(numbers), n), dtype=np.int8)
        for index in range(n):
            signs[:, index] = ((numbers >> index) & 1).astype(np.int8) * 2 - 1
        return signs, "exact", 0
    signs = np.random.default_rng(seed).integers(0, 2, size=(draws, n), dtype=np.int8) * 2 - 1
    return signs, "monte_carlo", 1


def sign_flip_test(values: list[float], *, theta=0, seed=20261005, draws=100000) -> dict:
    """Two-sided mean-statistic test, including ties and MC +1 correction.

    The deterministic sign set depends only on n/seed/draws, never theta. Small
    rounding differences in mathematically tied sums use a disclosed machine
    precision tolerance after scaling, not a data-selected statistical test.
    """
    values = _values(values)
    theta = _finite(theta, "theta")
    # Subtract before rescaling: scaling x and theta separately can lose the
    # tiny residuals and break exact ties near a nonzero location. Finite
    # subtraction is especially accurate for close values (Sterbenz's lemma).
    # Use absolute scaling only when the raw subtraction actually overflows.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = values - theta
    finite_subtraction = bool(np.isfinite(centered).all())
    if not finite_subtraction:
        scale = max(float(np.abs(values).max()), abs(theta), 1.0)
        centered = values / scale - theta / scale
    centered_scale = float(np.abs(centered).max())
    normalized = centered / centered_scale if centered_scale else centered
    signs, mode, correction = _signs(len(values), seed, draws)
    observed = abs(_mean(normalized.tolist()))
    statistics = np.abs(signs @ (normalized / len(values)))
    tolerance = 64 * np.finfo(np.float64).eps * (float(np.abs(normalized).max()) if centered_scale else 0)
    extreme = int(np.count_nonzero(statistics >= observed - tolerance))
    mean_delta = _mean(centered.tolist()) if finite_subtraction else _mean(values.tolist()) - theta
    if not math.isfinite(mean_delta):
        raise ValueError("location difference is outside finite float range")
    return {
        "n": len(values), "theta": theta, "mean_delta": mean_delta,
        "statistic": abs(mean_delta), "p_value": (extreme + correction) / (len(signs) + correction),
        "mode": mode, "sample_count": len(signs), "extreme_count": extreme,
        "seed": int(seed), "draws": int(draws), "mc_plus_one": bool(correction),
        "normalized_tie_tolerance": float(tolerance),
    }


def sign_flip_interval(values, *, alpha=0.05, seed=20261005, draws=100000) -> dict:
    """Invert the same location test and return the acceptance-set hull.

    For a sign vector, a=mean(sign*x), b=mean(sign), m=mean(x).
    Its tail contribution is the closed interval between (m-a)/(1-b) and
    (a+m)/(1+b). All-same signs tie everywhere. Every such interval contains m,
    so their coverage rises toward m and falls after m. The hull boundaries
    are order statistics of interval endpoints, requiring no arbitrary grid,
    finite search range, or post-hoc choice of confidence procedure.
    """
    values = _values(values)
    alpha = _finite(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    signs, mode, correction = _signs(len(values), seed, draws)
    sign_sums = signs.sum(axis=1, dtype=np.int32)
    always = np.abs(sign_sums) == len(values)
    always_count = int(np.count_nonzero(always))
    minimum_p = (always_count + correction) / (len(signs) + correction)
    result = {
        "n": len(values), "alpha": alpha, "coverage": 1 - alpha,
        "method": "sign_flip_test_inversion_hull", "mode": mode,
        "sample_count": len(signs), "seed": int(seed), "draws": int(draws),
        "mc_plus_one": bool(correction), "tail_minimum_p": minimum_p,
        "lower": None, "upper": None, "bounds_status": "unbounded",
        "rejection_rule": "p_value <= alpha",
    }
    if minimum_p > alpha:
        return result
    # Find the smallest total tail count whose (possibly +1) p exceeds alpha.
    rank = math.floor(alpha * (len(signs) + correction) - correction - always_count) + 1
    while (always_count + rank + correction) / (len(signs) + correction) <= alpha:
        rank += 1
    while rank > 1 and (always_count + rank - 1 + correction) / (len(signs) + correction) > alpha:
        rank -= 1
    varying = signs[~always]
    b = sign_sums[~always] / len(values)
    scale = max(float(np.abs(values).max()), 1.0)
    normalized = values / scale
    m = _mean(normalized.tolist())
    # Center first to avoid cancellation in a - m*b.
    displacement = varying @ ((normalized - m) / len(values))
    first = m - displacement / (1 - b)
    second = m + displacement / (1 + b)
    left = np.minimum(first, second)
    right = np.maximum(first, second)
    lower = float(np.partition(left, rank - 1)[rank - 1] * scale)
    upper = float(np.partition(right, len(right) - rank)[len(right) - rank] * scale)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("interval endpoint is outside finite float range")
    result.update(lower=lower, upper=upper, bounds_status="bounded")
    return result


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for the entire supplied family."""
    if not isinstance(pvalues, dict):
        raise ValueError("pvalues must be a mapping")
    checked = {key: _finite(value, f"pvalues[{key}]") for key, value in pvalues.items()}
    if any(not 0 <= value <= 1 for value in checked.values()):
        raise ValueError("p-values must be in [0, 1]")
    result, running = {}, 0.0
    for index, (key, value) in enumerate(sorted(checked.items(), key=lambda item: item[1])):
        running = max(running, min(1.0, (len(checked) - index) * value))
        result[key] = running
    return {key: result[key] for key in checked}


def _seed_list(seeds, name):
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{name} must be a nonempty seed list")
    checked = [_integer(seed, name) for seed in seeds]
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} contains duplicate seeds")
    return sorted(checked)


def analyze_records(records: list[dict], *, strategies: list[str], blocks: list[int], delta_strategy=0.1) -> dict:
    """Aggregate seeds, retain the registered denominator, analyze three pairs.

    Each record uses ``pairs=[{seed,status,delta_db}]``. Only status='valid'
    enters an aggregate. Include ``expected_seeds`` in records to prove seed
    coverage; absent a registered list, an observed union is descriptive and
    cannot establish that all groups did not omit the same seed. The native
    confirmation/provenance validator must run before creating these records.
    """
    if not isinstance(strategies, list) or not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("strategies must be a nonempty unique list")
    if any(strategy not in _STRATEGIES for strategy in strategies):
        raise ValueError("unknown strategy")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("blocks must be a nonempty unique list")
    blocks = [_integer(block, "block") for block in blocks]
    if len(set(blocks)) != len(blocks):
        raise ValueError("duplicate blocks")
    delta_strategy = _finite(delta_strategy, "delta_strategy")
    if delta_strategy < 0:
        raise ValueError("delta_strategy must be nonnegative")
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    indexed, registered, observed = {}, {}, {block: set() for block in blocks}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each record must be a mapping")
        strategy, block = record.get("strategy"), _integer(record.get("block"), "block")
        if strategy not in strategies or block not in blocks:
            raise ValueError("record outside preregistered strategies/blocks")
        key = (strategy, block)
        if key in indexed:
            raise ValueError("duplicate campaign record")
        if "expected_seeds" in record:
            expected = _seed_list(record["expected_seeds"], "expected_seeds")
            if block in registered and registered[block] != expected:
                raise ValueError("conflicting registered seed lists in paired block")
            registered[block] = expected
        pairs = record.get("pairs")
        if not isinstance(pairs, list):
            raise ValueError("pairs must be a list")
        checked = {}
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError("each pair must be a mapping")
            seed = _integer(pair.get("seed"), "seed")
            if seed in checked:
                raise ValueError("duplicate seed in campaign")
            status, delta = pair.get("status"), pair.get("delta_db")
            if status not in _PAIR_STATUSES:
                raise ValueError("unknown confirmation pair status")
            if status == "valid" or delta is not None:
                delta = _finite(delta, "delta_db")
            checked[seed] = {"seed": seed, "status": status, "delta_db": delta}
            observed[block].add(seed)
        indexed[key] = checked
    expected_by_block = {}
    for block in blocks:
        expected = registered.get(block, sorted(observed[block]))
        if observed[block] - set(expected):
            raise ValueError("confirmation seed outside registered seed list")
        expected_by_block[block] = expected

    campaigns, rows_by_key = [], {}
    for strategy in strategies:
        for block in blocks:
            key, expected = (strategy, block), expected_by_block[block]
            pairs = indexed.get(key, {})
            valid = [pair["delta_db"] for pair in pairs.values() if pair["status"] == "valid"]
            missing = [seed for seed in expected if seed not in pairs or pairs[seed]["status"] != "valid"]
            complete = bool(expected) and not missing and key in indexed
            row = {
                "strategy": strategy, "block": block, "status": "complete" if complete else "incomplete",
                "expected_seeds": expected, "valid_seed_count": len(valid), "expected_seed_count": len(expected),
                "missing_seeds": missing, "record_present": key in indexed,
                "mean_delta_db": _mean(valid) if complete else None,
                "available_seed_mean_delta_db": _mean(valid) if valid else None,
                "seed_denominator_source": "registered" if block in registered else "observed_union",
                "pairs": [pairs[seed] for seed in sorted(pairs)],
            }
            campaigns.append(row)
            rows_by_key[key] = row

    comparisons = {}
    for name, left, right in _COMPARISONS:
        differences, missing_blocks = [], []
        planned = left in strategies and right in strategies
        for block in blocks:
            a, b = rows_by_key.get((left, block)), rows_by_key.get((right, block))
            if a and b and a["status"] == b["status"] == "complete":
                differences.append({"block": block, "delta_db": a["mean_delta_db"] - b["mean_delta_db"]})
            else:
                missing_blocks.append(block)
        eligible = planned and not missing_blocks and all(block in registered for block in blocks)
        values = [item["delta_db"] for item in differences]
        comparisons[name] = {
            "left": left, "right": right, "expected_blocks": list(blocks),
            "n_blocks": len(values), "missing_blocks": missing_blocks, "differences": differences,
            "mean_difference_db": _mean(values) if values else None,
            "median_difference_db": float(np.median(values)) if values else None,
            "status": "complete" if eligible else ("not_planned" if not planned else "incomplete_or_unverified"),
            "eligible_for_primary_inference": eligible,
            "test": sign_flip_test(values) if eligible else None,
            "simultaneous_interval": sign_flip_interval(values, alpha=0.05 / 3) if eligible else None,
            "pointwise_interval": sign_flip_interval(values, alpha=0.05) if eligible else None,
            "holm_p_value": None, "conclusion": "inconclusive",
        }

    family_complete = all(item["eligible_for_primary_inference"] for item in comparisons.values())
    if family_complete:
        adjusted = holm_adjust({name: item["test"]["p_value"] for name, item in comparisons.items()})
        for name, item in comparisons.items():
            item["holm_p_value"] = adjusted[name]
            lower = item["simultaneous_interval"]["lower"]
            if adjusted[name] < 0.05 and lower is not None and lower > delta_strategy + 1e-12:
                item["conclusion"] = "useful_advantage"
            elif adjusted[name] < 0.05 and lower is not None and lower > 1e-12:
                item["conclusion"] = "small_positive_difference"
            else:
                item["conclusion"] = "no_demonstrated_advantage"
    complete_count = sum(row["status"] == "complete" for row in campaigns)
    all_campaigns_complete = complete_count == len(campaigns)
    seeds_registered = len(registered) == len(blocks)
    status = "incomplete" if not all_campaigns_complete else ("complete" if seeds_registered else "unverified_seed_denominator")
    return {
        "schema_version": 1, "status": status, "strategies": list(strategies), "blocks": list(blocks),
        "delta_strategy": delta_strategy, "expected_campaign_count": len(strategies) * len(blocks),
        "complete_campaign_count": complete_count, "all_campaigns_complete": all_campaigns_complete,
        "seed_coverage_verified": seeds_registered, "primary_family_status": "complete" if family_complete else "inconclusive",
        "primary_family_alpha": 0.05, "simultaneous_interval_alpha": 0.05 / 3,
        "campaigns": campaigns, "comparisons": comparisons,
        "inference_scope": "fixed_CONFIRM_campaign_differences_under_symmetric_location_model",
        "scene_generalization": "unavailable_without_scene_level_records",
        "notes": ["No missing or failed confirmation is replaced with zero.",
                  "Training seeds are averaged within each campaign, not counted as independent strategy repetitions.",
                  "Native primary independent-CONFIRM provenance must be verified before record ingestion.",
                  "Unknown actual costs and scene-level uncertainty are not inferred from these numeric records."],
    }


def export_report(report: dict, outputdir) -> dict:
    """Write JSON/CSV and, if matplotlib is installed, a campaign-gain plot."""
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    json_path = outputdir / "analysis.json"
    json_path.write_text(serialized + "\n", encoding="utf-8")
    artifacts = {"analysis_json": str(json_path)}
    campaign_fields = ["strategy", "block", "status", "expected_seed_count", "valid_seed_count", "mean_delta_db", "available_seed_mean_delta_db", "seed_denominator_source"]
    comparison_fields = ["comparison", "left", "right", "status", "n_blocks", "expected_block_count", "mean_difference_db", "median_difference_db", "p_value", "holm_p_value", "simultaneous_lower_db", "simultaneous_upper_db", "bounds_status", "conclusion"]
    for name, fields, rows in [
        ("campaigns", campaign_fields, report["campaigns"]),
        ("comparisons", comparison_fields, [
            dict(item, comparison=name, expected_block_count=len(item["expected_blocks"]),
                 p_value=item["test"]["p_value"] if item["test"] else None,
                 simultaneous_lower_db=item["simultaneous_interval"]["lower"] if item["simultaneous_interval"] else None,
                 simultaneous_upper_db=item["simultaneous_interval"]["upper"] if item["simultaneous_interval"] else None,
                 bounds_status=item["simultaneous_interval"]["bounds_status"] if item["simultaneous_interval"] else "unavailable")
            for name, item in report["comparisons"].items()
        ]),
    ]:
        path = outputdir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        artifacts[f"{name}_csv"] = str(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        artifacts["plot_status"] = "unavailable_matplotlib_not_installed"
        return artifacts
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for position, strategy in enumerate(report["strategies"]):
        rows = [row for row in report["campaigns"] if row["strategy"] == strategy and row["mean_delta_db"] is not None]
        axes.scatter([position] * len(rows), [row["mean_delta_db"] for row in rows], label=strategy, alpha=0.8)
    axes.set_xticks(range(len(report["strategies"])), report["strategies"], rotation=15)
    axes.axhline(0, color="grey", linewidth=0.8)
    axes.set_ylabel("Independent CONFIRM gain vs paired baseline (dB)")
    axes.set_title(f"Campaign means; coverage status: {report['status']}")
    figure.tight_layout()
    plot_path = outputdir / "confirmation_gains.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    artifacts.update(confirmation_gains_png=str(plot_path), plot_status="available")
    return artifacts
