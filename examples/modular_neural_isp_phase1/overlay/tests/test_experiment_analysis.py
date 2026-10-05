"""Independent checks of the preregistered campaign-level statistics."""

import itertools
import json
import math

import numpy as np
import pytest

from tm_research.experiment_analysis import (
    analyze_records,
    export_report,
    holm_adjust,
    sign_flip_interval,
    sign_flip_test,
)


STRATEGIES = ["random", "tpe", "naive_scalar", "naive_rich", "naive_fast_slow"]


def brute_p(values, theta=0):
    centered = [value - theta for value in values]
    observed = abs(math.fsum(centered) / len(values))
    statistics = [
        abs(math.fsum(sign * value for sign, value in zip(signs, centered)) / len(values))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    count = sum(statistic >= observed - 1e-12 for statistic in statistics)
    return count / len(statistics)


@pytest.mark.parametrize("values", [[1, 2, 3], [0, 0, 0], [-1, 0, 1], [1] * 10, [0.5, -0.5, 2, 3]])
@pytest.mark.parametrize("theta", [0, 0.1, 1])
def test_exact_p_matches_independent_enumeration(values, theta):
    result = sign_flip_test(values, theta=theta)
    assert result["p_value"] == pytest.approx(brute_p(values, theta))
    assert result["mode"] == "exact"
    assert result["sample_count"] == 2 ** len(values)


def test_ties_are_included_and_all_same_signs_are_not_dropped():
    result = sign_flip_test([1] * 10)
    assert result["extreme_count"] == 2
    assert result["p_value"] == 2 / 1024
    assert sign_flip_test([0] * 10)["p_value"] == 1


def test_location_centering_preserves_ties_at_closed_interval_boundary():
    values = [
        1.000091552734375, 1.0001068115234375, 1.0001220703125,
        1.000091552734375, 0.99993896484375, 0.999969482421875,
        1.0000762939453125, 0.999847412109375, 0.99993896484375,
        1.000152587890625,
    ]
    theta = 0.9999237060546875
    assert brute_p(values, theta) == 0.017578125
    assert sign_flip_test(values, theta=theta)["p_value"] == 0.017578125
    interval = sign_flip_interval(values, alpha=0.05 / 3)
    assert interval["lower"] == theta
    assert sign_flip_test(values, theta=interval["lower"])["p_value"] > 0.05 / 3


def test_large_location_offset_does_not_change_centered_test():
    centered = np.array([6, 7, 8, 6, -4, -2, 5, -10, -4, 10], dtype=float)
    # The offset and every increment are exactly representable in float64.
    values = 2 ** 40 + centered * 2 ** -12
    theta = 2 ** 40 - 5 * 2 ** -12
    assert sign_flip_test(values.tolist(), theta=theta)["p_value"] == brute_p(centered.tolist(), -5)


def test_overflow_in_raw_subtraction_uses_scaled_fallback():
    result = sign_flip_test([1e308, -1e308], theta=-1e308)
    assert result["p_value"] == 1
    assert result["mean_delta"] == 1e308
    json.dumps(result, allow_nan=False)


def test_monte_carlo_plus_one_and_fixed_sign_set():
    values = np.linspace(-0.7, 1.9, 21)
    draws = 1301
    seed = 4
    signs = np.random.default_rng(seed).integers(0, 2, size=(draws, len(values)), dtype=np.int8) * 2 - 1
    statistics = np.abs(signs @ (values / len(values)))
    extreme = int(np.count_nonzero(statistics >= abs(values.mean()) - 1e-12))
    result = sign_flip_test(values.tolist(), draws=draws, seed=seed)
    assert result["mode"] == "monte_carlo"
    assert result["extreme_count"] == extreme
    assert result["p_value"] == (extreme + 1) / (draws + 1)
    assert result == sign_flip_test(values.tolist(), draws=draws, seed=seed)


@pytest.mark.parametrize("n", [1, 2, 5, 6])
def test_small_samples_have_unbounded_simultaneous_interval(n):
    result = sign_flip_interval(list(range(n)), alpha=0.05 / 3)
    assert result["lower"] is None
    assert result["upper"] is None
    assert result["bounds_status"] == "unbounded"
    json.dumps(result, allow_nan=False)


def test_interval_uses_same_exact_test_and_rejects_outside_boundaries():
    values = [0.11, 0.16, 0.20, 0.22, 0.26, 0.31, 0.39, 0.42, 0.52, 0.61]
    alpha = 0.05 / 3
    result = sign_flip_interval(values, alpha=alpha)
    assert result["bounds_status"] == "bounded"
    lower, upper = result["lower"], result["upper"]
    for theta in np.linspace(lower + 1e-5, upper - 1e-5, 21):
        assert brute_p(values, theta) > alpha
    assert brute_p(values, lower - 1e-5) <= alpha
    assert brute_p(values, upper + 1e-5) <= alpha
    # Independently locate boundaries by bisection of the enumerated test.
    left, right = min(values) - 1, math.fsum(values) / len(values)
    for _ in range(40):
        midpoint = (left + right) / 2
        if brute_p(values, midpoint) > alpha:
            right = midpoint
        else:
            left = midpoint
    assert lower == pytest.approx(right, abs=1e-8)
    left, right = math.fsum(values) / len(values), max(values) + 1
    for _ in range(40):
        midpoint = (left + right) / 2
        if brute_p(values, midpoint) > alpha:
            left = midpoint
        else:
            right = midpoint
    assert upper == pytest.approx(left, abs=1e-8)


def test_interval_discrete_equal_alpha_is_rejected():
    # At infinity the exact p is 2/128. Equality with alpha must not accept.
    interval = sign_flip_interval([0, 1, 2, 3, 4, 5, 6], alpha=2 / 128)
    assert interval["bounds_status"] == "bounded"
    assert interval["lower"] == 0
    assert interval["upper"] == 6
    assert brute_p([0, 1, 2, 3, 4, 5, 6], -0.001) == 2 / 128


def test_interval_handles_tied_values_and_location_scale_equivariance():
    values = [2] * 10
    interval = sign_flip_interval(values, alpha=0.05 / 3)
    assert interval["lower"] == interval["upper"] == 2
    original = [0.11, 0.16, 0.20, 0.22, 0.26, 0.31, 0.39, 0.42, 0.52, 0.61]
    a = sign_flip_interval(original)
    b = sign_flip_interval([3 * value + 4 for value in original])
    assert b["lower"] == pytest.approx(3 * a["lower"] + 4)
    assert b["upper"] == pytest.approx(3 * a["upper"] + 4)


def test_monte_carlo_interval_matches_same_fixed_test():
    values = np.linspace(-0.1, 0.7, 23).tolist()
    interval = sign_flip_interval(values, alpha=0.05, seed=13, draws=3001)
    for theta in [interval["lower"] + 1e-6, interval["upper"] - 1e-6]:
        assert sign_flip_test(values, theta=theta, seed=13, draws=3001)["p_value"] > 0.05
    for theta in [interval["lower"] - 1e-6, interval["upper"] + 1e-6]:
        assert sign_flip_test(values, theta=theta, seed=13, draws=3001)["p_value"] <= 0.05


def test_holm_hand_calculation_and_ties():
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert holm_adjust({"a": 0.4, "b": 0.4, "c": 0.9}) == {"a": 1, "b": 1, "c": 1}


def records_for(blocks):
    offsets = dict(zip(STRATEGIES, [0, 0.02, 0.01, 0.15, 0.4]))
    return [
        {
            "strategy": strategy,
            "block": block,
            "expected_seeds": [100 + 3 * block, 101 + 3 * block, 102 + 3 * block],
            "pairs": [
                {"seed": seed, "status": "valid", "delta_db": offsets[strategy] + seed / 10000}
                for seed in [100 + 3 * block, 101 + 3 * block, 102 + 3 * block]
            ],
        }
        for block in blocks
        for strategy in STRATEGIES
    ]


def test_campaign_seed_aggregation_and_three_comparison_family():
    records = records_for(range(10))
    result = analyze_records(records, strategies=STRATEGIES, blocks=list(range(10)))
    assert result["status"] == "complete"
    assert result["expected_campaign_count"] == 50
    assert len(result["comparisons"]) == 3
    comparison = result["comparisons"]["naive_fast_slow_minus_tpe"]
    assert comparison["n_blocks"] == 10  # not 30 training seeds or image count
    assert comparison["mean_difference_db"] == pytest.approx(0.38)
    assert comparison["holm_p_value"] == pytest.approx(3 * (2 / 1024))
    assert comparison["simultaneous_interval"]["alpha"] == pytest.approx(0.05 / 3)
    assert comparison["conclusion"] == "useful_advantage"
    json.dumps(result, allow_nan=False)


def test_failure_is_missing_not_zero_and_preregistered_denominator_retained():
    records = records_for(range(10))
    failed = next(record for record in records if record["strategy"] == "naive_fast_slow" and record["block"] == 2)
    failed["pairs"][0] = {"seed": 106, "status": "failed", "delta_db": None}
    result = analyze_records(records, strategies=STRATEGIES, blocks=list(range(10)))
    assert result["status"] == "incomplete"
    assert result["expected_campaign_count"] == 50
    assert result["complete_campaign_count"] == 49
    comparison = result["comparisons"]["naive_fast_slow_minus_tpe"]
    assert comparison["missing_blocks"] == [2]
    assert comparison["n_blocks"] == 9
    assert comparison["test"] is None
    assert comparison["holm_p_value"] is None
    assert comparison["conclusion"] == "inconclusive"
    assert comparison["mean_difference_db"] == pytest.approx(0.38)
    campaign = next(row for row in result["campaigns"] if row["strategy"] == "naive_fast_slow" and row["block"] == 2)
    assert campaign["mean_delta_db"] is None
    assert campaign["available_seed_mean_delta_db"] is not None


def test_missing_entire_campaign_and_unobserved_registered_seed_detected():
    records = records_for(range(2))
    records = [record for record in records if not (record["strategy"] == "tpe" and record["block"] == 0)]
    for record in records:
        if record["block"] == 1:
            record["pairs"].pop()
    result = analyze_records(records, strategies=STRATEGIES, blocks=[0, 1])
    assert result["complete_campaign_count"] == 4
    assert result["comparisons"]["naive_fast_slow_minus_tpe"]["missing_blocks"] == [0, 1]
    assert result["status"] == "incomplete"


def test_missing_seed_manifest_cannot_claim_preregistered_coverage():
    records = records_for(range(10))
    for record in records:
        record.pop("expected_seeds")
    result = analyze_records(records, strategies=STRATEGIES, blocks=list(range(10)))
    assert result["status"] == "unverified_seed_denominator"
    assert result["comparisons"]["naive_fast_slow_minus_tpe"]["conclusion"] == "inconclusive"


@pytest.mark.parametrize("mutation", ["duplicate_record", "duplicate_seed", "nan", "extra_block", "seed_manifest_conflict"])
def test_invalid_records_rejected(mutation):
    records = records_for([0])
    if mutation == "duplicate_record":
        records.append(records[0])
    elif mutation == "duplicate_seed":
        records[0]["pairs"].append(records[0]["pairs"][0])
    elif mutation == "nan":
        records[0]["pairs"][0]["delta_db"] = float("nan")
    elif mutation == "extra_block":
        records[0]["block"] = 3
    elif mutation == "seed_manifest_conflict":
        records[0]["expected_seeds"] = [100, 101]
    with pytest.raises(ValueError):
        analyze_records(records, strategies=STRATEGIES, blocks=[0])


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [True], ["1"]])
def test_invalid_numeric_inputs(values):
    with pytest.raises(ValueError):
        sign_flip_test(values)
    with pytest.raises(ValueError):
        sign_flip_interval(values)


def test_export_preserves_missing_values_in_json_and_csv(tmp_path):
    report = analyze_records(records_for([0]), strategies=STRATEGIES, blocks=[0])
    artifacts = export_report(report, tmp_path)
    loaded = json.loads((tmp_path / "analysis.json").read_text())
    assert loaded == report
    assert (tmp_path / "campaigns.csv").is_file()
    assert (tmp_path / "comparisons.csv").is_file()
    assert artifacts["analysis_json"] == str(tmp_path / "analysis.json")
    assert "Infinity" not in (tmp_path / "analysis.json").read_text()
