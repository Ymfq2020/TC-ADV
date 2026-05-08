from tc_adv.experiments.sweeps import aggregate_rows, paired_t_test


def test_aggregate_rows_computes_mean_std():
    rows = [
        {"MRR": 0.50, "TVR": 0.10, "Hits@10": 0.60},
        {"MRR": 0.52, "TVR": 0.08, "Hits@10": 0.62},
        {"MRR": 0.48, "TVR": 0.12, "Hits@10": 0.58},
    ]
    summary = aggregate_rows(rows, group_by=None)["all"]
    assert summary["n"] == 3
    assert abs(summary["MRR"]["mean"] - 0.5) < 1e-6
    assert summary["MRR"]["std"] >= 0.0
    assert summary["TVR"]["mean"] > 0.0


def test_paired_t_test_basic_significance():
    treated = [49.0, 51.0, 50.5, 50.0, 49.5]
    baseline = [47.0, 48.5, 47.5, 48.0, 47.0]
    result = paired_t_test(treated, baseline)
    assert result["n"] == 5
    assert result["mean_diff"] > 0.0
    assert 0.0 <= result["p_two_sided"] <= 1.0
    assert result["ci95_low"] < result["mean_diff"] < result["ci95_high"]


def test_paired_t_test_zero_difference():
    series = [40.0, 41.0, 42.0]
    result = paired_t_test(series, series)
    assert abs(result["mean_diff"]) < 1e-9
    assert result["p_two_sided"] > 0.5
