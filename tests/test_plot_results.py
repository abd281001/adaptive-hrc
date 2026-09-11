"""Protect the Pareto plot's per-retrain cost definition."""

import json

import pytest

from plot_results import load_ablation, pareto_frontier, pareto_points


def points(totals, counts):
    keys = [dict(scenario="homogeneous", baseline="full", seed=i) for i in range(len(totals))]
    accuracy = [dict(key, value=0.8) for key in keys]
    systems = [dict(key, training_total_retrain_wall_s=total, training_retrain_count=count,
                    training_estimated_fit_flops=total * 1e6)
               for key, total, count in zip(keys, totals, counts)]
    return pareto_points(accuracy, systems)


def test_pareto_cost_is_invariant_to_run_length():
    short = points([8], [4])[0]
    long = points([80], [40])[0]
    assert short["mean_retrain_time_s"] == long["mean_retrain_time_s"] == 2
    assert short["mean_fit_flops"] == long["mean_fit_flops"] == 2e6


def test_normalize_each_seed_before_averaging():
    result = points([8, 60], [4, 10])[0]
    assert result["mean_retrain_time_s"] == 4
    assert result["mean_retrain_time_se_s"] == pytest.approx(2)
    assert result["mean_fit_flops"] == 4e6
    assert result["mean_fit_flops_se"] == pytest.approx(2e6)
    assert result["n_seeds"] == 2


def test_no_retrains_has_no_defined_mean_duration():
    assert points([0], [0]) == []
    assert points([0, 8], [0, 4])[0]["n_seeds"] == 1


def test_frontier_minimizes_mean_retrain_time():
    slow = dict(mean_retrain_time_s=2, accuracy=0.8)
    fast = dict(mean_retrain_time_s=1, accuracy=0.8)
    accurate = dict(mean_retrain_time_s=3, accuracy=0.9)
    assert pareto_frontier([slow, fast, accurate]) == [fast, accurate]


def test_flop_frontier_uses_flops_instead_of_time():
    fast = dict(mean_retrain_time_s=1, mean_fit_flops=4e6, accuracy=0.8)
    fewer_flops = dict(mean_retrain_time_s=2, mean_fit_flops=2e6, accuracy=0.8)
    assert pareto_frontier([fast, fewer_flops]) == [fast]
    assert pareto_frontier([fast, fewer_flops], "mean_fit_flops") == [fewer_flops]


@pytest.fixture
def ablation(tmp_path):
    methods = {"full": None, "full_no_support": None}
    rows, reference = [], {"systems": [], "pareto_accuracy": []}
    for seed in (7, 42):
        for arm in methods:
            key = dict(scenario="homogeneous", seed=seed, baseline=arm)
            system = dict(training_total_retrain_wall_s=8, training_retrain_count=4,
                          training_estimated_fit_flops=80)
            rows.append(dict(scenario="homogeneous", seed=seed, arm=arm, live_top_1=0.99,
                             detailed_metrics={"system": system,
                                               "assist": {"overall": {"live_top_1": 0.8, "teacher_forced_top_1": 0.99}}}))
            if arm == "full":
                reference["systems"].append(dict(system, **key, training_total_retrain_wall_s=12))
                reference["pareto_accuracy"].append(dict(key, value=0.8))
    path = tmp_path / "components.json"
    path.write_text(json.dumps({"rows": rows}))
    (tmp_path / "manifest.json").write_text(json.dumps({"state": "complete", "jobs": {"components": {"return_code": 0}}}))
    return path, methods, reference


def test_ablation_uses_own_timing_and_live_robot_accuracy(ablation):
    data = load_ablation(*ablation)
    for p in pareto_points(data["pareto_accuracy"], data["systems"]):
        assert p["mean_retrain_time_s"] == 2  # Not the standard run's 3 s/retrain.
        assert p["mean_fit_flops"] == 20
        assert p["accuracy"] == 0.8
        assert p["n_seeds"] == 2


@pytest.mark.parametrize("damage", ["missing", "duplicate", "different_full"])
def test_ablation_rejects_unmatched_data(ablation, damage):
    path, methods, reference = ablation
    data = json.loads(path.read_text())
    if damage == "missing":
        data["rows"].pop()
    elif damage == "duplicate":
        data["rows"].append(data["rows"][0])
    else:
        data["rows"][0]["detailed_metrics"]["system"]["training_retrain_count"] = 5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cohort|differ"):
        load_ablation(path, methods, reference)
