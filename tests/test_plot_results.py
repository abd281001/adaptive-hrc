"""Protect the Pareto plot's per-retrain cost definition."""

import json

import pytest

from plot_results import load_arms, pareto_frontier, pareto_points


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
def run(tmp_path):
    """A run directory holding both a roster arm and an ablation arm."""
    arms = ("full", "full_no_pin")
    jobs = [dict(scenario="homogeneous", seed=seed) for seed in (7, 42)]
    (tmp_path / "manifest.json").write_text(json.dumps({"expected_jobs": jobs}))
    for job in jobs:
        for arm in arms:
            folder = (tmp_path / "baselines" / arm / "scenarios" / job["scenario"]
                      / "seeds" / f"{job['seed']:010d}")
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps({
                "state": "complete",
                "per_baseline": {arm: {
                    "system": {"training_total_retrain_wall_s": 8,
                               "training_retrain_count": 4,
                               "training_estimated_fit_flops": 80},
                    "assist": {"overall": {"live_top_1": 0.8}},
                }},
            }))
    return tmp_path, arms


def test_arms_are_read_from_their_own_cells(run):
    path, arms = run
    data = load_arms(path, arms)
    for point in pareto_points(data["pareto_accuracy"], data["systems"]):
        assert point["mean_retrain_time_s"] == 2
        assert point["mean_fit_flops"] == 20
        assert point["accuracy"] == 0.8
        assert point["n_seeds"] == 2
    assert {row["baseline"] for row in data["systems"]} == set(arms)


def test_full_is_the_same_cell_for_baselines_and_ablation_arms(run):
    """There is nothing to reconcile: one run, one Full."""
    path, arms = run
    baselines = load_arms(path, ("full",))
    components = load_arms(path, arms)
    mine = [r for r in components["systems"] if r["baseline"] == "full"]
    assert mine == baselines["systems"]


def test_an_incomplete_cell_is_rejected(run):
    path, arms = run
    cell = (path / "baselines" / "full_no_pin" / "scenarios" / "homogeneous"
            / "seeds" / "0000000007" / "summary.json")
    payload = json.loads(cell.read_text())
    payload["state"] = "running"
    cell.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not complete"):
        load_arms(path, arms)
