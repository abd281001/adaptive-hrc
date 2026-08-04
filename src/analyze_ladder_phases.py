#!/usr/bin/env python3
"""Create phase-separated figures and preference-holdout statistics.

The figures deliberately keep the five scenario plans separate.  In the two
holdout plans, the plotted phases are the *heldout* climb/settled arms;
source-progression episodes are observation-only and are therefore not a
prediction comparison.

The statistical unit is a seed-level phase aggregate.  With five paired
seeds, exact sign-flip randomization tests are more defensible than tests on
individual episodes, which would be pseudoreplication.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


SCENARIO_PHASES = {
    "ladder_heterogeneous": (("climb", "climb"), ("settled", "settled")),
    "ladder_homogeneous": (("climb", "climb"), ("settled", "settled")),
    "ladder_deployment_random": (("climb", "climb"), ("settled", "settled")),
    "axis_holdout": (("heldout_climb", "held-out climb"), ("heldout_settled", "held-out settled")),
    "preference_holdout": (("heldout_climb", "held-out climb"), ("heldout_settled", "held-out settled")),
}

SCENARIO_TITLES = {
    "ladder_heterogeneous": "Heterogeneous ladder",
    "ladder_homogeneous": "Homogeneous ladder",
    "ladder_deployment_random": "Random-deployment ladder",
    "axis_holdout": "Axis holdout",
    "preference_holdout": "Preference holdout",
}

BASELINE_ORDER = (
    "full",
    "no_decay",
    "experience_replay_bc",
    "bc",
    "adaptive_decay",
    "ewc",
    "latest_only",
    "bigram",
    "fixed_decay",
    "offline_all_recipes_identity_frozen",
    "offline_pretrained_frozen",
    "clairvoyant_memory_oracle",
)

BASELINE_LABELS = {
    "full": "Full",
    "no_decay": "No decay",
    "experience_replay_bc": "ER-BC",
    "bc": "BC",
    "adaptive_decay": "Adaptive",
    "ewc": "EWC",
    "latest_only": "Latest",
    "bigram": "Bigram",
    "fixed_decay": "Fixed",
    "offline_all_recipes_identity_frozen": "Offline-ID",
    "offline_pretrained_frozen": "Offline-pre",
    "clairvoyant_memory_oracle": "Oracle†",
}

BASELINE_COLORS = {
    "full": "#0072B2",
    "no_decay": "#009E73",
    "experience_replay_bc": "#D55E00",
    "bc": "#CC79A7",
    "adaptive_decay": "#56B4E9",
    "ewc": "#E69F00",
    "latest_only": "#999999",
    "bigram": "#8C564B",
    "fixed_decay": "#C44E52",
    "offline_all_recipes_identity_frozen": "#8172B3",
    "offline_pretrained_frozen": "#64B5CD",
    "clairvoyant_memory_oracle": "#222222",
}

DEPLOYABLE_BASELINES = tuple(
    name for name in BASELINE_ORDER
    if name not in {"full", "clairvoyant_memory_oracle"}
)


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _load_summaries(root: Path, scenario: str) -> list[Mapping[str, Any]]:
    paths = sorted((root / scenario).glob("seed_*/summary.json"))
    if len(paths) != 5:
        raise RuntimeError(f"{scenario}: expected five seed summaries, found {len(paths)}")
    summaries = [json.loads(path.read_text()) for path in paths]
    return sorted(summaries, key=lambda row: int(row["seed"]))


def _metric_values(
    summaries: Sequence[Mapping[str, Any]], baseline: str, phase: str, field: str,
) -> list[float]:
    values = []
    for summary in summaries:
        record = summary["per_baseline"][baseline]["per_phase_role"].get(phase, {})
        values.append(_numeric(record.get(field)))
    return values


def _phase_training_values(
    summaries: Sequence[Mapping[str, Any]], baseline: str, phase: str, field: str,
) -> list[float]:
    values = []
    for summary in summaries:
        record = (
            summary["per_baseline"][baseline]
            .get("phase_training_cost", {})
            .get("per_phase_role", {})
            .get(phase, {})
        )
        values.append(_numeric(record.get(field)))
    return values


def _available_baselines(summaries: Sequence[Mapping[str, Any]], phase: str) -> list[str]:
    common = set.intersection(*(set(summary["per_baseline"]) for summary in summaries))
    return [
        baseline for baseline in BASELINE_ORDER
        if baseline in common and phase in summaries[0]["per_baseline"][baseline]["per_phase_role"]
    ]


def _bar(ax: Any, x: np.ndarray, values: Sequence[float], errors: Sequence[float], baselines: Sequence[str], *, label: str, offset: float = 0.0, width: float = 0.72, alpha: float = 1.0, hatch: str = "") -> None:
    colors = [BASELINE_COLORS[baseline] for baseline in baselines]
    bars = ax.bar(x + offset, values, width=width, yerr=errors, capsize=2.0, color=colors, alpha=alpha, label=label, edgecolor="#333333", linewidth=0.25)
    for bar, baseline in zip(bars, baselines):
        bar.set_hatch(hatch)
        if baseline == "clairvoyant_memory_oracle":
            bar.set_hatch(f"{hatch}xx")


def _format_axis(ax: Any, labels: Sequence[str], *, rotation: int = 45) -> None:
    ax.set_xticks(range(len(labels)), labels, rotation=rotation, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _make_phase_figure(
    summaries: Sequence[Mapping[str, Any]],
    scenario: str,
    phase: str,
    phase_title: str,
) -> plt.Figure:
    baselines = _available_baselines(summaries, phase)
    labels = [BASELINE_LABELS[baseline] for baseline in baselines]
    x = np.arange(len(baselines))
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
    fig.suptitle(
        f"{SCENARIO_TITLES[scenario]} — {phase_title}",
        fontsize=16,
        fontweight="bold",
    )

    # (a) Exact and top-k task prediction.
    ax = axes[0, 0]
    for metric, label, offset in (("live_top1", "Top-1", -0.20), ("live_topk", "Top-3", 0.20)):
        data = [_mean_sd(_metric_values(summaries, baseline, phase, metric)) for baseline in baselines]
        _bar(ax, x, [100.0 * m for m, _ in data], [100.0 * s for _, s in data], baselines, label=label, offset=offset, width=0.38, hatch="" if metric == "live_top1" else "///")
    ax.set_ylabel("Robot-turn rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Task prediction")
    ax.legend(frameon=False, ncols=2)
    _format_axis(ax, labels)

    # (b) Error behaviour that is hidden by top-1 alone.
    ax = axes[0, 1]
    for metric, label, offset in (("human_correction_rate", "Correction", -0.20), ("future_valid_wrong_rate", "Future-valid wrong", 0.20)):
        data = [_mean_sd(_metric_values(summaries, baseline, phase, metric)) for baseline in baselines]
        _bar(ax, x, [100.0 * m for m, _ in data], [100.0 * s for _, s in data], baselines, label=label, offset=offset, width=0.38, hatch="" if metric == "human_correction_rate" else "///")
    ax.set_ylabel("Robot-turn rate (%)")
    ax.set_title("Error and correction burden")
    ax.legend(frameon=False, ncols=2)
    _format_axis(ax, labels)

    # (c) HRC cost ratio.
    ax = axes[1, 0]
    data = [_mean_sd(_metric_values(summaries, baseline, phase, "testing_normalized_interaction_cost")) for baseline in baselines]
    _bar(ax, x, [m for m, _ in data], [s for _, s in data], baselines, label="Interaction cost")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Human-only")
    ax.set_ylabel("HRC / human-only action time")
    ax.set_title("Simulated interaction cost (lower is better)")
    ax.legend(frameon=False)
    _format_axis(ax, labels)

    # (d) The two components behind simulated human-facing action time.
    ax = axes[1, 1]
    for metric, label, offset in (("testing_total_action_time", "HRC action time", -0.20), ("testing_human_effort_time", "Human effort", 0.20)):
        per_episode = []
        for baseline in baselines:
            numerators = _metric_values(summaries, baseline, phase, metric)
            denominators = _metric_values(summaries, baseline, phase, "n_episodes")
            per_episode.append(_mean_sd([n / d if d else 0.0 for n, d in zip(numerators, denominators)]))
        _bar(ax, x, [m for m, _ in per_episode], [s for _, s in per_episode], baselines, label=label, offset=offset, width=0.38, hatch="" if metric == "testing_total_action_time" else "///")
    ax.set_ylabel("Seconds per assist episode")
    ax.set_title("Simulated time and human effort")
    ax.legend(frameon=False, ncols=2)
    _format_axis(ax, labels)

    # (e) Measured runtime.  Log scales retain the very fast symbolic controls.
    ax = axes[2, 0]
    wall = []
    for baseline in baselines:
        numerators = _metric_values(summaries, baseline, phase, "testing_episode_wall_s")
        denominators = _metric_values(summaries, baseline, phase, "n_episodes")
        wall.append(_mean_sd([n / d if d else 0.0 for n, d in zip(numerators, denominators)]))
    _bar(ax, x, [max(m, 1e-4) for m, _ in wall], [s for _, s in wall], baselines, label="End-to-end wall")
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per assist episode (log)")
    ax.set_title("Measured deployment runtime")
    latency = [_mean_sd(_metric_values(summaries, baseline, phase, "mean_prediction_wall_s")) for baseline in baselines]
    right = ax.twinx()
    right.plot(x, [max(1000.0 * m, 1e-4) for m, _ in latency], color="#222222", marker="o", linewidth=1.1, markersize=3, label="Prediction latency")
    right.set_yscale("log")
    right.set_ylabel("ms / robot turn (log)")
    _format_axis(ax, labels)
    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    ax.legend(left_handles + right_handles, left_labels + right_labels, frameon=False, fontsize=8, loc="upper right")

    # (f) Online retraining work.  Fit estimates are explicitly non-comparable.
    ax = axes[2, 1]
    train_wall = [_mean_sd(_phase_training_values(summaries, baseline, phase, "online_training_total_retrain_wall_s")) for baseline in baselines]
    _bar(ax, x, [max(m, 1e-4) for m, _ in train_wall], [s for _, s in train_wall], baselines, label="Online train wall")
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per phase (log)")
    ax.set_title("Online fitting work")
    fit_ops = [_mean_sd(_phase_training_values(summaries, baseline, phase, "online_training_estimated_fit_flops")) for baseline in baselines]
    right = ax.twinx()
    right.plot(x, [max(m / 1e9, 1e-5) for m, _ in fit_ops], color="#222222", marker="D", linewidth=1.1, markersize=3, label="Fit-op estimate")
    right.set_yscale("log")
    right.set_ylabel("Recorded fit-op estimate (GF, log)*")
    _format_axis(ax, labels)
    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    ax.legend(left_handles + right_handles, left_labels + right_labels, frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.01,
        0.005,
        "† Clairvoyant oracle is non-deployable. * Fit-operation estimates are model-specific diagnostics, not cross-model FLOP measurements. "
        "Phase-level fitting includes every event in that phase; climb may include shared observation updates.",
        fontsize=8,
    )
    return fig


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    return ordered[lo] if lo == hi else ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def _bootstrap_ci(deltas: Sequence[float], *, key: str, n_draws: int = 100_000) -> tuple[float, float]:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    values = list(map(float, deltas))
    n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_draws)]
    return _quantile(draws, 0.025), _quantile(draws, 0.975)


def _sign_flip_test(deltas: Sequence[float]) -> tuple[float, float]:
    """Return exact one-sided and two-sided sign-flip randomization p-values."""
    values = [float(value) for value in deltas if abs(float(value)) > 1e-15]
    if not values:
        return 1.0, 1.0
    observed = sum(values) / len(values)
    draws = [
        sum(sign * value for sign, value in zip(signs, values)) / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    eps = 1e-15
    one_sided = sum(draw >= observed - eps for draw in draws) / len(draws)
    two_sided = sum(abs(draw) >= abs(observed) - eps for draw in draws) / len(draws)
    return one_sided, two_sided


def _holm_adjust(p_values: Mapping[str, float]) -> Mapping[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m, running, adjusted = len(ordered), 0.0, {}
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * p_value))
        adjusted[name] = running
    return adjusted


def _holdout_tests(summaries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Phase-specific, seed-paired inference for the preference holdout arm."""
    endpoints = (
        ("heldout_climb", "live_top1", "higher_is_better", "confirmatory"),
        ("heldout_settled", "live_top1", "higher_is_better", "secondary"),
        ("heldout_climb", "testing_normalized_interaction_cost", "lower_is_better", "secondary"),
        ("heldout_settled", "testing_normalized_interaction_cost", "lower_is_better", "secondary"),
    )
    tests: dict[str, Any] = {}
    for phase, metric, direction, role in endpoints:
        family: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        full_values = _metric_values(summaries, "full", phase, metric)
        for baseline in DEPLOYABLE_BASELINES:
            baseline_values = _metric_values(summaries, baseline, phase, metric)
            deltas = [
                (full - other) if direction == "higher_is_better" else (other - full)
                for full, other in zip(full_values, baseline_values)
            ]
            one_sided, two_sided = _sign_flip_test(deltas)
            ci_low, ci_high = _bootstrap_ci(deltas, key=f"preference_holdout|{phase}|{metric}|{baseline}")
            raw_p[baseline] = one_sided
            mean, sd = _mean_sd(deltas)
            family[baseline] = {
                "full_advantage_direction": direction,
                "mean_full_advantage": mean,
                "sd_full_advantage": sd,
                "paired_seed_deltas_full_advantage": {
                    str(summary["seed"]): delta for summary, delta in zip(summaries, deltas)
                },
                "paired_bootstrap_ci95": [ci_low, ci_high],
                "exact_sign_flip_p_one_sided": one_sided,
                "exact_sign_flip_p_two_sided": two_sided,
            }
        adjusted = _holm_adjust(raw_p)
        for baseline, result in family.items():
            result["holm_adjusted_p_one_sided_within_endpoint"] = adjusted[baseline]
        tests[f"{phase}/{metric}"] = {
            "role": role,
            "phase": phase,
            "metric": metric,
            "comparison_family": "Full versus 11 deployable baselines",
            "results": family,
        }
    return {
        "scenario": "preference_holdout",
        "experimental_unit": "paired seed-level phase aggregate (n=5)",
        "confirmatory_hypothesis": "Full has higher held-out-climb Top-1 than each deployable baseline.",
        "secondary_endpoints": [
            "held-out-settled Top-1",
            "held-out-climb normalized interaction cost",
            "held-out-settled normalized interaction cost",
        ],
        "test": "Exact paired sign-flip randomization test; one-sided p-values are Holm-adjusted within each endpoint.",
        "caution": "With five seeds, inference is low-powered. Bootstrap intervals are descriptive and do not replace the exact tests.",
        "endpoints": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("eval_results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("eval_results/figures/ladder_phase"))
    args = parser.parse_args()
    root = args.results_root
    figure_dir = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    summaries_by_scenario = {
        scenario: _load_summaries(root, scenario)
        for scenario in SCENARIO_PHASES
    }
    pdf_path = figure_dir / "all_ladder_phase_figures.pdf"
    generated = []
    with PdfPages(pdf_path) as pdf:
        for scenario, phases in SCENARIO_PHASES.items():
            for phase, phase_title in phases:
                figure = _make_phase_figure(
                    summaries_by_scenario[scenario], scenario, phase, phase_title,
                )
                stem = f"{scenario}__{phase}"
                png_path = figure_dir / f"{stem}.png"
                figure.savefig(png_path, dpi=300, bbox_inches="tight")
                pdf.savefig(figure, bbox_inches="tight")
                generated.append(str(png_path))
                plt.close(figure)

    preference_tests = _holdout_tests(summaries_by_scenario["preference_holdout"])
    test_path = root / "preference_holdout_hypothesis_tests.json"
    test_path.write_text(json.dumps(preference_tests, indent=2, sort_keys=True) + "\n")
    print("Wrote figures:")
    for path in generated:
        print(path)
    print(pdf_path)
    print(f"Wrote hypothesis tests: {test_path}")


if __name__ == "__main__":
    main()
