#!/usr/bin/env python3
"""Plot seed-level longitudinal outcomes and paired holdout statistics."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .ablations import commit_confidence_calibration, parse_commit_record
from .evaluation import (
    BASELINE_ORDER, BASELINE_STYLES, HOLDOUT, SCENARIO_SPECS,
    _mean_std,
    _value as _numeric, _write_json as write_json, compare_groups,
)


SCENARIO_PHASES = {name: spec.phases for name, spec in SCENARIO_SPECS.items()}
SCENARIO_TITLES = {name: spec.title for name, spec in SCENARIO_SPECS.items()}
BASELINE_LABELS = {name: style.label for name, style in BASELINE_STYLES.items()}
BASELINE_COLORS = {name: style.color for name, style in BASELINE_STYLES.items()}
SWITCH_FIGURE_BASELINES = (
    "full",
    "no_decay",
    "unpinned",
    "latest",
    "fixed",
    "memory_oracle",
)
SWITCH_OPERATIONS = frozenset({"addition", "removal", "swap"})
SWITCH_METRICS = (
    (
        "teacher_forced_top_1",
        "Teacher-forced Top-1",
        ("teacher_forced_correct_count",),
        "teacher_forced_prediction_count",
    ),
    (
        "live_top_1",
        "Live Top-1",
        ("hrc_robot_correct_count",),
        "hrc_robot_turn_count",
    ),
    (
        "normalized_human_action_load",
        "Normalized human action load",
        ("hrc_human_turn_count", "hrc_human_correction_count"),
        "recipe_steps",
    ),
)


def _load_manifest(root: Path) -> Mapping[str, Any]:
    manifest_path = root / "manifest.json"
    status_path = root / "status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise RuntimeError(f"Not an evaluation run: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError(f"Evaluation run is not complete: {root}")
    return manifest


def _expected_seeds(manifest: Mapping[str, Any], scenario: str) -> set[int]:
    config = manifest.get("config", {})
    if scenario not in config.get("scenarios", []):
        raise RuntimeError(f"Scenario {scenario!r} was not run")
    return {int(seed) for seed in config.get("seeds", [])}


def _load_summaries(
    root: Path, scenario: str, expected: set[int],
) -> list[Mapping[str, Any]]:
    paths = sorted((root / "scenarios" / scenario / "seeds").glob("*/summary.json"))
    summaries = [json.loads(path.read_text()) for path in paths]
    actual = {int(summary["seed"]) for summary in summaries}
    if actual != expected:
        raise RuntimeError(f"{scenario}: expected seeds {sorted(expected)}, found {sorted(actual)}")
    return sorted(summaries, key=lambda row: int(row["seed"]))


def _load_episode_rows(
    root: Path, scenario: str, expected: set[int],
) -> list[Mapping[str, Any]]:
    paths = sorted(
        (root / "scenarios" / scenario / "seeds").glob("*/tables/episodes.jsonl.gz")
    )
    if len(paths) != len(expected):
        raise RuntimeError(f"{scenario}: expected {len(expected)} episode logs, found {len(paths)}")
    rows: list[Mapping[str, Any]] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _metric_values(
    summaries: Sequence[Mapping[str, Any]], baseline: str, phase: str, field: str,
) -> list[float]:
    values = []
    for summary in summaries:
        scope = "by_holdout_phase_role" if phase.startswith("heldout_") else "by_phase_role"
        record = summary["per_baseline"][baseline]["assist"][scope].get(phase, {})
        values.append(_numeric(record.get(field)))
    return values


def _phase_training_values(
    summaries: Sequence[Mapping[str, Any]], baseline: str, phase: str, field: str,
) -> list[float]:
    values = []
    for summary in summaries:
        phase_role = phase.removeprefix("heldout_")
        record = (
            summary["per_baseline"][baseline]
            .get("training", {})
            .get("per_phase_role", {})
            .get(phase_role, {})
        )
        values.append(_numeric(record.get(field)))
    return values


def _available_baselines(summaries: Sequence[Mapping[str, Any]], phase: str) -> list[str]:
    common = set.intersection(*(set(summary["per_baseline"]) for summary in summaries))
    scope = "by_holdout_phase_role" if phase.startswith("heldout_") else "by_phase_role"
    return [
        baseline for baseline in BASELINE_ORDER
        if baseline in common
        and phase in summaries[0]["per_baseline"][baseline]["assist"][scope]
    ]


def _bar(axis: Any, x: np.ndarray, values: Sequence[float], errors: Sequence[float], baselines: Sequence[str], *, label: str, offset: float = 0.0, width: float = 0.72, alpha: float = 1.0, hatch: str = "") -> None:
    colors = [BASELINE_COLORS[baseline] for baseline in baselines]
    bars = axis.bar(x + offset, values, width=width, yerr=errors, capsize=2.0, color=colors, alpha=alpha, label=label, edgecolor="#333333", linewidth=0.25)
    for bar, baseline in zip(bars, baselines):
        bar.set_hatch(hatch)
        if baseline == "memory_oracle":
            bar.set_hatch(f"{hatch}xx")


def _format_axis(axis: Any, labels: Sequence[str], *, rotation: int = 45) -> None:
    axis.set_xticks(range(len(labels)), labels, rotation=rotation, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.20, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def _switch_aligned_records(
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
    *,
    radius: int = 3,
) -> list[Mapping[str, Any]]:
    """Align same-recipe assist exposures around authoritative schedule switches."""
    aligned: list[Mapping[str, Any]] = []
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("baseline") == baseline:
            by_seed.setdefault(int(row["seed"]), []).append(row)

    for seed, seed_rows in sorted(by_seed.items()):
        ordered = sorted(seed_rows, key=lambda row: int(row.get("event_index", -1)))
        switches: dict[tuple[str, str], int] = {}
        for row in ordered:
            if (
                bool(row.get("recipe_changed_this_phase"))
                and row.get("lifecycle_operation") in SWITCH_OPERATIONS
            ):
                phase = str(row.get("phase_id", row.get("phase_index", "unknown")))
                recipe = str(row.get("recipe", "unknown"))
                start = int(row.get("phase_start", row.get("event_index", -1)))
                switches[(phase, recipe)] = min(
                    start, switches.get((phase, recipe), start)
                )

        starts_by_recipe: dict[str, list[tuple[str, int]]] = {}
        for (phase, recipe), start in switches.items():
            starts_by_recipe.setdefault(recipe, []).append((phase, start))

        for recipe, recipe_switches in sorted(starts_by_recipe.items()):
            recipe_assists = [
                row for row in ordered
                if row.get("mode") == "assist"
                and str(row.get("recipe", "unknown")) == recipe
            ]
            ordered_switches = sorted(recipe_switches, key=lambda item: item[1])
            for switch_index, (phase, start) in enumerate(ordered_switches):
                previous_start = (
                    ordered_switches[switch_index - 1][1]
                    if switch_index > 0 else None
                )
                next_start = (
                    ordered_switches[switch_index + 1][1]
                    if switch_index + 1 < len(ordered_switches) else None
                )
                before = [
                    row for row in recipe_assists
                    if int(row.get("event_index", -1)) < start
                    and (
                        previous_start is None
                        or int(row.get("event_index", -1)) >= previous_start
                    )
                ][-max(1, int(radius)):]
                after = [
                    row for row in recipe_assists
                    if int(row.get("event_index", -1)) >= start
                    and (
                        next_start is None
                        or int(row.get("event_index", -1)) < next_start
                    )
                ][:max(1, int(radius)) + 1]
                # No pre-switch assist history means this is a new task, not an
                # identifiable within-recipe preference change.
                if not before or not after:
                    continue
                switch_id = f"{seed}|{phase}|{recipe}"
                for relative_t, row in zip(range(-len(before), 0), before):
                    aligned.append({
                        "seed": seed,
                        "switch_id": switch_id,
                        "relative_t": relative_t,
                        "row": row,
                    })
                for relative_t, row in enumerate(after):
                    aligned.append({
                        "seed": seed,
                        "switch_id": switch_id,
                        "relative_t": relative_t,
                        "row": row,
                    })
    return aligned


def _seed_cluster_ci(
    values: Sequence[float],
    *,
    key: str,
    n_resamples: int = 5000,
) -> tuple[float, float]:
    """Return a deterministic pointwise percentile CI over seed-level values."""
    samples = np.asarray(values, dtype=float)
    if not len(samples):
        return float("nan"), float("nan")
    if len(samples) == 1:
        return float(samples[0]), float(samples[0])
    rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
    indices = rng.integers(0, len(samples), size=(max(1, n_resamples), len(samples)))
    means = samples[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _switch_metric_curve(
    records: Sequence[Mapping[str, Any]],
    numerator_fields: Sequence[str],
    denominator_field: str,
    *,
    key: str,
    radius: int = 3,
) -> list[Mapping[str, Any]]:
    """Pool switches within seeds, then estimate a seed-level mean and CI."""
    points: list[Mapping[str, Any]] = []
    for relative_t in range(-max(1, int(radius)), max(1, int(radius)) + 1):
        at_t = [record for record in records if record["relative_t"] == relative_t]
        by_seed: dict[int, list[Mapping[str, Any]]] = {}
        for record in at_t:
            row = record["row"]
            if _numeric(row.get(denominator_field)) > 0.0:
                by_seed.setdefault(int(record["seed"]), []).append(record)
        seed_rates = []
        n_switches = 0
        for seed_records in by_seed.values():
            denominator = sum(
                _numeric(record["row"].get(denominator_field))
                for record in seed_records
            )
            numerator = sum(
                sum(_numeric(record["row"].get(field)) for field in numerator_fields)
                for record in seed_records
            )
            if denominator > 0.0:
                seed_rates.append(numerator / denominator)
                n_switches += len(seed_records)
        mean = float(np.mean(seed_rates)) if seed_rates else float("nan")
        low, high = _seed_cluster_ci(
            seed_rates,
            key=f"{key}|t={relative_t}",
        )
        points.append({
            "relative_t": relative_t,
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "n_seeds": len(seed_rates),
            "n_switches": n_switches,
        })
    return points


def _make_switch_aligned_figure(
    rows: Sequence[Mapping[str, Any]],
    scenario: str,
    *,
    radius: int = 3,
) -> plt.Figure:
    """Plot unsmoothed outcomes aligned to recipe-specific preference switches."""
    available = {str(row.get("baseline")) for row in rows}
    baselines = [name for name in SWITCH_FIGURE_BASELINES if name in available]
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0.0, 0.12, 1.0, 0.78))
    figure.suptitle(
        f"{SCENARIO_TITLES[scenario]} — switch-aligned adaptation",
        fontsize=15,
        fontweight="bold",
    )
    aligned_by_baseline = {
        baseline: _switch_aligned_records(rows, baseline, radius=radius)
        for baseline in baselines
    }
    support_notes = []
    legend_handles = []
    legend_labels = []
    for axis, (metric, title, numerators, denominator) in zip(axes, SWITCH_METRICS):
        panel_support = []
        panel_highs = []
        for baseline in baselines:
            points = _switch_metric_curve(
                aligned_by_baseline[baseline],
                numerators,
                denominator,
                key=f"{scenario}|{baseline}|{metric}",
                radius=radius,
            )
            valid = [point for point in points if np.isfinite(point["mean"])]
            if not valid:
                continue
            x = np.asarray([point["relative_t"] for point in valid], dtype=float)
            mean = 100.0 * np.asarray([point["mean"] for point in valid])
            low = 100.0 * np.asarray([point["ci_low"] for point in valid])
            high = 100.0 * np.asarray([point["ci_high"] for point in valid])
            color = BASELINE_COLORS[baseline]
            linestyle = "--" if baseline == "memory_oracle" else "-"
            axis.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=BASELINE_LABELS[baseline],
            )
            axis.fill_between(x, low, high, color=color, alpha=0.09, linewidth=0)
            panel_highs.extend(high[np.isfinite(high)])
            panel_support.extend(
                (int(point["n_seeds"]), int(point["n_switches"]))
                for point in valid
            )
        axis.axvline(0.0, color="#222222", linestyle=":", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Same-recipe assist exposure relative to switch")
        axis.set_ylabel("Accuracy (%)" if metric != "normalized_human_action_load" else "Human actions / recipe steps (%)")
        axis.set_xticks(range(-max(1, int(radius)), max(1, int(radius)) + 1))
        upper = (
            max(100.0, 1.05 * max(panel_highs, default=100.0))
            if metric == "normalized_human_action_load" else 100.0
        )
        axis.set_ylim(0.0, upper)
        axis.grid(alpha=0.20, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axis.get_legend_handles_labels()
        if handles and not legend_handles:
            legend_handles = handles
            legend_labels = labels
        if panel_support:
            seeds = [support[0] for support in panel_support]
            switches = [support[1] for support in panel_support]
            support_notes.append(
                f"{title}: {min(seeds)}–{max(seeds)} seeds, "
                f"{min(switches)}–{max(switches)} switches/point"
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No supported switch-aligned outcomes",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            fontsize=8,
            ncols=len(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
        )
    figure.text(
        0.01,
        0.01,
        "t=0 is the first assist exposure after a non-initial recipe preference-set change; observations and new-task "
        "introductions are excluded. Points pool decision counts across switches within each seed, then average paired "
        "seeds; bands are pointwise 95% seed-clustered bootstrap CIs. Curves stop before the next same-recipe switch.\n"
        + " | ".join(support_notes)
        + " † Clairvoyant oracle is non-deployable.",
        fontsize=8,
    )
    return figure


def _make_commit_reliability_figure(
    rows: Sequence[Mapping[str, Any]],
    scenario: str,
) -> plt.Figure:
    """Plot commit-candidate confidence against empirical correctness."""
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    available = {str(row.get("baseline")) for row in rows}
    plotted = 0
    for baseline in (name for name in BASELINE_ORDER if name in available):
        baseline_rows = [
            row for row in rows
            if row.get("baseline") == baseline and row.get("mode") == "assist"
        ]
        records = tuple(
            parse_commit_record(row, row_index=index)
            for index, row in enumerate(baseline_rows)
        )
        calibration = commit_confidence_calibration(records)
        bins = [row for row in calibration.get("bins", ()) if row.get("n")]
        if not bins:
            continue
        axis.plot(
            [float(row["mean_confidence"]) for row in bins],
            [float(row["empirical_accuracy"]) for row in bins],
            marker="o",
            linewidth=1.2,
            markersize=4,
            color=BASELINE_COLORS[baseline],
            label=f"{BASELINE_LABELS[baseline]} (n={calibration['n']})",
        )
        plotted += 1
    axis.plot((0.0, 1.0), (0.0, 1.0), color="#333333", linestyle="--", linewidth=1.0, label="Perfect calibration")
    axis.set_title(f"{SCENARIO_TITLES[scenario]} — self-training reliability")
    axis.set_xlabel("Mean commit-candidate confidence")
    axis.set_ylabel("Empirical recipe-and-variant correctness")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.20, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    if plotted:
        axis.legend(frameon=False, fontsize=8)
    else:
        axis.text(0.5, 0.5, "No scored assist decisions", ha="center", va="center", transform=axis.transAxes)
    return figure


def _make_phase_figure(
    summaries: Sequence[Mapping[str, Any]],
    scenario: str,
    phase: str,
    phase_title: str,
) -> plt.Figure:
    """Plot primary prediction and human-effort outcomes without diagnostics."""
    baselines = _available_baselines(summaries, phase)
    labels = [BASELINE_LABELS[baseline] for baseline in baselines]
    x = np.arange(len(baselines))
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0.0, 0.09, 1.0, 0.89))
    figure.suptitle(
        f"{SCENARIO_TITLES[scenario]} — {phase_title}",
        fontsize=15,
        fontweight="bold",
    )

    axis = axes[0]
    accuracy = [
        _mean_std(
            _metric_values(summaries, baseline, phase, "teacher_forced_top_1")
        )
        for baseline in baselines
    ]
    _bar(
        axis,
        x,
        [100.0 * mean for mean, _ in accuracy],
        [100.0 * standard_deviation for _, standard_deviation in accuracy],
        baselines,
        label="Teacher-forced Top-1",
    )
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Identical-prefix prediction")
    _format_axis(axis, labels)

    axis = axes[1]
    live = [
        _mean_std(_metric_values(summaries, baseline, phase, "live_top_1"))
        for baseline in baselines
    ]
    _bar(
        axis,
        x,
        [100.0 * mean for mean, _ in live],
        [100.0 * standard_deviation for _, standard_deviation in live],
        baselines,
        label="Live Top-1",
    )
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Closed-loop prediction")
    _format_axis(axis, labels)

    axis = axes[2]
    load = [
        _mean_std(
            _metric_values(
                summaries, baseline, phase, "normalized_human_action_load"
            )
        )
        for baseline in baselines
    ]
    _bar(
        axis,
        x,
        [100.0 * m for m, _ in load],
        [100.0 * s for _, s in load],
        baselines,
        label="Normalized human action load",
    )
    axis.set_ylabel("Human actions / recipe steps (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Normalized human action load")
    _format_axis(axis, labels)
    figure.text(
        0.01,
        0.01,
        "Bars are means across paired seeds; error bars are ±1 SD. Teacher-forced Top-1 uses identical "
        "ground-truth prefixes; live Top-1 is closed-loop and selection-affected. Human load counts scheduled human "
        "actions and corrections, divided by recipe steps. † Clairvoyant oracle is non-deployable.",
        fontsize=8,
    )
    return figure


def _make_hrc_phase_figure(
    summaries: Sequence[Mapping[str, Any]],
    scenario: str,
    phase: str,
    phase_title: str,
) -> plt.Figure:
    """Plot secondary corrective burden separately from primary outcomes."""
    baselines = _available_baselines(summaries, phase)
    labels = [BASELINE_LABELS[baseline] for baseline in baselines]
    x = np.arange(len(baselines))
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0.0, 0.09, 1.0, 0.89))
    figure.suptitle(
        f"{SCENARIO_TITLES[scenario]} — {phase_title}: corrective burden",
        fontsize=15,
        fontweight="bold",
    )

    corrections = [
        _mean_std(
            _metric_values(
                summaries, baseline, phase, "mean_corrections_per_task"
            )
        )
        for baseline in baselines
    ]
    _bar(
        axis,
        x,
        [mean for mean, _ in corrections],
        [standard_deviation for _, standard_deviation in corrections],
        baselines,
        label="Corrections / task",
    )
    axis.set_ylabel("Mean corrections per assist task")
    correction_ceiling = max(
        (
            mean + standard_deviation
            for mean, standard_deviation in corrections
        ),
        default=0.0,
    )
    axis.set_ylim(0.0, max(1.0, 1.25 * correction_ceiling))
    axis.set_title("Corrective burden")
    _format_axis(axis, labels)
    figure.text(
        0.01,
        0.01,
        "Bars are means across paired seeds; error bars are ±1 SD. Corrections are counted per assist task. "
        "† Clairvoyant oracle is non-deployable.",
        fontsize=8,
    )
    return figure


def _make_compute_phase_figure(
    summaries: Sequence[Mapping[str, Any]],
    scenario: str,
    phase: str,
    phase_title: str,
) -> plt.Figure:
    """Plot measured and model-specific compute diagnostics in a separate figure."""
    baselines = _available_baselines(summaries, phase)
    labels = [BASELINE_LABELS[baseline] for baseline in baselines]
    x = np.arange(len(baselines))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0.0, 0.065, 1.0, 0.91))
    figure.suptitle(
        f"{SCENARIO_TITLES[scenario]} — {phase_title}: compute diagnostics",
        fontsize=15,
        fontweight="bold",
    )

    axis = axes[1, 0]
    wall = []
    for baseline in baselines:
        numerators = _metric_values(summaries, baseline, phase, "testing_episode_wall_s")
        denominators = _metric_values(summaries, baseline, phase, "n_episodes")
        wall.append(_mean_std([
            numerator / denominator if denominator else 0.0
            for numerator, denominator in zip(numerators, denominators)
        ]))
    _bar(axis, x, [max(m, 1e-4) for m, _ in wall], [s for _, s in wall], baselines, label="End-to-end wall")
    axis.set_yscale("log")
    axis.set_ylabel("Seconds per assist episode (log)")
    axis.set_title("Measured evaluator episode runtime")
    _format_axis(axis, labels)

    axis = axes[0, 1]
    latency = [_mean_std(_metric_values(summaries, baseline, phase, "mean_prediction_wall_s")) for baseline in baselines]
    _bar(
        axis,
        x,
        [max(1000.0 * mean, 1e-4) for mean, _ in latency],
        [1000.0 * standard_deviation for _, standard_deviation in latency],
        baselines,
        label="Prediction latency",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Milliseconds per robot turn (log)")
    axis.set_title("Measured prediction latency")
    _format_axis(axis, labels)

    axis = axes[0, 0]
    train_wall = [_mean_std(_phase_training_values(summaries, baseline, phase, "online_training_total_retrain_wall_s")) for baseline in baselines]
    _bar(axis, x, [max(m, 1e-4) for m, _ in train_wall], [s for _, s in train_wall], baselines, label="Online train wall")
    axis.set_yscale("log")
    axis.set_ylabel("Seconds per phase (log)")
    axis.set_title("Measured online fitting time")
    _format_axis(axis, labels)

    axis = axes[1, 1]
    fit_ops = [_mean_std(_phase_training_values(summaries, baseline, phase, "online_training_estimated_fit_flops")) for baseline in baselines]
    _bar(
        axis,
        x,
        [max(mean / 1e9, 1e-5) for mean, _ in fit_ops],
        [
            standard_deviation / 1e9
            for _, standard_deviation in fit_ops
        ],
        baselines,
        label="Fit-op estimate",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Recorded fit-op estimate (GF, log)*")
    axis.set_title("Model-specific fitting operations")
    _format_axis(axis, labels)
    figure.text(
        0.01,
        0.005,
        "Bars are means across paired seeds; error bars are ±1 SD. Runtime panels are measured wall-clock diagnostics. "
        "* Fit-operation estimates are model-specific diagnostics, not cross-model FLOP measurements.",
        fontsize=8,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("eval_results/latest"))
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--switch-window", type=int, default=3)
    args = parser.parse_args()
    root = args.results.expanduser().resolve(strict=True)
    manifest = _load_manifest(root)
    selected = tuple(manifest.get("config", {}).get("scenarios", ()))
    unsupported = sorted(set(selected) - set(SCENARIO_PHASES))
    if unsupported:
        raise RuntimeError(f"No plotting specification for scenarios: {unsupported}")
    figures = args.figures or root / "figures" / "learning_curves"
    figures.mkdir(parents=True, exist_ok=True)
    expected_seeds = {
        scenario: _expected_seeds(manifest, scenario)
        for scenario in selected
    }

    summaries_by_scenario = {
        scenario: _load_summaries(root, scenario, expected_seeds[scenario])
        for scenario in selected
    }
    episode_rows_by_scenario = {
        scenario: _load_episode_rows(root, scenario, expected_seeds[scenario])
        for scenario in selected
    }
    pdf_path = figures / "all_figures.pdf"
    generated = []
    with PdfPages(pdf_path) as pdf:
        for scenario in selected:
            phases = SCENARIO_PHASES[scenario]
            for phase, phase_title in phases:
                figure = _make_phase_figure(
                    summaries_by_scenario[scenario], scenario, phase, phase_title,
                )
                stem = f"{scenario}__{phase}"
                png_path = figures / f"{stem}.png"
                figure.savefig(png_path, dpi=300, bbox_inches="tight")
                pdf.savefig(figure, bbox_inches="tight")
                generated.append(str(png_path))
                plt.close(figure)
                hrc_figure = _make_hrc_phase_figure(
                    summaries_by_scenario[scenario], scenario, phase, phase_title,
                )
                hrc_path = figures / f"{stem}__hrc.png"
                hrc_figure.savefig(hrc_path, dpi=300, bbox_inches="tight")
                pdf.savefig(hrc_figure, bbox_inches="tight")
                generated.append(str(hrc_path))
                plt.close(hrc_figure)
                compute_figure = _make_compute_phase_figure(
                    summaries_by_scenario[scenario], scenario, phase, phase_title,
                )
                compute_path = figures / f"{stem}__compute.png"
                compute_figure.savefig(compute_path, dpi=300, bbox_inches="tight")
                pdf.savefig(compute_figure, bbox_inches="tight")
                generated.append(str(compute_path))
                plt.close(compute_figure)
            switch_aligned = _make_switch_aligned_figure(
                episode_rows_by_scenario[scenario],
                scenario,
                radius=max(1, int(args.switch_window)),
            )
            switch_path = figures / f"{scenario}__switch_aligned.png"
            switch_aligned.savefig(switch_path, dpi=300, bbox_inches="tight")
            pdf.savefig(switch_aligned, bbox_inches="tight")
            generated.append(str(switch_path))
            plt.close(switch_aligned)
            reliability = _make_commit_reliability_figure(
                episode_rows_by_scenario[scenario],
                scenario,
            )
            reliability_path = figures / f"{scenario}__commit_reliability.png"
            reliability.savefig(reliability_path, dpi=300, bbox_inches="tight")
            pdf.savefig(reliability, bbox_inches="tight")
            generated.append(str(reliability_path))
            plt.close(reliability)

    holdout_scenarios = tuple(
        scenario for scenario in selected
        if scenario == HOLDOUT
    )
    holdout_results = {
        scenario: compare_groups(scenario, summaries_by_scenario[scenario])
        for scenario in holdout_scenarios
    }
    test_path = root / "aggregate" / "holdout_hypothesis_tests.json"
    if holdout_results:
        write_json(test_path, holdout_results)
    print("Wrote figures:")
    for path in generated:
        print(path)
    print(pdf_path)
    if holdout_results:
        print(f"Wrote hypothesis tests: {test_path}")


if __name__ == "__main__":
    main()
