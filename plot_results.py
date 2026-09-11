"""Plot a completed standard evaluation without importing the experiment code.

    venv/bin/python plot_results.py [eval_results/latest] [--window 10]
    # Add a row showing individual component ablations in each Pareto:
    venv/bin/python plot_results.py --components

Outputs: accuracy, phase/workload summary, dataset growth, retention, two Paretos,
and compute CSVs. Curves/bars show seed means +/- one standard error (not a CI).
Accuracy pools robot-turn counts within each seed; observation-only executions
remain gaps. A window pools counts over trailing executions before averaging
seeds. Retention uses matched frozen robot-turn probes during preference absence.
The memory oracle is a future-informed reference, not an upper bound.
Pareto costs are mean time and estimated FLOPs per retrain within each seed.
Paretos include adaptive, no-decay and fixed-decay arms with support disabled.
With --components, a second row shows paired component removals, read from
the same run: the ablation arms live beside the baselines, so Full is the one
cell both rows are plotted from. The standard oracle supplies accuracy only.
"""

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


METHODS = {
    "full": ("Full", "#008577", "-"),
    "no_decay": ("No decay", "#4477AA", "--"),
    "fixed": ("Fixed decay", "#CC8800", "-."),
    "memory_oracle": ("Memory oracle", "#CC2233", ":"),
}
PARETO_METHODS = {
    "full": ("Full", "#008577", "o"),
    "unpinned": ("Adaptive decay (no support)", "#555555", "P"),
    "no_decay": ("No decay (no support)", "#4477AA", "s"),
    "fixed": ("Fixed decay (no support)", "#CC8800", "^"),
}
COMPONENT_METHODS = {
    "full": PARETO_METHODS["full"],
    "full_no_semantic_fallback": ("Without semantic", "#4477AA", "s"),
    "full_no_latent_residual": ("Without latent", "#CC8800", "^"),
    "full_no_pin": ("Without latest pin", "#AA4499", "v"),
    # Removing all three is settings-identical to the deployable baseline,
    # so the roster arm is read rather than re-run under a second name.
    "unpinned": ("Without all three", "#555555", "P"),
}
SCENARIOS = {"homogeneous": "Homogeneous", "heterogeneous": "Heterogeneous", "holdout": "Holdout"}
KEYS = ("scenario", "baseline", "seed")
FIELDS = {
    "episodes": ("pair", "mode", "phase_role", "hrc_robot_correct_count",
                 "hrc_robot_turn_count", "hrc_human_turn_count",
                 "hrc_human_correction_count", "recipe_steps"),
    "diagnostics": ("active_variants", "registry_size"),
    "frozen_probes": ("pair", "checkpoint", "preference_temporal_status", "closed_loop_live_top_1"),
}


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def read_table(path, table):
    """Read only the four methods and fields needed for plotting."""
    rows = []
    if not Path(path).is_file():
        return rows
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["baseline"] not in METHODS:
                continue
            if table == "diagnostics" and row["diagnostic_type"] != "memory_compute":
                continue
            fields = (*KEYS, "event_index", *FIELDS[table])
            rows.append({key: row[key] for key in fields})
    return rows


def cell_dir(run, baseline, scenario, seed):
    """One arm's own folder for one scenario/seed."""
    return run / "baselines" / baseline / "scenarios" / scenario / "seeds" / f"{seed:010d}"


def load_run(path):
    """Load final tables from every expected seed; reject incomplete runs.

    Arms are stored in separate folders, so only the plotted ones are read.
    Per-arm summaries are taken from the merged cell the suite writes, which
    is the only artifact that spans arms.
    """
    path = Path(path).resolve()
    if read_json(path / "status.json")["state"] != "complete":
        raise ValueError(f"Run is not complete: {path}")
    data = {name: [] for name in (*FIELDS, "systems", "pareto_accuracy")}
    for job in read_json(path / "manifest.json")["expected_jobs"]:
        scenario, seed = job["scenario"], job["seed"]
        folder = path / "aggregate" / "cells" / scenario / f"{seed:010d}"
        summary = read_json(folder / "summary.json")
        if summary["state"] != "complete":
            raise ValueError(f"Cell is not complete: {folder}")
        for baseline in (*METHODS, "unpinned"):
            arm = summary["per_baseline"][baseline]
            data["systems"].append({**arm["system"], **job, "baseline": baseline})
            data["pareto_accuracy"].append({**job, "baseline": baseline, "value": arm["assist"]["overall"]["live_top_1"]})
        for baseline in METHODS:
            tables = cell_dir(path, baseline, scenario, seed) / "tables"
            for table in FIELDS:
                data[table].extend(read_table(tables / f"{table}.jsonl.gz", table))
    for row in data["episodes"]:
        assist = row["mode"] == "assist"
        row["human_actions"] = row["hrc_human_turn_count"] + row["hrc_human_correction_count"] if assist else 0
        row["assist_steps"] = row["recipe_steps"] if assist else 0
    return data


def load_arms(run, arms):
    """Per-arm system and accuracy rows from a completed run.

    Ablation arms are contributed to the same run directory as the
    deployable roster, so there is no second run to reconcile: `full` here is
    the identical cell the baselines are plotted from.
    """
    run = Path(run).resolve()
    data = {"systems": [], "pareto_accuracy": []}
    for job in read_json(run / "manifest.json")["expected_jobs"]:
        scenario, seed = job["scenario"], job["seed"]
        for arm in arms:
            folder = cell_dir(run, arm, scenario, seed)
            summary = read_json(folder / "summary.json")
            if summary["state"] != "complete":
                raise ValueError(f"Cell is not complete: {folder}")
            metrics = summary["per_baseline"][arm]
            key = dict(scenario=scenario, baseline=arm, seed=seed)
            data["systems"].append({**metrics["system"], **key})
            data["pareto_accuracy"].append(
                {**key, "value": metrics["assist"]["overall"]["live_top_1"]})
    return data


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    return (values.mean() if n else np.nan,
            values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan, n)


def aggregate(rows, numerator, denominator=None, by=()):
    """Pool counts within each seed, or average measurements if no denominator."""
    result = []
    keys = (*KEYS, *by)
    for key, group in group_rows(rows, keys).items():
        total = sum(row[denominator] for row in group) if denominator else len(group)
        value = sum(row[numerator] for row in group) / total if total else np.nan
        result.append({**dict(zip(keys, key)), "value": value})
    return result


def time_series(rows, numerator, denominator=None, window=1):
    """One point per execution; optional trailing count pooling within each seed."""
    result = []
    for key, group in group_rows(rows, KEYS).items():
        group = sorted(group, key=lambda row: row["event_index"])
        num = np.array([row[numerator] for row in group], dtype=float)
        den = np.array([row[denominator] if denominator else 1 for row in group], dtype=float)
        totals = np.convolve(den, np.ones(window), mode="full")[:len(group)]
        sums = np.convolve(num, np.ones(window), mode="full")[:len(group)]
        values = np.divide(sums, totals, out=np.full(len(group), np.nan), where=(totals > 0) & (den > 0))
        result.extend({**dict(zip(KEYS, key)), "x": row["event_index"] + 1, "value": value}
                      for row, value in zip(group, values))
    return result


def retention_changes(episodes, probes):
    """Last minus first frozen accuracy per absence, with no intervening re-exposure.

    Only preferences still active or expected to recur are eligible. Require
    at least two intervening executions; never compare different preferences.
    This measures change over the sampled interval, not total lifetime forgetting.
    """
    history = {key: sorted(row["event_index"] + 1 for row in group)
               for key, group in group_rows(episodes, (*KEYS, "pair")).items()}
    references, changes = {}, {}
    for row in sorted(probes, key=lambda r: (r["event_index"], not r["checkpoint"].startswith("pre_"))):
        key = tuple(row[k] for k in (*KEYS, "pair"))
        time = row["event_index"] + (not row["checkpoint"].startswith("pre_"))
        seen = history.get(key, [])
        index = bisect_right(seen, time)
        if not index or row["preference_temporal_status"] not in ("current_active", "old_expected_to_recur"):
            continue
        gap = (*key, seen[index - 1])
        value = row["closed_loop_live_top_1"]
        if gap not in references:
            references[gap] = (time, value)
        start, before = references[gap]
        if time - start >= 2:
            changes[gap] = {**dict(zip((*KEYS, "pair"), key)), "start_execution": start,
                            "end_execution": time, "before": before, "after": value,
                            "value": 100 * (value - before)}
    return list(changes.values())


def legend(fig, methods=METHODS):
    handles = [Line2D([], [], color=color, linestyle=style, lw=1.8, label=label)
               for label, color, style in methods.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(methods), frameon=False)


def finish(fig, output, name, methods=METHODS):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        ax.set_axisbelow(True)
    if methods is not None:
        legend(fig, methods)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_lines(ax, rows):
    support = []
    for baseline, (label, color, style) in METHODS.items():
        selected = [r for r in rows if r["baseline"] == baseline]
        groups = group_rows(selected, ("x",))
        xs = sorted(key[0] for key in groups)
        stats = np.array([mean_sem([r["value"] for r in groups[(x,)]]) for x in xs])
        if not len(xs):
            continue
        support.extend(int(n) for n in stats[:, 2] if n)
        ax.plot(xs, stats[:, 0], color=color, linestyle=style, lw=1.5, zorder=4 if baseline == "memory_oracle" else 3)
        ax.fill_between(xs, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1], color=color, alpha=0.09, linewidth=0)
    ax.set_xlabel("Task execution")
    ax.margins(x=0)
    if support:
        count = str(min(support)) if min(support) == max(support) else f"{min(support)}–{max(support)}"
        ax.text(0.98, 0.03, f"n = {count} seeds", ha="right", transform=ax.transAxes, fontsize=7, color="0.4")


def draw_bars(ax, rows, categories, field):
    for i, category in enumerate(categories):
        for j, (baseline, (_, color, _)) in enumerate(METHODS.items()):
            values = [r["value"] for r in rows if r["baseline"] == baseline and r[field] == category]
            mean, se, n = mean_sem(values)
            if not n:
                continue
            if baseline == "memory_oracle":
                ax.hlines(mean, i - 0.36, i + 0.36, color=color, linestyles=":", linewidth=1.8, zorder=4)
            else:
                x = i + (j - 1) * 0.23
                ax.bar(x, mean, width=0.21, color=color, zorder=3)
                if np.isfinite(se):
                    ax.errorbar(x, mean, yerr=se, color="0.25", capsize=2, fmt="none", linewidth=0.7, zorder=4)
    ax.set_xticks(range(len(categories)), [str(c).capitalize() for c in categories])


def plot_accuracy(episodes, scenarios, output, window):
    rows = time_series(episodes, "hrc_robot_correct_count", "hrc_robot_turn_count", window)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.1 * len(scenarios), 2.7), layout="constrained", squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        draw_lines(ax, [r for r in rows if r["scenario"] == scenario])
        ax.set(title=SCENARIOS[scenario], ylim=(0, 1.02), ylabel="Robot Top-1 accuracy")
    fig.suptitle("Per-execution accuracy" if window == 1 else f"Accuracy · {window}-execution trailing window", fontsize=10)
    finish(fig, output, "accuracy")


def plot_phases(episodes, scenarios, output):
    fig, axes = plt.subplots(2, len(scenarios), figsize=(3.1 * len(scenarios), 4.8), layout="constrained", squeeze=False)
    summaries = []
    for i, (metric, numerator, denominator) in enumerate([
        ("Robot Top-1 accuracy", "hrc_robot_correct_count", "hrc_robot_turn_count"),
        ("Human action load", "human_actions", "assist_steps"),
    ]):
        rows = [{**r, "phase_role": "overall"} for r in aggregate(episodes, numerator, denominator)]
        rows += aggregate(episodes, numerator, denominator, ("phase_role",))
        summaries.extend({**r, "metric": metric} for r in rows)
        for ax, scenario in zip(axes[i], scenarios):
            draw_bars(ax, [r for r in rows if r["scenario"] == scenario], ["overall", "climb", "settled"], "phase_role")
            ax.set(ylabel=metric, ylim=(0, 1.02))
            if i == 0:
                ax.set_title(SCENARIOS[scenario])
            else:
                ax.axhline(0.5, color="0.5", linewidth=0.7, linestyle="--")
    finish(fig, output, "phase_summary")
    return summaries


def plot_memory(memory, scenarios, output):
    fig, axes = plt.subplots(2, len(scenarios), figsize=(3.1 * len(scenarios), 4.8), layout="constrained", squeeze=False)
    for i, (field, label) in enumerate([("active_variants", "Active training workflows"), ("registry_size", "Total stored workflows")]):
        rows = time_series(memory, field)
        for ax, scenario in zip(axes[i], scenarios):
            draw_lines(ax, [r for r in rows if r["scenario"] == scenario])
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(SCENARIOS[scenario])
    finish(fig, output, "dataset_growth")


def plot_retention(changes, scenarios, output):
    rows = aggregate(changes, "value")
    scenarios = [s for s in scenarios if any(r["scenario"] == s for r in rows)]
    if not scenarios:
        return []
    for row in rows:
        row["cohort"] = "matched absence"
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.1 * len(scenarios), 2.9), layout="constrained", squeeze=False, sharey=True)
    for ax, scenario in zip(axes[0], scenarios):
        selected = [r for r in rows if r["scenario"] == scenario]
        draw_bars(ax, selected, ["matched absence"], "cohort")
        n = len({r["seed"] for r in selected})
        pairs = sum(r["scenario"] == scenario and r["baseline"] == "full" for r in changes)
        ax.set_xlabel(f"{n} seeds · {pairs} matched intervals")
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.set_title(SCENARIOS[scenario])
    axes[0, 0].set_ylabel("Accuracy change (pp)\nduring preference absence")
    finish(fig, output, "retention")
    return [{**r, "metric": "Retention change (pp)", "phase_role": "matched absence"} for r in rows]


def pareto_points(accuracy, systems):
    """Pair accuracy with each seed's mean time and FLOPs per completed retrain.

    Normalize by retrain count before averaging seeds. Skipped retrains are
    excluded; a seed with no retrains has no defined per-retrain mean cost.
    """
    costs = {tuple(r[k] for k in KEYS): tuple(r[field] / r["training_retrain_count"]
             if r["training_retrain_count"] else np.nan for field in
             ("training_total_retrain_wall_s", "training_estimated_fit_flops")) for r in systems}
    points = []
    for (scenario, baseline), rows in group_rows(accuracy, ("scenario", "baseline")).items():
        samples = np.array([(*costs[tuple(r[k] for k in KEYS)], r["value"]) for r in rows])
        samples = samples[np.isfinite(samples).all(axis=1)]
        time, time_se, n = mean_sem(samples[:, 0])
        flops, flops_se, _ = mean_sem(samples[:, 1])
        score, score_se, _ = mean_sem(samples[:, 2])
        if n:
            points.append(dict(scenario=scenario, baseline=baseline, mean_retrain_time_s=time,
                               mean_retrain_time_se_s=time_se, mean_fit_flops=flops, mean_fit_flops_se=flops_se,
                               accuracy=score, accuracy_se=score_se, n_seeds=n))
    return points


def pareto_frontier(points, cost="mean_retrain_time_s"):
    """Minimize the chosen mean per-retrain cost and maximize accuracy."""
    return [p for p in points if not any(
        q[cost] <= p[cost] and q["accuracy"] >= p["accuracy"]
        and (q[cost] < p[cost] or q["accuracy"] > p["accuracy"])
        for q in points)]


def plot_pareto(comparisons, scenarios, output, flops=False, oracle=()):
    """One row per comparison, with its own measured Full reference and frontier."""
    cost = "mean_fit_flops" if flops else "mean_retrain_time_s"
    error = "mean_fit_flops_se" if flops else "mean_retrain_time_se_s"
    scale = 1e6 if flops else 0.001
    name = "pareto_flops" if flops else "pareto"
    fig = plt.figure(figsize=(3.1 * len(scenarios), 3.3 * len(comparisons)), layout="constrained")
    fig.suptitle("Mean ± SE across seeds · rings mark empirical Pareto frontiers", fontsize=9, color="0.35")
    panels = np.atleast_1d(fig.subfigures(len(comparisons), 1))
    exported = []
    for panel, (group, title, methods, points) in zip(panels, comparisons):
        points = [p for p in points if p["baseline"] in methods]
        exported.extend({"comparison": group, **p} for p in points)
        lower = [p["accuracy"] - np.nan_to_num(p["accuracy_se"]) for p in points]
        ymin = max(0, np.floor(min(lower) * 20 - 1) / 20)
        axes = panel.subplots(1, len(scenarios), squeeze=False, sharey=True)[0]
        for ax, scenario in zip(axes, scenarios):
            selected = [p for p in points if p["scenario"] == scenario]
            for p in selected:
                _, color, marker = methods[p["baseline"]]
                ax.errorbar(p[cost] / scale, p["accuracy"], xerr=p[error] / scale,
                            yerr=p["accuracy_se"], fmt=marker, color=color,
                            markersize=5, capsize=2, elinewidth=0.8, zorder=4)
            reference = [r["value"] for r in oracle if r["scenario"] == scenario]
            if reference:
                ax.axhline(mean_sem(reference)[0], color=METHODS["memory_oracle"][1], linestyle=":", linewidth=1.2)
            frontier = sorted(pareto_frontier(selected, cost), key=lambda p: p[cost])
            ax.plot([p[cost] / scale for p in frontier], [p["accuracy"] for p in frontier],
                    "o--", color="0.5", markerfacecolor="none", markersize=10, linewidth=0.8, zorder=3)
            ax.set(title=SCENARIOS[scenario], ylim=(ymin, 1.01),
                   xlabel="Mean estimated MFLOPs per retrain" if flops else "Mean time per retrain (ms)")
            ax.margins(x=0.2)
            ax.annotate("better", xy=(0.05, 0.28), xytext=(0.24, 0.10), xycoords="axes fraction",
                        fontsize=8, color="0.4", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
        axes[0].set_ylabel("Overall robot Top-1 accuracy")
        panel.suptitle(title, fontsize=10)
        handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=5, label=label)
                   for label, c, m in methods.values()]
        if oracle:
            handles.append(Line2D([], [], color=METHODS["memory_oracle"][1], linestyle=":", label="Memory oracle"))
        panel.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    finish(fig, output, name, methods=None)
    with (output / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comparison", "scenario", "baseline", cost, error,
                                                   "accuracy", "accuracy_se", "n_seeds"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(exported)


def write_summary(rows, path, keys):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*keys, "mean", "standard_error", "n_seeds"])
        for key, group in sorted(group_rows(rows, keys).items()):
            writer.writerow([*key, *mean_sem([r["value"] for r in group])])


def plot_run(run, output, window=1, ablations=None):
    if window < 1:
        raise ValueError("window must be at least 1")
    data = load_run(run)
    comparisons = [("baselines", "Decay baselines · support = semantic + latent + latest pin", PARETO_METHODS,
                    pareto_points(data["pareto_accuracy"], data["systems"]))]
    if ablations:
        components = load_arms(run, COMPONENT_METHODS)
        comparisons.append(("components", "Component ablation · adaptive decay throughout", COMPONENT_METHODS,
                            pareto_points(components["pareto_accuracy"], components["systems"])))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in SCENARIOS if any(r["scenario"] == s for r in data["episodes"])]
    style = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
             "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
             "ytick.labelsize": 8, "legend.fontsize": 9, "axes.linewidth": 0.7,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        plot_accuracy(data["episodes"], scenarios, output, window)
        summary = plot_phases(data["episodes"], scenarios, output)
        plot_memory(data["diagnostics"], scenarios, output)
        changes = retention_changes(data["episodes"], data["frozen_probes"])
        summary += plot_retention(changes, scenarios, output)
        oracle = [r for r in data["pareto_accuracy"] if r["baseline"] == "memory_oracle"]
        for flops in (False, True):
            plot_pareto(comparisons, scenarios, output, flops, oracle)
    write_summary(summary, output / "metrics.csv", ("scenario", "baseline", "metric", "phase_role"))
    if changes:
        with (output / "retention_pairs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
            writer.writeheader()
            writer.writerows(changes)
    costs = [{**row, "metric": field, "value": row[field]} for row in data["systems"] for field in
             ("training_total_retrain_wall_s", "training_fit_wall_s", "p95_retrain_fit_wall_s", "training_estimated_fit_flops")]
    write_summary(costs, output / "compute.csv", ("scenario", "baseline", "metric"))
    (output / "notes.txt").write_text(
        f"Run: {Path(run).resolve()}\nAccuracy window: {window} trailing executions.\n"
        f"Pareto baseline source: {Path(run).resolve()}\n"
        f"Pareto component source: {'component arms of this run' if ablations else 'not included'}\n"
        "Means +/- one standard error across seeds; missing values are not zero.\n"
        "Curves use available seeds at each execution; their support range is labelled.\n"
        "Unequal seed durations change the contributing cohort near the end of a curve.\n"
        "Overall and phase metrics pool robot/action counts within each seed.\n"
        "Human action load excludes observation-only demonstrations; 0.5 is the ideal even-length reference.\n"
        "Phase boundaries vary across seeds; phase results are therefore shown separately.\n"
        "Memory counts are workflows, not RAM/VRAM bytes. RAM/VRAM is not recorded in these tables.\n"
        "Compute: end-to-end retraining time, fit time, per-seed p95 fit latency, estimated fit-only FLOPs.\n"
        "Pareto: robot accuracy versus mean end-to-end time per completed retrain, shown separately by scenario.\n"
        "Per seed, mean retrain time = training_total_retrain_wall_s / training_retrain_count; skipped retrains are excluded.\n"
        "Pareto points average these per-retrain means across seeds; error bars are one SE across seeds on both axes.\n"
        "Standard No decay and Fixed decay disable semantic fallback, latent residual, and latest pin; this is not a retention-only comparison.\n"
        "The added unpinned arm is adaptive decay with all three support components disabled, not just the pin.\n"
        "Compare unpinned, no_decay and fixed to hold predictor support constant; Full also includes all support.\n"
        "With --components, a second Pareto row reads this run's component arms: Full and removals of semantic, latent, latest pin, and all three together.\n"
        "Each comparison uses its own suite's measured Full timing; ablation cohorts and Full accuracy/counts/FLOPs must match the run.\n"
        "Rings and dashed lines mark the empirical frontier of the displayed means; they do not establish statistical dominance.\n"
        "The oracle's dotted line is reference accuracy from the standard run; it has no cost coordinate and is excluded from frontiers.\n"
        "BC is excluded. Other figures retain the original four methods.\n"
        "FLOP Pareto: per-seed training_estimated_fit_flops / training_retrain_count, then mean across seeds.\n"
        "Pareto cost axes are linear: milliseconds and estimated MFLOPs; CSVs retain seconds and unscaled FLOPs.\n"
        "FLOPs are model-specific arithmetic estimates, not hardware measurements.\n"
        "Memory oracle uses future information and is not an upper bound; its actual values are plotted.\n"
        "Retention: last minus first frozen robot accuracy in each sampled absence, then mean within seed.\n"
        "Only active/recurrent preferences with no intervening re-exposure are matched.\n"
        f"Scenarios without matched retention probes: {', '.join(s for s in scenarios if not any(r['scenario'] == s for r in changes)) or 'none'}.\n"
        "Retention coverage is limited; this is not a complete catastrophic-forgetting evaluation.\n"
    )
    print(f"Saved figures (PDF + PNG), metrics.csv and compute.csv to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, default=Path("eval_results/latest"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--window", type=int, default=1, help="Trailing execution window for accuracy (default: raw)")
    parser.add_argument("--components", action="store_true", help="Add a Pareto row for this run's component-ablation arms.")
    args = parser.parse_args()
    plot_run(args.run, args.output, args.window, args.components)
