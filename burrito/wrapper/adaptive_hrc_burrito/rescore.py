"""Rescore a completed cooking run from its saved decision records.

This does **not** re-run anything. It reads an existing run's per-cell
checkpoints, rebuilds every episode-level quantity that is recoverable from the
stored decisions, and writes a separate derived analysis next to the original
artifacts, which are never modified.

What it corrects, all of it recoverable because every assist decision was
logged whether or not the arm answered it:

* denominators -- every assist decision counts, including the ones an arm
  failed to answer, which used to be filtered out of the rates they should have
  lowered;
* availability -- reported beside the rates instead of subtracted from them;
* correction accounting -- robot decisions come from the scheduled turns;
* opening-move metrics -- the decision at recipe step 0, which under
  ``human_first`` is never a robot turn and which no robot-turn metric can see;
* negative log likelihood -- one convention shared with the symbolic
  evaluator, ``-log(max(p, floor))`` with ``floor = min_probability``, pooled
  as total loss over total decisions.  A stored NLL is ignored entirely and
  recomputed from the stored ``ground_truth_probability``, so an unanswered
  decision recorded at the old 1e-12 floor (27.6310) is rescored to 13.8155
  rather than being dropped;
* holdout stages -- source training, axis introduction, and axis target
  composition, with recipe role and first/later exposure kept apart;
* seed-level cells and paired Full-minus-arm contrasts.

What it cannot correct, and why a rerun is still required: the candidate list
each decision was scored against was the physically-legal subset at the first
tick where the *ground truth* became executable, so it was a function of the
answer.  Fixing that changes which options the predictor saw, which no amount
of rescoring can reconstruct.  Every output here is therefore labelled a
rescoring of the original execution, not a corrected experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .task_graph import (
    CookingTaskGraph,
    is_preference_discriminating,
    is_prefix_conditioned_discriminating,
)
from .evaluation import (
    _atomic_json,
    _finite_mean,
    _holdout_stage,
    _paired_difference,
    _pooled_rate,
    _utc_now,
    FULL_ARM,
)


RESCORE_SCHEMA_VERSION = 1
DEFAULT_NLL_FLOOR = 1e-6


def _iter_cells(run_dir: Path) -> Iterable[Tuple[Path, Mapping[str, Any]]]:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.json"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints under {run_dir / 'checkpoints'}")
    for path in checkpoints:
        yield path, json.loads(path.read_text(encoding="utf-8"))


def rescore_episode(
    episode: Mapping[str, Any], *, nll_floor: float,
) -> Dict[str, Any]:
    """Rebuild one episode's counters from its stored decisions."""
    decisions = episode.get("decisions") or []
    # ``mode`` is a property of the episode; ``_decision_record`` does not
    # repeat it per decision.  Every decision of an assist episode is an assist
    # decision -- that is exactly what the runner records.
    assist = decisions if episode.get("mode") == "assist" else []

    # Ambiguity flags are deterministic functions of (recipe, prefix,
    # candidates), all three of which are stored, so they are recomputed here
    # rather than read back.  ``prefix_conditioned_discriminating`` post-dates
    # this run and is absent from its records entirely; recomputing also
    # revalidates the flag that is present.  The candidate list used is the one
    # the predictor actually saw, which is the point of a rescoring.
    graph = CookingTaskGraph.create(episode["recipe_id"])
    flags: Dict[int, Tuple[bool, bool]] = {}
    completed: List[str] = []
    for index, row in enumerate(decisions):
        legal = tuple(row.get("legal_actions") or ())
        flags[index] = (
            is_preference_discriminating(legal, graph),
            is_prefix_conditioned_discriminating(legal, graph, completed),
        )
        completed.append(row["executed_action"])
    offset = len(decisions) - len(assist)

    def discriminates(row_index: int) -> bool:
        return flags[row_index + offset][0]

    def conditioned_at(row_index: int) -> bool:
        return flags[row_index + offset][1]
    robot = [row for row in assist if row.get("scheduled_actor") == "robot"]
    answered = [row for row in assist if row.get("predicted_action") is not None]
    opening = assist[0] if assist else None

    def hits(rows: Sequence[Mapping[str, Any]], key: str = "correct_top_1") -> int:
        return sum(1 for row in rows if row.get(key))

    indexed = list(enumerate(assist))
    robot_indexed = [
        (index, row) for index, row in indexed
        if row.get("scheduled_actor") == "robot"
    ]
    discriminating = [row for index, row in robot_indexed if discriminates(index)]
    conditioned = [row for index, row in robot_indexed if conditioned_at(index)]
    tf_discriminating = [row for index, row in indexed if discriminates(index)]
    tf_conditioned = [row for index, row in indexed if conditioned_at(index)]
    forced = [row for row in robot if len(row.get("legal_actions") or ()) == 1]

    # One convention, recomputed from the stored ground-truth probability.  The
    # stored NLL is discarded: an unanswered decision recorded zero probability
    # and was charged the old 1e-12 floor.
    losses = [
        -math.log(max(float(row.get("ground_truth_probability") or 0.0), nll_floor))
        for row in assist
    ]
    corrections = sum(1 for row in robot if row.get("human_corrected"))

    return {
        "seed": episode["seed"],
        "arm": episode["arm"],
        "scenario": episode["scenario"],
        "environment": episode["environment"],
        "recipe_id": episode["recipe_id"],
        "preference": episode["preference"],
        "strategy": episode.get("strategy"),
        "holdout_target": episode.get("holdout_target"),
        "holdout_transfer_isomorphic": episode.get("holdout_transfer_isomorphic"),
        "exposure_after_change": episode.get("exposure_after_change", 0),
        "mode": episode["mode"],
        "cell_complete": episode.get("cell_complete", True),
        "recipe_steps": len(decisions),

        "teacher_forced_decision_count": len(assist),
        "teacher_forced_top_1_hits": hits(assist),
        "robot_decision_count": len(robot),
        "robot_top_1_hits": hits(robot),
        "single_legal_action_decision_count": len(forced),

        "preference_discriminating_robot_decisions": len(discriminating),
        "preference_discriminating_top_1_hits": hits(discriminating),
        "prefix_conditioned_robot_decisions": len(conditioned),
        "prefix_conditioned_top_1_hits": hits(conditioned),
        "teacher_forced_preference_discriminating_decisions": len(tf_discriminating),
        "teacher_forced_preference_discriminating_top_1_hits": hits(tf_discriminating),
        "teacher_forced_prefix_conditioned_decisions": len(tf_conditioned),
        "teacher_forced_prefix_conditioned_top_1_hits": hits(tf_conditioned),

        "opening_decision_count": int(opening is not None),
        "opening_top_1_hits": int(bool(opening and opening.get("correct_top_1"))),
        "opening_scheduled_actor": (
            None if opening is None else opening.get("scheduled_actor")
        ),
        "opening_discriminating_decision_count": int(
            bool(opening is not None and discriminates(0))
        ),
        "opening_discriminating_top_1_hits": int(bool(
            opening is not None
            and discriminates(0)
            and opening.get("correct_top_1")
        )),
        "opening_prefix_conditioned_decision_count": int(
            bool(opening is not None and conditioned_at(0))
        ),
        "opening_prefix_conditioned_top_1_hits": int(bool(
            opening is not None
            and conditioned_at(0)
            and opening.get("correct_top_1")
        )),

        "prediction_available_decisions": len(answered),
        "prediction_unavailable_decisions": len(assist) - len(answered),
        "robot_prediction_available_decisions": sum(
            1 for row in robot if row.get("predicted_action") is not None
        ),
        "human_corrections": corrections,
        "correction_free": corrections == 0,

        "teacher_forced_nll_total": sum(losses),
        "teacher_forced_nll_decisions": len(losses),
        "nll_probability_floor": nll_floor,
    }


def _rate(rows, numerator, denominator):
    return _pooled_rate(rows, numerator, denominator)


def _block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every rate for one group, each beside the denominator it was taken over."""
    decisions = sum(row["teacher_forced_decision_count"] for row in rows)
    robot = sum(row["robot_decision_count"] for row in rows)
    available = sum(row["prediction_available_decisions"] for row in rows)
    nll_decisions = sum(row["teacher_forced_nll_decisions"] for row in rows)
    return {
        "n_episodes": len(rows),
        "teacher_forced_top_1": _rate(
            rows, "teacher_forced_top_1_hits", "teacher_forced_decision_count",
        ),
        "teacher_forced_decision_count": decisions,
        "robot_top_1": _rate(rows, "robot_top_1_hits", "robot_decision_count"),
        "robot_decision_count": robot,
        "prefix_conditioned_top_1": _rate(
            rows, "prefix_conditioned_top_1_hits",
            "prefix_conditioned_robot_decisions",
        ),
        "prefix_conditioned_robot_decisions": sum(
            row["prefix_conditioned_robot_decisions"] for row in rows
        ),
        "teacher_forced_prefix_conditioned_top_1": _rate(
            rows, "teacher_forced_prefix_conditioned_top_1_hits",
            "teacher_forced_prefix_conditioned_decisions",
        ),
        "preference_discriminating_top_1": _rate(
            rows, "preference_discriminating_top_1_hits",
            "preference_discriminating_robot_decisions",
        ),
        "preference_discriminating_robot_decisions": sum(
            row["preference_discriminating_robot_decisions"] for row in rows
        ),
        "opening_top_1": _rate(
            rows, "opening_top_1_hits", "opening_decision_count",
        ),
        "opening_discriminating_top_1": _rate(
            rows, "opening_discriminating_top_1_hits",
            "opening_discriminating_decision_count",
        ),
        "opening_discriminating_decision_count": sum(
            row["opening_discriminating_decision_count"] for row in rows
        ),
        "opening_prefix_conditioned_top_1": _rate(
            rows, "opening_prefix_conditioned_top_1_hits",
            "opening_prefix_conditioned_decision_count",
        ),
        "opening_prefix_conditioned_decision_count": sum(
            row["opening_prefix_conditioned_decision_count"] for row in rows
        ),
        "single_legal_action_fraction": (
            sum(row["single_legal_action_decision_count"] for row in rows) / robot
            if robot else None
        ),
        "corrections_per_robot_decision": _rate(
            rows, "human_corrections", "robot_decision_count",
        ),
        "correction_free_rate": _finite_mean(row["correction_free"] for row in rows),
        "prediction_availability": (available / decisions if decisions else None),
        "prediction_unavailable_decisions": decisions - available,
        "teacher_forced_nll": (
            sum(row["teacher_forced_nll_total"] for row in rows) / nll_decisions
            if nll_decisions else None
        ),
        "teacher_forced_nll_decisions": nll_decisions,
    }


_CONTRAST_METRICS = (
    "teacher_forced_top_1",
    "prefix_conditioned_top_1",
    "teacher_forced_prefix_conditioned_top_1",
    "preference_discriminating_top_1",
    "opening_discriminating_top_1",
    "robot_top_1",
    "corrections_per_robot_decision",
)


def _paired_vs_full(
    cells: Mapping[Tuple[Any, ...], Mapping[str, Any]],
    key_without_arm,
) -> List[Dict[str, Any]]:
    indexed: Dict[Any, Dict[Any, Mapping[str, Any]]] = defaultdict(dict)
    for key, block in cells.items():
        arm, seed = key[0], key[1]
        indexed[(arm, *key_without_arm(key))][seed] = block
    contrasts = []
    reference_by_group = {
        group[1:]: by_seed for group, by_seed in indexed.items()
        if group[0] == FULL_ARM
    }
    for group, by_seed in sorted(indexed.items(), key=lambda item: tuple(map(str, item[0]))):
        arm = group[0]
        if arm == FULL_ARM:
            continue
        reference = reference_by_group.get(group[1:], {})
        shared = sorted(set(reference) & set(by_seed), key=str)
        if not shared:
            continue
        entry: Dict[str, Any] = {
            "arm": arm,
            "contrast": f"{FULL_ARM}_minus_{arm}",
            "group": [str(part) for part in group[1:]],
        }
        for metric in _CONTRAST_METRICS:
            entry[metric] = _paired_difference([
                reference[seed][metric] - by_seed[seed][metric]
                for seed in shared
                if reference[seed][metric] is not None
                and by_seed[seed][metric] is not None
            ])
        contrasts.append(entry)
    return contrasts


def rescore_run(run_dir: Path, *, nll_floor: float) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []
    for _path, cell in _iter_cells(run_dir):
        for episode in cell.get("episodes", ()):
            rescored = rescore_episode(episode, nll_floor=nll_floor)
            if not rescored["cell_complete"]:
                incomplete.append(rescored)
                continue
            episodes.append(rescored)

    assist = [row for row in episodes if row["mode"] == "assist"]

    groups = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in assist:
        grouped[(row["arm"], row["scenario"], row["environment"])].append(row)
    for (arm, scenario, environment), rows in sorted(grouped.items()):
        groups.append({
            "arm": arm, "scenario": scenario, "environment": environment,
            **_block(rows),
        })

    seed_cells: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    cell_rows: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in assist:
        cell_rows[(
            row["arm"], row["seed"], row["scenario"], row["environment"],
        )].append(row)
    per_cell = []
    for key, rows in sorted(cell_rows.items(), key=lambda item: tuple(map(str, item[0]))):
        arm, seed, scenario, environment = key
        block = _block(rows)
        seed_cells[key] = block
        per_cell.append({
            "arm": arm, "seed": seed, "scenario": scenario,
            "environment": environment, **block,
        })

    holdout_rows: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in assist:
        if row["scenario"] != "holdout":
            continue
        stage, role, axis, exposure = _holdout_stage(row)
        holdout_rows[(
            row["arm"], row["seed"], row["environment"], stage, role, axis,
            bool(row["holdout_transfer_isomorphic"]), exposure,
        )].append(row)
    holdout_groups = []
    transfer_cells: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for key, rows in sorted(holdout_rows.items(), key=lambda item: tuple(map(str, item[0]))):
        arm, seed, environment, stage, role, axis, isomorphic, exposure = key
        is_transfer = (
            stage == "holdout_axis_target_composition"
            and role == "target" and axis and exposure == "first"
        )
        block = _block(rows)
        holdout_groups.append({
            "arm": arm, "seed": seed, "environment": environment,
            "holdout_stage": stage, "holdout_recipe_role": role,
            "axis_preference": axis,
            "holdout_transfer_isomorphic": isomorphic,
            "target_exposure": exposure,
            "is_transfer_measurement": is_transfer,
            **block,
        })
        if is_transfer:
            transfer_cells[(arm, seed, environment, isomorphic)] = block

    return {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "kind": "rescoring_of_original_execution",
        "generated_at": _utc_now(),
        "source_run": str(run_dir),
        "nll_probability_floor": nll_floor,
        "caveat": (
            "Rescored from the saved decision records of one completed run. "
            "Denominators, availability, corrections, opening metrics, NLL and "
            "the holdout stage split are recomputed; the candidate list each "
            "decision was scored against is whatever the original execution "
            "produced, and that list was conditioned on the ground truth. A "
            "fresh run is required for structural-frontier candidate lists."
        ),
        "assist_episode_count": len(assist),
        "excluded_incomplete_episode_count": len(incomplete),
        "pooled": _block(assist),
        "groups": groups,
        "by_seed": {
            "unit": "seed_x_arm_x_scenario_x_environment",
            "per_cell": per_cell,
            "paired_vs_full": _paired_vs_full(
                {(k[0], k[1], k[2], k[3]): v for k, v in seed_cells.items()},
                lambda key: (key[2], key[3]),
            ),
        },
        "holdout_transfer_groups": holdout_groups,
        "holdout_transfer_paired_vs_full": _paired_vs_full(
            transfer_cells, lambda key: (key[2], key[3]),
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m adaptive_hrc_burrito.rescore",
        description=(
            "Rescore a completed cooking run from its saved decision records. "
            "The original artifacts are never modified."
        ),
    )
    parser.add_argument("run_dir", help="a completed run directory")
    parser.add_argument(
        "--out", default=None,
        help="output path (default: <run_dir>/rescored/analysis.json)",
    )
    parser.add_argument(
        "--nll-floor", type=float, default=DEFAULT_NLL_FLOOR,
        help=(
            "ground-truth probability floor, matching "
            "src Settings.min_probability (default: 1e-6)"
        ),
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    analysis = rescore_run(run_dir, nll_floor=float(args.nll_floor))
    out = Path(args.out) if args.out else run_dir / "rescored" / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(out, analysis)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
