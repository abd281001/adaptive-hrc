"""Versioned, manifest-backed Burrito experiments and progressive validation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .domain import (
    BurritoDomainAdapter,
    REWARD_FEATURE_VERSION,
    SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
    SEMANTIC_FEATURE_VERSION,
)
from .macros import macro_actions
from .options import (
    CONTROLLED_PARKING_POSITIONS,
    BurritoOptionExecutor,
    OptionExecutionError,
)
from .protocol import ASSIST, BurritoEpisodeResult, BurritoHrcRunner, BurritoTask
from .runtime import BurritoRuntime, UpstreamPaths, verify_pins


CONFIG_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
FULL_ARM = "latent_timing_lightweight"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _finite_mean(values: Iterable[Any]) -> float | None:
    usable = [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return statistics.fmean(usable) if usable else None


def load_config(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"config schema must be {CONFIG_SCHEMA_VERSION}"
        )
    for field in ("experiment", "seeds", "layouts", "arms", "conditions"):
        if not config.get(field):
            raise ValueError(f"config field {field!r} must be non-empty")
    config["_config_path"] = str(resolved)
    return config


def _arm_overrides(name: str) -> Dict[str, Any]:
    # Importing the registered symbolic arms guarantees Burrito uses the same
    # model ablation definitions rather than a divergent local transcription.
    from src.ablations import LATENT_STRATEGY_ABLATION_ARMS

    arms = {arm.name: arm.model_overrides() for arm in LATENT_STRATEGY_ABLATION_ARMS}
    if name == "full":
        name = FULL_ARM
    if name not in arms:
        raise ValueError(f"unknown Burrito evaluation arm {name!r}")
    return dict(arms[name])


def _expanded_tasks(condition: Mapping[str, Any]) -> Tuple[BurritoTask, ...]:
    tasks = []
    for row in condition.get("tasks", ()):
        repeat = max(1, int(row.get("repeat", 1)))
        for _ in range(repeat):
            tasks.append(BurritoTask.create(
                str(row["protein"]),
                str(row["preference"]),
                recipe_id=row.get("recipe_id"),
            ))
    if not tasks:
        raise ValueError(f"condition {condition.get('name')!r} has no tasks")
    return tuple(tasks)


def _model_storage_values(agent: Any) -> int:
    model = agent.maxent
    arrays = (
        model.reward_weights,
        model.features,
        model.semantic_features,
        model.values,
        model.latent_strategy.components,
        model.latent_strategy.codes,
        model.latent_strategy.fingerprints,
        model.latent_strategy.role_counts,
        model.latent_strategy.weights,
    )
    dense = sum(
        int(value.size) for value in arrays
        if isinstance(value, np.ndarray)
    )
    replay = sum(
        len(item.ordering) + 2 * len(item.transitions)
        for item in agent.replay.active.values()
    )
    return dense + len(model.q_values) + replay


def _memory_metrics(agent: Any) -> Dict[str, Any]:
    replay = agent.replay
    weights = [float(item.weight) for item in replay.active.values()]
    evidence = replay.horizon_evidence()
    horizons = [float(row["horizon_demos"]) for row in evidence.values()]
    return {
        "memory_active_variants": len(replay.active),
        "memory_pruned_variants": len(replay.pruned),
        "memory_nonunit_weights": sum(
            not math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for weight in weights
        ),
        "memory_min_active_weight": min(weights) if weights else None,
        "memory_reuse_gap_event_count": len(replay.reuse_gap_events),
        "memory_reentry_event_count": len(replay.reentry_events),
        "memory_pair_gap_sample_count": sum(
            int(row["pair_gap_samples"]) for row in evidence.values()
        ),
        "memory_horizon_min_demos": min(horizons) if horizons else None,
        "memory_horizon_max_demos": max(horizons) if horizons else None,
    }


def _recovery_metrics(decisions: Sequence[Any]) -> Dict[str, Any]:
    robot = [
        int(row.correct_top_1) for row in decisions
        if row.mode == ASSIST and row.scheduled_actor == "robot"
    ]
    if not robot:
        return {"adaptation_latency_robot_turns": None, "recovery_auc": None}
    latency = None
    for index in range(len(robot)):
        window = robot[index:index + 2]
        if len(window) == 2 and all(window):
            latency = index + 1
            break
    running = [statistics.fmean(robot[:index]) for index in range(1, len(robot) + 1)]
    return {
        "adaptation_latency_robot_turns": latency,
        "recovery_auc": statistics.fmean(running),
    }


def _decision_record(row: Any) -> Dict[str, Any]:
    return {
        "step": row.recipe_step,
        "scheduled_actor": row.scheduled_actor,
        "physical_actor_id": row.physical_actor_id,
        "executed_by": row.executed_by,
        "legal_actions": list(row.legal_actions),
        "acceptable_actions": list(row.acceptable_actions),
        "reference_action": row.reference_action,
        "executed_action": row.actual,
        "predicted_action": row.predicted,
        "acceptable_top_1": row.correct_top_1,
        "reference_top_1": row.exact_reference_match,
        "reference_probability": row.reference_probability,
        "reference_nll": (
            row.reference_nll if math.isfinite(row.reference_nll) else None
        ),
        "acceptable_probability_mass": row.acceptable_probability_mass,
        "acceptable_nll": (
            row.acceptable_nll if math.isfinite(row.acceptable_nll) else None
        ),
        "invalid_prediction": row.invalid_prediction,
        "human_corrected": row.human_corrected,
        "proposal_executed": row.proposal_executed,
        "prediction_wall_s": row.prediction_wall_s,
        "low_level_ticks": row.low_level_ticks,
        "passive_wait_ticks_before": row.passive_wait_ticks_before,
        "native_actions": list(row.primitive_calls),
        "native_action_ticks": list(row.primitive_ticks),
        "semantic_fallback_used": bool(
            row.prediction_stats.get("semantic_fallback_used", False)
        ),
        "latent_strategy_used": bool(
            row.prediction_stats.get("latent_strategy_used", False)
        ),
    }


def _episode_record(
    result: BurritoEpisodeResult,
    *,
    metadata: Mapping[str, Any],
    wall_s: float,
    train_events: Sequence[Mapping[str, Any]],
    storage_values: int,
    memory_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    decisions = [
        row for row in result.decisions
        if row.mode == ASSIST and row.predicted is not None
    ]
    recipe_steps = len(result.decisions)
    train_wall = sum(float(row.get("total_wall_s", 0.0)) for row in train_events)
    train_flops = sum(float(row.get("flop_estimate", 0.0)) for row in train_events)
    human_turns = sum(row.scheduled_actor == "human" for row in result.decisions)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        **dict(metadata),
        "recipe_id": result.task.recipe_id,
        "protein": result.task.protein,
        "preference": result.task.preference,
        "mode": result.mode,
        "recipe_steps": recipe_steps,
        "deliveries": result.deliveries,
        "task_completed": result.deliveries == 1,
        "task_wall_s": wall_s,
        "task_low_level_ticks": result.low_level_ticks,
        "passive_wait_ticks": result.passive_wait_ticks,
        "memory_age_delta": result.memory_age_delta,
        "acceptable_set_accuracy": _finite_mean(
            row.correct_top_1 for row in decisions
        ),
        "acceptable_set_nll": _finite_mean(
            row.acceptable_nll for row in decisions
        ),
        "reference_top_1": _finite_mean(
            row.exact_reference_match for row in decisions
        ),
        "reference_nll": _finite_mean(row.reference_nll for row in decisions),
        "robot_acceptable_top_1": result.robot_top_1,
        "robot_reference_top_1": result.robot_exact_reference_top_1,
        "human_shadow_acceptable_top_1": result.human_shadow_top_1,
        "human_interventions": result.corrections,
        "human_action_load": (
            (human_turns + result.corrections) / recipe_steps
            if recipe_steps else None
        ),
        "invalid_predictions": result.invalid_predictions,
        "prediction_wall_s": result.prediction_wall_s,
        "train_wall_s": train_wall,
        "train_flops": train_flops,
        "compute_wall_s_per_macro": (
            (result.prediction_wall_s + train_wall) / recipe_steps
            if recipe_steps else None
        ),
        "train_flops_per_macro": train_flops / max(1, recipe_steps),
        "model_storage_values": storage_values,
        "model_storage_values_per_macro": storage_values / max(1, recipe_steps),
        **dict(memory_metrics),
        "semantic_fallback_decisions": sum(
            bool(row.prediction_stats.get("semantic_fallback_used", False))
            for row in decisions
        ),
        "latent_strategy_decisions": sum(
            bool(row.prediction_stats.get("latent_strategy_used", False))
            for row in decisions
        ),
        **_recovery_metrics(result.decisions),
        "decisions": [_decision_record(row) for row in result.decisions],
    }


def _condition_run(
    runtime: BurritoRuntime,
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    seed: int,
    layout: str,
    arm: str,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    from src.adaptive_agent import AdaptiveAgent
    from src.models import Settings

    overrides = dict(config.get("settings", {}))
    overrides.update(_arm_overrides(arm))
    overrides.update({
        "verbose": False,
        "seed": int(seed),
        "semantic_fallback_max_rms_distance": float(
            overrides.get(
                "semantic_fallback_max_rms_distance",
                SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
            )
        ),
    })
    executor = BurritoOptionExecutor(
        runtime,
        layout=layout,
        horizon=int(config.get("horizon", 2400)),
        seed=int(config.get("planner_seed", 11)),
    )
    domain = BurritoDomainAdapter(
        executor.state,
        terrain_positions=executor.env.mdp.terrain_pos_dict,
    )
    agent = AdaptiveAgent(Settings(**overrides), domain=domain)
    runner = BurritoHrcRunner(agent, executor, domain, seed=seed)

    episodes: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    observed_preferences: Dict[str, set[str]] = {}
    global_preferences: set[str] = set()
    last_preference: Dict[str, str] = {}
    last_pair_index: Dict[Tuple[str, str], int] = {}
    for event_index, task in enumerate(_expanded_tasks(condition)):
        seen_recipe = task.recipe_id in runner.observed_recipes
        seen_here = task.preference in observed_preferences.get(task.recipe_id, set())
        cross_recipe = seen_recipe and not seen_here and task.preference in global_preferences
        preference_switch = bool(
            seen_recipe
            and task.recipe_id in last_preference
            and last_preference[task.recipe_id] != task.preference
        )
        pair = (task.recipe_id, task.preference)
        recurrence_gap = (
            event_index - last_pair_index[pair] - 1
            if pair in last_pair_index else None
        )
        event_metadata = {
            "condition": str(condition["name"]),
            "event_index": event_index,
            "seed": int(seed),
            "planner_seed": int(config.get("planner_seed", 11)),
            "layout": layout,
            "arm": arm,
            "is_full_arm": arm in {"full", FULL_ARM},
            "cross_recipe_transfer": cross_recipe,
            "preference_switch": preference_switch,
            "recurrence_gap": recurrence_gap,
        }
        event_start = time.perf_counter()
        train_before = len(agent.retrain_events)
        try:
            result = runner.run_task(task)
        except OptionExecutionError as error:
            failures.append({
                **event_metadata,
                "failure_type": "planner_failure",
                "error": str(error),
            })
            break
        except Exception as error:
            failures.append({
                **event_metadata,
                "failure_type": "system_failure",
                "error": f"{type(error).__name__}: {error}",
            })
            break
        episodes.append(_episode_record(
            result,
            metadata=event_metadata,
            wall_s=time.perf_counter() - event_start,
            train_events=agent.retrain_events[train_before:],
            storage_values=_model_storage_values(agent),
            memory_metrics=_memory_metrics(agent),
        ))
        observed_preferences.setdefault(task.recipe_id, set()).add(task.preference)
        global_preferences.add(task.preference)
        last_preference[task.recipe_id] = task.preference
        last_pair_index[pair] = event_index
    return episodes, failures


def _aggregate(episodes: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_names = (
        "acceptable_set_accuracy", "acceptable_set_nll", "reference_top_1",
        "reference_nll", "robot_acceptable_top_1", "human_action_load",
        "human_interventions", "task_wall_s", "task_low_level_ticks",
        "invalid_predictions", "adaptation_latency_robot_turns",
        "recovery_auc", "compute_wall_s_per_macro", "train_flops_per_macro",
        "model_storage_values_per_macro", "memory_active_variants",
        "memory_pruned_variants", "memory_nonunit_weights",
        "memory_pair_gap_sample_count", "memory_horizon_min_demos",
        "memory_horizon_max_demos",
    )
    assist = [row for row in episodes if row["mode"] == ASSIST]
    grouped: Dict[Tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in assist:
        grouped.setdefault(
            (str(row["arm"]), str(row["layout"]), str(row["condition"])), [],
        ).append(row)
    groups = []
    for (arm, layout, condition), rows in sorted(grouped.items()):
        groups.append({
            "arm": arm,
            "layout": layout,
            "condition": condition,
            "n_assist_episodes": len(rows),
            **{name: _finite_mean(row.get(name) for row in rows) for name in metric_names},
        })
    recurrence = [
        row for row in assist
        if isinstance(row.get("recurrence_gap"), int) and row["recurrence_gap"] > 0
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed" if not failures else "completed_with_failures",
        "episode_count": len(episodes),
        "assist_episode_count": len(assist),
        "failure_count": len(failures),
        "planner_failure_count": sum(
            row["failure_type"] == "planner_failure" for row in failures
        ),
        "system_failure_count": sum(
            row["failure_type"] == "system_failure" for row in failures
        ),
        "delivery_rate": _finite_mean(row["task_completed"] for row in episodes),
        "retention_after_gap_accuracy": _finite_mean(
            row["robot_acceptable_top_1"] for row in recurrence
        ),
        "retention_episode_count": len(recurrence),
        "groups": groups,
    }


def _hardware() -> Dict[str, Any]:
    memory_bytes = None
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
        "python": sys.version,
        "executable": sys.executable,
    }


def _manifest(config: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    paths = UpstreamPaths.discover()
    project = paths.integration_root.parent
    status = _git(project, "status", "--porcelain")
    if bool(config.get("require_clean_git", False)) and status:
        raise RuntimeError(
            "publication config requires a clean Git commit; commit or stash "
            "the current changes before running"
        )
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "run_dir": str(run_dir),
        "command": list(sys.argv),
        "config_path": config["_config_path"],
        "config_sha256": hashlib.sha256(_json_bytes(public_config)).hexdigest(),
        "config": public_config,
        "repositories": {
            "adaptive_hrc": {
                "commit": _git(project, "rev-parse", "HEAD"),
                "dirty": bool(status),
                "dirty_paths": status.splitlines(),
            },
            **{
                name: {"commit": commit}
                for name, commit in verify_pins(paths).items()
            },
        },
        "features": {
            "reward": REWARD_FEATURE_VERSION,
            "semantic": SEMANTIC_FEATURE_VERSION,
            "semantic_fallback_max_rms_distance": (
                SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
            ),
        },
        "macros": {
            protein: list(macro_actions(protein))
            for protein in ("steak", "mushroom")
        },
        "parking_positions": {
            layout: {
                str(actor): list(position)
                for actor, position in CONTROLLED_PARKING_POSITIONS.get(
                    layout, {}
                ).items()
            }
            for layout in map(str, config["layouts"])
        },
        "hardware": _hardware(),
        "dependencies": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
    }


def run_experiment(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    digest = hashlib.sha256(_json_bytes(public_config)).hexdigest()[:10]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(output_root or config.get("output", "burrito/results")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{config['experiment']}__{timestamp}__{digest}"
    manifest = _manifest(config, run_dir)
    run_dir.mkdir(parents=False, exist_ok=False)
    _atomic_json(run_dir / "config.json", public_config)
    _atomic_json(run_dir / "manifest.json", manifest)

    runtime = BurritoRuntime.discover()
    episodes: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    try:
        for seed in map(int, config["seeds"]):
            for layout in map(str, config["layouts"]):
                for arm in map(str, config["arms"]):
                    for condition in config["conditions"]:
                        rows, errors = _condition_run(
                            runtime, config, condition,
                            seed=seed, layout=layout, arm=arm,
                        )
                        episodes.extend(rows)
                        failures.extend(errors)
        summary = _aggregate(episodes, failures)
        _atomic_json(run_dir / "episodes.json", episodes)
        _atomic_json(run_dir / "failures.json", failures)
        _atomic_json(run_dir / "summary.json", summary)
        manifest.update({
            "status": summary["status"],
            "completed_at": _utc_now(),
            "episode_count": len(episodes),
            "failure_count": len(failures),
        })
        _atomic_json(run_dir / "manifest.json", manifest)
        return {"run_dir": str(run_dir), "summary": summary}
    except BaseException as error:
        manifest.update({
            "status": "failed",
            "completed_at": _utc_now(),
            "error": f"{type(error).__name__}: {error}",
        })
        _atomic_json(run_dir / "manifest.json", manifest)
        raise


def validate_result(result: Mapping[str, Any], config_path: str | Path) -> Dict[str, Any]:
    config = load_config(config_path)
    summary = result["summary"]
    failures = []
    if summary["failure_count"]:
        failures.append(f"{summary['failure_count']} execution failures")
    if summary["delivery_rate"] != 1.0:
        failures.append(f"delivery rate is {summary['delivery_rate']!r}")
    requirements = set(config.get("validation_requirements", ()))
    episodes = json.loads(
        (Path(result["run_dir"]) / "episodes.json").read_text(encoding="utf-8")
    )
    if "cross_recipe_transfer" in requirements and not any(
        row["cross_recipe_transfer"] and row["semantic_fallback_decisions"] > 0
        for row in episodes
    ):
        failures.append("cross-recipe semantic fallback was never exercised")
    if "preference_switch" in requirements and not any(
        row["preference_switch"] for row in episodes
    ):
        failures.append("no preference-switch assist episode was executed")
    if "long_gap_recurrence" in requirements and not any(
        isinstance(row.get("recurrence_gap"), int) and row["recurrence_gap"] > 0
        for row in episodes
    ):
        failures.append("no positive-gap recurrence was executed")
    if "memory_decay_adaptation" in requirements:
        memory_rows = [
            row for row in episodes
            if row["condition"] == "long_gap_recurrence"
        ]
        if not any(
            int(row["memory_pair_gap_sample_count"]) > 0
            for row in memory_rows
        ):
            failures.append("long-gap run produced no adaptive-horizon evidence")
        if not any(
            int(row["memory_pruned_variants"]) > 0
            or int(row["memory_nonunit_weights"]) > 0
            or int(row["memory_reentry_event_count"]) > 0
            for row in memory_rows
        ):
            failures.append("long-gap run never exercised decay or re-entry")
    if "multiple_layouts" in requirements and len({row["layout"] for row in episodes}) < 2:
        failures.append("fewer than two layouts completed")
    if "multiple_seeds" in requirements and len({row["seed"] for row in episodes}) < 2:
        failures.append("fewer than two seeds completed")
    if "registered_ablations" in requirements:
        expected = set(map(str, config["arms"]))
        if {row["arm"] for row in episodes} != expected:
            failures.append("not every registered ablation arm completed")
    if any(int(row["invalid_predictions"]) for row in episodes):
        failures.append("one or more task-graph-invalid predictions occurred")
    validation = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "requirements": sorted(requirements),
    }
    _atomic_json(Path(result["run_dir"]) / "validation.json", validation)
    if failures:
        raise RuntimeError("Burrito validation failed: " + "; ".join(failures))
    return validation


__all__ = [
    "FULL_ARM",
    "load_config",
    "run_experiment",
    "validate_result",
]
