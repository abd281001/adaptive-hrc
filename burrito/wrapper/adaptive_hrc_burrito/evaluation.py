"""Manifest-backed evaluation over natural Overcooked/Burrito ladders."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog import (
    BURRITO_RECIPE_IDS,
    STRATA,
    OVERCOOKED_RECIPE_IDS,
    RECIPES,
    TALENTS_LIKE_PREFERENCES,
    applicable_preferences,
    get_preference,
    get_recipe,
    preference_order,
)
from .domain import (
    CookingDomainAdapter,
    REWARD_FEATURE_VERSION,
    SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
    SEMANTIC_FEATURE_VERSION,
    STRATEGY_ROLE_VERSION,
)
from .ladder import LadderSettings, SCENARIOS, generate_ladder, ladder_audit
from .options import OptionExecutionError
from .protocol import (
    ASSIST,
    CookingEpisodeResult,
    CookingHrcRunner,
    CookingObservation,
    CookingTask,
)
from .runtime import BurritoRuntime, UpstreamPaths, verify_pins
from .task_graph import CookingPreferencePolicy, CookingTaskGraph, is_preference_discriminating


CONFIG_SCHEMA_VERSION = 4
RESULT_SCHEMA_VERSION = 6
FULL_ARM = "full"
ARM_NAMES: Tuple[str, ...] = (
    FULL_ARM,
    "frozen",
    "offline_default",
    "unpinned",
    "latest",
    "fixed",
    "no_decay",
    "bc",
    "ewc",
    "replay_bc",
    "memory_oracle",
)
VALIDATION_REQUIREMENTS: Tuple[str, ...] = (
    "adaptation_linkage",
    "holdout_transfer_is_generalisation",
    "stratum_separation",
    "behavioral_preference_coverage",
    "catalog_coverage",
    "adaptive_hrc_baseline_parity",
    "comparison_arms",
    "cross_environment_adaptation",
    "framework_components",
    "multiple_seeds",
    "natural_ladder",
    "open_set_recipe_separation",
    "post_update_recurrence",
    "preference_discrimination",
    "scenario_invariants",
    "transfer_mechanisms_exercised",
    "verified_shift_update",
)


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
    result = subprocess.run(
        ("git", "-C", str(repository), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _finite_mean(values: Iterable[Any]) -> float | None:
    usable = [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return statistics.fmean(usable) if usable else None


def _pooled_rate(
    rows: Iterable[Mapping[str, Any]], numerator: str, denominator: str,
) -> float | None:
    usable = list(rows)
    total = sum(int(row.get(denominator, 0) or 0) for row in usable)
    if not total:
        return None
    return sum(int(row.get(numerator, 0) or 0) for row in usable) / total


def _ladder_settings(value: Mapping[str, Any]) -> LadderSettings:
    normalized = dict(value)
    for name in ("active_size_weights", "lifecycle_weights"):
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    return LadderSettings(**normalized)


def load_config(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"evaluation config schema must be {CONFIG_SCHEMA_VERSION}")
    for key in (
        "experiment", "output", "seeds", "scenarios", "recipe_ids", "arms",
        "ladder",
    ):
        if key not in config:
            raise ValueError(f"evaluation config is missing {key!r}")
    if not config["seeds"] or not config["scenarios"] or not config["recipe_ids"]:
        raise ValueError("seeds, scenarios, and recipe_ids must be non-empty")
    unknown_scenarios = set(map(str, config["scenarios"])) - set(SCENARIOS)
    if unknown_scenarios:
        raise ValueError(f"unknown scenarios: {sorted(unknown_scenarios)}")
    unknown_recipes = set(map(str, config["recipe_ids"])) - set(RECIPES)
    if unknown_recipes:
        raise ValueError(f"unknown recipes: {sorted(unknown_recipes)}")
    unknown_arms = set(map(str, config["arms"])) - set(ARM_NAMES)
    if unknown_arms:
        raise ValueError(f"unknown arms: {sorted(unknown_arms)}")
    requirements = set(map(str, config.get("validation_requirements", ())))
    # Subset checking let the shipped config quietly drop a requirement, so the
    # only check that verifies preference coverage never ran.  Require the
    # exact set: opting out of a check must be a deliberate code change.
    if requirements != set(VALIDATION_REQUIREMENTS):
        raise ValueError(
            "validation_requirements must be exactly "
            f"{sorted(VALIDATION_REQUIREMENTS)}; missing "
            f"{sorted(set(VALIDATION_REQUIREMENTS) - requirements)}, unknown "
            f"{sorted(requirements - set(VALIDATION_REQUIREMENTS))}"
        )
    ladder = _ladder_settings(config["ladder"])
    ladder.validate(len(config["recipe_ids"]))
    if ladder != LadderSettings():
        raise ValueError("ladder settings must match the Adaptive-HRC publication defaults")
    trivial = [
        recipe_id for recipe_id in map(str, config["recipe_ids"])
        if len(RECIPES[recipe_id].ingredients) < 2
    ]
    if trivial:
        raise ValueError(f"single-ingredient recipes are prohibited: {trivial}")
    if tuple(map(str, config["arms"])) != ARM_NAMES:
        raise ValueError(
            "the full cooking evaluation must use the Adaptive-HRC baseline "
            f"roster in order: {ARM_NAMES}"
        )
    # Taken from Adaptive-HRC rather than restated, so the replication cannot
    # silently fall behind the paired grid the symbolic evaluation runs. It
    # already had: this stayed at five seeds after that grid grew to eight.
    from src.evaluation import PAPER_SEEDS

    if tuple(map(int, config["seeds"])) != tuple(int(s) for s in PAPER_SEEDS):
        raise ValueError(
            "the full evaluation uses the Adaptive-HRC paper seeds "
            f"{tuple(int(s) for s in PAPER_SEEDS)}"
        )
    if tuple(map(str, config["scenarios"])) != SCENARIOS:
        raise ValueError("the full evaluation must contain all three scenarios")
    if set(map(str, config["recipe_ids"])) != set(RECIPES):
        raise ValueError("the full evaluation must use the complete nontrivial catalog")
    expected_frozen_pairs = sum(
        len(applicable_preferences(str(recipe_id)))
        for recipe_id in config["recipe_ids"]
    )
    parity_flags = {
        "shared_routing": True,
        "pre_event_probes": True,
        "frozen_pairs": expected_frozen_pairs,
        "audit_period": 2,
        "audit_prefixes": 16,
        "audit_tolerance": 0.05,
        "top_k": 3,
        "offline_recipe_fraction": 0.50,
        "offline_preference_fraction": 0.50,
        "lead_actor_policy": "human_first",
    }
    mismatched = {
        name: (config.get(name), expected)
        for name, expected in parity_flags.items()
        if config.get(name) != expected
    }
    if mismatched:
        raise ValueError(f"Adaptive-HRC evaluation setting mismatch: {mismatched}")
    model_contract = {
        "irl_cold_steps": 100,
        "irl_warm_steps": 40,
        "irl_horizon": 45,
        "initial_grace": 50,
        "min_grace": 6,
        "replay_capacity": 64,
        "semantic_fallback_max_rms_distance": 0.20,
    }
    model_mismatch = {
        name: (config.get("settings", {}).get(name), expected)
        for name, expected in model_contract.items()
        if config.get("settings", {}).get(name) != expected
    }
    if model_mismatch:
        raise ValueError(f"Adaptive-HRC model setting mismatch: {model_mismatch}")
    unexpected_model_overrides = set(config.get("settings", {})) - set(model_contract)
    if unexpected_model_overrides:
        raise ValueError(
            "full evaluation may not override other Adaptive-HRC model settings: "
            f"{sorted(unexpected_model_overrides)}"
        )
    if "arm_settings" in config:
        raise ValueError("per-arm model overrides are prohibited in the full evaluation")
    workers = int(config.get("workers", 0) or 0)
    if workers < 0:
        raise ValueError("workers must be zero (auto) or positive")
    config["_config_path"] = str(resolved)
    return config


def _build_agent(arm: str, settings: Any, domain: CookingDomainAdapter) -> Any:
    from src.adaptive_agent import AdaptiveAgent

    if arm in {FULL_ARM, "memory_oracle"}:
        agent = AdaptiveAgent(settings, domain=domain)
        if arm == "memory_oracle":
            from src.memory import ReplayMemory

            # As in Adaptive-HRC, the oracle differs from Full only in its
            # future-aware retention policy.  It is not an action oracle.
            agent.replay = ReplayMemory(agent.settings, policy="none")
        return agent
    from src.baselines import BASELINE_AGENTS
    try:
        agent_type = BASELINE_AGENTS[arm]
    except KeyError as error:
        raise ValueError(f"unknown evaluation arm {arm!r}") from error
    return agent_type(settings, domain=domain)


def _offline_subset(
    values: Sequence[str], fraction: float, *, seed: int, axis: str,
) -> Tuple[str, ...]:
    """Use the same deterministic floor-based subset rule as Adaptive-HRC."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"offline {axis} fraction must lie in (0, 1]")
    candidates = sorted(set(map(str, values)))
    if not candidates:
        raise ValueError(f"cannot sample an empty offline {axis} set")
    count = max(1, min(
        len(candidates), int(math.floor(float(fraction) * len(candidates))),
    ))
    import random

    rng = random.Random(f"frozen|{axis}|{int(seed)}")
    return tuple(sorted(rng.sample(candidates, count)))


def _observe_offline(
    agent: Any,
    domain: CookingDomainAdapter,
    recipe_id: str,
    preference: str,
) -> None:
    """Fit one semantic task-option trajectory without running a simulator."""
    recipe = get_recipe(recipe_id)
    ordering = preference_order(recipe, get_preference(preference))
    domain.begin_task(recipe_id)
    agent.start_demo()
    completed: list[str] = []
    for action in ordering:
        before = domain.state_from_completed(recipe_id, completed)
        completed.append(action)
        after = domain.state_from_completed(recipe_id, completed)
        agent.observe(
            CookingObservation(before, action, after),
            ground_truth_recipe=recipe_id,
        )
    agent.end_demo()


def _prepare_offline_baseline(
    arm: str,
    agent: Any,
    domain: CookingDomainAdapter,
    tasks: Sequence[Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    scenario: str,
) -> Mapping[str, Any]:
    """Match Adaptive-HRC's two frozen training regimes."""
    if arm not in {"frozen", "offline_default"}:
        return {}
    started = time.perf_counter()
    if scenario == "holdout":
        source_tasks = [
            task for task in tasks if task.strategy == "holdout_source_training"
        ]
        pairs = [(task.recipe_id, task.preference) for task in source_tasks]
        design = "matched_holdout_source_progression"
    elif arm == "frozen":
        recipes = _offline_subset(
            [task.recipe_id for task in tasks],
            float(config.get("offline_recipe_fraction", 0.50)),
            seed=seed,
            axis="recipe",
        )
        preferences = _offline_subset(
            [task.preference for task in tasks],
            float(config.get("offline_preference_fraction", 0.50)),
            seed=seed,
            axis="preference",
        )
        pairs = [
            (recipe, preference)
            for recipe in recipes for preference in preferences
            if preference in applicable_preferences(recipe)
        ]
        design = "subset_recipes_subset_preferences"
    else:
        recipes = sorted({task.recipe_id for task in tasks})
        pairs = [(recipe, _canonical_preference(recipe)) for recipe in recipes]
        design = "all_recipes_default_only"
    if not pairs:
        raise ValueError(f"{arm} has no effective offline training pairs")
    for recipe, preference in pairs:
        _observe_offline(agent, domain, recipe, preference)
    metadata = {
        "offline_training_design": design,
        "offline_training_event_count": len(pairs),
        "offline_training_unique_pair_count": len(set(pairs)),
        "offline_training_recipe_count": len({recipe for recipe, _ in pairs}),
        "offline_training_preference_count": len({preference for _, preference in pairs}),
        "offline_training_end_to_end_wall_s": time.perf_counter() - started,
    }
    lock = getattr(agent, "lock_deployment", None)
    if not callable(lock):
        raise TypeError(f"{arm} does not implement lock_deployment()")
    return dict(lock(metadata))


def _canonical_preference(recipe_id: str) -> str:
    return applicable_preferences(recipe_id)[0]


def _apply_memory_oracle_pruning(
    agent: Any,
    future_tasks: Sequence[Any],
    learner_recipe_by_task: Mapping[str, str],
) -> Mapping[str, Any]:
    """Retain only known variants that occur again in the future stream."""
    from src.memory import make_variant_id

    future_keys = {
        (learner_recipe_by_task[task.recipe_id], make_variant_id(preference_order(
            get_recipe(task.recipe_id), get_preference(task.preference),
        )))
        for task in future_tasks
        if task.recipe_id in learner_recipe_by_task
    }
    active_before = set(agent.replay.active)
    pruned_before = set(agent.replay.pruned)
    discarded = sorted((active_before | pruned_before) - future_keys)
    if discarded:
        agent.discard(discarded)
    weight_changed = False
    for key in set(agent.replay.active) & future_keys:
        entry = agent.replay.active[key]
        weight_changed = weight_changed or not math.isclose(
            float(entry.weight), 1.0, abs_tol=1e-12,
        )
        entry.weight = 1.0
    if (discarded or weight_changed) and agent.replay.active:
        agent.refresh()
    return {
        "oracle_retention_policy": "future_filtered_known_variants",
        "oracle_decay_policy": "binary_keep_until_final_occurrence",
        "oracle_pruned_variant_count": len(discarded),
        "oracle_future_variant_count": len(future_keys),
        "oracle_active_variants_before": len(active_before),
        "oracle_active_variants_after": len(agent.replay.active),
    }


def _frozen_probe(
    agent: Any,
    domain: CookingDomainAdapter,
    task: Any,
    *,
    metadata: Mapping[str, Any],
    probe_kind: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Evaluate an entire task-option trajectory without mutating the agent."""
    graph = CookingTaskGraph.create(task.recipe_id)
    policy = CookingPreferencePolicy.create(task.preference)
    completed: list[str] = []
    rows = []
    agent.set_frozen(True)
    try:
        domain.begin_task(task.recipe_id)
        while not graph.is_complete(completed):
            legal = graph.frontier(completed)
            truth = policy.choose_action(legal, graph)
            distribution: Mapping[str, float] = {}
            # Probe the opening move too.  It is the most preference-informative
            # decision in these task graphs, and excluding it made the probe
            # blind to exactly the strategies the holdout is built around.
            state = domain.state_from_completed(task.recipe_id, completed)
            distribution = agent.predict_actions(
                tuple(completed), state=state, action_universe=legal,
            )
            ranked = tuple(agent.rank_actions(distribution, k=max(1, int(top_k))))
            predicted = ranked[0] if ranked else None
            rows.append({
                "legal_count": len(legal),
                "correct": predicted == truth,
                "correct_top_k": truth in ranked,
                "scored": predicted is not None,
                "discriminating": bool(
                    predicted is not None
                    and is_preference_discriminating(legal, graph)
                ),
                "probability": float(distribution.get(truth, 0.0)),
            })
            completed.append(truth)
    finally:
        agent.set_frozen(False)
        # AdaptiveAgent's generic freeze snapshot deep-copies its adapter.
        # Restore the shared cooking adapter required by the physical runner.
        agent.domain = domain
        if hasattr(agent, "maxent"):
            agent.maxent.domain = domain
        if hasattr(agent, "cloner"):
            agent.cloner.domain = domain
    scored = [row for row in rows if row["scored"]]
    discriminating = [row for row in rows if row["discriminating"]]
    nontrivial = [row for row in scored if row["legal_count"] > 1]
    return {
        **dict(metadata),
        "diagnostic_type": probe_kind,
        "recipe_id": task.recipe_id,
        "environment": RECIPES[task.recipe_id].stratum,
        "preference": task.preference,
        "decision_count": len(scored),
        "top_1_hits": sum(row["correct"] for row in scored),
        "top_1": _finite_mean(row["correct"] for row in scored),
        "top_k_hits": sum(row["correct_top_k"] for row in scored),
        "top_k": _finite_mean(row["correct_top_k"] for row in scored),
        "preference_discriminating_decision_count": len(discriminating),
        "preference_discriminating_top_1_hits": sum(
            row["correct"] for row in discriminating
        ),
        "preference_discriminating_top_1": _finite_mean(
            row["correct"] for row in discriminating
        ),
        "nontrivial_choice_decision_count": len(nontrivial),
        "nontrivial_choice_top_1_hits": sum(row["correct"] for row in nontrivial),
        "mean_ground_truth_probability": _finite_mean(
            row["probability"] for row in scored
        ),
        "mutation_free": True,
    }


def _model_storage_metrics(agent: Any) -> Dict[str, Any]:
    model = agent.maxent
    fit_stats = dict(getattr(model, "last_fit_stats", {}) or {})
    arrays = [
        value for value in vars(model).values() if isinstance(value, np.ndarray)
    ]
    replay = tuple(agent.replay.active.values()) + tuple(agent.replay.pruned.values())
    transition_count = sum(len(item.transitions) for item in replay)
    return {
        "learner_dense_array_bytes": sum(int(value.nbytes) for value in arrays),
        "learner_replay_variant_count": len(replay),
        "learner_replay_transition_count": transition_count,
        "learner_registry_recipe_count": sum(
            bool(slot) for slot in agent.library.variants.values()
        ),
        "learner_model_structure": str(fit_stats.get("model_structure", "unfitted")),
        "latent_strategy_prototypes": int(
            fit_stats.get("latent_strategy_prototypes", 0)
        ),
    }


def _memory_metrics(agent: Any) -> Dict[str, Any]:
    weights = [float(item.weight) for item in agent.replay.active.values()]
    return {
        "memory_active_variants": len(agent.replay.active),
        "memory_pruned_variants": len(agent.replay.pruned),
        "memory_nonunit_weights": sum(
            not math.isclose(weight, 1.0, abs_tol=1e-12) for weight in weights
        ),
        "memory_min_active_weight": min(weights) if weights else None,
    }


def _decision_record(row: Any) -> Dict[str, Any]:
    return {
        "step": row.recipe_step,
        "scheduled_actor": row.scheduled_actor,
        "physical_actor_id": row.physical_actor_id,
        "executed_by": row.executed_by,
        "legal_actions": list(row.legal_actions),
        "ground_truth_action": row.ground_truth_action,
        "executed_action": row.actual,
        "predicted_action": row.predicted,
        "correct_top_1": row.correct_top_1,
        "correct_top_k": row.correct_top_k,
        "ground_truth_probability": row.ground_truth_probability,
        "ground_truth_nll": (
            row.ground_truth_nll if math.isfinite(row.ground_truth_nll) else None
        ),
        "preference_discriminating": row.preference_discriminating,
        "human_corrected": row.human_corrected,
        "proposal_executed": row.proposal_executed,
        "invalid_prediction": row.invalid_prediction,
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
    result: CookingEpisodeResult,
    *,
    metadata: Mapping[str, Any],
    wall_s: float,
    agent: Any,
) -> Dict[str, Any]:
    assist = [row for row in result.decisions if row.mode == ASSIST and row.predicted is not None]
    robot = [row for row in assist if row.scheduled_actor == "robot"]
    discriminating = [row for row in robot if row.preference_discriminating]
    choice = [row for row in robot if len(row.legal_actions) > 1]
    forced = [row for row in robot if len(row.legal_actions) == 1]
    recipe = RECIPES[result.task.recipe_id]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        **dict(metadata),
        "recipe_id": result.task.recipe_id,
        # Stratum, not raw environment: compatibility recipes are executed by
        # wrapper-restored transitions and must never pool with natively
        # executed Burrito recipes.
        "environment": recipe.stratum,
        "upstream_environment": recipe.environment,
        "layout": recipe.layout,
        "ingredients": list(recipe.ingredients),
        "upstream_dish": recipe.upstream_dish,
        "preference": result.task.preference,
        "strategy": result.task.strategy,
        "adaptation_id": result.task.adaptation_id,
        "holdout_target": result.task.holdout_target,
        "phase": result.task.phase,
        "schedule_step": result.task.schedule_step,
        "phase_role": result.task.phase_role,
        "lifecycle": result.task.lifecycle,
        "preference_changed": result.task.preference_changed,
        "exposure_after_change": result.task.exposure_after_change,
        "mode": result.mode,
        "recipe_steps": len(result.decisions),
        "deliveries": result.deliveries,
        "expected_deliveries": recipe.expected_deliveries,
        "task_completed": result.deliveries == recipe.expected_deliveries,
        "task_wall_s": wall_s,
        "task_low_level_ticks": result.low_level_ticks,
        "passive_wait_ticks": result.passive_wait_ticks,
        "robot_decision_count": len(robot),
        "robot_top_1_hits": result.robot_top_1_hits,
        "robot_top_1": result.robot_top_1,
        "robot_top_k_hits": result.robot_top_k_hits,
        "robot_top_k": result.robot_top_k,
        "human_action_count": result.human_actions,
        "robot_action_count": result.robot_actions,
        "human_corrections": result.corrections,
        "correction_free": result.corrections == 0,
        "corrections_per_robot_decision": (
            result.corrections / len(robot) if robot else None
        ),
        "preference_discriminating_robot_decisions": len(discriminating),
        "preference_discriminating_top_1_hits": sum(
            row.correct_top_1 for row in discriminating
        ),
        "preference_discriminating_top_1": _finite_mean(
            row.correct_top_1 for row in discriminating
        ),
        "teacher_forced_top_1": _finite_mean(row.correct_top_1 for row in assist),
        "teacher_forced_decision_count": len(assist),
        "teacher_forced_top_1_hits": sum(row.correct_top_1 for row in assist),
        "teacher_forced_top_k_hits": sum(row.correct_top_k for row in assist),
        "teacher_forced_nll": _finite_mean(row.ground_truth_nll for row in assist),
        "teacher_forced_preference_discriminating_decisions": (
            result.scored_discriminating_decisions
        ),
        "teacher_forced_preference_discriminating_top_1_hits": (
            result.scored_discriminating_top_1_hits
        ),
        "teacher_forced_preference_discriminating_top_1": (
            result.scored_discriminating_top_1
        ),
        "nontrivial_choice_decision_count": len(choice),
        "nontrivial_choice_top_1_hits": sum(row.correct_top_1 for row in choice),
        "single_legal_action_decision_count": len(forced),
        "invalid_predictions": result.invalid_predictions,
        "memory_age_delta": result.memory_age_delta,
        "commit_kind": getattr(result.match, "kind", None),
        "matched_recipe_id": getattr(result.match, "recipe_id", None),
        "matched_variant_id": getattr(result.match, "variant_id", None),
        "commit_applied": result.commit_applied,
        "active_rehearsal": result.active_rehearsal,
        "retrain_executed": result.retrain_executed,
        "retrain_correctly_skipped": result.retrain_correctly_skipped,
        "retrain_event_count": len(result.retrain_events),
        "lead_actor_policy": result.lead_actor_policy,
        "semantic_fallback_decisions": sum(
            row.prediction_stats.get("semantic_fallback_used", False) for row in assist
        ),
        # Semantic features are identity-masked, so every structurally
        # identical recipe sits at distance zero and the fallback always fires
        # between them.  Split accuracy by whether it fired, otherwise the
        # mechanism is assumed to help rather than shown to.
        "semantic_fallback_top_1_hits": sum(
            row.correct_top_1 for row in assist
            if row.prediction_stats.get("semantic_fallback_used", False)
        ),
        "semantic_fallback_discriminating_decisions": sum(
            1 for row in assist
            if row.prediction_stats.get("semantic_fallback_used", False)
            and row.preference_discriminating
        ),
        "semantic_fallback_discriminating_top_1_hits": sum(
            row.correct_top_1 for row in assist
            if row.prediction_stats.get("semantic_fallback_used", False)
            and row.preference_discriminating
        ),
        "own_model_decisions": sum(
            1 for row in assist
            if not row.prediction_stats.get("semantic_fallback_used", False)
        ),
        "own_model_top_1_hits": sum(
            row.correct_top_1 for row in assist
            if not row.prediction_stats.get("semantic_fallback_used", False)
        ),
        "own_model_discriminating_decisions": sum(
            1 for row in assist
            if not row.prediction_stats.get("semantic_fallback_used", False)
            and row.preference_discriminating
        ),
        "own_model_discriminating_top_1_hits": sum(
            row.correct_top_1 for row in assist
            if not row.prediction_stats.get("semantic_fallback_used", False)
            and row.preference_discriminating
        ),
        "latent_strategy_decisions": sum(
            row.prediction_stats.get("latent_strategy_used", False) for row in assist
        ),
        "compatibility_dynamics": result.compatibility_dynamics,
        "compatibility_calls": list(result.compatibility_calls),
        **_model_storage_metrics(agent),
        **_memory_metrics(agent),
        "decisions": [_decision_record(row) for row in result.decisions],
    }


def _run_cell(
    runtime: BurritoRuntime,
    config: Mapping[str, Any],
    *,
    seed: int,
    scenario: str,
    arm: str,
) -> Tuple[
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    Mapping[str, Any],
]:
    from src.models import Settings

    ladder_settings = _ladder_settings(config["ladder"])
    tasks, generated_audit = generate_ladder(
        seed=seed,
        scenario=scenario,
        recipe_ids=tuple(map(str, config["recipe_ids"])),
        settings=ladder_settings,
        return_audit=True,
    )
    overrides = dict(config.get("settings", {}))
    overrides.update({
        "verbose": False,
        "seed": int(seed),
        "semantic_fallback_max_rms_distance": float(overrides.get(
            "semantic_fallback_max_rms_distance",
            SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
        )),
    })
    domain = CookingDomainAdapter()
    agent = _build_agent(arm, Settings(**overrides), domain)
    offline_context = _prepare_offline_baseline(
        arm,
        agent,
        domain,
        tasks,
        config,
        seed=seed,
        scenario=scenario,
    )
    runner = CookingHrcRunner(
        agent,
        runtime,
        domain,
        horizon=int(config.get("horizon", 1800)),
        planner_seed=int(config.get("planner_seed", 11)),
        top_k=int(config.get("top_k", 3)),
        memory_updates_enabled=arm not in {"frozen", "offline_default"},
        require_shift_update=arm == FULL_ARM,
        lead_actor_policy=str(config.get("lead_actor_policy", "human_first")),
    )
    episodes: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    probes: list[Dict[str, Any]] = []
    audits: list[Dict[str, Any]] = []
    oracle_pruning: list[Mapping[str, Any]] = []
    # The panel is every behaviourally distinct (recipe, preference) pair.
    # It used to be truncated to a configured 48 while the catalog could only
    # supply far fewer, so the audit reported a panel size that never existed.
    # ``load_config`` now requires the declared size to equal the real one.
    frozen_pair_tasks = [
        CookingTask.create(recipe, preference)
        for recipe in map(str, config["recipe_ids"])
        for preference in applicable_preferences(recipe)
    ]
    isomorphic_targets = set(
        generated_audit.get("holdout_isomorphic_target_recipe_ids", ())
    )
    for event_index, task in enumerate(tasks):
        metadata = {
            "event_index": event_index,
            # A holdout target whose task graph is isomorphic to its source is
            # a relabelling of the container axis, not a generalisation of it.
            "holdout_transfer_isomorphic": bool(
                task.holdout_target and task.recipe_id in isomorphic_targets
            ),
            "seed": int(seed),
            "planner_seed": int(config.get("planner_seed", 11)),
            "scenario": scenario,
            "arm": arm,
            "is_full_arm": arm == FULL_ARM,
            "mode_schedule_policy": "matched_full_realized_execution_schedule",
            **dict(offline_context),
        }
        started = time.perf_counter()
        try:
            if (
                bool(config.get("pre_event_probes", True))
                and task.phase_role == "climb"
                and task.recipe_id in runner.observed_recipes
            ):
                probes.append(_frozen_probe(
                    agent,
                    domain,
                    task,
                    metadata=metadata,
                    probe_kind="pre_event_climb_probe",
                    top_k=int(config.get("top_k", 3)),
                ))
            result = runner.run_task(task)
        except Exception as error:
            failures.append({
                **metadata,
                "recipe_id": task.recipe_id,
                "phase": task.phase,
                "lifecycle": task.lifecycle,
                "failure_type": (
                    "planner_failure" if isinstance(error, OptionExecutionError)
                    else "system_failure"
                ),
                "error": f"{type(error).__name__}: {error}",
            })
            break
        episodes.append(_episode_record(
            result,
            metadata=metadata,
            wall_s=time.perf_counter() - started,
            agent=agent,
        ))
        if arm == "memory_oracle":
            oracle_pruning.append(_apply_memory_oracle_pruning(
                agent,
                tasks[event_index + 1:],
                runner.learner_recipe_by_task,
            ))
        audit_period = int(config.get("audit_period", 2))
        if audit_period > 0 and (event_index + 1) % audit_period == 0:
            context = {
                **metadata,
                "diagnostic_type": "active_only_pruned_influence_audit",
                "active_variants": len(agent.replay.active),
                "pruned_variants": len(agent.replay.pruned),
            }
            try:
                audit_result = dict(agent.audit_pruning(
                    max_prefixes=int(config.get("audit_prefixes", 16)),
                    tolerance=float(config.get("audit_tolerance", 0.05)),
                ))
                audits.append({**context, **audit_result, "audit_available": True})
            except Exception as error:
                audits.append({
                    **context,
                    "audit_available": False,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                })
        phase_boundary = (
            event_index + 1 == len(tasks)
            or tasks[event_index + 1].schedule_step != task.schedule_step
        )
        if phase_boundary:
            for frozen_task in frozen_pair_tasks:
                probes.append(_frozen_probe(
                    agent,
                    domain,
                    frozen_task,
                    metadata=metadata,
                    probe_kind="phase_boundary_frozen_panel_probe",
                    top_k=int(config.get("top_k", 3)),
                ))
    # A cell that raised stops at the failing episode.  Mark every row so the
    # pooled summary can exclude truncated cells instead of silently averaging
    # a 20-episode cell against a 210-episode one.
    cell_complete = not failures and len(episodes) == len(tasks)
    for row in episodes:
        row["cell_complete"] = cell_complete
        row["cell_planned_episodes"] = len(tasks)
    audit = dict(generated_audit)
    audit.update({
        "cell_complete": cell_complete,
        "cell_planned_episodes": len(tasks),
        "cell_completed_episodes": len(episodes),
        "shared_routing": bool(config.get("shared_routing", True)),
        "pre_event_probes": bool(config.get("pre_event_probes", True)),
        "frozen_pairs": len(frozen_pair_tasks),
        "audit_period": int(config.get("audit_period", 2)),
        "audit_prefixes": int(config.get("audit_prefixes", 16)),
        "audit_tolerance": float(config.get("audit_tolerance", 0.05)),
        "offline_training": dict(offline_context),
        "oracle_pruning_event_count": len(oracle_pruning),
        "oracle_pruned_variant_count": sum(
            int(row["oracle_pruned_variant_count"]) for row in oracle_pruning
        ),
        "pre_event_probe_count": sum(
            row["diagnostic_type"] == "pre_event_climb_probe" for row in probes
        ),
        "frozen_panel_probe_count": sum(
            row["diagnostic_type"] == "phase_boundary_frozen_panel_probe"
            for row in probes
        ),
        "active_only_audit_count": len(audits),
        "active_only_audit_failures": sum(
            row.get("passed") is False for row in audits
        ),
    })
    return episodes, failures, probes, audits, audit


def _retrain_settled(row: Mapping[str, Any]) -> bool:
    """Did the acquisition leave the deployed model fit on the committed set?

    ``BurritoHrcRunner`` already accepts a retrain that ``TrainPolicy``
    correctly skipped: when the shifted variant was still resident in active
    replay from an earlier occurrence, the policy sees no membership or weight
    change on this step and returns ``"replay_unchanged"``.  That is a
    legitimate no-op rather than a missed update, and the runtime records it as
    ``retrain_correctly_skipped``.

    This validator runs behind that runtime, so it has to apply the same rule.
    Testing ``retrain_executed`` alone rejected episodes the protocol had
    already accepted, which is why a completed run with zero execution
    failures still reported a validation failure.  Records written before the
    flag existed carry no value and fall back to the strict test.
    """
    if bool(row.get("retrain_executed")):
        return True
    return bool(row.get("retrain_correctly_skipped", False))


def _adaptation_records(episodes: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        adaptation_id = row.get("adaptation_id")
        if adaptation_id is None or int(row.get("exposure_after_change", 0)) < 1:
            continue
        key = (
            row["seed"], row["scenario"], row["arm"], str(adaptation_id),
        )
        grouped[key].append(row)
    records = []
    for key, rows in grouped.items():
        acquisition = [row for row in rows if row["exposure_after_change"] == 1]
        recurrence = [row for row in rows if row["exposure_after_change"] >= 2]
        if len(acquisition) != 1 or not recurrence:
            continue
        first = acquisition[0]
        records.append({
            "seed": key[0],
            "scenario": key[1],
            "arm": key[2],
            "adaptation_id": key[3],
            "recipe_id": first["recipe_id"],
            "acquisition_phase": first["phase"],
            "last_recurrence_phase": max(row["phase"] for row in recurrence),
            "preference": first["preference"],
            "strategy": first["strategy"],
            "acquisition_event_index": first["event_index"],
            "first_recurrence_event_index": min(row["event_index"] for row in recurrence),
            "intervening_episode_count": min(row["event_index"] for row in recurrence) - first["event_index"] - 1,
            "update_verified": bool(
                first["commit_applied"]
                and first["active_rehearsal"]
                and _retrain_settled(first)
            ),
            "acquisition_corrections": first["human_corrections"],
            # Corrections only ever occur on robot turns, so a preference whose
            # discriminating decision is the opening move shows zero
            # corrections however badly the learner adapts.  Carry the
            # teacher-forced discriminating accuracy alongside it.
            "acquisition_teacher_forced_discriminating_top_1": first.get(
                "teacher_forced_preference_discriminating_top_1"
            ),
            "post_update_teacher_forced_discriminating_top_1": _pooled_rate(
                recurrence,
                "teacher_forced_preference_discriminating_top_1_hits",
                "teacher_forced_preference_discriminating_decisions",
            ),
            "post_update_episode_count": len(recurrence),
            "post_update_correction_free_rate": _finite_mean(
                row["correction_free"] for row in recurrence
            ),
            "post_update_corrections_per_robot_decision": _finite_mean(
                row["corrections_per_robot_decision"] for row in recurrence
            ),
        })
    return sorted(records, key=lambda row: (
        row["seed"], row["scenario"], row["arm"],
        row["acquisition_phase"], row["recipe_id"],
    ))


def _human_action_load(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Human action load, plus the floor the turn-taking protocol imposes.

    The protocol is human-first strict alternation, so even a perfect robot
    performs ceil(n/2) of an n-step recipe.  The raw ratio therefore lives in
    roughly [0.5, 1.0] and looks saturated when it is merely bounded.
    ``excess`` rescales it onto [0, 1], where 0 is a robot that took every turn
    available to it and 1 is a robot that took none.
    """
    steps = sum(int(row["recipe_steps"]) for row in rows)
    human = sum(int(row["human_action_count"]) for row in rows)
    floor = sum(-(-int(row["recipe_steps"]) // 2) for row in rows)
    if not steps:
        return {
            "normalized_human_action_load": None,
            "normalized_human_action_load_floor": None,
            "human_action_load_excess": None,
        }
    span = steps - floor
    return {
        "normalized_human_action_load": human / steps,
        "normalized_human_action_load_floor": floor / steps,
        "human_action_load_excess": (human - floor) / span if span else None,
    }


def _by_seed(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-seed rates plus their macro-average.

    Pooling sums numerators and denominators across every row, so a seed whose
    heterogeneous ladder ran 1,395 episodes outweighs one that ran 1,080.
    Seeds are the unit of replication, so the seed-mean is the figure to draw
    inferences from and the pooled value is descriptive.
    """
    grouped: Dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    metrics = (
        ("preference_discriminating_top_1",
         "preference_discriminating_top_1_hits",
         "preference_discriminating_robot_decisions"),
        ("teacher_forced_preference_discriminating_top_1",
         "teacher_forced_preference_discriminating_top_1_hits",
         "teacher_forced_preference_discriminating_decisions"),
        ("robot_top_1", "robot_top_1_hits", "robot_decision_count"),
    )
    per_seed = []
    for seed, seed_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        entry: Dict[str, Any] = {"seed": seed, "n_episodes": len(seed_rows)}
        for name, numerator, denominator in metrics:
            entry[name] = _pooled_rate(seed_rows, numerator, denominator)
        entry.update(_human_action_load(seed_rows))
        per_seed.append(entry)
    seed_means = {
        f"{name}_seed_mean": _finite_mean(entry[name] for entry in per_seed)
        for name, _numerator, _denominator in metrics
    }
    seed_means["normalized_human_action_load_seed_mean"] = _finite_mean(
        entry["normalized_human_action_load"] for entry in per_seed
    )
    seed_means["human_action_load_excess_seed_mean"] = _finite_mean(
        entry["human_action_load_excess"] for entry in per_seed
    )
    return {"per_seed": per_seed, **seed_means}


def _aggregate(
    episodes: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]] = (),
    audits: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    # Truncated cells are reported but never pooled: their episode counts are
    # not comparable and summing numerators across them biases every rate.
    complete = [row for row in episodes if row.get("cell_complete", True)]
    excluded = [row for row in episodes if not row.get("cell_complete", True)]
    assist = [row for row in complete if row["mode"] == ASSIST]
    post_update = [
        row for row in assist if int(row["exposure_after_change"]) >= 2
    ]
    adaptation = _adaptation_records(complete)
    groups = []
    grouped: Dict[Tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in assist:
        grouped[(row["arm"], row["scenario"], row["environment"])].append(row)
    for (arm, scenario, environment), rows in sorted(grouped.items()):
        groups.append({
            "arm": arm,
            "scenario": scenario,
            "environment": environment,
            "n_episodes": len(rows),
            "robot_top_1": _pooled_rate(
                rows, "robot_top_1_hits", "robot_decision_count",
            ),
            "robot_top_k": _pooled_rate(
                rows, "robot_top_k_hits", "robot_decision_count",
            ),
            "preference_discriminating_top_1": _pooled_rate(
                rows,
                "preference_discriminating_top_1_hits",
                "preference_discriminating_robot_decisions",
            ),
            "nontrivial_choice_top_1": _pooled_rate(
                rows, "nontrivial_choice_top_1_hits", "nontrivial_choice_decision_count",
            ),
            "single_legal_action_fraction": (
                sum(row["single_legal_action_decision_count"] for row in rows)
                / max(1, sum(row["robot_decision_count"] for row in rows))
            ),
            "teacher_forced_preference_discriminating_top_1": _pooled_rate(
                rows,
                "teacher_forced_preference_discriminating_top_1_hits",
                "teacher_forced_preference_discriminating_decisions",
            ),
            **_human_action_load(rows),
            "corrections_per_robot_decision": (
                sum(row["human_corrections"] for row in rows)
                / max(1, sum(row["robot_decision_count"] for row in rows))
            ),
            "correction_free_rate": _finite_mean(row["correction_free"] for row in rows),
        })
    full_post = [row for row in post_update if row["arm"] == FULL_ARM]
    primary_probes = [
        row for row in probes
        if row.get("diagnostic_type") == "pre_event_climb_probe"
    ]
    probe_groups = []
    grouped_probes: Dict[Tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary_probes:
        grouped_probes[(row["arm"], row["scenario"], row["environment"])].append(row)
    for (arm, scenario, environment), rows in sorted(grouped_probes.items()):
        probe_groups.append({
            "arm": arm,
            "scenario": scenario,
            "environment": environment,
            "n_probes": len(rows),
            "preference_discriminating_top_1": _pooled_rate(
                rows,
                "preference_discriminating_top_1_hits",
                "preference_discriminating_decision_count",
            ),
            "top_1": _pooled_rate(rows, "top_1_hits", "decision_count"),
            "top_k": _pooled_rate(rows, "top_k_hits", "decision_count"),
        })
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed" if not failures else "completed_with_failures",
        "episode_count": len(episodes),
        "pooled_episode_count": len(complete),
        "assist_episode_count": len(assist),
        "post_update_episode_count": len(post_update),
        "failure_count": len(failures),
        "delivery_rate": _finite_mean(row["task_completed"] for row in episodes),
        "covered_recipe_count": len({row["recipe_id"] for row in episodes}),
        "covered_recipe_count_by_stratum": {
            stratum: len({
                row["recipe_id"] for row in episodes
                if row["environment"] == stratum
            })
            for stratum in STRATA
        },
        "covered_preference_count": len({row["preference"] for row in episodes}),
        "covered_preference_ids": sorted({row["preference"] for row in episodes}),
        "covered_strategy_count": len({row["strategy"] for row in episodes}),
        "primary_metric": "normalized_human_action_load",
        "primary_accuracy_metric": "preference_discriminating_top_1",
        "teacher_forced_accuracy_metric": (
            "teacher_forced_preference_discriminating_top_1"
        ),
        "metric_warning": (
            "overall top-1 includes forced single-legal-action decisions; "
            "use preference_discriminating_top_1 as the primary accuracy"
        ),
        "robot_top_1": _pooled_rate(
            assist, "robot_top_1_hits", "robot_decision_count",
        ),
        "robot_top_k": _pooled_rate(
            assist, "robot_top_k_hits", "robot_decision_count",
        ),
        "preference_discriminating_top_1": _pooled_rate(
            assist,
            "preference_discriminating_top_1_hits",
            "preference_discriminating_robot_decisions",
        ),
        "nontrivial_choice_top_1": _pooled_rate(
            assist, "nontrivial_choice_top_1_hits", "nontrivial_choice_decision_count",
        ),
        "single_legal_action_fraction": (
            sum(row["single_legal_action_decision_count"] for row in assist)
            / max(1, sum(row["robot_decision_count"] for row in assist))
        ),
        **_human_action_load(assist),
        "teacher_forced_preference_discriminating_top_1": _pooled_rate(
            assist,
            "teacher_forced_preference_discriminating_top_1_hits",
            "teacher_forced_preference_discriminating_decisions",
        ),
        "semantic_fallback_discriminating_top_1": _pooled_rate(
            assist,
            "semantic_fallback_discriminating_top_1_hits",
            "semantic_fallback_discriminating_decisions",
        ),
        "own_model_discriminating_top_1": _pooled_rate(
            assist,
            "own_model_discriminating_top_1_hits",
            "own_model_discriminating_decisions",
        ),
        "lead_actor_policies": sorted({
            row.get("lead_actor_policy") for row in assist
            if row.get("lead_actor_policy")
        }),
        # (recipe, preference) cells whose preference-discriminating decisions
        # never fall on a robot turn: measured for prediction, not assistance.
        "assistance_unscored_cells": sorted(
            [recipe_id, preference]
            for recipe_id, preference in {
                (row["recipe_id"], row["preference"]) for row in assist
            }
            if not any(
                row["preference_discriminating_robot_decisions"]
                for row in assist
                if row["recipe_id"] == recipe_id
                and row["preference"] == preference
            )
        ),
        "by_seed": _by_seed(assist),
        "excluded_incomplete_cell_episode_count": len(excluded),
        "excluded_incomplete_cells": sorted({
            (row["seed"], row["scenario"], row["arm"]) for row in excluded
        }),
        "adaptation_record_count": len(adaptation),
        "full_post_update_correction_free_rate": _finite_mean(
            row["correction_free"] for row in full_post
        ),
        "full_post_update_corrections_per_robot_decision": _finite_mean(
            row["corrections_per_robot_decision"] for row in full_post
        ),
        "groups": groups,
        "pre_event_probe_groups": probe_groups,
        "active_only_audit_count": len(audits),
        "active_only_audit_pass_rate": _finite_mean(
            row.get("passed") for row in audits
        ),
        "adaptation_records": adaptation,
    }


def _manifest(config: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    paths = UpstreamPaths.discover()
    project = paths.integration_root.parent
    status = _git(project, "status", "--porcelain")
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "run_dir": str(run_dir),
        "command": list(sys.argv),
        "config_path": config["_config_path"],
        "config_sha256": hashlib.sha256(_json_bytes(public)).hexdigest(),
        "config": public,
        "repositories": {
            "adaptive_hrc": {
                "commit": _git(project, "rev-parse", "HEAD"),
                "dirty": bool(status),
                "dirty_paths": status.splitlines(),
            },
            **{name: {"commit": commit} for name, commit in verify_pins(paths).items()},
        },
        "catalog": {
            "recipe_count": len(RECIPES),
            "overcooked_recipe_ids": list(OVERCOOKED_RECIPE_IDS),
            "burrito_recipe_ids": list(BURRITO_RECIPE_IDS),
            "talents_like_preferences": list(TALENTS_LIKE_PREFERENCES),
            "behaviorally_distinct_preference_ids": sorted({
                preference for recipe_id in RECIPES
                for preference in applicable_preferences(recipe_id)
            }),
            "behaviorally_distinct_preferences_by_recipe": {
                recipe_id: list(applicable_preferences(recipe_id))
                for recipe_id in RECIPES
            },
            "compatibility_recipe_ids": [
                recipe.recipe_id for recipe in RECIPES.values()
                if recipe.compatibility_dynamics
            ],
        },
        "features": {
            "reward": REWARD_FEATURE_VERSION,
            "semantic": SEMANTIC_FEATURE_VERSION,
            "semantic_fallback_max_rms_distance": SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
            "strategy_roles": STRATEGY_ROLE_VERSION,
        },
        "protocol": {
            "decision_level": "completion_checked_task_option",
            "player_0": "human",
            "player_1": "robot",
            "human_first": True,
            "wrong_robot_proposal_executed": False,
            "robot_retries_after_correction": True,
            "preference_policy": "deterministic_event_priority",
            "learning_state": "actor_independent_recipe_progress",
            "acquisition": "first_natural_exposure_after_preference_change",
            "primary_adaptation_window": "later_natural_recurrences_after_retraining",
            "adaptation_link": "persistent_shift_id_across_phase_boundaries",
            "scripted_probe_episodes": False,
            "pre_event_frozen_probes": bool(config.get("pre_event_probes", True)),
            "phase_boundary_frozen_panel_size": int(config.get("frozen_pairs", 48)),
            "shared_routing": bool(config.get("shared_routing", True)),
            "action_mask": "exact completion-checked task frontier",
            "accuracy_caveat": (
                "single-legal-action decisions are structurally forced and are "
                "excluded from the primary preference-discriminating metric"
            ),
            "primary_workload_metric": "normalized_human_action_load",
            "primary_accuracy_metric": "preference_discriminating_top_1",
        "teacher_forced_accuracy_metric": (
            "teacher_forced_preference_discriminating_top_1"
        ),
        },
        "scenario_design": {
            "homogeneous": (
                "seven ordered shared-strategy macro phases with a fixed "
                "210-demonstration budget"
            ),
            "heterogeneous": (
                "seven recipe-specific macro phases serialized into heavy-tailed "
                "climb/settle steps; length may exceed 210"
            ),
            "holdout": (
                "eight 45-episode source stages without container-first, followed "
                "by six 45-episode held-out stages that introduce it cross-environment "
                "and compose it with already-known target recipes"
            ),
        },
        "adaptive_hrc_parity": {
            "paired_seeds": list(config["seeds"]),
            "scenarios": list(config["scenarios"]),
            "baseline_roster": list(config["arms"]),
            "offline_recipe_fraction": float(config.get("offline_recipe_fraction", 0.5)),
            "offline_preference_fraction": float(config.get("offline_preference_fraction", 0.5)),
            "model_overrides": dict(config.get("settings", {})),
            "environment_specific_differences": [
                "recipe catalog and task-option state/action representation",
                "lifecycle operations conditioned on feasibility because each cooking recipe has only two or three distinct preferences",
                "physical option execution and legality",
            ],
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version,
            "executable": sys.executable,
        },
    }


def _mp_context() -> "multiprocessing.context.BaseContext":
    # "spawn" so a worker never inherits the parent's copy of the upstream
    # global recipe caches or NumPy RNG state.
    return multiprocessing.get_context("spawn")


# Each cell worker holds its own Overcooked simulator, planner and learner, so
# the ceiling is memory rather than cores. Defaulting to the CPU count
# exhausted a 30 GB machine and a killed worker surfaces as BrokenProcessPool
# hours into a run. An explicit --workers above this is honoured; the default
# and the "use every CPU" path are both clamped.
DEFAULT_CELL_WORKER_CAP = 10


def _worker_count(config: Mapping[str, Any], cells: int) -> int:
    requested = int(config.get("workers", 0) or 0)
    if requested <= 0:
        requested = os.cpu_count() or 1
    # Hard cap, matching src.evaluation: an explicit request above it is
    # clamped rather than honoured, because the ceiling is the machine's
    # memory and exceeding it fails the run hours in.
    return max(1, min(requested, DEFAULT_CELL_WORKER_CAP, max(1, cells)))


def _detect_p_core_cpus(pmu_path: Path = Path("/sys/devices/cpu_core/cpus")) -> Optional[FrozenSet[int]]:
    """CPU ids of the performance cores on a hybrid Intel part.

    Duplicated from ``src/evaluation.py`` -- this wrapper runs in an isolated
    Python 3.10 venv and cannot import the core package. Linux exposes a
    ``cpu_core`` PMU on hybrid parts (12th-gen+) whose ``cpus`` file lists
    exactly the P-core ids in ``lscpu`` range syntax (``"0-7"`` or
    ``"0-3,8-11"``); a uniform part has no such file.
    """
    try:
        text = pmu_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                cpus.update(range(int(start), int(end) + 1))
            except ValueError:
                return None
        else:
            try:
                cpus.add(int(part))
            except ValueError:
                return None
    return frozenset(cpus) if cpus else None


def _pin_to_performance_cores() -> Optional[FrozenSet[int]]:
    """Restrict this process to P-cores so cross-arm wall-clock stays comparable.

    An unpinned worker can be scheduled onto an E-core under load and run
    slower than a sibling worker on a P-core for reasons that have nothing to
    do with the arm it is executing. Every worker process gets the identical
    restriction, so any contention from oversubscribing the P-core set is at
    least shared rather than distinguishing between arms.
    """
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpus = _detect_p_core_cpus()
    if not cpus:
        return None
    try:
        os.sched_setaffinity(0, cpus)
    except OSError:
        return None
    return cpus


def _run_cell_job(
    config_path: str, seed: int, scenario: str, arm: str,
) -> Dict[str, Any]:
    """Run one seed/scenario/arm cell in its own process.

    Cells are independent -- each builds its own agent, runner and executors --
    but they cannot share a process: the pinned environment keeps recipe
    configuration, the shared ``complete_orders`` default and the planner's RNG
    in process-global state, which is exactly what the executor locks guard.
    Threads would serialise on that state and corrupt each other's planner
    stream; separate processes have neither problem.
    """
    _pin_to_performance_cores()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    config = load_config(config_path)
    runtime = BurritoRuntime.discover()
    rows, errors, cell_probes, cell_audits, audit = _run_cell(
        runtime, config, seed=int(seed), scenario=str(scenario), arm=str(arm),
    )
    return {
        "seed": int(seed),
        "scenario": str(scenario),
        "arm": str(arm),
        "episodes": rows,
        "failures": errors,
        "probes": cell_probes,
        "audits": cell_audits,
        "schedule": audit,
    }


def _cell_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("seed")), str(row.get("scenario")), str(row.get("arm")),
        int(row.get("event_index", 0) or 0),
        str(row.get("diagnostic_type", "")),
        str(row.get("recipe_id", "")), str(row.get("preference", "")),
    )


def _load_checkpoint(
    checkpoint_dir: Path, seed: int, scenario: str, arm: str,
) -> Dict[str, Any] | None:
    path = checkpoint_dir / f"{seed}__{scenario}__{arm}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def run_experiment(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    resume_from: str | Path | None = None,
    workers: int | None = None,
    progress: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path)
    if workers is not None:
        config = {**config, "workers": int(workers)}
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    digest = hashlib.sha256(_json_bytes(public)).hexdigest()[:10]
    root = Path(
        config["output"] if output_root is None else output_root
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Cell checkpoints used to be written and never read, so an interrupted
    # run of ~115k physical episodes had to start over.  Resuming replays the
    # completed cells from disk and only re-runs what is missing; the config
    # digest must match so a resumed run cannot mix two configurations.
    if resume_from is not None:
        run_dir = Path(resume_from).resolve()
        if not run_dir.is_dir():
            raise ValueError(f"cannot resume: {run_dir} is not a run directory")
        previous = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        if hashlib.sha256(_json_bytes(previous)).hexdigest()[:10] != digest:
            raise ValueError(
                "cannot resume: the run directory was produced by a different "
                "configuration"
            )
        manifest = _manifest(config, run_dir)
        manifest["resumed_from"] = str(run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / f"{config['experiment']}__{timestamp}__{digest}"
        manifest = _manifest(config, run_dir)
        run_dir.mkdir(parents=False, exist_ok=False)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=False, exist_ok=False)
    _atomic_json(run_dir / "config.json", public)
    _atomic_json(run_dir / "manifest.json", manifest)
    episodes: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    probes: list[Dict[str, Any]] = []
    audits: list[Dict[str, Any]] = []
    schedules: list[Dict[str, Any]] = []
    completed_cells: list[Dict[str, Any]] = []
    reused_cells: list[Dict[str, Any]] = []
    pending: list[Tuple[int, str, str]] = []
    try:
        for seed in map(int, config["seeds"]):
            for scenario in map(str, config["scenarios"]):
                for arm in map(str, config["arms"]):
                    pending.append((seed, scenario, arm))

        def absorb(cell: Mapping[str, Any], *, reused: bool) -> None:
            episodes.extend(cell["episodes"])
            failures.extend(cell["failures"])
            probes.extend(cell["probes"])
            audits.extend(cell["audits"])
            schedules.append({
                "seed": cell["seed"], "scenario": cell["scenario"],
                "arm": cell["arm"], **cell["schedule"],
            })
            record = {
                "seed": cell["seed"], "scenario": cell["scenario"],
                "arm": cell["arm"],
            }
            if reused:
                reused_cells.append(record)
            else:
                _atomic_json(
                    checkpoint_dir
                    / f"{cell['seed']}__{cell['scenario']}__{cell['arm']}.json",
                    dict(cell),
                )
            completed_cells.append({
                **record,
                "episode_count": len(cell["episodes"]),
                "failure_count": len(cell["failures"]),
                "reused_checkpoint": reused,
            })
            manifest["completed_cells"] = list(completed_cells)
            manifest["completed_cell_count"] = len(completed_cells)
            _atomic_json(run_dir / "manifest.json", manifest)
            if progress:
                print(
                    f"[cooking] completed {len(completed_cells)}/{len(pending)} "
                    f"seed={cell['seed']} scenario={cell['scenario']} "
                    f"arm={cell['arm']} reused={reused}",
                    file=sys.stderr,
                    flush=True,
                )

        outstanding: list[Tuple[int, str, str]] = []
        for seed, scenario, arm in pending:
            cached = (
                _load_checkpoint(checkpoint_dir, seed, scenario, arm)
                if resume_from is not None else None
            )
            if cached is not None:
                absorb(cached, reused=True)
            else:
                outstanding.append((seed, scenario, arm))

        workers = _worker_count(config, len(outstanding))
        manifest["workers"] = workers
        config_path_text = str(config["_config_path"])
        if workers <= 1 or len(outstanding) <= 1:
            _pin_to_performance_cores()
            runtime = BurritoRuntime.discover()
            for seed, scenario, arm in outstanding:
                rows, errors, cell_probes, cell_audits, audit = _run_cell(
                    runtime, config, seed=seed, scenario=scenario, arm=arm,
                )
                absorb({
                    "seed": seed, "scenario": scenario, "arm": arm,
                    "episodes": rows, "failures": errors, "probes": cell_probes,
                    "audits": cell_audits, "schedule": audit,
                }, reused=False)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=_mp_context(),
            ) as executor:
                futures = {
                    executor.submit(
                        _run_cell_job, config_path_text, seed, scenario, arm,
                    ): (seed, scenario, arm)
                    for seed, scenario, arm in outstanding
                }
                for future in as_completed(futures):
                    seed, scenario, arm = futures[future]
                    try:
                        absorb(future.result(), reused=False)
                    except BrokenProcessPool as error:
                        # Record the cell as lost rather than discarding the
                        # whole run; _aggregate excludes incomplete cells.
                        failures.append({
                            "seed": seed, "scenario": scenario, "arm": arm,
                            "failure_type": "worker_terminated",
                            "error": f"{type(error).__name__}: {error}",
                        })

        # Completion order depends on worker scheduling, so sort before writing:
        # the artefacts must be byte-identical whatever the worker count.
        episodes.sort(key=_cell_sort_key)
        probes.sort(key=_cell_sort_key)
        audits.sort(key=_cell_sort_key)
        failures.sort(key=_cell_sort_key)
        schedules.sort(key=_cell_sort_key)
        completed_cells.sort(key=_cell_sort_key)
        reused_cells.sort(key=_cell_sort_key)
        summary = _aggregate(episodes, failures, probes, audits)
        _atomic_json(run_dir / "episodes.json", episodes)
        _atomic_json(run_dir / "failures.json", failures)
        _atomic_json(run_dir / "probes.json", probes)
        _atomic_json(run_dir / "audits.json", audits)
        _atomic_json(run_dir / "schedules.json", schedules)
        _atomic_json(run_dir / "summary.json", summary)
        manifest.update({
            "status": summary["status"],
            "completed_at": _utc_now(),
            "episode_count": len(episodes),
            "failure_count": len(failures),
            "completed_cells": completed_cells,
            "completed_cell_count": len(completed_cells),
            "reused_checkpoint_cells": reused_cells,
            "reused_checkpoint_cell_count": len(reused_cells),
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


def validate_result(
    result: Mapping[str, Any], config_path: str | Path,
) -> Dict[str, Any]:
    config = load_config(config_path)
    summary = result["summary"]
    episodes = json.loads(
        (Path(result["run_dir"]) / "episodes.json").read_text(encoding="utf-8")
    )
    schedules = json.loads(
        (Path(result["run_dir"]) / "schedules.json").read_text(encoding="utf-8")
    )
    probes = json.loads(
        (Path(result["run_dir"]) / "probes.json").read_text(encoding="utf-8")
    )
    audits = json.loads(
        (Path(result["run_dir"]) / "audits.json").read_text(encoding="utf-8")
    )
    requirements = set(config.get("validation_requirements", ()))
    failures = []
    if summary["failure_count"]:
        failures.append(f"{summary['failure_count']} execution failures")
    if summary["delivery_rate"] != 1.0:
        failures.append(f"delivery rate is {summary['delivery_rate']!r}")
    full = [row for row in episodes if row["arm"] == FULL_ARM]
    acquisitions = [row for row in full if row["lifecycle"] == "acquire_shift"]
    recurrences = [row for row in full if row["exposure_after_change"] >= 2]
    if "verified_shift_update" in requirements and (
        not acquisitions or not all(
            row["commit_applied"] and row["active_rehearsal"] and _retrain_settled(row)
            for row in acquisitions
        )
    ):
        failures.append("not every natural acquisition committed, rehearsed, and retrained")
    if "post_update_recurrence" in requirements and not recurrences:
        failures.append("no natural post-update recurrence completed")
    if "preference_discrimination" in requirements and not any(
        row["preference_discriminating_robot_decisions"] > 0 for row in full
    ):
        failures.append("no preference-discriminating robot decision was scored")
    if "multiple_seeds" in requirements and len({row["seed"] for row in episodes}) < 2:
        failures.append("fewer than two seeds completed")
    if "comparison_arms" in requirements and {
        row["arm"] for row in episodes
    } != set(config["arms"]):
        failures.append("not every configured arm completed")
    if "adaptive_hrc_baseline_parity" in requirements:
        if tuple(config["arms"]) != ARM_NAMES:
            failures.append("baseline roster differs from Adaptive-HRC")
        if not probes or not any(
            row.get("diagnostic_type") == "pre_event_climb_probe" for row in probes
        ):
            failures.append("matched pre-event climb probes were not recorded")
        if not audits:
            failures.append("active-only replay audits were not recorded")
        if any(row.get("passed") is False for row in audits):
            failures.append("one or more active-only replay audits failed")
        schedule_lengths = {
            scenario: {int(row["episodes"]) for row in schedules if row["scenario"] == scenario}
            for scenario in SCENARIOS
        }
        if schedule_lengths["homogeneous"] != {210}:
            failures.append("homogeneous ladder is not 210 episodes")
        if schedule_lengths["holdout"] != {630}:
            failures.append("controlled holdout ladder is not 630 episodes")
        if not schedule_lengths["heterogeneous"] or min(
            schedule_lengths["heterogeneous"]
        ) <= 210:
            failures.append("heterogeneous ladder was not longitudinally serialized")
        if any(
            len(row.get("ingredients", ())) < 2 for row in episodes
        ):
            failures.append("single-ingredient episode entered the full evaluation")
        if any(
            decision.get("prediction_stats", {}).get("predictor")
            == "ground_truth_oracle"
            for row in episodes for decision in row.get("decisions", ())
        ):
            failures.append("ground-truth action oracle was used")
    if "catalog_coverage" in requirements:
        expected = set(config["recipe_ids"])
        covered = {row["recipe_id"] for row in full}
        if covered != expected:
            failures.append(f"recipe coverage mismatch: missing {sorted(expected - covered)}")
    if "behavioral_preference_coverage" in requirements:
        expected_preferences = {
            preference for recipe_id in config["recipe_ids"]
            for preference in applicable_preferences(str(recipe_id))
        }
        covered_preferences = {row["preference"] for row in full}
        if not expected_preferences <= covered_preferences:
            failures.append(
                "behavioral preference coverage mismatch: missing "
                f"{sorted(expected_preferences - covered_preferences)}"
            )
    if "natural_ladder" in requirements and any(
        row["lifecycle"] == "acquire_shift"
        and (
            row["exposure_after_change"] != 1
            or not row["preference_changed"]
            or row.get("adaptation_id") is None
        ) for row in episodes
    ):
        failures.append("an acquisition was not a first natural post-shift exposure")
    if "scenario_invariants" in requirements and (
        not schedules or not all(
            row.get("scenario_invariants_passed") is True for row in schedules
        )
    ):
        failures.append("one or more generated ladders failed scenario invariants")
    if "adaptation_linkage" in requirements:
        full_acquisition_ids = {
            row.get("adaptation_id") for row in acquisitions
        }
        linked_ids = {
            row.get("adaptation_id")
            for row in summary.get("adaptation_records", ())
            if row.get("arm") == FULL_ARM
        }
        missing = full_acquisition_ids - linked_ids
        if None in full_acquisition_ids or missing:
            failures.append(
                f"natural acquisitions missing post-update linkage: {sorted(map(str, missing))}"
            )
    if "cross_environment_adaptation" in requirements and not {
        "overcooked", "burrito_native"
    } <= {row["environment"] for row in acquisitions}:
        failures.append(
            "full-system acquisitions did not cover Overcooked and natively "
            "executed Burrito"
        )
    if "holdout_transfer_is_generalisation" in requirements:
        holdout_rows = [
            row for row in full
            if row["scenario"] == "holdout" and row.get("holdout_target")
        ]
        if not holdout_rows:
            failures.append("the holdout scenario produced no target episodes")
        else:
            # Per stratum, not just overall: an Overcooked generalisation does
            # not license a cross-environment claim if every Burrito target is
            # a relabelling of its source.
            for stratum in sorted({row["environment"] for row in holdout_rows}):
                generalising = {
                    row["recipe_id"] for row in holdout_rows
                    if row["environment"] == stratum
                    and not row.get("holdout_transfer_isomorphic")
                }
                if not generalising:
                    failures.append(
                        f"every {stratum} holdout target is isomorphic to its "
                        "source; the container axis is relabelled there, not "
                        "generalised"
                    )
    if "stratum_separation" in requirements:
        # Wrapper-restored compatibility dynamics must never be pooled with
        # natively executed episodes.
        mixed = {
            row["environment"] for row in episodes
            if row["environment"] not in STRATA
        }
        if mixed:
            failures.append(f"episodes carry unknown strata: {sorted(mixed)}")
        for row in episodes:
            expected = "burrito_compat" if row.get("compatibility_dynamics") else None
            if expected is not None and row["environment"] != expected:
                failures.append(
                    "a compatibility episode was recorded outside the "
                    "burrito_compat stratum"
                )
                break
        if len({row["environment"] for row in full}) < len(STRATA):
            failures.append("the full arm did not cover every reporting stratum")
    if "framework_components" in requirements:
        if not full or not all(
            row["learner_model_structure"]
            == "maxent_irl_with_latent_strategy_residual"
            for row in full
        ):
            failures.append("full arm did not run MaxEnt IRL with latent strategy")
        if not any(row["semantic_fallback_decisions"] > 0 for row in full):
            failures.append("semantic-distance fallback was never exercised")
        if not any(row["latent_strategy_prototypes"] > 0 for row in full):
            failures.append("latent strategy was never fitted")
        if not any(
            row["memory_nonunit_weights"] > 0
            or row["memory_pruned_variants"] > 0 for row in full
        ):
            failures.append("adaptive rehearsal weighting was never exercised")
    if "transfer_mechanisms_exercised" in requirements:
        if not any(row["semantic_fallback_decisions"] > 0 for row in full):
            failures.append("semantic-distance transfer was never used")
        if not any(row["latent_strategy_decisions"] > 0 for row in full):
            failures.append("latent-strategy transfer was never used")
    if "open_set_recipe_separation" in requirements:
        cells: Dict[Tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in full:
            cells[(row["seed"], row["scenario"])].append(row)
        if not cells or any(
            max(row["learner_registry_recipe_count"] for row in rows)
            != len({row["recipe_id"] for row in rows})
            for rows in cells.values()
        ):
            failures.append("open-set learner did not preserve one identity per recipe")
    if any(row["invalid_predictions"] for row in episodes):
        failures.append("one or more task-graph-invalid predictions occurred")
    validation = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "requirements": sorted(requirements),
    }
    _atomic_json(Path(result["run_dir"]) / "validation.json", validation)
    if failures:
        raise RuntimeError("evaluation validation failed: " + "; ".join(failures))
    return validation


__all__ = [
    "ARM_NAMES", "FULL_ARM", "VALIDATION_REQUIREMENTS", "load_config",
    "run_experiment", "validate_result",
]
