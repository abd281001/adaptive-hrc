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
import re
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
from .ladder import (
    CONTAINER_FIRST_PREFERENCE,
    LadderSettings,
    SCENARIOS,
    generate_ladder,
    ladder_audit,
)
from .options import OptionExecutionError
from .protocol import (
    ASSIST,
    CookingEpisodeResult,
    CookingHrcRunner,
    CookingObservation,
    CookingTask,
)
from .runtime import BurritoRuntime, UpstreamPaths, verify_pins
from .null_agents import NULL_AGENTS
from .task_graph import (
    CookingPreferencePolicy,
    CookingTaskGraph,
    is_preference_discriminating,
    is_prefix_conditioned_discriminating,
)


CONFIG_SCHEMA_VERSION = 4
RESULT_SCHEMA_VERSION = 7
FULL_ARM = "full"
ARM_NAMES: Tuple[str, ...] = (
    FULL_ARM,
    "frozen",
    "offline_default",
    "offline_all",
    "unpinned",
    "latest",
    "fixed",
    "no_decay",
    "bc",
    "ewc",
    "replay_bc",
    "memory_oracle",
)
# Zero-learning references, not deployable systems.  They are excluded from the
# Adaptive-HRC parity roster on purpose: parity is about matching the symbolic
# baseline set, and these arms have no symbolic counterpart.  They are required
# in the full config because a Full-versus-baseline margin is uninterpretable
# without knowing what a fixed rule scores on the same episodes.
NULL_ARM_NAMES: Tuple[str, ...] = ("canonical_order",)
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
    "zero_learning_reference",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


# The core modules a cooking cell's result actually depends on.  The wrapper
# supplies the environment and the protocol; everything the learner does comes
# from src, so hashing only the wrapper left a checkpoint valid across an edit
# to the agent, the model, the memory policy or the baselines.
_CORE_SOURCE_MODULES: Tuple[str, ...] = (
    "adaptive_agent.py",
    "baselines.py",
    "domain.py",
    "environment.py",
    "latent_strategy.py",
    "memory.py",
    "models.py",
    "preferences.py",
    "representations.py",
)


def _source_fingerprint() -> str:
    """Digest of the wrapper package plus the core learner implementation.

    A checkpoint records the code that produced it.  The config digest alone
    was not enough: an edit to the evaluator, the protocol, or the learner
    changes what a cell means without changing the config, and a resume would
    then splice cells produced by two different implementations into one
    artifact.
    """
    paths = UpstreamPaths.discover()
    package = Path(__file__).resolve().parent
    core = paths.integration_root.parent / "src"
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    for name in _CORE_SOURCE_MODULES:
        path = core / name
        digest.update(f"src/{name}".encode("utf-8"))
        # A module that has moved or been renamed is itself a change worth
        # invalidating on, so record its absence rather than skipping it.
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()[:16]


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
    unknown_arms = (
        set(map(str, config["arms"])) - set(ARM_NAMES) - set(NULL_ARM_NAMES)
    )
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
    # The deployable roster must match Adaptive-HRC in order; the
    # zero-learning references follow it.  They are required, not optional: a
    # Full-versus-baseline margin cannot be read without knowing what a fixed
    # rule scores on the same episodes, and roughly half of this catalog's
    # robot turns have one legal option.
    if tuple(map(str, config["arms"])) != ARM_NAMES + NULL_ARM_NAMES:
        raise ValueError(
            "the full cooking evaluation must use the Adaptive-HRC baseline "
            f"roster in order, followed by the zero-learning references: "
            f"{ARM_NAMES + NULL_ARM_NAMES}"
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

    if arm in NULL_AGENTS:
        return NULL_AGENTS[arm](settings, domain=domain)
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
    """Match Adaptive-HRC's three frozen training regimes."""
    if arm not in {"frozen", "offline_default", "offline_all"}:
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
    elif arm == "offline_default":
        recipes = sorted({task.recipe_id for task in tasks})
        pairs = [(recipe, _canonical_preference(recipe)) for recipe in recipes]
        design = "all_recipes_default_only"
    else:
        # Every behaviourally distinct preference of every recipe this seed
        # schedules.  Recipes outside the seed's set stay out: no other arm
        # ever sees them.
        recipes = sorted({task.recipe_id for task in tasks})
        pairs = [
            (recipe, preference)
            for recipe in recipes for preference in applicable_preferences(recipe)
        ]
        design = "all_recipes_all_preferences"
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
                "available": predicted is not None,
                "discriminating": is_preference_discriminating(legal, graph),
                "conditioned": is_prefix_conditioned_discriminating(
                    legal, graph, completed,
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
    # As above: a probe decision the arm could not answer is a miss, not an
    # absence.
    scored = rows
    discriminating = [row for row in rows if row["discriminating"]]
    conditioned = [row for row in rows if row["conditioned"]]
    nontrivial = [row for row in scored if row["legal_count"] > 1]
    return {
        **dict(metadata),
        "diagnostic_type": probe_kind,
        "recipe_id": task.recipe_id,
        "environment": RECIPES[task.recipe_id].stratum,
        "preference": task.preference,
        "decision_count": len(scored),
        "prediction_available_decisions": sum(
            row["available"] for row in rows
        ),
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
        "prefix_conditioned_decision_count": len(conditioned),
        "prefix_conditioned_top_1_hits": sum(row["correct"] for row in conditioned),
        "prefix_conditioned_top_1": _finite_mean(
            row["correct"] for row in conditioned
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
        "prefix_conditioned_discriminating": row.prefix_conditioned_discriminating,
        "prediction_available": row.prediction_available,
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
    # Every assist decision is a scored opportunity.  Filtering on
    # ``predicted is not None`` removed the decisions an arm failed to answer
    # from the denominator of every rate, which inflated precisely the arms
    # that fail to answer; availability is reported instead of subtracted.
    assist = [row for row in result.decisions if row.mode == ASSIST]
    robot = [row for row in assist if row.scheduled_actor == "robot"]
    if len(robot) != result.robot_turns:
        raise RuntimeError(
            f"robot decision count {len(robot)} disagrees with the "
            f"{result.robot_turns} scheduled robot turns"
        )
    if result.corrections > len(robot):
        raise RuntimeError(
            f"{result.corrections} corrections exceed {len(robot)} robot turns"
        )
    discriminating = [row for row in robot if row.preference_discriminating]
    # The accuracy denominator: decisions where the preferences still
    # consistent with this episode's prefix disagree.  ``discriminating``
    # counts decisions the prefix has already settled.
    conditioned = [row for row in robot if row.prefix_conditioned_discriminating]
    scored_conditioned = [
        row for row in assist if row.prefix_conditioned_discriminating
    ]
    choice = [row for row in robot if len(row.legal_actions) > 1]
    forced = [row for row in robot if len(row.legal_actions) == 1]
    fits = [
        event for event in result.retrain_events
        if not bool(event.get("skipped", False))
    ]
    # The opening move is the most preference-informative decision in these
    # task graphs and under human_first it is never a robot turn, so no
    # robot-turn metric can see it.  On the container-axis transfer cell it is
    # discriminating in every episode, which makes it the transfer measurement.
    opening = assist[0] if assist else None
    # From the result, not the agent: the runner owns the effective floor and
    # may have been given an explicit override.
    nll_floor = float(result.nll_probability_floor)
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
        # Pooling needs the sum and its denominator.  Averaging episode means
        # is a macro-average over episodes of unequal length, which is not the
        # per-decision loss anyone reads it as.
        "teacher_forced_nll_total": sum(
            row.ground_truth_nll for row in assist
            if math.isfinite(row.ground_truth_nll)
        ),
        "teacher_forced_nll_decisions": sum(
            1 for row in assist if math.isfinite(row.ground_truth_nll)
        ),
        "nll_probability_floor": nll_floor,
        "teacher_forced_preference_discriminating_decisions": (
            result.scored_discriminating_decisions
        ),
        "teacher_forced_preference_discriminating_top_1_hits": (
            result.scored_discriminating_top_1_hits
        ),
        "teacher_forced_preference_discriminating_top_1": (
            result.scored_discriminating_top_1
        ),
        "prefix_conditioned_robot_decisions": len(conditioned),
        "prefix_conditioned_top_1_hits": sum(
            row.correct_top_1 for row in conditioned
        ),
        "prefix_conditioned_top_1": _finite_mean(
            row.correct_top_1 for row in conditioned
        ),
        "teacher_forced_prefix_conditioned_decisions": len(scored_conditioned),
        "teacher_forced_prefix_conditioned_top_1_hits": sum(
            row.correct_top_1 for row in scored_conditioned
        ),
        "teacher_forced_prefix_conditioned_top_1": _finite_mean(
            row.correct_top_1 for row in scored_conditioned
        ),
        "opening_scored": opening is not None,
        "opening_scheduled_actor": (
            None if opening is None else opening.scheduled_actor
        ),
        "opening_preference_discriminating": (
            None if opening is None else opening.preference_discriminating
        ),
        "opening_prefix_conditioned_discriminating": (
            None if opening is None
            else opening.prefix_conditioned_discriminating
        ),
        "opening_top_1_hits": (
            0 if opening is None else int(opening.correct_top_1)
        ),
        "opening_decision_count": int(opening is not None),
        "opening_discriminating_top_1_hits": (
            int(opening.correct_top_1)
            if opening is not None and opening.preference_discriminating else 0
        ),
        "opening_discriminating_decision_count": int(
            opening is not None and opening.preference_discriminating
        ),
        "prediction_wall_s": result.prediction_wall_s,
        # Reported next to every rate, never folded into one.
        "prediction_available_decisions": result.scored_available_decisions,
        "prediction_unavailable_decisions": (
            result.scored_turns - result.scored_available_decisions
        ),
        "robot_prediction_available_decisions": result.robot_available_decisions,
        "robot_prediction_unavailable_decisions": (
            result.robot_turns - result.robot_available_decisions
        ),
        "fit_count": len(fits),
        "fit_total_wall_s": sum(float(event.get("total_wall_s", 0.0)) for event in fits),
        "fit_wall_s_values": [float(event.get("fit_wall_s", 0.0)) for event in fits],
        "fit_flop_estimate": sum(
            float(event.get("flop_estimate", 0.0)) for event in fits
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
        memory_updates_enabled=arm not in {"frozen", "offline_default", "offline_all"},
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


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _performance(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute cost, latency, memory and calibration, aggregated.

    Every column here was already recorded per episode and reported nowhere,
    so a cross-environment claim about the system's cost could not be checked
    against this run at all.  Prediction latency is per decision (a blocking
    wait inside an episode); fit latency is per fit (the blocking wait between
    two demonstrations), which is why both a total and a p95 are reported
    rather than one mean over a mixture of the two.
    """
    def total(key: str) -> float:
        return sum(float(row.get(key, 0.0) or 0.0) for row in rows)

    fit_waits = [
        value for row in rows for value in (row.get("fit_wall_s_values") or ())
    ]
    decisions = sum(
        int(row.get("teacher_forced_decision_count", 0) or 0) for row in rows
    )
    prediction_wall = total("prediction_wall_s")
    available = sum(
        int(row.get("prediction_available_decisions", 0) or 0) for row in rows
    )
    robot_available = sum(
        int(row.get("robot_prediction_available_decisions", 0) or 0)
        for row in rows
    )
    robot_decisions = sum(
        int(row.get("robot_decision_count", 0) or 0) for row in rows
    )
    nll_decisions = sum(
        int(row.get("teacher_forced_nll_decisions", 0) or 0) for row in rows
    )
    sizes = [
        int(row["learner_dense_array_bytes"]) for row in rows
        if row.get("learner_dense_array_bytes") is not None
    ]
    return {
        "episode_wall_s": total("task_wall_s"),
        "prediction_wall_s": prediction_wall,
        "mean_prediction_wall_s": (
            prediction_wall / decisions if decisions else None
        ),
        "fit_count": sum(int(row.get("fit_count", 0) or 0) for row in rows),
        "fit_total_wall_s": total("fit_total_wall_s"),
        "p50_fit_wall_s": _percentile(fit_waits, 0.50),
        "p95_fit_wall_s": _percentile(fit_waits, 0.95),
        "fit_flop_estimate": total("fit_flop_estimate"),
        "peak_learner_dense_array_bytes": max(sizes, default=None),
        "mean_memory_active_variants": _finite_mean(
            row.get("memory_active_variants") for row in rows
        ),
        "mean_replay_transition_count": _finite_mean(
            row.get("learner_replay_transition_count") for row in rows
        ),
        # Total loss over total decisions.  This previously averaged the
        # per-episode means, which with unequal episode lengths is a different
        # number entirely -- 1.0 against a true 0.2 per decision in the
        # regression that now guards it.
        "teacher_forced_nll": (
            total("teacher_forced_nll_total") / nll_decisions
            if nll_decisions else None
        ),
        "teacher_forced_nll_decisions": nll_decisions,
        "nll_probability_floor": next(
            (
                row["nll_probability_floor"] for row in rows
                if row.get("nll_probability_floor") is not None
            ),
            None,
        ),
        # Availability is a property of the arm, reported beside the rates
        # rather than removed from their denominators.
        "prediction_availability": (available / decisions if decisions else None),
        "prediction_unavailable_decisions": decisions - available,
        "robot_prediction_availability": (
            robot_available / robot_decisions if robot_decisions else None
        ),
        "robot_prediction_unavailable_decisions": robot_decisions - robot_available,
        "invalid_prediction_count": sum(
            int(row.get("invalid_predictions", 0) or 0) for row in rows
        ),
        "task_completion_rate": _finite_mean(
            row.get("task_completed") for row in rows
        ),
    }


# Two-sided 95% Student-t multipliers by degrees of freedom, for the paired
# seed differences below.  With eight seeds the normal multiplier understates
# the interval by about 20%, and the whole point of reporting seeds as the unit
# of replication is not to overstate what eight of them support.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

SEED_UNIT_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("opening_discriminating_top_1",
     "opening_discriminating_top_1_hits",
     "opening_discriminating_decision_count"),
    ("teacher_forced_top_1",
     "teacher_forced_top_1_hits", "teacher_forced_decision_count"),
    ("prefix_conditioned_top_1",
     "prefix_conditioned_top_1_hits", "prefix_conditioned_robot_decisions"),
    ("teacher_forced_prefix_conditioned_top_1",
     "teacher_forced_prefix_conditioned_top_1_hits",
     "teacher_forced_prefix_conditioned_decisions"),
    ("preference_discriminating_top_1",
     "preference_discriminating_top_1_hits",
     "preference_discriminating_robot_decisions"),
    ("teacher_forced_preference_discriminating_top_1",
     "teacher_forced_preference_discriminating_top_1_hits",
     "teacher_forced_preference_discriminating_decisions"),
    ("robot_top_1", "robot_top_1_hits", "robot_decision_count"),
    ("corrections_per_robot_decision",
     "human_corrections", "robot_decision_count"),
)


def _paired_difference(values: Sequence[float]) -> Dict[str, Any]:
    """Mean, spread and a t-based 95% interval for one paired contrast."""
    usable = [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not usable:
        return {
            "n_seeds": 0, "mean": None, "sd": None,
            "ci95_half_width": None, "ci95": None,
        }
    mean = statistics.fmean(usable)
    if len(usable) < 2:
        return {
            "n_seeds": len(usable), "mean": mean, "sd": None,
            "ci95_half_width": None, "ci95": None,
        }
    sd = statistics.stdev(usable)
    half = _T95.get(len(usable) - 1, 1.96) * sd / math.sqrt(len(usable))
    return {
        "n_seeds": len(usable),
        "mean": mean,
        "sd": sd,
        "ci95_half_width": half,
        "ci95": [mean - half, mean + half],
    }


def _by_seed(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Seed-level cells, their macro-averages, and paired contrasts against Full.

    The unit of replication is one (seed, arm, scenario, environment) cell.
    Grouping by seed alone pooled all twelve arms into a single per-seed rate,
    which is not a quantity anyone can draw an inference from, and it left the
    run with no unit in which a Full-versus-arm difference or an
    environment-by-method interaction could be tested at all.

    Pooling within a cell still sums numerators and denominators across
    episodes, so a longer heterogeneous ladder weighs more inside its own cell;
    across seeds the macro-average is what the contrasts use.
    """
    cells: Dict[Tuple[Any, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(
            row["seed"], row["arm"], row["scenario"], row["environment"],
        )].append(row)

    per_cell = []
    indexed: Dict[Tuple[str, str, str], Dict[Any, Dict[str, Any]]] = defaultdict(dict)
    for (seed, arm, scenario, environment), cell_rows in sorted(
        cells.items(), key=lambda item: tuple(map(str, item[0])),
    ):
        entry: Dict[str, Any] = {
            "seed": seed, "arm": arm, "scenario": scenario,
            "environment": environment, "n_episodes": len(cell_rows),
        }
        for name, numerator, denominator in SEED_UNIT_METRICS:
            entry[name] = _pooled_rate(cell_rows, numerator, denominator)
        per_cell.append(entry)
        indexed[(arm, scenario, environment)][seed] = entry

    seed_means = []
    for (arm, scenario, environment), by_seed_entry in sorted(indexed.items()):
        row: Dict[str, Any] = {
            "arm": arm, "scenario": scenario, "environment": environment,
            "n_seeds": len(by_seed_entry),
        }
        for name, _numerator, _denominator in SEED_UNIT_METRICS:
            row[f"{name}_seed_mean"] = _finite_mean(
                entry[name] for entry in by_seed_entry.values()
            )
        seed_means.append(row)

    # Paired within seed, so a seed whose panel happens to be hard does not
    # count as evidence against a method.
    contrasts = []
    for (arm, scenario, environment), by_seed_entry in sorted(indexed.items()):
        if arm == FULL_ARM:
            continue
        reference = indexed.get((FULL_ARM, scenario, environment), {})
        shared = sorted(set(reference) & set(by_seed_entry), key=str)
        if not shared:
            continue
        contrast: Dict[str, Any] = {
            "arm": arm, "scenario": scenario, "environment": environment,
            "contrast": f"{FULL_ARM}_minus_{arm}",
        }
        for name, _numerator, _denominator in SEED_UNIT_METRICS:
            paired = [
                reference[seed][name] - by_seed_entry[seed][name]
                for seed in shared
                if reference[seed][name] is not None
                and by_seed_entry[seed][name] is not None
            ]
            contrast[name] = _paired_difference(paired)
        contrasts.append(contrast)

    return {
        "unit": "seed_x_arm_x_scenario_x_environment",
        "per_cell": per_cell,
        "seed_means": seed_means,
        "paired_vs_full": contrasts,
    }


HOLDOUT_SOURCE_TRAINING = "holdout_source_training"
HOLDOUT_AXIS_INTRODUCTION = "holdout_axis_source_introduction"
HOLDOUT_AXIS_TARGET = "holdout_axis_target_composition"


def _holdout_stage(row: Mapping[str, Any]) -> Tuple[str, str, bool, str]:
    """Classify one holdout episode: stage, recipe role, is-it-the-axis, exposure.

    Stage and recipe role are independent and both are needed.  The stage comes
    from the ladder's own ``strategy`` label -- which phase of the holdout this
    episode belongs to -- and the role comes from ``holdout_target``, which
    says whether this episode's *recipe* is a held-out target.  Source recipes
    keep appearing throughout the target-composition phase, so the stage alone
    does not isolate targets: on the saved run the axis "later exposure" group
    was 1,022 Overcooked episodes of which 250 were source recipes, and 384
    native-Burrito episodes of which 211 were.

    Neither does the role alone, which is what an earlier version tried:
    combining ``holdout_target`` with ``exposure_after_change <= 1`` counted
    ordinary source training and acquisitions of unrelated preferences as
    transfer, 1,500 and 257 episodes against 24 and 12 actual ones.
    """
    strategy = str(row.get("strategy", ""))
    role = "target" if row.get("holdout_target") else "source"
    axis = row.get("preference") == CONTAINER_FIRST_PREFERENCE
    if strategy == HOLDOUT_AXIS_TARGET and axis and role == "target":
        exposure = (
            "first" if int(row["exposure_after_change"]) == 1 else "later"
        )
    else:
        exposure = "not_applicable"
    return strategy or "unlabelled", role, axis, exposure


def _holdout_groups(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Split the holdout into its four stages, keeping seeds as separate rows.

    Only one cell is the transfer measurement: the *first* exposure of the
    container-first preference on a held-out target
    (``holdout_axis_target_composition``, ``axis_preference``,
    ``target_exposure == "first"``).  Source training never contains the axis
    at all, the introduction stage teaches it on an already-known source, and
    later target exposures are ordinary adaptation on a now-demonstrated pair
    -- and they outnumber the first ones by roughly fifty to one, so a pooled
    holdout figure is almost entirely adaptation reported as transfer.

    Seeds stay separate here rather than being pooled away, because with two
    dozen first exposures per environment the seed is the only honest unit.
    Read the teacher-forced columns: at 24 first exposures the robot-turn
    metrics miss every human-first opening, and the opening move is the most
    preference-informative decision in these task graphs.
    """
    holdout = [row for row in rows if row["scenario"] == "holdout"]
    grouped: Dict[Tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in holdout:
        stage, role, axis, exposure = _holdout_stage(row)
        grouped[(
            row["arm"], row["seed"], row["environment"], stage, role, axis,
            bool(row.get("holdout_transfer_isomorphic")), exposure,
        )].append(row)

    groups = []
    for key, cell in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        arm, seed, environment, stage, role, axis, isomorphic, exposure = key
        groups.append({
            "arm": arm,
            "seed": seed,
            "environment": environment,
            "holdout_stage": stage,
            "holdout_recipe_role": role,
            "axis_preference": axis,
            "holdout_transfer_isomorphic": isomorphic,
            "target_exposure": exposure,
            "is_transfer_measurement": (
                stage == HOLDOUT_AXIS_TARGET
                and role == "target"
                and axis
                and exposure == "first"
            ),
            "n_episodes": len(cell),
            "teacher_forced_top_1": _pooled_rate(
                cell, "teacher_forced_top_1_hits", "teacher_forced_decision_count",
            ),
            "teacher_forced_decision_count": sum(
                row.get("teacher_forced_decision_count", 0) or 0 for row in cell
            ),
            "teacher_forced_prefix_conditioned_top_1": _pooled_rate(
                cell,
                "teacher_forced_prefix_conditioned_top_1_hits",
                "teacher_forced_prefix_conditioned_decisions",
            ),
            "teacher_forced_prefix_conditioned_decisions": sum(
                row.get("teacher_forced_prefix_conditioned_decisions", 0) or 0
                for row in cell
            ),
            "prefix_conditioned_top_1": _pooled_rate(
                cell,
                "prefix_conditioned_top_1_hits",
                "prefix_conditioned_robot_decisions",
            ),
            "prefix_conditioned_robot_decisions": sum(
                row.get("prefix_conditioned_robot_decisions", 0) or 0
                for row in cell
            ),
            "opening_top_1": _pooled_rate(
                cell, "opening_top_1_hits", "opening_decision_count",
            ),
            "opening_discriminating_top_1": _pooled_rate(
                cell,
                "opening_discriminating_top_1_hits",
                "opening_discriminating_decision_count",
            ),
            "opening_discriminating_decision_count": sum(
                row.get("opening_discriminating_decision_count", 0) or 0
                for row in cell
            ),
            "corrections_per_robot_decision": _pooled_rate(
                cell, "human_corrections", "robot_decision_count",
            ),
        })
    return groups


def _holdout_transfer_contrasts(
    groups: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Full-minus-arm on the transfer cell alone, paired within seed."""
    transfer = [row for row in groups if row["is_transfer_measurement"]]
    indexed: Dict[Tuple[str, str, bool], Dict[Any, Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    for row in transfer:
        indexed[(
            row["arm"], row["environment"], row["holdout_transfer_isomorphic"],
        )][row["seed"]] = row

    metrics = (
        "teacher_forced_top_1",
        "teacher_forced_prefix_conditioned_top_1",
        "prefix_conditioned_top_1",
        "opening_discriminating_top_1",
    )
    contrasts = []
    for (arm, environment, isomorphic), by_seed in sorted(
        indexed.items(), key=lambda item: tuple(map(str, item[0])),
    ):
        if arm == FULL_ARM:
            continue
        reference = indexed.get((FULL_ARM, environment, isomorphic), {})
        shared = sorted(set(reference) & set(by_seed), key=str)
        if not shared:
            continue
        entry: Dict[str, Any] = {
            "arm": arm,
            "environment": environment,
            "holdout_transfer_isomorphic": isomorphic,
            "contrast": f"{FULL_ARM}_minus_{arm}",
            "n_transfer_episodes": sum(
                by_seed[seed]["n_episodes"] for seed in shared
            ),
        }
        for name in metrics:
            entry[name] = _paired_difference([
                reference[seed][name] - by_seed[seed][name]
                for seed in shared
                if reference[seed][name] is not None
                and by_seed[seed][name] is not None
            ])
        contrasts.append(entry)
    return contrasts


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
            "teacher_forced_top_1": _pooled_rate(
                rows, "teacher_forced_top_1_hits", "teacher_forced_decision_count",
            ),
            "teacher_forced_decision_count": sum(
                row.get("teacher_forced_decision_count", 0) or 0 for row in rows
            ),
            "prefix_conditioned_top_1": _pooled_rate(
                rows,
                "prefix_conditioned_top_1_hits",
                "prefix_conditioned_robot_decisions",
            ),
            "prefix_conditioned_robot_decisions": sum(
                row.get("prefix_conditioned_robot_decisions", 0) or 0 for row in rows
            ),
            "teacher_forced_prefix_conditioned_top_1": _pooled_rate(
                rows,
                "teacher_forced_prefix_conditioned_top_1_hits",
                "teacher_forced_prefix_conditioned_decisions",
            ),
            "opening_top_1": _pooled_rate(
                rows, "opening_top_1_hits", "opening_decision_count",
            ),
            "opening_discriminating_top_1": _pooled_rate(
                rows,
                "opening_discriminating_top_1_hits",
                "opening_discriminating_decision_count",
            ),
            "opening_discriminating_decision_count": sum(
                row.get("opening_discriminating_decision_count", 0) or 0
                for row in rows
            ),
            "teacher_forced_prefix_conditioned_decisions": sum(
                row.get("teacher_forced_prefix_conditioned_decisions", 0) or 0 for row in rows
            ),
            "preference_discriminating_top_1": _pooled_rate(
                rows,
                "preference_discriminating_top_1_hits",
                "preference_discriminating_robot_decisions",
            ),
            "preference_discriminating_robot_decisions": sum(
                row.get("preference_discriminating_robot_decisions", 0) or 0 for row in rows
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
            "performance": _performance(rows),
            "corrections_per_robot_decision": (
                sum(row["human_corrections"] for row in rows)
                / max(1, sum(row["robot_decision_count"] for row in rows))
            ),
            "correction_free_rate": _finite_mean(row["correction_free"] for row in rows),
        })
    holdout_groups = _holdout_groups(assist)
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
            "prefix_conditioned_top_1": _pooled_rate(
                rows,
                "prefix_conditioned_top_1_hits",
                "prefix_conditioned_decision_count",
            ),
            "prefix_conditioned_decision_count": sum(
                row.get("prefix_conditioned_decision_count", 0) or 0 for row in rows
            ),
            "top_1": _pooled_rate(rows, "top_1_hits", "decision_count"),
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
        # Primary is the metric the symbolic evaluation also reports as its
        # primary prediction metric (src.evaluation: primary_prediction_metric
        # == "teacher_forced_top_1"), so the two environments are compared on
        # one definition.  The preference-discriminating family is
        # cooking-specific and has no symbolic counterpart, which is why it is
        # a secondary here rather than the headline.
        "primary_accuracy_metric": "teacher_forced_top_1",
        "cross_environment_parity_metric": "teacher_forced_top_1",
        "secondary_accuracy_metrics": [
            "prefix_conditioned_top_1",
            "teacher_forced_prefix_conditioned_top_1",
            "preference_discriminating_top_1",
            "teacher_forced_preference_discriminating_top_1",
        ],
        "metric_warning": (
            "robot_top_1 pools structurally forced single-legal-action "
            "decisions and is a diagnostic only. Of the accuracy metrics, "
            "prefix_conditioned_top_1 is the only one whose denominator is "
            "restricted to decisions still ambiguous given the episode "
            "prefix; preference_discriminating_top_1 counts decisions the "
            "prefix has already settled, because it tests the recipe's whole "
            "declared preference set rather than the surviving one. Report "
            "the zero-learning canonical_order arm alongside any of them."
        ),
        "robot_top_1": _pooled_rate(
            assist, "robot_top_1_hits", "robot_decision_count",
        ),
        "teacher_forced_top_1": _pooled_rate(
            assist, "teacher_forced_top_1_hits", "teacher_forced_decision_count",
        ),
        "teacher_forced_decision_count": sum(
            row.get("teacher_forced_decision_count", 0) or 0 for row in assist
        ),
        "opening_top_1": _pooled_rate(
            assist, "opening_top_1_hits", "opening_decision_count",
        ),
        "opening_discriminating_top_1": _pooled_rate(
            assist,
            "opening_discriminating_top_1_hits",
            "opening_discriminating_decision_count",
        ),
        "opening_discriminating_decision_count": sum(
            row.get("opening_discriminating_decision_count", 0) or 0
            for row in assist
        ),
        "prefix_conditioned_top_1": _pooled_rate(
            assist,
            "prefix_conditioned_top_1_hits",
            "prefix_conditioned_robot_decisions",
        ),
        "prefix_conditioned_robot_decisions": sum(
            row.get("prefix_conditioned_robot_decisions", 0) or 0 for row in assist
        ),
        "teacher_forced_prefix_conditioned_top_1": _pooled_rate(
            assist,
            "teacher_forced_prefix_conditioned_top_1_hits",
            "teacher_forced_prefix_conditioned_decisions",
        ),
        "teacher_forced_prefix_conditioned_decisions": sum(
            row.get("teacher_forced_prefix_conditioned_decisions", 0) or 0 for row in assist
        ),
        "preference_discriminating_top_1": _pooled_rate(
            assist,
            "preference_discriminating_top_1_hits",
            "preference_discriminating_robot_decisions",
        ),
        "preference_discriminating_robot_decisions": sum(
            row.get("preference_discriminating_robot_decisions", 0) or 0 for row in assist
        ),
        "single_legal_action_fraction": (
            sum(row["single_legal_action_decision_count"] for row in assist)
            / max(1, sum(row["robot_decision_count"] for row in assist))
        ),
        "performance": _performance(assist),
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
        "holdout_transfer_groups": holdout_groups,
        "holdout_transfer_paired_vs_full": _holdout_transfer_contrasts(
            holdout_groups
        ),
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


# What the CLI prints when a run finishes.  This lives beside ``_aggregate``
# deliberately: it used to live in ``__main__`` and drifted from the summary it
# reads, so a completed two-hour run wrote every artifact, passed validation,
# and then died on ``KeyError: 'primary_metric'`` in its own success report.
CLI_REPORT_KEYS: Tuple[str, ...] = (
    "status",
    "episode_count",
    "assist_episode_count",
    "failure_count",
    "delivery_rate",
    "primary_accuracy_metric",
    "cross_environment_parity_metric",
    "teacher_forced_top_1",
    "teacher_forced_decision_count",
    "prefix_conditioned_top_1",
    "prefix_conditioned_robot_decisions",
    "teacher_forced_prefix_conditioned_top_1",
    "opening_top_1",
    "opening_discriminating_top_1",
    "opening_discriminating_decision_count",
    "preference_discriminating_top_1",
    "preference_discriminating_robot_decisions",
    "robot_top_1",
    "single_legal_action_fraction",
    "semantic_fallback_discriminating_top_1",
    "own_model_discriminating_top_1",
    "metric_warning",
    "excluded_incomplete_cells",
    "assistance_unscored_cells",
    "lead_actor_policies",
    "performance",
    "groups",
    "pre_event_probe_groups",
    "holdout_transfer_paired_vs_full",
)


def summary_report(
    result: Mapping[str, Any], validation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the end-of-run report, without being able to fail.

    Every artifact is already on disk by the time this is called, so a missing
    key must not raise: it is listed under ``missing_summary_keys`` so the
    drift is visible in the output instead of discarding the run.
    """
    summary = result["summary"]
    report: Dict[str, Any] = {"run_dir": result["run_dir"]}
    missing = []
    for key in CLI_REPORT_KEYS:
        if key in summary:
            report[key] = summary[key]
        else:
            missing.append(key)
    if missing:
        report["missing_summary_keys"] = missing
    if validation is not None:
        report["validation"] = dict(validation)
    return report


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
        "source_fingerprint": _source_fingerprint(),
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
                "single-legal-action decisions are structurally forced; "
                "prefix_conditioned_top_1 additionally excludes decisions the "
                "episode prefix has already settled, and canonical_order "
                "reports what a fixed rule scores on the same episodes"
            ),
            "primary_accuracy_metric": "teacher_forced_top_1",
            "cross_environment_parity_metric": "teacher_forced_top_1",
            "ambiguity_conditioned_accuracy_metric": "prefix_conditioned_top_1",
            "zero_learning_reference_arm": "canonical_order",
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


def cell_execution_order(
    config: Mapping[str, Any],
) -> list[Tuple[int, str, str]]:
    """Every (seed, scenario, arm) cell, arm-major.

    One arm completes the whole seed/scenario grid before the next starts.
    Cells are independent here -- unlike the symbolic evaluation there is no
    shared route to publish -- so the order is purely about where results
    land and what has to be recomputed: an arm whose behaviour changed can
    have its folder deleted and be re-run without touching the others.
    """
    return [
        (int(seed), str(scenario), str(arm))
        for arm in map(str, config["arms"])
        for seed in map(int, config["seeds"])
        for scenario in map(str, config["scenarios"])
    ]


def _arm_dir(checkpoint_dir: Path, arm: str) -> Path:
    """One arm's own folder, holding every cell it produced.

    Arms are stored apart so a changed arm can be deleted and re-run on its
    own: a resumed run finds every other arm's cells intact and recomputes
    only what is missing.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(arm)):
        raise ValueError(f"arm name is not usable as a directory: {arm!r}")
    return checkpoint_dir / str(arm)


def _cell_checkpoint_path(
    checkpoint_dir: Path, seed: int, scenario: str, arm: str,
) -> Path:
    return _arm_dir(checkpoint_dir, arm) / f"{int(seed)}__{scenario}.json"


def _load_checkpoint(
    checkpoint_dir: Path,
    seed: int,
    scenario: str,
    arm: str,
    *,
    config_digest: str,
    source_fingerprint: str,
) -> Dict[str, Any] | None:
    """Return a reusable checkpoint, or None so the cell re-runs.

    Returning None is always safe -- the cell is simply recomputed -- so every
    check here fails closed.  Two of them were missing: a checkpoint carried no
    record of the code or schema that produced it, and an *incomplete* cell was
    absorbed as though it had finished, so a cell that died partway through
    could never be recovered by resuming and stayed permanently excluded from
    every pooled statistic.
    """
    path = _cell_checkpoint_path(checkpoint_dir, seed, scenario, arm)
    if not path.exists():
        return None
    try:
        cell = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(cell, dict):
        return None
    provenance = cell.get("provenance")
    if not isinstance(provenance, dict):
        return None
    if (
        provenance.get("result_schema_version") != RESULT_SCHEMA_VERSION
        or provenance.get("config_digest") != config_digest
        or provenance.get("source_fingerprint") != source_fingerprint
    ):
        return None
    if (
        cell.get("seed") != seed
        or cell.get("scenario") != scenario
        or cell.get("arm") != arm
    ):
        return None
    episodes = cell.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return None
    if not all(isinstance(row, dict) for row in episodes):
        return None
    if not all(row.get("cell_complete", False) for row in episodes):
        return None
    if cell.get("failures"):
        return None
    # A completion flag is a claim, not evidence.  A checkpoint holding one of
    # its 630 planned episodes, with the flag set, was accepted as a finished
    # cell; so verify the count the cell itself recorded as planned, and that
    # the event indices cover the schedule exactly once.
    planned = {row.get("cell_planned_episodes") for row in episodes}
    if len(planned) != 1:
        return None
    expected = planned.pop()
    if not isinstance(expected, int) or expected <= 0:
        return None
    if len(episodes) != expected:
        return None
    indices = [row.get("event_index") for row in episodes]
    if any(index is None for index in indices):
        return None
    if sorted(int(index) for index in indices) != list(range(expected)):
        return None
    for row in episodes:
        if row.get("seed") != seed or row.get("scenario") != scenario:
            return None
        if row.get("arm") != arm:
            return None
    return cell


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
    source_fingerprint = _source_fingerprint()
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
        manifest["source_fingerprint"] = source_fingerprint
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / f"{config['experiment']}__{timestamp}__{digest}"
        manifest = _manifest(config, run_dir)
        run_dir.mkdir(parents=False, exist_ok=False)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=False, exist_ok=False)
    manifest["cell_execution"] = "arm-major: one arm completes the whole seed/scenario grid before the next starts"
    manifest["arm_order"] = [str(arm) for arm in config["arms"]]
    manifest["artifacts"] = {
        "arm_cells": "checkpoints/<arm>/<seed>__<scenario>.json",
        "arm_summary": "checkpoints/<arm>/summary.json",
        "combined": "episodes.json, probes.json, audits.json, summary.json",
    }
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
        pending.extend(cell_execution_order(config))

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
                    _cell_checkpoint_path(
                        checkpoint_dir, cell["seed"], cell["scenario"], cell["arm"],
                    ),
                    {
                        **dict(cell),
                        "provenance": {
                            "result_schema_version": RESULT_SCHEMA_VERSION,
                            "config_digest": digest,
                            "source_fingerprint": source_fingerprint,
                        },
                    },
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
                _load_checkpoint(
                    checkpoint_dir, seed, scenario, arm,
                    config_digest=digest,
                    source_fingerprint=source_fingerprint,
                )
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
        # A per-arm roll-up beside that arm's cells. The heavy per-episode
        # rows are not copied: they already live in the arm's own cell files.
        for arm in map(str, config["arms"]):
            arm_episodes = [row for row in episodes if row.get("arm") == arm]
            if not arm_episodes:
                continue
            _atomic_json(_arm_dir(checkpoint_dir, arm) / "summary.json", {
                "arm": arm,
                "episode_count": len(arm_episodes),
                "failure_count": sum(
                    1 for row in failures if row.get("arm") == arm
                ),
                "cells": sorted(
                    {
                        f"{row['scenario']}__{row['seed']}"
                        for row in arm_episodes
                        if "scenario" in row and "seed" in row
                    }
                ),
                "summary": _aggregate(
                    arm_episodes,
                    [row for row in failures if row.get("arm") == arm],
                    [row for row in probes if row.get("arm") == arm],
                    [row for row in audits if row.get("arm") == arm],
                ),
            })
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
    if "zero_learning_reference" in requirements:
        missing = set(NULL_ARM_NAMES) - {row["arm"] for row in episodes}
        if missing:
            failures.append(
                "zero-learning reference arms did not complete: "
                f"{sorted(missing)}"
            )
        null_rows = [row for row in episodes if row["arm"] in NULL_ARM_NAMES]
        # A null arm that fitted a model is not a null arm.  Its retrain
        # bookkeeping still runs (so routing and audits stay comparable), but
        # no weights may ever be learned.
        fitted = [
            row for row in null_rows
            if row["learner_model_structure"] != "unfitted"
        ]
        if fitted:
            failures.append(
                "a zero-learning reference arm fitted a predictor "
                f"({fitted[0]['learner_model_structure']})"
            )
        if any(row["semantic_fallback_decisions"] for row in null_rows) or any(
            row["latent_strategy_decisions"] for row in null_rows
        ):
            failures.append(
                "a zero-learning reference arm consulted a learned component"
            )
    if "multiple_seeds" in requirements and len({row["seed"] for row in episodes}) < 2:
        failures.append("fewer than two seeds completed")
    if "comparison_arms" in requirements and {
        row["arm"] for row in episodes
    } != set(config["arms"]):
        failures.append("not every configured arm completed")
    if "adaptive_hrc_baseline_parity" in requirements:
        # Parity is about the deployable roster.  The zero-learning references
        # are cooking-only diagnostics with no symbolic counterpart, so they
        # are held apart rather than counted as a roster difference.
        deployable = tuple(
            arm for arm in config["arms"] if arm not in NULL_ARM_NAMES
        )
        if deployable != ARM_NAMES:
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
    "ARM_NAMES", "CLI_REPORT_KEYS", "FULL_ARM", "NULL_ARM_NAMES",
    "cell_execution_order",
    "VALIDATION_REQUIREMENTS", "load_config", "run_experiment",
    "summary_report", "validate_result",
]
