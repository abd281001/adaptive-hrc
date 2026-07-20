"""Evaluation harness for the adaptive preference-learning HRC benchmark.

This module keeps one runner, three scenario generators, compact episode-level logging, frozen
evaluation, and the diagnostics needed for the paper claims.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_NATIVE_THREADS_PER_WORKER = 1
PAPER_SEEDS = (1337, 2024, 7, 9001, 31415)
PAIRED_BOOTSTRAP_SAMPLES = 10_000
# One-factor-at-a-time checks around the online self-training rule.  The
# identity/posterior pair preserves their combined 0.66 contribution, so that
# comparison changes evidence allocation rather than total score scale.
COMMIT_SENSITIVITY_SPECS: Tuple[Tuple[str, Mapping[str, float]], ...] = (
    ("default", {}),
    ("tentative_threshold_low", {"online_commit_tentative_threshold": 0.30}),
    ("tentative_threshold_high", {"online_commit_tentative_threshold": 0.60}),
    ("full_threshold_low", {"online_commit_full_threshold": 0.60}),
    ("full_threshold_high", {"online_commit_full_threshold": 0.90}),
    ("identity_evidence_low", {
        "online_commit_identity_weight": 0.30,
        "online_commit_recipe_posterior_weight": 0.36,
    }),
    ("identity_evidence_high", {
        "online_commit_identity_weight": 0.60,
        "online_commit_recipe_posterior_weight": 0.06,
    }),
)
NATIVE_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _set_native_thread_env(threads: int, *, override: bool) -> None:
    value = str(_positive_int(threads))
    for var in NATIVE_THREAD_ENV_VARS:
        if override or not os.environ.get(var):
            os.environ[var] = value


def _loaded_openblas_paths() -> Tuple[str, ...]:
    maps_path = Path("/proc/self/maps")
    try:
        lines = maps_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ()

    seen = set()
    paths: List[str] = []
    for line in lines:
        path = line.rsplit(maxsplit=1)[-1]
        if "/" not in path or "openblas" not in path.lower() or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return tuple(paths)


def _set_loaded_openblas_threads(threads: int) -> bool:
    symbols = (
        "scipy_openblas_set_num_threads64_",
        "scipy_openblas_set_num_threads_64_",
        "scipy_openblas_set_num_threads",
        "scipy_openblas_set_num_threads_",
        "openblas_set_num_threads",
    )
    applied = False
    for path in _loaded_openblas_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for symbol in symbols:
            try:
                setter = getattr(lib, symbol)
            except AttributeError:
                continue
            setter.argtypes = [ctypes.c_int]
            setter.restype = None
            setter(_positive_int(threads))
            applied = True
            break
    return applied


def _apply_native_thread_limit(threads: int) -> Dict[str, Any]:
    threads = _positive_int(threads, DEFAULT_NATIVE_THREADS_PER_WORKER)
    _set_native_thread_env(threads, override=True)
    return {
        "native_threads_per_worker": threads,
        "env": {var: os.environ.get(var) for var in NATIVE_THREAD_ENV_VARS},
        "openblas_runtime_limited": _set_loaded_openblas_threads(threads),
    }


_set_native_thread_env(DEFAULT_NATIVE_THREADS_PER_WORKER, override=False)

from .adaptive_agent import AdaptiveHRCAgent
from .baselines import BASELINE_AGENTS, OracleCeilingAgent
from .environment import gen
from .hrc_simulation import DEFAULT_HRC_TIMING, run_alternating_hrc_episode
from .memory import VariantKey, variant_hash
from .models import DEFAULT_CONFIG, Config
from .preferences import PRESET_PREFERENCES, materialize_with_report
from .representations import observations_from_actions


SCENARIO_LADDER_HETEROGENEOUS = "ladder_heterogeneous"
SCENARIO_LADDER_HOMOGENEOUS = "ladder_homogeneous"
SCENARIO_DEPLOYMENT_RANDOM = "ladder_deployment_random"
SCENARIOS = (
    SCENARIO_LADDER_HETEROGENEOUS,
    SCENARIO_LADDER_HOMOGENEOUS,
    SCENARIO_DEPLOYMENT_RANDOM,
)

DEFAULT_BASELINES = (
    "full",
    "latest_only",
    "fixed_decay",
    "no_decay",
    "bc",
    "ewc",
    "experience_replay_bc",
    "bigram",
)

CLAIRVOYANT_MEMORY_ORACLE = "clairvoyant_memory_oracle"
CLAIRVOYANT_REFERENCE_TAG = "dashed_reference_not_deployable"
CLAIRVOYANT_LEAKAGE_WARNING = (
    "Non-deployable oracle: prunes memory using future recipe support from the "
    "same evaluation stream. Use only as a reference curve."
)

DEFAULT_LADDER_PREFERENCES = (
    "identity",
    "p2_frontload",
    "p3_clean_eager",
    "p5_prep_stage_clean",
    "p6_full_restructure",
    "p9_deferred_cook_start",
    "p10_cleanup_before_serve",
    "p12_defer_cook_cleanup_before_serve",
    "p1_prep_first",
    "p4_prep_clean",
    "p7_late_seasoning",
    "p8_batch_container_loading",
    "p11_late_season_batch_load",
)

OBSERVATION_MODE_EXTRA_TIME_PER_STEP = 1.0
ADAPTATION_RECOVERY_WINDOWS = (1, 2, 3)
PRIMARY_ADAPTATION_RECOVERY_WINDOW = 1


@dataclass(frozen=True)
class RecipePreferencePair:
    recipe_name: str
    preference_name: str
    actions: Tuple[str, ...]
    axis_values: Mapping[str, str] = field(default_factory=dict)
    ordering_is_distinct: bool = True
    duplicate_of_preference: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.recipe_name}/{self.preference_name}"

    @property
    def non_default_axes(self) -> Tuple[str, ...]:
        defaults = PRESET_PREFERENCES["identity"].as_dict()
        return tuple(
            axis for axis, value in self.axis_values.items()
            if value != defaults.get(axis)
        )

    @property
    def is_composed_preference(self) -> bool:
        return len(self.non_default_axes) >= 2


@dataclass(frozen=True)
class ScenarioEvent:
    mode: str
    pair: RecipePreferencePair
    tags: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioPlan:
    scenario: str
    seed: int
    events: Tuple[ScenarioEvent, ...]
    eval_pairs: Tuple[RecipePreferencePair, ...]
    selected_recipes: Tuple[str, ...]
    selected_preferences: Tuple[str, ...]
    description: str


@dataclass(frozen=True)
class EvaluationConfig:
    # Five fixed, paired scenario draws are the minimum paper protocol.  The
    # runner still accepts any explicit seed tuple for smoke tests or larger
    # final studies.
    seeds: Tuple[int, ...] = PAPER_SEEDS
    scenarios: Tuple[str, ...] = SCENARIOS
    baselines: Tuple[str, ...] = DEFAULT_BASELINES
    output_dir: str = "results/evaluation"
    workers: int = 0
    native_threads_per_worker: int = DEFAULT_NATIVE_THREADS_PER_WORKER
    print_eta: bool = True
    include_clairvoyant_oracle: bool = True
    n_recipes: int = 6
    ladder_rungs: int = 5
    ladder_preferences: Tuple[str, ...] = DEFAULT_LADDER_PREFERENCES
    allow_repeated_ladder_orderings: bool = False
    min_distinct_ladder_orderings: int = 2
    settle_repeats_per_update: int = 1
    deployment_events: int = 80
    deployment_onboarding_recipes: int = 3
    # The probabilities define a fixed per-seed quota allocation; event order
    # is randomized only within feasibility constraints so each requested
    # condition has adequate support.
    deployment_quota_controlled: bool = True
    deployment_preference_shift_prob: float = 0.28
    deployment_transfer_probe_prob: float = 0.22
    deployment_reentry_prob: float = 0.14
    deployment_new_recipe_prob: float = 0.10
    # Retained only for backwards-compatible configuration parsing. Known
    # recipes are never silently routed to observation: all preference changes
    # use assist mode under the stated interaction protocol.
    deployment_random_observation_prob: float = 0.0
    deployment_reentry_oldest_fraction: float = 0.35
    deployment_reentry_prefer_displaced: bool = True
    deployment_transfer_fresh_gap_events: int = 2
    deployment_transfer_aged_gap_events: int = 8
    route_absent_recipe_assists_to_observe: bool = True
    # A matched, non-mutating probe immediately before each explicitly tagged
    # primary assist event.  Random deployment extends this to every routed
    # assist event.  This is the comparable prediction metric; live
    # interaction cost still uses the actual routed event.
    pre_event_frozen_probes: bool = True
    # Applies to the heterogeneous ladder only.  The homogeneous ladder uses
    # explicit rung boundaries; randomized deployment uses event-local probes
    # rather than repeated full-grid sweeps.
    frozen_eval_period: int = 4
    frozen_eval_max_pairs: int = 48
    active_only_audit_period: int = 2
    active_only_audit_max_prefixes: int = 16
    active_only_audit_tolerance: float = 5e-2
    topk: int = 3
    profile: bool = False
    run_commit_sensitivity: bool = False
    model_overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EventStreamRun:
    baseline: str
    scenario: str
    seed: int
    agent: AdaptiveHRCAgent
    name_to_rid: Dict[str, str]
    episode_rows: List[Dict[str, Any]]
    frozen_rows: List[Dict[str, Any]]
    memory_rows: List[Dict[str, Any]]
    prototype_rows: List[Dict[str, Any]]
    active_audit_rows: List[Dict[str, Any]]
    oracle_pruning_rows: List[Dict[str, Any]]
    turn_rows: List[Dict[str, Any]]
    wall_s: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _mean(values: Iterable[Any]) -> float:
    vals = _finite(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _numeric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else float(default)


def _p95(values: Iterable[Any]) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)
    return vals[idx]


def base_config(seed: int, eval_config: EvaluationConfig, **overrides: Any) -> Config:
    cfg = replace(
        DEFAULT_CONFIG,
        seed=int(seed),
        verbose=False,
        profile=bool(eval_config.profile),
    )
    merged = {**dict(eval_config.model_overrides), **overrides}
    return replace(cfg, **merged) if merged else cfg


def make_agent(name: str, cfg: Config) -> AdaptiveHRCAgent:
    registry: Dict[str, Callable[..., AdaptiveHRCAgent]] = {
        "full": AdaptiveHRCAgent,
        "oracle": OracleCeilingAgent,
        **BASELINE_AGENTS,
    }
    if name not in registry:
        raise KeyError(f"unknown baseline {name!r}; available={sorted(registry)}")
    # Session-local correction adaptation is a proposed Full component.  It
    # must not silently improve a comparator simply because the shared HRC
    # runner delivers feedback to every agent.
    if name != "full":
        cfg = replace(cfg, session_correction_adaptation=False)
    return registry[name](cfg=cfg)


def shuffled_recipe_builders(seed: int) -> List[Tuple[str, Callable[[], List[str]]]]:
    items = list(gen.recipe_library().items())
    rng = random.Random(int(seed))
    rng.shuffle(items)
    return items


def select_recipe_builders(seed: int, n_recipes: int) -> List[Tuple[str, Callable[[], List[str]]]]:
    items = shuffled_recipe_builders(seed)
    return items[: max(1, min(int(n_recipes), len(items)))]


def materialize_pair(recipe_name: str, preference_name: str, builder: Callable[[], List[str]]) -> RecipePreferencePair:
    if preference_name not in PRESET_PREFERENCES:
        raise KeyError(f"unknown preference {preference_name!r}")
    base = tuple(builder())
    report = materialize_with_report(base, preference_name)
    return RecipePreferencePair(
        recipe_name=recipe_name,
        preference_name=preference_name,
        actions=tuple(report.actions),
        axis_values=dict(report.axis_values),
    )


def distinct_pairs_for_recipe(
    recipe_name: str,
    builder: Callable[[], List[str]],
    preferences: Sequence[str],
    min_pairs: int,
    *,
    allow_repeated_orderings: bool,
) -> List[RecipePreferencePair]:
    pairs: List[RecipePreferencePair] = []
    seen_orderings: Dict[Tuple[str, ...], str] = {}
    for pref in preferences:
        try:
            pair = materialize_pair(recipe_name, pref, builder)
        except Exception:
            continue
        duplicate_of = seen_orderings.get(pair.actions)
        if duplicate_of is not None:
            if not allow_repeated_orderings:
                continue
            pair = replace(
                pair,
                ordering_is_distinct=False,
                duplicate_of_preference=duplicate_of,
            )
        else:
            seen_orderings[pair.actions] = pref
        pairs.append(pair)
        if len(pairs) >= min_pairs:
            break
    return pairs


def _preferences(config: EvaluationConfig) -> List[str]:
    prefs = [p for p in config.ladder_preferences if p in PRESET_PREFERENCES]
    return prefs if "identity" in prefs else ["identity", *prefs]


def _pair_matrix(config: EvaluationConfig, seed: int) -> Tuple[List[str], List[str], Dict[str, List[RecipePreferencePair]]]:
    min_pairs = max(2, int(config.ladder_rungs))
    strict = not bool(config.allow_repeated_ladder_orderings)
    min_distinct = min_pairs if strict else max(1, int(config.min_distinct_ladder_orderings))
    target_recipes = max(1, int(config.n_recipes))
    matrix: Dict[str, List[RecipePreferencePair]] = {}
    recipes: List[str] = []
    for recipe_name, builder in shuffled_recipe_builders(seed):
        pairs = distinct_pairs_for_recipe(
            recipe_name,
            builder,
            _preferences(config),
            min_pairs,
            allow_repeated_orderings=not strict,
        )
        distinct_count = sum(1 for pair in pairs if pair.ordering_is_distinct)
        if len(pairs) >= min_pairs and distinct_count >= min_distinct:
            matrix[recipe_name] = pairs[:min_pairs]
            recipes.append(recipe_name)
            if len(recipes) >= target_recipes:
                break
    if len(recipes) < target_recipes:
        raise RuntimeError(
            f"only {len(recipes)} recipes support {min_pairs} operative preference rungs"
        )
    prefs = sorted({p.preference_name for pairs in matrix.values() for p in pairs})
    return recipes, prefs, matrix


def _shared_preference_matrix(config: EvaluationConfig, seed: int) -> Tuple[List[str], List[str], Dict[str, List[RecipePreferencePair]]]:
    builders = shuffled_recipe_builders(seed)
    prefs = _preferences(config)
    n_rungs = max(2, int(config.ladder_rungs))
    target_recipes = max(1, int(config.n_recipes))
    strict = not bool(config.allow_repeated_ladder_orderings)
    min_distinct = n_rungs if strict else max(1, int(config.min_distinct_ladder_orderings))
    pair_cache: Dict[str, Dict[str, RecipePreferencePair]] = {}
    for recipe, builder in builders:
        by_pref: Dict[str, RecipePreferencePair] = {}
        for pref in prefs:
            try:
                by_pref[pref] = materialize_pair(recipe, pref, builder)
            except Exception:
                pass
        pair_cache[recipe] = by_pref

    combos = combinations(prefs, n_rungs)
    if "identity" in prefs:
        combos = (combo for combo in combos if "identity" in combo)

    best_preferences: Tuple[str, ...] = ()
    best_matrix: Dict[str, List[RecipePreferencePair]] = {}
    for combo in combos:
        candidate: Dict[str, List[RecipePreferencePair]] = {}
        for recipe, builder in builders:
            pairs: List[RecipePreferencePair] = []
            seen: Dict[Tuple[str, ...], str] = {}
            for pref in combo:
                pair = pair_cache.get(recipe, {}).get(pref)
                if pair is None:
                    pairs = []
                    break
                duplicate_of = seen.get(pair.actions)
                if duplicate_of is not None:
                    if strict:
                        pairs = []
                        break
                    pair = replace(pair, ordering_is_distinct=False, duplicate_of_preference=duplicate_of)
                else:
                    seen[pair.actions] = pref
                pairs.append(pair)
            if not pairs:
                continue
            if sum(1 for pair in pairs if pair.ordering_is_distinct) >= min_distinct:
                candidate[recipe] = pairs
        if len(candidate) > len(best_matrix):
            best_preferences = tuple(combo)
            best_matrix = candidate

    selected_recipes = [recipe for recipe, _ in builders if recipe in best_matrix][:target_recipes]
    if len(selected_recipes) < target_recipes:
        raise RuntimeError(
            f"only {len(selected_recipes)} homogeneous recipes support {n_rungs} operative shared rungs"
        )
    return selected_recipes, list(best_preferences), {recipe: best_matrix[recipe] for recipe in selected_recipes}


def _largest_common_latin_matrix(config: EvaluationConfig, seed: int) -> Tuple[List[str], List[str], Dict[str, List[RecipePreferencePair]]]:
    """Find the largest square recipe/preference set valid for Latin transfer.

    A heterogeneous cross-transfer rung is meaningful only when every selected
    preference is materially valid for every selected recipe.  Rather than
    silently substituting arbitrary per-recipe preferences or failing at an
    aspirational size, reduce to the largest common square subset.
    """
    max_size = min(max(2, int(config.n_recipes)), max(2, int(config.ladder_rungs)), len(_preferences(config)))
    for size in range(max_size, 1, -1):
        candidate = replace(config, n_recipes=size, ladder_rungs=size)
        try:
            recipes, preferences, matrix = _shared_preference_matrix(candidate, seed)
        except RuntimeError:
            continue
        if len(recipes) == size and len(preferences) == size:
            return recipes, preferences, matrix
    raise RuntimeError("no common recipe/preference subset of size at least two supports a valid Latin transfer ladder")


def _scenario_tag(config: EvaluationConfig, scenario: str, **extra: Any) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "single_user_id": "user_001",
        "one_user_updating_preferences": True,
        "topk": int(config.topk),
        **extra,
    }


def _preference_tags(pair: RecipePreferencePair) -> Dict[str, Any]:
    return {
        "preference_non_default_axes": list(pair.non_default_axes),
        "hypothesis_tags": [
            "known_recipe_new_preference_adaptation",
            "emergent_axis_composition" if pair.is_composed_preference else "single_axis_preference",
        ],
    }


def build_ladder_heterogeneous(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    recipes, preferences, matrix = _largest_common_latin_matrix(config, seed)
    events: List[ScenarioEvent] = []
    n_rungs = len(preferences)
    for rung in range(n_rungs):
        for recipe_idx, recipe in enumerate(recipes):
            pair = matrix[recipe][(recipe_idx + rung) % len(matrix[recipe])]
            if rung == 0:
                events.append(ScenarioEvent("observe", pair, _scenario_tag(
                    config,
                    SCENARIO_LADDER_HETEROGENEOUS,
                    event_type="onboarding_observation",
                    condition="onboarding_observation",
                    rung_idx=rung,
                    recipe_position=recipe_idx,
                    ladder_structure="heterogeneous_latin_square",
                    hypothesis_tags=["initial_learning"],
                )))
                continue
            source_idx = (recipe_idx + rung) % n_rungs
            source_recipe = recipes[source_idx]
            source_pair = matrix[source_recipe][source_idx]
            base_tags = _scenario_tag(
                config,
                SCENARIO_LADDER_HETEROGENEOUS,
                event_type="cross_recipe_transfer_first_exposure",
                condition="heterogeneous_cross_recipe_transfer_first_exposure",
                condition_family="cross_recipe_transfer",
                evaluation_phase="first_exposure",
                primary_probe=True,
                rung_idx=rung,
                recipe_position=recipe_idx,
                source_recipe=source_recipe,
                source_pair=source_pair.label,
                source_preference=pair.preference_name,
                source_initial_event_idx=source_idx,
                scheduled_source_age_events=len(events) - source_idx,
                target_pair_seen_before=False,
                ladder_structure="heterogeneous_latin_square",
                preference_non_default_axes=list(pair.non_default_axes),
                hypothesis_tags=[
                    "cross_recipe_transfer",
                    "emergent_axis_composition" if pair.is_composed_preference else "single_axis_preference",
                ],
            )
            events.append(ScenarioEvent("assist", pair, base_tags))
            for repeat in range(max(0, int(config.settle_repeats_per_update))):
                events.append(ScenarioEvent("assist", pair, {
                    **base_tags,
                    "event_type": "settled_reuse_probe",
                    "condition": "heterogeneous_settled_reuse",
                    "condition_family": "post_update_settling",
                    "evaluation_phase": "post_commit_settling",
                    "primary_probe": False,
                    "settle_repeat_idx": repeat,
                    "hypothesis_tags": ["post_update_settling", "retention_after_adaptation"],
                }))
    return ScenarioPlan(
        scenario=SCENARIO_LADDER_HETEROGENEOUS,
        seed=seed,
        events=tuple(events),
        eval_pairs=tuple(pair for pairs in matrix.values() for pair in pairs[:n_rungs]),
        selected_recipes=tuple(recipes),
        selected_preferences=tuple(preferences),
        description="Off-diagonal Latin ladder over the largest common valid recipe/preference subset. Each primary post-onboarding event is a labelled, first-exposure cross-recipe transfer probe with a scheduled source age; post-commit repeats are reported separately as settling.",
    )


def build_ladder_homogeneous(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    recipes, preferences, matrix = _shared_preference_matrix(config, seed + 101)
    events: List[ScenarioEvent] = []
    n_rungs = max(2, int(config.ladder_rungs))
    for rung in range(n_rungs):
        for recipe_idx, recipe in enumerate(recipes):
            pair = matrix[recipe][rung]
            if rung == 0:
                events.append(ScenarioEvent("observe", pair, _scenario_tag(
                    config,
                    SCENARIO_LADDER_HOMOGENEOUS,
                    event_type="onboarding_observation",
                    condition="onboarding_observation",
                    rung_idx=rung,
                    recipe_position=recipe_idx,
                    ladder_structure="homogeneous_shared_preference",
                    hypothesis_tags=["initial_learning"],
                )))
                continue
            base_tags = _scenario_tag(
                config,
                SCENARIO_LADDER_HOMOGENEOUS,
                event_type="shared_preference_transfer_probe",
                condition="homogeneous_shared_preference_update",
                condition_family="homogeneous_shared_preference_update",
                evaluation_phase="first_exposure",
                primary_probe=True,
                rung_idx=rung,
                recipe_position=recipe_idx,
                ladder_structure="homogeneous_shared_preference",
                **_preference_tags(pair),
            )
            events.append(ScenarioEvent("assist", pair, base_tags))
            for repeat in range(max(0, int(config.settle_repeats_per_update))):
                events.append(ScenarioEvent("assist", pair, {
                    **base_tags,
                    "event_type": "settled_reuse_probe",
                    "condition": "homogeneous_settled_reuse",
                    "condition_family": "post_update_settling",
                    "evaluation_phase": "post_commit_settling",
                    "primary_probe": False,
                    "settle_repeat_idx": repeat,
                    "hypothesis_tags": ["post_update_settling", "retention_after_adaptation"],
                }))
    return ScenarioPlan(
        scenario=SCENARIO_LADDER_HOMOGENEOUS,
        seed=seed,
        events=tuple(events),
        eval_pairs=tuple(pair for pairs in matrix.values() for pair in pairs[:n_rungs]),
        selected_recipes=tuple(recipes),
        selected_preferences=tuple(preferences),
        description="Structured ladder where every recipe shares the same preference at each rung.",
    )


def build_deployment_random(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    """Build one paired, quota-controlled stochastic deployment stream.

    The generator never observes an already known recipe.  Requested event
    frequencies are converted to quotas before randomisation, so a seed cannot
    accidentally contain no transfer or re-entry evidence.  The plan itself
    cannot inspect mutable baseline memory; actual active/pruned state remains
    evaluator-side metadata recorded immediately before each event.
    """
    rng = random.Random(int(seed) + 2027)
    recipes, preferences, matrix = _pair_matrix(config, seed + 202)
    events: List[ScenarioEvent] = []
    observed_recipes: set[str] = set()
    current_pref_idx = {recipe: 0 for recipe in recipes}
    seen_by_label: Dict[str, RecipePreferencePair] = {}
    last_event_by_label: Dict[str, int] = {}

    def append_event(mode: str, pair: RecipePreferencePair, *, stream_idx: int, **tags: Any) -> None:
        event_idx = len(events)
        events.append(ScenarioEvent(mode, pair, _scenario_tag(
            config,
            SCENARIO_DEPLOYMENT_RANDOM,
            deployment_event_idx=event_idx,
            stream_idx=stream_idx,
            deployment_randomized=True,
            deployment_schedule="quota_controlled_randomized",
            **tags,
        )))
        seen_by_label[pair.label] = pair
        last_event_by_label[pair.label] = event_idx

    onboarding = min(max(1, int(config.deployment_onboarding_recipes)), len(recipes))
    for idx, recipe in enumerate(recipes[:onboarding]):
        pair = matrix[recipe][0]
        observed_recipes.add(recipe)
        current_pref_idx[recipe] = 0
        append_event(
            "observe",
            pair,
            stream_idx=-1,
            event_type="deployment_onboarding_observation",
            condition="deployment_onboarding",
            condition_family="initial_learning",
            recipe_position=idx,
            rung_idx=0,
            primary_probe=False,
            hypothesis_tags=["initial_learning"],
        )

    n_stream_events = max(0, int(config.deployment_events))

    def quota_allocation() -> Dict[str, int]:
        requested = {
            "new_recipe": max(0.0, float(config.deployment_new_recipe_prob)),
            "preference_shift": max(0.0, float(config.deployment_preference_shift_prob)),
            "cross_transfer": max(0.0, float(config.deployment_transfer_probe_prob)),
            "reentry": max(0.0, float(config.deployment_reentry_prob)),
        }
        total = sum(requested.values())
        scale = 1.0 / total if total > 1.0 else 1.0
        raw = {name: n_stream_events * probability * scale for name, probability in requested.items()}
        counts = {name: int(math.floor(value)) for name, value in raw.items()}
        assigned = sum(counts.values())
        for name in sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True):
            if assigned >= n_stream_events:
                break
            counts[name] += 1
            assigned += 1
        # A requested transfer requires at least one source preference shift.
        if counts["cross_transfer"] > 0 and counts["preference_shift"] == 0:
            counts["preference_shift"] = 1
        counts["new_recipe"] = min(counts["new_recipe"], max(0, len(recipes) - onboarding))
        assigned = sum(counts.values())
        # The final residual is a direct-retrieval control.  If setup rounding
        # overcommits a very short stream, reduce new-recipe quota first.
        while assigned > n_stream_events and counts["new_recipe"] > 0:
            counts["new_recipe"] -= 1
            assigned -= 1
        while assigned > n_stream_events and counts["preference_shift"] > 1:
            counts["preference_shift"] -= 1
            assigned -= 1
        counts["routine"] = max(0, n_stream_events - assigned)
        return counts

    remaining = quota_allocation()

    def known_recipe(*, exclude: Optional[str] = None) -> Optional[str]:
        candidates = sorted(observed_recipes)
        if exclude is not None:
            candidates = [recipe for recipe in candidates if recipe != exclude]
        return rng.choice(candidates) if candidates else None

    def current_pair(recipe: str) -> RecipePreferencePair:
        return matrix[recipe][current_pref_idx[recipe]]

    def shift_candidates() -> List[Tuple[str, int, RecipePreferencePair]]:
        candidates: List[Tuple[str, int, RecipePreferencePair]] = []
        for recipe in sorted(observed_recipes):
            for pref_idx, pair in enumerate(matrix[recipe]):
                if pref_idx != current_pref_idx[recipe] and pair.label not in seen_by_label:
                    candidates.append((recipe, pref_idx, pair))
        return candidates

    def transfer_candidates(required_gap: int) -> List[Tuple[RecipePreferencePair, RecipePreferencePair, int, int]]:
        candidates: List[Tuple[RecipePreferencePair, RecipePreferencePair, int, int]] = []
        for source_label, source_pair in seen_by_label.items():
            source_event_idx = last_event_by_label[source_label]
            gap = len(events) - source_event_idx
            if gap < required_gap:
                continue
            for target_recipe in sorted(observed_recipes):
                if target_recipe == source_pair.recipe_name:
                    continue
                for target_idx, candidate in enumerate(matrix[target_recipe]):
                    if candidate.preference_name == source_pair.preference_name and candidate.label not in seen_by_label:
                        candidates.append((source_pair, candidate, target_idx, gap))
        return candidates

    def reentry_candidates() -> Tuple[List[str], List[str]]:
        displaced = [
            label for label, pair in seen_by_label.items()
            if pair.preference_name != current_pair(pair.recipe_name).preference_name
        ]
        eligible = displaced if (config.deployment_reentry_prefer_displaced and displaced) else list(seen_by_label)
        return eligible, displaced

    transfer_ordinal = 0
    for stream_idx in range(n_stream_events):
        fresh_gap = max(1, int(config.deployment_transfer_fresh_gap_events))
        aged_gap = max(fresh_gap, int(config.deployment_transfer_aged_gap_events))
        any_transfer_pool = transfer_candidates(1)
        fresh_transfer_pool = [candidate for candidate in any_transfer_pool if candidate[3] <= fresh_gap]
        aged_transfer_pool = [candidate for candidate in any_transfer_pool if candidate[3] >= aged_gap]
        desired_transfer_age = "fresh" if transfer_ordinal % 2 == 0 else "aged"
        transfer_pool = fresh_transfer_pool if desired_transfer_age == "fresh" else aged_transfer_pool

        selectable = [name for name, count in remaining.items() if count > 0]
        if not selectable:
            selectable = ["routine"]
        # A queued transfer without a source forces an available source shift
        # before random selection.  This is the minimal causal setup needed to
        # make a transfer probe valid rather than relabelling routine reuse.
        if (
            remaining.get("cross_transfer", 0) > 0
            and (not any_transfer_pool or (desired_transfer_age == "fresh" and not fresh_transfer_pool))
            and remaining.get("preference_shift", 0) > 0
        ):
            requested_type = "preference_shift"
        else:
            # Do not consume an aged-transfer quota before its designated
            # source has aged; schedule another requested event if one exists.
            deferred_cross = (
                remaining.get("cross_transfer", 0) > 0
                and not transfer_pool
                and any(name != "cross_transfer" for name in selectable)
            )
            candidate_types = [name for name in selectable if not (deferred_cross and name == "cross_transfer")]
            weights = [max(1, int(remaining.get(name, 0))) for name in candidate_types]
            requested_type = rng.choices(candidate_types, weights=weights, k=1)[0]
        if remaining.get(requested_type, 0) > 0:
            remaining[requested_type] -= 1

        event_idx = len(events)
        common = {
            "quota_requested_type": requested_type,
            "quota_remaining_after_selection": dict(remaining),
        }

        if requested_type == "new_recipe":
            candidates = sorted(set(recipes) - observed_recipes)
            if candidates:
                recipe = rng.choice(candidates)
                pair = matrix[recipe][0]
                observed_recipes.add(recipe)
                current_pref_idx[recipe] = 0
                append_event(
                    "observe",
                    pair,
                    stream_idx=stream_idx,
                    event_type="deployment_new_recipe_observation",
                    condition="deployment_new_recipe",
                    condition_family="new_recipe_learning",
                    rung_idx=0,
                    primary_probe=False,
                    hypothesis_tags=["new_recipe_learning", "unseen_unseen_control"],
                    **common,
                )
                continue

        if requested_type == "preference_shift":
            candidates = shift_candidates()
            if candidates:
                globally_new = [candidate for candidate in candidates if candidate[2].preference_name not in {pair.preference_name for pair in seen_by_label.values()}]
                recipe, pref_idx, pair = rng.choice(globally_new or candidates)
                preference_seen_elsewhere = any(
                    seen.preference_name == pair.preference_name and seen.recipe_name != recipe
                    for seen in seen_by_label.values()
                )
                current_pref_idx[recipe] = pref_idx
                hypothesis = ["known_recipe_new_preference_adaptation"] if not preference_seen_elsewhere else ["known_recipe_new_pair"]
                hypothesis.append("emergent_axis_composition" if pair.is_composed_preference else "single_axis_preference")
                append_event(
                    "assist",
                    pair,
                    stream_idx=stream_idx,
                    event_type="deployment_preference_shift",
                    condition="deployment_preference_shift",
                    condition_family=("within_recipe_new_preference" if not preference_seen_elsewhere else "known_recipe_new_pair_with_prior_transfer_support"),
                    evaluation_phase="first_exposure",
                    primary_probe=True,
                    rung_idx=pref_idx,
                    target_preference_seen_other_recipe_before=preference_seen_elsewhere,
                    hypothesis_tags=hypothesis,
                    preference_non_default_axes=list(pair.non_default_axes),
                    **common,
                )
                continue

        if requested_type == "cross_transfer":
            pool = transfer_pool or any_transfer_pool
            if pool:
                # Fresh/aged source exposure is counterbalanced by transfer
                # ordinal; if the desired aged support is unavailable we retain
                # the probe but mark the achieved source age explicitly.
                source_pair, pair, target_idx, source_age = rng.choice(pool)
                current_pref_idx[pair.recipe_name] = target_idx
                achieved_bucket = "aged" if source_age >= aged_gap else ("fresh" if source_age <= fresh_gap else "intermediate")
                target_met = source_age <= fresh_gap if desired_transfer_age == "fresh" else source_age >= aged_gap
                append_event(
                    "assist",
                    pair,
                    stream_idx=stream_idx,
                    event_type="deployment_cross_recipe_transfer_probe",
                    condition="deployment_cross_recipe_transfer",
                    condition_family="cross_recipe_transfer",
                    evaluation_phase="first_exposure",
                    primary_probe=True,
                    rung_idx=target_idx,
                    source_recipe=source_pair.recipe_name,
                    source_pair=source_pair.label,
                    source_preference=source_pair.preference_name,
                    target_pair_seen_before=False,
                    scheduled_source_age_events=source_age,
                    scheduled_source_age_target=desired_transfer_age,
                    scheduled_source_age_target_met=target_met,
                    source_age_bucket=achieved_bucket,
                    hypothesis_tags=["cross_recipe_transfer", "preference_reuse_under_new_recipe_context"],
                    preference_non_default_axes=list(pair.non_default_axes),
                    **common,
                )
                transfer_ordinal += 1
                continue

        if requested_type == "reentry":
            eligible, displaced = reentry_candidates()
            if eligible:
                ordered = sorted(eligible, key=lambda label: (last_event_by_label[label], label))
                oldest_pool = ordered[:max(
                    1,
                    int(math.ceil(len(ordered) * max(0.0, min(1.0, config.deployment_reentry_oldest_fraction)))),
                )]
                selected_label = rng.choice(oldest_pool)
                pair = seen_by_label[selected_label]
                prior_event_idx = last_event_by_label[selected_label]
                target_idx = next(idx for idx, candidate in enumerate(matrix[pair.recipe_name]) if candidate.label == pair.label)
                was_current_preference = pair.preference_name == current_pair(pair.recipe_name).preference_name
                current_pref_idx[pair.recipe_name] = target_idx
                append_event(
                    "assist",
                    pair,
                    stream_idx=stream_idx,
                    event_type="deployment_reentry_probe",
                    condition="deployment_reentry",
                    condition_family="selective_forgetting_reentry",
                    evaluation_phase="reentry",
                    primary_probe=True,
                    rung_idx=target_idx,
                    scheduled_reentry_gap_events=event_idx - prior_event_idx,
                    reentry_schedule_policy="oldest_displaced_preference_pool",
                    reentry_candidate_count=len(seen_by_label),
                    reentry_displaced_candidate_count=len(displaced),
                    reentry_target_is_current_preference=was_current_preference,
                    hypothesis_tags=["selective_forgetting_reentry", "retention_after_interference"],
                    preference_non_default_axes=list(pair.non_default_axes),
                    **common,
                )
                continue

        # Either a requested condition exhausted its feasible support or this
        # slot was allocated to controls.  Keep the fallback explicit rather
        # than mislabelling it as transfer or adaptation.
        recipe = known_recipe()
        if recipe is None:
            raise RuntimeError("deployment stream has no known recipe after onboarding")
        pair = current_pair(recipe)
        append_event(
            "assist",
            pair,
            stream_idx=stream_idx,
            event_type="deployment_routine_reuse",
            condition="deployment_routine_reuse",
            condition_family="direct_retrieval_control",
            evaluation_phase="direct_retrieval",
            primary_probe=True,
            rung_idx=current_pref_idx[recipe],
            quota_fallback_from=(requested_type if requested_type != "routine" else None),
            hypothesis_tags=["routine_reuse", "direct_retrieval_control"],
            preference_non_default_axes=list(pair.non_default_axes),
            **common,
        )

    return ScenarioPlan(
        scenario=SCENARIO_DEPLOYMENT_RANDOM,
        seed=seed,
        events=tuple(events),
        eval_pairs=tuple(pair for recipe in recipes for pair in matrix[recipe]),
        selected_recipes=tuple(recipes),
        selected_preferences=tuple(preferences),
        description="Quota-controlled randomized single-user deployment stream. It guarantees requested support for adaptation, cross-recipe transfer, direct retrieval, and re-entry while preserving randomized feasible ordering.",
    )


def build_scenario_plan(scenario: str, config: EvaluationConfig, seed: int) -> ScenarioPlan:
    if scenario == SCENARIO_LADDER_HETEROGENEOUS:
        return build_ladder_heterogeneous(config, seed)
    if scenario == SCENARIO_LADDER_HOMOGENEOUS:
        return build_ladder_homogeneous(config, seed)
    if scenario == SCENARIO_DEPLOYMENT_RANDOM:
        return build_deployment_random(config, seed)
    raise KeyError(f"unknown scenario {scenario!r}; available={SCENARIOS}")


def _axis_labels(pair: RecipePreferencePair) -> Tuple[str, ...]:
    defaults = PRESET_PREFERENCES["identity"].as_dict()
    return tuple(
        f"{axis}={value}"
        for axis, value in sorted(dict(pair.axis_values).items())
        if value != defaults.get(axis)
    )


def _exposure_tags(
    pair: RecipePreferencePair,
    observed_recipes: set[str],
    observed_preferences: set[str],
    observed_pairs: set[str],
    preferences_by_recipe: Mapping[str, set[str]],
    axis_values_by_recipe: Mapping[str, set[str]],
) -> Dict[str, Any]:
    seen_recipe = pair.recipe_name in observed_recipes
    seen_pref = pair.preference_name in observed_preferences
    seen_pair = pair.label in observed_pairs
    same_recipe_pref = pair.preference_name in preferences_by_recipe.get(pair.recipe_name, set())
    other_recipe_pref = any(
        pair.preference_name in prefs
        for recipe, prefs in preferences_by_recipe.items()
        if recipe != pair.recipe_name
    )
    if seen_pair or (seen_recipe and same_recipe_pref):
        transfer_cell = "direct_retrieval"
    elif seen_recipe and other_recipe_pref:
        transfer_cell = "seen_recipe_preference_seen_elsewhere"
    elif seen_recipe:
        transfer_cell = "seen_recipe_new_preference"
    elif seen_pref:
        transfer_cell = "unseen_recipe_seen_preference"
    else:
        transfer_cell = "unseen_unseen"

    labels = set(_axis_labels(pair))
    same_axis = labels & axis_values_by_recipe.get(pair.recipe_name, set())
    other_axis = {
        label
        for recipe, vals in axis_values_by_recipe.items()
        if recipe != pair.recipe_name
        for label in labels & vals
    }
    global_axis = {label for vals in axis_values_by_recipe.values() for label in labels & vals}
    if not labels:
        axis_cell = "identity_or_no_nondefault_axis"
    elif same_axis:
        axis_cell = "same_recipe_axis_value_reuse"
    elif other_axis:
        axis_cell = "cross_recipe_axis_value_transfer"
    elif global_axis:
        axis_cell = "axis_value_seen_global"
    else:
        axis_cell = "new_axis_value"

    if seen_recipe and seen_pref:
        four_cell = "known_recipe_known_preference"
    elif seen_recipe:
        four_cell = "seen_recipe_new_preference"
    elif seen_pref:
        four_cell = "unseen_recipe_seen_preference"
    else:
        four_cell = "unseen_unseen"

    return {
        "seen_recipe_before": seen_recipe,
        "seen_preference_before": seen_pref,
        "seen_pair_before": seen_pair,
        "four_cell_before": four_cell,
        "transfer_cell_before": transfer_cell,
        "axis_transfer_cell_before": axis_cell,
        "axis_value_seen_same_recipe_labels_before": sorted(same_axis),
        "axis_value_seen_other_recipe_labels_before": sorted(other_axis),
        "axis_value_seen_global_labels_before": sorted(global_axis),
    }


def _active_keys(agent: AdaptiveHRCAgent) -> set[VariantKey]:
    return set(getattr(agent.decay, "active", {}).keys())


def _pruned_keys(agent: AdaptiveHRCAgent) -> set[VariantKey]:
    return set(getattr(agent.decay, "pruned", {}).keys())


def _evaluator_tokens_from_observations(
    agent: AdaptiveHRCAgent,
    observations: Sequence[Any],
) -> Tuple[str, ...]:
    """Map observed transition vectors to existing opaque agent tokens.

    This helper deliberately does not accept symbolic action strings and does
    not mutate the agent's codebook.  It belongs to the evaluator solely to
    score predictions against the current observation stream.
    """
    codebook = getattr(agent, "action_vector_to_token", {})
    return tuple(codebook.get(obs.action_vector, "act_unseen") for obs in observations)


def _opaque_observation_label(observation: Any) -> str:
    """Stable log-only ID for an observed action vector, never its action name."""
    payload = ",".join(str(int(value)) for value in observation.action_vector).encode("ascii")
    return f"obs_{hashlib.sha256(payload).hexdigest()[:12]}"


def _pair_key(
    agent: AdaptiveHRCAgent,
    pair: RecipePreferencePair,
    name_to_rid: Mapping[str, str],
) -> Optional[VariantKey]:
    rid = name_to_rid.get(pair.recipe_name)
    if rid is None:
        return None
    try:
        observations = observations_from_actions(pair.actions)
        tokens = _evaluator_tokens_from_observations(agent, observations)
        # If a vector was never observed, its opaque token has not been
        # allocated and this cannot be an exact stored variant.
        if any(token == "act_unseen" for token in tokens):
            return None
        return (rid, variant_hash(tokens))
    except Exception:
        return None


def _recipe_has_active_variant(
    agent: AdaptiveHRCAgent,
    pair: RecipePreferencePair,
    name_to_rid: Mapping[str, str],
) -> bool:
    rid = name_to_rid.get(pair.recipe_name)
    return bool(rid and any(recipe_id == rid for recipe_id, _ in _active_keys(agent)))


def _memory_state(
    agent: AdaptiveHRCAgent,
    pair: RecipePreferencePair,
    name_to_rid: Mapping[str, str],
    observed_pairs: set[str],
    observed_recipes: set[str],
) -> str:
    key = _pair_key(agent, pair, name_to_rid)
    if key and key in _active_keys(agent):
        return "active_memory"
    if key and (key in _pruned_keys(agent) or key[1] in getattr(agent.memory, "variants", {}).get(key[0], {})):
        return "pruned_memory"
    if pair.recipe_name in observed_recipes:
        return "same_recipe_new_preference"
    return "no_memory"


def _pref_id_for_pair(
    agent: AdaptiveHRCAgent,
    pair: RecipePreferencePair,
    name_to_rid: Mapping[str, str],
    hint: Optional[str],
) -> Optional[str]:
    if hint is not None:
        return hint
    key = _pair_key(agent, pair, name_to_rid)
    return getattr(agent, "variant_pref_ids", {}).get(key) if key else None


def _time_fields(
    *,
    hrc_total_time: float,
    human_only_time: float,
    human_effort_time: float,
    robot_correct_time: float = 0.0,
    robot_wrong_time: float = 0.0,
    human_action_time: float = 0.0,
    human_correction_time: float = 0.0,
) -> Dict[str, Any]:
    return {
        "testing_total_action_time": float(hrc_total_time),
        "testing_human_only_action_time": float(human_only_time),
        "testing_time_delta_vs_human_only": float(hrc_total_time - human_only_time),
        "testing_normalized_interaction_cost": _safe_div(hrc_total_time, human_only_time),
        "testing_human_effort_time": float(human_effort_time),
        "testing_robot_correct_action_time": float(robot_correct_time),
        "testing_robot_wrong_action_time": float(robot_wrong_time),
        "testing_human_action_time": float(human_action_time),
        "testing_human_correction_time": float(human_correction_time),
        "testing_efficiency_metric": "testing_total_action_time",
        "testing_efficiency_metric_value": float(hrc_total_time),
        "testing_efficiency_excludes_training": True,
    }


def _adaptation_recovery(correct_flags: Sequence[bool], first_mismatch: Optional[int]) -> Dict[str, float]:
    flags = [bool(v) for v in correct_flags]
    fm = int(first_mismatch) if first_mismatch is not None and int(first_mismatch) >= 0 else -1
    has_mismatch = 0 <= fm < len(flags)
    out = {
        "first_mismatch_rate": 1.0 if has_mismatch else 0.0,
        "first_mismatch_robot_turn": float(fm if has_mismatch else -1),
    }
    for window in ADAPTATION_RECOVERY_WINDOWS:
        segment = flags[fm + 1:fm + 1 + window] if has_mismatch else []
        eligible = len(segment) == window
        out[f"adaptation_recovery_top1_w{window}"] = _mean(1.0 if v else 0.0 for v in segment) if eligible else 0.0
        out[f"adaptation_recovery_eligible_w{window}"] = 1.0 if eligible else 0.0
    out["primary_adaptation_recovery_top1"] = out[f"adaptation_recovery_top1_w{PRIMARY_ADAPTATION_RECOVERY_WINDOW}"]
    out["primary_adaptation_recovery_eligible"] = out[f"adaptation_recovery_eligible_w{PRIMARY_ADAPTATION_RECOVERY_WINDOW}"]
    return out


def observe_episode(
    agent: AdaptiveHRCAgent,
    pair: RecipePreferencePair,
    name_to_rid: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    if isinstance(agent, OracleCeilingAgent):
        agent.set_oracle_target(pair.actions)
    n_steps = len(pair.actions)
    human_only = float(n_steps * DEFAULT_HRC_TIMING.human_action_time)
    total_time = human_only + float(n_steps * OBSERVATION_MODE_EXTRA_TIME_PER_STEP)
    t0 = time.perf_counter()
    agent.start_demo()
    for obs in observations_from_actions(pair.actions):
        agent.observe_observation(obs)
    cls = agent.end_demo()
    if name_to_rid is not None and cls.recipe_id is not None:
        name_to_rid[pair.recipe_name] = cls.recipe_id
    return {
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "mode": "observe",
        "classification_kind": cls.kind,
        "classification_recipe_id": cls.recipe_id,
        "classification_variant_hash": cls.variant_hash,
        "episode_wall_s": float(time.perf_counter() - t0),
        "n_steps": 0,
        "n_recipe_steps": n_steps,
        "live_top1": 0.0,
        "live_topk": 0.0,
        "human_correction_rate": 0.0,
        "robot_wrong_rate": 0.0,
        "hrc_robot_turn_count": 0,
        "hrc_human_turn_count": n_steps,
        "hrc_human_correction_count": 0,
        "hrc_robot_correct_count": 0,
        "hrc_robot_wrong_count": 0,
        "hrc_robot_topk_hit_count": 0,
        "hrc_human_shadow_turn_count": 0,
        "hrc_human_shadow_correct_count": 0,
        "hrc_human_shadow_topk_hit_count": 0,
        "committed_latent_pref_id": getattr(agent, "last_pref_id", None),
        "commit_attempted": False,
        "_turn_records": [],
        "human_effort_time": human_only,
        "observation_mode_episode": 1.0,
        "user_observation_required": 1.0,
        **_time_fields(
            hrc_total_time=total_time,
            human_only_time=human_only,
            human_effort_time=human_only,
            human_action_time=human_only,
        ),
    }


def assist_episode(
    agent: AdaptiveHRCAgent,
    pair: RecipePreferencePair,
    name_to_rid: Mapping[str, str],
    *,
    config: EvaluationConfig,
    commit: bool = True,
    true_preference_id: Optional[str] = None,
    observed_pairs: Optional[set[str]] = None,
    observed_recipes: Optional[set[str]] = None,
    memory_state_before: Optional[str] = None,
    preserve_noncommitting_state: bool = True,
    capture_turn_records: bool = True,
) -> Dict[str, Any]:
    """Run one alternating HRC episode.

    Non-committing calls normally snapshot and restore the agent.  A caller
    evaluating several independent probes from one checkpoint may instead own
    that restore operation by setting ``preserve_noncommitting_state=False``.
    This avoids taking two full agent copies for every pair in a frozen sweep.
    ``capture_turn_records`` is disabled for frozen sweeps because their
    aggregate rows retain only episode-level outcomes.
    """
    snapshot = agent.snapshot() if (not commit and preserve_noncommitting_state) else None
    pre_observed_pairs = set(observed_pairs or ())
    pre_observed_recipes = set(observed_recipes or ())
    pre_memory_state = memory_state_before or _memory_state(
        agent,
        pair,
        name_to_rid,
        pre_observed_pairs,
        pre_observed_recipes,
    )
    true_pref_id = _pref_id_for_pair(agent, pair, name_to_rid, true_preference_id)
    observations = observations_from_actions(pair.actions)
    actual_tokens = _evaluator_tokens_from_observations(agent, observations)
    actual_labels = tuple(_opaque_observation_label(obs) for obs in observations)
    if isinstance(agent, OracleCeilingAgent):
        agent.set_oracle_target(actual_tokens)

    def predict(prefix: Sequence[str]) -> Mapping[str, float]:
        return agent.predict_next_tokens(list(prefix))

    def observe(obs: Any, distribution: Optional[Mapping[str, float]] = None) -> None:
        # The agent receives only the observed numeric state transition and its
        # own prior distribution. Ground-truth recipe/preference labels remain
        # evaluator-only metadata.
        agent.observe_observation(obs, precomputed_distribution=distribution)

    def robot_feedback(context: Any) -> None:
        agent.record_robot_feedback(
            prefix=context.prefix,
            predicted=context.predicted,
            actual=context.actual,
            correct_top1=context.correct_top1,
        )

    def robot_metadata(context: Any) -> Mapping[str, Any]:
        stats = agent.action_policy_stats()
        actual_probability = context.distribution.get(context.actual)
        return {
            **stats,
            "actual_action_probability": float(actual_probability) if actual_probability is not None else None,
        }

    t0 = time.perf_counter()
    trace = run_alternating_hrc_episode(
        observations=observations,
        actual_tokens=actual_tokens,
        actual_actions=actual_labels,
        current_prefix=lambda: list(agent.current_prefix),
        predict_distribution=predict,
        observe_ground_truth=observe,
        topk=int(config.topk),
        prob_floor=float(agent.cfg.prob_floor),
        timing=DEFAULT_HRC_TIMING,
        capture_robot_metadata=robot_metadata,
        on_robot_feedback=robot_feedback,
    )
    cls = agent.end_demo() if commit else None
    if commit and isinstance(name_to_rid, dict) and cls is not None and cls.recipe_id is not None:
        name_to_rid[pair.recipe_name] = cls.recipe_id

    post_observed_pairs = pre_observed_pairs | {pair.label}
    post_observed_recipes = pre_observed_recipes | {pair.recipe_name}
    post_memory_state = (
        _memory_state(agent, pair, name_to_rid, post_observed_pairs, post_observed_recipes)
        if commit else pre_memory_state
    )

    summary = trace.summary
    n_robot = max(1, int(summary.n_robot_turns))
    robot_turns = list(trace.robot_turns)
    human_shadow_turns = list(trace.human_shadow_turns)
    turn_records: List[Dict[str, Any]] = []
    if capture_turn_records:
        for turn in robot_turns:
            turn_records.append({
                "turn_kind": "robot",
                "recipe_step": int(turn.recipe_step),
                "turn_idx": int(turn.robot_turn_idx),
                "prefix_length": len(turn.prefix),
                "actual": turn.actual,
                "predicted": turn.predicted,
                "correct_top1": bool(turn.correct_top1),
                "correct_topk": bool(turn.correct_topk),
                "prediction_wall_s": float(turn.prediction_wall_s),
                "scheduled_actor": turn.scheduled_actor,
                "executed_by": turn.executed_by,
                "human_corrected": bool(turn.human_corrected),
                "future_valid_wrong": bool(turn.future_valid_wrong),
                "predicted_future_offset": turn.predicted_future_offset,
                **dict(turn.metadata),
            })
        for turn in human_shadow_turns:
            turn_records.append({
                "turn_kind": "human_shadow",
                "recipe_step": int(turn.recipe_step),
                "turn_idx": int(turn.human_turn_idx),
                "prefix_length": len(turn.prefix),
                "actual": turn.actual,
                "predicted": turn.predicted,
                "correct_top1": bool(turn.correct_top1),
                "correct_topk": bool(turn.correct_topk),
                "prediction_wall_s": float(turn.prediction_wall_s),
                "scheduled_actor": turn.scheduled_actor,
                "executed_by": turn.executed_by,
                "human_corrected": False,
                "future_valid_wrong": False,
                "predicted_future_offset": None,
                **dict(turn.metadata),
            })
    metrics = {
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "true_preference_id": true_pref_id,
        "mode": "assist",
        # ``memory_state_gt`` is retained as a backward-compatible alias for
        # the pre-episode state. Never use the post-commit state to stratify
        # first-pass forgetting or re-entry performance.
        "memory_state_gt": pre_memory_state,
        "memory_state_before": pre_memory_state,
        "memory_state_after": post_memory_state,
        "classification_kind": getattr(cls, "kind", None),
        "classification_recipe_id": getattr(cls, "recipe_id", None),
        "classification_variant_hash": getattr(cls, "variant_hash", None),
        "episode_wall_s": float(time.perf_counter() - t0),
        "n_steps": int(summary.n_robot_turns),
        "n_recipe_steps": int(summary.n_recipe_steps),
        "live_top1": _safe_div(summary.robot_correct_count, n_robot),
        "live_topk": _safe_div(summary.hitsk, n_robot),
        f"live_top{int(config.topk)}": _safe_div(summary.hitsk, n_robot),
        "total_robot_turn_nll": float(summary.nll),
        "mean_nll_per_robot_turn": _safe_div(summary.nll, n_robot),
        "mean_nll_per_recipe_step": _safe_div(summary.nll, summary.n_recipe_steps),
        "mean_prediction_wall_s": _mean(t.prediction_wall_s for t in robot_turns),
        "hrc_robot_turn_count": int(summary.n_robot_turns),
        "hrc_human_turn_count": int(summary.human_turn_count),
        "hrc_human_correction_count": int(summary.human_correction_count),
        "hrc_robot_correct_count": int(summary.robot_correct_count),
        "hrc_robot_wrong_count": int(summary.robot_wrong_count),
        "hrc_robot_topk_hit_count": int(summary.hitsk),
        "hrc_human_shadow_turn_count": int(len(human_shadow_turns)),
        "hrc_human_shadow_correct_count": int(sum(int(turn.correct_top1) for turn in human_shadow_turns)),
        "hrc_human_shadow_topk_hit_count": int(sum(int(turn.correct_topk) for turn in human_shadow_turns)),
        "human_correction_rate": _safe_div(summary.human_correction_count, n_robot),
        "robot_wrong_rate": _safe_div(summary.robot_wrong_count, n_robot),
        "human_effort_time": float(summary.human_effort_time),
        "future_valid_wrong_rate": _safe_div(summary.future_valid_wrong_count, n_robot),
        "wrong_prediction_future_valid_rate": _safe_div(summary.future_valid_wrong_count, summary.robot_wrong_count),
        "observation_mode_episode": 0.0,
        "user_observation_required": 1.0 if getattr(agent, "_needs_observation", False) else 0.0,
        "committed_latent_pref_id": getattr(agent, "last_pref_id", None) if commit else None,
        "commit_attempted": bool(commit),
        "_turn_records": turn_records,
        **_adaptation_recovery([t.correct_top1 for t in robot_turns], summary.first_mismatch_robot_turn),
        **_time_fields(
            hrc_total_time=float(summary.hrc_total_time),
            human_only_time=float(summary.human_only_time),
            human_effort_time=float(summary.human_effort_time),
            robot_correct_time=float(summary.hrc_robot_correct_time),
            robot_wrong_time=float(summary.hrc_robot_wrong_time),
            human_action_time=float(summary.hrc_human_action_time),
            human_correction_time=float(summary.hrc_human_correction_time),
        ),
    }

    if snapshot is not None:
        agent.restore_from(snapshot)
    return metrics


def frozen_eval(
    agent: AdaptiveHRCAgent,
    pairs: Sequence[RecipePreferencePair],
    name_to_rid: Mapping[str, str],
    *,
    config: EvaluationConfig,
    checkpoint: str,
    event_idx: int,
    context: Mapping[str, Any],
    observed_pairs: set[str],
    observed_recipes: set[str],
) -> List[Dict[str, Any]]:
    # The old path let each ``assist_episode(commit=False)`` snapshot before
    # and deep-copy again while restoring afterwards.  A frozen sweep contains
    # many independent pairs at exactly the same checkpoint, so retain one
    # immutable checkpoint and restore it between probes instead.  This is
    # outcome-equivalent while halving full-agent copies; failure paths still
    # restore the caller's agent in the ``finally`` block.
    checkpoint_state = agent.snapshot()
    rows: List[Dict[str, Any]] = []
    restore_required = False
    try:
        for pair in pairs[: max(0, int(config.frozen_eval_max_pairs))]:
            restore_required = True
            metrics = assist_episode(
                agent,
                pair,
                name_to_rid,
                config=config,
                commit=False,
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                preserve_noncommitting_state=False,
                capture_turn_records=False,
            )
            rows.append({
                **dict(context),
                "checkpoint": checkpoint,
                "event_idx": int(event_idx),
                "pair": pair.label,
                "recipe": pair.recipe_name,
                "preference": pair.preference_name,
                "top1": float(metrics.get("live_top1", 0.0)),
                "topk": float(metrics.get("live_topk", 0.0)),
                "human_correction_rate": float(metrics.get("human_correction_rate", 0.0)),
                "memory_state_gt": metrics.get("memory_state_gt"),
            })
            agent.restore_from(checkpoint_state)
            restore_required = False
    finally:
        if restore_required:
            agent.restore_from(checkpoint_state)
    return rows


def memory_snapshot(agent: AdaptiveHRCAgent, wall_s: float = 0.0) -> Dict[str, Any]:
    active = getattr(agent.decay, "active", {})
    pruned = getattr(agent.decay, "pruned", {})
    weights = [float(entry.weight) for entry in active.values()]
    fit_times = _finite(getattr(agent, "retrain_fit_wall_times", []))
    total_times = _finite(getattr(agent, "retrain_total_wall_times", []))
    build_times = _finite(getattr(agent, "retrain_build_wall_times", []))
    flop_estimates = _finite(getattr(agent, "retrain_flop_estimates", []))
    estimated_fit_flops = float(sum(flop_estimates))
    fit_wall_s = float(sum(fit_times))
    latest_fit_stats: Dict[str, Any] = {}
    latest_fit_stats_method = getattr(agent, "_latest_fit_stats", None)
    if callable(latest_fit_stats_method):
        try:
            candidate = latest_fit_stats_method()
            if isinstance(candidate, dict):
                latest_fit_stats = candidate
        except Exception:
            # Accounting must never make an evaluation fail after the model
            # has completed.  The event-level statistics remain available.
            latest_fit_stats = {}
    out = {
        "active_variants": int(len(active)),
        "pruned_variants": int(len(pruned)),
        "active_action_steps": int(sum(len(entry.ordering) for entry in active.values())),
        "latest_keys": int(len(getattr(agent.decay, "latest_keys", set()))),
        "mean_active_weight": _mean(weights),
        "min_active_weight": min(weights) if weights else 0.0,
        "max_active_weight": max(weights) if weights else 0.0,
        "retrain_cycle": int(getattr(agent, "retrain_cycle", 0)),
        "retrain_skipped_count": int(getattr(agent, "retrain_skipped_count", 0)),
        "wall_s": float(wall_s),
        "training_total_retrain_wall_s": float(sum(total_times)),
        "training_fit_wall_s": fit_wall_s,
        "training_build_wall_s": float(sum(build_times)),
        "training_retrain_count": int(len(total_times)),
        "training_skipped_retrain_count": int(getattr(agent, "retrain_skipped_count", 0)),
        # ``training_estimated_flops`` is retained as a backward-compatible
        # alias.  New analyses should use the explicitly scoped name below
        # and pair it only with ``training_fit_wall_s``.
        "training_estimated_flops": estimated_fit_flops,
        "training_estimated_fit_flops": estimated_fit_flops,
        "training_flop_accounting_scope": "fit_only_model_specific_arithmetic",
        "training_flop_cross_model_comparable": False,
        "training_fit_effective_gflops_s": (
            estimated_fit_flops / (1.0e9 * fit_wall_s) if fit_wall_s > 0.0 else 0.0
        ),
        "training_fit_effective_gflops_interpretation": "within_implementation_diagnostic_only",
        "training_total_retrain_wall_scope": "end_to_end_retrain_including_trajectory_build",
        "mean_retrain_fit_wall_s": _mean(fit_times),
        "p95_retrain_fit_wall_s": _p95(fit_times),
        "mean_estimated_flops": _mean(flop_estimates),
    }
    if latest_fit_stats:
        out["latest_fit_flop_accounting_scope"] = latest_fit_stats.get("flop_accounting_scope", "unspecified")
        out["latest_fit_flop_cross_model_comparable"] = bool(latest_fit_stats.get("flop_cross_model_comparable", False))
    for method_name, key in (
        ("_latest_fit_stats", "fit_stats"),
        ("replay_buffer_metadata", "replay_buffer"),
        ("baseline_memory_metadata", "baseline_model_memory"),
    ):
        method = getattr(agent, method_name, None)
        if callable(method):
            try:
                out[key] = method()
            except Exception:
                pass
    return out


def _prototype_rows(
    agent: AdaptiveHRCAgent,
    context: Mapping[str, Any],
    before_count: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, event in enumerate((getattr(agent, "prototype_events", []) or [])[before_count:]):
        rows.append({
            **dict(context),
            "diagnostic_type": "prototype_stability",
            "prototype_event_idx": int(before_count + idx),
            "prototype_event": event.get("event"),
            "prototype_new_count": len(event.get("prototype_new", []) or []),
            "prototype_retired_count": len(event.get("prototype_retired", []) or []),
            "prototype_persisted_count": len(event.get("prototype_persisted", []) or []),
            "prototype_split_count": int(event.get("prototype_split_count", 0) or 0),
            "prototype_merge_count": int(event.get("prototype_merge_count", 0) or 0),
            "prototype_stability_ari": event.get("prototype_stability_ari"),
        })
    return rows


def _active_only_audit_row(
    agent: AdaptiveHRCAgent,
    context: Mapping[str, Any],
    config: EvaluationConfig,
) -> Dict[str, Any]:
    row = {
        **dict(context),
        "diagnostic_type": "active_only_pruned_influence_audit",
        "active_variants": len(getattr(agent.decay, "active", {})),
        "pruned_variants": len(getattr(agent.decay, "pruned", {})),
    }
    audit = getattr(agent, "pruned_influence_audit", None)
    if not callable(audit):
        return {**row, "audit_available": False, "primary_active_only_contract_passed": None}
    try:
        result = dict(audit(
            max_prefixes=int(config.active_only_audit_max_prefixes),
            tolerance=float(config.active_only_audit_tolerance),
        ))
        primary = result.get("live_prediction_passed", result.get("passed"))
        return {
            **row,
            **result,
            "audit_available": True,
            "primary_active_only_contract": "live_prediction_excludes_pruned_registry",
            "primary_active_only_contract_passed": bool(primary) if primary is not None else None,
        }
    except Exception as exc:
        return {
            **row,
            "audit_available": False,
            "passed": False,
            "primary_active_only_contract": "live_prediction_excludes_pruned_registry",
            "primary_active_only_contract_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sync_latest_after_clairvoyant_prune(agent: AdaptiveHRCAgent) -> None:
    active_by_recipe: Dict[str, List[Any]] = defaultdict(list)
    for entry in getattr(agent.decay, "active", {}).values():
        active_by_recipe[str(entry.recipe_id)].append(entry)

    known_recipe_ids = set(getattr(agent.memory, "variants", {}).keys())
    known_recipe_ids.update(getattr(agent.memory, "latest", {}).keys())
    known_recipe_ids.update(getattr(agent.decay, "latest_by_recipe", {}).keys())
    for rid in known_recipe_ids:
        entries = active_by_recipe.get(rid, [])
        if not entries:
            getattr(agent.memory, "latest", {}).pop(rid, None)
            getattr(agent.decay, "latest_by_recipe", {}).pop(rid, None)
            for key in list(getattr(agent.decay, "latest_keys", set())):
                if key[0] == rid:
                    agent.decay.latest_keys.discard(key)
            continue
        newest = max(entries, key=lambda e: (int(getattr(e, "last_seen_step", -1)), str(getattr(e, "variant_hash", ""))))
        agent.memory.latest[rid] = newest.variant_hash
        agent.decay.mark_latest(rid, newest.variant_hash)

    remap = getattr(agent, "_remap_preference_ids_after_rebuild", None)
    if callable(remap):
        remap()


def _apply_clairvoyant_memory_pruning(
    agent: AdaptiveHRCAgent,
    future_events: Sequence[ScenarioEvent],
    name_to_rid: Mapping[str, str],
) -> Dict[str, Any]:
    future_recipe_ids = {
        str(name_to_rid[event.pair.recipe_name])
        for event in future_events
        if event.pair.recipe_name in name_to_rid
    }
    active_before = _active_keys(agent)
    discard = sorted(key for key in active_before if key[0] not in future_recipe_ids)
    for rid, variant_hash_ in discard:
        agent.decay.discard(rid, variant_hash_, allow_latest=True)
    if discard:
        _sync_latest_after_clairvoyant_prune(agent)
        agent.refresh_model_from_memory()
    return {
        "diagnostic_type": "clairvoyant_pruning",
        "oracle_pruned_active_variants": len(discard),
        "oracle_active_variants_before": len(active_before),
        "oracle_active_variants_after": len(_active_keys(agent)),
        "oracle_future_recipe_count": len(future_recipe_ids),
        "oracle_retention_policy": "future_recipe_support",
        "oracle_pruned_keys": [f"{rid}:{h}" for rid, h in discard],
    }


def _event_context(
    baseline: str,
    plan: ScenarioPlan,
    event_idx: int,
    requested_mode: str,
    executed_mode: str,
    pair: RecipePreferencePair,
    tags: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "baseline": baseline,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "event_idx": int(event_idx),
        "requested_mode": requested_mode,
        "mode": executed_mode,
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "preference_axis_values": dict(pair.axis_values),
        "preference_non_default_axes": list(pair.non_default_axes),
        "preference_axis_value_labels": list(_axis_labels(pair)),
        "preference_is_composed": pair.is_composed_preference,
        "preference_ordering_is_distinct": bool(pair.ordering_is_distinct),
        "preference_duplicate_of": pair.duplicate_of_preference,
        "no_op_or_duplicate_rung": not bool(pair.ordering_is_distinct),
        **dict(tags),
    }


def _record_observed(
    pair: RecipePreferencePair,
    observed_recipes: set[str],
    observed_preferences: set[str],
    observed_pairs: set[str],
    preferences_by_recipe: Dict[str, set[str]],
    axis_values_by_recipe: Dict[str, set[str]],
) -> None:
    observed_recipes.add(pair.recipe_name)
    observed_preferences.add(pair.preference_name)
    observed_pairs.add(pair.label)
    preferences_by_recipe[pair.recipe_name].add(pair.preference_name)
    axis_values_by_recipe[pair.recipe_name].update(_axis_labels(pair))


def _is_preference_rung_boundary(plan: ScenarioPlan, event_idx: int) -> bool:
    """Whether ``event_idx`` closes a non-onboarding ladder rung.

    This is deliberately tag-based rather than a fixed event interval: the
    number of recipes and settling repeats may change, while every completed
    homogeneous preference rung should still receive exactly one full-grid
    frozen evaluation.
    """
    if event_idx < 0 or event_idx >= len(plan.events):
        return False
    rung = plan.events[event_idx].tags.get("rung_idx")
    if not isinstance(rung, int) or rung <= 0:
        return False
    if event_idx == len(plan.events) - 1:
        return True
    return plan.events[event_idx + 1].tags.get("rung_idx") != rung


def _should_run_periodic_frozen_eval(
    plan: ScenarioPlan,
    event_idx: int,
    config: EvaluationConfig,
) -> bool:
    """Return whether to run a full evaluation-grid frozen sweep.

    The three paper scenarios intentionally use different diagnostic cadence:
    homogeneous learning is summarized at rung completion, heterogeneous
    learning every five events, and randomized deployment has only its
    event-local pre-interaction probe.  This changes diagnostic timing only;
    it never changes live interaction or model updates.
    """
    if plan.scenario == SCENARIO_LADDER_HOMOGENEOUS:
        return _is_preference_rung_boundary(plan, event_idx)
    if plan.scenario == SCENARIO_LADDER_HETEROGENEOUS:
        period = int(config.frozen_eval_period)
        return period > 0 and (event_idx + 1) % period == 0
    if plan.scenario == SCENARIO_DEPLOYMENT_RANDOM:
        return False
    return False


def _should_run_pre_event_frozen_probe(
    plan: ScenarioPlan,
    requested_mode: str,
    executed_mode: str,
    tags: Mapping[str, Any],
    config: EvaluationConfig,
) -> bool:
    """Select matched, single-pair pre-interaction probes.

    New recipes routed to observation have no deployable assistive prediction,
    so they are excluded.  Every routed assist event in randomized deployment
    is measured; ladders retain their explicitly tagged primary probes only.
    """
    if not bool(config.pre_event_frozen_probes) or requested_mode != "assist" or executed_mode != "assist":
        return False
    if plan.scenario == SCENARIO_DEPLOYMENT_RANDOM:
        return True
    return bool(tags.get("primary_probe", False))


def run_event_stream_for_baseline(
    baseline: str,
    plan: ScenarioPlan,
    config: EvaluationConfig,
) -> EventStreamRun:
    is_clairvoyant = baseline == CLAIRVOYANT_MEMORY_ORACLE
    agent = make_agent("full" if is_clairvoyant else baseline, base_config(plan.seed, config))
    name_to_rid: Dict[str, str] = {}
    episode_rows: List[Dict[str, Any]] = []
    frozen_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []
    prototype_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    turn_rows: List[Dict[str, Any]] = []
    observed_recipes: set[str] = set()
    observed_preferences: set[str] = set()
    observed_pairs: set[str] = set()
    preferences_by_recipe: Dict[str, set[str]] = defaultdict(set)
    axis_values_by_recipe: Dict[str, set[str]] = defaultdict(set)
    pref_to_pid: Dict[str, Optional[str]] = {}
    last_global_frozen_event_idx: Optional[int] = None
    t0 = time.perf_counter()

    for event_idx, event in enumerate(plan.events):
        pair = event.pair
        if is_clairvoyant:
            prune = _apply_clairvoyant_memory_pruning(agent, plan.events[event_idx:], name_to_rid)
            oracle_rows.append({
                **prune,
                "baseline": baseline,
                "scenario": plan.scenario,
                "seed": int(plan.seed),
                "event_idx": int(event_idx),
                "timing": "before_event",
                "oracle_reference": CLAIRVOYANT_MEMORY_ORACLE,
                "reported_as": CLAIRVOYANT_REFERENCE_TAG,
                "leakage_warning": CLAIRVOYANT_LEAKAGE_WARNING,
            })

        tags = {
            **dict(event.tags),
            **_exposure_tags(
                pair,
                observed_recipes,
                observed_preferences,
                observed_pairs,
                preferences_by_recipe,
                axis_values_by_recipe,
            ),
        }
        requested_mode = event.mode
        executed_mode = requested_mode
        route_reason = "user_selected"
        # Evaluator-only pre-episode label. It is captured before routing,
        # observations, or commits, and is never supplied to the agent.
        memory_state_before = _memory_state(
            agent,
            pair,
            name_to_rid,
            observed_pairs,
            observed_recipes,
        )
        target_key_before = _pair_key(agent, pair, name_to_rid)
        target_recipe_id_before = name_to_rid.get(pair.recipe_name)
        latest_before = (
            getattr(agent.memory, "latest", {}).get(target_recipe_id_before)
            if target_recipe_id_before is not None else None
        )
        if (
            requested_mode == "assist"
            and config.route_absent_recipe_assists_to_observe
            and not _recipe_has_active_variant(agent, pair, name_to_rid)
        ):
            executed_mode = "observe"
            route_reason = "assist_routed_to_observe_recipe_absent_from_active_memory"
        tags.update({"requested_mode": requested_mode, "executed_mode": executed_mode, "mode_route_reason": route_reason})
        if is_clairvoyant:
            tags.update({
                "oracle_reference": CLAIRVOYANT_MEMORY_ORACLE,
                "reported_as": CLAIRVOYANT_REFERENCE_TAG,
                "leakage_warning": CLAIRVOYANT_LEAKAGE_WARNING,
                "oracle_retention_policy": "future_recipe_support",
            })

        context = _event_context(baseline, plan, event_idx, requested_mode, executed_mode, pair, tags)
        active_before = _active_keys(agent)
        pruned_before = _pruned_keys(agent)
        retrain_before = len(getattr(agent, "retrain_events", []) or [])
        proto_before = len(getattr(agent, "prototype_events", []) or [])

        # A matched frozen probe evaluates the same pre-event memory state for
        # every baseline without changing the real interaction route or its
        # workload cost.  Only primary probes opt in; settling repeats remain
        # live-only outcomes rather than recursively evaluated traces.
        if _should_run_pre_event_frozen_probe(
            plan,
            requested_mode,
            executed_mode,
            tags,
            config,
        ):
            frozen_rows.extend(frozen_eval(
                agent,
                (pair,),
                name_to_rid,
                config=config,
                checkpoint=f"pre_event_{event_idx}",
                event_idx=event_idx,
                context={
                    **context,
                    "probe_phase": "pre_event",
                    "primary_probe": True,
                    "live_event_idx": int(event_idx),
                },
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
            ))

        if executed_mode == "observe":
            row = observe_episode(agent, pair, name_to_rid)
            post_observed_pairs = set(observed_pairs) | {pair.label}
            post_observed_recipes = set(observed_recipes) | {pair.recipe_name}
            row.update({
                "memory_state_gt": memory_state_before,
                "memory_state_before": memory_state_before,
                "memory_state_after": _memory_state(
                    agent,
                    pair,
                    name_to_rid,
                    post_observed_pairs,
                    post_observed_recipes,
                ),
            })
            pref_to_pid[pair.preference_name] = getattr(agent, "last_pref_id", None)
        else:
            row = assist_episode(
                agent,
                pair,
                name_to_rid,
                config=config,
                commit=True,
                true_preference_id=pref_to_pid.get(pair.preference_name),
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                memory_state_before=memory_state_before,
            )
            latest_pid = getattr(agent, "last_pref_id", None)
            if latest_pid is not None:
                pref_to_pid[pair.preference_name] = latest_pid
        row.update(context)
        target_key_after = _pair_key(agent, pair, name_to_rid)
        target_recipe_id_after = name_to_rid.get(pair.recipe_name)
        committed = bool(executed_mode == "assist" and row.get("commit_attempted"))
        classified_recipe = row.get("classification_recipe_id")
        committed_variant = row.get("classification_variant_hash")
        expected_variant_hash = target_key_after[1] if target_key_after is not None else None
        latest_after = (
            getattr(agent.memory, "latest", {}).get(target_recipe_id_after)
            if target_recipe_id_after is not None else None
        )
        scheduled_reentry = bool(
            "selective_forgetting_reentry" in (row.get("hypothesis_tags") or [])
        )
        if target_key_before and target_key_before in pruned_before:
            reentry_target_state_before = "pruned_exact_variant"
        elif target_key_before and target_key_before in active_before:
            reentry_target_state_before = "active_exact_variant"
        elif pair.recipe_name in observed_recipes:
            reentry_target_state_before = "known_recipe_no_exact_variant"
        else:
            reentry_target_state_before = "unknown_recipe"
        row.update({
            "target_variant_active_before": bool(target_key_before and target_key_before in active_before),
            "target_variant_pruned_before": bool(target_key_before and target_key_before in pruned_before),
            "scheduled_reentry_probe": scheduled_reentry,
            "reentry_probe_target_state_before": reentry_target_state_before if scheduled_reentry else None,
            "actual_reentry_from_pruned": bool(row.get("classification_kind") == "reentry_from_pruned"),
            "commit_recipe_correct": (
                bool(classified_recipe == target_recipe_id_after)
                if committed and target_recipe_id_after is not None else None
            ),
            "commit_variant_correct": (
                bool(committed_variant == expected_variant_hash)
                if committed and expected_variant_hash is not None else None
            ),
            "false_variant_creation": bool(
                committed
                and pair.label in observed_pairs
                and row.get("classification_kind") in {"preference_shift", "tentative_preference_shift"}
                and target_key_before is None
            ),
            "false_latest_promotion": bool(
                committed
                and target_recipe_id_after is not None
                and latest_after is not None
                and expected_variant_hash is not None
                and latest_after != expected_variant_hash
                and latest_after != latest_before
            ),
        })
        for turn in row.pop("_turn_records", []):
            turn_rows.append({**context, **turn})
        episode_rows.append(row)
        _record_observed(pair, observed_recipes, observed_preferences, observed_pairs, preferences_by_recipe, axis_values_by_recipe)

        after_prune: Dict[str, Any] = {}
        if is_clairvoyant:
            after_prune = _apply_clairvoyant_memory_pruning(agent, plan.events[event_idx + 1:], name_to_rid)
            oracle_rows.append({
                **after_prune,
                "baseline": baseline,
                "scenario": plan.scenario,
                "seed": int(plan.seed),
                "event_idx": int(event_idx),
                "timing": "after_event",
                "oracle_reference": CLAIRVOYANT_MEMORY_ORACLE,
                "reported_as": CLAIRVOYANT_REFERENCE_TAG,
                "leakage_warning": CLAIRVOYANT_LEAKAGE_WARNING,
            })

        active_after = _active_keys(agent)
        pruned_after = _pruned_keys(agent)
        memory_rows.append({
            **context,
            "diagnostic_type": "memory_compute",
            **memory_snapshot(agent),
            "active_added_count": len(active_after - active_before),
            "active_removed_count": len(active_before - active_after),
            "pruned_added_count": len(pruned_after - pruned_before),
            "pruned_removed_count": len(pruned_before - pruned_after),
            "retrain_event_count_delta": len((getattr(agent, "retrain_events", []) or [])[retrain_before:]),
            "post_event_oracle_pruned_active_variants": after_prune.get("oracle_pruned_active_variants"),
        })
        prototype_rows.extend(_prototype_rows(agent, context, proto_before))

        if config.active_only_audit_period > 0 and (event_idx + 1) % int(config.active_only_audit_period) == 0:
            audit_rows.append(_active_only_audit_row(agent, {**context, "audit_checkpoint": f"event_{event_idx}"}, config))

        if _should_run_periodic_frozen_eval(plan, event_idx, config):
            frozen_rows.extend(frozen_eval(
                agent,
                plan.eval_pairs,
                name_to_rid,
                config=config,
                checkpoint=f"event_{event_idx}",
                event_idx=event_idx,
                context={"baseline": baseline, "scenario": plan.scenario, "seed": int(plan.seed)},
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
            ))
            last_global_frozen_event_idx = event_idx

    final_context = {"baseline": baseline, "scenario": plan.scenario, "seed": int(plan.seed)}
    # Do not duplicate the final full-grid sweep when the final event already
    # closes a homogeneous rung or lands on the heterogeneous five-event
    # cadence.  Random deployment has no periodic full-grid sweeps, so it
    # retains one final global snapshot in addition to its event-local probes.
    if last_global_frozen_event_idx != len(plan.events) - 1:
        frozen_rows.extend(frozen_eval(
            agent,
            plan.eval_pairs,
            name_to_rid,
            config=config,
            checkpoint="final",
            event_idx=len(plan.events) - 1,
            context=final_context,
            observed_pairs=observed_pairs,
            observed_recipes=observed_recipes,
        ))
    audit_rows.append(_active_only_audit_row(
        agent,
        {**final_context, "event_idx": len(plan.events) - 1, "audit_checkpoint": "final"},
        config,
    ))

    return EventStreamRun(
        baseline=baseline,
        scenario=plan.scenario,
        seed=plan.seed,
        agent=agent,
        name_to_rid=name_to_rid,
        episode_rows=episode_rows,
        frozen_rows=frozen_rows,
        memory_rows=memory_rows,
        prototype_rows=prototype_rows,
        active_audit_rows=audit_rows,
        oracle_pruning_rows=oracle_rows,
        turn_rows=turn_rows,
        wall_s=float(time.perf_counter() - t0),
    )


def aggregate_episode_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n_episodes": 0.0, "n_steps": 0.0, "live_top1": 0.0, "live_topk": 0.0}
    robot_turns = sum(_numeric(row, "hrc_robot_turn_count") for row in rows)
    out = {
        "n_episodes": float(len(rows)),
        "n_steps": float(sum(_numeric(row, "n_steps") for row in rows)),
        "n_recipe_steps": float(sum(_numeric(row, "n_recipe_steps") for row in rows)),
        "live_top1": _safe_div(sum(_numeric(row, "hrc_robot_correct_count") for row in rows), robot_turns),
        "live_topk": _safe_div(sum(_numeric(row, "hrc_robot_topk_hit_count") for row in rows), robot_turns),
        "robot_wrong_rate": _safe_div(sum(_numeric(row, "hrc_robot_wrong_count") for row in rows), robot_turns),
        "human_correction_rate": _safe_div(sum(_numeric(row, "hrc_human_correction_count") for row in rows), robot_turns),
        "testing_total_action_time": float(sum(_numeric(row, "testing_total_action_time") for row in rows)),
        "testing_human_only_action_time": float(sum(_numeric(row, "testing_human_only_action_time") for row in rows)),
        "testing_human_effort_time": float(sum(_numeric(row, "testing_human_effort_time") for row in rows)),
        "testing_normalized_interaction_cost": _safe_div(
            sum(_numeric(row, "testing_total_action_time") for row in rows),
            sum(_numeric(row, "testing_human_only_action_time") for row in rows),
        ),
        "human_effort_time": float(sum(_numeric(row, "human_effort_time") for row in rows)),
        "mean_nll_per_robot_turn": _safe_div(sum(_numeric(row, "total_robot_turn_nll") for row in rows), robot_turns),
        "mean_prediction_wall_s": _mean(_numeric(row, "mean_prediction_wall_s") for row in rows),
        "human_shadow_top1": _safe_div(
            sum(_numeric(row, "hrc_human_shadow_correct_count") for row in rows),
            sum(_numeric(row, "hrc_human_shadow_turn_count") for row in rows),
        ),
        "human_shadow_topk": _safe_div(
            sum(_numeric(row, "hrc_human_shadow_topk_hit_count") for row in rows),
            sum(_numeric(row, "hrc_human_shadow_turn_count") for row in rows),
        ),
        "n_human_shadow_turns": float(sum(_numeric(row, "hrc_human_shadow_turn_count") for row in rows)),
        "future_valid_wrong_rate": _safe_div(sum(_numeric(row, "future_valid_wrong_rate") * _numeric(row, "hrc_robot_turn_count") for row in rows), robot_turns),
        "observation_mode_rate": _safe_div(sum(1.0 for row in rows if row.get("mode") == "observe"), len(rows)),
        "user_observation_required_rate": _mean(_numeric(row, "user_observation_required") for row in rows),
    }
    for window in ADAPTATION_RECOVERY_WINDOWS:
        eligible = [row for row in rows if _numeric(row, f"adaptation_recovery_eligible_w{window}") > 0.0]
        out[f"adaptation_recovery_top1_w{window}"] = _mean(_numeric(row, f"adaptation_recovery_top1_w{window}") for row in eligible)
        out[f"adaptation_recovery_eligible_rate_w{window}"] = _safe_div(len(eligible), len(rows))
    primary = PRIMARY_ADAPTATION_RECOVERY_WINDOW
    out["primary_adaptation_recovery_top1"] = out[f"adaptation_recovery_top1_w{primary}"]
    out["primary_adaptation_recovery_eligible_rate"] = out[f"adaptation_recovery_eligible_rate_w{primary}"]
    out["primary_prediction_metric"] = "live_top1"
    out["primary_prediction_metric_value"] = out["live_top1"]
    out["primary_hrc_metric"] = "human_correction_rate"
    out["primary_hrc_metric_value"] = out["human_correction_rate"]
    return out


def _group_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                grouped[str(item)].append(row)
        else:
            grouped[str(value if value is not None else "unknown")].append(row)
    return {group: aggregate_episode_metrics(vals) for group, vals in sorted(grouped.items())}


def frozen_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize frozen probes without treating an omitted audit as success.

    ``checkpoints`` is the canonical schema.  Completed summaries also retain
    checkpoint names at the top level for compatibility with early consumers
    that accessed, for example, ``summary[\"final\"]`` directly.
    """
    if not rows:
        return {
            "definition": "Frozen, non-mutating evaluation of the current active-memory model at scheduled checkpoints.",
            "status": "not_run",
            "n_checkpoints": 0,
            "n_rows": 0,
            "checkpoints": {},
        }
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("checkpoint", "unknown"))].append(row)
    checkpoints = {
        checkpoint: {
            "n_pairs": float(len(vals)),
            "top1": _mean(row.get("top1") for row in vals),
            "topk": _mean(row.get("topk") for row in vals),
            "human_correction_rate": _mean(row.get("human_correction_rate") for row in vals),
        }
        for checkpoint, vals in sorted(grouped.items())
    }
    return {
        "definition": "Frozen, non-mutating evaluation of the current active-memory model at scheduled checkpoints.",
        "status": "completed",
        "n_checkpoints": len(checkpoints),
        "n_rows": len(rows),
        "checkpoints": checkpoints,
        # Backward-compatible direct checkpoint access.
        **checkpoints,
    }


def policy_calibration_summary(turn_rows: Sequence[Mapping[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    """Calibration and arbitration diagnostics from actual robot decisions.

    This is a diagnostic of the deployed policy, not a post-hoc threshold
    selector.  Its input is limited to turn metadata emitted before the human
    correction is observed.
    """
    rows = [
        row for row in turn_rows
        if row.get("turn_kind") == "robot"
        and isinstance(row.get("final_action_confidence"), (int, float))
        and isinstance(row.get("correct_top1"), bool)
    ]
    if not rows:
        return {
            "definition": "Top-1 confidence calibration and structural/local expert arbitration on robot turns.",
            "status": "not_run",
            "n_robot_turns": 0,
            "ece": None,
            "top1_brier": None,
            "expert_disagreement_count": 0,
        }
    bins: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        confidence = max(0.0, min(1.0, float(row["final_action_confidence"])))
        idx = min(max(0, int(n_bins) - 1), int(confidence * max(1, int(n_bins))))
        bins[idx].append(row)
    ece = 0.0
    bin_rows: List[Dict[str, Any]] = []
    for idx in range(max(1, int(n_bins))):
        vals = bins.get(idx, [])
        if not vals:
            continue
        confidence = _mean(float(row["final_action_confidence"]) for row in vals)
        accuracy = _mean(1.0 if row["correct_top1"] else 0.0 for row in vals)
        ece += (len(vals) / len(rows)) * abs(accuracy - confidence)
        bin_rows.append({"bin": idx, "n": len(vals), "mean_confidence": confidence, "accuracy": accuracy})
    brier = _mean(
        (float(row["final_action_confidence"]) - (1.0 if row["correct_top1"] else 0.0)) ** 2
        for row in rows
    )
    disagreements = [
        row for row in rows
        if row.get("conditioned_top_token") is not None
        and row.get("ensemble_top_token") is not None
        and row.get("conditioned_top_token") != row.get("ensemble_top_token")
    ]
    return {
        "definition": "Top-1 confidence calibration and structural/local expert arbitration on robot turns.",
        "status": "completed",
        "n_robot_turns": len(rows),
        "ece": float(ece),
        "top1_brier": float(brier),
        "bins": bin_rows,
        "expert_disagreement_count": len(disagreements),
        "expert_disagreement_rate": _safe_div(len(disagreements), len(rows)),
        "accuracy_when_experts_disagree": _mean(1.0 if row["correct_top1"] else 0.0 for row in disagreements),
        "mean_structural_blend_when_disagree": _mean(row.get("blend_strength") for row in disagreements),
    }


def axis_value_transfer_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    defaults = PRESET_PREFERENCES["identity"].as_dict()
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("mode") != "assist" or not isinstance(row.get("preference_axis_values"), Mapping):
            continue
        same = set(row.get("axis_value_seen_same_recipe_labels_before", []) or [])
        other = set(row.get("axis_value_seen_other_recipe_labels_before", []) or [])
        global_seen = set(row.get("axis_value_seen_global_labels_before", []) or [])
        axis_values = row["preference_axis_values"]
        for axis, value in sorted((str(k), str(v)) for k, v in axis_values.items()):
            if value == defaults.get(axis):
                continue
            label = f"{axis}={value}"
            if label in same:
                cell = "same_recipe_axis_value_reuse"
            elif label in other:
                cell = "cross_recipe_axis_value_transfer"
            elif label in global_seen:
                cell = "axis_value_seen_global"
            else:
                cell = "new_axis_value"
            out.append({
                "baseline": row.get("baseline"),
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "event_idx": row.get("event_idx"),
                "pair": row.get("pair"),
                "recipe": row.get("recipe"),
                "preference": row.get("preference"),
                "axis": axis,
                "axis_value": value,
                "axis_value_label": label,
                "axis_transfer_cell": cell,
                "preference_is_composed": bool(row.get("preference_is_composed")),
                "transfer_cell_before": row.get("transfer_cell_before"),
                "live_top1": _numeric(row, "live_top1"),
                "live_topk": _numeric(row, "live_topk"),
                "human_correction_rate": _numeric(row, "human_correction_rate"),
                "robot_wrong_rate": _numeric(row, "robot_wrong_rate"),
                "hrc_robot_turn_count": _numeric(row, "hrc_robot_turn_count"),
                "hrc_human_correction_count": _numeric(row, "hrc_human_correction_count"),
                "hrc_robot_correct_count": _numeric(row, "hrc_robot_correct_count"),
                "hrc_robot_wrong_count": _numeric(row, "hrc_robot_wrong_count"),
                "hrc_robot_topk_hit_count": _numeric(row, "hrc_robot_topk_hit_count"),
            })
    return out


def axis_value_transfer_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    axis_rows = axis_value_transfer_rows(rows)
    grouped_axis: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    grouped_value: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    grouped_cell: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in axis_rows:
        grouped_axis[str(row.get("axis"))].append(row)
        grouped_value[str(row.get("axis_value_label"))].append(row)
        grouped_cell[str(row.get("axis_transfer_cell"))].append(row)
    return {
        "definition": "Per-axis-value transfer diagnostics; composed preferences contribute one row per non-default axis.",
        "n_axis_value_rows": len(axis_rows),
        "by_axis": {k: aggregate_episode_metrics(v) for k, v in sorted(grouped_axis.items())},
        "by_axis_value": {k: aggregate_episode_metrics(v) for k, v in sorted(grouped_value.items())},
        "by_axis_transfer_cell": {k: aggregate_episode_metrics(v) for k, v in sorted(grouped_cell.items())},
    }


def prototype_stability_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "definition": "Latent preference prototype rebuild stability; splits/merges indicate semantic ID instability.",
        "n_events": len(rows),
        "n_split_events": sum(1 for row in rows if _numeric(row, "prototype_split_count") > 0),
        "n_merge_events": sum(1 for row in rows if _numeric(row, "prototype_merge_count") > 0),
        "mean_split_count": _mean(row.get("prototype_split_count") for row in rows),
        "mean_merge_count": _mean(row.get("prototype_merge_count") for row in rows),
        "mean_prototype_stability_ari": _mean(row.get("prototype_stability_ari") for row in rows),
        "mean_new_count": _mean(row.get("prototype_new_count") for row in rows),
        "mean_retired_count": _mean(row.get("prototype_retired_count") for row in rows),
    }


def active_only_audit_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "definition": "Checks whether live prediction matches an active-only memory view after pruning.",
            "status": "not_run",
            "n_audits": 0,
            "n_available": 0,
            "n_primary_contract_failed": None,
            "primary_contract_failure_rate": None,
            "live_prediction_max_l1": None,
            "active_head_max_l1": None,
            "mean_active_variants": None,
            "mean_pruned_variants": None,
            "failed_checkpoints": [],
        }
    available = [row for row in rows if row.get("audit_available")]
    failed = [row for row in available if row.get("primary_active_only_contract_passed") is False]
    return {
        "definition": "Checks whether live prediction matches an active-only memory view after pruning.",
        "status": "completed" if available else "unavailable",
        "n_audits": len(rows),
        "n_available": len(available),
        "n_primary_contract_failed": len(failed),
        "primary_contract_failure_rate": _safe_div(len(failed), len(available)),
        "live_prediction_max_l1": max(_finite(row.get("live_prediction_max_l1") for row in available), default=0.0),
        "active_head_max_l1": max(_finite(row.get("active_head_max_l1") for row in available), default=0.0),
        "mean_active_variants": _mean(row.get("active_variants") for row in rows),
        "mean_pruned_variants": _mean(row.get("pruned_variants") for row in rows),
        "failed_checkpoints": [
            {"event_idx": row.get("event_idx"), "audit_checkpoint": row.get("audit_checkpoint")}
            for row in failed
        ],
    }


def _adjusted_rand_index_labels(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    if len(labels_a) != len(labels_b):
        raise ValueError("ARI inputs must have equal length")
    n = len(labels_a)
    if n < 2:
        return None

    def comb2(value: int) -> float:
        return float(value * (value - 1) / 2)

    counts_a: Counter = Counter(labels_a)
    counts_b: Counter = Counter(labels_b)
    table: Counter = Counter(zip(labels_a, labels_b))
    observed = sum(comb2(count) for count in table.values())
    expected = sum(comb2(count) for count in counts_a.values()) * sum(comb2(count) for count in counts_b.values()) / comb2(n)
    maximum = 0.5 * (sum(comb2(count) for count in counts_a.values()) + sum(comb2(count) for count in counts_b.values()))
    denom = maximum - expected
    return 1.0 if abs(denom) <= 1e-12 else max(-1.0, min(1.0, (observed - expected) / denom))


def preference_prototype_semantics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labelled = [
        row for row in rows
        if row.get("committed_latent_pref_id") is not None and row.get("preference") is not None
    ]
    if not labelled:
        return {
            "definition": "Agreement of learned latent preference IDs with evaluator-only preference labels; reported as a diagnostic, not supplied to the agent.",
            "n_labelled_episodes": 0,
            "adjusted_rand_index": None,
            "cluster_purity": None,
            "preference_identity_recovery_rate": None,
            "coverage": 0.0,
        }
    labels = [str(row["preference"]) for row in labelled]
    prototype_ids = [str(row["committed_latent_pref_id"]) for row in labelled]
    by_prototype: Dict[str, List[str]] = defaultdict(list)
    by_preference: Dict[str, List[str]] = defaultdict(list)
    for label, prototype_id in zip(labels, prototype_ids):
        by_prototype[prototype_id].append(label)
        by_preference[label].append(prototype_id)
    purity_hits = sum(max(Counter(values).values()) for values in by_prototype.values())
    recovery_eligible = 0
    recovery_hits = 0
    for values in by_preference.values():
        if len(values) < 2:
            continue
        reference = values[0]
        recovery_eligible += len(values) - 1
        recovery_hits += sum(value == reference for value in values[1:])
    return {
        "definition": "Agreement of learned latent preference IDs with evaluator-only preference labels; reported as a diagnostic, not supplied to the agent.",
        "n_labelled_episodes": len(labelled),
        "n_ground_truth_preferences": len(by_preference),
        "n_learned_prototypes": len(by_prototype),
        "adjusted_rand_index": _adjusted_rand_index_labels(labels, prototype_ids),
        "cluster_purity": _safe_div(purity_hits, len(labelled)),
        "preference_identity_recovery_rate": _safe_div(recovery_hits, recovery_eligible) if recovery_eligible else None,
        "preference_identity_recovery_eligible": recovery_eligible,
        "coverage": _safe_div(len(labelled), len(rows)),
    }


def online_commit_safety_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    committed = [row for row in rows if row.get("mode") == "assist" and row.get("commit_attempted")]

    def rate(key: str) -> Optional[float]:
        eligible = [row for row in committed if row.get(key) is not None]
        return _mean(1.0 if bool(row.get(key)) else 0.0 for row in eligible) if eligible else None

    return {
        "definition": "Evaluator-side correctness of online session commits; false-promotion rates are reported only where a ground-truth variant is identifiable.",
        "n_committed_assist_episodes": len(committed),
        "recipe_commit_accuracy": rate("commit_recipe_correct"),
        "variant_commit_accuracy": rate("commit_variant_correct"),
        "false_variant_creation_rate": _mean(1.0 if row.get("false_variant_creation") else 0.0 for row in committed) if committed else None,
        "false_latest_promotion_rate": _mean(1.0 if row.get("false_latest_promotion") else 0.0 for row in committed) if committed else None,
        "needs_observation_after_assist_rate": _mean(1.0 if row.get("classification_kind") == "needs_observation" else 0.0 for row in committed) if committed else None,
    }


def reentry_stratification_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scheduled = [row for row in rows if row.get("scheduled_reentry_probe")]
    scheduled_assist = [row for row in scheduled if row.get("mode") == "assist"]
    pruned_target = [
        row for row in scheduled
        if row.get("reentry_probe_target_state_before") == "pruned_exact_variant"
    ]
    active_target = [
        row for row in scheduled
        if row.get("reentry_probe_target_state_before") == "active_exact_variant"
    ]
    confirmed = [row for row in scheduled_assist if row.get("actual_reentry_from_pruned")]
    return {
        "definition": "Reentry performance is stratified by evaluator-observed pre-probe exact-variant state. Only confirmed pruned-variant reentries support the selective-forgetting recovery claim; active variants are an explicit retention control.",
        "scheduled_reentry_all_routes": aggregate_episode_metrics(scheduled),
        "scheduled_reentry_assist_only": aggregate_episode_metrics(scheduled_assist),
        "target_pruned_before_probe": aggregate_episode_metrics(pruned_target),
        "target_active_before_probe_control": aggregate_episode_metrics(active_target),
        "confirmed_reentry_from_pruned": aggregate_episode_metrics(confirmed),
        "n_scheduled_reentry_probes": len(scheduled),
        "n_scheduled_reentry_assist_probes": len(scheduled_assist),
        "n_target_pruned_before_probe": len(pruned_target),
        "n_target_active_before_probe_control": len(active_target),
        "n_confirmed_reentry_from_pruned": len(confirmed),
    }


ORACLE_HIGHER_IS_BETTER = {
    "live_top1",
    "live_topk",
    "primary_prediction_metric_value",
    "primary_adaptation_recovery_top1",
}
ORACLE_LOWER_IS_BETTER = {
    "human_correction_rate",
    "robot_wrong_rate",
    "mean_nll_per_robot_turn",
    "testing_total_action_time",
    "testing_normalized_interaction_cost",
}


def _oracle_row(
    baseline: str,
    oracle_reference: str,
    scope: str,
    group: str,
    metric: str,
    baseline_value: Any,
    oracle_value: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(baseline_value, (int, float)) or not isinstance(oracle_value, (int, float)):
        return None
    b = float(baseline_value)
    o = float(oracle_value)
    if not math.isfinite(b) or not math.isfinite(o):
        return None
    if metric in ORACLE_HIGHER_IS_BETTER:
        advantage = o - b
        direction = "higher_is_better"
    elif metric in ORACLE_LOWER_IS_BETTER:
        advantage = b - o
        direction = "lower_is_better"
    else:
        return None
    return {
        "baseline": baseline,
        "oracle_reference": oracle_reference,
        "scope": scope,
        "group": group,
        "metric": metric,
        "metric_direction": direction,
        "baseline_value": b,
        "oracle_value": o,
        "oracle_advantage": advantage,
        "regret_to_clairvoyant": max(0.0, advantage),
    }


def oracle_gap_rows(per_baseline: Mapping[str, Any], oracle_reference: str = CLAIRVOYANT_MEMORY_ORACLE) -> List[Dict[str, Any]]:
    oracle = per_baseline.get(oracle_reference)
    if not isinstance(oracle, Mapping):
        return []
    rows: List[Dict[str, Any]] = []
    metrics = tuple(sorted(ORACLE_HIGHER_IS_BETTER | ORACLE_LOWER_IS_BETTER))

    def add(baseline: str, scope: str, group: str, base_view: Mapping[str, Any], oracle_view: Mapping[str, Any]) -> None:
        for metric in metrics:
            row = _oracle_row(baseline, oracle_reference, scope, group, metric, base_view.get(metric), oracle_view.get(metric))
            if row is not None:
                rows.append(row)

    for baseline, summary in sorted(per_baseline.items()):
        if baseline == oracle_reference or not isinstance(summary, Mapping):
            continue
        for scope in ("assist_only", "all_episode_workload"):
            if isinstance(summary.get(scope), Mapping) and isinstance(oracle.get(scope), Mapping):
                add(baseline, scope, "all", summary[scope], oracle[scope])
        for scope in ("per_hypothesis", "per_transfer_cell", "per_memory_state", "per_event_type"):
            base_groups = summary.get(scope, {})
            oracle_groups = oracle.get(scope, {})
            if not isinstance(base_groups, Mapping) or not isinstance(oracle_groups, Mapping):
                continue
            for group in sorted(set(base_groups) & set(oracle_groups)):
                add(baseline, scope, str(group), base_groups[group], oracle_groups[group])
        base_axis = (summary.get("axis_value_transfer", {}) or {}).get("by_axis_transfer_cell", {})
        oracle_axis = (oracle.get("axis_value_transfer", {}) or {}).get("by_axis_transfer_cell", {})
        if isinstance(base_axis, Mapping) and isinstance(oracle_axis, Mapping):
            for group in sorted(set(base_axis) & set(oracle_axis)):
                add(baseline, "axis_value_transfer.by_axis_transfer_cell", str(group), base_axis[group], oracle_axis[group])
    return rows


def oracle_gap_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("baseline", "unknown")), str(row.get("metric", "unknown")))].append(row)
    references = sorted({str(row.get("oracle_reference")) for row in rows if row.get("oracle_reference")})
    return {
        "definition": "Positive oracle_advantage means the future-aware memory-pruning reference is better.",
        "oracle_reference": references[0] if len(references) == 1 else references,
        "n_rows": len(rows),
        "by_baseline": {
            baseline: {
                "by_metric": {
                    metric: {
                        "n_rows": len(vals),
                        "mean_oracle_advantage": _mean(row.get("oracle_advantage") for row in vals),
                        "mean_regret_to_clairvoyant": _mean(row.get("regret_to_clairvoyant") for row in vals),
                        "max_regret_to_clairvoyant": max(_finite(row.get("regret_to_clairvoyant") for row in vals), default=0.0),
                        "by_scope": {
                            scope: {
                                "n_rows": len(scope_rows),
                                "mean_oracle_advantage": _mean(row.get("oracle_advantage") for row in scope_rows),
                                "mean_regret_to_clairvoyant": _mean(row.get("regret_to_clairvoyant") for row in scope_rows),
                            }
                            for scope, scope_rows in sorted(
                                (
                                    (scope, [row for row in vals if str(row.get("scope")) == scope])
                                    for scope in {str(row.get("scope")) for row in vals}
                                ),
                                key=lambda item: item[0],
                            )
                        },
                    }
                    for (candidate_baseline, metric), vals in sorted(grouped.items())
                    if candidate_baseline == baseline
                },
            }
            for baseline in sorted({baseline for baseline, _metric in grouped})
        },
    }


def summarize_stream(stream: EventStreamRun) -> Dict[str, Any]:
    assist_rows = [row for row in stream.episode_rows if row.get("mode") == "assist"]
    reentry = reentry_stratification_summary(stream.episode_rows)
    metrics = {
        "baseline": stream.baseline,
        "scenario": stream.scenario,
        "seed": int(stream.seed),
        "assist_only": aggregate_episode_metrics(assist_rows),
        "all_episode_workload": {
            **aggregate_episode_metrics(stream.episode_rows),
            "scope_note": "Workload includes observation episodes; prediction rates are pooled over robot turns only.",
        },
        "per_event_type": _group_metrics(assist_rows, "event_type"),
        "per_condition": _group_metrics(assist_rows, "condition"),
        "per_hypothesis": _group_metrics(assist_rows, "hypothesis_tags"),
        "per_event_type_all_episodes": _group_metrics(stream.episode_rows, "event_type"),
        "per_hypothesis_all_episodes": _group_metrics(stream.episode_rows, "hypothesis_tags"),
        "per_transfer_cell": _group_metrics(assist_rows, "transfer_cell_before"),
        "per_memory_state": _group_metrics(assist_rows, "memory_state_gt"),
        "per_rung_effectiveness": _group_metrics(assist_rows, "no_op_or_duplicate_rung"),
        "per_rung": _group_metrics(stream.episode_rows, "rung_idx"),
        "frozen_eval": frozen_summary(stream.frozen_rows),
        "policy_calibration": policy_calibration_summary(stream.turn_rows),
        "memory": memory_snapshot(stream.agent),
        "compute": memory_snapshot(stream.agent, stream.wall_s),
        "axis_value_transfer": axis_value_transfer_summary(assist_rows),
        "prototype_stability": prototype_stability_summary(stream.prototype_rows),
        "active_only_pruned_influence_audit": active_only_audit_summary(stream.active_audit_rows),
        "online_commit_safety": online_commit_safety_summary(stream.episode_rows),
        "preference_prototype_semantics": preference_prototype_semantics(stream.episode_rows),
        "reentry_stratification": reentry,
    }
    metrics["paper_hypothesis_views"] = {
        "known_recipe_new_preference_adaptation": metrics["per_hypothesis"].get("known_recipe_new_preference_adaptation", {}),
        "cross_recipe_transfer": metrics["per_hypothesis"].get("cross_recipe_transfer", {}),
        "emergent_axis_composition": metrics["per_hypothesis"].get("emergent_axis_composition", {}),
        "selective_forgetting_reentry": metrics["reentry_stratification"]["confirmed_reentry_from_pruned"],
        "direct_retrieval_control": metrics["per_hypothesis"].get("direct_retrieval_control", {}),
    }
    return metrics


def _support_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "modes": dict(Counter(str(row.get("mode", "unknown")) for row in rows)),
        "event_types": dict(Counter(str(row.get("event_type", "unknown")) for row in rows)),
        "transfer_cell": dict(Counter(str(row.get("transfer_cell_before", "unknown")) for row in rows)),
        "axis_transfer_cell": dict(Counter(str(row.get("axis_transfer_cell_before", "unknown")) for row in rows)),
        "no_op_or_duplicate_rung": dict(Counter(str(row.get("no_op_or_duplicate_rung", "unknown")) for row in rows)),
    }


def run_plan(plan: ScenarioPlan, config: EvaluationConfig, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    _write_json(out_dir / "scenario_plan.json", {
        "scenario": plan.scenario,
        "seed": plan.seed,
        "description": plan.description,
        "selected_recipes": list(plan.selected_recipes),
        "selected_preferences": list(plan.selected_preferences),
        "n_events": len(plan.events),
        "n_eval_pairs": len(plan.eval_pairs),
        "events": [
            {"idx": idx, "mode": event.mode, "pair": event.pair.label, **dict(event.tags)}
            for idx, event in enumerate(plan.events)
        ],
    })
    _write_json(out_dir / "evaluation_config.json", asdict(config))

    per_baseline: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_frozen_rows: List[Dict[str, Any]] = []
    all_diagnostic_rows: List[Dict[str, Any]] = []
    all_turn_rows: List[Dict[str, Any]] = []

    baselines = [b for b in config.baselines if b != CLAIRVOYANT_MEMORY_ORACLE]
    for baseline in baselines:
        stream = run_event_stream_for_baseline(baseline, plan, config)
        per_baseline[baseline] = summarize_stream(stream)
        all_episode_rows.extend(stream.episode_rows)
        all_frozen_rows.extend(stream.frozen_rows)
        all_diagnostic_rows.extend(stream.memory_rows + stream.prototype_rows + stream.active_audit_rows + stream.oracle_pruning_rows)
        all_turn_rows.extend(stream.turn_rows)

    oracle_summary: Optional[Dict[str, Any]] = None
    if config.include_clairvoyant_oracle:
        stream = run_event_stream_for_baseline(CLAIRVOYANT_MEMORY_ORACLE, plan, config)
        oracle_summary = summarize_stream(stream)
        oracle_summary.update({
            "oracle_reference": CLAIRVOYANT_MEMORY_ORACLE,
            "reported_as": CLAIRVOYANT_REFERENCE_TAG,
            "leakage_warning": CLAIRVOYANT_LEAKAGE_WARNING,
            "oracle_retention_policy": "future_recipe_support",
            "clairvoyant_pruning": {
                "n_prune_decisions": len(stream.oracle_pruning_rows),
                "total_pruned_active_variants": int(sum(_numeric(row, "oracle_pruned_active_variants") for row in stream.oracle_pruning_rows)),
            },
        })
        per_baseline[CLAIRVOYANT_MEMORY_ORACLE] = oracle_summary
        all_episode_rows.extend(stream.episode_rows)
        all_frozen_rows.extend(stream.frozen_rows)
        all_diagnostic_rows.extend(stream.memory_rows + stream.prototype_rows + stream.active_audit_rows + stream.oracle_pruning_rows)
        all_turn_rows.extend(stream.turn_rows)

    axis_rows = axis_value_transfer_rows(all_episode_rows)
    oracle_rows = oracle_gap_rows(per_baseline)
    _append_jsonl(out_dir / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out_dir / "turn_metrics.jsonl", all_turn_rows)
    _append_jsonl(out_dir / "frozen_eval.jsonl", all_frozen_rows)
    _append_jsonl(out_dir / "diagnostics.jsonl", all_diagnostic_rows)
    _append_jsonl(out_dir / "axis_value_transfer_rows.jsonl", axis_rows)
    _append_jsonl(out_dir / "oracle_gap_rows.jsonl", oracle_rows)

    support_source = [row for row in all_episode_rows if baselines and row.get("baseline") == baselines[0]]
    if not support_source:
        support_source = [row for row in all_episode_rows if row.get("baseline") == CLAIRVOYANT_MEMORY_ORACLE]

    summary = {
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "description": plan.description,
        "n_events": len(plan.events),
        "n_eval_pairs": len(plan.eval_pairs),
        "selected_recipes": list(plan.selected_recipes),
        "selected_preferences": list(plan.selected_preferences),
        "support_counts": _support_counts(support_source),
        "n_turn_metric_rows": len(all_turn_rows),
        "per_baseline": per_baseline,
        "oracle_reference": oracle_summary,
        "axis_value_transfer_summary": axis_value_transfer_summary(all_episode_rows),
        "prototype_stability_summary": prototype_stability_summary([r for r in all_diagnostic_rows if r.get("diagnostic_type") == "prototype_stability"]),
        "active_only_pruned_influence_audit_summary": active_only_audit_summary([r for r in all_diagnostic_rows if r.get("diagnostic_type") == "active_only_pruned_influence_audit"]),
        "oracle_gap_summary": oracle_gap_summary(oracle_rows),
        "wall_s": float(time.perf_counter() - t0),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _mp_context() -> mp.context.BaseContext:
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")


def _worker_count(config: EvaluationConfig) -> int:
    n_seeds = max(1, len(config.seeds))
    return max(1, min(int(config.workers or n_seeds), n_seeds))


def _native_thread_count(config: EvaluationConfig) -> int:
    return _positive_int(getattr(config, "native_threads_per_worker", DEFAULT_NATIVE_THREADS_PER_WORKER))


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "unknown"
    seconds_i = int(round(float(seconds)))
    if seconds_i >= 3600:
        h, rem = divmod(seconds_i, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"
    if seconds_i >= 60:
        m, s = divmod(seconds_i, 60)
        return f"{m}m{s:02d}s"
    return f"{seconds_i}s"


def _eta(suite_t0: float, scenario_t0: float, scenario_idx: int, n_scenarios: int, done: int, n_seeds: int, past_scenario_wall: Sequence[float]) -> str:
    now = time.perf_counter()
    elapsed = now - suite_t0
    current_elapsed = now - scenario_t0
    if done > 0:
        current_remaining = max(0.0, current_elapsed * (n_seeds - done) / max(1, done))
    else:
        current_remaining = (sum(past_scenario_wall) / len(past_scenario_wall)) if past_scenario_wall else None
    future_est = (sum(past_scenario_wall) / len(past_scenario_wall)) if past_scenario_wall else None
    future_remaining = None if future_est is None else future_est * max(0, n_scenarios - scenario_idx - 1)
    total_remaining = None if current_remaining is None and future_remaining is None else float(current_remaining or 0.0) + float(future_remaining or 0.0)
    return f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(total_remaining)}"


def _run_seed_scenario_job(scenario: str, seed: int, config: EvaluationConfig) -> Dict[str, Any]:
    _apply_native_thread_limit(_native_thread_count(config))
    plan = build_scenario_plan(scenario, config, int(seed))
    out_dir = Path(config.output_dir) / scenario / f"seed_{int(seed)}"
    t0 = time.perf_counter()
    summary = run_plan(plan, config, out_dir)
    return {
        "scenario": scenario,
        "seed": int(seed),
        "key": f"{scenario}/seed_{int(seed)}",
        "summary": summary,
        "wall_s": float(time.perf_counter() - t0),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def _paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    seed: int,
    n_samples: int = PAIRED_BOOTSTRAP_SAMPLES,
) -> Dict[str, float]:
    """Deterministic percentile CI over paired per-seed differences."""
    values = [float(value) for value in deltas if math.isfinite(float(value))]
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "bootstrap_full_better_fraction": 0.0}
    rng = random.Random(int(seed))
    n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(max(1, int(n_samples)))]
    return {
        "mean": _mean(values),
        "ci95_low": _quantile(draws, 0.025),
        "ci95_high": _quantile(draws, 0.975),
        "bootstrap_full_better_fraction": _mean(value > 0.0 for value in draws),
    }


def paired_bootstrap_summary(
    scenario_summaries: Mapping[str, Mapping[str, Any]],
    *,
    n_samples: int = PAIRED_BOOTSTRAP_SAMPLES,
) -> Dict[str, Any]:
    """Compare the full agent with each deployable baseline on matched seeds.

    Positive deltas always favor ``full``.  The output is a confidence
    interval, not a p-value: with five seeds it communicates uncertainty but
    should not be over-interpreted as a decisive significance test.
    """
    by_scenario: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for summary in scenario_summaries.values():
        scenario = summary.get("scenario")
        seed = summary.get("seed")
        if isinstance(scenario, str) and isinstance(seed, int):
            by_scenario[scenario][int(seed)] = summary

    metric_specs = (
        ("live_top1", "higher_is_better"),
        ("human_correction_rate", "lower_is_better"),
        ("testing_normalized_interaction_cost", "lower_is_better"),
    )
    results: Dict[str, Any] = {}
    for scenario, by_seed in sorted(by_scenario.items()):
        all_baselines = {
            str(name)
            for summary in by_seed.values()
            for name in (summary.get("per_baseline") or {})
        }
        comparisons: Dict[str, Any] = {}
        for baseline in sorted(all_baselines - {"full", CLAIRVOYANT_MEMORY_ORACLE}):
            by_metric: Dict[str, Any] = {}
            for metric, direction in metric_specs:
                paired: List[Tuple[int, float]] = []
                for seed, summary in sorted(by_seed.items()):
                    per_baseline = summary.get("per_baseline") or {}
                    full_metrics = (per_baseline.get("full") or {}).get("assist_only") or {}
                    baseline_metrics = (per_baseline.get(baseline) or {}).get("assist_only") or {}
                    full_value = full_metrics.get(metric)
                    baseline_value = baseline_metrics.get(metric)
                    if not isinstance(full_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                        continue
                    if not math.isfinite(float(full_value)) or not math.isfinite(float(baseline_value)):
                        continue
                    delta = float(full_value) - float(baseline_value)
                    if direction == "lower_is_better":
                        delta = -delta
                    paired.append((int(seed), delta))
                stable_seed = int.from_bytes(
                    hashlib.sha256(f"{scenario}|{baseline}|{metric}".encode("utf-8")).digest()[:8],
                    byteorder="big",
                )
                interval = _paired_bootstrap_ci(
                    [delta for _seed, delta in paired],
                    seed=stable_seed,
                    n_samples=n_samples,
                )
                by_metric[metric] = {
                    "direction": direction,
                    "n_paired_seeds": len(paired),
                    "paired_seed_deltas_full_advantage": {str(seed): delta for seed, delta in paired},
                    **interval,
                }
            comparisons[baseline] = by_metric
        results[scenario] = {
            "n_available_seeds": len(by_seed),
            "comparisons_against_full": comparisons,
        }
    return {
        "definition": "Paired, non-parametric bootstrap percentile intervals over full-minus-baseline seed-matched differences. Positive values favor the full agent; bootstrap_full_better_fraction is a direction frequency, not a hypothesis-test p-value.",
        "bootstrap_samples": int(n_samples),
        "by_scenario": results,
    }


def _commit_sensitivity_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scenario")), str(row.get("sensitivity_setting")))].append(row)
    return {
        "definition": "One-factor-at-a-time sensitivity analysis of the online self-training confidence rule. Each setting is evaluated on the same scenario plans and seeds as the main full-agent run.",
        "settings": [
            {
                "scenario": scenario,
                "sensitivity_setting": setting,
                "n_seeds": len(group),
                "mean_live_top1": _mean(row.get("live_top1") for row in group),
                "mean_human_correction_rate": _mean(row.get("human_correction_rate") for row in group),
                "mean_normalized_interaction_cost": _mean(row.get("testing_normalized_interaction_cost") for row in group),
                "mean_recipe_commit_accuracy": _mean(row.get("recipe_commit_accuracy") for row in group),
                "mean_variant_commit_accuracy": _mean(row.get("variant_commit_accuracy") for row in group),
            }
            for (scenario, setting), group in sorted(grouped.items())
        ],
    }


def run_commit_sensitivity(config: EvaluationConfig) -> Dict[str, Any]:
    """Run the H3 audit only when explicitly enabled by the caller.

    This is deliberately separate from the main suite because it multiplies
    the full-agent workload by the number of sensitivity settings.
    """
    rows: List[Dict[str, Any]] = []
    for scenario in config.scenarios:
        for seed in config.seeds:
            plan = build_scenario_plan(scenario, config, int(seed))
            for setting, overrides in COMMIT_SENSITIVITY_SPECS:
                sensitivity_config = replace(
                    config,
                    model_overrides={**dict(config.model_overrides), **dict(overrides)},
                    run_commit_sensitivity=False,
                )
                stream = run_event_stream_for_baseline("full", plan, sensitivity_config)
                summary = summarize_stream(stream)
                assist = summary["assist_only"]
                safety = summary["online_commit_safety"]
                rows.append({
                    "scenario": scenario,
                    "seed": int(seed),
                    "baseline": "full",
                    "sensitivity_setting": setting,
                    "model_overrides": dict(overrides),
                    "live_top1": assist.get("live_top1"),
                    "live_topk": assist.get("live_topk"),
                    "human_correction_rate": assist.get("human_correction_rate"),
                    "testing_normalized_interaction_cost": assist.get("testing_normalized_interaction_cost"),
                    "recipe_commit_accuracy": safety.get("recipe_commit_accuracy"),
                    "variant_commit_accuracy": safety.get("variant_commit_accuracy"),
                    "false_variant_creation_rate": safety.get("false_variant_creation_rate"),
                    "false_latest_promotion_rate": safety.get("false_latest_promotion_rate"),
                })
    return {
        "specification": [
            {"sensitivity_setting": setting, "model_overrides": dict(overrides)}
            for setting, overrides in COMMIT_SENSITIVITY_SPECS
        ],
        "rows": rows,
        "summary": _commit_sensitivity_summary(rows),
    }


def run_suite(config: EvaluationConfig) -> Dict[str, Any]:
    workers = _worker_count(config)
    native_threads = _native_thread_count(config)
    thread_control = _apply_native_thread_limit(native_threads)
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    suite: Dict[str, Any] = {
        "config": asdict(config),
        "execution": {
            "parallelism": "processes",
            "seed_parallelism": "all seeds for one scenario complete before the next scenario starts",
            "workers": workers,
            "native_threads_per_worker": native_threads,
            "estimated_max_native_threads": workers * native_threads,
            "thread_control": thread_control,
            "clairvoyant_oracle_included": bool(config.include_clairvoyant_oracle),
        },
        "scenarios": {},
    }
    suite_t0 = time.perf_counter()
    past_scenario_wall: List[float] = []

    for scenario_idx, scenario in enumerate(config.scenarios):
        scenario_t0 = time.perf_counter()
        if config.print_eta:
            print(
                f"[evaluation] start scenario={scenario} seeds={len(config.seeds)} workers={workers} "
                f"{_eta(suite_t0, scenario_t0, scenario_idx, len(config.scenarios), 0, len(config.seeds), past_scenario_wall)}",
                flush=True,
            )
        if workers <= 1 or len(config.seeds) <= 1:
            for done, seed in enumerate(config.seeds, start=1):
                result = _run_seed_scenario_job(scenario, int(seed), config)
                suite["scenarios"][result["key"]] = result["summary"]
                if config.print_eta:
                    print(
                        f"[evaluation] done scenario={scenario} seed={seed} seed_wall={_format_seconds(result['wall_s'])} "
                        f"{_eta(suite_t0, scenario_t0, scenario_idx, len(config.scenarios), done, len(config.seeds), past_scenario_wall)}",
                        flush=True,
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context()) as executor:
                futures = {
                    executor.submit(_run_seed_scenario_job, scenario, int(seed), config): int(seed)
                    for seed in config.seeds
                }
                done = 0
                for future in as_completed(futures):
                    seed = futures[future]
                    result = future.result()
                    done += 1
                    suite["scenarios"][result["key"]] = result["summary"]
                    if config.print_eta:
                        print(
                            f"[evaluation] done scenario={scenario} seed={seed} seed_wall={_format_seconds(result['wall_s'])} "
                            f"{_eta(suite_t0, scenario_t0, scenario_idx, len(config.scenarios), done, len(config.seeds), past_scenario_wall)}",
                            flush=True,
                        )
        scenario_wall = time.perf_counter() - scenario_t0
        past_scenario_wall.append(float(scenario_wall))
        suite.setdefault("scenario_wall_s", {})[scenario] = float(scenario_wall)
        if config.print_eta:
            print(
                f"[evaluation] complete scenario={scenario} wall={_format_seconds(scenario_wall)} "
                f"{_eta(suite_t0, scenario_t0, scenario_idx, len(config.scenarios), len(config.seeds), len(config.seeds), past_scenario_wall)}",
                flush=True,
            )
        _write_json(root / "suite_summary.json", suite)

    suite["paired_bootstrap_statistics"] = paired_bootstrap_summary(suite["scenarios"])
    _write_json(root / "paired_bootstrap_statistics.json", suite["paired_bootstrap_statistics"])
    if config.run_commit_sensitivity:
        sensitivity = run_commit_sensitivity(config)
        suite["commit_sensitivity"] = sensitivity["summary"]
        _write_json(root / "commit_sensitivity.json", sensitivity)
    suite["wall_s"] = float(time.perf_counter() - suite_t0)
    _write_json(root / "suite_summary.json", suite)
    return suite


def _parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_args(argv: Optional[Sequence[str]] = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Run the adaptive-preference HRC evaluation harness.")
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in PAPER_SEEDS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--native-threads-per-worker", type=int, default=DEFAULT_NATIVE_THREADS_PER_WORKER)
    parser.add_argument("--no-eta", action="store_true")
    parser.add_argument("--no-clairvoyant-oracle", action="store_true")
    parser.add_argument("--n-recipes", type=int, default=6)
    parser.add_argument("--ladder-rungs", type=int, default=5)
    parser.add_argument("--allow-repeated-ladder-orderings", action="store_true")
    parser.add_argument("--strict-distinct-ladder-orderings", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--min-distinct-ladder-orderings", type=int, default=2)
    parser.add_argument("--deployment-events", type=int, default=80)
    parser.add_argument("--frozen-eval-period", type=int, default=4)
    parser.add_argument("--frozen-eval-max-pairs", type=int, default=48)
    parser.add_argument("--active-only-audit-period", type=int, default=2)
    parser.add_argument("--active-only-audit-max-prefixes", type=int, default=16)
    parser.add_argument("--active-only-audit-tolerance", type=float, default=5e-2)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--commit-sensitivity", action="store_true", help="Run the opt-in H3 online-commit sensitivity audit after the main suite.")
    parser.add_argument("--fast-smoke", action="store_true")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.fast_smoke:
        overrides = {
            "maxent_iters_cold": 1,
            "maxent_iters_warm": 1,
            "maxent_mc_rollouts": 1,
            "bc_epochs_cold": 1,
            "bc_epochs_warm": 1,
        }
    return EvaluationConfig(
        seeds=tuple(int(seed) for seed in _parse_csv(args.seeds)),
        scenarios=_parse_csv(args.scenarios),
        baselines=_parse_csv(args.baselines),
        output_dir=str(args.output_dir),
        workers=int(args.workers),
        native_threads_per_worker=_positive_int(args.native_threads_per_worker, DEFAULT_NATIVE_THREADS_PER_WORKER),
        print_eta=not bool(args.no_eta),
        include_clairvoyant_oracle=not bool(args.no_clairvoyant_oracle),
        n_recipes=int(args.n_recipes),
        ladder_rungs=int(args.ladder_rungs),
        allow_repeated_ladder_orderings=bool(args.allow_repeated_ladder_orderings) and not bool(args.strict_distinct_ladder_orderings),
        min_distinct_ladder_orderings=int(args.min_distinct_ladder_orderings),
        deployment_events=int(args.deployment_events),
        frozen_eval_period=int(args.frozen_eval_period),
        frozen_eval_max_pairs=int(args.frozen_eval_max_pairs),
        active_only_audit_period=int(args.active_only_audit_period),
        active_only_audit_max_prefixes=int(args.active_only_audit_max_prefixes),
        active_only_audit_tolerance=float(args.active_only_audit_tolerance),
        topk=int(args.topk),
        profile=bool(args.profile),
        run_commit_sensitivity=bool(args.commit_sensitivity),
        model_overrides=overrides,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    summary = run_suite(config)
    print(json.dumps(_jsonable({
        "output_dir": config.output_dir,
        "scenarios": sorted(summary.get("scenarios", {}).keys()),
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
