"""Evaluation harness for the adaptive preference-learning HRC benchmark.

This module keeps one runner, five scenario generators, compact episode-level logging, frozen
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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_NATIVE_THREADS_PER_WORKER = 1
# Fifteen fixed paired replications. The original five seeds are retained;
# the ten-seed extension was drawn once from ``random.Random(20250817)`` and
# stored literally so the paper protocol is independent of Python PRNG
# implementation details.
PAPER_SEEDS = (
    1337, 2024, 7, 9001, 31415,
    977683127, 391693862, 224405947, 1115789944, 206259923,
    369598007, 905883393, 1388713902, 1782798442, 448584643,
)
PAIRED_BOOTSTRAP_SAMPLES = 10_000
# One-factor-at-a-time checks around the deployable online self-training rule.
COMMIT_SENSITIVITY_SPECS: Tuple[Tuple[str, Mapping[str, float]], ...] = (
    ("default", {}),
    ("tentative_threshold_low", {"online_commit_tentative_threshold": 0.30}),
    ("tentative_threshold_high", {"online_commit_tentative_threshold": 0.60}),
    ("full_threshold_low", {"online_commit_full_threshold": 0.60}),
    ("full_threshold_high", {"online_commit_full_threshold": 0.90}),
    ("identity_evidence_low", {"online_commit_identity_weight": 0.30}),
    ("identity_evidence_high", {"online_commit_identity_weight": 0.60}),
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
from .baselines import BASELINE_AGENTS
from .environment import gen
from .hrc_simulation import DEFAULT_HRC_TIMING, run_alternating_hrc_episode
from .memory import VariantKey, clear_all_module_caches, variant_hash
from .models import DEFAULT_CONFIG, Config
from .preferences import PREFERENCE_NAMES, PRESET_PREFERENCES, materialize_with_report
from .representations import observations_from_actions


SCENARIO_LADDER_HETEROGENEOUS = "ladder_heterogeneous"
SCENARIO_LADDER_HOMOGENEOUS = "ladder_homogeneous"
SCENARIO_DEPLOYMENT_RANDOM = "ladder_deployment_random"
SCENARIO_AXIS_HOLDOUT = "axis_holdout"
SCENARIO_PREFERENCE_HOLDOUT = "preference_holdout"
SCENARIOS = (
    SCENARIO_LADDER_HETEROGENEOUS,
    SCENARIO_LADDER_HOMOGENEOUS,
    SCENARIO_DEPLOYMENT_RANDOM,
    SCENARIO_AXIS_HOLDOUT,
    SCENARIO_PREFERENCE_HOLDOUT,
)

DEFAULT_BASELINES = (
    "full",
    "offline_pretrained_frozen",
    "offline_all_recipes_identity_frozen",
    "adaptive_decay",
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
    # Primary 9-rung protocol: identity plus one non-default axis per rung.
    "identity",
    "p1_mise_en_place",
    "p2_equipment_just_in_time",
    "p3_frontload_serving_setup",
    "p4_load_just_in_time",
    "p5_shutdown_late",
    "p6_deferred_cook_start",
    "p7_clean_eager",
    "p8_cleanup_before_serve",
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
    # Fifteen fixed, paired scenario draws are the default paper protocol. The
    # runner still accepts any explicit seed tuple for smoke tests or larger
    # final studies.
    seeds: Tuple[int, ...] = PAPER_SEEDS
    scenarios: Tuple[str, ...] = SCENARIOS
    baselines: Tuple[str, ...] = DEFAULT_BASELINES
    # Used only by ``offline_pretrained_frozen``.  The selected subsets are
    # deterministic within each paired scenario seed; all deployment events
    # then run with a fixed policy and fixed memory.
    offline_pretrained_recipe_fraction: float = 0.50
    offline_pretrained_preference_fraction: float = 0.50
    output_dir: str = "results/evaluation"
    workers: int = 0
    native_threads_per_worker: int = DEFAULT_NATIVE_THREADS_PER_WORKER
    print_eta: bool = True
    include_clairvoyant_oracle: bool = True
    n_recipes: int = 15
    ladder_rungs: int = 9
    ladder_preferences: Tuple[str, ...] = DEFAULT_LADDER_PREFERENCES
    # Each settled block contains a seed-fixed total of 2--2.5 times its
    # climb subgroup size.
    settle_repeat_multiplier_min: float = 2.0
    settle_repeat_multiplier_max: float = 2.5
    # The randomized ladder samples a subgroup independently at each rung;
    # homogeneous and heterogeneous ladders always use every recipe.
    random_ladder_min_recipes_per_rung: int = 3
    # The random deployment ladder includes one mechanism-matched delayed
    # recurrence probe in each eligible rung. The probe is the current
    # (just-updated) recipe--preference pair, selected only after at least two
    # distinct conflicting variants for that recipe have entered history. Its
    # delay is calibrated to the full model's per-recipe grace horizon and
    # pruning rate; it replaces an ordinary settled repeat, so the episode
    # budget is unchanged.
    random_ladder_recurrence_min_prior_conflicts: int = 2
    random_ladder_recurrence_min_intervening_events: int = 4
    route_absent_recipe_assists_to_observe: bool = True
    # The full system's realized observation/assist route is the canonical
    # interaction schedule. Every deployable baseline is evaluated under that
    # same schedule, preventing a weaker memory from buying extra supervision
    # by routing more events to observation mode.
    match_baseline_execution_modes_to_full: bool = True
    # A matched, non-mutating probe immediately before each explicitly tagged
    # primary assist event.  Random deployment extends this to every routed
    # assist event.  This is the comparable prediction metric; live
    # interaction cost still uses the actual routed event.
    pre_event_frozen_probes: bool = True
    # Applies to the heterogeneous ladder only. Homogeneous and holdout
    # ladders use explicit rung boundaries; randomized deployment uses
    # event-local probes rather than repeated full-grid sweeps.
    frozen_eval_period: int = 4
    frozen_eval_max_pairs: int = 48
    active_only_audit_period: int = 2
    active_only_audit_max_prefixes: int = 16
    active_only_audit_tolerance: float = 5e-2
    topk: int = 3
    profile: bool = False
    run_commit_sensitivity: bool = False
    model_overrides: Mapping[str, Any] = field(default_factory=dict)
    # A descriptive label persisted in the existing evaluation_config and
    # suite summary.  It never changes agent behaviour.
    experiment_label: str = "standard_evaluation"


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
    active_audit_rows: List[Dict[str, Any]]
    oracle_pruning_rows: List[Dict[str, Any]]
    turn_rows: List[Dict[str, Any]]
    initial_memory: Dict[str, Any]
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
        **BASELINE_AGENTS,
    }
    if name not in registry:
        raise KeyError(f"unknown baseline {name!r}; available={sorted(registry)}")
    return registry[name](cfg=cfg)


def shuffled_recipe_builders(seed: int) -> List[Tuple[str, Callable[[], List[str]]]]:
    items = list(gen.recipe_library().items())
    rng = random.Random(int(seed))
    rng.shuffle(items)
    return items


def select_recipe_builders(seed: int, n_recipes: int) -> List[Tuple[str, Callable[[], List[str]]]]:
    items = shuffled_recipe_builders(seed)
    return items[: max(1, min(int(n_recipes), len(items)))]


def _sampled_recipe_builders(config: EvaluationConfig, seed: int) -> List[Tuple[str, Callable[[], List[str]]]]:
    """Draw the experimental recipe panel without filtering on axis support.

    A seed therefore samples from the full recipe population.  A non-identity
    preference that leaves one sampled recipe unchanged is omitted only from
    that rung; it must never cause the recipe itself to be replaced by a more
    convenient one.
    """
    return select_recipe_builders(seed, max(1, int(config.n_recipes)))


def _effective_pairs_by_preference(
    recipe_name: str,
    builder: Callable[[], List[str]],
    preferences: Sequence[str],
) -> Dict[str, RecipePreferencePair]:
    """Materialize useful preference variants for one fixed raw recipe.

    Identity remains an explicit control.  Other preferences are omitted when
    they reproduce identity or an earlier emitted ordering, avoiding no-op
    trials while retaining the recipe in every other applicable rung.
    """
    base = tuple(builder())
    pairs: Dict[str, RecipePreferencePair] = {}
    seen: Dict[Tuple[str, ...], str] = {}
    for preference_name in preferences:
        try:
            pair = materialize_pair(recipe_name, preference_name, builder)
        except Exception:
            continue
        if preference_name != "identity" and pair.actions == base:
            continue
        duplicate_of = seen.get(pair.actions)
        if duplicate_of is not None:
            continue
        seen[pair.actions] = preference_name
        pairs[preference_name] = pair
    return pairs


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


def _preferences(config: EvaluationConfig) -> List[str]:
    prefs = [p for p in config.ladder_preferences if p in PRESET_PREFERENCES]
    return prefs if "identity" in prefs else ["identity", *prefs]

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


def _settle_block_size(
    config: EvaluationConfig,
    subgroup_size: int,
    rng: random.Random,
) -> int:
    """Choose a phase-level settled-block budget without equalizing recipes.

    The default is intentionally specified at the subgroup level, rather than
    as a repeat-after-each-update count.  This is the experimental distinction
    between a climb (one exposure to every selected pair) and a settled block
    (random reuse of the complete just-climbed pair set).
    """
    n_pairs = max(0, int(subgroup_size))
    if n_pairs == 0:
        return 0
    lower = max(n_pairs, int(math.ceil(float(config.settle_repeat_multiplier_min) * n_pairs)))
    upper = max(lower, int(math.floor(float(config.settle_repeat_multiplier_max) * n_pairs)))
    return rng.randint(lower, upper)


def _randomized_settle_order(
    pairs: Sequence[RecipePreferencePair],
    total_repeats: int,
    rng: random.Random,
) -> List[RecipePreferencePair]:
    """Return an unequal random reuse sequence covering the full climb set.

    Every climbed pair occurs at least once.  The deliberate two-extra-draw
    allocation prevents an accidental exactly balanced block when there are
    multiple pairs, while the remaining draws are sampled uniformly and the
    sequence is shuffled.  Thus order and counts are not a hidden factorial
    control variable.
    """
    unique_pairs = list(pairs)
    if not unique_pairs or total_repeats <= 0:
        return []
    total = max(len(unique_pairs), int(total_repeats))
    counts = {pair.label: 1 for pair in unique_pairs}
    remaining = total - len(unique_pairs)
    if len(unique_pairs) > 1 and remaining >= 2:
        anchor = rng.choice(unique_pairs)
        counts[anchor.label] += 2
        remaining -= 2
    for _ in range(remaining):
        counts[rng.choice(unique_pairs).label] += 1
    by_label = {pair.label: pair for pair in unique_pairs}
    order = [by_label[label] for label, count in counts.items() for _ in range(count)]
    rng.shuffle(order)
    return order


def _settle_counts(order: Sequence[RecipePreferencePair]) -> Dict[str, int]:
    return dict(Counter(pair.label for pair in order))


def _place_horizon_calibrated_recurrence_probe(
    order: Sequence[RecipePreferencePair],
    target: RecipePreferencePair,
    *,
    required_delay_events: int,
    rng: random.Random,
) -> Tuple[List[RecipePreferencePair], Optional[int]]:
    """Place the first settled occurrence after the required delay.

    The sequence multiset is preserved exactly.  This is important because the
    delayed-recurrence condition must not receive more target exposures or a
    larger settled block than the ordinary random-ladder condition.  The
    caller places ``target`` last in its climb block, so this settled index is
    also the number of intervening episode events between update and probe.
    """
    original = list(order)
    target_count = sum(pair.label == target.label for pair in original)
    if target_count <= 0:
        return original, None

    probe_position = max(0, int(required_delay_events))
    if probe_position > len(original) - target_count:
        return original, None
    non_targets = [pair for pair in original if pair.label != target.label]
    prefix = non_targets[:probe_position]
    suffix = non_targets[probe_position:] + [target] * (target_count - 1)
    rng.shuffle(suffix)
    return prefix + [target] + suffix, probe_position


def build_ladder_heterogeneous(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    """Build an all-recipe, phase-level heterogeneous preference ladder.

    Each recipe receives its own seeded random permutation of the available
    preference pairs.  This keeps its per-rung assignment heterogeneous while
    allowing a shared preference to occur for several recipes naturally.  It
    deliberately does *not* impose an off-diagonal Latin-square constraint:
    at the requested 15 recipes / 9 rungs that constraint would silently drop
    seven recipes, contradicting the experimental population.
    """
    rng = random.Random(int(seed) + 1103)
    # Every seed keeps its randomly sampled recipe panel.  Each recipe then
    # draws independently from the full fifteen-preference pool, so several
    # recipes may naturally share a preference within one rung.
    recipe_builders = _sampled_recipe_builders(config, seed)
    recipes = [recipe for recipe, _builder in recipe_builders]
    preferences = list(PREFERENCE_NAMES)
    matrix = {
        recipe: _effective_pairs_by_preference(recipe, builder, preferences)
        for recipe, builder in recipe_builders
    }
    events: List[ScenarioEvent] = []
    n_rungs = max(2, int(config.ladder_rungs))
    # A recipe with fewer than nine *distinct, effective* orderings is kept
    # in the seed panel.  It simply has no item in the surplus rung(s), rather
    # than being replaced by a recipe selected for high axis coverage.  This
    # is the intended exception to the all-recipes-per-rung rule.
    pair_order: Dict[str, List[Optional[RecipePreferencePair]]] = {}
    for recipe in recipes:
        effective = list(matrix[recipe].values())
        sampled = rng.sample(effective, k=min(n_rungs, len(effective)))
        pair_order[recipe] = [*sampled, *([None] * max(0, n_rungs - len(sampled)))]
        rng.shuffle(pair_order[recipe])
    prior_preference_sources: Dict[str, RecipePreferencePair] = {}

    for rung in range(n_rungs):
        phase_id = f"rung_{rung:02d}"
        climb_pairs = [
            pair_order[recipe][rung]
            for recipe in recipes
            if pair_order[recipe][rung] is not None
        ]
        omitted_nonoperative_count = len(recipes) - len(climb_pairs)
        for recipe_idx, pair in enumerate(climb_pairs):
            assert pair is not None
            common = {
                "rung_idx": rung,
                "phase_id": phase_id,
                "phase_role": "climb",
                "phase_position": recipe_idx,
                "subgroup_id": phase_id,
                "subgroup_size": len(climb_pairs),
                "climb_block_size": len(climb_pairs),
                "recipe_position": recipe_idx,
                "ladder_structure": "heterogeneous_all_recipe_seeded_permutations",
                "target_pair_seen_before": False,
                "sampled_recipe_count": len(recipes),
                "omitted_nonoperative_recipe_count": omitted_nonoperative_count,
            }
            if rung == 0:
                events.append(ScenarioEvent("observe", pair, _scenario_tag(
                    config,
                    SCENARIO_LADDER_HETEROGENEOUS,
                    event_type="onboarding_observation",
                    condition="onboarding_observation",
                    condition_family="initial_learning",
                    evaluation_phase="climb",
                    primary_probe=False,
                    hypothesis_tags=["initial_learning"],
                    **common,
                )))
                continue
            source_pair = prior_preference_sources.get(pair.preference_name)
            is_cross_recipe_transfer = source_pair is not None and source_pair.recipe_name != recipe
            hypothesis_tags = [
                "cross_recipe_transfer" if is_cross_recipe_transfer else "known_recipe_new_preference_adaptation",
                "emergent_axis_composition" if pair.is_composed_preference else "single_axis_preference",
            ]
            base_tags = _scenario_tag(
                config,
                SCENARIO_LADDER_HETEROGENEOUS,
                event_type=(
                    "heterogeneous_climb_cross_recipe_transfer"
                    if is_cross_recipe_transfer else "heterogeneous_climb_preference_update"
                ),
                condition=(
                    "heterogeneous_cross_recipe_transfer"
                    if is_cross_recipe_transfer else "heterogeneous_preference_update"
                ),
                condition_family=("cross_recipe_transfer" if is_cross_recipe_transfer else "within_recipe_new_preference"),
                evaluation_phase="climb",
                primary_probe=True,
                source_recipe=(source_pair.recipe_name if is_cross_recipe_transfer else None),
                source_pair=(source_pair.label if is_cross_recipe_transfer else None),
                source_preference=(source_pair.preference_name if is_cross_recipe_transfer else None),
                preference_non_default_axes=list(pair.non_default_axes),
                hypothesis_tags=hypothesis_tags,
                **common,
            )
            events.append(ScenarioEvent("assist", pair, base_tags))
        settled_order = _randomized_settle_order(
            climb_pairs,
            _settle_block_size(config, len(climb_pairs), rng),
            rng,
        )
        counts = _settle_counts(settled_order)
        seen_settle_count: Counter[str] = Counter()
        for settle_idx, pair in enumerate(settled_order):
            seen_settle_count[pair.label] += 1
            events.append(ScenarioEvent("assist", pair, _scenario_tag(
                config,
                SCENARIO_LADDER_HETEROGENEOUS,
                event_type="heterogeneous_settled_phase_reuse",
                condition="heterogeneous_settled_phase_reuse",
                condition_family="settled_phase_reuse",
                evaluation_phase="settled",
                primary_probe=False,
                rung_idx=rung,
                phase_id=phase_id,
                phase_role="settled",
                phase_position=settle_idx,
                subgroup_id=phase_id,
                subgroup_size=len(climb_pairs),
                climb_block_size=len(climb_pairs),
                settled_block_size=len(settled_order),
                settle_repeat_idx=seen_settle_count[pair.label] - 1,
                settle_repeat_count_for_pair=counts[pair.label],
                ladder_structure="heterogeneous_all_recipe_seeded_permutations",
                hypothesis_tags=["settled_phase_reuse", "retention_after_adaptation"],
                preference_non_default_axes=list(pair.non_default_axes),
            )))
        for pair in climb_pairs:
            assert pair is not None
            prior_preference_sources.setdefault(pair.preference_name, pair)
    return ScenarioPlan(
        scenario=SCENARIO_LADDER_HETEROGENEOUS,
        seed=seed,
        events=tuple(events),
        eval_pairs=tuple(
            pair
            for recipe in recipes
            for pair in pair_order[recipe]
            if pair is not None
        ),
        selected_recipes=tuple(recipes),
        selected_preferences=tuple(preferences),
        description="Fifteen-recipe heterogeneous phase ladder. Each recipe independently samples effective preferences from the full fifteen-preference pool; shared preferences within a rung are allowed. A recipe with fewer effective variants than rungs is omitted only from those otherwise non-operative rungs, and every rung settles by random reuse of its climbed pairs.",
    )


def build_ladder_homogeneous(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    """Build an all-recipe shared-preference climb/settle ladder."""
    rng = random.Random(int(seed) + 1207)
    recipe_builders = _sampled_recipe_builders(config, seed + 101)
    recipes = [recipe for recipe, _builder in recipe_builders]
    preferences = _preferences(config)
    n_rungs = min(max(2, int(config.ladder_rungs)), len(preferences))
    matrix = {
        recipe: _effective_pairs_by_preference(recipe, builder, preferences[:n_rungs])
        for recipe, builder in recipe_builders
    }
    events: List[ScenarioEvent] = []
    for rung in range(n_rungs):
        phase_id = f"rung_{rung:02d}"
        preference_name = preferences[rung]
        climb_pairs = [
            matrix[recipe][preference_name]
            for recipe in recipes
            if preference_name in matrix[recipe]
        ]
        if not climb_pairs:
            raise RuntimeError(
                f"homogeneous rung {rung} ({preference_name}) has no effective recipe/preference pairs"
            )
        omitted_noop_count = len(recipes) - len(climb_pairs)
        for recipe_idx, pair in enumerate(climb_pairs):
            common = {
                "rung_idx": rung,
                "phase_id": phase_id,
                "phase_role": "climb",
                "phase_position": recipe_idx,
                "subgroup_id": phase_id,
                "subgroup_size": len(climb_pairs),
                "climb_block_size": len(climb_pairs),
                "recipe_position": recipe_idx,
                "ladder_structure": "homogeneous_all_recipe_shared_preference",
                "target_pair_seen_before": False,
                "sampled_recipe_count": len(recipes),
                "omitted_noop_recipe_count": omitted_noop_count,
                "shared_preference": preference_name,
            }
            if rung == 0:
                events.append(ScenarioEvent("observe", pair, _scenario_tag(
                    config,
                    SCENARIO_LADDER_HOMOGENEOUS,
                    event_type="onboarding_observation",
                    condition="onboarding_observation",
                    condition_family="initial_learning",
                    evaluation_phase="climb",
                    primary_probe=False,
                    hypothesis_tags=["initial_learning"],
                    **common,
                )))
                continue
            base_tags = _scenario_tag(
                config,
                SCENARIO_LADDER_HOMOGENEOUS,
                event_type="homogeneous_climb_shared_preference_update",
                condition="homogeneous_shared_preference_update",
                condition_family="homogeneous_shared_preference_update",
                evaluation_phase="climb",
                primary_probe=True,
                **_preference_tags(pair),
                **common,
            )
            events.append(ScenarioEvent("assist", pair, base_tags))
        settled_order = _randomized_settle_order(
            climb_pairs,
            _settle_block_size(config, len(climb_pairs), rng),
            rng,
        )
        counts = _settle_counts(settled_order)
        seen_settle_count: Counter[str] = Counter()
        for settle_idx, pair in enumerate(settled_order):
            seen_settle_count[pair.label] += 1
            events.append(ScenarioEvent("assist", pair, _scenario_tag(
                config,
                SCENARIO_LADDER_HOMOGENEOUS,
                event_type="homogeneous_settled_phase_reuse",
                condition="homogeneous_settled_phase_reuse",
                condition_family="settled_phase_reuse",
                evaluation_phase="settled",
                primary_probe=False,
                rung_idx=rung,
                phase_id=phase_id,
                phase_role="settled",
                phase_position=settle_idx,
                subgroup_id=phase_id,
                subgroup_size=len(climb_pairs),
                climb_block_size=len(climb_pairs),
                settled_block_size=len(settled_order),
                settle_repeat_idx=seen_settle_count[pair.label] - 1,
                settle_repeat_count_for_pair=counts[pair.label],
                ladder_structure="homogeneous_all_recipe_shared_preference",
                hypothesis_tags=["settled_phase_reuse", "retention_after_adaptation"],
                preference_non_default_axes=list(pair.non_default_axes),
            )))
    return ScenarioPlan(
        scenario=SCENARIO_LADDER_HOMOGENEOUS,
        seed=seed,
        events=tuple(events),
        eval_pairs=tuple(pair for recipe in recipes for pair in matrix[recipe].values()),
        selected_recipes=tuple(recipes),
        selected_preferences=tuple(preferences),
        description="Fifteen-recipe homogeneous phase ladder. Every rung uses one shared preference across the sampled panel; only recipe-specific no-op variants are omitted, then the effective rung pairs are randomly reused in a 2--2.5x settled block.",
    )


def build_deployment_random(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    """Build a phase-structured random ladder distinct from heterogeneity.

    At each rung, the climb subgroup is sampled independently (rather than
    containing every recipe or following a balanced recipe--preference design).
    New recipes are observed when first sampled; known recipes receive one
    unseen preference pair.  The subsequent settled block samples only the
    just-climbed pairs.  Cross-recipe reuse can arise naturally, but is neither
    quota-forced nor Latin-square-balanced, making this a stress test of the
    learned representation under variable recipe coverage and interference.
    """
    rng = random.Random(int(seed) + 2027)
    recipe_builders = _sampled_recipe_builders(config, seed + 202)
    recipes = [recipe for recipe, _builder in recipe_builders]
    preferences = list(PREFERENCE_NAMES)
    matrix = {
        recipe: list(_effective_pairs_by_preference(recipe, builder, preferences).values())
        for recipe, builder in recipe_builders
    }
    events: List[ScenarioEvent] = []
    observed_recipes: set[str] = set()
    climbed_pair_labels: set[str] = set()
    prior_preference_sources: Dict[str, List[RecipePreferencePair]] = defaultdict(list)
    # This is evaluator-only scheduling state.  It never enters an agent
    # prompt, feature vector, or action-selection interface.
    prior_pairs_by_recipe: Dict[str, List[RecipePreferencePair]] = defaultdict(list)
    # Mirror the full system's decay-clock inputs from the planned accepted
    # interaction stream. This is schedule construction only; agents receive
    # neither these counters nor the evaluator preference labels.
    decay_cfg = base_config(seed, config)
    reuse_window = max(1, int(getattr(decay_cfg, "decay_reuse_window", 3)))
    decay_horizon_floor = max(0, int(getattr(decay_cfg, "decay_horizon_floor", 6)))
    decay_horizon_init = max(0, int(getattr(decay_cfg, "decay_horizon_init", 15)))
    decay_after_grace_steps = max(1, int(getattr(decay_cfg, "decay_after_grace_steps", 3)))
    planned_recipe_last_session: Dict[str, int] = {}
    planned_recipe_reuse_gaps: Dict[str, List[int]] = defaultdict(list)

    for rung in range(max(2, int(config.ladder_rungs))):
        phase_start_event_idx = len(events)
        phase_id = f"rung_{rung:02d}"
        minimum = min(len(recipes), max(1, int(config.random_ladder_min_recipes_per_rung)))
        subgroup_size = rng.randint(minimum, len(recipes))
        subgroup = rng.sample(recipes, k=subgroup_size)
        prior_sources = {pref: list(pairs) for pref, pairs in prior_preference_sources.items()}
        climb_pairs: List[RecipePreferencePair] = []

        for recipe in subgroup:
            unseen = [pair for pair in matrix[recipe] if pair.label not in climbed_pair_labels]
            # With the standard rungs<=available pairs design this fallback is
            # unreachable.  Keep it explicit for intentional long stress runs.
            candidates = unseen or list(matrix[recipe])
            cross_candidates = [
                pair for pair in candidates
                if any(source.recipe_name != recipe for source in prior_sources.get(pair.preference_name, []))
            ]
            globally_novel = [pair for pair in candidates if pair.preference_name not in prior_sources]
            if cross_candidates and rng.random() < 0.55:
                pair = rng.choice(cross_candidates)
            else:
                pair = rng.choice(globally_novel or candidates)
            climb_pairs.append(pair)

        # The target is a newly updated pair whose recipe has accumulated at
        # least two *different* older variants.  Moving the target to the end
        # of the climb makes the later settled position an exact, transparent
        # update-to-recurrence lag.  We only relabel/reorder an existing
        # settled repeat; the pair multiset and total event count are fixed.
        settled_order = _randomized_settle_order(
            climb_pairs,
            _settle_block_size(config, len(climb_pairs), rng),
            rng,
        )
        recurrence_target: Optional[RecipePreferencePair] = None
        recurrence_probe_position: Optional[int] = None
        recurrence_conflicting_preferences: Tuple[str, ...] = ()
        recurrence_conflicting_pair_labels: Tuple[str, ...] = ()
        recurrence_current_reuse_gap: Optional[int] = None
        recurrence_grace_horizon: Optional[int] = None
        recurrence_required_delay: Optional[int] = None
        min_conflicts = max(1, int(config.random_ladder_recurrence_min_prior_conflicts))
        # The selected target is placed last in the climb, so this is its
        # planned session index and the exact point at which the old latest
        # variant becomes unpinned in the full system.
        target_session = phase_start_event_idx + len(climb_pairs)
        delayed_candidates: List[Dict[str, Any]] = []
        for pair in climb_pairs:
            if pair.label in climbed_pair_labels:
                continue
            earlier_pairs = [
                earlier for earlier in prior_pairs_by_recipe.get(pair.recipe_name, [])
                if earlier.label != pair.label
            ]
            conflicts = tuple(sorted({earlier.preference_name for earlier in earlier_pairs}))
            if len(conflicts) < min_conflicts:
                continue
            last_session = planned_recipe_last_session.get(pair.recipe_name)
            if last_session is None:
                continue
            current_reuse_gap = target_session - int(last_session)
            prospective_gaps = (
                list(planned_recipe_reuse_gaps.get(pair.recipe_name, []))[-(reuse_window - 1):]
                + [current_reuse_gap]
            )[-reuse_window:]
            grace_horizon = (
                max(decay_horizon_floor, max(prospective_gaps))
                if prospective_gaps else max(decay_horizon_init, decay_horizon_floor)
            )
            # The old latest has age ``current_reuse_gap`` at the target
            # update. Once unpinned, it first decays when age exceeds the
            # grace horizon and is pruned after the configured number of
            # overdue sessions. The fixed lower bound prevents a degenerate
            # immediate repeat when the horizon is already satisfied.
            required_delay = max(
                max(0, int(config.random_ladder_recurrence_min_intervening_events)),
                max(0, int(grace_horizon) - int(current_reuse_gap)) + decay_after_grace_steps,
            )
            delayed_candidates.append({
                "target": pair,
                "conflicting_preferences": conflicts,
                "conflicting_pair_labels": tuple(sorted({earlier.label for earlier in earlier_pairs})),
                "current_reuse_gap": int(current_reuse_gap),
                "grace_horizon": int(grace_horizon),
                "required_delay": int(required_delay),
            })

        # A target can be used only if every duplicate occurrence can remain
        # after the required horizon-calibrated delay. Otherwise a target
        # repeat would appear before the mechanism is engaged.
        viable_candidates = [
            candidate for candidate in delayed_candidates
            if len(settled_order) - _settle_counts(settled_order).get(candidate["target"].label, 0)
            >= int(candidate["required_delay"])
        ]
        if viable_candidates:
            selected = rng.choice(viable_candidates)
            recurrence_target = selected["target"]
            recurrence_conflicting_preferences = selected["conflicting_preferences"]
            recurrence_conflicting_pair_labels = selected["conflicting_pair_labels"]
            recurrence_current_reuse_gap = selected["current_reuse_gap"]
            recurrence_grace_horizon = selected["grace_horizon"]
            recurrence_required_delay = selected["required_delay"]
            target_idx = climb_pairs.index(recurrence_target)
            climb_pairs.append(climb_pairs.pop(target_idx))
            settled_order, recurrence_probe_position = _place_horizon_calibrated_recurrence_probe(
                settled_order,
                recurrence_target,
                required_delay_events=recurrence_required_delay,
                rng=rng,
            )
            # Defensive fallback for nonstandard small blocks or invalid user
            # overrides: retain the ordinary settled order instead of creating
            # a mislabeled mechanism probe.
            if recurrence_probe_position is None:
                recurrence_target = None
                recurrence_conflicting_preferences = ()
                recurrence_conflicting_pair_labels = ()
                recurrence_current_reuse_gap = None
                recurrence_grace_horizon = None
                recurrence_required_delay = None

        for climb_idx, pair in enumerate(climb_pairs):
            recipe = pair.recipe_name
            source_candidates = [
                source for source in prior_sources.get(pair.preference_name, [])
                if source.recipe_name != recipe
            ]
            source_pair = rng.choice(source_candidates) if source_candidates else None
            common = {
                "rung_idx": rung,
                "phase_id": phase_id,
                "phase_role": "climb",
                "phase_position": climb_idx,
                "subgroup_id": phase_id,
                "subgroup_size": subgroup_size,
                "climb_block_size": subgroup_size,
                "random_subgroup_recipe_coverage": float(subgroup_size) / max(1, len(recipes)),
                "ladder_structure": "random_variable_subgroup_phase_ladder",
                "target_pair_seen_before": pair.label in climbed_pair_labels,
            }
            if recipe not in observed_recipes:
                observed_recipes.add(recipe)
                events.append(ScenarioEvent("observe", pair, _scenario_tag(
                    config,
                    SCENARIO_DEPLOYMENT_RANDOM,
                    event_type="random_ladder_climb_new_recipe_observation",
                    condition="random_ladder_new_recipe",
                    condition_family="new_recipe_learning",
                    evaluation_phase="climb",
                    primary_probe=False,
                    hypothesis_tags=["new_recipe_learning", "random_climb"],
                    preference_non_default_axes=list(pair.non_default_axes),
                    **common,
                )))
            else:
                is_cross_recipe_transfer = source_pair is not None
                events.append(ScenarioEvent("assist", pair, _scenario_tag(
                    config,
                    SCENARIO_DEPLOYMENT_RANDOM,
                    event_type=(
                        "random_ladder_climb_cross_recipe_transfer"
                        if is_cross_recipe_transfer else "random_ladder_climb_preference_update"
                    ),
                    condition=(
                        "random_ladder_cross_recipe_transfer"
                        if is_cross_recipe_transfer else "random_ladder_preference_update"
                    ),
                    condition_family=("cross_recipe_transfer" if is_cross_recipe_transfer else "within_recipe_new_preference"),
                    evaluation_phase="climb",
                    primary_probe=True,
                    source_recipe=(source_pair.recipe_name if source_pair else None),
                    source_pair=(source_pair.label if source_pair else None),
                    source_preference=(source_pair.preference_name if source_pair else None),
                    hypothesis_tags=[
                        "cross_recipe_transfer" if is_cross_recipe_transfer else "known_recipe_new_preference_adaptation",
                        "random_climb",
                        "emergent_axis_composition" if pair.is_composed_preference else "single_axis_preference",
                    ],
                    preference_non_default_axes=list(pair.non_default_axes),
                    **common,
                )))

        counts = _settle_counts(settled_order)
        seen_settle_count: Counter[str] = Counter()
        for settle_idx, pair in enumerate(settled_order):
            seen_settle_count[pair.label] += 1
            is_delayed_recurrence_probe = bool(
                recurrence_target is not None
                and recurrence_probe_position is not None
                and pair.label == recurrence_target.label
                and settle_idx == recurrence_probe_position
            )
            event_type = (
                "random_ladder_delayed_recurrence_probe"
                if is_delayed_recurrence_probe else "random_ladder_settled_phase_reuse"
            )
            condition = (
                "random_ladder_delayed_recurrence_interference"
                if is_delayed_recurrence_probe else "random_ladder_settled_phase_reuse"
            )
            condition_family = (
                "delayed_recurrence_interference"
                if is_delayed_recurrence_probe else "settled_phase_reuse"
            )
            hypothesis_tags = ["settled_phase_reuse", "retention_after_adaptation", "random_settled_reuse"]
            if is_delayed_recurrence_probe:
                hypothesis_tags.append("delayed_recurrence_interference")
            events.append(ScenarioEvent("assist", pair, _scenario_tag(
                config,
                SCENARIO_DEPLOYMENT_RANDOM,
                event_type=event_type,
                condition=condition,
                condition_family=condition_family,
                evaluation_phase="settled",
                primary_probe=is_delayed_recurrence_probe,
                rung_idx=rung,
                phase_id=phase_id,
                phase_role="settled",
                phase_position=settle_idx,
                subgroup_id=phase_id,
                subgroup_size=subgroup_size,
                climb_block_size=subgroup_size,
                settled_block_size=len(settled_order),
                settle_repeat_idx=seen_settle_count[pair.label] - 1,
                settle_repeat_count_for_pair=counts[pair.label],
                random_subgroup_recipe_coverage=float(subgroup_size) / max(1, len(recipes)),
                ladder_structure="random_variable_subgroup_phase_ladder",
                delayed_recurrence_probe=is_delayed_recurrence_probe,
                delayed_recurrence_update_rung=(rung if is_delayed_recurrence_probe else None),
                delayed_recurrence_intervening_event_count=(
                    recurrence_probe_position if is_delayed_recurrence_probe else None
                ),
                delayed_recurrence_required_intervening_event_count=(
                    recurrence_required_delay if is_delayed_recurrence_probe else None
                ),
                delayed_recurrence_current_reuse_gap_events=(
                    recurrence_current_reuse_gap if is_delayed_recurrence_probe else None
                ),
                delayed_recurrence_predicted_grace_horizon_events=(
                    recurrence_grace_horizon if is_delayed_recurrence_probe else None
                ),
                delayed_recurrence_prune_confirmation_events=(
                    decay_after_grace_steps if is_delayed_recurrence_probe else None
                ),
                delayed_recurrence_historical_conflicting_preference_count=(
                    len(recurrence_conflicting_preferences) if is_delayed_recurrence_probe else None
                ),
                delayed_recurrence_historical_conflicting_preferences=(
                    list(recurrence_conflicting_preferences) if is_delayed_recurrence_probe else None
                ),
                delayed_recurrence_historical_conflicting_pair_labels=(
                    list(recurrence_conflicting_pair_labels) if is_delayed_recurrence_probe else None
                ),
                hypothesis_tags=hypothesis_tags,
                preference_non_default_axes=list(pair.non_default_axes),
            )))
        for pair in climb_pairs:
            climbed_pair_labels.add(pair.label)
            prior_preference_sources[pair.preference_name].append(pair)
            prior_pairs_by_recipe[pair.recipe_name].append(pair)
        # Advance the schedule mirror after the phase is complete. Settled
        # repeats count because each accepted HRC episode advances the decay
        # clock and updates the per-recipe reuse-gap window in the full agent.
        for session_idx, event in enumerate(events[phase_start_event_idx:], start=phase_start_event_idx + 1):
            recipe = event.pair.recipe_name
            previous = planned_recipe_last_session.get(recipe)
            if previous is not None:
                gaps = planned_recipe_reuse_gaps[recipe]
                gaps.append(int(session_idx) - int(previous))
                del gaps[:-reuse_window]
            planned_recipe_last_session[recipe] = int(session_idx)

    return ScenarioPlan(
        scenario=SCENARIO_DEPLOYMENT_RANDOM,
        seed=seed,
        events=tuple(events),
        eval_pairs=tuple(pair for recipe in recipes for pair in matrix[recipe]),
        selected_recipes=tuple(recipes),
        selected_preferences=tuple(preferences),
        description="Random variable-subgroup phase ladder. Each rung samples 3--15 recipes and random effective preferences from the full fifteen-preference pool, then randomly reuses only that rung's pairs in a 2--2.5x settled block. One eligible settled repeat is a tagged delayed-recurrence probe: a just-updated pair with at least two older conflicting recipe variants, first revisited only after its full-system, recipe-specific grace horizon and pruning ticks are satisfied without increasing the block budget. It has no common-preference or Latin-square constraint.",
    )


def _build_holdout_ladder(
    config: EvaluationConfig,
    seed: int,
    *,
    scenario: str,
    source_preferences: Sequence[str],
    heldout_preferences: Sequence[str],
    holdout_kind: str,
) -> ScenarioPlan:
    """Build one independent source-progression and holdout-adaptation arm."""
    rng = random.Random(f"{scenario}|{int(seed)}")
    recipe_builders = _sampled_recipe_builders(config, seed + 307)
    recipes = [recipe for recipe, _builder in recipe_builders]
    preferences = tuple(source_preferences) + tuple(heldout_preferences)
    matrix = {
        recipe: _effective_pairs_by_preference(recipe, builder, preferences)
        for recipe, builder in recipe_builders
    }
    events: List[ScenarioEvent] = []

    for rung, preference_name in enumerate(preferences):
        partition = "source" if preference_name in source_preferences else "heldout"
        phase_id = f"{partition}_{rung:02d}"
        climb_pairs = [
            matrix[recipe][preference_name]
            for recipe in recipes
            if preference_name in matrix[recipe]
        ]
        if not climb_pairs:
            raise RuntimeError(
                f"{scenario} rung {rung} ({preference_name}) has no effective recipe/preference pairs"
            )
        omitted_noop_count = len(recipes) - len(climb_pairs)
        requested_mode = "observe" if partition == "source" else "assist"
        phase_role = f"{partition}_climb"
        for position, pair in enumerate(climb_pairs):
            common = {
                "rung_idx": rung,
                "phase_id": phase_id,
                "phase_role": phase_role,
                "phase_position": position,
                "subgroup_id": phase_id,
                "subgroup_size": len(climb_pairs),
                "climb_block_size": len(climb_pairs),
                "recipe_position": position,
                "ladder_structure": f"{holdout_kind}_holdout_progression",
                "holdout_kind": holdout_kind,
                "holdout_partition": partition,
                "source_preference_count": len(source_preferences),
                "heldout_preference_count": len(heldout_preferences),
                "sampled_recipe_count": len(recipes),
                "omitted_noop_recipe_count": omitted_noop_count,
                "target_pair_seen_before": False,
            }
            tags = _scenario_tag(
                config,
                scenario,
                event_type=(
                    f"{holdout_kind}_source_progression_observation"
                    if partition == "source" else f"{holdout_kind}_heldout_preference_adaptation"
                ),
                condition=(
                    f"{holdout_kind}_source_progression"
                    if partition == "source" else f"{holdout_kind}_heldout_adaptation"
                ),
                condition_family=("holdout_source_progression" if partition == "source" else "holdout_adaptation"),
                evaluation_phase=("source" if partition == "source" else "heldout"),
                primary_probe=bool(partition == "heldout"),
                preference_non_default_axes=list(pair.non_default_axes),
                hypothesis_tags=(
                    ["source_preference_progression", holdout_kind]
                    if partition == "source" else ["heldout_preference_adaptation", holdout_kind]
                ),
                **common,
            )
            events.append(ScenarioEvent(requested_mode, pair, tags))

        settled_order = _randomized_settle_order(
            climb_pairs,
            _settle_block_size(config, len(climb_pairs), rng),
            rng,
        )
        counts = _settle_counts(settled_order)
        seen_settle_count: Counter[str] = Counter()
        for position, pair in enumerate(settled_order):
            seen_settle_count[pair.label] += 1
            events.append(ScenarioEvent(requested_mode, pair, _scenario_tag(
                config,
                scenario,
                event_type=(
                    f"{holdout_kind}_source_settled_reuse"
                    if partition == "source" else f"{holdout_kind}_heldout_settled_reuse"
                ),
                condition=("holdout_source_settled" if partition == "source" else "holdout_heldout_settled"),
                condition_family=("holdout_source_progression" if partition == "source" else "holdout_adaptation"),
                evaluation_phase=("source_settled" if partition == "source" else "heldout_settled"),
                primary_probe=False,
                rung_idx=rung,
                phase_id=phase_id,
                phase_role=f"{partition}_settled",
                phase_position=position,
                subgroup_id=phase_id,
                subgroup_size=len(climb_pairs),
                climb_block_size=len(climb_pairs),
                settled_block_size=len(settled_order),
                settle_repeat_idx=seen_settle_count[pair.label] - 1,
                settle_repeat_count_for_pair=counts[pair.label],
                ladder_structure=f"{holdout_kind}_holdout_progression",
                holdout_kind=holdout_kind,
                holdout_partition=partition,
                source_preference_count=len(source_preferences),
                heldout_preference_count=len(heldout_preferences),
                sampled_recipe_count=len(recipes),
                omitted_noop_recipe_count=omitted_noop_count,
                hypothesis_tags=(
                    ["source_preference_progression", "settled_phase_reuse", holdout_kind]
                    if partition == "source" else ["heldout_preference_adaptation", "settled_phase_reuse", holdout_kind]
                ),
                preference_non_default_axes=list(pair.non_default_axes),
            )))

    return ScenarioPlan(
        scenario=scenario,
        seed=seed,
        events=tuple(events),
        eval_pairs=tuple(
            pair
            for recipe in recipes
            for pair in matrix[recipe].values()
        ),
        selected_recipes=tuple(recipes),
        selected_preferences=tuple(preferences),
        description=(
            f"Fifteen-recipe {holdout_kind} holdout ladder. Source preferences are shown naturally in observation "
            f"rungs, then held-out preferences are tested in assistive climb/settled rungs. Non-identity no-op "
            f"recipe/preference pairs are omitted only from their own rung."
        ),
    )


def build_axis_holdout(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    """Four observed isolated axes followed by four held-out isolated axes."""
    return _build_holdout_ladder(
        config,
        seed,
        scenario=SCENARIO_AXIS_HOLDOUT,
        source_preferences=(
            "p1_mise_en_place",
            "p2_equipment_just_in_time",
            "p3_frontload_serving_setup",
            "p4_load_just_in_time",
        ),
        heldout_preferences=(
            "p5_shutdown_late",
            "p6_deferred_cook_start",
            "p7_clean_eager",
            "p8_cleanup_before_serve",
        ),
        holdout_kind="axis",
    )


def build_preference_holdout(config: EvaluationConfig, seed: int) -> ScenarioPlan:
    """Seven observed preferences covering all axes, then eight held preferences."""
    return _build_holdout_ladder(
        config,
        seed,
        scenario=SCENARIO_PREFERENCE_HOLDOUT,
        source_preferences=(
            "identity",
            "p5_shutdown_late",
            "p6_deferred_cook_start",
            "p8_cleanup_before_serve",
            "p9_equipment_jit_frontload_serving",
            "p10_mise_en_place_clean",
            "p12_multi_stage_reorganization",
        ),
        heldout_preferences=(
            "p1_mise_en_place",
            "p2_equipment_just_in_time",
            "p3_frontload_serving_setup",
            "p4_load_just_in_time",
            "p7_clean_eager",
            "p11_mise_en_place_serving_clean",
            "p13_equipment_jit_clean",
            "p14_mise_load_clean",
        ),
        holdout_kind="preference",
    )


def build_scenario_plan(scenario: str, config: EvaluationConfig, seed: int) -> ScenarioPlan:
    if scenario == SCENARIO_LADDER_HETEROGENEOUS:
        return build_ladder_heterogeneous(config, seed)
    if scenario == SCENARIO_LADDER_HOMOGENEOUS:
        return build_ladder_homogeneous(config, seed)
    if scenario == SCENARIO_DEPLOYMENT_RANDOM:
        return build_deployment_random(config, seed)
    if scenario == SCENARIO_AXIS_HOLDOUT:
        return build_axis_holdout(config, seed)
    if scenario == SCENARIO_PREFERENCE_HOLDOUT:
        return build_preference_holdout(config, seed)
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


def _delayed_recurrence_memory_audit_tags(
    agent: AdaptiveHRCAgent,
    pair: RecipePreferencePair,
    name_to_rid: Mapping[str, str],
    target_key: Optional[VariantKey],
    tags: Mapping[str, Any],
    pairs_by_label: Mapping[str, RecipePreferencePair],
) -> Dict[str, Any]:
    """Record whether a tagged recurrence actually engaged memory removal.

    The scheduler uses a full-system clock mirror, but a deployed agent can
    still decline a commit or route an event differently. These evaluator-only
    diagnostics make that distinction reportable instead of silently assuming
    that a nominally long delay caused forgetting.
    """
    if not bool(tags.get("delayed_recurrence_probe")):
        return {}
    conflict_labels = tuple(str(label) for label in (
        tags.get("delayed_recurrence_historical_conflicting_pair_labels") or ()
    ))
    active = _active_keys(agent)
    pruned = _pruned_keys(agent)
    conflict_keys = [
        _pair_key(agent, pairs_by_label[label], name_to_rid)
        for label in conflict_labels
        if label in pairs_by_label
    ]
    known_conflict_keys = [key for key in conflict_keys if key is not None]
    recipe_id = name_to_rid.get(pair.recipe_name)
    horizon = (
        float(agent.decay.horizon_for(target_key))
        if target_key is not None else None
    )
    target_is_latest = bool(
        target_key is not None
        and recipe_id is not None
        and getattr(agent.decay, "latest_by_recipe", {}).get(recipe_id) == target_key[1]
    )
    active_count = sum(key in active for key in known_conflict_keys)
    pruned_count = sum(key in pruned for key in known_conflict_keys)
    return {
        "delayed_recurrence_target_active_before": bool(target_key and target_key in active),
        "delayed_recurrence_target_is_latest_before": target_is_latest,
        "delayed_recurrence_actual_grace_horizon_before": horizon,
        "delayed_recurrence_agent_session_before": int(getattr(agent, "session_counter", 0)),
        "delayed_recurrence_known_conflicting_variant_count_before": len(known_conflict_keys),
        "delayed_recurrence_conflicting_active_count_before": int(active_count),
        "delayed_recurrence_conflicting_pruned_count_before": int(pruned_count),
        "delayed_recurrence_all_known_conflicts_pruned_before": bool(
            known_conflict_keys and pruned_count == len(known_conflict_keys)
        ),
    }


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
    observations = observations_from_actions(pair.actions)
    actual_tokens = _evaluator_tokens_from_observations(agent, observations)
    actual_labels = tuple(_opaque_observation_label(obs) for obs in observations)

    def predict(prefix: Sequence[str]) -> Mapping[str, float]:
        return agent.predict_next_tokens(list(prefix))

    def observe(obs: Any, distribution: Optional[Mapping[str, float]] = None) -> None:
        # The agent receives only the observed numeric state transition and its
        # own prior distribution. Ground-truth recipe/preference labels remain
        # evaluator-only metadata.
        agent.observe_observation(obs, precomputed_distribution=distribution)

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
        "mode": "assist",
        # Always stratify first-pass forgetting and re-entry performance by
        # the pre-episode state, never the post-commit state.
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
                "memory_state_before": metrics.get("memory_state_before"),
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
        ("offline_pretraining_metadata", "offline_pretraining"),
    ):
        method = getattr(agent, method_name, None)
        if callable(method):
            try:
                out[key] = method()
            except Exception:
                pass
    offline_pretraining = out.get("offline_pretraining")
    if isinstance(offline_pretraining, Mapping):
        # Keep the nested record for provenance while exposing the fields in
        # diagnostics.jsonl/CSV exports for direct efficiency comparisons.
        for key, value in offline_pretraining.items():
            out.setdefault(str(key), value)
    return out




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
        primary = result.get("passed")
        return {
            **row,
            **result,
            "audit_available": True,
            "primary_active_only_contract": "fitted_policy_matches_active_replay_reference",
            "primary_active_only_contract_passed": bool(primary) if primary is not None else None,
        }
    except Exception as exc:
        return {
            **row,
            "audit_available": False,
            "passed": False,
            "primary_active_only_contract": "fitted_policy_matches_active_replay_reference",
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
    """Whether ``event_idx`` closes a completed settled ladder rung.

    This is deliberately tag-based rather than a fixed event interval: the
    number of recipes and random settled repeats may change, while each
    completed phase still receives exactly one full-grid frozen evaluation.
    """
    if event_idx < 0 or event_idx >= len(plan.events):
        return False
    tags = plan.events[event_idx].tags
    if tags.get("phase_role") is not None:
        if tags.get("phase_role") != "settled":
            return False
        phase_id = tags.get("phase_id")
        if event_idx == len(plan.events) - 1:
            return True
        return plan.events[event_idx + 1].tags.get("phase_id") != phase_id
    rung = tags.get("rung_idx")
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

    Full-grid frozen evaluations are rung-level diagnostics for both structured
    ladders. Randomized phases use matched climb probes only, avoiding a large
    diagnostic multiplier on the deliberately repeated settled events. This
    changes diagnostic timing only; it never changes live interaction or model
    updates.
    """
    if plan.scenario in {
        SCENARIO_LADDER_HOMOGENEOUS,
        SCENARIO_LADDER_HETEROGENEOUS,
        SCENARIO_AXIS_HOLDOUT,
        SCENARIO_PREFERENCE_HOLDOUT,
    }:
        return _is_preference_rung_boundary(plan, event_idx)
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
    so they are excluded. In every phase ladder, only primary climb events
    receive a matched frozen probe; settled-block performance is the actual
    deployed, post-commit outcome rather than a diagnostic replay.
    """
    if not bool(config.pre_event_frozen_probes) or requested_mode != "assist" or executed_mode != "assist":
        return False
    return bool(tags.get("primary_probe", False))


def _offline_pretraining_subset(
    values: Sequence[str],
    fraction: float,
    *,
    seed: int,
    axis: str,
) -> Tuple[str, ...]:
    """Choose a reproducible 40--50% style offline-training subset."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"offline-pretrained {axis} fraction must be in (0, 1], got {fraction!r}")
    candidates = sorted({str(value) for value in values})
    if not candidates:
        raise ValueError(f"cannot pretrain offline baseline: scenario has no {axis}s")
    # floor(0.5 * 5) = 2 gives 40%, keeping the requested 40--50% coverage
    # when an odd number of preferences is available.
    n_selected = max(1, min(len(candidates), int(math.floor(float(fraction) * len(candidates)))))
    rng = random.Random(f"offline_pretrained_frozen|{axis}|{int(seed)}")
    return tuple(sorted(rng.sample(candidates, n_selected)))


def _fit_and_lock_offline_frozen_agent(
    agent: AdaptiveHRCAgent,
    *,
    recipe_names: Sequence[str],
    preference_names: Sequence[str],
    metadata: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Fit an offline reference before deployment without exposing labels to it."""
    lock_deployment = getattr(agent, "lock_deployment", None)
    if not callable(lock_deployment):
        raise TypeError("offline frozen registry entry must implement lock_deployment()")
    library = gen.recipe_library()
    pairs = [
        materialize_pair(recipe_name, preference_name, library[recipe_name])
        for recipe_name in recipe_names
        for preference_name in preference_names
    ]
    name_to_rid: Dict[str, str] = {}
    pretraining_t0 = time.perf_counter()
    for pair in pairs:
        # ``observe_episode`` supplies only anonymous transition observations
        # to the learner. Recipe/preference labels stay in evaluator metadata.
        observe_episode(agent, pair, name_to_rid)
    locked_metadata = lock_deployment({
        **dict(metadata),
        "offline_training_recipe_count": int(len(recipe_names)),
        "offline_training_preference_count": int(len(preference_names)),
        "offline_training_pair_count": int(len(pairs)),
        "offline_training_recipe_names": list(sorted(recipe_names)),
        "offline_training_preference_names": list(sorted(preference_names)),
        "offline_pretraining_end_to_end_wall_s": float(time.perf_counter() - pretraining_t0),
    })
    return name_to_rid, dict(locked_metadata)


def _prepare_offline_pretrained_frozen_agent(
    agent: AdaptiveHRCAgent,
    plan: ScenarioPlan,
    config: EvaluationConfig,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Pretrain on paired 40--50% recipe and preference subsets, then lock."""
    recipe_names = _offline_pretraining_subset(
        plan.selected_recipes,
        config.offline_pretrained_recipe_fraction,
        seed=plan.seed,
        axis="recipe",
    )
    preference_names = _offline_pretraining_subset(
        plan.selected_preferences,
        config.offline_pretrained_preference_fraction,
        seed=plan.seed,
        axis="preference",
    )
    return _fit_and_lock_offline_frozen_agent(
        agent,
        recipe_names=recipe_names,
        preference_names=preference_names,
        metadata={
            "offline_training_design": "subset_recipes_subset_preferences",
            "offline_training_recipe_fraction_requested": float(config.offline_pretrained_recipe_fraction),
            "offline_training_preference_fraction_requested": float(config.offline_pretrained_preference_fraction),
        },
    )


def _prepare_offline_all_recipes_identity_frozen_agent(
    agent: AdaptiveHRCAgent,
    plan: ScenarioPlan,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Pretrain every evaluated recipe on the identity preference only, then lock."""
    recipe_names = tuple(sorted({str(recipe_name) for recipe_name in plan.selected_recipes}))
    if not recipe_names:
        raise ValueError("cannot pretrain all-recipes identity baseline: scenario has no recipes")
    return _fit_and_lock_offline_frozen_agent(
        agent,
        recipe_names=recipe_names,
        preference_names=("identity",),
        metadata={
            "offline_training_design": "all_selected_recipes_identity_only",
            "offline_training_recipe_fraction_requested": 1.0,
            "offline_training_recipe_scope": "all_selected_scenario_recipes",
            "offline_training_preference_scope": "identity_only",
        },
    )


def _prepare_holdout_source_matched_frozen_agent(
    agent: AdaptiveHRCAgent,
    plan: ScenarioPlan,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Fit a frozen control on the exact observation-only source progression.

    The holdout arms compare adaptation after a common source curriculum.  An
    online method receives that curriculum during its source rungs; a frozen
    method must receive the same ordered demonstrations before deployment and
    then be locked.  Replaying the actual source events (including their
    unequal settled repeats) avoids turning either frozen control into a
    broader, differently distributed pretraining advantage.
    """
    lock_deployment = getattr(agent, "lock_deployment", None)
    if not callable(lock_deployment):
        raise TypeError("holdout frozen registry entry must implement lock_deployment()")
    source_events = [
        event for event in plan.events
        if str(event.tags.get("holdout_partition")) == "source"
    ]
    if not source_events:
        raise ValueError(f"{plan.scenario} has no source progression to pretrain a frozen control")
    if any(event.mode != "observe" for event in source_events):
        raise ValueError("holdout source progression must remain observation-only")

    name_to_rid: Dict[str, str] = {}
    pretraining_t0 = time.perf_counter()
    for event in source_events:
        # Labels stay evaluator-side; the learner receives the same anonymous
        # action observations it would receive online during a source rung.
        observe_episode(agent, event.pair, name_to_rid)
    unique_pairs = {event.pair.label for event in source_events}
    source_recipes = {event.pair.recipe_name for event in source_events}
    source_preferences = {event.pair.preference_name for event in source_events}
    locked_metadata = lock_deployment({
        "offline_training_design": "matched_holdout_source_progression",
        "offline_training_scope": "exact_ordered_source_climb_and_settled_events",
        "offline_training_event_count": int(len(source_events)),
        "offline_training_unique_pair_count": int(len(unique_pairs)),
        "offline_training_recipe_count": int(len(source_recipes)),
        "offline_training_preference_count": int(len(source_preferences)),
        "offline_training_recipe_names": sorted(source_recipes),
        "offline_training_preference_names": sorted(source_preferences),
        "offline_pretraining_end_to_end_wall_s": float(time.perf_counter() - pretraining_t0),
        "holdout_source_progression_matched": True,
        "heldout_deployment_updates_allowed": False,
    })
    return name_to_rid, dict(locked_metadata)


def run_event_stream_for_baseline(
    baseline: str,
    plan: ScenarioPlan,
    config: EvaluationConfig,
    *,
    execution_mode_schedule: Optional[Sequence[str]] = None,
    mode_schedule_policy: str = "baseline_local",
) -> EventStreamRun:
    if execution_mode_schedule is not None:
        if len(execution_mode_schedule) != len(plan.events):
            raise ValueError(
                "execution_mode_schedule must contain exactly one mode per scenario event; "
                f"got {len(execution_mode_schedule)} for {len(plan.events)} events"
            )
        invalid_modes = sorted({str(mode) for mode in execution_mode_schedule if mode not in {"assist", "observe"}})
        if invalid_modes:
            raise ValueError(f"execution_mode_schedule has invalid modes: {invalid_modes}")
    is_clairvoyant = baseline == CLAIRVOYANT_MEMORY_ORACLE
    agent = make_agent("full" if is_clairvoyant else baseline, base_config(plan.seed, config))
    name_to_rid: Dict[str, str] = {}
    episode_rows: List[Dict[str, Any]] = []
    frozen_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    turn_rows: List[Dict[str, Any]] = []
    observed_recipes: set[str] = set()
    observed_preferences: set[str] = set()
    observed_pairs: set[str] = set()
    preferences_by_recipe: Dict[str, set[str]] = defaultdict(set)
    axis_values_by_recipe: Dict[str, set[str]] = defaultdict(set)
    last_global_frozen_event_idx: Optional[int] = None
    t0 = time.perf_counter()
    baseline_context: Dict[str, Any] = {}
    holdout_scenario = plan.scenario in {SCENARIO_AXIS_HOLDOUT, SCENARIO_PREFERENCE_HOLDOUT}
    if holdout_scenario and baseline in {
        "offline_pretrained_frozen",
        "offline_all_recipes_identity_frozen",
    }:
        name_to_rid, baseline_context = _prepare_holdout_source_matched_frozen_agent(agent, plan)
    elif baseline == "offline_pretrained_frozen":
        name_to_rid, baseline_context = _prepare_offline_pretrained_frozen_agent(agent, plan, config)
    elif baseline == "offline_all_recipes_identity_frozen":
        name_to_rid, baseline_context = _prepare_offline_all_recipes_identity_frozen_agent(agent, plan)
    # Keep offline/pre-deployment work out of individual online phases. The
    # initial snapshot is separately reported in the phase-cost schema.
    initial_memory = memory_snapshot(agent)
    pairs_by_label = {event.pair.label: event.pair for event in plan.events}

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
        natural_executed_mode = requested_mode
        natural_route_reason = "user_selected"
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
            natural_executed_mode = "observe"
            natural_route_reason = "assist_routed_to_observe_recipe_absent_from_active_memory"
        full_execution_mode = (
            str(execution_mode_schedule[event_idx])
            if execution_mode_schedule is not None else None
        )
        executed_mode = full_execution_mode or natural_executed_mode
        route_reason = (
            "matched_full_realized_execution_schedule"
            if full_execution_mode is not None and executed_mode != natural_executed_mode
            else natural_route_reason
        )
        tags.update({
            "requested_mode": requested_mode,
            "executed_mode": executed_mode,
            "mode_route_reason": route_reason,
            "natural_executed_mode": natural_executed_mode,
            "natural_mode_route_reason": natural_route_reason,
            "full_realized_execution_mode": full_execution_mode,
            "mode_schedule_policy": mode_schedule_policy,
            "mode_matches_full_schedule": (
                bool(executed_mode == full_execution_mode)
                if full_execution_mode is not None else None
            ),
        })
        tags.update(_delayed_recurrence_memory_audit_tags(
            agent,
            pair,
            name_to_rid,
            target_key_before,
            tags,
            pairs_by_label,
        ))
        if is_clairvoyant:
            tags.update({
                "oracle_reference": CLAIRVOYANT_MEMORY_ORACLE,
                "reported_as": CLAIRVOYANT_REFERENCE_TAG,
                "leakage_warning": CLAIRVOYANT_LEAKAGE_WARNING,
                "oracle_retention_policy": "future_recipe_support",
            })

        context = {
            **_event_context(baseline, plan, event_idx, requested_mode, executed_mode, pair, tags),
            **baseline_context,
        }
        active_before = _active_keys(agent)
        pruned_before = _pruned_keys(agent)
        retrain_before = len(getattr(agent, "retrain_events", []) or [])

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
                "memory_state_before": memory_state_before,
                "memory_state_after": _memory_state(
                    agent,
                    pair,
                    name_to_rid,
                    post_observed_pairs,
                    post_observed_recipes,
                ),
            })
        else:
            row = assist_episode(
                agent,
                pair,
                name_to_rid,
                config=config,
                commit=True,
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                memory_state_before=memory_state_before,
            )
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
                context={
                    "baseline": baseline,
                    "scenario": plan.scenario,
                    "seed": int(plan.seed),
                    **baseline_context,
                },
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
            ))
            last_global_frozen_event_idx = event_idx

    final_context = {
        "baseline": baseline,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        **baseline_context,
    }
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
        active_audit_rows=audit_rows,
        oracle_pruning_rows=oracle_rows,
        turn_rows=turn_rows,
        initial_memory=initial_memory,
        wall_s=float(time.perf_counter() - t0),
    )


def aggregate_episode_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "status": "not_run",
            "n_episodes": 0.0,
            "n_steps": 0.0,
            "n_recipe_steps": 0.0,
            "live_top1": None,
            "live_topk": None,
            "robot_wrong_rate": None,
            "human_correction_rate": None,
            "testing_total_action_time": 0.0,
            "testing_human_only_action_time": 0.0,
            "testing_human_effort_time": 0.0,
            "testing_episode_wall_s": 0.0,
            "testing_normalized_interaction_cost": None,
            "human_effort_time": 0.0,
            "mean_nll_per_robot_turn": None,
            "mean_prediction_wall_s": None,
            "human_shadow_top1": None,
            "human_shadow_topk": None,
            "n_human_shadow_turns": 0.0,
            "future_valid_wrong_rate": None,
            "observation_mode_rate": None,
            "user_observation_required_rate": None,
            "primary_prediction_metric": "live_top1",
            "primary_prediction_metric_value": None,
            "primary_hrc_metric": "human_correction_rate",
            "primary_hrc_metric_value": None,
        }
    robot_turns = sum(_numeric(row, "hrc_robot_turn_count") for row in rows)
    out = {
        "status": "completed",
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
        "testing_episode_wall_s": float(sum(_numeric(row, "episode_wall_s") for row in rows)),
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


def _phase_rung_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate the explicit climb/settled blocks without interleaving them."""
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        phase_role = str(row.get("phase_role") or "unphased")
        rung = row.get("rung_idx")
        rung_label = f"rung_{int(rung):02d}" if isinstance(rung, int) else "rung_unknown"
        grouped[phase_role][rung_label].append(row)
    return {
        phase_role: {
            rung_label: aggregate_episode_metrics(group_rows)
            for rung_label, group_rows in sorted(by_rung.items())
        }
        for phase_role, by_rung in sorted(grouped.items())
    }


_PHASE_TRAINING_FIELDS = (
    "training_total_retrain_wall_s",
    "training_fit_wall_s",
    "training_build_wall_s",
    "training_estimated_fit_flops",
    "training_retrain_count",
)


def _phase_training_costs(
    memory_rows: Sequence[Mapping[str, Any]],
    initial_memory: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attribute *online* retraining deltas to each logged phase and rung.

    Snapshots are cumulative.  Differencing them is necessary to avoid
    charging a late settled phase for all prior fitting, and starting from the
    post-pretraining snapshot keeps offline frozen baselines' initial fit
    separate from deployment adaptation.  FLOPs remain model-specific fit-only
    estimates and therefore must not be compared across model families.
    """
    zero = {f"online_{field}": 0.0 for field in _PHASE_TRAINING_FIELDS}
    previous = {field: _numeric(initial_memory, field) for field in _PHASE_TRAINING_FIELDS}
    by_phase: Dict[str, Dict[str, float]] = defaultdict(lambda: dict(zero))
    by_phase_rung: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: dict(zero))
    )
    for row in sorted(memory_rows, key=lambda item: _numeric(item, "event_idx", -1.0)):
        phase_role = str(row.get("phase_role") or "unphased")
        rung = row.get("rung_idx")
        rung_label = f"rung_{int(rung):02d}" if isinstance(rung, int) else "rung_unknown"
        for field in _PHASE_TRAINING_FIELDS:
            current = _numeric(row, field)
            delta = max(0.0, current - previous[field])
            previous[field] = current
            by_phase[phase_role][f"online_{field}"] = by_phase[phase_role].get(f"online_{field}", 0.0) + delta
            phase_rung = by_phase_rung[phase_role][rung_label]
            phase_rung[f"online_{field}"] = phase_rung.get(f"online_{field}", 0.0) + delta
    offline = {
        f"upfront_{field}": _numeric(initial_memory, field)
        for field in _PHASE_TRAINING_FIELDS
    }
    return {
        "definition": (
            "Online training cost is the non-negative event-to-event delta of cumulative retraining snapshots; "
            "upfront cost is pre-deployment work and is not allocated to a phase. FLOPs are fit-only, model-specific estimates."
        ),
        "upfront_training": offline,
        "per_phase_role": {phase: dict(values) for phase, values in sorted(by_phase.items())},
        "per_phase_rung": {
            phase: {rung: dict(values) for rung, values in sorted(by_rung.items())}
            for phase, by_rung in sorted(by_phase_rung.items())
        },
    }


def _add_phase_training_costs(metrics: Dict[str, Any], costs: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach phase-attributed online training costs to phase metric tables."""
    merged = {phase: dict(values) for phase, values in metrics.items()}
    for phase, values in (costs.get("per_phase_role") or {}).items():
        merged.setdefault(str(phase), {}).update(dict(values))
    return merged


def _add_phase_rung_training_costs(metrics: Dict[str, Dict[str, Any]], costs: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Attach rung-attributed online training costs without dropping empty assist cells."""
    merged = {
        phase: {rung: dict(values) for rung, values in by_rung.items()}
        for phase, by_rung in metrics.items()
    }
    for phase, by_rung in (costs.get("per_phase_rung") or {}).items():
        target = merged.setdefault(str(phase), {})
        for rung, values in by_rung.items():
            target.setdefault(str(rung), {}).update(dict(values))
    return merged


def frozen_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize frozen probes without treating an omitted audit as success.

    ``checkpoints`` is the sole summary schema.
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
    }


def frozen_summary_by(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> Dict[str, Dict[str, Any]]:
    """Summarize frozen probes by an evaluator tag while retaining support."""
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            grouped[str(value)].append(row)
    return {
        value: frozen_summary(group_rows)
        for value, group_rows in sorted(grouped.items())
    }


def delayed_recurrence_audit_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize whether the planned recurrence condition engaged forgetting."""
    probes = [row for row in rows if bool(row.get("delayed_recurrence_probe"))]
    if not probes:
        return {
            "definition": "Pre-event memory state for horizon-calibrated delayed-recurrence probes.",
            "status": "not_run",
            "n_probes": 0,
        }

    def rate(field: str) -> Optional[float]:
        values = [row.get(field) for row in probes if isinstance(row.get(field), bool)]
        return _mean(1.0 if value else 0.0 for value in values) if values else None

    return {
        "definition": "Pre-event memory state for horizon-calibrated delayed-recurrence probes. A Full probe engages the intended removal mechanism when the target is active/latest and all known historical conflicts are pruned.",
        "status": "completed",
        "n_probes": len(probes),
        "target_active_before_rate": rate("delayed_recurrence_target_active_before"),
        "target_latest_before_rate": rate("delayed_recurrence_target_is_latest_before"),
        "all_known_conflicts_pruned_before_rate": rate("delayed_recurrence_all_known_conflicts_pruned_before"),
        "mean_known_conflicting_variant_count_before": _mean(
            row.get("delayed_recurrence_known_conflicting_variant_count_before") for row in probes
        ),
        "mean_conflicting_active_count_before": _mean(
            row.get("delayed_recurrence_conflicting_active_count_before") for row in probes
        ),
        "mean_conflicting_pruned_count_before": _mean(
            row.get("delayed_recurrence_conflicting_pruned_count_before") for row in probes
        ),
        "mean_intervening_event_count": _mean(
            row.get("delayed_recurrence_intervening_event_count") for row in probes
        ),
        "mean_required_intervening_event_count": _mean(
            row.get("delayed_recurrence_required_intervening_event_count") for row in probes
        ),
        "mean_actual_grace_horizon_before": _mean(
            row.get("delayed_recurrence_actual_grace_horizon_before") for row in probes
        ),
    }


def policy_calibration_summary(turn_rows: Sequence[Mapping[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    """Top-1 confidence calibration for the deployed ensemble policy."""
    rows = [
        row for row in turn_rows
        if row.get("turn_kind") == "robot"
        and isinstance(row.get("final_action_confidence"), (int, float))
        and isinstance(row.get("correct_top1"), bool)
    ]
    if not rows:
        return {
            "definition": "Top-1 confidence calibration on robot turns.",
            "status": "not_run",
            "n_robot_turns": 0,
            "ece": None,
            "top1_brier": None,
        }
    bins: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        confidence = max(0.0, min(1.0, float(row["final_action_confidence"])))
        bins[min(max(0, int(n_bins) - 1), int(confidence * max(1, int(n_bins))))].append(row)
    ece = 0.0
    bin_rows: List[Dict[str, Any]] = []
    for idx in range(max(1, int(n_bins))):
        values = bins.get(idx, [])
        if not values:
            continue
        confidence = _mean(float(row["final_action_confidence"]) for row in values)
        accuracy = _mean(1.0 if row["correct_top1"] else 0.0 for row in values)
        ece += (len(values) / len(rows)) * abs(accuracy - confidence)
        bin_rows.append({"bin": idx, "n": len(values), "mean_confidence": confidence, "accuracy": accuracy})
    return {
        "definition": "Top-1 confidence calibration on robot turns.",
        "status": "completed",
        "n_robot_turns": len(rows),
        "ece": float(ece),
        "top1_brier": _mean(
            (float(row["final_action_confidence"]) - (1.0 if row["correct_top1"] else 0.0)) ** 2
            for row in rows
        ),
        "bins": bin_rows,
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




def active_only_audit_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "definition": "Checks whether the fitted deployable policy matches a fresh active-replay reference.",
            "status": "not_run",
            "n_audits": 0,
            "n_available": 0,
            "n_failed": None,
            "failure_rate": None,
            "max_l1": None,
            "mean_active_variants": None,
            "mean_pruned_variants": None,
            "failed_checkpoints": [],
        }
    available = [row for row in rows if row.get("audit_available")]
    failed = [row for row in available if row.get("primary_active_only_contract_passed") is False]
    return {
        "definition": "Checks whether the fitted deployable policy matches a fresh active-replay reference.",
        "status": "completed" if available else "unavailable",
        "n_audits": len(rows),
        "n_available": len(available),
        "n_failed": len(failed),
        "failure_rate": _safe_div(len(failed), len(available)),
        "max_l1": max(_finite(row.get("max_l1") for row in available), default=0.0),
        "mean_active_variants": _mean(row.get("active_variants") for row in rows),
        "mean_pruned_variants": _mean(row.get("pruned_variants") for row in rows),
        "failed_checkpoints": [
            {"event_idx": row.get("event_idx"), "audit_checkpoint": row.get("audit_checkpoint")}
            for row in failed
        ],
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
        for scope in ("per_hypothesis", "per_transfer_cell", "per_memory_state", "per_event_type", "per_phase_role"):
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
    phase_training = _phase_training_costs(stream.memory_rows, stream.initial_memory)
    phase_role_assist = _add_phase_training_costs(
        _group_metrics(assist_rows, "phase_role"), phase_training,
    )
    phase_rung_assist = _add_phase_rung_training_costs(
        _phase_rung_metrics(assist_rows), phase_training,
    )
    phase_role_all = _add_phase_training_costs(
        _group_metrics(stream.episode_rows, "phase_role"), phase_training,
    )
    phase_rung_all = _add_phase_rung_training_costs(
        _phase_rung_metrics(stream.episode_rows), phase_training,
    )
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
        "per_phase_role": phase_role_assist,
        "per_phase_rung": phase_rung_assist,
        "per_event_type_all_episodes": _group_metrics(stream.episode_rows, "event_type"),
        "per_hypothesis_all_episodes": _group_metrics(stream.episode_rows, "hypothesis_tags"),
        "per_phase_role_all_episodes": phase_role_all,
        "per_phase_rung_all_episodes": phase_rung_all,
        "phase_training_cost": phase_training,
        "per_transfer_cell": _group_metrics(assist_rows, "transfer_cell_before"),
        "per_memory_state": _group_metrics(assist_rows, "memory_state_before"),
        "per_rung_effectiveness": _group_metrics(assist_rows, "no_op_or_duplicate_rung"),
        "per_rung": _group_metrics(stream.episode_rows, "rung_idx"),
        "frozen_eval": frozen_summary(stream.frozen_rows),
        "pre_event_frozen_by_condition": frozen_summary_by(
            [row for row in stream.frozen_rows if row.get("probe_phase") == "pre_event"],
            "condition",
        ),
        "delayed_recurrence_audit": delayed_recurrence_audit_summary(stream.episode_rows),
        "policy_calibration": policy_calibration_summary(stream.turn_rows),
        "memory": memory_snapshot(stream.agent),
        "compute": memory_snapshot(stream.agent, stream.wall_s),
        "axis_value_transfer": axis_value_transfer_summary(assist_rows),
        "mode_schedule": _mode_schedule_summary(stream.episode_rows),
        "active_only_pruned_influence_audit": active_only_audit_summary(stream.active_audit_rows),
        "online_commit_safety": online_commit_safety_summary(stream.episode_rows),
        "reentry_stratification": reentry,
    }
    metrics["paper_hypothesis_views"] = {
        "known_recipe_new_preference_adaptation": metrics["per_hypothesis"].get("known_recipe_new_preference_adaptation", {}),
        "cross_recipe_transfer": metrics["per_hypothesis"].get("cross_recipe_transfer", {}),
        "emergent_axis_composition": metrics["per_hypothesis"].get("emergent_axis_composition", {}),
        "selective_forgetting_reentry": metrics["reentry_stratification"]["confirmed_reentry_from_pruned"],
        "direct_retrieval_control": metrics["per_hypothesis"].get("direct_retrieval_control", {}),
        "delayed_recurrence_interference": metrics["per_condition"].get(
            "random_ladder_delayed_recurrence_interference", {}
        ),
        "delayed_recurrence_pre_event_frozen": metrics["pre_event_frozen_by_condition"].get(
            "random_ladder_delayed_recurrence_interference", {}
        ),
        "delayed_recurrence_memory_audit": metrics["delayed_recurrence_audit"],
        "climb_phase": metrics["per_phase_role"].get("climb", {}),
        "settled_phase": metrics["per_phase_role"].get("settled", {}),
        "holdout_heldout_climb_phase": metrics["per_phase_role"].get("heldout_climb", {}),
        "holdout_heldout_settled_phase": metrics["per_phase_role"].get("heldout_settled", {}),
        "holdout_source_progression_workload": {
            "climb": metrics["per_phase_role_all_episodes"].get("source_climb", {}),
            "settled": metrics["per_phase_role_all_episodes"].get("source_settled", {}),
        },
    }
    return metrics


def _support_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "modes": dict(Counter(str(row.get("mode", "unknown")) for row in rows)),
        "event_types": dict(Counter(str(row.get("event_type", "unknown")) for row in rows)),
        "phase_role": dict(Counter(str(row.get("phase_role", "unphased")) for row in rows)),
        "rung_idx": dict(Counter(str(row.get("rung_idx", "unknown")) for row in rows)),
        "transfer_cell": dict(Counter(str(row.get("transfer_cell_before", "unknown")) for row in rows)),
        "axis_transfer_cell": dict(Counter(str(row.get("axis_transfer_cell_before", "unknown")) for row in rows)),
        "no_op_or_duplicate_rung": dict(Counter(str(row.get("no_op_or_duplicate_rung", "unknown")) for row in rows)),
    }


def _mode_schedule_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Audit whether a baseline received exactly the full interaction route."""
    if not rows:
        return {
            "status": "not_run",
            "n_events": 0,
            "executed_mode_counts": {},
            "n_full_schedule_mismatches": 0,
            "n_extra_observations_prevented": 0,
        }
    matched_rows = [row for row in rows if row.get("full_realized_execution_mode") is not None]
    return {
        "status": "completed",
        "policy": sorted({str(row.get("mode_schedule_policy", "unknown")) for row in rows}),
        "n_events": len(rows),
        "executed_mode_counts": dict(Counter(str(row.get("mode", "unknown")) for row in rows)),
        "natural_mode_counts": dict(Counter(str(row.get("natural_executed_mode", row.get("mode", "unknown"))) for row in rows)),
        "n_full_schedule_constrained_events": len(matched_rows),
        "n_full_schedule_mismatches": sum(
            1 for row in matched_rows if row.get("mode_matches_full_schedule") is not True
        ),
        "n_extra_observations_prevented": sum(
            1 for row in matched_rows
            if row.get("natural_executed_mode") == "observe" and row.get("mode") == "assist"
        ),
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
    if config.match_baseline_execution_modes_to_full and "full" not in baselines:
        raise ValueError(
            "match_baseline_execution_modes_to_full=True requires the deployable 'full' system "
            "to be included in EvaluationConfig.baselines"
        )
    canonical_full_modes: Optional[Tuple[str, ...]] = None
    ordered_baselines = list(baselines)
    if config.match_baseline_execution_modes_to_full:
        ordered_baselines = ["full", *[baseline for baseline in baselines if baseline != "full"]]

    for baseline in ordered_baselines:
        # LCS caches are pure but can otherwise warm a later baseline in a
        # sequential run.  Reset only at stream boundaries, retaining cache
        # benefits within a baseline's own deployment trajectory.
        clear_all_module_caches()
        if baseline == "full" and config.match_baseline_execution_modes_to_full:
            stream = run_event_stream_for_baseline(
                baseline,
                plan,
                config,
                mode_schedule_policy="full_realized_canonical",
            )
            canonical_full_modes = tuple(str(row.get("mode")) for row in stream.episode_rows)
        else:
            stream = run_event_stream_for_baseline(
                baseline,
                plan,
                config,
                execution_mode_schedule=canonical_full_modes,
                mode_schedule_policy=(
                    "matched_full_realized_execution_schedule"
                    if canonical_full_modes is not None else "baseline_local"
                ),
            )
        per_baseline[baseline] = summarize_stream(stream)
        all_episode_rows.extend(stream.episode_rows)
        all_frozen_rows.extend(stream.frozen_rows)
        all_diagnostic_rows.extend(stream.memory_rows + stream.active_audit_rows + stream.oracle_pruning_rows)
        all_turn_rows.extend(stream.turn_rows)

    oracle_summary: Optional[Dict[str, Any]] = None
    if config.include_clairvoyant_oracle:
        clear_all_module_caches()
        stream = run_event_stream_for_baseline(
            CLAIRVOYANT_MEMORY_ORACLE,
            plan,
            config,
            execution_mode_schedule=canonical_full_modes,
            mode_schedule_policy=(
                "matched_full_realized_execution_schedule"
                if canonical_full_modes is not None else "oracle_local"
            ),
        )
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
        all_diagnostic_rows.extend(stream.memory_rows + stream.active_audit_rows + stream.oracle_pruning_rows)
        all_turn_rows.extend(stream.turn_rows)

    axis_rows = axis_value_transfer_rows(all_episode_rows)
    oracle_rows = oracle_gap_rows(per_baseline)
    _append_jsonl(out_dir / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out_dir / "turn_metrics.jsonl", all_turn_rows)
    _append_jsonl(out_dir / "frozen_eval.jsonl", all_frozen_rows)
    _append_jsonl(out_dir / "diagnostics.jsonl", all_diagnostic_rows)
    _append_jsonl(out_dir / "axis_value_transfer_rows.jsonl", axis_rows)
    _append_jsonl(out_dir / "oracle_gap_rows.jsonl", oracle_rows)

    support_baseline = "full" if "full" in baselines else (baselines[0] if baselines else None)
    support_source = [row for row in all_episode_rows if support_baseline and row.get("baseline") == support_baseline]
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
        "mode_schedule": {
            "policy": (
                "full_realized_shared_across_all_baselines"
                if config.match_baseline_execution_modes_to_full else "baseline_local_routing"
            ),
            "canonical_full_mode_counts": (
                dict(Counter(canonical_full_modes)) if canonical_full_modes is not None else None
            ),
            "per_baseline": {
                baseline: summary.get("mode_schedule", {})
                for baseline, summary in sorted(per_baseline.items())
            },
        },
        "n_turn_metric_rows": len(all_turn_rows),
        "per_baseline": per_baseline,
        "oracle_reference": oracle_summary,
        "axis_value_transfer_summary": axis_value_transfer_summary(all_episode_rows),
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
        "experiment_label": config.experiment_label,
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
    parser.add_argument("--offline-pretrained-recipe-fraction", type=float, default=0.50)
    parser.add_argument("--offline-pretrained-preference-fraction", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--native-threads-per-worker", type=int, default=DEFAULT_NATIVE_THREADS_PER_WORKER)
    parser.add_argument("--no-eta", action="store_true")
    parser.add_argument("--no-clairvoyant-oracle", action="store_true")
    parser.add_argument("--n-recipes", type=int, default=15)
    parser.add_argument("--ladder-rungs", type=int, default=9)
    parser.add_argument("--settle-repeat-multiplier-min", type=float, default=2.0)
    parser.add_argument("--settle-repeat-multiplier-max", type=float, default=2.5)
    parser.add_argument("--random-ladder-min-recipes-per-rung", type=int, default=3)
    parser.add_argument("--random-ladder-recurrence-min-prior-conflicts", type=int, default=2)
    parser.add_argument("--random-ladder-recurrence-min-intervening-events", type=int, default=4)
    parser.add_argument("--frozen-eval-period", type=int, default=4)
    parser.add_argument("--frozen-eval-max-pairs", type=int, default=48)
    parser.add_argument("--active-only-audit-period", type=int, default=2)
    parser.add_argument("--active-only-audit-max-prefixes", type=int, default=16)
    parser.add_argument("--active-only-audit-tolerance", type=float, default=5e-2)
    parser.add_argument(
        "--baseline-local-routing",
        action="store_true",
        help="Disable the default full-realized shared interaction schedule (diagnostic only; not comparable headline evaluation).",
    )
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
            "bc_epochs_cold": 1,
            "bc_epochs_warm": 1,
        }
    return EvaluationConfig(
        seeds=tuple(int(seed) for seed in _parse_csv(args.seeds)),
        scenarios=_parse_csv(args.scenarios),
        baselines=_parse_csv(args.baselines),
        offline_pretrained_recipe_fraction=float(args.offline_pretrained_recipe_fraction),
        offline_pretrained_preference_fraction=float(args.offline_pretrained_preference_fraction),
        output_dir=str(args.output_dir),
        workers=int(args.workers),
        native_threads_per_worker=_positive_int(args.native_threads_per_worker, DEFAULT_NATIVE_THREADS_PER_WORKER),
        print_eta=not bool(args.no_eta),
        include_clairvoyant_oracle=not bool(args.no_clairvoyant_oracle),
        n_recipes=int(args.n_recipes),
        ladder_rungs=int(args.ladder_rungs),
        settle_repeat_multiplier_min=float(args.settle_repeat_multiplier_min),
        settle_repeat_multiplier_max=float(args.settle_repeat_multiplier_max),
        random_ladder_min_recipes_per_rung=int(args.random_ladder_min_recipes_per_rung),
        random_ladder_recurrence_min_prior_conflicts=int(args.random_ladder_recurrence_min_prior_conflicts),
        random_ladder_recurrence_min_intervening_events=int(args.random_ladder_recurrence_min_intervening_events),
        frozen_eval_period=int(args.frozen_eval_period),
        frozen_eval_max_pairs=int(args.frozen_eval_max_pairs),
        active_only_audit_period=int(args.active_only_audit_period),
        active_only_audit_max_prefixes=int(args.active_only_audit_max_prefixes),
        active_only_audit_tolerance=float(args.active_only_audit_tolerance),
        match_baseline_execution_modes_to_full=not bool(args.baseline_local_routing),
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
