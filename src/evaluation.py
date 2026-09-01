"""Longitudinal evaluation, diagnostics, and artifact generation."""
from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import random
import re
import subprocess
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_NATIVE_THREADS_PER_WORKER = 1
DEFAULT_RESULTS_ROOT = "eval_results"
RUNS_DIRNAME = "runs"
LATEST_NAME = "latest"
LLM_CUDA_ALLOCATOR_CONFIG = "expandable_segments:True"
# Literal paired seeds keep defaults independent of PRNG implementation details.
PAPER_SEEDS = (
    1337, 2024, 7, 9001, 31415,
)
PAIRED_BOOTSTRAP_SAMPLES = 10_000
DEMOS_PER_GAP = 3
SHORT_DEMO_GAP_MAX = 21
MEDIUM_DEMO_GAP_MAX = 90
# One-factor-at-a-time online-commit sensitivity settings.
COMMIT_SENSITIVITY_SPECS: Tuple[Tuple[str, Mapping[str, float]], ...] = (
    ("default", {}),
    ("tentative_threshold_low", {"tentative_threshold": 0.30}),
    ("tentative_threshold_high", {"tentative_threshold": 0.60}),
    ("full_threshold_low", {"commit_threshold": 0.60}),
    ("full_threshold_high", {"commit_threshold": 0.90}),
    ("match_weight_low", {"match_weight": 0.30}),
    ("match_weight_high", {"match_weight": 0.60}),
)
NATIVE_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    structure: str
    holdout: str
    title: str
    phases: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class BaselineStyle:
    label: str
    color: str


HOMOGENEOUS = "homogeneous"
HETEROGENEOUS = "heterogeneous"
HOLDOUT = "holdout"

HOLDOUT_SOURCES = (
    "prep_first",
    "equipment_jit",
    "serving_early",
    "loading_jit",
    "shutdown_late",
    "cook_start_late",
    "cleanup_first",
    "equipment_jit_serving_early",
)
HOLDOUT_TARGETS = (
    "cleanup_when_free",
    "prep_first_cleanup",
    "prep_first_serving_cleanup",
    "prep_loading_serving_cleanup",
    "equipment_jit_cleanup",
    "prep_loading_cleanup",
)
HOLDOUT_AXIS = "cleanup"
HOLDOUT_VALUE = "when_free"

_DEPLOYMENT_PHASES = (("climb", "climb"), ("settled", "settled"))
_HOLDOUT_PHASES = (("heldout_climb", "held-out climb"), ("heldout_settled", "held-out settled"))
SCENARIO_SPECS: Mapping[str, ScenarioSpec] = {
    spec.name: spec for spec in (
        ScenarioSpec(HOMOGENEOUS, "homogeneous", "none", "Homogeneous longitudinal deployment", _DEPLOYMENT_PHASES),
        ScenarioSpec(HETEROGENEOUS, "heterogeneous", "none", "Heterogeneous longitudinal deployment", _DEPLOYMENT_PHASES),
        ScenarioSpec(HOLDOUT, "controlled", "axis", "Cleanup-axis compositional holdout", _HOLDOUT_PHASES),
    )
}
SCENARIOS = tuple(SCENARIO_SPECS)

DEFAULT_BASELINES = (
    "full", "frozen", "offline_default",
    "unpinned", "latest", "fixed", "no_decay", "bc", "ewc",
    "replay_bc",
)
MEMORY_ORACLE = "memory_oracle"
BASELINE_ORDER = (
    "full", "in_context_llm", "no_decay", "replay_bc", "bc", "unpinned", "ewc",
    "latest", "fixed", "offline_default",
    "frozen", MEMORY_ORACLE,
)
BASELINE_STYLES: Mapping[str, BaselineStyle] = {
    name: BaselineStyle(label, color) for name, label, color in (
        ("full", "Full", "#0072B2"),
        ("in_context_llm", "In-context LLM", "#6A3D9A"),
        ("no_decay", "No decay", "#009E73"),
        ("replay_bc", "ER-BC", "#D55E00"),
        ("bc", "BC", "#CC79A7"),
        ("unpinned", "Adaptive", "#56B4E9"),
        ("ewc", "EWC", "#E69F00"),
        ("latest", "Latest", "#999999"),
        ("fixed", "Fixed", "#C44E52"),
        ("offline_default", "Offline-ID", "#8172B3"),
        ("frozen", "Offline-pre", "#64B5CD"),
        (MEMORY_ORACLE, "Oracle†", "#222222"),
    )
}
DEPLOYABLE_BASELINES = tuple(
    name for name in BASELINE_ORDER if name not in {"full", MEMORY_ORACLE}
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return str(value)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(path, encoded)


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _finite(values: Iterable[Any]) -> List[float]:
    return [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def _mean(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = _finite(values)
    return float(sum(numbers) / len(numbers)) if numbers else float(default)


def _mean_std(values: Iterable[Any]) -> Tuple[float, float]:
    numbers = _finite(values)
    if not numbers:
        return 0.0, 0.0
    average = _mean(numbers)
    if len(numbers) == 1:
        return average, 0.0
    variance = sum((value - average) ** 2 for value in numbers) / (len(numbers) - 1)
    return average, math.sqrt(variance)


def _value(value: Any, default: float = 0.0) -> float:
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
        else float(default)
    )


def _numeric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return _value(row.get(key), default)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _p95(values: Iterable[Any]) -> float:
    numbers = sorted(_finite(values))
    if not numbers:
        return 0.0
    index = min(len(numbers) - 1, int(math.ceil(0.95 * len(numbers))) - 1)
    return numbers[index]


def bootstrap(
    values: Sequence[Any], *, key: str, draws: int = 100_000,
) -> np.ndarray:
    """Return deterministic resampled means for a stable experiment key."""
    numbers = np.asarray(_finite(values), dtype=float)
    if numbers.size == 0:
        return np.empty(0, dtype=float)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.choice(
        numbers, size=(max(1, int(draws)), numbers.size), replace=True,
    ).mean(axis=1)


def _bootstrap_ci(
    values: Sequence[Any], *, key: str, draws: int = 100_000,
) -> Tuple[float, float]:
    means = bootstrap(values, key=key, draws=draws)
    if means.size == 0:
        return 0.0, 0.0
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _sign_flip_test(deltas: Sequence[float]) -> Tuple[float, float]:
    """Return exact one-sided and two-sided paired randomization p-values."""
    values = [float(value) for value in deltas if abs(float(value)) > 1e-15]
    if not values:
        return 1.0, 1.0
    observed = sum(values) / len(values)
    draws = [
        sum(sign * value for sign, value in zip(signs, values)) / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    one_sided = sum(draw >= observed - 1e-15 for draw in draws) / len(draws)
    two_sided = sum(abs(draw) >= abs(observed) - 1e-15 for draw in draws) / len(draws)
    return one_sided, two_sided


def _holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    running = 0.0
    adjusted: Dict[str, float] = {}
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_value))
        adjusted[name] = running
    return adjusted


def _summary_metric_values(
    summaries: Sequence[Mapping[str, Any]],
    baseline: str,
    phase: str,
    metric: str,
) -> List[float]:
    scope = "by_holdout_phase_role" if phase.startswith("heldout_") else "by_phase_role"
    return [
        _value(
            summary["per_baseline"][baseline]["assist"][scope]
            .get(phase, {}).get(metric)
        )
        for summary in summaries
    ]


def compare_groups(
    scenario: str, summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Phase-specific, seed-paired inference for one holdout cell."""
    endpoints = (
        ("heldout_climb", "teacher_forced_top_1", "higher_is_better", "confirmatory"),
        ("heldout_settled", "teacher_forced_top_1", "higher_is_better", "secondary"),
        ("heldout_climb", "normalized_human_action_load", "lower_is_better", "secondary"),
        ("heldout_settled", "normalized_human_action_load", "lower_is_better", "secondary"),
    )
    common = set.intersection(*(set(summary["per_baseline"]) for summary in summaries))
    comparators = [baseline for baseline in DEPLOYABLE_BASELINES if baseline in common]
    tests: Dict[str, Any] = {}
    for phase, metric, direction, role in endpoints:
        family: Dict[str, Any] = {}
        raw_p: Dict[str, float] = {}
        full_values = _summary_metric_values(summaries, "full", phase, metric)
        for baseline in comparators:
            baseline_values = _summary_metric_values(summaries, baseline, phase, metric)
            deltas = [
                (full - other) if direction == "higher_is_better" else (other - full)
                for full, other in zip(full_values, baseline_values)
            ]
            one_sided, two_sided = _sign_flip_test(deltas)
            ci_low, ci_high = _bootstrap_ci(
                deltas, key=f"{scenario}|{phase}|{metric}|{baseline}",
            )
            raw_p[baseline] = one_sided
            mean, standard_deviation = _mean_std(deltas)
            family[baseline] = {
                "full_advantage_direction": direction,
                "mean_full_advantage": mean,
                "std_full_advantage": standard_deviation,
                "paired_seed_deltas_full_advantage": {
                    str(summary["seed"]): delta
                    for summary, delta in zip(summaries, deltas)
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
            "comparison_family": f"Full versus {len(comparators)} deployable baselines",
            "results": family,
        }
    return {
        "scenario": scenario,
        "experimental_unit": f"paired seed-level phase aggregate (n={len(summaries)})",
        "confirmatory_hypothesis": "Full has higher held-out-climb teacher-forced Top-1 than each deployable baseline.",
        "secondary_endpoints": [
            "held-out-settled teacher-forced Top-1",
            "held-out-climb normalized human action load",
            "held-out-settled normalized human action load",
        ],
        "test": "Exact paired sign-flip randomization test; one-sided p-values are Holm-adjusted within each endpoint.",
        "caution": "Inference may be low-powered. Bootstrap intervals are descriptive and do not replace the exact tests.",
        "endpoints": tests,
    }


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
    thread_count = _positive_int(
        threads, DEFAULT_NATIVE_THREADS_PER_WORKER,
    )
    _set_native_thread_env(thread_count, override=True)
    return {
        "threads": thread_count,
        "env": {var: os.environ.get(var) for var in NATIVE_THREAD_ENV_VARS},
        "openblas_runtime_limited": _set_loaded_openblas_threads(thread_count),
    }


_set_native_thread_env(DEFAULT_NATIVE_THREADS_PER_WORKER, override=False)

from .adaptive_agent import AdaptiveAgent
from .ablations import parse_commit_records, summarize_commit_decisions
from .baselines import BASELINE_AGENTS
from .environment import recipe_builders
from .hrc_simulation import DEFAULT_TIMING, simulate_episode
from .memory import MemoryItem, ReplayMemory, VariantKey, clear_caches, make_variant_id
from .models import DEFAULT_SETTINGS, Settings
from .preferences import DEFAULT_PREFERENCE, PREFERENCE_IDS, PREFERENCES, apply_preset
from .representations import (
    ACTION_REPRESENTATION,
    Observation,
    observe_actions,
)


TaskKey = Tuple[str, str]


@dataclass(frozen=True)
class ScheduleSettings:
    panel_size: int = 20
    phases: int = 7
    demos: int = 210
    min_recipes: int = 5
    max_recipes: int = 8
    transition_min: float = 0.20
    transition_max: float = 0.25
    gap_allocation: float = 0.80
    hetero_gap_mean: int = 15
    hetero_gap_min: int = 3
    hetero_gap_max: int = 40
    hetero_gap_shape: float = 0.35
    climb_decay: float = 0.50
    active_size_weights: Tuple[float, float, float] = (0.90, 0.08, 0.02)
    lifecycle_weights: Tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)
    reentry_rate: float = 0.05
    recipe_skew: float = 0.70
    pair_skew: float = 0.70
    holdout_start: float = 0.60
    holdout_demos: int = 3
    max_attempts: int = 1000

    def validate(self, recipe_count: int) -> None:
        if self.phases < 2:
            raise ValueError("schedule generation requires at least two phases")
        if self.demos <= 0:
            raise ValueError("demos must be positive")
        if self.demos % DEMOS_PER_GAP:
            raise ValueError("demos must be divisible by three")
        if self.demos // DEMOS_PER_GAP < self.phases:
            raise ValueError("the fixed gap budget must provide at least one unit per phase")
        if self.holdout_demos < 1:
            raise ValueError("holdout_demos must be positive")
        if not 0.20 <= self.transition_min <= self.transition_max <= 0.25:
            raise ValueError("transition fractions must satisfy 0.20 <= min <= max <= 0.25")
        if math.ceil(self.transition_min * self.demos) > math.floor(
            self.transition_max * self.demos
        ):
            raise ValueError("total demonstrations provide no integer climb budget in the configured range")
        if self.min_recipes < 1 or self.min_recipes > recipe_count:
            raise ValueError("invalid minimum recipes per phase")
        if self.max_recipes < self.min_recipes:
            raise ValueError("maximum recipes per phase is smaller than minimum")
        if self.gap_allocation <= 0.0:
            raise ValueError("gap_allocation must be positive")
        if not (
            1 <= self.hetero_gap_min
            <= self.hetero_gap_mean
            <= self.hetero_gap_max
        ):
            raise ValueError(
                "heterogeneous gap must satisfy 1 <= min <= mean <= max"
            )
        if self.hetero_gap_shape <= 0.0:
            raise ValueError("hetero_gap_shape must be positive")
        if not 0.0 < self.climb_decay < 1.0:
            raise ValueError("climb_decay must lie between zero and one")
        if len(self.active_size_weights) != 3 or any(weight < 0.0 for weight in self.active_size_weights):
            raise ValueError("active-cardinality weights must be three nonnegative values")
        if sum(self.active_size_weights) <= 0.0:
            raise ValueError("active-cardinality weights must have positive mass")
        max_active_size = max(
            index + 1
            for index, weight in enumerate(self.active_size_weights)
            if weight > 0.0
        )
        maximum_climb_support = math.floor(
            self.transition_max * self.demos
        )
        if maximum_climb_support < self.phases * max_active_size:
            raise ValueError(
                "the climb budget cannot reserve one complete active set per phase; "
                "increase demos, reduce phases, or lower the active-cardinality ceiling"
            )
        if len(self.lifecycle_weights) != 4 or any(
            value < 0.0 for value in self.lifecycle_weights
        ) or not math.isclose(sum(self.lifecycle_weights), 1.0, abs_tol=1e-9):
            raise ValueError("retain/add/remove/swap probabilities must be nonnegative and sum to one")
        if not 0.0 <= self.reentry_rate <= 1.0:
            raise ValueError("conditional swap reintroduction probability must lie in [0, 1]")
        for name, value in (("holdout_start", self.holdout_start),):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.recipe_skew <= 0.0 or self.pair_skew <= 0.0:
            raise ValueError("recipe and settled-pair weight shapes must be positive")


@dataclass(frozen=True)
class Holdout:
    kind: str
    preference: Optional[str]
    axis: Optional[str]
    axis_value: Optional[str]
    introduction_phase: Optional[int]
    required_source_preferences: Tuple[str, ...] = ()
    introduction_step: Optional[int] = None
    source_preferences: Tuple[str, ...] = ()
    target_preferences: Tuple[str, ...] = ()


def _holdout_step(target: Holdout) -> Optional[int]:
    return (
        target.introduction_step
        if target.introduction_step is not None
        else target.introduction_phase
    )


def _is_holdout_target(preference: str, target: Holdout) -> bool:
    if target.target_preferences:
        return preference in target.target_preferences
    return target.preference == preference


@dataclass(frozen=True)
class LifecycleChange:
    operation: str
    reason: str
    can_reenter: bool = False


@dataclass
class Lifecycle:
    active: set[str] = field(default_factory=set)
    ever: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Phase:
    index: int
    stage: int
    climb_recipes: Tuple[str, ...]
    gap: int
    start_demo: int
    recipes: Tuple[str, ...]
    requested_recipes: int
    climb_limited: bool
    climb_capacity: int
    active_preferences: Mapping[str, Tuple[str, ...]]
    climb_pairs: Tuple[TaskKey, ...]
    settled_pairs: Tuple[TaskKey, ...]
    lifecycle_by_pair: Mapping[str, str]
    added_pairs: Tuple[str, ...]
    retained_pairs: Tuple[str, ...]
    removed_pairs: Tuple[str, ...]
    reintroduced_pairs: Tuple[str, ...]
    settle_counts: Mapping[str, int]
    settle_weights: Mapping[str, Mapping[str, float]]
    omitted_pairs: Mapping[str, str]
    transitions: Mapping[str, LifecycleChange]
    transition_fraction: float


@dataclass(frozen=True)
class DeploymentSchedule:
    seed: int
    user_id: str
    structure: str
    holdout: Holdout
    phases: Tuple[Phase, ...]
    candidates: Tuple[str, ...]
    recipe_panel: Tuple[str, ...]
    recipe_popularity: Mapping[str, float]
    transition_fraction: float
    demos: int
    holdout_equivalents: Mapping[str, Tuple[str, ...]]
    config: ScheduleSettings
    attempts: int


@dataclass
class _PhasePlan:
    index: int
    stage: int
    climb_recipes: Tuple[str, ...]
    recipes: Tuple[str, ...]
    requested_recipes: int
    climb_limited: bool
    climb_capacity: int
    active_preferences: Dict[str, Tuple[str, ...]]
    active_pairs: Tuple[TaskKey, ...]
    climb_pairs: Tuple[TaskKey, ...]
    lifecycle_by_pair: Dict[str, str]
    added_pairs: Tuple[str, ...]
    retained_pairs: Tuple[str, ...]
    removed_pairs: Tuple[str, ...]
    reintroduced_pairs: Tuple[str, ...]
    omitted_pairs: Dict[str, str]
    transitions: Dict[str, LifecycleChange]


def _pair_label(pair: TaskKey) -> str:
    return f"{pair[0]}/{pair[1]}"


def _weighted_sample_without_replacement(
    values: Sequence[str],
    weights: Mapping[str, float],
    count: int,
    rng: random.Random,
) -> Tuple[str, ...]:
    remaining = list(values)
    selected = []
    for _ in range(min(max(0, count), len(remaining))):
        draw_weights = [max(1e-12, float(weights[value])) for value in remaining]
        chosen = rng.choices(remaining, weights=draw_weights, k=1)[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def _cardinality(config: ScheduleSettings, maximum: int, rng: random.Random) -> int:
    maximum = max(1, min(3, int(maximum)))
    choices = list(range(1, maximum + 1))
    weights = list(config.active_size_weights[:maximum])
    return int(rng.choices(choices, weights=weights, k=1)[0])


def _max_active_size(config: ScheduleSettings) -> int:
    """Largest active-set size supported by the behavioral cardinality prior."""
    return max(
        index + 1
        for index, weight in enumerate(config.active_size_weights)
        if weight > 0.0
    )


def _preference_axes(
    preference: str,
    preference_axis_values: Mapping[str, Mapping[str, str]],
    default_axis_values: Mapping[str, str],
) -> Dict[str, str]:
    return {
        axis: value
        for axis, value in preference_axis_values[preference].items()
        if value != default_axis_values.get(axis)
    }


def _select_holdout_target(
    holdout_kind: str,
    preferences: Sequence[str],
    effective_preferences_by_recipe: Mapping[str, Sequence[str]],
    preference_axis_values: Mapping[str, Mapping[str, str]],
    default_axis_values: Mapping[str, str],
    introduction_phase: int,
    rng: random.Random,
) -> Holdout:
    if holdout_kind == "none":
        return Holdout("none", None, None, None, None)

    support = Counter(
        preference
        for recipe_preferences in effective_preferences_by_recipe.values()
        for preference in recipe_preferences
    )
    if holdout_kind == "preference":
        options = []
        for preference in preferences:
            target_axes = _preference_axes(preference, preference_axis_values, default_axis_values)
            if len(target_axes) < 2 or support[preference] <= 0:
                continue
            source_options = []
            complete_source_coverage = True
            for axis, value in target_axes.items():
                isolated = [
                    candidate for candidate in preferences
                    if support[candidate] > 0
                    and _preference_axes(candidate, preference_axis_values, default_axis_values) == {axis: value}
                ]
                if isolated:
                    source_options.append(tuple(sorted(isolated)))
                else:
                    complete_source_coverage = False
            if complete_source_coverage and len(source_options) <= introduction_phase:
                options.append((len(target_axes), preference, tuple(source_options)))
        if not options:
            raise ValueError("preference holdout requires an effective composed preference")
        # Keep compositional complexity matched across paired seeds.
        minimum_axis_count = min(axis_count for axis_count, _get_preference, _required in options)
        matched_complexity = [
            (preference, source_options)
            for axis_count, preference, source_options in options
            if axis_count == minimum_axis_count
        ]
        target, source_options = rng.choice(sorted(matched_complexity))
        required = tuple(
            rng.choice(list(candidates)) for candidates in source_options
        )
        return Holdout(
            kind="preference",
            preference=target,
            axis=None,
            axis_value=None,
            introduction_phase=introduction_phase,
            required_source_preferences=required,
        )

    if holdout_kind == "axis":
        isolated = []
        for preference in preferences:
            axes = _preference_axes(preference, preference_axis_values, default_axis_values)
            if len(axes) == 1 and support[preference] > 0:
                axis, value = next(iter(axes.items()))
                isolated.append((preference, axis, value))
        if not isolated:
            raise ValueError("axis holdout requires an effective isolated-axis preference")
        target, axis, value = rng.choice(sorted(isolated))
        return Holdout("axis", target, axis, value, introduction_phase)

    raise ValueError(f"unknown holdout kind {holdout_kind!r}")


def _blocked_before_introduction(
    preference: str,
    phase_index: int,
    target: Holdout,
    preference_axis_values: Mapping[str, Mapping[str, str]],
) -> bool:
    if target.kind == "none" or target.introduction_phase is None or phase_index >= target.introduction_phase:
        return False
    if target.kind == "preference":
        return preference == target.preference
    assert target.axis is not None
    return preference_axis_values[preference].get(target.axis) == target.axis_value


def _holdout_equivalents(
    target: Holdout,
    recipes: Sequence[str],
    effective_preferences_by_recipe: Mapping[str, Sequence[str]],
    applicability_by_recipe: Mapping[str, Mapping[str, str]],
) -> Dict[str, Tuple[str, ...]]:
    """Resolve recipe-specific behavioral aliases of a preference holdout."""
    if target.kind != "preference" or target.preference is None:
        return {}

    equivalents: Dict[str, Tuple[str, ...]] = {}
    for recipe in recipes:
        effective = set(effective_preferences_by_recipe[recipe])
        status = str(applicability_by_recipe[recipe].get(target.preference, ""))
        aliases: set[str] = set()
        if target.preference in effective:
            aliases.add(target.preference)
        elif status == "recipe_specific_noop" and "default" in effective:
            aliases.add("default")
        elif status.startswith("semantic_duplicate_of:"):
            representative = status.split(":", 1)[1]
            if representative in effective:
                aliases.add(representative)
        equivalents[recipe] = tuple(sorted(aliases))
    return equivalents


def _holdout_blocked(
    recipe: str,
    preference: str,
    phase_index: int,
    target: Holdout,
    semantic_equivalents_by_recipe: Mapping[str, Sequence[str]],
) -> bool:
    return bool(
        target.kind == "preference"
        and target.introduction_phase is not None
        and phase_index < target.introduction_phase
        and preference in semantic_equivalents_by_recipe.get(recipe, ())
    )


def _sample_lifecycle_operation(
    config: ScheduleSettings, rng: random.Random,
) -> str:
    return rng.choices(
        ("retention", "addition", "removal", "swap"),
        weights=config.lifecycle_weights,
        k=1,
    )[0]


def _evolve_active_set(
    state: Lifecycle,
    candidates: Sequence[str],
    forced: Sequence[str],
    config: ScheduleSettings,
    rng: random.Random,
    *,
    operation: Optional[str] = None,
    allow_reintroduction: bool = True,
    max_active_size: int = 3,
) -> Tuple[set[str], Dict[str, set[str]], str]:
    """Apply one explicitly sampled add, remove, or swap operation.

    Operations are never silently converted or renormalized around active-set
    boundaries.  An infeasible sampled operation invalidates the schedule
    attempt so the generator can use another recipe panel or lifecycle draw.
    Reintroduction is a conditional incoming-source choice within ``swap``.
    """
    candidate_set = set(candidates)
    max_active_size = max(1, min(3, int(max_active_size)))
    previous = set(state.active) & candidate_set
    forced_set = set(forced) & candidate_set
    old_removed = set(state.removed)
    if not candidate_set:
        raise RuntimeError("no effective preference candidate is available")

    if not previous:
        operation = "initialization"
        desired = min(
            max_active_size,
            _cardinality(config, len(candidate_set - old_removed), rng),
        )
        available = sorted(candidate_set - old_removed)
        if not available:
            raise RuntimeError("initialization has no non-removed preference candidate")
        rng.shuffle(available)
        next_active = set(forced_set)
        if len(next_active) > max_active_size:
            raise RuntimeError("forced initialization exceeds the active-cardinality ceiling")
        desired = max(desired, len(next_active))
        for preference in available:
            if len(next_active) >= desired:
                break
            next_active.add(preference)
    elif len(previous) > max_active_size and not (forced_set - previous):
        next_active = set(previous)
        removable = sorted(next_active - forced_set)
        while len(next_active) > max_active_size and removable:
            removed = rng.choice(removable)
            removable.remove(removed)
            next_active.remove(removed)
        if len(next_active) > max_active_size:
            raise RuntimeError("active-cardinality ceiling cannot retain all forced preferences")
        operation = "removal"
    elif forced_set - previous:
        next_active = set(previous)
        incoming = sorted(forced_set - previous)
        returning = any(preference in old_removed for preference in incoming)
        removed_for_swap = False
        for preference in incoming:
            # A returning forced preference remains a swap subtype even when
            # spare capacity exists; reintroduction is never a silent add.
            must_swap = preference in old_removed or len(next_active) >= max_active_size
            while must_swap:
                removable = sorted(next_active - forced_set)
                if not removable:
                    raise RuntimeError("forced preferences exceed the maximum active-set cardinality")
                next_active.remove(rng.choice(removable))
                removed_for_swap = True
                must_swap = len(next_active) >= max_active_size
            next_active.add(preference)
        operation = "swap" if removed_for_swap or returning else "addition"
    else:
        fresh = sorted(candidate_set - state.ever - old_removed - previous)
        returning = sorted((old_removed & candidate_set) - previous)
        operation = operation or _sample_lifecycle_operation(config, rng)
        if operation == "retention":
            raise RuntimeError("retention must not enter the active-set mutator")
        next_active = set(previous)
        if operation == "addition":
            if not fresh or len(previous) >= max_active_size:
                raise RuntimeError("sampled addition is infeasible for the active preference set")
            next_active.add(rng.choice(fresh))
        elif operation == "removal":
            removable = sorted(next_active - forced_set)
            if len(next_active) <= 1 or not removable:
                raise RuntimeError("sampled removal is infeasible for the active preference set")
            next_active.remove(rng.choice(removable))
        elif operation == "swap":
            reintroduce_now = bool(
                allow_reintroduction
                and returning
                and rng.random() < config.reentry_rate
            )
            incoming_pool = returning if reintroduce_now else fresh
            removable = sorted(next_active - forced_set)
            if not incoming_pool or not removable:
                subtype = "reintroduction" if reintroduce_now else "ordinary"
                raise RuntimeError(
                    f"sampled {subtype} swap is infeasible for the active preference set"
                )
            incoming = rng.choice(incoming_pool)
            next_active.remove(rng.choice(removable))
            next_active.add(incoming)

    if not next_active or next_active == previous:
        raise RuntimeError("lifecycle change did not alter the active preference set")

    retained_now = next_active & previous
    entered = next_active - previous
    reintroduced_now = entered & old_removed
    added_now = entered - old_removed
    removed_now = previous - next_active
    state.active = next_active
    state.ever.update(next_active)
    state.removed.update(removed_now)
    state.removed.difference_update(next_active)
    return next_active, {
        "added": added_now,
        "retained": retained_now,
        "removed": removed_now,
        "reintroduced": reintroduced_now,
    }, operation


def _min_phase_gap(
    draft: _PhasePlan,
    target: Holdout,
    config: ScheduleSettings,
) -> int:
    """Return the minimum gap covering climb and settled support."""
    climb_count = len(draft.climb_pairs)
    required_settled = len(draft.active_pairs)
    if draft.index == _holdout_step(target) and target.kind != "none":
        assert target.preference is not None
        target_pairs = sum(
            _is_holdout_target(preference, target)
            for _recipe, preference in draft.active_pairs
        )
        required_settled += max(
            0, config.holdout_demos - target_pairs,
        )
    required_demos = climb_count + required_settled
    return max(1, math.ceil(required_demos / DEMOS_PER_GAP))


def _allocate_integer_budget(
    total: int, minimum: Sequence[int], shape: float, rng: random.Random,
) -> Tuple[int, ...]:
    allocation = [max(1, int(value)) for value in minimum]
    remaining = int(total) - sum(allocation)
    if remaining < 0:
        raise ValueError("budget cannot provide the requested minima")
    weights = [
        max(1e-12, rng.gammavariate(shape, 1.0)) for _value in minimum
    ]
    for _ in range(remaining):
        allocation[rng.choices(range(len(allocation)), weights=weights, k=1)[0]] += 1
    return tuple(allocation)


def _allocate_bounded_budget(
    total: int,
    minimum: Sequence[int],
    maximum: int,
    shape: float,
    rng: random.Random,
) -> Tuple[int, ...]:
    allocation = [max(1, int(value)) for value in minimum]
    maximum = int(maximum)
    if any(value > maximum for value in allocation):
        raise ValueError("phase support exceeds the maximum gap")
    remaining = int(total) - sum(allocation)
    capacity = sum(maximum - value for value in allocation)
    if remaining < 0 or remaining > capacity:
        raise ValueError("gap budget is incompatible with its phase bounds")
    weights = [
        max(1e-12, rng.gammavariate(shape, 1.0)) for _value in minimum
    ]
    while remaining:
        eligible = [
            index for index, value in enumerate(allocation)
            if value < maximum
        ]
        chosen = rng.choices(
            eligible,
            weights=[weights[index] for index in eligible],
            k=1,
        )[0]
        allocation[chosen] += 1
        remaining -= 1
    return tuple(allocation)


def _sample_phase_gap(
    config: ScheduleSettings,
    rng: random.Random,
    minimum_by_phase: Sequence[int],
) -> Tuple[int, ...]:
    """Sample a fixed-budget composition above realized support minima."""
    if len(minimum_by_phase) != config.phases:
        raise ValueError("minimum_by_phase must contain one value per phase")
    return _allocate_integer_budget(
        config.demos // DEMOS_PER_GAP,
        minimum_by_phase,
        config.gap_allocation,
        rng,
    )


def _sample_hetero_gap(
    config: ScheduleSettings,
    rng: random.Random,
    minimum_by_phase: Sequence[int],
) -> Tuple[int, ...]:
    minimum = [
        max(int(config.hetero_gap_min), int(value))
        for value in minimum_by_phase
    ]
    return _allocate_bounded_budget(
        int(config.hetero_gap_mean) * len(minimum),
        minimum,
        int(config.hetero_gap_max),
        float(config.hetero_gap_shape),
        rng,
    )


def _allocate_climb_budget(
    drafts: Sequence[_PhasePlan],
    config: ScheduleSettings,
    recipe_popularity: Mapping[str, float],
    rng: random.Random,
) -> int:
    """Fill the climb budget by repeating changed-pair support."""
    minimum = int(math.ceil(config.transition_min * config.demos))
    maximum = int(math.floor(config.transition_max * config.demos))
    mandatory = sum(len(draft.climb_pairs) for draft in drafts)
    if mandatory > maximum:
        raise ValueError(
            "mandatory changed-recipe support exceeds the configured climb budget; "
            "reduce heterogeneous change probability or increase total demonstrations"
        )
    target = rng.randint(max(minimum, mandatory), maximum)
    candidates = [
        (draft.index, pair)
        for draft in drafts
        for pair in draft.climb_pairs
    ]
    if not candidates:
        raise RuntimeError("a schedule must contain climb support")
    extras_by_phase: Dict[int, List[TaskKey]] = defaultdict(list)
    for _ in range(target - mandatory):
        phase_index, pair = rng.choices(
            candidates,
            weights=[recipe_popularity[item[1][0]] for item in candidates],
            k=1,
        )[0]
        extras_by_phase[phase_index].append(pair)
    for draft in drafts:
        climb = list(draft.climb_pairs) + extras_by_phase.get(draft.index, [])
        rng.shuffle(climb)
        draft.climb_pairs = tuple(climb)
    return target


def _repeat_phase_support(
    active_pairs: Sequence[TaskKey],
    total: int,
    recipe_popularity: Mapping[str, float],
    rng: random.Random,
    *,
    pair_weight_shape: float,
    priority_preference: Optional[str] = None,
    minimum_priority_count: int = 0,
) -> Tuple[Tuple[TaskKey, ...], Dict[str, int], Dict[str, Dict[str, float]]]:
    if total < len(active_pairs):
        raise ValueError("settled budget cannot cover every active pair")
    order = list(active_pairs)
    raw_pair_weights = {
        pair: max(
            1e-12,
            recipe_popularity[pair[0]] * rng.gammavariate(pair_weight_shape, 1.0),
        )
        for pair in active_pairs
    }
    extra = total - len(active_pairs)
    priority_pairs = [pair for pair in active_pairs if pair[1] == priority_preference]
    priority_present = sum(pair[1] == priority_preference for pair in order)
    priority_needed = max(0, int(minimum_priority_count) - priority_present)
    if priority_needed > extra:
        raise ValueError("settled allocation cannot meet holdout support minimum")
    for _ in range(priority_needed):
        order.append(rng.choice(priority_pairs))
    extra -= priority_needed
    population = list(active_pairs)
    population_weights = [raw_pair_weights[pair] for pair in population]
    order.extend(rng.choices(population, weights=population_weights, k=extra))
    rng.shuffle(order)
    counts = Counter(_pair_label(pair) for pair in order)
    by_recipe: Dict[str, Counter[str]] = {}
    for recipe, preference in order:
        by_recipe.setdefault(recipe, Counter())[preference] += 1
    weights = {
        recipe: {
            preference: count / sum(recipe_counts.values())
            for preference, count in sorted(recipe_counts.items())
        }
        for recipe, recipe_counts in sorted(by_recipe.items())
    }
    return tuple(order), dict(counts), weights


def _source_forcing(target: Holdout) -> Dict[int, str]:
    if target.kind != "preference" or target.introduction_phase is None:
        return {}
    return {
        phase: preference
        for phase, preference in enumerate(target.required_source_preferences)
        if phase < target.introduction_phase
    }


def _plan_drafts(
    structure: str,
    target: Holdout,
    recipes: Sequence[str],
    effective_preferences_by_recipe: Mapping[str, Sequence[str]],
    applicability_by_recipe: Mapping[str, Mapping[str, str]],
    preference_axis_values: Mapping[str, Mapping[str, str]],
    semantic_equivalents_by_recipe: Mapping[str, Sequence[str]],
    recipe_popularity: Mapping[str, float],
    config: ScheduleSettings,
    rng: random.Random,
) -> Optional[Tuple[_PhasePlan, ...]]:
    state_by_recipe = {recipe: Lifecycle() for recipe in recipes}
    shared_state = Lifecycle()
    seen_pair_labels: set[str] = set()
    removed_pair_labels: set[str] = set()
    source_forcing = _source_forcing(target)
    drafts = []
    max_active_size = _max_active_size(config)

    globally_effective = sorted({
        preference
        for recipe_preferences in effective_preferences_by_recipe.values()
        for preference in recipe_preferences
    })
    for phase_index in range(config.phases):
        requested_subgroup_size = rng.randint(
            min(len(recipes), config.min_recipes),
            min(len(recipes), config.max_recipes),
        )
        subgroup_size = requested_subgroup_size
        climb_limited = False
        phase_recipes = list(_weighted_sample_without_replacement(
            recipes, recipe_popularity, subgroup_size, rng,
        ))
        forced_preference = source_forcing.get(phase_index)
        if forced_preference is not None:
            source_already_exposed = (
                forced_preference in shared_state.ever
                if structure == "homogeneous"
                else any(
                    forced_preference in recipe_state.ever
                    for recipe_state in state_by_recipe.values()
                )
            )
            if source_already_exposed:
                forced_preference = None
        if target.introduction_phase == phase_index:
            forced_preference = target.preference
        if forced_preference is not None:
            compatible = [
                recipe for recipe in recipes
                if forced_preference in effective_preferences_by_recipe[recipe]
            ]
            if not compatible:
                return None
            if not any(recipe in compatible for recipe in phase_recipes):
                phase_recipes[-1] = rng.choice(compatible)
                phase_recipes = list(dict.fromkeys(phase_recipes))
                while len(phase_recipes) < subgroup_size:
                    remaining = [recipe for recipe in recipes if recipe not in phase_recipes]
                    phase_recipes.append(rng.choice(remaining))
        rng.shuffle(phase_recipes)

        lifecycle_by_pair: Dict[str, str] = {}
        active_for_phase: Dict[str, Tuple[str, ...]] = {}
        added_pairs = []
        retained_pairs = []
        removed_pairs = []
        reintroduced_pairs = []
        transitions: Dict[str, LifecycleChange] = {}
        reintroduction_eligible_by_recipe: Dict[str, bool] = {}
        changed_recipes: set[str] = set()

        if structure == "homogeneous":
            candidates = [
                preference for preference in globally_effective
                if not _blocked_before_introduction(
                    preference, phase_index, target, preference_axis_values,
                )
            ]
            forced = [forced_preference] if forced_preference is not None else []
            shared_reintroduction_eligible = bool(shared_state.removed & set(candidates))
            requested_operation = (
                _sample_lifecycle_operation(config, rng)
                if shared_state.active and not forced else None
            )
            if requested_operation == "retention":
                delta = {
                    "added": set(),
                    "retained": set(shared_state.active),
                    "removed": set(),
                    "reintroduced": set(),
                }
                operation = "retention"
            else:
                try:
                    _active, delta, operation = _evolve_active_set(
                        shared_state,
                        candidates, forced, config, rng,
                        operation=requested_operation,
                        allow_reintroduction=True,
                        max_active_size=max_active_size,
                    )
                except RuntimeError:
                    return None
            eligible_recipes = [
                recipe for recipe in recipes
                if any(
                    preference in effective_preferences_by_recipe[recipe]
                    for preference in shared_state.active
                )
                and not any(
                    _holdout_blocked(
                        recipe,
                        preference,
                        phase_index,
                        target,
                        semantic_equivalents_by_recipe,
                    )
                    for preference in shared_state.active
                )
            ]
            phase_recipes = [
                recipe for recipe in phase_recipes if recipe in eligible_recipes
            ]
            replacements = [
                recipe for recipe in eligible_recipes if recipe not in phase_recipes
            ]
            phase_recipes.extend(_weighted_sample_without_replacement(
                replacements,
                recipe_popularity,
                subgroup_size - len(phase_recipes),
                rng,
            ))
            if len(phase_recipes) < subgroup_size:
                return None
            rng.shuffle(phase_recipes)
            if forced_preference is not None:
                if not any(
                    forced_preference in effective_preferences_by_recipe[recipe]
                    for recipe in phase_recipes
                ):
                    compatible_replacements = [
                        recipe for recipe in eligible_recipes
                        if recipe not in phase_recipes
                        and forced_preference in effective_preferences_by_recipe[recipe]
                    ]
                    if not compatible_replacements:
                        return None
                    phase_recipes[-1] = rng.choice(compatible_replacements)
            rng.shuffle(phase_recipes)
            newly_removed = {
                label
                for label in seen_pair_labels
                if label.split("/", 1)[1] in delta["removed"]
                and label not in removed_pair_labels
            }
            removed_pair_labels.update(newly_removed)
            removed_pairs.extend(sorted(newly_removed))
            for recipe in phase_recipes:
                active_for_phase[recipe] = tuple(sorted(shared_state.active))
                transitions[recipe] = LifecycleChange(
                    operation=operation,
                    reason=(
                        "global_initialization" if operation == "initialization"
                        else "global_phase_retention" if operation == "retention"
                        else "global_phase_change"
                    ),
                    can_reenter=shared_reintroduction_eligible,
                )
            preference_role: Dict[TaskKey, str] = {}
        elif structure == "heterogeneous":
            preference_role = {}
            force_recipe: Optional[str] = None
            if forced_preference is not None:
                force_candidates = [
                    recipe for recipe in phase_recipes
                    if forced_preference in effective_preferences_by_recipe[recipe]
                ]
                known_force_candidates = [
                    recipe for recipe in force_candidates if state_by_recipe[recipe].ever
                ]
                force_recipe = rng.choice(known_force_candidates or force_candidates)
            for recipe in phase_recipes:
                candidates = [
                    preference for preference in effective_preferences_by_recipe[recipe]
                    if not _blocked_before_introduction(
                        preference, phase_index, target, preference_axis_values,
                    )
                    and not _holdout_blocked(
                        recipe,
                        preference,
                        phase_index,
                        target,
                        semantic_equivalents_by_recipe,
                    )
                ]
                current = set(state_by_recipe[recipe].active)
                can_reenter = bool(
                    state_by_recipe[recipe].removed & set(candidates)
                )
                forced = (
                    [forced_preference]
                    if recipe == force_recipe and forced_preference is not None
                    else []
                )
                requested = (
                    None
                    if not current or forced
                    else _sample_lifecycle_operation(config, rng)
                )
                if requested == "retention":
                    active = set(current)
                    delta = {
                        "added": set(), "retained": set(active),
                        "removed": set(), "reintroduced": set(),
                    }
                    operation = "retention"
                else:
                    try:
                        active, delta, operation = _evolve_active_set(
                            state_by_recipe[recipe],
                            candidates,
                            forced,
                            config,
                            rng,
                            operation=requested,
                            allow_reintroduction=True,
                            max_active_size=max_active_size,
                        )
                    except RuntimeError:
                        return None
                if active != current:
                    changed_recipes.add(recipe)
                active_for_phase[recipe] = tuple(sorted(active))
                transitions[recipe] = LifecycleChange(
                    operation=operation,
                    reason=(
                        "initialization" if not current
                        else "forced_preference_change" if forced
                        else "stochastic_retention" if operation == "retention"
                        else "stochastic_change"
                    ),
                    can_reenter=can_reenter,
                )
                for role in ("added", "retained", "reintroduced"):
                    for preference in delta[role]:
                        preference_role[(recipe, preference)] = role
                removed_pairs.extend(
                    _pair_label((recipe, preference)) for preference in delta["removed"]
                )
        else:
            raise ValueError(f"unknown deployment structure {structure!r}")

        active_pairs = []
        omitted_pairs: Dict[str, str] = {}
        for recipe in phase_recipes:
            for preference in active_for_phase[recipe]:
                if preference not in effective_preferences_by_recipe[recipe]:
                    omitted_pairs[_pair_label((recipe, preference))] = str(
                        applicability_by_recipe[recipe].get(preference, "not_effective")
                    )
                    continue
                pair = (recipe, preference)
                active_pairs.append(pair)
                if structure == "homogeneous":
                    label = _pair_label(pair)
                    if label in removed_pair_labels:
                        role = "reintroduced"
                    elif label in seen_pair_labels:
                        role = "retained"
                    else:
                        role = "added"
                else:
                    role = preference_role.get(pair, "retained")
                lifecycle_by_pair[_pair_label(pair)] = role
                if role == "added":
                    added_pairs.append(_pair_label(pair))
                elif role == "reintroduced":
                    reintroduced_pairs.append(_pair_label(pair))
                else:
                    retained_pairs.append(_pair_label(pair))
        if structure == "homogeneous":
            # A global change makes every participating homogeneous recipe climb.
            changed_recipes = (
                set(phase_recipes)
                if operation != "retention"
                else {
                    recipe
                    for recipe, preference in active_pairs
                    if _pair_label((recipe, preference)) not in seen_pair_labels
                }
            )
        else:
            # Heterogeneous macro-phases are serialized below; every selected
            # recipe receives a complete recipe-specific climb, including a
            # retained recipe whose membership did not change.
            changed_recipes = set(phase_recipes)
        climb_pairs = [
            pair for pair in active_pairs if pair[0] in changed_recipes
        ]
        if structure == "homogeneous":
            climbed_labels = {_pair_label(pair) for pair in climb_pairs}
            seen_pair_labels.update(climbed_labels)
            removed_pair_labels.difference_update(climbed_labels)
        rng.shuffle(climb_pairs)
        drafts.append(_PhasePlan(
            index=phase_index,
            stage=phase_index,
            climb_recipes=(),
            recipes=tuple(phase_recipes),
            requested_recipes=requested_subgroup_size,
            climb_limited=climb_limited,
            climb_capacity=len(climb_pairs),
            active_preferences=active_for_phase,
            active_pairs=tuple(active_pairs),
            climb_pairs=tuple(climb_pairs),
            lifecycle_by_pair=lifecycle_by_pair,
            added_pairs=tuple(sorted(set(added_pairs))),
            retained_pairs=tuple(sorted(set(retained_pairs))),
            removed_pairs=tuple(sorted(set(removed_pairs))),
            reintroduced_pairs=tuple(sorted(set(reintroduced_pairs))),
            omitted_pairs=omitted_pairs,
            transitions=transitions,
        ))

    if structure == "heterogeneous":
        if config.phases >= 3 and len({phase.recipes for phase in drafts}) < 2:
            return None
    elif sum(len(phase.climb_pairs) for phase in drafts) > math.floor(
        config.transition_max * config.demos
    ):
        return None

    if target.kind != "none":
        assert target.preference is not None and target.introduction_phase is not None
        pre_pairs = [pair for phase in drafts[:target.introduction_phase] for pair in phase.climb_pairs]
        intro_pairs = drafts[target.introduction_phase].climb_pairs
        if any(preference == target.preference for _recipe, preference in pre_pairs):
            return None
        if target.kind == "preference" and any(
            preference in semantic_equivalents_by_recipe.get(recipe, ())
            for recipe, preference in pre_pairs
        ):
            return None
        if target.kind == "axis" and any(
            preference_axis_values[preference].get(target.axis) == target.axis_value
            for _recipe, preference in pre_pairs
        ):
            return None
        if not any(preference == target.preference for _recipe, preference in intro_pairs):
            return None
        earlier_recipes = {recipe for recipe, _get_preference in pre_pairs}
        if not any(
            recipe in earlier_recipes
            for recipe, preference in intro_pairs
            if preference == target.preference
        ):
            return None
        if target.kind == "preference":
            pre_preferences = {preference for _recipe, preference in pre_pairs}
            if not set(target.required_source_preferences).issubset(pre_preferences):
                return None
    return tuple(drafts)


def _climb_groups(
    recipes: Sequence[str],
    decay: float,
    rng: random.Random,
) -> Tuple[Tuple[str, ...], ...]:
    remaining = list(recipes)
    groups = []
    while remaining:
        sizes = list(range(1, len(remaining) + 1))
        size = rng.choices(
            sizes,
            weights=[float(decay) ** (value - 1) for value in sizes],
            k=1,
        )[0]
        groups.append(tuple(remaining[:size]))
        del remaining[:size]
    return tuple(groups)


def _serialize_drafts(
    drafts: Sequence[_PhasePlan],
    recipe_panel: Sequence[str],
    group_decay: float,
    rng: random.Random,
) -> Tuple[_PhasePlan, ...]:
    """Expand each macro-phase into decreasing-probability recipe groups."""
    current: Dict[str, Tuple[str, ...]] = {}
    steps: List[_PhasePlan] = []
    panel_order = {recipe: index for index, recipe in enumerate(recipe_panel)}

    for macro in drafts:
        for climb_recipes in _climb_groups(
            macro.recipes, group_decay, rng,
        ):
            climb_set = set(climb_recipes)
            for recipe in climb_recipes:
                current[recipe] = tuple(
                    sorted(macro.active_preferences[recipe])
                )
            active_by_recipe = {
                recipe: current[recipe]
                for recipe in sorted(current, key=panel_order.__getitem__)
                if current[recipe]
            }
            active_pairs = tuple(
                (recipe, preference)
                for recipe, preferences in active_by_recipe.items()
                for preference in preferences
            )
            climb_pairs = [
                (recipe, preference)
                for recipe in climb_recipes
                for preference in active_by_recipe[recipe]
            ]
            rng.shuffle(climb_pairs)
            roles = {
                _pair_label(pair): (
                    macro.lifecycle_by_pair.get(_pair_label(pair), "retained")
                    if pair[0] in climb_set else "retained"
                )
                for pair in active_pairs
            }
            added = tuple(sorted(
                label for label, role in roles.items() if role == "added"
            ))
            reintroduced = tuple(sorted(
                label for label, role in roles.items() if role == "reintroduced"
            ))
            retained = tuple(sorted(
                label for label, role in roles.items() if role == "retained"
            ))
            removed = tuple(sorted(
                label for label in macro.removed_pairs
                if label.split("/", 1)[0] in climb_set
            ))
            omitted = {
                label: reason for label, reason in macro.omitted_pairs.items()
                if label.split("/", 1)[0] in climb_set
            }
            steps.append(_PhasePlan(
                index=len(steps),
                stage=macro.index,
                climb_recipes=climb_recipes,
                recipes=tuple(active_by_recipe),
                requested_recipes=macro.requested_recipes,
                climb_limited=False,
                climb_capacity=len(climb_pairs),
                active_preferences=active_by_recipe,
                active_pairs=active_pairs,
                climb_pairs=tuple(climb_pairs),
                lifecycle_by_pair=roles,
                added_pairs=added,
                retained_pairs=retained,
                removed_pairs=removed,
                reintroduced_pairs=reintroduced,
                omitted_pairs=omitted,
                transitions={
                    recipe: macro.transitions[recipe]
                    for recipe in climb_recipes
                },
            ))
    return tuple(steps)


def build_schedule(
    *,
    structure: str,
    holdout_kind: str,
    seed: int,
    recipes: Sequence[str],
    preferences: Sequence[str],
    effective_preferences_by_recipe: Mapping[str, Sequence[str]],
    applicability_by_recipe: Mapping[str, Mapping[str, str]],
    preference_axis_values: Mapping[str, Mapping[str, str]],
    default_axis_values: Mapping[str, str],
    config: ScheduleSettings,
) -> DeploymentSchedule:
    """Generate one reproducible user timeline satisfying all hard invariants."""
    recipes = tuple(recipes)
    preferences = tuple(preferences)
    config.validate(len(recipes))
    panel_size = min(
        len(recipes),
        max(1, int(config.panel_size)),
    )
    if structure not in {"homogeneous", "heterogeneous"}:
        raise ValueError(f"unknown structure {structure!r}")
    if holdout_kind != "none" and config.phases < 3:
        raise ValueError("holdout timelines require at least three phases: source, introduction, and post-introduction")

    intro_phase = min(
        config.phases - 2,
        max(1, int(round((config.phases - 1) * config.holdout_start))),
    )
    profile_rng = random.Random(f"longitudinal|user_profile|seed={int(seed)}")
    popularity_raw = {
        recipe: profile_rng.gammavariate(config.recipe_skew, 1.0)
        for recipe in recipes
    }
    popularity_total = sum(popularity_raw.values())
    recipe_popularity = {
        recipe: value / popularity_total for recipe, value in popularity_raw.items()
    }
    panel = _weighted_sample_without_replacement(
        recipes,
        recipe_popularity,
        panel_size,
        random.Random(f"longitudinal|recipe_panel|seed={int(seed)}"),
    )
    panel_preferences = {
        recipe: effective_preferences_by_recipe[recipe] for recipe in panel
    }
    target = _select_holdout_target(
        holdout_kind,
        preferences,
        panel_preferences,
        preference_axis_values,
        default_axis_values,
        intro_phase,
        random.Random(f"longitudinal|holdout={holdout_kind}|seed={int(seed)}"),
    )
    semantic_equivalents_by_recipe = _holdout_equivalents(
        target,
        panel,
        effective_preferences_by_recipe,
        applicability_by_recipe,
    )
    drafts: Optional[Tuple[_PhasePlan, ...]] = None
    accepted_rng: Optional[random.Random] = None
    accepted_attempt_count = 0
    for attempt in range(config.max_attempts):
        attempt_rng = random.Random(
            f"longitudinal|{structure}|{holdout_kind}|{int(seed)}|attempt={attempt}"
        )
        drafts = _plan_drafts(
            structure,
            target,
            panel,
            effective_preferences_by_recipe,
            applicability_by_recipe,
            preference_axis_values,
            semantic_equivalents_by_recipe,
            recipe_popularity,
            config,
            attempt_rng,
        )
        if drafts is not None:
            accepted_rng = attempt_rng
            accepted_attempt_count = attempt + 1
            break
    if drafts is None or accepted_rng is None:
        raise RuntimeError(
            "could not generate a schedule satisfying the configured budget and invariants; "
            "increase demos, widen recipe participation, or increase max_attempts"
        )

    if structure == "heterogeneous":
        macro_introduction = target.introduction_phase
        drafts = _serialize_drafts(
            drafts,
            panel,
            config.climb_decay,
            random.Random(
                f"longitudinal|climb_groups|{holdout_kind}|seed={int(seed)}"
            ),
        )
        if macro_introduction is not None:
            introductions = [
                draft.index for draft in drafts
                if draft.stage == macro_introduction
                and any(
                    preference == target.preference
                    for _recipe, preference in draft.climb_pairs
                )
            ]
            if not introductions:
                raise RuntimeError("serialized schedule lost the holdout introduction")
            target = replace(target, introduction_step=min(introductions))
        target_climb_count = sum(len(draft.climb_pairs) for draft in drafts)
        gaps = _sample_hetero_gap(
            config,
            random.Random(
                f"longitudinal|gap|{structure}|{holdout_kind}|seed={int(seed)}"
            ),
            tuple(
                _min_phase_gap(draft, target, config)
                for draft in drafts
            ),
        )
    else:
        target_climb_count = _allocate_climb_budget(
            drafts,
            config,
            recipe_popularity,
            random.Random(
                f"longitudinal|climb_budget|{structure}|{holdout_kind}|seed={int(seed)}"
            ),
        )
        gaps = _sample_phase_gap(
            config,
            random.Random(
                f"longitudinal|gap|{structure}|{holdout_kind}|seed={int(seed)}"
            ),
            tuple(
                _min_phase_gap(draft, target, config)
                for draft in drafts
            ),
        )

    climb_total = sum(len(phase.climb_pairs) for phase in drafts)
    if climb_total != target_climb_count:
        raise RuntimeError("explicit climb-budget allocation changed support unexpectedly")
    phases = []
    start_demo = 0
    cumulative_climb = 0
    cumulative_demos = 0
    for draft, gap in zip(drafts, gaps):
        phase_demo_count = DEMOS_PER_GAP * gap
        settled_count = phase_demo_count - len(draft.climb_pairs)
        settled, settle_counts, settle_weights = _repeat_phase_support(
            draft.active_pairs,
            settled_count,
            recipe_popularity,
            accepted_rng,
            pair_weight_shape=config.pair_skew,
            priority_preference=(
                target.preference if draft.index == _holdout_step(target) else None
            ),
            minimum_priority_count=(
                config.holdout_demos
                if draft.index == _holdout_step(target) and target.kind != "none" else 0
            ),
        )
        cumulative_climb += len(draft.climb_pairs)
        cumulative_demos += len(draft.climb_pairs) + len(settled)
        phases.append(Phase(
            index=draft.index,
            stage=draft.stage,
            climb_recipes=draft.climb_recipes,
            gap=gap,
            start_demo=start_demo,
            recipes=draft.recipes,
            requested_recipes=draft.requested_recipes,
            climb_limited=(
                draft.climb_limited
            ),
            climb_capacity=(
                draft.climb_capacity
            ),
            active_preferences=draft.active_preferences,
            climb_pairs=draft.climb_pairs,
            settled_pairs=settled,
            lifecycle_by_pair=draft.lifecycle_by_pair,
            added_pairs=draft.added_pairs,
            retained_pairs=draft.retained_pairs,
            removed_pairs=draft.removed_pairs,
            reintroduced_pairs=draft.reintroduced_pairs,
            settle_counts=settle_counts,
            settle_weights=settle_weights,
            omitted_pairs=draft.omitted_pairs,
            transitions=draft.transitions,
            transition_fraction=cumulative_climb / cumulative_demos,
        ))
        start_demo += phase_demo_count

    demos = DEMOS_PER_GAP * sum(gaps)
    if start_demo != demos:
        raise RuntimeError("gap allocation did not preserve the demonstration budget")

    return DeploymentSchedule(
        seed=int(seed),
        user_id=f"user_seed_{int(seed)}",
        structure=structure,
        holdout=target,
        phases=tuple(phases),
        candidates=tuple(recipes),
        recipe_panel=tuple(panel),
        recipe_popularity=recipe_popularity,
        transition_fraction=climb_total / demos,
        demos=demos,
        holdout_equivalents=semantic_equivalents_by_recipe,
        config=config,
        attempts=accepted_attempt_count,
    )


CLAIRVOYANT_REFERENCE_TAG = "dashed_reference_not_deployable"
CLAIRVOYANT_LEAKAGE_WARNING = (
    "Non-deployable oracle: compares future-filtered retention against an "
    "identical Full reference using ground-truth outcomes, and keeps the "
    "non-inferior state. Use only as a reference curve."
)

POST_MISMATCH_WINDOWS = (1, 2, 3)
RECOVERY_BASELINE_EXPOSURES = 3
RECOVERY_TARGET_FRACTION = 0.90


@dataclass(frozen=True)
class TaskVariant:
    recipe_name: str
    preference_name: str
    actions: Tuple[str, ...]
    values: Mapping[str, str] = field(default_factory=dict)
    @cached_property
    def observations(self) -> Tuple[Observation, ...]:
        return tuple(observe_actions(self.actions))

    @property
    def label(self) -> str:
        return f"{self.recipe_name}/{self.preference_name}"

    @property
    def non_default_axes(self) -> Tuple[str, ...]:
        return tuple(
            axis for axis, value in self.values.items()
            if value != DEFAULT_PREFERENCE.get(axis)
        )

    @property
    def is_composed_preference(self) -> bool:
        return len(self.non_default_axes) >= 2


@dataclass(frozen=True)
class Event:
    mode: str
    pair: TaskVariant
    tags: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    scenario: str
    seed: int
    events: Tuple[Event, ...]
    eval_pairs: Tuple[TaskVariant, ...]
    selected_recipes: Tuple[str, ...]
    selected_preferences: Tuple[str, ...]
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSettings:
    # Default paired paper replications; callers may supply more seeds.
    seeds: Tuple[int, ...] = PAPER_SEEDS
    scenarios: Tuple[str, ...] = SCENARIOS
    baselines: Tuple[str, ...] = DEFAULT_BASELINES
    # Deterministic offline-frozen training subsets.
    offline_recipe_fraction: float = 0.50
    offline_preference_fraction: float = 0.50
    output: str = DEFAULT_RESULTS_ROOT
    run: Optional[str] = None
    resume: bool = False
    workers: int = 0
    threads: int = DEFAULT_NATIVE_THREADS_PER_WORKER
    show_eta: bool = True
    include_oracle: bool = True
    recipe_count: int = 20
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    # Paper contract: the externally selected mode is authoritative. Missing
    # recipe recovery is disabled in the normal matched evaluation.
    observe_missing_recipes: bool = False
    # Ablation-only treatment: a baseline may request another complete
    # demonstration after it has forgotten every active variant of a recipe.
    # The standard evaluation keeps this False.
    allow_repeat_observation: bool = False
    # Share the full system's realized route across deployable baselines.
    shared_routing: bool = True
    # Probe primary climb assists before interaction without mutation.
    pre_event_probes: bool = True
    frozen_pairs: int = 48
    audit_period: int = 2
    audit_prefixes: int = 16
    audit_tolerance: float = 5e-2
    top_k: int = 3
    profile: bool = False
    sensitivity: bool = False
    model_settings: Mapping[str, Any] = field(default_factory=dict)
    # Reporting label with no behavioral effect.
    experiment: str = "standard_evaluation"


@dataclass
class RunState:
    baseline: str
    scenario: str
    seed: int
    agent: AdaptiveAgent
    recipe_ids: Dict[str, str]
    episode_rows: List[Dict[str, Any]]
    frozen_rows: List[Dict[str, Any]]
    memory_rows: List[Dict[str, Any]]
    active_audit_rows: List[Dict[str, Any]]
    oracle_pruning_rows: List[Dict[str, Any]]
    turn_rows: List[Dict[str, Any]]
    initial_memory: Dict[str, Any]
    wall_s: float


def build_settings(seed: int, eval_config: EvalSettings, **overrides: Any) -> Settings:
    settings = replace(
        DEFAULT_SETTINGS,
        seed=int(seed),
        verbose=False,
        profile=bool(eval_config.profile),
    )
    merged = {**dict(eval_config.model_settings), **overrides}
    return replace(settings, **merged) if merged else settings


def build_agent(name: str, settings: Settings) -> AdaptiveAgent:
    if name == "full":
        return AdaptiveAgent(settings=settings)
    try:
        agent_type = BASELINE_AGENTS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown baseline {name!r}; available={sorted(('full', *BASELINE_AGENTS))}"
        ) from exc
    return agent_type(settings=settings)


def _recipe_sample(
    seed: int, count: int,
) -> List[Tuple[str, Callable[[], List[str]]]]:
    """Select the seeded candidate pool used by every scenario cell."""
    items = list(recipe_builders().items())
    rng = random.Random(int(seed))
    rng.shuffle(items)
    return items[: max(1, min(int(count), len(items)))]


def _preference_panel(
    recipe_name: str,
    builder: Callable[[], List[str]],
    preferences: Sequence[str],
) -> Tuple[Dict[str, TaskVariant], Dict[str, str]]:
    """Return effective pairs and an auditable status for every preference."""
    base = tuple(builder())
    pairs: Dict[str, TaskVariant] = {}
    applicability: Dict[str, str] = {}
    seen: Dict[Tuple[str, ...], str] = {}
    for preference_name in preferences:
        try:
            pair = build_task(recipe_name, preference_name, builder)
        except (KeyError, ValueError):
            applicability[preference_name] = "materialization_error"
            continue
        if preference_name != "default" and pair.actions == base:
            applicability[preference_name] = "recipe_specific_noop"
            continue
        duplicate_of = seen.get(pair.actions)
        if duplicate_of is not None:
            applicability[preference_name] = f"semantic_duplicate_of:{duplicate_of}"
            continue
        seen[pair.actions] = preference_name
        pairs[preference_name] = pair
        applicability[preference_name] = "effective_applicable"
    return pairs, applicability


def build_task(recipe_name: str, preference_name: str, builder: Callable[[], List[str]]) -> TaskVariant:
    if preference_name not in PREFERENCES:
        raise KeyError(f"unknown preference {preference_name!r}")
    base = tuple(builder())
    report = apply_preset(base, preference_name)
    return TaskVariant(
        recipe_name=recipe_name,
        preference_name=preference_name,
        actions=tuple(report.actions),
        values=dict(report.values),
    )


def _scenario_tag(config: EvalSettings, scenario: str, **extra: Any) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "user_id": "user_001",
        "one_user_updating_preferences": True,
        "top_k": int(config.top_k),
        **extra,
    }


def _schedule_config(config: EvalSettings, recipe_count: int) -> ScheduleSettings:
    schedule = config.schedule
    minimum = min(max(1, int(schedule.min_recipes)), recipe_count)
    maximum = min(
        recipe_count,
        max(minimum, int(schedule.max_recipes)),
    )
    return replace(
        schedule,
        panel_size=min(recipe_count, max(1, int(schedule.panel_size))),
        phases=max(2, int(schedule.phases)),
        demos=max(1, int(schedule.demos)),
        min_recipes=minimum,
        max_recipes=maximum,
        transition_min=float(schedule.transition_min),
        transition_max=float(schedule.transition_max),
        gap_allocation=float(schedule.gap_allocation),
        hetero_gap_mean=int(schedule.hetero_gap_mean),
        hetero_gap_min=int(schedule.hetero_gap_min),
        hetero_gap_max=int(schedule.hetero_gap_max),
        hetero_gap_shape=float(schedule.hetero_gap_shape),
        climb_decay=float(schedule.climb_decay),
        active_size_weights=tuple(float(v) for v in schedule.active_size_weights),
        lifecycle_weights=tuple(
            float(value) for value in schedule.lifecycle_weights
        ),
        reentry_rate=float(schedule.reentry_rate),
        recipe_skew=float(schedule.recipe_skew),
        pair_skew=float(schedule.pair_skew),
        holdout_start=float(schedule.holdout_start),
        holdout_demos=max(1, int(schedule.holdout_demos)),
    )


def _effective_active_preferences(phase: Phase) -> Dict[str, Tuple[str, ...]]:
    by_recipe: Dict[str, set[str]] = {recipe: set() for recipe in phase.recipes}
    for recipe, preference in phase.settled_pairs:
        by_recipe[recipe].add(preference)
    return {
        recipe: tuple(sorted(preferences))
        for recipe, preferences in by_recipe.items()
    }


def _schedule_metadata(schedule: DeploymentSchedule) -> Dict[str, Any]:
    """Serialize canonical schedule facts and compact design audits."""
    all_pairs = [
        pair
        for phase in schedule.phases
        for pair in (*phase.climb_pairs, *phase.settled_pairs)
    ]
    operations = [
        transition.operation
        for phase in schedule.phases
        for transition in phase.transitions.values()
    ]
    operation_counts = Counter(operations)
    active_cardinalities = Counter(
        len(preferences)
        for phase in schedule.phases
        for preferences in _effective_active_preferences(phase).values()
    )
    introduction = _holdout_step(schedule.holdout)
    pre_introduction = [
        pair
        for phase in schedule.phases
        if introduction is not None and phase.index < introduction
        for pair in (*phase.climb_pairs, *phase.settled_pairs)
    ]
    preference_leaks = sum(
        preference == schedule.holdout.preference
        for _recipe, preference in pre_introduction
    )
    semantic_leaks = sum(
        preference in schedule.holdout_equivalents.get(recipe, ())
        for recipe, preference in pre_introduction
    )
    axis_leaks = 0
    if schedule.holdout.kind == "axis":
        axis_leaks = sum(
            PREFERENCES[preference].as_dict().get(schedule.holdout.axis)
            == schedule.holdout.axis_value
            for _recipe, preference in pre_introduction
        )

    phase_rows = []
    for phase in schedule.phases:
        phase_pairs = (*phase.climb_pairs, *phase.settled_pairs)
        phase_rows.append({
            "index": phase.index,
            "stage": phase.stage,
            "climb_recipes": list(phase.climb_recipes),
            "gap": phase.gap,
            "start_demo": phase.start_demo,
            "recipes": list(phase.recipes),
            "requested_recipes": phase.requested_recipes,
            "climb_limited": phase.climb_limited,
            "climb_support_capacity": phase.climb_capacity,
            "active_preferences": {
                recipe: list(preferences)
                for recipe, preferences in phase.active_preferences.items()
            },
            "climb_pairs": [_pair_label(pair) for pair in phase.climb_pairs],
            "settled_pair_counts": dict(Counter(
                _pair_label(pair) for pair in phase.settled_pairs
            )),
            "demo_count": len(phase_pairs),
            "transitions": {
                recipe: asdict(transition)
                for recipe, transition in phase.transitions.items()
            },
            "added_pairs": list(phase.added_pairs),
            "removed_pairs": list(phase.removed_pairs),
            "reintroduced_pairs": list(phase.reintroduced_pairs),
            "omitted_pairs": dict(phase.omitted_pairs),
            "transition_fraction": phase.transition_fraction,
        })

    existing = sum(operation != "initialization" for operation in operations)
    changed = sum(operation not in {"initialization", "retention"} for operation in operations)
    return {
        "user_id": schedule.user_id,
        "holdout": asdict(schedule.holdout),
        "generator": asdict(schedule.config),
        "attempts": schedule.attempts,
        "candidates": list(schedule.candidates),
        "recipe_panel": list(schedule.recipe_panel),
        "recipe_popularity": dict(schedule.recipe_popularity),
        "transition_fraction": schedule.transition_fraction,
        "totals": {
            "demos": schedule.demos,
            "gap": schedule.demos // DEMOS_PER_GAP,
            "climb_demos": sum(len(phase.climb_pairs) for phase in schedule.phases),
            "settled_demos": sum(len(phase.settled_pairs) for phase in schedule.phases),
            "unique_recipes": len({recipe for recipe, _get_preference in all_pairs}),
            "unique_pairs": len(set(all_pairs)),
        },
        "lifecycle": {
            "operation_counts": dict(operation_counts),
            "existing_opportunities": existing,
            "changed_existing": changed,
            "realized_change_rate": _safe_div(changed, existing),
            "reintroduced_pairs": sum(len(phase.reintroduced_pairs) for phase in schedule.phases),
            "active_cardinality_counts": {
                str(cardinality): count
                for cardinality, count in sorted(active_cardinalities.items())
            },
        },
        "exposure": {
            "recipe_demo_counts": dict(Counter(recipe for recipe, _get_preference in all_pairs)),
            "pair_demo_counts": dict(Counter(_pair_label(pair) for pair in all_pairs)),
        },
        "leakage_audit": {
            "preference_leak_count": preference_leaks,
            "semantic_preference_leak_count": semantic_leaks,
            "axis_value_leak_count": axis_leaks,
            "passed": preference_leaks == semantic_leaks == axis_leaks == 0,
        },
        "phases": phase_rows,
    }
def _schedule_events(
    config: EvalSettings,
    scenario: str,
    schedule: DeploymentSchedule,
    pair_matrix: Mapping[str, Mapping[str, TaskVariant]],
) -> Tuple[Event, ...]:
    events: List[Event] = []
    seen_recipes: set[str] = set()
    seen_pairs: set[str] = set()
    seen_preferences_by_recipe: Dict[str, set[str]] = defaultdict(set)
    preference_sources: Dict[str, TaskVariant] = {}
    last_recipe_event_index: Dict[str, int] = {}
    last_preference_event_index: Dict[str, int] = {}
    last_pair_event_index: Dict[str, int] = {}
    exposure_since_recipe_change: Counter[str] = Counter()
    holdout_seen = False
    holdout_step = _holdout_step(schedule.holdout)
    for phase in schedule.phases:
        phase_id = f"phase_{phase.index:02d}"
        effective_active_by_recipe = _effective_active_preferences(phase)
        partition = (
            "ordinary" if schedule.holdout.kind == "none"
            else "source" if phase.index < int(holdout_step)
            else "heldout"
        )
        occurrence_by_phase_role: Counter[Tuple[str, str]] = Counter()
        changed_recipes = (
            set(phase.recipes)
            if schedule.structure == "homogeneous"
            else {
                recipe
                for recipe, transition in phase.transitions.items()
                if transition.operation != "retention"
            }
        )
        for recipe in changed_recipes:
            for preference in effective_active_by_recipe[recipe]:
                exposure_since_recipe_change[
                    _pair_label((recipe, preference))
                ] = 0

        for phase_role, phase_names in (("climb", phase.climb_pairs), ("settled", phase.settled_pairs)):
            names = list(phase_names)
            if (
                phase_role == "climb"
                and schedule.holdout.kind != "none"
                and phase.index == holdout_step
            ):
                # Start holdout evaluation with first-exposure assist on a known recipe.
                names.sort(key=lambda item: (
                    item[1] != schedule.holdout.preference,
                    item[0] not in seen_recipes,
                    item[0],
                    item[1],
                ))
            for position, (recipe, preference) in enumerate(names):
                pair = pair_matrix[recipe][preference]
                label = pair.label
                exposure_since_recipe_change[label] += 1
                occurrence_by_phase_role[(phase_role, label)] += 1
                event_index = len(events)
                preference_history_depth_before = len(
                    seen_preferences_by_recipe[recipe]
                )
                preference_seen_for_recipe_before = (
                    preference in seen_preferences_by_recipe[recipe]
                )
                preference_history_depth_after = len(
                    seen_preferences_by_recipe[recipe] | {preference}
                )

                lifecycle = phase.lifecycle_by_pair.get(label, "retained")
                transition = phase.transitions.get(
                    recipe,
                    LifecycleChange(
                        operation="retention",
                        reason="settled_after_other_recipe_climb",
                    ),
                )
                target_event = _is_holdout_target(preference, schedule.holdout)
                first_holdout_exposure = bool(target_event and not holdout_seen)
                if first_holdout_exposure:
                    holdout_seen = True
                source_pair = preference_sources.get(preference)
                cross_recipe_transfer = bool(
                    phase_role == "climb"
                    and label not in seen_pairs
                    and source_pair is not None
                    and source_pair.recipe_name != recipe
                )
                hypothesis_tags = [
                    "climb" if phase_role == "climb" else "settled_reuse",
                    f"preference_{lifecycle}",
                ]
                if cross_recipe_transfer:
                    hypothesis_tags.append("cross_recipe_transfer")
                if target_event and schedule.holdout.kind != "none":
                    hypothesis_tags.append(f"{schedule.holdout.kind}_holdout")

                # A recipe receives exactly one full observation: its first
                # event anywhere in the stream. Every subsequent exposure,
                # including a new preference or another climb occurrence in
                # the same phase, is an assist episode.
                requested_mode = (
                    "observe" if recipe not in seen_recipes else "assist"
                )
                settled_repeat_count = int(phase.settle_counts.get(label, 0))
                demos_since_recipe_seen = (
                    None if recipe not in last_recipe_event_index
                    else event_index - last_recipe_event_index[recipe]
                )
                demos_since_preference_seen = (
                    None if preference not in last_preference_event_index
                    else event_index - last_preference_event_index[preference]
                )
                demos_since_pair_seen = (
                    None if label not in last_pair_event_index
                    else event_index - last_pair_event_index[label]
                )
                pair_gap_bin = (
                    "first_exposure" if demos_since_pair_seen is None
                    else "short_le_21_demos" if demos_since_pair_seen <= SHORT_DEMO_GAP_MAX
                    else "medium_le_90_demos" if demos_since_pair_seen <= MEDIUM_DEMO_GAP_MAX
                    else "long_gt_90_demos"
                )
                progress = (event_index + 1) / schedule.demos
                demo_progress_bin = (
                    "budget_q1" if progress <= 0.25
                    else "budget_q2" if progress <= 0.50
                    else "budget_q3" if progress <= 0.75
                    else "budget_q4"
                )
                holdout_exposure_role = (
                    "non_target" if not target_event
                    else "first_climb" if first_holdout_exposure
                    else "later_climb" if phase_role == "climb"
                    else "post_introduction_settled"
                )
                recipe_changed = recipe in changed_recipes
                tags = _scenario_tag(
                    config,
                    scenario,
                    user_id=schedule.user_id,
                    event_type=f"{schedule.structure}_{phase_role}",
                    condition=f"{schedule.structure}_{partition}_{phase_role}",
                    primary_probe=bool(phase_role == "climb" and requested_mode == "assist"),
                    event_index=event_index,
                    demonstration_number=event_index + 1,
                    recipe=recipe,
                    effective_preference=preference,
                    pair_label=label,
                    phase_id=phase_id,
                    phase_index=int(phase.index),
                    stage=int(phase.stage),
                    climb_recipes=list(phase.climb_recipes),
                    phase_role=phase_role,
                    phase_position=int(position),
                    gap=int(phase.gap),
                    phase_start=int(phase.start_demo),
                    phase_demo_count=len(phase.climb_pairs) + len(phase.settled_pairs),
                    demo_progress_bin=demo_progress_bin,
                    deployment_structure=schedule.structure,
                    active_preferences=list(effective_active_by_recipe[recipe]),
                    active_preference_count_for_recipe=len(
                        effective_active_by_recipe[recipe]
                    ),
                    pair_lifecycle_role=lifecycle,
                    lifecycle_phase_role=f"{lifecycle}_{phase_role}",
                    lifecycle_operation=transition.operation,
                    lifecycle_reason=transition.reason,
                    reintroduction_eligible_this_phase=transition.can_reenter,
                    recipe_changed_this_phase=recipe_changed,
                    unchanged_recipe_during_other_change=bool(
                        not recipe_changed and changed_recipes
                    ),
                    preference_seen_for_recipe_before=(
                        preference_seen_for_recipe_before
                    ),
                    preference_history_depth_before=(
                        preference_history_depth_before
                    ),
                    preference_history_depth_after=(
                        preference_history_depth_after
                    ),
                    demos_since_recipe_seen=demos_since_recipe_seen,
                    demos_since_preference_seen=demos_since_preference_seen,
                    demos_since_pair_seen=demos_since_pair_seen,
                    pair_gap_bin=pair_gap_bin,
                    retained_pair_gap_bin=(
                        pair_gap_bin if lifecycle == "retained" else "not_retained"
                    ),
                    exposure_index_since_recipe_change=int(
                        exposure_since_recipe_change[label]
                    ),
                    settle_repeat_index=(
                        None if phase_role == "climb" else int(
                            occurrence_by_phase_role[(phase_role, label)] - 1
                        )
                    ),
                    settle_repeat_count_for_pair=settled_repeat_count,
                    holdout_kind=schedule.holdout.kind,
                    holdout_partition=partition,
                    holdout_phase_role=f"{partition}_{phase_role}",
                    holdout_introduction_phase=schedule.holdout.introduction_phase,
                    holdout_introduction_step=holdout_step,
                    holdout_exposure_role=holdout_exposure_role,
                    holdout_target_lifecycle_role=(lifecycle if target_event else None),
                    holdout_target_type=(
                        "isolated_axis"
                        if target_event and preference == schedule.holdout.preference
                        else "axis_composition"
                        if target_event and schedule.holdout.kind == "axis"
                        else "preference"
                        if target_event else None
                    ),
                    is_first_holdout_exposure=first_holdout_exposure,
                    is_post_holdout_settled=bool(
                        partition == "heldout" and phase_role == "settled" and target_event
                    ),
                    applicability_status="effective_applicable",
                    omitted_pair_count=len(phase.omitted_pairs),
                    preference_non_default_axes=list(pair.non_default_axes),
                    hypothesis_tags=hypothesis_tags,
                )
                events.append(Event(requested_mode, pair, tags))
                seen_recipes.add(recipe)
                seen_pairs.add(label)
                seen_preferences_by_recipe[recipe].add(preference)
                preference_sources.setdefault(preference, pair)
                last_recipe_event_index[recipe] = event_index
                last_preference_event_index[preference] = event_index
                last_pair_event_index[label] = event_index
    return tuple(events)


def build_deployment_plan(
    config: EvalSettings,
    seed: int,
    *,
    structure: str,
    holdout: str = "none",
    scenario: Optional[str] = None,
) -> Plan:
    """Build an ordinary homogeneous or heterogeneous deployment."""
    sampled_recipes = _recipe_sample(seed + 101, config.recipe_count)
    recipes = [recipe for recipe, _builder in sampled_recipes]
    preferences = tuple(PREFERENCE_IDS)
    pair_matrix: Dict[str, Dict[str, TaskVariant]] = {}
    applicability_by_recipe: Dict[str, Dict[str, str]] = {}
    for recipe, builder in sampled_recipes:
        pairs, applicability = _preference_panel(recipe, builder, preferences)
        pair_matrix[recipe] = pairs
        applicability_by_recipe[recipe] = applicability
    schedule = build_schedule(
        structure=structure,
        holdout_kind=holdout,
        seed=seed,
        recipes=recipes,
        preferences=preferences,
        effective_preferences_by_recipe={
            recipe: tuple(pairs) for recipe, pairs in pair_matrix.items()
        },
        applicability_by_recipe=applicability_by_recipe,
        preference_axis_values={
            preference: PREFERENCES[preference].as_dict()
            for preference in preferences
        },
        default_axis_values=DEFAULT_PREFERENCE,
        config=_schedule_config(config, len(recipes)),
    )
    scenario_name = scenario or structure
    events = _schedule_events(config, scenario_name, schedule, pair_matrix)
    selected_preferences = tuple(sorted({event.pair.preference_name for event in events}))
    eval_pairs = tuple(
        pair
        for recipe in schedule.recipe_panel
        for pair in pair_matrix[recipe].values()
    )
    return Plan(
        scenario=scenario_name,
        seed=seed,
        events=events,
        eval_pairs=eval_pairs,
        selected_recipes=tuple(schedule.recipe_panel),
        selected_preferences=selected_preferences,
        description=(
            f"Stylized single-user longitudinal {structure} deployment with {len(schedule.phases)} variable-size "
            f"climb/settle phases and a {schedule.demos}-demonstration budget, "
            f"exactly {DEMOS_PER_GAP} demonstrations per gap unit, "
            f"{schedule.transition_fraction:.1%} climb support, settled reuse, explicit removals, "
            f"rare reintroduction, and {holdout} selection constraints."
        ),
        metadata=_schedule_metadata(schedule),
    )


def build_holdout_plan(
    config: EvalSettings,
    seed: int,
) -> Plan:
    """Build the controlled cleanup-axis transfer ladder."""
    sampled = _recipe_sample(seed + 101, config.recipe_count)
    recipes = tuple(recipe for recipe, _builder in sampled)
    preferences = HOLDOUT_SOURCES + HOLDOUT_TARGETS
    pairs: Dict[str, Dict[str, TaskVariant]] = {}
    for recipe, builder in sampled:
        pairs[recipe], _applicability = _preference_panel(
            recipe, builder, preferences,
        )

    schedule_config = _schedule_config(config, len(recipes))
    profile_rng = random.Random(f"longitudinal|user_profile|seed={int(seed)}")
    raw_popularity = {
        recipe: profile_rng.gammavariate(
            schedule_config.recipe_skew, 1.0,
        )
        for recipe in recipes
    }
    popularity_total = sum(raw_popularity.values())
    popularity = {
        recipe: value / popularity_total
        for recipe, value in raw_popularity.items()
    }
    panel = _weighted_sample_without_replacement(
        recipes,
        popularity,
        schedule_config.panel_size,
        random.Random(f"longitudinal|recipe_panel|seed={int(seed)}"),
    )
    eligible = [
        recipe for recipe in panel
        if all(preference in pairs[recipe] for preference in preferences)
    ]
    if len(eligible) < schedule_config.min_recipes:
        raise RuntimeError(
            "axis holdout has too few recipes on which all source and target "
            "preferences are behaviorally effective"
        )
    rng = random.Random(f"holdout|seed={int(seed)}")
    recipe_count = rng.randint(
        schedule_config.min_recipes,
        min(schedule_config.max_recipes, len(eligible)),
    )
    probe_recipes = _weighted_sample_without_replacement(
        eligible, popularity, recipe_count, rng,
    )

    target = Holdout(
        kind="axis",
        preference=HOLDOUT_TARGETS[0],
        axis=HOLDOUT_AXIS,
        axis_value=HOLDOUT_VALUE,
        introduction_phase=len(HOLDOUT_SOURCES),
        required_source_preferences=HOLDOUT_SOURCES,
        source_preferences=HOLDOUT_SOURCES,
        target_preferences=HOLDOUT_TARGETS,
    )
    gap = schedule_config.hetero_gap_mean
    demos_per_phase = DEMOS_PER_GAP * gap
    phases = []
    cumulative_climb = 0
    previous: Optional[str] = None
    for index, preference in enumerate(preferences):
        active_pairs = tuple((recipe, preference) for recipe in probe_recipes)
        climb = list(active_pairs)
        rng.shuffle(climb)
        settled, settle_counts, settle_weights = _repeat_phase_support(
            active_pairs,
            demos_per_phase - len(climb),
            popularity,
            rng,
            pair_weight_shape=schedule_config.pair_skew,
            priority_preference=preference,
            minimum_priority_count=schedule_config.holdout_demos,
        )
        cumulative_climb += len(climb)
        lifecycle = {_pair_label(pair): "added" for pair in active_pairs}
        phases.append(Phase(
            index=index,
            stage=index,
            climb_recipes=probe_recipes,
            gap=gap,
            start_demo=index * demos_per_phase,
            recipes=probe_recipes,
            requested_recipes=recipe_count,
            climb_limited=False,
            climb_capacity=len(climb),
            active_preferences={
                recipe: (preference,) for recipe in probe_recipes
            },
            climb_pairs=tuple(climb),
            settled_pairs=settled,
            lifecycle_by_pair=lifecycle,
            added_pairs=tuple(sorted(lifecycle)),
            retained_pairs=(),
            removed_pairs=(
                () if previous is None else tuple(
                    sorted(f"{recipe}/{previous}" for recipe in probe_recipes)
                )
            ),
            reintroduced_pairs=(),
            settle_counts=settle_counts,
            settle_weights=settle_weights,
            omitted_pairs={},
            transitions={
                recipe: LifecycleChange(
                    operation="initialization" if previous is None else "swap",
                    reason=(
                        "source_axis_training"
                        if preference in HOLDOUT_SOURCES
                        else "heldout_axis_introduction"
                        if preference == HOLDOUT_TARGETS[0]
                        else "heldout_axis_composition"
                    ),
                )
                for recipe in probe_recipes
            },
            transition_fraction=(
                cumulative_climb / ((index + 1) * demos_per_phase)
            ),
        ))
        previous = preference

    demos = demos_per_phase * len(phases)
    schedule = DeploymentSchedule(
        seed=int(seed),
        user_id=f"user_seed_{int(seed)}",
        structure="controlled",
        holdout=target,
        phases=tuple(phases),
        candidates=recipes,
        recipe_panel=panel,
        recipe_popularity=popularity,
        transition_fraction=cumulative_climb / demos,
        demos=demos,
        holdout_equivalents={},
        config=replace(
            schedule_config,
            phases=len(phases),
            demos=demos,
        ),
        attempts=1,
    )
    events = _schedule_events(
        config, HOLDOUT, schedule, pairs,
    )
    metadata = _schedule_metadata(schedule)
    metadata["probe_recipes"] = list(probe_recipes)
    return Plan(
        scenario=HOLDOUT,
        seed=seed,
        events=events,
        eval_pairs=tuple(
            pairs[recipe][preference]
            for recipe in probe_recipes
            for preference in preferences
        ),
        selected_recipes=panel,
        selected_preferences=preferences,
        description=(
            "Controlled cleanup-timing holdout: train without the when-free "
            "cleanup value, then introduce cleanup_when_free followed by its "
            "unseen compositions."
        ),
        metadata=metadata,
    )


def build_plan(scenario: str, config: EvalSettings, seed: int) -> Plan:
    try:
        spec = SCENARIO_SPECS[scenario]
    except KeyError as exc:
        raise KeyError(f"unknown scenario {scenario!r}; available={SCENARIOS}") from exc
    if scenario == HOLDOUT:
        return build_holdout_plan(config, seed)
    return build_deployment_plan(
        config,
        seed,
        structure=spec.structure,
        holdout=spec.holdout,
        scenario=scenario,
    )


def _axis_labels(pair: TaskVariant) -> Tuple[str, ...]:
    defaults = DEFAULT_PREFERENCE
    return tuple(
        f"{axis}={value}"
        for axis, value in sorted(dict(pair.values).items())
        if value != defaults.get(axis)
    )


def _exposure_tags(
    pair: TaskVariant,
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
        for recipe, axis_values in axis_values_by_recipe.items()
        if recipe != pair.recipe_name
        for label in labels & axis_values
    }
    global_axis = {
        label
        for axis_values in axis_values_by_recipe.values()
        for label in labels & axis_values
    }
    if not labels:
        axis_cell = "default_or_no_changed_axis"
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


def _active_variants(agent: AdaptiveAgent) -> set[VariantKey]:
    return set(agent.replay.active)


def _pruned_keys(agent: AdaptiveAgent) -> set[VariantKey]:
    return set(agent.replay.pruned)


def _pair_key(
    agent: AdaptiveAgent,
    pair: TaskVariant,
    recipe_ids: Mapping[str, str],
) -> Optional[VariantKey]:
    recipe_id = recipe_ids.get(pair.recipe_name)
    if recipe_id is None:
        return None
    return (recipe_id, make_variant_id(pair.actions))


def _recipe_has_active_variant(
    agent: AdaptiveAgent,
    pair: TaskVariant,
    recipe_ids: Mapping[str, str],
) -> bool:
    recipe_id = recipe_ids.get(pair.recipe_name)
    return bool(recipe_id and any(recipe_id == recipe_id for recipe_id, _ in _active_variants(agent)))


def _recurrence_tags(
    agent: AdaptiveAgent,
    pair: TaskVariant,
    recipe_ids: Mapping[str, str],
    target_key: Optional[VariantKey],
    tags: Mapping[str, Any],
    pairs_by_label: Mapping[str, TaskVariant],
) -> Dict[str, Any]:
    """Report whether a tagged recurrence actually engaged memory removal."""
    if not bool(tags.get("delayed_recurrence_probe")):
        return {}
    conflict_labels = tuple(str(label) for label in (
        tags.get("delayed_recurrence_historical_conflicting_pair_labels") or ()
    ))
    active = _active_variants(agent)
    pruned = _pruned_keys(agent)
    conflict_keys = [
        _pair_key(agent, pairs_by_label[label], recipe_ids)
        for label in conflict_labels
        if label in pairs_by_label
    ]
    known_conflict_keys = [key for key in conflict_keys if key is not None]
    recipe_id = recipe_ids.get(pair.recipe_name)
    horizon = (
        float(agent.replay.horizon(target_key))
        if target_key is not None else None
    )
    target_is_latest = bool(
        target_key is not None
        and recipe_id is not None
        and agent.replay.latest_by_recipe.get(recipe_id) == target_key[1]
    )
    active_count = sum(key in active for key in known_conflict_keys)
    pruned_count = sum(key in pruned for key in known_conflict_keys)
    return {
        "delayed_recurrence_target_active_before": bool(target_key and target_key in active),
        "delayed_recurrence_target_is_latest_before": target_is_latest,
        "delayed_recurrence_actual_grace_horizon_before": horizon,
        "delayed_recurrence_agent_demo_before": int(agent.demo_counter),
        "delayed_recurrence_known_conflicting_variant_count_before": len(known_conflict_keys),
        "delayed_recurrence_conflicting_active_count_before": int(active_count),
        "delayed_recurrence_conflicting_pruned_count_before": int(pruned_count),
        "delayed_recurrence_all_known_conflicts_pruned_before": bool(
            known_conflict_keys and pruned_count == len(known_conflict_keys)
        ),
    }


def _memory_state(
    agent: AdaptiveAgent,
    pair: TaskVariant,
    recipe_ids: Mapping[str, str],
    observed_recipes: set[str],
) -> str:
    key = _pair_key(agent, pair, recipe_ids)
    if key and key in _active_variants(agent):
        return "active_memory"
    if key and (key in _pruned_keys(agent) or key[1] in agent.library.variants.get(key[0], {})):
        return "pruned_memory"
    if pair.recipe_name in observed_recipes:
        return "same_recipe_new_preference"
    return "no_memory"

def _post_mismatch_accuracy(
    correct_flags: Sequence[bool],
    first_mismatch: Optional[int],
) -> Dict[str, float]:
    """Measure within-model accuracy after its first error in an episode.

    This diagnostic is conditional on a mismatch and sufficient following robot
    turns. It is not a switch-aligned adaptation metric, and ineligible episodes
    must not be interpreted as failed recovery.
    """
    flags = [bool(v) for v in correct_flags]
    mismatch_index = int(first_mismatch) if first_mismatch is not None and int(first_mismatch) >= 0 else -1
    has_mismatch = 0 <= mismatch_index < len(flags)
    out = {
        "first_mismatch_rate": 1.0 if has_mismatch else 0.0,
        "first_mismatch_robot_turn": float(mismatch_index if has_mismatch else -1),
    }
    for window in POST_MISMATCH_WINDOWS:
        segment = flags[mismatch_index + 1:mismatch_index + 1 + window] if has_mismatch else []
        eligible = len(segment) == window
        out[f"conditional_post_mismatch_top_1_w{window}"] = (
            _mean(1.0 if value else 0.0 for value in segment)
            if eligible else 0.0
        )
        out[f"conditional_post_mismatch_eligible_w{window}"] = (
            1.0 if eligible else 0.0
        )
    return out


def observe_demo(
    agent: AdaptiveAgent,
    pair: TaskVariant,
    recipe_ids: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    n_steps = len(pair.actions)
    start_time = time.perf_counter()
    agent.start_demo()
    for obs in pair.observations:
        agent.observe(obs)
    match = agent.end_demo()
    if recipe_ids is not None and match.recipe_id is not None:
        recipe_ids[pair.recipe_name] = match.recipe_id
    return {
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "mode": "observe",
        "classification_kind": match.kind,
        "classification_recipe_id": match.recipe_id,
        "matched_variant_id": match.variant_id,
        "episode_wall_s": float(time.perf_counter() - start_time),
        "n_steps": 0,
        "recipe_steps": n_steps,
        "live_top_1": 0.0,
        "live_top_k": 0.0,
        "normalized_human_action_load": None,
        "corrections_per_recipe_step": None,
        "hrc_robot_turn_count": 0,
        "hrc_human_turn_count": n_steps,
        "hrc_human_correction_count": 0,
        "hrc_robot_correct_count": 0,
        "hrc_robot_wrong_count": 0,
        "hrc_robot_top_k_hit_count": 0,
        "hrc_robot_top_1_tie_count": 0,
        "hrc_human_shadow_turn_count": 0,
        "hrc_human_shadow_correct_count": 0,
        "hrc_human_shadow_top_k_hit_count": 0,
        "teacher_forced_prediction_count": 0,
        "teacher_forced_correct_count": 0,
        "teacher_forced_top_k_hit_count": 0,
        "teacher_forced_top_1_tie_count": 0,
        "teacher_forced_total_nll": 0.0,
        "teacher_forced_top_1": None,
        "teacher_forced_top_k": None,
        "teacher_forced_mean_nll": None,
        "teacher_forced_position_coverage": None,
        "commit_attempted": False,
        "_turn_records": [],
        "observation_mode_episode": 1.0,
        "user_observation_required": 1.0,
    }


def assist_demo(
    agent: AdaptiveAgent,
    pair: TaskVariant,
    recipe_ids: Mapping[str, str],
    *,
    config: EvalSettings,
    commit: bool = True,
    observed_pairs: Optional[set[str]] = None,
    observed_recipes: Optional[set[str]] = None,
    memory_state_before: Optional[str] = None,
    preserve_noncommitting_state: bool = True,
    capture_turn_records: bool = True,
) -> Dict[str, Any]:
    """Run one HRC episode, optionally restoring state after a frozen probe."""
    snapshot = agent.snapshot() if (not commit and preserve_noncommitting_state) else None
    expected_recipe_id = recipe_ids.get(pair.recipe_name)
    expected_key_before = _pair_key(agent, pair, recipe_ids)
    expected_variant_known_before = bool(
        expected_key_before is not None
        and expected_key_before[1] in agent.library.variants.get(expected_key_before[0], {})
    )
    pre_observed_pairs = set(observed_pairs or ())
    pre_observed_recipes = set(observed_recipes or ())
    pre_memory_state = memory_state_before or _memory_state(
        agent,
        pair,
        recipe_ids,
        pre_observed_recipes,
    )
    observations = pair.observations
    actual_actions = tuple(observation.action for observation in observations)

    def predict(prefix: Sequence[str]) -> Mapping[str, float]:
        return agent.predict_actions(list(prefix))

    def observe(
        obs: Any,
        distribution: Optional[Mapping[str, float]] = None,
        predicted: Optional[str] = None,
    ) -> None:
        # Learner input excludes evaluator recipe and preference labels.
        agent.observe(
            obs,
            precomputed_distribution=distribution,
            precomputed_prediction=predicted,
        )

    def robot_metadata(context: Any) -> Mapping[str, Any]:
        stats = agent.policy_stats()
        actual_probability = context.distribution.get(context.actual)
        return {
            **stats,
            "actual_action_probability": float(actual_probability) if actual_probability is not None else None,
        }

    start_time = time.perf_counter()
    trace = simulate_episode(
        observations=observations,
        actual_actions=actual_actions,
        current_prefix=lambda: list(agent.current_prefix),
        predict_distribution=predict,
        observe_ground_truth=observe,
        top_k=int(config.top_k),
        min_probability=float(agent.settings.min_probability),
        tie_rng=agent._tie_break_rng,
        timing=DEFAULT_TIMING,
        capture_robot_metadata=robot_metadata,
    )
    match = agent.end_demo() if commit else None
    commit_diagnostics = dict(
        agent.last_commit_stats if commit else {}
    )
    expected_key_after = _pair_key(agent, pair, recipe_ids)
    expected_variant_id = (
        expected_key_after[1] if expected_key_after is not None else None
    )
    candidate_recipe_id = commit_diagnostics.get(
        "candidate_recipe_id", getattr(match, "recipe_id", None)
    )
    candidate_variant_id = commit_diagnostics.get(
        "candidate_variant_id", getattr(match, "variant_id", None)
    )
    candidate_recipe_correct = (
        bool(candidate_recipe_id == expected_recipe_id)
        if expected_recipe_id is not None and candidate_recipe_id is not None else None
    )
    candidate_variant_correct = (
        bool(candidate_variant_id == expected_variant_id)
        if expected_variant_id is not None and candidate_variant_id is not None else None
    )
    candidate_correct = (
        bool(candidate_recipe_correct and candidate_variant_correct)
        if candidate_recipe_correct is not None and candidate_variant_correct is not None else None
    )

    post_observed_pairs = pre_observed_pairs | {pair.label}
    post_observed_recipes = pre_observed_recipes | {pair.recipe_name}
    post_memory_state = (
        _memory_state(agent, pair, recipe_ids, post_observed_recipes)
        if commit else pre_memory_state
    )

    summary = trace.summary
    n_robot = max(1, int(summary.robot_turn_count))
    robot_turns = list(trace.robot_turns)
    human_shadow_turns = list(trace.human_shadow_turns)
    teacher_forced_turns = sorted(
        [*robot_turns, *human_shadow_turns],
        key=lambda turn: int(turn.recipe_step),
    )
    teacher_forced_positions = [int(turn.recipe_step) for turn in teacher_forced_turns]
    expected_positions = list(range(len(observations)))
    if teacher_forced_positions != expected_positions:
        raise RuntimeError(
            "teacher-forced prediction coverage invariant failed: "
            f"expected={expected_positions}, actual={teacher_forced_positions}"
        )
    teacher_forced_count = len(teacher_forced_turns)
    teacher_forced_correct = sum(
        int(turn.correct_top_1) for turn in teacher_forced_turns
    )
    teacher_forced_top_k_hits = sum(
        int(turn.correct_top_k) for turn in teacher_forced_turns
    )
    floor = max(float(agent.settings.min_probability), 1e-12)
    teacher_forced_nll = sum(
        -math.log(max(float(turn.distribution.get(turn.actual, floor)), floor))
        for turn in teacher_forced_turns
    )
    turn_records: List[Dict[str, Any]] = []
    if capture_turn_records:
        for turn in robot_turns:
            turn_records.append({
                "turn_kind": "robot",
                "recipe_step": int(turn.recipe_step),
                "turn_index": int(turn.robot_turn_index),
                "prefix_length": len(turn.prefix),
                "actual": turn.actual,
                "predicted": turn.predicted,
                "correct_top_1": bool(turn.correct_top_1),
                "correct_top_k": bool(turn.correct_top_k),
                "prediction_time": float(turn.prediction_time),
                "scheduled_actor": turn.scheduled_actor,
                "executed_by": turn.executed_by,
                "human_corrected": bool(turn.human_corrected),
                "matches_later_action": bool(
                    turn.matches_later_action
                ),
                "matching_action_offset": (
                    turn.matching_action_offset
                ),
                **dict(turn.metadata),
            })
        for turn in human_shadow_turns:
            turn_records.append({
                "turn_kind": "human_shadow",
                "recipe_step": int(turn.recipe_step),
                "turn_index": int(turn.human_turn_index),
                "prefix_length": len(turn.prefix),
                "actual": turn.actual,
                "predicted": turn.predicted,
                "correct_top_1": bool(turn.correct_top_1),
                "correct_top_k": bool(turn.correct_top_k),
                "prediction_time": float(turn.prediction_time),
                "scheduled_actor": turn.scheduled_actor,
                "executed_by": turn.executed_by,
                "human_corrected": False,
                "matches_later_action": False,
                "matching_action_offset": None,
                **dict(turn.metadata),
            })
    metrics = {
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "mode": "assist",
        # Stratify forgetting and reentry by pre-episode memory state.
        "memory_state_before": pre_memory_state,
        "memory_state_after": post_memory_state,
        "classification_kind": getattr(match, "kind", None),
        "classification_recipe_id": getattr(match, "recipe_id", None),
        "matched_variant_id": getattr(match, "variant_id", None),
        "expected_recipe_id": expected_recipe_id,
        "expected_variant_id": expected_variant_id,
        "expected_variant_known_before": expected_variant_known_before,
        "self_training_opportunity": bool(
            expected_recipe_id is not None and not expected_variant_known_before
        ),
        "commit_decision_id": commit_diagnostics.get("decision_id"),
        "commit_decision": commit_diagnostics.get("decision", "none"),
        "commit_applied": bool(commit_diagnostics.get("commit_applied", False)),
        "commit_confidence": commit_diagnostics.get("confidence"),
        "commit_confidence_parts": commit_diagnostics.get("confidence_parts"),
        "commit_candidate_recipe_id": candidate_recipe_id,
        "commit_variant_id": candidate_variant_id,
        "commit_candidate_recipe_correct": candidate_recipe_correct,
        "commit_candidate_variant_correct": candidate_variant_correct,
        "commit_candidate_correct": candidate_correct,
        "commit_new_variant": bool(commit_diagnostics.get("new_variant", False)),
        "commit_known_variant": bool(commit_diagnostics.get("known_variant", False)),
        "promoted_from_tentative": bool(commit_diagnostics.get("promoted_from_tentative", False)),
        "tentative_first_demo": commit_diagnostics.get("tentative_first_demo"),
        "promotion_delay_demos": commit_diagnostics.get("promotion_delay_demos"),
        "latest_pinned": bool(commit_diagnostics.get("latest_pinned", False)),
        "episode_wall_s": float(time.perf_counter() - start_time),
        "n_steps": int(summary.robot_turn_count),
        "recipe_steps": int(summary.recipe_steps),
        "live_top_1": _safe_div(summary.robot_correct_count, n_robot),
        "live_top_k": _safe_div(summary.top_k_hits, n_robot),
        f"live_top{int(config.top_k)}": _safe_div(summary.top_k_hits, n_robot),
        "total_robot_turn_nll": float(summary.log_loss),
        "mean_nll_per_robot_turn": _safe_div(summary.log_loss, n_robot),
        "mean_nll_per_recipe_step": _safe_div(summary.log_loss, summary.recipe_steps),
        "mean_prediction_wall_s": _mean(t.prediction_time for t in robot_turns),
        "hrc_robot_turn_count": int(summary.robot_turn_count),
        "hrc_human_turn_count": int(summary.human_turn_count),
        "hrc_human_correction_count": int(summary.human_correction_count),
        "hrc_robot_correct_count": int(summary.robot_correct_count),
        "hrc_robot_wrong_count": int(summary.robot_wrong_count),
        "later_action_errors": int(
            summary.later_action_errors
        ),
        "hrc_robot_top_k_hit_count": int(summary.top_k_hits),
        "hrc_robot_top_1_tie_count": int(summary.robot_top_1_tie_count),
        "hrc_human_shadow_turn_count": int(len(human_shadow_turns)),
        "hrc_human_shadow_correct_count": int(sum(int(turn.correct_top_1) for turn in human_shadow_turns)),
        "hrc_human_shadow_top_k_hit_count": int(sum(int(turn.correct_top_k) for turn in human_shadow_turns)),
        "teacher_forced_prediction_count": int(teacher_forced_count),
        "teacher_forced_correct_count": int(teacher_forced_correct),
        "teacher_forced_top_k_hit_count": int(teacher_forced_top_k_hits),
        "teacher_forced_top_1_tie_count": int(
            summary.teacher_forced_top_1_tie_count
        ),
        "teacher_forced_total_nll": float(teacher_forced_nll),
        "teacher_forced_top_1": _safe_div(
            teacher_forced_correct, teacher_forced_count,
        ),
        "teacher_forced_top_k": _safe_div(
            teacher_forced_top_k_hits, teacher_forced_count,
        ),
        "teacher_forced_mean_nll": _safe_div(
            teacher_forced_nll, teacher_forced_count,
        ),
        "teacher_forced_position_coverage": _safe_div(
            teacher_forced_count, len(observations),
        ),
        "normalized_human_action_load": _safe_div(
            summary.human_turn_count + summary.human_correction_count,
            summary.recipe_steps,
        ),
        "corrections_per_recipe_step": _safe_div(
            summary.human_correction_count, summary.recipe_steps,
        ),
        "later_action_error_share": _safe_div(
            summary.later_action_errors,
            summary.robot_wrong_count,
        ),
        "observation_mode_episode": 0.0,
        "user_observation_required": 1.0 if agent._needs_observation else 0.0,
        "commit_attempted": bool(commit),
        "commit_registry_size": commit_diagnostics.get("registry_size"),
        "commit_active_variants": commit_diagnostics.get("active_variants"),
        "archived_variants": commit_diagnostics.get("archived_variants"),
        "commit_registry_recipes": commit_diagnostics.get("registry_recipes"),
        "commit_variants_scored": commit_diagnostics.get("variants_scored"),
        "commit_scoring_wall_s": commit_diagnostics.get("scoring_wall_s"),
        "_turn_records": turn_records,
        **_post_mismatch_accuracy(
            [turn.correct_top_1 for turn in robot_turns],
            summary.first_mismatch_robot_turn,
        ),
    }

    if snapshot is not None:
        agent.restore_from(snapshot)
    return metrics


def _active_pair_labels(
    plan: Plan,
    event_index: int,
) -> set[str]:
    """Return the simulator's active preference pairs at one checkpoint."""
    if not plan.events or event_index < 0:
        return set()
    bounded_index = min(int(event_index), len(plan.events) - 1)
    phase_index = plan.events[bounded_index].tags.get("phase_index")
    for phase in plan.metadata.get("phases", ()):
        if phase.get("index") != phase_index:
            continue
        return {
            f"{recipe}/{preference}"
            for recipe, preferences in phase.get(
                "active_preferences", {}
            ).items()
            for preference in preferences
        }

    active_by_recipe: Dict[str, set[str]] = {}
    for event in plan.events[:bounded_index + 1]:
        preferences = event.tags.get("active_preferences")
        if isinstance(preferences, (list, tuple, set)):
            active_by_recipe[event.pair.recipe_name] = {
                str(preference) for preference in preferences
            }
        else:
            active_by_recipe[event.pair.recipe_name] = {
                event.pair.preference_name
            }
    return {
        f"{recipe}/{preference}"
        for recipe, preferences in active_by_recipe.items()
        for preference in preferences
    }


def _preference_cohort_tags(
    plan: Plan,
    event_index: int,
    learned_pairs: set[str],
    *,
    before_event: bool,
) -> Dict[str, Dict[str, Any]]:
    """Assign evaluator-only temporal and holdout cohorts without model leakage."""
    current = _active_pair_labels(plan, event_index)
    future_start = int(event_index) if before_event else int(event_index) + 1
    future = {
        event.pair.label for event in plan.events[max(0, future_start):]
    }
    holdout = plan.metadata.get("holdout", {})
    holdout_preferences = {
        str(preference)
        for preference in holdout.get("target_preferences", ())
    }
    if holdout.get("preference") is not None:
        holdout_preferences.add(str(holdout["preference"]))

    tags: Dict[str, Dict[str, Any]] = {}
    for pair in plan.eval_pairs:
        if pair.label in current:
            temporal_status = "current_active"
        elif pair.label in learned_pairs and pair.label in future:
            temporal_status = "old_expected_to_recur"
        elif pair.label in learned_pairs:
            temporal_status = "intentionally_stale"
        else:
            temporal_status = "unseen_not_current"
        tags[pair.label] = {
            "preference_temporal_status": temporal_status,
            "heldout_never_learned": bool(
                pair.preference_name in holdout_preferences
                and pair.label not in learned_pairs
            ),
        }
    return tags


def evaluate_frozen(
    agent: AdaptiveAgent,
    pairs: Sequence[TaskVariant],
    recipe_ids: Mapping[str, str],
    *,
    config: EvalSettings,
    checkpoint: str,
    event_index: int,
    context: Mapping[str, Any],
    observed_pairs: set[str],
    observed_recipes: set[str],
    preference_cohort_tags: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
) -> List[Dict[str, Any]]:
    # Reuse one immutable checkpoint across independent frozen probes.
    checkpoint_state = agent.snapshot()
    rows: List[Dict[str, Any]] = []
    restore_required = False
    try:
        for pair in pairs[: max(0, int(config.frozen_pairs))]:
            restore_required = True
            metrics = assist_demo(
                agent,
                pair,
                recipe_ids,
                config=config,
                commit=False,
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                preserve_noncommitting_state=False,
                capture_turn_records=False,
            )
            rows.append({
                **dict(context),
                **dict((preference_cohort_tags or {}).get(pair.label, {})),
                "checkpoint": checkpoint,
                "event_index": int(event_index),
                "pair": pair.label,
                "recipe": pair.recipe_name,
                "preference": pair.preference_name,
                "top_1": float(metrics.get("teacher_forced_top_1", 0.0)),
                "top_k": float(metrics.get("teacher_forced_top_k", 0.0)),
                "teacher_forced_mean_nll": metrics.get("teacher_forced_mean_nll"),
                "teacher_forced_position_coverage": metrics.get(
                    "teacher_forced_position_coverage"
                ),
                "closed_loop_live_top_1": float(metrics.get("live_top_1", 0.0)),
                "closed_loop_live_top_k": float(metrics.get("live_top_k", 0.0)),
                "recipe_steps": int(metrics.get("recipe_steps", 0)),
                "hrc_human_turn_count": int(
                    metrics.get("hrc_human_turn_count", 0)
                ),
                "hrc_human_correction_count": int(
                    metrics.get("hrc_human_correction_count", 0)
                ),
                "normalized_human_action_load": metrics.get(
                    "normalized_human_action_load"
                ),
                "corrections_per_recipe_step": metrics.get(
                    "corrections_per_recipe_step"
                ),
                "memory_state_before": metrics.get("memory_state_before"),
            })
            agent.restore_from(checkpoint_state)
            restore_required = False
    finally:
        if restore_required:
            agent.restore_from(checkpoint_state)
    return rows


def snapshot_memory(agent: AdaptiveAgent, wall_s: float = 0.0) -> Dict[str, Any]:
    active = agent.replay.active
    pruned = agent.replay.pruned
    registry = agent.library.variants
    registry_variant_count = sum(len(slot) for slot in registry.values())
    active_count = sum(
        (recipe_id, variant_id) in active
        for recipe_id, slot in registry.items() for variant_id in slot
    )
    weights = [float(entry.weight) for entry in active.values()]
    pair_horizon_snapshot = agent.replay.horizons()
    pair_horizon_evidence = agent.replay.horizon_evidence()
    pair_horizons = list(pair_horizon_snapshot.values())
    pair_gap_windows = agent.replay._pair_gap_window
    pair_gap_sample_counts = [len(gaps) for gaps in pair_gap_windows.values()]
    fit_times = _finite(agent.retrain_fit_wall_times)
    total_times = _finite(agent.retrain_total_wall_times)
    build_times = _finite(agent.retrain_build_wall_times)
    flop_estimates = _finite(agent.retrain_flop_estimates)
    estimated_fit_flops = float(sum(flop_estimates))
    fit_wall_s = float(sum(fit_times))
    out = {
        "predictor": agent.predictor_name(),
        "irl_features": agent.irl_feature_name(),
        "top_1_tie_breaking": "seeded_iid_gaussian_argmax_exact_ties",
        "tie_reporting": "aggregate_counts_and_rates",
        "memory_policy": str(agent.replay.policy),
        "latest_pin_enabled": bool(agent.settings.pin_latest),
        "semantic_fallback_enabled": bool(
            agent.settings.semantic_fallback_enabled
        ),
        "latent_strategy_enabled": bool(
            agent.settings.latent_strategy_enabled
        ),
        "active_variants": int(len(active)),
        "pruned_variants": int(len(pruned)),
        "registry_recipes": int(sum(bool(slot) for slot in registry.values())),
        "registry_size": int(registry_variant_count),
        "active_variants": int(active_count),
        "archived_variants": int(registry_variant_count - active_count),
        "registry_steps": int(sum(
            len(variant.ordering) for slot in registry.values() for variant in slot.values()
        )),
        "active_action_steps": int(sum(len(entry.ordering) for entry in active.values())),
        "latest_keys": int(len(agent.replay.latest_keys)),
        "mean_active_weight": _mean(weights),
        "min_active_weight": min(weights) if weights else 0.0,
        "max_active_weight": max(weights) if weights else 0.0,
        "adaptive_pair_horizon_count": int(len(pair_horizons)),
        "mean_pair_grace_horizon_demos": _mean(pair_horizons),
        "max_pair_grace_horizon_demos": max(pair_horizons) if pair_horizons else 0.0,
        "pair_grace_horizons_demos": dict(pair_horizon_snapshot),
        "pair_horizon_evidence": dict(pair_horizon_evidence),
        "pair_recurrence_history_count": int(len(pair_gap_windows)),
        "mean_pair_recurrence_samples": _mean(pair_gap_sample_counts),
        "max_pair_recurrence_samples": max(pair_gap_sample_counts) if pair_gap_sample_counts else 0,
        "pair_recurrence_window": int(agent.replay.reuse_window),
        "pair_recurrence_min_samples": int(agent.replay.reuse_min_samples),
        "pair_downward_half_life_samples": float(agent.replay.pair_downward_half_life),
        "pair_recurrence_quantile": float(agent.replay.reuse_quantile),
        "pair_recurrence_iqr_multiplier": float(agent.replay.reuse_iqr_multiplier),
        "recipe_recurrence_window": int(agent.replay.recipe_reuse_window),
        "global_recurrence_window": int(agent.replay.global_reuse_window),
        "pair_recurrence_gap_windows_demos": {
            f"{recipe_id}/{variant_id}": list(gaps)
            for (recipe_id, variant_id), gaps in sorted(pair_gap_windows.items())
        },
        "demo_counter": int(agent.demo_counter),
        "retrain_cycle": int(agent.retrain_cycle),
        "wall_s": float(wall_s),
        "training_total_retrain_wall_s": float(sum(total_times)),
        "training_fit_wall_s": fit_wall_s,
        "training_build_wall_s": float(sum(build_times)),
        "training_retrain_count": int(len(total_times)),
        "training_skipped_retrain_count": int(agent.skipped_trains),
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
    out.update(agent.diagnostics())
    return out

def _active_only_audit_row(
    agent: AdaptiveAgent,
    context: Mapping[str, Any],
    config: EvalSettings,
) -> Dict[str, Any]:
    row = {
        **dict(context),
        "diagnostic_type": "active_only_pruned_influence_audit",
        "active_variants": len(agent.replay.active),
        "pruned_variants": len(agent.replay.pruned),
    }
    try:
        result = dict(agent.audit_pruning(
            max_prefixes=int(config.audit_prefixes),
            tolerance=float(config.audit_tolerance),
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


def _apply_oracle_pruning(
    agent: AdaptiveAgent,
    future_events: Sequence[Event],
    recipe_ids: Mapping[str, str],
) -> Dict[str, Any]:
    future_pairs = {event.pair.label for event in future_events}
    future_recipes = {event.pair.recipe_name for event in future_events}
    future_keys = {
        key
        for event in future_events
        if (key := _pair_key(agent, event.pair, recipe_ids)) is not None
    }
    active_before = _active_variants(agent)
    pruned_before = _pruned_keys(agent)
    discard = sorted((active_before | pruned_before) - future_keys)
    active_discard = sorted(active_before - future_keys)
    pruned_discard = sorted(pruned_before - future_keys)
    if discard:
        agent.discard(discard)

    restored_future = sorted(_pruned_keys(agent) & future_keys)
    for key in restored_future:
        removed = agent.replay.pruned.pop(key)
        agent.replay.active[key] = MemoryItem(
            recipe_id=removed.recipe_id,
            variant_id=removed.variant_id,
            ordering=removed.ordering,
            weight=1.0,
            added_step=removed.added_step,
            added_cycle=removed.added_cycle,
            last_seen_step=removed.last_seen_step,
            transitions=removed.transitions,
        )

    # Oracle retention is binary: a known future-supported preference remains
    # active at full replay weight until its final occurrence, then disappears.
    weight_changed = False
    for key in _active_variants(agent) & future_keys:
        weight_changed = weight_changed or not math.isclose(
            float(agent.replay.active[key].weight),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        agent.replay.active[key].weight = 1.0

    for recipe_id in {key[0] for key in _active_variants(agent)}:
        latest = agent.replay.latest_by_recipe.get(recipe_id)
        if latest is not None and (recipe_id, latest) in agent.replay.active:
            continue
        replacement = max(
            agent.replay.recipe_items(recipe_id),
            key=lambda entry: (entry.last_seen_step, entry.variant_id),
            default=None,
        )
        if replacement is not None:
            agent.replay.mark_latest(recipe_id, replacement.variant_id)
            agent.library.latest[recipe_id] = replacement.variant_id

    if active_discard or restored_future or weight_changed:
        agent.refresh()

    active_after = _active_variants(agent)
    pruned_after = _pruned_keys(agent)
    registry = {
        (str(recipe_id), str(variant_id))
        for recipe_id, slot in agent.library.variants.items()
        for variant_id in slot
    }
    missing = sorted((future_keys & registry) - active_after)
    unexpected = sorted((active_after | pruned_after) - future_keys)
    if missing or unexpected:
        raise RuntimeError(
            "clairvoyant recipe-preference retention invariant failed: "
            f"missing_future={missing}, retained_without_future={unexpected}"
        )
    return {
        "diagnostic_type": "clairvoyant_pruning",
        "oracle_pruned_active_variants": len(active_discard),
        "oracle_pruned_archived_variants": len(pruned_discard),
        "oracle_restored_future_variants": len(restored_future),
        "oracle_reset_future_weights": int(weight_changed),
        "oracle_active_variants_before": len(active_before),
        "oracle_active_variants_after": len(active_after),
        "oracle_archived_variants_before": len(pruned_before),
        "oracle_archived_variants_after": len(pruned_after),
        "oracle_future_recipe_count": len(future_recipes),
        "oracle_future_pair_count": len(future_pairs),
        "oracle_future_variant_count": len(future_keys),
        "oracle_missing_future_variants": len(missing),
        "oracle_retained_without_future": len(unexpected),
        "oracle_retention_policy": "future_filtered_with_full_fallback",
        "oracle_pruning_rule": "retain_exact_known_future_variants",
        "oracle_decay_policy": "binary_keep_until_final_occurrence",
        "oracle_pruned_keys": [
            f"{recipe_id}:{variant_id}"
            for recipe_id, variant_id in discard
        ],
    }


_ORACLE_HIGHER_IS_BETTER = (
    "teacher_forced_top_1",
    "teacher_forced_top_k",
    "live_top_1",
    "live_top_k",
    "top_1",
    "top_k",
    "closed_loop_live_top_1",
    "closed_loop_live_top_k",
)
_ORACLE_LOWER_IS_BETTER = (
    "teacher_forced_mean_nll",
    "mean_nll_per_robot_turn",
    "normalized_human_action_load",
    "corrections_per_recipe_step",
)


def _oracle_candidate_noninferior(
    candidate: Mapping[str, Any],
    full_reference: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> Tuple[bool, Tuple[str, ...]]:
    """Return whether future filtering is no worse than Full on every headline metric."""
    regressions: List[str] = []
    for metric in _ORACLE_HIGHER_IS_BETTER:
        candidate_value = candidate.get(metric)
        reference_value = full_reference.get(metric)
        if (
            not isinstance(reference_value, (int, float))
            or not math.isfinite(float(reference_value))
        ):
            continue
        if (
            not isinstance(candidate_value, (int, float))
            or not math.isfinite(float(candidate_value))
            or float(candidate_value) + tolerance < float(reference_value)
        ):
            regressions.append(metric)
    for metric in _ORACLE_LOWER_IS_BETTER:
        candidate_value = candidate.get(metric)
        reference_value = full_reference.get(metric)
        if (
            not isinstance(reference_value, (int, float))
            or not math.isfinite(float(reference_value))
        ):
            continue
        if (
            not isinstance(candidate_value, (int, float))
            or not math.isfinite(float(candidate_value))
            or float(candidate_value) > float(reference_value) + tolerance
        ):
            regressions.append(metric)
    if full_reference.get("commit_correct") is True and candidate.get("commit_correct") is not True:
        regressions.append("commit_correct")
    for metric in ("false_variant_creation", "false_latest_promotion"):
        if candidate.get(metric) is True and full_reference.get(metric) is not True:
            regressions.append(metric)
    return not regressions, tuple(regressions)


def _restore_oracle_candidate(
    candidate: AdaptiveAgent,
    full_snapshot: AdaptiveAgent,
) -> None:
    """Copy Full state while retaining the oracle's future-filter policy."""
    candidate.restore_from(full_snapshot)
    candidate.replay.policy = "none"
    candidate.replay.post_grace_decay_rate = 0.0


def _select_oracle_probe_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    full_reference_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply the event-level oracle dominance rule to frozen probe rows."""
    if len(candidate_rows) != len(full_reference_rows):
        raise RuntimeError(
            "oracle and Full produced different frozen-probe row counts: "
            f"{len(candidate_rows)} != {len(full_reference_rows)}"
        )
    selected: List[Dict[str, Any]] = []
    for candidate, full_reference in zip(candidate_rows, full_reference_rows):
        if candidate.get("pair") != full_reference.get("pair"):
            raise RuntimeError(
                "oracle and Full frozen probes are not pair-aligned: "
                f"{candidate.get('pair')!r} != {full_reference.get('pair')!r}"
            )
        candidate_ok, regressions = _oracle_candidate_noninferior(
            candidate, full_reference,
        )
        row = dict(candidate if candidate_ok else full_reference)
        row.update({
            "oracle_probe_selection": (
                "future_filtered_noninferior"
                if candidate_ok else "full_dominance_fallback"
            ),
            "oracle_probe_regressed_metrics": list(regressions),
        })
        selected.append(row)
    return selected


def _event_context(
    baseline: str,
    plan: Plan,
    event_index: int,
    requested_mode: str,
    executed_mode: str,
    pair: TaskVariant,
    tags: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "baseline": baseline,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "event_index": int(event_index),
        "requested_mode": requested_mode,
        "mode": executed_mode,
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "preference_axis_values": dict(pair.values),
        "preference_non_default_axes": list(pair.non_default_axes),
        "preference_axis_value_labels": list(_axis_labels(pair)),
        "preference_is_composed": pair.is_composed_preference,
        **dict(tags),
    }


def _record_observed(
    pair: TaskVariant,
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


def _is_phase_boundary(plan: Plan, event_index: int) -> bool:
    """Return whether a tagged schedule phase closes at ``event_index``."""
    if event_index < 0 or event_index >= len(plan.events):
        return False
    tags = plan.events[event_index].tags
    if tags.get("phase_role") != "settled":
        return False
    phase_id = tags.get("phase_id")
    if event_index == len(plan.events) - 1:
        return True
    return plan.events[event_index + 1].tags.get("phase_id") != phase_id


def _periodic_probe_due(
    plan: Plan,
    event_index: int,
) -> bool:
    """Run one diagnostic sweep when a schedule phase closes."""
    return plan.scenario in SCENARIOS and _is_phase_boundary(plan, event_index)


def _pre_event_probe_due(
    requested_mode: str,
    executed_mode: str,
    tags: Mapping[str, Any],
    config: EvalSettings,
) -> bool:
    """Select matched primary-climb probes before assistive interaction."""
    if not bool(config.pre_event_probes) or requested_mode != "assist" or executed_mode != "assist":
        return False
    return bool(tags.get("primary_probe", False))


def _offline_subset(
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
    # Floor preserves 40--50% coverage for odd candidate counts.
    n_selected = max(1, min(len(candidates), int(math.floor(float(fraction) * len(candidates)))))
    rng = random.Random(f"frozen|{axis}|{int(seed)}")
    return tuple(sorted(rng.sample(candidates, n_selected)))


def _fit_frozen(
    agent: AdaptiveAgent,
    *,
    pairs: Sequence[TaskVariant],
    metadata: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Fit an offline reference before deployment without exposing labels to it."""
    lock_deployment = getattr(agent, "lock_deployment", None)
    if not callable(lock_deployment):
        raise TypeError("offline frozen registry entry must implement lock_deployment()")
    recipe_ids: Dict[str, str] = {}
    pretraining_t0 = time.perf_counter()
    for pair in pairs:
        # Offline learner receives semantic actions, never recipe or preference labels.
        observe_demo(agent, pair, recipe_ids)
    locked_metadata = lock_deployment({
        **dict(metadata),
        "offline_pretraining_end_to_end_wall_s": float(time.perf_counter() - pretraining_t0),
    })
    return recipe_ids, dict(locked_metadata)


def _prepare_frozen(
    agent: AdaptiveAgent,
    plan: Plan,
    config: EvalSettings,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Pretrain on paired 40--50% recipe and preference subsets, then lock."""
    recipe_names = _offline_subset(
        plan.selected_recipes,
        config.offline_recipe_fraction,
        seed=plan.seed,
        axis="recipe",
    )
    preference_names = _offline_subset(
        plan.selected_preferences,
        config.offline_preference_fraction,
        seed=plan.seed,
        axis="preference",
    )
    library = recipe_builders()
    pairs = [
        build_task(recipe, preference, library[recipe])
        for recipe in recipe_names for preference in preference_names
    ]
    return _fit_frozen(
        agent,
        pairs=pairs,
        metadata={
            "offline_training_design": "subset_recipes_subset_preferences",
            "offline_training_recipe_fraction_requested": float(config.offline_recipe_fraction),
            "offline_training_preference_fraction_requested": float(config.offline_preference_fraction),
            "offline_training_recipe_count": int(len(recipe_names)),
            "offline_training_preference_count": int(len(preference_names)),
            "offline_training_pair_count": int(len(pairs)),
            "offline_training_recipe_names": list(recipe_names),
            "offline_training_preference_names": list(preference_names),
        },
    )


def _prepare_offline_default(
    agent: AdaptiveAgent,
    plan: Plan,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Pretrain every evaluated recipe on the default preference only, then lock."""
    recipe_names = tuple(sorted({str(recipe_name) for recipe_name in plan.selected_recipes}))
    if not recipe_names:
        raise ValueError("cannot pretrain all-recipes default baseline: scenario has no recipes")
    library = recipe_builders()
    pairs = [
        build_task(recipe, "default", library[recipe])
        for recipe in recipe_names
    ]
    return _fit_frozen(
        agent,
        pairs=pairs,
        metadata={
            "offline_training_design": "all_recipes_default_only",
            "offline_training_recipe_fraction_requested": 1.0,
            "offline_training_recipe_scope": "all_selected_scenario_recipes",
            "offline_training_preference_scope": "default_only",
            "offline_training_recipe_count": int(len(recipe_names)),
            "offline_training_preference_count": 1,
            "offline_training_pair_count": int(len(pairs)),
            "offline_training_recipe_names": list(recipe_names),
            "offline_training_preference_names": ["default"],
        },
    )


def _prepare_holdout_frozen(
    agent: AdaptiveAgent,
    plan: Plan,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Fit and lock a control on the exact ordered holdout-source events."""
    source_events = [
        event for event in plan.events
        if str(event.tags.get("holdout_partition")) == "source"
    ]
    if not source_events:
        raise ValueError(f"{plan.scenario} has no source progression to pretrain a frozen control")
    unique_pairs = {event.pair.label for event in source_events}
    source_recipes = {event.pair.recipe_name for event in source_events}
    source_preferences = {event.pair.preference_name for event in source_events}
    return _fit_frozen(
        agent,
        pairs=[event.pair for event in source_events],
        metadata={
            "offline_training_design": "matched_holdout_source_progression",
            "offline_training_scope": "exact_ordered_source_climb_and_settled_events",
            "offline_training_event_count": int(len(source_events)),
            "offline_training_unique_pair_count": int(len(unique_pairs)),
            "offline_training_recipe_count": int(len(source_recipes)),
            "offline_training_preference_count": int(len(source_preferences)),
            "offline_training_recipe_names": sorted(source_recipes),
            "offline_training_preference_names": sorted(source_preferences),
            "holdout_source_progression_matched": True,
            "heldout_deployment_updates_allowed": False,
        },
    )


def run_stream(
    baseline: str,
    plan: Plan,
    config: EvalSettings,
    *,
    execution_mode_schedule: Optional[Sequence[str]] = None,
    mode_schedule_policy: str = "baseline_local",
) -> RunState:
    if execution_mode_schedule is not None:
        if len(execution_mode_schedule) != len(plan.events):
            raise ValueError(
                "execution_mode_schedule must contain exactly one mode per scenario event; "
                f"got {len(execution_mode_schedule)} for {len(plan.events)} events"
            )
        invalid_modes = sorted({str(mode) for mode in execution_mode_schedule if mode not in {"assist", "observe"}})
        if invalid_modes:
            raise ValueError(f"execution_mode_schedule has invalid modes: {invalid_modes}")
    is_clairvoyant = baseline == MEMORY_ORACLE
    settings = build_settings(plan.seed, config)
    # Independent Settings instances keep the two deterministic RNG streams
    # identical without allowing one fit to advance the other's generator.
    agent = build_agent(
        "full" if is_clairvoyant else baseline,
        replace(settings) if is_clairvoyant else settings,
    )
    full_reference_agent: Optional[AdaptiveAgent] = None
    full_reference_recipe_ids: Dict[str, str] = {}
    if is_clairvoyant:
        full_reference_agent = build_agent("full", replace(settings))
        # The candidate differs from Full only in replay retention. The
        # counterfactual Full agent remains available as a dominance fallback.
        agent.replay = ReplayMemory(agent.settings, policy="none")
    recipe_ids: Dict[str, str] = {}
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
    last_frozen_event: Optional[int] = None
    start_time = time.perf_counter()
    baseline_context: Dict[str, Any] = {}
    holdout_scenario = plan.scenario == HOLDOUT
    if holdout_scenario and baseline in {
        "frozen",
        "offline_default",
    }:
        recipe_ids, baseline_context = _prepare_holdout_frozen(agent, plan)
    elif baseline == "frozen":
        recipe_ids, baseline_context = _prepare_frozen(agent, plan, config)
    elif baseline == "offline_default":
        recipe_ids, baseline_context = _prepare_offline_default(agent, plan)
    # Report offline work separately from online phase costs.
    initial_memory = snapshot_memory(agent)
    offline_training_pairs = {
        f"{recipe}/{preference}"
        for recipe in baseline_context.get("offline_training_recipe_names", ())
        for preference in baseline_context.get(
            "offline_training_preference_names", ()
        )
    }
    pairs_by_label = {event.pair.label: event.pair for event in plan.events}

    for event_index, event in enumerate(plan.events):
        pair = event.pair
        pair_seen_before = pair.label in observed_pairs
        if is_clairvoyant and not pair_seen_before:
            assert full_reference_agent is not None
            _restore_oracle_candidate(
                agent, full_reference_agent.snapshot(),
            )
            recipe_ids = dict(full_reference_recipe_ids)
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
        routing_agent = full_reference_agent if is_clairvoyant else agent
        routing_recipe_ids = (
            full_reference_recipe_ids if is_clairvoyant else recipe_ids
        )
        assert routing_agent is not None
        # Capture evaluator-only memory state before routing or mutation.
        memory_state_before = _memory_state(
            agent,
            pair,
            recipe_ids,
            observed_recipes,
        )
        target_key_before = _pair_key(agent, pair, recipe_ids)
        target_recipe_id_before = recipe_ids.get(pair.recipe_name)
        target_variant_known_before = bool(
            target_key_before is not None
            and target_key_before[1] in agent.library.variants.get(target_key_before[0], {})
        )
        if (
            requested_mode == "assist"
            and config.observe_missing_recipes
            and (
                pair.recipe_name not in observed_recipes
                or config.allow_repeat_observation
            )
            and not _recipe_has_active_variant(
                routing_agent, pair, routing_recipe_ids,
            )
        ):
            natural_executed_mode = "observe"
            natural_route_reason = (
                "assist_routed_to_reobservation_recipe_absent_from_active_memory"
                if pair.recipe_name in observed_recipes else
                "assist_routed_to_observe_recipe_absent_from_active_memory"
            )
        full_execution_mode = (
            str(execution_mode_schedule[event_index])
            if execution_mode_schedule is not None else None
        )
        executed_mode = full_execution_mode or natural_executed_mode
        if (
            executed_mode == "observe"
            and pair.recipe_name in observed_recipes
            and not config.allow_repeat_observation
        ):
            raise ValueError(
                "observation mode is permitted only for a recipe's first "
                f"stream exposure: event={event_index}, recipe={pair.recipe_name!r}"
            )
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
            "repeat_observation": bool(
                executed_mode == "observe"
                and pair.recipe_name in observed_recipes
            ),
        })
        tags.update(_recurrence_tags(
            agent,
            pair,
            recipe_ids,
            target_key_before,
            tags,
            pairs_by_label,
        ))
        if is_clairvoyant:
            tags.update({
                "oracle_reference": MEMORY_ORACLE,
                "reported_as": CLAIRVOYANT_REFERENCE_TAG,
                "leakage_warning": CLAIRVOYANT_LEAKAGE_WARNING,
                "oracle_retention_policy": "future_filtered_with_full_fallback",
                "oracle_decay_policy": "binary_keep_until_final_occurrence",
            })

        context = {
            **_event_context(baseline, plan, event_index, requested_mode, executed_mode, pair, tags),
            **baseline_context,
        }
        active_before = _active_variants(agent)
        pruned_before = _pruned_keys(agent)
        retrain_before = len(agent.retrain_events)

        # Matched probes preserve route, state, and workload cost.
        if _pre_event_probe_due(
            requested_mode,
            executed_mode,
            tags,
            config,
        ):
            probe_context = {
                **context,
                "probe_phase": "pre_event",
                "primary_probe": True,
                "live_event_index": int(event_index),
            }
            candidate_probe_rows = evaluate_frozen(
                agent,
                (pair,),
                recipe_ids,
                config=config,
                checkpoint=f"pre_event_{event_index}",
                event_index=event_index,
                context=probe_context,
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                preference_cohort_tags=_preference_cohort_tags(
                    plan,
                    event_index,
                    set(observed_pairs) | offline_training_pairs,
                    before_event=True,
                ),
            )
            if is_clairvoyant:
                assert full_reference_agent is not None
                reference_probe_rows = evaluate_frozen(
                    full_reference_agent,
                    (pair,),
                    full_reference_recipe_ids,
                    config=config,
                    checkpoint=f"pre_event_{event_index}",
                    event_index=event_index,
                    context=probe_context,
                    observed_pairs=observed_pairs,
                    observed_recipes=observed_recipes,
                    preference_cohort_tags=_preference_cohort_tags(
                        plan,
                        event_index,
                        set(observed_pairs) | offline_training_pairs,
                        before_event=True,
                    ),
                )
                frozen_rows.extend(_select_oracle_probe_rows(
                    candidate_probe_rows, reference_probe_rows,
                ))
            else:
                frozen_rows.extend(candidate_probe_rows)

        def execute_event(
            target_agent: AdaptiveAgent,
            target_recipe_ids: Dict[str, str],
            pre_memory_state: str,
        ) -> Dict[str, Any]:
            if executed_mode == "observe":
                result = observe_demo(target_agent, pair, target_recipe_ids)
                result.update({
                    "memory_state_before": pre_memory_state,
                    "memory_state_after": _memory_state(
                        target_agent,
                        pair,
                        target_recipe_ids,
                        set(observed_recipes) | {pair.recipe_name},
                    ),
                })
                return result
            return assist_demo(
                target_agent,
                pair,
                target_recipe_ids,
                config=config,
                commit=True,
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                memory_state_before=pre_memory_state,
            )

        oracle_selection = "not_applicable"
        oracle_regressions: Tuple[str, ...] = ()
        if not is_clairvoyant:
            row = execute_event(agent, recipe_ids, memory_state_before)
        else:
            assert full_reference_agent is not None
            reference_before = full_reference_agent.snapshot()
            reference_recipe_ids_before = dict(full_reference_recipe_ids)
            reference_active_before = _active_variants(reference_before)
            reference_pruned_before = _pruned_keys(reference_before)
            reference_retrain_before = len(reference_before.retrain_events)
            reference_memory_state = _memory_state(
                reference_before,
                pair,
                full_reference_recipe_ids,
                observed_recipes,
            )
            full_row = execute_event(
                full_reference_agent,
                full_reference_recipe_ids,
                reference_memory_state,
            )
            expected_reference_key = _pair_key(
                reference_before, pair, reference_recipe_ids_before,
            )
            expected_reference_recipe = reference_recipe_ids_before.get(
                pair.recipe_name,
            )
            expected_reference_variant = (
                expected_reference_key[1]
                if expected_reference_key is not None else None
            )

            def annotate_commit_outcome(
                decision_row: Dict[str, Any],
                decision_agent: AdaptiveAgent,
            ) -> None:
                committed_here = bool(
                    executed_mode == "assist"
                    and decision_row.get("commit_applied")
                )
                commit_correct = (
                    bool(
                        decision_row.get("commit_candidate_recipe_id")
                        == expected_reference_recipe
                        and decision_row.get("commit_variant_id")
                        == expected_reference_variant
                    )
                    if committed_here
                    and expected_reference_recipe is not None
                    and expected_reference_variant is not None
                    else None
                )
                decision_row["commit_correct"] = commit_correct
                decision_row["false_variant_creation"] = bool(
                    committed_here
                    and decision_row.get("commit_new_variant")
                    and commit_correct is not True
                )
                latest = (
                    decision_agent.library.latest.get(expected_reference_recipe)
                    if expected_reference_recipe is not None else None
                )
                decision_row["false_latest_promotion"] = bool(
                    committed_here
                    and decision_row.get("latest_pinned")
                    and expected_reference_variant is not None
                    and latest != expected_reference_variant
                )

            annotate_commit_outcome(full_row, full_reference_agent)
            if not pair_seen_before:
                # First exposures must be exactly Full-equivalent. The oracle
                # may use future knowledge only after learning the new pair.
                _restore_oracle_candidate(
                    agent, full_reference_agent.snapshot(),
                )
                recipe_ids = dict(full_reference_recipe_ids)
                row = full_row
                active_before = reference_active_before
                pruned_before = reference_pruned_before
                retrain_before = reference_retrain_before
                memory_state_before = reference_memory_state
                target_key_before = _pair_key(
                    reference_before, pair, reference_recipe_ids_before,
                )
                target_recipe_id_before = reference_recipe_ids_before.get(
                    pair.recipe_name,
                )
                target_variant_known_before = bool(
                    target_key_before is not None
                    and target_key_before[1]
                    in reference_before.library.variants.get(
                        target_key_before[0], {},
                    )
                )
                oracle_selection = "full_for_unseen_pair"
            else:
                candidate_row = execute_event(
                    agent, recipe_ids, memory_state_before,
                )
                annotate_commit_outcome(candidate_row, agent)
                candidate_ok, oracle_regressions = _oracle_candidate_noninferior(
                    candidate_row, full_row,
                )
                if candidate_ok:
                    row = candidate_row
                    oracle_selection = "future_filtered_noninferior"
                else:
                    _restore_oracle_candidate(
                        agent, full_reference_agent.snapshot(),
                    )
                    recipe_ids = dict(full_reference_recipe_ids)
                    row = full_row
                    active_before = reference_active_before
                    pruned_before = reference_pruned_before
                    retrain_before = reference_retrain_before
                    memory_state_before = reference_memory_state
                    target_key_before = _pair_key(
                        reference_before, pair, reference_recipe_ids_before,
                    )
                    target_recipe_id_before = reference_recipe_ids_before.get(
                        pair.recipe_name,
                    )
                    target_variant_known_before = bool(
                        target_key_before is not None
                        and target_key_before[1] in reference_before.library.variants.get(
                            target_key_before[0], {},
                        )
                    )
                    oracle_selection = "full_dominance_fallback"
                    tags.update(_recurrence_tags(
                        reference_before,
                        pair,
                        reference_recipe_ids_before,
                        target_key_before,
                        tags,
                        pairs_by_label,
                    ))
                    context = {
                        **_event_context(
                            baseline,
                            plan,
                            event_index,
                            requested_mode,
                            executed_mode,
                            pair,
                            tags,
                        ),
                        **baseline_context,
                    }
            row.update({
                "oracle_selection": oracle_selection,
                "oracle_full_fallback": bool(
                    oracle_selection != "future_filtered_noninferior"
                ),
                "oracle_regressed_metrics": list(oracle_regressions),
            })
        row.update(context)
        target_key_after = _pair_key(agent, pair, recipe_ids)
        target_recipe_id_after = recipe_ids.get(pair.recipe_name)
        committed = bool(executed_mode == "assist" and row.get("commit_applied"))
        classified_recipe = row.get("commit_candidate_recipe_id")
        committed_variant = row.get("commit_variant_id")
        expected_variant_id = target_key_after[1] if target_key_after is not None else None
        latest_after = (
            agent.library.latest.get(target_recipe_id_after)
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
            "expected_recipe_id": target_recipe_id_before,
            "expected_variant_id": expected_variant_id,
            "expected_variant_known_before": target_variant_known_before,
            "self_training_opportunity": bool(
                executed_mode == "assist"
                and target_recipe_id_before is not None
                and not target_variant_known_before
            ),
            "target_variant_active_before": bool(target_key_before and target_key_before in active_before),
            "target_variant_pruned_before": bool(target_key_before and target_key_before in pruned_before),
            "scheduled_reentry_probe": scheduled_reentry,
            "reentry_probe_target_state_before": reentry_target_state_before if scheduled_reentry else None,
            "actual_reentry_from_pruned": bool(row.get("classification_kind") == "reentry_from_pruned"),
            "commit_recipe_correct": (
                bool(classified_recipe == target_recipe_id_before)
                if committed and target_recipe_id_before is not None else None
            ),
            "commit_variant_correct": (
                bool(committed_variant == expected_variant_id)
                if committed and expected_variant_id is not None else None
            ),
            "commit_correct": (
                bool(
                    classified_recipe == target_recipe_id_before
                    and committed_variant == expected_variant_id
                )
                if committed
                and target_recipe_id_before is not None
                and expected_variant_id is not None else None
            ),
            "false_variant_creation": bool(
                committed
                and row.get("commit_new_variant")
                and not (
                    classified_recipe == target_recipe_id_before
                    and committed_variant == expected_variant_id
                )
            ),
            "false_latest_promotion": bool(
                committed
                and row.get("latest_pinned")
                and target_recipe_id_before is not None
                and expected_variant_id is not None
                and (
                    target_recipe_id_after != target_recipe_id_before
                    or latest_after != expected_variant_id
                )
            ),
            "latest_pin_correct": (
                bool(
                    target_recipe_id_after == target_recipe_id_before
                    and latest_after == expected_variant_id
                )
                if committed and row.get("latest_pinned")
                and target_recipe_id_before is not None
                and expected_variant_id is not None else None
            ),
        })
        for turn in row.pop("_turn_records", []):
            turn_rows.append({**context, **turn})
        episode_rows.append(row)
        _record_observed(pair, observed_recipes, observed_preferences, observed_pairs, preferences_by_recipe, axis_values_by_recipe)

        after_prune: Dict[str, Any] = {}
        if is_clairvoyant:
            after_prune = _apply_oracle_pruning(agent, plan.events[event_index + 1:], recipe_ids)
            oracle_rows.append({
                **after_prune,
                "baseline": baseline,
                "scenario": plan.scenario,
                "seed": int(plan.seed),
                "event_index": int(event_index),
                "timing": "after_event",
                "oracle_reference": MEMORY_ORACLE,
                "reported_as": CLAIRVOYANT_REFERENCE_TAG,
                "leakage_warning": CLAIRVOYANT_LEAKAGE_WARNING,
                "oracle_selection": oracle_selection,
                "oracle_full_fallback": bool(
                    oracle_selection != "future_filtered_noninferior"
                ),
                "oracle_regressed_metrics": list(oracle_regressions),
            })

        active_after = _active_variants(agent)
        pruned_after = _pruned_keys(agent)
        memory_rows.append({
            **context,
            "diagnostic_type": "memory_compute",
            **snapshot_memory(agent),
            "active_added_count": len(active_after - active_before),
            "active_removed_count": len(active_before - active_after),
            "pruned_added_count": len(pruned_after - pruned_before),
            "pruned_removed_count": len(pruned_before - pruned_after),
            "retrain_event_count_delta": len(agent.retrain_events[retrain_before:]),
            "post_event_oracle_pruned_active_variants": after_prune.get("oracle_pruned_active_variants"),
        })

        if config.audit_period > 0 and (event_index + 1) % int(config.audit_period) == 0:
            audit_rows.append(_active_only_audit_row(agent, {**context, "audit_checkpoint": f"event_{event_index}"}, config))

        if _periodic_probe_due(plan, event_index):
            periodic_context = {
                "baseline": baseline,
                "scenario": plan.scenario,
                "seed": int(plan.seed),
                **baseline_context,
            }
            candidate_probe_rows = evaluate_frozen(
                agent,
                plan.eval_pairs,
                recipe_ids,
                config=config,
                checkpoint=f"event_{event_index}",
                event_index=event_index,
                context=periodic_context,
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                preference_cohort_tags=_preference_cohort_tags(
                    plan,
                    event_index,
                    set(observed_pairs) | offline_training_pairs,
                    before_event=False,
                ),
            )
            if is_clairvoyant:
                assert full_reference_agent is not None
                reference_probe_rows = evaluate_frozen(
                    full_reference_agent,
                    plan.eval_pairs,
                    full_reference_recipe_ids,
                    config=config,
                    checkpoint=f"event_{event_index}",
                    event_index=event_index,
                    context=periodic_context,
                    observed_pairs=observed_pairs,
                    observed_recipes=observed_recipes,
                    preference_cohort_tags=_preference_cohort_tags(
                        plan,
                        event_index,
                        set(observed_pairs) | offline_training_pairs,
                        before_event=False,
                    ),
                )
                frozen_rows.extend(_select_oracle_probe_rows(
                    candidate_probe_rows, reference_probe_rows,
                ))
            else:
                frozen_rows.extend(candidate_probe_rows)
            last_frozen_event = event_index

    final_context = {
        "baseline": baseline,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        **baseline_context,
    }
    # Avoid duplicating a final phase-boundary sweep.
    if last_frozen_event != len(plan.events) - 1:
        candidate_probe_rows = evaluate_frozen(
            agent,
            plan.eval_pairs,
            recipe_ids,
            config=config,
            checkpoint="final",
            event_index=len(plan.events) - 1,
            context=final_context,
            observed_pairs=observed_pairs,
            observed_recipes=observed_recipes,
            preference_cohort_tags=_preference_cohort_tags(
                plan,
                len(plan.events) - 1,
                set(observed_pairs) | offline_training_pairs,
                before_event=False,
            ),
        )
        if is_clairvoyant:
            assert full_reference_agent is not None
            reference_probe_rows = evaluate_frozen(
                full_reference_agent,
                plan.eval_pairs,
                full_reference_recipe_ids,
                config=config,
                checkpoint="final",
                event_index=len(plan.events) - 1,
                context=final_context,
                observed_pairs=observed_pairs,
                observed_recipes=observed_recipes,
                preference_cohort_tags=_preference_cohort_tags(
                    plan,
                    len(plan.events) - 1,
                    set(observed_pairs) | offline_training_pairs,
                    before_event=False,
                ),
            )
            frozen_rows.extend(_select_oracle_probe_rows(
                candidate_probe_rows, reference_probe_rows,
            ))
        else:
            frozen_rows.extend(candidate_probe_rows)
    if config.audit_period > 0:
        audit_rows.append(_active_only_audit_row(
            agent,
            {
                **final_context,
                "event_index": len(plan.events) - 1,
                "audit_checkpoint": "final",
            },
            config,
        ))

    return RunState(
        baseline=baseline,
        scenario=plan.scenario,
        seed=plan.seed,
        agent=agent,
        recipe_ids=recipe_ids,
        episode_rows=episode_rows,
        frozen_rows=frozen_rows,
        memory_rows=memory_rows,
        active_audit_rows=audit_rows,
        oracle_pruning_rows=oracle_rows,
        turn_rows=turn_rows,
        initial_memory=initial_memory,
        wall_s=float(time.perf_counter() - start_time),
    )


def aggregate_episodes(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "status": "not_run",
            "n_episodes": 0.0,
            "n_steps": 0.0,
            "recipe_steps": 0.0,
            "live_top_1": None,
            "live_top_k": None,
            "teacher_forced_top_1": None,
            "teacher_forced_top_k": None,
            "teacher_forced_mean_nll": None,
            "teacher_forced_position_coverage": None,
            "n_teacher_forced_predictions": 0.0,
            "n_assist_tasks": 0.0,
            "n_assist_human_actions": 0.0,
            "n_human_corrections": 0.0,
            "normalized_human_action_load": None,
            "mean_corrections_per_task": None,
            "corrections_per_recipe_step": None,
            "testing_episode_wall_s": 0.0,
            "mean_nll_per_robot_turn": None,
            "mean_prediction_wall_s": None,
            "human_shadow_top_1": None,
            "human_shadow_top_k": None,
            "n_human_shadow_turns": 0.0,
            "live_top_1_tie_count": 0.0,
            "live_top_1_tie_rate": None,
            "teacher_forced_top_1_tie_count": 0.0,
            "teacher_forced_top_1_tie_rate": None,
            "later_action_errors": 0.0,
            "later_action_error_share": None,
            "observation_mode_rate": None,
            "user_observation_required_rate": None,
            "primary_prediction_metric": "teacher_forced_top_1",
            "primary_prediction_metric_value": None,
            "primary_hrc_metric": "normalized_human_action_load",
            "primary_hrc_metric_value": None,
        }
    robot_turns = sum(_numeric(row, "hrc_robot_turn_count") for row in rows)
    teacher_forced_predictions = sum(
        _numeric(row, "teacher_forced_prediction_count") for row in rows
    )
    assist_recipe_steps = sum(
        _numeric(row, "recipe_steps")
        for row in rows if row.get("mode") == "assist"
    )
    assist_rows = [row for row in rows if row.get("mode") == "assist"]
    human_turns = sum(_numeric(row, "hrc_human_turn_count") for row in assist_rows)
    corrections = sum(
        _numeric(row, "hrc_human_correction_count") for row in assist_rows
    )
    human_actions = human_turns + corrections
    wrong_predictions = sum(
        _numeric(row, "hrc_robot_wrong_count") for row in rows
    )
    later_reference_action_wrongs = sum(
        _numeric(row, "later_action_errors") for row in rows
    )
    out = {
        "status": "completed",
        "n_episodes": float(len(rows)),
        "n_steps": float(sum(_numeric(row, "n_steps") for row in rows)),
        "recipe_steps": float(sum(_numeric(row, "recipe_steps") for row in rows)),
        "live_top_1": _safe_div(sum(_numeric(row, "hrc_robot_correct_count") for row in rows), robot_turns),
        "live_top_k": _safe_div(sum(_numeric(row, "hrc_robot_top_k_hit_count") for row in rows), robot_turns),
        "live_top_1_tie_count": float(sum(
            _numeric(row, "hrc_robot_top_1_tie_count") for row in rows
        )),
        "live_top_1_tie_rate": _safe_div(
            sum(_numeric(row, "hrc_robot_top_1_tie_count") for row in rows),
            robot_turns,
        ),
        "teacher_forced_top_1": _safe_div(
            sum(_numeric(row, "teacher_forced_correct_count") for row in rows),
            teacher_forced_predictions,
        ),
        "teacher_forced_top_k": _safe_div(
            sum(_numeric(row, "teacher_forced_top_k_hit_count") for row in rows),
            teacher_forced_predictions,
        ),
        "teacher_forced_top_1_tie_count": float(sum(
            _numeric(row, "teacher_forced_top_1_tie_count") for row in rows
        )),
        "teacher_forced_top_1_tie_rate": _safe_div(
            sum(_numeric(row, "teacher_forced_top_1_tie_count") for row in rows),
            teacher_forced_predictions,
        ),
        "teacher_forced_mean_nll": _safe_div(
            sum(_numeric(row, "teacher_forced_total_nll") for row in rows),
            teacher_forced_predictions,
        ),
        "teacher_forced_position_coverage": _safe_div(
            teacher_forced_predictions, assist_recipe_steps,
        ),
        "n_teacher_forced_predictions": float(teacher_forced_predictions),
        "n_assist_tasks": float(len(assist_rows)),
        "n_assist_recipe_steps": float(assist_recipe_steps),
        "n_assist_human_actions": float(human_actions),
        "n_human_corrections": float(corrections),
        "normalized_human_action_load": _safe_div(
            human_actions, assist_recipe_steps,
        ),
        "mean_corrections_per_task": _safe_div(corrections, len(assist_rows)),
        "corrections_per_recipe_step": _safe_div(
            corrections, assist_recipe_steps,
        ),
        "testing_episode_wall_s": float(sum(_numeric(row, "episode_wall_s") for row in rows)),
        "mean_nll_per_robot_turn": _safe_div(sum(_numeric(row, "total_robot_turn_nll") for row in rows), robot_turns),
        "mean_prediction_wall_s": _mean(_numeric(row, "mean_prediction_wall_s") for row in rows),
        "human_shadow_top_1": _safe_div(
            sum(_numeric(row, "hrc_human_shadow_correct_count") for row in rows),
            sum(_numeric(row, "hrc_human_shadow_turn_count") for row in rows),
        ),
        "human_shadow_top_k": _safe_div(
            sum(_numeric(row, "hrc_human_shadow_top_k_hit_count") for row in rows),
            sum(_numeric(row, "hrc_human_shadow_turn_count") for row in rows),
        ),
        "n_human_shadow_turns": float(sum(_numeric(row, "hrc_human_shadow_turn_count") for row in rows)),
        "later_action_errors": float(
            later_reference_action_wrongs
        ),
        "later_action_error_share": _safe_div(
            later_reference_action_wrongs, wrong_predictions,
        ),
        "observation_mode_rate": _safe_div(sum(1.0 for row in rows if row.get("mode") == "observe"), len(rows)),
        "user_observation_required_rate": _mean(_numeric(row, "user_observation_required") for row in rows),
    }
    for window in POST_MISMATCH_WINDOWS:
        eligible = [
            row for row in rows
            if _numeric(
                row, f"conditional_post_mismatch_eligible_w{window}",
            ) > 0.0
        ]
        out[f"conditional_post_mismatch_top_1_w{window}"] = _mean(
            _numeric(row, f"conditional_post_mismatch_top_1_w{window}")
            for row in eligible
        )
        out[f"conditional_post_mismatch_eligible_rate_w{window}"] = (
            _safe_div(len(eligible), len(rows))
        )
    out["primary_prediction_metric"] = "teacher_forced_top_1"
    out["primary_prediction_metric_value"] = out["teacher_forced_top_1"]
    out["primary_hrc_metric"] = "normalized_human_action_load"
    out["primary_hrc_metric_value"] = out["normalized_human_action_load"]
    return out


def summarize_adaptation(
    episode_rows: Sequence[Mapping[str, Any]],
    turn_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize immediate and model-specific post-switch adaptation.

    The early window is one complete demonstration. Recovery is measured
    against each model's own teacher-forced performance on its last three
    same-recipe demonstrations before the switch.
    """
    changed_rows = [
        row for row in episode_rows
        if row.get("mode") == "assist"
        and bool(row.get("recipe_changed_this_phase"))
        and row.get("lifecycle_operation") in {"addition", "removal", "swap"}
        and _numeric(row, "teacher_forced_prediction_count") > 0.0
    ]
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in changed_rows:
        grouped[
            (
                str(row.get("phase_id", row.get("phase_index", "unknown"))),
                str(row.get("recipe", "unknown")),
                str(row.get("pair", row.get("pair_label", "unknown"))),
            )
        ].append(row)

    first_turn_by_event: Dict[int, Mapping[str, Any]] = {}
    for turn in turn_rows:
        event_index = turn.get("event_index")
        recipe_step = turn.get("recipe_step")
        if not isinstance(event_index, int) or not isinstance(recipe_step, int):
            continue
        current = first_turn_by_event.get(event_index)
        if current is None or recipe_step < int(current.get("recipe_step", recipe_step)):
            first_turn_by_event[event_index] = turn

    history_by_recipe: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        if (
            row.get("mode") == "assist"
            and _numeric(row, "teacher_forced_prediction_count") > 0.0
        ):
            history_by_recipe[str(row.get("recipe", "unknown"))].append(row)
    for rows in history_by_recipe.values():
        rows.sort(key=lambda row: int(row.get("event_index", -1)))

    switch_starts_by_recipe: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (phase, recipe, _pair), rows in grouped.items():
        switch_starts_by_recipe[recipe].append((
            phase,
            min(int(row.get("phase_start", -1)) for row in rows),
        ))

    records: List[Dict[str, Any]] = []
    for (phase, recipe, pair_label), rows in sorted(grouped.items()):
        initial_by_exposure = {
            int(row.get("exposure_index_since_recipe_change")): row
            for row in rows
            if isinstance(row.get("exposure_index_since_recipe_change"), int)
        }
        first = initial_by_exposure.get(1)
        if first is None:
            continue
        first_event_index = int(first.get("event_index", -1))
        first_turn = first_turn_by_event.get(first_event_index)
        phase_start = int(first.get("phase_start", first_event_index))
        next_switch_starts = [
            start for other_phase, start in switch_starts_by_recipe[recipe]
            if other_phase != phase and start > phase_start
        ]
        next_switch_start = min(next_switch_starts, default=None)
        continuation = [
            row for row in episode_rows
            if row.get("mode") == "assist"
            and str(row.get("recipe", "unknown")) == recipe
            and str(row.get("pair", row.get("pair_label", "unknown")))
            == pair_label
            and int(row.get("event_index", -1)) >= first_event_index
            and (
                next_switch_start is None
                or int(row.get("event_index", -1)) < next_switch_start
            )
            and _numeric(row, "teacher_forced_prediction_count") > 0.0
        ]
        by_exposure = {
            int(row.get("exposure_index_since_recipe_change")): row
            for row in continuation
            if isinstance(row.get("exposure_index_since_recipe_change"), int)
        }
        previous = [
            row for row in history_by_recipe.get(recipe, ())
            if int(row.get("event_index", -1)) < phase_start
        ][-RECOVERY_BASELINE_EXPOSURES:]
        previous_top_1 = _finite(row.get("teacher_forced_top_1") for row in previous)
        baseline = _mean(previous_top_1) if previous_top_1 else None
        threshold = (
            RECOVERY_TARGET_FRACTION * baseline
            if baseline is not None else None
        )

        recovery_exposure: Optional[int] = None
        recovery_decisions: Optional[int] = None
        cumulative_decisions = 0
        for exposure, row in sorted(by_exposure.items()):
            cumulative_decisions += int(
                _numeric(row, "teacher_forced_prediction_count")
            )
            if (
                threshold is not None
                and _numeric(row, "teacher_forced_top_1") >= threshold
            ):
                recovery_exposure = exposure
                recovery_decisions = cumulative_decisions
                break

        records.append({
            "first_decision_correct": (
                bool(first_turn.get("correct_top_1"))
                if first_turn is not None else None
            ),
            "first_exposure_teacher_forced_top_1": first.get(
                "teacher_forced_top_1"
            ),
            "first_exposure_corrections": _numeric(
                first, "hrc_human_correction_count",
            ),
            "pre_switch_teacher_forced_top_1": baseline,
            "recovery_exposure": recovery_exposure,
            "recovery_decisions": recovery_decisions,
            "recovery_eligible": baseline is not None,
        })

    if not records:
        return {
            "definition": "Model-specific adaptation after non-initial preference changes.",
            "status": "not_run",
            "early_window_exposures": 1,
            "n_switch_targets": 0,
        }

    first_decisions = [
        record for record in records
        if isinstance(record.get("first_decision_correct"), bool)
    ]
    recovery_eligible = [
        record for record in records if bool(record.get("recovery_eligible"))
    ]
    recovered = [
        record for record in recovery_eligible
        if isinstance(record.get("recovery_exposure"), int)
    ]
    return {
        "definition": (
            "The first decision is a non-controlling teacher-forced prediction "
            "at recipe step zero. The early endpoint uses one complete "
            "post-switch demonstration. Recovery is the first post-switch "
            "exposure whose teacher-forced Top-1 reaches 90% of that model's "
            "mean over its last three same-recipe pre-switch demonstrations."
        ),
        "status": "completed",
        "early_window_exposures": 1,
        "recovery_baseline_exposures": RECOVERY_BASELINE_EXPOSURES,
        "recovery_target_fraction": RECOVERY_TARGET_FRACTION,
        "n_switch_targets": len(records),
        "n_first_post_switch_teacher_forced_decisions": len(first_decisions),
        "first_post_switch_teacher_forced_decision_top_1": _mean(
            1.0 if record["first_decision_correct"] else 0.0
            for record in first_decisions
        ),
        "first_post_switch_exposure_teacher_forced_top_1": _mean(
            record.get("first_exposure_teacher_forced_top_1")
            for record in records
        ),
        "mean_cumulative_corrections_first_1_exposure": _mean(
            record.get("first_exposure_corrections") for record in records
        ),
        "total_corrections_first_1_exposure": float(sum(
            _value(record.get("first_exposure_corrections"))
            for record in records
        )),
        "n_recovery_eligible": len(recovery_eligible),
        "n_recovered": len(recovered),
        "recovered_rate": _safe_div(len(recovered), len(recovery_eligible)),
        "mean_exposures_to_recover_90pct": _mean(
            record.get("recovery_exposure") for record in recovered
        ),
        "mean_teacher_forced_decisions_to_recover_90pct": _mean(
            record.get("recovery_decisions") for record in recovered
        ),
        "n_right_censored": len(recovery_eligible) - len(recovered),
        "mean_pre_switch_teacher_forced_top_1": _mean(
            record.get("pre_switch_teacher_forced_top_1")
            for record in recovery_eligible
        ),
    }


def _group_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                grouped[str(item)].append(row)
        else:
            grouped[str(value if value is not None else "unknown")].append(row)
    return {
        group: aggregate_episodes(group_rows)
        for group, group_rows in sorted(grouped.items())
    }


def _phase_index_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate the explicit climb/settled blocks without interleaving them."""
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        phase_role = str(row.get("phase_role") or "unphased")
        phase_index = row.get("phase_index")
        phase_label = (
            f"phase_{int(phase_index):02d}"
            if isinstance(phase_index, int) else "phase_unknown"
        )
        grouped[phase_role][phase_label].append(row)
    return {
        phase_role: {
            phase_label: aggregate_episodes(group_rows)
            for phase_label, group_rows in sorted(by_phase_index.items())
        }
        for phase_role, by_phase_index in sorted(grouped.items())
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
    """Attribute cumulative-snapshot deltas to each phase and phase index."""
    zero = {f"online_{field}": 0.0 for field in _PHASE_TRAINING_FIELDS}
    previous = {field: _numeric(initial_memory, field) for field in _PHASE_TRAINING_FIELDS}
    by_phase: Dict[str, Dict[str, float]] = defaultdict(lambda: dict(zero))
    by_phase_index: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: dict(zero))
    )
    for row in sorted(memory_rows, key=lambda item: _numeric(item, "event_index", -1.0)):
        phase_role = str(row.get("phase_role") or "unphased")
        phase_index = row.get("phase_index")
        phase_label = (
            f"phase_{int(phase_index):02d}"
            if isinstance(phase_index, int) else "phase_unknown"
        )
        for field in _PHASE_TRAINING_FIELDS:
            current = _numeric(row, field)
            delta = max(0.0, current - previous[field])
            previous[field] = current
            by_phase[phase_role][f"online_{field}"] = by_phase[phase_role].get(f"online_{field}", 0.0) + delta
            phase_values = by_phase_index[phase_role][phase_label]
            phase_values[f"online_{field}"] = phase_values.get(f"online_{field}", 0.0) + delta
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
        "per_phase_index": {
            phase: {
                phase_index: dict(values)
                for phase_index, values in sorted(by_index.items())
            }
            for phase, by_index in sorted(by_phase_index.items())
        },
    }


def _add_phase_training_costs(metrics: Dict[str, Any], costs: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach phase-attributed online training costs to phase metric tables."""
    merged = {phase: dict(values) for phase, values in metrics.items()}
    for phase, values in (costs.get("per_phase_role") or {}).items():
        merged.setdefault(str(phase), {}).update(dict(values))
    return merged


def _add_phase_index_training_costs(metrics: Dict[str, Dict[str, Any]], costs: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Attach phase-indexed online training costs without dropping empty assist cells."""
    merged = {
        phase: {index: dict(values) for index, values in by_index.items()}
        for phase, by_index in metrics.items()
    }
    for phase, by_index in (costs.get("per_phase_index") or {}).items():
        target = merged.setdefault(str(phase), {})
        for phase_index, values in by_index.items():
            target.setdefault(str(phase_index), {}).update(dict(values))
    return merged


def summarize_frozen(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize frozen probes without treating an omitted audit as success."""
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
            "n_pairs": float(len(checkpoint_rows)),
            "top_1": _mean(row.get("top_1") for row in checkpoint_rows),
            "top_k": _mean(row.get("top_k") for row in checkpoint_rows),
            "memory_state_counts": dict(Counter(
                str(row.get("memory_state_before", "unknown"))
                for row in checkpoint_rows
            )),
            "normalized_human_action_load": _safe_div(
                sum(
                    _numeric(row, "hrc_human_turn_count")
                    + _numeric(row, "hrc_human_correction_count")
                    for row in checkpoint_rows
                ),
                sum(
                    _numeric(row, "recipe_steps")
                    for row in checkpoint_rows
                ),
            ),
            "mean_corrections_per_task": _safe_div(
                sum(
                    _numeric(row, "hrc_human_correction_count")
                    for row in checkpoint_rows
                ),
                len(checkpoint_rows),
            ),
            "corrections_per_recipe_step": _safe_div(
                sum(
                    _numeric(row, "hrc_human_correction_count")
                    for row in checkpoint_rows
                ),
                sum(
                    _numeric(row, "recipe_steps")
                    for row in checkpoint_rows
                ),
            ),
        }
        for checkpoint, checkpoint_rows in sorted(grouped.items())
    }
    return {
        "definition": "Frozen, non-mutating evaluation of the current active-memory model at scheduled checkpoints.",
        "status": "completed",
        "n_checkpoints": len(checkpoints),
        "n_rows": len(rows),
        "checkpoints": checkpoints,
    }


def summarize_frozen_by(
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
        value: summarize_frozen(group_rows)
        for value, group_rows in sorted(grouped.items())
    }


def summarize_cohorts(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Separate current, recurrent, stale, and held-out frozen probes."""
    cohorts = {
        "current_active": [
            row for row in rows
            if row.get("preference_temporal_status") == "current_active"
        ],
        "old_expected_to_recur": [
            row for row in rows
            if row.get("preference_temporal_status") == "old_expected_to_recur"
        ],
        "intentionally_stale": [
            row for row in rows
            if row.get("preference_temporal_status") == "intentionally_stale"
        ],
        "heldout_never_learned": [
            row for row in rows if bool(row.get("heldout_never_learned"))
        ],
    }
    return {
        "definition": (
            "Evaluator-only temporal cohorts from the fixed simulator schedule. "
            "Old preferences are recurrent when another scheduled occurrence "
            "remains and intentionally stale when none remains. Held-out status "
            "requires a designated holdout pair absent from that model's offline "
            "and deployment training history."
        ),
        "uses_future_schedule_for_evaluation_only": True,
        **{
            cohort: summarize_frozen(cohort_rows)
            for cohort, cohort_rows in cohorts.items()
        },
    }


def summarize_recurrence(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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


def summarize_calibration(turn_rows: Sequence[Mapping[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    """Top-1 confidence calibration for the deployed configured policy."""
    rows = [
        row for row in turn_rows
        if row.get("turn_kind") == "robot"
        and isinstance(row.get("final_confidence"), (int, float))
        and isinstance(row.get("correct_top_1"), bool)
    ]
    if not rows:
        return {
            "definition": "Top-1 confidence calibration on robot turns.",
            "status": "not_run",
            "robot_turn_count": 0,
            "ece": None,
            "top_1_brier": None,
        }
    bins: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        confidence = max(0.0, min(1.0, float(row["final_confidence"])))
        bins[min(max(0, int(n_bins) - 1), int(confidence * max(1, int(n_bins))))].append(row)
    ece = 0.0
    bin_rows: List[Dict[str, Any]] = []
    for index in range(max(1, int(n_bins))):
        values = bins.get(index, [])
        if not values:
            continue
        confidence = _mean(float(row["final_confidence"]) for row in values)
        accuracy = _mean(1.0 if row["correct_top_1"] else 0.0 for row in values)
        ece += (len(values) / len(rows)) * abs(accuracy - confidence)
        bin_rows.append({"bin": index, "n": len(values), "mean_confidence": confidence, "accuracy": accuracy})
    return {
        "definition": "Top-1 confidence calibration on robot turns.",
        "status": "completed",
        "robot_turn_count": len(rows),
        "ece": float(ece),
        "top_1_brier": _mean(
            (float(row["final_confidence"]) - (1.0 if row["correct_top_1"] else 0.0)) ** 2
            for row in rows
        ),
        "bins": bin_rows,
    }


def transfer_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    defaults = DEFAULT_PREFERENCE
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("mode") != "assist" or not isinstance(row.get("preference_axis_values"), Mapping):
            continue
        same = set(row.get("axis_value_seen_same_recipe_labels_before", []) or [])
        other = set(row.get("axis_value_seen_other_recipe_labels_before", []) or [])
        global_seen = set(row.get("axis_value_seen_global_labels_before", []) or [])
        values = row["preference_axis_values"]
        for axis, value in sorted((str(k), str(v)) for k, v in values.items()):
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
                "event_index": row.get("event_index"),
                "pair": row.get("pair"),
                "recipe": row.get("recipe"),
                "preference": row.get("preference"),
                "axis": axis,
                "axis_value": value,
                "axis_value_label": label,
                "axis_transfer_cell": cell,
                "preference_is_composed": bool(row.get("preference_is_composed")),
                "transfer_cell_before": row.get("transfer_cell_before"),
                "mode": "assist",
                "recipe_steps": _numeric(row, "recipe_steps"),
                "live_top_1": _numeric(row, "live_top_1"),
                "live_top_k": _numeric(row, "live_top_k"),
                "teacher_forced_top_1": _numeric(row, "teacher_forced_top_1"),
                "teacher_forced_top_k": _numeric(row, "teacher_forced_top_k"),
                "teacher_forced_prediction_count": _numeric(
                    row, "teacher_forced_prediction_count"
                ),
                "teacher_forced_correct_count": _numeric(
                    row, "teacher_forced_correct_count"
                ),
                "teacher_forced_top_k_hit_count": _numeric(
                    row, "teacher_forced_top_k_hit_count"
                ),
                "teacher_forced_total_nll": _numeric(
                    row, "teacher_forced_total_nll"
                ),
                "corrections_per_recipe_step": _numeric(
                    row, "corrections_per_recipe_step"
                ),
                "hrc_robot_turn_count": _numeric(row, "hrc_robot_turn_count"),
                "hrc_human_turn_count": _numeric(row, "hrc_human_turn_count"),
                "hrc_human_correction_count": _numeric(row, "hrc_human_correction_count"),
                "hrc_robot_correct_count": _numeric(row, "hrc_robot_correct_count"),
                "hrc_robot_wrong_count": _numeric(row, "hrc_robot_wrong_count"),
                "later_action_errors": _numeric(
                    row, "later_action_errors"
                ),
                "hrc_robot_top_k_hit_count": _numeric(row, "hrc_robot_top_k_hit_count"),
            })
    return out


def summarize_transfer(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    axis_rows = transfer_rows(rows)
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
        "by_axis": {k: aggregate_episodes(v) for k, v in sorted(grouped_axis.items())},
        "by_axis_value": {k: aggregate_episodes(v) for k, v in sorted(grouped_value.items())},
        "by_axis_transfer_cell": {k: aggregate_episodes(v) for k, v in sorted(grouped_cell.items())},
    }

def summarize_active_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
            {"event_index": row.get("event_index"), "audit_checkpoint": row.get("audit_checkpoint")}
            for row in failed
        ],
    }

def summarize_commit_safety(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    assist = [row for row in rows if row.get("mode") == "assist"]
    committed = [row for row in assist if row.get("commit_applied")]
    novel_commits = [row for row in committed if row.get("commit_new_variant")]
    latest_pins = [row for row in committed if row.get("latest_pinned")]
    scored = [row for row in committed if row.get("commit_scoring_wall_s") is not None]
    lifecycle = summarize_commit_decisions(parse_commit_records(assist))

    def rate(key: str) -> Optional[float]:
        eligible = [row for row in committed if row.get(key) is not None]
        return _mean(1.0 if bool(row.get(key)) else 0.0 for row in eligible) if eligible else None

    return {
        **lifecycle,
        "definition": "Ground-truth-aware candidate, commit, promotion, latest-pin, recovery, and calibration audit for assist-mode self-training.",
        "commit_scoring_wall_scope": "online_identity_classification_and_confidence_scoring_excluding_retraining",
        "identity_registry_scope": "persistent default archive; active/pruned decay state is unchanged",
        "n_assist_episodes": len(assist),
        "n_committed_assist_episodes": len(committed),
        "recipe_commit_accuracy": rate("commit_recipe_correct"),
        "variant_commit_accuracy": rate("commit_variant_correct"),
        "false_variant_creation_rate": _mean(
            1.0 if row.get("false_variant_creation") else 0.0
            for row in novel_commits
        ) if novel_commits else None,
        "false_latest_promotion_rate": _mean(
            1.0 if row.get("false_latest_promotion") else 0.0
            for row in latest_pins
        ) if latest_pins else None,
        "needs_observation_after_assist_rate": _mean(
            1.0 if row.get("classification_kind") == "needs_observation" else 0.0
            for row in assist
        ) if assist else None,
        "uncertain_known_recipe_abstention_rate": _mean(
            1.0 if row.get("classification_kind") == "known_recipe_uncertain" else 0.0
            for row in assist
        ) if assist else None,
        "assist_unavailable_protocol_error_rate": _mean(
            1.0 if row.get("classification_kind") == "assist_unavailable" else 0.0
            for row in assist
        ) if assist else None,
        "n_registry_scored_commits": len(scored),
        "mean_commit_registry_size": _mean(row.get("commit_registry_size") for row in scored) if scored else None,
        "max_commit_registry_size": max(_finite(row.get("commit_registry_size") for row in scored), default=None),
        "mean_variants_scored_per_commit": _mean(row.get("commit_variants_scored") for row in scored) if scored else None,
        "mean_commit_scoring_wall_s": _mean(row.get("commit_scoring_wall_s") for row in scored) if scored else None,
        "p95_commit_scoring_wall_s": _p95(row.get("commit_scoring_wall_s") for row in scored) if scored else None,
    }


def summarize_reentry(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
        "scheduled_reentry_all_routes": aggregate_episodes(scheduled),
        "scheduled_reentry_assist_only": aggregate_episodes(scheduled_assist),
        "target_pruned_before_probe": aggregate_episodes(pruned_target),
        "target_active_before_probe_control": aggregate_episodes(active_target),
        "confirmed_reentry_from_pruned": aggregate_episodes(confirmed),
        "n_scheduled_reentry_probes": len(scheduled),
        "n_scheduled_reentry_assist_probes": len(scheduled_assist),
        "n_target_pruned_before_probe": len(pruned_target),
        "n_target_active_before_probe_control": len(active_target),
        "n_confirmed_reentry_from_pruned": len(confirmed),
    }


ORACLE_HIGHER_IS_BETTER = {
    "live_top_1",
    "live_top_k",
    "teacher_forced_top_1",
    "teacher_forced_top_k",
    "primary_prediction_metric_value",
}
ORACLE_LOWER_IS_BETTER = {
    "normalized_human_action_load",
    "mean_corrections_per_task",
    "corrections_per_recipe_step",
    "mean_nll_per_robot_turn",
    "teacher_forced_mean_nll",
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
    baseline_score = float(baseline_value)
    oracle_score = float(oracle_value)
    if not math.isfinite(baseline_score) or not math.isfinite(oracle_score):
        return None
    if metric in ORACLE_HIGHER_IS_BETTER:
        advantage = oracle_score - baseline_score
        direction = "higher_is_better"
    elif metric in ORACLE_LOWER_IS_BETTER:
        advantage = baseline_score - oracle_score
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
        "baseline_value": baseline_score,
        "oracle_value": oracle_score,
        "oracle_advantage": advantage,
        "regret_to_clairvoyant": max(0.0, advantage),
    }


def build_oracle_rows(
    per_baseline: Mapping[str, Any],
    oracle_reference: str = MEMORY_ORACLE,
) -> List[Dict[str, Any]]:
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
        for scope in ("assist", "episodes"):
            base_scope = summary.get(scope, {})
            oracle_scope = oracle.get(scope, {})
            if isinstance(base_scope, Mapping) and isinstance(oracle_scope, Mapping):
                add(
                    baseline, f"{scope}.overall", "all",
                    base_scope.get("overall", {}), oracle_scope.get("overall", {}),
                )
        assist = summary.get("assist", {})
        oracle_assist = oracle.get("assist", {})
        for group_name in (
            "by_hypothesis", "by_transfer_cell", "by_memory_state", "by_event_type",
            "by_phase_role", "by_pair_lifecycle", "by_lifecycle_phase_role",
            "by_active_preference_count", "by_preference_history_depth",
            "by_pair_gap_bin", "by_retained_pair_gap_bin",
            "by_post_change_exposure_index", "by_demo_progress_bin",
            "by_holdout_exposure_role", "by_holdout_target_lifecycle",
            "by_holdout_target_type",
            "by_unchanged_recipe_during_other_change", "by_lifecycle_operation",
            "by_lifecycle_decision",
        ):
            base_groups = assist.get(group_name, {})
            oracle_groups = oracle_assist.get(group_name, {})
            if not isinstance(base_groups, Mapping) or not isinstance(oracle_groups, Mapping):
                continue
            for group in sorted(set(base_groups) & set(oracle_groups)):
                add(
                    baseline, f"assist.{group_name}", str(group),
                    base_groups[group], oracle_groups[group],
                )
        base_axis = (
            (summary.get("diagnostics", {}) or {}).get("axis_value_transfer", {}) or {}
        ).get("by_axis_transfer_cell", {})
        oracle_axis = (
            (oracle.get("diagnostics", {}) or {}).get("axis_value_transfer", {}) or {}
        ).get("by_axis_transfer_cell", {})
        if isinstance(base_axis, Mapping) and isinstance(oracle_axis, Mapping):
            for group in sorted(set(base_axis) & set(oracle_axis)):
                add(baseline, "axis_value_transfer.by_axis_transfer_cell", str(group), base_axis[group], oracle_axis[group])
    return rows


def summarize_oracle(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
                        "n_rows": len(metric_rows),
                        "mean_oracle_advantage": _mean(row.get("oracle_advantage") for row in metric_rows),
                        "mean_regret_to_clairvoyant": _mean(row.get("regret_to_clairvoyant") for row in metric_rows),
                        "max_regret_to_clairvoyant": max(_finite(row.get("regret_to_clairvoyant") for row in metric_rows), default=0.0),
                        "by_scope": {
                            scope: {
                                "n_rows": len(scope_rows),
                                "mean_oracle_advantage": _mean(row.get("oracle_advantage") for row in scope_rows),
                                "mean_regret_to_clairvoyant": _mean(row.get("regret_to_clairvoyant") for row in scope_rows),
                            }
                            for scope, scope_rows in sorted(
                                (
                                    (scope, [row for row in metric_rows if str(row.get("scope")) == scope])
                                    for scope in {str(row.get("scope")) for row in metric_rows}
                                ),
                                key=lambda item: item[0],
                            )
                        },
                    }
                    for (candidate_baseline, metric), metric_rows in sorted(grouped.items())
                    if candidate_baseline == baseline
                },
            }
            for baseline in sorted({baseline for baseline, _metric in grouped})
        },
    }


def _scope_metrics(
    rows: Sequence[Mapping[str, Any]],
    phase_training: Mapping[str, Any],
) -> Dict[str, Any]:
    retained = [row for row in rows if row.get("pair_lifecycle_role") == "retained"]
    holdout_target = [
        row for row in rows
        if row.get("holdout_exposure_role") not in {None, "non_target"}
    ]
    post_change = [
        row for row in rows
        if bool(row.get("recipe_changed_this_phase"))
        and row.get("lifecycle_operation") in {"addition", "removal", "swap"}
    ]
    groups = {
        "event_type": "event_type",
        "condition": "condition",
        "hypothesis": "hypothesis_tags",
        "pair_lifecycle": "pair_lifecycle_role",
        "lifecycle_phase_role": "lifecycle_phase_role",
        "active_preference_count": "active_preference_count_for_recipe",
        "preference_history_depth": "preference_history_depth_before",
        "pair_gap_bin": "pair_gap_bin",
        "demo_progress_bin": "demo_progress_bin",
        "holdout_partition": "holdout_partition",
        "holdout_phase_role": "holdout_phase_role",
        "unchanged_recipe_during_other_change": "unchanged_recipe_during_other_change",
        "lifecycle_operation": "lifecycle_operation",
        "lifecycle_decision": "lifecycle_decision",
        "applicability_status": "applicability_status",
        "transfer_cell": "transfer_cell_before",
        "memory_state": "memory_state_before",
    }
    return {
        "overall": aggregate_episodes(rows),
        **{
            f"by_{name}": _group_metrics(rows, field)
            for name, field in groups.items()
        },
        "by_post_change_exposure_index": _group_metrics(
            post_change, "exposure_index_since_recipe_change",
        ),
        "by_retained_pair_gap_bin": _group_metrics(
            retained, "retained_pair_gap_bin",
        ),
        "by_holdout_exposure_role": _group_metrics(
            holdout_target, "holdout_exposure_role",
        ),
        "by_holdout_target_lifecycle": _group_metrics(
            holdout_target, "holdout_target_lifecycle_role",
        ),
        "by_holdout_target_type": _group_metrics(
            holdout_target, "holdout_target_type",
        ),
        "by_phase_role": _add_phase_training_costs(
            _group_metrics(rows, "phase_role"), phase_training,
        ),
        "by_phase_index": _add_phase_index_training_costs(
            _phase_index_metrics(rows), phase_training,
        ),
    }


def summarize_stream(stream: RunState) -> Dict[str, Any]:
    assist_rows = [row for row in stream.episode_rows if row.get("mode") == "assist"]
    phase_training = _phase_training_costs(stream.memory_rows, stream.initial_memory)
    episodes = _scope_metrics(stream.episode_rows, phase_training)
    episodes["overall"]["scope_note"] = (
        "Workload includes observations; closed-loop rates pool robot turns, "
        "while teacher-forced rates pool identical ground-truth positions."
    )
    return {
        "assist": _scope_metrics(assist_rows, phase_training),
        "episodes": episodes,
        "training": phase_training,
        "diagnostics": {
            "adaptation_speed": summarize_adaptation(
                stream.episode_rows, stream.turn_rows,
            ),
            "evaluate_frozen": summarize_frozen(stream.frozen_rows),
            "preference_cohorts": summarize_cohorts(
                stream.frozen_rows,
            ),
            "pre_event_frozen_by_condition": summarize_frozen_by(
                [
                    row for row in stream.frozen_rows
                    if row.get("probe_phase") == "pre_event"
                ],
                "condition",
            ),
            "delayed_recurrence": summarize_recurrence(
                stream.episode_rows,
            ),
            "policy_calibration": summarize_calibration(stream.turn_rows),
            "axis_value_transfer": summarize_transfer(assist_rows),
            "mode_schedule": _mode_schedule_summary(stream.episode_rows),
            "active_only_pruned_influence": summarize_active_audit(
                stream.active_audit_rows,
            ),
            "online_commit_safety": summarize_commit_safety(
                stream.episode_rows,
            ),
            "reentry": summarize_reentry(stream.episode_rows),
        },
        "system": snapshot_memory(stream.agent, stream.wall_s),
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


def run_plan(plan: Plan, config: EvalSettings, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    _write_json(out_dir / "plan.json", {
        "scenario": plan.scenario,
        "seed": plan.seed,
        "selected_recipes": list(plan.selected_recipes),
        "selected_preferences": list(plan.selected_preferences),
        "eval_pairs": [pair.label for pair in plan.eval_pairs],
        "metadata": _jsonable(plan.metadata),
        "events": [
            {
                "index": index,
                "mode": event.mode,
                "pair": event.pair.label,
                "tags": {
                    key: value
                    for key, value in event.tags.items()
                    if key not in {
                        "scenario", "user_id", "event_index",
                        "demonstration_number", "recipe",
                        "effective_preference", "pair_label",
                        "one_user_updating_preferences", "top_k",
                        "deployment_structure", "phase_id",
                        "stage", "climb_recipes",
                        "phase_start", "phase_demo_count",
                        "demo_progress_bin", "active_preference_count_for_recipe",
                        "lifecycle_phase_role", "retained_pair_gap_bin",
                        "holdout_kind", "holdout_introduction_phase",
                        "holdout_introduction_step",
                        "holdout_phase_role", "is_post_holdout_settled",
                        "applicability_status", "omitted_pair_count",
                    }
                },
            }
            for index, event in enumerate(plan.events)
        ],
    })
    per_baseline: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_frozen_rows: List[Dict[str, Any]] = []
    all_diagnostic_rows: List[Dict[str, Any]] = []
    all_turn_rows: List[Dict[str, Any]] = []

    baselines = [b for b in config.baselines if b != MEMORY_ORACLE]
    if config.shared_routing and "full" not in baselines:
        raise ValueError(
            "shared_routing=True requires the deployable 'full' system "
            "to be included in EvalSettings.baselines"
        )
    canonical_full_modes: Optional[Tuple[str, ...]] = None
    ordered_baselines = list(baselines)
    if config.shared_routing:
        ordered_baselines = ["full", *[baseline for baseline in baselines if baseline != "full"]]

    run_order = list(ordered_baselines)
    if config.include_oracle:
        run_order.append(MEMORY_ORACLE)

    for baseline in run_order:
        clear_caches()
        if baseline == "full" and config.shared_routing:
            stream = run_stream(
                baseline,
                plan,
                config,
                mode_schedule_policy="full_realized_canonical",
            )
            canonical_full_modes = tuple(
                str(row.get("mode")) for row in stream.episode_rows
            )
        else:
            stream = run_stream(
                baseline,
                plan,
                config,
                execution_mode_schedule=(
                    canonical_full_modes
                ),
                mode_schedule_policy=(
                    "matched_full_realized_execution_schedule_oracle"
                    if baseline == MEMORY_ORACLE
                    else "matched_full_realized_execution_schedule"
                    if canonical_full_modes is not None
                    else "baseline_local"
                ),
            )

        baseline_summary = summarize_stream(stream)
        if baseline == MEMORY_ORACLE:
            baseline_summary["oracle"] = {
                "presentation": CLAIRVOYANT_REFERENCE_TAG,
                "warning": CLAIRVOYANT_LEAKAGE_WARNING,
                "retention": "future_filtered_with_full_fallback",
                "decay": "binary_keep_until_final_occurrence",
                "selection": {
                    "full_for_unseen_pair": int(sum(
                        row.get("oracle_selection") == "full_for_unseen_pair"
                        for row in stream.oracle_pruning_rows
                    )),
                    "future_filtered_noninferior": int(sum(
                        row.get("oracle_selection") == "future_filtered_noninferior"
                        for row in stream.oracle_pruning_rows
                    )),
                    "full_dominance_fallback": int(sum(
                        row.get("oracle_selection") == "full_dominance_fallback"
                        for row in stream.oracle_pruning_rows
                    )),
                },
                "pruning": {
                    "n_prune_decisions": len(stream.oracle_pruning_rows),
                    "total_pruned_active_variants": int(sum(
                        _numeric(row, "oracle_pruned_active_variants")
                        for row in stream.oracle_pruning_rows
                    )),
                    "total_pruned_archived_variants": int(sum(
                        _numeric(row, "oracle_pruned_archived_variants")
                        for row in stream.oracle_pruning_rows
                    )),
                },
            }
        per_baseline[baseline] = baseline_summary
        all_episode_rows.extend(stream.episode_rows)
        all_frozen_rows.extend(stream.frozen_rows)
        all_diagnostic_rows.extend(
            stream.memory_rows
            + stream.active_audit_rows
            + stream.oracle_pruning_rows
        )
        all_turn_rows.extend(stream.turn_rows)
    axis_rows = transfer_rows(all_episode_rows)
    oracle_comparisons = build_oracle_rows(per_baseline)
    tables = out_dir / "tables"
    _write_jsonl_gz(tables / "episodes.jsonl.gz", all_episode_rows)
    _write_jsonl_gz(tables / "turns.jsonl.gz", all_turn_rows)
    _write_jsonl_gz(tables / "frozen_probes.jsonl.gz", all_frozen_rows)
    _write_jsonl_gz(tables / "diagnostics.jsonl.gz", all_diagnostic_rows)
    _write_jsonl_gz(tables / "axis_transfer.jsonl.gz", axis_rows)
    _write_jsonl_gz(tables / "oracle_gaps.jsonl.gz", oracle_comparisons)

    summary = {
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "plan": "plan.json",
        "mode_schedule": {
            "policy": (
                "full_realized_shared_across_all_methods"
                if config.shared_routing else "local_routing"
            ),
            "canonical_full_mode_counts": (
                dict(Counter(canonical_full_modes)) if canonical_full_modes is not None else None
            ),
        },
        "table_rows": {
            "episodes": len(all_episode_rows),
            "turns": len(all_turn_rows),
            "frozen_probes": len(all_frozen_rows),
            "diagnostics": len(all_diagnostic_rows),
            "axis_transfer": len(axis_rows),
            "oracle_gaps": len(oracle_comparisons),
        },
        "per_baseline": per_baseline,
        "oracle_summary": summarize_oracle(oracle_comparisons),
        "wall_s": float(time.perf_counter() - start_time),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _mp_context() -> mp.context.BaseContext:
    return mp.get_context("spawn")


def _worker_count(config: EvalSettings) -> int:
    # One worker owns the quantized GPU model; seed-level GPU replication can
    # otherwise exhaust memory before evaluation begins.
    if "in_context_llm" in config.baselines:
        return 1
    n_seeds = max(1, len(config.seeds))
    return max(1, min(int(config.workers or n_seeds), n_seeds))


def _native_thread_count(config: EvalSettings) -> int:
    return _positive_int(config.threads, DEFAULT_NATIVE_THREADS_PER_WORKER)


def _configure_llm_runtime(config: EvalSettings) -> Dict[str, Any]:
    """Use a fresh CUDA allocator per seed and reduce large-allocation churn."""
    enabled = "in_context_llm" in config.baselines
    if enabled:
        os.environ.setdefault(
            "PYTORCH_ALLOC_CONF",
            LLM_CUDA_ALLOCATOR_CONFIG,
        )
    runtime: Dict[str, Any] = {
        "seed_process_isolation": bool(enabled),
        "native_retry_attempts": 0,
        "pytorch_allocator_config": (
            os.environ.get("PYTORCH_ALLOC_CONF") if enabled else None
        ),
    }
    if not enabled:
        return runtime
    for package in ("torch", "bitsandbytes", "transformers", "accelerate"):
        try:
            runtime[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[f"{package}_version"] = None
    try:
        import torch
    except ImportError:
        runtime.update({"torch_cuda_version": None, "cuda_device": None})
    else:
        runtime["torch_cuda_version"] = str(torch.version.cuda)
        runtime["cuda_device"] = (
            str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available() else None
        )
    return runtime


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "unknown"
    rounded_seconds = int(round(float(seconds)))
    if rounded_seconds >= 3600:
        hours, remainder = divmod(rounded_seconds, 3600)
        minutes, remaining_seconds = divmod(remainder, 60)
        return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
    if rounded_seconds >= 60:
        minutes, remaining_seconds = divmod(rounded_seconds, 60)
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{rounded_seconds}s"


def _eta(
    suite_start: float,
    scenario_start: float,
    scenario_index: int,
    scenario_count: int,
    completed_seeds: int,
    seed_count: int,
    past_scenario_wall: Sequence[float],
) -> str:
    now = time.perf_counter()
    elapsed = now - suite_start
    current_elapsed = now - scenario_start
    if completed_seeds > 0:
        current_remaining = max(
            0.0,
            current_elapsed * (seed_count - completed_seeds)
            / max(1, completed_seeds),
        )
    else:
        current_remaining = (sum(past_scenario_wall) / len(past_scenario_wall)) if past_scenario_wall else None
    future_est = (sum(past_scenario_wall) / len(past_scenario_wall)) if past_scenario_wall else None
    future_remaining = (
        None
        if future_est is None
        else future_est * max(0, scenario_count - scenario_index - 1)
    )
    total_remaining = None if current_remaining is None and future_remaining is None else float(current_remaining or 0.0) + float(future_remaining or 0.0)
    return f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(total_remaining)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _config_payload(config: EvalSettings) -> Dict[str, Any]:
    return dict(_jsonable(asdict(config)))


def _resolved_model_config(config: EvalSettings) -> Dict[str, Any]:
    payload = dict(_jsonable(asdict(build_settings(0, config))))
    payload.pop("seed", None)
    payload["seed_source"] = "evaluation job seed"
    return payload


def _config_hash(config: EvalSettings) -> str:
    payload = _config_payload(config)
    for key in ("output", "run", "resume", "workers", "show_eta", "experiment"):
        payload.pop(key, None)
    payload["resolved_model_config"] = _resolved_model_config(config)
    payload["action_representation"] = ACTION_REPRESENTATION
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_component(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_', or '-': {value!r}")
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return slug[:48] or "evaluation"


def _git_provenance() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ("git", "status", "--porcelain"), cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _runtime_provenance() -> Dict[str, Any]:
    """Record the interpreter and direct CPU dependencies used by a run."""
    package_versions: Dict[str, Optional[str]] = {}
    for package in ("numpy", "matplotlib", "packaging"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions,
    }


def _seed_dir(run_dir: Path, scenario: str, seed: int) -> Path:
    scenario = _safe_component(str(scenario), "scenario")
    return run_dir / "scenarios" / scenario / "seeds" / f"{int(seed):010d}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _create_or_resume_run(config: EvalSettings) -> Tuple[Path, Path, str]:
    root = Path(config.output).expanduser().resolve()
    runs = root / RUNS_DIRNAME
    runs.mkdir(parents=True, exist_ok=True)
    latest = root / LATEST_NAME
    if latest.exists() and not latest.is_symlink():
        raise RuntimeError(f"Expected {latest} to be a symlink")
    digest = _config_hash(config)
    if config.resume and not config.run:
        raise ValueError("resume=True requires an explicit run")
    run = config.run or (
        f"{_slug(config.experiment)}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}__{digest[:10]}"
    )
    run = _safe_component(run, "run")
    run_dir = runs / run
    manifest_path = run_dir / "manifest.json"
    if config.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume run without manifest: {run_dir}")
        manifest = _load_json(manifest_path)
        if manifest.get("config_hash") != digest:
            raise ValueError(f"Run {run!r} does not match the requested evaluation config")
        if manifest.get("experiment") != config.experiment:
            raise ValueError(f"Run {run!r} has a different experiment label")
        _write_json(run_dir / "status.json", {
            "state": "running", "resumed_at_utc": _utc_now(),
            "completed_jobs": 0, "expected_jobs": len(config.scenarios) * len(config.seeds),
        })
        return root, run_dir, digest

    run_dir.mkdir(parents=False, exist_ok=False)
    created_at = _utc_now()
    _write_json(manifest_path, {
        "run": run,
        "experiment": config.experiment,
        "config_hash": digest,
        "created_at_utc": created_at,
        "config": _config_payload(config),
        "resolved_model_config": _resolved_model_config(config),
        "action_representation": ACTION_REPRESENTATION,
        "code": _git_provenance(),
        "runtime": _runtime_provenance(),
        "expected_jobs": [
            {"scenario": scenario, "seed": int(seed)}
            for scenario in config.scenarios for seed in config.seeds
        ],
        "artifacts": {
            "summary": "scenarios/<scenario>/seeds/<seed>/summary.json",
            "tables": "scenarios/<scenario>/seeds/<seed>/tables/*.jsonl.gz",
            "aggregate": "aggregate/",
        },
    })
    _write_json(run_dir / "status.json", {
        "state": "running", "started_at_utc": created_at,
        "completed_jobs": 0, "expected_jobs": len(config.scenarios) * len(config.seeds),
    })
    return root, run_dir, digest


def _update_latest(root: Path, run_dir: Path) -> None:
    latest = root / LATEST_NAME
    if latest.exists() and not latest.is_symlink():
        raise RuntimeError(f"Refusing to replace non-symlink result path: {latest}")
    temporary = root / f".{LATEST_NAME}.{os.getpid()}.{time.time_ns()}"
    try:
        temporary.symlink_to(run_dir.relative_to(root), target_is_directory=True)
        os.replace(temporary, latest)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def _completed_seed_result(run_dir: Path, scenario: str, seed: int) -> Optional[Dict[str, Any]]:
    seed_dir = _seed_dir(run_dir, scenario, seed)
    status_path = seed_dir / "status.json"
    summary_path = seed_dir / "summary.json"
    if not status_path.is_file() or not summary_path.is_file():
        return None
    status = _load_json(status_path)
    if status.get("state") != "complete":
        return None
    return {
        "scenario": scenario,
        "seed": int(seed),
        "key": f"{scenario}/{int(seed):010d}",
        "summary": _load_json(summary_path),
        "summary_path": str(summary_path.relative_to(run_dir)),
        "wall_s": float(status.get("wall_s", 0.0)),
    }


def _run_seed_scenario_job(
    scenario: str, seed: int, config: EvalSettings, run_dir: str,
) -> Dict[str, Any]:
    _apply_native_thread_limit(_native_thread_count(config))
    out_dir = _seed_dir(Path(run_dir), scenario, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    started_at = _utc_now()
    _write_json(out_dir / "status.json", {"state": "running", "started_at_utc": started_at})
    try:
        plan = build_plan(scenario, config, int(seed))
        summary = run_plan(plan, config, out_dir)
    except BaseException as error:
        _write_json(out_dir / "status.json", {
            "state": "failed", "started_at_utc": started_at, "failed_at_utc": _utc_now(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "wall_s": float(time.perf_counter() - start_time),
        })
        raise
    wall_s = float(time.perf_counter() - start_time)
    _write_json(out_dir / "status.json", {
        "state": "complete", "started_at_utc": started_at, "completed_at_utc": _utc_now(),
        "wall_s": wall_s,
    })
    return {
        "scenario": scenario,
        "seed": int(seed),
        "key": f"{scenario}/{int(seed):010d}",
        "summary": summary,
        "summary_path": str((out_dir / "summary.json").relative_to(run_dir)),
        "wall_s": wall_s,
    }


def _pending_seed_results(
    scenario: str,
    seeds: Sequence[int],
    config: EvalSettings,
    run_dir: Path,
    workers: int,
) -> Iterable[Mapping[str, Any]]:
    if "in_context_llm" in config.baselines:
        # Use a fresh CUDA process per seed. Native failures are not retried:
        # repeating a long seed on an unqualified runtime hides instability and
        # wastes compute without improving scientific reproducibility.
        for seed in seeds:
            try:
                with ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=_mp_context(),
                ) as executor:
                    future = executor.submit(
                        _run_seed_scenario_job,
                        scenario,
                        int(seed),
                        config,
                        str(run_dir),
                    )
                    result = dict(future.result())
            except BrokenProcessPool as exc:
                raise RuntimeError(
                    "LLM GPU worker terminated natively for "
                    f"scenario={scenario}, seed={int(seed)}; automatic retry "
                    "is disabled because this runtime is not reproducible"
                ) from exc
            result.update({
                "native_worker_attempts": 1,
                "native_worker_failed_attempts": 0,
                "native_worker_failed_wall_s": 0.0,
            })
            yield result
        return
    if workers <= 1 or len(seeds) <= 1:
        for seed in seeds:
            yield _run_seed_scenario_job(scenario, int(seed), config, str(run_dir))
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=_mp_context(),
    ) as executor:
        futures = [
            executor.submit(
                _run_seed_scenario_job, scenario, int(seed), config, str(run_dir),
            )
            for seed in seeds
        ]
        for future in as_completed(futures):
            yield future.result()


def _paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    key: str,
    n_samples: int = PAIRED_BOOTSTRAP_SAMPLES,
) -> Dict[str, float]:
    """Deterministic percentile CI over paired per-seed differences."""
    values = [float(value) for value in deltas if math.isfinite(float(value))]
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "bootstrap_full_better_fraction": 0.0}
    draws = bootstrap(values, key=key, draws=n_samples)
    return {
        "mean": _mean(values),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_full_better_fraction": float(np.mean(draws > 0.0)),
    }


def summarize_pairs(
    scenario_summaries: Mapping[str, Mapping[str, Any]],
    *,
    n_samples: int = PAIRED_BOOTSTRAP_SAMPLES,
) -> Dict[str, Any]:
    """Bootstrap full-minus-baseline deltas on matched seeds; positive favors full."""
    by_scenario: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for summary in scenario_summaries.values():
        scenario = summary.get("scenario")
        seed = summary.get("seed")
        if isinstance(scenario, str) and isinstance(seed, int):
            by_scenario[scenario][int(seed)] = summary

    metric_specs = (
        ("teacher_forced_top_1", "higher_is_better"),
        ("normalized_human_action_load", "lower_is_better"),
    )
    results: Dict[str, Any] = {}
    for scenario, by_seed in sorted(by_scenario.items()):
        all_baselines = {
            str(name)
            for summary in by_seed.values()
            for name in (summary.get("per_baseline") or {})
        }
        comparisons: Dict[str, Any] = {}
        for baseline in sorted(all_baselines - {"full", MEMORY_ORACLE}):
            by_metric: Dict[str, Any] = {}
            for metric, direction in metric_specs:
                paired: List[Tuple[int, float]] = []
                for seed, summary in sorted(by_seed.items()):
                    per_baseline = summary.get("per_baseline") or {}
                    full_metrics = (
                        (per_baseline.get("full") or {}).get("assist", {}) or {}
                    ).get("overall", {})
                    baseline_metrics = (
                        (per_baseline.get(baseline) or {}).get("assist", {}) or {}
                    ).get("overall", {})
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
                interval = _paired_bootstrap_ci(
                    [delta for _seed, delta in paired],
                    key=f"{scenario}|{baseline}|{metric}",
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


def _summarize_sensitivity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
                "mean_live_top_1": _mean(row.get("live_top_1") for row in group),
                "mean_teacher_forced_top_1": _mean(
                    row.get("teacher_forced_top_1") for row in group
                ),
                "mean_normalized_human_action_load": _mean(
                    row.get("normalized_human_action_load") for row in group
                ),
                "mean_corrections_per_task": _mean(
                    row.get("mean_corrections_per_task") for row in group
                ),
                "mean_corrections_per_recipe_step": _mean(
                    row.get("corrections_per_recipe_step") for row in group
                ),
                "mean_recipe_commit_accuracy": _mean(row.get("recipe_commit_accuracy") for row in group),
                "mean_variant_commit_accuracy": _mean(row.get("variant_commit_accuracy") for row in group),
            }
            for (scenario, setting), group in sorted(grouped.items())
        ],
    }


def run_sensitivity(config: EvalSettings) -> Dict[str, Any]:
    """Run the optional H3 audit separately from the main suite."""
    rows: List[Dict[str, Any]] = []
    for scenario in config.scenarios:
        for seed in config.seeds:
            plan = build_plan(scenario, config, int(seed))
            for setting, overrides in COMMIT_SENSITIVITY_SPECS:
                sensitivity_config = replace(
                    config,
                    model_settings={**dict(config.model_settings), **dict(overrides)},
                    sensitivity=False,
                )
                stream = run_stream("full", plan, sensitivity_config)
                summary = summarize_stream(stream)
                assist = summary["assist"]["overall"]
                safety = summary["diagnostics"]["online_commit_safety"]
                rows.append({
                    "scenario": scenario,
                    "seed": int(seed),
                    "baseline": "full",
                    "sensitivity_setting": setting,
                    "model_settings": dict(overrides),
                    "live_top_1": assist.get("live_top_1"),
                    "live_top_k": assist.get("live_top_k"),
                    "teacher_forced_top_1": assist.get("teacher_forced_top_1"),
                    "teacher_forced_top_k": assist.get("teacher_forced_top_k"),
                    "teacher_forced_mean_nll": assist.get(
                        "teacher_forced_mean_nll"
                    ),
                    "normalized_human_action_load": assist.get(
                        "normalized_human_action_load"
                    ),
                    "mean_corrections_per_task": assist.get(
                        "mean_corrections_per_task"
                    ),
                    "corrections_per_recipe_step": assist.get(
                        "corrections_per_recipe_step"
                    ),
                    "recipe_commit_accuracy": safety.get("recipe_commit_accuracy"),
                    "variant_commit_accuracy": safety.get("variant_commit_accuracy"),
                    "false_variant_creation_rate": safety.get("false_variant_creation_rate"),
                    "false_latest_promotion_rate": safety.get("false_latest_promotion_rate"),
                })
    return {
        "specification": [
            {"sensitivity_setting": setting, "model_settings": dict(overrides)}
            for setting, overrides in COMMIT_SENSITIVITY_SPECS
        ],
        "rows": rows,
        "summary": _summarize_sensitivity(rows),
    }


def run_evaluation(config: EvalSettings) -> Dict[str, Any]:
    llm_runtime = _configure_llm_runtime(config)
    workers = _worker_count(config)
    native_threads = _native_thread_count(config)
    thread_control = _apply_native_thread_limit(native_threads)
    root, run_dir, config_hash = _create_or_resume_run(config)
    expected_jobs = len(config.scenarios) * len(config.seeds)
    suite: Dict[str, Any] = {
        "state": "running",
        "run": run_dir.name,
        "experiment": config.experiment,
        "config_hash": config_hash,
        "execution": {
            "parallelism": "processes",
            "seed_parallelism": "all seeds for one scenario complete before the next scenario starts",
            "workers": workers,
            "threads": native_threads,
            "estimated_max_native_threads": workers * native_threads,
            "thread_control": thread_control,
            "clairvoyant_oracle_included": bool(config.include_oracle),
            "llm_runtime": llm_runtime,
        },
        "scenarios": {},
    }
    scenario_summaries: Dict[str, Mapping[str, Any]] = {}
    suite_start = time.perf_counter()
    past_scenario_wall: List[float] = []
    completed_jobs = 0
    aggregate_dir = run_dir / "aggregate"

    def persist() -> None:
        suite["completed_jobs"] = completed_jobs
        suite["expected_jobs"] = expected_jobs
        _write_json(aggregate_dir / "suite_summary.json", suite)

    def record(result: Mapping[str, Any]) -> None:
        nonlocal completed_jobs
        key = str(result["key"])
        if key in scenario_summaries:
            return
        scenario_summaries[key] = result["summary"]
        suite["scenarios"][key] = {
            "scenario": result["scenario"],
            "seed": int(result["seed"]),
            "state": "complete",
            "summary_path": result["summary_path"],
            "wall_s": float(result["wall_s"]),
            "native_worker_attempts": int(
                result.get("native_worker_attempts", 1)
            ),
            "native_worker_failed_attempts": int(
                result.get("native_worker_failed_attempts", 0)
            ),
            "native_worker_failed_wall_s": float(
                result.get("native_worker_failed_wall_s", 0.0)
            ),
        }
        completed_jobs += 1
        persist()
        _write_json(run_dir / "status.json", {
            "state": "running", "updated_at_utc": _utc_now(),
            "completed_jobs": completed_jobs, "expected_jobs": expected_jobs,
        })

    try:
        for scenario_index, scenario in enumerate(config.scenarios):
            scenario_start = time.perf_counter()
            completed = [
                result for seed in config.seeds
                if (result := _completed_seed_result(run_dir, scenario, int(seed))) is not None
            ]
            for result in completed:
                record(result)
            pending = [
                int(seed) for seed in config.seeds
                if f"{scenario}/{int(seed):010d}" not in scenario_summaries
            ]
            completed_seeds = len(completed)
            _write_json(run_dir / "status.json", {
                "state": "running", "updated_at_utc": _utc_now(),
                "completed_jobs": completed_jobs, "expected_jobs": expected_jobs,
            })
            if config.show_eta:
                print(
                    f"[evaluation] start scenario={scenario} seeds={len(config.seeds)} "
                    f"pending={len(pending)} workers={workers} "
                    f"{_eta(suite_start, scenario_start, scenario_index, len(config.scenarios), completed_seeds, len(config.seeds), past_scenario_wall)}",
                    flush=True,
                )
            for result in _pending_seed_results(
                scenario, pending, config, run_dir, workers,
            ):
                completed_seeds += 1
                record(result)
                if config.show_eta:
                    print(
                        f"[evaluation] done scenario={scenario} seed={result['seed']} "
                        f"seed_wall={_format_seconds(result['wall_s'])} "
                        f"{_eta(suite_start, scenario_start, scenario_index, len(config.scenarios), completed_seeds, len(config.seeds), past_scenario_wall)}",
                        flush=True,
                    )
            scenario_wall = time.perf_counter() - scenario_start
            past_scenario_wall.append(float(scenario_wall))
            suite.setdefault("scenario_wall_s", {})[scenario] = float(scenario_wall)
            persist()
            if config.show_eta:
                print(
                    f"[evaluation] complete scenario={scenario} wall={_format_seconds(scenario_wall)} "
                    f"{_eta(suite_start, scenario_start, scenario_index, len(config.scenarios), len(config.seeds), len(config.seeds), past_scenario_wall)}",
                    flush=True,
                )

        paired = summarize_pairs(scenario_summaries)
        _write_json(aggregate_dir / "paired_bootstrap.json", paired)
        suite["paired_bootstrap_path"] = "aggregate/paired_bootstrap.json"
        if config.sensitivity:
            sensitivity = run_sensitivity(config)
            _write_json(aggregate_dir / "sensitivity.json", sensitivity)
            suite["sensitivity_path"] = "aggregate/sensitivity.json"
        suite["state"] = "complete"
        suite["wall_s"] = float(time.perf_counter() - suite_start)
        suite["completed_at_utc"] = _utc_now()
        persist()
        _write_json(run_dir / "status.json", {
            "state": "complete", "completed_at_utc": suite["completed_at_utc"],
            "completed_jobs": completed_jobs, "expected_jobs": expected_jobs,
            "wall_s": suite["wall_s"],
        })
        _update_latest(root, run_dir)
    except BaseException as error:
        suite["state"] = "failed"
        suite["wall_s"] = float(time.perf_counter() - suite_start)
        suite["error"] = f"{type(error).__name__}: {error}"
        persist()
        _write_json(run_dir / "status.json", {
            "state": "failed", "failed_at_utc": _utc_now(),
            "completed_jobs": completed_jobs, "expected_jobs": expected_jobs,
            "error": suite["error"], "traceback": traceback.format_exc(),
            "wall_s": suite["wall_s"],
        })
        raise
    suite["run_dir"] = str(run_dir)
    suite["latest_dir"] = str(root / LATEST_NAME)
    return suite


def _parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_float_csv(value: str) -> Tuple[float, ...]:
    return tuple(float(item) for item in _parse_csv(value))


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    description: str = "Run the adaptive-preference HRC evaluation harness.",
    default_baselines: Sequence[str] = DEFAULT_BASELINES,
) -> EvalSettings:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run", help="Explicit run directory name; required with --resume.")
    parser.add_argument("--resume", action="store_true", help="Resume completed seeds in an existing run.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in PAPER_SEEDS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--baselines", default=",".join(default_baselines))
    parser.add_argument("--offline-recipe-fraction", type=float, default=0.50)
    parser.add_argument("--offline-preference-fraction", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=DEFAULT_NATIVE_THREADS_PER_WORKER)
    parser.add_argument("--no-eta", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--recipe-count", type=int, default=20)
    parser.add_argument("--panel-size", type=int, default=20)
    parser.add_argument("--phases", type=int, default=7)
    parser.add_argument("--demos", type=int, default=210)
    parser.add_argument("--gap-allocation", type=float, default=0.80)
    parser.add_argument("--hetero-gap-mean", type=int, default=15)
    parser.add_argument("--hetero-gap-min", type=int, default=3)
    parser.add_argument("--hetero-gap-max", type=int, default=40)
    parser.add_argument("--hetero-gap-shape", type=float, default=0.35)
    parser.add_argument("--climb-decay", type=float, default=0.50)
    parser.add_argument("--min-recipes", type=int, default=5)
    parser.add_argument("--max-recipes", type=int, default=8)
    parser.add_argument("--transition-min", type=float, default=0.20)
    parser.add_argument("--transition-max", type=float, default=0.25)
    parser.add_argument("--active-size-weights", default="0.90,0.08,0.02")
    parser.add_argument(
        "--lifecycle-weights", default="0.40,0.20,0.20,0.20",
        help="Comma-separated probabilities for retention, addition, removal, and swap.",
    )
    parser.add_argument("--reentry-rate", type=float, default=0.05)
    parser.add_argument("--recipe-skew", type=float, default=0.70)
    parser.add_argument("--pair-skew", type=float, default=0.70)
    parser.add_argument("--holdout-start", type=float, default=0.60)
    parser.add_argument("--holdout-demos", type=int, default=3)
    parser.add_argument("--frozen-pairs", type=int, default=48)
    parser.add_argument("--audit-period", type=int, default=2)
    parser.add_argument("--audit-prefixes", type=int, default=16)
    parser.add_argument("--audit-tolerance", type=float, default=5e-2)
    parser.add_argument(
        "--local-routing",
        action="store_true",
        help="Disable the default full-realized shared interaction schedule (diagnostic only; not comparable headline evaluation).",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--llm-model",
        help="Local model snapshot or an already-cached Hugging Face model ID.",
    )
    parser.add_argument(
        "--llm-context-tokens",
        type=int,
        help="Maximum prompt plus candidate tokens; defaults to the model setting.",
    )
    parser.add_argument(
        "--llm-candidate-batch",
        type=int,
        help="Number of semantic actions scored together; defaults to one.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--sensitivity", action="store_true", help="Run the opt-in H3 online-commit sensitivity audit after the main suite.")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.llm_model is not None:
        overrides["llm_model"] = str(args.llm_model)
    if args.llm_context_tokens is not None:
        overrides["llm_context_tokens"] = int(args.llm_context_tokens)
    if args.llm_candidate_batch is not None:
        overrides["llm_candidate_batch"] = int(args.llm_candidate_batch)
    return EvalSettings(
        seeds=tuple(int(seed) for seed in _parse_csv(args.seeds)),
        scenarios=_parse_csv(args.scenarios),
        baselines=_parse_csv(args.baselines),
        offline_recipe_fraction=float(args.offline_recipe_fraction),
        offline_preference_fraction=float(args.offline_preference_fraction),
        output=str(args.output),
        run=args.run,
        resume=bool(args.resume),
        workers=int(args.workers),
        threads=_positive_int(args.threads, DEFAULT_NATIVE_THREADS_PER_WORKER),
        show_eta=not bool(args.no_eta),
        include_oracle=not bool(args.no_oracle),
        recipe_count=int(args.recipe_count),
        schedule=ScheduleSettings(
            panel_size=int(args.panel_size),
            phases=int(args.phases),
            demos=int(args.demos),
            gap_allocation=float(args.gap_allocation),
            hetero_gap_mean=int(args.hetero_gap_mean),
            hetero_gap_min=int(args.hetero_gap_min),
            hetero_gap_max=int(args.hetero_gap_max),
            hetero_gap_shape=float(args.hetero_gap_shape),
            climb_decay=float(args.climb_decay),
            min_recipes=int(args.min_recipes),
            max_recipes=int(args.max_recipes),
            transition_min=float(args.transition_min),
            transition_max=float(args.transition_max),
            active_size_weights=tuple(
                _parse_float_csv(args.active_size_weights)
            ),
            lifecycle_weights=tuple(
                _parse_float_csv(args.lifecycle_weights)
            ),
            reentry_rate=float(
                args.reentry_rate
            ),
            recipe_skew=float(args.recipe_skew),
            pair_skew=float(args.pair_skew),
            holdout_start=float(args.holdout_start),
            holdout_demos=int(args.holdout_demos),
        ),
        frozen_pairs=int(args.frozen_pairs),
        audit_period=int(args.audit_period),
        audit_prefixes=int(args.audit_prefixes),
        audit_tolerance=float(args.audit_tolerance),
        shared_routing=not bool(args.local_routing),
        top_k=int(args.top_k),
        profile=bool(args.profile),
        sensitivity=bool(args.sensitivity),
        model_settings=overrides,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    summary = run_evaluation(config)
    print(json.dumps(_jsonable({
        "run_dir": summary["run_dir"],
        "latest_dir": summary["latest_dir"],
        "scenarios": sorted(summary.get("scenarios", {})),
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
