"""Longitudinal evaluation, diagnostics, and artifact generation."""
from __future__ import annotations

import argparse
import ctypes
import faulthandler
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
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
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_NATIVE_THREADS_PER_WORKER = 1
# Each worker holds its own agent, replay and fitted model. Sizing this to
# the CPU count exhausted memory on a 30 GB machine and a killed worker
# surfaces as BrokenProcessPool hours into a run, so the cap is set by
# memory rather than cores. Raise it only with headroom to spare.
DEFAULT_EVALUATION_WORKER_CAP = 10
DEFAULT_RESULTS_ROOT = "eval_results"
RUNS_DIRNAME = "runs"
LATEST_NAME = "latest"
LLM_CUDA_ALLOCATOR_CONFIG = "expandable_segments:True"
# Literal paired seeds keep defaults independent of PRNG implementation details.
PAPER_SEEDS = (
    1337, 2024, 7, 9001, 31415, 42, 271828, 8675309,
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
    "full", "frozen", "offline_default", "offline_all",
    "unpinned", "latest", "fixed", "no_decay", "bc", "ewc",
    "replay_bc",
)
MEMORY_ORACLE = "memory_oracle"
# The optional GPU arm. It owns a quantized model and cannot share a process
# with another cell, so the scheduler treats it differently from every
# symbolic arm.
LLM_BASELINE = "in_context_llm"
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
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        _fsync_directory(path.parent)
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


def _detect_p_core_cpus(pmu_path: Path = Path("/sys/devices/cpu_core/cpus")) -> Optional[FrozenSet[int]]:
    """CPU ids of the performance cores on a hybrid Intel part.

    Linux exposes a ``cpu_core`` PMU on hybrid parts (12th-gen+) whose
    ``cpus`` file lists exactly the P-core ids in ``lscpu`` range syntax
    (``"0-7"`` or ``"0-3,8-11"``); a uniform part has no such file. Absent
    that signal there is nothing to distinguish, so callers should treat
    ``None`` as "don't pin" rather than guess a core set.
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
    """Restrict this process to P-cores so cross-baseline wall-clock stays comparable.

    An unpinned worker can be scheduled onto an E-core under load and run
    slower than a sibling worker on a P-core for reasons that have nothing to
    do with the baseline it is fitting -- the retrain/predict wall-clock
    metrics this evaluation reports and compares across baselines would then
    partly reflect scheduling luck. Every worker process gets the identical
    restriction, so if the pool oversubscribes the P-core set the resulting
    contention is at least shared rather than distinguishing between arms.
    Returns the CPU set actually applied, or ``None`` if pinning did not
    happen (no ``sched_setaffinity`` -- not Linux, or no hybrid P-core PMU).
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

from .adaptive_agent import ACTION_MASK_CONTRACT, AdaptiveAgent
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
    # Save each stream's agent and accumulated rows after every event, and
    # continue from there on a later attempt. Seed-level resume alone requires
    # a completed seed, so a native crash part-way through discards the whole
    # seed; on a runtime that crashes every few million GPU calls that means a
    # long scenario never finishes.
    event_resume: bool = False
    # Process cap for complete scenario cohorts.  With the paper schedule this
    # admits all 3 scenarios x 5 seeds at once (15 workers).
    workers: int = DEFAULT_EVALUATION_WORKER_CAP
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
    # The pruned-influence audit refits a throwaway model; it is read-only with
    # respect to the agent (no state, no RNG, no deployed weights change), so its
    # frequency is a diagnostic-density choice, not a method parameter. Every
    # 2 events made it the single largest cost in the suite.
    audit_period: int = 10
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


def decision_ceiling(pairs: Sequence[TaskVariant]) -> Dict[str, Any]:
    """Best Top-1 any predictor could reach on this stream, and why.

    At prediction time the learner sees a state and a prefix, never a
    preference label. Where one context is followed by different actions in
    different demonstrations, the correct action is not identified by that
    context and no predictor can do better than picking the most frequent
    continuation. Reporting that bound alongside accuracy is what separates
    "the model is weak here" from "this decision was not decidable".

    A ceiling of 1.0 means the stream contains no such contest, so it cannot
    evidence preference learning however well a model scores on it.
    """
    by_state: Dict[Tuple[str, Tuple[int, ...]], Counter] = defaultdict(Counter)
    by_prefix: Dict[Tuple[str, Tuple[str, ...]], Counter] = defaultdict(Counter)
    for pair in pairs:
        prefix: Tuple[str, ...] = ()
        for observation in pair.observations:
            by_state[(pair.recipe_name, tuple(observation.state))][observation.action] += 1
            by_prefix[(pair.recipe_name, prefix)][observation.action] += 1
            prefix = prefix + (observation.action,)

    def bound(table: Mapping[Any, Counter]) -> Dict[str, Any]:
        decisions = sum(sum(counts.values()) for counts in table.values())
        if not decisions: return {"n_decisions": 0, "n_contexts": 0, "ambiguous_decision_share": None, "max_achievable_top_1": None}
        best = sum(max(counts.values()) for counts in table.values())
        ambiguous = sum(sum(counts.values()) for counts in table.values() if len(counts) >= 2)
        return {"n_decisions": decisions, "n_contexts": len(table), "ambiguous_decision_share": _safe_div(ambiguous, decisions), "max_achievable_top_1": _safe_div(best, decisions)}

    state_bound = bound(by_state)
    prefix_bound = bound(by_prefix)
    return {
        "given_state": state_bound,
        "given_prefix": prefix_bound,
        # Positive means history carries preference information the state does
        # not, which is what a prefix-conditioned component can exploit.
        "prefix_information_gain": (None if prefix_bound["max_achievable_top_1"] is None or state_bound["max_achievable_top_1"] is None
                                    else prefix_bound["max_achievable_top_1"] - state_bound["max_achievable_top_1"]),
        "n_pairs": len(pairs),
    }


def build_plan(scenario: str, config: EvalSettings, seed: int) -> Plan:
    try:
        spec = SCENARIO_SPECS[scenario]
    except KeyError as exc:
        raise KeyError(f"unknown scenario {scenario!r}; available={SCENARIOS}") from exc
    if scenario == HOLDOUT:
        plan = build_holdout_plan(config, seed)
    else:
        plan = build_deployment_plan(
            config,
            seed,
            structure=spec.structure,
            holdout=spec.holdout,
            scenario=scenario,
        )
    # Record what this schedule makes decidable, so every reported accuracy can
    # be read against its own bound rather than against 1.0.
    demonstrated = tuple(event.pair for event in plan.events)
    return replace(plan, metadata={**dict(plan.metadata), "decision_ceiling": decision_ceiling(demonstrated)})


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
    """Whether this agent still holds any active variant of the pair's recipe.

    The comprehension variable is deliberately named apart from the target:
    binding it to ``recipe_id`` shadowed the target and made the test
    ``recipe_id == recipe_id``, so this returned True whenever active memory was
    non-empty regardless of which recipe it held. That silently disabled the
    only trigger for baseline-local re-observation, which is why the routing
    ablation recorded zero extra observations for every baseline.
    """
    recipe_id = recipe_ids.get(pair.recipe_name)
    if recipe_id is None: return False
    return any(active_recipe_id == recipe_id for active_recipe_id, _variant_id in _active_variants(agent))


def _recurrence_tags(
    agent: AdaptiveAgent,
    target_key: Optional[VariantKey],
) -> Dict[str, Any]:
    """Classify and describe a delayed recurrence of an already-seen variant.

    The probe condition is read off realized state rather than planned in the
    schedule: an event is a delayed recurrence when this exact variant has been
    demonstrated before and the gap since then has passed the retention horizon
    the agent itself learned for it. The earlier version gated on a
    ``delayed_recurrence_probe`` tag that no code path ever sets, so the whole
    diagnostic reported ``not_run`` in every completed run.

    Conflicts are the recipe's other known variants: the mechanism under test is
    that the target survives, or reenters, while its stale siblings do not.
    """
    if target_key is None: return {}
    recipe_id, variant_id = target_key
    gaps = agent.replay.pair_history(target_key)
    last_seen = agent.replay.pair_last_seen(target_key)
    if not gaps and last_seen is None: return {}

    horizon = float(agent.replay.horizon(target_key))
    elapsed = (
        float(int(agent.demo_counter) - int(last_seen))
        if last_seen is not None else None
    )
    is_probe = bool(elapsed is not None and elapsed > horizon)
    if not is_probe: return {"delayed_recurrence_probe": False}

    active = _active_variants(agent)
    pruned = _pruned_keys(agent)
    conflict_keys = [
        (recipe_id, other_variant_id)
        for other_variant_id in agent.library.variants.get(recipe_id, {})
        if other_variant_id != variant_id
    ]
    active_count = sum(key in active for key in conflict_keys)
    pruned_count = sum(key in pruned for key in conflict_keys)
    return {
        "delayed_recurrence_probe": True,
        "delayed_recurrence_target_active_before": bool(target_key in active),
        "delayed_recurrence_target_is_latest_before": bool(
            agent.replay.latest_by_recipe.get(recipe_id) == variant_id
        ),
        "delayed_recurrence_actual_grace_horizon_before": horizon,
        "delayed_recurrence_agent_demo_before": int(agent.demo_counter),
        "delayed_recurrence_intervening_event_count": elapsed,
        "delayed_recurrence_required_intervening_event_count": horizon,
        "delayed_recurrence_known_conflicting_variant_count_before": len(conflict_keys),
        "delayed_recurrence_conflicting_active_count_before": int(active_count),
        "delayed_recurrence_conflicting_pruned_count_before": int(pruned_count),
        "delayed_recurrence_all_known_conflicts_pruned_before": bool(
            conflict_keys and pruned_count == len(conflict_keys)
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

    pending_prediction: Dict[str, Any] = {}

    def predict(prefix: Sequence[str]) -> Mapping[str, float]:
        deployed_t0 = time.perf_counter()
        distribution = agent.predict_actions(list(prefix))
        deployed_wall_s = time.perf_counter() - deployed_t0
        # Capture decision-time telemetry immediately. This is also what
        # makes human-shadow rows complete.
        decision_stats = dict(agent.varying_policy_stats())
        pending_prediction.clear()
        pending_prediction.update({
            "prefix": tuple(prefix),
            "decision_stats": decision_stats,
            "deployed_prediction_wall_s": float(deployed_wall_s),
        })
        return distribution

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

    def prediction_metadata(context: Any) -> Mapping[str, Any]:
        if tuple(context.prefix) != pending_prediction.get("prefix"):
            raise RuntimeError(
                "prediction metadata did not match the immediately preceding prefix"
            )
        stats = dict(pending_prediction.get("decision_stats") or {})
        actual_probability = context.distribution.get(context.actual)
        metadata: Dict[str, Any] = {
            **stats,
            "actual_action_probability": float(actual_probability) if actual_probability is not None else None,
            "deployed_prediction_wall_s": pending_prediction.get("deployed_prediction_wall_s"),
        }
        pending_prediction.clear()
        return metadata

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
        capture_prediction_metadata=prediction_metadata,
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
            "primary_active_only_contract": "training_inputs_exclude_pruned_variants",
            "primary_active_only_contract_passed": bool(primary) if primary is not None else None,
        }
    except Exception as exc:
        return {
            **row,
            "audit_available": False,
            "passed": False,
            "primary_active_only_contract": "training_inputs_exclude_pruned_variants",
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
            # Pass the replacement's own clock so a recent-set pin scope keeps
            # the sibling window it would have kept during normal operation.
            agent.replay.mark_latest(
                recipe_id, replacement.variant_id, now=replacement.last_seen_step,
            )
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
        "oracle_retention_policy": "future_filtered_after_first_exposure",
        "oracle_pruning_rule": "retain_exact_known_future_variants",
        "oracle_decay_policy": "binary_keep_until_final_occurrence",
        "oracle_pruned_keys": [
            f"{recipe_id}:{variant_id}"
            for recipe_id, variant_id in discard
        ],
    }


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
    """Return the oracle's own frozen probes, pair-aligned against Full.

    The oracle keeps whatever its future-filtered retention produced, including
    when that is worse than Full. Selecting per event on realized outcomes made
    ``oracle_advantage >= 0`` true by construction and left the reference unable
    to be worse than the system it was benchmarking.
    """
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
        row = dict(candidate)
        row["oracle_probe_selection"] = "future_filtered"
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


def _prepare_offline_all(
    agent: AdaptiveAgent,
    plan: Plan,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Pretrain on every selected recipe under every preference, then lock.

    "Every recipe" is this seed's panel, not the whole library: a seed draws
    20 of the 30 recipes and never schedules the other 10, so training on them
    would hand this control a corpus no other arm could have. Preferences need
    no such restriction because the candidate preference set is identical for
    every seed. The corpus is therefore the seed's frozen-probe panel, which
    makes this arm the offline reference for those probes: what one frozen
    model scores when it was handed every behaviour upfront instead of
    acquiring them one demonstration at a time. Non-effective and semantically
    duplicate presets are dropped per recipe, exactly as the probe panel drops
    them, so a recipe contributes one demonstration per distinct behavior.

    In the holdout this arm trains on the source progression like the other two
    frozen controls: training it on the full preference library there would
    hand it the held-out axis value the scenario exists to withhold.
    """
    recipe_names = tuple(sorted({str(recipe_name) for recipe_name in plan.selected_recipes}))
    if not recipe_names:
        raise ValueError("cannot pretrain all-pairs baseline: scenario has no recipes")
    library = recipe_builders()
    pairs: List[TaskVariant] = []
    preference_names: set[str] = set()
    for recipe in recipe_names:
        effective, _applicability = _preference_panel(
            recipe, library[recipe], PREFERENCE_IDS,
        )
        for preference_name, pair in effective.items():
            pairs.append(pair)
            preference_names.add(str(preference_name))
    if not pairs:
        raise ValueError("cannot pretrain all-pairs baseline: no effective pairs")
    return _fit_frozen(
        agent,
        pairs=pairs,
        metadata={
            "offline_training_design": "all_recipes_all_preferences",
            "offline_training_recipe_fraction_requested": 1.0,
            "offline_training_preference_fraction_requested": 1.0,
            "offline_training_recipe_scope": "all_selected_scenario_recipes",
            "offline_training_preference_scope": "all_declared_preferences",
            "offline_training_declared_preference_count": int(len(PREFERENCE_IDS)),
            "offline_training_recipe_count": int(len(recipe_names)),
            "offline_training_preference_count": int(len(preference_names)),
            "offline_training_pair_count": int(len(pairs)),
            "offline_training_recipe_names": list(recipe_names),
            "offline_training_preference_names": sorted(preference_names),
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


EVENT_RESUME_FORMAT = 1


def _event_resume_path(out_dir: Path, baseline: str) -> Path:
    return out_dir / "partial" / "resume" / f"{baseline}.pickle"


def _save_event_resume(
    path: Path,
    baseline: str,
    plan: Plan,
    state: "RunState",
    event_index: int,
    loop_state: Mapping[str, Any],
) -> None:
    """Persist enough state to continue this stream at the next event.

    The GPU model is not part of it: the scorer is reattached on restore, and
    its prompt memoization is a pure cache that costs only time to rebuild.
    Written to a temporary file and renamed, so a crash during the write cannot
    leave a half-written state that a later run would trust.
    """
    agent = state.agent
    scorer = getattr(agent, "scorer", None)
    payload = {
        "format": EVENT_RESUME_FORMAT,
        "baseline": baseline,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "event_count": len(plan.events),
        "event_index": int(event_index),
        "wall_s": float(state.wall_s),
        "loop_state": dict(loop_state),
        "rows": {
            "episode_rows": state.episode_rows,
            "frozen_rows": state.frozen_rows,
            "memory_rows": state.memory_rows,
            "active_audit_rows": state.active_audit_rows,
            "oracle_pruning_rows": state.oracle_pruning_rows,
            "turn_rows": state.turn_rows,
        },
        "initial_memory": state.initial_memory,
        "recipe_ids": dict(state.recipe_ids),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pickle.partial")
    try:
        if scorer is not None:
            agent.scorer = None
        payload["agent"] = agent
        with open(temporary, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if scorer is not None:
            agent.scorer = scorer
        temporary.unlink(missing_ok=True)


def _load_event_resume(
    path: Path,
    baseline: str,
    plan: Plan,
) -> Optional[Dict[str, Any]]:
    """Return a usable resume payload for this exact stream, or None.

    Every mismatch is treated as "start over" rather than an error: a resume
    state is an optimization, and continuing a different plan from it would
    silently produce a run that never happened.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = {
        "format": EVENT_RESUME_FORMAT,
        "baseline": baseline,
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "event_count": len(plan.events),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    event_index = payload.get("event_index")
    if not isinstance(event_index, int) or not 0 <= event_index < len(plan.events):
        return None
    if payload.get("agent") is None:
        return None
    return payload


def run_stream(
    baseline: str,
    plan: Plan,
    config: EvalSettings,
    *,
    execution_mode_schedule: Optional[Sequence[str]] = None,
    mode_schedule_policy: str = "baseline_local",
    event_progress: Optional[Callable[[RunState, int], None]] = None,
    resume_dir: Optional[Path] = None,
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
    # Events already completed by an earlier attempt at this stream. The
    # oracle arm is excluded because it also carries a counterfactual Full
    # agent, which this state does not describe.
    resumed_wall_s = 0.0
    resume_from = -1
    resume_path = (
        _event_resume_path(resume_dir, baseline)
        if resume_dir is not None and not is_clairvoyant else None
    )
    holdout_scenario = plan.scenario == HOLDOUT
    if holdout_scenario and baseline in {
        "frozen",
        "offline_default",
        "offline_all",
    }:
        recipe_ids, baseline_context = _prepare_holdout_frozen(agent, plan)
    elif baseline == "frozen":
        recipe_ids, baseline_context = _prepare_frozen(agent, plan, config)
    elif baseline == "offline_default":
        recipe_ids, baseline_context = _prepare_offline_default(agent, plan)
    elif baseline == "offline_all":
        recipe_ids, baseline_context = _prepare_offline_all(agent, plan)
    # Report offline work separately from online phase costs.
    initial_memory = snapshot_memory(agent)
    offline_training_pairs = {
        f"{recipe}/{preference}"
        for recipe in baseline_context.get("offline_training_recipe_names", ())
        for preference in baseline_context.get(
            "offline_training_preference_names", ()
        )
    }

    if resume_path is not None:
        payload = _load_event_resume(resume_path, baseline, plan)
        if payload is not None:
            scorer = getattr(agent, "scorer", None)
            agent = payload["agent"]
            if scorer is not None:
                # The restored agent was saved without its GPU scorer.
                agent.scorer = scorer
            loop_state = payload["loop_state"]
            recipe_ids = dict(payload["recipe_ids"])
            observed_recipes = set(loop_state["observed_recipes"])
            observed_preferences = set(loop_state["observed_preferences"])
            observed_pairs = set(loop_state["observed_pairs"])
            preferences_by_recipe = defaultdict(set, {
                key: set(value)
                for key, value in loop_state["preferences_by_recipe"].items()
            })
            axis_values_by_recipe = defaultdict(set, {
                key: set(value)
                for key, value in loop_state["axis_values_by_recipe"].items()
            })
            last_frozen_event = loop_state["last_frozen_event"]
            baseline_context = dict(loop_state["baseline_context"])
            offline_training_pairs = set(loop_state["offline_training_pairs"])
            initial_memory = payload["initial_memory"]
            rows = payload["rows"]
            episode_rows = list(rows["episode_rows"])
            frozen_rows = list(rows["frozen_rows"])
            memory_rows = list(rows["memory_rows"])
            audit_rows = list(rows["active_audit_rows"])
            oracle_rows = list(rows["oracle_pruning_rows"])
            turn_rows = list(rows["turn_rows"])
            resumed_wall_s = float(payload["wall_s"])
            resume_from = int(payload["event_index"])
            # A previous attempt may have written a checkpoint and then died
            # before recording it as resumable. Those events are about to be
            # replayed, so their old rows must not survive alongside the new.
            _trim_checkpoints_after(resume_dir, baseline, resume_from)
            print(
                f"[evaluation] resuming {baseline} {plan.scenario} "
                f"seed={int(plan.seed)} after event {resume_from + 1}"
                f"/{len(plan.events)}; "
                f"{resumed_wall_s:.0f}s of earlier work retained",
                flush=True,
            )

    for event_index, event in enumerate(plan.events):
        if event_index <= resume_from:
            continue
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
        tags.update(_recurrence_tags(agent, target_key_before))
        if is_clairvoyant:
            tags.update({
                "oracle_reference": MEMORY_ORACLE,
                "reported_as": CLAIRVOYANT_REFERENCE_TAG,
                "oracle_retention_policy": "future_filtered_after_first_exposure",
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
                # The oracle keeps its own future-filtered outcome, whatever it
                # is. Reverting to Full whenever the future-aware policy did
                # worse on this event's realized metrics made the reference a
                # per-event maximum over {Full, oracle}, so it could never lose
                # to the system it exists to bound.
                row = execute_event(agent, recipe_ids, memory_state_before)
                annotate_commit_outcome(row, agent)
                oracle_selection = "future_filtered"
            row.update({
                "oracle_selection": oracle_selection,
                "oracle_full_fallback": bool(
                    oracle_selection == "full_for_unseen_pair"
                ),
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
            # Recorded on every episode. This was previously gated on a
            # "selective_forgetting_reentry" hypothesis tag that the schedule
            # builder never emits, so the reentry diagnostic saw zero probes in
            # every run while the evidence sat unused on each row.
            "reentry_probe_target_state_before": reentry_target_state_before,
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
                "oracle_selection": oracle_selection,
                "oracle_full_fallback": bool(
                    oracle_selection == "full_for_unseen_pair"
                ),
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

        event_state = RunState(
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
            # Includes work retained from an earlier attempt, so a resumed run
            # reports the cost of producing the results, not of this process.
            wall_s=resumed_wall_s + float(time.perf_counter() - start_time),
        )
        if event_progress is not None:
            event_progress(event_state, event_index)
        # Written after the checkpoint, never before: a resume state that named
        # an event whose rows were not persisted would leave a hole in the
        # per-event series that nothing could later detect.
        if resume_path is not None:
            _save_event_resume(
                resume_path,
                baseline,
                plan,
                event_state,
                event_index,
                {
                    "observed_recipes": set(observed_recipes),
                    "observed_preferences": set(observed_preferences),
                    "observed_pairs": set(observed_pairs),
                    "preferences_by_recipe": {
                        key: set(value)
                        for key, value in preferences_by_recipe.items()
                    },
                    "axis_values_by_recipe": {
                        key: set(value)
                        for key, value in axis_values_by_recipe.items()
                    },
                    "last_frozen_event": last_frozen_event,
                    "baseline_context": dict(baseline_context),
                    "offline_training_pairs": set(offline_training_pairs),
                },
            )

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
        wall_s=resumed_wall_s + float(time.perf_counter() - start_time),
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


def _phase_retrain_latency(
    agent: AdaptiveAgent,
    memory_rows: Sequence[Mapping[str, Any]],
    initial_memory: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Attribute individual retrain fit times to the phase they occurred in.

    Cumulative wall-clock totals hide the quantity a person actually waits on,
    which is one retrain between two demonstrations. ``training_retrain_count``
    counts exactly the entries appended to ``retrain_fit_wall_times``, so the
    cumulative count at each event boundary slices that list by phase and the
    tail of the distribution survives aggregation.
    """
    fit_times = [float(value) for value in agent.retrain_fit_wall_times]
    by_phase: Dict[str, List[float]] = defaultdict(list)
    by_phase_index: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    previous = int(_numeric(initial_memory, "training_retrain_count"))
    for row in sorted(memory_rows, key=lambda item: _numeric(item, "event_index", -1.0)):
        current = int(_numeric(row, "training_retrain_count"))
        if current <= previous:
            previous = max(previous, current)
            continue
        window = fit_times[previous:current]
        previous = current
        if not window: continue
        phase_role = str(row.get("phase_role") or "unphased")
        phase_index = row.get("phase_index")
        phase_label = (
            f"phase_{int(phase_index):02d}"
            if isinstance(phase_index, int) else "phase_unknown"
        )
        by_phase[phase_role].extend(window)
        by_phase_index[phase_role][phase_label].extend(window)

    def stats(values: Sequence[float]) -> Dict[str, float]:
        return {
            "online_retrain_fit_count": float(len(values)),
            "online_mean_retrain_fit_wall_s": _mean(values),
            "online_p95_retrain_fit_wall_s": _p95(values),
            "online_max_retrain_fit_wall_s": float(max(values)) if values else 0.0,
        }

    return (
        {phase: stats(values) for phase, values in sorted(by_phase.items())},
        {
            phase: {label: stats(values) for label, values in sorted(by_label.items())}
            for phase, by_label in sorted(by_phase_index.items())
        },
    )


def _phase_training_costs(
    agent: AdaptiveAgent,
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
    latency_by_phase, latency_by_phase_index = _phase_retrain_latency(
        agent, memory_rows, initial_memory,
    )
    return {
        "upfront_training": offline,
        "per_phase_role": {
            phase: {**dict(values), **latency_by_phase.get(phase, {})}
            for phase, values in sorted(by_phase.items())
        },
        "per_phase_index": {
            phase: {
                phase_index: {
                    **dict(values),
                    **latency_by_phase_index.get(phase, {}).get(phase_index, {}),
                }
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
            "status": "not_run",
            "n_probes": 0,
        }

    def rate(field: str) -> Optional[float]:
        values = [row.get(field) for row in probes if isinstance(row.get(field), bool)]
        return _mean(1.0 if value else 0.0 for value in values) if values else None

    return {
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
        "status": "completed",
        "robot_turn_count": len(rows),
        "ece": float(ece),
        "top_1_brier": _mean(
            (float(row["final_confidence"]) - (1.0 if row["correct_top_1"] else 0.0)) ** 2
            for row in rows
        ),
        "bins": bin_rows,
    }


DECISION_REGIMES: Tuple[str, ...] = ("single_option_lookup", "branching", "unseen_state_fallback")


def _regime_action_count(row: Mapping[str, Any]) -> Any:
    """Prefer the model-agnostic count; fall back for runs recorded before it.

    ``observed_actions_at_state`` is computed from the replayed demonstrations
    and is therefore identical across arms. Earlier runs only carry
    ``exact_state_learned_action_count``, which for the MaxEnt arms is the same
    quantity -- the learned state-action entries at a state are exactly the
    actions demonstrated there -- so those runs can still be stratified.
    """
    count = row.get("observed_actions_at_state")
    return count if count is not None else row.get("exact_state_learned_action_count")


def _decision_regime(learned_action_count: Any) -> Optional[str]:
    """Classify one decision by how much choice the learner actually faced.

    ``exact_state_learned_action_count`` is the number of actions the model had
    observed at this exact state. One means the demonstrated continuation is
    unique and any predictor that memorised the state graph is correct, so the
    reward function cannot influence the outcome. Two or more is a genuine
    ranking problem. Zero means the state was never observed and the answer
    comes from generalisation rather than lookup.
    """
    if not isinstance(learned_action_count, (int, float)) or isinstance(learned_action_count, bool): return None
    count = int(learned_action_count)
    if count <= 0: return "unseen_state_fallback"
    return "single_option_lookup" if count == 1 else "branching"


def summarize_decision_regimes(turn_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Split accuracy by decision regime instead of averaging over them.

    An aggregate Top-1 over this stream is dominated by single-option lookups,
    which every arm answers almost perfectly, so it mostly measures whether the
    demonstrated state graph was memorised. Reporting the three regimes
    separately keeps the two that require generalisation visible, and makes the
    share of each regime -- itself a consequence of the memory policy -- an
    explicit part of the comparison rather than a hidden weighting.
    """
    buckets: Dict[str, List[Mapping[str, Any]]] = {regime: [] for regime in DECISION_REGIMES}
    unclassified = 0
    for row in turn_rows:
        regime = _decision_regime(_regime_action_count(row))
        if regime is None or not isinstance(row.get("correct_top_1"), bool):
            unclassified += 1
            continue
        buckets[regime].append(row)
    classified = sum(len(rows) for rows in buckets.values())
    if not classified:
        return {"status": "not_run", "n_classified": 0, "n_unclassified": unclassified, "by_regime": {}}
    by_regime: Dict[str, Any] = {}
    for regime in DECISION_REGIMES:
        rows = buckets[regime]
        by_regime[regime] = {
            "n": len(rows),
            "share": _safe_div(len(rows), classified),
            "top_1": _mean(1.0 if row["correct_top_1"] else 0.0 for row in rows) if rows else None,
            "top_k": (_mean(1.0 if row.get("correct_top_k") else 0.0 for row in rows) if rows else None),
            "mean_observed_action_count": (_mean(_regime_action_count(row) for row in rows) if rows else None),
        }
    return {
        "status": "completed",
        "n_classified": classified,
        # Turns with no exact-state count: predictors that never consult the
        # learned state graph, and turns where no prediction was produced.
        "n_unclassified": unclassified,
        "aggregate_top_1_over_classified": _mean(1.0 if row["correct_top_1"] else 0.0 for rows in buckets.values() for row in rows),
        "by_regime": by_regime,
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
        "n_axis_value_rows": len(axis_rows),
        "by_axis": {k: aggregate_episodes(v) for k, v in sorted(grouped_axis.items())},
        "by_axis_value": {k: aggregate_episodes(v) for k, v in sorted(grouped_value.items())},
        "by_axis_transfer_cell": {k: aggregate_episodes(v) for k, v in sorted(grouped_cell.items())},
    }

def summarize_active_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "status": "not_run",
            "n_audits": 0,
            "n_available": 0,
            "n_contract_violations": None,
            "contract_violation_rate": None,
            "max_redundancy_l1": None,
            "mean_redundancy_l1": None,
            "max_deployed_path_dependence_l1": None,
            "mean_deployed_path_dependence_l1": None,
            "mean_active_variants": None,
            "mean_pruned_variants": None,
            "contract_violation_checkpoints": [],
        }
    available = [row for row in rows if row.get("audit_available")]
    violations = [
        row for row in available
        if row.get("primary_active_only_contract_passed") is False
    ]
    with_pruned = [row for row in available if row.get("pruned_available")]
    return {
        "status": "completed" if available else "unavailable",
        "n_audits": len(rows),
        "n_available": len(available),
        "n_audits_with_pruned_memory": len(with_pruned),
        "n_contract_violations": len(violations),
        "contract_violation_rate": _safe_div(len(violations), len(available)),
        "max_redundancy_l1": max(
            _finite(row.get("redundancy_max_l1") for row in with_pruned), default=0.0,
        ),
        "mean_redundancy_l1": _mean(row.get("redundancy_mean_l1") for row in with_pruned),
        "max_deployed_path_dependence_l1": max(
            _finite(row.get("deployed_path_dependence_max_l1") for row in available), default=0.0,
        ),
        "mean_deployed_path_dependence_l1": _mean(
            row.get("deployed_path_dependence_mean_l1") for row in available
        ),
        "mean_active_variants": _mean(row.get("active_variants") for row in rows),
        "mean_pruned_variants": _mean(row.get("pruned_variants") for row in rows),
        "contract_violation_checkpoints": [
            {"event_index": row.get("event_index"), "audit_checkpoint": row.get("audit_checkpoint")}
            for row in violations
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
    """Stratify assist performance by the target variant's pre-episode state.

    Keyed off what memory actually held when the episode began, not off a
    planned probe tag: the schedule emits no reentry probes, so the earlier
    tag-gated version reported ``n_probes = 0`` in every run for every agent.
    A pruned-target episode is the selective-forgetting recovery case; an
    active-target episode is the retention control.
    """
    assist = [row for row in rows if row.get("mode") == "assist"]
    pruned_target = [
        row for row in assist
        if row.get("reentry_probe_target_state_before") == "pruned_exact_variant"
    ]
    active_target = [
        row for row in assist
        if row.get("reentry_probe_target_state_before") == "active_exact_variant"
    ]
    known_recipe_new_variant = [
        row for row in assist
        if row.get("reentry_probe_target_state_before") == "known_recipe_no_exact_variant"
    ]
    confirmed = [row for row in assist if row.get("actual_reentry_from_pruned")]
    return {
        "target_pruned_before_episode": aggregate_episodes(pruned_target),
        "target_active_before_episode_control": aggregate_episodes(active_target),
        "target_known_recipe_new_variant": aggregate_episodes(known_recipe_new_variant),
        "confirmed_reentry_from_pruned": aggregate_episodes(confirmed),
        "n_assist_episodes": len(assist),
        "n_target_pruned_before_episode": len(pruned_target),
        "n_target_active_before_episode_control": len(active_target),
        "n_target_known_recipe_new_variant": len(known_recipe_new_variant),
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
    phase_training = _phase_training_costs(stream.agent, stream.memory_rows, stream.initial_memory)
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
            "decision_regimes": summarize_decision_regimes(stream.turn_rows),
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


class _EventProgressWriter:
    """Write one atomic checkpoint per completed event, plus a partial summary.

    Held as an object rather than a closure so a per-baseline cell and a
    whole-plan run can share it: the row offsets and the partial roll-up are
    keyed by baseline either way, so a directory holding one arm and a
    directory holding all of them are read back the same way.
    """

    def __init__(
        self,
        plan: Plan,
        out_dir: Path,
        progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.plan = plan
        self.out_dir = out_dir
        self.progress_callback = progress_callback
        self.offsets: Dict[str, Dict[str, int]] = {}
        self.partial_baselines: Dict[str, Dict[str, Any]] = {}
        self.phase_ids = tuple(dict.fromkeys(
            str(event.tags.get("phase_id", "unknown"))
            for event in plan.events
        ))
        self.phase_numbers = {
            phase_id: index + 1
            for index, phase_id in enumerate(self.phase_ids)
        }
        self.stages = tuple(dict.fromkeys(
            str(event.tags.get("stage", "unknown"))
            for event in plan.events
        ))
        self.stage_numbers = {
            stage: index + 1 for index, stage in enumerate(self.stages)
        }

    def __call__(self, stream: "RunState", event_index: int) -> None:
        plan = self.plan
        out_dir = self.out_dir
        event = plan.events[event_index]
        phase_id = str(event.tags.get("phase_id", "unknown"))
        stage = str(event.tags.get("stage", "unknown"))
        row_groups = {
            "episodes": stream.episode_rows,
            "turns": stream.turn_rows,
            "frozen_probes": stream.frozen_rows,
            "diagnostics": (
                stream.memory_rows
                + stream.active_audit_rows
                + stream.oracle_pruning_rows
            ),
        }
        offsets = self.offsets.setdefault(
            stream.baseline,
            {name: 0 for name in row_groups},
        )
        new_rows = {
            name: rows[offsets[name]:]
            for name, rows in row_groups.items()
        }
        checkpoint_path = (
            out_dir / "partial" / "checkpoints" / stream.baseline
            / f"event_{event_index:06d}.json"
        )
        payload = {
            "state": "event_complete",
            "scenario": plan.scenario,
            "seed": int(plan.seed),
            "baseline": stream.baseline,
            "event_index": int(event_index),
            "events_completed": len(stream.episode_rows),
            "events_total": len(plan.events),
            "phase_id": phase_id,
            "phase_number": self.phase_numbers[phase_id],
            "phase_count": len(self.phase_ids),
            "stage": stage,
            "stage_number": self.stage_numbers[stage],
            "stage_count": len(self.stages),
            "phase_complete": _is_phase_boundary(plan, event_index),
            "pair": event.pair.label,
            "mode": event.mode,
            "elapsed_s": float(stream.wall_s),
            "created_at_utc": _utc_now(),
            "row_counts": {
                name: len(rows) for name, rows in row_groups.items()
            },
            "tables": new_rows,
        }
        _write_json(checkpoint_path, payload)
        for name, rows in row_groups.items():
            offsets[name] = len(rows)

        self.partial_baselines[stream.baseline] = {
            "events_completed": len(stream.episode_rows),
            "events_total": len(plan.events),
            "latest_event_index": int(event_index),
            "elapsed_s": float(stream.wall_s),
            "episodes": aggregate_episodes(stream.episode_rows),
            "assist": aggregate_episodes([
                row for row in stream.episode_rows
                if row.get("mode") == "assist"
            ]),
            "row_counts": payload["row_counts"],
        }
        relative_checkpoint = str(checkpoint_path.relative_to(out_dir))
        progress = {
            key: value for key, value in payload.items()
            if key != "tables"
        }
        progress["checkpoint"] = relative_checkpoint
        _write_json(out_dir / "partial" / "summary.json", {
            "state": "running",
            "scenario": plan.scenario,
            "seed": int(plan.seed),
            "updated_at_utc": progress["created_at_utc"],
            "latest": progress,
            "per_baseline": self.partial_baselines,
            "analysis_note": (
                "Each checkpoint is atomic and contains only rows added by its "
                "completed event. Concatenate checkpoints by baseline and event_index."
            ),
        })
        if self.progress_callback is not None:
            self.progress_callback(progress)


def _write_plan(plan: Plan, out_dir: Path) -> None:
    """Record the schedule a cell was run against, next to its results."""
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


def baseline_run_order(config: EvalSettings) -> Tuple[str, ...]:
    """Order the arms so the shared route is produced before it is consumed.

    Under shared routing every other arm replays ``full``'s realized
    observe/assist schedule, so ``full`` has to go first whatever order the
    caller asked for. The memory oracle goes last: it is a reference curve read
    against the arms above it, not a system competing with them.
    """
    baselines = [name for name in config.baselines if name != MEMORY_ORACLE]
    if config.shared_routing and "full" not in baselines:
        raise ValueError(
            "shared_routing=True requires the deployable 'full' system "
            "to be included in EvalSettings.baselines"
        )
    if config.shared_routing:
        baselines = ["full", *[name for name in baselines if name != "full"]]
    order = list(dict.fromkeys(str(name) for name in baselines))
    if config.include_oracle:
        order.append(MEMORY_ORACLE)
    return tuple(order)


def produces_shared_route(baseline: str, config: EvalSettings) -> bool:
    """Whether this arm is the one that defines the schedule others replay."""
    return str(baseline) == "full" and bool(config.shared_routing)


def realized_route(stream: "RunState") -> Tuple[str, ...]:
    """The execution mode actually taken at each event of a finished stream."""
    return tuple(str(row.get("mode")) for row in stream.episode_rows)


def execute_baseline_stream(
    baseline: str,
    plan: Plan,
    config: EvalSettings,
    *,
    route: Optional[Sequence[str]] = None,
    event_progress: Optional[Callable[["RunState", int], None]] = None,
    resume_dir: Optional[Path] = None,
) -> Tuple["RunState", Dict[str, Any]]:
    """Run one arm over one plan and summarize it.

    ``route`` is ``full``'s realized execution schedule. It is None for the arm
    that produces it and whenever routing is not shared; passing it in is what
    lets an arm be run on its own, long after ``full`` finished, and still be
    scored on the identical sequence of observations and assists.
    """
    clear_caches()
    if produces_shared_route(baseline, config):
        stream = run_stream(
            baseline,
            plan,
            config,
            mode_schedule_policy="full_realized_canonical",
            event_progress=event_progress,
            resume_dir=resume_dir,
        )
    else:
        stream = run_stream(
            baseline,
            plan,
            config,
            execution_mode_schedule=(
                tuple(str(mode) for mode in route) if route is not None else None
            ),
            mode_schedule_policy=(
                "matched_full_realized_execution_schedule_oracle"
                if baseline == MEMORY_ORACLE
                else "matched_full_realized_execution_schedule"
                if route is not None
                else "baseline_local"
            ),
            event_progress=event_progress,
            resume_dir=resume_dir,
        )
    baseline_summary = summarize_stream(stream)
    if baseline == MEMORY_ORACLE:
        baseline_summary["oracle"] = {
            "presentation": CLAIRVOYANT_REFERENCE_TAG,
            "retention": "future_filtered_after_first_exposure",
            "decay": "binary_keep_until_final_occurrence",
            "selection": {
                "full_for_unseen_pair": int(sum(
                    row.get("oracle_selection") == "full_for_unseen_pair"
                    for row in stream.oracle_pruning_rows
                )),
                "future_filtered": int(sum(
                    row.get("oracle_selection") == "future_filtered"
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
    return stream, baseline_summary


def merge_cell(
    run_dir: Path,
    scenario: str,
    seed: int,
    baselines: Sequence[str],
) -> Dict[str, Any]:
    """Stitch one scenario/seed's per-arm folders into a cross-arm view.

    The arms are run and stored separately, but several claims are only
    defined across them -- the oracle gaps, and every paired full-minus-
    baseline delta. Rather than keep a second copy of the data, this reads
    each arm's own summary back and writes only what the comparison adds.
    """
    per_baseline: Dict[str, Any] = {}
    wall_s = 0.0
    table_rows: Dict[str, int] = defaultdict(int)
    for baseline in baselines:
        summary_path = (
            _baseline_cell_dir(run_dir, baseline, scenario, int(seed)) / "summary.json"
        )
        if not summary_path.is_file():
            continue
        cell = _load_json(summary_path)
        for name, value in dict(cell.get("per_baseline", {})).items():
            per_baseline[str(name)] = value
        wall_s += float(cell.get("wall_s", 0.0))
        for name, count in dict(cell.get("table_rows", {})).items():
            table_rows[str(name)] += int(count)
    oracle_comparisons = build_oracle_rows(per_baseline)
    out_dir = (
        run_dir / "aggregate" / "cells"
        / _safe_component(str(scenario), "scenario") / f"{int(seed):010d}"
    )
    _write_jsonl_gz(out_dir / "tables" / "oracle_gaps.jsonl.gz", oracle_comparisons)
    table_rows["oracle_gaps"] = len(oracle_comparisons)
    summary = {
        "state": "complete",
        "scenario": str(scenario),
        "seed": int(seed),
        "baselines": [str(name) for name in per_baseline],
        "per_baseline": per_baseline,
        "oracle_summary": summarize_oracle(oracle_comparisons),
        "table_rows": dict(table_rows),
        "cell_paths": {
            str(baseline): str(
                _baseline_cell_dir(run_dir, baseline, scenario, int(seed))
                .relative_to(run_dir)
            )
            for baseline in per_baseline
        },
        "wall_s": float(wall_s),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


BASELINES_DIRNAME = "baselines"
SHARED_DIRNAME = "shared"


def _baseline_cell_dir(
    run_dir: Path, baseline: str, scenario: str, seed: int,
) -> Path:
    """Where one (arm, scenario, seed) cell keeps everything it produced."""
    return (
        run_dir / BASELINES_DIRNAME / _safe_component(str(baseline), "baseline")
        / "scenarios" / _safe_component(str(scenario), "scenario")
        / "seeds" / f"{int(seed):010d}"
    )


def _baseline_dir(run_dir: Path, baseline: str) -> Path:
    return run_dir / BASELINES_DIRNAME / _safe_component(str(baseline), "baseline")


def _route_path(run_dir: Path, scenario: str, seed: int) -> Path:
    """Where ``full``'s realized schedule is published for the other arms.

    It lives outside every arm's directory on purpose: an arm's folder can be
    deleted and re-run without destroying the route its scoring depends on.
    """
    return (
        run_dir / SHARED_DIRNAME / "routing"
        / _safe_component(str(scenario), "scenario") / f"{int(seed):010d}.json"
    )


def save_route(
    run_dir: Path, scenario: str, seed: int, route: Sequence[str],
) -> None:
    _write_json(_route_path(run_dir, scenario, seed), {
        "scenario": str(scenario),
        "seed": int(seed),
        "source": "full",
        "policy": "full_realized_canonical",
        "event_count": len(tuple(route)),
        "modes": [str(mode) for mode in route],
    })


def load_route(
    run_dir: Path, scenario: str, seed: int,
) -> Optional[Tuple[str, ...]]:
    path = _route_path(run_dir, scenario, seed)
    if not path.is_file():
        return None
    try:
        payload = _load_json(path)
    except (OSError, ValueError):
        return None
    modes = payload.get("modes")
    if not isinstance(modes, list):
        return None
    return tuple(str(mode) for mode in modes)


def run_baseline_cell(
    baseline: str,
    plan: Plan,
    config: EvalSettings,
    out_dir: Path,
    *,
    route: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run exactly one arm over one plan and write its own result folder.

    This is the unit the suite schedules. Everything a cell needs is either in
    its arguments or in the shared route, so an arm whose implementation
    changed can be deleted and re-run on its own without disturbing the arms
    that did not change.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    _write_plan(plan, out_dir)
    stream, baseline_summary = execute_baseline_stream(
        baseline,
        plan,
        config,
        route=route,
        event_progress=_EventProgressWriter(plan, out_dir, progress_callback),
        resume_dir=out_dir if config.event_resume else None,
    )
    diagnostic_rows = (
        stream.memory_rows
        + stream.active_audit_rows
        + stream.oracle_pruning_rows
    )
    axis_rows = transfer_rows(stream.episode_rows)
    tables = out_dir / "tables"
    _write_jsonl_gz(tables / "episodes.jsonl.gz", stream.episode_rows)
    _write_jsonl_gz(tables / "turns.jsonl.gz", stream.turn_rows)
    _write_jsonl_gz(tables / "frozen_probes.jsonl.gz", stream.frozen_rows)
    _write_jsonl_gz(tables / "diagnostics.jsonl.gz", diagnostic_rows)
    _write_jsonl_gz(tables / "axis_transfer.jsonl.gz", axis_rows)

    realized = (
        realized_route(stream) if produces_shared_route(baseline, config) else None
    )
    summary = {
        "state": "complete",
        "scenario": plan.scenario,
        "seed": int(plan.seed),
        "baseline": str(baseline),
        "plan": "plan.json",
        "mode_schedule": {
            "policy": (
                "full_realized_shared_across_all_methods"
                if config.shared_routing else "local_routing"
            ),
            "produced_shared_route": realized is not None,
            "replayed_shared_route": route is not None,
            "canonical_full_mode_counts": (
                dict(Counter(realized)) if realized is not None
                else dict(Counter(route)) if route is not None
                else None
            ),
        },
        "table_rows": {
            "episodes": len(stream.episode_rows),
            "turns": len(stream.turn_rows),
            "frozen_probes": len(stream.frozen_rows),
            "diagnostics": len(diagnostic_rows),
            "axis_transfer": len(axis_rows),
        },
        # Keyed by arm even though there is only one, so a single-arm folder
        # and a merged cell are read back with the same code.
        "per_baseline": {str(baseline): baseline_summary},
        "wall_s": float(time.perf_counter() - start_time),
    }
    if realized is not None:
        summary["realized_route"] = list(realized)
    _write_json(out_dir / "summary.json", summary)
    partial_summary_path = out_dir / "partial" / "summary.json"
    if partial_summary_path.is_file():
        partial_summary = _load_json(partial_summary_path)
        partial_summary.update({
            "state": "complete",
            "completed_at_utc": _utc_now(),
            "final_summary": "../summary.json",
        })
        _write_json(partial_summary_path, partial_summary)
    return summary


def run_plan(
    plan: Plan,
    config: EvalSettings,
    out_dir: Path,
    *,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    _write_plan(plan, out_dir)
    per_baseline: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_frozen_rows: List[Dict[str, Any]] = []
    all_diagnostic_rows: List[Dict[str, Any]] = []
    all_turn_rows: List[Dict[str, Any]] = []
    persist_event_progress = _EventProgressWriter(plan, out_dir, progress_callback)

    canonical_full_modes: Optional[Tuple[str, ...]] = None
    for baseline in baseline_run_order(config):
        stream, baseline_summary = execute_baseline_stream(
            baseline,
            plan,
            config,
            route=canonical_full_modes,
            event_progress=persist_event_progress,
            resume_dir=out_dir if config.event_resume else None,
        )
        if produces_shared_route(baseline, config):
            canonical_full_modes = realized_route(stream)
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
        "state": "complete",
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
    partial_summary_path = out_dir / "partial" / "summary.json"
    if partial_summary_path.is_file():
        partial_summary = _load_json(partial_summary_path)
        partial_summary.update({
            "state": "complete",
            "completed_at_utc": _utc_now(),
            "final_summary": "../summary.json",
        })
        _write_json(partial_summary_path, partial_summary)
    return summary


def _mp_context() -> mp.context.BaseContext:
    return mp.get_context("spawn")


def _worker_count(
    config: EvalSettings,
    pending_jobs: Optional[int] = None,
    baseline: Optional[str] = None,
) -> int:
    # One worker owns the quantized GPU model; cell-level GPU replication can
    # otherwise exhaust memory before evaluation begins. Arms run one at a
    # time, so only the GPU arm itself is held to a single process -- the
    # symbolic arms in the same run keep the full cohort.
    if (LLM_BASELINE in config.baselines) if baseline is None else (baseline == LLM_BASELINE):
        return 1
    # Admit only complete scenario cohorts: every seed for an admitted scenario
    # starts together, and additional scenarios wait when the process cap would
    # admit only part of their seed cohort.
    total_jobs = max(1, len(config.scenarios) * len(config.seeds))
    cap = max(1, min(
        DEFAULT_EVALUATION_WORKER_CAP,
        int(config.workers or DEFAULT_EVALUATION_WORKER_CAP),
    ))
    seeds_per_scenario = max(1, len(config.seeds))
    concurrent_scenarios = max(
        1, min(len(config.scenarios), cap // seeds_per_scenario),
    )
    workers = min(cap, concurrent_scenarios * seeds_per_scenario, total_jobs)
    if pending_jobs is not None:
        workers = min(workers, max(1, int(pending_jobs)))
    return max(1, workers)


def _scenario_job_batches(
    jobs: Sequence[Tuple[str, int]],
    config: EvalSettings,
) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """Group jobs so a process wave never admits a partial scenario cohort."""
    grouped: Dict[str, List[Tuple[str, int]]] = {}
    for scenario, seed in jobs:
        grouped.setdefault(str(scenario), []).append((str(scenario), int(seed)))
    seeds_per_scenario = max(1, len(config.seeds))
    cap = max(1, min(
        DEFAULT_EVALUATION_WORKER_CAP,
        int(config.workers or DEFAULT_EVALUATION_WORKER_CAP),
    ))
    scenarios_per_batch = max(1, cap // seeds_per_scenario)
    scenarios = tuple(grouped)
    return tuple(
        tuple(
            job
            for scenario in scenarios[index:index + scenarios_per_batch]
            for job in grouped[scenario]
        )
        for index in range(0, len(scenarios), scenarios_per_batch)
    )


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


def _job_eta(suite_start: float, completed: int, expected: int) -> str:
    """Remaining-time estimate over all (scenario, seed) jobs."""
    elapsed = time.perf_counter() - suite_start
    if completed <= 0:
        return f"elapsed={_format_seconds(elapsed)} eta=unknown"
    remaining = max(0.0, elapsed * (expected - completed) / completed)
    return (
        f"elapsed={_format_seconds(elapsed)} "
        f"done={completed}/{expected} eta={_format_seconds(remaining)}"
    )


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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_resumable_run(config: EvalSettings) -> Optional[str]:
    """Name the newest incomplete run this exact config can continue.

    Without this, per-event resume is unreachable: every invocation mints a new
    timestamped directory, so the state a crashed run left behind is never
    looked at. Matching on the config hash and experiment label is the same
    check ``_create_or_resume_run`` applies to an explicit ``--resume``, so a
    directory found here is one that resume would already have accepted.
    """
    runs = Path(config.output).expanduser().resolve() / RUNS_DIRNAME
    if not runs.is_dir():
        return None
    digest = _config_hash(config)
    candidates: List[Tuple[float, str]] = []
    for run_dir in runs.iterdir():
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = _load_json(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.get("config_hash") != digest:
            continue
        if manifest.get("experiment") != config.experiment:
            continue
        status = run_dir / "status.json"
        try:
            state = _load_json(status).get("state") if status.is_file() else None
        except (OSError, ValueError):
            state = None
        if state == "complete":
            continue
        # Any per-event state to continue from? A run that never reached one
        # offers nothing a fresh directory does not.
        if not any(run_dir.glob("baselines/*/scenarios/*/seeds/*/partial/resume/*.pickle")):
            continue
        candidates.append((manifest_path.stat().st_mtime, run_dir.name))
    if not candidates:
        return None
    return max(candidates)[1]


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
            "completed_jobs": 0,
            "expected_jobs": _expected_cell_count(config),
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
        # Stated once per run rather than per turn row. Asserted by the test
        # suite (see tests/test_agent.py), not by repetition in the data.
        "action_mask_contract": dict(ACTION_MASK_CONTRACT),
        "code": _git_provenance(),
        "runtime": _runtime_provenance(),
        "baseline_order": list(baseline_run_order(config)),
        "expected_jobs": [
            {"scenario": scenario, "seed": int(seed)}
            for scenario in config.scenarios for seed in config.seeds
        ],
        "expected_cells": [
            {"baseline": baseline, "scenario": scenario, "seed": int(seed)}
            for baseline in baseline_run_order(config)
            for scenario in config.scenarios for seed in config.seeds
        ],
        "artifacts": {
            "cell_summary": (
                "baselines/<baseline>/scenarios/<scenario>/seeds/<seed>/summary.json"
            ),
            "cell_tables": (
                "baselines/<baseline>/scenarios/<scenario>/seeds/<seed>/tables/*.jsonl.gz"
            ),
            "partial_progress": (
                "baselines/<baseline>/scenarios/<scenario>/seeds/<seed>/partial/summary.json"
            ),
            "event_checkpoints": (
                "baselines/<baseline>/scenarios/<scenario>/seeds/<seed>/partial/"
                "checkpoints/<baseline>/event_<index>.json"
            ),
            "baseline_status": "baselines/<baseline>/status.json",
            "shared_route": "shared/routing/<scenario>/<seed>.json",
            "merged_cell": "aggregate/cells/<scenario>/<seed>/summary.json",
            "aggregate": "aggregate/",
        },
    })
    _write_json(run_dir / "status.json", {
        "state": "running", "started_at_utc": created_at,
        "completed_jobs": 0, "expected_jobs": _expected_cell_count(config),
    })
    return root, run_dir, digest


def _expected_cell_count(config: EvalSettings) -> int:
    return (
        len(baseline_run_order(config))
        * len(config.scenarios)
        * len(config.seeds)
    )


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


def _completed_cell_result(
    run_dir: Path, baseline: str, scenario: str, seed: int,
) -> Optional[Dict[str, Any]]:
    """Reuse one finished arm/scenario/seed cell instead of re-running it.

    This is what makes an arm independently re-runnable: delete one arm's
    folder and every other cell still reports complete, so a resumed run
    re-runs only the arm that changed.
    """
    cell_dir = _baseline_cell_dir(run_dir, baseline, scenario, seed)
    status_path = cell_dir / "status.json"
    summary_path = cell_dir / "summary.json"
    if not status_path.is_file() or not summary_path.is_file():
        return None
    status = _load_json(status_path)
    if status.get("state") != "complete":
        return None
    summary = _load_json(summary_path)
    # The published route can be reconstructed from the arm that produced it,
    # so losing the shared directory does not invalidate a finished full cell.
    route = summary.get("realized_route")
    if isinstance(route, list) and route:
        if load_route(run_dir, scenario, seed) is None:
            save_route(run_dir, scenario, seed, [str(mode) for mode in route])
    return {
        "baseline": str(baseline),
        "scenario": scenario,
        "seed": int(seed),
        "key": f"{baseline}/{scenario}/{int(seed):010d}",
        "summary": summary,
        "summary_path": str(summary_path.relative_to(run_dir)),
        "wall_s": float(status.get("wall_s", 0.0)),
    }


def _clear_stale_checkpoints(out_dir: Path) -> int:
    """Drop checkpoints from an earlier attempt at this seed.

    A seed that did not finish is re-run from its first event, because resume
    is seed-level. Its old checkpoints are not overwritten past the point the
    new attempt reaches, so a directory can end up holding a low range from one
    attempt and a high range from another. Concatenating those by event index --
    which is what the partial summary tells readers to do -- would splice two
    different runs of the same seed into one series.
    """
    checkpoint_root = out_dir / "partial" / "checkpoints"
    if not checkpoint_root.is_dir():
        return 0
    removed = 0
    for path in checkpoint_root.glob("*/event_*.json"):
        path.unlink()
        removed += 1
    return removed


def _trim_checkpoints_after(
    out_dir: Path, baseline: str, event_index: int,
) -> int:
    """Delete one baseline's checkpoints past a resume point."""
    directory = out_dir / "partial" / "checkpoints" / baseline
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("event_*.json"):
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if index > int(event_index):
            path.unlink()
            removed += 1
    return removed


def _open_worker_fault_log(out_dir: Path) -> Optional[Any]:
    """Point faulthandler at a file so a native crash leaves a stack behind.

    A worker that dies on a fatal signal prints its traceback to stderr, which
    for a pooled worker is the launching terminal and is kept nowhere. The
    parent then reports only ``BrokenProcessPool``, which says the process died
    and nothing about where. On this GPU stack that has happened twice, hours
    into a run, so the evidence has to survive in the run directory.

    The handle is deliberately never closed: it has to stay valid through a
    signal that terminates the process.
    """
    try:
        handle = open(out_dir / "worker_fault.log", "a", buffering=1)
        faulthandler.enable(file=handle, all_threads=True)
        return handle
    except (OSError, RuntimeError, ValueError):
        # Diagnostics must never be the reason a run cannot start.
        return None


def _run_baseline_cell_job(
    baseline: str, scenario: str, seed: int, config: EvalSettings, run_dir: str,
) -> Dict[str, Any]:
    """Process entry point for one arm/scenario/seed cell."""
    _pin_to_performance_cores()
    _apply_native_thread_limit(_native_thread_count(config))
    run_path = Path(run_dir)
    out_dir = _baseline_cell_dir(run_path, baseline, scenario, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    started_at = _utc_now()
    latest_progress: Dict[str, Any] = {}
    fault_log = _open_worker_fault_log(out_dir)

    def report_progress(progress: Mapping[str, Any]) -> None:
        latest_progress.clear()
        latest_progress.update(dict(progress))
        _write_json(out_dir / "status.json", {
            "state": "running",
            "baseline": str(baseline),
            "started_at_utc": started_at,
            "updated_at_utc": _utc_now(),
            "progress": latest_progress,
        })
        if (
            progress.get("baseline") == "in_context_llm"
            or bool(progress.get("phase_complete"))
        ):
            print(
                "[evaluation] progress "
                f"baseline={progress.get('baseline')} "
                f"scenario={scenario} seed={int(seed)} "
                f"event={progress.get('events_completed')}/{progress.get('events_total')} "
                f"stage={progress.get('stage_number')}/{progress.get('stage_count')} "
                f"elapsed={_format_seconds(progress.get('elapsed_s'))} "
                f"saved={progress.get('checkpoint')}",
                flush=True,
            )

    if not config.event_resume:
        # With event resume the low range belongs to the same logical run and
        # is not stale; run_stream trims only what sits past its resume point.
        _clear_stale_checkpoints(out_dir)
    _write_json(out_dir / "status.json", {
        "state": "running",
        "baseline": str(baseline),
        "started_at_utc": started_at,
    })
    try:
        plan = build_plan(scenario, config, int(seed))
        route: Optional[Tuple[str, ...]] = None
        if config.shared_routing and not produces_shared_route(baseline, config):
            route = load_route(run_path, scenario, int(seed))
            if route is None:
                raise FileNotFoundError(
                    f"{baseline} replays full's realized schedule, but no route "
                    f"is published for scenario={scenario} seed={int(seed)}. "
                    "Run the 'full' arm for this cell first."
                )
            if len(route) != len(plan.events):
                raise ValueError(
                    f"published route for scenario={scenario} seed={int(seed)} "
                    f"has {len(route)} events but the plan has {len(plan.events)}; "
                    "the schedule changed since full was run, so every arm must "
                    "be re-run"
                )
        summary = run_baseline_cell(
            baseline,
            plan,
            config,
            out_dir,
            route=route,
            progress_callback=report_progress,
        )
        realized = summary.get("realized_route")
        if realized:
            # Published before this cell is marked complete: a resumed run
            # must never see a finished full cell with no route beside it.
            save_route(run_path, scenario, int(seed), [str(m) for m in realized])
    except BaseException as error:
        _write_json(out_dir / "status.json", {
            "state": "failed", "baseline": str(baseline),
            "started_at_utc": started_at, "failed_at_utc": _utc_now(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "wall_s": float(time.perf_counter() - start_time),
            "progress": latest_progress or None,
        })
        raise
    wall_s = float(time.perf_counter() - start_time)
    _write_json(out_dir / "status.json", {
        "state": "complete", "baseline": str(baseline),
        "started_at_utc": started_at, "completed_at_utc": _utc_now(),
        "wall_s": wall_s,
        "progress": latest_progress or None,
    })
    return {
        "baseline": str(baseline),
        "scenario": scenario,
        "seed": int(seed),
        "key": f"{baseline}/{scenario}/{int(seed):010d}",
        "summary": summary,
        "summary_path": str((out_dir / "summary.json").relative_to(run_path)),
        "wall_s": wall_s,
    }


def _pending_cell_results(
    baseline: str,
    cells: Sequence[Tuple[str, int]],
    config: EvalSettings,
    run_dir: Path,
    workers: int,
) -> Iterable[Mapping[str, Any]]:
    """Run one arm's outstanding cells in process-cap-bounded cohorts.

    Only the scenario/seed grid is parallel here. Arms are sequential by
    construction: the suite finishes one before starting the next, which is
    what keeps the shared route available and lets a single arm be re-run.
    """
    if baseline == LLM_BASELINE:
        yield from _pending_llm_cell_results(baseline, cells, config, run_dir)
        return
    if workers <= 1 or len(cells) <= 1:
        for scenario, seed in cells:
            yield _run_baseline_cell_job(
                str(baseline), str(scenario), int(seed), config, str(run_dir),
            )
        return
    for batch in _scenario_job_batches(cells, config):
        with ProcessPoolExecutor(
            max_workers=min(workers, len(batch)), mp_context=_mp_context(),
        ) as executor:
            futures = [
                executor.submit(
                    _run_baseline_cell_job,
                    str(baseline), str(scenario), int(seed), config, str(run_dir),
                )
                for scenario, seed in batch
            ]
            for future in as_completed(futures):
                yield future.result()


def _pending_llm_cell_results(
    baseline: str,
    cells: Sequence[Tuple[str, int]],
    config: EvalSettings,
    run_dir: Path,
) -> Iterable[Mapping[str, Any]]:
    """One fresh CUDA process per cell for the GPU arm.

    Native failures are not retried: repeating a long cell on an unqualified
    runtime hides instability and wastes compute without improving
    reproducibility.
    """
    for scenario, seed in cells:
        try:
            with ProcessPoolExecutor(
                max_workers=1, mp_context=_mp_context(),
            ) as executor:
                future = executor.submit(
                    _run_baseline_cell_job,
                    str(baseline), str(scenario), int(seed), config, str(run_dir),
                )
                result = dict(future.result())
        except BrokenProcessPool as exc:
            # BrokenProcessPool says only that the child died. What it died in
            # is in the worker's own fault log, if it took a signal
            # faulthandler could catch; name the file either way so the first
            # thing anyone reads is the one that has the answer.
            fault_log = (
                _baseline_cell_dir(Path(run_dir), baseline, scenario, int(seed))
                / "worker_fault.log"
            )
            detail = (
                f"see {fault_log} for the native stack"
                if fault_log.is_file() and fault_log.stat().st_size
                else f"no stack was written to {fault_log}, so the worker "
                "was killed by an uncatchable signal (SIGKILL) rather than "
                "faulting"
            )
            raise RuntimeError(
                "LLM GPU worker terminated natively for "
                f"baseline={baseline}, scenario={scenario}, seed={int(seed)}; "
                "automatic retry is disabled because this runtime is not "
                f"reproducible. {detail}"
            ) from exc
        result.update({
            "native_worker_attempts": 1,
            "native_worker_failed_attempts": 0,
            "native_worker_failed_wall_s": 0.0,
        })
        yield result


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
        "bootstrap_samples": int(n_samples),
        "by_scenario": results,
    }


def _summarize_sensitivity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scenario")), str(row.get("sensitivity_setting")))].append(row)
    return {
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
    run_order = baseline_run_order(config)
    expected_jobs = len(run_order) * len(config.scenarios) * len(config.seeds)
    suite: Dict[str, Any] = {
        "state": "running",
        "run": run_dir.name,
        "experiment": config.experiment,
        "config_hash": config_hash,
        "execution": {
            "parallelism": "processes",
            "unit": "one (baseline, scenario, seed) cell",
            "baseline_execution": "sequential; one arm completes the whole scenario/seed grid before the next starts",
            "baseline_order": list(run_order),
            "seed_parallelism": "within one arm, complete seed cohorts are scheduled for as many scenarios as fit within the 20-process cap",
            "workers": workers,
            "threads": native_threads,
            "estimated_max_native_threads": workers * native_threads,
            "thread_control": thread_control,
            "clairvoyant_oracle_included": bool(config.include_oracle),
            "llm_runtime": llm_runtime,
        },
        "cells": {},
    }
    cell_summaries: Dict[str, Mapping[str, Any]] = {}
    suite_start = time.perf_counter()
    completed_jobs = 0
    aggregate_dir = run_dir / "aggregate"

    def persist() -> None:
        suite["completed_jobs"] = completed_jobs
        suite["expected_jobs"] = expected_jobs
        _write_json(aggregate_dir / "suite_summary.json", suite)

    def record(result: Mapping[str, Any]) -> None:
        nonlocal completed_jobs
        key = str(result["key"])
        if key in cell_summaries:
            return
        cell_summaries[key] = result["summary"]
        suite["cells"][key] = {
            "baseline": str(result["baseline"]),
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
        grid = [
            (str(scenario), int(seed))
            for scenario in config.scenarios for seed in config.seeds
        ]
        job_wall: Dict[str, List[float]] = defaultdict(list)
        baseline_wall: Dict[str, float] = {}
        if config.show_eta:
            print(
                f"[evaluation] start arms={len(run_order)} "
                f"x cells={len(grid)} "
                f"(scenarios={len(config.scenarios)} x seeds={len(config.seeds)}) "
                f"order={', '.join(run_order)}",
                flush=True,
            )
        # One arm at a time, each over the whole scenario/seed grid. The order
        # matters: 'full' publishes the route the other arms replay.
        for baseline in run_order:
            baseline_start = time.perf_counter()
            pending: List[Tuple[str, int]] = []
            for scenario, seed in grid:
                result = _completed_cell_result(run_dir, baseline, scenario, seed)
                if result is not None:
                    record(result)
                else:
                    pending.append((scenario, seed))
            workers = _worker_count(config, len(pending), baseline)
            suite["execution"]["workers"] = workers
            suite["execution"]["estimated_max_native_threads"] = (
                workers * native_threads
            )
            if config.show_eta:
                print(
                    f"[evaluation] arm {baseline} "
                    f"pending={len(pending)}/{len(grid)} workers={workers}",
                    flush=True,
                )
            for result in _pending_cell_results(
                baseline, pending, config, run_dir, workers,
            ):
                record(result)
                job_wall[str(result["scenario"])].append(float(result["wall_s"]))
                if config.show_eta:
                    print(
                        f"[evaluation] done baseline={result['baseline']} "
                        f"scenario={result['scenario']} "
                        f"seed={result['seed']} "
                        f"cell_wall={_format_seconds(result['wall_s'])} "
                        f"{_job_eta(suite_start, completed_jobs, expected_jobs)}",
                        flush=True,
                    )
            baseline_wall[baseline] = float(time.perf_counter() - baseline_start)
            _write_json(_baseline_dir(run_dir, baseline) / "status.json", {
                "state": "complete",
                "baseline": baseline,
                "completed_at_utc": _utc_now(),
                "cells": len(grid),
                "wall_s": baseline_wall[baseline],
            })
            persist()

        # Arms are stored apart; the cross-arm claims are assembled here.
        merged: Dict[str, Mapping[str, Any]] = {}
        for scenario, seed in grid:
            merged[f"{scenario}/{seed:010d}"] = merge_cell(
                run_dir, scenario, seed, run_order,
            )
        suite["cells_path"] = "aggregate/cells/<scenario>/<seed>/summary.json"
        # Scenarios no longer own a wall-clock phase, so report the cell time
        # each one consumed and the suite's actual span separately.
        suite["scenario_job_wall_s"] = {
            scenario: float(sum(values)) for scenario, values in job_wall.items()
        }
        suite["baseline_wall_s"] = dict(baseline_wall)
        suite["suite_wall_s"] = float(time.perf_counter() - suite_start)
        persist()

        paired = summarize_pairs(merged)
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
    default_seeds: Sequence[int] = PAPER_SEEDS,
) -> EvalSettings:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run", help="Explicit run directory name; required with --resume.")
    parser.add_argument("--resume", action="store_true", help="Resume completed seeds in an existing run.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in default_seeds))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--baselines", default=",".join(default_baselines))
    parser.add_argument("--offline-recipe-fraction", type=float, default=0.50)
    parser.add_argument("--offline-preference-fraction", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=EvalSettings.workers)
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
    # Derived, not restated: a CLI default that duplicates the dataclass default
    # is a second source of truth and silently overrides it (see the CLI-default test).
    parser.add_argument("--audit-period", type=int, default=EvalSettings.audit_period)
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
    parser.add_argument(
        "--llm-prefill-chunk-tokens",
        type=int,
        help=(
            "Prefill the prompt this many positions at a time. Raises the "
            "prompt budget on a display-sharing GPU by about half, costs "
            "prefill wall time, and perturbs candidate probabilities by a few "
            "percent relative. 0 keeps one forward and today's exact scores."
        ),
    )
    parser.add_argument(
        "--llm-context-encoding",
        choices=("auto", "state_delta", "action_only"),
        help=(
            "Demonstration encoding. 'auto' annotates per-step state and drops "
            "the annotation only when the prompt stops fitting, so a long run "
            "may change encoding partway. Pin one to keep a run consistent; "
            "'action_only' is about 2.7x smaller."
        ),
    )
    parser.add_argument(
        "--llm-vram-headroom-gib",
        type=float,
        help=(
            "VRAM left for the display server and other GPU clients. The "
            "prompt budget is sized from what remains, so this is what keeps a "
            "growing prompt from hanging the desktop on a single-GPU machine. "
            "Use 0 only on a GPU that drives no display."
        ),
    )
    parser.add_argument(
        "--event-resume",
        action="store_true",
        help=(
            "Checkpoint each stream after every event and continue from the "
            "last completed one. Makes a scenario survive a native GPU crash "
            "instead of restarting from its first event."
        ),
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
    if args.llm_vram_headroom_gib is not None:
        overrides["llm_vram_headroom_gib"] = float(args.llm_vram_headroom_gib)
    if args.llm_prefill_chunk_tokens is not None:
        overrides["llm_prefill_chunk_tokens"] = int(args.llm_prefill_chunk_tokens)
    if args.llm_context_encoding is not None:
        overrides["llm_context_encoding"] = str(args.llm_context_encoding)
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
        event_resume=bool(args.event_resume),
        audit_prefixes=int(args.audit_prefixes),
        audit_tolerance=float(args.audit_tolerance),
        shared_routing=not bool(args.local_routing),
        top_k=int(args.top_k),
        profile=bool(args.profile),
        sensitivity=bool(args.sensitivity),
        model_settings=overrides,
        experiment="standard_evaluation",
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
