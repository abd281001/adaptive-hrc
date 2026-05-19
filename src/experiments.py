"""Clean experimental harness for the adaptive HRC project.

This module is intentionally separate from ``src.evaluations``. It keeps the
paper-facing benchmark modular: each experiment has a logical name, a typed
configuration, deterministic seed-based scenario selection, seed-level
``result.json`` files, aggregate experiment summaries, and figure data that can
be replotted without reruns.
"""
from __future__ import annotations
import csv
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import random
import socket
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

for _v in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hrc")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adaptive_agent import AdaptiveHRCAgent
from .baselines import BASELINE_AGENTS, OracleCeilingAgent
from .environment import gen
from .memory import jaccard, kendall_tau_distance, variant_hash
from .models import Config, DEFAULT_CONFIG, top_k
from .preferences import (AXES as WORKFLOW_AXES, AXIS_VALUES as WORKFLOW_AXIS_VALUES, PRESET_PREFERENCES, PREFERENCE_NAMES, WorkflowPreference, WorkflowPreferenceModifier)
from .representations import observations_from_actions


DEFAULT_SEEDS: Tuple[int, ...] = (1337, 2024, 7, 9001, 31415)
PAPER_BASELINES: Tuple[str, ...] = (
    "full",
    "adaptive",
    "fixed_decay",
    "experience_replay_bc",
    "ewc",
    "online_ewc",
    "l2_anchor",
    "bigram",
    "frequency_conditioned_bigram",
    "latest_only",
    "no_decay",
    "no_preference_prototype",
    "no_recipe_prototype",
    "oracle_ceiling",
)
APPENDIX_BASELINES: Tuple[str, ...] = (
    "no_replay",
    "uniform_weight",
    "uniform_valid",
    "most_frequent",
    "bc",
)
FACTOR_BASELINES: Tuple[str, ...] = (
    "full",
    "irl_ngram_graph_only",
    "no_preference_prototype",
    "no_recipe_prototype",
    "no_state_graph",
    "no_posterior",
    "no_weighted_rehearsal",
    "latest_only",
    "bigram",
    "oracle_preference_label",
)

_BASELINE_COLORS: Dict[str, str] = {
    "full": "#0f766e",
    "adaptive": "#0369a1",
    "fixed_decay": "#9333ea",
    "no_decay": "#dc2626",
    "no_replay": "#ea580c",
    "latest_only": "#facc15",
    "l2_anchor": "#a16207",
    "ewc": "#0891b2",
    "online_ewc": "#7c3aed",
    "experience_replay_bc": "#be185d",
    "bc": "#475569",
    "bigram": "#94a3b8",
    "frequency_conditioned_bigram": "#64748b",
    "most_frequent": "#9ca3af",
    "uniform_valid": "#cbd5e1",
    "uniform_weight": "#6b7280",
    "no_preference_prototype": "#b45309",
    "no_recipe_prototype": "#7c2d12",
    "no_state_graph": "#4b5563",
    "no_posterior": "#111827",
    "no_weighted_rehearsal": "#ea580c",
    "irl_ngram_graph_only": "#2563eb",
    "oracle_preference_label": "#16a34a",
    "oracle_recipe_and_preference_label": "#15803d",
    "oracle_ceiling": "#000000",
}

_BASELINE_DISPLAY_NAMES: Dict[str, str] = {
    "full": "Full",
    "adaptive": "Adaptive decay",
    "fixed_decay": "Fixed decay",
    "no_decay": "NoDecay (unbounded memory)",
    "latest_only": "Latest only",
    "experience_replay_bc": "Experience replay BC",
    "bigram": "Bigram",
    "frequency_conditioned_bigram": "Weighted bigram",
    "oracle_ceiling": "Oracle ceiling",
}

_BASELINE_GROUPS: Dict[str, str] = {
    "full": "Proposed variants",
    "adaptive": "Proposed variants",
    "fixed_decay": "Proposed variants",
    "experience_replay_bc": "CL baselines",
    "ewc": "CL baselines",
    "online_ewc": "CL baselines",
    "l2_anchor": "CL baselines",
    "bigram": "Weak baselines / ablations",
    "frequency_conditioned_bigram": "Weak baselines / ablations",
    "latest_only": "Weak baselines / ablations",
    "no_decay": "Weak baselines / ablations",
    "no_preference_prototype": "Weak baselines / ablations",
    "no_recipe_prototype": "Weak baselines / ablations",
    "oracle_ceiling": "Oracle",
}

_BASELINE_GROUP_ORDER: Tuple[str, ...] = (
    "Proposed variants",
    "CL baselines",
    "Weak baselines / ablations",
    "Oracle",
)


def _baseline_color(name: str) -> str:
    return _BASELINE_COLORS.get(str(name), "#888888")


def _baseline_label(name: str) -> str:
    return _BASELINE_DISPLAY_NAMES.get(str(name), str(name))


def _baseline_group(name: str) -> str:
    return _BASELINE_GROUPS.get(str(name), "Other")


def _ordered_baseline_names(names: Iterable[str]) -> List[str]:
    names_set = list(dict.fromkeys(str(n) for n in names))
    preferred = list(PAPER_BASELINES) + list(APPENDIX_BASELINES) + list(FACTOR_BASELINES)
    by_name = {name: i for i, name in enumerate(preferred)}
    group_rank = {group: i for i, group in enumerate(_BASELINE_GROUP_ORDER)}
    return sorted(
        names_set,
        key=lambda n: (
            group_rank.get(_baseline_group(n), len(group_rank)),
            by_name.get(n, len(by_name)),
            n,
        ),
    )

CANONICAL_SCENARIOS: Tuple[str, ...] = (
    "materiality_preflight",
    "deployment_stream",
    "cross_recipe_transfer",
    "decay_reentry",
    "disambiguation_audit",
)

DERIVED_VIEWS: Tuple[str, ...] = (
    "memory_efficiency",
    "compute_tradeoff",
    "forgetting_curve",
    "per_user_accuracy",
    "coverage_accuracy_curve",
    "transfer_heatmap",
    "composition_curve",
    "decay_rate_trace",
)

# Public defaults now follow the five-scenario harness design. Legacy runner
# functions remain registered below for explicit debugging/back-compat only.
MAIN_EXPERIMENTS: Tuple[str, ...] = CANONICAL_SCENARIOS

APPENDIX_EXPERIMENTS: Tuple[str, ...] = (
    "zipf_usage_sweep",
    "cl_regularizer_comparison",
    "posterior_ablation_matrix",
)

LEGACY_EXPERIMENTS: Tuple[str, ...] = (
    "materiality_audit",
    "demo_count_sample_efficiency",
    "single_shot_reuse",
    "deployment_gate_preference_shift",
    "cross_recipe_preference_transfer",
    "preference_axis_holdout",
    "novel_preference_composition",
    "adaptive_decay_reentry",
    "multi_user_continual_stream",
    "bounded_memory_tradeoff",
    "compute_tradeoff",
    "short_term_capacity_sweep",
    "disambiguation_threshold_calibration",
    "action_gate_threshold_calibration",
    "boundary_degradation_disambiguation",
    "frequency_gap_decay_sweep",
    "mwr_window_sensitivity",
    "reentry_gap_neutral_vs_distractor",
    "active_pruned_decay_probe",
    "sparse_first_exposure_pool_sweep",
    "cycle_width_sparsity_sweep",
    "baseline_anchor_sweep",
    "continual_learning_regularizer_sweep",
    "recipe_preference_factor_ablation",
    "confidence_calibration_reliability",
    "seed_recipe_selection_audit",
    "runtime_scaling_sweep",
)

ALL_EXPERIMENTS: Tuple[str, ...] = MAIN_EXPERIMENTS + APPENDIX_EXPERIMENTS
FOUR_CELL_KEYS: Tuple[str, ...] = ("seen_seen", "seen_unseen", "unseen_seen", "unseen_unseen")
DEPLOYMENT_REPORT_CELLS: Tuple[str, ...] = ("seen_seen", "seen_unseen")
ASSIST_WRONG_PENALTY: float = 1.0

SCENARIO_NARRATIVES: Dict[str, Dict[str, str]] = {
    "deployment_stream": {
        "claim": "A single deployment stream can test routine reuse, preference shifts, gradual drift, user switches, transfer probes, reentry, memory, compute, and per-step latency without redundant reruns.",
        "pressure": "Shared Phase A followed by a structured Phase B event mix with forgetting checkpoints, user-return blocks, latency logging, and coverage/accuracy curves.",
    },
    "cross_recipe_transfer": {
        "claim": "Preference structure should transfer across recipes, compose across held-out axis values, and show single-shot sufficiency under one shared training checkpoint.",
        "pressure": "Exclusive diagonal training snapshot reused for off-diagonal, axis-holdout, novel-composition, and diagonal-cycle sufficiency probes.",
    },
    "decay_reentry": {
        "claim": "Adaptive decay should prune stale variants while recovering when a task re-enters after neutral time or distractor growth.",
        "pressure": "Neutral and distractor gap arms from the same prefix checkpoint.",
    },
    "disambiguation_audit": {
        "claim": "Classification and action-gate thresholds should be justified by held-out calibration, not hard-coded by inspection.",
        "pressure": "Jaccard threshold sweep, online-prefix degradation, observation-vs-assist gate decisions, and action-gate coverage/accuracy.",
    },
    "materiality_preflight": {
        "claim": "Preference transformations must materially change action orderings before learning claims are interpretable.",
        "pressure": "Seed-level duplicate/no-op filtering and per-recipe discriminability checks.",
    },
    "materiality_audit": {
        "claim": "Preference labels must induce materially different action orderings before downstream learning claims are meaningful.",
        "pressure": "Detect duplicate or no-op preference transformations per recipe.",
    },
    "single_shot_reuse": {
        "claim": "After one observation, the robot should assist on first reuse without re-entering observation mode.",
        "pressure": "First online reuse after a single observed demonstration.",
    },
    "deployment_gate_preference_shift": {
        "claim": "The robot should assist known recipes with new preferences but safe-fail or request observation for truly novel recipes.",
        "pressure": "Contrasts same-recipe preference shift against novel-recipe assist attempts.",
    },
    "cross_recipe_preference_transfer": {
        "claim": "A preference learned on one recipe should transfer to another recipe without seeing that exact recipe-preference pair.",
        "pressure": "Diagonal train set followed by off-diagonal transfer probes.",
    },
    "preference_axis_holdout": {
        "claim": "Latent preference structure should generalize along held-out workflow axes.",
        "pressure": "Train identity-axis values and probe non-identity values by axis.",
    },
    "novel_preference_composition": {
        "claim": "The robot should compose unseen combinations of known preference axes.",
        "pressure": "Train named presets and probe held-out axis-value compositions.",
    },
    "adaptive_decay_reentry": {
        "claim": "Bounded memory should recover gracefully when a decayed task re-enters after a gap.",
        "pressure": "Target, variant, gap, then reentry probe across gap sizes.",
    },
    "multi_user_continual_stream": {
        "claim": "The robot should track changing preferences across users without manual preference labels.",
        "pressure": "Narrative stream mixing routine reuse, preference shifts, user conflicts, reentry, and transfer probes.",
    },
    "bounded_memory_tradeoff": {
        "claim": "Useful assistance should be retained with fewer active variants than unbounded memory.",
        "pressure": "Multi-user continual stream evaluated as assistance per active-memory footprint.",
    },
    "compute_tradeoff": {
        "claim": "Improved assistance should be justified against runtime and refit cost.",
        "pressure": "Shared continual stream with wall-time and fit-count accounting.",
    },
}


@dataclass(frozen=True)
class RunConfig:
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    output_root: str = "eval_results"
    timestamp_subdir: bool = True
    workers: Optional[int] = None
    baselines: Tuple[str, ...] = PAPER_BASELINES
    appendix_baselines: Tuple[str, ...] = PAPER_BASELINES + APPENDIX_BASELINES
    quick: bool = False
    profile: bool = False
    dpi: int = 140
    topk: int = 3
    log_full_distributions: bool = False
    write_debug_jsonl: bool = False
    valid_action_expansion: bool = False


@dataclass(frozen=True)
class MaterialityAuditConfig:
    n_recipes: int = 15
    n_preferences: int = 7


@dataclass(frozen=True)
class SingleShotReuseConfig:
    n_recipes: int = 10
    n_preferences: int = 1
    observe_preference: str = "identity"
    test_preferences: Tuple[str, ...] = ("identity", "p1_prep_first", "p3_clean_eager")


@dataclass(frozen=True)
class DeploymentGateConfig:
    n_train_recipes: int = 8
    train_preference: str = "identity"
    shift_preferences: Tuple[str, ...] = ("p1_prep_first", "p2_frontload", "p3_clean_eager")


@dataclass(frozen=True)
class CrossRecipeTransferConfig:
    n_recipes: int = 7
    n_preferences: int = 7
    diagonal_cycles: int = 1
    repeats: int = 2
    baselines: Tuple[str, ...] = FACTOR_BASELINES


@dataclass(frozen=True)
class PreferenceAxisHoldoutConfig:
    n_recipes: int = 10
    settle_cycles: int = 2


@dataclass(frozen=True)
class NovelPreferenceCompositionConfig:
    n_recipes: int = 10
    settle_cycles: int = 2


@dataclass(frozen=True)
class AdaptiveDecayReentryConfig:
    n_recipes: int = 12
    gap_sweep: Tuple[int, ...] = (5, 10, 15, 30, 45)
    neutral_filler: bool = True


@dataclass(frozen=True)
class MultiUserStreamConfig:
    n_users: int = 5
    n_recipes: int = 10
    n_events: int = 120
    observe_first_recipe: bool = True
    switch_probability: float = 0.45
    zipf_alpha: float = 1.2


@dataclass(frozen=True)
class MemoryTradeoffConfig:
    n_recipes: int = 12
    n_events: int = 150
    zipf_alpha: float = 1.0


@dataclass(frozen=True)
class ComputeTradeoffConfig:
    n_recipes: int = 8
    n_events: int = 60


@dataclass(frozen=True)
class DeploymentStreamConfig:
    n_recipes: int = 10
    n_users: int = 4
    n_phase_b_events: int = 120
    zipf_alpha: float = 1.2
    event_mix: Dict[str, float] = field(default_factory=lambda: {
        "routine_reuse": 0.40,
        "preference_shift": 0.12,
        "gradual_shift": 0.10,
        "user_switch": 0.16,
        "cross_transfer_probe": 0.12,
        "rare_reentry": 0.10,
    })
    phase_a_non_identity_prob: float = 0.50
    stay_same_user_prob: float = 0.70
    preference_switch_prob: float = 0.20
    tail_reentry_rate: float = 0.15
    cross_product_probe_rate: float = 0.10
    new_recipe_obs_rate: float = 0.05
    user_block_size: int = 20
    transfer_warmup_events: int = 30
    forgetting_checkpoint_interval: int = 10
    coverage_curve_thresholds: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class TransferSuiteConfig:
    n_recipes: int = 7
    n_preferences: int = 7
    diagonal_cycles: int = 1
    offdiag_repeats: int = 2
    diagonal_cycle_sweep: Tuple[int, ...] = (1, 2, 3, 5)
    include_axis_holdout: bool = True
    include_novel_composition: bool = True
    baselines: Tuple[str, ...] = FACTOR_BASELINES


@dataclass(frozen=True)
class DecayReentrySuiteConfig:
    n_recipes: int = 12
    gap_sweep: Tuple[int, ...] = (5, 10, 15, 30, 45)
    n_target_recipes: int = 5
    run_neutral_arm: bool = True
    run_distractor_arm: bool = True
    baselines: Tuple[str, ...] = ("full", "adaptive", "fixed_decay", "no_decay", "latest_only", "experience_replay_bc", "bigram")


@dataclass(frozen=True)
class DisambiguationAuditConfig:
    n_recipes: int = 12
    thresholds: Tuple[float, ...] = (0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98, 0.99)
    action_gate_thresholds: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    validation_fraction: float = 0.30
    drop_fractions: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)


@dataclass(frozen=True)
class MaterialityPreflightConfig:
    n_recipes: int = 15
    n_preferences: int = 7
    min_effective_preferences: int = 5
    fail_on_noop: bool = True


@dataclass(frozen=True)
class CapacitySweepConfig:
    demo_counts: Tuple[int, ...] = (3, 5, 10, 15)
    n_preferences: int = 3


@dataclass(frozen=True)
class ThresholdCalibrationConfig:
    n_recipes: int = 12
    thresholds: Tuple[float, ...] = (0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98, 0.99)
    validation_fraction: float = 0.30
    drop_fractions: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)


@dataclass(frozen=True)
class GapSweepConfig:
    n_recipes: int = 12
    gaps: Tuple[int, ...] = (5, 10, 15, 30, 45)


@dataclass(frozen=True)
class MWRWindowSensitivityConfig:
    n_recipes: int = 12
    windows: Tuple[int, ...] = (10, 20, 30, 50)
    gap: int = 30


@dataclass(frozen=True)
class SparsePoolConfig:
    pool_sizes: Tuple[int, ...] = (8, 12, 20, 30)
    n_explore: int = 40
    n_settle: int = 60


@dataclass(frozen=True)
class ZipfUsageConfig:
    n_recipes: int = 12
    n_events: int = 120
    alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)


@dataclass(frozen=True)
class CycleWidthConfig:
    n_recipes: int = 15
    n_events: int = 100
    cycle_widths: Tuple[int, ...] = (1, 2, 4, 8, 15)


@dataclass(frozen=True)
class StressConfig:
    n_recipes: int = 12
    reps: int = 20
    gap: int = 30


@dataclass(frozen=True)
class RuntimeScalingConfig:
    recipe_counts: Tuple[int, ...] = (3, 5, 10, 15)
    n_events: int = 50


@dataclass(frozen=True)
class ExperimentSuiteConfig:
    deployment_stream: DeploymentStreamConfig = field(default_factory=DeploymentStreamConfig)
    cross_recipe_transfer: TransferSuiteConfig = field(default_factory=TransferSuiteConfig)
    decay_reentry: DecayReentrySuiteConfig = field(default_factory=DecayReentrySuiteConfig)
    disambiguation_audit: DisambiguationAuditConfig = field(default_factory=DisambiguationAuditConfig)
    materiality_preflight: MaterialityPreflightConfig = field(default_factory=MaterialityPreflightConfig)
    materiality_audit: MaterialityAuditConfig = field(default_factory=MaterialityAuditConfig)
    single_shot_reuse: SingleShotReuseConfig = field(default_factory=SingleShotReuseConfig)
    deployment_gate_preference_shift: DeploymentGateConfig = field(default_factory=DeploymentGateConfig)
    cross_recipe_preference_transfer: CrossRecipeTransferConfig = field(default_factory=CrossRecipeTransferConfig)
    preference_axis_holdout: PreferenceAxisHoldoutConfig = field(default_factory=PreferenceAxisHoldoutConfig)
    novel_preference_composition: NovelPreferenceCompositionConfig = field(default_factory=NovelPreferenceCompositionConfig)
    adaptive_decay_reentry: AdaptiveDecayReentryConfig = field(default_factory=AdaptiveDecayReentryConfig)
    multi_user_continual_stream: MultiUserStreamConfig = field(default_factory=MultiUserStreamConfig)
    bounded_memory_tradeoff: MemoryTradeoffConfig = field(default_factory=MemoryTradeoffConfig)
    compute_tradeoff: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    short_term_capacity_sweep: CapacitySweepConfig = field(default_factory=CapacitySweepConfig)
    demo_count_sample_efficiency: CapacitySweepConfig = field(default_factory=CapacitySweepConfig)
    disambiguation_threshold_calibration: ThresholdCalibrationConfig = field(default_factory=ThresholdCalibrationConfig)
    action_gate_threshold_calibration: ThresholdCalibrationConfig = field(default_factory=ThresholdCalibrationConfig)
    boundary_degradation_disambiguation: ThresholdCalibrationConfig = field(default_factory=ThresholdCalibrationConfig)
    frequency_gap_decay_sweep: GapSweepConfig = field(default_factory=GapSweepConfig)
    mwr_window_sensitivity: MWRWindowSensitivityConfig = field(default_factory=MWRWindowSensitivityConfig)
    reentry_gap_neutral_vs_distractor: GapSweepConfig = field(default_factory=GapSweepConfig)
    active_pruned_decay_probe: GapSweepConfig = field(default_factory=GapSweepConfig)
    sparse_first_exposure_pool_sweep: SparsePoolConfig = field(default_factory=SparsePoolConfig)
    zipf_usage_sweep: ZipfUsageConfig = field(default_factory=ZipfUsageConfig)
    cl_regularizer_comparison: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    cycle_width_sparsity_sweep: CycleWidthConfig = field(default_factory=CycleWidthConfig)
    baseline_anchor_sweep: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    continual_learning_regularizer_sweep: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    posterior_ablation_matrix: CrossRecipeTransferConfig = field(default_factory=CrossRecipeTransferConfig)
    recipe_preference_factor_ablation: CrossRecipeTransferConfig = field(default_factory=CrossRecipeTransferConfig)
    coverage_accuracy_curve: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    confidence_calibration_reliability: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    memory_exhaustion_stress: StressConfig = field(default_factory=StressConfig)
    prefix_collision_stress: StressConfig = field(default_factory=StressConfig)
    preference_thrash_stress: StressConfig = field(default_factory=StressConfig)
    rare_reentry_stress: StressConfig = field(default_factory=StressConfig)
    late_distractor_stress: StressConfig = field(default_factory=StressConfig)
    seed_recipe_selection_audit: MaterialityAuditConfig = field(default_factory=MaterialityAuditConfig)
    runtime_scaling_sweep: RuntimeScalingConfig = field(default_factory=RuntimeScalingConfig)


@dataclass(frozen=True)
class RecipePrefPair:
    recipe_name: str
    preference_name: str
    label: str
    actions: Tuple[str, ...]
    preference: Optional[WorkflowPreference] = None
    base_pref: bool = False
    applied_axes: Tuple[str, ...] = ()
    failed_axes: Tuple[str, ...] = ()
    unchanged_axes: Tuple[str, ...] = ()

    @property
    def recipe_id(self) -> str:
        return self.recipe_name


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: str
    completed_experiments: Tuple[str, ...]
    failed_experiments: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "out_dir": self.run_dir,
            "completed_experiments": list(self.completed_experiments),
            "completed_tests": list(self.completed_experiments),
            "failed_experiments": self.failed_experiments,
        }


@dataclass(frozen=True)
class ScenarioEvent:
    pair: RecipePrefPair
    mode: Literal["observe", "assist"]
    phase: str
    user_id: str = "U1"
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveStepRecord:
    step: int
    predicted: Optional[str]
    actual: str
    correct_top1: bool
    correct_topk: bool
    topk: Tuple[str, ...]
    inferred_recipe: Optional[str]
    inferred_pref: Optional[str]
    inferred_latent_pref: Optional[str] = None
    posterior_entropy: Optional[float] = None
    posterior_confidence: Optional[float] = None
    posterior_raw_entropy: Optional[float] = None
    posterior_max_prob: Optional[float] = None
    posterior_n_hypotheses: Optional[int] = None
    assist_used: bool = False
    action_marginal_entropy: Optional[float] = None
    prediction_wall_s: Optional[float] = None


@dataclass(frozen=True)
class LiveEpisodeRecord:
    pair_label: str
    recipe_id: str
    recipe_name: str
    preference_name: str
    memory_state: str
    mode: str
    steps: Tuple[LiveStepRecord, ...]
    live_top1: float
    live_topk: float
    n: int
    first_mismatch_step: Optional[int]


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (set, tuple)):
            return list(obj)
        return super().default(obj)


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, cls=_JSONEncoder))


class _SeedResultCollector:
    def __init__(self, experiment: str, seed: int, seed_dir: Path, config: Any) -> None:
        self.experiment = experiment
        self.seed = int(seed)
        self.seed_dir = Path(seed_dir)
        self.config = config
        self.metadata: Dict[str, Any] = {
            "experiment": experiment,
            "seed": int(seed),
            "created_at": _dt.datetime.now().isoformat(),
        }
        self.scenario: Dict[str, Any] = {
            "events": [],
            "selected_recipes": [],
            "selected_preferences": [],
            "narrative": SCENARIO_NARRATIVES.get(experiment, {}),
        }
        self.records: Dict[str, List[Dict[str, Any]]] = {}
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.figures: Dict[str, Dict[str, Any]] = {}
        self.warnings: List[str] = []
        self.captured_json: Dict[str, Any] = {}

    def capture_json(self, path: Path, obj: Any) -> None:
        name = path.name
        stem = path.stem
        if name == "experiment_config.json":
            self.captured_json["experiment_config"] = _jsonable(obj)
        elif name == "selected_recipes.json":
            self.scenario["selected_recipes"] = _jsonable(obj)
        elif name == "selected_preferences.json":
            self.scenario["selected_preferences"] = _jsonable(obj)
        elif name == "metrics.json":
            self.captured_json["metrics"] = _jsonable(obj)
        else:
            self.captured_json[stem] = _jsonable(obj)

    def capture_rows(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        key = path.stem
        clean = [_jsonable(r) for r in rows]
        self.records.setdefault(key, []).extend(clean)
        if key == "scenario_events":
            self.scenario.setdefault("events", []).extend(clean)

    def capture_table(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        self.tables[path.stem] = [_jsonable(r) for r in rows]

    def capture_figure(self, path: Path, data: Dict[str, Any]) -> None:
        self.figures[path.name] = {
            "path": path.name,
            "data": _jsonable(data),
        }

    def finalize(self, metrics: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
        metrics = _jsonable(metrics)
        scenario = self._scenario_summary()
        baseline_health = _baseline_health(metrics, self.records)
        result = {
            "metadata": self.metadata,
            "status": _jsonable(status),
            "config": _jsonable(self.config),
            "scenario": scenario,
            "baseline_health": baseline_health,
            "baselines": metrics.get("per_baseline", {}),
            "metrics": metrics,
            "comparisons": _comparison_summary(metrics),
            "records": _jsonable(self.records),
            "tables": _jsonable(self.tables),
            "figures": _jsonable(self.figures),
            "warnings": list(self.warnings) + _scenario_warnings(scenario) + _baseline_warnings(baseline_health),
        }
        _write_json_file(self.seed_dir / "result.json", result)
        self._remove_empty_subdirs()
        return result

    def _scenario_summary(self) -> Dict[str, Any]:
        scenario = _jsonable(self.scenario)
        events = list(scenario.get("events", []))
        if not events and "episode_metrics" in self.records:
            events = [
                {
                    "mode": r.get("mode"),
                    "pair": r.get("pair"),
                    "recipe": r.get("recipe"),
                    "preference": r.get("preference"),
                    "condition": r.get("condition") or r.get("experiment_condition"),
                    "event_idx": r.get("event_idx", i),
                }
                for i, r in enumerate(self.records.get("episode_metrics", []))
            ]
            scenario["events"] = events
        mode_counts = Counter(str(e.get("mode", e.get("condition", "unknown"))) for e in events)
        event_type_counts = Counter(str(e.get("event_type", e.get("condition", "unknown"))) for e in events)
        recipes = {str(e.get("recipe") or str(e.get("pair", "")).split("/")[0]) for e in events if e.get("recipe") or e.get("pair")}
        prefs = {str(e.get("preference") or str(e.get("pair", "")).split("/")[-1]) for e in events if e.get("preference") or e.get("pair")}
        signature = tuple(
            (
                str(e.get("pair", "")),
                str(e.get("mode", "")),
                str(e.get("phase", "")),
                str(e.get("user_id", "")),
                str(e.get("event_type", e.get("condition", ""))),
            )
            for e in events
        )
        scenario.update({
            "n_events": len(events),
            "mode_counts": dict(mode_counts),
            "event_type_counts": dict(event_type_counts),
            "n_unique_recipes": len(recipes),
            "n_unique_preferences": len(prefs),
            "event_stream_signature": signature,
            "support_counts": _support_counts(events),
        })
        return scenario

    def _remove_empty_subdirs(self) -> None:
        for path in sorted((p for p in self.seed_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass


_ACTIVE_RESULT: Optional[_SeedResultCollector] = None


def _set_active_result(collector: Optional[_SeedResultCollector]) -> None:
    global _ACTIVE_RESULT
    _ACTIVE_RESULT = collector


def _write_json_file(path: Path, obj: Any) -> Path:
    _ensure(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, cls=_JSONEncoder), encoding="utf-8")
    return path


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d_%H%M")


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, obj: Any) -> Path:
    if _ACTIVE_RESULT is not None and path.name not in {"result.json", "aggregate.json", "run.json"}:
        _ACTIVE_RESULT.capture_json(path, obj)
        return path
    _write_json_file(path, obj)
    return path


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    materialized = list(rows)
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_rows(path, materialized)
        return path
    _ensure(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in materialized:
            f.write(json.dumps(row, sort_keys=True, cls=_JSONEncoder) + "\n")
    return path


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_table(path, rows)
        return path
    _ensure(path.parent)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _scalar(row.get(k)) for k in keys})
    return path


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, cls=_JSONEncoder)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _stderr95(values: Sequence[float]) -> float:
    """95% CI half-width using the t-distribution.

    The default run uses few independent seeds; a normal 1.96 multiplier is
    anti-conservative at that scale (for n=3, t=4.30).
    """
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    _T95 = {
        1: 12.71,
        2: 4.30,
        3: 3.18,
        4: 2.78,
        5: 2.57,
        6: 2.45,
        7: 2.36,
        8: 2.31,
        9: 2.26,
        10: 2.23,
        11: 2.20,
        12: 2.18,
        13: 2.16,
        14: 2.14,
        15: 2.13,
        16: 2.12,
        17: 2.11,
        18: 2.10,
        19: 2.09,
        20: 2.09,
        24: 2.06,
        29: 2.05,
        39: 2.02,
        59: 2.00,
        119: 1.98,
    }
    df = len(vals) - 1
    if df in _T95:
        t = _T95[df]
    else:
        larger = sorted(k for k in _T95 if k >= df)
        t = _T95[larger[0]] if larger else 1.96
    return float(t * math.sqrt(var / len(vals)))


def mean_ci95(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not vals:
        return float("nan"), 0.0
    return _mean(vals), _stderr95(vals)


def _p95(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = int(math.ceil(0.95 * len(vals))) - 1
    return float(vals[max(0, min(idx, len(vals) - 1))])


def binary_decision_metrics(scores: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.size == 0 or y.size == 0 or s.size != y.size:
        return {"auroc": 0.0, "auprc": 0.0, "ece": 0.0, "brier": 0.0, "calibration": []}
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
    else:
        sum_pos_ranks = float(ranks[y == 1].sum())
        auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    desc = np.argsort(-s)
    y_desc = y[desc]
    if n_pos == 0:
        auprc = 0.0
    else:
        tp = np.cumsum(y_desc == 1)
        fp = np.cumsum(y_desc == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / n_pos
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        auprc = float(np.trapezoid(precision, recall)) if hasattr(np, "trapezoid") else float(np.trapz(precision, recall))
    if s.min() < 0 or s.max() > 1:
        s_norm = (s - float(s.min())) / max(float(s.max() - s.min()), 1e-9)
    else:
        s_norm = s
    edges = np.linspace(0, 1, max(1, int(n_bins)) + 1)
    calibration: List[Tuple[float, float, int]] = []
    ece = 0.0
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        mask = ((s_norm >= lo) & (s_norm <= hi)) if k == len(edges) - 2 else ((s_norm >= lo) & (s_norm < hi))
        if int(mask.sum()) == 0:
            continue
        mean_score = float(s_norm[mask].mean())
        empirical = float(y[mask].mean())
        count = int(mask.sum())
        ece += (count / len(s_norm)) * abs(mean_score - empirical)
        calibration.append((mean_score, empirical, count))
    brier = float(np.mean((s_norm - y) ** 2))
    return {"auroc": float(auroc), "auprc": float(auprc), "ece": float(ece), "brier": brier, "calibration": calibration}


def behavioral_steps_to_lock(steps: Sequence[LiveStepRecord], window: int = 3, threshold: float = 0.75) -> int:
    """Baseline-agnostic lock time based on sustained behavioral accuracy."""
    n = len(steps)
    if n <= 0:
        return -1
    width = max(1, int(window))
    for k in range(n):
        stable = True
        for j in range(k, n):
            chunk = steps[j:min(n, j + width)]
            if not chunk:
                continue
            if _mean([1.0 if s.correct_top1 else 0.0 for s in chunk]) < float(threshold):
                stable = False
                break
        if stable:
            return k
    return -1


def posterior_steps_to_lock(steps: Sequence[LiveStepRecord], true_rid: Optional[str]) -> int:
    if true_rid is None:
        return -1
    for k in range(len(steps)):
        if all(steps[j].inferred_recipe == true_rid for j in range(k, len(steps))):
            return k
    return -1


def live_episode_metrics(record: LiveEpisodeRecord, *, true_rid: Optional[str] = None) -> Dict[str, float]:
    n = max(1, int(record.n))
    if record.n <= 0:
        return {
            "live_top1": 0.0,
            "live_topk": 0.0,
            "post_divergence_top1": 0.0,
            "first_mismatch_step": -1.0,
            "steps_to_lock": -1.0,
            "behavioral_steps_to_lock": -1.0,
            "posterior_steps_to_lock": -1.0,
            "adaptation_latency_steps": -1.0,
            "assistance_coverage": 0.0,
            "conditional_top1": 0.0,
            "useful_assistance_rate": 0.0,
            "confidence_ece": 0.0,
            "confidence_brier": 0.0,
            "mean_prediction_wall_s": 0.0,
            "p95_prediction_wall_s": 0.0,
        }
    fm = record.first_mismatch_step if record.first_mismatch_step is not None else -1
    post_steps = record.steps[fm + 1:] if fm >= 0 else record.steps
    assist_steps = [s for s in record.steps if s.assist_used]
    closed_steps = [s for s in record.steps if not s.assist_used]
    wrong_steps = [s for s in record.steps if not s.correct_top1]
    confidences = [float(s.posterior_confidence) for s in record.steps if s.posterior_confidence is not None]
    entropies = [float(s.posterior_entropy) for s in record.steps if s.posterior_entropy is not None]
    action_entropies = [float(s.action_marginal_entropy) for s in record.steps if s.action_marginal_entropy is not None]
    n_hypotheses = [int(s.posterior_n_hypotheses) for s in record.steps if s.posterior_n_hypotheses is not None]
    scores = [float(s.posterior_confidence) for s in record.steps if s.posterior_confidence is not None]
    labels = [1 if s.correct_top1 else 0 for s in record.steps if s.posterior_confidence is not None]
    calib = binary_decision_metrics(scores, labels, n_bins=10) if scores else {"ece": 0.0, "brier": 0.0}
    steps_to_lock = behavioral_steps_to_lock(record.steps)
    posterior_lock = posterior_steps_to_lock(record.steps, true_rid)
    recovery_latency = -1
    if fm >= 0:
        for k in range(fm + 1, len(record.steps)):
            chunk = record.steps[k:k + 3]
            if chunk and _mean([1.0 if s.correct_top1 else 0.0 for s in chunk]) >= 0.75:
                recovery_latency = k - fm
                break
    recipe_vocab = {s.actual for s in record.steps}
    recipe_vocab_hits = [1.0 if s.predicted in recipe_vocab else 0.0 for s in record.steps]
    recipe_vocab_correct = [s for s in record.steps if s.predicted in recipe_vocab]
    prediction_wall = [float(s.prediction_wall_s) for s in record.steps if s.prediction_wall_s is not None]
    recipe_vocab_conditional = _mean([1.0 if s.correct_top1 else 0.0 for s in recipe_vocab_correct])
    return {
        "live_top1": float(record.live_top1),
        "live_topk": float(record.live_topk),
        "post_divergence_top1": _mean([1.0 if s.correct_top1 else 0.0 for s in post_steps]),
        "first_mismatch_step": float(fm),
        "steps_to_lock": float(steps_to_lock),
        "behavioral_steps_to_lock": float(steps_to_lock),
        "posterior_steps_to_lock": float(posterior_lock),
        "adaptation_latency_steps": float(recovery_latency),
        "posterior_correct_recipe": _mean([1.0 if s.inferred_recipe == true_rid else 0.0 for s in record.steps]) if true_rid is not None else 0.0,
        "recipe_vocab_top1": _mean(recipe_vocab_hits),
        "recipe_vocab_conditional_top1": recipe_vocab_conditional,
        "preference_consistent_top1": recipe_vocab_conditional,
        "assistance_coverage": len(assist_steps) / n,
        "coverage": len(assist_steps) / n,
        "conditional_top1": _mean([1.0 if s.correct_top1 else 0.0 for s in assist_steps]),
        "useful_assistance_rate": (len(assist_steps) / n) * _mean([1.0 if s.correct_top1 else 0.0 for s in assist_steps]),
        "assist_correct_rate": len([s for s in assist_steps if s.correct_top1]) / n,
        "assist_wrong_rate": len([s for s in assist_steps if not s.correct_top1]) / n,
        "net_assistance_value": (
            len([s for s in assist_steps if s.correct_top1])
            - ASSIST_WRONG_PENALTY * len([s for s in assist_steps if not s.correct_top1])
        ) / n,
        "abstention_error_rate": _mean([1.0 if not s.correct_top1 else 0.0 for s in closed_steps]),
        "abstention_recall": len([s for s in closed_steps if not s.correct_top1]) / max(1, len(wrong_steps)),
        "mean_action_confidence": _mean(confidences),
        "mean_posterior_entropy": _mean(entropies),
        "mean_action_marginal_entropy": _mean(action_entropies),
        "posterior_degenerate_rate": _mean([1.0 if h <= 1 else 0.0 for h in n_hypotheses]),
        "confidence_ece": float(calib.get("ece", 0.0)),
        "confidence_brier": float(calib.get("brier", 0.0)),
        "mean_prediction_wall_s": _mean(prediction_wall),
        "p95_prediction_wall_s": _p95(prediction_wall),
    }


def aggregate_live_episodes(records: Sequence[LiveEpisodeRecord]) -> Dict[str, float]:
    if not records:
        return _aggregate_episode_metrics([])
    rows = [live_episode_metrics(r) for r in records]
    weights = [max(1, int(r.n)) for r in records]
    total = max(1, sum(weights))
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    out: Dict[str, float] = {"n_episodes": float(len(records)), "n_steps": float(total), "n": float(total)}
    for key in keys:
        out[key] = sum(float(row.get(key, 0.0)) * w for row, w in zip(rows, weights)) / total
    pooled_scores: List[float] = []
    pooled_labels: List[int] = []
    for record in records:
        for step in record.steps:
            if step.posterior_confidence is not None:
                pooled_scores.append(float(step.posterior_confidence))
                pooled_labels.append(1 if step.correct_top1 else 0)
    if pooled_scores:
        calib = binary_decision_metrics(pooled_scores, pooled_labels, n_bins=10)
        out["pooled_ece"] = float(calib.get("ece", 0.0))
        out["pooled_brier"] = float(calib.get("brier", 0.0))
        out["pooled_auroc"] = float(calib.get("auroc", 0.5))
        out["n_calib_pooled"] = float(len(pooled_scores))
    else:
        out["pooled_ece"] = 0.0
        out["pooled_brier"] = 0.0
        out["pooled_auroc"] = 0.5
        out["n_calib_pooled"] = 0.0
    return out


def bwt_fwt_checkpoints(
    phase_a_eval: Dict[str, Dict[str, float]],
    final_eval: Dict[str, Dict[str, float]],
    phase_a_seen_labels: Sequence[str],
) -> Dict[str, Any]:
    labels = sorted(k for k in final_eval if k in phase_a_eval)
    seen = set(phase_a_seen_labels)
    bwt_terms = [
        float(final_eval[label].get("top1", 0.0)) - float(phase_a_eval[label].get("top1", 0.0))
        for label in labels
        if label in seen
    ]
    fwt_terms = [
        float(phase_a_eval[label].get("top1", 0.0))
        for label in labels
        if label not in seen
    ]
    return {
        "bwt": float(_mean(bwt_terms)) if bwt_terms else float("nan"),
        "fwt": float(_mean(fwt_terms)) if fwt_terms else float("nan"),
        "n_bwt_tasks": len(bwt_terms),
        "n_fwt_tasks": len(fwt_terms),
        "bwt_terms": bwt_terms,
        "fwt_terms": fwt_terms,
    }


MEMORY_STATE_LABELS: Tuple[str, ...] = ("no_memory", "active_memory", "pruned_memory", "same_recipe_new_preference")


def confusion_labels(
    predicted: Sequence[str],
    ground_truth: Sequence[str],
    labels: Sequence[str] = MEMORY_STATE_LABELS,
) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for pred, gt in zip(predicted, ground_truth):
        if pred in idx and gt in idx:
            matrix[idx[gt], idx[pred]] += 1
    return matrix


def confusion_3label(
    predicted: Sequence[str],
    ground_truth: Sequence[str],
    labels: Sequence[str] = MEMORY_STATE_LABELS,
) -> np.ndarray:
    return confusion_labels(predicted, ground_truth, labels=labels)


def classifier_report(matrix: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    n_total = int(matrix.sum())
    if n_total <= 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}
    accuracy = float(np.trace(matrix) / n_total)
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fn = int(matrix[i, :].sum() - tp)
        fp = int(matrix[:, i].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(matrix[i, :].sum()),
        }
        f1s.append(float(f1))
    return {"accuracy": accuracy, "macro_f1": _mean(f1s), "per_class": per_class}


def preference_lock_purity(
    records: Sequence[LiveEpisodeRecord],
    preset_of_pair_label: Dict[str, str],
) -> float:
    by_preset: Dict[str, List[str]] = {}
    for record in records:
        preset = preset_of_pair_label.get(record.pair_label)
        if preset is None:
            continue
        for step in record.steps:
            pid = step.inferred_latent_pref or step.inferred_pref
            if pid is not None:
                by_preset.setdefault(preset, []).append(str(pid))
    total = 0
    modal_total = 0
    for pids in by_preset.values():
        if not pids:
            continue
        counts = Counter(pids)
        modal_total += counts.most_common(1)[0][1]
        total += len(pids)
    return float(modal_total / total) if total > 0 else 0.0


def preference_axis_top1(
    records_by_label: Dict[str, Dict[str, float]],
    pairs: Sequence[RecipePrefPair],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for axis in WORKFLOW_AXES:
        for value in WORKFLOW_AXIS_VALUES.get(axis, []):
            vals: List[float] = []
            for pair in pairs:
                pref = PRESET_PREFERENCES.get(pair.preference_name)
                if pref is None or pref.as_dict().get(axis) != value:
                    continue
                if pair.label in records_by_label:
                    vals.append(float(records_by_label[pair.label].get("top1", 0.0)))
            if vals:
                out[f"{axis}={value}"] = _mean(vals)
    return out


def per_recipe_accuracy_matrix(records_by_label: Dict[str, Dict[str, float]], pairs: Sequence[RecipePrefPair]) -> Dict[str, Dict[str, float]]:
    matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    for pair in pairs:
        if pair.label in records_by_label:
            matrix[pair.recipe_name][pair.preference_name] = float(records_by_label[pair.label].get("top1", 0.0))
    return {recipe: dict(vals) for recipe, vals in matrix.items()}


def live_step_trace(step_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    episodes: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(step_rows):
        key = (
            row.get("baseline"),
            row.get("event_idx"),
            row.get("pair"),
            row.get("condition"),
            row.get("repeat"),
            row.get("arm"),
            row.get("gap"),
        )
        if key[1] is None:
            key = (row.get("baseline"), row.get("pair"), row.get("condition"), i)
        episodes[key].append(row)
    if episodes:
        lengths = [len(rows) for rows in episodes.values()]
        target_len = min(20, max(lengths))
        eligible = [rows for rows in episodes.values() if len(rows) >= target_len]
        if not eligible:
            target_len = min(lengths)
            eligible = [rows for rows in episodes.values() if len(rows) >= target_len]
        rows_for_trace = [
            row
            for rows in eligible
            for row in rows
            if int(row.get("step", 0)) < target_len
        ]
    else:
        rows_for_trace = list(step_rows)
    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows_for_trace:
        by_step[int(row.get("step", 0))].append(row)
    out: List[Dict[str, float]] = []
    for step, rows in sorted(by_step.items()):
        out.append({
            "step": float(step),
            "top1": _mean([1.0 if r.get("correct_top1") else 0.0 for r in rows]),
            "assist_rate": _mean([1.0 if r.get("assist_used") else 0.0 for r in rows]),
            "posterior_entropy": _mean([float(r.get("posterior_entropy")) for r in rows if r.get("posterior_entropy") is not None]),
            "action_confidence": _mean([float(r.get("action_confidence")) for r in rows if r.get("action_confidence") is not None]),
            "n_episodes": float(len({(r.get("baseline"), r.get("event_idx"), r.get("pair"), r.get("condition"), r.get("repeat"), r.get("arm"), r.get("gap")) for r in rows})),
            "n_steps": float(len(rows)),
        })
    return out


def _flatten_numeric(prefix: str, obj: Any, out: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            _flatten_numeric(key, v, out)
    elif isinstance(obj, (int, float, bool)) and not isinstance(obj, bool):
        if math.isfinite(float(obj)):
            out[prefix] = float(obj)
    elif isinstance(obj, bool):
        out[prefix] = 1.0 if obj else 0.0
    return out


def _flat_artifact_path(path: Path) -> Path:
    path = Path(path)
    if _ACTIVE_RESULT is not None:
        try:
            rel = path.relative_to(_ACTIVE_RESULT.seed_dir)
            parts = [p for p in rel.parts if p != "figures"]
            if len(parts) > 1:
                return _ACTIVE_RESULT.seed_dir / "_".join(parts)
            return _ACTIVE_RESULT.seed_dir / parts[0]
        except ValueError:
            pass
    if path.parent.name == "figures":
        return path.parent.parent / path.name
    return path


def _remove_stale_pngs(root: Path, patterns: Sequence[str]) -> None:
    for pattern in patterns:
        for path in Path(root).glob(pattern):
            if path.is_file() and path.suffix.lower() == ".png":
                try:
                    path.unlink()
                except OSError:
                    pass


def _plot_bar(
    path: Path,
    title: str,
    values: Dict[str, float],
    ylabel: str = "value",
    data: Optional[Dict[str, Any]] = None,
    colors: Optional[Sequence[str]] = None,
) -> None:
    if not values:
        return
    path = _flat_artifact_path(path)
    _ensure(path.parent)
    labels = list(values.keys())
    ys = [float(values[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(labels) + 2), 3.6), dpi=140)
    bar_colors = list(colors) if colors is not None else [_baseline_color(label) for label in labels]
    ax.bar(np.arange(len(labels)), ys, color=bar_colors)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([_baseline_label(label) for label in labels], rotation=25, ha="right")
    groups = _add_baseline_group_separators(ax, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    fig_data = data or {"values": values, "title": title, "ylabel": ylabel, "baseline_groups": groups}
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_figure(path, fig_data)


def _add_baseline_group_separators(ax: Any, labels: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    if not labels:
        return groups
    group_names = [_baseline_group(label) for label in labels]
    for label, group in zip(labels, group_names):
        groups.setdefault(group, []).append(label)
    for i in range(1, len(labels)):
        if group_names[i] != group_names[i - 1]:
            ax.axvline(i - 0.5, color="#9ca3af", linestyle=":", linewidth=1.0, alpha=0.8)
    return groups


def _plot_bar_ci(path: Path, title: str, values: Dict[str, float], ci95: Optional[Dict[str, float]] = None, ylabel: str = "value") -> None:
    if not values:
        return
    path = _flat_artifact_path(path)
    _ensure(path.parent)
    labels = list(values.keys())
    ys = [float(values[k]) for k in labels]
    yerr = [float((ci95 or {}).get(k, 0.0)) for k in labels]
    fig, ax = plt.subplots(figsize=(max(5.2, 0.62 * len(labels) + 2), 3.8), dpi=140)
    ax.bar(np.arange(len(labels)), ys, yerr=yerr, capsize=3, color=[_baseline_color(label) for label in labels])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([_baseline_label(label) for label in labels], rotation=25, ha="right")
    groups = _add_baseline_group_separators(ax, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_figure(path, {"values": values, "ci95": ci95 or {}, "title": title, "ylabel": ylabel, "baseline_groups": groups})


def _plot_lines(
    path: Path,
    title: str,
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    vlines: Optional[Sequence[Tuple[float, str]]] = None,
    hlines: Optional[Sequence[Tuple[float, str]]] = None,
) -> None:
    if not series:
        return
    path = _flat_artifact_path(path)
    _ensure(path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=140)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#888888"])
    for idx, (label, (xs, ys)) in enumerate(series.items()):
        linestyle = "--" if str(label) == "oracle_ceiling" else "-"
        color = _baseline_color(label) if str(label) in _BASELINE_COLORS else cycle[idx % len(cycle)]
        ax.plot(list(xs), list(ys), marker="o", lw=1.4, label=_baseline_label(label), color=color, linestyle=linestyle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for x, label in vlines or ():
        ax.axvline(float(x), color="#991b1b", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.text(float(x), 0.98, str(label), transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=7, color="#991b1b")
    for y, label in hlines or ():
        ax.axhline(float(y), color="#374151", linestyle=":", linewidth=1.0, alpha=0.85)
        ax.text(0.99, float(y), str(label), transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=7, color="#374151")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    fig_data = {"series": series, "title": title, "xlabel": xlabel, "ylabel": ylabel, "vlines": list(vlines or ()), "hlines": list(hlines or ())}
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_figure(path, fig_data)


def _plot_lines_ci(
    path: Path,
    title: str,
    series: Dict[str, Tuple[Sequence[float], Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    vlines: Optional[Sequence[Tuple[float, str]]] = None,
    hlines: Optional[Sequence[Tuple[float, str]]] = None,
) -> None:
    if not series:
        return
    path = _flat_artifact_path(path)
    _ensure(path.parent)
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=140)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#888888"])
    for idx, (label, (xs, ys, ci)) in enumerate(series.items()):
        x_vals = list(xs)
        y_vals = [float(y) for y in ys]
        err_vals = [float(e) for e in ci]
        linestyle = "--" if str(label) == "oracle_ceiling" else "-"
        color = _baseline_color(label) if str(label) in _BASELINE_COLORS else cycle[idx % len(cycle)]
        ax.errorbar(x_vals, y_vals, yerr=err_vals, marker="o", lw=1.4, capsize=2, label=_baseline_label(label), color=color, linestyle=linestyle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for x, label in vlines or ():
        ax.axvline(float(x), color="#991b1b", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.text(float(x), 0.98, str(label), transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=7, color="#991b1b")
    for y, label in hlines or ():
        ax.axhline(float(y), color="#374151", linestyle=":", linewidth=1.0, alpha=0.85)
        ax.text(0.99, float(y), str(label), transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=7, color="#374151")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_figure(path, {"series": series, "title": title, "xlabel": xlabel, "ylabel": ylabel, "vlines": list(vlines or ()), "hlines": list(hlines or ())})


def _plot_heatmap(path: Path, title: str, matrix: Sequence[Sequence[float]], x_labels: Sequence[str], y_labels: Sequence[str], cmap: str = "Blues") -> None:
    if not matrix:
        return
    path = _flat_artifact_path(path)
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return
    _ensure(path.parent)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * len(x_labels) + 2), max(4.0, 0.35 * len(y_labels) + 2)), dpi=140)
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(list(x_labels), rotation=35, ha="right")
    ax.set_yticklabels(list(y_labels))
    ax.set_title(title)
    vmax = float(np.nanmax(arr)) if arr.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if vmax and val > vmax / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    fig_data = {"matrix": arr.tolist(), "x_labels": list(x_labels), "y_labels": list(y_labels), "title": title}
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_figure(path, fig_data)


def _plot_grouped_bar(
    path: Path,
    title: str,
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    ylabel: str = "value",
) -> None:
    if not groups or not series:
        return
    path = _flat_artifact_path(path)
    _ensure(path.parent)
    labels = list(series.keys())
    x = np.arange(len(groups), dtype=float)
    width = min(0.8 / max(1, len(labels)), 0.18)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.75 * len(groups) + 2), 3.8), dpi=140)
    offset0 = -0.5 * width * (len(labels) - 1)
    for i, label in enumerate(labels):
        vals = [float(v) for v in series[label]]
        ax.bar(x + offset0 + i * width, vals, width=width, label=_baseline_label(label), color=_baseline_color(label))
    ax.set_xticks(x)
    ax.set_xticklabels(list(groups), rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_figure(path, {"groups": list(groups), "series": {k: list(v) for k, v in series.items()}, "title": title, "ylabel": ylabel})


def _plot_two_heatmaps(
    path: Path,
    title: str,
    left_title: str,
    left_matrix: Sequence[Sequence[Optional[float]]],
    right_title: str,
    right_matrix: Sequence[Sequence[Optional[float]]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    mask_label: str = "train",
) -> None:
    if not left_matrix or not right_matrix:
        return
    path = _flat_artifact_path(path)
    _ensure(path.parent)
    arrays = [
        np.asarray([[np.nan if v is None else float(v) for v in row] for row in left_matrix], dtype=float),
        np.asarray([[np.nan if v is None else float(v) for v in row] for row in right_matrix], dtype=float),
    ]
    cmap_obj = plt.get_cmap("Blues").copy()
    cmap_obj.set_bad("#d1d5db")
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 0.8 * len(x_labels) + 4), max(4.0, 0.35 * len(y_labels) + 2)), dpi=140, sharey=True)
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if arr.size and np.isfinite(arr).any()])
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    for ax, arr, panel_title in zip(axes, arrays, (left_title, right_title)):
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, cmap=cmap_obj, aspect="auto", vmin=0.0, vmax=max(1e-9, vmax))
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(list(x_labels), rotation=35, ha="right")
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(list(y_labels))
        ax.set_title(panel_title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if not np.isfinite(val):
                    text, color = mask_label, "#374151"
                else:
                    text = f"{val:.2f}"
                    color = "white" if vmax and val > vmax / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.04)
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.12)
    fig.savefig(path)
    plt.close(fig)
    if _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.capture_figure(path, {"title": title, "left_title": left_title, "right_title": right_title, "x_labels": list(x_labels), "y_labels": list(y_labels)})


def _plot_heatmap_masked(
    path: Path,
    title: str,
    matrix: Sequence[Sequence[Optional[float]]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    mask_label: str = "train",
    cmap: str = "Blues",
) -> None:
    if not matrix:
        return
    path = _flat_artifact_path(path)
    arr = np.asarray([[np.nan if v is None else float(v) for v in row] for row in matrix], dtype=float)
    if arr.size == 0:
        return
    _ensure(path.parent)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#d1d5db")
    masked = np.ma.masked_invalid(arr)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * len(x_labels) + 2), max(4.0, 0.35 * len(y_labels) + 2)), dpi=140)
    im = ax.imshow(masked, cmap=cmap_obj, aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(list(x_labels), rotation=35, ha="right")
    ax.set_yticklabels(list(y_labels))
    ax.set_title(title)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.nanmax(finite)) if finite.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not np.isfinite(val):
                text, color = mask_label, "#374151"
            else:
                text = f"{val:.2f}"
                color = "white" if vmax and val > vmax / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    if _ACTIVE_RESULT is not None:
        matrix_json = [[None if not np.isfinite(v) else float(v) for v in row] for row in arr.tolist()]
        _ACTIVE_RESULT.capture_figure(path, {"matrix": matrix_json, "x_labels": list(x_labels), "y_labels": list(y_labels), "title": title, "mask_label": mask_label})


def _nested_get(obj: Mapping[str, Any], path: Sequence[Any], default: Any = None) -> Any:
    cur: Any = obj
    for part in path:
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _completed_seed_results(seed_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in seed_results if r.get("status", {}).get("state") == "completed"]


def _baseline_names_from_seed_results(seed_results: Sequence[Dict[str, Any]]) -> List[str]:
    names = sorted({b for r in _completed_seed_results(seed_results) for b in (r.get("metrics", {}).get("per_baseline", {}) or {})})
    return _ordered_baseline_names(names)


def _aggregate_seed_path_values(seed_results: Sequence[Dict[str, Any]], path: Sequence[Any]) -> Tuple[float, float, List[float]]:
    vals: List[float] = []
    for result in _completed_seed_results(seed_results):
        value = _nested_get(result.get("metrics", {}) or {}, path)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(float(value))
    return _mean(vals), _stderr95(vals), vals


def _aggregate_baseline_metric(seed_results: Sequence[Dict[str, Any]], metric_path: Sequence[Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    means: Dict[str, float] = {}
    ci: Dict[str, float] = {}
    for baseline in _baseline_names_from_seed_results(seed_results):
        mean, err, vals = _aggregate_seed_path_values(seed_results, ("per_baseline", baseline, *metric_path))
        if vals:
            means[baseline] = mean
            ci[baseline] = err
    return means, ci


def _curve_series(curve: Mapping[str, Mapping[str, Any]], metric: str) -> Tuple[List[float], List[float]]:
    points: List[Tuple[float, float]] = []
    for key, row in curve.items():
        try:
            x = float(key)
            y = float(row.get(metric, 0.0))
        except (TypeError, ValueError, AttributeError):
            continue
        points.append((x, y))
    points.sort(key=lambda p: p[0])
    return [p[0] for p in points], [p[1] for p in points]


def _render_seed_publication_figures(seed_dir: Path, experiment: str, metrics: Mapping[str, Any]) -> None:
    """Regenerate the F-series seed figures from stored metrics only."""
    out = Path(seed_dir)
    if experiment in {"deployment_stream", "cl_regularizer_comparison"}:
        _remove_stale_pngs(out, ("F1_main_results_*.png", "F2_four_cell_heatmap.png"))
        per_baseline = dict(metrics.get("per_baseline", {}) or {})
        _plot_bar(out / "figures/F1a_live_top1.png", "F1a deployment stream live top-1", _label_to_metric_rows(per_baseline, "live_top1"), ylabel="top-1")
        _plot_bar(out / "figures/F1b_assistance_coverage.png", "F1b assistance coverage", _label_to_metric_rows(per_baseline, "assistance_coverage"), ylabel="coverage")
        _plot_bar(out / "figures/F1c_conditional_top1.png", "F1c conditional top-1 when assisting", _label_to_metric_rows(per_baseline, "conditional_top1"), ylabel="conditional top-1")
        _plot_bar(out / "figures/F1d_net_assistance_value.png", "F1d net assistance value", _label_to_metric_rows(per_baseline, "net_assistance_value"), ylabel="correct assists - wrong assists")
        _plot_bar(out / "figures/F1e_useful_assistance_secondary.png", "F1e useful assistance (coverage x conditional top-1)", _label_to_metric_rows(per_baseline, "useful_assistance_rate"), ylabel="rate")
        _plot_bar(out / "figures/F1f_recovery_latency.png", "F1f recovery latency after first mismatch", _label_to_metric_rows(per_baseline, "adaptation_latency_steps"), ylabel="steps")
        _plot_bar(out / "figures/F1g_p95_prediction_latency.png", "F1g p95 prediction latency", _label_to_metric_rows(per_baseline, "p95_prediction_wall_s"), ylabel="seconds")
        heatmap_baselines = [b for b in ("full", "no_preference_prototype", "no_recipe_prototype", "latest_only", "experience_replay_bc", "bigram", "oracle_ceiling") if b in per_baseline]
        grouped_bar_baselines = _ordered_baseline_names(per_baseline)
        if heatmap_baselines:
            matrix = [
                [per_baseline[b].get("four_cell", {}).get(cell, {}).get("live_top1", 0.0) for cell in FOUR_CELL_KEYS]
                for b in heatmap_baselines
            ]
            _plot_heatmap(out / "figures/F2_four_cell_heatmap.png", "F2 four-cell accuracy by baseline", matrix, FOUR_CELL_KEYS, heatmap_baselines)
        if grouped_bar_baselines:
            _plot_grouped_bar(
                out / "figures/F2_four_cell_grouped_bar.png",
                "F2 four-cell accuracy by baseline",
                FOUR_CELL_KEYS,
                {b: [per_baseline[b].get("four_cell", {}).get(cell, {}).get("live_top1", 0.0) for cell in FOUR_CELL_KEYS] for b in grouped_bar_baselines},
                ylabel="top-1",
            )
        forgetting_series: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {}
        for name in ("full", "no_decay", "latest_only", "fixed_decay"):
            ft = [r for r in metrics.get("derived_views", {}).get("forgetting_curve", []) if r.get("baseline") == name]
            if ft:
                forgetting_series[name] = ([r["phase_b_count"] for r in ft], [r["mean_phase_a_top1"] for r in ft])
        if forgetting_series:
            _plot_lines(out / "figures/F3_forgetting_curve.png", "F3 retained Phase-A accuracy", forgetting_series, "Phase-B events", "frozen top-1")
        trace_series: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {}
        trace_baselines = [b for b in ("full", "experience_replay_bc", "bigram", "latest_only") if b in per_baseline]
        for baseline in trace_baselines:
            row = per_baseline[baseline]
            trace = row.get("live_step_trace", [])
            if trace:
                trace_series[baseline] = ([r["step"] for r in trace], [r["top1"] for r in trace])
        if trace_series:
            _plot_lines(out / "figures/F4_per_step_adaptation.png", "F4 adaptation within session", trace_series, "step", "top-1")
        curve = (metrics.get("derived_views", {}).get("coverage_accuracy_curve", {}) or {}).get("full", {})
        if curve:
            xs_cov, ys_cov = _curve_series(curve, "coverage")
            xs_acc, ys_acc = _curve_series(curve, "conditional_top1")
            selected = float(metrics.get("selected_action_confidence_threshold", DEFAULT_CONFIG.posterior_action_confidence_threshold))
            _plot_lines(
                out / "figures/F5_coverage_accuracy.png",
                "F5 coverage vs accuracy",
                {"coverage": (xs_cov, ys_cov), "conditional_top1": (xs_acc, ys_acc)},
                "threshold",
                "rate",
                vlines=((selected, "selected"),),
            )
        _plot_bar(out / "figures/S1_memory_active_variants.png", "S1 active memory footprint", {b: float(row.get("memory", {}).get("active_variants", 0.0)) for b, row in per_baseline.items()}, ylabel="active variants")
        _plot_bar(out / "figures/S1_compute_wall_time.png", "S1 wall time by baseline", {b: float(row.get("compute", {}).get("wall_s", 0.0)) for b, row in per_baseline.items()}, ylabel="seconds")
    elif experiment in {"cross_recipe_transfer", "posterior_ablation_matrix"}:
        recipes = list(metrics.get("recipes", []) or [])
        prefs = list(metrics.get("preferences", []) or [])
        heat = dict(metrics.get("derived_views", {}).get("transfer_heatmap", {}) or {})
        if recipes and prefs and heat:
            train_cells = set()
            for label in metrics.get("diagonal_training_pairs", []) or []:
                if "/" in str(label):
                    recipe, pref = str(label).split("/", 1)
                    train_cells.add((recipe, pref))
            matrix = [
                [None if (recipe, pref) in train_cells else float(heat.get(f"{recipe}/{pref}", 0.0)) for pref in prefs]
                for recipe in recipes
            ]
            _plot_heatmap_masked(out / "figures/F6_transfer_heatmap.png", "F6 full transfer heatmap", matrix, prefs, recipes, mask_label="train")
        heat_by_baseline = dict(metrics.get("derived_views", {}).get("transfer_heatmap_by_baseline", {}) or {})
        compare_matrices: Dict[str, List[List[Optional[float]]]] = {}
        for baseline in ("full", "latest_only"):
            matrix_by_recipe = heat_by_baseline.get(baseline, {})
            if recipes and prefs and matrix_by_recipe:
                matrix = [
                    [matrix_by_recipe.get(recipe, {}).get(pref, None) for pref in prefs]
                    for recipe in recipes
                ]
                compare_matrices[baseline] = matrix
                _plot_heatmap_masked(out / f"figures/F6_{baseline}_transfer_heatmap.png", f"F6 {baseline} transfer heatmap", matrix, prefs, recipes, mask_label="train")
        if "full" in compare_matrices and "latest_only" in compare_matrices:
            _plot_two_heatmaps(
                out / "figures/F6_full_vs_latest_transfer_heatmap.png",
                "F6 transfer heatmap: full vs latest-only",
                "Full",
                compare_matrices["full"],
                "Latest only",
                compare_matrices["latest_only"],
                prefs,
                recipes,
                mask_label="train",
            )
        per_baseline = dict(metrics.get("per_baseline", {}) or {})
        _plot_bar(out / "figures/S2_offdiag_live_top1_by_baseline.png", "S2 off-diagonal live top-1", _label_to_metric_rows(per_baseline, "live_top1"), ylabel="top-1")
        _plot_bar(out / "figures/S2_primary_seen_seen_transfer_top1.png", "S2 primary seen-seen transfer top-1", _label_to_metric_rows(per_baseline, "primary_transfer_live_top1"), ylabel="top-1")
        _plot_bar(out / "figures/S2_preference_cluster_purity.png", "S2 preference cluster purity", _label_to_metric_rows(per_baseline, "preference_cluster_purity"), ylabel="purity")
        _plot_bar(out / "figures/S2_offdiag_conditional_top1_by_baseline.png", "S2 off-diagonal conditional top-1", _label_to_metric_rows(per_baseline, "conditional_top1"), ylabel="conditional top-1")
        cycle_series: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {}
        for baseline, row in per_baseline.items():
            curve = row.get("diagonal_cycle_curve", {}) or {}
            if curve:
                xs = sorted(int(k) for k in curve)
                cycle_series[baseline] = (xs, [curve[str(x)].get("frozen_diagonal_top1", 0.0) for x in xs])
        if cycle_series:
            _plot_lines(out / "figures/S2_single_shot_sufficiency_curve.png", "S2 single-shot sufficiency curve", cycle_series, "diagonal cycles", "frozen diagonal top-1")
        comp = (metrics.get("per_baseline", {}).get("full", {}) or {}).get("novel_composition", {})
        if comp:
            xs = sorted(int(k) for k in comp)
            _plot_lines(out / "figures/F7_composition_curve.png", "F7 novel composition by hamming distance", {"full": (xs, [comp[str(x)].get("live_top1", 0.0) for x in xs])}, "hamming distance", "live top-1")
    elif experiment == "decay_reentry":
        f8_series: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {}
        f8_active_series: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {}
        f9_series: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {}
        crossing_vlines: List[Tuple[float, str]] = []
        for baseline, row in (metrics.get("per_baseline", {}) or {}).items():
            if baseline not in ("full", "adaptive", "fixed_decay", "no_decay", "experience_replay_bc", "bigram"):
                continue
            for arm_name, arm_row in (row.get("arms", {}) or {}).items():
                per_gap = arm_row.get("per_gap", {}) or {}
                gap_keys = sorted(per_gap, key=lambda g: int(g))
                xs = [int(g) for g in gap_keys]
                if xs:
                    f8_series[f"{baseline}_{arm_name}"] = (xs, [per_gap[str(g)]["live_top1"] for g in xs])
                    f8_active_series[f"{baseline}_{arm_name}_active_before"] = (xs, [per_gap[str(g)].get("active_before_rate", 0.0) for g in xs])
                    if baseline in ("full", "adaptive"):
                        f9_series[f"{baseline}_{arm_name}"] = (xs, [per_gap[str(g)]["global_rate_after"] for g in xs])
                    if baseline == "full":
                        active_vals = [float(per_gap[str(g)].get("active_before_rate", 0.0)) for g in xs]
                        below = [x for x, y in zip(xs, active_vals) if y <= 0.5]
                        if below:
                            crossing_vlines.append((float(below[0]), f"{arm_name} active<=0.5"))
        if f8_series:
            _plot_lines(out / "figures/F8_decay_reentry_curve.png", "F8 decay reentry by gap", f8_series, "gap", "live top-1", vlines=crossing_vlines, hlines=((0.5, "0.5"),))
        if f8_active_series:
            _plot_lines(out / "figures/F8_active_before_curve.png", "F8 active-before by gap", f8_active_series, "gap", "active-before rate", vlines=crossing_vlines, hlines=((0.5, "pruning midpoint"),))
        if f9_series:
            _plot_lines(out / "figures/F9_decay_rate_adaptation.png", "F9 decay rate adaptation", f9_series, "gap", "global rate", hlines=((1.0 / max(1, int(DEFAULT_CONFIG.decay_horizon_init)), "initial horizon rate"),))
        for arm_name in ("neutral", "distractor"):
            arm_live = {k.replace(f"_{arm_name}", ""): v for k, v in f8_series.items() if k.endswith(f"_{arm_name}")}
            arm_active = {k.replace(f"_{arm_name}_active_before", ""): v for k, v in f8_active_series.items() if k.endswith(f"_{arm_name}_active_before")}
            arm_rate = {k.replace(f"_{arm_name}", ""): v for k, v in f9_series.items() if k.endswith(f"_{arm_name}")}
            if arm_live:
                _plot_lines(out / f"figures/F8_{arm_name}_decay_reentry_curve.png", f"F8 {arm_name} decay reentry by gap", arm_live, "gap", "live top-1", hlines=((0.5, "0.5"),))
            if arm_active:
                _plot_lines(out / f"figures/F8_{arm_name}_active_before_curve.png", f"F8 {arm_name} active-before by gap", arm_active, "gap", "active-before rate", hlines=((0.5, "pruning midpoint"),))
            if arm_rate:
                _plot_lines(out / f"figures/F9_{arm_name}_decay_rate_adaptation.png", f"F9 {arm_name} decay rate adaptation", arm_rate, "gap", "global rate", hlines=((1.0 / max(1, int(DEFAULT_CONFIG.decay_horizon_init)), "initial horizon rate"),))
    elif experiment == "disambiguation_audit":
        per_threshold = metrics.get("per_threshold", {}) or {}
        if per_threshold:
            xs = sorted(float(t) for t in per_threshold)
            _plot_lines(out / "figures/F10_disambiguation_macro_f1.png", "F10 disambiguation macro-F1", {"macro_f1": (xs, [per_threshold[str(t)]["macro_f1"] for t in xs])}, "jaccard threshold", "macro-F1", vlines=((0.95, "production"),))
            gate_f1 = [per_threshold[str(t)].get("gate_decision", {}).get("macro_f1", 0.0) for t in xs]
            _plot_lines(out / "figures/F10_gate_decision_macro_f1.png", "F10 gate decision macro-F1", {"gate_macro_f1": (xs, gate_f1)}, "jaccard threshold", "macro-F1", vlines=((0.95, "production"),))
            best = str(metrics.get("best_threshold") or xs[0])
            rows = per_threshold.get(best, {}).get("rows", [])
            labels = MEMORY_STATE_LABELS
            if rows:
                matrix = [[float(sum(1 for r in rows if r.get("gt") == gt and r.get("pred") == pred)) for pred in labels] for gt in labels]
                _plot_heatmap(out / "figures/S4_confusion_matrix_best_threshold.png", f"S4 confusion matrix at threshold={best}", matrix, labels, labels, cmap="Purples")
        gate_curve = metrics.get("action_gate_curve", {}) or {}
        if gate_curve:
            xs_cov, ys_cov = _curve_series(gate_curve, "coverage")
            xs_acc, ys_acc = _curve_series(gate_curve, "conditional_top1")
            _plot_lines(out / "figures/S4_action_gate_coverage_accuracy.png", "S4 action gate coverage/accuracy", {"coverage": (xs_cov, ys_cov), "conditional_top1": (xs_acc, ys_acc)}, "threshold", "rate")
        boundary = metrics.get("boundary_degradation", {}) or {}
        if boundary:
            xs = sorted(float(k) for k in boundary)
            _plot_lines(out / "figures/S4_boundary_degradation.png", "S4 online-prefix boundary degradation", {"macro_f1": (xs, [boundary[str(x)]["macro_f1"] for x in xs]), "accuracy": (xs, [boundary[str(x)]["accuracy"] for x in xs])}, "drop fraction", "score")
        axis_deg = metrics.get("preference_axis_degradation", {}) or {}
        if axis_deg:
            xs = sorted(int(k) for k in axis_deg)
            _plot_lines(out / "figures/S4_preference_axis_degradation.png", "S4 preference-axis materiality", {"jaccard": (xs, [axis_deg[str(x)]["mean_jaccard_from_identity"] for x in xs]), "kendall": (xs, [axis_deg[str(x)]["mean_kendall_from_identity"] for x in xs])}, "changed axes", "distance")
    elif experiment == "demo_count_sample_efficiency":
        _remove_stale_pngs(out, ("*_capacity.png", "n_*_live_top1.png", "n_*_frozen_top1.png"))
        rows = metrics.get("per_demo_count", {}) or {}
        baselines = sorted({b for row in rows.values() for b in (row.get("per_baseline", {}) or {})})
        demo_counts = sorted((int(n) for n in rows), key=int)
        series = {
            baseline: (
                demo_counts,
                [rows[str(n)].get("per_baseline", {}).get(baseline, {}).get("live_top1", 0.0) for n in demo_counts],
            )
            for baseline in baselines
        }
        _plot_lines(out / "figures/sample_efficiency_live_top1.png", "sample efficiency by baseline", series, "demo count", "live top-1")
    elif experiment == "zipf_usage_sweep":
        per_alpha = metrics.get("per_alpha", {}) or {}
        alpha_values = sorted(float(alpha) for alpha in per_alpha)
        baselines = sorted({baseline for row in per_alpha.values() for baseline in row})
        for metric, ylabel in (
            ("utility_top1", "utility top-1"),
            ("fairness_top1", "fairness top-1"),
            ("head_tail_gap", "head-tail gap"),
        ):
            series = {
                baseline: (
                    alpha_values,
                    [per_alpha[str(alpha)].get(baseline, {}).get(metric, 0.0) for alpha in alpha_values],
                )
                for baseline in baselines
            }
            _plot_lines(out / f"figures/zipf_{metric}.png", f"Zipf sweep {metric}", series, "alpha", ylabel)
        for metric, ylabel in (
            ("active_variants", "active variants"),
            ("pruned_variants", "pruned variants"),
        ):
            series = {
                baseline: (
                    alpha_values,
                    [per_alpha[str(alpha)].get(baseline, {}).get("memory", {}).get(metric, 0.0) for alpha in alpha_values],
                )
                for baseline in baselines
            }
            _plot_lines(out / f"figures/zipf_memory_{metric}.png", f"Zipf sweep memory {metric}", series, "alpha", ylabel)


def _line_aggregate_from_traces(
    seed_results: Sequence[Dict[str, Any]],
    trace_path: Sequence[Any],
    x_key: str,
    y_key: str,
    label_filter: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    grouped: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for result in _completed_seed_results(seed_results):
        rows = _nested_get(result.get("metrics", {}) or {}, trace_path, []) or []
        for row in rows:
            if not isinstance(row, Mapping) or x_key not in row or y_key not in row:
                continue
            label = label_filter(dict(row)) if label_filter else "mean"
            grouped[str(label)][float(row[x_key])].append(float(row[y_key]))
    series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for label, by_x in grouped.items():
        xs = sorted(by_x)
        series[label] = (xs, [_mean(by_x[x]) for x in xs], [_stderr95(by_x[x]) for x in xs])
    return series


def _aggregate_curve_by_threshold(
    seed_results: Sequence[Dict[str, Any]],
    curve_path: Sequence[Any],
    metric: str,
) -> Tuple[List[float], List[float], List[float]]:
    by_thr: Dict[float, List[float]] = defaultdict(list)
    for result in _completed_seed_results(seed_results):
        curve = _nested_get(result.get("metrics", {}) or {}, curve_path, {}) or {}
        for key, row in curve.items():
            if isinstance(row, Mapping) and metric in row:
                by_thr[float(key)].append(float(row[metric]))
    xs = sorted(by_thr)
    return xs, [_mean(by_thr[x]) for x in xs], [_stderr95(by_thr[x]) for x in xs]


def _render_deployment_aggregate_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]], prefix: str = "aggregate") -> None:
    aggregate_dir = _ensure(exp_dir / "aggregate")
    for metric, ylabel, title in (
        ("live_top1", "top-1", "F1a aggregate live top-1"),
        ("assistance_coverage", "coverage", "F1b aggregate assistance coverage"),
        ("conditional_top1", "conditional top-1", "F1c aggregate conditional top-1"),
        ("net_assistance_value", "correct assists - wrong assists", "F1d aggregate net assistance value"),
        ("useful_assistance_rate", "rate", "F1e aggregate useful assistance secondary"),
        ("adaptation_latency_steps", "steps", "F1f aggregate recovery latency"),
        ("p95_prediction_wall_s", "seconds", "F1g aggregate p95 prediction latency"),
    ):
        means, ci = _aggregate_baseline_metric(seed_results, (metric,))
        _plot_bar_ci(aggregate_dir / f"{prefix}_{metric}.png", title, means, ci, ylabel=ylabel)
    for metric, ylabel, title in (
        ("active_variants", "active variants", "Aggregate active memory footprint"),
        ("pruned_variants", "pruned variants", "Aggregate pruned memory footprint"),
        ("latest_keys", "latest keys", "Aggregate latest-key memory footprint"),
    ):
        means, ci = _aggregate_baseline_metric(seed_results, ("memory", metric))
        _plot_bar_ci(aggregate_dir / f"{prefix}_memory_{metric}.png", title, means, ci, ylabel=ylabel)
    for metric, ylabel, title in (
        ("wall_s", "seconds", "Aggregate wall time"),
        ("n_fits_executed", "fit count", "Aggregate fit count"),
        ("last_retrain_fit_wall_s", "seconds", "Aggregate last retrain fit time"),
        ("mean_retrain_fit_wall_s", "seconds", "Aggregate mean retrain fit time"),
        ("p95_retrain_fit_wall_s", "seconds", "Aggregate p95 retrain fit time"),
        ("total_retrain_fit_wall_s", "seconds", "Aggregate total retrain fit time"),
        ("estimated_flops", "estimated FLOPs", "Aggregate estimated training FLOPs"),
    ):
        means, ci = _aggregate_baseline_metric(seed_results, ("compute", metric))
        _plot_bar_ci(aggregate_dir / f"{prefix}_compute_{metric}.png", title, means, ci, ylabel=ylabel)
    # Four-cell deployment report across available baselines.
    heatmap_baselines = [b for b in ("full", "no_preference_prototype", "no_recipe_prototype", "latest_only", "experience_replay_bc", "bigram", "oracle_ceiling") if b in _baseline_names_from_seed_results(seed_results)]
    grouped_bar_baselines = _baseline_names_from_seed_results(seed_results)
    matrix: List[List[float]] = []
    for baseline in heatmap_baselines:
        row_vals: List[float] = []
        for cell in FOUR_CELL_KEYS:
            vals = [
                float(_nested_get(result.get("metrics", {}) or {}, ("per_baseline", baseline, "four_cell", cell, "live_top1"), 0.0))
                for result in _completed_seed_results(seed_results)
            ]
            row_vals.append(_mean(vals))
        matrix.append(row_vals)
    if matrix:
        _plot_heatmap(
            aggregate_dir / f"{prefix}_F2_four_cell_heatmap.png",
            "F2 aggregate four-cell accuracy by baseline",
            matrix,
            FOUR_CELL_KEYS,
            heatmap_baselines,
        )
    grouped_matrix: List[List[float]] = []
    for baseline in grouped_bar_baselines:
        grouped_matrix.append([
            _mean([
                float(_nested_get(result.get("metrics", {}) or {}, ("per_baseline", baseline, "four_cell", cell, "live_top1"), 0.0))
                for result in _completed_seed_results(seed_results)
            ])
            for cell in FOUR_CELL_KEYS
        ])
    if grouped_matrix:
        _plot_grouped_bar(
            aggregate_dir / f"{prefix}_F2_four_cell_grouped_bar.png",
            "F2 aggregate four-cell accuracy by baseline",
            FOUR_CELL_KEYS,
            {baseline: row for baseline, row in zip(grouped_bar_baselines, grouped_matrix)},
            ylabel="top-1",
        )
    cell_vals: Dict[str, List[float]] = defaultdict(list)
    for result in _completed_seed_results(seed_results):
        full = _nested_get(result.get("metrics", {}) or {}, ("per_baseline", "full", "four_cell"), {}) or {}
        for cell in DEPLOYMENT_REPORT_CELLS:
            cell_vals[cell].append(float((full.get(cell) or {}).get("live_top1", 0.0)))
    _plot_bar_ci(
        aggregate_dir / f"{prefix}_F2_seen_recipe_preference.png",
        "F2 aggregate seen recipe: same vs new preference",
        {cell: _mean(vals) for cell, vals in cell_vals.items()},
        {cell: _stderr95(vals) for cell, vals in cell_vals.items()},
        ylabel="top-1",
    )
    f3 = _line_aggregate_from_traces(
        seed_results,
        ("derived_views", "forgetting_curve"),
        "phase_b_count",
        "mean_phase_a_top1",
        label_filter=lambda row: str(row.get("baseline", "unknown")),
    )
    if "full" in f3:
        keep = {k: v for k, v in f3.items() if k in ("full", "no_decay", "latest_only", "fixed_decay")}
        _plot_lines_ci(aggregate_dir / f"{prefix}_F3_forgetting_curve.png", "F3 aggregate retained Phase-A accuracy", keep, "Phase-B events", "frozen top-1")
    trace_series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for baseline in ("full", "experience_replay_bc", "bigram", "latest_only"):
        by_step: Dict[float, List[float]] = defaultdict(list)
        for result in _completed_seed_results(seed_results):
            trace = _nested_get(result.get("metrics", {}) or {}, ("per_baseline", baseline, "live_step_trace"), []) or []
            for row in trace:
                by_step[float(row["step"])].append(float(row["top1"]))
        if by_step:
            xs = sorted(by_step)
            trace_series[baseline] = (xs, [_mean(by_step[x]) for x in xs], [_stderr95(by_step[x]) for x in xs])
    _plot_lines_ci(aggregate_dir / f"{prefix}_F4_per_step_adaptation.png", "F4 aggregate adaptation within session", trace_series, "step", "top-1")
    cov = _aggregate_curve_by_threshold(seed_results, ("derived_views", "coverage_accuracy_curve", "full"), "coverage")
    acc = _aggregate_curve_by_threshold(seed_results, ("derived_views", "coverage_accuracy_curve", "full"), "conditional_top1")
    if cov[0] and acc[0]:
        selected_vals = [
            float(_nested_get(result.get("metrics", {}) or {}, ("selected_action_confidence_threshold",), DEFAULT_CONFIG.posterior_action_confidence_threshold))
            for result in _completed_seed_results(seed_results)
        ]
        selected = _mean(selected_vals) if selected_vals else float(DEFAULT_CONFIG.posterior_action_confidence_threshold)
        _plot_lines_ci(
            aggregate_dir / f"{prefix}_F5_coverage_accuracy.png",
            "F5 aggregate coverage vs accuracy",
            {"coverage": cov, "conditional_top1": acc},
            "threshold",
            "rate",
            vlines=((selected, "selected"),),
        )
    for group_name, path in (("per_user", ("per_user",)), ("per_event_type", ("per_event_type",))):
        labels = sorted({label for result in _completed_seed_results(seed_results) for label in (_nested_get(result.get("metrics", {}) or {}, ("per_baseline", "full", *path), {}) or {})})
        values: Dict[str, float] = {}
        ci: Dict[str, float] = {}
        for label in labels:
            vals = [
                float(_nested_get(result.get("metrics", {}) or {}, ("per_baseline", "full", *path, label, "live_top1"), 0.0))
                for result in _completed_seed_results(seed_results)
            ]
            values[label], ci[label] = _mean(vals), _stderr95(vals)
        _plot_bar_ci(aggregate_dir / f"{prefix}_{group_name}_live_top1.png", f"Aggregate full-model {group_name} live top-1", values, ci, ylabel="top-1")


def _render_transfer_aggregate_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]], prefix: str = "aggregate") -> None:
    aggregate_dir = _ensure(exp_dir / "aggregate")
    for metric, ylabel, title in (
        ("live_top1", "top-1", "S2 aggregate off-diagonal live top-1"),
        ("primary_transfer_live_top1", "top-1", "S2 aggregate primary seen-seen transfer top-1"),
        ("zero_shot_preference_live_top1", "top-1", "S2 aggregate zero-shot preference top-1"),
        ("assistance_coverage", "coverage", "S2 aggregate off-diagonal assistance coverage"),
        ("conditional_top1", "conditional top-1", "S2 aggregate off-diagonal conditional top-1"),
        ("preference_transfer_rate", "rate", "S2 aggregate preference transfer rate"),
        ("preference_cluster_purity", "purity", "S2 aggregate preference cluster purity"),
        ("confidence_ece", "ECE", "S2 aggregate confidence ECE"),
        ("confidence_brier", "Brier", "S2 aggregate confidence Brier"),
    ):
        means, ci = _aggregate_baseline_metric(seed_results, (metric,))
        _plot_bar_ci(aggregate_dir / f"{prefix}_{metric}.png", title, means, ci, ylabel=ylabel)
    comp_series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for baseline in ("full", "latest_only", "bigram", "no_preference_prototype", "no_state_graph"):
        by_h: Dict[float, List[float]] = defaultdict(list)
        for result in _completed_seed_results(seed_results):
            comp = _nested_get(result.get("metrics", {}) or {}, ("per_baseline", baseline, "novel_composition"), {}) or {}
            for h, row in comp.items():
                by_h[float(h)].append(float(row.get("live_top1", 0.0)))
        if by_h:
            xs = sorted(by_h)
            comp_series[baseline] = (xs, [_mean(by_h[x]) for x in xs], [_stderr95(by_h[x]) for x in xs])
    _plot_lines_ci(aggregate_dir / f"{prefix}_F7_composition_curve_by_baseline.png", "F7 aggregate composition by baseline", comp_series, "hamming distance", "live top-1")
    cycle_series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for baseline in _baseline_names_from_seed_results(seed_results):
        by_cycle: Dict[float, List[float]] = defaultdict(list)
        for result in _completed_seed_results(seed_results):
            curve = _nested_get(result.get("metrics", {}) or {}, ("per_baseline", baseline, "diagonal_cycle_curve"), {}) or {}
            for cycle, row in curve.items():
                by_cycle[float(cycle)].append(float(row.get("frozen_diagonal_top1", 0.0)))
        if by_cycle:
            xs = sorted(by_cycle)
            cycle_series[baseline] = (xs, [_mean(by_cycle[x]) for x in xs], [_stderr95(by_cycle[x]) for x in xs])
    _plot_lines_ci(aggregate_dir / f"{prefix}_S2_single_shot_sufficiency_curve.png", "S2 aggregate single-shot sufficiency curve", cycle_series, "diagonal cycles", "frozen diagonal top-1")


def _render_decay_aggregate_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]], prefix: str = "aggregate") -> None:
    aggregate_dir = _ensure(exp_dir / "aggregate")
    for metric, ylabel, title in (
        ("live_top1", "top-1", "S3 aggregate reentry live top-1"),
        ("active_before_rate", "rate", "S3 aggregate active-before rate"),
        ("post_reentry_top1", "top-1", "S3 aggregate post-reentry top-1"),
    ):
        means, ci = _aggregate_baseline_metric(seed_results, (metric,))
        _plot_bar_ci(aggregate_dir / f"{prefix}_{metric}.png", title, means, ci, ylabel=ylabel)
    for arm in ("neutral", "distractor"):
        series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
        active_series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
        crossing_vlines: List[Tuple[float, str]] = []
        for baseline in ("full", "adaptive", "fixed_decay", "no_decay", "experience_replay_bc", "bigram"):
            by_gap: Dict[float, List[float]] = defaultdict(list)
            by_active: Dict[float, List[float]] = defaultdict(list)
            for result in _completed_seed_results(seed_results):
                per_gap = _nested_get(result.get("metrics", {}) or {}, ("per_baseline", baseline, "arms", arm, "per_gap"), {}) or {}
                for gap, row in per_gap.items():
                    by_gap[float(gap)].append(float(row.get("live_top1", 0.0)))
                    by_active[float(gap)].append(float(row.get("active_before_rate", 0.0)))
            if by_gap:
                xs = sorted(by_gap)
                series[baseline] = (xs, [_mean(by_gap[x]) for x in xs], [_stderr95(by_gap[x]) for x in xs])
                if baseline == "full":
                    active_vals = [_mean(by_active[x]) for x in xs]
                    below = [x for x, y in zip(xs, active_vals) if y <= 0.5]
                    if below:
                        crossing_vlines.append((float(below[0]), "active<=0.5"))
            if by_active:
                xs = sorted(by_active)
                active_series[baseline] = (xs, [_mean(by_active[x]) for x in xs], [_stderr95(by_active[x]) for x in xs])
        _plot_lines_ci(aggregate_dir / f"{prefix}_F8_{arm}_live_top1_by_gap.png", f"F8 aggregate {arm} reentry by gap", series, "gap", "live top-1", vlines=crossing_vlines, hlines=((0.5, "0.5"),))
        _plot_lines_ci(aggregate_dir / f"{prefix}_F8_{arm}_active_before_by_gap.png", f"F8 aggregate {arm} active-before by gap", active_series, "gap", "active-before rate", vlines=crossing_vlines, hlines=((0.5, "pruning midpoint"),))
    rate_series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for baseline in ("full", "adaptive"):
        for arm in ("neutral", "distractor"):
            by_gap: Dict[float, List[float]] = defaultdict(list)
            for result in _completed_seed_results(seed_results):
                per_gap = _nested_get(result.get("metrics", {}) or {}, ("per_baseline", baseline, "arms", arm, "per_gap"), {}) or {}
                for gap, row in per_gap.items():
                    by_gap[float(gap)].append(float(row.get("global_rate_after", 0.0)))
            if by_gap:
                xs = sorted(by_gap)
                rate_series[f"{baseline}_{arm}"] = (xs, [_mean(by_gap[x]) for x in xs], [_stderr95(by_gap[x]) for x in xs])
    _plot_lines_ci(aggregate_dir / f"{prefix}_F9_decay_rate_adaptation.png", "F9 aggregate decay rate adaptation", rate_series, "gap", "global rate", hlines=((1.0 / max(1, int(DEFAULT_CONFIG.decay_horizon_init)), "initial horizon rate"),))
    for arm in ("neutral", "distractor"):
        arm_series = {label.replace(f"_{arm}", ""): vals for label, vals in rate_series.items() if label.endswith(f"_{arm}")}
        if arm_series:
            _plot_lines_ci(aggregate_dir / f"{prefix}_F9_{arm}_decay_rate_adaptation.png", f"F9 aggregate {arm} decay rate adaptation", arm_series, "gap", "global rate", hlines=((1.0 / max(1, int(DEFAULT_CONFIG.decay_horizon_init)), "initial horizon rate"),))


def _render_disambiguation_aggregate_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]], prefix: str = "aggregate") -> None:
    aggregate_dir = _ensure(exp_dir / "aggregate")
    for metric in ("macro_f1", "accuracy"):
        by_thr: Dict[float, List[float]] = defaultdict(list)
        for result in _completed_seed_results(seed_results):
            per_threshold = _nested_get(result.get("metrics", {}) or {}, ("per_threshold",), {}) or {}
            for thr, row in per_threshold.items():
                by_thr[float(thr)].append(float(row.get(metric, 0.0)))
        if by_thr:
            xs = sorted(by_thr)
            _plot_lines_ci(aggregate_dir / f"{prefix}_F10_{metric}_by_threshold.png", f"F10 aggregate {metric} by Jaccard threshold", {metric: (xs, [_mean(by_thr[x]) for x in xs], [_stderr95(by_thr[x]) for x in xs])}, "threshold", metric, vlines=((0.95, "production"),))
    gate_by_thr: Dict[float, List[float]] = defaultdict(list)
    for result in _completed_seed_results(seed_results):
        per_threshold = _nested_get(result.get("metrics", {}) or {}, ("per_threshold",), {}) or {}
        for thr, row in per_threshold.items():
            gate_by_thr[float(thr)].append(float((row.get("gate_decision") or {}).get("macro_f1", 0.0)))
    if gate_by_thr:
        xs = sorted(gate_by_thr)
        _plot_lines_ci(aggregate_dir / f"{prefix}_F10_gate_decision_macro_f1_by_threshold.png", "F10 aggregate gate decision macro-F1 by threshold", {"gate_macro_f1": (xs, [_mean(gate_by_thr[x]) for x in xs], [_stderr95(gate_by_thr[x]) for x in xs])}, "threshold", "macro-F1", vlines=((0.95, "production"),))
    cov = _aggregate_curve_by_threshold(seed_results, ("action_gate_curve",), "coverage")
    acc = _aggregate_curve_by_threshold(seed_results, ("action_gate_curve",), "conditional_top1")
    if cov[0] and acc[0]:
        _plot_lines_ci(aggregate_dir / f"{prefix}_S4_action_gate_coverage_accuracy.png", "S4 aggregate action gate coverage/accuracy", {"coverage": cov, "conditional_top1": acc}, "threshold", "rate")
    for metric in ("macro_f1", "accuracy"):
        by_drop: Dict[float, List[float]] = defaultdict(list)
        for result in _completed_seed_results(seed_results):
            boundary = _nested_get(result.get("metrics", {}) or {}, ("boundary_degradation",), {}) or {}
            for drop, row in boundary.items():
                by_drop[float(drop)].append(float(row.get(metric, 0.0)))
        if by_drop:
            xs = sorted(by_drop)
            _plot_lines_ci(aggregate_dir / f"{prefix}_S4_boundary_{metric}.png", f"S4 aggregate boundary degradation {metric}", {metric: (xs, [_mean(by_drop[x]) for x in xs], [_stderr95(by_drop[x]) for x in xs])}, "drop fraction", metric)
    labels = MEMORY_STATE_LABELS
    counts = np.zeros((len(labels), len(labels)), dtype=float)
    for result in _completed_seed_results(seed_results):
        metrics = result.get("metrics", {}) or {}
        best = str(metrics.get("best_threshold") or "")
        rows = _nested_get(metrics, ("per_threshold", best, "rows"), []) or []
        for row in rows:
            if row.get("gt") in labels and row.get("pred") in labels:
                counts[labels.index(row["gt"]), labels.index(row["pred"])] += 1.0
    if counts.sum() > 0:
        _plot_heatmap(aggregate_dir / f"{prefix}_S4_confusion_matrix_best_threshold.png", "S4 aggregate confusion matrix at selected threshold", counts.tolist(), labels, labels, cmap="Purples")


def _render_materiality_aggregate_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]], prefix: str = "aggregate") -> None:
    aggregate_dir = _ensure(exp_dir / "aggregate")
    by_recipe_eff: Dict[str, List[float]] = defaultdict(list)
    by_recipe_dup: Dict[str, List[float]] = defaultdict(list)
    for result in _completed_seed_results(seed_results):
        for row in result.get("metrics", {}).get("summary", []) or []:
            by_recipe_eff[str(row.get("recipe"))].append(float(row.get("effective_preference_count", 0.0)))
            by_recipe_dup[str(row.get("recipe"))].append(float(row.get("duplicate_count", 0.0)))
    _plot_bar_ci(
        aggregate_dir / f"{prefix}_materiality_effective_preferences.png",
        "Materiality preflight effective preferences per recipe",
        {k: _mean(v) for k, v in sorted(by_recipe_eff.items())},
        {k: _stderr95(v) for k, v in sorted(by_recipe_eff.items())},
        ylabel="count",
    )
    _plot_bar_ci(
        aggregate_dir / f"{prefix}_materiality_duplicate_noop_count.png",
        "Materiality preflight duplicate/no-op count per recipe",
        {k: _mean(v) for k, v in sorted(by_recipe_dup.items())},
        {k: _stderr95(v) for k, v in sorted(by_recipe_dup.items())},
        ylabel="count",
    )


def _render_demo_count_aggregate_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]], prefix: str = "aggregate") -> None:
    aggregate_dir = _ensure(exp_dir / "aggregate")
    series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    baselines = sorted({b for result in _completed_seed_results(seed_results) for row in (result.get("metrics", {}).get("per_demo_count", {}) or {}).values() for b in (row.get("per_baseline", {}) or {})})
    for baseline in baselines:
        by_n: Dict[float, List[float]] = defaultdict(list)
        for result in _completed_seed_results(seed_results):
            for n, row in (result.get("metrics", {}).get("per_demo_count", {}) or {}).items():
                value = _nested_get(row, ("per_baseline", baseline, "live_top1"))
                if isinstance(value, (int, float)):
                    by_n[float(n)].append(float(value))
        if by_n:
            xs = sorted(by_n)
            series[baseline] = (xs, [_mean(by_n[x]) for x in xs], [_stderr95(by_n[x]) for x in xs])
    _plot_lines_ci(aggregate_dir / f"{prefix}_sample_efficiency_live_top1.png", "Sample efficiency by baseline", series, "demo count", "live top-1")


def _render_zipf_aggregate_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]], prefix: str = "aggregate") -> None:
    aggregate_dir = _ensure(exp_dir / "aggregate")
    baselines = sorted({b for result in _completed_seed_results(seed_results) for row in (result.get("metrics", {}).get("per_alpha", {}) or {}).values() for b in row})
    for metric, ylabel in (
        ("utility_top1", "utility top-1"),
        ("fairness_top1", "fairness top-1"),
        ("head_tail_gap", "head-tail gap"),
        ("memory_active_variants", "active variants"),
        ("memory_pruned_variants", "pruned variants"),
    ):
        series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
        for baseline in baselines:
            by_alpha: Dict[float, List[float]] = defaultdict(list)
            for result in _completed_seed_results(seed_results):
                for alpha, row in (result.get("metrics", {}).get("per_alpha", {}) or {}).items():
                    if metric.startswith("memory_"):
                        value = _nested_get(row, (baseline, "memory", metric.replace("memory_", "")))
                    else:
                        value = _nested_get(row, (baseline, metric))
                    if isinstance(value, (int, float)):
                        by_alpha[float(alpha)].append(float(value))
            if by_alpha:
                xs = sorted(by_alpha)
                series[baseline] = (xs, [_mean(by_alpha[x]) for x in xs], [_stderr95(by_alpha[x]) for x in xs])
        _plot_lines_ci(aggregate_dir / f"{prefix}_zipf_{metric}.png", f"Zipf sweep {metric}", series, "alpha", ylabel)


def _render_aggregate_publication_figures(exp_dir: Path, seed_results: Sequence[Dict[str, Any]]) -> None:
    name = exp_dir.name
    if name in {"deployment_stream", "cl_regularizer_comparison"}:
        _render_deployment_aggregate_figures(exp_dir, seed_results)
    elif name in {"cross_recipe_transfer", "posterior_ablation_matrix"}:
        _render_transfer_aggregate_figures(exp_dir, seed_results)
    elif name == "decay_reentry":
        _render_decay_aggregate_figures(exp_dir, seed_results)
    elif name == "disambiguation_audit":
        _render_disambiguation_aggregate_figures(exp_dir, seed_results)
    elif name == "materiality_preflight":
        _render_materiality_aggregate_figures(exp_dir, seed_results)
    elif name == "demo_count_sample_efficiency":
        _render_demo_count_aggregate_figures(exp_dir, seed_results)
    elif name == "zipf_usage_sweep":
        _render_zipf_aggregate_figures(exp_dir, seed_results)


def _rng(seed: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(int(seed) * 1009 + int(salt))


def stable_recipe_builders(seed: int) -> List[Tuple[str, Callable[[], List[str]]]]:
    library = list(gen.recipe_library().items())
    rng = _rng(seed, 17)
    order = list(range(len(library)))
    rng.shuffle(order)
    return [library[int(i)] for i in order]


def select_recipe_builders(seed: int, n_recipes: int, offset: int = 0) -> List[Tuple[str, Callable[[], List[str]]]]:
    ordered = stable_recipe_builders(seed)
    start = max(0, int(offset))
    end = min(len(ordered), start + max(0, int(n_recipes)))
    return ordered[start:end]


def select_preference_names(n_preferences: Optional[int] = None, preferences: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    names = tuple(preferences) if preferences is not None else tuple(PREFERENCE_NAMES)
    if n_preferences is None:
        return names
    return names[: max(0, min(int(n_preferences), len(names)))]


def materialize_pair(recipe_name: str, pref_name: str, generator_fn: Callable[[], List[str]]) -> RecipePrefPair:
    base_actions = list(generator_fn())
    modifier = WorkflowPreferenceModifier()
    if pref_name == "identity":
        actions = tuple(base_actions)
        pref = PRESET_PREFERENCES["identity"]
        report = modifier.modify_recipe_with_report(base_actions, pref)
    else:
        pref = PRESET_PREFERENCES[pref_name]
        report = modifier.modify_recipe_with_report(base_actions, pref)
        actions = tuple(report.actions)
    return RecipePrefPair(
        recipe_name=recipe_name,
        preference_name=pref_name,
        label=f"{recipe_name}/{pref_name}",
        actions=actions,
        preference=pref,
        base_pref=(pref_name == "identity"),
        applied_axes=tuple(report.applied_axes),
        failed_axes=tuple(report.failed_axes),
        unchanged_axes=tuple(report.unchanged_axes),
    )


def materialize_custom_pair(recipe_name: str, pref: WorkflowPreference, generator_fn: Callable[[], List[str]]) -> RecipePrefPair:
    report = WorkflowPreferenceModifier().modify_recipe_with_report(generator_fn(), pref)
    return RecipePrefPair(recipe_name, pref.label, f"{recipe_name}/{pref.label}", tuple(report.actions), pref, False, tuple(report.applied_axes), tuple(report.failed_axes), tuple(report.unchanged_axes))


def build_pairs(
    seed: int,
    n_recipes: int,
    n_preferences: Optional[int] = None,
    preferences: Optional[Sequence[str]] = None,
    offset: int = 0,
    require_material: bool = True,
) -> Tuple[List[str], List[RecipePrefPair]]:
    builders = select_recipe_builders(seed, n_recipes, offset=offset)
    prefs = select_preference_names(n_preferences, preferences)
    pairs: List[RecipePrefPair] = []
    recipes: List[str] = []
    for recipe_name, fn in builders:
        recipes.append(recipe_name)
        for pref_name in prefs:
            try:
                pair = materialize_pair(recipe_name, pref_name, fn)
            except Exception:
                continue
            if require_material and not pair.base_pref and not pair.applied_axes:
                if _ACTIVE_RESULT is not None:
                    _ACTIVE_RESULT.warnings.append(f"filtered_no_op_pair:{pair.label}")
                continue
            pairs.append(pair)
    return recipes, pairs


def _order_hash(pair: RecipePrefPair) -> str:
    return "\x1f".join(pair.actions)


def edit_distance(a: Sequence[str], b: Sequence[str]) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return float(prev[lb] / max(la, lb))


def materiality_rows(pairs: Sequence[RecipePrefPair]) -> List[Dict[str, Any]]:
    by_recipe: Dict[str, List[RecipePrefPair]] = defaultdict(list)
    for pair in pairs:
        by_recipe[pair.recipe_name].append(pair)
    rows: List[Dict[str, Any]] = []
    for recipe, vals in sorted(by_recipe.items()):
        identity = next((p for p in vals if p.preference_name == "identity"), vals[0])
        seen_hashes: Dict[str, str] = {}
        for pair in vals:
            oh = _order_hash(pair)
            rows.append({
                "recipe": recipe,
                "preference": pair.preference_name,
                "label": pair.label,
                "length": len(pair.actions),
                "edit_from_identity": edit_distance(identity.actions, pair.actions),
                "kendall_from_identity": kendall_tau_distance(identity.actions, pair.actions),
                "jaccard_from_identity": jaccard(identity.actions, pair.actions),
                "duplicate_of": seen_hashes.get(oh),
                "is_duplicate": oh in seen_hashes,
                "applied_axes": list(pair.applied_axes),
                "failed_axes": list(pair.failed_axes),
                "unchanged_axes": list(pair.unchanged_axes),
            })
            seen_hashes.setdefault(oh, pair.preference_name)
    return rows


def _build_gap_preferences() -> List[WorkflowPreference]:
    named = {p.label for p in PRESET_PREFERENCES.values()}
    out: List[WorkflowPreference] = []
    for ingredient_flow in WORKFLOW_AXIS_VALUES["ingredient_flow"]:
        for equipment_setup in WORKFLOW_AXIS_VALUES["equipment_setup"]:
            for serving_setup in WORKFLOW_AXIS_VALUES["serving_setup"]:
                for cleanup_timing in WORKFLOW_AXIS_VALUES["cleanup_timing"]:
                    pref = WorkflowPreference(
                        ingredient_flow=ingredient_flow,
                        equipment_setup=equipment_setup,
                        serving_setup=serving_setup,
                        cleanup_timing=cleanup_timing,
                    )
                    if pref.label not in named:
                        out.append(pref)
    return out


def base_config(seed: int, run_config: RunConfig, **overrides: Any) -> Config:
    cfg = replace(
        DEFAULT_CONFIG,
        seed=int(seed),
        verbose=False,
        profile=bool(run_config.profile),
        maxent_valid_action_expansion=bool(run_config.valid_action_expansion),
    )
    if run_config.quick:
        cfg = replace(
            cfg,
            maxent_mc_rollouts=min(cfg.maxent_mc_rollouts, 30),
            maxent_iters_cold=min(cfg.maxent_iters_cold, 25),
            maxent_iters_warm=min(cfg.maxent_iters_warm, 12),
            bc_epochs_cold=min(cfg.bc_epochs_cold, 30),
            bc_epochs_warm=min(cfg.bc_epochs_warm, 15),
        )
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def make_agent(name: str, cfg: Config) -> AdaptiveHRCAgent:
    key = str(name)
    if key == "full":
        return AdaptiveHRCAgent(cfg=cfg)
    aliases = {
        "adaptive": "adaptive_decay",
        "fixed_decay": "FixedDecay",
        "latest_only": "latest_only",
        "latest_pref_only": "latest_only",
        "no_decay": "NoDecay",
        "no_replay": "NoReplay",
        "uniform_weight": "UniformWeight",
        "experience_replay_bc": "ExperienceReplay",
        "experience_replay": "ExperienceReplay",
        "bc": "BC",
        "l2_anchor": "L2Anchor",
        "ewc": "EWC",
        "online_ewc": "online_ewc",
        "bigram": "bigram",
        "frequency_conditioned_bigram": "frequency_conditioned_bigram",
        "uniform_valid": "uniform_valid",
        "most_frequent": "MostFrequentNext",
        "oracle_ceiling": "oracle_ceiling",
        "no_weighted_rehearsal": "UniformWeight",
    }
    ablation_cfg: Optional[Config] = None
    if key in ("irl_ngram_graph_only", "current_ensemble_only", "no_posterior"):
        ablation_cfg = replace(cfg, ablation_disable_posterior=True)
    elif key in ("no_preference_prototype", "no_preference_head"):
        ablation_cfg = replace(cfg, ablation_disable_preference_head=True)
    elif key == "no_recipe_prototype":
        ablation_cfg = replace(cfg, ablation_disable_recipe_prototype=True)
    elif key == "no_state_graph":
        ablation_cfg = replace(cfg, ensemble_delta=0.0, conditioned_state_support_weight=0.0)
    elif key == "no_pruned_memory_prior":
        ablation_cfg = replace(cfg, ablation_disable_pruned_memory_prior=True)
    elif key == "oracle_preference_label":
        ablation_cfg = replace(cfg, ablation_oracle_preference_label=True)
    elif key == "oracle_recipe_and_preference_label":
        ablation_cfg = replace(cfg, ablation_oracle_recipe_and_preference_label=True)
    if ablation_cfg is not None:
        return AdaptiveHRCAgent(cfg=ablation_cfg)
    cls_key = aliases.get(key, key)
    if cls_key not in BASELINE_AGENTS:
        raise KeyError(f"unknown baseline {name!r}")
    return BASELINE_AGENTS[cls_key](cfg=cfg)


def observe_episode(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if isinstance(agent, OracleCeilingAgent):
        agent.set_oracle_target(pair.actions)
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
        "recipe_id": cls.recipe_id,
        "pref_hash": cls.pref_hash,
        "jaccard": cls.jaccard,
        "order_distance": cls.order_distance,
    }


def _memory_state(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Dict[str, str]) -> str:
    rid = name_to_rid.get(pair.recipe_name)
    if rid is None:
        return "no_memory"
    tokens = agent._tokens_from_action_labels(pair.actions)
    h = variant_hash(tokens)
    if (rid, h) in agent.decay.active:
        return "active_memory"
    if rid in agent.memory.variants and h in agent.memory.variants[rid]:
        return "pruned_memory"
    return "same_recipe_new_preference"


def _memory_state_gt(
    agent: AdaptiveHRCAgent,
    pair: RecipePrefPair,
    name_to_rid: Dict[str, str],
    simulator_observed_labels: Optional[set] = None,
    simulator_observed_recipes: Optional[set] = None,
) -> str:
    """Leakage-free memory-state label based on simulator history.

    The active/pruned distinction necessarily consults the agent's decay state,
    but seen/unseen status comes from the event stream, not from memory entries
    that the disambiguator may itself have created.
    """
    observed_labels = set(simulator_observed_labels or ())
    observed_recipes = set(simulator_observed_recipes or ())
    if pair.label not in observed_labels:
        if pair.recipe_name in observed_recipes:
            return "same_recipe_new_preference"
        return "no_memory"
    rid = name_to_rid.get(pair.recipe_name)
    ph = variant_hash(agent._tokens_from_action_labels(list(pair.actions)))
    if rid is not None and (rid, ph) in agent.decay.active:
        return "active_memory"
    return "pruned_memory"


def _posterior_stats(agent: AdaptiveHRCAgent) -> Dict[str, Any]:
    joint = agent.posterior.joint()
    gate = agent.assist_gate_stats()
    if not joint:
        return {
            "posterior_entropy": None,
            "posterior_raw_entropy": None,
            "posterior_max_prob": None,
            "posterior_n_hypotheses": 0,
            "posterior_recipe": agent.inferred_recipe,
            "posterior_preference": None,
            "assist_used": bool(gate.get("assist_used", False)),
            "assist_reason": gate.get("reason"),
            "action_confidence": gate.get("action_confidence"),
            "action_entropy": gate.get("action_entropy"),
        }
    probs = [float(p) for p in joint.values()]
    raw_h = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    norm_h = raw_h / math.log(len(probs)) if len(probs) > 1 else None
    return {
        "posterior_entropy": norm_h,
        "posterior_raw_entropy": raw_h,
        "posterior_max_prob": max(probs),
        "posterior_n_hypotheses": len(probs),
        "posterior_recipe": agent.posterior.argmax_recipe(),
        "posterior_preference": agent.posterior.argmax_preference(),
        "assist_used": bool(gate.get("assist_used", False)),
        "assist_reason": gate.get("reason"),
        "action_confidence": gate.get("action_confidence"),
        "action_entropy": gate.get("action_entropy"),
    }


def assist_episode(
    agent: AdaptiveHRCAgent,
    pair: RecipePrefPair,
    name_to_rid: Optional[Dict[str, str]] = None,
    *,
    run_config: RunConfig,
    commit: bool = True,
    oracle_preference_id: Optional[str] = None,
    oracle_recipe_id: Optional[str] = None,
    event_tag: Optional[Dict[str, Any]] = None,
    simulator_observed_labels: Optional[set] = None,
    simulator_observed_recipes: Optional[set] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if isinstance(agent, OracleCeilingAgent):
        agent.set_oracle_target(pair.actions)
    rid = (name_to_rid or {}).get(pair.recipe_name)
    if simulator_observed_labels is not None or simulator_observed_recipes is not None:
        memory_state = _memory_state_gt(agent, pair, name_to_rid or {}, simulator_observed_labels, simulator_observed_recipes)
    else:
        memory_state = _memory_state(agent, pair, name_to_rid or {})
    observations = observations_from_actions(pair.actions)
    actual_tokens = agent._tokens_from_action_labels(pair.actions)
    steps: List[Dict[str, Any]] = []
    live_steps: List[LiveStepRecord] = []
    hits1 = hits3 = 0
    nll = 0.0
    first_mismatch = -1
    floor = max(float(agent.cfg.prob_floor), 1e-12)
    for idx, (obs, actual) in enumerate(zip(observations, actual_tokens)):
        prefix = list(agent.current_prefix)
        if oracle_recipe_id is not None or oracle_preference_id is not None:
            t_pred = time.perf_counter()
            dist = agent.predict_with_oracle(prefix, oracle_recipe_id or rid, oracle_preference_id)
            prediction_wall_s = time.perf_counter() - t_pred
        else:
            t_pred = time.perf_counter()
            dist = agent.predict_next_tokens(prefix)
            prediction_wall_s = time.perf_counter() - t_pred
        ranked = top_k(dist, k=max(1, run_config.topk)) if dist else []
        correct1 = bool(ranked and ranked[0] == actual)
        correct3 = bool(actual in ranked)
        hits1 += int(correct1)
        hits3 += int(correct3)
        if first_mismatch < 0 and not correct1:
            first_mismatch = idx
        nll -= math.log(max(float(dist.get(actual, floor)), floor))
        post = _posterior_stats(agent)
        row = {
            "step": idx,
            "pair": pair.label,
            "recipe": pair.recipe_name,
            "preference": pair.preference_name,
            "memory_state": memory_state,
            "actual": actual,
            "actual_action": pair.actions[idx],
            "predicted": ranked[0] if ranked else None,
            "correct_top1": correct1,
            "correct_topk": correct3,
            "topk": ranked,
            "topk_probs": {k: float(dist.get(k, 0.0)) for k in ranked},
            "prediction_wall_s": float(prediction_wall_s),
            **post,
        }
        if run_config.log_full_distributions:
            row["distribution"] = {k: float(v) for k, v in sorted(dist.items(), key=lambda kv: -kv[1])}
        steps.append(row)
        live_steps.append(LiveStepRecord(
            step=idx,
            predicted=ranked[0] if ranked else None,
            actual=actual,
            correct_top1=correct1,
            correct_topk=correct3,
            topk=tuple(ranked),
            inferred_recipe=post.get("posterior_recipe"),
            inferred_pref=getattr(agent, "inferred_pref", None),
            inferred_latent_pref=post.get("posterior_preference"),
            posterior_entropy=post.get("posterior_entropy"),
            posterior_confidence=post.get("action_confidence"),
            posterior_raw_entropy=post.get("posterior_raw_entropy"),
            posterior_max_prob=post.get("posterior_max_prob"),
            posterior_n_hypotheses=post.get("posterior_n_hypotheses"),
            assist_used=bool(post.get("assist_used", False)),
            action_marginal_entropy=post.get("action_entropy"),
            prediction_wall_s=float(prediction_wall_s),
        ))
        agent.observe_observation(obs, ground_truth_recipe=rid)
    cls = None
    if commit:
        cls = agent.end_demo()
        if name_to_rid is not None and cls.recipe_id is not None:
            name_to_rid[pair.recipe_name] = cls.recipe_id
    else:
        agent.current_prefix = []
        agent.inferred_recipe = None
        agent.inferred_pref = None
        agent.posterior.reset()
    n = max(1, len(actual_tokens))
    assist_steps = [s for s in steps if s.get("assist_used")]
    closed_wrong = [s for s in steps if not s.get("assist_used") and not s.get("correct_top1")]
    post_div = [s for s in steps[first_mismatch + 1:]] if first_mismatch >= 0 else steps
    recipe_hits = [s for s in steps if rid is not None and s.get("posterior_recipe") == rid]
    confidence_vals = [float(s["action_confidence"]) for s in steps if s.get("action_confidence") is not None]
    entropy_vals = [float(s["posterior_entropy"]) for s in steps if s.get("posterior_entropy") is not None]
    degenerate = [s for s in steps if int(s.get("posterior_n_hypotheses") or 0) <= 1]
    live_record = LiveEpisodeRecord(
        pair_label=pair.label,
        recipe_id=rid or "",
        recipe_name=pair.recipe_name,
        preference_name=pair.preference_name,
        memory_state=memory_state,
        mode="assist",
        steps=tuple(live_steps),
        live_top1=hits1 / n,
        live_topk=hits3 / n,
        n=len(actual_tokens),
        first_mismatch_step=None if first_mismatch < 0 else first_mismatch,
    )
    rich_metrics = live_episode_metrics(live_record, true_rid=rid)
    metrics = {
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "mode": "assist",
        "memory_state": memory_state,
        "n_steps": len(actual_tokens),
        "live_top1": hits1 / n,
        "live_topk": hits3 / n,
        "cross_entropy": nll / n,
        "first_mismatch_step": first_mismatch,
        "post_divergence_top1": _mean([1.0 if s["correct_top1"] else 0.0 for s in post_div]),
        "assistance_coverage": len(assist_steps) / n,
        "conditional_top1": _mean([1.0 if s["correct_top1"] else 0.0 for s in assist_steps]),
        "useful_assistance_rate": (len(assist_steps) / n) * _mean([1.0 if s["correct_top1"] else 0.0 for s in assist_steps]),
        "abstention_error_rate": len(closed_wrong) / max(1, n - len(assist_steps)),
        "posterior_correct_recipe": len(recipe_hits) / n if rid is not None else 0.0,
        "mean_action_confidence": _mean(confidence_vals),
        "mean_posterior_entropy": _mean(entropy_vals),
        "posterior_degenerate_rate": len(degenerate) / n,
        "classification_kind": getattr(cls, "kind", None),
        "classification_recipe_id": getattr(cls, "recipe_id", None),
        "classification_pref_hash": getattr(cls, "pref_hash", None),
        "needs_observation": getattr(agent, "_needs_observation", False) or getattr(cls, "kind", None) == "needs_observation",
        "live_record": live_record,
        **rich_metrics,
        **(event_tag or {}),
    }
    return metrics, steps


def frozen_eval(agent: AdaptiveHRCAgent, pairs: Sequence[RecipePrefPair], name_to_rid: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with agent.frozen():
        for pair in pairs:
            if isinstance(agent, OracleCeilingAgent):
                agent.set_oracle_target(pair.actions)
            rid = (name_to_rid or {}).get(pair.recipe_name)
            out[pair.label] = agent.evaluate_autonomous_tokens(pair.actions, topn=3, ground_truth_recipe=rid)
    return out


def coverage_vs_accuracy_curve(
    agent: AdaptiveHRCAgent,
    pairs: Sequence[RecipePrefPair],
    name_to_rid: Optional[Dict[str, str]],
    run_config: RunConfig,
    thresholds: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    snap = agent.snapshot()
    original_cfg = agent.cfg
    out: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        agent.restore_from(snap)
        agent.cfg = replace(agent.cfg, posterior_action_confidence_threshold=float(threshold))
        rows: List[Dict[str, Any]] = []
        for pair in pairs:
            row, _steps = assist_episode(agent, pair, dict(name_to_rid or {}), run_config=run_config, commit=False, event_tag={"threshold": threshold})
            rows.append(row)
        agg = _aggregate_episode_metrics(rows)
        out[str(threshold)] = {
            "live_top1": float(agg.get("live_top1", 0.0)),
            "live_topk": float(agg.get("live_topk", 0.0)),
            "coverage": float(agg.get("assistance_coverage", 0.0)),
            "conditional_top1": float(agg.get("conditional_top1", 0.0)),
            "useful_assistance_rate": float(agg.get("useful_assistance_rate", 0.0)),
            "n": float(agg.get("n_steps", 0.0)),
        }
    agent.restore_from(snap)
    agent.cfg = original_cfg
    return out


def memory_snapshot(agent: AdaptiveHRCAgent) -> Dict[str, Any]:
    weights = agent.decay.weights()
    return {
        "active_variants": len(agent.decay.active),
        "pruned_variants": len(agent.decay.pruned),
        "latest_keys": len(agent.decay.latest_keys),
        "base_rate": float(agent.decay.base_rate),
        "global_rate": float(agent.decay.global_rate),
        "reuse_gaps": list(agent.decay.reuse_gaps),
        "reuse_gap_window": list(agent.decay.window_snapshot()),
        "active_count_window": list(agent.decay.active_count_window_snapshot()),
        "mean_active_weight": _mean(list(weights.values())),
        "min_active_weight": min(weights.values()) if weights else 0.0,
        "max_active_weight": max(weights.values()) if weights else 0.0,
        "retrain_cycle": int(agent.retrain_cycle),
        "retrain_skipped_count": int(getattr(agent, "retrain_skipped_count", 0)),
        "observation_mode_entries": int(getattr(agent, "observation_mode_entries", 0)),
    }


def compute_snapshot(agent: AdaptiveHRCAgent, wall_s: float) -> Dict[str, Any]:
    profile = {k: {"calls": int(v[0]), "wall_s": float(v[1])} for k, v in getattr(agent, "profile", {}).items()}
    fit_times = [float(v) for v in getattr(agent, "retrain_fit_wall_times", [])]
    total_times = [float(v) for v in getattr(agent, "retrain_total_wall_times", [])]
    build_times = [float(v) for v in getattr(agent, "retrain_build_wall_times", [])]
    flop_estimates = [float(v) for v in getattr(agent, "retrain_flop_estimates", [])]
    return {
        "wall_s": float(wall_s),
        "retrain_cycle": int(agent.retrain_cycle),
        "retrain_skipped_count": int(getattr(agent, "retrain_skipped_count", 0)),
        "n_fits_executed": int(len(fit_times)),
        "active_variants": len(agent.decay.active),
        "pruned_variants": len(agent.decay.pruned),
        "last_retrain_total_wall_s": total_times[-1] if total_times else 0.0,
        "last_retrain_fit_wall_s": fit_times[-1] if fit_times else 0.0,
        "mean_retrain_total_wall_s": _mean(total_times),
        "mean_retrain_fit_wall_s": _mean(fit_times),
        "p95_retrain_fit_wall_s": _p95(fit_times),
        "total_retrain_fit_wall_s": float(sum(fit_times)),
        "total_retrain_build_wall_s": float(sum(build_times)),
        "estimated_flops": float(sum(flop_estimates)),
        "last_estimated_flops": flop_estimates[-1] if flop_estimates else 0.0,
        "retrain_fit_wall_times": fit_times,
        "retrain_total_wall_times": total_times,
        "retrain_build_wall_times": build_times,
        "retrain_flop_estimates": flop_estimates,
        "retrain_events": _jsonable(getattr(agent, "retrain_events", [])),
        "profile": profile,
    }


def _four_cell(seen_recipe: bool, seen_preference: bool) -> str:
    if seen_recipe and seen_preference:
        return "seen_seen"
    if seen_recipe and not seen_preference:
        return "seen_unseen"
    if (not seen_recipe) and seen_preference:
        return "unseen_seen"
    return "unseen_unseen"


def _aggregate_episode_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "live_top1": 0.0,
            "live_topk": 0.0,
            "post_divergence_top1": 0.0,
            "assistance_coverage": 0.0,
            "conditional_top1": 0.0,
            "useful_assistance_rate": 0.0,
            "assist_correct_rate": 0.0,
            "assist_wrong_rate": 0.0,
            "net_assistance_value": 0.0,
            "n_episodes": 0.0,
        }
    weights = [max(1, int(r.get("n_steps", 1))) for r in rows]
    total = max(1, sum(weights))
    out: Dict[str, float] = {"n_episodes": float(len(rows)), "n_steps": float(total)}
    for key in (
        "live_top1", "live_topk", "post_divergence_top1", "assistance_coverage",
        "coverage", "conditional_top1", "useful_assistance_rate",
        "assist_correct_rate", "assist_wrong_rate", "net_assistance_value",
        "posterior_correct_recipe", "recipe_vocab_top1",
        "preference_consistent_top1", "recipe_vocab_conditional_top1",
        "mean_action_confidence",
        "mean_posterior_entropy", "mean_action_marginal_entropy",
        "posterior_degenerate_rate", "cross_entropy", "steps_to_lock",
        "behavioral_steps_to_lock", "posterior_steps_to_lock",
        "adaptation_latency_steps", "confidence_ece", "confidence_brier",
        "pooled_ece", "pooled_brier", "p95_prediction_wall_s",
        "mean_prediction_wall_s",
    ):
        out[key] = sum(float(r.get(key, 0.0)) * w for r, w in zip(rows, weights)) / total
    for bucket, predicate in (
        ("early", lambda fm: 0 <= fm < 5),
        ("middle", lambda fm: 5 <= fm <= 15),
        ("late", lambda fm: fm > 15),
    ):
        bucket_rows = [r for r in rows if predicate(int(r.get("first_mismatch_step", -1)))]
        if bucket_rows:
            bw = [max(1, int(r.get("n_steps", 1))) for r in bucket_rows]
            bt = max(1, sum(bw))
            out[f"post_divergence_top1_first_mismatch_{bucket}"] = sum(float(r.get("post_divergence_top1", 0.0)) * w for r, w in zip(bucket_rows, bw)) / bt
            out[f"n_episodes_first_mismatch_{bucket}"] = float(len(bucket_rows))
    pooled_scores: List[float] = []
    pooled_labels: List[int] = []
    for row in rows:
        record = row.get("live_record")
        if not isinstance(record, LiveEpisodeRecord):
            continue
        for step in record.steps:
            if step.posterior_confidence is not None:
                pooled_scores.append(float(step.posterior_confidence))
                pooled_labels.append(1 if step.correct_top1 else 0)
    if pooled_scores:
        pooled = binary_decision_metrics(pooled_scores, pooled_labels, n_bins=10)
        out["pooled_ece"] = float(pooled.get("ece", 0.0))
        out["pooled_brier"] = float(pooled.get("brier", 0.0))
        out["pooled_auroc"] = float(pooled.get("auroc", 0.5))
        out["confidence_ece"] = out["pooled_ece"]
        out["confidence_brier"] = out["pooled_brier"]
        out["n_calib_pooled"] = float(len(pooled_scores))
    return out


def _pair_parts(event: Dict[str, Any]) -> Tuple[str, str]:
    recipe = event.get("recipe")
    preference = event.get("preference")
    pair = str(event.get("pair", ""))
    if recipe is None and "/" in pair:
        recipe = pair.split("/", 1)[0]
    if preference is None and "/" in pair:
        preference = pair.rsplit("/", 1)[-1]
    return str(recipe or ""), str(preference or "")


def _support_counts(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    seen_recipes: set = set()
    seen_preferences: set = set()
    cells = Counter()
    modes = Counter()
    conditions = Counter()
    for event in events:
        recipe, preference = _pair_parts(event)
        seen_recipe = bool(recipe and recipe in seen_recipes)
        seen_preference = bool(preference and preference in seen_preferences)
        cells[_four_cell(seen_recipe, seen_preference)] += 1
        mode = str(event.get("mode", "unknown"))
        modes[mode] += 1
        conditions[str(event.get("condition", event.get("event_type", "unknown")))] += 1
        if recipe:
            seen_recipes.add(recipe)
        if preference:
            seen_preferences.add(preference)
    return {
        "four_cell": {k: int(cells.get(k, 0)) for k in FOUR_CELL_KEYS},
        "modes": dict(modes),
        "conditions": dict(conditions),
    }


def _scenario_warnings(scenario: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    support = scenario.get("support_counts", {})
    cells = support.get("four_cell", {}) if isinstance(support, dict) else {}
    if scenario.get("n_events", 0) == 0:
        warnings.append("scenario_has_no_recorded_events")
    if cells and sum(int(v) for v in cells.values()) > 0:
        if int(cells.get("seen_unseen", 0)) == 0:
            warnings.append("scenario_has_no_seen_recipe_unseen_preference_events")
        if int(cells.get("unseen_unseen", 0)) == 0 and int(support.get("modes", {}).get("observe", 0)) == 0:
            warnings.append("scenario_has_no_explicit_new_recipe_or_unseen_unseen_events")
    return warnings


def _baseline_health(metrics: Dict[str, Any], records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    per_baseline = metrics.get("per_baseline", {})
    if not isinstance(per_baseline, dict):
        return {}
    episodes = records.get("episode_metrics", [])
    steps = records.get("episode_steps", [])
    out: Dict[str, Any] = {}
    for baseline, row in per_baseline.items():
        b_eps = [r for r in episodes if r.get("baseline") == baseline]
        b_steps = [r for r in steps if r.get("baseline") == baseline]
        if not b_steps and isinstance(row, dict):
            flat = _flatten_numeric("", row)
            n_steps = int(flat.get("n_steps", 0.0))
            live_top1 = float(flat.get("live_top1", 0.0))
            empty_rate = 1.0 if n_steps > 0 and live_top1 == 0.0 and float(flat.get("cross_entropy", 0.0)) > 10.0 else 0.0
            out[baseline] = {
                "n_train_demos": 0,
                "n_assist_episodes": int(flat.get("n_episodes", 0.0)),
                "n_steps": n_steps,
                "n_policy_actions": 0,
                "empty_prediction_rate": empty_rate,
                "assist_gate_reason_counts": {},
                "model_fitted": empty_rate < 1.0,
            }
            continue
        empty_steps = [s for s in b_steps if not s.get("topk") and s.get("predicted") is None]
        reason_counts = Counter(str(s.get("assist_reason", "unknown")) for s in b_steps)
        actions = {str(s.get("actual")) for s in b_steps if s.get("actual") is not None}
        out[baseline] = {
            "n_train_demos": sum(1 for r in b_eps if r.get("mode") == "observe"),
            "n_assist_episodes": sum(1 for r in b_eps if r.get("mode") == "assist"),
            "n_steps": len(b_steps),
            "n_policy_actions": len(actions),
            "empty_prediction_rate": len(empty_steps) / max(1, len(b_steps)),
            "assist_gate_reason_counts": dict(reason_counts),
            "model_fitted": bool(len(actions) > 0 and len(empty_steps) < len(b_steps)),
        }
    return out


def _baseline_warnings(health: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for baseline, row in health.items():
        if float(row.get("empty_prediction_rate", 0.0)) >= 0.95 and int(row.get("n_steps", 0)) > 0:
            warnings.append(f"{baseline}_empty_prediction_rate_high")
        if not row.get("model_fitted", True):
            warnings.append(f"{baseline}_model_not_fitted_or_not_predicting")
    return warnings


def _comparison_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    per_baseline = metrics.get("per_baseline", {})
    if not isinstance(per_baseline, dict) or "full" not in per_baseline:
        return {}
    full = _flatten_numeric("", per_baseline.get("full", {}))
    comparisons: Dict[str, Any] = {}
    preferred = (
        "live_top1", "live_topk", "post_divergence_top1", "assistance_coverage",
        "conditional_top1", "useful_assistance_rate", "frozen_top1",
        "post_adapt_top1", "safe_fail_rate", "preference_transfer_rate",
        "known_shift_live_top1", "novel_recipe_live_top1",
        "primary_transfer_live_top1", "adaptation_latency_steps",
        "p95_prediction_wall_s", "behavioral_steps_to_lock",
        "pooled_ece", "preference_cluster_purity",
    )
    for baseline, row in per_baseline.items():
        if baseline == "full":
            continue
        other = _flatten_numeric("", row)
        diffs = {}
        for key in preferred:
            if key in full and key in other:
                diffs[key] = float(full[key] - other[key])
        if diffs:
            comparisons[f"full_minus_{baseline}"] = diffs
    return comparisons


def _label_to_metric_rows(per_baseline: Dict[str, Dict[str, Any]], metric: str) -> Dict[str, float]:
    return {b: float(per_baseline[b].get(metric, 0.0)) for b in _ordered_baseline_names(per_baseline)}


def _record_common_artifacts(seed_dir: Path, seed: int, config: Any, selected_recipes: Sequence[str], selected_preferences: Sequence[str]) -> None:
    _write_json(seed_dir / "experiment_config.json", config)
    _write_json(seed_dir / "selected_recipes.json", list(selected_recipes))
    _write_json(seed_dir / "selected_preferences.json", list(selected_preferences))


def _run_baseline_stream(
    baseline: str,
    cfg: Config,
    events: Sequence[Any],
    run_config: RunConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    agent = make_agent(baseline, cfg)
    name_to_rid: Dict[str, str] = {}
    episode_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, event in enumerate(events):
        if isinstance(event, ScenarioEvent):
            mode, pair, tags = event.mode, event.pair, {"phase": event.phase, "user_id": event.user_id, **dict(event.tags)}
        else:
            mode, pair, tags = event
        if mode == "observe":
            row = observe_episode(agent, pair, name_to_rid)
            row.update(tags)
            episode_rows.append(row)
        else:
            row, steps = assist_episode(agent, pair, name_to_rid, run_config=run_config, event_tag=tags)
            row.update(tags)
            episode_rows.append(row)
            step_rows.extend({"event_idx": idx, "baseline": baseline, **s} for s in steps)
        memory_rows.append({"event_idx": idx, "baseline": baseline, **memory_snapshot(agent)})
    wall_s = time.perf_counter() - t0
    metrics = {
        **_aggregate_episode_metrics([r for r in episode_rows if r.get("mode") == "assist"]),
        "compute": compute_snapshot(agent, wall_s),
        "memory": memory_snapshot(agent),
    }
    return metrics, episode_rows, step_rows, memory_rows


def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    if n <= 0:
        return np.ones(0)
    if alpha <= 0:
        return np.full(n, 1.0 / n)
    w = 1.0 / (np.arange(1, n + 1, dtype=float) ** float(alpha))
    return w / w.sum()


def run_materiality_audit(seed: int, out: Path, run_config: RunConfig, cfg_obj: MaterialityAuditConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, cfg_obj.n_preferences, require_material=False)
    prefs = select_preference_names(cfg_obj.n_preferences)
    _record_common_artifacts(out, seed, cfg_obj, recipes, prefs)
    rows = materiality_rows(pairs)
    _append_jsonl(out / "materiality_rows.jsonl", rows)
    _write_csv(out / "materiality_rows.csv", rows)
    by_recipe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_recipe[row["recipe"]].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for recipe, vals in sorted(by_recipe.items()):
        hashes = {r["preference"]: _order_hash(next(p for p in pairs if p.label == r["label"])) for r in vals}
        summary_rows.append({
            "recipe": recipe,
            "n_preferences": len(vals),
            "effective_preference_count": len(set(hashes.values())),
            "duplicate_count": sum(1 for r in vals if r["is_duplicate"]),
            "mean_edit_from_identity": _mean([r["edit_from_identity"] for r in vals]),
            "mean_kendall_from_identity": _mean([r["kendall_from_identity"] for r in vals]),
        })
    _write_csv(out / "recipe_materiality_summary.csv", summary_rows)
    metrics = {
        "seed": seed,
        "n_recipes": len(recipes),
        "n_pairs": len(pairs),
        "n_duplicate_pairs": sum(int(r["is_duplicate"]) for r in rows),
        "mean_effective_preference_count": _mean([r["effective_preference_count"] for r in summary_rows]),
        "mean_kendall_from_identity": _mean([r["mean_kendall_from_identity"] for r in summary_rows]),
        "rows": summary_rows,
    }
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/effective_preference_count.png", "effective preferences per recipe", {r["recipe"]: r["effective_preference_count"] for r in summary_rows}, ylabel="count")
    _plot_bar(out / "figures/duplicate_count.png", "duplicate/no-op preferences per recipe", {r["recipe"]: r["duplicate_count"] for r in summary_rows}, ylabel="count")
    return metrics


def run_single_shot_reuse(seed: int, out: Path, run_config: RunConfig, cfg_obj: SingleShotReuseConfig) -> Dict[str, Any]:
    observe_pref = cfg_obj.observe_preference if cfg_obj.observe_preference in PRESET_PREFERENCES else "identity"
    test_prefs = tuple(p for p in cfg_obj.test_preferences if p in PRESET_PREFERENCES) or (observe_pref,)
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    recipes = [r for r, _ in builders]
    pairs = [materialize_pair(recipe, observe_pref, fn) for recipe, fn in builders]
    if run_config.quick:
        pairs = pairs[: max(1, min(len(pairs), 4))]
        builders = builders[:len(pairs)]
        recipes = recipes[:len(pairs)]
    _record_common_artifacts(out, seed, cfg_obj, recipes, (observe_pref,) + tuple(p for p in test_prefs if p != observe_pref))
    _append_jsonl(
        out / "scenario_events.jsonl",
        [
            {"mode": "observe", "phase": "phase_a", "event_type": "single_shot_observe", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
            for p in pairs
        ] + [
            {"mode": "assist", "phase": "first_reuse_probe", "event_type": "single_shot_first_reuse", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
            for p in pairs
        ],
    )
    per_baseline: Dict[str, Dict[str, Any]] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    for baseline in run_config.baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        t0 = time.perf_counter()
        for pair in pairs:
            observe_episode(agent, pair, name_to_rid)
        phase_a_snapshot = agent.snapshot()
        frozen = frozen_eval(agent, pairs, name_to_rid)
        heldout_by_pref: Dict[str, float] = {}
        for test_pref in test_prefs:
            heldout_evals: List[float] = []
            for recipe_name, fn in builders:
                try:
                    heldout_pair = materialize_pair(recipe_name, test_pref, fn)
                except Exception:
                    continue
                with agent.frozen():
                    result = frozen_eval(agent, [heldout_pair], name_to_rid).get(heldout_pair.label, {})
                heldout_evals.append(float(result.get("top1", 0.0)))
            heldout_by_pref[f"heldout_top1_{test_pref}"] = _mean(heldout_evals)
        episodes: List[Dict[str, Any]] = []
        for pair in pairs:
            agent.restore_from(phase_a_snapshot)
            row, steps = assist_episode(agent, pair, name_to_rid, run_config=run_config, event_tag={"experiment_condition": "first_reuse"})
            row["baseline"] = baseline
            episodes.append(row)
            all_episode_rows.append(row)
            all_step_rows.extend({"baseline": baseline, **s} for s in steps)
        wall_s = time.perf_counter() - t0
        frozen_top1 = _mean([v.get("top1", 0.0) for v in frozen.values()])
        frozen_top3 = _mean([v.get("top3", 0.0) for v in frozen.values()])
        per_baseline[baseline] = {
            **_aggregate_episode_metrics(episodes),
            "frozen_top1": frozen_top1,
            "frozen_top3": frozen_top3,
            **heldout_by_pref,
            "compute": compute_snapshot(agent, wall_s),
            "memory": memory_snapshot(agent),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    metrics = {"seed": seed, "n_pairs": len(pairs), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/frozen_top1.png", "single-shot frozen top-1", _label_to_metric_rows(per_baseline, "frozen_top1"), ylabel="top-1")
    _plot_bar(out / "figures/live_top1.png", "single-shot online first-reuse top-1", _label_to_metric_rows(per_baseline, "live_top1"), ylabel="top-1")
    for test_pref in test_prefs:
        _plot_bar(out / f"figures/heldout_top1_{test_pref}.png", f"single-shot held-out {test_pref} top-1", _label_to_metric_rows(per_baseline, f"heldout_top1_{test_pref}"), ylabel="top-1")
    return metrics


def run_deployment_gate_preference_shift(seed: int, out: Path, run_config: RunConfig, cfg_obj: DeploymentGateConfig) -> Dict[str, Any]:
    n_train = max(1, int(cfg_obj.n_train_recipes))
    train_builders = select_recipe_builders(seed, n_train)
    novel_builders = select_recipe_builders(seed, n_train, offset=n_train)
    train_pairs = [materialize_pair(r, cfg_obj.train_preference, fn) for r, fn in train_builders]
    shift_prefs = tuple(p for p in cfg_obj.shift_preferences if p in PRESET_PREFERENCES)
    shift_pairs: List[RecipePrefPair] = []
    for r, fn in train_builders:
        for pref in shift_prefs:
            try:
                shifted = materialize_pair(r, pref, fn)
            except Exception:
                continue
            if shifted.actions != materialize_pair(r, cfg_obj.train_preference, fn).actions:
                shift_pairs.append(shifted)
                break
    novel_pairs = [materialize_pair(r, cfg_obj.train_preference, fn) for r, fn in novel_builders[:len(shift_pairs) or n_train]]
    _record_common_artifacts(out, seed, cfg_obj, [p.recipe_name for p in train_pairs], (cfg_obj.train_preference,) + shift_prefs)
    events_log = [
        {"mode": "assist", "phase": "probe", "event_type": "known_recipe_new_preference", "condition": "known_shift", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for p in shift_pairs
    ] + [
        {"mode": "assist", "phase": "probe", "event_type": "novel_recipe_assist_attempt", "condition": "novel_recipe", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for p in novel_pairs
    ]
    _append_jsonl(out / "scenario_events.jsonl", events_log)
    per_baseline: Dict[str, Dict[str, Any]] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    for baseline in run_config.baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        for pair in train_pairs:
            observe_episode(agent, pair, name_to_rid)
        checkpoint = agent.snapshot()
        known_rows: List[Dict[str, Any]] = []
        novel_rows: List[Dict[str, Any]] = []
        for pair in shift_pairs:
            agent.restore_from(checkpoint)
            local_map = dict(name_to_rid)
            row, steps = assist_episode(agent, pair, local_map, run_config=run_config, event_tag={"condition": "known_recipe_new_preference"})
            post = frozen_eval(agent, [pair], local_map)
            row["post_commit_top1"] = _mean([v.get("top1", 0.0) for v in post.values()])
            row["baseline"] = baseline
            known_rows.append(row)
            all_episode_rows.append(row)
            all_step_rows.extend({"baseline": baseline, "condition": "known_recipe_new_preference", **s} for s in steps)
        for pair in novel_pairs:
            agent.restore_from(checkpoint)
            local_map = dict(name_to_rid)
            row, steps = assist_episode(agent, pair, local_map, run_config=run_config, event_tag={"condition": "novel_recipe_assist_attempt"})
            safe_fail = bool(row.get("needs_observation")) or float(row.get("assistance_coverage", 0.0)) < 0.25
            row["safe_fail"] = safe_fail
            row["baseline"] = baseline
            novel_rows.append(row)
            all_episode_rows.append(row)
            all_step_rows.extend({"baseline": baseline, "condition": "novel_recipe_assist_attempt", **s} for s in steps)
            observe_episode(agent, pair, local_map)
        per_baseline[baseline] = {
            "known_shift": _aggregate_episode_metrics(known_rows),
            "novel_recipe": _aggregate_episode_metrics(novel_rows),
            "post_commit_top1": _mean([r.get("post_commit_top1", 0.0) for r in known_rows]),
            "safe_fail_rate": _mean([1.0 if r.get("safe_fail") else 0.0 for r in novel_rows]),
            "false_new_recipe_rate": _mean([1.0 if r.get("needs_observation") else 0.0 for r in known_rows]),
            "false_known_recipe_rate": _mean([0.0 if r.get("needs_observation") else 1.0 for r in novel_rows]),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    metrics = {"seed": seed, "n_known_shift": len(shift_pairs), "n_novel": len(novel_pairs), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/known_shift_live_top1.png", "known recipe, new preference: live top-1", {b: r["known_shift"]["live_top1"] for b, r in per_baseline.items()}, ylabel="top-1")
    _plot_bar(out / "figures/novel_safe_fail.png", "novel recipe safe-fail rate", {b: r["safe_fail_rate"] for b, r in per_baseline.items()}, ylabel="rate")
    return metrics


def _preference_pid_map(agent: AdaptiveHRCAgent, pairs: Sequence[RecipePrefPair]) -> Dict[str, Optional[str]]:
    # The agent's preference ids are latent; after diagonal training we map
    # simulator preference names to the most recent learned id for oracle probes.
    out: Dict[str, Optional[str]] = {}
    for pair in pairs:
        out[pair.preference_name] = agent.last_pref_id
    return out


def run_cross_recipe_preference_transfer(seed: int, out: Path, run_config: RunConfig, cfg_obj: CrossRecipeTransferConfig) -> Dict[str, Any]:
    prefs = select_preference_names(cfg_obj.n_preferences)
    recipes, all_pairs = build_pairs(seed, cfg_obj.n_recipes, preferences=prefs)
    by_recipe_pref = {(p.recipe_name, p.preference_name): p for p in all_pairs}
    diagonal: List[RecipePrefPair] = []
    for i, recipe in enumerate(recipes):
        pref = prefs[i % len(prefs)]
        pair = by_recipe_pref.get((recipe, pref))
        if pair is not None:
            diagonal.append(pair)
    diagonal_pref_by_recipe = {p.recipe_name: p.preference_name for p in diagonal}
    offdiag = [p for p in all_pairs if p.recipe_name in diagonal_pref_by_recipe and p.preference_name != diagonal_pref_by_recipe[p.recipe_name]]
    if run_config.quick:
        offdiag = offdiag[: max(1, min(12, len(offdiag)))]
    _record_common_artifacts(out, seed, cfg_obj, recipes, prefs)
    _append_jsonl(
        out / "scenario_events.jsonl",
        [
            {"mode": "observe", "phase": "diagonal_train", "event_type": "diagonal_train", "condition": "diagonal_train", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
            for p in diagonal
        ] + [
            {"mode": "assist", "phase": "offdiagonal_probe", "event_type": "offdiagonal_transfer", "condition": "offdiagonal_transfer", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
            for p in offdiag
        ],
    )
    baselines = cfg_obj.baselines
    per_baseline: Dict[str, Dict[str, Any]] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    matrix_rows: Dict[str, Dict[Tuple[str, str], float]] = {}
    for baseline in baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        pref_to_pid: Dict[str, Optional[str]] = {}
        t0 = time.perf_counter()
        for _ in range(max(1, int(cfg_obj.diagonal_cycles))):
            for pair in diagonal:
                observe_episode(agent, pair, name_to_rid)
                pref_to_pid[pair.preference_name] = agent.last_pref_id
        checkpoint = agent.snapshot()
        frozen_diag = frozen_eval(agent, diagonal, name_to_rid)
        frozen_off = frozen_eval(agent, offdiag, name_to_rid)
        rows: List[Dict[str, Any]] = []
        matrix_rows[baseline] = {}
        for pair in offdiag:
            for rep in range(max(1, int(cfg_obj.repeats))):
                agent.restore_from(checkpoint)
                local_map = dict(name_to_rid)
                oracle_pid = pref_to_pid.get(pair.preference_name)
                oracle_rid = local_map.get(pair.recipe_name)
                if baseline == "oracle_preference_label":
                    row, steps = assist_episode(agent, pair, local_map, run_config=run_config, oracle_preference_id=oracle_pid, event_tag={"condition": "offdiagonal_transfer", "repeat": rep})
                elif baseline == "oracle_recipe_and_preference_label":
                    row, steps = assist_episode(agent, pair, local_map, run_config=run_config, oracle_preference_id=oracle_pid, oracle_recipe_id=oracle_rid, event_tag={"condition": "offdiagonal_transfer", "repeat": rep})
                else:
                    row, steps = assist_episode(agent, pair, local_map, run_config=run_config, event_tag={"condition": "offdiagonal_transfer", "repeat": rep})
                row["baseline"] = baseline
                row["seen_recipe"] = pair.recipe_name in name_to_rid
                row["seen_preference_global"] = pair.preference_name in {p.preference_name for p in diagonal}
                row["seen_preference_for_recipe"] = pair.preference_name == diagonal_pref_by_recipe.get(pair.recipe_name)
                row["four_cell_global"] = _four_cell(row["seen_recipe"], row["seen_preference_global"])
                row["preference_transfer_correct"] = row.get("classification_kind") in ("known", "preference_shift", "reentry_from_pruned")
                rows.append(row)
                all_episode_rows.append(row)
                all_step_rows.extend({"baseline": baseline, **s} for s in steps)
            matrix_rows[baseline][(pair.recipe_name, pair.preference_name)] = _mean([r["live_top1"] for r in rows if r["pair"] == pair.label])
        wall_s = time.perf_counter() - t0
        per_baseline[baseline] = {
            **_aggregate_episode_metrics(rows),
            "frozen_diagonal_top1": _mean([v.get("top1", 0.0) for v in frozen_diag.values()]),
            "frozen_offdiagonal_top1": _mean([v.get("top1", 0.0) for v in frozen_off.values()]),
            "preference_transfer_rate": _mean([1.0 if r.get("preference_transfer_correct") else 0.0 for r in rows]),
            "compute": compute_snapshot(agent, wall_s),
            "memory": memory_snapshot(agent),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    metrics = {"seed": seed, "recipes": recipes, "preferences": list(prefs), "diagonal_training_pairs": [p.label for p in diagonal], "n_offdiag": len(offdiag), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/live_top1.png", "cross-recipe off-diagonal live top-1", _label_to_metric_rows(per_baseline, "live_top1"), ylabel="top-1")
    _plot_bar(out / "figures/preference_transfer_rate.png", "cross-recipe preference transfer rate", _label_to_metric_rows(per_baseline, "preference_transfer_rate"), ylabel="rate")
    if "full" in matrix_rows:
        mat = [[matrix_rows["full"].get((recipe, pref), 0.0) for pref in prefs] for recipe in recipes]
        _plot_heatmap(out / "figures/full_transfer_heatmap.png", "full model off-diagonal live top-1", mat, prefs, recipes)
    if "full" in per_baseline and "irl_ngram_graph_only" in per_baseline:
        gain = float(per_baseline["full"]["live_top1"] - per_baseline["irl_ngram_graph_only"]["live_top1"])
        metrics["full_minus_ensemble_live_top1"] = gain
        _write_json(out / "metrics.json", metrics)
    return metrics


def _axis_holdout_split(axis: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    identity = WorkflowPreference().as_dict()[axis]
    train = tuple(n for n, pref in PRESET_PREFERENCES.items() if pref.as_dict()[axis] == identity)
    holdout = tuple(n for n, pref in PRESET_PREFERENCES.items() if pref.as_dict()[axis] != identity)
    return train, holdout


def run_preference_axis_holdout(seed: int, out: Path, run_config: RunConfig, cfg_obj: PreferenceAxisHoldoutConfig) -> Dict[str, Any]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    _record_common_artifacts(out, seed, cfg_obj, [r for r, _ in builders], PREFERENCE_NAMES)
    per_axis: Dict[str, Any] = {}
    for axis in WORKFLOW_AXES:
        train_prefs, holdout_prefs = _axis_holdout_split(axis)
        train_pairs = [materialize_pair(r, p, fn) for r, fn in builders for p in train_prefs]
        raw_holdout = [materialize_pair(r, p, fn) for r, fn in builders for p in holdout_prefs]
        train_hashes = {_order_hash(p) for p in train_pairs}
        holdout_pairs = [p for p in raw_holdout if _order_hash(p) not in train_hashes]
        axis_row: Dict[str, Any] = {"n_train_pairs": len(train_pairs), "n_holdout_pairs": len(holdout_pairs), "n_filtered_duplicates": len(raw_holdout) - len(holdout_pairs)}
        for baseline in run_config.baselines:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            for _ in range(max(1, int(cfg_obj.settle_cycles))):
                for pair in train_pairs:
                    observe_episode(agent, pair, name_to_rid)
            checkpoint = agent.snapshot()
            frozen = frozen_eval(agent, holdout_pairs, name_to_rid)
            rows: List[Dict[str, Any]] = []
            for pair in holdout_pairs:
                agent.restore_from(checkpoint)
                local_map = dict(name_to_rid)
                row, steps = assist_episode(agent, pair, local_map, run_config=run_config, event_tag={"axis": axis})
                post = frozen_eval(agent, [pair], local_map)
                row["post_adapt_top1"] = _mean([v.get("top1", 0.0) for v in post.values()])
                row["baseline"] = baseline
                rows.append(row)
                _append_jsonl(out / "episode_steps.jsonl", ({"baseline": baseline, **s} for s in steps))
            agg = _aggregate_episode_metrics(rows)
            axis_row[baseline] = {
                **agg,
                "frozen_top1": _mean([v.get("top1", 0.0) for v in frozen.values()]),
                "post_adapt_top1": _mean([r.get("post_adapt_top1", 0.0) for r in rows]),
            }
        per_axis[axis] = axis_row
    metrics = {"seed": seed, "per_axis": per_axis}
    for baseline in run_config.baselines:
        metrics[f"{baseline}_live_top1"] = _mean([per_axis[a].get(baseline, {}).get("live_top1", 0.0) for a in WORKFLOW_AXES])
        metrics[f"{baseline}_post_adapt_top1"] = _mean([per_axis[a].get(baseline, {}).get("post_adapt_top1", 0.0) for a in WORKFLOW_AXES])
        metrics[f"{baseline}_frozen_top1"] = _mean([per_axis[a].get(baseline, {}).get("frozen_top1", 0.0) for a in WORKFLOW_AXES])
    _write_json(out / "metrics.json", metrics)
    matrix = [[per_axis[axis].get(b, {}).get("live_top1", 0.0) for axis in WORKFLOW_AXES] for b in run_config.baselines]
    _plot_heatmap(out / "figures/live_top1_by_axis.png", "axis holdout live top-1", matrix, WORKFLOW_AXES, run_config.baselines)
    return metrics


def run_novel_preference_composition(seed: int, out: Path, run_config: RunConfig, cfg_obj: NovelPreferenceCompositionConfig) -> Dict[str, Any]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    gap_prefs = _build_gap_preferences()
    train_pairs = [materialize_pair(r, p, fn) for r, fn in builders for p in PREFERENCE_NAMES]
    train_hashes = {_order_hash(p) for p in train_pairs}
    gap_pairs: List[Tuple[RecipePrefPair, int]] = []
    identity = WorkflowPreference().as_dict()
    for r, fn in builders:
        for pref in gap_prefs:
            try:
                pair = materialize_custom_pair(r, pref, fn)
            except Exception:
                continue
            if _order_hash(pair) in train_hashes:
                continue
            hamming = sum(1 for axis in WORKFLOW_AXES if pref.as_dict()[axis] != identity[axis])
            gap_pairs.append((pair, hamming))
    if run_config.quick:
        gap_pairs = gap_pairs[: max(1, min(20, len(gap_pairs)))]
    _record_common_artifacts(out, seed, cfg_obj, [r for r, _ in builders], tuple(PREFERENCE_NAMES) + tuple(p.label for p in gap_prefs))
    per_baseline: Dict[str, Dict[str, Any]] = {}
    for baseline in run_config.baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        for _ in range(max(1, int(cfg_obj.settle_cycles))):
            for pair in train_pairs:
                observe_episode(agent, pair, name_to_rid)
        checkpoint = agent.snapshot()
        frozen = frozen_eval(agent, [p for p, _h in gap_pairs], name_to_rid)
        rows: List[Dict[str, Any]] = []
        for pair, hamming in gap_pairs:
            agent.restore_from(checkpoint)
            local_map = dict(name_to_rid)
            row, steps = assist_episode(agent, pair, local_map, run_config=run_config, event_tag={"hamming": hamming})
            row["hamming"] = hamming
            post = frozen_eval(agent, [pair], local_map)
            row["post_adapt_top1"] = _mean([v.get("top1", 0.0) for v in post.values()])
            rows.append(row)
            _append_jsonl(out / "episode_steps.jsonl", ({"baseline": baseline, **s} for s in steps))
        per_h: Dict[str, Any] = {}
        for h in sorted({h for _p, h in gap_pairs}):
            h_rows = [r for r in rows if r["hamming"] == h]
            h_pairs = [p for p, hh in gap_pairs if hh == h]
            per_h[str(h)] = {
                **_aggregate_episode_metrics(h_rows),
                "frozen_top1": _mean([frozen.get(p.label, {}).get("top1", 0.0) for p in h_pairs]),
                "post_adapt_top1": _mean([r.get("post_adapt_top1", 0.0) for r in h_rows]),
            }
        per_baseline[baseline] = {**_aggregate_episode_metrics(rows), "per_hamming": per_h, "frozen_top1": _mean([v.get("top1", 0.0) for v in frozen.values()]), "post_adapt_top1": _mean([r.get("post_adapt_top1", 0.0) for r in rows])}
    metrics = {"seed": seed, "n_gap_pairs": len(gap_pairs), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    h_vals = sorted({str(h) for _p, h in gap_pairs})
    matrix = [[per_baseline[b]["per_hamming"].get(h, {}).get("live_top1", 0.0) for h in h_vals] for b in run_config.baselines]
    _plot_heatmap(out / "figures/live_top1_by_hamming.png", "novel preference composition live top-1", matrix, h_vals, run_config.baselines)
    return metrics


def _target_and_variant(seed: int, n_recipes: int) -> Tuple[RecipePrefPair, RecipePrefPair, List[RecipePrefPair]]:
    _recipes, pairs = build_pairs(seed, max(2, n_recipes), n_preferences=7)
    by_recipe: Dict[str, List[RecipePrefPair]] = defaultdict(list)
    for pair in pairs:
        by_recipe[pair.recipe_name].append(pair)
    for vals in by_recipe.values():
        unique = []
        seen = set()
        for pair in vals:
            h = _order_hash(pair)
            if h not in seen:
                unique.append(pair)
                seen.add(h)
        if len(unique) >= 2:
            target, variant = unique[0], unique[1]
            distractors = [p for p in pairs if p.recipe_name != target.recipe_name]
            return target, variant, distractors
    return pairs[0], pairs[1], pairs[2:]


def _target_variants(seed: int, n_recipes: int, n_targets: int) -> List[Tuple[RecipePrefPair, RecipePrefPair, List[RecipePrefPair]]]:
    _recipes, pairs = build_pairs(seed, max(2, n_recipes), n_preferences=7)
    by_recipe: Dict[str, List[RecipePrefPair]] = defaultdict(list)
    for pair in pairs:
        by_recipe[pair.recipe_name].append(pair)
    out: List[Tuple[RecipePrefPair, RecipePrefPair, List[RecipePrefPair]]] = []
    for _recipe, vals in by_recipe.items():
        unique: List[RecipePrefPair] = []
        seen = set()
        for pair in vals:
            h = _order_hash(pair)
            if h not in seen:
                unique.append(pair)
                seen.add(h)
        if len(unique) >= 2:
            target, variant = unique[0], unique[1]
            distractors = [p for p in pairs if p.recipe_name != target.recipe_name]
            out.append((target, variant, distractors))
        if len(out) >= max(1, int(n_targets)):
            break
    if out:
        return out
    target, variant, distractors = _target_and_variant(seed, n_recipes)
    return [(target, variant, distractors)]


def _advance_gap(agent: AdaptiveHRCAgent, gap: int, distractors: Sequence[RecipePrefPair], name_to_rid: Dict[str, str], neutral: bool) -> None:
    if neutral or not distractors:
        for _ in range(max(0, int(gap))):
            agent.session_counter += 1
            agent.decay.step(agent.session_counter, agent.retrain_cycle)
            agent.decay._recompute_effective(agent.session_counter)
    else:
        for i in range(max(0, int(gap))):
            observe_episode(agent, distractors[i % len(distractors)], name_to_rid)


def run_adaptive_decay_reentry(seed: int, out: Path, run_config: RunConfig, cfg_obj: AdaptiveDecayReentryConfig) -> Dict[str, Any]:
    target, variant, distractors = _target_and_variant(seed, cfg_obj.n_recipes)
    _record_common_artifacts(out, seed, cfg_obj, sorted({target.recipe_name, variant.recipe_name} | {p.recipe_name for p in distractors}), PREFERENCE_NAMES)
    per_baseline: Dict[str, Dict[str, Any]] = {}
    for baseline in run_config.baselines:
        gap_rows: Dict[str, Any] = {}
        for gap in cfg_obj.gap_sweep:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            observe_episode(agent, target, name_to_rid)
            observe_episode(agent, variant, name_to_rid)
            rid = name_to_rid.get(target.recipe_name)
            target_hash = variant_hash(agent._tokens_from_action_labels(target.actions))
            _advance_gap(agent, int(gap), distractors, name_to_rid, bool(cfg_obj.neutral_filler))
            active_before = bool(rid is not None and (rid, target_hash) in agent.decay.active)
            pruned_before = bool(rid is not None and ((rid, target_hash) in agent.decay.pruned or target_hash in agent.memory.variants.get(rid, {})) and not active_before)
            row, steps = assist_episode(agent, target, name_to_rid, run_config=run_config, event_tag={"gap": gap})
            post = frozen_eval(agent, [target], name_to_rid)
            row.update({
                "target_active_before": active_before,
                "target_pruned_before": pruned_before,
                "post_reentry_top1": _mean([v.get("top1", 0.0) for v in post.values()]),
                "base_rate_after": float(agent.decay.base_rate),
                "global_rate_after": float(agent.decay.global_rate),
                "active_variants_end": len(agent.decay.active),
                "pruned_variants_end": len(agent.decay.pruned),
            })
            gap_rows[str(gap)] = row
            _append_jsonl(out / "episode_steps.jsonl", ({"baseline": baseline, **s} for s in steps))
        per_baseline[baseline] = {
            "per_gap": gap_rows,
            "live_top1": _mean([r["live_top1"] for r in gap_rows.values()]),
            "post_reentry_top1": _mean([r["post_reentry_top1"] for r in gap_rows.values()]),
            "active_before_rate": _mean([1.0 if r["target_active_before"] else 0.0 for r in gap_rows.values()]),
            "pruned_before_rate": _mean([1.0 if r["target_pruned_before"] else 0.0 for r in gap_rows.values()]),
        }
    metrics = {"seed": seed, "target": target.label, "same_recipe_variant": variant.label, "neutral_filler": bool(cfg_obj.neutral_filler), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/reentry_live_top1.png", "reentry live top-1", _label_to_metric_rows(per_baseline, "live_top1"), ylabel="top-1")
    for baseline, row in per_baseline.items():
        _plot_lines(out / f"figures/{baseline}_gap_curve.png", f"{baseline}: reentry by gap", {baseline: ([int(g) for g in row["per_gap"]], [v["live_top1"] for v in row["per_gap"].values()])}, "gap", "live top-1")
    return metrics


def _user_preferences(seed: int, n_users: int) -> Dict[str, str]:
    prefs = list(PREFERENCE_NAMES)
    rng = _rng(seed, 311)
    rng.shuffle(prefs)
    return {f"U{i + 1}": prefs[i % len(prefs)] for i in range(max(1, n_users))}


def _scenario_event_row(event: Any) -> Dict[str, Any]:
    if isinstance(event, ScenarioEvent):
        return {
            "mode": event.mode,
            "phase": event.phase,
            "user_id": event.user_id,
            "pair": event.pair.label,
            "recipe": event.pair.recipe_name,
            "preference": event.pair.preference_name,
            **dict(event.tags),
        }
    mode, pair, tags = event
    return {"mode": mode, "pair": pair.label, "recipe": pair.recipe_name, "preference": pair.preference_name, **dict(tags)}


def _materialize_or_identity(recipe_name: str, pref_name: str, fn: Callable[[], List[str]]) -> RecipePrefPair:
    try:
        return materialize_pair(recipe_name, pref_name, fn)
    except Exception:
        return materialize_pair(recipe_name, "identity", fn)


def _stream_events(seed: int, cfg_obj: MultiUserStreamConfig, *, observe_new: bool = True) -> Tuple[List[str], List[str], List[ScenarioEvent]]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    users = _user_preferences(seed, cfg_obj.n_users)
    rng = _rng(seed, 719)
    probs = _zipf_probs(len(builders), cfg_obj.zipf_alpha)
    current_user = list(users)[0]
    known_recipes: set = set()
    observed_labels: set = set()
    seen_preferences: set = set()
    last_by_recipe: Dict[str, RecipePrefPair] = {}
    all_by_recipe: Dict[str, List[RecipePrefPair]] = {}
    for recipe_name, fn in builders:
        vals = []
        for pref_name in PREFERENCE_NAMES:
            try:
                vals.append(materialize_pair(recipe_name, pref_name, fn))
            except Exception:
                continue
        all_by_recipe[recipe_name] = vals
    events: List[ScenarioEvent] = []
    event_types = (
        "routine_known_recipe_pref",
        "same_recipe_different_pref",
        "alternating_user_conflict",
        "rare_reentry_candidate",
        "cross_product_transfer_probe",
    )
    event_probs = np.asarray([0.40, 0.22, 0.16, 0.10, 0.12], dtype=float)
    event_probs = event_probs / event_probs.sum()
    for idx in range(max(1, int(cfg_obj.n_events))):
        if rng.random() < float(cfg_obj.switch_probability):
            current_user = list(users)[int(rng.integers(0, len(users)))]
        ridx = int(rng.choice(len(builders), p=probs))
        recipe_name, fn = builders[ridx]
        seen_recipe_before = recipe_name in known_recipes
        event_type = "initial_recipe_observation"
        mode: Literal["observe", "assist"] = "assist"
        if observe_new and not seen_recipe_before:
            pref_name = users[current_user]
            pair = _materialize_or_identity(recipe_name, pref_name, fn)
            mode = "observe"
            phase = "phase_a"
        else:
            phase = "phase_b"
            kind = str(rng.choice(event_types, p=event_probs))
            event_type = kind
            recipe_pairs = all_by_recipe.get(recipe_name, [])
            if kind == "routine_known_recipe_pref":
                pair = last_by_recipe.get(recipe_name) or _materialize_or_identity(recipe_name, users[current_user], fn)
            elif kind == "same_recipe_different_pref":
                base = last_by_recipe.get(recipe_name)
                choices = [p for p in recipe_pairs if base is None or p.preference_name != base.preference_name]
                pair = choices[int(rng.integers(0, len(choices)))] if choices else _materialize_or_identity(recipe_name, users[current_user], fn)
            elif kind == "alternating_user_conflict":
                pair = _materialize_or_identity(recipe_name, users[current_user], fn)
            elif kind == "rare_reentry_candidate":
                older = [e.pair for e in events[:-max(1, min(8, len(events)))] if e.pair.recipe_name in known_recipes]
                pair = older[int(rng.integers(0, len(older)))] if older else (last_by_recipe.get(recipe_name) or _materialize_or_identity(recipe_name, users[current_user], fn))
            elif kind == "cross_product_transfer_probe":
                choices = [p for p in recipe_pairs if p.preference_name in seen_preferences and p.label not in observed_labels]
                pair = choices[int(rng.integers(0, len(choices)))] if choices else _materialize_or_identity(recipe_name, users[current_user], fn)
            else:
                pair = _materialize_or_identity(recipe_name, users[current_user], fn)
        seen_pref_before = pair.preference_name in seen_preferences
        seen_pair_before = pair.label in observed_labels
        events.append(ScenarioEvent(
            pair=pair,
            mode=mode,
            phase=phase,
            user_id=current_user,
            tags={
                "event_idx": idx,
                "event_type": event_type,
                "user_preference": users[current_user],
                "recipe_rank": ridx,
                "seen_recipe_before": seen_recipe_before,
                "seen_preference_before": seen_pref_before,
                "seen_pair_before": seen_pair_before,
                "four_cell_before": _four_cell(seen_recipe_before, seen_pref_before),
            },
        ))
        known_recipes.add(recipe_name)
        observed_labels.add(pair.label)
        seen_preferences.add(pair.preference_name)
        last_by_recipe[recipe_name] = pair
    return [r for r, _ in builders], list(users), events


def run_multi_user_continual_stream(seed: int, out: Path, run_config: RunConfig, cfg_obj: MultiUserStreamConfig) -> Dict[str, Any]:
    recipes, users, events = _stream_events(seed, cfg_obj, observe_new=cfg_obj.observe_first_recipe)
    _record_common_artifacts(out, seed, cfg_obj, recipes, [f"{u}:{_user_preferences(seed, cfg_obj.n_users)[u]}" for u in users])
    _append_jsonl(out / "scenario_events.jsonl", (_scenario_event_row(e) for e in events))
    per_baseline: Dict[str, Dict[str, Any]] = {}
    for baseline in run_config.baselines:
        metrics, episode_rows, step_rows, memory_rows = _run_baseline_stream(baseline, base_config(seed, run_config), events, run_config)
        per_user: Dict[str, Any] = {}
        for user in users:
            rows = [r for r in episode_rows if r.get("user_id") == user and r.get("mode") == "assist"]
            per_user[user] = _aggregate_episode_metrics(rows)
        metrics["per_user"] = per_user
        metrics["switch_count"] = sum(1 for i in range(1, len(events)) if events[i].user_id != events[i - 1].user_id)
        per_baseline[baseline] = metrics
        _append_jsonl(out / "episode_metrics.jsonl", ({"baseline": baseline, **r} for r in episode_rows))
        _append_jsonl(out / "episode_steps.jsonl", step_rows)
        _append_jsonl(out / "memory_trace.jsonl", memory_rows)
    metrics = {"seed": seed, "n_users": len(users), "n_recipes": len(recipes), "n_events": len(events), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/live_top1.png", "multi-user stream live top-1", _label_to_metric_rows(per_baseline, "live_top1"), ylabel="top-1")
    _plot_bar(out / "figures/useful_assistance.png", "multi-user useful assistance", _label_to_metric_rows(per_baseline, "useful_assistance_rate"), ylabel="rate")
    return metrics


def run_bounded_memory_tradeoff(seed: int, out: Path, run_config: RunConfig, cfg_obj: MemoryTradeoffConfig) -> Dict[str, Any]:
    stream_cfg = MultiUserStreamConfig(n_users=4, n_recipes=cfg_obj.n_recipes, n_events=cfg_obj.n_events, zipf_alpha=cfg_obj.zipf_alpha)
    metrics = run_multi_user_continual_stream(seed, out, run_config, stream_cfg)
    values = {}
    for b, row in metrics["per_baseline"].items():
        mem = row.get("memory", {})
        values[b] = float(row.get("useful_assistance_rate", 0.0)) / max(1.0, math.log1p(float(mem.get("active_variants", 0.0))))
    metrics["memory_efficiency"] = values
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/memory_efficiency.png", "useful assistance per log active memory", values, ylabel="efficiency")
    return metrics


def run_compute_tradeoff(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    stream_cfg = MultiUserStreamConfig(n_users=3, n_recipes=cfg_obj.n_recipes, n_events=cfg_obj.n_events, zipf_alpha=0.8)
    recipes, users, events = _stream_events(seed, stream_cfg)
    _record_common_artifacts(out, seed, cfg_obj, recipes, users)
    per_baseline: Dict[str, Dict[str, Any]] = {}
    for baseline in run_config.baselines:
        row, episode_rows, step_rows, memory_rows = _run_baseline_stream(baseline, base_config(seed, run_config), events, run_config)
        per_baseline[baseline] = row
        _append_jsonl(out / "episode_metrics.jsonl", ({"baseline": baseline, **r} for r in episode_rows))
        _append_jsonl(out / "episode_steps.jsonl", step_rows)
        _append_jsonl(out / "memory_trace.jsonl", memory_rows)
    metrics = {"seed": seed, "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/wall_s.png", "compute tradeoff: wall time", {b: r["compute"]["wall_s"] for b, r in per_baseline.items()}, ylabel="seconds")
    _plot_bar(out / "figures/useful_assistance.png", "compute tradeoff: useful assistance", _label_to_metric_rows(per_baseline, "useful_assistance_rate"), ylabel="rate")
    return metrics


def run_short_term_capacity_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: CapacitySweepConfig) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for n in cfg_obj.demo_counts:
        sub = SingleShotReuseConfig(n_recipes=int(n), n_preferences=cfg_obj.n_preferences)
        sub_out = _ensure(out / f"n_{n}")
        rows[str(n)] = run_single_shot_reuse(seed, sub_out, run_config, sub)
        for png in out.glob(f"n_{n}_*.png"):
            try:
                png.unlink()
            except OSError:
                pass
        if _ACTIVE_RESULT is not None:
            for key in list(_ACTIVE_RESULT.figures):
                if key.startswith(f"n_{n}_"):
                    _ACTIVE_RESULT.figures.pop(key, None)
    metrics = {"seed": seed, "per_demo_count": rows}
    _write_json(out / "metrics.json", metrics)
    series = {
        baseline: (list(cfg_obj.demo_counts), [rows[str(n)]["per_baseline"].get(baseline, {}).get("live_top1", 0.0) for n in cfg_obj.demo_counts])
        for baseline in run_config.baselines
    }
    _plot_lines(out / "figures/sample_efficiency_live_top1.png", "sample efficiency by baseline", series, "demo count", "live top-1")
    return metrics


def run_demo_count_sample_efficiency(seed: int, out: Path, run_config: RunConfig, cfg_obj: CapacitySweepConfig) -> Dict[str, Any]:
    metrics = run_short_term_capacity_sweep(seed, out, run_config, cfg_obj)
    efficiency: Dict[str, Dict[str, float]] = {}
    for n, row in metrics["per_demo_count"].items():
        efficiency[n] = {b: row["per_baseline"].get(b, {}).get("live_top1", 0.0) / max(1, int(n)) for b in run_config.baselines}
    metrics["sample_efficiency"] = efficiency
    _write_json(out / "metrics.json", metrics)
    return metrics


def _classify_pair(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Dict[str, str], threshold: Optional[float] = None, actions: Optional[Sequence[str]] = None) -> str:
    seq = agent._tokens_from_action_labels(actions or pair.actions)
    lib = agent.memory.library()
    cls = agent.disambig.classify(seq, lib, threshold=threshold)
    if cls.kind == "new_recipe":
        return "new_recipe"
    rid = name_to_rid.get(pair.recipe_name)
    if rid is None:
        return "new_recipe"
    h = variant_hash(seq)
    if rid in agent.memory.variants and h in agent.memory.variants[rid]:
        return "known"
    return "same_recipe_new_preference"


def _classify_memory_state_pair(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Dict[str, str], threshold: Optional[float] = None, actions: Optional[Sequence[str]] = None) -> str:
    seq = agent._tokens_from_action_labels(actions or pair.actions)
    lib = agent.memory.library()
    cls = agent.disambig.classify(seq, lib, threshold=threshold)
    if cls.kind == "new_recipe":
        return "no_memory"
    rid = name_to_rid.get(pair.recipe_name)
    if rid is None:
        return "no_memory"
    h = variant_hash(seq)
    if (rid, h) in agent.decay.active:
        return "active_memory"
    if rid in agent.memory.variants and h in agent.memory.variants[rid]:
        return "pruned_memory"
    return "same_recipe_new_preference"


def _gate_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = ("assist", "observe")
    matrix = confusion_labels([str(r.get("pred")) for r in rows], [str(r.get("gt")) for r in rows], labels=labels)
    report = classifier_report(matrix, labels)
    return {**report, "confusion_matrix": matrix.tolist(), "rows": list(rows)}


def run_disambiguation_threshold_calibration(seed: int, out: Path, run_config: RunConfig, cfg_obj: ThresholdCalibrationConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=7)
    rng = _rng(seed, 911)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cut = max(1, int((1.0 - cfg_obj.validation_fraction) * len(order)))
    train_pairs = [pairs[i] for i in order[:cut]]
    val_pairs = [pairs[i] for i in order[cut:]]
    _record_common_artifacts(out, seed, cfg_obj, recipes, PREFERENCE_NAMES)
    per_thr: Dict[str, Any] = {}
    for thr in cfg_obj.thresholds:
        agent = make_agent("full", base_config(seed, run_config, jaccard_threshold=float(thr)))
        name_to_rid: Dict[str, str] = {}
        for pair in train_pairs:
            observe_episode(agent, pair, name_to_rid)
        observed_labels = {p.label for p in train_pairs}
        observed_recipes = {p.recipe_name for p in train_pairs}
        rows: List[Dict[str, Any]] = []
        for pair in val_pairs:
            gt = _memory_state_gt(agent, pair, name_to_rid, observed_labels, observed_recipes)
            pred = _classify_memory_state_pair(agent, pair, name_to_rid, threshold=float(thr))
            rows.append({"threshold": thr, "pair": pair.label, "ground_truth": gt, "predicted": pred, "gt": gt, "pred": pred, "correct": gt == pred})
        matrix = confusion_labels([r["predicted"] for r in rows], [r["ground_truth"] for r in rows])
        report = classifier_report(matrix, MEMORY_STATE_LABELS)
        per_thr[str(thr)] = {**report, "confusion_matrix": matrix.tolist(), "rows": rows}
        _append_jsonl(out / "classification_rows.jsonl", rows)
    metrics = {"seed": seed, "per_threshold": per_thr}
    _write_json(out / "metrics.json", metrics)
    _plot_lines(out / "figures/threshold_accuracy.png", "disambiguation threshold validation", {"accuracy": (list(cfg_obj.thresholds), [per_thr[str(t)]["accuracy"] for t in cfg_obj.thresholds])}, "threshold", "accuracy", vlines=((0.95, "production"),))
    return metrics


def run_action_gate_threshold_calibration(seed: int, out: Path, run_config: RunConfig, cfg_obj: ThresholdCalibrationConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=3)
    train = [p for p in pairs if p.preference_name == "identity"]
    probes = [p for p in pairs if p.preference_name != "identity"]
    per_thr: Dict[str, Any] = {}
    for thr in cfg_obj.thresholds:
        cfg = base_config(seed, run_config, posterior_action_confidence_threshold=float(thr))
        agent = make_agent("full", cfg)
        name_to_rid: Dict[str, str] = {}
        for pair in train:
            observe_episode(agent, pair, name_to_rid)
        checkpoint = agent.snapshot()
        rows: List[Dict[str, Any]] = []
        for pair in probes:
            agent.restore_from(checkpoint)
            row, steps = assist_episode(agent, pair, dict(name_to_rid), run_config=run_config, event_tag={"threshold": thr})
            rows.append(row)
            _append_jsonl(out / "episode_steps.jsonl", ({"threshold": thr, **s} for s in steps))
        per_thr[str(thr)] = _aggregate_episode_metrics(rows)
    metrics = {"seed": seed, "per_threshold": per_thr}
    _write_json(out / "metrics.json", metrics)
    _plot_lines(out / "figures/coverage_accuracy.png", "action gate calibration", {
        "coverage": (list(cfg_obj.thresholds), [per_thr[str(t)]["assistance_coverage"] for t in cfg_obj.thresholds]),
        "conditional_top1": (list(cfg_obj.thresholds), [per_thr[str(t)]["conditional_top1"] for t in cfg_obj.thresholds]),
    }, "threshold", "rate")
    return metrics


def _drop_interior(actions: Sequence[str], frac: float, seed: int) -> List[str]:
    if len(actions) <= 2 or frac <= 0:
        return list(actions)
    rng = _rng(seed, int(frac * 10000))
    idxs = list(range(1, len(actions) - 1))
    rng.shuffle(idxs)
    drop = set(idxs[: max(1, int(round(frac * len(actions))))])
    return [a for i, a in enumerate(actions) if i not in drop]


def run_boundary_degradation_disambiguation(seed: int, out: Path, run_config: RunConfig, cfg_obj: ThresholdCalibrationConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=3)
    train = [p for p in pairs if p.preference_name == "identity"]
    probes = [p for p in pairs if p.preference_name != "identity"]
    agent = make_agent("full", base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    for pair in train:
        observe_episode(agent, pair, name_to_rid)
    per_drop: Dict[str, Any] = {}
    for frac in cfg_obj.drop_fractions:
        rows = []
        for pair in probes:
            dropped = _drop_interior(pair.actions, float(frac), seed)
            pred = _classify_pair(agent, pair, name_to_rid, actions=dropped)
            gt = "same_recipe_new_preference" if pair.recipe_name in name_to_rid else "new_recipe"
            rows.append({"drop_fraction": frac, "pair": pair.label, "pred": pred, "gt": gt, "correct": pred == gt})
        per_drop[str(frac)] = {"accuracy": _mean([1.0 if r["correct"] else 0.0 for r in rows]), "rows": rows}
        _append_jsonl(out / "boundary_rows.jsonl", rows)
    metrics = {"seed": seed, "per_drop_fraction": per_drop}
    _write_json(out / "metrics.json", metrics)
    _plot_lines(out / "figures/boundary_degradation.png", "classification under dropped actions", {"accuracy": (list(cfg_obj.drop_fractions), [per_drop[str(f)]["accuracy"] for f in cfg_obj.drop_fractions])}, "dropped fraction", "accuracy")
    return metrics


def run_frequency_gap_decay_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: GapSweepConfig) -> Dict[str, Any]:
    target, variant, distractors = _target_and_variant(seed, cfg_obj.n_recipes)
    per_baseline: Dict[str, Dict[str, Any]] = {}
    for baseline in run_config.baselines:
        per_gap: Dict[str, Any] = {}
        for gap in cfg_obj.gaps:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            observe_episode(agent, target, name_to_rid)
            observe_episode(agent, variant, name_to_rid)
            rates = []
            for _rep in range(3):
                _advance_gap(agent, int(gap), distractors, name_to_rid, True)
                observe_episode(agent, target, name_to_rid)
                rates.append(float(agent.decay.global_rate))
            per_gap[str(gap)] = {"mean_global_rate": _mean(rates), "active_variants": len(agent.decay.active), "reuse_gaps": list(agent.decay.reuse_gaps)}
        per_baseline[baseline] = {"per_gap": per_gap, "mean_global_rate": _mean([v["mean_global_rate"] for v in per_gap.values()])}
    metrics = {"seed": seed, "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    for baseline, row in per_baseline.items():
        _plot_lines(out / f"figures/{baseline}_global_rate.png", f"{baseline}: gap decay rate", {baseline: (list(cfg_obj.gaps), [row["per_gap"][str(g)]["mean_global_rate"] for g in cfg_obj.gaps])}, "gap", "global rate")
    return metrics


def run_mwr_window_sensitivity(seed: int, out: Path, run_config: RunConfig, cfg_obj: MWRWindowSensitivityConfig) -> Dict[str, Any]:
    per_window: Dict[str, Any] = {}
    for window in cfg_obj.windows:
        # Reuse the reentry implementation with the configured window.
        target, variant, distractors = _target_and_variant(seed, cfg_obj.n_recipes)
        rows: Dict[str, Any] = {}
        for baseline in run_config.baselines:
            agent = make_agent(baseline, base_config(seed, run_config, mwr_window=int(window)))
            name_to_rid: Dict[str, str] = {}
            observe_episode(agent, target, name_to_rid)
            observe_episode(agent, variant, name_to_rid)
            _advance_gap(agent, cfg_obj.gap, distractors, name_to_rid, True)
            row, _steps = assist_episode(agent, target, name_to_rid, run_config=run_config, event_tag={"window": window})
            row["global_rate_after"] = float(agent.decay.global_rate)
            rows[baseline] = row
        per_window[str(window)] = rows
    metrics = {"seed": seed, "per_window": per_window}
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_reentry_gap_neutral_vs_distractor(seed: int, out: Path, run_config: RunConfig, cfg_obj: GapSweepConfig) -> Dict[str, Any]:
    neutral = run_adaptive_decay_reentry(seed, _ensure(out / "neutral"), run_config, AdaptiveDecayReentryConfig(cfg_obj.n_recipes, cfg_obj.gaps, True))
    distractor = run_adaptive_decay_reentry(seed, _ensure(out / "distractor"), run_config, AdaptiveDecayReentryConfig(cfg_obj.n_recipes, cfg_obj.gaps, False))
    metrics = {"seed": seed, "neutral": neutral, "distractor": distractor}
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_active_pruned_decay_probe(seed: int, out: Path, run_config: RunConfig, cfg_obj: GapSweepConfig) -> Dict[str, Any]:
    metrics = run_adaptive_decay_reentry(seed, out, run_config, AdaptiveDecayReentryConfig(cfg_obj.n_recipes, cfg_obj.gaps, True))
    metrics["probe_family"] = "active_pruned_decay_probe"
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_sparse_first_exposure_pool_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: SparsePoolConfig) -> Dict[str, Any]:
    prefs = select_preference_names(7)
    per_pool: Dict[str, Any] = {}
    for pool_size in cfg_obj.pool_sizes:
        recipes, pool = build_pairs(seed + int(pool_size), max(1, int(pool_size)), preferences=prefs)
        rng = _rng(seed, int(pool_size) + 1200)
        explore = [pool[int(i)] for i in rng.integers(0, len(pool), size=max(1, int(cfg_obj.n_explore)))] if pool else []
        settle_candidates = list({p.label: p for p in explore}.values()) or pool
        settle = [settle_candidates[int(i) % len(settle_candidates)] for i in range(max(1, int(cfg_obj.n_settle)))] if settle_candidates else []
        row: Dict[str, Any] = {}
        for baseline in run_config.baselines:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            seen_labels: set = set()
            seen_recipes: set = set()
            seen_prefs: set = set()
            cell_rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in FOUR_CELL_KEYS}
            for pair in explore + settle:
                first = pair.label not in seen_labels
                cell = _four_cell(pair.recipe_name in seen_recipes, pair.preference_name in seen_prefs)
                if first and pair.recipe_name not in seen_recipes:
                    observe_episode(agent, pair, name_to_rid)
                else:
                    live, steps = assist_episode(agent, pair, name_to_rid, run_config=run_config, event_tag={"cell": cell, "pool_size": pool_size})
                    cell_rows[cell].append(live)
                    _append_jsonl(out / "episode_steps.jsonl", ({"baseline": baseline, **s} for s in steps))
                seen_labels.add(pair.label)
                seen_recipes.add(pair.recipe_name)
                seen_prefs.add(pair.preference_name)
            evals = frozen_eval(agent, pool, name_to_rid)
            row[baseline] = {
                "all_top1": _mean([v.get("top1", 0.0) for v in evals.values()]),
                "active_variants": len(agent.decay.active),
                "pool_coverage": len(seen_labels) / max(1, len(pool)),
                **{f"{metric}_{cell}": vals.get(metric, 0.0) for cell, rows in cell_rows.items() for vals in [_aggregate_episode_metrics(rows)] for metric in ("live_top1", "assistance_coverage", "conditional_top1", "n_episodes")},
            }
        per_pool[str(pool_size)] = row
    metrics = {"seed": seed, "per_pool": per_pool}
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_zipf_usage_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: ZipfUsageConfig) -> Dict[str, Any]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    base_pairs = [materialize_pair(r, "identity", fn) for r, fn in builders]
    per_alpha: Dict[str, Any] = {}
    for alpha in cfg_obj.alphas:
        probs = _zipf_probs(len(base_pairs), float(alpha))
        rng = _rng(seed, int(alpha * 1000) + 707)
        events: List[Tuple[str, RecipePrefPair, Dict[str, Any]]] = [("observe", p, {"phase": "phase_a"}) for p in base_pairs]
        counts = Counter()
        for i in range(max(1, int(cfg_obj.n_events))):
            idx = int(rng.choice(len(base_pairs), p=probs))
            counts[base_pairs[idx].label] += 1
            events.append(("assist", base_pairs[idx], {"phase": "phase_b", "event_idx": i, "zipf_alpha": alpha, "rank": idx}))
        alpha_row: Dict[str, Any] = {}
        total = max(1, sum(counts.values()))
        head_labels = {p.label for p in base_pairs[: max(1, len(base_pairs) // 4)]}
        tail_labels = {p.label for p in base_pairs[-max(1, len(base_pairs) // 4):]}
        for baseline in run_config.baselines:
            row, episode_rows, step_rows, memory_rows = _run_baseline_stream(baseline, base_config(seed, run_config), events, run_config)
            # Use stream rows for utility/fairness, avoiding an extra replay-heavy pass.
            assist_rows = [r for r in episode_rows if r.get("mode") == "assist"]
            by_label = defaultdict(list)
            for r in assist_rows:
                by_label[r["pair"]].append(r)
            label_acc = {lbl: _aggregate_episode_metrics(rows)["live_top1"] for lbl, rows in by_label.items()}
            utility = sum(label_acc.get(lbl, 0.0) * (cnt / total) for lbl, cnt in counts.items())
            fairness = _mean(list(label_acc.values()))
            row.update({
                "utility_top1": utility,
                "fairness_top1": fairness,
                "head_top1": _mean([label_acc.get(lbl, 0.0) for lbl in head_labels]),
                "tail_top1": _mean([label_acc.get(lbl, 0.0) for lbl in tail_labels]),
                "head_tail_gap": _mean([label_acc.get(lbl, 0.0) for lbl in head_labels]) - _mean([label_acc.get(lbl, 0.0) for lbl in tail_labels]),
                "frequency_hist": dict(counts),
            })
            alpha_row[baseline] = row
            _append_jsonl(out / "episode_steps.jsonl", step_rows)
            _append_jsonl(out / "memory_trace.jsonl", memory_rows)
        per_alpha[str(alpha)] = alpha_row
    metrics = {"seed": seed, "per_alpha": per_alpha}
    _write_json(out / "metrics.json", metrics)
    baselines = sorted({baseline for row in per_alpha.values() for baseline in row})
    for metric, ylabel in (
        ("utility_top1", "utility top-1"),
        ("fairness_top1", "fairness top-1"),
        ("head_tail_gap", "head-tail gap"),
    ):
        series = {
            baseline: (
                list(cfg_obj.alphas),
                [per_alpha[str(alpha)].get(baseline, {}).get(metric, 0.0) for alpha in cfg_obj.alphas],
            )
            for baseline in baselines
        }
        _plot_lines(out / f"figures/zipf_{metric}.png", f"Zipf sweep {metric}", series, "alpha", ylabel)
    for metric, ylabel in (
        ("active_variants", "active variants"),
        ("pruned_variants", "pruned variants"),
    ):
        series = {
            baseline: (
                list(cfg_obj.alphas),
                [per_alpha[str(alpha)].get(baseline, {}).get("memory", {}).get(metric, 0.0) for alpha in cfg_obj.alphas],
            )
            for baseline in baselines
        }
        _plot_lines(out / f"figures/zipf_memory_{metric}.png", f"Zipf sweep memory {metric}", series, "alpha", ylabel)
    return metrics


def run_cycle_width_sparsity_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: CycleWidthConfig) -> Dict[str, Any]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    pairs = [materialize_pair(r, "identity", fn) for r, fn in builders]
    per_width: Dict[str, Any] = {}
    for width in cfg_obj.cycle_widths:
        chosen = pairs[: max(1, min(int(width), len(pairs)))]
        events = [("observe", p, {"phase": "phase_a"}) for p in pairs]
        for i in range(max(1, int(cfg_obj.n_events))):
            events.append(("assist", chosen[i % len(chosen)], {"phase": "phase_b", "cycle_width": width, "event_idx": i}))
        row: Dict[str, Any] = {}
        for baseline in run_config.baselines:
            metrics, episode_rows, step_rows, memory_rows = _run_baseline_stream(baseline, base_config(seed, run_config), events, run_config)
            row[baseline] = metrics
            _append_jsonl(out / "episode_steps.jsonl", step_rows)
            _append_jsonl(out / "memory_trace.jsonl", memory_rows)
        per_width[str(width)] = row
    metrics = {"seed": seed, "per_cycle_width": per_width}
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_baseline_anchor_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    anchor_run = replace(run_config, baselines=("full", "adaptive", "l2_anchor", "ewc", "online_ewc", "experience_replay_bc", "no_replay"))
    stream_cfg = DeploymentStreamConfig(n_recipes=cfg_obj.n_recipes, n_phase_b_events=cfg_obj.n_events)
    return run_deployment_stream(seed, out, anchor_run, stream_cfg)


def run_continual_learning_regularizer_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    cl_run = replace(run_config, baselines=("bc", "l2_anchor", "ewc", "online_ewc", "experience_replay_bc", "full"))
    stream_cfg = DeploymentStreamConfig(n_recipes=cfg_obj.n_recipes, n_phase_b_events=cfg_obj.n_events)
    return run_deployment_stream(seed, out, cl_run, stream_cfg)


def run_posterior_ablation_matrix(seed: int, out: Path, run_config: RunConfig, cfg_obj: CrossRecipeTransferConfig) -> Dict[str, Any]:
    ab_cfg = TransferSuiteConfig(
        n_recipes=cfg_obj.n_recipes,
        n_preferences=cfg_obj.n_preferences,
        diagonal_cycles=cfg_obj.diagonal_cycles,
        offdiag_repeats=cfg_obj.repeats,
        baselines=("full", "irl_ngram_graph_only", "no_posterior", "no_pruned_memory_prior", "oracle_preference_label"),
    )
    return run_cross_recipe_transfer_suite(seed, out, run_config, ab_cfg)


def run_recipe_preference_factor_ablation(seed: int, out: Path, run_config: RunConfig, cfg_obj: CrossRecipeTransferConfig) -> Dict[str, Any]:
    ab_cfg = TransferSuiteConfig(
        n_recipes=cfg_obj.n_recipes,
        n_preferences=cfg_obj.n_preferences,
        diagonal_cycles=cfg_obj.diagonal_cycles,
        offdiag_repeats=cfg_obj.repeats,
        baselines=("full", "no_preference_prototype", "no_recipe_prototype", "no_state_graph", "no_weighted_rehearsal", "bigram"),
    )
    return run_cross_recipe_transfer_suite(seed, out, run_config, ab_cfg)


def run_coverage_accuracy_curve(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    thresholds = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7)
    per_thr: Dict[str, Any] = {}
    for thr in thresholds:
        thr_run = replace(run_config, baselines=("full",))
        stream_cfg = MultiUserStreamConfig(n_users=3, n_recipes=cfg_obj.n_recipes, n_events=cfg_obj.n_events, zipf_alpha=0.8)
        recipes, users, events = _stream_events(seed, stream_cfg)
        cfg = base_config(seed, thr_run, posterior_action_confidence_threshold=float(thr))
        metrics, episode_rows, step_rows, memory_rows = _run_baseline_stream("full", cfg, events, thr_run)
        per_thr[str(thr)] = metrics
        _append_jsonl(out / "episode_steps.jsonl", ({"threshold": thr, **s} for s in step_rows))
    metrics = {"seed": seed, "per_threshold": per_thr}
    _write_json(out / "metrics.json", metrics)
    _plot_lines(out / "figures/coverage_accuracy_curve.png", "coverage-accuracy curve", {
        "coverage": (list(thresholds), [per_thr[str(t)]["assistance_coverage"] for t in thresholds]),
        "conditional_top1": (list(thresholds), [per_thr[str(t)]["conditional_top1"] for t in thresholds]),
    }, "threshold", "rate")
    return metrics


def _calibration_bins(step_rows: Sequence[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    pairs = [(float(r["action_confidence"]), 1.0 if r.get("correct_top1") else 0.0) for r in step_rows if r.get("action_confidence") is not None]
    if not pairs:
        return {"ece": 0.0, "brier": 0.0, "bins": []}
    bins = []
    ece = 0.0
    brier = _mean([(s - y) ** 2 for s, y in pairs])
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        vals = [(s, y) for s, y in pairs if (s >= lo and (s < hi or i == n_bins - 1))]
        if not vals:
            continue
        conf = _mean([s for s, _y in vals])
        acc = _mean([y for _s, y in vals])
        ece += (len(vals) / len(pairs)) * abs(conf - acc)
        bins.append({"lo": lo, "hi": hi, "mean_confidence": conf, "accuracy": acc, "n": len(vals)})
    return {"ece": ece, "brier": brier, "bins": bins}


def run_confidence_calibration_reliability(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    stream_cfg = MultiUserStreamConfig(n_users=3, n_recipes=cfg_obj.n_recipes, n_events=cfg_obj.n_events)
    recipes, users, events = _stream_events(seed, stream_cfg)
    per_baseline: Dict[str, Any] = {}
    for baseline in run_config.baselines:
        metrics, episode_rows, step_rows, memory_rows = _run_baseline_stream(baseline, base_config(seed, run_config), events, run_config)
        calib = _calibration_bins(step_rows)
        metrics["calibration"] = calib
        per_baseline[baseline] = metrics
        _append_jsonl(out / "episode_steps.jsonl", step_rows)
    metrics = {"seed": seed, "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    return metrics


def _stress_prefix_collision(seed: int, baseline: str, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=2)
    best_pair: Optional[Tuple[RecipePrefPair, RecipePrefPair, int]] = None
    for i, a in enumerate(pairs):
        for b in pairs[i + 1:]:
            shared = 0
            for x, y in zip(a.actions, b.actions):
                if x == y:
                    shared += 1
                else:
                    break
            if best_pair is None or shared > best_pair[2]:
                best_pair = (a, b, shared)
    if best_pair is None:
        return {"score": 0.0}
    a, b, shared = best_pair
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    observe_episode(agent, a, name_to_rid)
    observe_episode(agent, b, name_to_rid)
    row, _steps = assist_episode(agent, b, name_to_rid, run_config=run_config, event_tag={"stress": "prefix_collision"})
    return {"score": row["live_top1"], "live_top1": row["live_top1"], "shared_prefix": shared}


def _stress_preference_thrash(seed: int, baseline: str, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, 1, n_preferences=7)
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    rows = []
    for i in range(max(1, cfg_obj.reps)):
        pair = pairs[i % len(pairs)]
        if i == 0:
            observe_episode(agent, pair, name_to_rid)
        else:
            row, _steps = assist_episode(agent, pair, name_to_rid, run_config=run_config, event_tag={"stress": "preference_thrash"})
            rows.append(row)
    agg = _aggregate_episode_metrics(rows)
    return {"score": agg["live_top1"], **agg, "active_variants": len(agent.decay.active)}


def _stress_reentry(seed: int, baseline: str, run_config: RunConfig, cfg_obj: StressConfig, distractor: bool = False) -> Dict[str, Any]:
    target, variant, distractors = _target_and_variant(seed, cfg_obj.n_recipes)
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    observe_episode(agent, target, name_to_rid)
    observe_episode(agent, variant, name_to_rid)
    _advance_gap(agent, cfg_obj.gap, distractors, name_to_rid, neutral=not distractor)
    row, _steps = assist_episode(agent, target, name_to_rid, run_config=run_config, event_tag={"stress": "rare_reentry"})
    return {"score": row["live_top1"], **row, "active_variants": len(agent.decay.active), "pruned_variants": len(agent.decay.pruned)}


def _stress_memory_exhaustion(seed: int, baseline: str, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=7)
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    for pair in pairs[: max(1, min(len(pairs), cfg_obj.reps * 4))]:
        observe_episode(agent, pair, name_to_rid)
    evals = frozen_eval(agent, pairs, name_to_rid)
    top1 = _mean([v.get("top1", 0.0) for v in evals.values()])
    return {"score": top1, "frozen_top1": top1, "active_variants": len(agent.decay.active), "pruned_variants": len(agent.decay.pruned)}


def _run_stress(seed: int, out: Path, run_config: RunConfig, cfg_obj: StressConfig, kind: str) -> Dict[str, Any]:
    fns = {
        "prefix_collision_stress": _stress_prefix_collision,
        "preference_thrash_stress": _stress_preference_thrash,
        "rare_reentry_stress": lambda s, b, r, c: _stress_reentry(s, b, r, c, False),
        "late_distractor_stress": lambda s, b, r, c: _stress_reentry(s, b, r, c, True),
        "memory_exhaustion_stress": _stress_memory_exhaustion,
    }
    per_baseline: Dict[str, Any] = {}
    baselines = tuple(b for b in run_config.baselines if b != "oracle_ceiling") or ("full",)
    for baseline in baselines:
        rows = [fns[kind](seed + i * 37, baseline, run_config, cfg_obj) for i in range(max(1, int(cfg_obj.reps if run_config.quick else min(cfg_obj.reps, 10))))]
        per_baseline[baseline] = {key: _mean([r.get(key, 0.0) for r in rows]) for key in sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float, bool))})}
        per_baseline[baseline]["rep_rows"] = rows
    metrics = {"seed": seed, "stress_kind": kind, "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    _plot_bar(out / "figures/stress_score.png", f"{kind} score", {b: r.get("score", 0.0) for b, r in per_baseline.items()}, ylabel="score")
    return metrics


def run_memory_exhaustion_stress(seed: int, out: Path, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    return _run_stress(seed, out, run_config, cfg_obj, "memory_exhaustion_stress")


def run_prefix_collision_stress(seed: int, out: Path, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    return _run_stress(seed, out, run_config, cfg_obj, "prefix_collision_stress")


def run_preference_thrash_stress(seed: int, out: Path, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    return _run_stress(seed, out, run_config, cfg_obj, "preference_thrash_stress")


def run_rare_reentry_stress(seed: int, out: Path, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    return _run_stress(seed, out, run_config, cfg_obj, "rare_reentry_stress")


def run_late_distractor_stress(seed: int, out: Path, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    return _run_stress(seed, out, run_config, cfg_obj, "late_distractor_stress")


def run_seed_recipe_selection_audit(seed: int, out: Path, run_config: RunConfig, cfg_obj: MaterialityAuditConfig) -> Dict[str, Any]:
    ordered = stable_recipe_builders(seed)
    selected = ordered[: cfg_obj.n_recipes]
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, cfg_obj.n_preferences, require_material=False)
    rows = materiality_rows(pairs)
    metrics = {
        "seed": seed,
        "recipe_order": [r for r, _ in ordered],
        "selected_recipes": [r for r, _ in selected],
        "n_selected": len(selected),
        "mean_sequence_length": _mean([len(p.actions) for p in pairs]),
        "duplicate_rate": _mean([1.0 if r["is_duplicate"] else 0.0 for r in rows]),
        "mean_kendall_from_identity": _mean([r["kendall_from_identity"] for r in rows]),
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_runtime_scaling_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: RuntimeScalingConfig) -> Dict[str, Any]:
    per_n: Dict[str, Any] = {}
    small_run = replace(run_config, baselines=("full", "bigram"))
    for n in cfg_obj.recipe_counts:
        metrics = run_compute_tradeoff(seed, _ensure(out / f"n_{n}"), small_run, ComputeTradeoffConfig(n_recipes=int(n), n_events=cfg_obj.n_events))
        per_n[str(n)] = metrics
    result = {"seed": seed, "per_recipe_count": per_n}
    _write_json(out / "metrics.json", result)
    return result


def _material_pair_candidates(recipe_name: str, fn: Callable[[], List[str]], prefs: Sequence[str]) -> List[RecipePrefPair]:
    pairs: List[RecipePrefPair] = []
    for pref in prefs:
        try:
            pair = materialize_pair(recipe_name, pref, fn)
        except Exception:
            continue
        if pair.base_pref or pair.applied_axes:
            pairs.append(pair)
    return pairs


def _blend_preference_pair(base: RecipePrefPair, target: RecipePrefPair, alpha: float, rng: np.random.Generator) -> RecipePrefPair:
    base_order = {action: i for i, action in enumerate(base.actions)}
    target_order = {action: i for i, action in enumerate(target.actions)}
    actions = sorted(
        set(base.actions) | set(target.actions),
        key=lambda action: (
            (1.0 - float(alpha)) * base_order.get(action, len(base_order))
            + float(alpha) * target_order.get(action, len(target_order))
            + float(rng.random()) * 1e-6
        ),
    )
    return RecipePrefPair(
        recipe_name=target.recipe_name,
        preference_name=target.preference_name,
        label=f"{target.recipe_name}/gradual_{base.preference_name}_to_{target.preference_name}_{int(alpha * 100)}",
        actions=tuple(actions),
        preference=target.preference,
        base_pref=False,
        applied_axes=target.applied_axes,
        failed_axes=target.failed_axes,
        unchanged_axes=target.unchanged_axes,
    )


def _blocked_user_for_event(user_ids: Sequence[str], phase_b_idx: int, block_size: int) -> Tuple[str, int, bool]:
    if not user_ids:
        return "U1", 0, False
    if len(user_ids) == 1:
        return user_ids[0], 0, False
    block = int(phase_b_idx) // max(1, int(block_size))
    if block % 2 == 0:
        return user_ids[0], block, block >= 2
    other_idx = ((block // 2) % (len(user_ids) - 1)) + 1
    return user_ids[other_idx], block, False


def _deployment_stream_events(seed: int, cfg_obj: DeploymentStreamConfig, run_config: RunConfig) -> Tuple[List[str], List[str], List[ScenarioEvent], List[RecipePrefPair]]:
    n_recipes = min(cfg_obj.n_recipes, 4) if run_config.quick else cfg_obj.n_recipes
    n_events = min(cfg_obj.n_phase_b_events, 24) if run_config.quick else cfg_obj.n_phase_b_events
    builders = select_recipe_builders(seed, n_recipes)
    n_new_candidates = max(1, int(math.ceil(max(0.0, float(cfg_obj.new_recipe_obs_rate)) * max(1, n_events))))
    new_builders = select_recipe_builders(seed, n_new_candidates, offset=n_recipes)
    users = _user_preferences(seed, cfg_obj.n_users)
    rng = _rng(seed, 5101)
    probs = _zipf_probs(len(builders), cfg_obj.zipf_alpha)
    by_recipe: Dict[str, List[RecipePrefPair]] = {
        r: _material_pair_candidates(r, fn, PREFERENCE_NAMES)
        for r, fn in list(builders) + list(new_builders)
    }
    phase_a_pairs: List[RecipePrefPair] = []
    for recipe_name, fn in builders:
        candidates = by_recipe.get(recipe_name) or [materialize_pair(recipe_name, "identity", fn)]
        identity = next((p for p in candidates if p.preference_name == "identity"), candidates[0])
        non_identity = [p for p in candidates if p.preference_name != "identity"]
        if non_identity and rng.random() < float(cfg_obj.phase_a_non_identity_prob):
            phase_a_pairs.append(non_identity[int(rng.integers(0, len(non_identity)))])
        else:
            phase_a_pairs.append(identity)
    events: List[ScenarioEvent] = []
    seen_recipes: set = set()
    seen_preferences: set = set()
    seen_pairs: set = set()
    last_by_recipe: Dict[str, RecipePrefPair] = {}
    last_by_user_recipe: Dict[Tuple[str, str], RecipePrefPair] = {}
    for idx, pair in enumerate(phase_a_pairs):
        seen_preference_before = pair.preference_name in seen_preferences
        events.append(ScenarioEvent(pair, "observe", "phase_a", "U1", {
            "event_idx": idx,
            "event_type": "phase_a_observe",
            "phase_a_preference": pair.preference_name,
            "four_cell_before": "unseen_unseen",
            "seen_recipe_before": False,
            "seen_preference_before": seen_preference_before,
            "seen_pair_before": False,
        }))
        seen_recipes.add(pair.recipe_name)
        seen_preferences.add(pair.preference_name)
        seen_pairs.add(pair.label)
        last_by_recipe[pair.recipe_name] = pair
        last_by_user_recipe[("U1", pair.recipe_name)] = pair
    event_mix = dict(cfg_obj.event_mix)
    event_types = tuple(event_mix)
    weights = np.asarray([max(0.0, float(event_mix[k])) for k in event_types], dtype=float)
    weights = weights / max(float(weights.sum()), 1e-9)
    user_ids = list(users)
    history: List[RecipePrefPair] = list(phase_a_pairs)
    unused_new = list(new_builders)
    pending_gradual_followups: List[Tuple[RecipePrefPair, int, str]] = []
    for i in range(max(0, int(n_events))):
        current_user, user_block, is_return_block = _blocked_user_for_event(user_ids, i, cfg_obj.user_block_size)
        block_start = i > 0 and (i % max(1, int(cfg_obj.user_block_size)) == 0)
        if unused_new and rng.random() < float(cfg_obj.new_recipe_obs_rate):
            recipe_name, fn = unused_new.pop(0)
            candidates = by_recipe.get(recipe_name) or [materialize_pair(recipe_name, "identity", fn)]
            pref = users.get(current_user, "identity")
            pair = next((p for p in candidates if p.preference_name == pref), candidates[0])
            events.append(ScenarioEvent(pair, "observe", "phase_b_observe", current_user, {
                "event_idx": len(events),
                "phase_b_idx": i,
                "event_type": "new_recipe_observe",
                "recipe_rank": n_recipes + len(new_builders) - len(unused_new) - 1,
                "user_preference": users[current_user],
                "user_block": user_block,
                "is_return_user_block": is_return_block,
                "seen_recipe_before": False,
                "seen_preference_before": pair.preference_name in seen_preferences,
                "seen_pair_before": False,
                "four_cell_before": _four_cell(False, pair.preference_name in seen_preferences),
            }))
            seen_recipes.add(pair.recipe_name)
            seen_preferences.add(pair.preference_name)
            seen_pairs.add(pair.label)
            last_by_recipe[pair.recipe_name] = pair
            last_by_user_recipe[(current_user, pair.recipe_name)] = pair
            history.append(pair)
            continue
        if pending_gradual_followups:
            pair, recipe_idx, requested_event_type = pending_gradual_followups.pop(0)
            seen_recipe_before = pair.recipe_name in seen_recipes
            seen_preference_before = pair.preference_name in seen_preferences
            events.append(ScenarioEvent(pair, "assist", "phase_b", current_user, {
                "event_idx": len(events),
                "phase_b_idx": i,
                "event_type": "gradual_shift_followup",
                "requested_event_type": requested_event_type,
                "event_substituted": False,
                "recipe_rank": recipe_idx,
                "user_preference": users[current_user],
                "user_block": user_block,
                "is_return_user_block": is_return_block,
                "block_start": block_start,
                "eventual_preference": pair.preference_name,
                "seen_recipe_before": seen_recipe_before,
                "seen_preference_before": seen_preference_before,
                "seen_pair_before": pair.label in seen_pairs,
                "four_cell_before": _four_cell(seen_recipe_before, seen_preference_before),
                "followup_for": "gradual_shift",
            }))
            seen_recipes.add(pair.recipe_name)
            seen_preferences.add(pair.preference_name)
            seen_pairs.add(pair.label)
            last_by_recipe[pair.recipe_name] = pair
            last_by_user_recipe[(current_user, pair.recipe_name)] = pair
            history.append(pair)
            continue
        recipe_idx = int(rng.choice(len(builders), p=probs))
        recipe_name, fn = builders[recipe_idx]
        candidates = by_recipe.get(recipe_name) or [materialize_pair(recipe_name, "identity", fn)]
        requested_event_type = str(rng.choice(event_types, p=weights))
        event_type = "user_switch" if block_start else requested_event_type
        base = last_by_user_recipe.get((current_user, recipe_name)) or last_by_recipe.get(recipe_name) or candidates[0]
        target_after_event: Optional[RecipePrefPair] = None
        if event_type == "routine_reuse":
            pair = base
        elif event_type == "preference_shift":
            choices = [p for p in candidates if p.preference_name != base.preference_name]
            pair = choices[int(rng.integers(0, len(choices)))] if choices else base
        elif event_type == "gradual_shift":
            choices = [p for p in candidates if p.preference_name != base.preference_name]
            target = choices[int(rng.integers(0, len(choices)))] if choices else base
            pair = _blend_preference_pair(base, target, alpha=0.30, rng=rng) if target is not base else base
            target_after_event = target
            if target is not base:
                pending_gradual_followups.append((target, recipe_idx, "gradual_shift"))
        elif event_type == "user_switch":
            pref = users[current_user]
            pair = next((p for p in candidates if p.preference_name == pref), base)
        elif event_type == "cross_transfer_probe":
            seen_on_other = {
                h.preference_name
                for h in history
                if h.recipe_name != recipe_name and h.preference_name != "identity"
            }
            seen_on_this = {
                h.preference_name
                for h in history
                if h.recipe_name == recipe_name
            }
            choices = [
                p for p in candidates
                if p.preference_name in seen_on_other
                and p.preference_name not in seen_on_this
                and p.label not in seen_pairs
            ]
            if i >= int(cfg_obj.transfer_warmup_events) and choices:
                pair = choices[int(rng.integers(0, len(choices)))]
            else:
                choices = [p for p in candidates if p.preference_name != base.preference_name]
                pair = choices[int(rng.integers(0, len(choices)))] if choices else base
                event_type = "preference_shift"
        elif event_type == "rare_reentry":
            horizon = int(getattr(DEFAULT_CONFIG, "decay_horizon_init", 10))
            old = history[:-horizon] if len(history) > horizon else []
            pair = old[int(rng.integers(0, len(old)))] if old else base
        else:
            pair = base
        seen_recipe_before = pair.recipe_name in seen_recipes
        seen_preference_before = pair.preference_name in seen_preferences
        events.append(ScenarioEvent(pair, "assist", "phase_b", current_user, {
            "event_idx": len(events),
            "phase_b_idx": i,
            "event_type": event_type,
            "requested_event_type": requested_event_type,
            "event_substituted": event_type != requested_event_type,
            "recipe_rank": recipe_idx,
            "user_preference": users[current_user],
            "user_block": user_block,
            "is_return_user_block": is_return_block,
            "block_start": block_start,
            "eventual_preference": target_after_event.preference_name if target_after_event is not None else pair.preference_name,
            "seen_recipe_before": seen_recipe_before,
            "seen_preference_before": seen_preference_before,
            "seen_pair_before": pair.label in seen_pairs,
            "four_cell_before": _four_cell(seen_recipe_before, seen_preference_before),
        }))
        seen_recipes.add(pair.recipe_name)
        seen_preferences.add(pair.preference_name)
        seen_pairs.add(pair.label)
        last_pair = target_after_event or pair
        last_by_recipe[pair.recipe_name] = last_pair
        last_by_user_recipe[(current_user, pair.recipe_name)] = last_pair
        history.append(pair)
    return [r for r, _ in builders], user_ids, events, phase_a_pairs


def run_deployment_stream(seed: int, out: Path, run_config: RunConfig, cfg_obj: DeploymentStreamConfig) -> Dict[str, Any]:
    recipes, users, events, phase_a_pairs = _deployment_stream_events(seed, cfg_obj, run_config)
    _record_common_artifacts(out, seed, cfg_obj, recipes, [f"{u}:{_user_preferences(seed, cfg_obj.n_users)[u]}" for u in users])
    _append_jsonl(out / "scenario_events.jsonl", (_scenario_event_row(e) for e in events))
    per_baseline: Dict[str, Dict[str, Any]] = {}
    coverage_curves: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    all_memory_rows: List[Dict[str, Any]] = []
    forgetting_rows: List[Dict[str, Any]] = []
    interval = max(1, int(cfg_obj.forgetting_checkpoint_interval))
    phase_b_pairs = [e.pair for e in events if e.mode == "assist"]
    eval_pool = list({event.pair.label: event.pair for event in events}.values())
    for baseline in run_config.baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        simulator_observed_labels: set = set()
        simulator_observed_recipes: set = set()
        phase_a_eval_at_start: Optional[Dict[str, Dict[str, float]]] = None
        episode_rows: List[Dict[str, Any]] = []
        step_rows: List[Dict[str, Any]] = []
        memory_rows: List[Dict[str, Any]] = []
        phase_b_count = 0
        t0 = time.perf_counter()
        for idx, event in enumerate(events):
            tags = {"phase": event.phase, "user_id": event.user_id, **dict(event.tags)}
            if phase_a_eval_at_start is None and event.phase != "phase_a":
                phase_a_eval_at_start = frozen_eval(agent, eval_pool, name_to_rid)
            if event.mode == "observe":
                row = observe_episode(agent, event.pair, name_to_rid)
                row.update(tags)
                episode_rows.append(row)
                simulator_observed_labels.add(event.pair.label)
                simulator_observed_recipes.add(event.pair.recipe_name)
            else:
                row, steps = assist_episode(
                    agent,
                    event.pair,
                    name_to_rid,
                    run_config=run_config,
                    event_tag=tags,
                    simulator_observed_labels=simulator_observed_labels,
                    simulator_observed_recipes=simulator_observed_recipes,
                )
                row.update(tags)
                episode_rows.append(row)
                step_rows.extend({"event_idx": idx, "baseline": baseline, **s} for s in steps)
                phase_b_count += 1
                if phase_b_count % interval == 0:
                    evals = frozen_eval(agent, phase_a_pairs, name_to_rid)
                    forgetting_rows.append({
                        "baseline": baseline,
                        "phase_b_count": phase_b_count,
                        "mean_phase_a_top1": _mean([v.get("top1", 0.0) for v in evals.values()]),
                        "mean_phase_a_top3": _mean([v.get("top3", 0.0) for v in evals.values()]),
                    })
            memory_rows.append({"event_idx": idx, "baseline": baseline, **memory_snapshot(agent)})
        wall_s = time.perf_counter() - t0
        assist_rows = [r for r in episode_rows if r.get("mode") == "assist"]
        metrics = _aggregate_episode_metrics(assist_rows)
        if phase_a_eval_at_start is None:
            phase_a_eval_at_start = frozen_eval(agent, eval_pool, name_to_rid)
        final_frozen = frozen_eval(agent, eval_pool, name_to_rid)
        bwt_fwt = bwt_fwt_checkpoints(phase_a_eval_at_start, final_frozen, [p.label for p in phase_a_pairs])
        metrics["compute"] = compute_snapshot(agent, wall_s)
        metrics["memory"] = memory_snapshot(agent)
        metrics["per_user"] = {u: _aggregate_episode_metrics([r for r in assist_rows if r.get("user_id") == u]) for u in users}
        metrics["per_user_block"] = {
            f"{u}_block_{block}": _aggregate_episode_metrics([
                r for r in assist_rows
                if r.get("user_id") == u and int(r.get("user_block", -1)) == block
            ])
            for u in users
            for block in sorted({int(r.get("user_block", -1)) for r in assist_rows if r.get("user_id") == u and r.get("user_block") is not None})
        }
        user_block_retention: Dict[str, Any] = {}
        for u in users:
            blocks = sorted({int(r.get("user_block", -1)) for r in assist_rows if r.get("user_id") == u and r.get("user_block") is not None})
            return_blocks = [b for b in blocks if b > min(blocks or [0])]
            if blocks and return_blocks:
                first_key = f"{u}_block_{blocks[0]}"
                return_key = f"{u}_block_{return_blocks[-1]}"
                first_top1 = float(metrics["per_user_block"].get(first_key, {}).get("live_top1", 0.0))
                return_top1 = float(metrics["per_user_block"].get(return_key, {}).get("live_top1", 0.0))
                user_block_retention[u] = {
                    "first_block": blocks[0],
                    "return_block": return_blocks[-1],
                    "first_block_live_top1": first_top1,
                    "return_block_live_top1": return_top1,
                    "return_minus_first_live_top1": return_top1 - first_top1,
                }
        metrics["per_user_return_retention"] = user_block_retention
        metrics["per_event_type"] = {k: _aggregate_episode_metrics([r for r in assist_rows if r.get("event_type") == k]) for k in sorted({str(r.get("event_type")) for r in assist_rows})}
        metrics["four_cell"] = {cell: _aggregate_episode_metrics([r for r in assist_rows if r.get("four_cell_before") == cell]) for cell in FOUR_CELL_KEYS}
        for cell, vals in metrics["four_cell"].items():
            for metric_name, metric_val in vals.items():
                if isinstance(metric_val, (int, float)):
                    metrics[f"{metric_name}_{cell}"] = float(metric_val)
        metrics["memory_efficiency"] = float(metrics.get("useful_assistance_rate", 0.0)) / max(1.0, math.log1p(float(metrics["memory"].get("active_variants", 0.0))))
        metrics["live_step_trace"] = live_step_trace(step_rows)
        prediction_walls = [float(s.get("prediction_wall_s")) for s in step_rows if s.get("prediction_wall_s") is not None]
        metrics["mean_prediction_wall_s"] = _mean(prediction_walls)
        metrics["p95_prediction_wall_s"] = _p95(prediction_walls)
        metrics["bwt"] = bwt_fwt["bwt"]
        metrics["fwt"] = bwt_fwt["fwt"]
        metrics["bwt_fwt_detail"] = bwt_fwt
        if baseline == "full" and phase_b_pairs:
            probe_pairs = list({p.label: p for p in phase_b_pairs}.values())[: max(1, min(20, len(phase_b_pairs)))]
            coverage_curves[baseline] = coverage_vs_accuracy_curve(agent, probe_pairs, name_to_rid, run_config, cfg_obj.coverage_curve_thresholds)
            metrics["coverage_accuracy_curve"] = coverage_curves[baseline]
        per_baseline[baseline] = metrics
        all_episode_rows.extend({"baseline": baseline, **r} for r in episode_rows)
        all_step_rows.extend(step_rows)
        all_memory_rows.extend(memory_rows)
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    _append_jsonl(out / "memory_trace.jsonl", all_memory_rows)
    _append_jsonl(out / "forgetting_trace.jsonl", forgetting_rows)
    metrics = {
        "seed": seed,
        "n_recipes": len(recipes),
        "n_users": len(users),
        "n_events": len(events),
        "n_phase_b_events": sum(1 for e in events if e.mode == "assist"),
        "selected_action_confidence_threshold": float(base_config(seed, run_config).posterior_action_confidence_threshold),
        "per_baseline": per_baseline,
        "derived_views": {
            "memory_efficiency": {b: row.get("memory_efficiency", 0.0) for b, row in per_baseline.items()},
            "compute_tradeoff": {b: row.get("compute", {}) for b, row in per_baseline.items()},
            "forgetting_curve": forgetting_rows,
            "per_user_accuracy": {b: row.get("per_user", {}) for b, row in per_baseline.items()},
            "coverage_accuracy_curve": coverage_curves,
        },
    }
    _write_json(out / "metrics.json", metrics)
    _render_seed_publication_figures(out, "deployment_stream", metrics)
    return metrics


def _axis_probe_pairs(builders: Sequence[Tuple[str, Callable[[], List[str]]]]) -> Dict[str, List[RecipePrefPair]]:
    out: Dict[str, List[RecipePrefPair]] = {}
    for axis in WORKFLOW_AXES:
        _train, holdout = _axis_holdout_split(axis)
        rows: List[RecipePrefPair] = []
        for recipe, fn in builders:
            for pref in holdout:
                try:
                    pair = materialize_pair(recipe, pref, fn)
                except Exception:
                    continue
                if pair.applied_axes:
                    rows.append(pair)
        out[axis] = rows
    return out


def _novel_composition_pairs(builders: Sequence[Tuple[str, Callable[[], List[str]]]], named_pairs: Sequence[RecipePrefPair]) -> List[Tuple[RecipePrefPair, int]]:
    train_hashes = {_order_hash(p) for p in named_pairs}
    identity = WorkflowPreference().as_dict()
    rows: List[Tuple[RecipePrefPair, int]] = []
    for recipe, fn in builders:
        for pref in _build_gap_preferences():
            try:
                pair = materialize_custom_pair(recipe, pref, fn)
            except Exception:
                continue
            if _order_hash(pair) in train_hashes:
                continue
            hamming = sum(1 for axis in WORKFLOW_AXES if pref.as_dict()[axis] != identity[axis])
            rows.append((pair, hamming))
    return rows


def _build_exclusive_diagonal(
    seed: int,
    pref_list: Sequence[str],
    n: int,
) -> Optional[Dict[str, Any]]:
    rng = _rng(seed, 6203)
    library = list(gen.recipe_library().items())
    rng.shuffle(library)
    selected: List[Tuple[str, Callable[[], List[str]], Dict[str, RecipePrefPair]]] = []
    for recipe_name, fn in library:
        by_pref: Dict[str, RecipePrefPair] = {}
        for pref in pref_list:
            try:
                by_pref[pref] = materialize_pair(recipe_name, pref, fn)
            except Exception:
                continue
        if len(by_pref) < len(pref_list):
            continue
        hashes = {_order_hash(pair) for pair in by_pref.values()}
        if len(hashes) < len(pref_list):
            continue
        selected.append((recipe_name, fn, by_pref))
        if len(selected) >= n:
            break
    if len(selected) < n:
        return None
    diagonal = [selected[i][2][pref_list[i]] for i in range(n)]
    offdiag = [
        selected[i][2][pref_list[j]]
        for i in range(n)
        for j in range(len(pref_list))
        if i != j
    ]
    all_pairs = [selected[i][2][pref] for i in range(n) for pref in pref_list]
    return {
        "recipes": [selected[i][0] for i in range(n)],
        "builders": [(selected[i][0], selected[i][1]) for i in range(n)],
        "diagonal": diagonal,
        "offdiag": offdiag,
        "all_pairs": all_pairs,
        "selected": selected,
    }


def run_cross_recipe_transfer_suite(seed: int, out: Path, run_config: RunConfig, cfg_obj: TransferSuiteConfig) -> Dict[str, Any]:
    prefs = select_preference_names(cfg_obj.n_preferences)
    n_diag = min(int(cfg_obj.n_recipes), len(prefs))
    active_prefs = tuple(prefs[:n_diag])
    diagonal_spec = _build_exclusive_diagonal(seed, active_prefs, n_diag)
    if diagonal_spec is None:
        raise RuntimeError(
            f"Cross-recipe transfer requires {n_diag} recipes with {n_diag} distinct effective preference orderings; "
            "run materiality_preflight and fix no-op preference transformations."
        )
    recipes = list(diagonal_spec["recipes"])
    builders = list(diagonal_spec["builders"])
    all_pairs = list(diagonal_spec["all_pairs"])
    diagonal = list(diagonal_spec["diagonal"])
    diag_pref_by_recipe = {p.recipe_name: p.preference_name for p in diagonal}
    offdiag = list(diagonal_spec["offdiag"])
    repeats = 1 if run_config.quick else max(1, int(cfg_obj.offdiag_repeats))
    if run_config.quick:
        offdiag = offdiag[: min(8, len(offdiag))]
    axis_probes = _axis_probe_pairs(builders) if cfg_obj.include_axis_holdout else {}
    novel_probes = _novel_composition_pairs(builders, all_pairs) if cfg_obj.include_novel_composition else []
    if run_config.quick:
        axis_probes = {k: v[: min(6, len(v))] for k, v in axis_probes.items()}
        novel_probes = novel_probes[: min(8, len(novel_probes))]
    _record_common_artifacts(out, seed, cfg_obj, recipes, active_prefs)
    scenario_rows = [
        {"mode": "observe", "phase": "shared_diagonal_train", "event_type": "diagonal_train", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for p in diagonal
    ] + [
        {"mode": "assist", "phase": "offdiagonal_probe", "event_type": "offdiagonal_transfer", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for p in offdiag
    ]
    _append_jsonl(out / "scenario_events.jsonl", scenario_rows)
    per_baseline: Dict[str, Dict[str, Any]] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    full_matrix: Dict[Tuple[str, str], float] = {}
    baselines = tuple(cfg_obj.baselines)
    for baseline in baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        pref_to_pid: Dict[str, Optional[str]] = {}
        t0 = time.perf_counter()
        for _cycle in range(max(1, int(cfg_obj.diagonal_cycles))):
            for pair in diagonal:
                observe_episode(agent, pair, name_to_rid)
                pref_to_pid[pair.preference_name] = agent.last_pref_id
        diagonal_cycle_curve: Dict[str, Dict[str, float]] = {}
        if cfg_obj.diagonal_cycle_sweep:
            curve_agent = make_agent(baseline, base_config(seed, run_config))
            curve_map: Dict[str, str] = {}
            max_cycle = max(max(1, int(c)) for c in cfg_obj.diagonal_cycle_sweep)
            wanted = {max(1, int(c)) for c in cfg_obj.diagonal_cycle_sweep}
            for cycle in range(1, max_cycle + 1):
                for pair in diagonal:
                    observe_episode(curve_agent, pair, curve_map)
                if cycle in wanted:
                    cycle_eval = frozen_eval(curve_agent, diagonal, curve_map)
                    diagonal_cycle_curve[str(cycle)] = {
                        "frozen_diagonal_top1": _mean([v.get("top1", 0.0) for v in cycle_eval.values()]),
                        "frozen_diagonal_top3": _mean([v.get("top3", 0.0) for v in cycle_eval.values()]),
                    }
        checkpoint = agent.snapshot()
        frozen_diag = frozen_eval(agent, diagonal, name_to_rid)
        frozen_off = frozen_eval(agent, offdiag, name_to_rid)
        off_rows: List[Dict[str, Any]] = []
        live_records: List[LiveEpisodeRecord] = []
        off_rows_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pair in offdiag:
            pair_rows: List[Dict[str, Any]] = []
            for rep in range(repeats):
                agent.restore_from(checkpoint)
                local_map = dict(name_to_rid)
                oracle_pid = pref_to_pid.get(pair.preference_name)
                oracle_rid = local_map.get(pair.recipe_name)
                kwargs = {"oracle_preference_id": oracle_pid} if baseline == "oracle_preference_label" else {}
                if baseline == "oracle_recipe_and_preference_label":
                    kwargs = {"oracle_preference_id": oracle_pid, "oracle_recipe_id": oracle_rid}
                row, steps = assist_episode(agent, pair, local_map, run_config=run_config, event_tag={"condition": "offdiagonal_transfer", "repeat": rep}, **kwargs)
                row.update({
                    "baseline": baseline,
                    "seen_recipe": True,
                    "seen_preference_global": pair.preference_name in pref_to_pid,
                    "seen_preference_for_recipe": False,
                    "four_cell_global": "seen_seen" if pair.preference_name in pref_to_pid else "seen_unseen",
                    "preference_transfer_correct": row.get("classification_kind") in ("known", "preference_shift", "reentry_from_pruned"),
                })
                pair_rows.append(row)
                off_rows.append(row)
                off_rows_by_label[pair.label].append(row)
                if isinstance(row.get("live_record"), LiveEpisodeRecord):
                    live_records.append(row["live_record"])
                all_episode_rows.append(row)
                all_step_rows.extend({"baseline": baseline, "condition": "offdiagonal_transfer", **s} for s in steps)
            if baseline == "full":
                full_matrix[(pair.recipe_name, pair.preference_name)] = _mean([r["live_top1"] for r in pair_rows])
        axis_metrics: Dict[str, Any] = {}
        for axis, pairs in axis_probes.items():
            rows: List[Dict[str, Any]] = []
            for pair in pairs:
                agent.restore_from(checkpoint)
                row, steps = assist_episode(agent, pair, dict(name_to_rid), run_config=run_config, event_tag={"condition": "axis_holdout", "axis": axis})
                rows.append(row)
                all_episode_rows.append({"baseline": baseline, **row})
                all_step_rows.extend({"baseline": baseline, "condition": "axis_holdout", "axis": axis, **s} for s in steps)
            axis_metrics[axis] = _aggregate_episode_metrics(rows)
        hamming_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pair, hamming in novel_probes:
            agent.restore_from(checkpoint)
            row, steps = assist_episode(agent, pair, dict(name_to_rid), run_config=run_config, event_tag={"condition": "novel_composition", "hamming": hamming})
            hamming_rows[str(hamming)].append(row)
            all_episode_rows.append({"baseline": baseline, **row})
            all_step_rows.extend({"baseline": baseline, "condition": "novel_composition", "hamming": hamming, **s} for s in steps)
        novel_metrics = {h: _aggregate_episode_metrics(rows) for h, rows in hamming_rows.items()}
        wall_s = time.perf_counter() - t0
        per_four_cell = {cell: _aggregate_episode_metrics([r for r in off_rows if r.get("four_cell_global") == cell]) for cell in FOUR_CELL_KEYS}
        live_records_by_label = {
            label: {"top1": _mean([float(r.get("live_top1", 0.0)) for r in rows])}
            for label, rows in off_rows_by_label.items()
        }
        frozen_records_by_label = {
            label: {"top1": float(row.get("top1", 0.0))}
            for label, row in frozen_off.items()
        }
        preset_of_label = {p.label: p.preference_name for p in all_pairs}
        per_baseline[baseline] = {
            **_aggregate_episode_metrics(off_rows),
            "frozen_diagonal_top1": _mean([v.get("top1", 0.0) for v in frozen_diag.values()]),
            "frozen_offdiagonal_top1": _mean([v.get("top1", 0.0) for v in frozen_off.values()]),
            "preference_transfer_rate": _mean([1.0 if r.get("preference_transfer_correct") else 0.0 for r in off_rows]),
            "per_four_cell": per_four_cell,
            **{f"live_top1_{cell}": float(per_four_cell[cell].get("live_top1", 0.0)) for cell in FOUR_CELL_KEYS},
            "primary_transfer_live_top1": float(per_four_cell["seen_seen"].get("live_top1", 0.0)),
            "zero_shot_preference_live_top1": float(per_four_cell["seen_unseen"].get("live_top1", 0.0)),
            "diagonal_cycle_curve": diagonal_cycle_curve,
            "preference_cluster_purity": preference_lock_purity(live_records, preset_of_label),
            "preference_axis_top1": preference_axis_top1(frozen_records_by_label, offdiag),
            "per_recipe_accuracy_matrix": per_recipe_accuracy_matrix(frozen_records_by_label, offdiag),
            "live_per_recipe_accuracy_matrix": per_recipe_accuracy_matrix(live_records_by_label, offdiag),
            "axis_holdout": axis_metrics,
            "novel_composition": novel_metrics,
            "compute": compute_snapshot(agent, wall_s),
            "memory": memory_snapshot(agent),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    metrics = {
        "seed": seed,
        "recipes": recipes,
        "preferences": list(active_prefs),
        "diagonal_training_pairs": [p.label for p in diagonal],
        "n_offdiag": len(offdiag),
        "n_axis_probe_pairs": sum(len(v) for v in axis_probes.values()),
        "n_novel_composition_pairs": len(novel_probes),
        "per_baseline": per_baseline,
        "derived_views": {
            "transfer_heatmap": {f"{r}/{p}": v for (r, p), v in full_matrix.items()},
            "transfer_heatmap_by_baseline": {
                b: row.get("live_per_recipe_accuracy_matrix", {})
                for b, row in per_baseline.items()
            },
            "frozen_transfer_matrix_by_baseline": {
                b: row.get("per_recipe_accuracy_matrix", {})
                for b, row in per_baseline.items()
            },
            "composition_curve": {b: row.get("novel_composition", {}) for b, row in per_baseline.items()},
        },
    }
    _write_json(out / "metrics.json", metrics)
    _render_seed_publication_figures(out, "cross_recipe_transfer", metrics)
    return metrics


def run_decay_reentry_suite(seed: int, out: Path, run_config: RunConfig, cfg_obj: DecayReentrySuiteConfig) -> Dict[str, Any]:
    n_targets = min(3, cfg_obj.n_target_recipes) if run_config.quick else cfg_obj.n_target_recipes
    target_specs = _target_variants(seed, cfg_obj.n_recipes, n_targets)
    selected_recipes = sorted({p.recipe_name for spec in target_specs for p in (spec[0], spec[1], *spec[2])})
    _record_common_artifacts(out, seed, cfg_obj, selected_recipes, PREFERENCE_NAMES)
    arms: List[Tuple[str, bool]] = []
    if cfg_obj.run_neutral_arm:
        arms.append(("neutral", True))
    if cfg_obj.run_distractor_arm:
        arms.append(("distractor", False))
    per_baseline: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    for baseline in cfg_obj.baselines:
        arm_rows: Dict[str, Dict[str, Any]] = {}
        t0 = time.perf_counter()
        last_agent_for_snapshot: Optional[AdaptiveHRCAgent] = None
        for arm_name, neutral in arms:
            gap_rows: Dict[str, Any] = {}
            for target_idx, (target, variant, distractors) in enumerate(target_specs):
                agent0 = make_agent(baseline, base_config(seed, run_config))
                name_to_rid0: Dict[str, str] = {}
                observe_episode(agent0, target, name_to_rid0)
                observe_episode(agent0, variant, name_to_rid0)
                checkpoint = agent0.snapshot()
                target_rid = name_to_rid0.get(target.recipe_name)
                target_hash = variant_hash(agent0._tokens_from_action_labels(target.actions))
                for gap in cfg_obj.gap_sweep:
                    agent = make_agent(baseline, base_config(seed, run_config))
                    agent.restore_from(checkpoint)
                    name_to_rid = dict(name_to_rid0)
                    _advance_gap(agent, int(gap), distractors, name_to_rid, neutral)
                    active_before = bool(target_rid is not None and (target_rid, target_hash) in agent.decay.active)
                    pruned_before = bool(target_rid is not None and ((target_rid, target_hash) in agent.decay.pruned or target_hash in agent.memory.variants.get(target_rid, {})) and not active_before)
                    row, steps = assist_episode(agent, target, name_to_rid, run_config=run_config, event_tag={"arm": arm_name, "gap": gap, "condition": "decay_reentry", "target_idx": target_idx})
                    post = frozen_eval(agent, [target], name_to_rid)
                    row.update({
                        "baseline": baseline,
                        "arm": arm_name,
                        "gap": gap,
                        "target": target.label,
                        "same_recipe_variant": variant.label,
                        "target_idx": target_idx,
                        "target_active_before": active_before,
                        "target_pruned_before": pruned_before,
                        "post_reentry_top1": _mean([v.get("top1", 0.0) for v in post.values()]),
                        "base_rate_after": float(agent.decay.base_rate),
                        "global_rate_after": float(agent.decay.global_rate),
                        "active_variants_end": len(agent.decay.active),
                        "pruned_variants_end": len(agent.decay.pruned),
                    })
                    gap_rows[f"{target_idx}:{gap}"] = row
                    all_rows.append(row)
                    all_steps.extend({"baseline": baseline, "arm": arm_name, **s} for s in steps)
                    last_agent_for_snapshot = agent
            per_gap_summary: Dict[str, Dict[str, float]] = {}
            for gap in cfg_obj.gap_sweep:
                rows_for_gap = [r for r in gap_rows.values() if int(r.get("gap", -1)) == int(gap)]
                per_gap_summary[str(gap)] = {
                    **_aggregate_episode_metrics(rows_for_gap),
                    "active_before_rate": _mean([1.0 if r["target_active_before"] else 0.0 for r in rows_for_gap]),
                    "pruned_before_rate": _mean([1.0 if r["target_pruned_before"] else 0.0 for r in rows_for_gap]),
                    "post_reentry_top1": _mean([r["post_reentry_top1"] for r in rows_for_gap]),
                    "global_rate_after": _mean([r["global_rate_after"] for r in rows_for_gap]),
                }
            arm_rows[arm_name] = {
                "per_target_gap": gap_rows,
                "per_gap": per_gap_summary,
                "live_top1": _mean([r["live_top1"] for r in gap_rows.values()]),
                "active_before_rate": _mean([1.0 if r["target_active_before"] else 0.0 for r in gap_rows.values()]),
                "pruned_before_rate": _mean([1.0 if r["target_pruned_before"] else 0.0 for r in gap_rows.values()]),
                "post_reentry_top1": _mean([r["post_reentry_top1"] for r in gap_rows.values()]),
                "global_rate_after": _mean([r["global_rate_after"] for r in gap_rows.values()]),
            }
        per_baseline[baseline] = {
            "arms": arm_rows,
            "live_top1": _mean([r["live_top1"] for r in arm_rows.values()]),
            "active_before_rate": _mean([r["active_before_rate"] for r in arm_rows.values()]),
            "post_reentry_top1": _mean([r["post_reentry_top1"] for r in arm_rows.values()]),
            "compute": compute_snapshot(last_agent_for_snapshot or make_agent(baseline, base_config(seed, run_config)), time.perf_counter() - t0),
            "memory": memory_snapshot(last_agent_for_snapshot or make_agent(baseline, base_config(seed, run_config))),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_steps)
    metrics = {
        "seed": seed,
        "targets": [target.label for target, _variant, _distractors in target_specs],
        "same_recipe_variants": [variant.label for _target, variant, _distractors in target_specs],
        "arms": [a for a, _ in arms],
        "per_baseline": per_baseline,
        "derived_views": {"decay_rate_trace": {b: r.get("arms", {}) for b, r in per_baseline.items()}},
    }
    _write_json(out / "metrics.json", metrics)
    _render_seed_publication_figures(out, "decay_reentry", metrics)
    return metrics


def _macro_f1(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = sorted({str(r.get("gt")) for r in rows} | {str(r.get("pred")) for r in rows})
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for label in labels:
        tp = sum(1 for r in rows if r.get("gt") == label and r.get("pred") == label)
        fp = sum(1 for r in rows if r.get("gt") != label and r.get("pred") == label)
        fn = sum(1 for r in rows if r.get("gt") == label and r.get("pred") != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": float(sum(1 for r in rows if r.get("gt") == label))}
        f1s.append(f1)
    return {"macro_f1": _mean(f1s), "accuracy": _mean([1.0 if r.get("gt") == r.get("pred") else 0.0 for r in rows]), "per_class": per_class}


def run_disambiguation_audit(seed: int, out: Path, run_config: RunConfig, cfg_obj: DisambiguationAuditConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=7)
    rng = _rng(seed, 911)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cut = max(1, int((1.0 - cfg_obj.validation_fraction) * len(order)))
    train_pairs = [pairs[i] for i in order[:cut]]
    val_pairs = [pairs[i] for i in order[cut:]]
    _record_common_artifacts(out, seed, cfg_obj, recipes, PREFERENCE_NAMES)
    per_threshold: Dict[str, Any] = {}
    for thr in cfg_obj.thresholds:
        agent = make_agent("full", base_config(seed, run_config, jaccard_threshold=float(thr)))
        name_to_rid: Dict[str, str] = {}
        for pair in train_pairs:
            observe_episode(agent, pair, name_to_rid)
        train_labels = {p.label for p in train_pairs}
        train_recipes = {p.recipe_name for p in train_pairs}
        checkpoint = agent.snapshot()
        rows = []
        gate_rows = []
        for pair in val_pairs:
            gt = _memory_state_gt(agent, pair, name_to_rid, train_labels, train_recipes)
            pred = _classify_memory_state_pair(agent, pair, name_to_rid, threshold=float(thr))
            rows.append({"threshold": thr, "pair": pair.label, "recipe": pair.recipe_name, "preference": pair.preference_name, "ground_truth": gt, "predicted": pred, "gt": gt, "pred": pred, "correct": gt == pred})
            agent.restore_from(checkpoint)
            local_map = dict(name_to_rid)
            gate_row, _steps = assist_episode(agent, pair, local_map, run_config=run_config, commit=True, event_tag={"threshold": thr, "condition": "gate_decision"})
            gt_gate = "observe" if pair.recipe_name not in name_to_rid else "assist"
            pred_gate = "observe" if bool(gate_row.get("needs_observation")) else "assist"
            gate_rows.append({"threshold": thr, "pair": pair.label, "gt": gt_gate, "pred": pred_gate, "correct": gt_gate == pred_gate})
        matrix = confusion_labels([r["predicted"] for r in rows], [r["ground_truth"] for r in rows])
        report = classifier_report(matrix, MEMORY_STATE_LABELS)
        per_threshold[str(thr)] = {**report, "confusion_matrix": matrix.tolist(), "rows": rows, "gate_decision": _gate_report(gate_rows)}
        _append_jsonl(out / "classification_rows.jsonl", rows)
        _append_jsonl(out / "gate_decision_rows.jsonl", gate_rows)
    best_threshold = max(per_threshold, key=lambda t: per_threshold[t].get("macro_f1", 0.0)) if per_threshold else None
    boundary: Dict[str, Any] = {}
    preference_axis_degradation: Dict[str, Any] = {}
    base_agent = make_agent("full", base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    for pair in train_pairs:
        observe_episode(base_agent, pair, name_to_rid)
    observed_labels = {p.label for p in train_pairs}
    observed_recipes = {p.recipe_name for p in train_pairs}
    for frac in cfg_obj.drop_fractions:
        rows = []
        for pair in val_pairs:
            dropped = _drop_interior(pair.actions, float(frac), seed)
            gt = _memory_state_gt(base_agent, pair, name_to_rid, observed_labels, observed_recipes)
            pred = _classify_memory_state_pair(base_agent, pair, name_to_rid, actions=dropped)
            rows.append({"drop_fraction": frac, "pair": pair.label, "gt": gt, "pred": pred, "correct": gt == pred})
        matrix = confusion_labels([r["pred"] for r in rows], [r["gt"] for r in rows])
        boundary[str(frac)] = {**classifier_report(matrix, MEMORY_STATE_LABELS), "confusion_matrix": matrix.tolist(), "rows": rows, "interpretation": "online_prefix_robustness"}
        _append_jsonl(out / "boundary_rows.jsonl", rows)
    identity_by_recipe = {
        recipe: materialize_pair(recipe, "identity", fn)
        for recipe, fn in select_recipe_builders(seed, cfg_obj.n_recipes)
    }
    identity_axes = WorkflowPreference().as_dict()
    by_hamming: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for pair in val_pairs:
        pref = PRESET_PREFERENCES.get(pair.preference_name)
        if pref is None:
            continue
        hamming = sum(1 for axis in WORKFLOW_AXES if pref.as_dict().get(axis) != identity_axes.get(axis))
        identity_pair = identity_by_recipe.get(pair.recipe_name)
        by_hamming[hamming].append({
            "pair": pair.label,
            "hamming": hamming,
            "jaccard_from_identity": jaccard(identity_pair.actions, pair.actions) if identity_pair else 0.0,
            "kendall_from_identity": kendall_tau_distance(identity_pair.actions, pair.actions) if identity_pair else 0.0,
        })
    for hamming, rows in sorted(by_hamming.items()):
        preference_axis_degradation[str(hamming)] = {
            "mean_jaccard_from_identity": _mean([r["jaccard_from_identity"] for r in rows]),
            "mean_kendall_from_identity": _mean([r["kendall_from_identity"] for r in rows]),
            "n": len(rows),
            "rows": rows,
        }
    train_identity = [p for p in pairs if p.preference_name == "identity"]
    probes = [p for p in pairs if p.preference_name != "identity"][: min(20, len(pairs))]
    gate_agent = make_agent("full", base_config(seed, run_config))
    gate_map: Dict[str, str] = {}
    for pair in train_identity:
        observe_episode(gate_agent, pair, gate_map)
    gate_curve = coverage_vs_accuracy_curve(gate_agent, probes, gate_map, run_config, cfg_obj.action_gate_thresholds)
    metrics = {
        "seed": seed,
        "per_threshold": per_threshold,
        "best_threshold": best_threshold,
        "boundary_degradation": boundary,
        "online_prefix_boundary_degradation": boundary,
        "preference_axis_degradation": preference_axis_degradation,
        "action_gate_curve": gate_curve,
    }
    _write_json(out / "metrics.json", metrics)
    _render_seed_publication_figures(out, "disambiguation_audit", metrics)
    return metrics


def run_materiality_preflight(seed: int, out: Path, run_config: RunConfig, cfg_obj: MaterialityPreflightConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, cfg_obj.n_preferences, require_material=False)
    prefs = select_preference_names(cfg_obj.n_preferences)
    _record_common_artifacts(out, seed, cfg_obj, recipes, prefs)
    rows = materiality_rows(pairs)
    _append_jsonl(out / "materiality_rows.jsonl", rows)
    by_recipe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pair_by_label = {p.label: p for p in pairs}
    filtered_pairs: List[str] = []
    summary: List[Dict[str, Any]] = []
    for row in rows:
        by_recipe[row["recipe"]].append(row)
        pair = pair_by_label.get(row["label"])
        if pair is not None and not pair.base_pref and not pair.applied_axes:
            filtered_pairs.append(pair.label)
    for recipe, vals in sorted(by_recipe.items()):
        effective = len({pair_by_label[v["label"]].actions for v in vals if v["label"] in pair_by_label})
        kendalls = [float(v["kendall_from_identity"]) for v in vals if v["preference"] != "identity"]
        summary.append({
            "recipe": recipe,
            "n_preferences": len(vals),
            "effective_preference_count": effective,
            "duplicate_count": sum(1 for v in vals if v["is_duplicate"]),
            "min_kendall_from_identity": min(kendalls) if kendalls else 0.0,
            "mean_kendall_from_identity": _mean(kendalls),
            "passes_min_effective": effective >= int(cfg_obj.min_effective_preferences),
        })
    pass_preflight = all(r["passes_min_effective"] for r in summary)
    metrics = {
        "seed": seed,
        "n_recipes": len(recipes),
        "n_pairs": len(pairs),
        "n_duplicate_pairs": sum(1 for r in rows if r["is_duplicate"]),
        "n_filtered_no_op_pairs": len(filtered_pairs),
        "filtered_no_op_pairs": filtered_pairs,
        "summary": summary,
        "pass_preflight": bool(pass_preflight and (not cfg_obj.fail_on_noop or not filtered_pairs)),
        "materiality_report": {
            "filtered_pairs": filtered_pairs,
            "recipe_summary": summary,
        },
    }
    if not metrics["pass_preflight"] and _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.warnings.append("materiality_preflight_failed")
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_cl_regularizer_comparison(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    cl_run = replace(run_config, baselines=("full", "adaptive", "l2_anchor", "ewc", "online_ewc", "experience_replay_bc", "no_replay"))
    return run_deployment_stream(seed, out, cl_run, DeploymentStreamConfig(n_recipes=cfg_obj.n_recipes, n_phase_b_events=cfg_obj.n_events))


EXPERIMENT_RUNNERS: Dict[str, Callable[[int, Path, RunConfig, Any], Dict[str, Any]]] = {
    "deployment_stream": run_deployment_stream,
    "cross_recipe_transfer": run_cross_recipe_transfer_suite,
    "decay_reentry": run_decay_reentry_suite,
    "disambiguation_audit": run_disambiguation_audit,
    "materiality_preflight": run_materiality_preflight,
    "cl_regularizer_comparison": run_cl_regularizer_comparison,
    "materiality_audit": run_materiality_audit,
    "single_shot_reuse": run_single_shot_reuse,
    "deployment_gate_preference_shift": run_deployment_gate_preference_shift,
    "cross_recipe_preference_transfer": run_cross_recipe_preference_transfer,
    "preference_axis_holdout": run_preference_axis_holdout,
    "novel_preference_composition": run_novel_preference_composition,
    "adaptive_decay_reentry": run_adaptive_decay_reentry,
    "multi_user_continual_stream": run_multi_user_continual_stream,
    "bounded_memory_tradeoff": run_bounded_memory_tradeoff,
    "compute_tradeoff": run_compute_tradeoff,
    "short_term_capacity_sweep": run_short_term_capacity_sweep,
    "demo_count_sample_efficiency": run_demo_count_sample_efficiency,
    "disambiguation_threshold_calibration": run_disambiguation_threshold_calibration,
    "action_gate_threshold_calibration": run_action_gate_threshold_calibration,
    "boundary_degradation_disambiguation": run_boundary_degradation_disambiguation,
    "frequency_gap_decay_sweep": run_frequency_gap_decay_sweep,
    "mwr_window_sensitivity": run_mwr_window_sensitivity,
    "reentry_gap_neutral_vs_distractor": run_reentry_gap_neutral_vs_distractor,
    "active_pruned_decay_probe": run_active_pruned_decay_probe,
    "sparse_first_exposure_pool_sweep": run_sparse_first_exposure_pool_sweep,
    "zipf_usage_sweep": run_zipf_usage_sweep,
    "cycle_width_sparsity_sweep": run_cycle_width_sparsity_sweep,
    "baseline_anchor_sweep": run_baseline_anchor_sweep,
    "continual_learning_regularizer_sweep": run_continual_learning_regularizer_sweep,
    "posterior_ablation_matrix": run_posterior_ablation_matrix,
    "recipe_preference_factor_ablation": run_recipe_preference_factor_ablation,
    "confidence_calibration_reliability": run_confidence_calibration_reliability,
    "seed_recipe_selection_audit": run_seed_recipe_selection_audit,
    "runtime_scaling_sweep": run_runtime_scaling_sweep,
}


def _experiment_config(suite_config: ExperimentSuiteConfig, name: str) -> Any:
    if not hasattr(suite_config, name):
        raise KeyError(f"ExperimentSuiteConfig has no config for {name!r}")
    return getattr(suite_config, name)


def make_run_dir(output_root: str | Path = "eval_results", timestamp_subdir: bool = True) -> Path:
    root = _ensure(Path(output_root))
    return _ensure(root / _now_stamp()) if timestamp_subdir else root


def aggregate_experiment(exp_dir: Path) -> Dict[str, Any]:
    seed_results: List[Dict[str, Any]] = []
    for path in sorted(exp_dir.glob("seed_*/result.json")):
        try:
            seed_results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    aggregate_dir = _ensure(exp_dir / "aggregate")
    if not seed_results:
        out = {"status": {"state": "failed", "n_seeds": 0}, "seeds": [], "metrics": {}}
        _write_json_file(aggregate_dir / "aggregate.json", out)
        return out
    seed_metrics = [r.get("metrics", {}) for r in seed_results if r.get("status", {}).get("state") == "completed"]
    flat_rows = [_flatten_numeric("", row) for row in seed_metrics]
    keys = sorted({k for row in flat_rows for k in row})
    failed = [r for r in seed_results if r.get("status", {}).get("state") != "completed"]
    aggregate: Dict[str, Any] = {
        "metadata": {"experiment": exp_dir.name, "created_at": _dt.datetime.now().isoformat()},
        "status": {
            "state": "completed" if not failed else ("partial" if seed_metrics else "failed"),
            "n_requested_seeds": len(seed_results),
            "n_completed_seeds": len(seed_metrics),
            "n_failed_seeds": len(failed),
            "failed_seeds": [r.get("metadata", {}).get("seed") for r in failed],
        },
        "seeds": [
            {
                "seed": r.get("metadata", {}).get("seed"),
                "state": r.get("status", {}).get("state"),
                "result_file": str(Path(f"seed_{r.get('metadata', {}).get('seed')}/result.json")),
                "warnings": r.get("warnings", []),
            }
            for r in seed_results
        ],
        "metrics": {},
        "comparisons": {},
        "warnings": sorted({w for r in seed_results for w in r.get("warnings", [])}),
    }
    rows: List[Dict[str, Any]] = []
    for key in keys:
        vals = [row[key] for row in flat_rows if key in row]
        if not vals:
            continue
        aggregate["metrics"][key] = {"mean": _mean(vals), "ci95": _stderr95(vals), "n": len(vals), "values": vals}
        rows.append({"metric": key, **aggregate["metrics"][key]})
    aggregate["metric_rows"] = rows
    full_comp: Dict[str, Any] = {}
    for r in seed_results:
        for name, vals in (r.get("comparisons", {}) or {}).items():
            full_comp.setdefault(name, defaultdict(list))
            for metric, value in vals.items():
                full_comp[name][metric].append(float(value))
    aggregate["comparisons"] = {
        name: {metric: {"mean": _mean(vals), "ci95": _stderr95(vals), "values": vals} for metric, vals in metric_map.items()}
        for name, metric_map in full_comp.items()
    }
    _render_aggregate_publication_figures(exp_dir, seed_results)
    aggregate["figures"] = [
        {"path": path.name, "bytes": int(path.stat().st_size)}
        for path in sorted(aggregate_dir.glob("*.png"))
    ]
    _write_json_file(aggregate_dir / "aggregate.json", aggregate)
    return aggregate


def aggregate_run(run_dir: str | Path) -> Dict[str, Any]:
    root = Path(run_dir)
    exp_aggs: Dict[str, Any] = {}
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("aggregate")):
        if any(exp_dir.glob("seed_*/result.json")):
            exp_aggs[exp_dir.name] = aggregate_experiment(exp_dir)
    aggregate_root = _ensure(root / "aggregate")
    run_summary = {
        "run_dir": str(root),
        "experiments": sorted(exp_aggs),
        "n_experiments": len(exp_aggs),
        "failed_or_partial_experiments": [
            name for name, agg in exp_aggs.items()
            if agg.get("status", {}).get("state") != "completed"
        ],
    }
    _write_json_file(aggregate_root / "run.json", run_summary)
    return run_summary


def render_figures(run_dir: str | Path) -> Dict[str, Any]:
    """Regenerate F-series publication figures from stored result JSON files."""
    root = Path(run_dir)
    for result_path in sorted(root.glob("*/seed_*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiment = str(result.get("metadata", {}).get("experiment") or result_path.parents[1].name)
        if experiment in set(CANONICAL_SCENARIOS) | set(APPENDIX_EXPERIMENTS):
            _render_seed_publication_figures(result_path.parent, experiment, result.get("metrics", {}) or {})
            existing = result.get("figures", {}) or {}
            refreshed: Dict[str, Any] = {}
            for path in sorted(result_path.parent.glob("*.png")):
                row = dict(existing.get(path.name, {}) or {})
                row["path"] = path.name
                row["bytes"] = int(path.stat().st_size)
                refreshed[path.name] = row
            result["figures"] = refreshed
            _write_json_file(result_path, result)
    return aggregate_run(run_dir)


def _resolve_experiments(experiments: Optional[Sequence[str]], include_appendix: bool = False) -> Tuple[str, ...]:
    if experiments is None:
        return MAIN_EXPERIMENTS + (APPENDIX_EXPERIMENTS if include_appendix else ())
    if isinstance(experiments, str):
        if experiments in {"main", "canonical"}:
            return MAIN_EXPERIMENTS
        if experiments == "appendix":
            return APPENDIX_EXPERIMENTS
        if experiments == "all":
            return ALL_EXPERIMENTS
        if experiments == "legacy":
            return LEGACY_EXPERIMENTS
        return (experiments,)
    return tuple(experiments)


def _eta_text(start_s: float, completed: int, total: int) -> str:
    if completed <= 0:
        return "ETA unknown"
    elapsed = time.perf_counter() - start_s
    remaining = max(0, total - completed)
    eta = elapsed * remaining / max(1, completed)
    return f"elapsed={elapsed:.1f}s ETA={eta:.1f}s"


def _mp_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else "spawn")


def _git_sha() -> Optional[str]:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def _materiality_preflight_failures(seed: int, cfg_obj: MaterialityPreflightConfig) -> List[str]:
    _recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, cfg_obj.n_preferences, require_material=False)
    rows = materiality_rows(pairs)
    pair_by_label = {p.label: p for p in pairs}
    by_recipe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    filtered_noops: List[str] = []
    for row in rows:
        by_recipe[row["recipe"]].append(row)
        pair = pair_by_label.get(row["label"])
        if pair is not None and not pair.base_pref and not pair.applied_axes:
            filtered_noops.append(pair.label)
    failures: List[str] = []
    for recipe, vals in sorted(by_recipe.items()):
        effective = len({pair_by_label[v["label"]].actions for v in vals if v["label"] in pair_by_label})
        if effective < int(cfg_obj.min_effective_preferences):
            failures.append(f"{recipe}:effective={effective}")
    if cfg_obj.fail_on_noop and filtered_noops:
        failures.append(f"noop_pairs={len(filtered_noops)}")
    return failures


def _run_seed_job(name: str, seed: int, run_dir: str, run_config: RunConfig, cfg_obj: Any) -> Dict[str, Any]:
    exp_dir = _ensure(Path(run_dir) / name)
    seed_dir = _ensure(exp_dir / f"seed_{seed}")
    runner = EXPERIMENT_RUNNERS[name]
    collector = _SeedResultCollector(name, int(seed), seed_dir, cfg_obj)
    _set_active_result(collector)
    t0 = time.perf_counter()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        metrics = runner(int(seed), seed_dir, run_config, cfg_obj)
        status = {
            "state": "completed",
            "wall_s": float(time.perf_counter() - t0),
            "error": None,
        }
        result = collector.finalize(metrics, status)
        return {"ok": True, "seed": int(seed), "wall_s": status["wall_s"], "result_file": str(seed_dir / "result.json"), "warnings": result.get("warnings", [])}
    except Exception as exc:
        status = {
            "state": "failed",
            "wall_s": float(time.perf_counter() - t0),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        collector.warnings.append("seed_failed")
        collector.finalize({}, status)
        return {"ok": False, "seed": int(seed), "wall_s": status["wall_s"], "result_file": str(seed_dir / "result.json"), "error": status["error"]}
    finally:
        _set_active_result(None)
        plt.close("all")


def run_suite(
    run_config: Optional[RunConfig] = None,
    suite_config: Optional[ExperimentSuiteConfig] = None,
    experiments: Optional[Sequence[str]] = None,
    include_appendix: bool = False,
) -> Dict[str, Any]:
    run_config = run_config or RunConfig()
    suite_config = suite_config or ExperimentSuiteConfig()
    names = _resolve_experiments(experiments, include_appendix=include_appendix)
    unknown = [name for name in names if name not in EXPERIMENT_RUNNERS]
    if unknown:
        raise KeyError(f"unknown experiments: {unknown}; valid={list(EXPERIMENT_RUNNERS)}")
    run_dir = make_run_dir(run_config.output_root, run_config.timestamp_subdir)
    workers = int(run_config.workers) if run_config.workers else max(1, len(run_config.seeds))
    manifest = {
        "created_at": _dt.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "run_config": run_config,
        "suite_config": suite_config,
        "main_experiments": MAIN_EXPERIMENTS,
        "appendix_experiments": APPENDIX_EXPERIMENTS,
        "requested_experiments": names,
        "selected_recipes_by_seed": {str(seed): [r for r, _ in stable_recipe_builders(seed)] for seed in run_config.seeds},
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "workers": workers,
        "status": {"state": "running", "completed": [], "partial": [], "failed": []},
    }
    _write_json_file(run_dir / "run.json", manifest)
    if (
        suite_config.materiality_preflight.fail_on_noop
        and any(name in set(CANONICAL_SCENARIOS) - {"materiality_preflight"} for name in names)
    ):
        failures_by_seed = {
            int(seed): _materiality_preflight_failures(int(seed), suite_config.materiality_preflight)
            for seed in run_config.seeds
        }
        failures_by_seed = {seed: failures for seed, failures in failures_by_seed.items() if failures}
        if failures_by_seed:
            manifest["status"] = {
                "state": "failed",
                "completed": [],
                "partial": [],
                "failed": ["materiality_preflight"],
            }
            manifest["materiality_preflight_failures"] = failures_by_seed
            _write_json_file(run_dir / "run.json", manifest)
            raise RuntimeError(
                "Materiality preflight failed before canonical scenario execution: "
                f"{failures_by_seed}. Fix the recipe library or lower min_effective_preferences."
            )
    completed: List[str] = []
    partial: List[str] = []
    failures: Dict[str, str] = {}
    total_jobs = len(names) * len(run_config.seeds)
    done_jobs = 0
    suite_t0 = time.perf_counter()
    for name in names:
        exp_dir = _ensure(run_dir / name)
        cfg_obj = _experiment_config(suite_config, name)
        print(f"[{name}] start: {len(run_config.seeds)} seeds, workers={workers}", flush=True)
        seed_results: List[Dict[str, Any]] = []
        if workers <= 1 or len(run_config.seeds) <= 1:
            for seed in run_config.seeds:
                r = _run_seed_job(name, int(seed), str(run_dir), run_config, cfg_obj)
                seed_results.append(r)
                done_jobs += 1
                state = "done" if r.get("ok") else "FAILED"
                print(f"  [{name}] seed {seed} {state} in {r.get('wall_s', 0.0):.1f}s ({done_jobs}/{total_jobs}, {_eta_text(suite_t0, done_jobs, total_jobs)})", flush=True)
        else:
            ctx = _mp_context()
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                fut_map = {pool.submit(_run_seed_job, name, int(seed), str(run_dir), run_config, cfg_obj): int(seed) for seed in run_config.seeds}
                for fut in as_completed(fut_map):
                    seed = fut_map[fut]
                    try:
                        r = fut.result()
                    except BaseException as exc:
                        r = {"ok": False, "seed": seed, "error": f"{type(exc).__name__}: {exc}", "wall_s": 0.0}
                    seed_results.append(r)
                    done_jobs += 1
                    state = "done" if r.get("ok") else "FAILED"
                    if not r.get("ok"):
                        failures[f"{name}/seed_{seed}"] = str(r.get("error", "unknown error"))
                    print(f"  [{name}] seed {seed} {state} in {r.get('wall_s', 0.0):.1f}s ({done_jobs}/{total_jobs}, {_eta_text(suite_t0, done_jobs, total_jobs)})", flush=True)
        for r in seed_results:
            if not r.get("ok"):
                failures[f"{name}/seed_{r.get('seed')}"] = str(r.get("error", "unknown error"))
        aggregate = aggregate_experiment(exp_dir)
        state = aggregate.get("status", {}).get("state", "failed")
        if state == "completed":
            completed.append(name)
        elif state == "partial":
            partial.append(name)
        else:
            failures.setdefault(name, "all seeds failed")
        manifest["status"] = {"state": "running", "completed": completed, "partial": partial, "failed": sorted(failures)}
        _write_json_file(run_dir / "run.json", manifest)
        print(f"[{name}] {state}", flush=True)
    run_summary = aggregate_run(run_dir)
    result = ExperimentResult(str(run_dir), tuple(completed), failures).as_dict()
    result["partial_experiments"] = partial
    result["run_summary"] = run_summary
    manifest["status"] = {
        "state": "completed" if not failures and not partial else ("partial" if completed or partial else "failed"),
        "completed": completed,
        "partial": partial,
        "failed": sorted(failures),
    }
    manifest["result"] = result
    _write_json_file(run_dir / "run.json", manifest)
    return result


def run_main(run_config: Optional[RunConfig] = None, suite_config: Optional[ExperimentSuiteConfig] = None) -> Dict[str, Any]:
    return run_suite(run_config=run_config, suite_config=suite_config, experiments=MAIN_EXPERIMENTS)


def run_appendix(run_config: Optional[RunConfig] = None, suite_config: Optional[ExperimentSuiteConfig] = None) -> Dict[str, Any]:
    return run_suite(run_config=run_config, suite_config=suite_config, experiments=APPENDIX_EXPERIMENTS)


def main(
    *,
    run_config: Optional[RunConfig] = None,
    suite_config: Optional[ExperimentSuiteConfig] = None,
    experiments: Optional[Sequence[str]] = None,
    include_appendix: bool = False,
) -> Dict[str, Any]:
    return run_suite(run_config=run_config, suite_config=suite_config, experiments=experiments, include_appendix=include_appendix)


__all__ = [
    "RunConfig",
    "ExperimentSuiteConfig",
    "CANONICAL_SCENARIOS",
    "DERIVED_VIEWS",
    "MAIN_EXPERIMENTS",
    "APPENDIX_EXPERIMENTS",
    "LEGACY_EXPERIMENTS",
    "ALL_EXPERIMENTS",
    "DeploymentStreamConfig",
    "TransferSuiteConfig",
    "DecayReentrySuiteConfig",
    "DisambiguationAuditConfig",
    "MaterialityPreflightConfig",
    "LiveStepRecord",
    "LiveEpisodeRecord",
    "binary_decision_metrics",
    "live_episode_metrics",
    "aggregate_live_episodes",
    "coverage_vs_accuracy_curve",
    "run_suite",
    "run_main",
    "run_appendix",
    "render_figures",
    "aggregate_run",
    "main",
]
