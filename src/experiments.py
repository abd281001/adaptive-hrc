"""Clean experimental harness for the adaptive HRC project.

The paper-facing benchmark keeps each experiment under one current API: a
logical name, a typed configuration, deterministic seed-based scenario
selection, seed-level ``result.json`` files, aggregate experiment summaries,
and structured logs for external analysis.
"""
from __future__ import annotations
import datetime as _dt
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
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
import numpy as np

from .adaptive_agent import AdaptiveHRCAgent
from .baselines import BASELINE_AGENTS, OracleCeilingAgent
from .environment import gen
from .hrc_simulation import DEFAULT_HRC_TIMING, HRCDecisionContext, policy_distribution_diagnostics, run_alternating_hrc_episode
from .memory import VariantKey, jaccard, kendall_tau_distance, variant_hash
from .models import Config, DEFAULT_CONFIG
from .preferences import (AXES as WORKFLOW_AXES, AXIS_VALUES as WORKFLOW_AXIS_VALUES, PRESET_PREFERENCES, PREFERENCE_NAMES, WorkflowPreference, WorkflowPreferenceModifier)
from .representations import observations_from_actions


DEFAULT_SEEDS: Tuple[int, ...] = (1337, 2024, 7, 9001, 31415)
PAPER_BASELINES: Tuple[str, ...] = (
    "full",
    "no_decay",
    "fixed_decay",
    "latest_only",
    "bc",
    "experience_replay_bc",
    "recency_prioritized_replay",
    "ewc",
    "online_ewc",
    "bigram",
)
ORACLE_REFERENCE_BASELINES: Tuple[str, ...] = (
    "oracle_recipe_label",
    "oracle_preference_label",
    "oracle_recipe_and_preference_label",
    "oracle_ceiling",
)
TRANSFER_BASELINES: Tuple[str, ...] = (
    "full",
    "latest_only",
    "bc",
    "experience_replay_bc",
    "recency_prioritized_replay",
    "ewc",
    "online_ewc",
    "bigram",
    "no_preference_prototype",
)

EXPERIMENTS: Tuple[str, ...] = (
    "materiality_preflight",
    "deployment_stream",
    "deployment_ladder_heterogeneous_prefs",
    "deployment_ladder_shared_pref",
    "cross_recipe_transfer",
    "confidence_threshold_accuracy",
    "zipf_usage_sweep",
    "demo_count_sample_efficiency",
    "adversarial_stress_suite",
    "baseline_hyperparameter_tuning",
    "single_shot_reuse",
    "deployment_gate_preference_shift",
    "short_term_capacity_sweep",
    "frequency_gap_decay_sweep",
    "mwr_window_sensitivity",
    "sparse_first_exposure_pool_sweep",
    "cycle_width_sparsity_sweep",
    "baseline_anchor_sweep",
    "continual_learning_regularizer_sweep",
    "cl_regularizer_comparison",
    "posterior_ablation_matrix",
    "recipe_preference_factor_ablation",
    "memory_exhaustion_stress",
    "prefix_collision_stress",
    "preference_thrash_stress",
    "rare_reentry_stress",
    "late_distractor_stress",
    "seed_recipe_selection_audit",
    "runtime_scaling_sweep",
    "catastrophic_vs_selective_forgetting",
    "adaptation_speed_curve",
    "decay_reentry",
    "disambiguation_audit",
)

FOUR_CELL_KEYS: Tuple[str, ...] = (
    "known_recipe_known_preference",
    "seen_recipe_new_preference",
    "unseen_recipe_seen_preference",
    "unseen_unseen",
)
TRANSFER_CELL_KEYS: Tuple[str, ...] = (
    "direct_retrieval",
    "seen_recipe_preference_seen_elsewhere",
    "seen_recipe_new_preference",
    "unseen_recipe_seen_preference",
    "unseen_unseen",
)
PRIMARY_EVENT_TYPES: Tuple[str, ...] = (
    "routine_reuse",
    "preference_shift",
    "cross_transfer_probe",
    "rare_reentry",
)
ASSIST_WRONG_PENALTY: float = 1.0
OBSERVATION_MODE_EXTRA_TIME_PER_STEP: float = 1.0

SCENARIO_NARRATIVES: Dict[str, Dict[str, str]] = {
    "deployment_stream": {
        "claim": "A single deployment stream can test routine reuse, preference shifts, user switches, transfer probes, reentry, memory, compute, and per-step latency without redundant reruns.",
        "pressure": "Shared Phase A followed by a structured Phase B event mix with online forgetting traces, user-return blocks, latency logging, and confidence-threshold diagnostics.",
    },
    "deployment_ladder_heterogeneous_prefs": {
        "claim": "A deployed assistant should adapt when every known recipe receives a recipe-specific preference update in synchronized ladder rungs.",
        "pressure": "First rung observes one variant per recipe; subsequent rungs present all recipes in assist mode with per-recipe seed-shuffled preference shifts followed by short settle phases.",
    },
    "deployment_ladder_shared_pref": {
        "claim": "A deployed assistant should exploit a globally shared preference style when all recipes shift together, then stabilize after each synchronized update.",
        "pressure": "First rung observes the same seed-shuffled preference across recipes; subsequent rungs use a new shared preference for all recipes in assist mode followed by settle phases.",
    },
    "cross_recipe_transfer": {
        "claim": "Preference structure should transfer across recipes, across held-out axis values, and under whole-axis family holdout.",
        "pressure": "Three explicit conditions are reported separately: diagonal-to-offdiagonal recipe/preference transfer, cross-axis-value transfer across recipe splits, and axis-family holdout where non-default values for one preference axis are absent from training.",
    },
    "confidence_threshold_accuracy": {
        "claim": "The assistant should know when its scheduled robot-turn predictions are reliable enough to be useful.",
        "pressure": "Posthoc confidence-threshold accuracy curves over alternating-action robot turns; thresholds are reporting cuts on confidence, not action-gate policies.",
    },
    "decay_reentry": {
        "claim": "Adaptive decay should prune stale variants while recovering when a task re-enters after neutral time or distractor growth.",
        "pressure": "Neutral and distractor gap arms from the same prefix checkpoint.",
    },
    "disambiguation_audit": {
        "claim": "Classification thresholds should be justified by held-out calibration, not hard-coded by inspection.",
        "pressure": "Jaccard threshold sweep, online-prefix degradation, active-memory availability diagnostics, and confidence calibration.",
    },
    "materiality_preflight": {
        "claim": "Preference transformations must materially change action orderings before learning claims are interpretable.",
        "pressure": "Seed-level duplicate/no-op filtering and per-recipe discriminability checks.",
    },
    "single_shot_reuse": {
        "claim": "After one observation, the robot should assist on first reuse without re-entering observation mode.",
        "pressure": "First online reuse after a single observed demonstration.",
    },
    "deployment_gate_preference_shift": {
        "claim": "For a known recipe, the robot should remain in assist mode while adapting to a new material preference.",
        "pressure": "Known-recipe, new-preference assist-mode probes only; unseen recipes are handled by deployment_stream observation events.",
    },
    "zipf_usage_sweep": {
        "claim": "A deployed kitchen assistant should remain useful when demand concentrates on house favorites without losing rare customized workflows.",
        "pressure": "Balanced prep-book onboarding followed by branched rush-hour Zipf streams; utility is demand-weighted while fairness evaluates the whole menu, including unsampled tail workflows.",
    },
}

DEPLOYMENT_EVENT_NARRATIVES: Dict[str, Dict[str, str]] = {
    "routine_reuse": {
        "claim": "Previously demonstrated recipe-preference pairs should remain directly reusable.",
        "pressure": "Assist episodes reuse the current user/recipe pair without a preference change.",
    },
    "preference_shift": {
        "claim": "A known recipe may be reused with a different preference ordering.",
        "pressure": "Assist episodes keep the recipe fixed while changing the preference label.",
    },
    "user_switch": {
        "claim": "Persistent user identity should modulate expected preferences across blocks.",
        "pressure": "Blocked user returns test retention after other users intervene.",
    },
    "cross_transfer_probe": {
        "claim": "Known preferences should transfer to recipes where that preference has not been shown.",
        "pressure": "Probe pairs require both a seen recipe and a globally seen preference in a new combination.",
    },
    "rare_reentry": {
        "claim": "Stale or likely-pruned tasks should be recoverable after long absence.",
        "pressure": "Probe pairs are sampled from older stream history beyond the decay horizon.",
    },
    "new_recipe_observe": {
        "claim": "The stream should continue acquiring genuinely new recipes during deployment.",
        "pressure": "Observation events introduce recipes outside the initial Phase-A recipe set.",
    },
    "ladder_initial_observe": {
        "claim": "The first ladder rung seeds recipe identities through normal observation.",
        "pressure": "Each selected recipe is demonstrated exactly once before all later ladder additions switch to assist mode.",
    },
    "ladder_pref_update_heterogeneous": {
        "claim": "Known recipes may each receive their own updated preference at the same deployment time.",
        "pressure": "Every recipe is presented in assist mode with a new seed-shuffled preference distinct from its prior rung preference.",
    },
    "ladder_shared_pref_shift": {
        "claim": "A shared style shift across all recipes should be reusable as a cross-recipe preference signal.",
        "pressure": "Every recipe is presented in assist mode under the same new preference label for that rung.",
    },
    "ladder_settle": {
        "claim": "After a synchronized preference update, short repeated use should stabilize online inference and assistance.",
        "pressure": "Assist-only repeats over the current rung's recipe-preference set measure recovery after the ladder step.",
    },
}


@dataclass(frozen=True)
class RunConfig:
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    output_root: str = "eval_results"
    timestamp_subdir: bool = True
    workers: Optional[int] = None
    baselines: Tuple[str, ...] = PAPER_BASELINES
    profile: bool = False
    topk: int = 3
    log_full_distributions: bool = False
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
    baselines: Tuple[str, ...] = TRANSFER_BASELINES


@dataclass(frozen=True)
class MultiUserStreamConfig:
    n_users: int = 5
    n_recipes: int = 10
    n_events: int = 120
    switch_probability: float = 0.45
    zipf_alpha: float = 1.2


@dataclass(frozen=True)
class ComputeTradeoffConfig:
    n_recipes: int = 8
    n_events: int = 60


@dataclass(frozen=True)
class DeploymentStreamConfig:
    n_recipes: int = 10
    n_users: int = 4
    n_phase_b_events: int = 250
    zipf_alpha: float = 1.2
    event_mix: Dict[str, float] = field(default_factory=lambda: {
        "routine_reuse": 0.44,
        "preference_shift": 0.14,
        "user_switch": 0.18,
        "cross_transfer_probe": 0.14,
        "rare_reentry": 0.10,
    })
    phase_a_non_identity_prob: float = 0.50
    new_recipe_obs_rate: float = 0.05
    user_block_size: int = 20
    transfer_warmup_events: int = 30
    event_mix_sweep_routine_fracs: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)


@dataclass(frozen=True)
class DeploymentLadderConfig:
    n_recipes: int = 10
    n_rungs: Optional[int] = None
    max_rungs: int = 4
    settle_repeats_per_recipe: int = 2
    preferences: Tuple[str, ...] = PREFERENCE_NAMES


@dataclass(frozen=True)
class TransferSuiteConfig:
    n_recipes: int = 7
    n_preferences: int = 7
    diagonal_cycles: int = 1
    offdiag_repeats: int = 2
    diagonal_cycle_sweep: Tuple[int, ...] = (1, 2, 3, 5)
    include_cross_axis_value_transfer: bool = True
    include_axis_family_holdout: bool = True
    include_novel_composition: bool = True
    baselines: Tuple[str, ...] = TRANSFER_BASELINES


@dataclass(frozen=True)
class DecayReentrySuiteConfig:
    n_recipes: int = 12
    gap_sweep: Tuple[int, ...] = (0, 5, 10, 15, 20, 30)
    distractor_counts: Tuple[int, ...] = (0, 2, 5, 10)
    n_target_recipes: int = 5
    run_neutral_arm: bool = True
    run_distractor_arm: bool = True
    baselines: Tuple[str, ...] = PAPER_BASELINES


@dataclass(frozen=True)
class DisambiguationAuditConfig:
    n_recipes: int = 12
    thresholds: Tuple[float, ...] = (0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98, 0.99)
    online_partial_thresholds: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    min_classify_lengths: Tuple[int, ...] = (3, 6, 9)
    min_classify_fractions: Tuple[float, ...] = (0.25, 0.35, 0.50)
    validation_fraction: float = 0.30
    drop_fractions: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
    prefix_fractions: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    posterior_alpha_grid: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    posterior_global_temperature_grid: Tuple[float, ...] = (0.5, 1.0, 2.0)
    posterior_calibration_max_pairs: int = 12
    posterior_calibration_max_grid_points: int = 75


@dataclass(frozen=True)
class MaterialityPreflightConfig:
    n_recipes: int = 15
    n_preferences: int = 7
    min_effective_preferences: int = 3
    min_classify_length: int = 6
    fail_on_noop: bool = True
    fail_on_duplicate_ordering: bool = True
    near_duplicate_tau_threshold: float = 0.05


@dataclass(frozen=True)
class CapacitySweepConfig:
    demo_counts: Tuple[int, ...] = (1, 3, 5, 10, 15)
    n_preferences: int = 3


@dataclass(frozen=True)
class ConfidenceThresholdAccuracyConfig:
    n_recipes: int = 7
    n_preferences: int = 7
    diagonal_cycles: int = 1
    offdiag_repeats: int = 2
    confidence_thresholds: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    baselines: Tuple[str, ...] = TRANSFER_BASELINES


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
    n_preferences: int = 4
    n_events: int = 120
    alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    phase_a_cycles: int = 1


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
class BaselineHyperparameterTuningConfig:
    n_recipes: int = 8
    n_preferences: int = 4
    baselines: Tuple[str, ...] = ("bc", "experience_replay_bc", "recency_prioritized_replay", "ewc", "online_ewc")
    validation_fraction: float = 0.35
    max_eval_pairs: int = 24
    max_grid_points_per_baseline: int = 24
    bc_learning_rates: Tuple[float, ...] = (0.03, 0.10)
    bc_l2_values: Tuple[float, ...] = (1e-4, 1e-3)
    bc_history_lens: Tuple[int, ...] = (2, 3)
    bc_history_bins_values: Tuple[int, ...] = (32, 64)
    bc_epoch_pairs: Tuple[Tuple[int, int], ...] = ((60, 30), (120, 60))
    er_buffer_sizes: Tuple[int, ...] = (64, 256)
    er_batch_sizes: Tuple[int, ...] = (64, 256)
    er_recency_alphas: Tuple[float, ...] = (0.5, 1.0, 2.0)
    er_uniform_mixes: Tuple[float, ...] = (0.05,)
    ewc_lambdas: Tuple[float, ...] = (0.1, 0.4, 1.0)
    maxent_learning_rates: Tuple[float, ...] = (0.02, 0.05)
    maxent_l2_values: Tuple[float, ...] = (0.005, 0.01)
    maxent_iter_pairs: Tuple[Tuple[int, int], ...] = ((50, 20), (100, 40))


@dataclass(frozen=True)
class RuntimeScalingConfig:
    recipe_counts: Tuple[int, ...] = (3, 5, 10, 15)
    n_events: int = 50


@dataclass(frozen=True)
class ExperimentSuiteConfig:
    deployment_stream: DeploymentStreamConfig = field(default_factory=DeploymentStreamConfig)
    deployment_ladder_heterogeneous_prefs: DeploymentLadderConfig = field(default_factory=DeploymentLadderConfig)
    deployment_ladder_shared_pref: DeploymentLadderConfig = field(default_factory=DeploymentLadderConfig)
    cross_recipe_transfer: TransferSuiteConfig = field(default_factory=TransferSuiteConfig)
    confidence_threshold_accuracy: ConfidenceThresholdAccuracyConfig = field(default_factory=ConfidenceThresholdAccuracyConfig)
    decay_reentry: DecayReentrySuiteConfig = field(default_factory=DecayReentrySuiteConfig)
    disambiguation_audit: DisambiguationAuditConfig = field(default_factory=DisambiguationAuditConfig)
    materiality_preflight: MaterialityPreflightConfig = field(default_factory=MaterialityPreflightConfig)
    single_shot_reuse: SingleShotReuseConfig = field(default_factory=SingleShotReuseConfig)
    deployment_gate_preference_shift: DeploymentGateConfig = field(default_factory=DeploymentGateConfig)
    short_term_capacity_sweep: CapacitySweepConfig = field(default_factory=CapacitySweepConfig)
    demo_count_sample_efficiency: CapacitySweepConfig = field(default_factory=CapacitySweepConfig)
    frequency_gap_decay_sweep: GapSweepConfig = field(default_factory=GapSweepConfig)
    mwr_window_sensitivity: MWRWindowSensitivityConfig = field(default_factory=MWRWindowSensitivityConfig)
    sparse_first_exposure_pool_sweep: SparsePoolConfig = field(default_factory=SparsePoolConfig)
    zipf_usage_sweep: ZipfUsageConfig = field(default_factory=ZipfUsageConfig)
    cl_regularizer_comparison: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    cycle_width_sparsity_sweep: CycleWidthConfig = field(default_factory=CycleWidthConfig)
    baseline_anchor_sweep: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    continual_learning_regularizer_sweep: ComputeTradeoffConfig = field(default_factory=ComputeTradeoffConfig)
    posterior_ablation_matrix: CrossRecipeTransferConfig = field(default_factory=CrossRecipeTransferConfig)
    recipe_preference_factor_ablation: CrossRecipeTransferConfig = field(default_factory=CrossRecipeTransferConfig)
    memory_exhaustion_stress: StressConfig = field(default_factory=StressConfig)
    prefix_collision_stress: StressConfig = field(default_factory=StressConfig)
    preference_thrash_stress: StressConfig = field(default_factory=StressConfig)
    rare_reentry_stress: StressConfig = field(default_factory=StressConfig)
    late_distractor_stress: StressConfig = field(default_factory=StressConfig)
    adversarial_stress_suite: StressConfig = field(default_factory=StressConfig)
    baseline_hyperparameter_tuning: BaselineHyperparameterTuningConfig = field(default_factory=BaselineHyperparameterTuningConfig)
    seed_recipe_selection_audit: MaterialityAuditConfig = field(default_factory=MaterialityAuditConfig)
    runtime_scaling_sweep: RuntimeScalingConfig = field(default_factory=RuntimeScalingConfig)
    catastrophic_vs_selective_forgetting: Any = field(default_factory=lambda: CatastrophicSelectiveConfig())
    adaptation_speed_curve: Any = field(default_factory=lambda: AdaptationSpeedConfig())


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
    inferred_variant_hash: Optional[str]
    inferred_latent_pref_id: Optional[str] = None
    posterior_entropy: Optional[float] = None
    posterior_confidence: Optional[float] = None
    posterior_raw_entropy: Optional[float] = None
    posterior_max_prob: Optional[float] = None
    posterior_n_hypotheses: Optional[int] = None
    would_assist_under_gate: bool = False
    policy_confidence: Optional[float] = None
    policy_raw_confidence: Optional[float] = None
    policy_gate_score: Optional[float] = None
    policy_margin: Optional[float] = None
    gate_threshold: Optional[float] = None
    policy_entropy: Optional[float] = None
    gate_source: Optional[str] = None
    gate_reason: Optional[str] = None
    policy_final_confidence: Optional[float] = None
    policy_final_margin: Optional[float] = None
    policy_final_entropy: Optional[float] = None
    conditioned_policy_confidence: Optional[float] = None
    ensemble_policy_confidence: Optional[float] = None
    policy_agreement: Optional[bool] = None
    blend_strength: Optional[float] = None
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
    first_mismatch_robot_turn: Optional[int]


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
        21: 2.08,
        22: 2.07,
        23: 2.07,
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
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return float("nan"), 0.0
    return _mean(vals), _stderr95(vals)


def _p95(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = int(math.ceil(0.95 * len(vals))) - 1
    return float(vals[max(0, min(idx, len(vals) - 1))])


def _histogram(values: Sequence[float], bins: Sequence[float] = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)) -> Dict[str, Any]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    edges = [float(v) for v in bins]
    if len(edges) < 2:
        edges = [0.0, 1.0]
    counts = [0 for _ in range(len(edges) - 1)]
    for val in vals:
        placed = False
        for idx in range(len(edges) - 1):
            lo, hi = edges[idx], edges[idx + 1]
            if (val >= lo and val < hi) or (idx == len(edges) - 2 and val <= hi):
                counts[idx] += 1
                placed = True
                break
        if not placed and val > edges[-1]:
            counts[-1] += 1
    return {
        "bins": [
            {"low": edges[i], "high": edges[i + 1], "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "n": int(len(vals)),
    }


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
    n_eff_bins = max(1, min(int(n_bins), len(s_norm)))
    ordered = np.argsort(s_norm)
    bins = [idx for idx in np.array_split(ordered, n_eff_bins) if len(idx) > 0]
    calibration: List[Tuple[float, float, int]] = []
    ece = 0.0
    for idx in bins:
        mean_score = float(s_norm[idx].mean())
        empirical = float(y[idx].mean())
        count = int(len(idx))
        ece += (count / len(s_norm)) * abs(mean_score - empirical)
        calibration.append((mean_score, empirical, count))
    brier = float(np.mean((s_norm - y) ** 2))
    return {"auroc": float(auroc), "auprc": float(auprc), "ece": float(ece), "brier": brier, "calibration": calibration}


def behavioral_steps_to_lock(
    steps: Sequence[LiveStepRecord],
    window_fraction: float = 0.10,
    threshold: float = 0.75,
) -> int:
    """Step at which the agent first achieves sustained behavioral accuracy.

    The window size is ``max(2, min(5, round(window_fraction * n)))`` so short
    and long recipes are treated proportionally rather than using a fixed count.
    Returns -1 when the episode is too short to produce a meaningful window or
    the threshold is never reached.
    """
    n = len(steps)
    if n < 2:
        return -1
    width = max(2, min(5, round(float(window_fraction) * n)))
    if n < width:
        return -1
    for k in range(n - width + 1):
        chunk = steps[k:k + width]
        if _mean([1.0 if s.correct_top1 else 0.0 for s in chunk]) >= float(threshold):
            return k
    return -1


def posterior_steps_to_lock(steps: Sequence[LiveStepRecord], true_rid: Optional[str]) -> int:
    if true_rid is None:
        return -1
    for k in range(len(steps)):
        if all(steps[j].inferred_recipe == true_rid for j in range(k, len(steps))):
            return k
    return -1


ADAPTATION_RECOVERY_WINDOWS = (1, 2, 3)
FIRST_MISMATCH_POSITION_BINS = 10


def _adaptation_recovery_metrics(
    correct_flags: Sequence[bool],
    first_mismatch_robot_turn: Optional[int],
) -> Dict[str, float]:
    """Fixed-window recovery after the first wrong robot turn.

    Recovery windows are censored unless the episode has the full requested
    number of later robot turns. That keeps late first mismatches from being
    averaged with less evidence than early first mismatches.
    """
    flags = [bool(v) for v in correct_flags]
    n = len(flags)
    try:
        fm = int(first_mismatch_robot_turn) if first_mismatch_robot_turn is not None else -1
    except (TypeError, ValueError):
        fm = -1
    has_mismatch = 0 <= fm < n
    after = max(0, n - fm - 1) if has_mismatch else 0
    out: Dict[str, float] = {
        "first_mismatch_rate": 1.0 if has_mismatch else 0.0,
        "no_mismatch_rate": 1.0 if n > 0 and not has_mismatch else 0.0,
        "first_mismatch_robot_turn": float(fm if has_mismatch else -1),
        "first_mismatch_robot_turn_normalized_position": (
            float(fm / max(1, n - 1)) if has_mismatch else -1.0
        ),
        "robot_turns_after_first_mismatch": float(after),
    }
    for window in ADAPTATION_RECOVERY_WINDOWS:
        eligible = has_mismatch and after >= window
        censored = has_mismatch and not eligible
        segment = flags[fm + 1:fm + 1 + window] if eligible else []
        out[f"adaptation_recovery_top1_w{window}"] = _mean([1.0 if v else 0.0 for v in segment]) if eligible else 0.0
        out[f"adaptation_recovery_eligible_w{window}"] = 1.0 if eligible else 0.0
        out[f"adaptation_recovery_censored_w{window}"] = 1.0 if censored else 0.0
        out[f"adaptation_recovery_support_w{window}"] = float(len(segment))
    return out


def _row_first_mismatch_robot_turn(row: Mapping[str, Any]) -> float:
    value = row.get("first_mismatch_robot_turn", -1.0)
    return float(value if isinstance(value, (int, float)) else -1.0)


def _row_first_mismatch_robot_turn_position(row: Mapping[str, Any]) -> float:
    value = row.get("first_mismatch_robot_turn_normalized_position", -1.0)
    return float(value if isinstance(value, (int, float)) else -1.0)


def _adaptation_recovery_window_curve(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    mismatch_rows = [r for r in rows if float(r.get("first_mismatch_rate", 0.0)) > 0.0]
    out: List[Dict[str, float]] = []
    for window in ADAPTATION_RECOVERY_WINDOWS:
        eligible_key = f"adaptation_recovery_eligible_w{window}"
        censored_key = f"adaptation_recovery_censored_w{window}"
        top1_key = f"adaptation_recovery_top1_w{window}"
        support_key = f"adaptation_recovery_support_w{window}"
        eligible = [r for r in mismatch_rows if float(r.get(eligible_key, 0.0)) > 0.0]
        support = [max(1.0, float(r.get(support_key, window))) for r in eligible]
        denom = max(1.0, sum(support))
        recovery = (
            sum(float(r.get(top1_key, 0.0)) * w for r, w in zip(eligible, support)) / denom
            if eligible else 0.0
        )
        out.append({
            "window": float(window),
            "top1": float(recovery),
            "eligible_rate": float(len(eligible) / max(1, len(mismatch_rows))),
            "censored_rate": float(sum(float(r.get(censored_key, 0.0)) for r in mismatch_rows) / max(1, len(mismatch_rows))),
            "n_eligible": float(len(eligible)),
            "n_mismatch_episodes": float(len(mismatch_rows)),
        })
    return out


def _first_mismatch_position_curve(
    rows: Sequence[Dict[str, Any]],
    *,
    n_bins: int = FIRST_MISMATCH_POSITION_BINS,
) -> List[Dict[str, float]]:
    n_bins = max(1, int(n_bins))
    bins: List[List[Dict[str, Any]]] = [[] for _ in range(n_bins)]
    for row in rows:
        if float(row.get("first_mismatch_rate", 0.0)) <= 0.0:
            continue
        pos = _row_first_mismatch_robot_turn_position(row)
        if pos < 0.0:
            continue
        idx = min(n_bins - 1, max(0, int(pos * n_bins)))
        bins[idx].append(row)
    curve: List[Dict[str, float]] = []
    for idx, bucket in enumerate(bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        eligible = [r for r in bucket if float(r.get("adaptation_recovery_eligible_w3", 0.0)) > 0.0]
        support = [max(1.0, float(r.get("adaptation_recovery_support_w3", 3.0))) for r in eligible]
        recovery = (
            sum(float(r.get("adaptation_recovery_top1_w3", 0.0)) * w for r, w in zip(eligible, support)) / max(1.0, sum(support))
            if eligible else 0.0
        )
        curve.append({
            "bin": float(idx),
            "position_min": float(lo),
            "position_max": float(hi),
            "position_center": float((lo + hi) / 2.0),
            "n_episodes": float(len(bucket)),
            "mean_first_mismatch_robot_turn": _mean([_row_first_mismatch_robot_turn(r) for r in bucket]),
            "mean_episode_top1": _mean([float(r.get("live_top1", 0.0)) for r in bucket]),
            "adaptation_recovery_top1_w3": float(recovery),
            "adaptation_recovery_eligible_rate_w3": float(len(eligible) / max(1, len(bucket))),
            "adaptation_recovery_censored_rate_w3": float(sum(float(r.get("adaptation_recovery_censored_w3", 0.0)) for r in bucket) / max(1, len(bucket))),
        })
    return curve


def _step_variant_hash(step: LiveStepRecord) -> Optional[str]:
    return step.inferred_variant_hash


def _step_latent_pref_id(step: LiveStepRecord) -> Optional[str]:
    return step.inferred_latent_pref_id


def _step_action_calibration_score(step: LiveStepRecord) -> Optional[float]:
    for value in (
        step.policy_final_confidence,
        step.policy_confidence,
        step.policy_gate_score,
        step.policy_raw_confidence,
    ):
        if value is not None:
            return float(value)
    return None


def live_episode_metrics(record: LiveEpisodeRecord, *, true_rid: Optional[str] = None) -> Dict[str, float]:
    n = max(1, int(record.n))
    if record.n <= 0:
        return {
            "live_top1": 0.0,
            "live_topk": 0.0,
            **_adaptation_recovery_metrics([], -1),
            "behavioral_steps_to_lock": -1.0,
            "posterior_steps_to_lock": -1.0,
            "adaptation_latency_steps": -1.0,
            "confidence_ece": 0.0,
            "confidence_brier": 0.0,
            "robot_wrong_rate": 0.0,
            "net_robot_action_value": 0.0,
            "mean_policy_confidence": 0.0,
            "mean_posterior_confidence": 0.0,
            "mean_policy_final_confidence": 0.0,
            "mean_conditioned_policy_confidence": 0.0,
            "mean_ensemble_policy_confidence": 0.0,
            "policy_agreement_rate": 0.0,
            "mean_blend_strength": 0.0,
            "mean_prediction_wall_s": 0.0,
            "p95_prediction_wall_s": 0.0,
            "posterior_pref_lock_consistency": 0.0,
            "posterior_variant_lock_consistency": 0.0,
        }
    fm = record.first_mismatch_robot_turn if record.first_mismatch_robot_turn is not None else -1
    recovery_metrics = _adaptation_recovery_metrics([s.correct_top1 for s in record.steps], fm)
    wrong_steps = [s for s in record.steps if not s.correct_top1]
    correct_steps = [s for s in record.steps if s.correct_top1]
    posterior_confidences = [float(s.posterior_confidence) for s in record.steps if s.posterior_confidence is not None]
    policy_confidences = [float(s.policy_confidence) for s in record.steps if s.policy_confidence is not None]
    raw_policy_confidences = [float(s.policy_raw_confidence) for s in record.steps if s.policy_raw_confidence is not None]
    policy_gate_scores = [float(s.policy_gate_score) for s in record.steps if s.policy_gate_score is not None]
    policy_margins = [float(s.policy_margin) for s in record.steps if s.policy_margin is not None]
    final_policy_confidences = [float(s.policy_final_confidence) for s in record.steps if s.policy_final_confidence is not None]
    conditioned_policy_confidences = [float(s.conditioned_policy_confidence) for s in record.steps if s.conditioned_policy_confidence is not None]
    ensemble_policy_confidences = [float(s.ensemble_policy_confidence) for s in record.steps if s.ensemble_policy_confidence is not None]
    agreement_flags = [1.0 if bool(s.policy_agreement) else 0.0 for s in record.steps if s.policy_agreement is not None]
    blend_strengths = [float(s.blend_strength) for s in record.steps if s.blend_strength is not None]
    entropies = [float(s.posterior_entropy) for s in record.steps if s.posterior_entropy is not None]
    policy_entropies = [float(s.policy_entropy) for s in record.steps if s.policy_entropy is not None]
    n_hypotheses = [int(s.posterior_n_hypotheses) for s in record.steps if s.posterior_n_hypotheses is not None]
    score_label_pairs = [
        (score, 1 if s.correct_top1 else 0)
        for s in record.steps
        for score in [_step_action_calibration_score(s)]
        if score is not None
    ]
    scores = [score for score, _label in score_label_pairs]
    labels = [label for _score, label in score_label_pairs]
    calib = binary_decision_metrics(scores, labels, n_bins=10) if scores else {"ece": 0.0, "brier": 0.0}
    behavioral_lock = behavioral_steps_to_lock(record.steps)
    posterior_lock = posterior_steps_to_lock(record.steps, true_rid)
    # adaptation_latency_steps: robot turns from first mismatch until recovery.
    # Recovery requires a FULL 3-turn window with >=75% accuracy so that a
    # single terminal correct prediction cannot falsely declare recovery.
    # Episodes where fewer than 3 turns remain after the mismatch are censored
    # (set to -1), consistent with _adaptation_recovery_metrics censoring logic.
    recovery_latency = -1
    if fm >= 0:
        steps_after_mismatch = len(record.steps) - fm - 1
        if steps_after_mismatch >= 3:
            for k in range(fm + 1, len(record.steps) - 2):
                chunk = record.steps[k:k + 3]
                if _mean([1.0 if s.correct_top1 else 0.0 for s in chunk]) >= 0.75:
                    recovery_latency = k - fm
                    break
    recipe_vocab = {s.actual for s in record.steps}
    recipe_vocab_hits = [1.0 if s.predicted in recipe_vocab else 0.0 for s in record.steps]
    prediction_wall = [float(s.prediction_wall_s) for s in record.steps if s.prediction_wall_s is not None]
    single_record_preset = {record.pair_label: record.preference_name}
    return {
        "live_top1": float(record.live_top1),
        "live_topk": float(record.live_topk),
        **recovery_metrics,
        "behavioral_steps_to_lock": float(behavioral_lock),
        "posterior_steps_to_lock": float(posterior_lock),
        "adaptation_latency_steps": float(recovery_latency),
        "posterior_correct_recipe": _mean([1.0 if s.inferred_recipe == true_rid else 0.0 for s in record.steps]) if true_rid is not None else 0.0,
        "recipe_vocab_top1": _mean(recipe_vocab_hits),
        "robot_wrong_rate": len(wrong_steps) / n,
        "net_robot_action_value": (
            len(correct_steps)
            - ASSIST_WRONG_PENALTY * len(wrong_steps)
        ) / n,
        "mean_policy_confidence": _mean(policy_confidences or final_policy_confidences),
        "mean_posterior_confidence": _mean(posterior_confidences),
        "mean_policy_raw_confidence": _mean(raw_policy_confidences),
        "mean_policy_gate_score": _mean(policy_gate_scores),
        "mean_policy_margin": _mean(policy_margins),
        "mean_policy_final_confidence": _mean(final_policy_confidences),
        "mean_conditioned_policy_confidence": _mean(conditioned_policy_confidences),
        "mean_ensemble_policy_confidence": _mean(ensemble_policy_confidences),
        # would_assist_under_gate_rate intentionally omitted: always 1.0 under
        # the forced alternating protocol (robot never abstains).  It is not a
        # meaningful aggregate metric in this setting.
        "policy_agreement_rate": _mean(agreement_flags),
        "mean_blend_strength": _mean(blend_strengths),
        "mean_posterior_entropy": _mean(entropies),
        "mean_policy_entropy": _mean(policy_entropies),
        "posterior_degenerate_rate": _mean([1.0 if h <= 1 else 0.0 for h in n_hypotheses]),
        "confidence_ece": float(calib.get("ece", 0.0)),
        "confidence_brier": float(calib.get("brier", 0.0)),
        "mean_prediction_wall_s": _mean(prediction_wall),
        "p95_prediction_wall_s": _p95(prediction_wall),
        "posterior_pref_lock_consistency": preference_lock_purity([record], single_record_preset, use_latent_pref=True),
        "posterior_variant_lock_consistency": preference_lock_purity([record], single_record_preset, use_latent_pref=False),
    }


def bwt_zero_shot_checkpoints(
    phase_a_eval: Dict[str, Dict[str, float]],
    final_eval: Dict[str, Dict[str, float]],
    phase_a_seen_labels: Sequence[str],
    isolated_eval: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Compute BWT and zero-shot transfer from checkpoint evals.

    BWT (backward transfer / forgetting):
        Measures how much performance on Phase A tasks changed from their
        isolated single-task peak to the end of the full deployment stream.
        When ``isolated_eval`` is supplied it is used as the baseline (correct);
        otherwise ``phase_a_eval`` is used as the Phase-A checkpoint baseline.

    Zero-shot transfer:
        Performance on tasks that were *not* in Phase A, evaluated at the Phase A
        checkpoint.
    """
    labels = sorted(k for k in final_eval if k in phase_a_eval)
    seen = set(phase_a_seen_labels)
    bwt_baseline = isolated_eval if isolated_eval is not None else phase_a_eval
    if isolated_eval is not None:
        # Correct reference: single-task peak evaluated on a fresh agent that
        # observed only task i before evaluation.  Approximates R_{i,i} from
        # the standard BWT formula (Lopez-Paz & Ranzato 2017).
        bwt_baseline_name = "isolated_single_task"
    else:
        # Fallback: Phase-A checkpoint AFTER all Phase-A tasks have been
        # learned.  Phase-A within-stream interference is already baked in,
        # so this reference is LOWER than the true R_{i,i}.  BWT values from
        # this path may understate forgetting.  Label is distinct so that
        # paper-facing results can be filtered to the correct reference only.
        bwt_baseline_name = "bwt_phase_a_checkpoint_approx"
    bwt_terms = [
        float(final_eval[label].get("top1", 0.0)) - float(bwt_baseline.get(label, {}).get("top1", 0.0))
        for label in labels
        if label in seen
    ]
    zero_shot_terms = [
        float(phase_a_eval[label].get("top1", 0.0))
        for label in labels
        if label not in seen
    ]
    return {
        "bwt": float(_mean(bwt_terms)) if bwt_terms else float("nan"),
        "zero_shot_transfer": float(_mean(zero_shot_terms)) if zero_shot_terms else float("nan"),
        "bwt_baseline": bwt_baseline_name,
        "n_bwt_tasks": len(bwt_terms),
        "n_zero_shot_tasks": len(zero_shot_terms),
        "bwt_terms": bwt_terms,
        "zero_shot_terms": zero_shot_terms,
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
    *,
    use_latent_pref: bool = True,
) -> float:
    by_preset: Dict[str, List[str]] = {}
    for record in records:
        preset = preset_of_pair_label.get(record.pair_label)
        if preset is None:
            continue
        for step in record.steps:
            pid = _step_latent_pref_id(step) if use_latent_pref else _step_variant_hash(step)
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


def preference_cluster_metrics(
    records: Sequence[LiveEpisodeRecord],
    preset_of_pair_label: Dict[str, str],
    *,
    use_latent_pref: bool = True,
) -> Dict[str, Any]:
    true_labels: List[str] = []
    pred_labels: List[str] = []
    for record in records:
        preset = preset_of_pair_label.get(record.pair_label)
        if preset is None:
            continue
        for step in record.steps:
            pid = _step_latent_pref_id(step) if use_latent_pref else _step_variant_hash(step)
            if pid is None:
                continue
            true_labels.append(str(preset))
            pred_labels.append(str(pid))
    n = len(true_labels)
    if n == 0:
        return {
            "preference_cluster_purity": 0.0,
            "preference_nmi": 0.0,
            "preference_ari": 0.0,
            "preference_collapse_rate": 0.0,
            "latent_preference_entropy": 0.0,
            "preference_confusion": {},
            "n_preference_assignments": 0,
        }
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    joint = Counter(zip(true_labels, pred_labels))

    def comb2(x: int) -> float:
        return float(x * (x - 1) / 2)

    mi = 0.0
    for (true_label, pred_label), count in joint.items():
        if count > 0:
            mi += (count / n) * math.log((count * n) / max(true_counts[true_label] * pred_counts[pred_label], 1))
    h_true = -sum((count / n) * math.log(count / n) for count in true_counts.values() if count > 0)
    h_pred = -sum((count / n) * math.log(count / n) for count in pred_counts.values() if count > 0)
    nmi = (2.0 * mi / (h_true + h_pred)) if (h_true + h_pred) > 0 else 0.0

    sum_joint = sum(comb2(count) for count in joint.values())
    sum_true = sum(comb2(count) for count in true_counts.values())
    sum_pred = sum(comb2(count) for count in pred_counts.values())
    total_pairs = comb2(n)
    expected = (sum_true * sum_pred / total_pairs) if total_pairs > 0 else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    ari = ((sum_joint - expected) / (max_index - expected)) if max_index > expected else 0.0

    max_cluster_fraction = max(pred_counts.values()) / n
    unique_ratio = len(pred_counts) / max(len(true_counts), 1)
    collapse_rate = max(0.0, 1.0 - unique_ratio) * max_cluster_fraction
    norm_entropy = (h_pred / math.log(len(pred_counts))) if len(pred_counts) > 1 else 0.0
    confusion: Dict[str, Dict[str, int]] = {}
    for (true_label, pred_label), count in joint.items():
        confusion.setdefault(true_label, {})[pred_label] = int(count)
    return {
        "preference_cluster_purity": preference_lock_purity(records, preset_of_pair_label, use_latent_pref=use_latent_pref),
        "preference_nmi": float(max(0.0, min(1.0, nmi))),
        "preference_ari": float(ari),
        "preference_collapse_rate": float(collapse_rate),
        "latent_preference_entropy": float(norm_entropy),
        "preference_confusion": confusion,
        "n_preference_assignments": int(n),
    }


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


_POLICY_DIAGNOSTIC_FALLBACK_KEYS: Dict[str, Tuple[str, ...]] = {
    "would_assist_under_gate": ("assist_used",),
    "gate_reason": ("assist_reason", "reason"),
    "gate_source": ("assist_source",),
    "policy_confidence": ("final_action_confidence", "action_confidence"),
    "policy_raw_confidence": ("raw_action_confidence",),
    "policy_gate_score": ("action_gate_score",),
    "policy_margin": ("final_action_margin", "action_margin"),
    "policy_entropy": ("final_action_entropy", "action_entropy", "action_marginal_entropy"),
    "gate_threshold": ("action_gate_threshold",),
    "policy_final_confidence": ("final_action_confidence",),
    "policy_final_margin": ("final_action_margin",),
    "policy_final_entropy": ("final_action_entropy",),
    "conditioned_policy_confidence": ("conditioned_action_confidence",),
    "ensemble_policy_confidence": ("ensemble_action_confidence",),
    "policy_agreement": ("policy_agreement",),
    "blend_strength": ("blend_strength",),
}


def _row_policy_diagnostics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = row.get("policy_diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _row_policy_raw(row: Mapping[str, Any], key: str) -> Any:
    diagnostics = _row_policy_diagnostics(row)
    if key in diagnostics:
        return diagnostics.get(key)
    for fallback_key in _POLICY_DIAGNOSTIC_FALLBACK_KEYS.get(key, ()):
        if fallback_key in row:
            return row.get(fallback_key)
    return None


def _row_policy_value(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = _row_policy_raw(row, key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _row_policy_bool(row: Mapping[str, Any], key: str) -> Optional[bool]:
    value = _row_policy_raw(row, key)
    if value is None:
        return None
    return bool(value)


def _row_policy_text(row: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = _row_policy_raw(row, key)
    return str(value if value is not None else default)


def _row_robot_wrong(row: Mapping[str, Any]) -> bool:
    return not bool(row.get("correct_top1", False))


def live_step_trace(step_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    rows_for_trace = list(step_rows)
    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows_for_trace:
        by_step[int(row.get("step", 0))].append(row)
    out: List[Dict[str, float]] = []
    for step, rows in sorted(by_step.items()):
        top1_vals = [1.0 if r.get("correct_top1") else 0.0 for r in rows]
        top1 = _mean(top1_vals)
        n_steps = max(1, len(rows))
        binom_ci = 1.96 * math.sqrt(max(0.0, top1 * (1.0 - top1)) / n_steps)
        continuous: Dict[str, float] = {}
        direct_keys = ("posterior_confidence", "posterior_entropy")
        policy_keys = ("policy_confidence", "policy_margin")
        for key in direct_keys:
            vals = [float(r.get(key)) for r in rows if r.get(key) is not None]
            if vals:
                arr = np.asarray(vals, dtype=float)
                continuous[key] = float(np.mean(arr))
                continuous[f"{key}_p10"] = float(np.percentile(arr, 10))
                continuous[f"{key}_p50"] = float(np.percentile(arr, 50))
                continuous[f"{key}_p90"] = float(np.percentile(arr, 90))
        for key in policy_keys:
            vals = [v for r in rows for v in (_row_policy_value(r, key),) if v is not None]
            if vals:
                arr = np.asarray(vals, dtype=float)
                continuous[key] = float(np.mean(arr))
                continuous[f"{key}_p10"] = float(np.percentile(arr, 10))
                continuous[f"{key}_p50"] = float(np.percentile(arr, 50))
                continuous[f"{key}_p90"] = float(np.percentile(arr, 90))
        out.append({
            "step": float(step),
            "top1": top1,
            "top1_ci95": float(binom_ci),
            **continuous,
            "n_episodes": float(len({(r.get("baseline"), r.get("event_idx"), r.get("pair"), r.get("condition"), r.get("repeat"), r.get("arm"), r.get("gap")) for r in rows})),
            "n_steps": float(len(rows)),
        })
    return out


def calibration_curve_from_steps(step_rows: Sequence[Dict[str, Any]], n_bins: int = 10) -> List[Dict[str, float]]:
    rows = [
        (float(score), 1.0 if r.get("correct_top1") else 0.0)
        for r in step_rows
        for score in (_row_policy_value(r, "policy_confidence"),)
        if score is not None
    ]
    if not rows:
        return []
    scores = np.asarray([r[0] for r in rows], dtype=float)
    labels = np.asarray([r[1] for r in rows], dtype=float)
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = (scores - float(scores.min())) / max(float(scores.max() - scores.min()), 1e-9)
    n_eff_bins = max(1, min(int(n_bins), len(scores)))
    ordered = np.argsort(scores)
    curve: List[Dict[str, float]] = []
    for idx, bin_idx in enumerate(np.array_split(ordered, n_eff_bins)):
        count = int(len(bin_idx))
        if count <= 0:
            continue
        curve.append({
            "bin": float(idx),
            "confidence": float(scores[bin_idx].mean()),
            "accuracy": float(labels[bin_idx].mean()),
            "n_steps": float(count),
            "score_min": float(scores[bin_idx].min()),
            "score_max": float(scores[bin_idx].max()),
        })
    return curve


def _step_row_confidence(row: Mapping[str, Any]) -> Optional[float]:
    for key in (
        "policy_final_confidence",
        "policy_confidence",
        "policy_gate_score",
        "policy_raw_confidence",
    ):
        value = _row_policy_value(row, key)
        if value is not None:
            return float(value)
    return None


def confidence_threshold_accuracy_curve_from_steps(
    step_rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    scored = [
        (
            float(score),
            1.0 if row.get("correct_top1") else 0.0,
            1.0 if row.get("correct_topk") else 0.0,
        )
        for row in step_rows
        for score in (_step_row_confidence(row),)
        if score is not None
    ]
    n_steps = len(scored)
    out: Dict[str, Dict[str, float]] = {}
    for threshold in sorted({float(t) for t in thresholds}):
        retained = [(top1, topk) for score, top1, topk in scored if score >= threshold]
        n_retained = len(retained)
        conditional_top1 = _mean([top1 for top1, _topk in retained]) if retained else float("nan")
        conditional_topk = _mean([topk for _top1, topk in retained]) if retained else float("nan")
        out[f"{threshold:.2f}"] = {
            "confidence_threshold": float(threshold),
            "retained_robot_turn_fraction": float(n_retained / max(1, n_steps)),
            "conditional_top1": float(conditional_top1),
            "conditional_topk": float(conditional_topk),
            "retained_wrong_rate": float(1.0 - conditional_top1) if math.isfinite(float(conditional_top1)) else float("nan"),
            "unconditional_top1": _mean([top1 for _score, top1, _topk in scored]),
            "unconditional_topk": _mean([topk for _score, _top1, topk in scored]),
            "n_retained_robot_turns": float(n_retained),
            "n_scored_robot_turns": float(n_steps),
        }
    return out


def latency_cdf_from_steps(step_rows: Sequence[Dict[str, Any]], max_points: int = 101) -> List[Dict[str, float]]:
    vals = sorted(float(r.get("prediction_wall_s")) for r in step_rows if r.get("prediction_wall_s") is not None)
    if not vals:
        return []
    qs = np.linspace(0.0, 1.0, max(2, int(max_points)))
    arr = np.asarray(vals, dtype=float)
    return [{"latency_s": float(np.quantile(arr, q)), "cdf": float(q)} for q in qs]


def policy_gate_reason_summary(step_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(_row_policy_text(r, "gate_reason") for r in step_rows)
    sources = Counter(_row_policy_text(r, "gate_source") for r in step_rows)
    total = max(1, sum(counts.values()))
    source_total = max(1, sum(sources.values()))
    return {
        "counts": {k: int(v) for k, v in sorted(counts.items())},
        "rates": {k: float(v / total) for k, v in sorted(counts.items())},
        "source_counts": {k: int(v) for k, v in sorted(sources.items())},
        "source_rates": {k: float(v / source_total) for k, v in sorted(sources.items())},
    }


def assist_gate_reason_summary(step_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Backward-compatible alias for historical callers."""
    return policy_gate_reason_summary(step_rows)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _humanize_identifier(value: str) -> str:
    stem = Path(str(value)).stem
    stem = re.sub(r"^(aggregate_)?", "", stem)
    stem = re.sub(r"^[FS][0-9]+[a-z]?_", "", stem)
    return stem.replace("_", " ").strip()












GUIDE_EVENT_TYPE_ORDER: Tuple[str, ...] = (
    "phase_a_observe",
    "diagonal_train",
    "initial_recipe_observation",
    "prep_book_onboarding",
    "single_shot_observe",
    "routine_reuse",
    "rush_hour_repeat_order",
    "single_shot_first_reuse",
    "preference_shift",
    "known_recipe_new_preference",
    "user_switch",
    "cross_transfer_probe",
    "offdiagonal_transfer",
    "axis_value_train",
    "axis_value_transfer_probe",
    "axis_family_train",
    "axis_family_holdout_probe",
    "regular_menu_rotation",
    "rare_reentry",
    "rare_custom_order_reentry",
    "new_recipe_observe",
    "ladder_initial_observe",
    "ladder_pref_update_heterogeneous",
    "ladder_shared_pref_shift",
    "ladder_settle",
)

EVENT_TYPE_EXPLANATIONS: Dict[str, str] = {
    "phase_a_observe": "initial demonstrations before deployment; these seed the recipe/preference library rather than testing assistance.",
    "diagonal_train": "training demonstrations where each recipe is paired with one assigned preference; this creates the diagonal training set for transfer tests.",
    "initial_recipe_observation": "a first observation pass over recipes before the assistant is evaluated.",
    "prep_book_onboarding": "a balanced pre-service demonstration pass over recipe/preference workflows before the rush-hour demand stream begins.",
    "single_shot_observe": "one demonstration of a recipe/preference pair before asking whether one exposure is enough.",
    "routine_reuse": "an assist episode for a recipe/preference pair that has already been demonstrated; this tests direct recall.",
    "rush_hour_repeat_order": "a high-frequency house-favorite workflow repeatedly requested during the simulated service window.",
    "single_shot_first_reuse": "the first assist attempt immediately after a single demonstration; this tests one-shot reuse.",
    "preference_shift": "a known recipe is requested with a different preference ordering; this tests whether the assistant adapts to a changed human style.",
    "known_recipe_new_preference": "the recipe is familiar but the requested preference has not been seen on that recipe before.",
    "user_switch": "the active user context changes; this tests whether user-specific preferences are retained across blocks.",
    "cross_transfer_probe": "a seen preference is applied to a recipe where that exact pairing was not demonstrated; this tests cross-recipe transfer.",
    "confidence_threshold_probe": "a scheduled robot-turn assist episode used to build confidence-threshold accuracy curves without changing the action policy.",
    "offdiagonal_transfer": "a held-out recipe/preference combination after diagonal training; this tests whether preference structure transfers rather than memorizing pairs.",
    "axis_value_train": "training support where one non-default preference-axis value is demonstrated on a recipe subset.",
    "axis_value_transfer_probe": "a held-out recipe subset using the same non-default axis value; this tests whether an axis value transfers across recipes.",
    "axis_family_train": "training support that excludes non-default values for one preference axis family while other axes may vary.",
    "axis_family_holdout_probe": "probe episodes using non-default values from an axis family absent from training; this is the cleanest test of cross-axis generalization.",
    "regular_menu_rotation": "a mid-frequency workflow that appears during ordinary menu rotation, neither a house favorite nor a rare custom order.",
    "rare_reentry": "an older recipe/preference pair returns after a long gap; this tests recovery from stale or pruned memory.",
    "rare_custom_order_reentry": "a low-frequency customized workflow returns during the Zipf branch; fairness should still count it even if finite sampling barely shows it.",
    "new_recipe_observe": "a deployment-time observation event for a new recipe; this expands the library instead of measuring assistance.",
    "ladder_initial_observe": "the first ladder rung observes one recipe-preference variant per selected recipe.",
    "ladder_pref_update_heterogeneous": "all known recipes receive recipe-specific seed-shuffled preference updates in assist mode.",
    "ladder_shared_pref_shift": "all known recipes receive the same seed-shuffled preference update in assist mode.",
    "ladder_settle": "assist-only repeats after a ladder update; this measures whether inference stabilizes after the new rung.",
}

TRANSFER_CELL_EXPLANATIONS: Dict[str, str] = {
    "direct_retrieval": "the exact recipe/preference pair was previously observed, so the assistant can retrieve a known workflow.",
    "seen_recipe_preference_seen_elsewhere": "the recipe is known and the preference is known from another recipe, but this exact pairing is new.",
    "seen_recipe_new_preference": "the recipe is known but the preference has not been seen before.",
    "unseen_recipe_seen_preference": "the recipe is new but the preference style has appeared elsewhere.",
    "unseen_unseen": "neither the recipe nor the preference style has been seen before.",
}

FOUR_CELL_EXPLANATIONS: Dict[str, str] = {
    "known_recipe_known_preference": "both the recipe and preference style have appeared previously; this is a coarse exposure label, not necessarily exact-pair retrieval.",
    "seen_recipe_new_preference": "the recipe has appeared before but the preference style is new to the stream.",
    "unseen_recipe_seen_preference": "the recipe is new but the preference style has appeared elsewhere.",
    "unseen_unseen": "both recipe and preference style are new to the stream.",
}

MEMORY_STATE_EXPLANATIONS: Dict[str, str] = {
    "no_memory": "the system has no retained variant for the requested workflow.",
    "active_memory": "the requested variant is still active in the bounded memory set.",
    "pruned_memory": "the requested variant was previously learned but has decayed into the pruned/archive set.",
    "same_recipe_new_preference": "the recipe is known, but the current preference variant for that recipe is new.",
}


def _ordered_known_labels(labels: Iterable[str], preferred: Sequence[str]) -> List[str]:
    seen = {str(label) for label in labels if str(label)}
    ordered: List[str] = []
    for label in preferred:
        if label in seen and label not in ordered:
            ordered.append(label)
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def _event_type_description(event_type: str) -> str:
    event_type = str(event_type)
    if event_type in EVENT_TYPE_EXPLANATIONS:
        return EVENT_TYPE_EXPLANATIONS[event_type]
    narrative = DEPLOYMENT_EVENT_NARRATIVES.get(event_type)
    if narrative:
        parts = [str(narrative.get("claim", "")).strip(), str(narrative.get("pressure", "")).strip()]
        return " ".join(p for p in parts if p)
    return f"`{_humanize_identifier(event_type)}` workflow category recorded by the experiment harness."


def _prior_state_description(label: str) -> str:
    label = str(label)
    if label in TRANSFER_CELL_EXPLANATIONS:
        return TRANSFER_CELL_EXPLANATIONS[label]
    if label in FOUR_CELL_EXPLANATIONS:
        return FOUR_CELL_EXPLANATIONS[label]
    if label in MEMORY_STATE_EXPLANATIONS:
        return MEMORY_STATE_EXPLANATIONS[label]
    return f"`{_humanize_identifier(label)}` prior exposure state recorded by the harness."


def _event_type_glossary_lines(event_types: Iterable[str]) -> List[str]:
    labels = _ordered_known_labels(event_types, GUIDE_EVENT_TYPE_ORDER)
    if not labels:
        return []
    lines = [
        "",
        "### Event Type Glossary",
        "These labels describe the kitchen situation being evaluated, not just category names.",
    ]
    for label in labels:
        lines.append(f"- `{label}`: {_event_type_description(label)}")
    return lines


def _prior_state_glossary_lines(prior_states: Iterable[str]) -> List[str]:
    labels = _ordered_known_labels(
        prior_states,
        tuple(TRANSFER_CELL_KEYS) + tuple(FOUR_CELL_KEYS) + tuple(MEMORY_STATE_LABELS),
    )
    if not labels:
        return []
    lines = [
        "",
        "### Prior State Glossary",
        "The prior state column describes what the system had seen before that episode begins.",
    ]
    for label in labels:
        lines.append(f"- `{label}`: {_prior_state_description(label)}")
    return lines


def _scenario_event_type_counts(result: Mapping[str, Any]) -> Dict[str, int]:
    scenario = result.get("scenario", {}) or {}
    counts = scenario.get("event_type_counts")
    if isinstance(counts, Mapping):
        return {str(k): int(v) for k, v in counts.items()}
    counter: Counter[str] = Counter()
    for event in scenario.get("events", []) or []:
        if isinstance(event, Mapping):
            event_type = event.get("event_type", event.get("requested_event_type", event.get("condition")))
            if event_type:
                counter[str(event_type)] += 1
    return dict(counter)


def _scenario_prior_state_counts(result: Mapping[str, Any]) -> Dict[str, int]:
    scenario = result.get("scenario", {}) or {}
    counter: Counter[str] = Counter()
    for event in scenario.get("events", []) or []:
        if not isinstance(event, Mapping):
            continue
        state = event.get("transfer_cell_before") or event.get("four_cell_before") or event.get("memory_state")
        if state:
            counter[str(state)] += 1
    return dict(counter)


def _aggregate_seed_results(folder: Path) -> List[Dict[str, Any]]:
    seed_results: List[Dict[str, Any]] = []
    for result_path in sorted(folder.parent.glob("seed_*/result.json")):
        data = _safe_read_json(result_path)
        if data:
            seed_results.append(data)
    return seed_results






















def _event_brief(row: Mapping[str, Any]) -> str:
    idx = row.get("event_idx", row.get("phase_b_idx", ""))
    mode = row.get("mode", row.get("condition", ""))
    event_type = row.get("event_type", row.get("requested_event_type", row.get("condition", "")))
    recipe = row.get("recipe") or str(row.get("pair", "")).split("/")[0]
    pref = row.get("preference") or (str(row.get("pair", "")).split("/")[-1] if row.get("pair") else "")
    user = row.get("user_id", "")
    phase = row.get("phase", "")
    memory = row.get("transfer_cell_before", row.get("four_cell_before", ""))
    return f"| {idx} | {mode} | {phase} | {event_type} | {user} | {recipe} | {pref} | {memory} |"


def _scenario_lines_for_seed(result: Mapping[str, Any]) -> List[str]:
    meta = result.get("metadata", {}) or {}
    scenario = result.get("scenario", {}) or {}
    narrative = scenario.get("narrative", {}) or {}
    events = list(scenario.get("events", []) or [])
    event_type_counts = _scenario_event_type_counts(result)
    prior_state_counts = _scenario_prior_state_counts(result)
    lines = [
        "## Scenario / Workflow",
        f"- **Experiment:** `{meta.get('experiment', 'unknown')}`.",
        f"- **Seed:** `{meta.get('seed', 'unknown')}`.",
    ]
    if narrative.get("claim"):
        lines.append(f"- **Scientific claim:** {narrative.get('claim')}")
    if narrative.get("pressure"):
        lines.append(f"- **Scenario pressure:** {narrative.get('pressure')}")
    if scenario.get("mode_counts"):
        lines.append(f"- **Mode counts:** {dict(scenario.get('mode_counts') or {})}.")
    if event_type_counts:
        lines.append(f"- **Event type counts:** {dict(event_type_counts)}.")
    if scenario.get("selected_recipes"):
        lines.append(f"- **Selected recipes:** {', '.join(str(x) for x in scenario.get('selected_recipes', [])[:20])}.")
    if scenario.get("selected_preferences"):
        lines.append(f"- **Selected preferences:** {', '.join(str(x) for x in scenario.get('selected_preferences', [])[:20])}.")
    phase_a = [e for e in events if str(e.get("phase")) == "phase_a"]
    if phase_a:
        shown = [f"{e.get('recipe') or str(e.get('pair', '')).split('/')[0]}/{e.get('preference') or str(e.get('pair', '')).split('/')[-1]}" for e in phase_a[:30]]
        lines.append(f"- **Initial demonstrations in order:** {', '.join(shown)}.")
    lines.extend(_event_type_glossary_lines(event_type_counts.keys()))
    lines.extend(_prior_state_glossary_lines(prior_state_counts.keys()))
    if events:
        lines.extend([
            "",
            "### Event Flow",
            "This table is the exact seed-level workflow order recorded by the harness.",
            "",
            "| idx | mode | phase | event type | user | recipe | preference | prior state |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        max_events = 180
        for row in events[:max_events]:
            lines.append(_event_brief(row))
        if len(events) > max_events:
            lines.append(f"| ... | ... | ... | ... | ... | ... | ... | truncated after {max_events} of {len(events)} events |")
    lines.append("")
    return lines


def _scenario_lines_for_aggregate(folder: Path, aggregate: Mapping[str, Any]) -> List[str]:
    seed_results = _aggregate_seed_results(folder)
    lines = [
        "## Scenario / Workflow",
        f"- **Experiment aggregate:** `{folder.parent.name}`.",
        f"- **Seeds summarized:** {len(seed_results)}.",
    ]
    if seed_results:
        experiment = str(seed_results[0].get("metadata", {}).get("experiment", folder.parent.name))
        narrative = (seed_results[0].get("scenario", {}) or {}).get("narrative", {}) or SCENARIO_NARRATIVES.get(experiment, {})
        if narrative.get("claim"):
            lines.append(f"- **Scientific claim:** {narrative.get('claim')}")
        if narrative.get("pressure"):
            lines.append(f"- **Scenario pressure:** {narrative.get('pressure')}")
        count_rows = []
        aggregate_event_counts: Counter[str] = Counter()
        aggregate_prior_counts: Counter[str] = Counter()
        for seed_result in seed_results:
            scenario = seed_result.get("scenario", {}) or {}
            meta = seed_result.get("metadata", {}) or {}
            event_type_counts = _scenario_event_type_counts(seed_result)
            aggregate_event_counts.update(event_type_counts)
            aggregate_prior_counts.update(_scenario_prior_state_counts(seed_result))
            count_rows.append({
                "seed": meta.get("seed"),
                "events": scenario.get("n_events", 0),
                "event_types": event_type_counts,
            })
        lines.append(f"- **Aggregate status:** {dict(aggregate.get('status', {}) or {})}.")
        lines.extend(["", "### Seed Event Counts", "", "| seed | events | event type counts |", "| --- | ---: | --- |"])
        for row in count_rows:
            lines.append(f"| {row['seed']} | {row['events']} | {row['event_types']} |")
        lines.extend(_event_type_glossary_lines(aggregate_event_counts.keys()))
        lines.extend(_prior_state_glossary_lines(aggregate_prior_counts.keys()))
        lines.extend(["", "### Seed Initial Demonstration Orders"])
        for seed_result in seed_results:
            meta = seed_result.get("metadata", {}) or {}
            events = list((seed_result.get("scenario", {}) or {}).get("events", []) or [])
            phase_a = [e for e in events if str(e.get("phase")) == "phase_a"]
            shown = [f"{e.get('recipe') or str(e.get('pair', '')).split('/')[0]}/{e.get('preference') or str(e.get('pair', '')).split('/')[-1]}" for e in phase_a[:20]]
            lines.append(f"- Seed `{meta.get('seed')}`: {', '.join(shown) if shown else 'no ordered demonstrations recorded'}.")
    else:
        lines.append("- No seed-level result files were found beside this aggregate folder.")
    lines.append("")
    return lines




def _artifact_guide_text(folder: Path) -> Optional[str]:
    result_path = folder / "result.json"
    aggregate_path = folder / "aggregate.json"
    if not result_path.exists() and not aggregate_path.exists():
        return None
    result = _safe_read_json(result_path) if result_path.exists() else {}
    aggregate = _safe_read_json(aggregate_path) if aggregate_path.exists() else {}
    lines = [
        "# Artifact Guide",
        "",
        "This file is generated from stored result logs.",
        "",
    ]
    if result:
        lines.extend(_scenario_lines_for_seed(result))
    elif aggregate:
        lines.extend(_scenario_lines_for_aggregate(folder, aggregate))
    else:
        lines.extend(["## Scenario / Workflow", "- No structured result JSON was found in this folder.", ""])
    return "\n".join(lines).rstrip() + "\n"


def generate_artifact_guides(run_dir: str | Path) -> Dict[str, Any]:
    """Write per-seed and per-aggregate markdown guides for stored logs."""
    root = Path(run_dir)
    written: List[str] = []
    for folder in sorted([p for p in root.rglob("*") if p.is_dir()] + [root]):
        if folder.name == "__pycache__":
            continue
        text = _artifact_guide_text(folder)
        if text is None:
            continue
        guide_path = folder / "artifact_guide.md"
        guide_path.write_text(text, encoding="utf-8")
        try:
            written.append(str(guide_path.relative_to(root)))
        except ValueError:
            written.append(str(guide_path))
    return {"n_guides": len(written), "guides": written}


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






def _nested_get(obj: Mapping[str, Any], path: Sequence[Any], default: Any = None) -> Any:
    cur: Any = obj
    for part in path:
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


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
    axes = tuple(WORKFLOW_AXES)
    values_by_axis = [tuple(WORKFLOW_AXIS_VALUES[axis]) for axis in axes]
    for values in itertools.product(*values_by_axis):
        pref = WorkflowPreference(**dict(zip(axes, values)))
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
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def make_agent(name: str, cfg: Config) -> AdaptiveHRCAgent:
    key = str(name)
    if key == "full":
        return AdaptiveHRCAgent(cfg=cfg)
    ablation_cfg: Optional[Config] = None
    if key == "no_posterior":
        ablation_cfg = replace(cfg, ablation_disable_posterior=True)
    elif key == "no_preference_prototype":
        ablation_cfg = replace(cfg, ablation_disable_preference_head=True)
    elif key == "no_recipe_prototype":
        ablation_cfg = replace(cfg, ablation_disable_recipe_prototype=True)
    elif key == "no_pruned_memory_prior":
        ablation_cfg = replace(cfg, ablation_disable_pruned_memory_prior=True)
    elif key == "oracle_preference_label":
        ablation_cfg = replace(cfg, ablation_oracle_preference_label=True)
    elif key == "oracle_recipe_and_preference_label":
        ablation_cfg = replace(cfg, ablation_oracle_recipe_and_preference_label=True)
    if ablation_cfg is not None:
        return AdaptiveHRCAgent(cfg=ablation_cfg)
    if key not in BASELINE_AGENTS:
        raise KeyError(f"unknown baseline {name!r}")
    return BASELINE_AGENTS[key](cfg=cfg)


TIMING_DETAILS_WARNING = (
    "Wall-clock HRC timing diagnostics are secondary metrics and are meaningful "
    "only when HRCTimingConfig is calibrated for the robot, task, and site. "
    "Use human_action_fraction as the primary HRC utility metric and "
    "human_effort_fraction when correction burden should be time-weighted."
)


def _hrc_timing_details(
    *,
    hrc_total_time: float,
    human_only_time: float,
    hrc_time_penalty_vs_human_only: float,
    observation_mode_extra_time: float,
    hrc_mean_time_per_recipe_step: float,
    hrc_mean_time_per_robot_turn: float,
    hrc_robot_correct_time: float,
    hrc_robot_wrong_time: float,
    hrc_human_action_time: float,
    hrc_human_correction_time: float,
    hrc_wrong_action_extra_penalty_time: float,
) -> Dict[str, Any]:
    """Return secondary wall-clock timing diagnostics.

    Warning: these metrics are interpretable only with a calibrated
    HRCTimingConfig. For kitchen HRC, the primary assistance outcome is human
    physical workload, represented by human_action_fraction.
    """
    hrc_total_time = float(hrc_total_time)
    human_only_time = float(human_only_time)
    return {
        "warning": TIMING_DETAILS_WARNING,
        "primary_hrc_metric": "human_action_fraction",
        "timing_config_seconds": asdict(DEFAULT_HRC_TIMING),
        "hrc_total_time": hrc_total_time,
        "human_only_time": human_only_time,
        "hrc_time_saved_vs_human_only": float(human_only_time - hrc_total_time),
        "hrc_time_over_human_only": float(hrc_total_time / max(human_only_time, 1e-12)) if human_only_time > 0.0 else 0.0,
        "normalized_interaction_cost": float(hrc_total_time / max(human_only_time, 1e-12)) if human_only_time > 0.0 else 0.0,
        "hrc_time_penalty_vs_human_only": float(hrc_time_penalty_vs_human_only),
        "observation_mode_extra_time": float(observation_mode_extra_time),
        "hrc_mean_time_per_recipe_step": float(hrc_mean_time_per_recipe_step),
        "hrc_mean_time_per_robot_turn": float(hrc_mean_time_per_robot_turn),
        "hrc_robot_correct_time": float(hrc_robot_correct_time),
        "hrc_robot_wrong_time": float(hrc_robot_wrong_time),
        "hrc_human_action_time": float(hrc_human_action_time),
        "hrc_human_correction_time": float(hrc_human_correction_time),
        "hrc_wrong_action_extra_penalty_time": float(hrc_wrong_action_extra_penalty_time),
    }


def observe_episode(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if isinstance(agent, OracleCeilingAgent):
        agent.set_oracle_target(pair.actions)
    n_recipe_steps = len(pair.actions)
    human_only_time = float(n_recipe_steps * DEFAULT_HRC_TIMING.human_action_time)
    observation_extra_time = float(n_recipe_steps * OBSERVATION_MODE_EXTRA_TIME_PER_STEP)
    observation_total_time = human_only_time + observation_extra_time
    human_effort_time = human_only_time
    human_effort_fraction = human_effort_time / max(human_only_time, 1e-12) if human_only_time > 0.0 else 0.0
    timing_details = _hrc_timing_details(
        hrc_total_time=observation_total_time,
        human_only_time=human_only_time,
        hrc_time_penalty_vs_human_only=observation_extra_time,
        observation_mode_extra_time=observation_extra_time,
        hrc_mean_time_per_recipe_step=observation_total_time / max(1, n_recipe_steps),
        hrc_mean_time_per_robot_turn=0.0,
        hrc_robot_correct_time=0.0,
        hrc_robot_wrong_time=0.0,
        hrc_human_action_time=human_only_time,
        hrc_human_correction_time=0.0,
        hrc_wrong_action_extra_penalty_time=0.0,
    )
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
        "variant_hash": cls.variant_hash,
        "jaccard": cls.jaccard,
        "order_distance": cls.order_distance,
        "n_steps": 0,
        "n_recipe_steps": n_recipe_steps,
        "live_top1": 0.0,
        "live_topk": 0.0,
        "total_robot_turn_nll": 0.0,
        "nll_scored_robot_turns": 0.0,
        "nll_normalization_recipe_steps": 0.0,
        "mean_nll_per_robot_turn": 0.0,
        "mean_nll_per_recipe_step": 0.0,
        **_adaptation_recovery_metrics([], -1),
        "robot_wrong_rate": 0.0,
        "net_robot_action_value": 0.0,
        "hrc_robot_turn_count": 0,
        "hrc_human_turn_count": n_recipe_steps,
        "hrc_human_correction_count": 0,
        "hrc_human_action_count": n_recipe_steps,
        "hrc_robot_correct_count": 0,
        "hrc_robot_wrong_count": 0,
        "human_correction_rate": 0.0,
        "human_action_fraction": 1.0,
        "human_effort_time": human_effort_time,
        "human_effort_fraction": human_effort_fraction,
        "primary_hrc_metric": "human_action_fraction",
        "primary_hrc_metric_value": 1.0,
        "human_workload_reduction_vs_human_only": 0.0,
        "hrc_future_valid_wrong_count": 0,
        "future_valid_wrong_rate": 0.0,
        "wrong_prediction_future_valid_rate": 0.0,
        "timing_details": timing_details,
        "observation_mode_episode": 1.0,
        "user_observation_required": 1.0,
        "mode_decision_source": "user",
        "mode_route_reason": "user_selected_observation",
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


def _recipe_has_active_variant(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Dict[str, str]) -> bool:
    rid = name_to_rid.get(pair.recipe_name)
    if rid is None:
        return False
    return any(recipe_id == rid for recipe_id, _variant_hash in agent.decay.active.keys())


def _user_selected_mode_for_active_memory(
    requested_mode: str,
    agent: AdaptiveHRCAgent,
    pair: RecipePrefPair,
    name_to_rid: Dict[str, str],
) -> Tuple[str, Dict[str, Any]]:
    active_recipe_before = _recipe_has_active_variant(agent, pair, name_to_rid)
    if requested_mode == "observe":
        mode = "observe"
        reason = "user_selected_observation"
    elif active_recipe_before:
        mode = "assist"
        reason = "user_selected_assist_active_recipe"
    else:
        mode = "observe"
        reason = "user_selected_observation_recipe_absent_from_active_memory"
    return mode, {
        "requested_mode": requested_mode,
        "executed_mode": mode,
        "mode_decision_source": "user",
        "mode_route_reason": reason,
        "active_recipe_before": bool(active_recipe_before),
        "recipe_active_before": bool(active_recipe_before),
    }


def _policy_diagnostics_from_gate(gate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "would_assist_under_gate": bool(gate.get("assist_used", False)),
        "gate_reason": gate.get("reason"),
        "gate_source": gate.get("assist_source"),
        "policy_confidence": gate.get("final_action_confidence", gate.get("action_confidence")),
        "policy_raw_confidence": gate.get("raw_action_confidence"),
        "policy_gate_score": gate.get("action_gate_score"),
        "policy_margin": gate.get("final_action_margin", gate.get("action_margin")),
        "policy_entropy": gate.get("final_action_entropy", gate.get("action_entropy")),
        "gate_threshold": gate.get("action_gate_threshold"),
        "policy_final_confidence": gate.get("final_action_confidence"),
        "policy_final_margin": gate.get("final_action_margin"),
        "policy_final_entropy": gate.get("final_action_entropy"),
        "conditioned_policy_confidence": gate.get("conditioned_action_confidence"),
        "ensemble_policy_confidence": gate.get("ensemble_action_confidence"),
        "policy_agreement": gate.get("policy_agreement"),
        "blend_strength": gate.get("blend_strength"),
    }


def _posterior_stats(agent: AdaptiveHRCAgent) -> Dict[str, Any]:
    joint = agent.posterior.joint()
    gate = agent.assist_gate_stats()
    policy_diagnostics = _policy_diagnostics_from_gate(gate)
    if not joint:
        return {
            "posterior_entropy": None,
            "posterior_confidence": None,
            "posterior_raw_entropy": None,
            "posterior_max_prob": None,
            "posterior_n_hypotheses": 0,
            "posterior_recipe": agent.inferred_recipe,
            "posterior_preference": None,
            "policy_diagnostics": policy_diagnostics,
        }
    probs = [float(p) for p in joint.values()]
    raw_h = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    norm_h = raw_h / math.log(len(probs)) if len(probs) > 1 else None
    return {
        "posterior_entropy": norm_h,
        "posterior_confidence": max(probs),
        "posterior_raw_entropy": raw_h,
        "posterior_max_prob": max(probs),
        "posterior_n_hypotheses": len(probs),
        "posterior_recipe": agent.posterior.argmax_recipe(),
        "posterior_preference": agent.posterior.argmax_preference(),
        "policy_diagnostics": policy_diagnostics,
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
    probe_snapshot = agent.snapshot() if not commit else None
    rid = (name_to_rid or {}).get(pair.recipe_name)
    if simulator_observed_labels is not None or simulator_observed_recipes is not None:
        memory_state = _memory_state_gt(agent, pair, name_to_rid or {}, simulator_observed_labels, simulator_observed_recipes)
    else:
        memory_state = _memory_state(agent, pair, name_to_rid or {})
    observations = observations_from_actions(pair.actions)
    actual_tokens = agent._tokens_from_action_labels(pair.actions)
    if isinstance(agent, OracleCeilingAgent):
        agent.set_oracle_target(actual_tokens)
    floor = max(float(agent.cfg.prob_floor), 1e-12)

    uses_oracle_label_prediction = oracle_recipe_id is not None or oracle_preference_id is not None

    def predict_for_prefix(prefix: Sequence[str]) -> Mapping[str, float]:
        if uses_oracle_label_prediction:
            return agent.predict_with_oracle(list(prefix), oracle_recipe_id or rid, oracle_preference_id)
        return agent.predict_next_tokens(list(prefix))

    def observe_ground_truth(obs: Any, distribution: Optional[Mapping[str, float]] = None) -> None:
        precomputed_distribution = None if uses_oracle_label_prediction else distribution
        agent.observe_observation(
            obs,
            ground_truth_recipe=rid,
            precomputed_distribution=precomputed_distribution,
        )

    def capture_robot_metadata(context: HRCDecisionContext) -> Dict[str, Any]:
        post = _posterior_stats(agent)
        policy_diagnostics = dict(post.get("policy_diagnostics") or {})
        policy_diagnostics.update(
            policy_distribution_diagnostics(context.distribution, context.ranked, context.predicted)
        )
        post["policy_diagnostics"] = policy_diagnostics
        return post

    trace = run_alternating_hrc_episode(
        observations=observations,
        actual_tokens=actual_tokens,
        actual_actions=pair.actions,
        current_prefix=lambda: list(agent.current_prefix),
        predict_distribution=predict_for_prefix,
        observe_ground_truth=observe_ground_truth,
        topk=max(1, int(run_config.topk)),
        prob_floor=floor,
        capture_robot_metadata=capture_robot_metadata,
    )
    steps: List[Dict[str, Any]] = []
    live_steps: List[LiveStepRecord] = []
    for turn in trace.robot_turns:
        ranked = list(turn.ranked)
        dist = dict(turn.distribution)
        post = dict(turn.metadata)
        policy = dict(post.get("policy_diagnostics") or {})
        row = {
            "step": turn.recipe_step,
            "recipe_step": turn.recipe_step,
            "robot_turn_idx": turn.robot_turn_idx,
            "pair": pair.label,
            "recipe": pair.recipe_name,
            "preference": pair.preference_name,
            "memory_state": memory_state,
            "memory_state_gt": memory_state,
            "actual": turn.actual,
            "actual_action": turn.actual_action,
            "predicted": turn.predicted,
            "correct_top1": turn.correct_top1,
            "correct_topk": turn.correct_topk,
            "topk": ranked,
            "topk_probs": {k: float(dist.get(k, 0.0)) for k in ranked},
            "valid_next_action_count": int(max(1, len(dist))),
            "prediction_wall_s": float(turn.prediction_wall_s),
            "scheduled_actor": turn.scheduled_actor,
            "executed_by": turn.executed_by,
            "human_corrected": turn.human_corrected,
            "future_valid_wrong": turn.future_valid_wrong,
            "predicted_future_offset": turn.predicted_future_offset,
            "hrc_step_time": float(turn.hrc_step_time),
            "hrc_total_time_so_far": float(turn.hrc_total_time_so_far),
            "human_actions_so_far": int(turn.human_actions_so_far),
            "human_effort_time_so_far": float(turn.human_effort_time_so_far),
            "inferred_recipe": post.get("posterior_recipe"),
            "inferred_variant_hash": getattr(agent, "inferred_variant_hash", None),
            "inferred_latent_pref_id": getattr(agent, "inferred_latent_pref_id", None),
            "locked_variant_key": (
                list(getattr(agent, "locked_variant_key"))
                if getattr(agent, "locked_variant_key", None) is not None else None
            ),
            **post,
        }
        if event_tag:
            row.update(event_tag)
        if run_config.log_full_distributions:
            row["distribution"] = {k: float(v) for k, v in sorted(dist.items(), key=lambda kv: -kv[1])}
        steps.append(row)
        live_steps.append(LiveStepRecord(
            step=turn.robot_turn_idx,
            predicted=turn.predicted,
            actual=turn.actual,
            correct_top1=turn.correct_top1,
            correct_topk=turn.correct_topk,
            topk=tuple(ranked),
            inferred_recipe=post.get("posterior_recipe"),
            inferred_variant_hash=getattr(agent, "inferred_variant_hash", None),
            inferred_latent_pref_id=post.get("posterior_preference"),
            posterior_entropy=post.get("posterior_entropy"),
            posterior_confidence=post.get("posterior_confidence"),
            posterior_raw_entropy=post.get("posterior_raw_entropy"),
            posterior_max_prob=post.get("posterior_max_prob"),
            posterior_n_hypotheses=post.get("posterior_n_hypotheses"),
            would_assist_under_gate=bool(policy.get("would_assist_under_gate", False)),
            policy_confidence=policy.get("policy_confidence"),
            policy_raw_confidence=policy.get("policy_raw_confidence"),
            policy_gate_score=policy.get("policy_gate_score"),
            policy_margin=policy.get("policy_margin"),
            gate_threshold=policy.get("gate_threshold"),
            policy_entropy=policy.get("policy_entropy"),
            gate_source=policy.get("gate_source"),
            gate_reason=policy.get("gate_reason"),
            policy_final_confidence=policy.get("policy_final_confidence"),
            policy_final_margin=policy.get("policy_final_margin"),
            policy_final_entropy=policy.get("policy_final_entropy"),
            conditioned_policy_confidence=policy.get("conditioned_policy_confidence"),
            ensemble_policy_confidence=policy.get("ensemble_policy_confidence"),
            policy_agreement=policy.get("policy_agreement"),
            blend_strength=policy.get("blend_strength"),
            prediction_wall_s=float(turn.prediction_wall_s),
        ))
    cls = None
    if commit:
        cls = agent.end_demo()
        if name_to_rid is not None and cls.recipe_id is not None:
            name_to_rid[pair.recipe_name] = cls.recipe_id
    else:
        agent.current_prefix = []
        agent.inferred_recipe = None
        agent.inferred_variant_hash = None
        if hasattr(agent, "inferred_latent_pref_id"):
            agent.inferred_latent_pref_id = None
        if hasattr(agent, "_needs_observation"):
            agent._needs_observation = False
        if hasattr(agent, "_pending_argmax"):
            agent._pending_argmax = None
        if hasattr(agent, "_pending_count"):
            agent._pending_count = 0
        if hasattr(agent, "locked_variant_key"):
            agent.locked_variant_key = None
        if hasattr(agent, "locked_pref_id"):
            agent.locked_pref_id = None
        agent.posterior.reset()
    summary = trace.summary
    n_robot_turns = summary.n_robot_turns
    n = max(1, n_robot_turns)
    n_recipe_steps = max(1, int(summary.n_recipe_steps))
    total_robot_turn_nll = float(summary.nll)
    mean_nll_per_robot_turn = float(total_robot_turn_nll / max(1, n_robot_turns))
    mean_nll_per_recipe_step = float(total_robot_turn_nll / n_recipe_steps)
    first_mismatch = summary.first_mismatch_robot_turn
    first_mismatch_recipe_step = summary.first_mismatch_recipe_step
    recipe_hits = [s for s in steps if rid is not None and s.get("posterior_recipe") == rid]
    confidence_vals = [v for s in steps for v in (_row_policy_value(s, "policy_confidence"),) if v is not None]
    raw_confidence_vals = [v for s in steps for v in (_row_policy_value(s, "policy_raw_confidence"),) if v is not None]
    gate_score_vals = [v for s in steps for v in (_row_policy_value(s, "policy_gate_score"),) if v is not None]
    margin_vals = [v for s in steps for v in (_row_policy_value(s, "policy_margin"),) if v is not None]
    policy_entropy_vals = [v for s in steps for v in (_row_policy_value(s, "policy_entropy"),) if v is not None]
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
        live_top1=summary.hits1 / n,
        live_topk=summary.hitsk / n,
        n=n_robot_turns,
        first_mismatch_robot_turn=None if first_mismatch < 0 else first_mismatch,
    )
    rich_metrics = live_episode_metrics(live_record, true_rid=rid)
    human_only_time = float(summary.human_only_time)
    hrc_time_penalty = float(summary.hrc_total_time - human_only_time)
    human_action_count = int(summary.human_turn_count + summary.human_correction_count)
    human_correction_rate = float(summary.human_correction_count / max(1, n_robot_turns))
    human_action_fraction = float(summary.human_action_fraction)
    human_effort_time = float(summary.human_effort_time)
    human_effort_fraction = float(summary.human_effort_fraction)
    timing_details = _hrc_timing_details(
        hrc_total_time=float(summary.hrc_total_time),
        human_only_time=human_only_time,
        hrc_time_penalty_vs_human_only=hrc_time_penalty,
        observation_mode_extra_time=0.0,
        hrc_mean_time_per_recipe_step=float(summary.hrc_total_time / max(1, len(actual_tokens))),
        hrc_mean_time_per_robot_turn=float(summary.hrc_total_time / max(1, n_robot_turns)),
        hrc_robot_correct_time=float(summary.hrc_robot_correct_time),
        hrc_robot_wrong_time=float(summary.hrc_robot_wrong_time),
        hrc_human_action_time=float(summary.hrc_human_action_time),
        hrc_human_correction_time=float(summary.hrc_human_correction_time),
        hrc_wrong_action_extra_penalty_time=float(summary.hrc_wrong_action_extra_penalty_time),
    )
    metrics = {
        "pair": pair.label,
        "recipe": pair.recipe_name,
        "preference": pair.preference_name,
        "mode": "assist",
        "memory_state": memory_state,
        "memory_state_gt": memory_state,
        "n_steps": n_robot_turns,
        "n_recipe_steps": len(actual_tokens),
        "live_top1": summary.hits1 / n,
        "live_topk": summary.hitsk / n,
        "total_robot_turn_nll": total_robot_turn_nll,
        "nll_scored_robot_turns": float(n_robot_turns),
        "nll_normalization_recipe_steps": float(summary.n_recipe_steps),
        "mean_nll_per_robot_turn": mean_nll_per_robot_turn,
        "mean_nll_per_recipe_step": mean_nll_per_recipe_step,
        "first_mismatch_robot_turn": first_mismatch,
        "first_mismatch_recipe_step": first_mismatch_recipe_step,
        "posterior_correct_recipe": len(recipe_hits) / n if rid is not None else 0.0,
        "mean_policy_confidence": _mean(confidence_vals),
        "mean_policy_raw_confidence": _mean(raw_confidence_vals),
        "mean_policy_gate_score": _mean(gate_score_vals),
        "mean_policy_margin": _mean(margin_vals),
        "mean_posterior_entropy": _mean(entropy_vals),
        "mean_policy_entropy": _mean(policy_entropy_vals),
        "posterior_degenerate_rate": len(degenerate) / n,
        "classification_kind": getattr(cls, "kind", None),
        "classification_recipe_id": getattr(cls, "recipe_id", None),
        "classification_variant_hash": getattr(cls, "variant_hash", None),
        "needs_observation": getattr(agent, "_needs_observation", False) or getattr(cls, "kind", None) == "needs_observation",
        "live_record": live_record,
        **rich_metrics,
        "hrc_robot_turn_count": n_robot_turns,
        "hrc_human_turn_count": summary.human_turn_count,
        "hrc_human_correction_count": summary.human_correction_count,
        "hrc_human_action_count": human_action_count,
        "hrc_robot_correct_count": summary.robot_correct_count,
        "hrc_robot_wrong_count": summary.robot_wrong_count,
        "human_correction_rate": human_correction_rate,
        "human_action_fraction": human_action_fraction,
        "human_effort_time": human_effort_time,
        "human_effort_fraction": human_effort_fraction,
        "primary_hrc_metric": "human_action_fraction",
        "primary_hrc_metric_value": human_action_fraction,
        "human_workload_reduction_vs_human_only": float(1.0 - human_action_fraction),
        "hrc_future_valid_wrong_count": summary.future_valid_wrong_count,
        # future_valid_wrong_rate omitted from primary metrics: a wrong
        # prediction is wrong regardless of whether it matches a future action.
        # future_valid_wrong_count is kept for per-turn diagnostic inspection.
        "wrong_prediction_future_valid_rate": (
            float(summary.future_valid_wrong_count / summary.robot_wrong_count)
            if summary.robot_wrong_count > 0 else float("nan")
            # NaN when no wrong predictions: the rate is undefined, not zero.
            # _mean() already filters NaN before aggregating across episodes.
        ),
        "timing_details": timing_details,
        "observation_mode_episode": 0.0,
        "user_observation_required": 0.0,
        "mode_decision_source": "user",
        "mode_route_reason": "user_selected_assist_active_recipe",
        **(event_tag or {}),
    }
    if probe_snapshot is not None:
        agent.restore_from(probe_snapshot)
    return metrics, steps


def frozen_eval(agent: AdaptiveHRCAgent, pairs: Sequence[RecipePrefPair], name_to_rid: Optional[Dict[str, str]] = None, run_config: Optional[RunConfig] = None) -> Dict[str, Dict[str, float]]:
    """Evaluate the agent using the HRC protocol without updating any state.

    Accuracy is measured only on the steps the robot is scheduled to carry out —
    matching the live deployment metric exactly. Human turns and corrections are
    accounted for in timing but not counted toward accuracy.

    This intentionally uses snapshot/restore rather than ``agent.frozen()``.
    Frozen mode blocks ``observe_observation`` from advancing the online prefix,
    which would make later robot turns condition on the wrong interaction history.
    """
    if getattr(agent, "_frozen", False):
        raise RuntimeError("frozen_eval must not be called inside agent.frozen(); it manages its own snapshot/restore")
    _run_config = run_config if run_config is not None else RunConfig()
    out: Dict[str, Dict[str, float]] = {}
    for pair in pairs:
        checkpoint = agent.snapshot()
        try:
            row, _ = assist_episode(
                agent, pair, dict(name_to_rid or {}),
                run_config=_run_config,
                commit=False,
            )
        finally:
            agent.restore_from(checkpoint)
        topk_value = max(1, int(_run_config.topk))
        row_out = {
            "top1": float(row.get("live_top1", 0.0)),
            "topk": float(row.get("live_topk", 0.0)),
            "n": float(row.get("n_steps", 0)),
            "live_top1": float(row.get("live_top1", 0.0)),
            "live_topk": float(row.get("live_topk", 0.0)),
        }
        row_out[f"top{topk_value}"] = float(row.get("live_topk", 0.0))
        out[pair.label] = row_out
    return out


def _hrc_net_action_value(row: Mapping[str, Any], *, wrong_assist_penalty: float = ASSIST_WRONG_PENALTY) -> float:
    if isinstance(row.get("net_robot_action_value"), (int, float)):
        return float(row.get("net_robot_action_value", 0.0))
    if isinstance(row.get("net_assistance_value"), (int, float)):
        return float(row.get("net_assistance_value", 0.0))
    live_top1 = float(row.get("live_top1", 0.0))
    wrong_rate = float(row.get("robot_wrong_rate", row.get("assist_wrong_rate", row.get("wrong_assistance_rate", 0.0))))
    return float(live_top1 - float(wrong_assist_penalty) * wrong_rate)


def _oracle_reference_metrics(
    seed: int,
    run_config: RunConfig,
    train_pairs: Sequence[RecipePrefPair],
    eval_pairs: Sequence[RecipePrefPair],
    *,
    max_eval_pairs: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate non-deployable oracle references on a fixed train/eval split.

    These are reported as reference lines, not deployable baselines. The recipe
    and preference oracles only reveal labels that can be mapped through the
    training checkpoint; the next-action oracle reads the simulator target.
    """
    eval_list = list(eval_pairs)
    if max_eval_pairs is not None:
        eval_list = eval_list[: max(0, int(max_eval_pairs))]
    if not eval_list:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for oracle_name in ORACLE_REFERENCE_BASELINES:
        agent_name = "oracle_ceiling" if oracle_name == "oracle_ceiling" else "full"
        agent = make_agent(agent_name, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        pref_to_pid: Dict[str, Optional[str]] = {}
        for pair in train_pairs:
            observe_episode(agent, pair, name_to_rid)
            if pair.preference_name not in pref_to_pid:
                pref_to_pid[pair.preference_name] = getattr(agent, "last_pref_id", None)
        rows: List[Dict[str, Any]] = []
        for pair in eval_list:
            kwargs: Dict[str, Any] = {}
            if oracle_name in {"oracle_recipe_label", "oracle_recipe_and_preference_label"}:
                kwargs["oracle_recipe_id"] = name_to_rid.get(pair.recipe_name)
            if oracle_name in {"oracle_preference_label", "oracle_recipe_and_preference_label"}:
                kwargs["oracle_preference_id"] = pref_to_pid.get(pair.preference_name)
            row, _steps = assist_episode(
                agent,
                pair,
                dict(name_to_rid),
                run_config=run_config,
                commit=False,
                event_tag={"oracle_reference": oracle_name},
                **kwargs,
            )
            rows.append(row)
        agg = _aggregate_episode_metrics(rows)
        ci95 = {
            metric: _stderr95([
                float(r.get(metric, 0.0))
                for r in rows
                if isinstance(r.get(metric, None), (int, float))
            ])
            for metric in (
                "live_top1",
                "live_topk",
                "adaptation_recovery_top1_w3",
                "robot_wrong_rate",
                "net_robot_action_value",
            )
        }
        out[oracle_name] = {
            **agg,
            "oracle_type": oracle_name,
            "mean": {
                metric: float(agg.get(metric, 0.0))
                for metric in ci95
            },
            "ci95": ci95,
            "n": float(agg.get("n_steps", 0.0)),
            "n_eval_pairs": len(eval_list),
            "reported_as": "dashed_reference_not_deployable",
        }
    return out


def _event_distribution(events: Sequence[ScenarioEvent]) -> Dict[str, Any]:
    actual = Counter(str(e.tags.get("event_type", "unknown")) for e in events if e.phase != "phase_a")
    requested = Counter(str(e.tags.get("requested_event_type", e.tags.get("event_type", "unknown"))) for e in events if e.phase != "phase_a")
    assist_events = [e for e in events if e.mode == "assist" and e.phase != "phase_a"]
    substituted = [e for e in assist_events if bool(e.tags.get("event_substituted", False))]
    total = max(1, len(assist_events))
    return {
        "requested_counts": dict(requested),
        "actual_counts": dict(actual),
        "requested_distribution": {k: v / max(1, sum(requested.values())) for k, v in requested.items()},
        "actual_distribution": {k: v / max(1, sum(actual.values())) for k, v in actual.items()},
        "event_substituted_rate": len(substituted) / total,
        "n_assist_events": len(assist_events),
    }


def _scaled_event_mix(base_mix: Mapping[str, float], routine_frac: float) -> Dict[str, float]:
    routine = max(0.0, min(1.0, float(routine_frac)))
    others = {k: max(0.0, float(v)) for k, v in base_mix.items() if k != "routine_reuse"}
    total_other = sum(others.values())
    if total_other <= 0.0:
        return {"routine_reuse": 1.0}
    scaled = {"routine_reuse": routine}
    for key, value in others.items():
        scaled[key] = (1.0 - routine) * value / total_other
    return scaled


def _event_mix_sensitivity(
    seed: int,
    run_config: RunConfig,
    cfg_obj: DeploymentStreamConfig,
) -> Dict[str, Any]:
    fractions = tuple(float(x) for x in getattr(cfg_obj, "event_mix_sweep_routine_fracs", ()) or ())
    if not fractions:
        return {}
    baselines = tuple(b for b in PAPER_BASELINES if b in set(run_config.baselines)) or ("full",)
    max_events = min(60, int(cfg_obj.n_phase_b_events))
    out: Dict[str, Any] = {}
    for frac in fractions:
        sub_cfg = replace(
            cfg_obj,
            event_mix=_scaled_event_mix(cfg_obj.event_mix, frac),
            n_phase_b_events=max_events,
            event_mix_sweep_routine_fracs=(),
        )
        _recipes, _users, events, _phase_a_pairs, _new_recipe_rejections = _deployment_stream_events(seed + int(round(frac * 1000)), sub_cfg, run_config)
        per_baseline: Dict[str, Any] = {}
        for baseline in baselines:
            row, _episode_rows, _step_rows, _memory_rows = _run_baseline_stream(
                baseline,
                base_config(seed, run_config),
                events,
                replace(run_config, baselines=(baseline,)),
            )
            per_baseline[baseline] = row
        full_top1 = float(per_baseline.get("full", {}).get("live_top1", 0.0))
        out[str(frac)] = {
            "routine_reuse_frac": frac,
            "event_distribution": _event_distribution(events),
            "per_baseline": per_baseline,
            "full_minus_baseline_live_top1": {
                b: full_top1 - float(row.get("live_top1", 0.0))
                for b, row in per_baseline.items()
                if b != "full"
            },
        }
    return out


def _memory_key_for_pair(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Mapping[str, str]) -> Optional[VariantKey]:
    rid = name_to_rid.get(pair.recipe_name)
    if rid is None:
        return None
    return (rid, variant_hash(agent._tokens_from_action_labels(pair.actions)))


def _online_best_so_far_forgetting_rows(
    agent: AdaptiveHRCAgent,
    pairs: Sequence[RecipePrefPair],
    name_to_rid: Mapping[str, str],
    evals: Mapping[str, Mapping[str, float]],
    best_top1_by_pair: Dict[str, float],
    *,
    baseline: str,
    event_idx: int,
    phase_b_count: int,
    phase: str,
    mode: str,
    event_type: str,
    displaced_at: Mapping[VariantKey, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        eval_row = evals.get(pair.label, {}) or {}
        current_top1 = float(eval_row.get("top1", 0.0))
        current_topk = float(eval_row.get("topk", 0.0))
        prior_best = float(best_top1_by_pair.get(pair.label, current_top1))
        forgetting = max(0.0, prior_best - current_top1)
        best_top1_by_pair[pair.label] = max(prior_best, current_top1)
        key = _memory_key_for_pair(agent, pair, name_to_rid)
        rid = key[0] if key is not None else None
        h = key[1] if key is not None else None
        active_entry = agent.decay.active.get(key) if key is not None else None
        pruned_entry = agent.decay.pruned.get(key) if key is not None else None
        latest_hash = agent.decay.latest_by_recipe.get(rid) if rid is not None else None
        is_latest = bool(h is not None and latest_hash == h)
        if key is None:
            subgroup = "unmapped"
        elif is_latest:
            subgroup = "latest_pin"
        elif active_entry is not None:
            subgroup = "displaced_active"
        elif pruned_entry is not None:
            subgroup = "pruned"
        else:
            subgroup = "known_in_registry"
        rows.append({
            "baseline": baseline,
            "event_idx": int(event_idx),
            "phase_b_count": int(phase_b_count),
            "phase": phase,
            "mode": mode,
            "event_type": event_type,
            "pair": pair.label,
            "recipe": pair.recipe_name,
            "preference": pair.preference_name,
            "subgroup": subgroup,
            "top1": current_top1,
            "topk": current_topk,
            "best_top1_before": prior_best,
            "best_top1_ever": float(best_top1_by_pair[pair.label]),
            "best_so_far_forgetting_top1": forgetting,
            "weight": float(active_entry.weight) if active_entry is not None else 0.0,
            "sessions_since_displaced": 0 if is_latest or key is None else max(0, int(phase_b_count) - int(displaced_at.get(key, 0))),
            "sessions_since_pruned": (
                max(0, int(agent.session_counter) - int(pruned_entry.removed_step))
                if pruned_entry is not None else 0
            ),
            "variant_is_latest_pin": bool(is_latest),
            "variant_is_active": bool(active_entry is not None),
            "variant_is_pruned": bool(pruned_entry is not None),
        })
    return rows


def _summarize_online_forgetting_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vals = [float(r.get("best_so_far_forgetting_top1", 0.0)) for r in rows]
    worst_by_pair: Dict[str, float] = {}
    final_by_pair: Dict[str, float] = {}
    for row in rows:
        pair = str(row.get("pair", ""))
        val = float(row.get("best_so_far_forgetting_top1", 0.0))
        worst_by_pair[pair] = max(float(worst_by_pair.get(pair, 0.0)), val)
        final_by_pair[pair] = val
    by_subgroup: Dict[str, Any] = {}
    for subgroup in sorted({str(r.get("subgroup", "unknown")) for r in rows}):
        subgroup_vals = [
            float(r.get("best_so_far_forgetting_top1", 0.0))
            for r in rows
            if str(r.get("subgroup", "unknown")) == subgroup
        ]
        by_subgroup[subgroup] = {
            "n_rows": float(len(subgroup_vals)),
            "mean_forgetting_top1": _mean(subgroup_vals),
            "p95_forgetting_top1": _p95(subgroup_vals),
            "max_forgetting_top1": max(subgroup_vals, default=0.0),
            "forgetting_event_rate": _mean([1.0 if v > 1e-12 else 0.0 for v in subgroup_vals]),
            "histogram": _histogram(subgroup_vals),
        }
    return {
        "metric_definition": "max(0, best_top1_ever_seen_for_pair_before_event - top1_after_learning_event)",
        "sampling": "after_every_learning_event_after_initial_onboarding",
        "n_rows": float(len(rows)),
        "n_pairs": float(len(worst_by_pair)),
        "n_events": float(len({int(r.get("event_idx", -1)) for r in rows})),
        "mean_forgetting_top1": _mean(vals),
        "p95_forgetting_top1": _p95(vals),
        "max_forgetting_top1": max(vals, default=0.0),
        "forgetting_event_rate": _mean([1.0 if v > 1e-12 else 0.0 for v in vals]),
        "all_row_histogram": _histogram(vals),
        "worst_per_pair_mean": _mean(list(worst_by_pair.values())),
        "worst_per_pair_p95": _p95(list(worst_by_pair.values())),
        "worst_per_pair_max": max(worst_by_pair.values(), default=0.0),
        "worst_per_pair_histogram": _histogram(list(worst_by_pair.values())),
        "final_per_pair_mean": _mean(list(final_by_pair.values())),
        "final_per_pair_histogram": _histogram(list(final_by_pair.values())),
        "by_subgroup": by_subgroup,
    }


def _summarize_preference_reentry_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        "n_reentries": float(len(rows)),
        "pre_reentry_top1": _mean([float(r.get("pre_reentry_top1", 0.0)) for r in rows]),
        "during_reentry_step_top1": _mean([float(r.get("during_reentry_step_top1", 0.0)) for r in rows]),
        "post_reentry_top1": _mean([float(r.get("post_reentry_top1", 0.0)) for r in rows]),
        "steps_to_recover": _mean([float(r.get("steps_to_recover", -1.0)) for r in rows if float(r.get("steps_to_recover", -1.0)) >= 0.0]),
        "recovery_failure_rate": _mean([0.0 if r.get("recovered") else 1.0 for r in rows]),
        "robot_wrong_rate_before_recovery": _mean([float(r.get("robot_wrong_rate_before_recovery", r.get("wrong_assist_rate_before_recovery", 0.0))) for r in rows]),
    }


def _variant_state_for_pair(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Mapping[str, str]) -> str:
    key = _memory_key_for_pair(agent, pair, name_to_rid)
    if key is None:
        return "unseen"
    rid, h = key
    if agent.decay.latest_by_recipe.get(rid) == h:
        return "latest_pin"
    if key in agent.decay.active:
        return "displaced_active"
    if key in agent.decay.pruned:
        return "pruned"
    return "known_in_registry"


def _preference_decay_profile(seed: int, run_config: RunConfig, cfg_obj: DeploymentStreamConfig) -> Dict[str, Any]:
    builders = select_recipe_builders(seed + 7919, max(3, int(cfg_obj.n_recipes)))
    target_recipe = ""
    target_fn: Optional[Callable[[], List[str]]] = None
    target_pairs: List[RecipePrefPair] = []
    for recipe, fn in builders:
        pairs: List[RecipePrefPair] = []
        seen_orderings: set = set()
        for pref in PREFERENCE_NAMES:
            try:
                pair = materialize_pair(recipe, pref, fn)
            except Exception:
                continue
            if pair.actions in seen_orderings:
                continue
            seen_orderings.add(pair.actions)
            pairs.append(pair)
            if len(pairs) >= 3:
                break
        if len(pairs) >= 3:
            target_recipe = recipe
            target_fn = fn
            target_pairs = pairs[:3]
            break
    if target_fn is None or len(target_pairs) < 3:
        return {
            "status": "insufficient_material_preferences",
            "rows": [],
            "displacements": [],
        }
    distractors: List[RecipePrefPair] = []
    for recipe, fn in builders:
        if recipe == target_recipe:
            continue
        for pref in ("identity", target_pairs[0].preference_name, target_pairs[1].preference_name):
            try:
                distractors.append(materialize_pair(recipe, pref, fn))
            except Exception:
                continue
            if len(distractors) >= 12:
                break
        if len(distractors) >= 12:
            break
    if not distractors:
        for pref in PREFERENCE_NAMES:
            if pref not in {p.preference_name for p in target_pairs}:
                try:
                    distractors.append(materialize_pair(target_recipe, pref, target_fn))
                except Exception:
                    continue
            if distractors:
                break
    agent = make_agent("full", base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    displacements: List[Dict[str, Any]] = []

    def record(elapsed: int, checkpoint: str) -> None:
        evals = frozen_eval(agent, target_pairs, name_to_rid)
        for idx, pair in enumerate(target_pairs, start=1):
            rows.append({
                "elapsed_sessions": int(elapsed),
                "checkpoint": checkpoint,
                "recipe": pair.recipe_name,
                "pair": pair.label,
                "preference_index": idx,
                "preference": pair.preference_name,
                "top1": float((evals.get(pair.label, {}) or {}).get("top1", 0.0)),
                "topk": float((evals.get(pair.label, {}) or {}).get("topk", 0.0)),
                "state": _variant_state_for_pair(agent, pair, name_to_rid),
            })

    observe_episode(agent, target_pairs[0], name_to_rid)
    record(0, "after_p1")
    observe_episode(agent, target_pairs[1], name_to_rid)
    displacements.append({
        "elapsed_sessions": 1,
        "label": "p1 displaced by p2",
        "displaced_pair": target_pairs[0].label,
        "displaced_preference": target_pairs[0].preference_name,
        "incoming_pair": target_pairs[1].label,
        "incoming_preference": target_pairs[1].preference_name,
    })
    record(1, "after_p2")
    observe_episode(agent, target_pairs[2], name_to_rid)
    displacements.append({
        "elapsed_sessions": 2,
        "label": "p2 displaced by p3",
        "displaced_pair": target_pairs[1].label,
        "displaced_preference": target_pairs[1].preference_name,
        "incoming_pair": target_pairs[2].label,
        "incoming_preference": target_pairs[2].preference_name,
    })
    record(2, "after_p3")
    max_tail = min(max(8, int(cfg_obj.n_phase_b_events)), 30)
    checkpoints = sorted({3, 5, 10, max_tail})
    elapsed = 2
    for target_elapsed in checkpoints:
        while elapsed < target_elapsed:
            if distractors:
                observe_episode(agent, distractors[elapsed % len(distractors)], name_to_rid)
            else:
                agent.decay._recompute_effective(agent.session_counter + 1)
            elapsed += 1
        record(elapsed, "distractor_gap")
    return {
        "status": "ok",
        "baseline": "full",
        "recipe": target_recipe,
        "tracked_pairs": [p.label for p in target_pairs],
        "distractor_pairs": [p.label for p in distractors],
        "rows": rows,
        "displacements": displacements,
        "caption_note": "Single-recipe p1->p2->p3 profile with unrelated distractor sessions; evaluation is frozen top-1 on the three tracked variants.",
    }


def _step_trace_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for step in sorted({int(r.get("step", 0)) for r in rows}):
        vals = [r for r in rows if int(r.get("step", 0)) == step]
        out.append({
            "step_idx": float(step),
            "top1": _mean([1.0 if r.get("correct_top1") else 0.0 for r in vals]),
            "topk": _mean([1.0 if r.get("correct_topk") else 0.0 for r in vals]),
            "robot_wrong_rate": _mean([1.0 if _row_robot_wrong(r) else 0.0 for r in vals]),
            "posterior_confidence": _mean([float(r.get("posterior_confidence", 0.0) or 0.0) for r in vals]),
            "n": float(len(vals)),
        })
    return out


def memory_snapshot(agent: AdaptiveHRCAgent) -> Dict[str, Any]:
    weights = agent.decay.weights()
    active_action_steps = sum(len(entry.ordering) for entry in agent.decay.active.values())
    prototype_events = list(getattr(agent, "prototype_events", []))
    latest_proto_event = prototype_events[-1] if prototype_events else {}
    latest_new = latest_proto_event.get("prototype_new", []) if isinstance(latest_proto_event, Mapping) else []
    latest_retired = latest_proto_event.get("prototype_retired", []) if isinstance(latest_proto_event, Mapping) else []
    latest_persisted = latest_proto_event.get("prototype_persisted", []) if isinstance(latest_proto_event, Mapping) else []
    latest_split_count = int(latest_proto_event.get("prototype_split_count", 0) or 0) if isinstance(latest_proto_event, Mapping) else 0
    latest_merge_count = int(latest_proto_event.get("prototype_merge_count", 0) or 0) if isinstance(latest_proto_event, Mapping) else 0
    latest_total = len(latest_new or []) + len(latest_retired or []) + len(latest_persisted or [])
    latest_churn = len(latest_new or []) + len(latest_retired or []) + latest_split_count + latest_merge_count
    replay_meta = {}
    if hasattr(agent, "replay_buffer_metadata"):
        try:
            replay_meta = dict(agent.replay_buffer_metadata())  # type: ignore[attr-defined]
        except Exception:
            replay_meta = dict(getattr(agent, "_last_replay_metadata", {}) or {})
    out = {
        "active_variants": len(agent.decay.active),
        "active_action_steps": int(active_action_steps),
        "active_examples": int(active_action_steps),
        "pruned_variants": len(agent.decay.pruned),
        "latest_keys": len(agent.decay.latest_keys),
        "base_rate": float(agent.decay.base_rate),
        "global_rate": float(agent.decay.global_rate),
        "reuse_gaps": list(agent.decay.reuse_gaps),
        "reuse_gap_window": list(agent.decay.window_snapshot()),
        "active_recipe_count_window": list(agent.decay.active_recipe_count_window_snapshot()),
        "mean_active_weight": _mean(list(weights.values())),
        "min_active_weight": min(weights.values()) if weights else 0.0,
        "max_active_weight": max(weights.values()) if weights else 0.0,
        "retrain_cycle": int(agent.retrain_cycle),
        "retrain_skipped_count": int(getattr(agent, "retrain_skipped_count", 0)),
        "observation_mode_entries": int(getattr(agent, "observation_mode_entries", 0)),
        "prototype_rebuild_count": len(prototype_events),
        "prototype_churn_rate": latest_churn / max(1, latest_total),
        "prototype_new_count": len(latest_new or []),
        "prototype_retired_count": len(latest_retired or []),
        "prototype_split_count": latest_split_count,
        "prototype_merge_count": latest_merge_count,
        "prototype_stability_ari": latest_proto_event.get("prototype_stability_ari") if isinstance(latest_proto_event, Mapping) else None,
        "prototype_active_intersection": latest_proto_event.get("prototype_active_intersection") if isinstance(latest_proto_event, Mapping) else 0,
    }
    if replay_meta:
        out["replay_buffer"] = replay_meta
        out["replay_policy"] = replay_meta.get("policy")
        out["er_buffer_size"] = int(replay_meta.get("er_buffer_size", agent.cfg.er_buffer_size))
        out["er_batch_size"] = int(replay_meta.get("er_batch_size", agent.cfg.er_batch_size))
        out["replay_memory_footprint"] = int(replay_meta.get("replay_memory_footprint", 0))
    return out


def variant_weight_for_pair(agent: AdaptiveHRCAgent, pair: RecipePrefPair, name_to_rid: Mapping[str, str]) -> Optional[float]:
    rid = name_to_rid.get(pair.recipe_name)
    if rid is None:
        return None
    h = variant_hash(agent._tokens_from_action_labels(pair.actions))
    entry = agent.decay.active.get((rid, h))
    return float(entry.weight) if entry is not None else None


def _learner_hyperparameter_metadata(agent: AdaptiveHRCAgent) -> Dict[str, Any]:
    cfg = agent.cfg
    return {
        "bc_learning_rate": float(cfg.bc_learning_rate),
        "bc_l2": float(cfg.bc_l2),
        "bc_history_len": int(cfg.bc_history_len),
        "bc_history_bins": int(cfg.bc_history_bins),
        "bc_epochs_cold": int(cfg.bc_epochs_cold),
        "bc_epochs_warm": int(cfg.bc_epochs_warm),
        "er_buffer_size": int(cfg.er_buffer_size),
        "er_batch_size": int(cfg.er_batch_size),
        "er_recency_alpha": float(cfg.er_recency_alpha),
        "er_uniform_mix": float(cfg.er_uniform_mix),
        "ewc_lambda": float(cfg.ewc_lambda),
        "ewc_fisher_clip": float(cfg.ewc_fisher_clip),
        "maxent_learning_rate": float(cfg.maxent_learning_rate),
        "maxent_l2": float(cfg.maxent_l2),
        "maxent_iters_cold": int(cfg.maxent_iters_cold),
        "maxent_iters_warm": int(cfg.maxent_iters_warm),
    }


def compute_snapshot(agent: AdaptiveHRCAgent, wall_s: float) -> Dict[str, Any]:
    profile = {k: {"calls": int(v[0]), "wall_s": float(v[1])} for k, v in getattr(agent, "profile", {}).items()}
    fit_times = [float(v) for v in getattr(agent, "retrain_fit_wall_times", [])]
    total_times = [float(v) for v in getattr(agent, "retrain_total_wall_times", [])]
    build_times = [float(v) for v in getattr(agent, "retrain_build_wall_times", [])]
    flop_estimates = [float(v) for v in getattr(agent, "retrain_flop_estimates", [])]
    replay_meta = dict(getattr(agent, "_last_replay_metadata", {}) or {})
    out = {
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
        "mean_estimated_flops": _mean(flop_estimates),
        "p95_estimated_flops": _p95(flop_estimates),
        "retrain_fit_wall_times": fit_times,
        "retrain_total_wall_times": total_times,
        "retrain_build_wall_times": build_times,
        "retrain_flop_estimates": flop_estimates,
        "retrain_events": _jsonable(getattr(agent, "retrain_events", [])),
        "profile": profile,
        "learner_hyperparameters": _learner_hyperparameter_metadata(agent),
    }
    if replay_meta:
        out["replay_buffer"] = _jsonable(replay_meta)
    return out


def _four_cell(seen_recipe: bool, seen_preference: bool) -> str:
    if seen_recipe and seen_preference:
        return "known_recipe_known_preference"
    if seen_recipe and not seen_preference:
        return "seen_recipe_new_preference"
    if (not seen_recipe) and seen_preference:
        return "unseen_recipe_seen_preference"
    return "unseen_unseen"


def _transfer_cell(
    *,
    seen_recipe: bool,
    seen_pair: bool,
    seen_preference_global: bool,
    seen_preference_same_recipe: bool,
    seen_preference_other_recipe: bool,
) -> str:
    if seen_pair or (seen_recipe and seen_preference_same_recipe):
        return "direct_retrieval"
    if seen_recipe and seen_preference_other_recipe:
        return "seen_recipe_preference_seen_elsewhere"
    if seen_recipe and not seen_preference_global:
        return "seen_recipe_new_preference"
    if (not seen_recipe) and seen_preference_global:
        return "unseen_recipe_seen_preference"
    return "unseen_unseen"


def _transfer_cell_for_pair(
    pair: RecipePrefPair,
    *,
    seen_recipes: set,
    seen_pairs: set,
    seen_preferences: set,
    seen_preferences_by_recipe: Mapping[str, set],
) -> str:
    same_recipe_prefs = seen_preferences_by_recipe.get(pair.recipe_name, set())
    return _transfer_cell(
        seen_recipe=pair.recipe_name in seen_recipes,
        seen_pair=pair.label in seen_pairs,
        seen_preference_global=pair.preference_name in seen_preferences,
        seen_preference_same_recipe=pair.preference_name in same_recipe_prefs,
        seen_preference_other_recipe=any(
            pair.preference_name in vals
            for recipe, vals in seen_preferences_by_recipe.items()
            if recipe != pair.recipe_name
        ),
    )


def _episode_timing_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    timing_details = row.get("timing_details")
    if isinstance(timing_details, Mapping) and key in timing_details:
        return float(timing_details.get(key, default) or 0.0)
    return float(row.get(key, default) or 0.0)


_EPISODE_METRIC_FALLBACK_KEYS: Dict[str, Tuple[str, ...]] = {
    "robot_wrong_rate": ("assist_wrong_rate", "wrong_assistance_rate", "wrong_assists_per_step"),
    "net_robot_action_value": ("net_assistance_value",),
    "mean_policy_confidence": ("mean_action_confidence", "mean_final_action_confidence"),
    "mean_policy_raw_confidence": ("mean_raw_action_confidence",),
    "mean_policy_gate_score": ("mean_action_gate_score",),
    "mean_policy_margin": ("mean_action_margin",),
    "mean_policy_entropy": ("mean_action_marginal_entropy",),
    "mean_policy_final_confidence": ("mean_final_action_confidence",),
    "mean_conditioned_policy_confidence": ("mean_conditioned_action_confidence",),
    "mean_ensemble_policy_confidence": ("mean_ensemble_action_confidence",),
}


def _episode_metric_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    for fallback_key in _EPISODE_METRIC_FALLBACK_KEYS.get(key, ()):
        value = row.get(fallback_key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(default)


def _episode_robot_turn_count(row: Mapping[str, Any]) -> float:
    for key in ("nll_scored_robot_turns", "hrc_robot_turn_count", "n_steps"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
    return 0.0


def _episode_nll_recipe_steps(row: Mapping[str, Any]) -> float:
    value = row.get("nll_normalization_recipe_steps")
    if isinstance(value, (int, float)) and float(value) > 0.0:
        return float(value)
    if _episode_robot_turn_count(row) <= 0.0:
        return 0.0
    value = row.get("n_recipe_steps")
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return 0.0


def _episode_total_robot_turn_nll(row: Mapping[str, Any]) -> float:
    value = row.get("total_robot_turn_nll")
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    robot_turns = _episode_robot_turn_count(row)
    recipe_steps = _episode_nll_recipe_steps(row)
    for key, denom in (
        ("mean_nll_per_robot_turn", robot_turns),
        ("cross_entropy", robot_turns),
        ("mean_nll_per_recipe_step", recipe_steps),
    ):
        value = row.get(key)
        if isinstance(value, (int, float)) and denom > 0.0:
            return max(0.0, float(value) * float(denom))
    return 0.0


def _aggregate_episode_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "live_top1": 0.0,
            "live_topk": 0.0,
            "total_robot_turn_nll": 0.0,
            "nll_scored_robot_turns": 0.0,
            "nll_normalization_recipe_steps": 0.0,
            "mean_nll_per_robot_turn": 0.0,
            "mean_nll_per_recipe_step": 0.0,
            **_adaptation_recovery_metrics([], -1),
            "mean_first_mismatch_robot_turn": -1.0,
            "mean_first_mismatch_robot_turn_normalized_position": -1.0,
            "adaptation_recovery_window_curve": _adaptation_recovery_window_curve([]),
            "first_mismatch_position_curve": _first_mismatch_position_curve([]),
            **{f"adaptation_recovery_eligible_rate_w{w}": 0.0 for w in ADAPTATION_RECOVERY_WINDOWS},
            **{f"adaptation_recovery_censored_rate_w{w}": 0.0 for w in ADAPTATION_RECOVERY_WINDOWS},
            **{f"n_adaptation_recovery_eligible_w{w}": 0.0 for w in ADAPTATION_RECOVERY_WINDOWS},
            "robot_wrong_rate": 0.0,
            "net_robot_action_value": 0.0,
            "n_episodes": 0.0,
            "n_steps": 0.0,
            "n_recipe_steps": 0.0,
            "hrc_robot_turn_count": 0.0,
            "hrc_human_turn_count": 0.0,
            "hrc_human_correction_count": 0.0,
            "hrc_human_action_count": 0.0,
            "hrc_robot_correct_count": 0.0,
            "hrc_robot_wrong_count": 0.0,
            "human_correction_rate": 0.0,
            "human_action_fraction": 0.0,
            "human_effort_time": 0.0,
            "human_effort_fraction": 0.0,
            "primary_hrc_metric": "human_action_fraction",
            "primary_hrc_metric_value": 0.0,
            "human_workload_reduction_vs_human_only": 0.0,
            "hrc_future_valid_wrong_count": 0.0,
            "future_valid_wrong_rate": 0.0,
            "wrong_prediction_future_valid_rate": 0.0,
            "behavioral_steps_to_lock": -1.0,
            "posterior_steps_to_lock": -1.0,
            "n_episodes_behavioral_lock_eligible": 0.0,
            "lock_success_rate": 0.0,
            "mean_behavioral_steps_to_lock_conditional": -1.0,
            "timing_details": _hrc_timing_details(
                hrc_total_time=0.0,
                human_only_time=0.0,
                hrc_time_penalty_vs_human_only=0.0,
                observation_mode_extra_time=0.0,
                hrc_mean_time_per_recipe_step=0.0,
                hrc_mean_time_per_robot_turn=0.0,
                hrc_robot_correct_time=0.0,
                hrc_robot_wrong_time=0.0,
                hrc_human_action_time=0.0,
                hrc_human_correction_time=0.0,
                hrc_wrong_action_extra_penalty_time=0.0,
            ),
            "observation_mode_episode": 0.0,
            "observation_mode_rate": 0.0,
            "user_observation_required_rate": 0.0,
        }
    step_support = [max(0, int(r.get("n_steps", 0))) for r in rows]
    weights = [max(1, n) for n in step_support]
    total = max(1, sum(weights))
    out: Dict[str, Any] = {"n_episodes": float(len(rows)), "n_steps": float(sum(step_support))}
    for key in (
        "live_top1", "live_topk",
        "robot_wrong_rate", "net_robot_action_value",
        "posterior_correct_recipe", "recipe_vocab_top1",
        "mean_policy_confidence", "mean_policy_raw_confidence",
        "mean_posterior_confidence",
        "mean_policy_gate_score", "mean_policy_margin",
        "mean_posterior_entropy", "mean_policy_entropy",
        "mean_policy_final_confidence", "mean_conditioned_policy_confidence",
        "mean_ensemble_policy_confidence",
        "posterior_degenerate_rate",
        "posterior_steps_to_lock",
        "adaptation_latency_steps", "confidence_ece", "confidence_brier",
        "pooled_ece", "pooled_brier", "p95_prediction_wall_s",
        "mean_prediction_wall_s",
        "observation_mode_episode",
        "user_observation_required",
    ):
        out[key] = sum(_episode_metric_value(r, key, 0.0) * w for r, w in zip(rows, weights)) / total
    for key in (
        "n_recipe_steps",
        "hrc_robot_turn_count",
        "hrc_human_turn_count",
        "hrc_human_correction_count",
        "hrc_human_action_count",
        "hrc_robot_correct_count",
        "hrc_robot_wrong_count",
        "hrc_future_valid_wrong_count",
        "human_effort_time",
    ):
        out[key] = sum(float(r.get(key, 0.0)) for r in rows)
    total_robot_turn_nll = sum(_episode_total_robot_turn_nll(r) for r in rows)
    nll_scored_robot_turns = sum(_episode_robot_turn_count(r) for r in rows)
    nll_normalization_recipe_steps = sum(_episode_nll_recipe_steps(r) for r in rows)
    out["total_robot_turn_nll"] = float(total_robot_turn_nll)
    out["nll_scored_robot_turns"] = float(nll_scored_robot_turns)
    out["nll_normalization_recipe_steps"] = float(nll_normalization_recipe_steps)
    out["mean_nll_per_robot_turn"] = float(total_robot_turn_nll / max(1.0, nll_scored_robot_turns))
    out["mean_nll_per_recipe_step"] = float(total_robot_turn_nll / max(1.0, nll_normalization_recipe_steps))
    timing_totals = {
        key: sum(_episode_timing_value(r, key) for r in rows)
        for key in (
            "hrc_total_time",
            "human_only_time",
            "observation_mode_extra_time",
            "hrc_robot_correct_time",
            "hrc_robot_wrong_time",
            "hrc_human_action_time",
            "hrc_human_correction_time",
            "hrc_wrong_action_extra_penalty_time",
        )
    }
    out["observation_mode_rate"] = sum(1.0 for r in rows if r.get("mode") == "observe") / max(1.0, len(rows))
    out["user_observation_required_rate"] = sum(float(r.get("user_observation_required", 0.0)) for r in rows) / max(1.0, len(rows))
    out["human_correction_rate"] = out["hrc_human_correction_count"] / max(1.0, out["hrc_robot_turn_count"])
    out["human_action_fraction"] = out["hrc_human_action_count"] / max(1.0, out["n_recipe_steps"])
    if out["human_effort_time"] <= 0.0 and out["n_recipe_steps"] > 0.0:
        out["human_effort_time"] = (
            out["hrc_human_turn_count"] * float(DEFAULT_HRC_TIMING.human_action_time)
            + out["hrc_human_correction_count"] * float(DEFAULT_HRC_TIMING.human_correction_time)
        )
    out["human_effort_fraction"] = out["human_effort_time"] / max(1.0e-12, timing_totals["human_only_time"]) if timing_totals["human_only_time"] > 0.0 else 0.0
    out["primary_hrc_metric"] = "human_action_fraction"
    out["primary_hrc_metric_value"] = out["human_action_fraction"]
    out["human_workload_reduction_vs_human_only"] = 1.0 - out["human_action_fraction"]
    out["future_valid_wrong_rate"] = out["hrc_future_valid_wrong_count"] / max(1.0, out["hrc_robot_turn_count"])
    out["wrong_prediction_future_valid_rate"] = out["hrc_future_valid_wrong_count"] / max(1.0, out["hrc_robot_wrong_count"])
    out["timing_details"] = _hrc_timing_details(
        hrc_total_time=timing_totals["hrc_total_time"],
        human_only_time=timing_totals["human_only_time"],
        hrc_time_penalty_vs_human_only=timing_totals["hrc_total_time"] - timing_totals["human_only_time"],
        observation_mode_extra_time=timing_totals["observation_mode_extra_time"],
        hrc_mean_time_per_recipe_step=timing_totals["hrc_total_time"] / max(1.0, out["n_recipe_steps"]),
        hrc_mean_time_per_robot_turn=timing_totals["hrc_total_time"] / max(1.0, out["hrc_robot_turn_count"]),
        hrc_robot_correct_time=timing_totals["hrc_robot_correct_time"],
        hrc_robot_wrong_time=timing_totals["hrc_robot_wrong_time"],
        hrc_human_action_time=timing_totals["hrc_human_action_time"],
        hrc_human_correction_time=timing_totals["hrc_human_correction_time"],
        hrc_wrong_action_extra_penalty_time=timing_totals["hrc_wrong_action_extra_penalty_time"],
    )
    lock_rows = [r for r in rows if int(r.get("n_steps", 0)) >= 5]
    lock_weights = [max(1, int(r.get("n_steps", 1))) for r in lock_rows]
    lock_total = max(1, sum(lock_weights))
    out["n_episodes_behavioral_lock_eligible"] = float(len(lock_rows))
    out["lock_success_rate"] = _mean([1.0 if float(r.get("behavioral_steps_to_lock", -1.0)) >= 0.0 else 0.0 for r in lock_rows]) if lock_rows else 0.0
    successful_lock_rows = [r for r in lock_rows if float(r.get("behavioral_steps_to_lock", -1.0)) >= 0.0]
    out["mean_behavioral_steps_to_lock_conditional"] = _mean([float(r.get("behavioral_steps_to_lock", 0.0)) for r in successful_lock_rows]) if successful_lock_rows else -1.0
    out["behavioral_steps_to_lock"] = (
        sum(float(r.get("behavioral_steps_to_lock", -1.0)) * w for r, w in zip(lock_rows, lock_weights)) / lock_total
        if lock_rows else -1.0
    )
    robot_turn_rows = [r for r in rows if int(r.get("n_steps", 0)) > 0]
    mismatch_rows = [r for r in robot_turn_rows if float(r.get("first_mismatch_rate", 0.0)) > 0.0]
    out["first_mismatch_rate"] = _mean([float(r.get("first_mismatch_rate", 0.0)) for r in robot_turn_rows])
    out["no_mismatch_rate"] = _mean([float(r.get("no_mismatch_rate", 0.0)) for r in robot_turn_rows])
    out["n_first_mismatch_episodes"] = float(len(mismatch_rows))
    out["mean_first_mismatch_robot_turn"] = (
        _mean([_row_first_mismatch_robot_turn(r) for r in mismatch_rows])
        if mismatch_rows else -1.0
    )
    out["mean_first_mismatch_robot_turn_normalized_position"] = (
        _mean([_row_first_mismatch_robot_turn_position(r) for r in mismatch_rows])
        if mismatch_rows else -1.0
    )
    out["mean_robot_turns_after_first_mismatch"] = _mean([
        float(r.get("robot_turns_after_first_mismatch", 0.0)) for r in mismatch_rows
    ])
    for window in ADAPTATION_RECOVERY_WINDOWS:
        eligible_key = f"adaptation_recovery_eligible_w{window}"
        censored_key = f"adaptation_recovery_censored_w{window}"
        top1_key = f"adaptation_recovery_top1_w{window}"
        support_key = f"adaptation_recovery_support_w{window}"
        eligible_rows = [r for r in mismatch_rows if float(r.get(eligible_key, 0.0)) > 0.0]
        support = [max(1.0, float(r.get(support_key, window))) for r in eligible_rows]
        out[f"adaptation_recovery_top1_w{window}"] = (
            sum(float(r.get(top1_key, 0.0)) * w for r, w in zip(eligible_rows, support)) / max(1.0, sum(support))
            if eligible_rows else 0.0
        )
        out[f"adaptation_recovery_eligible_rate_w{window}"] = float(len(eligible_rows) / max(1, len(mismatch_rows)))
        out[f"adaptation_recovery_censored_rate_w{window}"] = float(
            sum(float(r.get(censored_key, 0.0)) for r in mismatch_rows) / max(1, len(mismatch_rows))
        )
        out[f"n_adaptation_recovery_eligible_w{window}"] = float(len(eligible_rows))
    out["adaptation_recovery_window_curve"] = _adaptation_recovery_window_curve(robot_turn_rows)
    out["first_mismatch_position_curve"] = _first_mismatch_position_curve(robot_turn_rows)
    pooled_scores: List[float] = []
    pooled_labels: List[int] = []
    for row in rows:
        record = row.get("live_record")
        if not isinstance(record, LiveEpisodeRecord):
            continue
        for step in record.steps:
            score = _step_action_calibration_score(step)
            if score is not None:
                pooled_scores.append(float(score))
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
    transfer_cells = Counter()
    modes = Counter()
    conditions = Counter()
    for event in events:
        recipe, preference = _pair_parts(event)
        seen_recipe = bool(recipe and recipe in seen_recipes)
        seen_preference = bool(preference and preference in seen_preferences)
        cells[_four_cell(seen_recipe, seen_preference)] += 1
        transfer_cell = str(event.get("transfer_cell_before", ""))
        if transfer_cell:
            transfer_cells[transfer_cell] += 1
        mode = str(event.get("mode", "unknown"))
        modes[mode] += 1
        conditions[str(event.get("condition", event.get("event_type", "unknown")))] += 1
        if recipe:
            seen_recipes.add(recipe)
        if preference:
            seen_preferences.add(preference)
    return {
        "exposure_cell": {k: int(cells.get(k, 0)) for k in FOUR_CELL_KEYS},
        "four_cell": {k: int(cells.get(k, 0)) for k in FOUR_CELL_KEYS},
        "transfer_cell": {k: int(transfer_cells.get(k, 0)) for k in TRANSFER_CELL_KEYS},
        "modes": dict(modes),
        "conditions": dict(conditions),
    }


def _scenario_warnings(scenario: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    support = scenario.get("support_counts", {})
    cells = support.get("exposure_cell", support.get("four_cell", {})) if isinstance(support, dict) else {}
    event_type_counts = scenario.get("event_type_counts", {}) or {}
    is_ladder_stream = any(str(k).startswith("ladder_") for k in event_type_counts)
    if scenario.get("n_events", 0) == 0:
        warnings.append("scenario_has_no_recorded_events")
    if cells and sum(int(v) for v in cells.values()) > 0:
        if not is_ladder_stream and int(cells.get("seen_recipe_new_preference", 0)) == 0:
            warnings.append("scenario_has_no_seen_recipe_new_preference_events")
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
            prediction_nll = float(flat.get("mean_nll_per_robot_turn", flat.get("cross_entropy", 0.0)))
            empty_rate = 1.0 if n_steps > 0 and live_top1 == 0.0 and prediction_nll > 10.0 else 0.0
            out[baseline] = {
                "n_train_demos": 0,
                "n_assist_episodes": int(flat.get("n_episodes", 0.0)),
                "n_steps": n_steps,
                "n_policy_actions": 0,
                "empty_prediction_rate": empty_rate,
                "policy_gate_reason_counts": {},
                "model_fitted": empty_rate < 1.0,
            }
            continue
        empty_steps = [s for s in b_steps if not s.get("topk") and s.get("predicted") is None]
        reason_counts = Counter(_row_policy_text(s, "gate_reason") for s in b_steps)
        actions = {str(s.get("actual")) for s in b_steps if s.get("actual") is not None}
        out[baseline] = {
            "n_train_demos": sum(1 for r in b_eps if r.get("mode") == "observe"),
            "n_assist_episodes": sum(1 for r in b_eps if r.get("mode") == "assist"),
            "n_steps": len(b_steps),
            "n_policy_actions": len(actions),
            "empty_prediction_rate": len(empty_steps) / max(1, len(b_steps)),
            "policy_gate_reason_counts": dict(reason_counts),
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
        "live_top1", "live_topk", "adaptation_recovery_top1_w3",
        "adaptation_recovery_eligible_rate_w3", "adaptation_recovery_censored_rate_w3",
        "first_mismatch_rate", "no_mismatch_rate", "mean_first_mismatch_robot_turn_normalized_position",
        "robot_wrong_rate", "net_robot_action_value", "frozen_top1",
        "frozen_transfer_top1", "live_online_transfer_top1",
        "frozen_to_live_transfer_gain", "inference_time_adaptation_gain",
        "prototype_level_transfer_top1",
        "post_adapt_top1", "known_recipe_assist_continuation_rate",
        "known_recipe_assistance_suppression_rate", "preference_transfer_rate",
        "preference_gate_accuracy",
        "known_shift_live_top1",
        "primary_transfer_live_top1", "adaptation_latency_steps",
        "human_correction_rate", "human_action_fraction", "human_effort_fraction", "primary_hrc_metric_value",
        "human_workload_reduction_vs_human_only",
        "observation_mode_rate", "user_observation_required_rate",
        "p95_prediction_wall_s", "behavioral_steps_to_lock",
        "lock_success_rate", "mean_behavioral_steps_to_lock_conditional",
        "pooled_ece", "preference_cluster_purity",
        "preference_nmi", "preference_ari", "preference_collapse_rate",
        "latent_preference_entropy",
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
        requested_mode = str(mode)
        mode, route_tags = _user_selected_mode_for_active_memory(requested_mode, agent, pair, name_to_rid)
        tags = {**tags, **route_tags}
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
        **_aggregate_episode_metrics(episode_rows),
        "assist_only": _aggregate_episode_metrics([r for r in episode_rows if r.get("mode") == "assist"]),
        "compute": compute_snapshot(agent, wall_s),
        "memory": memory_snapshot(agent),
        "policy_gate": policy_gate_reason_summary(step_rows),
    }
    return metrics, episode_rows, step_rows, memory_rows


def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    if n <= 0:
        return np.ones(0)
    if alpha <= 0:
        return np.full(n, 1.0 / n)
    w = 1.0 / (np.arange(1, n + 1, dtype=float) ** float(alpha))
    return w / w.sum()


def run_single_shot_reuse(seed: int, out: Path, run_config: RunConfig, cfg_obj: SingleShotReuseConfig) -> Dict[str, Any]:
    observe_pref = cfg_obj.observe_preference if cfg_obj.observe_preference in PRESET_PREFERENCES else "identity"
    test_prefs = tuple(p for p in cfg_obj.test_preferences if p in PRESET_PREFERENCES) or (observe_pref,)
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    recipes = [r for r, _ in builders]
    pairs = [materialize_pair(recipe, observe_pref, fn) for recipe, fn in builders]
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
        frozen_topk = _mean([v.get("topk", 0.0) for v in frozen.values()])
        per_baseline[baseline] = {
            **_aggregate_episode_metrics(episodes),
            "frozen_top1": frozen_top1,
            "frozen_topk": frozen_topk,
            **heldout_by_pref,
            "compute": compute_snapshot(agent, wall_s),
            "memory": memory_snapshot(agent),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    metrics = {"seed": seed, "n_pairs": len(pairs), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_deployment_gate_preference_shift(seed: int, out: Path, run_config: RunConfig, cfg_obj: DeploymentGateConfig) -> Dict[str, Any]:
    n_train = max(1, int(cfg_obj.n_train_recipes))
    train_builders = select_recipe_builders(seed, n_train)
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
    _record_common_artifacts(out, seed, cfg_obj, [p.recipe_name for p in train_pairs], (cfg_obj.train_preference,) + shift_prefs)
    events_log = [
        {"mode": "assist", "phase": "probe", "event_type": "known_recipe_new_preference", "condition": "known_shift", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for p in shift_pairs
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
        known_suppression = _mean([1.0 if r.get("needs_observation") else 0.0 for r in known_rows])
        per_baseline[baseline] = {
            "known_shift": _aggregate_episode_metrics(known_rows),
            "post_commit_top1": _mean([r.get("post_commit_top1", 0.0) for r in known_rows]),
            "known_recipe_assistance_suppression_rate": known_suppression,
            "known_recipe_assist_continuation_rate": 1.0 - known_suppression,
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    metrics = {"seed": seed, "n_known_shift": len(shift_pairs), "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
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
        if int(gap) > 0:
            if hasattr(agent, "refresh_model_from_memory"):
                agent.refresh_model_from_memory()
            elif hasattr(agent, "_retrain"):
                agent._retrain()
    else:
        for i in range(max(0, int(gap))):
            observe_episode(agent, distractors[i % len(distractors)], name_to_rid)
            agent.decay._recompute_effective(agent.session_counter)


def _advance_mixed_gap(
    agent: AdaptiveHRCAgent,
    gap_sessions: int,
    n_distractors: int,
    distractors: Sequence[RecipePrefPair],
    name_to_rid: Dict[str, str],
) -> None:
    gap = max(0, int(gap_sessions))
    n_dist = max(0, min(int(n_distractors), gap))
    for i in range(n_dist):
        if distractors:
            observe_episode(agent, distractors[i % len(distractors)], name_to_rid)
            agent.decay._recompute_effective(agent.session_counter)
    neutral_ticks = gap - n_dist
    if neutral_ticks > 0:
        _advance_gap(agent, neutral_ticks, (), name_to_rid, neutral=True)


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


def _run_compute_tradeoff(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
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
    return metrics


def run_short_term_capacity_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: CapacitySweepConfig) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for n in cfg_obj.demo_counts:
        sub = SingleShotReuseConfig(n_recipes=int(n), n_preferences=cfg_obj.n_preferences)
        sub_out = _ensure(out / f"n_{n}")
        rows[str(n)] = run_single_shot_reuse(seed, sub_out, run_config, sub)
    metrics = {"seed": seed, "per_demo_count": rows}
    _write_json(out / "metrics.json", metrics)
    return metrics


def _unique_material_preference_pairs(
    recipe_name: str,
    generator_fn: Callable[[], List[str]],
    *,
    include_identity: bool = True,
) -> List[RecipePrefPair]:
    pairs: List[RecipePrefPair] = []
    seen_orderings: set = set()
    for pref_name in PREFERENCE_NAMES:
        if pref_name == "identity" and not include_identity:
            continue
        try:
            pair = materialize_pair(recipe_name, pref_name, generator_fn)
        except Exception:
            continue
        order = _order_hash(pair)
        if order in seen_orderings:
            continue
        seen_orderings.add(order)
        pairs.append(pair)
    return pairs


def _recipe_onboarding_sample_efficiency(
    seed: int,
    run_config: RunConfig,
    cfg_obj: CapacitySweepConfig,
) -> Dict[str, Any]:
    builders = select_recipe_builders(seed + 43, 1)
    if not builders:
        return {}
    recipe_name, fn = builders[0]
    pair = materialize_pair(recipe_name, "identity", fn)
    rows: Dict[str, Any] = {}
    for k_raw in cfg_obj.demo_counts:
        k = max(1, int(k_raw))
        train_pairs = [pair for _ in range(k)]
        per_baseline: Dict[str, Any] = {}
        for baseline in run_config.baselines:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            for train_pair in train_pairs:
                observe_episode(agent, train_pair, name_to_rid)
            frozen = frozen_eval(agent, [pair], name_to_rid)
            row, _steps = assist_episode(
                agent,
                pair,
                dict(name_to_rid),
                run_config=run_config,
                commit=False,
                event_tag={"condition": "recipe_onboarding_sample_efficiency", "demo_count": k},
            )
            row.update({
                "frozen_top1": float((frozen.get(pair.label, {}) or {}).get("top1", 0.0)),
                "frozen_topk": float((frozen.get(pair.label, {}) or {}).get("topk", 0.0)),
                "n_train_demos": k,
            })
            per_baseline[baseline] = row
        rows[str(k)] = {
            "curve_type": "recipe_onboarding",
            "train_pair": pair.label,
            "test_pair": pair.label,
            "recipe": pair.recipe_name,
            "preference": pair.preference_name,
            "n_train_demos": k,
            "per_baseline": per_baseline,
            "oracle_references": _oracle_reference_metrics(
                seed,
                run_config,
                train_pairs,
                [pair],
                max_eval_pairs=1,
            ),
        }
    return rows


def _preference_transfer_sample_efficiency(
    seed: int,
    run_config: RunConfig,
    cfg_obj: CapacitySweepConfig,
) -> Dict[str, Any]:
    max_k = max(int(k) for k in cfg_obj.demo_counts) if cfg_obj.demo_counts else 1
    builders = select_recipe_builders(seed + 71, max_k + 1)
    rows: Dict[str, Any] = {}
    for k_raw in cfg_obj.demo_counts:
        k = max(1, int(k_raw))
        if len(builders) <= k:
            continue
        candidate_prefs = [p for p in PREFERENCE_NAMES if p != "identity"]
        train_pairs: List[RecipePrefPair] = []
        test_pair: Optional[RecipePrefPair] = None
        selected_pref: Optional[str] = None
        for pref in candidate_prefs:
            try:
                trial_train = [materialize_pair(recipe, pref, fn) for recipe, fn in builders[:k]]
                trial_test = materialize_pair(builders[k][0], pref, builders[k][1])
            except Exception:
                continue
            if len({p.actions for p in trial_train + [trial_test]}) >= 2:
                train_pairs = trial_train
                test_pair = trial_test
                selected_pref = pref
                break
        if test_pair is None:
            continue
        per_baseline: Dict[str, Any] = {}
        for baseline in run_config.baselines:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            for pair in train_pairs:
                observe_episode(agent, pair, name_to_rid)
            row, _steps = assist_episode(
                agent,
                test_pair,
                name_to_rid,
                run_config=run_config,
                commit=False,
                event_tag={"condition": "preference_transfer_sample_efficiency", "preference_demo_count": k},
            )
            per_baseline[baseline] = row
        rows[str(k)] = {
            "preference": selected_pref,
            "train_pairs": [p.label for p in train_pairs],
            "test_pair": test_pair.label,
            "per_baseline": per_baseline,
            "oracle_references": _oracle_reference_metrics(
                seed,
                run_config,
                train_pairs,
                [test_pair],
                max_eval_pairs=1,
            ),
        }
    return rows


def _preference_variant_sample_efficiency(
    seed: int,
    run_config: RunConfig,
    cfg_obj: CapacitySweepConfig,
) -> Dict[str, Any]:
    max_k = max(int(k) for k in cfg_obj.demo_counts) if cfg_obj.demo_counts else 1
    builders = select_recipe_builders(seed + 109, max(4, min(len(gen.recipe_library()), max_k + 2)))
    selected_pairs: List[RecipePrefPair] = []
    for recipe_name, fn in builders:
        variants = _unique_material_preference_pairs(recipe_name, fn)
        if len(variants) > len(selected_pairs):
            selected_pairs = variants
    if len(selected_pairs) < 2:
        return {}
    rows: Dict[str, Any] = {}
    for k_raw in cfg_obj.demo_counts:
        k = max(1, int(k_raw))
        if k >= len(selected_pairs):
            continue
        train_pairs = selected_pairs[:k]
        test_pair = selected_pairs[k]
        per_baseline: Dict[str, Any] = {}
        for baseline in run_config.baselines:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            for pair in train_pairs:
                observe_episode(agent, pair, name_to_rid)
            row, _steps = assist_episode(
                agent,
                test_pair,
                dict(name_to_rid),
                run_config=run_config,
                commit=False,
                event_tag={"condition": "preference_variant_sample_efficiency", "preference_variant_count": k},
            )
            per_baseline[baseline] = row
        rows[str(k)] = {
            "curve_type": "preference_variant",
            "recipe": test_pair.recipe_name,
            "train_pairs": [p.label for p in train_pairs],
            "test_pair": test_pair.label,
            "n_train_variants": k,
            "per_baseline": per_baseline,
            "oracle_references": _oracle_reference_metrics(
                seed,
                run_config,
                train_pairs,
                [test_pair],
                max_eval_pairs=1,
            ),
        }
    return rows


def run_demo_count_sample_efficiency(seed: int, out: Path, run_config: RunConfig, cfg_obj: CapacitySweepConfig) -> Dict[str, Any]:
    onboarding_rows = _recipe_onboarding_sample_efficiency(seed, run_config, cfg_obj)
    metrics: Dict[str, Any] = {
        "seed": seed,
        "per_demo_count": onboarding_rows,
    }
    efficiency: Dict[str, Dict[str, float]] = {}
    for n, row in metrics["per_demo_count"].items():
        efficiency[n] = {b: row["per_baseline"].get(b, {}).get("live_top1", 0.0) / max(1, int(n)) for b in run_config.baselines}
    metrics["sample_efficiency"] = efficiency
    transfer_rows = _preference_transfer_sample_efficiency(seed, run_config, cfg_obj)
    variant_rows = _preference_variant_sample_efficiency(seed, run_config, cfg_obj)
    metrics["recipe_onboarding_sample_efficiency"] = metrics.get("per_demo_count", {})
    metrics["preference_transfer_sample_efficiency"] = transfer_rows
    metrics["preference_variant_sample_efficiency"] = variant_rows
    metrics["sample_efficiency_curves"] = {
        "recipe_onboarding": {
            baseline: {
                str(n): (metrics.get("per_demo_count", {}).get(str(n), {}).get("per_baseline", {}).get(baseline, {}) or {}).get("live_top1", 0.0)
                for n in cfg_obj.demo_counts
            }
            for baseline in run_config.baselines
        },
        "preference_transfer": {
            baseline: {
                str(n): (transfer_rows.get(str(n), {}).get("per_baseline", {}).get(baseline, {}) or {}).get("live_top1", 0.0)
                for n in cfg_obj.demo_counts
                if str(n) in transfer_rows
            }
            for baseline in run_config.baselines
        },
        "preference_variant": {
            baseline: {
                str(n): (variant_rows.get(str(n), {}).get("per_baseline", {}).get(baseline, {}) or {}).get("live_top1", 0.0)
                for n in cfg_obj.demo_counts
                if str(n) in variant_rows
            }
            for baseline in run_config.baselines
        },
    }
    metrics["sample_efficiency_metric_curves"] = {}
    for curve_name, source_rows in (
        ("recipe_onboarding", metrics.get("per_demo_count", {}) or {}),
        ("preference_transfer", transfer_rows),
        ("preference_variant", variant_rows),
    ):
        metrics["sample_efficiency_metric_curves"][curve_name] = {}
        for metric_name in ("live_top1", "robot_wrong_rate", "net_robot_action_value"):
            metric_curves = {
                baseline: {
                    str(n): (source_rows.get(str(n), {}).get("per_baseline", {}).get(baseline, {}) or {}).get(metric_name, 0.0)
                    for n in cfg_obj.demo_counts
                    if str(n) in source_rows
                }
                for baseline in run_config.baselines
            }
            if curve_name == "recipe_onboarding":
                metric_curves["oracle_ceiling"] = {
                    str(n): (0.0 if metric_name == "robot_wrong_rate" else 1.0)
                    for n in cfg_obj.demo_counts
                    if str(n) in source_rows
                }
            else:
                for oracle_name in ORACLE_REFERENCE_BASELINES:
                    vals = {
                        str(n): (source_rows.get(str(n), {}).get("oracle_references", {}).get(oracle_name, {}) or {}).get(metric_name, 0.0)
                        for n in cfg_obj.demo_counts
                        if str(n) in source_rows
                    }
                    if vals:
                        metric_curves[oracle_name] = vals
            metrics["sample_efficiency_metric_curves"][curve_name][metric_name] = metric_curves
    _write_json(out / "metrics.json", metrics)
    return metrics


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


def _route_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = ("assist", "observe")
    matrix = confusion_labels([str(r.get("pred")) for r in rows], [str(r.get("gt")) for r in rows], labels=labels)
    report = classifier_report(matrix, labels)
    return {**report, "confusion_matrix": matrix.tolist(), "rows": list(rows)}


def _drop_interior(actions: Sequence[str], frac: float, seed: int) -> List[str]:
    if len(actions) <= 2 or frac <= 0:
        return list(actions)
    rng = _rng(seed, int(frac * 10000))
    idxs = list(range(1, len(actions) - 1))
    rng.shuffle(idxs)
    drop = set(idxs[: max(1, int(round(frac * len(actions))))])
    return [a for i, a in enumerate(actions) if i not in drop]


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
                "pool_seen_fraction": len(seen_labels) / max(1, len(pool)),
                **{f"{metric}_{cell}": vals.get(metric, 0.0) for cell, rows in cell_rows.items() for vals in [_aggregate_episode_metrics(rows)] for metric in ("live_top1", "robot_wrong_rate", "n_episodes")},
            }
        per_pool[str(pool_size)] = row
    metrics = {"seed": seed, "per_pool": per_pool}
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_zipf_usage_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: ZipfUsageConfig) -> Dict[str, Any]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    prefs = select_preference_names(max(1, int(cfg_obj.n_preferences)))
    by_recipe: Dict[str, List[RecipePrefPair]] = {}
    for recipe_name, fn in builders:
        seen_orders: set = set()
        rows: List[RecipePrefPair] = []
        for pref in prefs:
            try:
                pair = materialize_pair(recipe_name, pref, fn)
            except Exception:
                continue
            if pair.actions in seen_orders:
                continue
            seen_orders.add(pair.actions)
            rows.append(pair)
        if rows:
            by_recipe[recipe_name] = rows
    ranked_pairs: List[RecipePrefPair] = []
    recipe_rank_by_name = {recipe_name: idx for idx, (recipe_name, _fn) in enumerate(builders)}
    preference_rank_by_label: Dict[str, int] = {}
    for recipe_name, _fn in builders:
        rows = _sorted_preference_choices(by_recipe.get(recipe_name, []))
        by_recipe[recipe_name] = rows
        for pref_idx, pair in enumerate(rows):
            preference_rank_by_label[pair.label] = pref_idx
            ranked_pairs.append(pair)
    if not ranked_pairs:
        raise RuntimeError("zipf_usage_sweep requires at least one materialized recipe/preference workflow")
    n_events = int(cfg_obj.n_events)
    phase_a_pairs: List[RecipePrefPair] = []
    for _cycle in range(max(1, int(cfg_obj.phase_a_cycles))):
        phase_a_pairs.extend(ranked_pairs)
    recipes = [r for r, _ in builders]
    _record_common_artifacts(out, seed, cfg_obj, recipes, prefs)
    phase_a_events = [
        ScenarioEvent(
            pair=pair,
            mode="observe",
            phase="phase_a_onboarding",
            user_id="kitchen_staff",
            tags={
                "event_idx": idx,
                "event_type": "prep_book_onboarding",
                "workflow_rank": ranked_pairs.index(pair),
                "scenario_note": "balanced onboarding demonstration before rush-hour skew",
            },
        )
        for idx, pair in enumerate(phase_a_pairs)
    ]
    per_alpha: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    all_memory_rows: List[Dict[str, Any]] = []
    scenario_rows: List[Dict[str, Any]] = [_scenario_event_row(e) for e in phase_a_events]
    phase_a_snapshots: Dict[str, Any] = {}
    phase_a_name_to_rid: Dict[str, Dict[str, str]] = {}
    phase_a_wall_s: Dict[str, float] = {}
    for baseline in run_config.baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        t_phase = time.perf_counter()
        for pair in phase_a_pairs:
            observe_episode(agent, pair, name_to_rid)
        phase_a_wall_s[baseline] = float(time.perf_counter() - t_phase)
        phase_a_snapshots[baseline] = agent.snapshot()
        phase_a_name_to_rid[baseline] = dict(name_to_rid)
    for alpha in cfg_obj.alphas:
        recipe_probs = _zipf_probs(len(builders), float(alpha))
        raw_probs: List[float] = []
        for pair in ranked_pairs:
            r_idx = recipe_rank_by_name.get(pair.recipe_name, 0)
            pref_probs = _zipf_probs(len(by_recipe.get(pair.recipe_name, [])), float(alpha))
            p_idx = preference_rank_by_label.get(pair.label, 0)
            raw_probs.append(float(recipe_probs[r_idx]) * float(pref_probs[min(p_idx, len(pref_probs) - 1)]))
        probs = np.asarray(raw_probs, dtype=float)
        probs = probs / max(float(probs.sum()), 1e-9)
        rng = _rng(seed, int(alpha * 1000) + 707)
        counts = Counter()
        alpha_events: List[ScenarioEvent] = []
        probability_order = list(np.argsort(-probs))
        rank_by_idx = {int(idx): rank for rank, idx in enumerate(probability_order)}
        head_cut = max(1, len(ranked_pairs) // 4)
        tail_rank_start = max(0, len(ranked_pairs) - head_cut)
        for i in range(max(1, n_events)):
            idx = int(rng.choice(len(ranked_pairs), p=probs))
            pair = ranked_pairs[idx]
            counts[pair.label] += 1
            prob_rank = int(rank_by_idx.get(idx, idx))
            if prob_rank < head_cut:
                event_type = "rush_hour_repeat_order"
            elif prob_rank >= tail_rank_start:
                event_type = "rare_custom_order_reentry"
            else:
                event_type = "regular_menu_rotation"
            alpha_events.append(ScenarioEvent(
                pair=pair,
                mode="assist",
                phase="zipf_phase_b",
                user_id="service_window",
                tags={
                    "event_idx": i,
                    "event_type": event_type,
                    "zipf_alpha": float(alpha),
                    "workflow_rank": prob_rank,
                    "recipe_rank": int(recipe_rank_by_name.get(pair.recipe_name, 0)),
                    "preference_rank": int(preference_rank_by_label.get(pair.label, 0)),
                    "intended_probability": float(probs[idx]),
                    "head_tail_bin": "head" if prob_rank < head_cut else ("tail" if prob_rank >= tail_rank_start else "middle"),
                },
            ))
        scenario_rows.extend(_scenario_event_row(e) for e in alpha_events)
        alpha_row: Dict[str, Any] = {}
        total = max(1, sum(counts.values()))
        head_labels = {ranked_pairs[idx].label for idx in probability_order[:head_cut]}
        tail_labels = {ranked_pairs[idx].label for idx in probability_order[tail_rank_start:]}
        prob_by_label = {p.label: float(probs[idx]) for idx, p in enumerate(ranked_pairs)}
        for baseline in run_config.baselines:
            agent = make_agent(baseline, base_config(seed, run_config))
            agent.restore_from(phase_a_snapshots[baseline])
            name_to_rid = dict(phase_a_name_to_rid[baseline])
            episode_rows: List[Dict[str, Any]] = []
            step_rows: List[Dict[str, Any]] = []
            memory_rows: List[Dict[str, Any]] = []
            t0 = time.perf_counter()
            for event_idx, event in enumerate(alpha_events):
                tags = {"phase": event.phase, "user_id": event.user_id, **dict(event.tags)}
                row, steps = assist_episode(agent, event.pair, name_to_rid, run_config=run_config, event_tag=tags)
                row.update(tags)
                episode_rows.append(row)
                step_rows.extend({"event_idx": event_idx, "baseline": baseline, **s} for s in steps)
                memory_rows.append({"event_idx": event_idx, "baseline": baseline, "zipf_alpha": float(alpha), **memory_snapshot(agent)})
            phase_b_wall = float(time.perf_counter() - t0)
            assist_rows = [r for r in episode_rows if r.get("mode") == "assist"]
            by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in assist_rows:
                by_label[r["pair"]].append(r)
            label_live_acc = {lbl: _aggregate_episode_metrics(rows)["live_top1"] for lbl, rows in by_label.items()}
            posthoc = frozen_eval(agent, ranked_pairs, name_to_rid)
            label_acc = {lbl: float(row.get("top1", 0.0)) for lbl, row in posthoc.items()}
            utility = sum(label_acc.get(lbl, 0.0) * prob_by_label.get(lbl, 0.0) for lbl in label_acc)
            realized_utility = sum(label_acc.get(lbl, 0.0) * (cnt / total) for lbl, cnt in counts.items())
            fairness = _mean([label_acc.get(p.label, 0.0) for p in ranked_pairs])
            observed_labels = set(counts)
            zero_count_labels = {p.label for p in ranked_pairs if p.label not in observed_labels}
            live_row = _aggregate_episode_metrics(assist_rows)
            row = {
                **live_row,
                "compute": compute_snapshot(agent, phase_a_wall_s.get(baseline, 0.0) + phase_b_wall),
                "memory": memory_snapshot(agent),
                "utility_top1": utility,
                "realized_utility_top1": realized_utility,
                "fairness_top1": fairness,
                "fairness_top1_all_onboarded": fairness,
                "fairness_top1_observed_phase_b": _mean([label_acc.get(lbl, 0.0) for lbl in observed_labels]),
                "fairness_top1_unsampled_phase_b": _mean([label_acc.get(lbl, 0.0) for lbl in zero_count_labels]),
                "head_top1": _mean([label_acc.get(lbl, 0.0) for lbl in head_labels]),
                "tail_top1": _mean([label_acc.get(lbl, 0.0) for lbl in tail_labels]),
                "head_tail_gap": _mean([label_acc.get(lbl, 0.0) for lbl in head_labels]) - _mean([label_acc.get(lbl, 0.0) for lbl in tail_labels]),
                "live_utility_top1": sum(label_live_acc.get(lbl, 0.0) * (cnt / total) for lbl, cnt in counts.items()),
                "live_fairness_top1_observed": _mean(list(label_live_acc.values())),
                "frequency_hist": dict(counts),
                "head_realised_n": int(sum(counts.get(lbl, 0) for lbl in head_labels)),
                "tail_realised_n": int(sum(counts.get(lbl, 0) for lbl in tail_labels)),
                "n_eval_pairs": len(ranked_pairs),
                "n_unique_phase_b_pairs": len(counts),
                "n_zero_count_fairness_pairs": len(zero_count_labels),
                "fairness_denominator": "all_onboarded_recipe_preference_workflows",
                "zipf_distribution": "joint_recipe_zipf_times_preference_zipf",
                "phase_a_wall_s": phase_a_wall_s.get(baseline, 0.0),
                "phase_b_wall_s": phase_b_wall,
            }
            alpha_row[baseline] = row
            all_episode_rows.extend({"baseline": baseline, **r} for r in episode_rows)
            all_step_rows.extend(step_rows)
            all_memory_rows.extend(memory_rows)
        per_alpha[str(alpha)] = alpha_row
    _append_jsonl(out / "scenario_events.jsonl", scenario_rows)
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    _append_jsonl(out / "memory_trace.jsonl", all_memory_rows)
    metrics = {
        "seed": seed,
        "n_recipes": len(recipes),
        "n_preferences": len(prefs),
        "n_workflows": len(ranked_pairs),
        "scenario": {
            "description": "Balanced prep-book onboarding followed by branched rush-hour Zipf demand over recipe/preference workflows.",
            "phase_a_cycles": max(1, int(cfg_obj.phase_a_cycles)),
            "phase_b_events_per_alpha": max(1, n_events),
            "head_definition": "top quartile of intended Zipf workflow ranks",
            "tail_definition": "bottom quartile of intended Zipf workflow ranks, including workflows not sampled in a finite branch",
        },
        "per_alpha": per_alpha,
    }
    _write_json(out / "metrics.json", metrics)
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
    anchor_run = replace(run_config, baselines=PAPER_BASELINES)
    stream_cfg = DeploymentStreamConfig(n_recipes=cfg_obj.n_recipes, n_phase_b_events=cfg_obj.n_events)
    return run_deployment_stream(seed, out, anchor_run, stream_cfg)


def run_continual_learning_regularizer_sweep(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    cl_run = replace(run_config, baselines=PAPER_BASELINES)
    stream_cfg = DeploymentStreamConfig(n_recipes=cfg_obj.n_recipes, n_phase_b_events=cfg_obj.n_events)
    return run_deployment_stream(seed, out, cl_run, stream_cfg)


def run_posterior_ablation_matrix(seed: int, out: Path, run_config: RunConfig, cfg_obj: CrossRecipeTransferConfig) -> Dict[str, Any]:
    ab_cfg = TransferSuiteConfig(
        n_recipes=cfg_obj.n_recipes,
        n_preferences=cfg_obj.n_preferences,
        diagonal_cycles=cfg_obj.diagonal_cycles,
        offdiag_repeats=cfg_obj.repeats,
        baselines=PAPER_BASELINES,
    )
    return run_cross_recipe_transfer_suite(seed, out, run_config, ab_cfg)


def run_recipe_preference_factor_ablation(seed: int, out: Path, run_config: RunConfig, cfg_obj: CrossRecipeTransferConfig) -> Dict[str, Any]:
    ab_cfg = TransferSuiteConfig(
        n_recipes=cfg_obj.n_recipes,
        n_preferences=cfg_obj.n_preferences,
        diagonal_cycles=cfg_obj.diagonal_cycles,
        offdiag_repeats=cfg_obj.repeats,
        baselines=PAPER_BASELINES,
    )
    return run_cross_recipe_transfer_suite(seed, out, run_config, ab_cfg)


def _baseline_tuning_grid_ranges(cfg_obj: BaselineHyperparameterTuningConfig) -> Dict[str, Any]:
    return {
        "bc_learning_rate": list(cfg_obj.bc_learning_rates),
        "bc_l2": list(cfg_obj.bc_l2_values),
        "bc_history_len": list(cfg_obj.bc_history_lens),
        "bc_history_bins": list(cfg_obj.bc_history_bins_values),
        "bc_epochs_cold_warm": [list(v) for v in cfg_obj.bc_epoch_pairs],
        "er_buffer_size": list(cfg_obj.er_buffer_sizes),
        "er_batch_size": list(cfg_obj.er_batch_sizes),
        "er_recency_alpha": list(cfg_obj.er_recency_alphas),
        "er_uniform_mix": list(cfg_obj.er_uniform_mixes),
        "ewc_lambda": list(cfg_obj.ewc_lambdas),
        "maxent_learning_rate": list(cfg_obj.maxent_learning_rates),
        "maxent_l2": list(cfg_obj.maxent_l2_values),
        "maxent_iters_cold_warm": [list(v) for v in cfg_obj.maxent_iter_pairs],
    }


def _dedupe_override_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in candidates:
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _cap_tuning_candidates(candidates: Sequence[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    defaults = _learner_hyperparameter_metadata(AdaptiveHRCAgent(cfg=DEFAULT_CONFIG))
    deduped = _dedupe_override_candidates(candidates)
    if len(deduped) <= max(1, int(max_points)):
        return deduped
    ranked = sorted(
        deduped,
        key=lambda row: (
            sum(abs(float(v) - float(defaults.get(k, v))) for k, v in row.items() if isinstance(v, (int, float))),
            tuple(sorted((str(k), str(v)) for k, v in row.items())),
        ),
    )
    stride = max(1, len(deduped) // max(1, int(max_points)))
    spread = deduped[::stride]
    return _dedupe_override_candidates(ranked[: max(1, int(max_points)) // 2] + spread)[: max(1, int(max_points))]


def _baseline_hyperparameter_candidates(
    baseline: str,
    cfg_obj: BaselineHyperparameterTuningConfig,
) -> List[Dict[str, Any]]:
    bc_epoch_pairs = tuple((int(c), int(w)) for c, w in cfg_obj.bc_epoch_pairs)
    maxent_iter_pairs = tuple((int(c), int(w)) for c, w in cfg_obj.maxent_iter_pairs)
    max_points = int(cfg_obj.max_grid_points_per_baseline)
    candidates: List[Dict[str, Any]] = []
    if baseline in {"bc", "experience_replay_bc", "recency_prioritized_replay"}:
        for lr in cfg_obj.bc_learning_rates:
            for l2 in cfg_obj.bc_l2_values:
                for hist_len in cfg_obj.bc_history_lens:
                    for hist_bins in cfg_obj.bc_history_bins_values:
                        for cold, warm in bc_epoch_pairs:
                            base = {
                                "bc_learning_rate": float(lr),
                                "bc_l2": float(l2),
                                "bc_history_len": int(hist_len),
                                "bc_history_bins": int(hist_bins),
                                "bc_epochs_cold": int(cold),
                                "bc_epochs_warm": int(warm),
                            }
                            if baseline in {"experience_replay_bc", "recency_prioritized_replay"}:
                                for buffer_size in cfg_obj.er_buffer_sizes:
                                    for batch_size in cfg_obj.er_batch_sizes:
                                        row = {
                                            **base,
                                            "er_buffer_size": int(buffer_size),
                                            "er_batch_size": int(batch_size),
                                        }
                                        if baseline == "recency_prioritized_replay":
                                            for alpha in cfg_obj.er_recency_alphas:
                                                for mix in cfg_obj.er_uniform_mixes:
                                                    candidates.append({
                                                        **row,
                                                        "er_recency_alpha": float(alpha),
                                                        "er_uniform_mix": float(mix),
                                                    })
                                        else:
                                            candidates.append(row)
                            else:
                                candidates.append(base)
    elif baseline in {"ewc", "online_ewc"}:
        for lam in cfg_obj.ewc_lambdas:
            for lr in cfg_obj.maxent_learning_rates:
                for l2 in cfg_obj.maxent_l2_values:
                    for cold, warm in maxent_iter_pairs:
                        candidates.append({
                            "ewc_lambda": float(lam),
                            "maxent_learning_rate": float(lr),
                            "maxent_l2": float(l2),
                            "maxent_iters_cold": int(cold),
                            "maxent_iters_warm": int(warm),
                        })
    else:
        candidates.append({})
    return _cap_tuning_candidates(candidates or [{}], max_points)


def run_baseline_hyperparameter_tuning(
    seed: int,
    out: Path,
    run_config: RunConfig,
    cfg_obj: BaselineHyperparameterTuningConfig,
) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=cfg_obj.n_preferences)
    rng = _rng(seed, 2701)
    usable = [p for p in pairs if p.preference_name in PRESET_PREFERENCES]
    rng.shuffle(usable)
    cut = max(1, int(round((1.0 - float(cfg_obj.validation_fraction)) * len(usable))))
    train_pairs = usable[:cut]
    calibration_pairs = usable[cut:] or usable[: max(1, min(4, len(usable)))]
    calibration_pairs = calibration_pairs[: max(1, int(cfg_obj.max_eval_pairs))]
    selected_prefs = sorted({p.preference_name for p in train_pairs + calibration_pairs})
    _record_common_artifacts(out, seed, cfg_obj, recipes, selected_prefs)
    rows: List[Dict[str, Any]] = []
    per_baseline: Dict[str, Any] = {}
    objective = "hrc_net_action_value_shared_heldout_calibration"
    for baseline in cfg_obj.baselines:
        if baseline != "full":
            try:
                make_agent(baseline, base_config(seed, run_config))
            except Exception:
                continue
        baseline_rows: List[Dict[str, Any]] = []
        candidates = _baseline_hyperparameter_candidates(baseline, cfg_obj)
        for idx, overrides in enumerate(candidates):
            cfg = base_config(seed, run_config, **overrides)
            agent = make_agent(baseline, cfg)
            name_to_rid: Dict[str, str] = {}
            for pair in train_pairs:
                observe_episode(agent, pair, name_to_rid)
            eval_rows: List[Dict[str, Any]] = []
            for pair in calibration_pairs:
                row, _steps = assist_episode(
                    agent,
                    pair,
                    dict(name_to_rid),
                    run_config=run_config,
                    commit=False,
                    event_tag={"condition": "baseline_hyperparameter_calibration", "candidate_idx": idx},
                )
                eval_rows.append(row)
            agg = _aggregate_episode_metrics(eval_rows)
            net_value = _hrc_net_action_value(agg)
            metadata = _learner_hyperparameter_metadata(agent)
            replay_meta = {}
            if hasattr(agent, "replay_buffer_metadata"):
                replay_meta = dict(agent.replay_buffer_metadata())  # type: ignore[attr-defined]
            candidate_row = {
                "baseline": baseline,
                "candidate_idx": int(idx),
                "overrides": dict(overrides),
                "selected_values": {k: metadata.get(k) for k in overrides},
                "hrc_net_action_value": float(net_value),
                "live_top1": float(agg.get("live_top1", 0.0)),
                "robot_wrong_rate": float(agg.get("robot_wrong_rate", 0.0)),
                "net_robot_action_value": float(agg.get("net_robot_action_value", 0.0)),
                "n_calibration_pairs": len(calibration_pairs),
                "n_train_pairs": len(train_pairs),
                "replay_buffer": replay_meta,
            }
            baseline_rows.append(candidate_row)
            rows.append(candidate_row)
        eligible = baseline_rows or [{
            "baseline": baseline,
            "candidate_idx": -1,
            "overrides": {},
            "hrc_net_action_value": 0.0,
            "live_top1": 0.0,
            "robot_wrong_rate": 1.0,
        }]
        best = max(
            eligible,
            key=lambda r: (
                float(r.get("hrc_net_action_value", 0.0)),
                float(r.get("live_top1", 0.0)),
                -float(r.get("robot_wrong_rate", 0.0)),
            ),
        )
        per_baseline[baseline] = {
            "recommendation": dict(best),
            "n_candidates": len(baseline_rows),
            "grid_points_evaluated": len(baseline_rows),
            "grid_capped": False,
            "reported_only_does_not_mutate_default_config": True,
        }
    _append_jsonl(out / "baseline_hyperparameter_candidates.jsonl", rows)
    metrics = {
        "seed": seed,
        "objective": objective,
        "shared_calibration_protocol": {
            "train_pairs": [p.label for p in train_pairs],
            "calibration_pairs": [p.label for p in calibration_pairs],
            "validation_fraction": float(cfg_obj.validation_fraction),
            "same_split_for_all_systems": True,
            "not_used_to_tune_final_test_stream": True,
        },
        "grid_ranges": _baseline_tuning_grid_ranges(cfg_obj),
        "per_baseline": per_baseline,
        "candidate_rows": rows,
        "reported_only_does_not_mutate_default_config": True,
        "default_values": _learner_hyperparameter_metadata(AdaptiveHRCAgent(cfg=DEFAULT_CONFIG)),
    }
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
    row, steps = assist_episode(agent, b, name_to_rid, run_config=run_config, event_tag={"stress": "prefix_collision"})
    before_div = [s for s in steps if int(s.get("step", 0)) < int(shared)]
    after_div = [s for s in steps if int(s.get("step", 0)) >= int(shared)]
    return {
        "score": row["live_top1"],
        "live_top1": row["live_top1"],
        "shared_prefix": shared,
        "wrong_recipe_commit_rate": 1.0 if row.get("classification_recipe_id") not in (None, name_to_rid.get(b.recipe_name)) else 0.0,
        "observation_suppression_rate": 0.0 if row.get("needs_observation") else 1.0,
        "posterior_switch_latency": float(row.get("posterior_steps_to_lock", -1.0)),
        "top1_before_diverging_step": _mean([1.0 if s.get("correct_top1") else 0.0 for s in before_div]),
        "top1_after_diverging_step": _mean([1.0 if s.get("correct_top1") else 0.0 for s in after_div]),
    }


def _stress_preference_thrash(seed: int, baseline: str, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, 1, n_preferences=7)
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    rows = []
    step_rows: List[Dict[str, Any]] = []
    latest_flips = 0
    last_latest = None
    for i in range(max(1, cfg_obj.reps)):
        pair = pairs[i % len(pairs)]
        if i == 0:
            observe_episode(agent, pair, name_to_rid)
        else:
            before_latest = dict(agent.decay.latest_by_recipe)
            row, steps = assist_episode(agent, pair, name_to_rid, run_config=run_config, event_tag={"stress": "preference_thrash", "switch_index": i})
            after_latest = tuple(sorted(agent.decay.latest_by_recipe.items()))
            if last_latest is not None and after_latest != last_latest:
                latest_flips += 1
            if before_latest != agent.decay.latest_by_recipe:
                latest_flips += 1
            last_latest = after_latest
            rows.append(row)
            step_rows.extend(steps)
    agg = _aggregate_episode_metrics(rows)
    return {
        "score": agg["live_top1"],
        **agg,
        "mean_steps_to_stable_inference": float(agg.get("posterior_steps_to_lock", -1.0)),
        "latest_pin_flip_count": float(latest_flips),
        "robot_wrong_rate_after_switch": _mean([1.0 if _row_robot_wrong(s) else 0.0 for s in step_rows]),
        "top1_by_switch_index": agg["live_top1"],
        "active_variants": len(agent.decay.active),
    }


def _stress_reentry(seed: int, baseline: str, run_config: RunConfig, cfg_obj: StressConfig, distractor: bool = False) -> Dict[str, Any]:
    target, variant, distractors = _target_and_variant(seed, cfg_obj.n_recipes)
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    observe_episode(agent, target, name_to_rid)
    observe_episode(agent, variant, name_to_rid)
    _advance_gap(agent, cfg_obj.gap, distractors, name_to_rid, neutral=not distractor)
    target_rid = name_to_rid.get(target.recipe_name)
    target_hash = variant_hash(agent._tokens_from_action_labels(target.actions))
    pruned_before = bool(target_rid is not None and (target_rid, target_hash) in agent.decay.pruned)
    episode_pair = target
    distractor_midpoint = -1
    if distractor and distractors:
        distractor_midpoint = max(1, len(target.actions) // 2)
        distractor_action = distractors[0].actions[0]
        late_actions = tuple(target.actions[:distractor_midpoint]) + (distractor_action,) + tuple(target.actions[distractor_midpoint:])
        episode_pair = RecipePrefPair(
            target.recipe_name,
            f"{target.preference_name}_late_distractor",
            f"{target.label}/late_distractor",
            late_actions,
            target.preference,
            target.base_pref,
            target.applied_axes,
            target.failed_axes,
            target.unchanged_axes,
        )
    row, steps = assist_episode(
        agent,
        episode_pair,
        name_to_rid,
        run_config=run_config,
        event_tag={"stress": "late_distractor" if distractor else "rare_reentry", "distractor_midpoint": distractor_midpoint},
    )
    post = frozen_eval(agent, [target], name_to_rid)
    recovery = next((int(s.get("step", 0)) for s in steps if s.get("correct_top1")), -1)
    before_distractor = [s for s in steps if distractor_midpoint < 0 or int(s.get("step", 0)) < distractor_midpoint]
    after_distractor = [s for s in steps if distractor_midpoint >= 0 and int(s.get("step", 0)) > distractor_midpoint]
    recovery_after_distractor = next((int(s.get("step", 0)) - distractor_midpoint for s in after_distractor if s.get("correct_top1")), -1)
    return {
        "score": row["live_top1"],
        **row,
        "reentry_live_top1": row["live_top1"],
        "pruned_step_top1": _mean([1.0 if s.get("correct_top1") else 0.0 for s in steps]) if pruned_before else 0.0,
        "post_reentry_top1": _mean([v.get("top1", 0.0) for v in post.values()]),
        "reentry_recovery_steps": recovery,
        "target_live_top1_before_distractor": _mean([1.0 if s.get("correct_top1") else 0.0 for s in before_distractor]),
        "target_live_top1_after_distractor": _mean([1.0 if s.get("correct_top1") else 0.0 for s in after_distractor]),
        "recovery_steps_after_distractor": recovery_after_distractor if distractor else -1,
        "posterior_recipe_flip_rate": _mean([
            1.0 if target_rid is not None and s.get("posterior_recipe") not in (None, target_rid) else 0.0
            for s in steps
        ]),
        "robot_wrong_rate": _mean([1.0 if _row_robot_wrong(s) else 0.0 for s in steps]),
        "robot_wrong_rate_after_distractor": _mean([1.0 if _row_robot_wrong(s) else 0.0 for s in after_distractor]),
        "active_variants": len(agent.decay.active),
        "pruned_variants": len(agent.decay.pruned),
    }


def _stress_memory_exhaustion(seed: int, baseline: str, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=7)
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    for pair in pairs[: max(1, min(len(pairs), cfg_obj.reps * 4))]:
        observe_episode(agent, pair, name_to_rid)
    evals = frozen_eval(agent, pairs, name_to_rid)
    top1 = _mean([v.get("top1", 0.0) for v in evals.values()])
    pin_violations = sum(
        1 for rid, latest in agent.decay.latest_by_recipe.items()
        if (rid, latest) not in agent.decay.active
    )
    capacity_sweep: List[Dict[str, Any]] = []
    capacity = max(1, int(agent.cfg.max_variants_per_recipe))
    for ratio in (0.8, 0.9, 1.0, 1.2):
        probe = make_agent(baseline, base_config(seed, run_config))
        probe_map: Dict[str, str] = {}
        n_train = max(1, min(len(pairs), int(math.ceil(capacity * ratio))))
        train_pairs = pairs[:n_train]
        for pair in train_pairs:
            observe_episode(probe, pair, probe_map)
        probe_eval = frozen_eval(probe, train_pairs, probe_map)
        probe_pin_violations = sum(
            1 for rid, latest in probe.decay.latest_by_recipe.items()
            if (rid, latest) not in probe.decay.active
        )
        capacity_sweep.append({
            "capacity_ratio_target": float(ratio),
            "n_train_pairs": n_train,
            "live_top1": _mean([v.get("top1", 0.0) for v in probe_eval.values()]),
            "frozen_top1": _mean([v.get("top1", 0.0) for v in probe_eval.values()]),
            "active_variants": len(probe.decay.active),
            "pruned_variants": len(probe.decay.pruned),
            "latest_pin_violation_count": float(probe_pin_violations),
        })
    return {
        "score": top1,
        "live_top1": top1,
        "frozen_top1": top1,
        "capacity_ratio": len(agent.decay.active) / max(1.0, float(agent.cfg.max_variants_per_recipe)),
        "active_variants": len(agent.decay.active),
        "pruned_variants": len(agent.decay.pruned),
        "latest_pin_violation_count": float(pin_violations),
        "capacity_sweep": capacity_sweep,
    }


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
        rows = [fns[kind](seed + i * 37, baseline, run_config, cfg_obj) for i in range(max(1, int(cfg_obj.reps)))]
        per_baseline[baseline] = {key: _mean([r.get(key, 0.0) for r in rows]) for key in sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float, bool))})}
        per_baseline[baseline]["rep_rows"] = rows
    metrics = {"seed": seed, "stress_kind": kind, "per_baseline": per_baseline}
    _write_json(out / "metrics.json", metrics)
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


def run_adversarial_stress_suite(seed: int, out: Path, run_config: RunConfig, cfg_obj: StressConfig) -> Dict[str, Any]:
    per_stress: Dict[str, Any] = {}
    for kind in (
        "memory_exhaustion_stress",
        "prefix_collision_stress",
        "preference_thrash_stress",
        "rare_reentry_stress",
        "late_distractor_stress",
    ):
        per_stress[kind] = _run_stress(seed, _ensure(out / kind), run_config, cfg_obj, kind)
    per_baseline: Dict[str, Any] = {}
    baselines = sorted({
        baseline
        for stress_row in per_stress.values()
        for baseline in (stress_row.get("per_baseline", {}) or {})
    })
    for baseline in baselines:
        rows = [
            stress_row.get("per_baseline", {}).get(baseline, {})
            for stress_row in per_stress.values()
            if baseline in (stress_row.get("per_baseline", {}) or {})
        ]
        per_baseline[baseline] = {
            "stress_macro_score": _mean([float(r.get("score", 0.0)) for r in rows]),
            "stress_macro_live_top1": _mean([float(r.get("live_top1", r.get("score", 0.0))) for r in rows]),
            "stress_macro_robot_wrong_rate": _mean([float(r.get("robot_wrong_rate", r.get("robot_wrong_rate_after_switch", 0.0))) for r in rows]),
            "n_stresses": len(rows),
        }
    metrics = {
        "seed": seed,
        "per_stress": per_stress,
        "per_baseline": per_baseline,
        "standardized_metrics": {
            "memory_exhaustion": ("live_top1", "frozen_top1", "active_variants", "pruned_variants", "latest_pin_violation_count"),
            "prefix_collision": ("wrong_recipe_commit_rate", "observation_suppression_rate", "posterior_switch_latency", "top1_before_diverging_step", "top1_after_diverging_step"),
            "preference_thrash": ("mean_steps_to_stable_inference", "latest_pin_flip_count", "robot_wrong_rate_after_switch", "top1_by_switch_index"),
            "rare_reentry": ("reentry_live_top1", "pruned_step_top1", "post_reentry_top1", "reentry_recovery_steps"),
            "late_distractor": ("reentry_live_top1", "posterior_recipe_flip_rate", "recovery_steps_after_distractor", "robot_wrong_rate"),
        },
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


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
        metrics = _run_compute_tradeoff(seed, _ensure(out / f"n_{n}"), small_run, ComputeTradeoffConfig(n_recipes=int(n), n_events=cfg_obj.n_events))
        per_n[str(n)] = metrics
    result = {"seed": seed, "per_recipe_count": per_n}
    _write_json(out / "metrics.json", result)
    return result


def _material_pair_candidates(recipe_name: str, fn: Callable[[], List[str]], prefs: Sequence[str]) -> List[RecipePrefPair]:
    pairs: List[RecipePrefPair] = []
    seen_orders: set = set()
    for pref in prefs:
        try:
            pair = materialize_pair(recipe_name, pref, fn)
        except Exception:
            continue
        if pair.actions in seen_orders:
            continue
        seen_orders.add(pair.actions)
        if pair.base_pref or pair.applied_axes:
            pairs.append(pair)
    return pairs


def _sorted_preference_choices(pairs: Sequence[RecipePrefPair]) -> List[RecipePrefPair]:
    pref_rank = {pref: idx for idx, pref in enumerate(PREFERENCE_NAMES)}
    return sorted(
        list(pairs),
        key=lambda p: (pref_rank.get(p.preference_name, len(pref_rank)), p.preference_name, p.label),
    )


def _choose_preference_zipf(pairs: Sequence[RecipePrefPair], alpha: float, rng: np.random.Generator) -> Optional[RecipePrefPair]:
    choices = _sorted_preference_choices(pairs)
    if not choices:
        return None
    probs = _zipf_probs(len(choices), float(alpha))
    return choices[int(rng.choice(len(choices), p=probs))]


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


def _valid_ladder_pairs_for_recipe(
    recipe_name: str,
    generator_fn: Callable[[], List[str]],
    preferences: Sequence[str],
    rng: np.random.Generator,
) -> List[RecipePrefPair]:
    pref_names = [p for p in preferences if p in PRESET_PREFERENCES]
    rng.shuffle(pref_names)
    pairs: List[RecipePrefPair] = []
    seen_orderings: set = set()
    for pref_name in pref_names:
        try:
            pair = materialize_pair(recipe_name, pref_name, generator_fn)
        except Exception:
            continue
        if not pair.base_pref and not pair.applied_axes:
            continue
        ordering = _order_hash(pair)
        if ordering in seen_orderings:
            continue
        seen_orderings.add(ordering)
        pairs.append(pair)
    return pairs


def _resolve_ladder_rung_count(cfg_obj: DeploymentLadderConfig, available: int) -> int:
    requested = cfg_obj.n_rungs if cfg_obj.n_rungs is not None else cfg_obj.max_rungs
    n_rungs = min(max(1, int(requested)), max(0, int(available)))
    if n_rungs < 2:
        raise RuntimeError(
            "Deployment ladder requires at least two effective preference rungs "
            "so the stream contains an update after initial observation."
        )
    return n_rungs


def _deployment_ladder_plan(
    seed: int,
    cfg_obj: DeploymentLadderConfig,
    scenario_kind: Literal["heterogeneous", "shared"],
) -> Tuple[List[str], List[List[RecipePrefPair]], Dict[str, Any]]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    if not builders:
        raise RuntimeError("deployment ladder requires at least one selected recipe")
    recipes = [recipe for recipe, _fn in builders]
    prefs = tuple(cfg_obj.preferences or PREFERENCE_NAMES)
    rng = _rng(seed, 8729 if scenario_kind == "heterogeneous" else 8731)
    if scenario_kind == "heterogeneous":
        by_recipe: Dict[str, List[RecipePrefPair]] = {}
        for idx, (recipe_name, fn) in enumerate(builders):
            by_recipe[recipe_name] = _valid_ladder_pairs_for_recipe(
                recipe_name,
                fn,
                prefs,
                _rng(seed, 8800 + idx),
            )
        available = min((len(v) for v in by_recipe.values()), default=0)
        n_rungs = _resolve_ladder_rung_count(cfg_obj, available)
        rungs = [
            [by_recipe[recipe_name][rung_idx] for recipe_name in recipes]
            for rung_idx in range(n_rungs)
        ]
        selection = {
            "scenario_kind": scenario_kind,
            "selection_policy": "per_recipe_seed_shuffled_unique_effective_preferences",
            "available_preferences_by_recipe": {
                recipe: [p.preference_name for p in pairs]
                for recipe, pairs in by_recipe.items()
            },
        }
    else:
        pref_names = [p for p in prefs if p in PRESET_PREFERENCES]
        rng.shuffle(pref_names)
        seen_by_recipe: Dict[str, set] = {recipe: set() for recipe in recipes}
        rungs = []
        selected_preferences: List[str] = []
        rejected_preferences: List[Dict[str, Any]] = []
        desired = int(cfg_obj.n_rungs if cfg_obj.n_rungs is not None else cfg_obj.max_rungs)
        for pref_name in pref_names:
            rung_pairs: List[RecipePrefPair] = []
            rejection_reason = ""
            for recipe_name, fn in builders:
                try:
                    pair = materialize_pair(recipe_name, pref_name, fn)
                except Exception as exc:
                    rejection_reason = f"{recipe_name}:materialization_failed:{type(exc).__name__}"
                    break
                if not pair.base_pref and not pair.applied_axes:
                    rejection_reason = f"{recipe_name}:no_effective_axis"
                    break
                ordering = _order_hash(pair)
                if ordering in seen_by_recipe[recipe_name]:
                    rejection_reason = f"{recipe_name}:duplicate_ordering_for_recipe"
                    break
                rung_pairs.append(pair)
            if rejection_reason:
                rejected_preferences.append({"preference": pref_name, "reason": rejection_reason})
                continue
            rungs.append(rung_pairs)
            selected_preferences.append(pref_name)
            for pair in rung_pairs:
                seen_by_recipe[pair.recipe_name].add(_order_hash(pair))
            if len(rungs) >= max(1, desired):
                break
        n_rungs = _resolve_ladder_rung_count(cfg_obj, len(rungs))
        rungs = rungs[:n_rungs]
        selection = {
            "scenario_kind": scenario_kind,
            "selection_policy": "seed_shuffled_shared_preference_valid_for_all_recipes",
            "selected_shared_preferences": selected_preferences[:n_rungs],
            "rejected_shared_preferences": rejected_preferences,
        }
    rung_plan = [
        {
            "rung_idx": rung_idx,
            "rung_number": rung_idx + 1,
            "shared_preference": (
                rungs[rung_idx][0].preference_name
                if scenario_kind == "shared" and rungs[rung_idx]
                else None
            ),
            "pairs": [
                {
                    "recipe": pair.recipe_name,
                    "preference": pair.preference_name,
                    "pair": pair.label,
                    "applied_axes": list(pair.applied_axes),
                }
                for pair in rungs[rung_idx]
            ],
        }
        for rung_idx in range(len(rungs))
    ]
    selection["rung_plan"] = rung_plan
    return recipes, rungs, selection


def _ladder_seen_tags(
    pair: RecipePrefPair,
    seen_recipes: set,
    seen_preferences: set,
    seen_pairs: set,
    seen_preferences_by_recipe: Mapping[str, set],
) -> Dict[str, Any]:
    seen_recipe = pair.recipe_name in seen_recipes
    seen_preference = pair.preference_name in seen_preferences
    seen_pair = pair.label in seen_pairs
    same_recipe_seen = pair.preference_name in seen_preferences_by_recipe.get(pair.recipe_name, set())
    other_recipe_seen = any(
        pair.preference_name in vals
        for recipe, vals in seen_preferences_by_recipe.items()
        if recipe != pair.recipe_name
    )
    return {
        "seen_recipe_before": seen_recipe,
        "seen_preference_before": seen_preference,
        "seen_preference_same_recipe_before": same_recipe_seen,
        "seen_preference_other_recipe_before": other_recipe_seen,
        "seen_pair_before": seen_pair,
        "four_cell_before": _four_cell(seen_recipe, seen_preference),
        "exposure_cell_before": _four_cell(seen_recipe, seen_preference),
        "transfer_cell_before": _transfer_cell(
            seen_recipe=seen_recipe,
            seen_pair=seen_pair,
            seen_preference_global=seen_preference,
            seen_preference_same_recipe=same_recipe_seen,
            seen_preference_other_recipe=other_recipe_seen,
        ),
    }


def _mark_ladder_seen(
    pair: RecipePrefPair,
    seen_recipes: set,
    seen_preferences: set,
    seen_pairs: set,
    seen_preferences_by_recipe: Dict[str, set],
) -> None:
    seen_recipes.add(pair.recipe_name)
    seen_preferences.add(pair.preference_name)
    seen_pairs.add(pair.label)
    seen_preferences_by_recipe[pair.recipe_name].add(pair.preference_name)


def _deployment_ladder_events(
    rungs: Sequence[Sequence[RecipePrefPair]],
    cfg_obj: DeploymentLadderConfig,
    scenario_kind: Literal["heterogeneous", "shared"],
) -> List[ScenarioEvent]:
    seen_recipes: set = set()
    seen_preferences: set = set()
    seen_pairs: set = set()
    seen_preferences_by_recipe: Dict[str, set] = defaultdict(set)
    settle_repeats = max(1, int(cfg_obj.settle_repeats_per_recipe))
    events: List[ScenarioEvent] = []
    update_event_type = (
        "ladder_pref_update_heterogeneous"
        if scenario_kind == "heterogeneous"
        else "ladder_shared_pref_shift"
    )
    for rung_idx, rung_pairs_raw in enumerate(rungs):
        rung_pairs = list(rung_pairs_raw)
        rung_signature = tuple(pair.preference_name for pair in rung_pairs)
        shared_pref = rung_pairs[0].preference_name if scenario_kind == "shared" and rung_pairs else ""
        for recipe_pos, pair in enumerate(rung_pairs):
            event_type = "ladder_initial_observe" if rung_idx == 0 else update_event_type
            mode: Literal["observe", "assist"] = "observe" if rung_idx == 0 else "assist"
            tags = {
                "event_idx": len(events),
                "event_type": event_type,
                "requested_event_type": event_type,
                "event_substituted": False,
                "ladder_scenario": scenario_kind,
                "ladder_stage": "add",
                "rung_idx": int(rung_idx),
                "rung_number": int(rung_idx + 1),
                "recipe_position": int(recipe_pos),
                "settle_cycle": -1,
                "rung_preference_signature": list(rung_signature),
                "shared_preference": shared_pref,
                **_ladder_seen_tags(pair, seen_recipes, seen_preferences, seen_pairs, seen_preferences_by_recipe),
            }
            events.append(ScenarioEvent(
                pair=pair,
                mode=mode,
                phase="phase_a" if rung_idx == 0 else "ladder_update",
                user_id="ladder_user",
                tags=tags,
            ))
            _mark_ladder_seen(pair, seen_recipes, seen_preferences, seen_pairs, seen_preferences_by_recipe)
        for settle_cycle in range(settle_repeats):
            for recipe_pos, pair in enumerate(rung_pairs):
                tags = {
                    "event_idx": len(events),
                    "event_type": "ladder_settle",
                    "requested_event_type": "ladder_settle",
                    "event_substituted": False,
                    "ladder_scenario": scenario_kind,
                    "ladder_stage": "settle",
                    "rung_idx": int(rung_idx),
                    "rung_number": int(rung_idx + 1),
                    "recipe_position": int(recipe_pos),
                    "settle_cycle": int(settle_cycle),
                    "rung_preference_signature": list(rung_signature),
                    "shared_preference": shared_pref,
                    **_ladder_seen_tags(pair, seen_recipes, seen_preferences, seen_pairs, seen_preferences_by_recipe),
                }
                events.append(ScenarioEvent(
                    pair=pair,
                    mode="assist",
                    phase="ladder_settle",
                    user_id="ladder_user",
                    tags=tags,
                ))
                _mark_ladder_seen(pair, seen_recipes, seen_preferences, seen_pairs, seen_preferences_by_recipe)
    return events


def _ladder_rung_memory_trace(memory_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rung in sorted({int(r.get("rung_idx", -1)) for r in memory_rows if r.get("rung_idx") is not None}):
        rows = [r for r in memory_rows if int(r.get("rung_idx", -1)) == rung]
        if not rows:
            continue
        last = rows[-1]
        out.append({
            "rung_idx": int(rung),
            "rung_number": int(rung + 1),
            "active_variants": float(last.get("active_variants", 0.0)),
            "pruned_variants": float(last.get("pruned_variants", 0.0)),
            "latest_keys": float(last.get("latest_keys", 0.0)),
            "global_rate": float(last.get("global_rate", 0.0)),
            "mean_active_weight": float(last.get("mean_active_weight", 0.0)),
        })
    return out


def _run_deployment_ladder(
    seed: int,
    out: Path,
    run_config: RunConfig,
    cfg_obj: DeploymentLadderConfig,
    scenario_kind: Literal["heterogeneous", "shared"],
) -> Dict[str, Any]:
    recipes, rungs, selection = _deployment_ladder_plan(seed, cfg_obj, scenario_kind)
    events = _deployment_ladder_events(rungs, cfg_obj, scenario_kind)
    selected_preferences = sorted({pair.preference_name for rung in rungs for pair in rung})
    _record_common_artifacts(out, seed, cfg_obj, recipes, selected_preferences)
    _append_jsonl(out / "scenario_events.jsonl", (_scenario_event_row(e) for e in events))
    per_baseline: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    all_memory_rows: List[Dict[str, Any]] = []
    for baseline in run_config.baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        simulator_observed_labels: set = set()
        simulator_observed_recipes: set = set()
        episode_rows: List[Dict[str, Any]] = []
        step_rows: List[Dict[str, Any]] = []
        memory_rows: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        for idx, event in enumerate(events):
            tags = {"phase": event.phase, "user_id": event.user_id, **dict(event.tags)}
            if event.mode == "observe":
                row = observe_episode(agent, event.pair, name_to_rid)
                row.update(tags)
                episode_rows.append(row)
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
            simulator_observed_labels.add(event.pair.label)
            simulator_observed_recipes.add(event.pair.recipe_name)
            memory_rows.append({
                "event_idx": idx,
                "baseline": baseline,
                "rung_idx": int(tags.get("rung_idx", -1)),
                "rung_number": int(tags.get("rung_number", 0)),
                "ladder_stage": str(tags.get("ladder_stage", "")),
                "event_type": str(tags.get("event_type", "")),
                **memory_snapshot(agent),
            })
        wall_s = time.perf_counter() - t0
        assist_rows = [r for r in episode_rows if r.get("mode") == "assist"]
        metrics = _aggregate_episode_metrics(assist_rows)
        metrics["compute"] = compute_snapshot(agent, wall_s)
        metrics["memory"] = memory_snapshot(agent)
        metrics["per_stage"] = {
            stage: _aggregate_episode_metrics([r for r in assist_rows if r.get("ladder_stage") == stage])
            for stage in ("add", "settle")
        }
        metrics["per_rung"] = {
            str(rung_idx): _aggregate_episode_metrics([
                r for r in assist_rows
                if int(r.get("rung_idx", -1)) == int(rung_idx)
            ])
            for rung_idx in range(len(rungs))
        }
        metrics["per_rung_stage"] = {
            str(rung_idx): {
                stage: _aggregate_episode_metrics([
                    r for r in assist_rows
                    if int(r.get("rung_idx", -1)) == int(rung_idx)
                    and r.get("ladder_stage") == stage
                ])
                for stage in ("add", "settle")
            }
            for rung_idx in range(len(rungs))
        }
        metrics["per_rung_update_to_settle_gain"] = {}
        for rung_idx in range(len(rungs)):
            add_row = metrics["per_rung_stage"][str(rung_idx)]["add"]
            settle_row = metrics["per_rung_stage"][str(rung_idx)]["settle"]
            has_add = float(add_row.get("n_episodes", 0.0)) > 0.0
            metrics["per_rung_update_to_settle_gain"][str(rung_idx)] = {
                "has_add_assist": has_add,
                "add_live_top1": float(add_row.get("live_top1", 0.0)) if has_add else None,
                "settle_live_top1": float(settle_row.get("live_top1", 0.0)),
                "settle_minus_add_live_top1": (
                    float(settle_row.get("live_top1", 0.0)) - float(add_row.get("live_top1", 0.0))
                    if has_add else None
                ),
                "add_robot_wrong_rate": float(add_row.get("robot_wrong_rate", 0.0)) if has_add else None,
                "settle_robot_wrong_rate": float(settle_row.get("robot_wrong_rate", 0.0)),
            }
        metrics["per_event_type"] = {
            event_type: _aggregate_episode_metrics([
                r for r in assist_rows
                if str(r.get("event_type")) == event_type
            ])
            for event_type in sorted({str(r.get("event_type")) for r in assist_rows})
        }
        metrics["per_transfer_cell"] = {
            cell: _aggregate_episode_metrics([
                r for r in assist_rows
                if r.get("transfer_cell_before") == cell
            ])
            for cell in TRANSFER_CELL_KEYS
        }
        metrics["coarse_exposure_cell"] = {
            cell: _aggregate_episode_metrics([
                r for r in assist_rows
                if r.get("exposure_cell_before", r.get("four_cell_before")) == cell
            ])
            for cell in FOUR_CELL_KEYS
        }
        metrics["four_cell"] = metrics["coarse_exposure_cell"]
        metrics["rung_memory_trace"] = _ladder_rung_memory_trace(memory_rows)
        metrics["valid_random_floor"] = _mean([
            1.0 / max(1, int(s.get("valid_next_action_count", 1)))
            for s in step_rows
        ])
        metrics["live_step_trace"] = live_step_trace(step_rows)
        metrics["calibration_curve"] = calibration_curve_from_steps(step_rows)
        metrics["latency_cdf"] = latency_cdf_from_steps(step_rows)
        metrics["policy_gate"] = policy_gate_reason_summary(step_rows)
        prediction_walls = [float(s.get("prediction_wall_s")) for s in step_rows if s.get("prediction_wall_s") is not None]
        metrics["mean_prediction_wall_s"] = _mean(prediction_walls)
        metrics["p95_prediction_wall_s"] = _p95(prediction_walls)
        per_baseline[baseline] = metrics
        all_episode_rows.extend({"baseline": baseline, **r} for r in episode_rows)
        all_step_rows.extend(step_rows)
        all_memory_rows.extend(memory_rows)
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    _append_jsonl(out / "memory_trace.jsonl", all_memory_rows)
    metrics = {
        "seed": seed,
        "scenario_kind": scenario_kind,
        "n_recipes": len(recipes),
        "n_rungs": len(rungs),
        "settle_repeats_per_recipe": int(max(1, cfg_obj.settle_repeats_per_recipe)),
        "settle_events_per_rung": int(len(recipes) * max(1, cfg_obj.settle_repeats_per_recipe)),
        "n_events": len(events),
        "n_assist_events": sum(1 for e in events if e.mode == "assist"),
        "n_observe_events": sum(1 for e in events if e.mode == "observe"),
        "event_distribution": _event_distribution(events),
        "ladder_selection": selection,
        "rung_plan": selection.get("rung_plan", []),
        "valid_random_floor": float((per_baseline.get("full", {}) or {}).get("valid_random_floor", 0.0)),
        "per_baseline": per_baseline,
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_deployment_ladder_heterogeneous_prefs(
    seed: int,
    out: Path,
    run_config: RunConfig,
    cfg_obj: DeploymentLadderConfig,
) -> Dict[str, Any]:
    return _run_deployment_ladder(seed, out, run_config, cfg_obj, "heterogeneous")


def run_deployment_ladder_shared_pref(
    seed: int,
    out: Path,
    run_config: RunConfig,
    cfg_obj: DeploymentLadderConfig,
) -> Dict[str, Any]:
    return _run_deployment_ladder(seed, out, run_config, cfg_obj, "shared")


def _deployment_stream_events(seed: int, cfg_obj: DeploymentStreamConfig, run_config: RunConfig) -> Tuple[List[str], List[str], List[ScenarioEvent], List[RecipePrefPair], List[Dict[str, Any]]]:
    n_recipes = cfg_obj.n_recipes
    n_events = cfg_obj.n_phase_b_events
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
    new_recipe_rejections: List[Dict[str, Any]] = []
    phase_a_pairs: List[RecipePrefPair] = []
    for recipe_name, fn in builders:
        candidates = by_recipe.get(recipe_name) or [materialize_pair(recipe_name, "identity", fn)]
        identity = next((p for p in candidates if p.preference_name == "identity"), candidates[0])
        non_identity = [p for p in candidates if p.preference_name != "identity"]
        if non_identity and rng.random() < float(cfg_obj.phase_a_non_identity_prob):
            phase_a_pairs.append(_choose_preference_zipf(non_identity, cfg_obj.zipf_alpha, rng) or non_identity[0])
        else:
            phase_a_pairs.append(identity)
    events: List[ScenarioEvent] = []
    seen_recipes: set = set()
    seen_preferences: set = set()
    seen_pairs: set = set()
    seen_preferences_by_recipe: Dict[str, set] = defaultdict(set)
    last_by_recipe: Dict[str, RecipePrefPair] = {}
    last_by_user_recipe: Dict[Tuple[str, str], RecipePrefPair] = {}
    for idx, pair in enumerate(phase_a_pairs):
        seen_preference_before = pair.preference_name in seen_preferences
        seen_preference_same_recipe_before = pair.preference_name in seen_preferences_by_recipe.get(pair.recipe_name, set())
        seen_preference_other_recipe_before = any(
            pair.preference_name in vals
            for recipe, vals in seen_preferences_by_recipe.items()
            if recipe != pair.recipe_name
        )
        events.append(ScenarioEvent(pair, "observe", "phase_a", "U1", {
            "event_idx": idx,
            "event_type": "phase_a_observe",
            "phase_a_preference": pair.preference_name,
            "four_cell_before": "unseen_unseen",
            "transfer_cell_before": _transfer_cell(
                seen_recipe=False,
                seen_pair=False,
                seen_preference_global=seen_preference_before,
                seen_preference_same_recipe=seen_preference_same_recipe_before,
                seen_preference_other_recipe=seen_preference_other_recipe_before,
            ),
            "seen_recipe_before": False,
            "seen_preference_before": seen_preference_before,
            "seen_preference_same_recipe_before": seen_preference_same_recipe_before,
            "seen_preference_other_recipe_before": seen_preference_other_recipe_before,
            "seen_pair_before": False,
        }))
        seen_recipes.add(pair.recipe_name)
        seen_preferences.add(pair.preference_name)
        seen_pairs.add(pair.label)
        seen_preferences_by_recipe[pair.recipe_name].add(pair.preference_name)
        last_by_recipe[pair.recipe_name] = pair
        last_by_user_recipe[("U1", pair.recipe_name)] = pair
    event_mix = dict(cfg_obj.event_mix)
    event_types = tuple(event_mix)
    weights = np.asarray([max(0.0, float(event_mix[k])) for k in event_types], dtype=float)
    weights = weights / max(float(weights.sum()), 1e-9)
    user_ids = list(users)
    history: List[RecipePrefPair] = list(phase_a_pairs)
    unused_new = list(new_builders)
    for i in range(max(0, int(n_events))):
        current_user, user_block, is_return_block = _blocked_user_for_event(user_ids, i, cfg_obj.user_block_size)
        block_start = i > 0 and (i % max(1, int(cfg_obj.user_block_size)) == 0)
        if unused_new and rng.random() < float(cfg_obj.new_recipe_obs_rate):
            accepted_new = False
            while unused_new:
                recipe_name, fn = unused_new.pop(0)
                candidates = by_recipe.get(recipe_name, [])
                material_variant_count = len({p.actions for p in candidates})
                if material_variant_count < 2:
                    new_recipe_rejections.append({
                        "phase_b_idx": i,
                        "recipe": recipe_name,
                        "reason": "fewer_than_two_material_preference_variants",
                        "material_variant_count": material_variant_count,
                    })
                    continue
                pref = users.get(current_user, "identity")
                pair = next((p for p in candidates if p.preference_name == pref), None)
                if pair is None:
                    pair = _choose_preference_zipf(candidates, cfg_obj.zipf_alpha, rng) or candidates[0]
                seen_preference_before = pair.preference_name in seen_preferences
                seen_preference_same_recipe_before = pair.preference_name in seen_preferences_by_recipe.get(pair.recipe_name, set())
                seen_preference_other_recipe_before = any(
                    pair.preference_name in vals
                    for recipe, vals in seen_preferences_by_recipe.items()
                    if recipe != pair.recipe_name
                )
                events.append(ScenarioEvent(pair, "observe", "phase_b_observe", current_user, {
                    "event_idx": len(events),
                    "phase_b_idx": i,
                    "event_type": "new_recipe_observe",
                    "recipe_rank": n_recipes + len(new_builders) - len(unused_new) - 1,
                    "user_preference": users[current_user],
                    "user_block": user_block,
                    "is_return_user_block": is_return_block,
                    "seen_recipe_before": False,
                    "seen_preference_before": seen_preference_before,
                    "seen_preference_same_recipe_before": seen_preference_same_recipe_before,
                    "seen_preference_other_recipe_before": seen_preference_other_recipe_before,
                    "seen_pair_before": False,
                    "four_cell_before": _four_cell(False, seen_preference_before),
                    "transfer_cell_before": _transfer_cell(
                        seen_recipe=False,
                        seen_pair=False,
                        seen_preference_global=seen_preference_before,
                        seen_preference_same_recipe=seen_preference_same_recipe_before,
                        seen_preference_other_recipe=seen_preference_other_recipe_before,
                    ),
                    "new_recipe_material_variant_count": material_variant_count,
                }))
                seen_recipes.add(pair.recipe_name)
                seen_preferences.add(pair.preference_name)
                seen_pairs.add(pair.label)
                seen_preferences_by_recipe[pair.recipe_name].add(pair.preference_name)
                last_by_recipe[pair.recipe_name] = pair
                last_by_user_recipe[(current_user, pair.recipe_name)] = pair
                history.append(pair)
                accepted_new = True
                break
            if accepted_new:
                continue
        recipe_idx = int(rng.choice(len(builders), p=probs))
        recipe_name, fn = builders[recipe_idx]
        candidates = by_recipe.get(recipe_name) or [materialize_pair(recipe_name, "identity", fn)]
        requested_event_type = str(rng.choice(event_types, p=weights))
        event_type = "user_switch" if block_start else requested_event_type
        base = last_by_user_recipe.get((current_user, recipe_name)) or last_by_recipe.get(recipe_name) or candidates[0]
        if event_type == "routine_reuse":
            pair = base
        elif event_type == "preference_shift":
            choices = [p for p in candidates if p.preference_name != base.preference_name]
            pair = _choose_preference_zipf(choices, cfg_obj.zipf_alpha, rng) if choices else None
            if pair is None:
                pair = base
                event_type = "routine_reuse"
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
                pair = _choose_preference_zipf(choices, cfg_obj.zipf_alpha, rng) or choices[0]
            else:
                choices = [p for p in candidates if p.preference_name != base.preference_name]
                pair = _choose_preference_zipf(choices, cfg_obj.zipf_alpha, rng) if choices else None
                if pair is None:
                    pair = base
                    event_type = "routine_reuse"
                else:
                    event_type = "preference_shift"
        elif event_type == "rare_reentry":
            horizon = int(getattr(DEFAULT_CONFIG, "decay_horizon_init", 10))
            old = history[:-horizon] if len(history) > horizon else []
            pair = old[int(rng.integers(0, len(old)))] if old else base
        else:
            pair = base
        seen_recipe_before = pair.recipe_name in seen_recipes
        seen_preference_before = pair.preference_name in seen_preferences
        seen_pair_before = pair.label in seen_pairs
        seen_preference_same_recipe_before = pair.preference_name in seen_preferences_by_recipe.get(pair.recipe_name, set())
        seen_preference_other_recipe_before = any(
            pair.preference_name in vals
            for recipe, vals in seen_preferences_by_recipe.items()
            if recipe != pair.recipe_name
        )
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
            "seen_recipe_before": seen_recipe_before,
            "seen_preference_before": seen_preference_before,
            "seen_preference_same_recipe_before": seen_preference_same_recipe_before,
            "seen_preference_other_recipe_before": seen_preference_other_recipe_before,
            "seen_pair_before": seen_pair_before,
            "four_cell_before": _four_cell(seen_recipe_before, seen_preference_before),
            "transfer_cell_before": _transfer_cell(
                seen_recipe=seen_recipe_before,
                seen_pair=seen_pair_before,
                seen_preference_global=seen_preference_before,
                seen_preference_same_recipe=seen_preference_same_recipe_before,
                seen_preference_other_recipe=seen_preference_other_recipe_before,
            ),
        }))
        seen_recipes.add(pair.recipe_name)
        seen_preferences.add(pair.preference_name)
        seen_pairs.add(pair.label)
        seen_preferences_by_recipe[pair.recipe_name].add(pair.preference_name)
        last_by_recipe[pair.recipe_name] = pair
        last_by_user_recipe[(current_user, pair.recipe_name)] = pair
        history.append(pair)
    for event in events:
        if "four_cell_before" in event.tags and "exposure_cell_before" not in event.tags:
            event.tags["exposure_cell_before"] = event.tags["four_cell_before"]
    return [r for r, _ in builders], user_ids, events, phase_a_pairs, new_recipe_rejections


def run_deployment_stream(seed: int, out: Path, run_config: RunConfig, cfg_obj: DeploymentStreamConfig) -> Dict[str, Any]:
    recipes, users, events, phase_a_pairs, new_recipe_rejections = _deployment_stream_events(seed, cfg_obj, run_config)
    _record_common_artifacts(out, seed, cfg_obj, recipes, [f"{u}:{_user_preferences(seed, cfg_obj.n_users)[u]}" for u in users])
    _append_jsonl(out / "scenario_events.jsonl", (_scenario_event_row(e) for e in events))
    if new_recipe_rejections:
        _append_jsonl(out / "new_recipe_rejections.jsonl", new_recipe_rejections)
    per_baseline: Dict[str, Dict[str, Any]] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    all_memory_rows: List[Dict[str, Any]] = []
    online_forgetting_rows: List[Dict[str, Any]] = []
    user_switch_recovery_rows: List[Dict[str, Any]] = []
    preference_reentry_recovery_rows: List[Dict[str, Any]] = []
    eval_pool = list({event.pair.label: event.pair for event in events}.values())
    for baseline in run_config.baselines:
        cfg = base_config(seed, run_config)
        agent = make_agent(baseline, cfg)
        name_to_rid: Dict[str, str] = {}
        simulator_observed_labels: set = set()
        simulator_observed_recipes: set = set()
        phase_a_eval_at_start: Optional[Dict[str, Dict[str, float]]] = None
        episode_rows: List[Dict[str, Any]] = []
        step_rows: List[Dict[str, Any]] = []
        memory_rows: List[Dict[str, Any]] = []
        baseline_online_forgetting_rows: List[Dict[str, Any]] = []
        best_top1_by_pair: Dict[str, float] = {}
        online_eval_pairs_by_label: Dict[str, RecipePrefPair] = {}
        # Compute isolated single-task performance for each Phase A pair.
        # Each pair is trained in isolation (fresh agent, one observe episode) and
        # evaluated immediately.  This provides the correct BWT baseline: the peak
        # performance the agent architecture can achieve on a task when trained only
        # on that task, free of multi-task interference.
        isolated_eval: Dict[str, Dict[str, float]] = {}
        for _iso_pair in phase_a_pairs:
            _iso_agent = make_agent(baseline, base_config(seed, run_config))
            _iso_rid: Dict[str, str] = {}
            observe_episode(_iso_agent, _iso_pair, _iso_rid)
            isolated_eval.update(frozen_eval(_iso_agent, [_iso_pair], _iso_rid, run_config))
        baseline_user_switch_recovery: List[Dict[str, Any]] = []
        baseline_preference_reentry_recovery: List[Dict[str, Any]] = []
        label_by_key: Dict[VariantKey, str] = {}
        displaced_at: Dict[VariantKey, int] = {}
        phase_b_count = 0
        t0 = time.perf_counter()
        for idx, event in enumerate(events):
            tags = {"phase": event.phase, "user_id": event.user_id, **dict(event.tags)}
            requested_mode = str(event.mode)
            mode, route_tags = _user_selected_mode_for_active_memory(requested_mode, agent, event.pair, name_to_rid)
            tags = {**tags, **route_tags}
            before_latest_by_recipe = dict(agent.decay.latest_by_recipe)
            if phase_a_eval_at_start is None and event.phase != "phase_a":
                phase_a_eval_at_start = frozen_eval(agent, eval_pool, name_to_rid)
            if mode == "observe":
                row = observe_episode(agent, event.pair, name_to_rid)
                row.update(tags)
                episode_rows.append(row)
                simulator_observed_labels.add(event.pair.label)
                simulator_observed_recipes.add(event.pair.recipe_name)
                if event.phase != "phase_a":
                    phase_b_count += 1
            else:
                user_switch_probe: Optional[Dict[str, Any]] = None
                reentry_probe: Optional[Dict[str, Any]] = None
                target_key_before = _memory_key_for_pair(agent, event.pair, name_to_rid)
                if target_key_before is not None and target_key_before in displaced_at:
                    target_rid, target_hash = target_key_before
                    pinned_hash = agent.decay.latest_by_recipe.get(target_rid)
                    if pinned_hash is not None and pinned_hash != target_hash:
                        pre_eval = frozen_eval(agent, [event.pair], name_to_rid).get(event.pair.label, {})
                        reentry_probe = {
                            "baseline": baseline,
                            "event_idx": idx,
                            "phase_b_count": int(phase_b_count),
                            "pair": event.pair.label,
                            "recipe": event.pair.recipe_name,
                            "preference": event.pair.preference_name,
                            "true_preference_name": event.pair.preference_name,
                            "true_variant_hash": target_hash,
                            "pinned_variant_hash_before": pinned_hash,
                            "subgroup_before": (
                                "displaced_active" if target_key_before in agent.decay.active
                                else "pruned" if target_key_before in agent.decay.pruned
                                else "known_in_registry"
                            ),
                            "sessions_since_displaced": max(0, int(phase_b_count) - int(displaced_at.get(target_key_before, phase_b_count))),
                            "pre_reentry_top1": float(pre_eval.get("top1", 0.0)),
                            "pre_reentry_topk": float(pre_eval.get("topk", 0.0)),
                        }
                if str(tags.get("event_type")) == "user_switch":
                    rid = name_to_rid.get(event.pair.recipe_name)
                    true_hash = variant_hash(agent._tokens_from_action_labels(event.pair.actions))
                    pinned_hash = agent.decay.latest_by_recipe.get(rid) if rid is not None else None
                    pinned_mismatch = bool(pinned_hash is not None and pinned_hash != true_hash)
                    tags["user_switch_pinned_variant_mismatch"] = pinned_mismatch
                    tags["user_switch_true_variant_hash"] = true_hash
                    tags["user_switch_pinned_variant_hash"] = pinned_hash
                    if pinned_mismatch:
                        user_switch_probe = {
                            "baseline": baseline,
                            "event_idx": idx,
                            "user_id": event.user_id,
                            "pair": event.pair.label,
                            "recipe": event.pair.recipe_name,
                            "preference": event.pair.preference_name,
                            "true_preference_name": event.pair.preference_name,
                            "true_variant_hash": true_hash,
                            "pinned_variant_hash": pinned_hash,
                        }
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
                if user_switch_probe is not None:
                    recovery_step = -1
                    for step_row in steps:
                        step_idx = int(step_row.get("step", 0))
                        if step_row.get("inferred_variant_hash") == user_switch_probe.get("true_variant_hash"):
                            recovery_step = step_idx
                            break
                    before_recovery = [
                        s for s in steps
                        if recovery_step < 0 or int(s.get("step", 0)) < recovery_step
                    ]
                    user_switch_probe.update({
                        "recovery_step": recovery_step,
                        "recovered": bool(recovery_step >= 0),
                        "robot_wrong_rate_before_recovery": _mean([
                            1.0 if _row_robot_wrong(s) else 0.0
                            for s in before_recovery
                        ]),
                        "live_top1": float(row.get("live_top1", 0.0)),
                        "classification_variant_hash": row.get("classification_variant_hash"),
                        "step_top1_trace": [
                            {
                                "step": int(s.get("step", 0)),
                                "top1_correct": bool(s.get("correct_top1")),
                                "correct_top1": bool(s.get("correct_top1")),
                                "correct_topk": bool(s.get("correct_topk")),
                                "would_assist_under_gate": bool(_row_policy_bool(s, "would_assist_under_gate")),
                                "robot_wrong": bool(_row_robot_wrong(s)),
                                "posterior_confidence": float(s.get("posterior_confidence", 0.0) or 0.0),
                                "posterior_entropy": float(s.get("posterior_entropy", 0.0) or 0.0),
                                "policy_confidence": float(_row_policy_value(s, "policy_confidence") or 0.0),
                                "policy_margin": float(_row_policy_value(s, "policy_margin") or 0.0),
                                "inferred_variant_hash": s.get("inferred_variant_hash"),
                                "inferred_latent_pref_id": s.get("inferred_latent_pref_id"),
                            }
                            for s in steps
                        ],
                    })
                    baseline_user_switch_recovery.append(user_switch_probe)
                    user_switch_recovery_rows.append(user_switch_probe)
                if reentry_probe is not None:
                    post_eval = frozen_eval(agent, [event.pair], name_to_rid).get(event.pair.label, {})
                    recovery_step = -1
                    for step_row in steps:
                        step_idx = int(step_row.get("step", 0))
                        if step_row.get("inferred_variant_hash") == reentry_probe.get("true_variant_hash"):
                            recovery_step = step_idx
                            break
                    before_recovery = [
                        s for s in steps
                        if recovery_step < 0 or int(s.get("step", 0)) < recovery_step
                    ]
                    reentry_probe.update({
                        "post_reentry_top1": float(post_eval.get("top1", 0.0)),
                        "post_reentry_topk": float(post_eval.get("topk", 0.0)),
                        "during_reentry_step_top1": _mean([1.0 if s.get("correct_top1") else 0.0 for s in steps]),
                        "during_reentry_step_topk": _mean([1.0 if s.get("correct_topk") else 0.0 for s in steps]),
                        "steps_to_recover": recovery_step,
                        "reentry_recovery_steps": recovery_step,
                        "recovered": bool(recovery_step >= 0),
                        "robot_wrong_rate_before_recovery": _mean([
                            1.0 if _row_robot_wrong(s) else 0.0
                            for s in before_recovery
                        ]),
                        "step_top1_trace": [
                            {
                                "step": int(s.get("step", 0)),
                                "top1_correct": bool(s.get("correct_top1")),
                                "would_assist_under_gate": bool(_row_policy_bool(s, "would_assist_under_gate")),
                                "robot_wrong": bool(_row_robot_wrong(s)),
                                "inferred_variant_hash": s.get("inferred_variant_hash"),
                                "inferred_latent_pref_id": s.get("inferred_latent_pref_id"),
                            }
                            for s in steps
                        ],
                    })
                    baseline_preference_reentry_recovery.append(reentry_probe)
                    preference_reentry_recovery_rows.append(reentry_probe)
            key = _memory_key_for_pair(agent, event.pair, name_to_rid)
            if key is not None:
                label_by_key[key] = event.pair.label
            for rid, old_hash in before_latest_by_recipe.items():
                new_hash = agent.decay.latest_by_recipe.get(rid)
                if old_hash is not None and old_hash != new_hash:
                    old_key = (rid, old_hash)
                    if old_key in label_by_key:
                        displaced_at.setdefault(old_key, int(phase_b_count))
            online_eval_pairs_by_label[event.pair.label] = event.pair
            online_pairs = list(online_eval_pairs_by_label.values())
            online_evals = frozen_eval(agent, online_pairs, name_to_rid, run_config) if online_pairs else {}
            online_rows = _online_best_so_far_forgetting_rows(
                agent,
                online_pairs,
                name_to_rid,
                online_evals,
                best_top1_by_pair,
                baseline=baseline,
                event_idx=idx,
                phase_b_count=phase_b_count,
                phase=str(event.phase),
                mode=str(mode),
                event_type=str(tags.get("event_type", "")),
                displaced_at=displaced_at,
            )
            if event.phase != "phase_a":
                baseline_online_forgetting_rows.extend(online_rows)
                online_forgetting_rows.extend(online_rows)
            memory_rows.append({"event_idx": idx, "baseline": baseline, **memory_snapshot(agent)})
        wall_s = time.perf_counter() - t0
        assist_rows = [r for r in episode_rows if r.get("mode") == "assist"]
        deployment_rows = list(episode_rows)
        metrics = _aggregate_episode_metrics(deployment_rows)
        metrics["assist_only"] = _aggregate_episode_metrics(assist_rows)
        if phase_a_eval_at_start is None:
            phase_a_eval_at_start = frozen_eval(agent, eval_pool, name_to_rid)
        final_frozen = frozen_eval(agent, eval_pool, name_to_rid)
        bwt_zero_shot = bwt_zero_shot_checkpoints(phase_a_eval_at_start, final_frozen, [p.label for p in phase_a_pairs], isolated_eval=isolated_eval)
        metrics["compute"] = compute_snapshot(agent, wall_s)
        metrics["memory"] = memory_snapshot(agent)
        metrics["per_user"] = {u: _aggregate_episode_metrics([r for r in deployment_rows if r.get("user_id") == u]) for u in users}
        metrics["per_user_block"] = {
            f"{u}_block_{block}": _aggregate_episode_metrics([
                r for r in deployment_rows
                if r.get("user_id") == u and int(r.get("user_block", -1)) == block
            ])
            for u in users
            for block in sorted({int(r.get("user_block", -1)) for r in deployment_rows if r.get("user_id") == u and r.get("user_block") is not None})
        }
        user_block_retention: Dict[str, Any] = {}
        for u in users:
            blocks = sorted({int(r.get("user_block", -1)) for r in assist_rows if r.get("user_id") == u and r.get("user_block") is not None})
            return_blocks = [b for b in blocks if b > min(blocks or [0])]
            if blocks and return_blocks:
                first_key = f"{u}_block_{blocks[0]}"
                first_return_key = f"{u}_block_{return_blocks[0]}"
                last_return_key = f"{u}_block_{return_blocks[-1]}"
                first_top1 = float(metrics["per_user_block"].get(first_key, {}).get("live_top1", 0.0))
                first_return_top1 = float(metrics["per_user_block"].get(first_return_key, {}).get("live_top1", 0.0))
                last_return_top1 = float(metrics["per_user_block"].get(last_return_key, {}).get("live_top1", 0.0))
                user_block_retention[u] = {
                    "first_block": blocks[0],
                    "first_return_block": return_blocks[0],
                    "last_return_block": return_blocks[-1],
                    "return_block": return_blocks[-1],
                    "first_block_live_top1": first_top1,
                    "first_return_live_top1": first_return_top1,
                    "last_return_live_top1": last_return_top1,
                    "return_block_live_top1": last_return_top1,
                    "first_return_minus_first_live_top1": first_return_top1 - first_top1,
                    "last_return_minus_first_live_top1": last_return_top1 - first_top1,
                    "return_minus_first_live_top1": last_return_top1 - first_top1,
                }
        metrics["per_user_return_retention"] = user_block_retention
        metrics["per_event_type"] = {k: _aggregate_episode_metrics([r for r in deployment_rows if r.get("event_type") == k]) for k in sorted({str(r.get("event_type")) for r in deployment_rows})}
        metrics["per_event_type_support"] = {k: float(v.get("n_episodes", 0.0)) for k, v in metrics["per_event_type"].items()}
        metrics["macro_event_top1"] = _mean([
            float(row.get("live_top1", 0.0))
            for event_type, row in metrics["per_event_type"].items()
            if event_type in PRIMARY_EVENT_TYPES and float(row.get("n_episodes", 0.0)) > 0.0
        ])
        metrics["calibration_curve"] = calibration_curve_from_steps(step_rows)
        # Calibration by event type surfaces whether the confidence signal is
        # trustworthy specifically on hard events (preference_shift, rare_reentry),
        # which is the operationally important range for the action gate.
        metrics["calibration_curve_by_event_type"] = {
            et: calibration_curve_from_steps([s for s in step_rows if s.get("event_type") == et])
            for et in PRIMARY_EVENT_TYPES
            if any(s.get("event_type") == et for s in step_rows)
        }
        hard_cases = {"preference_shift", "user_switch", "rare_reentry", "cross_transfer_probe"}
        metrics["hard_case_top1"] = _mean([
            float(r.get("live_top1", 0.0))
            for r in deployment_rows
            if str(r.get("event_type")) in hard_cases
        ])
        metrics["hard_case_net_robot_action_value"] = _mean([
            float(r.get("net_robot_action_value", r.get("net_assistance_value", 0.0)))
            for r in deployment_rows
            if str(r.get("event_type")) in hard_cases
        ])
        metrics["per_memory_state_gt"] = {
            k: _aggregate_episode_metrics([
                r for r in deployment_rows
                if str(r.get("memory_state_gt", r.get("memory_state", "unknown"))) == k
            ])
            for k in sorted({str(r.get("memory_state_gt", r.get("memory_state", "unknown"))) for r in deployment_rows})
        }
        metrics["per_event_type_memory_state_gt"] = {
            event_type: {
                state: _aggregate_episode_metrics([
                    r for r in deployment_rows
                    if str(r.get("event_type")) == event_type
                    and str(r.get("memory_state_gt", r.get("memory_state", "unknown"))) == state
                ])
                for state in sorted({str(r.get("memory_state_gt", r.get("memory_state", "unknown"))) for r in deployment_rows if str(r.get("event_type")) == event_type})
            }
            for event_type in sorted({str(r.get("event_type")) for r in deployment_rows})
        }
        metrics["per_transfer_cell"] = {
            cell: _aggregate_episode_metrics([r for r in deployment_rows if r.get("transfer_cell_before") == cell])
            for cell in TRANSFER_CELL_KEYS
        }
        exposure_cell_metrics = {
            cell: _aggregate_episode_metrics([
                r for r in deployment_rows
                if r.get("exposure_cell_before", r.get("four_cell_before")) == cell
            ])
            for cell in FOUR_CELL_KEYS
        }
        metrics["coarse_exposure_cell"] = exposure_cell_metrics
        metrics["four_cell"] = exposure_cell_metrics
        for cell, vals in exposure_cell_metrics.items():
            for metric_name, metric_val in vals.items():
                if isinstance(metric_val, (int, float)):
                    metrics[f"{metric_name}_{cell}"] = float(metric_val)
        metrics["valid_random_floor"] = _mean([
            1.0 / max(1, int(s.get("valid_next_action_count", 1)))
            for s in step_rows
        ])
        metrics["live_step_trace"] = live_step_trace(step_rows)
        event_trace_types = tuple(dict.fromkeys((*PRIMARY_EVENT_TYPES, "user_switch")))
        metrics["live_step_trace_by_event_type"] = {
            event_type: live_step_trace([s for s in step_rows if str(s.get("event_type")) == event_type])
            for event_type in event_trace_types
        }
        prediction_walls = [float(s.get("prediction_wall_s")) for s in step_rows if s.get("prediction_wall_s") is not None]
        metrics["mean_prediction_wall_s"] = _mean(prediction_walls)
        metrics["p95_prediction_wall_s"] = _p95(prediction_walls)
        metrics["calibration_curve"] = calibration_curve_from_steps(step_rows)
        metrics["latency_cdf"] = latency_cdf_from_steps(step_rows)
        metrics["policy_gate"] = policy_gate_reason_summary(step_rows)
        metrics["selected_policy_confidence_threshold"] = float(agent.cfg.posterior_action_confidence_threshold)
        metrics["bwt"] = bwt_zero_shot["bwt"]
        metrics["zero_shot_transfer"] = bwt_zero_shot["zero_shot_transfer"]
        metrics["bwt_zero_shot_detail"] = bwt_zero_shot
        metrics["primary_forgetting_metric"] = "online_best_so_far_forgetting"
        metrics["online_best_so_far_forgetting"] = _summarize_online_forgetting_rows(baseline_online_forgetting_rows)
        metrics["preference_reentry_recovery"] = {
            "rows": baseline_preference_reentry_recovery,
            "summary": _summarize_preference_reentry_rows(baseline_preference_reentry_recovery),
        }
        metrics["user_switch_recovery"] = {
            "rows": baseline_user_switch_recovery,
            "n_qualifying_episodes": len(baseline_user_switch_recovery),
            "user_mismatch_recovery_steps": _mean([float(r.get("recovery_step", -1)) for r in baseline_user_switch_recovery if int(r.get("recovery_step", -1)) >= 0]),
            "recovery_failure_rate": _mean([0.0 if r.get("recovered") else 1.0 for r in baseline_user_switch_recovery]),
            "robot_wrong_rate_before_recovery": _mean([float(r.get("robot_wrong_rate_before_recovery", r.get("wrong_assist_rate_before_recovery", 0.0))) for r in baseline_user_switch_recovery]),
            "step_trace": _step_trace_from_rows([
                step
                for row in baseline_user_switch_recovery
                for step in (row.get("step_top1_trace", []) or [])
                if isinstance(step, Mapping)
            ]),
        }
        per_baseline[baseline] = metrics
        all_episode_rows.extend({"baseline": baseline, **r} for r in episode_rows)
        all_step_rows.extend(step_rows)
        all_memory_rows.extend(memory_rows)
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    _append_jsonl(out / "memory_trace.jsonl", all_memory_rows)
    _append_jsonl(out / "online_forgetting_trace.jsonl", online_forgetting_rows)
    _append_jsonl(out / "user_switch_recovery.jsonl", user_switch_recovery_rows)
    _append_jsonl(out / "preference_reentry_recovery.jsonl", preference_reentry_recovery_rows)
    oracle_eval_pairs = [e.pair for e in events if e.mode == "assist" and e.phase != "phase_a"]
    oracle_references = _oracle_reference_metrics(
        seed,
        run_config,
        phase_a_pairs,
        oracle_eval_pairs,
        max_eval_pairs=None,
    )
    preference_decay_profile = _preference_decay_profile(seed, run_config, cfg_obj)
    valid_random_floor = float((per_baseline.get("full", {}) or {}).get("valid_random_floor", 0.0))
    transfer_taxonomy_policy = {
        "transfer_claim_cells": list(TRANSFER_CELL_KEYS),
        "forgetting_and_reentry_memory_states": ["latest_pin", "displaced_active", "pruned", "known_in_registry"],
        "exposure_cells": list(FOUR_CELL_KEYS),
        "four_cell_is_secondary_taxonomy": True,
    }
    metrics = {
        "seed": seed,
        "mode_policy": "user selects observation when the recipe is absent from the baseline's active memory; otherwise the user selects assist",
        "n_recipes": len(recipes),
        "n_users": len(users),
        "n_events": len(events),
        "n_phase_b_events": sum(1 for e in events if e.mode == "assist"),
        "n_new_recipe_rejections": len(new_recipe_rejections),
        "new_recipe_rejections": new_recipe_rejections,
        "event_distribution": _event_distribution(events),
        "event_mix_sensitivity": _event_mix_sensitivity(seed, run_config, cfg_obj),
        "oracle_references": oracle_references,
        "valid_random_floor": valid_random_floor,
        "preference_decay_profile": preference_decay_profile,
        "primary_forgetting_metric": "online_best_so_far_forgetting",
        "online_best_so_far_forgetting_summary": {
            baseline: row.get("online_best_so_far_forgetting", {})
            for baseline, row in per_baseline.items()
        },
        "transfer_taxonomy_policy": transfer_taxonomy_policy,
        "selected_policy_confidence_threshold": float((per_baseline.get("full", {}) or {}).get("selected_policy_confidence_threshold", per_baseline.get("full", {}).get("selected_action_confidence_threshold", base_config(seed, run_config).posterior_action_confidence_threshold))),
        "selected_policy_confidence_thresholds": {b: row.get("selected_policy_confidence_threshold", row.get("selected_action_confidence_threshold", base_config(seed, run_config).posterior_action_confidence_threshold)) for b, row in per_baseline.items()},
        "per_baseline": per_baseline,
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


def _default_axis_values() -> Dict[str, str]:
    return WorkflowPreference().as_dict()


def _single_axis_preference(axis: str, value: str) -> WorkflowPreference:
    axes = _default_axis_values()
    axes[axis] = value
    return WorkflowPreference(**axes)


def _non_default_axis_values(axis: str) -> Tuple[str, ...]:
    default = _default_axis_values()[axis]
    return tuple(v for v in WORKFLOW_AXIS_VALUES.get(axis, ()) if v != default)


def _material_single_axis_pair(
    recipe: str,
    fn: Callable[[], List[str]],
    axis: str,
    value: str,
) -> Optional[RecipePrefPair]:
    try:
        pair = materialize_custom_pair(recipe, _single_axis_preference(axis, value), fn)
    except Exception:
        return None
    if axis in pair.failed_axes:
        return None
    if axis not in pair.applied_axes:
        return None
    return pair


def _axis_materiality_report(builders: Sequence[Tuple[str, Callable[[], List[str]]]]) -> Dict[str, Dict[str, int]]:
    report: Dict[str, Dict[str, int]] = {}
    for axis in WORKFLOW_AXES:
        attempted = valid = failed = noop = 0
        for recipe, fn in builders:
            for value in _non_default_axis_values(axis):
                attempted += 1
                try:
                    pair = materialize_custom_pair(recipe, _single_axis_preference(axis, value), fn)
                except Exception:
                    failed += 1
                    continue
                if axis in pair.failed_axes:
                    failed += 1
                elif axis not in pair.applied_axes:
                    noop += 1
                else:
                    valid += 1
        report[axis] = {
            "pairs_attempted": attempted,
            "valid_material_pairs": valid,
            "excluded_target_axis_failed": failed,
            "excluded_target_axis_noop": noop,
        }
    return report


def _cross_axis_value_transfer_specs(
    builders: Sequence[Tuple[str, Callable[[], List[str]]]],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if len(builders) < 2:
        return specs
    split = max(1, len(builders) // 2)
    train_builders = list(builders[:split])
    test_builders = list(builders[split:]) or list(builders[:split])
    for axis in WORKFLOW_AXES:
        for value in _non_default_axis_values(axis):
            train_pairs = [
                pair
                for recipe, fn in train_builders
                for pair in [_material_single_axis_pair(recipe, fn, axis, value)]
                if pair is not None
            ]
            test_pairs = [
                pair
                for recipe, fn in test_builders
                for pair in [_material_single_axis_pair(recipe, fn, axis, value)]
                if pair is not None
            ]
            if train_pairs and test_pairs:
                specs.append({
                    "condition": "cross_axis_value_transfer",
                    "axis": axis,
                    "axis_value": value,
                    "train_pairs": train_pairs,
                    "test_pairs": test_pairs,
                })
    return specs


def _preset_excludes_axis(pref: WorkflowPreference, axis: str) -> bool:
    return pref.as_dict()[axis] == _default_axis_values()[axis]


def _preset_varies_other_axis(pref: WorkflowPreference, heldout_axis: str) -> bool:
    defaults = _default_axis_values()
    return any(
        axis != heldout_axis and pref.as_dict()[axis] != defaults[axis]
        for axis in WORKFLOW_AXES
    )


def _axis_family_holdout_specs(
    builders: Sequence[Tuple[str, Callable[[], List[str]]]],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if not builders:
        return specs
    split = max(1, int(math.ceil(len(builders) * 0.67)))
    train_builders = list(builders[:split])
    test_builders = list(builders[split:]) or list(builders)
    for axis in WORKFLOW_AXES:
        train_pref_names = [
            name
            for name, pref in PRESET_PREFERENCES.items()
            if _preset_excludes_axis(pref, axis) and (name == "identity" or _preset_varies_other_axis(pref, axis))
        ]
        if not train_pref_names:
            train_pref_names = ["identity"]
        train_pairs: List[RecipePrefPair] = []
        for i, (recipe, fn) in enumerate(train_builders):
            pref_name = train_pref_names[i % len(train_pref_names)]
            try:
                pair = materialize_pair(recipe, pref_name, fn)
            except Exception:
                continue
            train_pairs.append(pair)
        by_value: Dict[str, List[RecipePrefPair]] = {}
        test_pairs: List[RecipePrefPair] = []
        for value in _non_default_axis_values(axis):
            value_pairs = [
                pair
                for recipe, fn in test_builders
                for pair in [_material_single_axis_pair(recipe, fn, axis, value)]
                if pair is not None
            ]
            if value_pairs:
                by_value[value] = value_pairs
                test_pairs.extend(value_pairs)
        if train_pairs and test_pairs:
            specs.append({
                "condition": "axis_family_holdout",
                "axis": axis,
                "train_preference_names": tuple(train_pref_names),
                "heldout_axis_values": tuple(by_value.keys()),
                "train_pairs": train_pairs,
                "test_pairs": test_pairs,
                "test_pairs_by_value": by_value,
            })
    return specs


def _preference_axis_difference_count(pref: WorkflowPreference, reference: Optional[WorkflowPreference] = None) -> int:
    reference_axes = (reference or WorkflowPreference()).as_dict()
    pref_axes = pref.as_dict()
    return sum(1 for axis in WORKFLOW_AXES if pref_axes.get(axis) != reference_axes.get(axis))


def _novel_composition_pairs(builders: Sequence[Tuple[str, Callable[[], List[str]]]], named_pairs: Sequence[RecipePrefPair]) -> List[Tuple[RecipePrefPair, int]]:
    train_hashes = {_order_hash(p) for p in named_pairs}
    rows: List[Tuple[RecipePrefPair, int]] = []
    for recipe, fn in builders:
        for pref in _build_gap_preferences():
            try:
                pair = materialize_custom_pair(recipe, pref, fn)
            except Exception:
                continue
            if _order_hash(pair) in train_hashes:
                continue
            axis_diff_count = _preference_axis_difference_count(pref)
            rows.append((pair, axis_diff_count))
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
        pair_vals = list(by_pref.values())
        effective_pairs, duplicate_pairs = _deduplicate_preference_pairs(
            pair_vals,
            tau_threshold=float(MaterialityPreflightConfig().near_duplicate_tau_threshold),
        )
        if len(effective_pairs) < len(pref_list) or duplicate_pairs:
            continue
        divergences = [
            earliest_divergence_step(a.actions, b.actions)
            for i, a in enumerate(pair_vals)
            for b in pair_vals[i + 1:]
            if a.actions != b.actions
        ]
        min_divergence = min(divergences, default=0)
        min_len = min((len(p.actions) for p in pair_vals), default=1)
        if min_divergence > max(1, min_len // 3):
            continue
        identity_pair = by_pref.get("identity")
        if identity_pair is not None:
            too_close_to_selected = False
            for _sel_name, _sel_fn, sel_by_pref in selected:
                sel_identity = sel_by_pref.get("identity")
                if sel_identity is not None and jaccard(identity_pair.actions, sel_identity.actions) >= float(DEFAULT_CONFIG.jaccard_threshold):
                    too_close_to_selected = True
                    break
            if too_close_to_selected:
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


def _run_transfer_condition_probe(
    baseline: str,
    seed: int,
    run_config: RunConfig,
    condition: str,
    train_pairs: Sequence[RecipePrefPair],
    test_pairs: Sequence[RecipePrefPair],
    event_tag: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    agent = make_agent(baseline, base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    pref_to_pid: Dict[str, Optional[str]] = {}
    for pair in train_pairs:
        observe_episode(agent, pair, name_to_rid)
        pref_to_pid[pair.preference_name] = agent.last_pref_id
    checkpoint = agent.snapshot()
    frozen = frozen_eval(agent, test_pairs, name_to_rid) if test_pairs else {}
    rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    tag = {"condition": condition, **(event_tag or {})}
    for pair in test_pairs:
        agent.restore_from(checkpoint)
        try:
            local_map = dict(name_to_rid)
            oracle_pid = pref_to_pid.get(pair.preference_name)
            oracle_rid = local_map.get(pair.recipe_name)
            kwargs = {"oracle_preference_id": oracle_pid} if baseline == "oracle_preference_label" else {}
            if baseline == "oracle_recipe_and_preference_label":
                kwargs = {"oracle_preference_id": oracle_pid, "oracle_recipe_id": oracle_rid}
            row, steps = assist_episode(
                agent,
                pair,
                local_map,
                run_config=run_config,
                commit=False,
                event_tag=tag,
                **kwargs,
            )
        finally:
            agent.restore_from(checkpoint)
        row.update({"baseline": baseline, **tag})
        rows.append(row)
        step_rows.extend({"baseline": baseline, **tag, **s} for s in steps)
    live = _aggregate_episode_metrics(rows)
    frozen_top1 = _mean([v.get("top1", 0.0) for v in frozen.values()])
    return {
        **live,
        "condition": condition,
        "n_train_pairs": float(len(train_pairs)),
        "n_test_pairs": float(len(test_pairs)),
        "train_pairs": [p.label for p in train_pairs],
        "test_pairs": [p.label for p in test_pairs],
        "frozen_top1": frozen_top1,
        "frozen_to_live_transfer_gain": float(live.get("live_top1", 0.0)) - frozen_top1,
    }, rows, step_rows


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
    effective_preferences_by_recipe: Dict[str, int] = {}
    duplicate_orderings_by_recipe: Dict[str, List[Dict[str, Any]]] = {}
    for recipe in recipes:
        recipe_pairs = [p for p in all_pairs if p.recipe_name == recipe]
        deduped, dropped = _deduplicate_preference_pairs(
            recipe_pairs,
            tau_threshold=float(MaterialityPreflightConfig().near_duplicate_tau_threshold),
        )
        effective_preferences_by_recipe[recipe] = len(deduped)
        duplicate_orderings_by_recipe[recipe] = dropped
    repeats = max(1, int(cfg_obj.offdiag_repeats))
    axis_value_specs = _cross_axis_value_transfer_specs(builders) if cfg_obj.include_cross_axis_value_transfer else []
    axis_family_specs = _axis_family_holdout_specs(builders) if cfg_obj.include_axis_family_holdout else []
    axis_materiality = _axis_materiality_report(builders) if (axis_value_specs or axis_family_specs) else {}
    novel_probes = _novel_composition_pairs(builders, all_pairs) if cfg_obj.include_novel_composition else []
    _record_common_artifacts(out, seed, cfg_obj, recipes, active_prefs)
    scenario_rows = [
        {"mode": "observe", "phase": "shared_diagonal_train", "event_type": "diagonal_train", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for p in diagonal
    ] + [
        {"mode": "assist", "phase": "cross_recipe_pref_transfer", "event_type": "offdiagonal_transfer", "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for p in offdiag
    ] + [
        {"mode": "observe", "phase": "cross_axis_value_transfer_train", "event_type": "axis_value_train", "axis": spec["axis"], "axis_value": spec["axis_value"], "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for spec in axis_value_specs
        for p in spec["train_pairs"]
    ] + [
        {"mode": "assist", "phase": "cross_axis_value_transfer_probe", "event_type": "axis_value_transfer_probe", "axis": spec["axis"], "axis_value": spec["axis_value"], "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for spec in axis_value_specs
        for p in spec["test_pairs"]
    ] + [
        {"mode": "observe", "phase": "axis_family_holdout_train", "event_type": "axis_family_train", "heldout_axis": spec["axis"], "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for spec in axis_family_specs
        for p in spec["train_pairs"]
    ] + [
        {"mode": "assist", "phase": "axis_family_holdout_probe", "event_type": "axis_family_holdout_probe", "heldout_axis": spec["axis"], "pair": p.label, "recipe": p.recipe_name, "preference": p.preference_name}
        for spec in axis_family_specs
        for p in spec["test_pairs"]
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
        interference_rows: List[Dict[str, Any]] = []
        seen_train_pairs: List[RecipePrefPair] = []
        t0 = time.perf_counter()
        for _cycle in range(max(1, int(cfg_obj.diagonal_cycles))):
            for pair in diagonal:
                before_seen = frozen_eval(agent, seen_train_pairs, name_to_rid) if seen_train_pairs else {}
                observe_episode(agent, pair, name_to_rid)
                pref_to_pid[pair.preference_name] = agent.last_pref_id
                after_seen = frozen_eval(agent, seen_train_pairs, name_to_rid) if seen_train_pairs else {}
                for prior in seen_train_pairs:
                    key = _memory_key_for_pair(agent, prior, name_to_rid)
                    active = bool(key is not None and key in agent.decay.active)
                    latest = bool(key is not None and agent.decay.latest_by_recipe.get(key[0]) == key[1])
                    pruned = bool(key is not None and key in agent.decay.pruned)
                    same_axis = bool(set(prior.applied_axes) & set(pair.applied_axes))
                    interference_rows.append({
                        "added_pair": pair.label,
                        "evaluated_pair": prior.label,
                        "added_recipe": pair.recipe_name,
                        "evaluated_recipe": prior.recipe_name,
                        "same_preference_axis": same_axis,
                        "same_recipe": pair.recipe_name == prior.recipe_name,
                        "variant_is_active": active,
                        "variant_is_pruned": pruned,
                        "variant_is_latest_pin": latest,
                        "before_top1": float((before_seen.get(prior.label, {}) or {}).get("top1", 0.0)),
                        "after_top1": float((after_seen.get(prior.label, {}) or {}).get("top1", 0.0)),
                        "delta_top1": float((after_seen.get(prior.label, {}) or {}).get("top1", 0.0)) - float((before_seen.get(prior.label, {}) or {}).get("top1", 0.0)),
                    })
                if pair not in seen_train_pairs:
                    seen_train_pairs.append(pair)
        train_seen_recipes = {p.recipe_name for p in diagonal}
        train_seen_pairs = {p.label for p in diagonal}
        train_seen_preferences = {p.preference_name for p in diagonal}
        train_seen_preferences_by_recipe: Dict[str, set] = defaultdict(set)
        for p in diagonal:
            train_seen_preferences_by_recipe[p.recipe_name].add(p.preference_name)
        diagonal_weights_at_checkpoint = {
            p.label: variant_weight_for_pair(agent, p, name_to_rid)
            for p in diagonal
        }
        diagonal_latest_pin_integrity = all(
            weight is not None and abs(float(weight) - 1.0) <= 1e-9
            for weight in diagonal_weights_at_checkpoint.values()
        )
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
                        "frozen_diagonal_topk": _mean([v.get("topk", 0.0) for v in cycle_eval.values()]),
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
                try:
                    local_map = dict(name_to_rid)
                    oracle_pid = pref_to_pid.get(pair.preference_name)
                    oracle_rid = local_map.get(pair.recipe_name)
                    kwargs = {"oracle_preference_id": oracle_pid} if baseline == "oracle_preference_label" else {}
                    if baseline == "oracle_recipe_and_preference_label":
                        kwargs = {"oracle_preference_id": oracle_pid, "oracle_recipe_id": oracle_rid}
                    transfer_cell = _transfer_cell_for_pair(
                        pair,
                        seen_recipes=train_seen_recipes,
                        seen_pairs=train_seen_pairs,
                        seen_preferences=train_seen_preferences,
                        seen_preferences_by_recipe=train_seen_preferences_by_recipe,
                    )
                    eval_cls = agent.disambig.classify(
                        agent._tokens_from_action_labels(pair.actions),
                        agent.memory.library(allowed_keys=agent._active_keys()),
                    )
                    row, steps = assist_episode(
                        agent,
                        pair,
                        local_map,
                        run_config=run_config,
                        commit=False,
                        event_tag={"condition": "cross_recipe_pref_transfer", "repeat": rep, "transfer_cell": transfer_cell},
                        **kwargs,
                    )
                finally:
                    agent.restore_from(checkpoint)
                row.update({
                    "baseline": baseline,
                    "seen_recipe": True,
                    "seen_preference_global": pair.preference_name in pref_to_pid,
                    "seen_preference_for_recipe": False,
                    "transfer_cell": transfer_cell,
                    "classification_kind_eval": eval_cls.kind,
                    "preference_transfer_correct": eval_cls.kind in ("known", "preference_shift", "reentry_from_pruned"),
                    "transfer_evaluation_commit": False,
                })
                pair_rows.append(row)
                off_rows.append(row)
                off_rows_by_label[pair.label].append(row)
                if isinstance(row.get("live_record"), LiveEpisodeRecord):
                    live_records.append(row["live_record"])
                all_episode_rows.append(row)
                all_step_rows.extend({"baseline": baseline, "condition": "cross_recipe_pref_transfer", **s} for s in steps)
            if baseline == "full":
                full_matrix[(pair.recipe_name, pair.preference_name)] = _mean([r["live_top1"] for r in pair_rows])
        axis_value_metrics: Dict[str, Any] = {}
        axis_value_rows_all: List[Dict[str, Any]] = []
        for spec in axis_value_specs:
            spec_key = f"{spec['axis']}={spec['axis_value']}"
            result, rows, steps = _run_transfer_condition_probe(
                baseline,
                seed,
                run_config,
                "cross_axis_value_transfer",
                spec["train_pairs"],
                spec["test_pairs"],
                {"axis": spec["axis"], "axis_value": spec["axis_value"]},
            )
            axis_value_metrics[spec_key] = result
            axis_value_rows_all.extend(rows)
            all_episode_rows.extend(rows)
            all_step_rows.extend(steps)
        cross_axis_value_transfer = {
            **_aggregate_episode_metrics(axis_value_rows_all),
            "condition": "cross_axis_value_transfer",
            "by_axis_value": axis_value_metrics,
            "n_specs": float(len(axis_value_specs)),
        }
        axis_family_metrics: Dict[str, Any] = {}
        axis_family_rows_all: List[Dict[str, Any]] = []
        for spec in axis_family_specs:
            result, rows, steps = _run_transfer_condition_probe(
                baseline,
                seed,
                run_config,
                "axis_family_holdout",
                spec["train_pairs"],
                spec["test_pairs"],
                {"heldout_axis": spec["axis"]},
            )
            by_value: Dict[str, Any] = {}
            for value, pairs_for_value in spec.get("test_pairs_by_value", {}).items():
                value_labels = {p.label for p in pairs_for_value}
                by_value[value] = _aggregate_episode_metrics([r for r in rows if r.get("pair") in value_labels])
            result["by_heldout_value"] = by_value
            result["train_preference_names"] = list(spec.get("train_preference_names", ()))
            axis_family_metrics[spec["axis"]] = result
            axis_family_rows_all.extend(rows)
            all_episode_rows.extend(rows)
            all_step_rows.extend(steps)
        axis_family_holdout = {
            **_aggregate_episode_metrics(axis_family_rows_all),
            "condition": "axis_family_holdout",
            "by_axis": axis_family_metrics,
            "n_specs": float(len(axis_family_specs)),
        }
        axis_diff_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        frozen_novel = frozen_eval(agent, [p for p, _h in novel_probes], name_to_rid) if novel_probes else {}
        for pair, axis_diff_count in novel_probes:
            agent.restore_from(checkpoint)
            try:
                row, steps = assist_episode(agent, pair, dict(name_to_rid), run_config=run_config, commit=False, event_tag={"condition": "novel_composition", "axis_diff_count": axis_diff_count})
            finally:
                agent.restore_from(checkpoint)
            axis_diff_rows[str(axis_diff_count)].append(row)
            all_episode_rows.append({"baseline": baseline, **row})
            all_step_rows.extend({"baseline": baseline, "condition": "novel_composition", "axis_diff_count": axis_diff_count, **s} for s in steps)
        novel_metrics = {}
        for h, rows in axis_diff_rows.items():
            live_h = _aggregate_episode_metrics(rows)
            frozen_h = _mean([frozen_novel.get(r.get("pair"), {}).get("top1", 0.0) for r in rows])
            novel_metrics[h] = {
                **live_h,
                "frozen_top1": frozen_h,
                "frozen_to_live_transfer_gain": float(live_h.get("live_top1", 0.0)) - frozen_h,
            }
        wall_s = time.perf_counter() - t0
        live_offdiag = _aggregate_episode_metrics(off_rows)
        frozen_offdiag_top1 = _mean([v.get("top1", 0.0) for v in frozen_off.values()])
        per_transfer_cell = {cell: _aggregate_episode_metrics([r for r in off_rows if r.get("transfer_cell") == cell]) for cell in TRANSFER_CELL_KEYS}
        cross_recipe_pref_transfer = {
            **live_offdiag,
            "condition": "cross_recipe_pref_transfer",
            "frozen_top1": frozen_offdiag_top1,
            "frozen_to_live_transfer_gain": float(live_offdiag.get("live_top1", 0.0)) - frozen_offdiag_top1,
            "per_transfer_cell": per_transfer_cell,
            "train_pairs": [p.label for p in diagonal],
            "test_pairs": [p.label for p in offdiag],
        }
        live_records_by_label = {
            label: {"top1": _mean([float(r.get("live_top1", 0.0)) for r in rows])}
            for label, rows in off_rows_by_label.items()
        }
        frozen_records_by_label = {
            label: {"top1": float(row.get("top1", 0.0))}
            for label, row in frozen_off.items()
        }
        preset_of_label = {p.label: p.preference_name for p in all_pairs}
        pref_cluster = preference_cluster_metrics(live_records, preset_of_label)
        slot_matrix: List[List[Optional[float]]] = []
        slot_support: List[List[int]] = []
        slot_cell_text: List[List[str]] = []
        for recipe in recipes:
            matrix_row: List[Optional[float]] = []
            support_row: List[int] = []
            text_row: List[str] = []
            for pref in active_prefs:
                label = f"{recipe}/{pref}"
                if diag_pref_by_recipe.get(recipe) == pref:
                    matrix_row.append(None)
                    support_row.append(0)
                    text_row.append("train")
                elif label in off_rows_by_label:
                    rows_for_label = off_rows_by_label[label]
                    matrix_row.append(_mean([float(r.get("live_top1", 0.0)) for r in rows_for_label]))
                    support_row.append(len(rows_for_label))
                    text_row.append("")
                else:
                    matrix_row.append(None)
                    support_row.append(0)
                    text_row.append("n=0")
            slot_matrix.append(matrix_row)
            slot_support.append(support_row)
            slot_cell_text.append(text_row)
        per_baseline[baseline] = {
            **live_offdiag,
            "frozen_diagonal_top1": _mean([v.get("top1", 0.0) for v in frozen_diag.values()]),
            "frozen_offdiagonal_top1": frozen_offdiag_top1,
            "frozen_transfer_top1": frozen_offdiag_top1,
            "live_online_transfer_top1": float(live_offdiag.get("live_top1", 0.0)),
            "frozen_to_live_transfer_gain": float(live_offdiag.get("live_top1", 0.0)) - frozen_offdiag_top1,
            "inference_time_adaptation_gain": float(live_offdiag.get("live_top1", 0.0)) - frozen_offdiag_top1,
            "prototype_level_transfer_top1": frozen_offdiag_top1,
            "preference_gate_accuracy": _mean([1.0 if r.get("preference_transfer_correct") else 0.0 for r in off_rows]),
            "preference_transfer_rate": _mean([1.0 if r.get("preference_transfer_correct") else 0.0 for r in off_rows]),
            "per_transfer_cell": per_transfer_cell,
            **{f"live_top1_{cell}": float(per_transfer_cell[cell].get("live_top1", 0.0)) for cell in TRANSFER_CELL_KEYS},
            "primary_transfer_live_top1": float(per_transfer_cell["seen_recipe_preference_seen_elsewhere"].get("live_top1", 0.0)),
            "seen_recipe_new_preference_live_top1": float(per_transfer_cell["seen_recipe_new_preference"].get("live_top1", 0.0)),
            "diagonal_cycle_curve": diagonal_cycle_curve,
            "diagonal_weights_at_checkpoint": diagonal_weights_at_checkpoint,
            "diagonal_latest_pin_integrity": bool(diagonal_latest_pin_integrity),
            "diagonal_min_weight_at_checkpoint": min((float(w) for w in diagonal_weights_at_checkpoint.values() if w is not None), default=0.0),
            "cross_recipe_interference": {
                "rows": interference_rows,
                "mean_delta_top1": _mean([float(r.get("delta_top1", 0.0)) for r in interference_rows]),
                "same_axis_mean_delta_top1": _mean([float(r.get("delta_top1", 0.0)) for r in interference_rows if r.get("same_preference_axis")]),
                "different_axis_mean_delta_top1": _mean([float(r.get("delta_top1", 0.0)) for r in interference_rows if not r.get("same_preference_axis")]),
                "active_mean_delta_top1": _mean([float(r.get("delta_top1", 0.0)) for r in interference_rows if r.get("variant_is_active")]),
                "pruned_mean_delta_top1": _mean([float(r.get("delta_top1", 0.0)) for r in interference_rows if r.get("variant_is_pruned")]),
                "latest_pinned_mean_delta_top1": _mean([float(r.get("delta_top1", 0.0)) for r in interference_rows if r.get("variant_is_latest_pin")]),
            },
            **pref_cluster,
            "preference_cluster_metrics": pref_cluster,
            "preference_axis_top1": preference_axis_top1(frozen_records_by_label, offdiag),
            "per_recipe_accuracy_matrix": per_recipe_accuracy_matrix(frozen_records_by_label, offdiag),
            "live_per_recipe_accuracy_matrix": per_recipe_accuracy_matrix(live_records_by_label, offdiag),
            "transfer_slot_heatmap": {
                "x_slots": [f"P{i + 1}" for i in range(len(active_prefs))],
                "y_slots": [f"R{i + 1}" for i in range(len(recipes))],
                "matrix": slot_matrix,
                "support_matrix": slot_support,
                "cell_text": slot_cell_text,
                "diagonal_label": "train",
                "zero_support_label": "n=0",
            },
            "conditions": {
                "cross_recipe_pref_transfer": cross_recipe_pref_transfer,
                "cross_axis_value_transfer": cross_axis_value_transfer,
                "axis_family_holdout": axis_family_holdout,
            },
            "cross_recipe_pref_transfer": cross_recipe_pref_transfer,
            "cross_axis_value_transfer": cross_axis_value_transfer,
            "axis_family_holdout": axis_family_holdout,
            "novel_composition": novel_metrics,
            "compute": compute_snapshot(agent, wall_s),
            "memory": memory_snapshot(agent),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    transfer_slot_heatmap = {
        "x_slots": [f"P{i + 1}" for i in range(len(active_prefs))],
        "y_slots": [f"R{i + 1}" for i in range(len(recipes))],
        "recipe_names_by_slot": {f"R{i + 1}": recipe for i, recipe in enumerate(recipes)},
        "preference_names_by_slot": {f"P{i + 1}": pref for i, pref in enumerate(active_prefs)},
        "diagonal_label": "train",
        "zero_support_label": "n=0",
        "per_baseline": {
            b: row.get("transfer_slot_heatmap", {})
            for b, row in per_baseline.items()
        },
        "seed_local_slot_taxonomy": "Slots R1..Rn/P1..Pn are deterministic within seed; aggregate reports must not merge literal recipe names across seeds.",
    }
    # Difference heatmap: full agent gain over the no-preference-prototype ablation.
    # Positive cells (shown in the paper as the primary figure) directly prove that
    # preference prototypes contribute to transfer.  Cells near zero indicate the
    # transfer is explained by recipe-level generalisation alone.
    _full_mat = (per_baseline.get("full", {}) or {}).get("transfer_slot_heatmap", {}).get("matrix", [])
    _nop_mat  = (per_baseline.get("no_preference_prototype", {}) or {}).get("transfer_slot_heatmap", {}).get("matrix", [])
    if _full_mat and _nop_mat and len(_full_mat) == len(_nop_mat):
        transfer_slot_heatmap["prototype_transfer_gain_matrix"] = [
            [
                (float(_full_mat[i][j]) - float(_nop_mat[i][j]))
                if _full_mat[i][j] is not None and _nop_mat[i][j] is not None
                else None
                for j in range(len(_full_mat[i]))
            ]
            for i in range(len(_full_mat))
        ]
    transfer_taxonomy_policy = {
        "transfer_claim_cells": list(TRANSFER_CELL_KEYS),
        "main_heatmap_uses_seed_local_slots": True,
        "diagonal_cells_are_training_support": True,
        "zero_support_cells_are_marked_n0": True,
        "exposure_cells": list(FOUR_CELL_KEYS),
        "four_cell_is_secondary_taxonomy": True,
    }
    metrics = {
        "seed": seed,
        "recipes": recipes,
        "preferences": list(active_prefs),
        "diagonal_training_pairs": [p.label for p in diagonal],
        "n_effective_preferences": min(effective_preferences_by_recipe.values(), default=0),
        "effective_preferences_by_recipe": effective_preferences_by_recipe,
        "duplicate_orderings_by_recipe": duplicate_orderings_by_recipe,
        "axis_materiality_report": axis_materiality,
        "n_offdiag": len(offdiag),
        "n_cross_axis_value_specs": len(axis_value_specs),
        "n_cross_axis_value_test_pairs": sum(len(spec["test_pairs"]) for spec in axis_value_specs),
        "n_axis_family_holdout_specs": len(axis_family_specs),
        "n_axis_family_holdout_test_pairs": sum(len(spec["test_pairs"]) for spec in axis_family_specs),
        "n_novel_composition_pairs": len(novel_probes),
        "per_baseline": per_baseline,
        "transfer_condition_order": ["cross_recipe_pref_transfer", "cross_axis_value_transfer", "axis_family_holdout"],
        "transfer_condition_definitions": {
            "cross_recipe_pref_transfer": "Train diagonal recipe/preference pairs (R_i, P_i); test off-diagonal pairs (R_i, P_j) where recipe and preference were each seen but not together.",
            "cross_axis_value_transfer": "Train one non-default axis value on one recipe subset; test that same axis value on a held-out recipe subset.",
            "axis_family_holdout": "Train with non-default values for one target axis absent while other axes may vary; test non-default values of the held-out axis.",
        },
        "transfer_conditions": {
            condition: {
                baseline: row.get(condition, {})
                for baseline, row in per_baseline.items()
            }
            for condition in ("cross_recipe_pref_transfer", "cross_axis_value_transfer", "axis_family_holdout")
        },
        "transfer_slot_heatmap": transfer_slot_heatmap,
        "transfer_taxonomy_policy": transfer_taxonomy_policy,
        "oracle_references": _oracle_reference_metrics(
            seed,
            run_config,
            diagonal,
            offdiag,
        ),
    }
    full_transfer = float((per_baseline.get("full", {}) or {}).get("live_online_transfer_top1", 0.0))
    no_pref_transfer = float((per_baseline.get("no_preference_prototype", {}) or {}).get("live_online_transfer_top1", 0.0))
    metrics["cross_recipe_pref_transfer_gain"] = {
        "full_minus_no_preference_transfer_ablation": full_transfer - no_pref_transfer,
        "full_live_online_transfer_top1": full_transfer,
        "no_preference_transfer_ablation_live_online_transfer_top1": no_pref_transfer,
    }
    # Within-recipe preference generalisation vs. true cross-recipe transfer.
    # These two effects are computed from separate cells of the transfer taxonomy:
    #   within_recipe  — seen_recipe_new_preference: the recipe was in training
    #                    but this specific preference was not.
    #   cross_recipe   — seen_recipe_preference_seen_elsewhere: the preference was
    #                    seen on a different recipe; this is the genuine transfer signal.
    # Reporting them separately makes the transfer claim verifiable.
    def _cell_top1(bl: str, cell: str) -> float:
        return float((per_baseline.get(bl, {}) or {}).get("per_transfer_cell", {}).get(cell, {}).get("live_top1", 0.0))
    metrics["within_vs_cross_recipe_decomposition"] = {
        baseline: {
            "within_recipe_pref_generalisation": _cell_top1(baseline, "seen_recipe_new_preference"),
            "cross_recipe_pref_transfer":        _cell_top1(baseline, "seen_recipe_preference_seen_elsewhere"),
            "cross_minus_within":                _cell_top1(baseline, "seen_recipe_preference_seen_elsewhere") - _cell_top1(baseline, "seen_recipe_new_preference"),
            "cross_minus_no_pref_prototype":     _cell_top1(baseline, "seen_recipe_preference_seen_elsewhere") - _cell_top1("no_preference_prototype", "seen_recipe_preference_seen_elsewhere"),
        }
        for baseline in per_baseline
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_confidence_threshold_accuracy(seed: int, out: Path, run_config: RunConfig, cfg_obj: ConfidenceThresholdAccuracyConfig) -> Dict[str, Any]:
    prefs = select_preference_names(cfg_obj.n_preferences)
    n_diag = min(int(cfg_obj.n_recipes), len(prefs))
    active_prefs = tuple(prefs[:n_diag])
    diagonal_spec = _build_exclusive_diagonal(seed, active_prefs, n_diag)
    if diagonal_spec is None:
        raise RuntimeError(
            "confidence_threshold_accuracy requires material recipe/preference pairs; "
            "run materiality_preflight and fix no-op preference transformations."
        )
    recipes = list(diagonal_spec["recipes"])
    diagonal = list(diagonal_spec["diagonal"])
    offdiag = list(diagonal_spec["offdiag"])
    _record_common_artifacts(out, seed, cfg_obj, recipes, active_prefs)
    scenario_rows = [
        {
            "mode": "observe",
            "phase": "confidence_threshold_train",
            "event_type": "diagonal_train",
            "pair": pair.label,
            "recipe": pair.recipe_name,
            "preference": pair.preference_name,
        }
        for pair in diagonal
    ] + [
        {
            "mode": "assist",
            "phase": "confidence_threshold_probe",
            "event_type": "confidence_threshold_probe",
            "condition": condition,
            "pair": pair.label,
            "recipe": pair.recipe_name,
            "preference": pair.preference_name,
        }
        for condition, pairs in (
            ("direct_retrieval", diagonal),
            ("offdiagonal_transfer", offdiag),
        )
        for pair in pairs
    ]
    _append_jsonl(out / "scenario_events.jsonl", scenario_rows)

    thresholds = tuple(float(t) for t in cfg_obj.confidence_thresholds)
    per_baseline: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    for baseline in cfg_obj.baselines:
        agent = make_agent(baseline, base_config(seed, run_config))
        name_to_rid: Dict[str, str] = {}
        t0 = time.perf_counter()
        for _cycle in range(max(1, int(cfg_obj.diagonal_cycles))):
            for pair in diagonal:
                observe_episode(agent, pair, name_to_rid)
        checkpoint = agent.snapshot()
        episode_rows: List[Dict[str, Any]] = []
        step_rows: List[Dict[str, Any]] = []
        for condition, pairs, repeats in (
            ("direct_retrieval", diagonal, 1),
            ("offdiagonal_transfer", offdiag, max(1, int(cfg_obj.offdiag_repeats))),
        ):
            for pair in pairs:
                for rep in range(repeats):
                    agent.restore_from(checkpoint)
                    row, steps = assist_episode(
                        agent,
                        pair,
                        dict(name_to_rid),
                        run_config=run_config,
                        commit=False,
                        event_tag={
                            "condition": condition,
                            "event_type": "confidence_threshold_probe",
                            "repeat": rep,
                            "confidence_threshold_semantics": "posthoc_reporting_cut_not_action_gate",
                        },
                    )
                    row.update({"baseline": baseline, "condition": condition, "repeat": rep})
                    episode_rows.append(row)
                    all_episode_rows.append(row)
                    enriched_steps = [
                        {
                            "baseline": baseline,
                            "condition": condition,
                            "repeat": rep,
                            "pair": pair.label,
                            **step,
                        }
                        for step in steps
                    ]
                    step_rows.extend(enriched_steps)
                    all_step_rows.extend(enriched_steps)
        aggregate = _aggregate_episode_metrics(episode_rows)
        by_condition = {
            condition: {
                **_aggregate_episode_metrics([r for r in episode_rows if r.get("condition") == condition]),
                "confidence_threshold_accuracy_curve": confidence_threshold_accuracy_curve_from_steps(
                    [s for s in step_rows if s.get("condition") == condition],
                    thresholds,
                ),
            }
            for condition in ("direct_retrieval", "offdiagonal_transfer")
        }
        per_baseline[baseline] = {
            **aggregate,
            "confidence_threshold_accuracy_curve": confidence_threshold_accuracy_curve_from_steps(step_rows, thresholds),
            "confidence_threshold_accuracy_by_condition": by_condition,
            "n_scored_robot_turns": float(len([s for s in step_rows if _step_row_confidence(s) is not None])),
            "confidence_threshold_semantics": "posthoc_reporting_cut_not_action_gate",
            "robot_turn_denominator": "scheduled_robot_turns_in_alternating_action_protocol",
            "compute": compute_snapshot(agent, time.perf_counter() - t0),
            "memory": memory_snapshot(agent),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_step_rows)
    metrics = {
        "seed": seed,
        "thresholds": list(thresholds),
        "conditions": ["direct_retrieval", "offdiagonal_transfer"],
        "policy_note": (
            "Curves are posthoc confidence diagnostics over scheduled robot turns. "
            "They do not calibrate or change an action-gate threshold."
        ),
        "per_baseline": per_baseline,
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_decay_reentry_suite(seed: int, out: Path, run_config: RunConfig, cfg_obj: DecayReentrySuiteConfig) -> Dict[str, Any]:
    n_targets = cfg_obj.n_target_recipes
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
    pruned_step_rows: List[Dict[str, Any]] = []
    distractor_grid_rows: List[Dict[str, Any]] = []
    for baseline in cfg_obj.baselines:
        arm_rows: Dict[str, Dict[str, Any]] = {}
        baseline_episode_rows: List[Dict[str, Any]] = []
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
                    if neutral:
                        distractor_counts = (0,)
                    else:
                        distractor_counts = tuple(
                            c for c in cfg_obj.distractor_counts
                            if 0 < int(c) <= int(gap)
                        ) or ((int(gap),) if int(gap) > 0 else (0,))
                    for n_distractors in distractor_counts:
                        agent = make_agent(baseline, base_config(seed, run_config))
                        agent.restore_from(checkpoint)
                        name_to_rid = dict(name_to_rid0)
                        _advance_mixed_gap(agent, int(gap), int(n_distractors), distractors, name_to_rid)
                        active_before = bool(target_rid is not None and (target_rid, target_hash) in agent.decay.active)
                        pruned_before = bool(target_rid is not None and ((target_rid, target_hash) in agent.decay.pruned or target_hash in agent.memory.variants.get(target_rid, {})) and not active_before)
                        target_weight_before = (
                            float(agent.decay.active[(target_rid, target_hash)].weight)
                            if target_rid is not None and (target_rid, target_hash) in agent.decay.active else 0.0
                        )
                        global_rate_before = float(agent.decay.global_rate)
                        active_variants_before = len(agent.decay.active)
                        pruned_variants_before = len(agent.decay.pruned)
                        row, steps = assist_episode(agent, target, name_to_rid, run_config=run_config, event_tag={"arm": arm_name, "gap": gap, "gap_sessions": gap, "n_distractors": n_distractors, "neutral_ticks": int(gap) - int(n_distractors), "condition": "decay_reentry", "target_idx": target_idx})
                        post = frozen_eval(agent, [target], name_to_rid)
                        recovery_steps = next((int(s.get("step", 0)) for s in steps if s.get("correct_top1")), -1)
                        robot_wrong_rate = _mean([1.0 if _row_robot_wrong(s) else 0.0 for s in steps])
                        row.update({
                            "baseline": baseline,
                            "arm": arm_name,
                            "gap": gap,
                            "gap_sessions": int(gap),
                            "n_distractors": int(n_distractors),
                            "neutral_ticks": int(gap) - int(n_distractors),
                            "target": target.label,
                            "same_recipe_variant": variant.label,
                            "target_idx": target_idx,
                            "target_active_before": active_before,
                            "target_pruned_before": pruned_before,
                            "target_weight_before": target_weight_before,
                            "active_variants_before": active_variants_before,
                            "pruned_variants_before": pruned_variants_before,
                            "global_decay_rate_before": global_rate_before,
                            "post_reentry_top1": _mean([v.get("top1", 0.0) for v in post.values()]),
                            "reentry_recovery_steps": recovery_steps,
                            "robot_wrong_rate": robot_wrong_rate,
                            "base_rate_after": float(agent.decay.base_rate),
                            "global_rate_after": float(agent.decay.global_rate),
                            "active_variants_end": len(agent.decay.active),
                            "pruned_variants_end": len(agent.decay.pruned),
                        })
                        gap_rows[f"{target_idx}:{gap}:{n_distractors}"] = row
                        baseline_episode_rows.append(row)
                        all_rows.append(row)
                        distractor_grid_rows.append(row)
                        for step_row in steps:
                            enriched = {
                                "baseline": baseline,
                                "arm": arm_name,
                                "gap_sessions": int(gap),
                                "n_distractors": int(n_distractors),
                                "target_pruned_before": pruned_before,
                                "target_active_before": active_before,
                                "step_idx": int(step_row.get("step", 0)),
                                "top1_correct": bool(step_row.get("correct_top1")),
                                "topk_correct": bool(step_row.get("correct_topk")),
                                "posterior_recipe_confidence": step_row.get("posterior_confidence"),
                                "posterior_pref_confidence": step_row.get("posterior_confidence"),
                                "would_assist_under_gate": bool(_row_policy_bool(step_row, "would_assist_under_gate")),
                                "robot_wrong": bool(_row_robot_wrong(step_row)),
                                **step_row,
                            }
                            all_steps.append(enriched)
                            if pruned_before:
                                pruned_step_rows.append(enriched)
                        last_agent_for_snapshot = agent
            per_gap_summary: Dict[str, Dict[str, float]] = {}
            for gap in cfg_obj.gap_sweep:
                rows_for_gap = [r for r in gap_rows.values() if int(r.get("gap", -1)) == int(gap)]
                active_rows = [r for r in rows_for_gap if r.get("target_active_before")]
                pruned_rows = [r for r in rows_for_gap if r.get("target_pruned_before")]
                other_rows  = [r for r in rows_for_gap if not r.get("target_active_before") and not r.get("target_pruned_before")]
                per_gap_summary[str(gap)] = {
                    **_aggregate_episode_metrics(rows_for_gap),
                    "active_before_rate": _mean([1.0 if r["target_active_before"] else 0.0 for r in rows_for_gap]),
                    "pruned_before_rate": _mean([1.0 if r["target_pruned_before"] else 0.0 for r in rows_for_gap]),
                    "target_weight_before": _mean([r["target_weight_before"] for r in rows_for_gap]),
                    "active_variants_before": _mean([r["active_variants_before"] for r in rows_for_gap]),
                    "pruned_variants_before": _mean([r["pruned_variants_before"] for r in rows_for_gap]),
                    "global_decay_rate_before": _mean([r["global_decay_rate_before"] for r in rows_for_gap]),
                    "reentry_recovery_steps": _mean([r["reentry_recovery_steps"] for r in rows_for_gap if int(r["reentry_recovery_steps"]) >= 0]),
                    "robot_wrong_rate": _mean([r["robot_wrong_rate"] for r in rows_for_gap]),
                    "post_reentry_top1": _mean([r["post_reentry_top1"] for r in rows_for_gap]),
                    "global_rate_after": _mean([r["global_rate_after"] for r in rows_for_gap]),
                    # Memory-state stratification: the three sub-populations at re-encounter
                    # tell distinct stories (interference, forgetting, unseen) and must not
                    # be conflated in the primary figure.
                    "live_top1_when_active": _mean([r["live_top1"] for r in active_rows]) if active_rows else float("nan"),
                    "live_top1_when_pruned": _mean([r["live_top1"] for r in pruned_rows]) if pruned_rows else float("nan"),
                    "live_top1_when_other":  _mean([r["live_top1"] for r in other_rows])  if other_rows  else float("nan"),
                    "n_active_before": float(len(active_rows)),
                    "n_pruned_before": float(len(pruned_rows)),
                    "n_other_before":  float(len(other_rows)),
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
        baseline_episode_agg = _aggregate_episode_metrics(baseline_episode_rows)
        per_baseline[baseline] = {
            **baseline_episode_agg,
            "arms": arm_rows,
            "live_top1": _mean([r["live_top1"] for r in arm_rows.values()]),
            "active_before_rate": _mean([r["active_before_rate"] for r in arm_rows.values()]),
            "post_reentry_top1": _mean([r["post_reentry_top1"] for r in arm_rows.values()]),
            "compute": compute_snapshot(last_agent_for_snapshot or make_agent(baseline, base_config(seed, run_config)), time.perf_counter() - t0),
            "memory": memory_snapshot(last_agent_for_snapshot or make_agent(baseline, base_config(seed, run_config))),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_rows)
    _append_jsonl(out / "episode_steps.jsonl", all_steps)
    _append_jsonl(out / "pruned_reentry_step_trace.jsonl", pruned_step_rows)
    _append_jsonl(out / "distractor_grid_rows.jsonl", distractor_grid_rows)
    pruned_step_trace = {
        baseline: {
            "pruned": _step_trace_from_rows([
                r for r in all_steps
                if r.get("baseline") == baseline and r.get("target_pruned_before")
            ]),
            "active": _step_trace_from_rows([
                r for r in all_steps
                if r.get("baseline") == baseline and r.get("target_active_before")
            ]),
        }
        for baseline in sorted({str(r.get("baseline")) for r in all_steps})
    }
    distractor_grid = {
        baseline: {
            f"{int(r.get('gap_sessions', 0))}:{int(r.get('n_distractors', 0))}": {
                "gap_sessions": int(r.get("gap_sessions", 0)),
                "n_distractors": int(r.get("n_distractors", 0)),
                "reentry_live_top1": float(r.get("live_top1", 0.0)),
                "post_reentry_top1": float(r.get("post_reentry_top1", 0.0)),
                "target_active_before": 1.0 if r.get("target_active_before") else 0.0,
                "target_pruned_before": 1.0 if r.get("target_pruned_before") else 0.0,
                "target_weight_before": float(r.get("target_weight_before", 0.0)),
                "active_variants_before": float(r.get("active_variants_before", 0.0)),
                "pruned_variants_before": float(r.get("pruned_variants_before", 0.0)),
                "global_decay_rate_before": float(r.get("global_decay_rate_before", 0.0)),
                "reentry_recovery_steps": float(r.get("reentry_recovery_steps", -1.0)),
                "robot_wrong_rate": float(r.get("robot_wrong_rate", r.get("wrong_assist_rate", 0.0))),
            }
            for r in distractor_grid_rows
            if r.get("baseline") == baseline
        }
        for baseline in sorted({str(r.get("baseline")) for r in distractor_grid_rows})
    }
    metrics = {
        "seed": seed,
        "targets": [target.label for target, _variant, _distractors in target_specs],
        "same_recipe_variants": [variant.label for _target, variant, _distractors in target_specs],
        "arms": [a for a, _ in arms],
        "per_baseline": per_baseline,
        "pruned_step_trace": pruned_step_trace,
        "distractor_grid": distractor_grid,
    }
    no_decay_neutral = _nested_get(metrics, ("per_baseline", "no_decay", "arms", "neutral", "per_gap"), {}) or {}
    if no_decay_neutral:
        min_active = min(float(row.get("active_before_rate", 0.0)) for row in no_decay_neutral.values())
        metrics["no_decay_neutral_sanity"] = {
            "min_active_before_rate": min_active,
            "passes": bool(min_active >= 0.99),
        }
        if min_active < 0.99 and _ACTIVE_RESULT is not None:
            _ACTIVE_RESULT.warnings.append("no_decay_neutral_active_before_below_one")
    _write_json(out / "metrics.json", metrics)
    return metrics


def _disambiguation_shared_validation_protocol(
    seed: int,
    run_config: RunConfig,
    cfg_obj: DisambiguationAuditConfig,
    pairs: Sequence[RecipePrefPair],
) -> Dict[str, Any]:
    identity_by_recipe = {
        p.recipe_name: p
        for p in sorted(pairs, key=lambda pair: pair.label)
        if p.preference_name == "identity"
    }
    rng = _rng(seed, 1313)
    recipe_names = sorted(identity_by_recipe)
    rng.shuffle(recipe_names)
    n_holdout = max(1, int(round(float(cfg_obj.validation_fraction) * len(recipe_names)))) if recipe_names else 0
    heldout_recipes = set(recipe_names[:n_holdout])
    train_recipe_names = [r for r in recipe_names if r not in heldout_recipes]
    identity_pairs = [identity_by_recipe[r] for r in train_recipe_names if r in identity_by_recipe]
    known_shift_pairs = [
        p for p in sorted(pairs, key=lambda pair: pair.label)
        if p.recipe_name in set(train_recipe_names)
        and p.preference_name != "identity"
        and p.actions != identity_by_recipe[p.recipe_name].actions
    ]
    unseen_recipe_pairs = [
        p for p in sorted(pairs, key=lambda pair: pair.label)
        if p.recipe_name in heldout_recipes
        and p.preference_name == "identity"
    ]
    if not identity_pairs:
        return {
            "protocol": "shared_prefix_full_validation",
            "validation_rows": [],
            "partial_precision_recall_curve": {},
            "full_precision_recall_curve": {},
            "joint_grid": [],
            "recommendations": {},
            "error": "no_train_identity_pairs",
        }

    agent = make_agent("full", base_config(seed, run_config))
    name_to_rid: Dict[str, str] = {}
    for pair in identity_pairs:
        observe_episode(agent, pair, name_to_rid)
    active_lib = agent.memory.library(allowed_keys=agent._active_keys())
    train_identity_pairs = [p for p in identity_pairs if p.preference_name == "identity"]
    train_identity_actions = list(train_identity_pairs)

    prefix_fracs = tuple(float(f) for f in cfg_obj.prefix_fractions)
    validation_cases: List[Dict[str, Any]] = []

    def _nearest_train_identity(pair: RecipePrefPair) -> float:
        return max((jaccard(pair.actions, train_pair.actions) for train_pair in train_identity_actions), default=0.0)

    def _case(pair: RecipePrefPair, case_type: str, expected_memory_state: str, expected_gate: str) -> Dict[str, Any]:
        return {
            "case_type": case_type,
            "pair": pair,
            "expected_memory_state": expected_memory_state,
            "expected_gate": expected_gate,
            "expected_assist": expected_gate == "assist",
            "true_recipe_id": name_to_rid.get(pair.recipe_name),
            "nearest_train_identity_jaccard": _nearest_train_identity(pair),
        }

    for pair in identity_pairs:
        validation_cases.append(_case(pair, "known_recipe_seen_variant", "active_memory", "assist"))
    for pair in known_shift_pairs:
        validation_cases.append(_case(pair, "known_recipe_unseen_preference", "same_recipe_new_preference", "assist"))
    for pair in unseen_recipe_pairs:
        validation_cases.append(_case(pair, "unseen_recipe", "no_memory", "observe"))

    def _scores_for_prefix(pair: RecipePrefPair, n_prefix: int) -> Dict[str, Any]:
        tokens = agent._tokens_from_action_labels(pair.actions)
        n = max(1, min(int(n_prefix), len(tokens)))
        prefix = tokens[:n]
        ranked = agent.disambig.score_partial(prefix, active_lib)
        partial_recipe = ranked[0][0].recipe_id if ranked else None
        partial_score = float(ranked[0][1]) if ranked else 0.0
        cls = agent.disambig.classify(prefix, active_lib, threshold=0.0) if active_lib else None
        return {
            "n_prefix": int(n),
            "prefix_fraction_actual": float(n / max(1, len(tokens))),
            "partial_score": partial_score,
            "partial_recipe_id": partial_recipe,
            "full_jaccard_score": float(cls.jaccard) if cls is not None else 0.0,
            "full_recipe_id": cls.recipe_id if cls is not None else None,
            "full_order_distance": float(cls.order_distance) if cls is not None else 0.0,
        }

    validation_rows: List[Dict[str, Any]] = []
    for case in validation_cases:
        pair = case["pair"]
        tokens = agent._tokens_from_action_labels(pair.actions)
        for frac in prefix_fracs:
            n_prefix = max(1, int(math.ceil(float(frac) * len(tokens))))
            scores = _scores_for_prefix(pair, n_prefix)
            validation_rows.append({
                "pair": pair.label,
                "recipe": pair.recipe_name,
                "preference": pair.preference_name,
                "case_type": case["case_type"],
                "expected_memory_state": case["expected_memory_state"],
                "expected_gate": case["expected_gate"],
                "expected_assist": bool(case["expected_assist"]),
                "true_recipe_id": case["true_recipe_id"],
                "nearest_train_identity_jaccard": float(case["nearest_train_identity_jaccard"]),
                "requested_prefix_fraction": float(frac),
                "recipe_length": len(tokens),
                **scores,
            })

    def _safe_rate(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    def _decision_report(rows: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        tp = fp = tn = fn = 0
        recipe_correct = []
        by_case: Dict[str, List[Tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        for row, decision in zip(rows, decisions):
            expected_assist = bool(row.get("expected_assist"))
            pred_assist = bool(decision.get("pred_assist"))
            if pred_assist and expected_assist:
                tp += 1
            elif pred_assist and not expected_assist:
                fp += 1
            elif (not pred_assist) and expected_assist:
                fn += 1
            else:
                tn += 1
            if expected_assist and pred_assist and row.get("true_recipe_id") is not None:
                recipe_correct.append(1.0 if decision.get("pred_recipe_id") == row.get("true_recipe_id") else 0.0)
            by_case[str(row.get("case_type"))].append((row, decision))
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = _safe_rate(2.0 * precision * recall, precision + recall)
        case_reports = {}
        for case_type, vals in by_case.items():
            expected_known = [v for v in vals if bool(v[0].get("expected_assist"))]
            expected_observe = [v for v in vals if not bool(v[0].get("expected_assist"))]
            case_reports[case_type] = {
                "n": len(vals),
                "assist_rate": _mean([1.0 if v[1].get("pred_assist") else 0.0 for v in vals]),
                "false_observe_rate": _mean([1.0 if not v[1].get("pred_assist") else 0.0 for v in expected_known]) if expected_known else 0.0,
                "observation_recall": _mean([1.0 if not v[1].get("pred_assist") else 0.0 for v in expected_observe]) if expected_observe else 0.0,
            }
        return {
            "precision_known_assist": precision,
            "recall_known_assist": recall,
            "f1_known_assist": f1,
            "known_recipe_false_observe_rate": _safe_rate(fn, tp + fn),
            "unseen_recipe_false_assist_rate": _safe_rate(fp, fp + tn),
            "novel_recipe_observation_recall": _safe_rate(tn, fp + tn),
            "recipe_id_accuracy_when_assisting_known": _mean(recipe_correct),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n": len(rows),
            "by_case_type": case_reports,
        }

    partial_curve: Dict[str, Any] = {}
    for threshold in cfg_obj.online_partial_thresholds:
        t = float(threshold)
        decisions = [
            {
                "pred_assist": float(row.get("partial_score", 0.0)) >= t,
                "pred_recipe_id": row.get("partial_recipe_id"),
            }
            for row in validation_rows
        ]
        partial_curve[str(t)] = {
            "online_new_recipe_partial_threshold": t,
            **_decision_report(validation_rows, decisions),
        }

    full_curve: Dict[str, Any] = {}
    for threshold in cfg_obj.thresholds:
        t = float(threshold)
        decisions = [
            {
                "pred_assist": float(row.get("full_jaccard_score", 0.0)) >= t,
                "pred_recipe_id": row.get("full_recipe_id"),
            }
            for row in validation_rows
        ]
        full_curve[str(t)] = {
            "jaccard_threshold": t,
            **_decision_report(validation_rows, decisions),
        }

    lengths = sorted(len(agent._tokens_from_action_labels(case["pair"].actions)) for case in validation_cases)
    median_len = lengths[len(lengths) // 2] if lengths else int(DEFAULT_CONFIG.min_classify_length)
    min_length_rules: List[Dict[str, Any]] = []
    for length in cfg_obj.min_classify_lengths:
        min_length_rules.append({"rule": "absolute", "value": int(length), "min_classify_length": max(1, int(length)), "fraction": None})
    for frac in cfg_obj.min_classify_fractions:
        min_length_rules.append({
            "rule": "median_recipe_fraction",
            "value": float(frac),
            "fraction": float(frac),
            "min_classify_length": max(3, int(math.ceil(float(frac) * max(1, median_len)))),
        })
    deduped_rules: List[Dict[str, Any]] = []
    seen_rule_keys = set()
    for rule in min_length_rules:
        key = (rule["rule"], rule["value"], rule["min_classify_length"])
        if key not in seen_rule_keys:
            deduped_rules.append(rule)
            seen_rule_keys.add(key)

    def _transition_report(partial_threshold: float, jaccard_threshold: float, min_len: int) -> Dict[str, Any]:
        transition_rows = []
        for case in validation_cases:
            pair = case["pair"]
            tokens = agent._tokens_from_action_labels(pair.actions)
            if min_len <= 1 or len(tokens) < min_len:
                continue
            pre = _scores_for_prefix(pair, min_len - 1)
            post = _scores_for_prefix(pair, min_len)
            pre_assist = float(pre["partial_score"]) >= float(partial_threshold)
            post_assist = float(post["full_jaccard_score"]) >= float(jaccard_threshold)
            transition_rows.append({
                "pair": pair.label,
                "case_type": case["case_type"],
                "expected_assist": bool(case["expected_assist"]),
                "pre_n_prefix": min_len - 1,
                "post_n_prefix": min_len,
                "pre_partial_score": float(pre["partial_score"]),
                "post_full_jaccard_score": float(post["full_jaccard_score"]),
                "pre_assist": bool(pre_assist),
                "post_assist": bool(post_assist),
                "assist_decision_flipped": bool(pre_assist != post_assist),
                "pre_recipe_id": pre["partial_recipe_id"],
                "post_recipe_id": post["full_recipe_id"],
                "recipe_id_flipped": bool(pre_assist and post_assist and pre["partial_recipe_id"] != post["full_recipe_id"]),
            })
        return {
            "transition_flip_rate": _mean([1.0 if r["assist_decision_flipped"] else 0.0 for r in transition_rows]),
            "transition_recipe_flip_rate": _mean([1.0 if r["recipe_id_flipped"] else 0.0 for r in transition_rows]),
            "n_transition_cases": len(transition_rows),
            "rows": transition_rows,
        }

    joint_grid: List[Dict[str, Any]] = []
    for partial_threshold in cfg_obj.online_partial_thresholds:
        for jaccard_threshold in cfg_obj.thresholds:
            for rule in deduped_rules:
                min_len = int(rule["min_classify_length"])
                decisions = []
                for row in validation_rows:
                    use_partial = int(row.get("n_prefix", 0)) < min_len
                    if use_partial:
                        decisions.append({
                            "pred_assist": float(row.get("partial_score", 0.0)) >= float(partial_threshold),
                            "pred_recipe_id": row.get("partial_recipe_id"),
                            "classifier": "partial_prefix",
                        })
                    else:
                        decisions.append({
                            "pred_assist": float(row.get("full_jaccard_score", 0.0)) >= float(jaccard_threshold),
                            "pred_recipe_id": row.get("full_recipe_id"),
                            "classifier": "full_jaccard",
                        })
                report = _decision_report(validation_rows, decisions)
                transition = _transition_report(float(partial_threshold), float(jaccard_threshold), min_len)
                objective = (
                    0.30 * float(report["recall_known_assist"])
                    + 0.25 * float(report["novel_recipe_observation_recall"])
                    + 0.20 * float(report["recipe_id_accuracy_when_assisting_known"])
                    + 0.15 * float(report["f1_known_assist"])
                    + 0.10 * (1.0 - float(transition["transition_flip_rate"]))
                    - 0.10 * float(report["unseen_recipe_false_assist_rate"])
                )
                joint_grid.append({
                    "online_new_recipe_partial_threshold": float(partial_threshold),
                    "jaccard_threshold": float(jaccard_threshold),
                    "min_classify_length": min_len,
                    "min_classify_rule": rule["rule"],
                    "min_classify_fraction": rule.get("fraction"),
                    "median_recipe_length": int(median_len),
                    "objective": float(objective),
                    **report,
                    "transition_consistency": transition,
                })

    best = max(
        joint_grid,
        key=lambda row: (
            float(row.get("objective", 0.0)),
            -float(row.get("known_recipe_false_observe_rate", 1.0)),
            -float(row.get("unseen_recipe_false_assist_rate", 1.0)),
            -float(row.get("transition_consistency", {}).get("transition_flip_rate", 1.0)),
            -abs(float(row.get("online_new_recipe_partial_threshold", 0.0)) - float(DEFAULT_CONFIG.online_new_recipe_partial_threshold)),
            -abs(float(row.get("jaccard_threshold", 0.0)) - float(DEFAULT_CONFIG.jaccard_threshold)),
        ),
    ) if joint_grid else {}

    validation_case_counts = Counter(str(case["case_type"]) for case in validation_cases)
    return {
        "protocol": "shared_prefix_full_validation",
        "decision_target": "known recipes should remain assistable; held-out recipes should require observation",
        "train_pairs": [p.label for p in identity_pairs],
        "heldout_recipes": sorted(heldout_recipes),
        "validation_case_counts": {k: int(v) for k, v in sorted(validation_case_counts.items())},
        "validation_rows": validation_rows,
        "partial_precision_recall_curve": partial_curve,
        "full_precision_recall_curve": full_curve,
        "joint_grid": joint_grid,
        "transition_metric": "decision flip between partial-prefix prefix_length=min_len-1 and full-jaccard prefix_length=min_len",
        "recommendations": {
            "online_new_recipe_partial_threshold": best.get("online_new_recipe_partial_threshold"),
            "jaccard_threshold": best.get("jaccard_threshold"),
            "min_classify_length": best.get("min_classify_length"),
            "min_classify_rule": best.get("min_classify_rule"),
            "min_classify_fraction": best.get("min_classify_fraction"),
            "median_recipe_length": best.get("median_recipe_length"),
            "objective": best.get("objective"),
            "selected_metrics": {
                k: best.get(k)
                for k in (
                    "precision_known_assist",
                    "recall_known_assist",
                    "f1_known_assist",
                    "known_recipe_false_observe_rate",
                    "novel_recipe_observation_recall",
                    "unseen_recipe_false_assist_rate",
                    "recipe_id_accuracy_when_assisting_known",
                )
            },
            "transition_consistency": best.get("transition_consistency", {}),
            "defaults": {
                "online_new_recipe_partial_threshold": float(DEFAULT_CONFIG.online_new_recipe_partial_threshold),
                "jaccard_threshold": float(DEFAULT_CONFIG.jaccard_threshold),
                "min_classify_length": int(DEFAULT_CONFIG.min_classify_length),
            },
        },
    }


def _posterior_hyperparameter_calibration(
    seed: int,
    run_config: RunConfig,
    cfg_obj: DisambiguationAuditConfig,
    pairs: Sequence[RecipePrefPair],
) -> Dict[str, Any]:
    identity_by_recipe = {
        p.recipe_name: p
        for p in pairs
        if p.preference_name == "identity"
    }
    train_pairs = list(identity_by_recipe.values())
    calibration_pairs = [
        p for p in pairs
        if p.recipe_name in identity_by_recipe
        and p.preference_name != "identity"
        and p.actions != identity_by_recipe[p.recipe_name].actions
    ]
    alpha_grid = tuple(float(x) for x in cfg_obj.posterior_alpha_grid)
    temp_grid = tuple(float(x) for x in cfg_obj.posterior_global_temperature_grid)
    calibration_pairs = calibration_pairs[: max(1, int(cfg_obj.posterior_calibration_max_pairs))]
    if not train_pairs or not calibration_pairs:
        return {"rows": [], "recommendation": {}, "objective": "hrc_utility"}
    rows: List[Dict[str, Any]] = []
    grid_points = [
        (float(ar), float(ap), float(am), float(gt))
        for ar in alpha_grid
        for ap in alpha_grid
        for am in alpha_grid
        for gt in temp_grid
    ]
    max_grid = max(1, int(cfg_obj.posterior_calibration_max_grid_points))
    if len(grid_points) > max_grid:
        defaults = (
            float(DEFAULT_CONFIG.posterior_alpha_recipe),
            float(DEFAULT_CONFIG.posterior_alpha_pref),
            float(DEFAULT_CONFIG.posterior_alpha_memory),
            float(DEFAULT_CONFIG.posterior_global_temperature),
        )
        ranked = sorted(
            grid_points,
            key=lambda p: (
                abs(p[0] - defaults[0]) + abs(p[1] - defaults[1]) + abs(p[2] - defaults[2]) + abs(p[3] - defaults[3]),
                p,
            ),
        )
        stride = max(1, len(grid_points) // max_grid)
        spread = grid_points[::stride]
        grid_points = list(dict.fromkeys([defaults] + ranked[: max_grid // 2] + spread))[:max_grid]
    for alpha_recipe, alpha_pref, alpha_memory, global_temp in grid_points:
        cfg = base_config(
            seed,
            run_config,
            posterior_alpha_recipe=float(alpha_recipe),
            posterior_alpha_pref=float(alpha_pref),
            posterior_alpha_memory=float(alpha_memory),
            posterior_global_temperature=float(global_temp),
        )
        agent = make_agent("full", cfg)
        name_to_rid: Dict[str, str] = {}
        for pair in train_pairs:
            observe_episode(agent, pair, name_to_rid)
        eval_rows = []
        for pair in calibration_pairs:
            row, _steps = assist_episode(
                agent,
                pair,
                dict(name_to_rid),
                run_config=run_config,
                commit=False,
                event_tag={"condition": "posterior_calibration"},
            )
            eval_rows.append(row)
        agg = _aggregate_episode_metrics(eval_rows)
        net_value = _hrc_net_action_value(agg)
        rows.append({
            "posterior_alpha_recipe": float(alpha_recipe),
            "posterior_alpha_pref": float(alpha_pref),
            "posterior_alpha_memory": float(alpha_memory),
            "posterior_global_temperature": float(global_temp),
            "hrc_net_action_value": float(net_value),
            "robot_wrong_rate": float(agg.get("robot_wrong_rate", 0.0)),
            "live_top1": float(agg.get("live_top1", 0.0)),
            "net_robot_action_value": float(agg.get("net_robot_action_value", 0.0)),
            "confidence_ece": float(agg.get("confidence_ece", 0.0)),
            "confidence_brier": float(agg.get("confidence_brier", 0.0)),
            "n_calibration_pairs": len(calibration_pairs),
        })
    eligible = [
        r for r in rows
        if float(r.get("robot_wrong_rate", 1.0)) <= float(DEFAULT_CONFIG.posterior_action_confidence_threshold + 0.45)
    ] or rows
    best = max(
        eligible,
        key=lambda r: (
            float(r.get("hrc_net_action_value", 0.0)),
            float(r.get("live_top1", 0.0)),
            -float(r.get("confidence_ece", 0.0)),
        ),
    )
    return {
        "objective": "hrc_net_action_value_with_calibration_diagnostics",
        "rows": rows,
        "recommendation": dict(best),
        "reported_only_does_not_mutate_default_config": True,
        "default_values": {
            "posterior_alpha_recipe": float(DEFAULT_CONFIG.posterior_alpha_recipe),
            "posterior_alpha_pref": float(DEFAULT_CONFIG.posterior_alpha_pref),
            "posterior_alpha_memory": float(DEFAULT_CONFIG.posterior_alpha_memory),
            "posterior_global_temperature": float(DEFAULT_CONFIG.posterior_global_temperature),
        },
    }


def run_disambiguation_audit(seed: int, out: Path, run_config: RunConfig, cfg_obj: DisambiguationAuditConfig) -> Dict[str, Any]:
    recipes, pairs = build_pairs(seed, cfg_obj.n_recipes, n_preferences=7)
    rng = _rng(seed, 911)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cut = max(1, int((1.0 - cfg_obj.validation_fraction) * len(order)))
    train_pairs = [pairs[i] for i in order[:cut]]
    val_pairs = [pairs[i] for i in order[cut:]]
    recipe_order = list(recipes)
    rng.shuffle(recipe_order)
    n_val_recipes = max(1, int(round(float(cfg_obj.validation_fraction) * len(recipe_order))))
    val_recipe_names = set(recipe_order[:n_val_recipes])
    train_recipe_pairs = [p for p in pairs if p.recipe_name not in val_recipe_names]
    val_recipe_pairs = [p for p in pairs if p.recipe_name in val_recipe_names]
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
        active_route_rows = []
        for pair in val_pairs:
            gt = _memory_state_gt(agent, pair, name_to_rid, train_labels, train_recipes)
            pred = _classify_memory_state_pair(agent, pair, name_to_rid, threshold=float(thr))
            rows.append({"threshold": thr, "pair": pair.label, "recipe": pair.recipe_name, "preference": pair.preference_name, "ground_truth": gt, "predicted": pred, "gt": gt, "pred": pred, "correct": gt == pred})
            agent.restore_from(checkpoint)
            route_mode, route_tags = _user_selected_mode_for_active_memory("assist", agent, pair, name_to_rid)
            gt_route = "assist" if route_tags["active_recipe_before"] else "observe"
            active_route_rows.append({"threshold": thr, "pair": pair.label, "gt": gt_route, "pred": route_mode, "correct": gt_route == route_mode, **route_tags})
        matrix = confusion_labels([r["predicted"] for r in rows], [r["ground_truth"] for r in rows])
        report = classifier_report(matrix, MEMORY_STATE_LABELS)
        recipe_agent = make_agent("full", base_config(seed, run_config, jaccard_threshold=float(thr)))
        recipe_map: Dict[str, str] = {}
        for pair in train_recipe_pairs:
            observe_episode(recipe_agent, pair, recipe_map)
        recipe_train_labels = {p.label for p in train_recipe_pairs}
        recipe_train_recipes = {p.recipe_name for p in train_recipe_pairs}
        recipe_checkpoint = recipe_agent.snapshot()
        recipe_rows = []
        recipe_route_rows = []
        for pair in val_recipe_pairs:
            gt = _memory_state_gt(recipe_agent, pair, recipe_map, recipe_train_labels, recipe_train_recipes)
            pred = _classify_memory_state_pair(recipe_agent, pair, recipe_map, threshold=float(thr))
            recipe_rows.append({"threshold": thr, "pair": pair.label, "recipe": pair.recipe_name, "preference": pair.preference_name, "gt": gt, "pred": pred, "correct": gt == pred})
            recipe_agent.restore_from(recipe_checkpoint)
            route_mode, route_tags = _user_selected_mode_for_active_memory("assist", recipe_agent, pair, recipe_map)
            recipe_route_rows.append({"threshold": thr, "pair": pair.label, "gt": "observe", "pred": route_mode, "correct": route_mode == "observe", **route_tags})
        recipe_matrix = confusion_labels([r["pred"] for r in recipe_rows], [r["gt"] for r in recipe_rows])
        per_threshold[str(thr)] = {
            **report,
            "confusion_matrix": matrix.tolist(),
            "rows": rows,
            "active_memory_route_decision": _route_report(active_route_rows),
            "recipe_holdout_classification": {**classifier_report(recipe_matrix, MEMORY_STATE_LABELS), "confusion_matrix": recipe_matrix.tolist(), "rows": recipe_rows},
            "recipe_holdout_active_memory_route_decision": _route_report(recipe_route_rows),
            "mode_policy": "user selects observation when the recipe is absent from active memory; otherwise user selects assist",
        }
        _append_jsonl(out / "classification_rows.jsonl", rows)
        _append_jsonl(out / "active_memory_route_rows.jsonl", active_route_rows)
        _append_jsonl(out / "recipe_holdout_classification_rows.jsonl", recipe_rows)
        _append_jsonl(out / "recipe_holdout_active_memory_route_rows.jsonl", recipe_route_rows)
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
        boundary[str(frac)] = {**classifier_report(matrix, MEMORY_STATE_LABELS), "confusion_matrix": matrix.tolist(), "rows": rows, "interpretation": "missing_observation_robustness"}
        _append_jsonl(out / "boundary_rows.jsonl", rows)
    prefix_robustness: Dict[str, Any] = {}
    for frac in cfg_obj.prefix_fractions:
        rows = []
        for pair in val_pairs:
            n_prefix = max(1, int(math.ceil(float(frac) * len(pair.actions))))
            prefix_actions = list(pair.actions[:n_prefix])
            gt = _memory_state_gt(base_agent, pair, name_to_rid, observed_labels, observed_recipes)
            pred = _classify_memory_state_pair(base_agent, pair, name_to_rid, actions=prefix_actions)
            rows.append({"prefix_fraction": frac, "pair": pair.label, "n_prefix": n_prefix, "gt": gt, "pred": pred, "correct": gt == pred})
        matrix = confusion_labels([r["pred"] for r in rows], [r["gt"] for r in rows])
        prefix_robustness[str(frac)] = {**classifier_report(matrix, MEMORY_STATE_LABELS), "confusion_matrix": matrix.tolist(), "rows": rows, "interpretation": "growing_prefix_online_robustness"}
        _append_jsonl(out / "prefix_rows.jsonl", rows)
    identity_by_recipe = {
        recipe: materialize_pair(recipe, "identity", fn)
        for recipe, fn in select_recipe_builders(seed, cfg_obj.n_recipes)
    }
    by_axis_diff_count: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for pair in val_pairs:
        pref = PRESET_PREFERENCES.get(pair.preference_name)
        if pref is None:
            continue
        axis_diff_count = _preference_axis_difference_count(pref)
        identity_pair = identity_by_recipe.get(pair.recipe_name)
        by_axis_diff_count[axis_diff_count].append({
            "pair": pair.label,
            "axis_diff_count": axis_diff_count,
            "jaccard_from_identity": jaccard(identity_pair.actions, pair.actions) if identity_pair else 0.0,
            "kendall_from_identity": kendall_tau_distance(identity_pair.actions, pair.actions) if identity_pair else 0.0,
        })
    for axis_diff_count, rows in sorted(by_axis_diff_count.items()):
        preference_axis_degradation[str(axis_diff_count)] = {
            "mean_jaccard_from_identity": _mean([r["jaccard_from_identity"] for r in rows]),
            "mean_kendall_from_identity": _mean([r["kendall_from_identity"] for r in rows]),
            "n": len(rows),
            "rows": rows,
        }
    threshold_calibration = _disambiguation_shared_validation_protocol(seed, run_config, cfg_obj, pairs)
    posterior_calibration = _posterior_hyperparameter_calibration(seed, run_config, cfg_obj, pairs)
    metrics = {
        "seed": seed,
        "per_threshold": per_threshold,
        "best_threshold": best_threshold,
        "threshold_calibration": threshold_calibration,
        "recommended_thresholds": threshold_calibration.get("recommendations", {}),
        "posterior_hyperparameter_calibration": posterior_calibration,
        "boundary_degradation": boundary,
        "online_prefix_boundary_degradation": boundary,
        "prefix_robustness": prefix_robustness,
        "preference_axis_degradation": preference_axis_degradation,
        "mode_policy": "user selects observation when the recipe is absent from active memory; pruned recipes therefore require re-observation for baselines that do not keep an active variant",
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


def earliest_divergence_step(a: Sequence[str], b: Sequence[str]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _recipe_family_jaccard_ambiguities(
    pairs: Sequence[RecipePrefPair],
    threshold: float,
) -> List[Dict[str, Any]]:
    identities = [p for p in pairs if p.preference_name == "identity"]
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(identities):
        for b in identities[i + 1:]:
            score = jaccard(a.actions, b.actions)
            if score >= float(threshold):
                out.append({
                    "recipe_a": a.recipe_name,
                    "recipe_b": b.recipe_name,
                    "jaccard": float(score),
                    "threshold": float(threshold),
                })
    return out


def _deduplicate_preference_pairs(
    pairs: Sequence[RecipePrefPair],
    tau_threshold: float = 0.05,
) -> Tuple[List[RecipePrefPair], List[Dict[str, Any]]]:
    seen_hashes: set = set()
    kept: List[RecipePrefPair] = []
    dropped: List[Dict[str, Any]] = []
    for pair in pairs:
        h = tuple(pair.actions)
        if h in seen_hashes:
            match = next((p for p in kept if tuple(p.actions) == h), None)
            dropped.append({
                "label": pair.label,
                "preference": pair.preference_name,
                "reason": "exact_duplicate_ordering",
                "representative_label": match.label if match else None,
                "representative_preference": match.preference_name if match else None,
            })
            continue
        near_match: Optional[RecipePrefPair] = None
        near_tau = 1.0
        for prior in kept:
            tau = kendall_tau_distance(pair.actions, prior.actions)
            if tau < float(tau_threshold):
                near_match = prior
                near_tau = tau
                break
        if near_match is not None:
            dropped.append({
                "label": pair.label,
                "preference": pair.preference_name,
                "reason": "near_duplicate_ordering",
                "kendall_tau": float(near_tau),
                "representative_label": near_match.label,
                "representative_preference": near_match.preference_name,
            })
            continue
        seen_hashes.add(h)
        kept.append(pair)
    return kept, dropped


def _materiality_failure_reasons(
    summary: Sequence[Mapping[str, Any]],
    filtered_pairs: Sequence[str],
    recipe_family_ambiguities: Sequence[Mapping[str, Any]],
    cfg_obj: MaterialityPreflightConfig,
) -> List[str]:
    reasons: List[str] = []
    for row in summary:
        recipe = str(row.get("recipe", "unknown_recipe"))
        if not row.get("passes_min_effective", False):
            reasons.append(f"{recipe}: only {int(row.get('n_effective_preferences', 0))} effective preferences after dedup")
        if not row.get("passes_duplicate_ordering", False):
            reasons.append(f"{recipe}: {int(row.get('near_duplicate_count', 0))} duplicate or near-duplicate orderings")
    if cfg_obj.fail_on_noop and filtered_pairs:
        reasons.append(f"noop_pairs={len(filtered_pairs)}")
    if recipe_family_ambiguities:
        reasons.append(f"recipe_family_jaccard_ambiguities={len(recipe_family_ambiguities)}")
    return reasons


def _materiality_preflight_pairs(
    seed: int,
    cfg_obj: MaterialityPreflightConfig,
) -> Tuple[List[str], List[RecipePrefPair], Dict[str, Any]]:
    builders = select_recipe_builders(seed, cfg_obj.n_recipes)
    selected_pairs: List[RecipePrefPair] = []
    selection_rows: List[Dict[str, Any]] = []
    for recipe_name, fn in builders:
        candidates: List[RecipePrefPair] = []
        seen_orders: set = set()
        for pair in _material_pair_candidates(recipe_name, fn, PREFERENCE_NAMES):
            h = _order_hash(pair)
            if h in seen_orders:
                continue
            seen_orders.add(h)
            candidates.append(pair)
        for pref in _build_gap_preferences():
            try:
                pair = materialize_custom_pair(recipe_name, pref, fn)
            except Exception:
                continue
            if not pair.base_pref and not pair.applied_axes:
                continue
            h = _order_hash(pair)
            if h in seen_orders:
                continue
            seen_orders.add(h)
            candidates.append(pair)
        deduped, duplicate_pairs = _deduplicate_preference_pairs(
            candidates,
            tau_threshold=float(cfg_obj.near_duplicate_tau_threshold),
        )
        pref_rank = {pref: idx for idx, pref in enumerate(PREFERENCE_NAMES)}
        ordered = sorted(
            deduped,
            key=lambda p: (
                0 if p.preference_name == "identity" else 1,
                pref_rank.get(p.preference_name, len(pref_rank)),
                p.preference_name,
                p.label,
            ),
        )
        kept = ordered[: max(1, int(cfg_obj.n_preferences))]
        selected_pairs.extend(kept)
        selection_rows.append({
            "recipe": recipe_name,
            "n_candidate_preferences": len(candidates),
            "n_effective_preferences_available": len(deduped),
            "n_selected_preferences": len(kept),
            "dropped_duplicate_or_near_duplicate_pairs": duplicate_pairs,
            "selected_preferences": [p.preference_name for p in kept],
        })
    return [recipe for recipe, _fn in builders], selected_pairs, {
        "selection_policy": "per_recipe_effective_material_candidates_then_stable_order",
        "selection_rows": selection_rows,
    }


def run_materiality_preflight(seed: int, out: Path, run_config: RunConfig, cfg_obj: MaterialityPreflightConfig) -> Dict[str, Any]:
    recipes, pairs, selection_report = _materiality_preflight_pairs(seed, cfg_obj)
    prefs = sorted({p.preference_name for p in pairs})
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
        recipe_pairs = [pair_by_label[v["label"]] for v in vals if v["label"] in pair_by_label]
        deduped_pairs, duplicate_pairs = _deduplicate_preference_pairs(
            recipe_pairs,
            tau_threshold=float(cfg_obj.near_duplicate_tau_threshold),
        )
        effective = len(deduped_pairs)
        kendalls = [float(v["kendall_from_identity"]) for v in vals if v["preference"] != "identity"]
        pairwise_kendalls = [
            kendall_tau_distance(a.actions, b.actions)
            for i, a in enumerate(recipe_pairs)
            for b in recipe_pairs[i + 1:]
            if a.actions != b.actions
        ]
        divergence_rows = [
            {
                "pair_a": a.label,
                "pair_b": b.label,
                "earliest_divergence_step": earliest_divergence_step(a.actions, b.actions),
                "recipe_length": min(len(a.actions), len(b.actions)),
            }
            for i, a in enumerate(recipe_pairs)
            for b in recipe_pairs[i + 1:]
            if a.actions != b.actions
        ]
        late_divergence_cutoff = max(1, min((len(p.actions) for p in recipe_pairs), default=1) // 3)
        late_diverging = [
            r for r in divergence_rows
            if int(r["earliest_divergence_step"]) > late_divergence_cutoff
        ]
        classify_late = [
            r for r in divergence_rows
            if int(r["earliest_divergence_step"]) > int(cfg_obj.min_classify_length)
        ]
        min_divergence = min((int(r["earliest_divergence_step"]) for r in divergence_rows), default=-1)
        late_diverging_recipe = bool(min_divergence > late_divergence_cutoff) if min_divergence >= 0 else False
        summary.append({
            "recipe": recipe,
            "n_preferences": len(vals),
            "effective_preference_count": effective,
            "n_effective_preferences": effective,
            "n_effective_preferences_after_dedup": effective,
            "duplicate_count": sum(1 for v in vals if v["is_duplicate"]),
            "near_duplicate_count": len(duplicate_pairs),
            "duplicate_or_near_duplicate_pairs": duplicate_pairs,
            "min_kendall_from_identity": min(kendalls) if kendalls else 0.0,
            "mean_kendall_from_identity": _mean(kendalls),
            "min_pairwise_kendall_per_recipe": min(pairwise_kendalls) if pairwise_kendalls else 0.0,
            "min_earliest_divergence_per_recipe": min_divergence,
            "mean_earliest_divergence_per_recipe": _mean([int(r["earliest_divergence_step"]) for r in divergence_rows]),
            "late_divergence_cutoff": int(late_divergence_cutoff),
            "n_late_diverging_preference_pairs": len(late_diverging),
            "n_after_min_classify_length_pairs": len(classify_late),
            "late_diverging_preference_pairs": late_diverging,
            "late_diverging_recipe": late_diverging_recipe,
            "passes_early_divergence": not late_diverging_recipe,
            "passes_min_classify_divergence": len(classify_late) == 0,
            "passes_min_effective": effective >= int(cfg_obj.min_effective_preferences),
            "passes_duplicate_ordering": not (cfg_obj.fail_on_duplicate_ordering and duplicate_pairs),
        })
    recipe_family_ambiguities = _recipe_family_jaccard_ambiguities(
        [p for p in pairs if p.preference_name == "identity"],
        float(DEFAULT_CONFIG.jaccard_threshold),
    )
    pass_preflight = all(
        r["passes_min_effective"]
        and r["passes_duplicate_ordering"]
        for r in summary
    )
    failure_reasons = _materiality_failure_reasons(summary, filtered_pairs, recipe_family_ambiguities, cfg_obj)
    diagnostic_warnings: List[str] = []
    for row in summary:
        recipe = str(row.get("recipe", "unknown_recipe"))
        if not row.get("passes_early_divergence", False):
            diagnostic_warnings.append(f"{recipe}: late-diverging preference pairs")
        if not row.get("passes_min_classify_divergence", False):
            diagnostic_warnings.append(f"{recipe}: divergence after min_classify_length")
    metrics = {
        "seed": seed,
        "n_recipes": len(recipes),
        "n_pairs": len(pairs),
        "n_duplicate_pairs": sum(1 for r in rows if r["is_duplicate"]),
        "n_near_duplicate_pairs": sum(int(r.get("near_duplicate_count", 0)) for r in summary),
        "n_filtered_no_op_pairs": len(filtered_pairs),
        "filtered_no_op_pairs": filtered_pairs,
        "n_late_diverging_recipes": sum(1 for r in summary if not r.get("passes_early_divergence", True)),
        "recipe_family_jaccard_threshold": float(DEFAULT_CONFIG.jaccard_threshold),
        "recipe_family_jaccard_ambiguities": recipe_family_ambiguities,
        "n_recipe_family_jaccard_ambiguities": len(recipe_family_ambiguities),
        "summary": summary,
        "failure_reasons": failure_reasons,
        "diagnostic_warnings": diagnostic_warnings,
        "pass_preflight": bool(pass_preflight and not recipe_family_ambiguities and (not cfg_obj.fail_on_noop or not filtered_pairs)),
        "materiality_selection": selection_report,
        "materiality_report": {
            "filtered_pairs": filtered_pairs,
            "recipe_summary": summary,
            "recipe_family_jaccard_ambiguities": recipe_family_ambiguities,
            "failure_reasons": failure_reasons,
            "diagnostic_warnings": diagnostic_warnings,
            "selection": selection_report,
        },
    }
    if not metrics["pass_preflight"] and _ACTIVE_RESULT is not None:
        _ACTIVE_RESULT.warnings.append("materiality_preflight_failed")
    # Discriminability test: does the model actually produce different predictions
    # after training on P_A vs P_B for the same recipe?  A preference may be
    # materially distinct (Kendall tau > threshold) but completely indiscriminable
    # by the model architecture if the state features don't encode the distinction.
    # We test on a sample of recipes to avoid inflating runtime.
    discrim_rows: List[Dict[str, Any]] = []
    test_recipes = sorted(set(p.recipe_name for p in pairs))[:min(len(set(p.recipe_name for p in pairs)), 4)]
    pair_by_recipe: Dict[str, List[RecipePrefPair]] = defaultdict(list)
    for p in pairs:
        pair_by_recipe[p.recipe_name].append(p)
    for recipe in test_recipes:
        recipe_pairs = [p for p in pair_by_recipe[recipe] if not p.base_pref]
        if len(recipe_pairs) < 2:
            continue
        pa, pb = recipe_pairs[0], recipe_pairs[1]
        # Train two separate agents — one on pa, one on pb — then compare predictions.
        agent_a = make_agent("full", base_config(seed, run_config))
        ntr_a: Dict[str, str] = {}
        observe_episode(agent_a, pa, ntr_a)
        agent_b = make_agent("full", base_config(seed, run_config))
        ntr_b: Dict[str, str] = {}
        observe_episode(agent_b, pb, ntr_b)
        mid = max(1, len(pa.actions) // 2)
        prefix_a = tuple(agent_a._tokens_from_action_labels(pa.actions[:mid]))
        prefix_b = tuple(agent_b._tokens_from_action_labels(pb.actions[:mid]))
        rid_a = ntr_a.get(recipe)
        rid_b = ntr_b.get(recipe)
        dist_a = dict(agent_a._conditioned_distribution(list(prefix_a), rid_a, None)) if rid_a else {}
        dist_b = dict(agent_b._conditioned_distribution(list(prefix_b), rid_b, None)) if rid_b else {}
        if not dist_a or not dist_b:
            continue
        all_acts = set(dist_a) | set(dist_b)
        floor = 1e-12
        kl_ab = sum(
            max(dist_a.get(a, floor), floor) * math.log(max(dist_a.get(a, floor), floor) / max(dist_b.get(a, floor), floor))
            for a in all_acts
        )
        top1_a = max(dist_a, key=dist_a.get)
        top1_b = max(dist_b, key=dist_b.get)
        discrim_rows.append({
            "recipe": recipe,
            "preference_a": pa.preference_name,
            "preference_b": pb.preference_name,
            "kl_divergence_a_from_b": float(kl_ab),
            "top1_differ": top1_a != top1_b,
            "top1_a": top1_a,
            "top1_b": top1_b,
        })
    metrics["discriminability"] = {
        "rows": discrim_rows,
        "mean_kl_divergence": _mean([r["kl_divergence_a_from_b"] for r in discrim_rows]) if discrim_rows else float("nan"),
        "fraction_top1_differ": _mean([1.0 if r["top1_differ"] else 0.0 for r in discrim_rows]) if discrim_rows else float("nan"),
        "n_tested_recipes": len(discrim_rows),
        "interpretation": (
            "mean_kl_divergence > 0.1 and fraction_top1_differ > 0.5 indicate that "
            "preference differences are learnable by the model architecture. "
            "Low values mean the preference distinction is material in the data but "
            "not expressed in the prediction distribution — a model-side limitation."
        ),
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


def run_cl_regularizer_comparison(seed: int, out: Path, run_config: RunConfig, cfg_obj: ComputeTradeoffConfig) -> Dict[str, Any]:
    cl_run = replace(run_config, baselines=PAPER_BASELINES)
    return run_deployment_stream(seed, out, cl_run, DeploymentStreamConfig(n_recipes=cfg_obj.n_recipes, n_phase_b_events=cfg_obj.n_events))


# ---------------------------------------------------------------------------
# Experiment: catastrophic vs. selective forgetting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CatastrophicSelectiveConfig:
    """Config for the catastrophic-vs-selective forgetting experiment.

    Phase 1: observe recipe A with P1.
    Phase 2: add N distractor sessions (separate recipes).
    Phase 3: assist recipe A with P1 (retention) then with P2 (adaptation).

    Three distinct questions:
      Retention       – is A still correctly assisted after N distractors?
      Interference    – does adding distractors degrade frozen accuracy on A?
      Selective prune – after A is pruned, do its patterns bleed into other predictions?
    """
    n_distractors: int = 5
    n_recipes: int = 6
    gap_sessions: Tuple[int, ...] = (0, 3, 7, 12)
    baselines: Tuple[str, ...] = ()


def run_catastrophic_vs_selective_forgetting(
    seed: int, out: Path, run_config: RunConfig, cfg_obj: CatastrophicSelectiveConfig
) -> Dict[str, Any]:
    """Test catastrophic vs. selective forgetting."""
    target_specs = _target_variants(seed, cfg_obj.n_recipes, 1)
    if not target_specs:
        return {"status": "skip", "reason": "not_enough_recipes"}
    target, variant, distractors = target_specs[0]
    baselines = list(cfg_obj.baselines) or list(run_config.baselines)
    per_baseline: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        gap_rows: Dict[str, Any] = {}
        interference_rows: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        last_agent = None
        for gap in cfg_obj.gap_sessions:
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            observe_episode(agent, target, name_to_rid)
            observe_episode(agent, variant, name_to_rid)
            pre_eval = frozen_eval(agent, [target], name_to_rid, run_config)
            _advance_mixed_gap(agent, int(gap), int(gap), distractors, name_to_rid)
            post_eval = frozen_eval(agent, [target], name_to_rid, run_config)
            t_rid = name_to_rid.get(target.recipe_name)
            t_hash = variant_hash(agent._tokens_from_action_labels(target.actions))
            target_active = bool(t_rid and (t_rid, t_hash) in agent.decay.active)
            target_pruned = bool(t_rid and (t_rid, t_hash) in agent.decay.pruned)
            pre_top1 = float(pre_eval.get(target.label, {}).get("top1", 0.0))
            post_top1 = float(post_eval.get(target.label, {}).get("top1", 0.0))
            retention_row, retention_steps = assist_episode(
                agent, target, name_to_rid, run_config=run_config,
                commit=False, event_tag={"condition": "retention", "gap": gap},
            )
            new_pref_row: Dict[str, Any] = {}
            if variant.actions != target.actions:
                new_pref_row, _ = assist_episode(
                    agent, variant, name_to_rid, run_config=run_config,
                    commit=False, event_tag={"condition": "new_preference", "gap": gap},
                )
            interference_rows.append({
                "gap": gap, "pre_top1": pre_top1, "post_top1": post_top1,
                "interference_delta": post_top1 - pre_top1,
                "target_active_after_gap": target_active,
                "target_pruned_after_gap": target_pruned,
            })
            gap_rows[str(gap)] = {
                "gap": gap,
                "target_active_after_gap": target_active,
                "target_pruned_after_gap": target_pruned,
                "pre_distractor_frozen_top1": pre_top1,
                "post_distractor_frozen_top1": post_top1,
                "interference_delta": post_top1 - pre_top1,
                "retention_live_top1": float(retention_row.get("live_top1", 0.0)),
                "new_pref_live_top1": float(new_pref_row.get("live_top1", 0.0)) if new_pref_row else float("nan"),
                "retention_first_correct_step": next(
                    (int(s.get("step", 0)) for s in retention_steps if s.get("correct_top1")), -1
                ),
            }
            all_rows.append({"baseline": baseline, **gap_rows[str(gap)]})
            last_agent = agent
        per_baseline[baseline] = {
            "per_gap": gap_rows,
            "mean_retention_live_top1": _mean([r["retention_live_top1"] for r in gap_rows.values()]),
            "mean_interference_delta": _mean([r["interference_delta"] for r in gap_rows.values()]),
            "interference_rows": interference_rows,
            "compute": compute_snapshot(last_agent or make_agent(baseline, base_config(seed, run_config)), time.perf_counter() - t0),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_rows)
    metrics = {
        "seed": seed,
        "target": target.label,
        "variant": variant.label,
        "gap_sessions": list(cfg_obj.gap_sessions),
        "per_baseline": per_baseline,
        "caption": (
            "retention_live_top1: accuracy on known recipe after N distractor sessions. "
            "interference_delta: frozen accuracy drop on target after distractors (negative = forgetting). "
            "new_pref_live_top1: cold-start accuracy on a new preference after the gap."
        ),
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Experiment: adaptation speed curve
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptationSpeedConfig:
    """Config for the adaptation-speed-curve experiment.

    After observing recipe A/P1 (Phase A), the human performs A/P2 across
    n_assist_episodes sequential assist episodes with commit after each.
    Per-episode accuracy traces the online adaptation trajectory.
    """
    n_assist_episodes: int = 5
    n_recipes: int = 4
    baselines: Tuple[str, ...] = ()


def run_adaptation_speed_curve(
    seed: int, out: Path, run_config: RunConfig, cfg_obj: AdaptationSpeedConfig
) -> Dict[str, Any]:
    """Measure how quickly the agent adapts to a preference shift over sequential episodes.

    Primary outputs:
      cold_start_top1         – accuracy on episode 1 (no prior exposure to P2)
      final_episode_top1      – accuracy after n_assist_episodes corrective episodes
      episodes_to_above_50pct – first episode where live_top1 ≥ 0.50

    Compare oracle_preference_label (upper bound) vs. fixed_decay (no online learning)
    to bracket the achievable range.
    """
    target_specs = _target_variants(seed, cfg_obj.n_recipes, min(2, cfg_obj.n_recipes))
    if not target_specs:
        return {"status": "skip", "reason": "not_enough_recipes"}
    baselines = list(cfg_obj.baselines) or list(run_config.baselines)
    per_baseline: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        per_recipe_curves: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        last_agent = None
        for target, variant, _ in target_specs:
            if target.actions == variant.actions:
                continue
            agent = make_agent(baseline, base_config(seed, run_config))
            name_to_rid: Dict[str, str] = {}
            observe_episode(agent, target, name_to_rid)
            episode_curve: List[Dict[str, Any]] = []
            for ep_idx in range(int(cfg_obj.n_assist_episodes)):
                row, steps = assist_episode(
                    agent, variant, name_to_rid, run_config=run_config,
                    commit=False,
                    event_tag={
                        "condition": "adaptation_curve",
                        "episode_idx": ep_idx,
                        "original_pref": target.preference_name,
                        "new_pref": variant.preference_name,
                    },
                )
                agent.end_demo()
                episode_curve.append({
                    "episode_idx": ep_idx,
                    "live_top1": float(row.get("live_top1", 0.0)),
                    "live_topk": float(row.get("live_topk", 0.0)),
                    "wrong_prediction_future_valid_rate": float(row.get("wrong_prediction_future_valid_rate", 0.0)),
                    "human_correction_rate": float(row.get("human_correction_rate", 0.0)),
                    "human_action_fraction": float(row.get("human_action_fraction", 0.0)),
                    "human_effort_fraction": float(row.get("human_effort_fraction", 0.0)),
                    "human_workload_reduction_vs_human_only": float(row.get("human_workload_reduction_vs_human_only", 0.0)),
                    "n_steps": float(row.get("n_steps", 0)),
                })
                all_episode_rows.append({
                    "baseline": baseline,
                    "recipe": target.recipe_name,
                    "episode_idx": ep_idx,
                    **row,
                })
            per_recipe_curves.append({
                "recipe": target.recipe_name,
                "original_pref": target.preference_name,
                "new_pref": variant.preference_name,
                "episode_curve": episode_curve,
                "cold_start_top1": float(episode_curve[0]["live_top1"]) if episode_curve else float("nan"),
                "final_episode_top1": float(episode_curve[-1]["live_top1"]) if episode_curve else float("nan"),
                "episodes_to_above_50pct": next(
                    (e["episode_idx"] for e in episode_curve if e["live_top1"] >= 0.50), -1
                ),
            })
            last_agent = agent
        per_baseline[baseline] = {
            "per_recipe": per_recipe_curves,
            "mean_cold_start_top1": _mean([r["cold_start_top1"] for r in per_recipe_curves if not math.isnan(r["cold_start_top1"])]),
            "mean_final_episode_top1": _mean([r["final_episode_top1"] for r in per_recipe_curves if not math.isnan(r["final_episode_top1"])]),
            "mean_episodes_to_above_50pct": _mean([float(r["episodes_to_above_50pct"]) for r in per_recipe_curves if r["episodes_to_above_50pct"] >= 0]),
            "compute": compute_snapshot(last_agent or make_agent(baseline, base_config(seed, run_config)), time.perf_counter() - t0),
        }
    _append_jsonl(out / "episode_metrics.jsonl", all_episode_rows)
    metrics = {
        "seed": seed,
        "n_assist_episodes": cfg_obj.n_assist_episodes,
        "per_baseline": per_baseline,
        "caption": (
            "cold_start_top1 is episode-1 accuracy under the new preference (no prior exposure). "
            f"final_episode_top1 is accuracy after {cfg_obj.n_assist_episodes} corrective episodes. "
            "Bracket against oracle_preference_label (ceiling) and fixed_decay (no learning floor)."
        ),
    }
    _write_json(out / "metrics.json", metrics)
    return metrics


EXPERIMENT_RUNNERS: Dict[str, Callable[[int, Path, RunConfig, Any], Dict[str, Any]]] = {
    "deployment_stream": run_deployment_stream,
    "deployment_ladder_heterogeneous_prefs": run_deployment_ladder_heterogeneous_prefs,
    "deployment_ladder_shared_pref": run_deployment_ladder_shared_pref,
    "cross_recipe_transfer": run_cross_recipe_transfer_suite,
    "confidence_threshold_accuracy": run_confidence_threshold_accuracy,
    "decay_reentry": run_decay_reentry_suite,
    "disambiguation_audit": run_disambiguation_audit,
    "materiality_preflight": run_materiality_preflight,
    "cl_regularizer_comparison": run_cl_regularizer_comparison,
    "single_shot_reuse": run_single_shot_reuse,
    "deployment_gate_preference_shift": run_deployment_gate_preference_shift,
    "short_term_capacity_sweep": run_short_term_capacity_sweep,
    "demo_count_sample_efficiency": run_demo_count_sample_efficiency,
    "frequency_gap_decay_sweep": run_frequency_gap_decay_sweep,
    "mwr_window_sensitivity": run_mwr_window_sensitivity,
    "sparse_first_exposure_pool_sweep": run_sparse_first_exposure_pool_sweep,
    "zipf_usage_sweep": run_zipf_usage_sweep,
    "cycle_width_sparsity_sweep": run_cycle_width_sparsity_sweep,
    "baseline_anchor_sweep": run_baseline_anchor_sweep,
    "continual_learning_regularizer_sweep": run_continual_learning_regularizer_sweep,
    "posterior_ablation_matrix": run_posterior_ablation_matrix,
    "recipe_preference_factor_ablation": run_recipe_preference_factor_ablation,
    "memory_exhaustion_stress": run_memory_exhaustion_stress,
    "prefix_collision_stress": run_prefix_collision_stress,
    "preference_thrash_stress": run_preference_thrash_stress,
    "rare_reentry_stress": run_rare_reentry_stress,
    "late_distractor_stress": run_late_distractor_stress,
    "adversarial_stress_suite": run_adversarial_stress_suite,
    "baseline_hyperparameter_tuning": run_baseline_hyperparameter_tuning,
    "seed_recipe_selection_audit": run_seed_recipe_selection_audit,
    "runtime_scaling_sweep": run_runtime_scaling_sweep,
    "catastrophic_vs_selective_forgetting": run_catastrophic_vs_selective_forgetting,
    "adaptation_speed_curve": run_adaptation_speed_curve,
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
    try:
        run_summary["artifact_guides"] = generate_artifact_guides(root)
        _write_json_file(aggregate_root / "run.json", run_summary)
    except Exception as exc:
        run_summary["artifact_guide_error"] = str(exc)
        _write_json_file(aggregate_root / "run.json", run_summary)
    return run_summary


def _resolve_experiments(experiments: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if experiments is None:
        return EXPERIMENTS
    if isinstance(experiments, str):
        if experiments in {"all", "experiments"}:
            return EXPERIMENTS
        return (experiments,)
    return tuple(experiments)


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)):
        return "unknown"
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _parallel_batch_duration(wall_times: Sequence[float], workers: int) -> float:
    """Estimate elapsed wall time for seed jobs packed onto `workers` slots."""
    slots = [0.0] * max(1, int(workers))
    for wall_s in sorted((float(v) for v in wall_times if isinstance(v, (int, float)) and math.isfinite(float(v))), reverse=True):
        idx = min(range(len(slots)), key=lambda i: slots[i])
        slots[idx] += max(0.0, wall_s)
    return float(max(slots) if slots else 0.0)


_STATUS_STATE_RE = re.compile(r'"state"\s*:\s*"([^"]+)"')
_STATUS_WALL_RE = re.compile(r'"wall_s"\s*:\s*([0-9]+(?:\.[0-9]+)?)')


def _read_result_status_fast(result_path: Path) -> Dict[str, Any]:
    try:
        path = Path(result_path)
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            head = handle.read(8192)
            try:
                handle.seek(max(0, path.stat().st_size - 8192))
                head += handle.read(8192)
            except OSError:
                pass
    except Exception:
        return {}
    state = _STATUS_STATE_RE.search(head)
    wall_s = _STATUS_WALL_RE.search(head)
    if not state or not wall_s:
        return {}
    return {"state": state.group(1), "wall_s": float(wall_s.group(1))}


def _historical_experiment_duration_estimates(output_root: str | Path, experiments: Sequence[str], workers: int, n_seeds: Optional[int] = None) -> Dict[str, float]:
    """Median per-experiment batch duration from prior result.json files.

    The suite runs experiments sequentially, but seed jobs inside one experiment
    usually run in parallel. Historical ETA therefore uses the packed seed-batch
    duration for each experiment rather than treating every seed as serial work.
    """
    root = Path(output_root)
    wanted = set(str(e) for e in experiments)
    samples: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    if not root.exists():
        return {}
    candidate_roots = [p for p in root.iterdir() if p.is_dir()]
    if any((root / exp).is_dir() for exp in wanted):
        candidate_roots.append(root)
    for run_root in candidate_roots:
        for exp in wanted:
            exp_dir = run_root / exp
            if not exp_dir.is_dir():
                continue
            seed_walls: List[float] = []
            for result_path in exp_dir.glob("seed_*/result.json"):
                status = _read_result_status_fast(result_path)
                if status.get("state") != "completed":
                    continue
                wall_s = status.get("wall_s")
                if isinstance(wall_s, (int, float)) and math.isfinite(float(wall_s)):
                    seed_walls.append(float(wall_s))
            if seed_walls:
                samples[exp].append((len(seed_walls), _parallel_batch_duration(seed_walls, workers)))
    estimates: Dict[str, float] = {}
    for exp, rows in samples.items():
        vals = [duration for count, duration in rows if n_seeds is None or int(count) == int(n_seeds)]
        if not vals:
            vals = [duration for _count, duration in rows]
        med = _median(vals[-5:])
        if med is not None:
            estimates[exp] = float(med)
    return estimates


def _suite_eta_status(
    names: Sequence[str],
    current_index: int,
    suite_start_s: float,
    experiment_start_s: float,
    completed_experiment_wall_s: Mapping[str, float],
    historical_estimates: Mapping[str, float],
) -> str:
    now = time.perf_counter()
    elapsed = now - suite_start_s
    current_name = str(names[current_index]) if 0 <= current_index < len(names) else ""
    current_elapsed = now - experiment_start_s
    current_est = historical_estimates.get(current_name)
    current_remaining = max(0.0, float(current_est) - current_elapsed) if current_est is not None else 0.0
    fallback_pool = list(historical_estimates.values()) or list(completed_experiment_wall_s.values())
    fallback = _median(fallback_pool)
    future_known = 0
    future_missing = 0
    future_remaining = 0.0
    for future_name in names[current_index + 1:]:
        est = historical_estimates.get(str(future_name))
        if est is None:
            future_missing += 1
            if fallback is not None:
                future_remaining += float(fallback)
        else:
            future_known += 1
            future_remaining += float(est)
    eta = current_remaining + future_remaining
    detail = f"history={future_known}/{max(0, len(names) - current_index - 1)} future"
    if future_missing and fallback is None:
        detail += f", {future_missing} unknown"
        return f"elapsed={_format_seconds(elapsed)} ETA=unknown ({detail})"
    if future_missing:
        detail += f", {future_missing} fallback"
    return f"elapsed={_format_seconds(elapsed)} ETA={_format_seconds(eta)} ({detail})"


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
    _recipes, pairs, _selection_report = _materiality_preflight_pairs(seed, cfg_obj)
    rows = materiality_rows(pairs)
    pair_by_label = {p.label: p for p in pairs}
    by_recipe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    filtered_noops: List[str] = []
    for row in rows:
        by_recipe[row["recipe"]].append(row)
        pair = pair_by_label.get(row["label"])
        if pair is not None and not pair.base_pref and not pair.applied_axes:
            filtered_noops.append(pair.label)
    summary: List[Dict[str, Any]] = []
    for recipe, vals in sorted(by_recipe.items()):
        recipe_pairs = [pair_by_label[v["label"]] for v in vals if v["label"] in pair_by_label]
        deduped, duplicate_pairs = _deduplicate_preference_pairs(
            recipe_pairs,
            tau_threshold=float(cfg_obj.near_duplicate_tau_threshold),
        )
        divergence_rows = [
            earliest_divergence_step(a.actions, b.actions)
            for i, a in enumerate(recipe_pairs)
            for b in recipe_pairs[i + 1:]
            if a.actions != b.actions
        ]
        late_divergence_cutoff = max(1, min((len(p.actions) for p in recipe_pairs), default=1) // 3)
        min_divergence = min(divergence_rows, default=-1)
        summary.append({
            "recipe": recipe,
            "n_effective_preferences": len(deduped),
            "near_duplicate_count": len(duplicate_pairs),
            "passes_min_effective": len(deduped) >= int(cfg_obj.min_effective_preferences),
            "passes_duplicate_ordering": not (cfg_obj.fail_on_duplicate_ordering and duplicate_pairs),
            "passes_early_divergence": not (min_divergence >= 0 and min_divergence > late_divergence_cutoff),
            "passes_min_classify_divergence": not any(step > int(cfg_obj.min_classify_length) for step in divergence_rows),
        })
    recipe_family_ambiguities = _recipe_family_jaccard_ambiguities(
        [p for p in pairs if p.preference_name == "identity"],
        float(DEFAULT_CONFIG.jaccard_threshold),
    )
    return _materiality_failure_reasons(summary, filtered_noops, recipe_family_ambiguities, cfg_obj)


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
        if name == "deployment_stream":
            scenario_warnings = set(_scenario_warnings(collector._scenario_summary()))
            critical = {
                "scenario_has_no_seen_recipe_new_preference_events",
                "scenario_has_no_recorded_events",
            }
            failing = critical & scenario_warnings
            if failing:
                raise RuntimeError(
                    f"Seed {seed}: deployment stream missing critical conditions: {sorted(failing)}."
                )
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


def run_suite(
    run_config: Optional[RunConfig] = None,
    suite_config: Optional[ExperimentSuiteConfig] = None,
    experiments: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    run_config = run_config or RunConfig()
    suite_config = suite_config or ExperimentSuiteConfig()
    names = _resolve_experiments(experiments)
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
        "experiments": EXPERIMENTS,
        "requested_experiments": names,
        "selected_recipes_by_seed": {str(seed): [r for r, _ in stable_recipe_builders(seed)] for seed in run_config.seeds},
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "workers": workers,
        "status": {"state": "running", "completed": [], "partial": [], "failed": []},
    }
    _write_json_file(run_dir / "run.json", manifest)
    preflight_required = {
        "deployment_stream",
        "cross_recipe_transfer",
        "confidence_threshold_accuracy",
        "decay_reentry",
        "disambiguation_audit",
    }
    if any(name in preflight_required for name in names):
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
                "Materiality preflight failed before materiality-sensitive experiment execution: "
                f"{failures_by_seed}. Fix the recipe library or lower min_effective_preferences."
            )
    completed: List[str] = []
    partial: List[str] = []
    failures: Dict[str, str] = {}
    total_jobs = len(names) * len(run_config.seeds)
    done_jobs = 0
    suite_t0 = time.perf_counter()
    historical_eta = _historical_experiment_duration_estimates(run_config.output_root, names, workers, len(run_config.seeds))
    completed_experiment_wall_s: Dict[str, float] = {}
    for exp_index, name in enumerate(names):
        exp_dir = _ensure(run_dir / name)
        cfg_obj = _experiment_config(suite_config, name)
        exp_t0 = time.perf_counter()
        estimated = historical_eta.get(str(name))
        start_eta = _suite_eta_status(names, exp_index, suite_t0, exp_t0, completed_experiment_wall_s, historical_eta)
        estimate_text = f", historical batch estimate={_format_seconds(estimated)}" if estimated is not None else ", historical batch estimate=unknown"
        print(f"[{name}] start: {len(run_config.seeds)} seeds, workers={workers}{estimate_text}; {start_eta}", flush=True)
        seed_results: List[Dict[str, Any]] = []
        if workers <= 1 or len(run_config.seeds) <= 1:
            for seed in run_config.seeds:
                r = _run_seed_job(name, int(seed), str(run_dir), run_config, cfg_obj)
                seed_results.append(r)
                done_jobs += 1
                state = "done" if r.get("ok") else "FAILED"
                eta_status = _suite_eta_status(names, exp_index, suite_t0, exp_t0, completed_experiment_wall_s, historical_eta)
                print(f"  [{name}] seed {seed} {state} in {_format_seconds(r.get('wall_s', 0.0))} ({done_jobs}/{total_jobs}, {eta_status})", flush=True)
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
                    eta_status = _suite_eta_status(names, exp_index, suite_t0, exp_t0, completed_experiment_wall_s, historical_eta)
                    print(f"  [{name}] seed {seed} {state} in {_format_seconds(r.get('wall_s', 0.0))} ({done_jobs}/{total_jobs}, {eta_status})", flush=True)
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
        completed_experiment_wall_s[name] = float(time.perf_counter() - exp_t0)
        manifest["status"] = {"state": "running", "completed": completed, "partial": partial, "failed": sorted(failures)}
        _write_json_file(run_dir / "run.json", manifest)
        done_eta = _suite_eta_status(names, exp_index + 1, suite_t0, time.perf_counter(), completed_experiment_wall_s, historical_eta)
        print(f"[{name}] {state} in {_format_seconds(completed_experiment_wall_s[name])}; {done_eta}", flush=True)
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


def main(
    *,
    run_config: Optional[RunConfig] = None,
    suite_config: Optional[ExperimentSuiteConfig] = None,
    experiments: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return run_suite(run_config=run_config, suite_config=suite_config, experiments=experiments)


__all__ = [
    "RunConfig",
    "ExperimentSuiteConfig",
    "EXPERIMENTS",
    "ORACLE_REFERENCE_BASELINES",
    "DeploymentStreamConfig",
    "DeploymentLadderConfig",
    "TransferSuiteConfig",
    "ConfidenceThresholdAccuracyConfig",
    "DecayReentrySuiteConfig",
    "DisambiguationAuditConfig",
    "MaterialityPreflightConfig",
    "BaselineHyperparameterTuningConfig",
    "LiveStepRecord",
    "LiveEpisodeRecord",
    "binary_decision_metrics",
    "live_episode_metrics",
    "run_suite",
    "aggregate_run",
    "generate_artifact_guides",
    "main",
]
