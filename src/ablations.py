"""Ablations and stress tests for recipe/preference disambiguation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .environment import parse_action_label, recipe_builders, task_goal_signature, validate_ordering
from .memory import (
    MatchResult,
    RecipeMatcher,
    KnownVariant,
    _jaccard_counters,
    jaccard,
    kendall_tau_distance,
    make_variant_id,
)
from .models import Settings, DEFAULT_SETTINGS
from .preferences import PREFERENCE_IDS, apply_preset


KNOWN = "known"
PREFERENCE_SHIFT = "preference_shift"
NEW_RECIPE = "new_recipe"
TENTATIVE_PREFERENCE_SHIFT = "tentative_preference_shift"

EXPECTED_CLASSES = (KNOWN, PREFERENCE_SHIFT, NEW_RECIPE)

COMMIT_NONE = "none"
COMMIT_TENTATIVE = "tentative"
COMMIT_FULL = "full"
COMMIT_PROMOTION = "promotion"
COMMIT_KNOWN_REFRESH = "known_refresh"
COMMIT_DECISIONS = (
    COMMIT_NONE,
    COMMIT_TENTATIVE,
    COMMIT_FULL,
    COMMIT_PROMOTION,
    COMMIT_KNOWN_REFRESH,
)


SHARED_ROUTE = "shared"
LOCAL_ROUTE = "local"
REFERENCE_ROUTE = "reference"
ROUTING_BASELINES: Tuple[str, ...] = (
    "full",
    "unpinned",
    "latest",
    "fixed",
    "no_decay",
    "bc",
    "ewc",
    "replay_bc",
)
DEFAULT_ABLATION_RESULTS_ROOT = "eval_results/ablation_runs"
# Suites run one at a time, so a suite may use the whole performance-core budget for its own scenario-seed grid. 
DEFAULT_ABLATION_WORKERS = 8

PREDICTOR_LEVELS: Tuple[str, str] = ("maxent", "behavior_cloning")
MEMORY_LEVELS: Tuple[str, str] = ("adaptive_pinned", "retain_all")
MEMORY_PREDICTOR_ABLATION_CELLS: Tuple[Dict[str, Any], ...] = (
    {"arm": "full", "baseline": "full", "label": "MaxEnt + adaptive memory",
     "predictor": "maxent", "memory": "adaptive_pinned", "overrides": {}},
    {"arm": "bc_adaptive", "baseline": "bc_adaptive", "label": "BC + adaptive memory",
     "predictor": "behavior_cloning", "memory": "adaptive_pinned", "overrides": {}},
    {"arm": "maxent_retain_all", "baseline": "full", "label": "MaxEnt + retain-all",
     "predictor": "maxent", "memory": "retain_all",
     "overrides": {"retention_policy": "none", "pin_latest": False}},
    {"arm": "bc", "baseline": "bc", "label": "BC + retain-all",
     "predictor": "behavior_cloning", "memory": "retain_all", "overrides": {}},
)
MEMORY_PREDICTOR_ABLATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("teacher_forced_top_1", "higher_is_better"),
    ("teacher_forced_top_k", "higher_is_better"),
    ("teacher_forced_mean_nll", "lower_is_better"),
    ("live_top_1", "higher_is_better"),
    ("normalized_human_action_load", "lower_is_better"),
    ("corrections_per_recipe_step", "lower_is_better"),
)


@dataclass(frozen=True)
class LatentStrategyAblationArm:
    """One controlled MaxEnt/latent-strategy ablation condition. Correction-confirmation gating remains part of the shared agent protocol;
    it is deliberately not varied here.
    """

    name: str
    label: str
    latent_strategy_enabled: bool
    latent_strategy_rank: int
    latent_strategy_knn: int
    latent_strategy_sequence_weight: float

    def model_overrides(self) -> Dict[str, Any]:
        return {
            "latent_strategy_enabled": bool(self.latent_strategy_enabled),
            "latent_strategy_rank": int(self.latent_strategy_rank),
            "latent_strategy_knn": int(self.latent_strategy_knn),
            "latent_strategy_strength": 1.0,
            "latent_strategy_sequence_weight": float(
                self.latent_strategy_sequence_weight
            ),
        }

    def settings(self, base: Settings = DEFAULT_SETTINGS) -> Settings:
        """Apply this arm without changing any non-latent model setting."""
        return replace(base, **self.model_overrides())


LATENT_STRATEGY_ABLATION_ARMS: Tuple[LatentStrategyAblationArm, ...] = (
    LatentStrategyAblationArm(
        name="maxent_only",
        label="MaxEnt only",
        latent_strategy_enabled=False,
        latent_strategy_rank=8,
        latent_strategy_knn=3,
        latent_strategy_sequence_weight=0.0,
    ),
    LatentStrategyAblationArm(
        name="latent_timing_lightweight",
        label="Lightweight latent timing",
        latent_strategy_enabled=True,
        latent_strategy_rank=8,
        latent_strategy_knn=3,
        latent_strategy_sequence_weight=0.0,
    ),
    LatentStrategyAblationArm(
        name="latent_timing_capacity_matched",
        label="Capacity-matched latent timing",
        latent_strategy_enabled=True,
        latent_strategy_rank=32,
        latent_strategy_knn=8,
        latent_strategy_sequence_weight=0.0,
    ),
    LatentStrategyAblationArm(
        name="latent_timing_trajectory_hybrid",
        label="Heavy timing-trajectory hybrid",
        latent_strategy_enabled=True,
        latent_strategy_rank=32,
        latent_strategy_knn=8,
        latent_strategy_sequence_weight=0.5,
    ),
)


LATENT_STRATEGY_ABLATION_CONTRASTS: Tuple[Mapping[str, str], ...] = (
    {
        "name": "lightweight_latent_contribution",
        "reference": "maxent_only",
        "treatment": "latent_timing_lightweight",
        "interpretation": (
            "Contribution of low-cost cross-recipe latent timing over MaxEnt."
        ),
    },
    {
        "name": "latent_capacity_sensitivity",
        "reference": "latent_timing_lightweight",
        "treatment": "latent_timing_capacity_matched",
        "interpretation": (
            "Accuracy/compute sensitivity to rank and neighbor count; this is "
            "not an architectural-component contrast."
        ),
    },
    {
        "name": "trajectory_alignment_contribution",
        "reference": "latent_timing_capacity_matched",
        "treatment": "latent_timing_trajectory_hybrid",
        "interpretation": (
            "Causal contribution of partial-trajectory alignment because rank, "
            "KNN, strength, correction gating, and all non-latent settings match."
        ),
    },
)


def latent_strategy_ablation_design() -> Dict[str, Any]:
    """Return the auditable four-arm design used by the longitudinal runner."""
    return {
        "arms": [
            {
                "name": arm.name,
                "label": arm.label,
                "model_overrides": arm.model_overrides(),
            }
            for arm in LATENT_STRATEGY_ABLATION_ARMS
        ],
        "controlled_invariants": {
            "base_predictor": "MaxEnt IRL",
            "latent_activation": "only_after_human_correction_support",
            "interaction_schedule": "paired_within_scenario_and_seed",
            "non_latent_model_settings": "identical_across_arms",
        },
        "planned_contrasts": [
            dict(contrast) for contrast in LATENT_STRATEGY_ABLATION_CONTRASTS
        ],
        "invalid_causal_contrast": (
            "Do not attribute the lightweight-to-hybrid difference solely to "
            "trajectory alignment because capacity changes simultaneously."
        ),
    }


# --- predictor-representation ablation -------------------------------------
# The memory policy is held at Full's adaptive decay with the latest pin in
# every arm, so the only factor that moves is what the predictor is allowed to
# --------------------------------------------------------------------------
# Component ablation: which part of Full's predictor support earns its place.
COMPONENT_LEVELS: Tuple[str, ...] = (
    "all", "no_latest_pin", "no_semantic_fallback", "no_latent_residual", "none",
)


@dataclass(frozen=True)
class ComponentAblationArm:
    """One predictor-support condition, holding retention at Full's policy."""

    name: str
    label: str
    component_level: str
    removes: Tuple[str, ...]
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def model_overrides(self) -> Dict[str, Any]:
        return dict(self.overrides)

    def settings(self, base: Settings = DEFAULT_SETTINGS) -> Settings:
        """Apply this arm without changing any non-component setting."""
        return replace(base, **self.model_overrides())


COMPONENT_ABLATION_ARMS: Tuple[ComponentAblationArm, ...] = (
    ComponentAblationArm(
        name="full",
        label="Full",
        component_level="all",
        removes=(),
        overrides={},
    ),
    ComponentAblationArm(
        name="full_no_pin",
        label="Full without the latest pin",
        component_level="no_latest_pin",
        removes=("latest_pin",),
        overrides={"pin_latest": False},
    ),
    ComponentAblationArm(
        name="full_no_semantic_fallback",
        label="Full without the semantic fallback",
        component_level="no_semantic_fallback",
        removes=("semantic_fallback",),
        overrides={"semantic_fallback_enabled": False},
    ),
    ComponentAblationArm(
        name="full_no_latent_residual",
        label="Full without the latent residual",
        component_level="no_latent_residual",
        removes=("latent_residual",),
        overrides={"latent_strategy_enabled": False},
    ),
    ComponentAblationArm(
        name="full_no_support",
        label="Full without any predictor support",
        component_level="none",
        removes=("latest_pin", "semantic_fallback", "latent_residual"),
        overrides={
            "pin_latest": False,
            "semantic_fallback_enabled": False,
            "latent_strategy_enabled": False,
        },
    ),
)


# --------------------------------------------------------------------------
# Retention ablation: which part of the recurrence-aware memory policy works.
RETENTION_FACTORS: Tuple[str, ...] = (
    "reference", "retention_rule", "horizon_estimator", "pair_adaptation",
    "horizon_assignment", "pin_scope", "consolidation_schedule",
)


@dataclass(frozen=True)
class RetentionAblationArm:
    """One retention-mechanism condition under Full's predictor support."""

    name: str
    label: str
    factor: str
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def model_overrides(self) -> Dict[str, Any]:
        return dict(self.overrides)

    def settings(self, base: Settings = DEFAULT_SETTINGS) -> Settings:
        """Apply this arm without changing any non-retention setting."""
        return replace(base, **self.model_overrides())


RETENTION_ABLATION_ARMS: Tuple[RetentionAblationArm, ...] = (
    RetentionAblationArm(
        name="full",
        label="Full",
        factor="reference",
        overrides={},
    ),
    RetentionAblationArm(
        name="full_retain_all",
        label="Full, retain everything",
        factor="retention_rule",
        overrides={"retention_policy": "none"},
    ),
    RetentionAblationArm(
        name="constant_grace",
        label="Constant grace horizon",
        factor="horizon_estimator",
        overrides={"horizon_estimator": "constant"},
    ),
    RetentionAblationArm(
        name="pair_only_horizon",
        label="Pair evidence only, no parent pooling",
        factor="horizon_estimator",
        overrides={"horizon_estimator": "pair_only"},
    ),
    RetentionAblationArm(
        name="shuffled_horizon",
        label="Recurrence evidence reassigned across pairs",
        factor="horizon_assignment",
        overrides={"horizon_estimator": "shuffled"},
    ),
    RetentionAblationArm(
        name="symmetric_adaptation",
        label="Symmetric horizon adaptation",
        factor="pair_adaptation",
        overrides={"pair_adaptation": "symmetric"},
    ),
    RetentionAblationArm(
        name="recent_set_pin",
        label="Pin the recently active preference set",
        factor="pin_scope",
        overrides={"pin_mode": "recent_set"},
    ),
    RetentionAblationArm(
        name="schedule_warm_on_weight_change",
        label="Warm fit on every weight change",
        factor="consolidation_schedule",
        overrides={"retrain_warm_on_weight_change": True},
    ),
    RetentionAblationArm(
        name="schedule_removals_count_cold",
        label="Removals advance the cold-start counter",
        factor="consolidation_schedule",
        overrides={"retrain_cold_counts_removals": True},
    ),
)


# see. Two families are varied on their own axis: the reward representation
# the MaxEnt policy scores states with, and the within-episode history the
# cloner conditions on. That separates "this model family is weaker" from
# "this model family cannot see the thing that disambiguates the decision".
REPRESENTATION_FAMILIES: Tuple[str, str] = ("maxent", "behavior_cloning")


@dataclass(frozen=True)
class RepresentationAblationArm:
    """One predictor-representation condition under Full's memory policy.

    ``within_episode_history`` states, in the arm itself, how much of the
    current episode the predictor can condition on. A MaxEnt policy indexes
    its value function by environment state, so it reads ``none`` whatever its
    reward representation; the cloner reads whatever its prefix block encodes.
    Preferences here are goal-preserving reorderings, so two variants of one
    recipe pass through the same states and diverge on history alone -- which
    makes this column, rather than the model family, the quantity the ablation
    is about.
    """

    name: str
    label: str
    baseline: str
    family: str
    representation: str
    within_episode_history: str
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def model_overrides(self) -> Dict[str, Any]:
        return dict(self.overrides)

    def settings(self, base: Settings = DEFAULT_SETTINGS) -> Settings:
        """Apply this arm without changing any other model setting."""
        return replace(base, **self.model_overrides())


REPRESENTATION_ABLATION_ARMS: Tuple[RepresentationAblationArm, ...] = (
    RepresentationAblationArm(
        name="maxent_engineered",
        label="MaxEnt, engineered reward features",
        baseline="full", family="maxent",
        representation="engineered_reward_features",
        within_episode_history="none",
        overrides={"irl_features": "engineered"},
    ),
    RepresentationAblationArm(
        name="maxent_semantic",
        label="MaxEnt, semantic reward features",
        baseline="full", family="maxent",
        representation="semantic_reward_features",
        within_episode_history="none",
        overrides={"irl_features": "semantic"},
    ),
    RepresentationAblationArm(
        name="maxent_raw_state",
        label="MaxEnt, raw state features",
        baseline="full", family="maxent",
        representation="raw_state_features",
        within_episode_history="none",
        overrides={"irl_features": "raw_state"},
    ),
    RepresentationAblationArm(
        name="cloner_history_3",
        label="Cloner, three action lags and step count",
        baseline="bc_adaptive", family="behavior_cloning",
        representation="state_plus_action_history",
        within_episode_history="three_action_lags_and_step_count",
        overrides={"bc_history": 3, "bc_prefix_length_feature": True},
    ),
    RepresentationAblationArm(
        name="cloner_history_1",
        label="Cloner, one action lag and step count",
        baseline="bc_adaptive", family="behavior_cloning",
        representation="state_plus_action_history",
        within_episode_history="one_action_lag_and_step_count",
        overrides={"bc_history": 1, "bc_prefix_length_feature": True},
    ),
    RepresentationAblationArm(
        name="cloner_step_count_only",
        label="Cloner, step count only",
        baseline="bc_adaptive", family="behavior_cloning",
        representation="state_plus_step_count",
        within_episode_history="step_count_only",
        overrides={"bc_history": 0, "bc_prefix_length_feature": True},
    ),
    RepresentationAblationArm(
        name="cloner_state_only",
        label="Cloner, state only",
        baseline="bc_adaptive", family="behavior_cloning",
        representation="state_only",
        within_episode_history="none",
        overrides={"bc_history": 0, "bc_prefix_length_feature": False},
    ),
)

REPRESENTATION_ABLATION_CONTRASTS: Tuple[Mapping[str, str], ...] = (
    {
        "name": "cloner_history_contribution",
        "reference": "cloner_state_only",
        "treatment": "cloner_history_3",
        "interpretation": (
            "Causal contribution of within-episode action history to the "
            "cloner, with the memory policy, the stored demonstrations and "
            "every other setting matched. The ablation's primary contrast."
        ),
    },
    {
        "name": "cloner_action_identity_contribution",
        "reference": "cloner_step_count_only",
        "treatment": "cloner_history_3",
        "interpretation": (
            "Separates knowing which actions were taken from knowing how far "
            "the episode has progressed."
        ),
    },
    {
        "name": "maxent_reward_representation_sensitivity",
        "reference": "maxent_engineered",
        "treatment": "maxent_raw_state",
        "interpretation": (
            "Bounds how much of the MaxEnt gap the reward representation can "
            "account for; all three MaxEnt arms stay history-free."
        ),
    },
    {
        "name": "history_free_family_contrast",
        "reference": "maxent_engineered",
        "treatment": "cloner_state_only",
        "interpretation": (
            "Model family compared at matched access to history. A small gap "
            "here alongside a large history contribution locates the "
            "difference in the representation rather than the learner. "
            "Descriptive: the two families do not carry identical components."
        ),
    },
)

REPRESENTATION_ABLATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("teacher_forced_top_1", "higher_is_better"),
    ("teacher_forced_top_k", "higher_is_better"),
    ("teacher_forced_mean_nll", "lower_is_better"),
    ("live_top_1", "higher_is_better"),
    ("normalized_human_action_load", "lower_is_better"),
    ("corrections_per_recipe_step", "lower_is_better"),
    ("unseen_state_fallback_top_1", "higher_is_better"),
    ("branching_top_1", "higher_is_better"),
    ("single_option_lookup_top_1", "higher_is_better"),
    ("direct_retrieval_top_1", "higher_is_better"),
    ("new_preference_top_1", "higher_is_better"),
    ("preference_seen_elsewhere_top_1", "higher_is_better"),
    ("coexistence_degradation", "lower_is_better"),
)


def representation_ablation_design() -> Dict[str, Any]:
    """Return the auditable arm roster and contrasts used by the runner."""
    return {
        "families": list(REPRESENTATION_FAMILIES),
        "memory_policy_held_fixed": "adaptive_decay_with_latest_pin",
        "arms": [
            {
                "name": arm.name,
                "label": arm.label,
                "baseline": arm.baseline,
                "family": arm.family,
                "representation": arm.representation,
                "within_episode_history": arm.within_episode_history,
                "model_overrides": arm.model_overrides(),
            }
            for arm in REPRESENTATION_ABLATION_ARMS
        ],
        "contrasts": [
            dict(contrast) for contrast in REPRESENTATION_ABLATION_CONTRASTS
        ],
        "metrics": [
            {"metric": metric, "direction": direction}
            for metric, direction in REPRESENTATION_ABLATION_METRICS
        ],
        "component_parity_note": (
            "Full's semantic value fallback and latent-strategy residual are "
            "reached through the MaxEnt head, so they are present in the "
            "MaxEnt arms and absent from the cloner arms. Every within-family "
            "contrast above is component-matched; the cross-family contrast "
            "is not, and is reported as descriptive."
        ),
    }


@dataclass(frozen=True)
class MatcherSettings:
    """Configuration for the standalone matcher stress suite."""

    seed: int = 1337
    recipe_count: int = 30
    preferences: Tuple[str, ...] = tuple(p for p in PREFERENCE_IDS if p != "default")
    case_limit: int = 120
    omission_sizes: Tuple[int, ...] = (1, 2)
    repeat_sizes: Tuple[int, ...] = (1,)
    substitution_sizes: Tuple[int, ...] = (1,)
    overlap_targets: Tuple[float, ...] = (0.80, 0.90, 0.95, 0.9583, 0.96, 0.98)
    overlap_repeats: int = 8
    prefix_lengths: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    include_prefixes: bool = True
    prefix_case_limit: int = 16
    calibrate: bool = True
    calibration_seed_offset: int = 104729
    calibration_case_limit: int = 8


@dataclass(frozen=True)
class MatcherVariant:
    """One stored recipe/preference variant in matcher-friendly form."""

    recipe_id: str
    preference_id: str
    actions: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def variant_id(self) -> str:
        return make_variant_id(self.actions)

    def to_known_variant(self) -> KnownVariant:
        return KnownVariant(
            recipe_id=self.recipe_id,
            variant_id=self.variant_id,
            ordering=tuple(self.actions),
        )


@dataclass(frozen=True)
class MatchCase:
    """One query against a fixed library of seen variants."""

    case_id: str
    family: str
    expected_class: str
    expected_recipe_id: Optional[str]
    query_actions: Tuple[str, ...]
    library: Tuple[MatcherVariant, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "expected_class": self.expected_class,
            "expected_recipe_id": self.expected_recipe_id,
            "query_len": len(self.query_actions),
            "library_size": len(self.library),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MatchPrediction:
    matcher: str
    case_id: str
    family: str
    expected_class: str
    predicted_class: str
    expected_recipe_id: Optional[str]
    predicted_recipe_id: Optional[str]
    score: float
    distance: float
    correct_class: bool
    correct_recipe: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommitRecord:
    """Ground-truth-aware record for tentative/full online self-training audits.

    This deliberately uses explicit expected/predicted identifiers instead of
    evaluator-only recipe names.  The row adapter below can still consume the
    current evaluation logs when they expose correctness booleans.
    """

    decision_id: str
    expected_recipe_id: Optional[str]
    predicted_recipe_id: Optional[str]
    decision: str
    confidence: Optional[float] = None
    expected_variant_id: Optional[str] = None
    predicted_variant_id: Optional[str] = None
    event_index: Optional[int] = None
    promoted_from_tentative: bool = False
    promotion_event_index: Optional[int] = None
    latest_pinned: bool = False
    recovered_after_error: Optional[bool] = None
    recovery_steps: Optional[int] = None
    correct_override: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_commit(self) -> bool:
        return self.decision in {COMMIT_TENTATIVE, COMMIT_FULL, COMMIT_PROMOTION}

    @property
    def is_full_commit(self) -> bool:
        return self.decision in {COMMIT_FULL, COMMIT_PROMOTION}

    @property
    def is_promotion(self) -> bool:
        return bool(
            self.decision == COMMIT_PROMOTION
            or (self.promoted_from_tentative and self.is_full_commit)
        )

    @property
    def is_update(self) -> bool:
        return bool(self.is_commit or self.decision == COMMIT_KNOWN_REFRESH)

    @property
    def expected_committable(self) -> bool:
        if "expected_committable" in self.metadata:
            return bool(self.metadata["expected_committable"])
        return (
            self.expected_recipe_id is not None
            or self.expected_variant_id is not None
            or self.correct_override is not None
        )

    @property
    def correct(self) -> Optional[bool]:
        if self.correct_override is not None:
            return bool(self.correct_override)
        if self.expected_recipe_id is None and self.expected_variant_id is None:
            return None
        recipe_ok = (
            True
            if self.expected_recipe_id is None
            else self.predicted_recipe_id == self.expected_recipe_id
        )
        variant_ok = (
            True
            if self.expected_variant_id is None
            else self.predicted_variant_id == self.expected_variant_id
        )
        return bool(recipe_ok and variant_ok)

    def to_jsonable(self) -> Dict[str, Any]:
        row = asdict(self)
        row["is_commit"] = self.is_commit
        row["is_full_commit"] = self.is_full_commit
        row["is_promotion"] = self.is_promotion
        row["expected_committable"] = self.expected_committable
        row["correct"] = self.correct
        return row


def build_variant(
    recipe_id: str,
    preference_id: str,
    actions: Sequence[str],
    **metadata: Any,
) -> MatcherVariant:
    return MatcherVariant(
        recipe_id=str(recipe_id),
        preference_id=str(preference_id),
        actions=tuple(actions),
        metadata=dict(metadata),
    )


def _recipe_panel(seed: int, recipe_count: int) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    builders = list(recipe_builders().items())
    rng = random.Random(int(seed))
    rng.shuffle(builders)
    selected = builders[: max(1, min(int(recipe_count), len(builders)))]
    return tuple((name, tuple(builder())) for name, builder in selected)


def build_variants(
    seed: int = 1337,
    recipe_count: int = 16,
    preferences: Sequence[str] = (),
) -> Tuple[Dict[str, MatcherVariant], Dict[str, List[MatcherVariant]]]:
    """Build default and effective preference variants from the simulator."""

    identities: Dict[str, MatcherVariant] = {}
    preference_variants: Dict[str, List[MatcherVariant]] = defaultdict(list)
    for recipe_name, base_actions in _recipe_panel(seed, recipe_count):
        identities[recipe_name] = build_variant(recipe_name, "default", base_actions)
        seen = {tuple(base_actions)}
        for preference in preferences:
            if preference == "default":
                continue
            try:
                report = apply_preset(base_actions, preference)
            except (KeyError, ValueError):
                continue
            actions = tuple(report.actions)
            if actions in seen or actions == tuple(base_actions):
                continue
            seen.add(actions)
            preference_variants[recipe_name].append(
                build_variant(
                    recipe_name,
                    preference,
                    actions,
                    applied=tuple(report.applied),
                    failed=tuple(report.failed),
                )
            )
    return identities, dict(preference_variants)


def multiset_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    return _jaccard_counters(Counter(a), Counter(b))


def max_library_overlap(actions: Sequence[str], library: Sequence[MatcherVariant]) -> float:
    return max((multiset_jaccard(actions, variant.actions) for variant in library), default=0.0)


@lru_cache(maxsize=16384)
def _valid_goal_cached(
    actions: Tuple[str, ...],
) -> Tuple[bool, Optional[Tuple[int, ...]]]:
    try:
        return True, tuple(task_goal_signature(actions))
    except Exception:
        return False, None


def _valid_goal(actions: Sequence[str]) -> Tuple[bool, Optional[Tuple[int, ...]]]:
    return _valid_goal_cached(tuple(actions))


def _append_limited(
    buckets: Dict[str, List[MatchCase]],
    case: MatchCase,
    limit: int,
) -> None:
    # Balance is applied after generation so early recipes cannot consume a
    # family-wide cap before later recipes are considered.
    buckets[case.family].append(case)


def _balanced_family_sample(
    cases: Sequence[MatchCase],
    limit: int,
) -> Tuple[MatchCase, ...]:
    if len(cases) <= limit:
        return tuple(cases)
    groups: Dict[str, List[MatchCase]] = defaultdict(list)
    for case in cases:
        key = str(case.metadata.get("source_recipe", case.case_id))
        groups[key].append(case)
    selected: List[MatchCase] = []
    offsets = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            selected.append(groups[key][offset])
            offsets[key] = offset + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return tuple(selected)


def _case(
    case_id: str,
    family: str,
    expected_class: str,
    expected_recipe_id: Optional[str],
    actions: Sequence[str],
    library: Sequence[MatcherVariant],
    **metadata: Any,
) -> MatchCase:
    valid, goal = _valid_goal(actions) if actions else (False, None)
    meta = {
        "valid_ordering": bool(valid),
        "goal_signature_available": goal is not None,
        "max_action_overlap_to_seen": max_library_overlap(actions, library),
        **metadata,
    }
    return MatchCase(
        case_id=case_id,
        family=family,
        expected_class=expected_class,
        expected_recipe_id=expected_recipe_id,
        query_actions=tuple(actions),
        library=tuple(library),
        metadata=meta,
    )


def _drop_positions(seq: Sequence[Any], positions: Sequence[int]) -> Tuple[Any, ...]:
    remove = set(int(pos) for pos in positions)
    return tuple(item for index, item in enumerate(seq) if index not in remove)


def _repeat_positions(seq: Sequence[Any], positions: Sequence[int]) -> Tuple[Any, ...]:
    out: List[Any] = []
    repeat = Counter(int(pos) for pos in positions)
    for index, item in enumerate(seq):
        out.append(item)
        out.extend([item] * repeat.get(index, 0))
    return tuple(out)


def _replace_positions(
    seq: Sequence[Any],
    replacements: Mapping[int, Any],
) -> Tuple[Any, ...]:
    return tuple(replacements.get(index, item) for index, item in enumerate(seq))


def _sample_positions(rng: random.Random, length: int, count: int) -> Tuple[int, ...]:
    if length <= 0 or count <= 0:
        return ()
    return tuple(sorted(rng.sample(range(length), min(count, length))))


def _library_with_variant_history(
    identities: Mapping[str, MatcherVariant],
    preferences: Mapping[str, Sequence[MatcherVariant]],
    *,
    exclude_recipe: Optional[str] = None,
    exclude_variant_id: Optional[str] = None,
) -> Tuple[MatcherVariant, ...]:
    """Construct a realistic library with default plus one prior variant/recipe."""
    out: List[MatcherVariant] = []
    for recipe_id, default in identities.items():
        if recipe_id == exclude_recipe:
            continue
        out.append(default)
        prior = next(
            (
                variant for variant in preferences.get(recipe_id, ())
                if variant.variant_id != exclude_variant_id
            ),
            None,
        )
        if prior is not None:
            out.append(prior)
    return tuple(out)


def _valid_single_action_task_changes(
    variant: MatcherVariant,
    action_pool: Sequence[str],
    rng: random.Random,
    *,
    max_per_kind: int = 2,
) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
    """Find executable one-action changes whose symbolic task goal changes."""
    base_goal = task_goal_signature(variant.actions)
    pool = list(action_pool)
    rng.shuffle(pool)
    positions = list(range(len(variant.actions) + 1))
    rng.shuffle(positions)
    additions: List[Tuple[str, ...]] = []
    substitutions: List[Tuple[str, ...]] = []

    for position in positions:
        for action in pool:
            candidate = (
                variant.actions[:position] + (action,) + variant.actions[position:]
            )
            if validate_ordering(candidate):
                goal = task_goal_signature(candidate)
                if goal != base_goal and candidate not in additions:
                    additions.append(candidate)
                    break
        if len(additions) >= max_per_kind:
            break

    substitution_positions = list(range(len(variant.actions)))
    rng.shuffle(substitution_positions)
    for position in substitution_positions:
        for action in pool:
            if action == variant.actions[position]:
                continue
            candidate = (
                variant.actions[:position] + (action,) + variant.actions[position + 1:]
            )
            if validate_ordering(candidate):
                goal = task_goal_signature(candidate)
                if goal != base_goal and candidate not in substitutions:
                    substitutions.append(candidate)
                    break
        if len(substitutions) >= max_per_kind:
            break
    return {
        "addition": tuple(additions),
        "substitution": tuple(substitutions),
    }


def generate_match_cases(
    config: MatcherSettings = MatcherSettings(),
) -> Tuple[MatchCase, ...]:
    """Generate reviewer-facing stress cases.

    The case families include exact variants, same-recipe preference reorderings,
    leave-one-recipe-out natural new recipes, recognition noise, one-action
    semantic deltas, and synthetic controlled-overlap probes.
    """

    rng = random.Random(int(config.seed))
    identities, preferences = build_variants(
        seed=config.seed,
        recipe_count=config.recipe_count,
        preferences=config.preferences,
    )
    default_variants = tuple(identities.values())
    history_library = _library_with_variant_history(identities, preferences)
    action_pool = tuple(sorted({
        action for variant in default_variants for action in variant.actions
    }))
    buckets: Dict[str, List[MatchCase]] = defaultdict(list)
    limit = max(1, int(config.case_limit))

    for recipe_id, variant in identities.items():
        library = history_library
        _append_limited(
            buckets,
            _case(
                f"exact::{recipe_id}",
                "exact_known_variant",
                KNOWN,
                recipe_id,
                variant.actions,
                library,
                source_recipe=recipe_id,
                case_category="natural_task",
                primary_accuracy=True,
            ),
            limit,
        )

        for pref_variant in preferences.get(recipe_id, ()):
            preference_library = _library_with_variant_history(
                identities,
                preferences,
                exclude_variant_id=pref_variant.variant_id,
            )
            _append_limited(
                buckets,
                _case(
                    f"preference::{recipe_id}::{pref_variant.preference_id}",
                    "same_recipe_reordered_preference",
                    PREFERENCE_SHIFT,
                    recipe_id,
                    pref_variant.actions,
                    preference_library,
                    source_recipe=recipe_id,
                    preference=pref_variant.preference_id,
                    case_category="executable_task_variation",
                    primary_accuracy=True,
                ),
                limit,
            )

        base_goal = task_goal_signature(variant.actions)
        optional_positions = tuple(
            index for index in range(len(variant.actions))
            if validate_ordering(
                variant.actions[:index] + variant.actions[index + 1:],
                expected_goal=base_goal,
            )
        )
        for count in config.omission_sizes:
            for positions in itertools.combinations(optional_positions, int(count)):
                optional_actions = _drop_positions(variant.actions, positions)
                if not validate_ordering(optional_actions, expected_goal=base_goal):
                    continue
                _append_limited(
                    buckets,
                    _case(
                        f"optional_drop::{recipe_id}::{','.join(map(str, positions))}",
                        "same_recipe_optional_action_omitted",
                        PREFERENCE_SHIFT,
                        recipe_id,
                        optional_actions,
                        library,
                        source_recipe=recipe_id,
                        omitted_positions=positions,
                        omitted_count=len(positions),
                        expected_goal_preserved=True,
                        case_category="executable_task_variation",
                        primary_accuracy=True,
                    ),
                    limit,
                )

            positions = _sample_positions(rng, len(variant.actions), int(count))
            if not positions:
                continue
            _append_limited(
                buckets,
                _case(
                    f"noise_drop::{recipe_id}::{count}::{','.join(map(str, positions))}",
                    "same_recipe_recognition_drop",
                    PREFERENCE_SHIFT,
                    recipe_id,
                    _drop_positions(variant.actions, positions),
                    library,
                    source_recipe=recipe_id,
                    dropped_positions=positions,
                    dropped_count=len(positions),
                    physical_execution_valid=True,
                    case_category="recognition_corruption",
                    primary_accuracy=True,
                ),
                limit,
            )

        for count in config.repeat_sizes:
            positions = _sample_positions(rng, len(variant.actions), int(count))
            if not positions:
                continue
            _append_limited(
                buckets,
                _case(
                    f"noise_repeat::{recipe_id}::{count}::{','.join(map(str, positions))}",
                    "same_recipe_recognition_repeat",
                    PREFERENCE_SHIFT,
                    recipe_id,
                    _repeat_positions(variant.actions, positions),
                    library,
                    source_recipe=recipe_id,
                    repeated_positions=positions,
                    repeated_count=len(positions),
                    physical_execution_valid=True,
                    case_category="recognition_corruption",
                    primary_accuracy=True,
                ),
                limit,
            )
            repeated_actions = _repeat_positions(variant.actions, positions)
            _append_limited(
                buckets,
                _case(
                    f"execution_repeat::{recipe_id}::{count}::{','.join(map(str, positions))}",
                    "same_recipe_execution_repeat",
                    PREFERENCE_SHIFT,
                    recipe_id,
                    repeated_actions,
                    library,
                    source_recipe=recipe_id,
                    repeated_positions=positions,
                    repeated_count=len(positions),
                    expected_redundant_or_no_effect_action=True,
                    case_category="physical_execution_anomaly",
                    primary_accuracy=True,
                ),
                limit,
            )

        donor_choices = [candidate for candidate in default_variants if candidate.recipe_id != recipe_id]
        if donor_choices:
            donor = rng.choice(donor_choices)
            for count in config.substitution_sizes:
                source_positions = _sample_positions(rng, len(variant.actions), int(count))
                donor_positions = _sample_positions(rng, len(donor.actions), int(count))
                if not source_positions or not donor_positions:
                    continue
                action_replacements = {
                    pos: donor.actions[donor_positions[index]]
                    for index, pos in enumerate(source_positions)
                }
                _append_limited(
                    buckets,
                    _case(
                        f"noise_sub::{recipe_id}::{donor.recipe_id}::{count}",
                        "same_recipe_recognition_substitution",
                        PREFERENCE_SHIFT,
                        recipe_id,
                        _replace_positions(variant.actions, action_replacements),
                        library,
                        source_recipe=recipe_id,
                        donor_recipe=donor.recipe_id,
                        substituted_positions=source_positions,
                        donor_positions=donor_positions,
                        physical_execution_valid=True,
                        case_category="recognition_corruption",
                        primary_accuracy=True,
                    ),
                    limit,
                )

        valid_changes = _valid_single_action_task_changes(
            variant,
            action_pool,
            rng,
            max_per_kind=max(1, max(config.substitution_sizes, default=1)),
        )
        for change_kind, candidates in valid_changes.items():
            for change_index, changed_actions in enumerate(candidates):
                family = (
                    "new_recipe_one_action_added_valid"
                    if change_kind == "addition"
                    else "new_recipe_one_action_substitution_valid"
                )
                _append_limited(
                    buckets,
                    _case(
                        f"valid_{change_kind}::{recipe_id}::{change_index}",
                        family,
                        NEW_RECIPE,
                        None,
                        changed_actions,
                        library,
                        source_recipe=recipe_id,
                        edit_count=1,
                        symbolic_goal_changed=True,
                        case_category="executable_task_variation",
                        primary_accuracy=True,
                    ),
                    limit,
                )

    # Natural leave-one-recipe-out probes: target recipe has not been observed.
    for recipe_id, variant in identities.items():
        library = _library_with_variant_history(
            identities,
            preferences,
            exclude_recipe=recipe_id,
        )
        if not library:
            continue
        _append_limited(
            buckets,
            _case(
                f"natural_new::{recipe_id}",
                "natural_new_recipe_leave_one_out",
                NEW_RECIPE,
                None,
                variant.actions,
                library,
                source_recipe=recipe_id,
                case_category="natural_task",
                primary_accuracy=True,
            ),
            limit,
        )

        nearest_edit = min(
            (levenshtein_distance(variant.actions, candidate.actions) for candidate in library),
            default=10**9,
        )
        if nearest_edit == 1:
            _append_limited(
                buckets,
                _case(
                    f"natural_one_edit::{recipe_id}",
                    "natural_new_recipe_one_action_difference",
                    NEW_RECIPE,
                    None,
                    variant.actions,
                    library,
                    source_recipe=recipe_id,
                    nearest_natural_action_edit_distance=nearest_edit,
                    case_category="natural_task",
                    primary_accuracy=True,
                ),
                limit,
            )

        for pref_variant in preferences.get(recipe_id, ()):
            _append_limited(
                buckets,
                _case(
                    f"natural_pref_new::{recipe_id}::{pref_variant.preference_id}",
                    "natural_new_recipe_preference_leave_one_out",
                    NEW_RECIPE,
                    None,
                    pref_variant.actions,
                    library,
                    source_recipe=recipe_id,
                    preference=pref_variant.preference_id,
                    case_category="natural_task",
                    primary_accuracy=True,
                ),
                limit,
            )

    # Exact rational boundary probes are reported separately from natural-task
    # accuracy because their recipe identities are abstract by construction.
    for target in config.overlap_targets:
        for replicate in range(max(1, int(config.overlap_repeats))):
            reference_actions, query_actions, exact_overlap = _controlled_overlap_pair(
                float(target),
                replicate,
                rng,
            )
            reference = MatcherVariant(
                recipe_id=f"controlled_reference_{target:.4f}_{replicate}",
                preference_id="default",
                actions=reference_actions,
                metadata={"synthetic_boundary_probe": True},
            )
            distractor_actions = tuple(
                f"synthetic_distractor_{target:.4f}_{replicate}_{index}"
                for index in range(max(2, len(reference_actions)))
            )
            distractor = MatcherVariant(
                recipe_id=f"controlled_distractor_{target:.4f}_{replicate}",
                preference_id="default",
                actions=distractor_actions,
                metadata={"synthetic_boundary_probe": True},
            )
            _append_limited(
                buckets,
                _case(
                    f"controlled::{target:.4f}::{replicate}",
                    "controlled_overlap_new_recipe",
                    NEW_RECIPE,
                    None,
                    query_actions,
                    (reference, distractor),
                    source_recipe=reference.recipe_id,
                    target_overlap=float(target),
                    exact_rational_overlap=exact_overlap,
                    actual_overlap=multiset_jaccard(query_actions, reference_actions),
                    synthetic_actions=True,
                    case_category="controlled_boundary",
                    primary_accuracy=False,
                ),
                limit,
            )

    cases = tuple(
        case
        for family in sorted(buckets)
        for case in _balanced_family_sample(buckets[family], limit)
    )
    if not config.include_prefixes:
        return cases
    return cases + add_prefix_cases(
        cases,
        config.prefix_lengths,
        case_limit=config.prefix_case_limit,
    )


def _controlled_overlap_pair(
    target_overlap: float,
    replicate: int,
    rng: random.Random,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], str]:
    """Construct distinct abstract recipes with an exact rational Jaccard."""
    fraction = Fraction(str(float(target_overlap))).limit_denominator(100)
    if not 0 < fraction < 1:
        raise ValueError("controlled overlap must be strictly between zero and one")
    intersection = int(fraction.numerator)
    union = int(fraction.denominator)
    difference = union - intersection
    source_only_count = difference // 2
    query_only_count = difference - source_only_count
    stem = f"{fraction.numerator}_{fraction.denominator}_{replicate}"
    common = [f"synthetic_controlled_{stem}_common_{index}" for index in range(intersection)]
    source_only = [f"synthetic_controlled_{stem}_source_{index}" for index in range(source_only_count)]
    query_only = [f"synthetic_controlled_{stem}_query_{index}" for index in range(query_only_count)]
    source = common + source_only
    query = common + query_only
    rng.shuffle(source)
    rng.shuffle(query)
    return tuple(source), tuple(query), f"{fraction.numerator}/{fraction.denominator}"


def add_prefix_cases(
    cases: Sequence[MatchCase],
    prefix_lengths: Sequence[int],
    *,
    case_limit: Optional[int] = None,
) -> Tuple[MatchCase, ...]:
    """Return prefix-only variants of cases for early-recognition stress tests."""

    out: List[MatchCase] = []
    for case in cases:
        if case.metadata.get("case_category") not in {
            "natural_task",
            "executable_task_variation",
            "physical_execution_anomaly",
        }:
            continue
        if case.expected_class == KNOWN:
            expected_class = PREFERENCE_SHIFT
        else:
            expected_class = case.expected_class
        for length in prefix_lengths:
            k = min(int(length), len(case.query_actions))
            if k <= 0 or k >= len(case.query_actions):
                continue
            exact_prefix_candidates = sorted({
                variant.recipe_id
                for variant in case.library
                if tuple(variant.actions[:k]) == tuple(case.query_actions[:k])
            })
            prefix_identifiable = (
                exact_prefix_candidates == [case.expected_recipe_id]
                if case.expected_recipe_id is not None
                else not exact_prefix_candidates
            )
            out.append(
                MatchCase(
                    case_id=f"prefix{k}::{case.case_id}",
                    family=f"prefix::{case.family}",
                    expected_class=expected_class,
                    expected_recipe_id=case.expected_recipe_id,
                    query_actions=case.query_actions[:k],
                    library=case.library,
                    metadata={
                        **dict(case.metadata),
                        "prefix_length": k,
                        "prefix_case": True,
                        "primary_accuracy": False,
                        "exact_prefix_candidate_recipe_ids": exact_prefix_candidates,
                        "prefix_identifiable_under_exact_library": bool(prefix_identifiable),
                    },
                )
            )
    if case_limit is None:
        return tuple(out)
    grouped: Dict[Tuple[str, int], List[MatchCase]] = defaultdict(list)
    for case in out:
        grouped[(case.family, int(case.metadata["prefix_length"]))].append(case)
    return tuple(
        case
        for key in sorted(grouped)
        for case in _balanced_family_sample(
            grouped[key],
            max(1, int(case_limit)),
        )
    )


class Matcher:
    """Matcher interface for stress-suite baselines."""

    name = "base"
    representation = "semantic_actions"

    def classify(self, case: MatchCase) -> MatchResult:
        raise NotImplementedError

    def predict(self, case: MatchCase) -> MatchPrediction:
        match = self.classify(case)
        predicted = _normalize_predicted_class(match.kind)
        correct_class = predicted == case.expected_class
        correct_recipe = (
            match.recipe_id == case.expected_recipe_id
            if case.expected_recipe_id is not None
            else match.recipe_id is None
        )
        return MatchPrediction(
            matcher=self.name,
            case_id=case.case_id,
            family=case.family,
            expected_class=case.expected_class,
            predicted_class=predicted,
            expected_recipe_id=case.expected_recipe_id,
            predicted_recipe_id=match.recipe_id,
            score=float(match.jaccard),
            distance=float(match.order_distance),
            correct_class=bool(correct_class),
            correct_recipe=bool(correct_recipe),
            metadata={
                **dict(case.metadata),
                "matcher_representation": self.representation,
            },
        )


def _normalize_predicted_class(kind: str) -> str:
    if kind == TENTATIVE_PREFERENCE_SHIFT:
        return PREFERENCE_SHIFT
    if kind in EXPECTED_CLASSES:
        return kind
    return str(kind)


class JaccardMatcher(Matcher):
    """Production semantic-action Jaccard/Kendall matcher."""

    name = "current_jaccard_kendall"

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        self.matcher = RecipeMatcher(settings)

    def classify(self, case: MatchCase) -> MatchResult:
        return self.matcher.classify(
            case.query_actions,
            [variant.to_known_variant() for variant in case.library],
        )


class EditDistanceMatcher(Matcher):
    """Normalized Levenshtein baseline over semantic-action sequences."""

    name = "edit_distance"

    def __init__(
        self,
        threshold: float = DEFAULT_SETTINGS.match_threshold,
        margin: float = DEFAULT_SETTINGS.match_margin,
    ):
        self.threshold = float(threshold)
        self.margin = float(margin)

    def classify(self, case: MatchCase) -> MatchResult:
        exact = _exact_variant(case)
        if exact is not None:
            return exact
        best = _best_by_recipe(case, self._score)
        return _classification_from_scores(best, self.threshold, self.margin)

    @staticmethod
    def _score(query: Sequence[str], candidate: Sequence[str]) -> Tuple[float, float]:
        distance = levenshtein_distance(query, candidate)
        denom = max(len(query), len(candidate), 1)
        score = 1.0 - (distance / denom)
        return max(0.0, score), 1.0 - max(0.0, score)


class LcsMatcher(Matcher):
    """Longest-common-subsequence baseline over semantic-action sequences."""

    name = "lcs"

    def __init__(
        self,
        threshold: float = DEFAULT_SETTINGS.match_threshold,
        margin: float = DEFAULT_SETTINGS.match_margin,
    ):
        self.threshold = float(threshold)
        self.margin = float(margin)

    def classify(self, case: MatchCase) -> MatchResult:
        exact = _exact_variant(case)
        if exact is not None:
            return exact
        best = _best_by_recipe(case, self._score)
        return _classification_from_scores(best, self.threshold, self.margin)

    @staticmethod
    def _score(query: Sequence[str], candidate: Sequence[str]) -> Tuple[float, float]:
        lcs = lcs_length(query, candidate)
        denom = max(len(query), len(candidate), 1)
        score = lcs / denom
        return score, 1.0 - score


class BeliefMatcher(Matcher):
    """Closed known-recipe beliefs plus an explicit unknown-recipe hypothesis."""

    name = "probabilistic_recipe_belief"

    def __init__(
        self,
        temperature: float = 0.12,
        unknown_prior: float = 0.20,
        unknown_distance: float = 0.40,
        min_known_posterior: float = 0.50,
        posterior_margin: float = 0.05,
    ):
        self.temperature = max(float(temperature), 1e-6)
        self.unknown_prior = min(0.99, max(1e-6, float(unknown_prior)))
        self.unknown_distance = min(1.0, max(0.0, float(unknown_distance)))
        self.min_known_posterior = min(1.0, max(0.0, float(min_known_posterior)))
        self.posterior_margin = max(0.0, float(posterior_margin))

    def classify(self, case: MatchCase) -> MatchResult:
        exact = _exact_variant(case)
        if exact is not None:
            return exact
        if not case.library:
            return MatchResult(NEW_RECIPE, None, None, 1.0, 0.0)
        best_distances: Dict[str, Tuple[MatcherVariant, float]] = {}
        for variant in case.library:
            distance = levenshtein_distance(
                case.query_actions,
                variant.actions,
            ) / max(len(case.query_actions), len(variant.actions), 1)
            previous = best_distances.get(variant.recipe_id)
            if previous is None or distance < previous[1]:
                best_distances[variant.recipe_id] = (variant, float(distance))

        recipe_count = max(1, len(best_distances))
        known_prior = (1.0 - self.unknown_prior) / recipe_count
        logits = {
            recipe_id: math.log(known_prior) - distance / self.temperature
            for recipe_id, (_variant, distance) in best_distances.items()
        }
        unknown_logit = (
            math.log(self.unknown_prior)
            - self.unknown_distance / self.temperature
        )
        max_logit = max([unknown_logit, *logits.values()])
        normalizer = math.exp(unknown_logit - max_logit) + sum(
            math.exp(value - max_logit) for value in logits.values()
        )
        unknown_posterior = math.exp(unknown_logit - max_logit) / normalizer
        posteriors = {
            recipe_id: math.exp(value - max_logit) / normalizer
            for recipe_id, value in logits.items()
        }
        ranked = sorted(posteriors.items(), key=lambda item: item[1], reverse=True)
        best_recipe, best_posterior = ranked[0]
        runner_posterior = ranked[1][1] if len(ranked) > 1 else 0.0
        distance = best_distances[best_recipe][1]
        if (
            unknown_posterior >= best_posterior
            or best_posterior < self.min_known_posterior
            or best_posterior - runner_posterior < self.posterior_margin
        ):
            return MatchResult(NEW_RECIPE, None, None, unknown_posterior, distance)
        return MatchResult(PREFERENCE_SHIFT, best_recipe, None, best_posterior, distance)


@dataclass(frozen=True)
class PartialOrderGraph:
    """Action-occurrence DAG induced by shared physical resources."""

    nodes: Tuple[Tuple[str, int], ...]
    edges: Tuple[Tuple[Tuple[str, int], Tuple[str, int]], ...]


def _occurrence_nodes(tokens: Sequence[str]) -> Tuple[Tuple[str, int], ...]:
    counts: Counter = Counter()
    nodes: List[Tuple[str, int]] = []
    for token in tokens:
        counts[token] += 1
        nodes.append((str(token), int(counts[token])))
    return tuple(nodes)


def _action_resources(action: str) -> frozenset[str]:
    parsed = parse_action_label(action)
    return frozenset(
        str(value)
        for key, value in parsed.args.items()
        if key not in {"from", "to", "location"}
    )


@lru_cache(maxsize=16384)
def _build_partial_order_graph_cached(
    actions: Tuple[str, ...],
) -> PartialOrderGraph:
    nodes = _occurrence_nodes(actions)
    try:
        resources = tuple(_action_resources(action) for action in actions)
    except ValueError:
        return PartialOrderGraph(
            nodes=nodes,
            edges=tuple((nodes[index], nodes[index + 1]) for index in range(len(nodes) - 1)),
        )
    edges = tuple(
        (nodes[left], nodes[right])
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
        if resources[left] & resources[right]
    )
    return PartialOrderGraph(nodes=nodes, edges=edges)


def build_partial_order_graph(
    actions: Sequence[str],
) -> PartialOrderGraph:
    return _build_partial_order_graph_cached(tuple(actions))


def partial_order_similarity(
    query: PartialOrderGraph,
    candidate: PartialOrderGraph,
) -> Tuple[float, float]:
    support = multiset_jaccard(
        [node[0] for node in query.nodes],
        [node[0] for node in candidate.nodes],
    )
    query_positions = {node: index for index, node in enumerate(query.nodes)}
    comparable = [
        (left, right) for left, right in candidate.edges
        if left in query_positions and right in query_positions
    ]
    precedence = (
        sum(query_positions[left] < query_positions[right] for left, right in comparable)
        / len(comparable)
        if comparable else (1.0 if support > 0.0 else 0.0)
    )
    return float(support), float(precedence)


class GraphMatcher(Matcher):
    """Resource-dependency DAG baseline with goal and node-support evidence."""

    name = "partial_order_task"

    def __init__(
        self,
        threshold: float = 0.88,
        margin: float = DEFAULT_SETTINGS.match_margin,
        support_weight: float = 0.55,
        precedence_weight: float = 0.25,
        goal_weight: float = 0.20,
    ):
        total = support_weight + precedence_weight + goal_weight
        if total <= 0.0:
            raise ValueError("partial-order matcher weights must sum to a positive value")
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.support_weight = float(support_weight) / total
        self.precedence_weight = float(precedence_weight) / total
        self.goal_weight = float(goal_weight) / total

    def classify(self, case: MatchCase) -> MatchResult:
        exact = _exact_variant(case)
        if exact is not None:
            return exact
        query_goal_valid, query_goal = _valid_goal(case.query_actions)
        query_graph = build_partial_order_graph(case.query_actions)
        best: Dict[str, Tuple[MatcherVariant, float, float]] = {}
        for variant in case.library:
            candidate_graph = build_partial_order_graph(variant.actions)
            support_score, precedence_score = partial_order_similarity(
                query_graph,
                candidate_graph,
            )
            variant_goal_valid, variant_goal = _valid_goal(variant.actions)
            goal_score = (
                1.0
                if query_goal_valid and variant_goal_valid and query_goal == variant_goal
                else 0.0
            )
            score = (
                self.support_weight * support_score
                + self.precedence_weight * precedence_score
                + self.goal_weight * goal_score
            )
            previous = best.get(variant.recipe_id)
            if previous is None or score > previous[1]:
                best[variant.recipe_id] = (variant, float(score), 1.0 - float(score))
        return _classification_from_scores(best, self.threshold, self.margin)


def _exact_variant(case: MatchCase) -> Optional[MatchResult]:
    query = case.query_actions
    query_hash = make_variant_id(query)
    for variant in case.library:
        candidate = variant.actions
        if make_variant_id(candidate) == query_hash and candidate == query:
            return MatchResult(KNOWN, variant.recipe_id, make_variant_id(candidate), 1.0, 0.0)
    return None


def _best_by_recipe(
    case: MatchCase,
    scorer,
) -> Dict[str, Tuple[MatcherVariant, float, float]]:
    best: Dict[str, Tuple[MatcherVariant, float, float]] = {}
    query = case.query_actions
    for variant in case.library:
        score, distance = scorer(query, variant.actions)
        previous = best.get(variant.recipe_id)
        if previous is None or score > previous[1]:
            best[variant.recipe_id] = (variant, float(score), float(distance))
    return best


def _classification_from_scores(
    best: Mapping[str, Tuple[MatcherVariant, float, float]],
    threshold: float,
    margin: float,
) -> MatchResult:
    if not best:
        return MatchResult(NEW_RECIPE, None, None, 0.0, 0.0)
    ranked = sorted(best.items(), key=lambda item: (item[1][1], -item[1][2]), reverse=True)
    best_recipe, (variant, score, distance) = ranked[0]
    runner = ranked[1][1][1] if len(ranked) > 1 else None
    if score < threshold or (runner is not None and score - runner < margin):
        return MatchResult(NEW_RECIPE, None, None, score, distance)
    return MatchResult(PREFERENCE_SHIFT, best_recipe, None, score, distance)


def levenshtein_distance(
    first: Sequence[str], second: Sequence[str],
) -> int:
    longer, shorter = (
        (first, second) if len(first) >= len(second) else (second, first)
    )
    previous = list(range(len(shorter) + 1))
    for first_index, first_token in enumerate(longer, start=1):
        current = [first_index]
        for second_index, second_token in enumerate(shorter, start=1):
            current.append(
                min(
                    previous[second_index] + 1,
                    current[second_index - 1] + 1,
                    previous[second_index - 1]
                    + (0 if first_token == second_token else 1),
                )
            )
        previous = current
    return previous[-1]


def lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    if not first or not second:
        return 0
    previous = [0] * (len(second) + 1)
    for first_token in first:
        current = [0]
        for second_index, second_token in enumerate(second, start=1):
            current.append(
                previous[second_index - 1] + 1
                if first_token == second_token
                else max(
                    previous[second_index], current[second_index - 1],
                )
            )
        previous = current
    return previous[-1]


def default_matchers(settings: Settings = DEFAULT_SETTINGS) -> Tuple[Matcher, ...]:
    return (
        JaccardMatcher(settings),
        EditDistanceMatcher(settings.match_threshold, settings.match_margin),
        LcsMatcher(settings.match_threshold, settings.match_margin),
        BeliefMatcher(),
        GraphMatcher(),
    )


def _matcher_calibration_objective(
    predictions: Sequence[MatchPrediction],
) -> Tuple[float, float, float]:
    rows = [
        row for row in predictions
        if row.expected_class != KNOWN
        and not bool(row.metadata.get("prefix_case"))
        and bool(row.metadata.get("primary_accuracy", True))
    ]
    known_recipe = [row for row in rows if row.expected_class != NEW_RECIPE]
    new_recipe = [row for row in rows if row.expected_class == NEW_RECIPE]
    known_acceptance = _mean(row.predicted_class != NEW_RECIPE for row in known_recipe)
    new_rejection = _mean(row.predicted_class == NEW_RECIPE for row in new_recipe)
    balanced_open_set = 0.5 * (known_acceptance + new_rejection)
    closed_set_recipe = _mean(row.correct_recipe for row in known_recipe)
    objective = 0.70 * balanced_open_set + 0.30 * closed_set_recipe
    return objective, balanced_open_set, closed_set_recipe


def _grid_calibrate_matcher(
    name: str,
    cases: Sequence[MatchCase],
    candidates: Sequence[Tuple[Mapping[str, float], Matcher]],
) -> Tuple[Matcher, Dict[str, Any]]:
    if not candidates:
        raise ValueError(f"no calibration candidates for {name}")
    scored: List[Tuple[Tuple[float, float, float], Mapping[str, float], Matcher]] = []
    for parameters, matcher in candidates:
        objective = _matcher_calibration_objective(evaluate_cases(cases, (matcher,)))
        scored.append((objective, parameters, matcher))
    scored.sort(key=lambda item: item[0], reverse=True)
    objective, parameters, matcher = scored[0]
    return matcher, {
        "matcher": name,
        "selection_data": "independent_seed_primary_nonprefix_cases",
        "objective": "0.70*open_set_balanced_accuracy+0.30*closed_set_recipe_accuracy",
        "selected_parameters": dict(parameters),
        "objective_value": objective[0],
        "open_set_balanced_accuracy": objective[1],
        "closed_set_recipe_accuracy": objective[2],
        "n_candidates": len(candidates),
    }


def calibrated_matchers(
    calibration_cases: Sequence[MatchCase],
    settings: Settings = DEFAULT_SETTINGS,
) -> Tuple[Tuple[Matcher, ...], Dict[str, Any]]:
    """Tune each baseline on an independent seed and keep production fixed."""
    thresholds = (0.40, 0.55, 0.70, 0.80, 0.88, 0.92, 0.95, 0.96, 0.98, 1.0)
    margins = (0.0, 0.03, 0.08)
    selected: List[Matcher] = [JaccardMatcher(settings)]
    reports: Dict[str, Any] = {
        "current_jaccard_kendall": {
            "matcher": "current_jaccard_kendall",
            "selection_data": "fixed_production_operating_point",
            "selected_parameters": {
                "threshold": settings.match_threshold,
                "margin": settings.match_margin,
            },
        },
    }

    bounded_specs = (
        (
            "edit_distance", lambda threshold, margin: EditDistanceMatcher(threshold, margin),
        ),
        ("lcs", lambda threshold, margin: LcsMatcher(threshold, margin)),
        ("partial_order_task", lambda threshold, margin: GraphMatcher(threshold, margin)),
    )
    for name, factory in bounded_specs:
        matcher, report = _grid_calibrate_matcher(
            name,
            calibration_cases,
            tuple(
                (
                    {"threshold": threshold, "margin": margin},
                    factory(threshold, margin),
                )
                for threshold in thresholds for margin in margins
            ),
        )
        selected.append(matcher)
        reports[name] = report

    matcher, report = _grid_calibrate_matcher(
        "probabilistic_recipe_belief",
        calibration_cases,
        tuple(
            (
                {"min_known_posterior": threshold, "posterior_margin": margin},
                BeliefMatcher(
                    min_known_posterior=threshold,
                    posterior_margin=margin,
                ),
            )
            for threshold in (0.20, 0.35, 0.50, 0.65, 0.80)
            for margin in margins
        ),
    )
    selected.append(matcher)
    reports["probabilistic_recipe_belief"] = report
    return tuple(selected), reports


def evaluate_cases(
    cases: Sequence[MatchCase],
    matchers: Sequence[Matcher],
) -> Tuple[MatchPrediction, ...]:
    return tuple(matcher.predict(case) for matcher in matchers for case in cases)


def confusion_matrix(predictions: Sequence[MatchPrediction]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {
        expected: {predicted: 0 for predicted in EXPECTED_CLASSES}
        for expected in EXPECTED_CLASSES
    }
    for pred in predictions:
        matrix.setdefault(pred.expected_class, {}).setdefault(pred.predicted_class, 0)
        matrix[pred.expected_class][pred.predicted_class] += 1
    return matrix


def summarize_predictions(predictions: Sequence[MatchPrediction]) -> Dict[str, Any]:
    rows = list(predictions)
    total = len(rows)
    if total == 0:
        return {
            "n": 0,
            "class_accuracy": 0.0,
            "recipe_accuracy": 0.0,
            "confusion": confusion_matrix(rows),
        }
    new_rows = [row for row in rows if row.expected_class == NEW_RECIPE]
    same_recipe_rows = [row for row in rows if row.expected_class != NEW_RECIPE]
    non_known_rows = [row for row in rows if row.expected_class != KNOWN]
    identifiable_prefix_rows = [
        row for row in rows
        if row.metadata.get("prefix_case")
        and row.metadata.get("prefix_identifiable_under_exact_library")
    ]
    known_recipe_acceptance = _mean(
        row.predicted_class != NEW_RECIPE for row in same_recipe_rows
    )
    new_recipe_rejection = _mean(
        row.predicted_class == NEW_RECIPE for row in new_rows
    )
    return {
        "n": total,
        "class_accuracy": _mean(row.correct_class for row in rows),
        "recipe_accuracy": _mean(row.correct_recipe for row in rows),
        "false_same_recipe_rate": _mean(row.predicted_class != NEW_RECIPE for row in new_rows),
        "false_new_recipe_rate": _mean(row.predicted_class == NEW_RECIPE for row in same_recipe_rows),
        "false_known_rate": _mean(row.predicted_class == KNOWN for row in non_known_rows),
        "known_recall": _recall(rows, KNOWN),
        "preference_shift_recall": _recall(rows, PREFERENCE_SHIFT),
        "new_recipe_recall": _recall(rows, NEW_RECIPE),
        "open_set_balanced_accuracy": 0.5 * (
            known_recipe_acceptance + new_recipe_rejection
        ),
        "closed_set_recipe_accuracy": _mean(
            row.predicted_recipe_id == row.expected_recipe_id
            for row in same_recipe_rows
        ),
        "identifiable_prefix_recipe_accuracy": (
            _mean(row.correct_recipe for row in identifiable_prefix_rows)
            if identifiable_prefix_rows else None
        ),
        "n_identifiable_prefix_cases": len(identifiable_prefix_rows),
        "confusion": confusion_matrix(rows),
    }


def summarize_assist_contract(predictions: Sequence[MatchPrediction]) -> Dict[str, Any]:
    """Summarize violations of an observation-authoritative protocol.

    Protocol interpreted for the paper:
        observation mode: new recipes may be introduced;
        assist mode: recipe is already known, but preference may be new.

    In this stress suite, KNOWN/PREFERENCE_SHIFT cases stand in for assist-mode
    known-recipe executions. NEW_RECIPE cases stand in for episodes that should
    be routed to observation before assist, not learned by assist self-labeling.
    """

    rows = list(predictions)
    assist_known = [row for row in rows if row.expected_class != NEW_RECIPE]
    observation_required = [row for row in rows if row.expected_class == NEW_RECIPE]
    assist_violations = [row for row in assist_known if row.predicted_class == NEW_RECIPE]
    false_assist_accepts = [
        row for row in observation_required
        if row.predicted_class != NEW_RECIPE
    ]
    return {
        "definition": (
            "Checks the contract observation=new task allowed, assist=known "
            "recipe with possible new preference. A production needs_observation "
            "or new_recipe return during assist is counted as a contract violation."
        ),
        "n_cases": len(rows),
        "n_assist_known_recipe_cases": len(assist_known),
        "n_observation_required_cases": len(observation_required),
        "assist_contract_violation_rate": _mean(
            row.predicted_class == NEW_RECIPE for row in assist_known
        ),
        "assist_known_recipe_recall": _mean(
            row.predicted_class != NEW_RECIPE for row in assist_known
        ),
        "false_assist_acceptance_on_new_recipe_rate": _mean(
            row.predicted_class != NEW_RECIPE for row in observation_required
        ),
        "observation_required_recall": _mean(
            row.predicted_class == NEW_RECIPE for row in observation_required
        ),
        "assist_violation_examples": [
            {
                "case_id": row.case_id,
                "family": row.family,
                "expected_class": row.expected_class,
                "predicted_class": row.predicted_class,
                "expected_recipe_id": row.expected_recipe_id,
                "predicted_recipe_id": row.predicted_recipe_id,
                "score": row.score,
                "max_action_overlap_to_seen": row.metadata.get("max_action_overlap_to_seen"),
            }
            for row in assist_violations[:10]
        ],
        "false_assist_accept_examples": [
            {
                "case_id": row.case_id,
                "family": row.family,
                "expected_class": row.expected_class,
                "predicted_class": row.predicted_class,
                "predicted_recipe_id": row.predicted_recipe_id,
                "score": row.score,
                "max_action_overlap_to_seen": row.metadata.get("max_action_overlap_to_seen"),
            }
            for row in false_assist_accepts[:10]
        ],
    }


def summarize_assist_by_matcher(
    predictions: Sequence[MatchPrediction],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[MatchPrediction]] = defaultdict(list)
    for pred in predictions:
        grouped[pred.matcher].append(pred)
    return {
        matcher: summarize_assist_contract(rows)
        for matcher, rows in sorted(grouped.items())
    }


def summarize_by(
    predictions: Sequence[MatchPrediction],
    key: str,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[MatchPrediction]] = defaultdict(list)
    for pred in predictions:
        if key == "matcher":
            group_key = pred.matcher
        elif key == "family":
            group_key = pred.family
        elif key == "overlap_bin":
            group_key = overlap_bin(float(pred.metadata.get("max_action_overlap_to_seen", 0.0)))
        else:
            group_key = str(pred.metadata.get(key, "NA"))
        grouped[group_key].append(pred)
    return {group: summarize_predictions(rows) for group, rows in sorted(grouped.items())}


def threshold_sweep(
    cases: Sequence[MatchCase],
    thresholds: Sequence[float],
    settings: Settings = DEFAULT_SETTINGS,
) -> Tuple[Dict[str, Any], ...]:
    """Evaluate the production matcher across semantic-action Jaccard thresholds."""

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        local_settings = replace(settings, match_threshold=float(threshold))
        matcher = JaccardMatcher(local_settings)
        predictions = evaluate_cases(cases, (matcher,))
        summary = summarize_predictions(predictions)
        rows.append({
            "match_threshold": float(threshold),
            **summary,
        })
    return tuple(rows)


def parse_commit_records(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[CommitRecord, ...]:
    """Parse evaluator rows using the current commit-log schema."""

    records = list(
        parse_commit_record(row, row_index=index)
        for index, row in enumerate(rows)
    )
    for index, record in enumerate(tuple(records)):
        if not record.is_update or record.correct is not False:
            continue
        recovery = next(
            (
                candidate for candidate in records[index + 1:]
                if candidate.is_update
                and candidate.correct is True
                and candidate.expected_recipe_id == record.expected_recipe_id
                and candidate.expected_variant_id == record.expected_variant_id
            ),
            None,
        )
        if recovery is None:
            records[index] = replace(record, recovered_after_error=False)
            continue
        recovery_steps = (
            recovery.event_index - record.event_index
            if recovery.event_index is not None and record.event_index is not None
            else None
        )
        records[index] = replace(
            record,
            recovered_after_error=True,
            recovery_steps=recovery_steps,
        )
    return tuple(records)


def parse_commit_record(
    row: Mapping[str, Any],
    *,
    row_index: int = 0,
) -> CommitRecord:
    kind = row.get("commit_decision")
    decision = _decision_from_kind(kind)
    commit_applied = _as_optional_bool(row.get("commit_applied"))
    if commit_applied is False:
        decision = COMMIT_NONE

    candidate_correct = _as_optional_bool(
        row.get("commit_candidate_correct")
    )
    recipe_correct = _as_optional_bool(
        row.get("commit_candidate_recipe_correct")
    )
    variant_correct = _as_optional_bool(
        row.get("commit_candidate_variant_correct")
    )
    if candidate_correct is not None:
        correct_override = candidate_correct
    elif recipe_correct is not None and variant_correct is not None:
        correct_override = bool(recipe_correct and variant_correct)
    elif variant_correct is not None:
        correct_override = bool(variant_correct)
    elif recipe_correct is not None:
        correct_override = bool(recipe_correct)
    else:
        correct_override = None

    latest_pinned = _as_bool(row.get("latest_pinned"))
    expected_committable = _as_optional_bool(
        row.get("self_training_opportunity")
    )
    if expected_committable is None:
        expected_committable = bool(
            row.get("mode") == "assist"
            and not bool(row.get("expected_variant_known_before", False))
        )

    metadata_keys = (
        "pair", "recipe", "preference", "mode", "requested_mode",
        "executed_mode", "classification_kind", "memory_state_before",
        "memory_state_after", "false_variant_creation", "false_latest_promotion",
        "commit_registry_size", "commit_variants_scored",
        "commit_scoring_wall_s", "commit_candidate_correct", "commit_applied",
        "commit_new_variant", "commit_known_variant", "latest_pin_correct",
        "promotion_delay_demos", "tentative_first_demo",
    )
    metadata = {
        key: row.get(key)
        for key in metadata_keys
        if key in row
    }
    metadata["expected_committable"] = bool(expected_committable)

    return CommitRecord(
        decision_id=str(row.get("commit_decision_id") or f"row_{row_index}"),
        expected_recipe_id=_as_optional_str(row.get("expected_recipe_id")),
        predicted_recipe_id=_as_optional_str(
            row.get("commit_candidate_recipe_id")
        ),
        decision=decision,
        confidence=_as_optional_float(row.get("commit_confidence")),
        expected_variant_id=_as_optional_str(row.get("expected_variant_id")),
        predicted_variant_id=_as_optional_str(row.get("commit_variant_id")),
        event_index=_as_optional_int(row.get("event_index", row_index)),
        promoted_from_tentative=_as_bool(
            row.get("promoted_from_tentative")
        ),
        promotion_event_index=_as_optional_int(
            row.get("event_index") if decision == COMMIT_PROMOTION else None
        ),
        latest_pinned=latest_pinned,
        recovered_after_error=_as_optional_bool(
            row.get("recovered_after_error")
        ),
        recovery_steps=_as_optional_int(row.get("recovery_steps")),
        correct_override=correct_override,
        metadata=metadata,
    )


def summarize_commit_decisions(
    records: Sequence[CommitRecord],
    *,
    n_calibration_bins: int = 10,
) -> Dict[str, Any]:
    """Reviewer-facing audit for confidence-gated online self-training."""

    rows = list(records)
    commits = [row for row in rows if row.is_commit]
    full_commits = [row for row in rows if row.is_full_commit]
    tentative = [row for row in rows if row.decision == COMMIT_TENTATIVE]
    known_refreshes = [row for row in rows if row.decision == COMMIT_KNOWN_REFRESH]
    expected = [row for row in rows if row.expected_committable]
    expected_commits = [row for row in commits if row.expected_committable]
    correct_expected_commits = [row for row in expected_commits if row.correct is True]
    scored_commits = [row for row in commits if row.correct is not None]
    promotions = [row for row in rows if row.is_promotion]
    scored_promotions = [row for row in promotions if row.correct is not None]
    latest_pins = [
        row for row in rows
        if row.latest_pinned
    ]
    scored_latest_pins = [row for row in latest_pins if row.correct is not None]
    mistaken_commits = [row for row in scored_commits if row.correct is False]
    recovery_rows = [
        row for row in mistaken_commits
        if row.recovered_after_error is not None
    ]
    promotion_times = []
    for row in promotions:
        logged_delay = _as_optional_int(row.metadata.get("promotion_delay_demos"))
        if logged_delay is not None:
            promotion_times.append(logged_delay)
        elif row.promotion_event_index is not None and row.event_index is not None:
            promotion_times.append(row.promotion_event_index - row.event_index)
    recovery_steps = [
        row.recovery_steps for row in recovery_rows
        if row.recovery_steps is not None
    ]
    return {
        "definition": (
            "Audits the confidence-gated tentative/full online self-training "
            "path. Commit precision/recall require ground-truth correctness; "
            "calibration uses confidence only when logged."
        ),
        "n_records": len(rows),
        "n_expected_committable": len(expected),
        "n_commits": len(commits),
        "n_full_commits": len(full_commits),
        "n_tentative_commits": len(tentative),
        "n_known_variant_refreshes": len(known_refreshes),
        "n_promotions": len(promotions),
        "n_latest_pins": len(latest_pins),
        "commit_precision": _optional_rate(row.correct is True for row in scored_commits),
        "commit_recall": _safe_div(len(correct_expected_commits), len(expected)),
        "commit_coverage": _safe_div(len(expected_commits), len(expected)),
        "full_commit_precision": _optional_rate(row.correct is True for row in full_commits if row.correct is not None),
        "tentative_commit_precision": _optional_rate(row.correct is True for row in tentative if row.correct is not None),
        "false_promotion_rate": _optional_rate(row.correct is False for row in scored_promotions),
        "tentative_to_full_promotion_accuracy": _optional_rate(row.correct is True for row in scored_promotions),
        "known_variant_refresh_accuracy": _optional_rate(
            row.correct is True for row in known_refreshes if row.correct is not None
        ),
        "false_latest_pin_rate": _optional_rate(
            not bool(
                row.metadata.get("latest_pin_correct")
                if row.metadata.get("latest_pin_correct") is not None
                else row.correct
            )
            for row in scored_latest_pins
        ),
        "mean_time_to_promotion": (
            sum(float(value) for value in promotion_times) / len(promotion_times)
            if promotion_times else None
        ),
        "mean_recovery_steps_after_mistaken_commit": (
            sum(float(value) for value in recovery_steps) / len(recovery_steps)
            if recovery_steps else None
        ),
        "recovery_after_mistaken_commit_rate": _optional_rate(
            row.recovered_after_error is True for row in recovery_rows
        ),
        "calibration": commit_confidence_calibration(
            rows,
            n_bins=n_calibration_bins,
        ),
    }


def commit_confidence_calibration(
    records: Sequence[CommitRecord],
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Return reliability-bin data for commit confidence."""

    usable = [
        row for row in records
        if row.confidence is not None and row.correct is not None
    ]
    if not usable:
        return {
            "status": "unavailable",
            "reason": "no assist decisions with both confidence and candidate correctness",
            "n": 0,
            "bins": [],
            "expected_calibration_error": None,
            "brier_score": None,
        }

    bin_count = max(1, int(n_bins))
    buckets: List[List[CommitRecord]] = [[] for _ in range(bin_count)]
    for row in usable:
        confidence = min(1.0, max(0.0, float(row.confidence)))
        index = min(bin_count - 1, int(confidence * bin_count))
        buckets[index].append(row)

    bins: List[Dict[str, Any]] = []
    calibration_error = 0.0
    for index, bucket in enumerate(buckets):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if not bucket:
            bins.append({
                "bin": index,
                "lower": lower,
                "upper": upper,
                "n": 0,
                "mean_confidence": None,
                "empirical_accuracy": None,
                "abs_calibration_error": None,
                "commit_coverage": None,
            })
            continue
        mean_confidence = sum(float(row.confidence) for row in bucket) / len(bucket)
        empirical_accuracy = _mean(row.correct is True for row in bucket)
        abs_error = abs(empirical_accuracy - mean_confidence)
        calibration_error += (len(bucket) / len(usable)) * abs_error
        bins.append({
            "bin": index,
            "lower": lower,
            "upper": upper,
            "n": len(bucket),
            "mean_confidence": mean_confidence,
            "empirical_accuracy": empirical_accuracy,
            "abs_calibration_error": abs_error,
            "commit_coverage": _mean(row.is_commit for row in bucket),
        })

    brier_score = sum(
        (float(row.confidence) - (1.0 if row.correct else 0.0)) ** 2
        for row in usable
    ) / len(usable)
    return {
        "status": "completed",
        "n": len(usable),
        "bins": bins,
        "expected_calibration_error": calibration_error,
        "brier_score": brier_score,
    }


def commit_audit_spec() -> Dict[str, Any]:
    """Document the online self-training metrics expected by reviewers."""

    return {
        "definition": (
            "Metric schema for confidence-gated tentative/full self-training. "
            "Use parse_commit_records(...) for existing evaluator rows or "
            "emit CommitRecord directly in a dedicated stress run."
        ),
        "record_type": "CommitRecord",
        "required_for_precision_recall": (
            "decision, expected_committable, predicted recipe/variant or "
            "correctness booleans"
        ),
        "required_for_calibration": "confidence plus correctness per committed decision",
        "metrics": (
            "commit_precision",
            "commit_recall",
            "full_commit_precision",
            "tentative_commit_precision",
            "false_promotion_rate",
            "tentative_to_full_promotion_accuracy",
            "false_latest_pin_rate",
            "mean_time_to_promotion",
            "recovery_after_mistaken_commit_rate",
            "mean_recovery_steps_after_mistaken_commit",
            "reliability_bins",
            "expected_calibration_error",
            "brier_score",
        ),
    }


def overlap_bin(value: float) -> str:
    bins = (
        (0.00, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 0.95),
        (0.95, 0.98),
        (0.98, 1.01),
    )
    for lower, upper in bins:
        if lower <= value < upper:
            return f"[{lower:.2f},{upper:.2f})"
    return "out_of_range"


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return (float(numerator) / float(denominator)) if denominator else None


def _mean(values: Iterable[Any]) -> float:
    numeric_values = [
        1.0 if value is True else 0.0 if value is False else float(value)
        for value in values
    ]
    return (
        sum(numeric_values) / len(numeric_values)
        if numeric_values else 0.0
    )


def _optional_rate(values: Iterable[Any]) -> Optional[float]:
    numeric_values = [
        1.0 if value is True else 0.0 if value is False else float(value)
        for value in values
    ]
    return (
        sum(numeric_values) / len(numeric_values)
        if numeric_values else None
    )


def _recall(rows: Sequence[MatchPrediction], expected_class: str) -> float:
    relevant = [row for row in rows if row.expected_class == expected_class]
    return _mean(row.predicted_class == expected_class for row in relevant)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "na"}:
            return None
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _decision_from_kind(kind: Any) -> str:
    if kind is None:
        return COMMIT_NONE
    text = str(kind).strip().lower()
    return text if text in COMMIT_DECISIONS else COMMIT_NONE


def run_matcher_ablation(
    config: MatcherSettings = MatcherSettings(),
    settings: Settings = DEFAULT_SETTINGS,
    matchers: Optional[Sequence[Matcher]] = None,
) -> Dict[str, Any]:
    """Run the standalone stress suite and return JSON-serializable results."""

    cases = generate_match_cases(config)
    calibration_report: Dict[str, Any] = {}
    if matchers is not None:
        matcher_list = tuple(matchers)
        calibration_report = {
            "status": "caller_supplied_matchers",
            "matchers": [matcher.name for matcher in matcher_list],
        }
    elif config.calibrate:
        calibration_config = replace(
            config,
            seed=int(config.seed) + int(config.calibration_seed_offset),
            case_limit=int(config.calibration_case_limit),
            include_prefixes=False,
            calibrate=False,
        )
        calibration_cases = generate_match_cases(calibration_config)
        matcher_list, reports = calibrated_matchers(calibration_cases, settings)
        calibration_report = {
            "status": "completed",
            "seed": calibration_config.seed,
            "n_cases": len(calibration_cases),
            "reports": reports,
        }
    else:
        matcher_list = default_matchers(settings)
        calibration_report = {"status": "disabled"}
    predictions = evaluate_cases(cases, matcher_list)
    matcher_names = sorted({row.matcher for row in predictions})
    primary_predictions = [
        row for row in predictions
        if bool(row.metadata.get("primary_accuracy", True))
        and not bool(row.metadata.get("prefix_case"))
    ]
    prefix_predictions = [
        row for row in predictions if bool(row.metadata.get("prefix_case"))
    ]
    boundary_predictions = [
        row for row in predictions
        if row.metadata.get("case_category") == "controlled_boundary"
    ]
    by_matcher = {
        matcher: summarize_predictions([
            row for row in primary_predictions if row.matcher == matcher
        ])
        for matcher in matcher_names
    }
    result = {
        "definition": (
            "Standalone recipe-vs-preference matcher stress suite. "
            "Executable task variations are state-replayed and separated from "
            "recognition corruption and redundant-action anomalies. Exact rational "
            "overlap probes are reported outside natural-task accuracy."
        ),
        "config": asdict(config),
        "model_config": {
            "match_threshold": settings.match_threshold,
            "match_margin": settings.match_margin,
        },
        "n_cases": len(cases),
        "n_primary_cases": len({row.case_id for row in primary_predictions}),
        "n_prefix_cases": len({row.case_id for row in prefix_predictions}),
        "n_controlled_boundary_cases": len({row.case_id for row in boundary_predictions}),
        "case_family_counts": dict(Counter(case.family for case in cases)),
        "case_validity_by_family": {
            family: {
                "n": len(rows),
                "n_executable": sum(bool(row.metadata.get("valid_ordering")) for row in rows),
                "case_category": sorted({str(row.metadata.get("case_category")) for row in rows}),
            }
            for family, rows in sorted({
                family: [case for case in cases if case.family == family]
                for family in {case.family for case in cases}
            }.items())
        },
        "baseline_calibration": calibration_report,
        "summary_by_matcher": by_matcher,
        "summary_all_cases_by_matcher": {
            matcher: summarize_predictions([row for row in predictions if row.matcher == matcher])
            for matcher in matcher_names
        },
        "open_set_behavior_if_used_directly_in_assist": summarize_assist_by_matcher(
            primary_predictions,
        ),
        "authoritative_assist_contract": {
            "observation_mode": "open_set_recipe_classification_and_recipe_creation",
            "assist_mode": "closed_set_known_recipe_inference",
            "low_confidence": "abstain_from_self_training_without_requesting_observation",
            "no_known_candidates": "assist_unavailable_protocol_error",
        },
        "summary_by_matcher_and_family": {
            matcher: summarize_by(
                [row for row in primary_predictions if row.matcher == matcher],
                "family",
            )
            for matcher in matcher_names
        },
        "summary_by_matcher_and_overlap_bin": {
            matcher: summarize_by(
                [row for row in primary_predictions if row.matcher == matcher],
                "overlap_bin",
            )
            for matcher in matcher_names
        },
        "prefix_summary_by_matcher_and_length": {
            matcher: summarize_by(
                [row for row in prefix_predictions if row.matcher == matcher],
                "prefix_length",
            )
            for matcher in matcher_names
        },
        "controlled_boundary_by_matcher_and_target": {
            matcher: summarize_by(
                [row for row in boundary_predictions if row.matcher == matcher],
                "target_overlap",
            )
            for matcher in matcher_names
        },
        "threshold_sweep": threshold_sweep(
            [case for case in cases if bool(case.metadata.get("primary_accuracy", True)) and not case.metadata.get("prefix_case")],
            (0.90, 0.93, 0.95, 0.9583, 0.96, 0.97, 0.98, 0.99),
            settings,
        ),
        "production_controlled_boundary_threshold_sweep": threshold_sweep(
            [
                case for case in cases
                if case.metadata.get("case_category") == "controlled_boundary"
            ],
            (0.90, 0.93, 0.95, 0.9583, 0.96, 0.97, 0.98, 0.99),
            settings,
        ),
        "confidence_gated_self_training_audit": commit_audit_spec(),
        "cases": [case.to_jsonable() for case in cases],
        "predictions": [prediction.to_jsonable() for prediction in predictions],
    }
    return result



def _trend(stream: Any) -> List[Dict[str, Any]]:
    """Summarize periodic performance at each ladder phase."""
    rows = list(stream.episode_rows)
    phases = sorted({
        int(row["phase_index"])
        for row in rows if isinstance(row.get("phase_index"), int)
    })

    def total(items: Sequence[Mapping[str, Any]], key: str) -> float:
        return float(sum(
            float(row[key]) for row in items
            if isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
        ))

    out: List[Dict[str, Any]] = []
    for phase in phases:
        current = [row for row in rows if row.get("phase_index") == phase]
        seen = [
            row for row in rows
            if isinstance(row.get("phase_index"), int)
            and int(row["phase_index"]) <= phase
        ]
        assists = [row for row in current if row.get("mode") == "assist"]
        past_assists = [row for row in seen if row.get("mode") == "assist"]
        last_event = max((int(row.get("event_index", -1)) for row in seen), default=-1)
        probes = [
            row for row in stream.frozen_rows
            if int(row.get("event_index", -1)) <= last_event
            and not str(row.get("checkpoint", "")).startswith("pre_event")
        ]
        memory = [
            row for row in stream.memory_rows
            if int(row.get("event_index", -1)) <= last_event
        ]
        snapshot = memory[-1] if memory else {}
        out.append({
            "phase": phase,
            "events": len(current),
            "events_seen": len(seen),
            "demos_seen": sum(row.get("mode") == "observe" for row in seen),
            "demo_actions_seen": total(
                [row for row in seen if row.get("mode") == "observe"],
                "recipe_steps",
            ),
            "phase_top_1": _safe_div(
                total(assists, "teacher_forced_correct_count"),
                total(assists, "teacher_forced_prediction_count"),
            ),
            "top_1_seen": _safe_div(
                total(past_assists, "teacher_forced_correct_count"),
                total(past_assists, "teacher_forced_prediction_count"),
            ),
            "phase_nll": _safe_div(
                total(assists, "teacher_forced_total_nll"),
                total(assists, "teacher_forced_prediction_count"),
            ),
            "probe_top_1": _finite_mean(row.get("top_1") for row in probes),
            "probe_count": len(probes),
            "active_variants": snapshot.get("active_variants"),
            "pruned_variants": snapshot.get("pruned_variants"),
            "retrain_count": snapshot.get("training_retrain_count"),
        })
    return out


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return sum(numeric) / len(numeric) if numeric else None



def teaching_metrics(
    episode_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Measure explicit demonstrations and corrections over the complete stream."""
    observations = [row for row in episode_rows if row.get("mode") == "observe"]
    assists = [row for row in episode_rows if row.get("mode") == "assist"]
    planned_observations = [
        row for row in observations if row.get("requested_mode") == "observe"
    ]
    extra_observations = [
        row for row in observations if row.get("requested_mode") == "assist"
    ]

    def total(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return float(sum(
            float(row.get(key))
            for row in rows
            if isinstance(row.get(key), (int, float))
            and math.isfinite(float(row.get(key)))
        ))

    n_observation_actions = total(observations, "recipe_steps")
    n_planned_observation_actions = total(
        planned_observations, "recipe_steps",
    )
    n_extra_observation_actions = total(extra_observations, "recipe_steps")
    n_corrections = total(assists, "hrc_human_correction_count")
    n_human_turns = total(assists, "hrc_human_turn_count")
    n_human_actions = n_human_turns + n_corrections
    robot_turn_count = total(assists, "hrc_robot_turn_count")
    n_robot_correct = total(assists, "hrc_robot_correct_count")
    n_assist_recipe_steps = total(assists, "recipe_steps")
    fixed_checkpoint_rows = [
        row for row in frozen_rows if row.get("probe_phase") != "pre_event"
    ]
    checkpoints = {
        str(row.get("checkpoint", "unknown")) for row in fixed_checkpoint_rows
    }
    return {
        "n_events": len(episode_rows),
        "n_assist_episodes": len(assists),
        "n_observation_episodes": len(observations),
        "observation_episode_rate": _safe_div(len(observations), len(episode_rows)),
        "n_planned_observation_episodes": len(planned_observations),
        "n_extra_observation_episodes": len(extra_observations),
        "n_explicit_demonstration_actions": n_observation_actions,
        "n_planned_demonstration_actions": n_planned_observation_actions,
        "n_extra_demonstration_actions": n_extra_observation_actions,
        "n_corrective_teaching_actions": n_corrections,
        "n_total_explicit_teaching_actions": n_observation_actions + n_corrections,
        "assist_robot_turns": robot_turn_count,
        "assist_recipe_steps": n_assist_recipe_steps,
        "assist_human_actions": n_human_actions,
        "assist_live_top_1_selection_affected": _safe_div(
            n_robot_correct, robot_turn_count,
        ),
        "assist_teacher_forced_top_1": _safe_div(
            total(assists, "teacher_forced_correct_count"),
            total(assists, "teacher_forced_prediction_count"),
        ),
        "assist_normalized_human_action_load": _safe_div(
            n_human_actions, n_assist_recipe_steps,
        ),
        "assist_mean_corrections_per_task": _safe_div(
            n_corrections, len(assists),
        ),
        "assist_corrections_per_recipe_step": _safe_div(
            n_corrections, n_assist_recipe_steps,
        ),
        "fixed_checkpoint_n_checkpoints": len(checkpoints),
        "fixed_checkpoint_n_rows": len(fixed_checkpoint_rows),
        "fixed_checkpoint_top_1": _finite_mean(
            row.get("top_1") for row in fixed_checkpoint_rows
        ),
        "fixed_checkpoint_top_k": _finite_mean(
            row.get("top_k") for row in fixed_checkpoint_rows
        ),
        "fixed_checkpoint_normalized_human_action_load": _safe_div(
            total(fixed_checkpoint_rows, "hrc_human_turn_count")
            + total(fixed_checkpoint_rows, "hrc_human_correction_count"),
            total(fixed_checkpoint_rows, "recipe_steps"),
        ),
        "fixed_checkpoint_mean_corrections_per_task": _safe_div(
            total(fixed_checkpoint_rows, "hrc_human_correction_count"),
            len(fixed_checkpoint_rows),
        ),
        "fixed_checkpoint_corrections_per_recipe_step": _safe_div(
            total(fixed_checkpoint_rows, "hrc_human_correction_count"),
            total(fixed_checkpoint_rows, "recipe_steps"),
        ),
    }


def _teaching_row(
    stream: Any,
    routing_condition: str,
) -> Dict[str, Any]:
    system = dict(stream.agent.diagnostics().get("fit_stats") or {})
    return {
        "scenario": str(stream.scenario),
        "seed": int(stream.seed),
        "baseline": str(stream.baseline),
        "routing_condition": str(routing_condition),
        **teaching_metrics(stream.episode_rows, stream.frozen_rows),
        "training_retrain_count": len(stream.agent.retrain_fit_wall_times),
        "training_estimated_fit_flops": float(sum(stream.agent.retrain_flop_estimates)),
        "model_family": system.get("model_family"),
        "trend": _trend(stream),
    }


LATENT_ABLATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("teacher_forced_top_1", "higher_is_better"),
    ("teacher_forced_top_k", "higher_is_better"),
    ("teacher_forced_mean_nll", "lower_is_better"),
    ("live_top_1", "higher_is_better"),
    ("normalized_human_action_load", "lower_is_better"),
    ("corrections_per_recipe_step", "lower_is_better"),
    ("cross_recipe_teacher_forced_top_1", "higher_is_better"),
    (
        "seen_recipe_preference_seen_elsewhere_top_1",
        "higher_is_better",
    ),
    ("mean_prediction_wall_s", "lower_is_better"),
    ("testing_episode_wall_s", "lower_is_better"),
    ("training_fit_wall_s", "lower_is_better"),
    ("training_estimated_fit_flops", "lower_is_better"),
)


def _latent_ablation_row(
    stream: Any,
    arm: LatentStrategyAblationArm,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Flatten headline accuracy/transfer/compute metrics for one arm."""
    assist = summary.get("assist") or {}
    overall = assist.get("overall") or {}
    by_hypothesis = assist.get("by_hypothesis") or {}
    cross_recipe = by_hypothesis.get("cross_recipe_transfer") or {}
    by_transfer = assist.get("by_transfer_cell") or {}
    seen_elsewhere = (
        by_transfer.get("seen_recipe_preference_seen_elsewhere") or {}
    )
    system = summary.get("system") or {}
    fit_stats = system.get("fit_stats") or {}
    return {
        "scenario": str(stream.scenario),
        "seed": int(stream.seed),
        "arm": arm.name,
        "label": arm.label,
        "model_overrides": arm.model_overrides(),
        "teacher_forced_top_1": overall.get("teacher_forced_top_1"),
        "teacher_forced_top_k": overall.get("teacher_forced_top_k"),
        "teacher_forced_mean_nll": overall.get("teacher_forced_mean_nll"),
        "live_top_1": overall.get("live_top_1"),
        "normalized_human_action_load": overall.get(
            "normalized_human_action_load"
        ),
        "corrections_per_recipe_step": overall.get(
            "corrections_per_recipe_step"
        ),
        "cross_recipe_teacher_forced_top_1": cross_recipe.get(
            "teacher_forced_top_1"
        ),
        "cross_recipe_prediction_count": cross_recipe.get(
            "n_teacher_forced_predictions"
        ),
        "seen_recipe_preference_seen_elsewhere_top_1": seen_elsewhere.get(
            "teacher_forced_top_1"
        ),
        "seen_recipe_preference_seen_elsewhere_prediction_count": (
            seen_elsewhere.get("n_teacher_forced_predictions")
        ),
        "mean_prediction_wall_s": overall.get("mean_prediction_wall_s"),
        "testing_episode_wall_s": overall.get("testing_episode_wall_s"),
        "stream_wall_s": float(stream.wall_s),
        "training_fit_wall_s": system.get("training_fit_wall_s"),
        "training_estimated_fit_flops": system.get(
            "training_estimated_fit_flops"
        ),
        "model_family": fit_stats.get("model_family"),
        "latent_strategy_parameter_count": fit_stats.get(
            "latent_strategy_parameter_count"
        ),
        "latent_strategy_prototype_storage_values": fit_stats.get(
            "latent_strategy_prototype_storage_values"
        ),
        "trend": _trend(stream),
        # Preserve the detailed phase, transfer-cell, adaptation, and memory
        # summaries without writing dense per-turn logs in the ablation JSON.
        "detailed_metrics": dict(summary),
    }


def summarize_latent_strategy_ablation(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate arm means and matched-seed treatment-minus-reference deltas."""
    metric_directions = dict(LATENT_ABLATION_METRICS)
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    indexed: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        scenario = str(row.get("scenario"))
        arm = str(row.get("arm"))
        grouped[(scenario, arm)].append(row)
        seed = row.get("seed")
        if isinstance(seed, int):
            indexed[(scenario, int(seed), arm)] = row

    means = []
    for (scenario, arm), group in sorted(grouped.items()):
        means.append({
            "scenario": scenario,
            "arm": arm,
            "n_seeds": len(group),
            **{
                f"mean_{metric}": _finite_mean(
                    row.get(metric) for row in group
                )
                for metric in metric_directions
            },
        })

    paired = []
    scenarios = sorted({scenario for scenario, _seed, _arm in indexed})
    for contrast in LATENT_STRATEGY_ABLATION_CONTRASTS:
        reference = contrast["reference"]
        treatment = contrast["treatment"]
        for scenario in scenarios:
            seeds = sorted({
                seed for row_scenario, seed, arm in indexed
                if row_scenario == scenario
                and arm == treatment
                and (scenario, seed, reference) in indexed
            })
            metric_deltas: Dict[str, Any] = {}
            for metric, direction in LATENT_ABLATION_METRICS:
                deltas = []
                for seed in seeds:
                    treatment_value = indexed[(
                        scenario, seed, treatment,
                    )].get(metric)
                    reference_value = indexed[(
                        scenario, seed, reference,
                    )].get(metric)
                    if not isinstance(treatment_value, (int, float)) or not (
                        isinstance(reference_value, (int, float))
                    ):
                        continue
                    delta = float(treatment_value) - float(reference_value)
                    if math.isfinite(delta):
                        deltas.append(delta)
                metric_deltas[metric] = {
                    "direction": direction,
                    "n_paired_seeds": len(deltas),
                    "mean_treatment_minus_reference": _finite_mean(deltas),
                }
            paired.append({
                **dict(contrast),
                "scenario": scenario,
                "n_paired_seeds": len(seeds),
                "metrics": metric_deltas,
            })
    return {
        "mean_by_scenario_arm": means,
        "paired_seed_contrasts": paired,
    }


def run_latent_strategy_ablation(
    evaluation_config: Optional[Any] = None,
    arms: Sequence[LatentStrategyAblationArm] = LATENT_STRATEGY_ABLATION_ARMS,
    arms_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the four paired longitudinal MaxEnt/latent conditions."""
    from .evaluation import EvalSettings

    selected_arms = tuple(arms)
    names = tuple(arm.name for arm in selected_arms)
    if len(selected_arms) != 4 or len(set(names)) != 4:
        raise ValueError(
            "latent strategy ablation requires the four unique registered arms"
        )
    expected_names = tuple(arm.name for arm in LATENT_STRATEGY_ABLATION_ARMS)
    if set(names) != set(expected_names):
        raise ValueError(
            f"latent strategy ablation arms must be {expected_names}"
        )

    config = evaluation_config or EvalSettings(
        seeds=(1337,),
        baselines=("full",),
        include_oracle=False,
        experiment="latent_strategy_ablation",
    )
    config = replace(
        config,
        baselines=("full",),
        include_oracle=False,
        shared_routing=True,
        observe_missing_recipes=False,
        allow_repeat_observation=False,
        audit_period=0,
        sensitivity=False,
        experiment="latent_strategy_ablation",
    )
    rows = run_arm_major_suite(
        "latent_strategy_ablation", selected_arms, config, arms_dir=arms_dir,
    )
    return {
        "definition": (
            "Four-arm paired longitudinal ablation of MaxEnt, lightweight "
            "latent timing, higher-capacity timing, and capacity-matched "
            "timing plus masked-role trajectory alignment."
        ),
        "design_type": "paired_latent_strategy_component_ablation",
        "design": latent_strategy_ablation_design(),
        "primary_trajectory_contrast": (
            "latent_timing_trajectory_hybrid minus "
            "latent_timing_capacity_matched"
        ),
        "rows": rows,
        "summary": summarize_latent_strategy_ablation(rows),
    }


ARM_ABLATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("teacher_forced_top_1", "higher_is_better"),
    ("teacher_forced_top_k", "higher_is_better"),
    ("teacher_forced_mean_nll", "lower_is_better"),
    ("live_top_1", "higher_is_better"),
    ("normalized_human_action_load", "lower_is_better"),
    ("corrections_per_recipe_step", "lower_is_better"),
)


def _arm_ablation_row(stream: Any, arm: Any, summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten one settings-override arm, plus the state its factor moves.

    Shared by the component and retention suites: both hold everything except
    one declared factor at Full's configuration, so both need the same
    accuracy, workload, retention-state and deployed-cost columns. The
    retention-state columns are what let a summary verify that an arm's factor
    actually moved rather than merely being labelled as moved.
    """
    assist = summary.get("assist") or {}
    overall = assist.get("overall") or {}
    diagnostics = summary.get("diagnostics") or {}
    by_memory_state = assist.get("by_memory_state") or {}
    by_preference_count = assist.get("by_active_preference_count") or {}
    adaptation = diagnostics.get("adaptation_speed") or {}
    system = summary.get("system") or {}
    fit_stats = system.get("fit_stats") or {}
    training = summary.get("training") or {}
    per_phase_role = training.get("per_phase_role") or {}

    def phase_max(metric: str) -> Optional[float]:
        values = [
            float(entry[metric]) for entry in per_phase_role.values()
            if isinstance(entry, Mapping) and isinstance(entry.get(metric), (int, float))
        ]
        return max(values) if values else None

    def phase_sum(metric: str) -> Optional[float]:
        values = [
            float(entry[metric]) for entry in per_phase_role.values()
            if isinstance(entry, Mapping) and isinstance(entry.get(metric), (int, float))
        ]
        return sum(values) if values else None

    def memory_state(name: str, metric: str) -> Optional[float]:
        cell = by_memory_state.get(name) or {}
        value = cell.get(metric)
        return float(value) if isinstance(value, (int, float)) else None

    row: Dict[str, Any] = {
        "scenario": str(stream.scenario),
        "seed": int(stream.seed),
        "arm": arm.name,
        "label": arm.label,
        "model_overrides": arm.model_overrides(),
        # Realized configuration, read back from the run rather than from the
        # arm definition, so a factor that silently failed to apply is visible.
        "memory_policy": system.get("memory_policy"),
        "latest_pin_enabled": system.get("latest_pin_enabled"),
        "semantic_fallback_enabled": fit_stats.get("semantic_fallback_enabled"),
        "latent_strategy_enabled": system.get("latent_strategy_enabled"),
        "active_variants": system.get("active_variants"),
        "pruned_variants": system.get("pruned_variants"),
        "mean_active_weight": system.get("mean_active_weight"),
        "mean_pair_grace_horizon_demos": system.get("mean_pair_grace_horizon_demos"),
        "max_pair_grace_horizon_demos": system.get("max_pair_grace_horizon_demos"),
        "mean_pair_recurrence_samples": system.get("mean_pair_recurrence_samples"),
        # Selective-versus-catastrophic forgetting: accuracy on pairs whose
        # variant is currently pruned out of active fitting, next to how many
        # decisions were taken in that state. A retention rule is only better
        # than another if it wins on both together.
        "pruned_state_top_1": memory_state("pruned_memory", "teacher_forced_top_1"),
        "pruned_state_decisions": memory_state("pruned_memory", "n_teacher_forced_predictions"),
        "active_state_top_1": memory_state("active_memory", "teacher_forced_top_1"),
        "active_state_decisions": memory_state("active_memory", "n_teacher_forced_predictions"),
        "new_preference_top_1": memory_state("same_recipe_new_preference", "teacher_forced_top_1"),
        "new_preference_decisions": memory_state("same_recipe_new_preference", "n_teacher_forced_predictions"),
        # Concurrency: the regime a single per-recipe pin is expected to hurt.
        "top_1_by_active_preference_count": {
            str(count): (cell or {}).get("teacher_forced_top_1")
            for count, cell in sorted(by_preference_count.items())
        },
        "decisions_by_active_preference_count": {
            str(count): (cell or {}).get("n_teacher_forced_predictions")
            for count, cell in sorted(by_preference_count.items())
        },
        "recovered_rate": adaptation.get("recovered_rate"),
        "mean_exposures_to_recover_90pct": adaptation.get("mean_exposures_to_recover_90pct"),
        "first_post_switch_exposure_top_1": adaptation.get(
            "first_post_switch_exposure_teacher_forced_top_1"
        ),
        "mean_pre_switch_top_1": adaptation.get("mean_pre_switch_teacher_forced_top_1"),
        # Deployed cost. p95 is the blocking wait between two demonstrations,
        # which cumulative totals cannot express.
        "online_p95_retrain_fit_wall_s": phase_max("online_p95_retrain_fit_wall_s"),
        "online_training_fit_wall_s": phase_sum("online_training_fit_wall_s"),
        "online_retrain_fit_count": phase_sum("online_retrain_fit_count"),
        "online_training_estimated_fit_flops": phase_sum("online_training_estimated_fit_flops"),
        "mean_prediction_wall_s": overall.get("mean_prediction_wall_s"),
        "stream_wall_s": float(stream.wall_s),
        "trend": _trend(stream),
        "detailed_metrics": dict(summary),
    }
    row.update({metric: overall.get(metric) for metric, _direction in ARM_ABLATION_METRICS})
    return row


def _paired_arm_deltas(
    indexed: Mapping[Tuple[str, int, str], Mapping[str, Any]],
    scenario: str,
    treatment: str,
    reference: str,
    metrics: Sequence[Tuple[str, str]] = ARM_ABLATION_METRICS,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Seed-matched treatment-minus-reference deltas, signed so positive favours treatment."""
    seeds = sorted({
        seed for row_scenario, seed, arm in indexed
        if row_scenario == scenario and arm == treatment
        and (scenario, seed, reference) in indexed
    })
    metric_deltas: Dict[str, Any] = {}
    raw: Dict[str, List[float]] = {}
    for metric, direction in metrics:
        deltas: List[float] = []
        for seed in seeds:
            treatment_value = indexed[(scenario, seed, treatment)].get(metric)
            reference_value = indexed[(scenario, seed, reference)].get(metric)
            if not isinstance(treatment_value, (int, float)) or not isinstance(reference_value, (int, float)):
                continue
            delta = float(treatment_value) - float(reference_value)
            if direction == "lower_is_better":
                delta = -delta
            if math.isfinite(delta):
                deltas.append(delta)
        raw[metric] = deltas
        metric_deltas[metric] = {
            "direction": direction,
            "n_paired_seeds": len(deltas),
            "mean_treatment_advantage": _finite_mean(deltas),
        }
    return {"scenario": scenario, "n_paired_seeds": len(seeds), "metrics": metric_deltas}, raw


def _index_arm_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[str, str], List[Mapping[str, Any]]], Dict[Tuple[str, int, str], Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    indexed: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        scenario = str(row.get("scenario"))
        arm = str(row.get("arm"))
        grouped[(scenario, arm)].append(row)
        seed = row.get("seed")
        if isinstance(seed, int):
            indexed[(scenario, int(seed), arm)] = row
    return grouped, indexed


def _arm_cell_means(
    grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    extra_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for (scenario, arm), group in sorted(grouped.items()):
        first = group[0]
        cells.append({
            "scenario": scenario,
            "arm": arm,
            "label": first.get("label"),
            "n_seeds": len(group),
            "model_overrides": first.get("model_overrides"),
            **{key: first.get(key) for key in ("memory_policy", "latest_pin_enabled",
                                               "semantic_fallback_enabled", "latent_strategy_enabled")},
            **{
                f"mean_{key}": _finite_mean(row.get(key) for row in group)
                for key in extra_keys
            },
            **{
                f"mean_{metric}": _finite_mean(row.get(metric) for row in group)
                for metric, _direction in ARM_ABLATION_METRICS
            },
        })
    return cells


_ARM_SHARED_KEYS: Tuple[str, ...] = (
    "active_variants", "pruned_variants", "mean_active_weight",
    "mean_pair_grace_horizon_demos", "mean_pair_recurrence_samples",
    "pruned_state_top_1", "pruned_state_decisions",
    "active_state_top_1", "active_state_decisions",
    "new_preference_top_1", "new_preference_decisions",
    "recovered_rate", "mean_exposures_to_recover_90pct",
    "first_post_switch_exposure_top_1",
    "online_p95_retrain_fit_wall_s", "online_training_fit_wall_s",
    "online_retrain_fit_count", "online_training_estimated_fit_flops",
)


def _concurrency_profile(
    grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """Per-arm accuracy against the number of concurrently active preferences.

    Reported separately because a single per-recipe pin is expected to degrade
    precisely as this count rises, and the scenario where retention helps most
    is also the one that never exercises the regime.
    """
    profile: List[Dict[str, Any]] = []
    for (scenario, arm), group in sorted(grouped.items()):
        counts: Dict[str, List[float]] = defaultdict(list)
        decisions: Dict[str, List[float]] = defaultdict(list)
        for row in group:
            for count, value in (row.get("top_1_by_active_preference_count") or {}).items():
                if isinstance(value, (int, float)):
                    counts[str(count)].append(float(value))
            for count, value in (row.get("decisions_by_active_preference_count") or {}).items():
                if isinstance(value, (int, float)):
                    decisions[str(count)].append(float(value))
        ordered = sorted(counts, key=lambda item: (len(item), item))
        single = _finite_mean(counts.get("1", []))
        highest = _finite_mean(counts.get(ordered[-1], [])) if ordered else None
        profile.append({
            "scenario": scenario,
            "arm": arm,
            "n_seeds": len(group),
            "mean_top_1_by_active_preference_count": {
                count: _finite_mean(counts[count]) for count in ordered
            },
            "total_decisions_by_active_preference_count": {
                count: sum(decisions.get(count, [])) for count in ordered
            },
            "highest_observed_active_preference_count": ordered[-1] if ordered else None,
            # Negative means the arm loses accuracy as preferences pile up.
            "concurrency_degradation_single_minus_highest": (
                None if single is None or highest is None else float(single - highest)
            ),
        })
    return profile


def summarize_component_ablation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-component simple effects and an additivity check against the joint drop."""
    grouped, indexed = _index_arm_rows(rows)
    scenarios = sorted({scenario for scenario, _seed, _arm in indexed})
    single_factor = ("full_no_pin", "full_no_semantic_fallback", "full_no_latent_residual")

    simple_effects: List[Dict[str, Any]] = []
    raw_by_arm: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for arm in (*single_factor, "full_no_support"):
        for scenario in scenarios:
            # Signed so a positive value means removing the component HURT,
            # i.e. the component was contributing that much.
            entry, raw = _paired_arm_deltas(indexed, scenario, "full", arm)
            simple_effects.append({
                "component_removed": arm,
                "question": f"what does Full lose when {arm} drops its component(s)",
                **entry,
            })
            raw_by_arm[(arm, scenario)] = raw

    additivity: List[Dict[str, Any]] = []
    for scenario in scenarios:
        metrics: Dict[str, Any] = {}
        for metric, direction in ARM_ABLATION_METRICS:
            joint = raw_by_arm.get(("full_no_support", scenario), {}).get(metric) or []
            parts = [raw_by_arm.get((arm, scenario), {}).get(metric) or [] for arm in single_factor]
            paired = min([len(joint)] + [len(part) for part in parts]) if joint and all(parts) else 0
            summed = [sum(part[index] for part in parts) for index in range(paired)]
            metrics[metric] = {
                "direction": direction,
                "n_paired_seeds": paired,
                "mean_joint_removal_cost": _finite_mean(joint[:paired]),
                "mean_summed_single_removal_cost": _finite_mean(summed),
                # Positive means the joint removal costs more than the parts
                # sum to, i.e. the components are complementary rather than
                # independent contributors.
                "mean_superadditivity": _finite_mean(
                    joint[index] - summed[index] for index in range(paired)
                ),
            }
        additivity.append({
            "scenario": scenario,
            "definition": (
                "joint removal of all three components minus the sum of the "
                "three single-factor removals; near zero means the components "
                "contribute independently"
            ),
            "metrics": metrics,
        })

    # Each arm must have moved exactly the component it names and nothing else.
    expected = {
        "full": (True, True, True),
        "full_no_pin": (False, True, True),
        "full_no_semantic_fallback": (True, False, True),
        "full_no_latent_residual": (True, True, False),
        "full_no_support": (False, False, False),
    }
    realized = {
        arm: sorted({
            (bool(row.get("latest_pin_enabled")), bool(row.get("semantic_fallback_enabled")),
             bool(row.get("latent_strategy_enabled")))
            for row in rows if str(row.get("arm")) == arm
        })
        for arm in expected
    }
    factors_applied = all(
        len(realized.get(arm, [])) == 1 and realized[arm][0] == expected[arm]
        for arm in expected if realized.get(arm)
    )
    return {
        "mean_by_scenario_arm": _arm_cell_means(grouped, _ARM_SHARED_KEYS),
        "paired_seed_component_costs": simple_effects,
        "component_additivity": additivity,
        "concurrency_profile": _concurrency_profile(grouped),
        "realized_component_flags_by_arm": {
            arm: [list(item) for item in value] for arm, value in realized.items()
        },
        "declared_factors_were_applied": bool(factors_applied),
        "retention_held_constant": len({
            str(row.get("memory_policy")) for row in rows
        }) == 1,
    }


def summarize_retention_ablation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-mechanism simple effects against Full, grouped by the factor moved."""
    grouped, indexed = _index_arm_rows(rows)
    scenarios = sorted({scenario for scenario, _seed, _arm in indexed})
    arms_by_name = {arm.name: arm for arm in RETENTION_ABLATION_ARMS}
    treatments = [name for name in arms_by_name if name != "full"]

    simple_effects: List[Dict[str, Any]] = []
    for arm in treatments:
        for scenario in scenarios:
            entry, _raw = _paired_arm_deltas(indexed, scenario, "full", arm)
            simple_effects.append({
                "reference_arm": arm,
                "factor": arms_by_name[arm].factor,
                "question": f"what does Full lose against {arm}",
                **entry,
            })

    # Retention arms are only interpretable if predictor support really was
    # held constant, which is the failure the roster baselines have.
    support = sorted({
        (bool(row.get("semantic_fallback_enabled")), bool(row.get("latent_strategy_enabled")))
        for row in rows
    })
    return {
        "mean_by_scenario_arm": _arm_cell_means(grouped, _ARM_SHARED_KEYS),
        "paired_seed_simple_effects": simple_effects,
        "concurrency_profile": _concurrency_profile(grouped),
        "factors_by_arm": {
            arm.name: {"factor": arm.factor, "overrides": arm.model_overrides()}
            for arm in RETENTION_ABLATION_ARMS
        },
        "realized_predictor_support": [list(item) for item in support],
        "predictor_support_held_constant": len(support) == 1,
    }


# --- Arm-major ablation driver ------------------------------------------
#
# Every longitudinal ablation suite has the same shape: a list of arms that
# differ only in settings, run over the same scenario/seed grid, with the
# first arm's realized observe/assist sequence becoming the route the rest
# replay. They used to be scheduled cell-major -- one process per
# scenario/seed running every arm -- which meant one changed arm invalidated
# the whole grid. They are now scheduled arm-major: an arm finishes the whole
# grid before the next starts, and its rows are written to its own folder.


def _arm_suite_baseline_fixed_full(_arm: Any) -> str:
    return "full"


def _arm_suite_baseline_attribute(arm: Any) -> str:
    return str(arm.baseline)


def _arm_suite_baseline_mapping(arm: Any) -> str:
    return str(arm["baseline"])


def _arm_suite_overrides_method(arm: Any) -> Mapping[str, Any]:
    return dict(arm.model_overrides())


def _arm_suite_overrides_mapping(arm: Any) -> Mapping[str, Any]:
    return dict(arm.get("overrides") or {})


def _arm_suite_name_attribute(arm: Any) -> str:
    return str(arm.name)


def _arm_suite_name_mapping(arm: Any) -> str:
    return str(arm["arm"])


@dataclass(frozen=True)
class ArmSuiteSpec:
    """How one ablation suite names, configures and records its arms."""

    reference_policy: str
    matched_policy: str
    baseline: Callable[[Any], str]
    overrides: Callable[[Any], Mapping[str, Any]]
    row: Callable[..., Dict[str, Any]]
    name: Callable[[Any], str]
    # The representation suite also narrows the roster to its arm's predictor.
    set_roster: bool = False


def _arm_suite_specs() -> Dict[str, ArmSuiteSpec]:
    """Built lazily: the row builders are defined later in this module."""
    return {
        "component_ablation": ArmSuiteSpec(
            "component_ablation_full_reference_route",
            "component_ablation_matched_full_route",
            _arm_suite_baseline_fixed_full, _arm_suite_overrides_method,
            _arm_ablation_row, _arm_suite_name_attribute,
        ),
        "retention_ablation": ArmSuiteSpec(
            "retention_ablation_full_reference_route",
            "retention_ablation_matched_full_route",
            _arm_suite_baseline_fixed_full, _arm_suite_overrides_method,
            _arm_ablation_row, _arm_suite_name_attribute,
        ),
        "latent_strategy_ablation": ArmSuiteSpec(
            "latent_ablation_maxent_reference_route",
            "latent_ablation_matched_maxent_route",
            _arm_suite_baseline_fixed_full, _arm_suite_overrides_method,
            _latent_ablation_row, _arm_suite_name_attribute,
        ),
        "memory_predictor_ablation": ArmSuiteSpec(
            "memory_predictor_ablation_full_reference_route",
            "memory_predictor_ablation_matched_full_route",
            _arm_suite_baseline_mapping, _arm_suite_overrides_mapping,
            _memory_predictor_row, _arm_suite_name_mapping,
        ),
        "representation_ablation": ArmSuiteSpec(
            "representation_ablation_reference_route",
            "representation_ablation_matched_route",
            _arm_suite_baseline_attribute, _arm_suite_overrides_method,
            _representation_row, _arm_suite_name_attribute, set_roster=True,
        ),
    }


def _arm_cell_job(
    job: Tuple[str, str, int, Any, Any, Optional[Tuple[str, ...]]],
) -> Tuple[Dict[str, Any], Optional[Tuple[str, ...]]]:
    """Run one arm on one scenario/seed plan.

    Returns the arm's row and, when this cell defined the route, the schedule
    the remaining arms must replay.
    """
    suite, scenario, seed, config, arm, route = job
    from .evaluation import (
        _apply_native_thread_limit,
        _native_thread_count,
        _pin_to_performance_cores,
        build_plan,
        run_stream,
        summarize_stream,
    )
    from .memory import clear_caches

    spec = _arm_suite_specs()[str(suite)]
    _pin_to_performance_cores()
    _apply_native_thread_limit(_native_thread_count(config))
    plan = build_plan(scenario, config, int(seed))
    baseline = spec.baseline(arm)
    replacements: Dict[str, Any] = {
        "model_settings": {
            **dict(config.model_settings), **dict(spec.overrides(arm)),
        },
    }
    if spec.set_roster:
        replacements["baselines"] = (baseline,)
    arm_config = replace(config, **replacements)
    clear_caches()
    stream = run_stream(
        baseline,
        plan,
        arm_config,
        execution_mode_schedule=route,
        mode_schedule_policy=(
            spec.reference_policy if route is None else spec.matched_policy
        ),
    )
    realized = (
        tuple(str(row.get("mode")) for row in stream.episode_rows)
        if route is None else None
    )
    return spec.row(stream, arm, summarize_stream(stream)), realized


def run_arm_major_suite(
    suite: str,
    arms: Sequence[Any],
    config: Any,
    *,
    arms_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run one ablation suite arm by arm over the whole scenario/seed grid.

    The first arm is the reference: it runs unconstrained and publishes the
    route every later arm replays, so no arm can win or lose by being asked
    for a different amount of teaching. Because that route is data rather
    than a live variable, each later arm is independently re-runnable.
    """
    spec = _arm_suite_specs()[str(suite)]
    selected = tuple(arms)
    if not selected:
        raise ValueError(f"{suite} needs at least one arm")
    names = tuple(spec.name(arm) for arm in selected)
    if len(set(names)) != len(names):
        raise ValueError(f"{suite} arms must have distinct names")
    grid = [
        (str(scenario), int(seed))
        for scenario in config.scenarios for seed in config.seeds
    ]
    workers = max(1, min(int(config.workers or 1), len(grid)))
    routes: Dict[Tuple[str, int], Tuple[str, ...]] = {}
    rows: List[Dict[str, Any]] = []
    for index, arm in enumerate(selected):
        arm_name = names[index]
        is_reference = index == 0
        jobs = {
            cell: (
                str(suite), cell[0], cell[1], config, arm,
                None if is_reference else routes[cell],
            )
            for cell in grid
        }
        results: Dict[Tuple[str, int], Tuple[Dict[str, Any], Optional[Tuple[str, ...]]]] = {}
        if workers == 1 or len(grid) == 1:
            for cell, job in jobs.items():
                results[cell] = _arm_cell_job(job)
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {
                    pool.submit(_arm_cell_job, job): cell
                    for cell, job in jobs.items()
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
        arm_rows: List[Dict[str, Any]] = []
        for cell in grid:
            row, realized = results[cell]
            if realized is not None:
                routes[cell] = realized
            arm_rows.append(row)
        rows.extend(arm_rows)
        if arms_dir is not None:
            Path(arms_dir).mkdir(parents=True, exist_ok=True)
            _atomic_json(Path(arms_dir) / f"{_safe_arm_component(arm_name)}.json", {
                "suite": str(suite),
                "arm": arm_name,
                "reference_arm": bool(is_reference),
                "route_policy": (
                    spec.reference_policy if is_reference else spec.matched_policy
                ),
                "n_cells": len(arm_rows),
                "rows": arm_rows,
            })
    order = {name: index for index, name in enumerate(names)}
    rows.sort(key=lambda row: (
        str(row["scenario"]), int(row["seed"]), order[str(row["arm"])],
    ))
    return rows


def _safe_arm_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(value)):
        raise ValueError(f"arm name is not usable as a filename: {value!r}")
    return str(value)


def _run_arm_override_suite(
    arms: Sequence[Any],
    experiment: str,
    route_tag: str,
    evaluation_config: Optional[Any] = None,
    arms_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Shared paired scenario/seed grid for the component and retention suites."""
    from .evaluation import EvalSettings

    selected = tuple(arms)
    names = tuple(arm.name for arm in selected)
    if len(set(names)) != len(names):
        raise ValueError(f"{experiment} arms must have distinct names")
    if names[0] != "full":
        raise ValueError(f"{experiment} requires the 'full' arm first; it sets the shared route")

    config = evaluation_config or EvalSettings(
        seeds=(1337,), baselines=("full",), include_oracle=False, experiment=experiment,
    )
    config = replace(
        config,
        baselines=("full",),
        include_oracle=False,
        shared_routing=True,
        observe_missing_recipes=False,
        allow_repeat_observation=False,
        audit_period=0,
        sensitivity=False,
        experiment=experiment,
    )
    return run_arm_major_suite(
        route_tag, selected, config, arms_dir=arms_dir,
    )


def run_component_ablation(
    evaluation_config: Optional[Any] = None,
    arms: Sequence[ComponentAblationArm] = COMPONENT_ABLATION_ARMS,
    arms_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the one-factor-at-a-time predictor-support ablation on Full."""
    rows = _run_arm_override_suite(
        arms, "component_ablation", "component_ablation", evaluation_config,
        arms_dir=arms_dir,
    )
    return {
        "definition": (
            "One-factor-at-a-time paired longitudinal ablation of Full's three "
            "predictor-support components -- the latest-preference pin, bounded "
            "semantic successor interpolation, and the workflow-strategy "
            "residual -- plus their joint removal. Retention, admission, the "
            "action mask and the consolidation schedule stay at Full's "
            "configuration in every arm, and all arms follow Full's realized "
            "interaction schedule on a shared plan."
        ),
        "design_type": "paired_predictor_support_component_ablation",
        "rationale": (
            "Every deployable memory baseline is constructed through "
            "_without_proposed_components, which removes all three components "
            "together, so a Full-versus-baseline margin cannot be assigned to "
            "any one of them and the deployed adaptive-decay arm in particular "
            "changes retention and support at the same time. Dropping one "
            "component at a time reports what each is worth, and the joint arm "
            "reports whether they are additive."
        ),
        "primary_contrast": "full minus full_no_pin",
        "component_levels": list(COMPONENT_LEVELS),
        "arms": [
            {"name": arm.name, "label": arm.label, "component_level": arm.component_level,
             "removes": list(arm.removes),
             "overrides": arm.model_overrides()}
            for arm in arms
        ],
        "rows": rows,
        "summary": summarize_component_ablation(rows),
    }


def run_retention_ablation(
    evaluation_config: Optional[Any] = None,
    arms: Sequence[RetentionAblationArm] = RETENTION_ABLATION_ARMS,
    arms_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the retention-mechanism ablation under Full's predictor support."""
    rows = _run_arm_override_suite(
        arms, "retention_ablation", "retention_ablation", evaluation_config,
        arms_dir=arms_dir,
    )
    return {
        "definition": (
            "Paired longitudinal ablation of the recurrence-aware retention "
            "policy. Arms move one retention mechanism each -- the retention "
            "rule, the horizon estimator, the pair adaptation asymmetry, the "
            "assignment of recurrence evidence to pairs, the pin scope, and "
            "the consolidation schedule -- while predictor support, admission, "
            "the action mask and the shared interaction route stay at Full's "
            "configuration."
        ),
        "design_type": "paired_retention_mechanism_ablation",
        "rationale": (
            "The deployed 'fixed' baseline is not a control for horizon "
            "estimation: ReplayMemory.step gates the grace check on the "
            "adaptive policy, so that arm has no grace period at all and also "
            "uses a different post-grace decrement, changing three things at "
            "once. The constant-grace arm here keeps the grace period, the "
            "decrement, the pin and every component and replaces only the "
            "per-pair horizon, which is what isolates estimating horizons from "
            "recurrence. The shuffled arm additionally holds the horizon "
            "distribution fixed and destroys only its assignment, separating "
            "the policy from the amount it happens to retain."
        ),
        "primary_contrast": "full minus constant_grace",
        "factors": list(RETENTION_FACTORS),
        "arms": [
            {"name": arm.name, "label": arm.label, "factor": arm.factor,
             "overrides": arm.model_overrides()}
            for arm in arms
        ],
        "rows": rows,
        "summary": summarize_retention_ablation(rows),
    }


def _memory_predictor_row(
    stream: Any,
    cell: Mapping[str, str],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Flatten one factorial cell, plus the retention state that defines it."""
    assist = summary.get("assist") or {}
    overall = assist.get("overall") or {}
    system = summary.get("system") or {}
    fit_stats = system.get("fit_stats") or {}
    row: Dict[str, Any] = {
        "scenario": str(stream.scenario),
        "seed": int(stream.seed),
        "arm": str(cell["arm"]),
        "baseline": str(cell["baseline"]),
        "model_overrides": dict(cell.get("overrides") or {}),
        "label": str(cell["label"]),
        "predictor_level": str(cell["predictor"]),
        "memory_level": str(cell["memory"]),
        # Retention state is what the memory factor is supposed to move. It is
        # recorded per cell so the summary can verify the two arms of a level
        # really shared a policy rather than merely being labelled alike.
        "memory_policy": system.get("memory_policy"),
        "latest_pin_enabled": system.get("latest_pin_enabled"),
        # Predictor support must be identical across the two MaxEnt cells for
        # the memory factor to be attributable; the summary asserts it.
        "semantic_fallback_enabled": fit_stats.get("semantic_fallback_enabled"),
        "latent_strategy_enabled": system.get("latent_strategy_enabled"),
        "active_variants": system.get("active_variants"),
        "pruned_variants": system.get("pruned_variants"),
        "mean_active_weight": system.get("mean_active_weight"),
        "predictor": system.get("predictor"),
        "model_family": fit_stats.get("model_family"),
        "stream_wall_s": float(stream.wall_s),
        "training_fit_wall_s": system.get("training_fit_wall_s"),
        "training_estimated_fit_flops": system.get(
            "training_estimated_fit_flops"
        ),
        "trend": _trend(stream),
        "detailed_metrics": dict(summary),
    }
    row.update({
        metric: overall.get(metric)
        for metric, _direction in MEMORY_PREDICTOR_ABLATION_METRICS
    })
    return row


def _route_job(job: Tuple[str, int, Any, Tuple[str, ...]]) -> List[Dict[str, Any]]:
    """Run one scenario's routing comparisons in an isolated process."""
    scenario, seed, config, baselines = job
    from .evaluation import (
        _apply_native_thread_limit,
        _native_thread_count,
        _pin_to_performance_cores,
        build_plan,
        run_stream,
    )
    from .memory import clear_caches

    _pin_to_performance_cores()
    _apply_native_thread_limit(_native_thread_count(config))
    strict_config = replace(config, allow_repeat_observation=False)
    local_config = replace(config, allow_repeat_observation=True)
    plan = build_plan(scenario, strict_config, seed)
    clear_caches()
    full = run_stream(
        "full",
        plan,
        strict_config,
        mode_schedule_policy="full_reference_strict_observation_route",
    )
    rows = [_teaching_row(full, REFERENCE_ROUTE)]
    schedule = tuple(str(row.get("mode")) for row in full.episode_rows)
    for baseline in baselines:
        if baseline == "full":
            continue
        clear_caches()
        shared = run_stream(
            baseline,
            plan,
            strict_config,
            execution_mode_schedule=schedule,
            mode_schedule_policy="matched_full_realized_execution_schedule",
        )
        rows.append(_teaching_row(shared, SHARED_ROUTE))
        clear_caches()
        local = run_stream(
            baseline,
            plan,
            local_config,
            mode_schedule_policy="baseline_local_reobservation_when_recipe_absent",
        )
        rows.append(_teaching_row(local, LOCAL_ROUTE))
    return rows


MEMORY_PREDICTOR_CONTRASTS: Tuple[Dict[str, str], ...] = (
    {"name": "memory_effect_within_maxent",
     "treatment": "full", "reference": "maxent_retain_all",
     "question": "does the memory policy help the proposed predictor"},
    {"name": "memory_effect_within_behavior_cloning",
     "treatment": "bc_adaptive", "reference": "bc",
     "question": "does the same memory policy help a linear cloner"},
    {"name": "predictor_effect_under_adaptive_memory",
     "treatment": "full", "reference": "bc_adaptive",
     "question": "predictor gap once both arms share Full's memory policy"},
    {"name": "predictor_effect_under_retain_all",
     "treatment": "maxent_retain_all", "reference": "bc",
     "question": "predictor gap when neither arm forgets"},
)


def _paired_cell_deltas(
    indexed: Mapping[Tuple[str, int, str], Mapping[str, Any]],
    scenario: str,
    treatment: str,
    reference: str,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Seed-matched treatment-minus-reference deltas for one scenario."""
    seeds = sorted({
        seed for row_scenario, seed, arm in indexed
        if row_scenario == scenario and arm == treatment
        and (scenario, seed, reference) in indexed
    })
    metric_deltas: Dict[str, Any] = {}
    raw: Dict[str, List[float]] = {}
    for metric, direction in MEMORY_PREDICTOR_ABLATION_METRICS:
        deltas: List[float] = []
        for seed in seeds:
            treatment_value = indexed[(scenario, seed, treatment)].get(metric)
            reference_value = indexed[(scenario, seed, reference)].get(metric)
            if not isinstance(treatment_value, (int, float)) or not (
                isinstance(reference_value, (int, float))
            ):
                continue
            delta = float(treatment_value) - float(reference_value)
            # Sign every contrast so positive favours the treatment, whichever
            # way the underlying metric reads.
            if direction == "lower_is_better":
                delta = -delta
            if math.isfinite(delta):
                deltas.append(delta)
        raw[metric] = deltas
        metric_deltas[metric] = {
            "direction": direction,
            "n_paired_seeds": len(deltas),
            "mean_treatment_advantage": _finite_mean(deltas),
        }
    return {"scenario": scenario, "n_paired_seeds": len(seeds),
            "metrics": metric_deltas}, raw


def summarize_memory_predictor_ablation(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Cell means, the four simple effects, and the memory x predictor term."""
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    indexed: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        scenario = str(row.get("scenario"))
        arm = str(row.get("arm", row.get("baseline")))
        grouped[(scenario, arm)].append(row)
        seed = row.get("seed")
        if isinstance(seed, int):
            indexed[(scenario, int(seed), arm)] = row

    cell_means = []
    for (scenario, arm), group in sorted(grouped.items()):
        first = group[0]
        cell_means.append({
            "scenario": scenario,
            "arm": arm,
            "baseline": first.get("baseline"),
            "label": first.get("label"),
            "predictor_level": first.get("predictor_level"),
            "memory_level": first.get("memory_level"),
            "n_seeds": len(group),
            "memory_policy": first.get("memory_policy"),
            "latest_pin_enabled": first.get("latest_pin_enabled"),
            "mean_active_variants": _finite_mean(
                row.get("active_variants") for row in group
            ),
            "mean_pruned_variants": _finite_mean(
                row.get("pruned_variants") for row in group
            ),
            **{
                f"mean_{metric}": _finite_mean(row.get(metric) for row in group)
                for metric, _direction in MEMORY_PREDICTOR_ABLATION_METRICS
            },
        })

    scenarios = sorted({scenario for scenario, _seed, _arm in indexed})
    simple_effects: List[Dict[str, Any]] = []
    raw_by_contrast: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for contrast in MEMORY_PREDICTOR_CONTRASTS:
        for scenario in scenarios:
            entry, raw = _paired_cell_deltas(
                indexed, scenario,
                str(contrast["treatment"]), str(contrast["reference"]),
            )
            simple_effects.append({**dict(contrast), **entry})
            raw_by_contrast[(str(contrast["name"]), scenario)] = raw

    # The interaction is the question the arm was added to answer: if the
    # memory policy is worth as much to a linear cloner as to MaxEnt, the gain
    # is the memory, not the model family.
    interactions = []
    for scenario in scenarios:
        maxent = raw_by_contrast.get(
            ("memory_effect_within_maxent", scenario), {},
        )
        cloner = raw_by_contrast.get(
            ("memory_effect_within_behavior_cloning", scenario), {},
        )
        metrics: Dict[str, Any] = {}
        for metric, direction in MEMORY_PREDICTOR_ABLATION_METRICS:
            left = maxent.get(metric) or []
            right = cloner.get(metric) or []
            paired = min(len(left), len(right))
            metrics[metric] = {
                "direction": direction,
                "n_paired_seeds": paired,
                "mean_memory_effect_within_maxent": _finite_mean(left),
                "mean_memory_effect_within_behavior_cloning": _finite_mean(
                    right,
                ),
                "mean_interaction": _finite_mean(
                    a - b for a, b in zip(left[:paired], right[:paired])
                ),
            }
        interactions.append({
            "scenario": scenario,
            "definition": (
                "memory effect within MaxEnt minus memory effect within "
                "behaviour cloning; near zero means the retention policy pays "
                "off independently of the model family"
            ),
            "metrics": metrics,
        })

    # A mislabelled cell would silently invalidate every contrast above.
    policy_check = {
        level: sorted({
            (str(row.get("memory_policy")), bool(row.get("latest_pin_enabled")))
            for row in rows if str(row.get("memory_level")) == level
        })
        for level in MEMORY_LEVELS
    }
    # Both MaxEnt cells must hold the two predictor-support components at
    # Full's level; if they do not, the memory factor is confounded again and
    # the interaction term below is not interpretable.
    support_check = {
        arm: sorted({
            (bool(row.get("semantic_fallback_enabled")), bool(row.get("latent_strategy_enabled")))
            for row in rows if str(row.get("arm", row.get("baseline"))) == arm
        })
        for arm in ("full", "maxent_retain_all")
    }
    maxent_support_matched = (
        all(len(value) == 1 for value in support_check.values())
        and len({value[0] for value in support_check.values() if value}) == 1
    )
    return {
        "mean_by_scenario_cell": cell_means,
        "paired_seed_simple_effects": simple_effects,
        "memory_by_predictor_interaction": interactions,
        "realized_memory_policy_by_level": {
            level: [list(item) for item in value]
            for level, value in policy_check.items()
        },
        "memory_levels_are_internally_consistent": all(
            len(value) == 1 for value in policy_check.values()
        ),
        "realized_predictor_support_by_maxent_arm": {
            arm: [list(item) for item in value]
            for arm, value in support_check.items()
        },
        "maxent_cells_share_predictor_support": bool(maxent_support_matched),
    }


def run_memory_predictor_ablation(
    evaluation_config: Optional[Any] = None,
    cells: Sequence[Mapping[str, str]] = MEMORY_PREDICTOR_ABLATION_CELLS,
    arms_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the paired 2x2 memory-policy by predictor longitudinal ablation."""
    from .evaluation import EvalSettings

    selected = tuple(dict(cell) for cell in cells)
    arms = tuple(str(cell["arm"]) for cell in selected)
    # Two cells may share a baseline class and differ only in their settings
    # overrides (the MaxEnt row does exactly that), so the arm name -- not the
    # baseline -- is the cell identity.
    if len(set(arms)) != len(arms):
        raise ValueError("memory ablation cells must name distinct arms")
    if arms[0] != "full":
        raise ValueError(
            "memory ablation requires the 'full' arm first; it sets the shared route"
        )
    baselines = tuple(dict.fromkeys(str(cell["baseline"]) for cell in selected))
    observed = {
        (str(cell["predictor"]), str(cell["memory"])) for cell in selected
    }
    expected = {
        (predictor, memory)
        for predictor in PREDICTOR_LEVELS for memory in MEMORY_LEVELS
    }
    if observed != expected:
        raise ValueError(
            "memory ablation requires every predictor x memory cell exactly "
            f"once; missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )

    config = evaluation_config or EvalSettings(
        seeds=(1337,),
        baselines=baselines,
        include_oracle=False,
        experiment="memory_predictor_ablation",
    )
    config = replace(
        config,
        baselines=baselines,
        include_oracle=False,
        shared_routing=True,
        observe_missing_recipes=False,
        allow_repeat_observation=False,
        audit_period=0,
        sensitivity=False,
        experiment="memory_predictor_ablation",
    )

    rows = run_arm_major_suite(
        "memory_predictor_ablation", selected, config, arms_dir=arms_dir,
    )
    return {
        "definition": (
            "Paired 2x2 longitudinal ablation crossing the predictor (MaxEnt "
            "against a linear behaviour cloner) with the retention policy "
            "(Full's adaptive decay and latest pin against retaining every "
            "variant at unit weight). All four cells follow Full's realized "
            "interaction schedule on a shared plan."
        ),
        "design_type": "paired_memory_policy_by_predictor_factorial_ablation",
        "rationale": (
            "The publication 'bc' arm moves the predictor and the retention "
            "policy at the same time, so its margin over Full cannot be "
            "attributed to either. Holding the cloner fixed and giving it "
            "Full's memory policy isolates the retention contribution, and "
            "the interaction term reports whether that contribution depends "
            "on the model family. The MaxEnt retain-all cell is Full with its "
            "retention policy switched off rather than the 'no_decay' "
            "baseline, which would also have removed the semantic fallback "
            "and the latent residual and so reintroduced the confound this "
            "design exists to remove."
        ),
        "primary_contrast": "memory_effect_within_behavior_cloning",
        "predictor_levels": list(PREDICTOR_LEVELS),
        "memory_levels": list(MEMORY_LEVELS),
        "cells": [dict(cell) for cell in selected],
        "rows": rows,
        "summary": summarize_memory_predictor_ablation(rows),
    }


def _regime_top_1(summary: Mapping[str, Any], regime: str) -> Optional[float]:
    """Top-1 within one decision regime, or None when the regime is empty."""
    regimes = ((summary.get("diagnostics") or {}).get("decision_regimes") or {})
    row = (regimes.get("by_regime") or {}).get(regime) or {}
    return _as_optional_float(row.get("top_1"))


def _regime_share(summary: Mapping[str, Any], regime: str) -> Optional[float]:
    regimes = ((summary.get("diagnostics") or {}).get("decision_regimes") or {})
    row = (regimes.get("by_regime") or {}).get(regime) or {}
    return _as_optional_float(row.get("share"))


def _transfer_cell_top_1(summary: Mapping[str, Any], cell: str) -> Optional[float]:
    """Teacher-forced top-1 in one transfer cell of the assist stream."""
    cells = ((summary.get("assist") or {}).get("by_transfer_cell") or {})
    row = cells.get(cell) or {}
    if not _as_optional_float(row.get("n_teacher_forced_predictions")):
        return None
    return _as_optional_float(row.get("teacher_forced_top_1"))


def _coexistence_profile(
    summary: Mapping[str, Any],
) -> Tuple[Dict[str, Optional[float]], Optional[float]]:
    """Accuracy against the number of preferences active for one recipe.

    The degradation term is the single-preference cell minus the most crowded
    populated cell. It is the quantity a state-indexed predictor should lose
    on: coexisting variants of one recipe share states and separate only on
    what has already happened this episode.
    """
    counts = ((summary.get("assist") or {}).get("by_active_preference_count") or {})
    profile: Dict[str, Optional[float]] = {}
    populated: List[Tuple[int, float]] = []
    for key, row in counts.items():
        value = _as_optional_float((row or {}).get("teacher_forced_top_1"))
        profile[str(key)] = value
        try:
            index = int(str(key))
        except (TypeError, ValueError):
            continue
        if value is not None:
            populated.append((index, float(value)))
    if len(populated) < 2:
        return profile, None
    populated.sort()
    return profile, float(populated[0][1] - populated[-1][1])


def _representation_row(
    stream: Any,
    arm: RepresentationAblationArm,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Flatten one representation arm and the endpoints it is judged on."""
    assist = summary.get("assist") or {}
    overall = assist.get("overall") or {}
    system = summary.get("system") or {}
    fit_stats = system.get("fit_stats") or {}
    coexistence, degradation = _coexistence_profile(summary)
    row: Dict[str, Any] = {
        "scenario": str(stream.scenario),
        "seed": int(stream.seed),
        "arm": arm.name,
        "label": arm.label,
        "baseline": arm.baseline,
        "family": arm.family,
        "representation": arm.representation,
        "within_episode_history": arm.within_episode_history,
        "model_overrides": arm.model_overrides(),
        # Retention state is held fixed by design, so it is recorded to prove
        # the arms really shared a memory policy rather than being labelled
        # alike -- the same check the 2x2 memory suite makes.
        "memory_policy": system.get("memory_policy"),
        "latest_pin_enabled": system.get("latest_pin_enabled"),
        "active_variants": system.get("active_variants"),
        "pruned_variants": system.get("pruned_variants"),
        "predictor": system.get("predictor"),
        "model_family": fit_stats.get("model_family"),
        "irl_features": system.get("irl_features"),
        "bc_history": arm.model_overrides().get("bc_history"),
        "bc_prefix_length_feature": arm.model_overrides().get(
            "bc_prefix_length_feature"
        ),
        "parameter_count": fit_stats.get("parameter_count"),
        "history_feature_dim": fit_stats.get("history_feature_dim"),
        "stream_wall_s": float(stream.wall_s),
        "training_fit_wall_s": system.get("training_fit_wall_s"),
        "single_option_lookup_share": _regime_share(
            summary, "single_option_lookup"
        ),
        "branching_share": _regime_share(summary, "branching"),
        "unseen_state_fallback_share": _regime_share(
            summary, "unseen_state_fallback"
        ),
        "coexistence_profile": coexistence,
        "trend": _trend(stream),
        "detailed_metrics": dict(summary),
    }
    row.update({
        metric: overall.get(metric)
        for metric, _direction in REPRESENTATION_ABLATION_METRICS
        if metric in overall
    })
    row.update({
        "single_option_lookup_top_1": _regime_top_1(
            summary, "single_option_lookup"
        ),
        "branching_top_1": _regime_top_1(summary, "branching"),
        "unseen_state_fallback_top_1": _regime_top_1(
            summary, "unseen_state_fallback"
        ),
        "direct_retrieval_top_1": _transfer_cell_top_1(
            summary, "direct_retrieval"
        ),
        "new_preference_top_1": _transfer_cell_top_1(
            summary, "seen_recipe_new_preference"
        ),
        "preference_seen_elsewhere_top_1": _transfer_cell_top_1(
            summary, "seen_recipe_preference_seen_elsewhere"
        ),
        "coexistence_degradation": degradation,
    })
    return row


def summarize_representation_ablation(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-arm means and paired within-family contrasts.

    Contrasts carry the exact paired randomization p-value the evaluation
    module already uses for holdout inference, so a component claim from this
    suite is tested the same way as a scenario claim.
    """
    from .evaluation import _sign_flip_test

    by_scenario_arm: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    by_key: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        arm = str(row["arm"])
        by_scenario_arm[(scenario, arm)].append(row)
        by_key[(scenario, int(row["seed"]), arm)] = row

    means: List[Dict[str, Any]] = []
    for (scenario, arm), group in sorted(by_scenario_arm.items()):
        entry: Dict[str, Any] = {
            "scenario": scenario,
            "arm": arm,
            "label": str(group[0]["label"]),
            "family": str(group[0]["family"]),
            "representation": str(group[0]["representation"]),
            "within_episode_history": str(group[0]["within_episode_history"]),
            "memory_policy": group[0].get("memory_policy"),
            "latest_pin_enabled": group[0].get("latest_pin_enabled"),
            "n_seeds": len(group),
        }
        for metric, _direction in REPRESENTATION_ABLATION_METRICS:
            entry[f"mean_{metric}"] = _finite_mean(
                row.get(metric) for row in group
            )
        for metric in (
            "active_variants", "pruned_variants", "parameter_count",
            "history_feature_dim", "training_fit_wall_s",
            "single_option_lookup_share", "branching_share",
            "unseen_state_fallback_share",
        ):
            entry[f"mean_{metric}"] = _finite_mean(
                row.get(metric) for row in group
            )
        means.append(entry)

    scenarios = sorted({str(row["scenario"]) for row in rows})
    contrasts: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for spec in REPRESENTATION_ABLATION_CONTRASTS:
            reference = str(spec["reference"])
            treatment = str(spec["treatment"])
            seeds = sorted({
                int(row["seed"]) for row in rows
                if str(row["scenario"]) == scenario
            })
            metrics: Dict[str, Any] = {}
            for metric, direction in REPRESENTATION_ABLATION_METRICS:
                deltas: List[float] = []
                for seed in seeds:
                    treated = by_key.get((scenario, seed, treatment))
                    control = by_key.get((scenario, seed, reference))
                    if treated is None or control is None:
                        continue
                    high = _as_optional_float(treated.get(metric))
                    low = _as_optional_float(control.get(metric))
                    if high is None or low is None:
                        continue
                    advantage = high - low
                    deltas.append(
                        advantage if direction == "higher_is_better"
                        else -advantage
                    )
                if not deltas:
                    continue
                one_sided, two_sided = _sign_flip_test(deltas)
                metrics[metric] = {
                    "direction": direction,
                    "mean_treatment_advantage": _finite_mean(deltas),
                    "n_paired_seeds": len(deltas),
                    "seeds_favouring_treatment": sum(
                        1 for value in deltas if value > 0.0
                    ),
                    "sign_flip_p_one_sided": one_sided,
                    "sign_flip_p_two_sided": two_sided,
                }
            contrasts.append({
                "scenario": scenario,
                "name": str(spec["name"]),
                "reference": reference,
                "treatment": treatment,
                "interpretation": str(spec["interpretation"]),
                "metrics": metrics,
            })

    held_fixed = sorted({
        (str(row.get("memory_policy")), bool(row.get("latest_pin_enabled")))
        for row in rows
    })
    return {
        "mean_by_scenario_arm": means,
        "paired_within_family_contrasts": contrasts,
        "memory_policy_levels_observed": [
            {"memory_policy": policy, "latest_pin_enabled": pinned}
            for policy, pinned in held_fixed
        ],
        "memory_policy_held_fixed": len(held_fixed) == 1,
    }


def run_representation_ablation(
    evaluation_config: Optional[Any] = None,
    arms: Sequence[RepresentationAblationArm] = REPRESENTATION_ABLATION_ARMS,
    arms_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the paired predictor-representation longitudinal ablation."""
    from .evaluation import EvalSettings

    selected = tuple(arms)
    names = tuple(arm.name for arm in selected)
    if len(set(names)) != len(names):
        raise ValueError("representation ablation arms must be uniquely named")
    families = {arm.family for arm in selected}
    unknown = families - set(REPRESENTATION_FAMILIES)
    if unknown:
        raise ValueError(f"unknown representation families: {sorted(unknown)}")
    for family in REPRESENTATION_FAMILIES:
        if sum(1 for arm in selected if arm.family == family) < 2:
            raise ValueError(
                f"family {family!r} needs at least two arms for a "
                "within-family contrast"
            )
    from .baselines import BASELINE_AGENTS

    baselines = {arm.baseline for arm in selected}
    unsupported = baselines - {"full", *BASELINE_AGENTS}
    if unsupported:
        raise ValueError(f"unknown ablation baselines: {sorted(unsupported)}")

    config = evaluation_config or EvalSettings(
        seeds=(1337,),
        baselines=("full",),
        include_oracle=False,
        experiment="representation_ablation",
    )
    config = replace(
        config,
        include_oracle=False,
        shared_routing=True,
        observe_missing_recipes=False,
        allow_repeat_observation=False,
        audit_period=0,
        sensitivity=False,
        experiment="representation_ablation",
    )

    rows = run_arm_major_suite(
        "representation_ablation", selected, config, arms_dir=arms_dir,
    )
    return {
        "definition": (
            "Paired longitudinal ablation of what the predictor is allowed to "
            "see, with Full's adaptive-decay and latest-pin memory policy held "
            "fixed in every arm. The MaxEnt arms vary the reward "
            "representation; the cloner arms vary the within-episode action "
            "history, down to a state-only condition that sees exactly what "
            "the MaxEnt head sees."
        ),
        "design_type": "paired_predictor_representation_component_ablation",
        "design": representation_ablation_design(),
        "rationale": (
            "Preferences in this domain are goal-preserving reorderings, so "
            "coexisting variants of one recipe traverse the same environment "
            "states and separate only on what has already happened in the "
            "episode. A state-indexed policy cannot represent that "
            "distinction and a history-conditioned one can, which predicts "
            "the cloner's advantage on unseen states, on branching decisions "
            "and under coexisting preferences. Removing the cloner's history "
            "while holding storage and memory policy fixed tests that "
            "explanation directly, and sweeping the MaxEnt reward "
            "representation bounds the competing explanation."
        ),
        "primary_contrast": "cloner_history_contribution",
        "primary_endpoint": "unseen_state_fallback_top_1",
        "rows": rows,
        "summary": summarize_representation_ablation(rows),
    }


def summarize_routing_ablation(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate condition means and local-minus-shared matched-seed deltas."""
    metrics = (
        "n_observation_episodes",
        "n_extra_observation_episodes",
        "n_explicit_demonstration_actions",
        "n_extra_demonstration_actions",
        "n_corrective_teaching_actions",
        "n_total_explicit_teaching_actions",
        "fixed_checkpoint_top_1",
        "fixed_checkpoint_top_k",
        "fixed_checkpoint_normalized_human_action_load",
        "fixed_checkpoint_mean_corrections_per_task",
        "fixed_checkpoint_corrections_per_recipe_step",
        "assist_live_top_1_selection_affected",
        "assist_teacher_forced_top_1",
        "assist_normalized_human_action_load",
        "assist_mean_corrections_per_task",
        "assist_corrections_per_recipe_step",
    )
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    indexed: Dict[Tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        scenario = str(row.get("scenario"))
        baseline = str(row.get("baseline"))
        condition = str(row.get("routing_condition"))
        grouped[(scenario, baseline, condition)].append(row)
        if isinstance(row.get("seed"), int):
            indexed[(scenario, int(row["seed"]), baseline, condition)] = row

    paired = []
    scenario_baselines = sorted({
        (scenario, baseline)
        for scenario, _seed, baseline, condition in indexed
        if condition in {SHARED_ROUTE, LOCAL_ROUTE}
    })
    for scenario, baseline in scenario_baselines:
        seeds = sorted({
            seed
            for row_scenario, seed, row_baseline, condition in indexed
            if row_scenario == scenario
            and row_baseline == baseline
            and condition == LOCAL_ROUTE
            and (
                scenario, seed, baseline, SHARED_ROUTE
            ) in indexed
        })
        metric_deltas: Dict[str, Any] = {}
        teaching_efficiency = []
        for metric in metrics:
            deltas = []
            for seed in seeds:
                local = indexed[(
                    scenario, seed, baseline, LOCAL_ROUTE,
                )].get(metric)
                shared = indexed[(
                    scenario, seed, baseline, SHARED_ROUTE,
                )].get(metric)
                if not isinstance(local, (int, float)) or not isinstance(
                    shared, (int, float)
                ):
                    continue
                delta = float(local) - float(shared)
                if math.isfinite(delta):
                    deltas.append(delta)
            metric_deltas[metric] = {
                "n_paired_seeds": len(deltas),
                "mean_local_minus_shared": _finite_mean(deltas),
            }
        for seed in seeds:
            local = indexed[(
                scenario, seed, baseline, LOCAL_ROUTE,
            )]
            shared = indexed[(
                scenario, seed, baseline, SHARED_ROUTE,
            )]
            extra_demonstration_actions = (
                float(local.get("n_extra_demonstration_actions", 0.0))
                - float(shared.get("n_extra_demonstration_actions", 0.0))
            )
            frozen_gain = (
                float(local.get("fixed_checkpoint_top_1", 0.0) or 0.0)
                - float(shared.get("fixed_checkpoint_top_1", 0.0) or 0.0)
            )
            if extra_demonstration_actions > 0.0:
                teaching_efficiency.append(
                    frozen_gain / extra_demonstration_actions
                )
        paired.append({
            "scenario": scenario,
            "baseline": baseline,
            "contrast": "baseline_local_minus_shared_full_route",
            "n_paired_seeds": len(seeds),
            "metrics": metric_deltas,
            "mean_fixed_checkpoint_top_1_gain_per_additional_demonstration_action": (
                _finite_mean(teaching_efficiency)
            ),
        })

    return {
        "by_scenario_baseline_condition": [
            {
                "scenario": scenario,
                "baseline": baseline,
                "routing_condition": condition,
                "n_seeds": len(group),
                **{
                    f"mean_{metric}": _finite_mean(
                        row.get(metric) for row in group
                    )
                    for metric in metrics
                },
            }
            for (scenario, baseline, condition), group in sorted(grouped.items())
        ],
        "paired_seed_local_minus_shared": paired,
    }


def run_routing_ablation(
    evaluation_config: Optional[Any] = None,
    baselines: Sequence[str] = ROUTING_BASELINES,
) -> Dict[str, Any]:
    """Compare matched full routing with unrestricted baseline-local teaching."""
    from .evaluation import EvalSettings

    baseline_names = tuple(dict.fromkeys(str(name) for name in baselines))
    if "full" not in baseline_names:
        raise ValueError("routing ablation requires 'full' as the reference")
    frozen_references = {
        "frozen", "offline_default", "offline_all",
    }
    invalid = sorted(set(baseline_names) & frozen_references)
    if invalid:
        raise ValueError(
            "unrestricted observation is not a meaningful treatment for frozen "
            f"deployment references: {invalid}"
        )

    config = evaluation_config or EvalSettings(
        baselines=baseline_names,
        include_oracle=False,
        shared_routing=False,
        observe_missing_recipes=True,
        allow_repeat_observation=False,
        experiment="teaching_burden_routing_ablation",
    )
    config = replace(
        config,
        baselines=baseline_names,
        include_oracle=False,
        shared_routing=False,
        observe_missing_recipes=True,
        allow_repeat_observation=False,
        audit_period=0,
        sensitivity=False,
        experiment="teaching_burden_routing_ablation",
    )

    jobs = [
        (str(scenario), int(seed), config, baseline_names)
        for scenario in config.scenarios for seed in config.seeds
    ]
    workers = max(1, min(int(config.workers or 1), len(jobs)))
    if workers == 1:
        groups = [_route_job(job) for job in jobs]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            groups = [future.result() for future in as_completed(
                pool.submit(_route_job, job) for job in jobs
            )]
    rows = [row for group in groups for row in group]
    rows.sort(key=lambda row: (
        str(row["scenario"]), int(row["seed"]), str(row["baseline"]),
        str(row["routing_condition"]),
    ))

    return {
        "definition": (
            "One full-system route is the shared interaction reference. Each "
            "online baseline is compared under that route and under unrestricted "
            "baseline-local recovery, where an assist becomes a complete "
            "observation demonstration only after that baseline has forgotten "
            "every active variant of the recipe."
        ),
        "design_type": "paired_routing_policy_ablation",
        "primary_estimand": (
            "additional explicit teaching burden and fixed-checkpoint performance "
            "from allowing each baseline to request its own observations"
        ),
        "primary_burden_unit": (
            "total explicit teaching time; demonstration and correction action "
            "counts are reported separately because their costs differ"
        ),
        "performance_comparison": (
            "frozen evaluations at common scheduled checkpoints; live assist "
            "accuracy is selection-affected under local routing"
        ),
        "normal_evaluation_observation_contract": (
            "only the first stream exposure of each recipe is observed; all "
            "baselines follow Full's realized interaction schedule"
        ),
        "local_ablation_observation_trigger": (
            "requested assist and no active variant of that recipe in the "
            "evaluated baseline"
        ),
        "excluded_diagnostics": {
            "active_only_pruned_influence_audit": (
                "not used by this ablation's burden or performance estimands"
            ),
        },
        "excluded_references": sorted(frozen_references),
        "baselines": list(baseline_names),
        "n_unique_executions_per_seed_scenario": 1 + 2 * (
            len(baseline_names) - 1
        ),
        "rows": rows,
        "summary": summarize_routing_ablation(rows),
    }


def _parse_args() -> argparse.Namespace:
    from .evaluation import PAPER_SEEDS

    paired_seed_csv = ",".join(str(seed) for seed in PAPER_SEEDS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "matcher", "routing", "latent", "memory", "representation",
                 "components", "retention"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=MatcherSettings.seed)
    parser.add_argument("--recipe-count", type=int, default=MatcherSettings.recipe_count)
    parser.add_argument("--case-limit", type=int, default=MatcherSettings.case_limit)
    parser.add_argument("--no-prefixes", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--routing-seeds", default=paired_seed_csv)
    parser.add_argument(
        "--routing-scenarios",
        default="homogeneous,heterogeneous,holdout",
    )
    parser.add_argument(
        "--routing-baselines", default=",".join(ROUTING_BASELINES),
    )
    parser.add_argument("--routing-recipes", type=int, default=20)
    parser.add_argument("--components-seeds", default=paired_seed_csv)
    parser.add_argument(
        "--components-scenarios",
        default="homogeneous,heterogeneous,holdout",
    )
    parser.add_argument("--components-recipes", type=int, default=20)
    parser.add_argument("--retention-seeds", default=paired_seed_csv)
    parser.add_argument(
        "--retention-scenarios",
        default="homogeneous,heterogeneous,holdout",
    )
    parser.add_argument("--retention-recipes", type=int, default=20)
    parser.add_argument("--memory-seeds", default=paired_seed_csv)
    parser.add_argument(
        "--memory-scenarios",
        default="homogeneous,heterogeneous,holdout",
    )
    parser.add_argument("--memory-recipes", type=int, default=20)
    parser.add_argument("--latent-seeds", default=paired_seed_csv)
    parser.add_argument(
        "--latent-scenarios",
        default="homogeneous,heterogeneous,holdout",
    )
    parser.add_argument("--latent-recipes", type=int, default=20)
    parser.add_argument("--representation-seeds", default=paired_seed_csv)
    parser.add_argument(
        "--representation-scenarios",
        default="homogeneous,heterogeneous,holdout",
    )
    parser.add_argument("--representation-recipes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=DEFAULT_ABLATION_WORKERS)
    parser.add_argument(
        "--output",
        help=(
            "JSON file for one suite, or the immutable run-directory root "
            "for --suite all"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def _emit(result: Mapping[str, Any], args: argparse.Namespace) -> None:
    encoded = json.dumps(result, indent=args.indent, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    if not args.quiet:
        print(encoded, end="")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_ablation_process(
    name: str,
    command: Sequence[str],
    run_dir: Path,
    environment: Mapping[str, str],
) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    stdout_path = run_dir / f"{name}.stdout.log"
    stderr_path = run_dir / f"{name}.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_stream,
        stderr_path.open("w", encoding="utf-8") as stderr_stream,
    ):
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=dict(environment),
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
            text=True,
        )
    return {
        "name": name,
        "command": list(command),
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "return_code": int(process.returncode),
        "wall_s": float(time.perf_counter() - started),
        "output": str(run_dir / f"{name}.json"),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _validate_all_ablation_results(
    results: Mapping[str, Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
) -> None:
    expected_seeds = {int(value) for value in seeds}
    matcher = results.get("matcher", {})
    if not isinstance(matcher.get("summary_by_matcher"), dict):
        raise RuntimeError("matcher ablation output is incomplete")

    routing_rows = results.get("routing", {}).get("rows")
    if not isinstance(routing_rows, list) or not routing_rows:
        raise RuntimeError("routing ablation has no result rows")
    if {
        row.get("scenario") for row in routing_rows
    } != set(scenarios) or {row.get("seed") for row in routing_rows} != expected_seeds:
        raise RuntimeError("routing ablation did not cover its scenario-seed grid")
    if any(
        not isinstance(row.get("trend"), list) or not row["trend"]
        for row in routing_rows
    ):
        raise RuntimeError("routing ablation contains a row without phase trends")

    latent_rows = results.get("latent", {}).get("rows")
    if not isinstance(latent_rows, list) or not latent_rows:
        raise RuntimeError("latent-strategy ablation has no result rows")
    expected_arms = {arm.name for arm in LATENT_STRATEGY_ABLATION_ARMS}
    if {row.get("arm") for row in latent_rows} != expected_arms:
        raise RuntimeError("latent-strategy ablation did not execute all arms")
    if {
        row.get("scenario") for row in latent_rows
    } != set(scenarios) or {row.get("seed") for row in latent_rows} != expected_seeds:
        raise RuntimeError(
            "latent-strategy ablation did not cover its scenario-seed grid"
        )

    memory = results.get("memory", {})
    memory_rows = memory.get("rows")
    if not isinstance(memory_rows, list) or not memory_rows:
        raise RuntimeError("memory-policy ablation has no result rows")
    expected_cells = {
        str(cell["arm"]) for cell in MEMORY_PREDICTOR_ABLATION_CELLS
    }
    if {row.get("arm") for row in memory_rows} != expected_cells:
        raise RuntimeError(
            "memory-policy ablation did not execute every factorial cell"
        )
    if {
        row.get("scenario") for row in memory_rows
    } != set(scenarios) or {row.get("seed") for row in memory_rows} != expected_seeds:
        raise RuntimeError(
            "memory-policy ablation did not cover its scenario-seed grid"
        )
    # Every contrast in the summary assumes the two arms of a memory level
    # really ran the same retention policy; a mislabelled cell would leave the
    # deltas looking valid while measuring nothing.
    if not (memory.get("summary") or {}).get(
        "memory_levels_are_internally_consistent"
    ):
        raise RuntimeError(
            "memory-policy ablation cells disagree with their declared "
            "retention level"
        )
    # The repaired factorial only means anything if both MaxEnt cells kept
    # Full's predictor support; if they did not, the memory factor is
    # confounded again and the interaction term is not interpretable.
    if not (memory.get("summary") or {}).get(
        "maxent_cells_share_predictor_support"
    ):
        raise RuntimeError(
            "memory-policy ablation MaxEnt cells do not share Full's "
            "predictor support"
        )

    components = results.get("components", {})
    component_rows = components.get("rows")
    if not isinstance(component_rows, list) or not component_rows:
        raise RuntimeError("component ablation has no result rows")
    expected_component_arms = {arm.name for arm in COMPONENT_ABLATION_ARMS}
    if {row.get("arm") for row in component_rows} != expected_component_arms:
        raise RuntimeError("component ablation did not execute all arms")
    if {
        row.get("scenario") for row in component_rows
    } != set(scenarios) or {
        row.get("seed") for row in component_rows
    } != expected_seeds:
        raise RuntimeError(
            "component ablation did not cover its scenario-seed grid"
        )
    component_summary = components.get("summary") or {}
    # Each arm must have dropped exactly the component it names, and none of
    # them may have moved retention; otherwise the per-component costs are not
    # attributable to the component.
    if not component_summary.get("declared_factors_were_applied"):
        raise RuntimeError(
            "component ablation arms did not realize their declared component "
            "configuration"
        )
    if not component_summary.get("retention_held_constant"):
        raise RuntimeError(
            "component ablation moved the retention policy; the per-component "
            "costs would not be attributable"
        )

    retention = results.get("retention", {})
    retention_rows = retention.get("rows")
    if not isinstance(retention_rows, list) or not retention_rows:
        raise RuntimeError("retention ablation has no result rows")
    expected_retention_arms = {arm.name for arm in RETENTION_ABLATION_ARMS}
    if {row.get("arm") for row in retention_rows} != expected_retention_arms:
        raise RuntimeError("retention ablation did not execute all arms")
    if {
        row.get("scenario") for row in retention_rows
    } != set(scenarios) or {
        row.get("seed") for row in retention_rows
    } != expected_seeds:
        raise RuntimeError(
            "retention ablation did not cover its scenario-seed grid"
        )
    # The whole point of this suite is that predictor support does not move,
    # which is exactly what the deployed retention baselines get wrong.
    if not (retention.get("summary") or {}).get(
        "predictor_support_held_constant"
    ):
        raise RuntimeError(
            "retention ablation arms disagree on predictor support; the "
            "retention factor would be confounded"
        )

    representation = results.get("representation", {})
    representation_rows = representation.get("rows")
    if not isinstance(representation_rows, list) or not representation_rows:
        raise RuntimeError("representation ablation has no result rows")
    expected_representation_arms = {
        arm.name for arm in REPRESENTATION_ABLATION_ARMS
    }
    if {
        row.get("arm") for row in representation_rows
    } != expected_representation_arms:
        raise RuntimeError("representation ablation did not execute all arms")
    if {
        row.get("scenario") for row in representation_rows
    } != set(scenarios) or {
        row.get("seed") for row in representation_rows
    } != expected_seeds:
        raise RuntimeError(
            "representation ablation did not cover its scenario-seed grid"
        )
    # The suite only isolates the representation if every arm really shared
    # Full's retention policy. A drifting memory level would leave the
    # contrasts looking valid while measuring two factors at once.
    if not (representation.get("summary") or {}).get(
        "memory_policy_held_fixed"
    ):
        raise RuntimeError(
            "representation ablation arms did not share one retention policy"
        )


def run_all_ablations(
    output_root: str | Path = DEFAULT_ABLATION_RESULTS_ROOT,
    *,
    workers: int = DEFAULT_ABLATION_WORKERS,
) -> Dict[str, Any]:
    """Run and validate the fixed reviewer-facing ablation collection.

    Suites run one at a time, and each suite spreads its own scenario-seed
    grid across ``workers`` processes. The alternative -- all four suites at
    once, each internally near-sequential -- puts unrelated suites in
    contention for the same performance cores, which is exactly what the
    per-arm wall-clock metrics are supposed to be able to compare.
    """
    from .evaluation import NATIVE_THREAD_ENV_VARS, PAPER_SEEDS, SCENARIOS

    seeds = tuple(int(seed) for seed in PAPER_SEEDS)
    longitudinal_workers = int(workers)
    maximum_workers = len(SCENARIOS) * len(seeds)
    if not 1 <= longitudinal_workers <= maximum_workers:
        raise ValueError(
            f"workers must be between 1 and {maximum_workers}"
        )
    seed = int(MatcherSettings.seed)
    seed_csv = ",".join(str(value) for value in seeds)
    scenario_csv = ",".join(SCENARIOS)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "ablations__%Y%m%dT%H%M%S%fZ"
    )
    run_dir = root / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    commands = {
        "matcher": [
            sys.executable, "-X", "faulthandler", "-m", "src.ablations",
            "--suite", "matcher", "--seed", str(seed),
            "--output", str(run_dir / "matcher.json"), "--quiet",
        ],
        "routing": [
            sys.executable, "-X", "faulthandler", "-m", "src.ablations",
            "--suite", "routing", "--routing-seeds", seed_csv,
            "--routing-scenarios", scenario_csv,
            "--workers", str(longitudinal_workers),
            "--output", str(run_dir / "routing.json"), "--quiet",
        ],
        "latent": [
            sys.executable, "-X", "faulthandler", "-m", "src.ablations",
            "--suite", "latent", "--latent-seeds", seed_csv,
            "--latent-scenarios", scenario_csv,
            "--workers", str(longitudinal_workers),
            "--output", str(run_dir / "latent.json"), "--quiet",
        ],
        "memory": [
            sys.executable, "-X", "faulthandler", "-m", "src.ablations",
            "--suite", "memory", "--memory-seeds", seed_csv,
            "--memory-scenarios", scenario_csv,
            "--workers", str(longitudinal_workers),
            "--output", str(run_dir / "memory.json"), "--quiet",
        ],
        "representation": [
            sys.executable, "-X", "faulthandler", "-m", "src.ablations",
            "--suite", "representation",
            "--representation-seeds", seed_csv,
            "--representation-scenarios", scenario_csv,
            "--workers", str(longitudinal_workers),
            "--output", str(run_dir / "representation.json"), "--quiet",
        ],
        "components": [
            sys.executable, "-X", "faulthandler", "-m", "src.ablations",
            "--suite", "components", "--components-seeds", seed_csv,
            "--components-scenarios", scenario_csv,
            "--workers", str(longitudinal_workers),
            "--output", str(run_dir / "components.json"), "--quiet",
        ],
        "retention": [
            sys.executable, "-X", "faulthandler", "-m", "src.ablations",
            "--suite", "retention", "--retention-seeds", seed_csv,
            "--retention-scenarios", scenario_csv,
            "--workers", str(longitudinal_workers),
            "--output", str(run_dir / "retention.json"), "--quiet",
        ],
    }
    environment = os.environ.copy()
    environment.update({name: "1" for name in NATIVE_THREAD_ENV_VARS})
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    manifest_path = run_dir / "manifest.json"
    manifest: Dict[str, Any] = {
        "run": run_name,
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "matcher_seed": seed,
        "longitudinal_seeds": list(seeds),
        "longitudinal_scenarios": list(SCENARIOS),
        "suite_execution": "sequential",
        "suite_parallelism": 1,
        "longitudinal_workers_per_suite": longitudinal_workers,
        "commands": commands,
        "jobs": {},
    }
    _atomic_json(manifest_path, manifest)

    try:
        for name, command in commands.items():
            record = _run_ablation_process(
                name, command, run_dir, environment,
            )
            manifest["jobs"][record["name"]] = record
            _atomic_json(manifest_path, manifest)
            print(
                f"[ablation] {record['name']} "
                f"return_code={record['return_code']} "
                f"wall_s={record['wall_s']:.1f}",
                flush=True,
            )

        failed = sorted(
            name for name, record in manifest["jobs"].items()
            if record["return_code"] != 0
        )
        if failed:
            raise RuntimeError(
                "ablation suites failed; inspect stderr logs for "
                + ", ".join(failed)
            )
        results = {
            name: json.loads(Path(record["output"]).read_text(encoding="utf-8"))
            for name, record in manifest["jobs"].items()
        }
        _validate_all_ablation_results(
            results, seeds=seeds, scenarios=SCENARIOS,
        )
    except BaseException as error:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        _atomic_json(manifest_path, manifest)
        raise

    manifest["state"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    manifest["jobs"] = {
        name: manifest["jobs"][name] for name in sorted(manifest["jobs"])
    }
    _atomic_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "state": manifest["state"],
        "outputs": {
            name: record["output"] for name, record in manifest["jobs"].items()
        },
    }


def _arms_dir_for(args: Any) -> Optional[Path]:
    """Give each arm its own folder beside the suite's combined JSON.

    ``--output run/components.json`` also writes ``run/components/arms/*.json``
    so one arm's rows can be inspected, diffed or replaced on their own.
    """
    if not getattr(args, "output", None):
        return None
    output = Path(args.output)
    return output.parent / output.stem / "arms"


def main() -> None:
    args = _parse_args()
    if args.suite == "all":
        result = run_all_ablations(
            args.output or DEFAULT_ABLATION_RESULTS_ROOT,
            workers=int(args.workers),
        )
        if not args.quiet:
            print(json.dumps(result, indent=args.indent, sort_keys=True))
        return
    if args.suite in {"components", "retention"}:
        from .evaluation import EvalSettings, ScheduleSettings

        is_components = args.suite == "components"
        seeds = tuple(
            int(value) for value in str(
                args.components_seeds if is_components else args.retention_seeds
            ).split(",") if value.strip()
        )
        scenarios = tuple(
            value.strip() for value in str(
                args.components_scenarios if is_components else args.retention_scenarios
            ).split(",") if value.strip()
        )
        experiment = "component_ablation" if is_components else "retention_ablation"
        # Every arm runs the 'full' agent and differs only in its settings
        # overrides, so the roster here just has to be non-empty and valid.
        evaluation_config = EvalSettings(
            seeds=seeds,
            scenarios=scenarios,
            baselines=("full",),
            include_oracle=False,
            show_eta=False,
            recipe_count=int(
                args.components_recipes if is_components else args.retention_recipes
            ),
            schedule=ScheduleSettings(),
            frozen_pairs=EvalSettings.frozen_pairs,
            audit_period=0,
            model_settings={},
            experiment=experiment,
            workers=max(0, int(args.workers)),
        )
        runner = run_component_ablation if is_components else run_retention_ablation
        _emit(runner(evaluation_config, arms_dir=_arms_dir_for(args)), args)
        return
    if args.suite == "memory":
        from .evaluation import EvalSettings, ScheduleSettings

        seeds = tuple(
            int(value) for value in str(args.memory_seeds).split(",")
            if value.strip()
        )
        scenarios = tuple(
            value.strip() for value in str(args.memory_scenarios).split(",")
            if value.strip()
        )
        baselines = tuple(
            str(cell["baseline"]) for cell in MEMORY_PREDICTOR_ABLATION_CELLS
        )
        evaluation_config = EvalSettings(
            seeds=seeds,
            scenarios=scenarios,
            baselines=baselines,
            include_oracle=False,
            show_eta=False,
            recipe_count=int(args.memory_recipes),
            schedule=ScheduleSettings(),
            frozen_pairs=EvalSettings.frozen_pairs,
            audit_period=0,
            model_settings={},
            experiment="memory_predictor_ablation",
            workers=max(0, int(args.workers)),
        )
        _emit(run_memory_predictor_ablation(
            evaluation_config, arms_dir=_arms_dir_for(args),
        ), args)
        return
    if args.suite == "latent":
        from .evaluation import EvalSettings, ScheduleSettings

        seeds = tuple(
            int(value) for value in str(args.latent_seeds).split(",")
            if value.strip()
        )
        scenarios = tuple(
            value.strip() for value in str(args.latent_scenarios).split(",")
            if value.strip()
        )
        recipe_count = int(args.latent_recipes)
        schedule = ScheduleSettings()
        frozen_pairs = EvalSettings.frozen_pairs
        model_settings: Dict[str, Any] = {}
        evaluation_config = EvalSettings(
            seeds=seeds,
            scenarios=scenarios,
            baselines=("full",),
            include_oracle=False,
            show_eta=False,
            recipe_count=recipe_count,
            schedule=schedule,
            frozen_pairs=frozen_pairs,
            audit_period=0,
            model_settings=model_settings,
            experiment="latent_strategy_ablation",
            workers=max(0, int(args.workers)),
        )
        _emit(run_latent_strategy_ablation(
            evaluation_config, arms_dir=_arms_dir_for(args),
        ), args)
        return
    if args.suite == "representation":
        from .evaluation import EvalSettings, ScheduleSettings

        seeds = tuple(
            int(value) for value in str(args.representation_seeds).split(",")
            if value.strip()
        )
        scenarios = tuple(
            value.strip()
            for value in str(args.representation_scenarios).split(",")
            if value.strip()
        )
        # Every arm names its own baseline, so the roster here only has to be
        # non-empty and valid; `_representation_job` overrides it per arm.
        evaluation_config = EvalSettings(
            seeds=seeds,
            scenarios=scenarios,
            baselines=("full",),
            include_oracle=False,
            show_eta=False,
            recipe_count=int(args.representation_recipes),
            schedule=ScheduleSettings(),
            frozen_pairs=EvalSettings.frozen_pairs,
            audit_period=0,
            model_settings={},
            experiment="representation_ablation",
            workers=max(0, int(args.workers)),
        )
        _emit(run_representation_ablation(
            evaluation_config, arms_dir=_arms_dir_for(args),
        ), args)
        return
    if args.suite == "routing":
        from .evaluation import EvalSettings, ScheduleSettings

        seeds = tuple(
            int(value) for value in str(args.routing_seeds).split(",")
            if value.strip()
        )
        scenarios = tuple(
            value.strip() for value in str(args.routing_scenarios).split(",")
            if value.strip()
        )
        baselines = tuple(
            value.strip() for value in str(args.routing_baselines).split(",")
            if value.strip()
        )
        recipe_count = int(args.routing_recipes)
        schedule = ScheduleSettings()
        frozen_pairs = EvalSettings.frozen_pairs
        audit_period = EvalSettings.audit_period
        model_settings: Dict[str, Any] = {}
        evaluation_config = EvalSettings(
            seeds=seeds,
            scenarios=scenarios,
            baselines=baselines,
            include_oracle=False,
            shared_routing=False,
            observe_missing_recipes=True,
            show_eta=False,
            recipe_count=recipe_count,
            schedule=schedule,
            frozen_pairs=frozen_pairs,
            audit_period=audit_period,
            model_settings=model_settings,
            experiment="teaching_burden_routing_ablation",
            workers=max(0, int(args.workers)),
        )
        _emit(run_routing_ablation(
            evaluation_config, baselines=baselines,
        ), args)
        return
    config = MatcherSettings(
        seed=int(args.seed),
        recipe_count=int(args.recipe_count),
        case_limit=int(args.case_limit),
        include_prefixes=not bool(args.no_prefixes),
        calibrate=not bool(args.no_calibration),
    )
    _emit(run_matcher_ablation(config), args)


if __name__ == "__main__":
    main()
