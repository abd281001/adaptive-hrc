"""Ablations: the matcher stress suite, and the longitudinal arm grid.

Two independent programs live here, separated below.

SECTION 1 runs no deployment streams at all. It generates synthetic
recipe/preference identification cases and scores several matchers on them.

SECTION 2 is the longitudinal grid. An ablation there is an arm plus a
contrast: arms are run by the evaluation's own per-cell runner into its run
directory, so an arm shared between groups -- or with the deployable roster --
is computed once and read by every contrast that names it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
import gzip
import itertools
import json
from functools import lru_cache
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .environment import parse_action_label, recipe_builders, task_goal_signature, validate_ordering
from .memory import (
    MatchResult,
    RecipeMatcher,
    KnownVariant,
    _jaccard_counters,
    jaccard,
    make_variant_id,
)
from .models import Settings, DEFAULT_SETTINGS
from .preferences import PREFERENCE_IDS, apply_preset


# ==========================================================================
# SECTION 1 -- MATCHER STRESS SUITE
#
# Offline and self-contained: synthetic query/library cases, several
# matchers, and a commit audit over recorded decisions. It runs no
# deployment stream and shares nothing with SECTION 2 below.
# ==========================================================================
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


# ==========================================================================
# SECTION 2 -- LONGITUDINAL ARM GRID
#
# An ablation is not a suite. It is an arm plus a contrast.
#
# Every longitudinal ablation does the same three things: run a settings
# variant of an agent on the standard plan, replay ``full``'s realized route,
# and difference it against a reference arm. That is what the evaluation
# already does for its deployable roster, so ablation arms are run by the
# same per-cell runner into the same run directory. A cell that exists is
# reused, which is why no arm here is ever computed twice -- ``full`` is run
# once and every group that references it reads that one result.
#
# A "group" is therefore a label on contrasts, not a unit of execution.
# ==========================================================================

# How an arm gets its interaction schedule.
PUBLISH = "publish"   # runs unconstrained and publishes the route (only 'full')
REPLAY = "replay"     # replays the published route -- the default
LOCAL = "local"       # free to request its own re-observations (routing only)
ROUTES = (PUBLISH, REPLAY, LOCAL)

DEFAULT_ABLATION_RESULTS_ROOT = "eval_results"
DEFAULT_ABLATION_WORKERS = 8


@dataclass(frozen=True)
class Arm:
    """One condition to run, named once and run once.

    ``name`` labels every row and names the result folder; ``agent`` says
    which agent to build. They differ for a settings variant, which is a
    named override of an existing agent rather than a new class.
    """

    name: str
    label: str
    agent: str = "full"
    overrides: Mapping[str, Any] = field(default_factory=dict)
    route: str = REPLAY
    groups: Tuple[str, ...] = ()
    # Descriptive columns a group reports; never used to compute anything.
    facets: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"{self.name}: route must be one of {ROUTES}")
        if not self.groups:
            raise ValueError(f"{self.name}: an arm must belong to a group")


@dataclass(frozen=True)
class Contrast:
    """One seed-matched treatment-minus-reference comparison.

    Deltas are signed so a positive value always favours the treatment,
    whatever direction the underlying metric reads.
    """

    name: str
    group: str
    treatment: str
    reference: str
    interpretation: str
    primary: bool = False
    # Set when the two arms do not carry identical components, so the
    # difference cannot be attributed to the factor the name suggests.
    descriptive: bool = False


@dataclass(frozen=True)
class Invariant:
    """A claim about the arms that a group's contrasts depend on.

    With ``expected`` the named arms must show those exact field values; with
    ``expected`` omitted the arms must merely agree with each other. Both
    read the recorded state of the run, not the declared configuration, so a
    group cannot pass by asserting what it meant to do.
    """

    name: str
    group: str
    arms: Tuple[str, ...]
    fields: Tuple[str, ...]
    expected: Optional[Mapping[str, Mapping[str, Any]]] = None


# Every group reports these; they are the evaluation's own headline metrics.
CORE_METRICS: Tuple[Tuple[str, str], ...] = (
    ("teacher_forced_top_1", "higher_is_better"),
    ("teacher_forced_top_k", "higher_is_better"),
    ("teacher_forced_mean_nll", "lower_is_better"),
    ("live_top_1", "higher_is_better"),
    ("normalized_human_action_load", "lower_is_better"),
    ("corrections_per_recipe_step", "lower_is_better"),
)

# Extra endpoints a group needs on top of the core. No direction may conflict
# with the core table; ``_validate_tables`` enforces that.
GROUP_METRICS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "latent": (
        ("cross_recipe_teacher_forced_top_1", "higher_is_better"),
        ("seen_recipe_preference_seen_elsewhere_top_1", "higher_is_better"),
        ("training_estimated_fit_flops", "lower_is_better"),
    ),
    "representation": (
        ("unseen_state_fallback_top_1", "higher_is_better"),
        ("branching_top_1", "higher_is_better"),
        ("single_option_lookup_top_1", "higher_is_better"),
        ("direct_retrieval_top_1", "higher_is_better"),
        ("new_preference_top_1", "higher_is_better"),
        ("preference_seen_elsewhere_top_1", "higher_is_better"),
    ),
    "routing": (
        ("n_extra_demonstration_actions", "lower_is_better"),
        ("n_total_explicit_teaching_actions", "lower_is_better"),
        ("n_corrective_teaching_actions", "lower_is_better"),
    ),
}

# --------------------------------------------------------------------------
# Arms.
#
# Only the diagnostic arms are declared here. The deployable roster lives in
# src/evaluation.py and is referenced by name: re-declaring it is exactly the
# duplication this table exists to remove. `_validate_tables` rejects an arm
# whose condition already exists in that roster.
# --------------------------------------------------------------------------
ARMS: Tuple[Arm, ...] = (
    # -- components: which part of Full's predictor support earns its place.
    #    'full' is the reference and 'full_no_support' is the deployable
    #    'unpinned' baseline; both are read from the standard roster.
    Arm("full_no_pin", "Full without the latest pin", groups=("components",),
        overrides={"pin_latest": False},
        facets={"component_level": "no_latest_pin", "removes": "latest_pin"}),
    Arm("full_no_semantic_fallback", "Full without the semantic fallback",
        groups=("components",),
        overrides={"semantic_fallback_enabled": False},
        facets={"component_level": "no_semantic_fallback",
                "removes": "semantic_fallback"}),
    Arm("full_no_latent_residual", "Full without the latent residual",
        # Also the latent group's MaxEnt-only reference: with the residual
        # off, its rank/knn/strength/sequence settings are inert, so the old
        # 'maxent_only' arm was this same condition under another name.
        groups=("components", "latent"),
        overrides={"latent_strategy_enabled": False},
        facets={"component_level": "no_latent_residual",
                "removes": "latent_residual"}),

    # -- retention: which part of the recurrence-aware memory policy works.
    Arm("full_retain_all", "Full, retain everything", groups=("retention",),
        overrides={"retention_policy": "none"},
        facets={"factor": "retention_rule"}),
    Arm("constant_grace", "Constant grace horizon", groups=("retention",),
        overrides={"horizon_estimator": "constant"},
        facets={"factor": "horizon_estimator"}),
    Arm("pair_only_horizon", "Pair evidence only, no parent pooling",
        groups=("retention",), overrides={"horizon_estimator": "pair_only"},
        facets={"factor": "horizon_estimator"}),
    Arm("shuffled_horizon", "Recurrence evidence reassigned across pairs",
        groups=("retention",), overrides={"horizon_estimator": "shuffled"},
        facets={"factor": "horizon_assignment"}),
    Arm("symmetric_adaptation", "Symmetric horizon adaptation",
        groups=("retention",), overrides={"pair_adaptation": "symmetric"},
        facets={"factor": "pair_adaptation"}),
    Arm("recent_set_pin", "Pin the recently active preference set",
        groups=("retention",), overrides={"pin_mode": "recent_set"},
        facets={"factor": "pin_scope"}),
    Arm("schedule_warm_on_weight_change", "Warm fit on every weight change",
        groups=("retention",),
        overrides={"retrain_warm_on_weight_change": True},
        facets={"factor": "consolidation_schedule"}),
    Arm("schedule_removals_count_cold", "Removals advance the cold counter",
        groups=("retention",),
        overrides={"retrain_cold_counts_removals": True},
        facets={"factor": "consolidation_schedule"}),

    # -- latent: capacity x trajectory alignment.
    #    The deployed residual is rank 8 / knn 3 / sequence weight 0.5, which
    #    is the standard 'full' arm. Declaring the other three corners here
    #    completes the 2x2 -- the design used to omit the shipped corner and
    #    could only price trajectory alignment at high capacity.
    Arm("latent_timing_lightweight", "Lightweight latent timing",
        groups=("latent",),
        overrides={"latent_strategy_enabled": True, "latent_strategy_rank": 8,
                   "latent_strategy_knn": 3, "latent_strategy_strength": 1.0,
                   "latent_strategy_sequence_weight": 0.0},
        facets={"capacity": "low", "sequence_alignment": "off"}),
    Arm("latent_timing_capacity_matched", "Capacity-matched latent timing",
        groups=("latent",),
        overrides={"latent_strategy_enabled": True, "latent_strategy_rank": 32,
                   "latent_strategy_knn": 8, "latent_strategy_strength": 1.0,
                   "latent_strategy_sequence_weight": 0.0},
        facets={"capacity": "high", "sequence_alignment": "off"}),
    Arm("latent_timing_trajectory_hybrid", "Heavy timing-trajectory hybrid",
        groups=("latent",),
        overrides={"latent_strategy_enabled": True, "latent_strategy_rank": 32,
                   "latent_strategy_knn": 8, "latent_strategy_strength": 1.0,
                   "latent_strategy_sequence_weight": 0.5},
        facets={"capacity": "high", "sequence_alignment": "on"}),

    # -- memory: predictor x retention policy, 2x2.
    #    'full' and 'bc' are the standard roster's own arms.
    Arm("bc_adaptive", "BC + adaptive memory", agent="bc_adaptive",
        # Also the representation group's full-history cloner: its declared
        # overrides were the defaults, so it was this same condition.
        groups=("memory", "representation"),
        facets={"predictor": "behavior_cloning", "memory": "adaptive_pinned",
                "family": "behavior_cloning",
                "representation": "state_plus_action_history",
                "within_episode_history": "three_action_lags_and_step_count"}),
    Arm("maxent_retain_all", "MaxEnt + retain-all", groups=("memory",),
        overrides={"retention_policy": "none", "pin_latest": False},
        facets={"predictor": "maxent", "memory": "retain_all"}),

    # -- representation: what the predictor is allowed to see.
    Arm("maxent_semantic", "MaxEnt, semantic reward features",
        groups=("representation",), overrides={"irl_features": "semantic"},
        facets={"family": "maxent", "representation": "semantic_reward_features",
                "within_episode_history": "none"}),
    Arm("maxent_raw_state", "MaxEnt, raw state features",
        groups=("representation",), overrides={"irl_features": "raw_state"},
        facets={"family": "maxent", "representation": "raw_state_features",
                "within_episode_history": "none"}),
    Arm("cloner_history_1", "Cloner, one action lag and step count",
        agent="bc_adaptive", groups=("representation",),
        overrides={"bc_history": 1, "bc_prefix_length_feature": True},
        facets={"family": "behavior_cloning",
                "representation": "state_plus_action_history",
                "within_episode_history": "one_action_lag_and_step_count"}),
    Arm("cloner_step_count_only", "Cloner, step count only",
        agent="bc_adaptive", groups=("representation",),
        overrides={"bc_history": 0, "bc_prefix_length_feature": True},
        facets={"family": "behavior_cloning",
                "representation": "state_plus_step_count",
                "within_episode_history": "step_count_only"}),
    Arm("cloner_state_only", "Cloner, state only", agent="bc_adaptive",
        groups=("representation",),
        overrides={"bc_history": 0, "bc_prefix_length_feature": False},
        facets={"family": "behavior_cloning", "representation": "state_only",
                "within_episode_history": "none"}),

    # -- routing: teaching burden when a method may request its own demos.
    #    Only a method that can lose every active variant of a recipe can
    #    route differently, so only the three decaying baselines get a local
    #    arm. Retain-all methods never prune, so their local run is provably
    #    the shared run and used to be computed anyway.
    Arm("unpinned_local", "Adaptive decay, local re-observation",
        agent="unpinned", route=LOCAL, groups=("routing",),
        facets={"shared_arm": "unpinned"}),
    Arm("latest_local", "Latest-only, local re-observation",
        agent="latest", route=LOCAL, groups=("routing",),
        facets={"shared_arm": "latest"}),
    Arm("fixed_local", "Fixed decay, local re-observation",
        agent="fixed", route=LOCAL, groups=("routing",),
        facets={"shared_arm": "fixed"}),
)


# --------------------------------------------------------------------------
# Contrasts. One row per planned comparison; the summary computes nothing
# that is not declared here, so the reported set cannot drift from the design.
# --------------------------------------------------------------------------
CONTRASTS: Tuple[Contrast, ...] = (
    # -- components: positive means Full lost that much when the part went.
    Contrast("cost_of_removing_latest_pin", "components", "full", "full_no_pin",
             "What Full loses without latest-preference protection.", primary=True),
    Contrast("cost_of_removing_semantic_fallback", "components",
             "full", "full_no_semantic_fallback",
             "What Full loses without bounded semantic successor interpolation."),
    Contrast("cost_of_removing_latent_residual", "components",
             "full", "full_no_latent_residual",
             "What Full loses without the workflow-strategy residual."),
    Contrast("cost_of_removing_all_support", "components", "full", "unpinned",
             "Joint removal of all three. 'unpinned' is the deployable "
             "baseline, which is settings-identical to removing the three "
             "components together, so the roster arm is reused rather than "
             "re-run under a second name."),

    # -- retention: positive means Full lost that much when the mechanism moved.
    Contrast("cost_of_retaining_everything", "retention", "full",
             "full_retain_all", "Value of retiring anything at all.",
             primary=True),
    Contrast("cost_of_a_constant_horizon", "retention", "full",
             "constant_grace",
             "Value of estimating a per-pair horizon rather than fixing one."),
    Contrast("cost_of_pair_only_evidence", "retention", "full",
             "pair_only_horizon",
             "Value of pooling recurrence evidence across a pair's parents."),
    Contrast("cost_of_reassigned_evidence", "retention", "full",
             "shuffled_horizon",
             "Holds the horizon distribution fixed and destroys only its "
             "assignment to pairs: separates the policy from how much it "
             "happens to retain."),
    Contrast("cost_of_symmetric_adaptation", "retention", "full",
             "symmetric_adaptation",
             "Value of adapting a pair's horizon asymmetrically."),
    Contrast("cost_of_widening_the_pin", "retention", "full", "recent_set_pin",
             "Effect of protecting every recently demonstrated sibling "
             "instead of one variant per recipe."),
    Contrast("cost_of_warm_fitting_weight_changes", "retention", "full",
             "schedule_warm_on_weight_change",
             "Effect of refitting whenever a rehearsal weight moves."),
    Contrast("cost_of_counting_removals_cold", "retention", "full",
             "schedule_removals_count_cold",
             "Effect of letting removals advance the cold-restart counter."),

    # -- latent: a 2x2 of capacity x trajectory alignment.
    #    'full' is the deployed corner (rank 8, knn 3, sequence weight 0.5).
    Contrast("latent_contribution_as_deployed", "latent", "full",
             "full_no_latent_residual",
             "What the residual contributes in the configuration that ships.",
             primary=True),
    Contrast("lightweight_latent_contribution", "latent",
             "latent_timing_lightweight", "full_no_latent_residual",
             "Contribution of low-cost cross-recipe latent timing over MaxEnt."),
    Contrast("trajectory_alignment_at_low_capacity", "latent", "full",
             "latent_timing_lightweight",
             "Causal contribution of partial-trajectory alignment at the "
             "deployed capacity: rank, KNN, strength and every non-latent "
             "setting match."),
    Contrast("trajectory_alignment_at_high_capacity", "latent",
             "latent_timing_trajectory_hybrid", "latent_timing_capacity_matched",
             "The same contribution at raised capacity. Reading it against "
             "the low-capacity contrast gives the capacity x alignment "
             "interaction the three-arm design could not estimate."),
    Contrast("latent_capacity_sensitivity", "latent",
             "latent_timing_capacity_matched", "latent_timing_lightweight",
             "Sensitivity to rank and neighbour count together. Rank and KNN "
             "move at once, so this bounds capacity rather than attributing "
             "to either.", descriptive=True),

    # -- memory: predictor x retention, 2x2.
    Contrast("memory_effect_within_maxent", "memory", "full",
             "maxent_retain_all",
             "Does the memory policy help the proposed predictor."),
    Contrast("memory_effect_within_behavior_cloning", "memory", "bc_adaptive",
             "bc", "Does the same memory policy help a linear cloner.",
             primary=True),
    Contrast("predictor_effect_under_adaptive_memory", "memory", "full",
             "bc_adaptive",
             "Predictor gap once both arms share Full's memory policy. The "
             "MaxEnt arm also carries the semantic fallback and the latent "
             "residual and the cloner carries neither, so this is not a "
             "component-matched comparison.", descriptive=True),
    Contrast("predictor_effect_under_retain_all", "memory",
             "maxent_retain_all", "bc",
             "Predictor gap when neither arm forgets. Not component-matched, "
             "for the same reason.", descriptive=True),

    # -- representation: what the predictor can see, memory held fixed.
    Contrast("cloner_history_contribution", "representation", "bc_adaptive",
             "cloner_state_only",
             "Causal contribution of within-episode action history to the "
             "cloner, with the memory policy, the stored demonstrations and "
             "every other setting matched.", primary=True),
    Contrast("cloner_action_identity_contribution", "representation",
             "bc_adaptive", "cloner_step_count_only",
             "Separates knowing which actions were taken from knowing how "
             "far the episode has progressed."),
    Contrast("maxent_reward_representation_sensitivity", "representation",
             "full", "maxent_raw_state",
             "Bounds how much of the MaxEnt gap the reward representation "
             "can account for. All three MaxEnt arms keep the semantic "
             "fallback, so its separate representation still supplies values "
             "where exact support is missing.", descriptive=True),
    Contrast("history_free_family_contrast", "representation", "full",
             "cloner_state_only",
             "Model family at matched access to history. A small gap here "
             "beside a large history contribution locates the difference in "
             "the representation rather than the learner. The families do "
             "not carry identical components.", descriptive=True),

    # -- routing: what a method spends when it may teach itself.
    #    The reference is the roster arm, which already ran on the shared
    #    route; only the local arm is new work.
    Contrast("local_routing_burden_adaptive_decay", "routing", "unpinned",
             "unpinned_local",
             "Teaching bought by letting adaptive decay re-observe at will.",
             primary=True),
    Contrast("local_routing_burden_latest_only", "routing", "latest",
             "latest_local",
             "The same for latest-only storage."),
    Contrast("local_routing_burden_fixed_decay", "routing", "fixed",
             "fixed_local",
             "The same for fixed-rate forgetting."),
)


# --------------------------------------------------------------------------
# Invariants. Each is read back from the recorded state of the run, so a
# group cannot pass by asserting what it meant to configure.
# --------------------------------------------------------------------------
_SUPPORT_FIELDS = (
    "latest_pin_enabled", "semantic_fallback_enabled", "latent_strategy_enabled",
)
INVARIANTS: Tuple[Invariant, ...] = (
    Invariant(
        "each_component_arm_moved_exactly_its_own_component", "components",
        ("full", "full_no_pin", "full_no_semantic_fallback",
         "full_no_latent_residual", "unpinned"),
        _SUPPORT_FIELDS,
        expected={
            "full": dict(zip(_SUPPORT_FIELDS, (True, True, True))),
            "full_no_pin": dict(zip(_SUPPORT_FIELDS, (False, True, True))),
            "full_no_semantic_fallback": dict(zip(_SUPPORT_FIELDS, (True, False, True))),
            "full_no_latent_residual": dict(zip(_SUPPORT_FIELDS, (True, True, False))),
            "unpinned": dict(zip(_SUPPORT_FIELDS, (False, False, False))),
        },
    ),
    Invariant("component_arms_held_retention_fixed", "components",
              ("full", "full_no_semantic_fallback", "full_no_latent_residual"),
              ("memory_policy",)),
    Invariant("retention_arms_held_predictor_support_fixed", "retention",
              ("full", "constant_grace", "pair_only_horizon", "shuffled_horizon",
               "symmetric_adaptation", "schedule_warm_on_weight_change",
               "schedule_removals_count_cold"),
              ("semantic_fallback_enabled", "latent_strategy_enabled")),
    Invariant("adaptive_memory_level_is_internally_consistent", "memory",
              ("full", "bc_adaptive"), ("memory_policy", "latest_pin_enabled")),
    Invariant("retain_all_memory_level_is_internally_consistent", "memory",
              ("maxent_retain_all", "bc"), ("memory_policy", "latest_pin_enabled")),
    Invariant("maxent_cells_share_predictor_support", "memory",
              ("full", "maxent_retain_all"), _SUPPORT_FIELDS[1:]),
    Invariant("representation_arms_held_memory_fixed", "representation",
              ("full", "maxent_semantic", "maxent_raw_state", "bc_adaptive",
               "cloner_history_1", "cloner_step_count_only", "cloner_state_only"),
              ("memory_policy", "latest_pin_enabled")),
    Invariant("latent_arms_share_the_maxent_head", "latent",
              ("full", "full_no_latent_residual", "latent_timing_lightweight",
               "latent_timing_capacity_matched", "latent_timing_trajectory_hybrid"),
              ("predictor", "semantic_fallback_enabled", "memory_policy")),
)


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


# --------------------------------------------------------------------------
# Table access and startup consistency.
# --------------------------------------------------------------------------
def arms_by_name() -> Dict[str, Arm]:
    return {arm.name: arm for arm in ARMS}


def groups() -> Tuple[str, ...]:
    """Every declared contrast group, in table order."""
    return tuple(dict.fromkeys(contrast.group for contrast in CONTRASTS))


def group_contrasts(group: str) -> Tuple[Contrast, ...]:
    return tuple(c for c in CONTRASTS if c.group == group)


def group_metrics(group: str) -> Tuple[Tuple[str, str], ...]:
    return CORE_METRICS + tuple(GROUP_METRICS.get(group, ()))


def group_arms(group: str) -> Tuple[str, ...]:
    """Every arm a group's contrasts and invariants name, in stable order."""
    names: List[str] = []
    for contrast in group_contrasts(group):
        names.extend((contrast.treatment, contrast.reference))
    for invariant in INVARIANTS:
        if invariant.group == group:
            names.extend(invariant.arms)
    return tuple(dict.fromkeys(names))


def roster_arms(selected: Sequence[str]) -> Tuple[str, ...]:
    """Arms a group needs that the deployable roster already provides.

    These are never run here. Re-running them under a second name is the
    duplication this module exists to remove, so a missing one is an error
    telling the caller to complete the standard evaluation first.
    """
    declared = arms_by_name()
    names: List[str] = []
    for group in selected:
        names.extend(n for n in group_arms(group) if n not in declared)
    return tuple(dict.fromkeys(names))


def declared_arms(selected: Sequence[str]) -> Tuple[Arm, ...]:
    """Arms this module must run for the selected groups."""
    wanted = {n for group in selected for n in group_arms(group)}
    return tuple(arm for arm in ARMS if arm.name in wanted)


def _validate_tables() -> None:
    """Reject a table that cannot mean what it says. Runs at import."""
    from .evaluation import DEFAULT_BASELINES, MEMORY_ORACLE

    seen: Dict[str, Arm] = {}
    for arm in ARMS:
        if arm.name in seen:
            raise ValueError(f"duplicate arm name {arm.name!r}")
        if arm.name in DEFAULT_BASELINES or arm.name == MEMORY_ORACLE:
            raise ValueError(
                f"{arm.name!r} is already a deployable roster arm; reference "
                "it by name instead of redeclaring it"
            )
        seen[arm.name] = arm
    # Two arms with the same agent and the same overrides are one condition
    # run twice. Names may differ; the work must not.
    conditions: Dict[Tuple[str, str, Tuple[Tuple[str, Any], ...]], str] = {}
    for arm in ARMS:
        key = (arm.agent, arm.route, tuple(sorted(arm.overrides.items())))
        if key in conditions:
            raise ValueError(
                f"{arm.name!r} and {conditions[key]!r} are the same condition; "
                "give one arm both groups instead of declaring it twice"
            )
        conditions[key] = arm.name
    known = set(seen) | set(DEFAULT_BASELINES) | {MEMORY_ORACLE}
    for contrast in CONTRASTS:
        for side in (contrast.treatment, contrast.reference):
            if side not in known:
                raise ValueError(f"{contrast.name!r} names unknown arm {side!r}")
        if contrast.treatment == contrast.reference:
            raise ValueError(f"{contrast.name!r} compares an arm with itself")
    for invariant in INVARIANTS:
        for name in invariant.arms:
            if name not in known:
                raise ValueError(f"{invariant.name!r} names unknown arm {name!r}")
        if invariant.expected is not None:
            missing = set(invariant.arms) - set(invariant.expected)
            if missing:
                raise ValueError(
                    f"{invariant.name!r} expects values for {sorted(missing)}"
                )
    # A metric must read the same direction everywhere it appears.
    directions: Dict[str, str] = dict(CORE_METRICS)
    for extras in GROUP_METRICS.values():
        for metric, direction in extras:
            if directions.setdefault(metric, direction) != direction:
                raise ValueError(f"metric {metric!r} declares two directions")
    for group in groups():
        if not any(c.primary for c in group_contrasts(group)):
            raise ValueError(f"group {group!r} declares no primary contrast")


_validate_tables()


# --------------------------------------------------------------------------
# Reading a finished cell.
#
# Everything below reads the per-arm cell folders the evaluation already
# writes. Nothing is recomputed and nothing is copied: the full per-stream
# summary stays where it was written.
# --------------------------------------------------------------------------
_TRANSFER_CELL_METRICS: Mapping[str, str] = {
    "direct_retrieval_top_1": "direct_retrieval",
    "new_preference_top_1": "seen_recipe_new_preference",
    "preference_seen_elsewhere_top_1": "seen_recipe_preference_seen_elsewhere",
    "seen_recipe_preference_seen_elsewhere_top_1":
        "seen_recipe_preference_seen_elsewhere",
}
_REGIME_METRICS: Mapping[str, str] = {
    "single_option_lookup_top_1": "single_option_lookup",
    "branching_top_1": "branching",
    "unseen_state_fallback_top_1": "unseen_state_fallback",
}
_PATH_METRICS: Mapping[str, Tuple[str, ...]] = {
    "cross_recipe_teacher_forced_top_1": (
        "assist", "by_hypothesis", "cross_recipe_transfer", "teacher_forced_top_1",
    ),
    "training_estimated_fit_flops": ("system", "training_estimated_fit_flops"),
}
# Configuration an arm holds fixed across every cell. The invariants read
# these, and the means table reports them verbatim.
STATE_FIELDS: Tuple[str, ...] = (
    "predictor", "irl_features", "memory_policy", "latest_pin_enabled",
    "semantic_fallback_enabled", "latent_strategy_enabled",
)
# Quantities that differ from seed to seed. Reporting one seed's value as the
# arm's would be wrong, so these are averaged like any other metric.
COST_FIELDS: Tuple[str, ...] = (
    "active_variants", "pruned_variants", "training_retrain_count",
    "training_total_retrain_wall_s", "training_estimated_fit_flops",
)


def _dig(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def metric_value(summary: Mapping[str, Any], metric: str) -> Optional[float]:
    """One metric out of one arm's stream summary."""
    if metric in _PATH_METRICS:
        return _as_optional_float(_dig(summary, _PATH_METRICS[metric]))
    if metric in _TRANSFER_CELL_METRICS:
        cell = _dig(summary, ("assist", "by_transfer_cell",
                             _TRANSFER_CELL_METRICS[metric])) or {}
        if not _as_optional_float(cell.get("n_teacher_forced_predictions")):
            return None
        return _as_optional_float(cell.get("teacher_forced_top_1"))
    if metric in _REGIME_METRICS:
        return _as_optional_float(_dig(summary, (
            "diagnostics", "decision_regimes", "by_regime",
            _REGIME_METRICS[metric], "top_1",
        )))
    return _as_optional_float(_dig(summary, ("assist", "overall", metric)))


def _gz_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def cell_row(
    run_dir: Path,
    arm: str,
    scenario: str,
    seed: int,
    metrics: Sequence[Tuple[str, str]],
    *,
    teaching: bool = False,
) -> Optional[Dict[str, Any]]:
    """Flatten one finished cell into a comparison row, or None if absent."""
    from .evaluation import _baseline_cell_dir

    cell = _baseline_cell_dir(run_dir, arm, scenario, int(seed))
    path = cell / "summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "complete":
        return None
    summary = (payload.get("per_baseline") or {}).get(arm)
    if not isinstance(summary, Mapping):
        return None
    system = summary.get("system") or {}
    row: Dict[str, Any] = {
        "arm": arm, "scenario": str(scenario), "seed": int(seed),
        **{field: system.get(field) for field in STATE_FIELDS + COST_FIELDS},
        **{metric: metric_value(summary, metric) for metric, _d in metrics},
    }
    if teaching:
        # The only quantity that needs the tables rather than the summary:
        # how much extra teaching a self-routing arm actually bought.
        row.update(teaching_metrics(
            _gz_rows(cell / "tables" / "episodes.jsonl.gz"),
            _gz_rows(cell / "tables" / "frozen_probes.jsonl.gz"),
        ))
    return row


def read_rows(
    run_dir: Path,
    selected: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, List[Dict[str, Any]]]:
    """One row per (arm, scenario, seed) for each selected group."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for group in selected:
        metrics = group_metrics(group)
        teaching = group == "routing"
        rows: List[Dict[str, Any]] = []
        for arm in group_arms(group):
            for scenario in scenarios:
                for seed in seeds:
                    row = cell_row(run_dir, arm, scenario, int(seed), metrics,
                                   teaching=teaching)
                    if row is not None:
                        rows.append(row)
        out[group] = rows
    return out


# --------------------------------------------------------------------------
# Running the arms.
#
# Arms are filled into the standard evaluation's run directory by its own
# per-cell runner, so a cell that already exists is reused rather than
# recomputed. That is the whole mechanism: there is no cache, no reference
# flag and no import path -- the run directory is the shared result.
# --------------------------------------------------------------------------
def arm_config(arm: Arm, config: Any) -> Any:
    """The evaluation config for one arm: its overrides and its route.

    Nothing that determines the plan is touched, so every arm is scored on
    the identical schedule the standard evaluation built.
    """
    from .evaluation import EvalSettings  # noqa: F401  (documents the type)

    if arm.route == LOCAL:
        routing = {
            "shared_routing": False,
            "observe_missing_recipes": True,
            "allow_repeat_observation": True,
        }
    else:
        routing = {
            "shared_routing": True,
            "observe_missing_recipes": False,
            "allow_repeat_observation": False,
        }
    return replace(
        config,
        baselines=(arm.name,),
        include_oracle=False,
        sensitivity=False,
        model_settings={**dict(config.model_settings), **dict(arm.overrides)},
        **routing,
    )


def _rebuild_config(payload: Mapping[str, Any]) -> Any:
    """An EvalSettings from a manifest's recorded configuration."""
    from .evaluation import EvalSettings, ScheduleSettings

    def restore(value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    config = dict(payload)
    schedule = config.pop("schedule", None)
    fields = {key: restore(value) for key, value in config.items()}
    if isinstance(schedule, Mapping):
        fields["schedule"] = ScheduleSettings(**{
            key: restore(value) for key, value in schedule.items()
        })
    return EvalSettings(**fields)


def find_run(
    output_root: str | Path = DEFAULT_ABLATION_RESULTS_ROOT,
    run: Optional[str] = None,
) -> Tuple[Path, Any]:
    """The completed evaluation run to contribute arms to, and its config.

    The grid is adopted from the run rather than reconstructed, so an
    ablation cannot silently describe a different experiment than the one it
    is adding arms to. Only the arm selection is overridden.
    """
    from .evaluation import RUNS_DIRNAME, _config_hash, manifest_identity

    runs = Path(output_root).expanduser().resolve() / RUNS_DIRNAME
    candidates: List[Tuple[float, Path, Mapping[str, Any]]] = []
    if runs.is_dir():
        for run_dir in sorted(runs.iterdir()):
            if run is not None and run_dir.name != run:
                continue
            manifest = run_dir / "manifest.json"
            status = run_dir / "status.json"
            if not manifest.is_file() or not status.is_file():
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                state = json.loads(status.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if state.get("state") != "complete":
                continue
            candidates.append((manifest.stat().st_mtime, run_dir, payload))
    if not candidates:
        where = f"{runs}" + (f"/{run}" if run else "")
        raise SystemExit(
            f"no completed evaluation run under {where}. Ablation arms are "
            "contributed to the standard run so the roster is never computed "
            "twice; finish `./hrc run` first."
        )
    _mtime, run_dir, payload = max(candidates, key=lambda item: item[0])
    config = _rebuild_config(payload["config"])
    # The rebuild must be faithful, or the arms would be scored on a
    # different plan than the roster they are compared against.
    recomputed = manifest_identity(payload)
    if recomputed is not None and _config_hash(config) != recomputed:
        raise SystemExit(
            f"{run_dir.name} records a configuration this build cannot "
            "reproduce, so its cells are not comparable with new arms."
        )
    return run_dir, config


def resolve_run(config: Any) -> Path:
    """The run directory this configuration's arms belong in."""
    from .evaluation import RUNS_DIRNAME, _config_hash, manifest_identity

    runs = Path(config.output).expanduser().resolve() / RUNS_DIRNAME
    digest = _config_hash(config)
    matches = []
    if runs.is_dir():
        for run_dir in sorted(runs.iterdir()):
            manifest = run_dir / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            # Recomputed from the manifest, not the digest the run saved, so
            # a run recorded under an older definition still resolves.
            if manifest_identity(payload) == digest:
                matches.append(run_dir)
    if not matches:
        raise SystemExit(
            "no evaluation run matches this configuration. Ablation arms are "
            "contributed to the standard run so the roster is never computed "
            "twice; run `./hrc run` first."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def _require_roster(run_dir: Path, selected: Sequence[str],
                    scenarios: Sequence[str], seeds: Sequence[int]) -> None:
    """Every roster arm a contrast references must already be complete."""
    from .evaluation import _completed_cell_result

    missing = [
        f"{arm}/{scenario}/{int(seed)}"
        for arm in roster_arms(selected)
        for scenario in scenarios for seed in seeds
        if _completed_cell_result(run_dir, arm, scenario, int(seed)) is None
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} roster cells are missing from {run_dir.name}, "
            f"starting with {missing[0]}. These are the deployable arms the "
            "contrasts reference; complete `./hrc run` before ablating."
        )


def run_arms(
    selected: Sequence[str],
    config: Any,
    *,
    workers: int = DEFAULT_ABLATION_WORKERS,
    progress: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Fill every missing cell the selected groups need. Returns the run dir."""
    from .evaluation import (
        _completed_cell_result, _pending_cell_results, _worker_count,
    )

    run_dir = resolve_run(config)
    scenarios = tuple(str(s) for s in config.scenarios)
    seeds = tuple(int(s) for s in config.seeds)
    _require_roster(run_dir, selected, scenarios, seeds)

    executed: Dict[str, int] = {}
    reused: Dict[str, int] = {}
    for arm in declared_arms(selected):
        pending = [
            (scenario, int(seed))
            for scenario in scenarios for seed in seeds
            if _completed_cell_result(run_dir, arm.name, scenario, int(seed)) is None
        ]
        reused[arm.name] = len(scenarios) * len(seeds) - len(pending)
        executed[arm.name] = len(pending)
        if not pending:
            if progress:
                print(f"[ablation] {arm.name}: all cells reused", flush=True)
            continue
        cell_config = arm_config(arm, config)
        count = _worker_count(cell_config, len(pending), arm.name)
        if progress:
            print(f"[ablation] {arm.name}: {len(pending)} cells, "
                  f"{count} workers", flush=True)
        for _result in _pending_cell_results(
            arm.name, pending, cell_config, run_dir, count, arm.agent,
        ):
            pass
    return run_dir, {
        "run": run_dir.name,
        "groups": list(selected),
        "cells_executed": executed,
        "cells_reused": reused,
        "roster_arms_reused": list(roster_arms(selected)),
    }


# --------------------------------------------------------------------------
# Summarising. One signed paired delta, one test, one output schema, for
# every group. Positive always favours the treatment.
# --------------------------------------------------------------------------
GROUP_NOTES: Mapping[str, str] = {
    "components": (
        "One-factor-at-a-time removal of Full's three predictor-support "
        "components, plus their joint removal, with retention held at Full's "
        "policy in every arm."
    ),
    "retention": (
        "One mechanism of the recurrence-aware retention policy moved per "
        "arm, under Full's predictor support."
    ),
    "latent": (
        "A 2x2 of residual capacity against partial-trajectory alignment. "
        "The deployed residual is the low-capacity, alignment-on corner, so "
        "the shipped system is inside the design rather than beside it."
    ),
    "memory": (
        "A 2x2 crossing the predictor with the retention policy. The "
        "within-family memory effects and their interaction are "
        "component-matched; the cross-family predictor effects are not."
    ),
    "representation": (
        "What the predictor is allowed to see, with Full's memory policy "
        "held fixed. MaxEnt arms vary the reward representation; cloner arms "
        "vary within-episode history down to a state-only condition."
    ),
    "routing": (
        "Teaching burden when a method may request its own re-observations "
        "instead of replaying Full's schedule. Only methods that can retire "
        "a variant can route differently, so only those have a local arm."
    ),
}


def _holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjustment across one group's tests."""
    values = [float(p) for p in pvalues]
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[Tuple[str, int], Mapping[str, Any]]]:
    """Rows by scenario, then by (arm, seed). Built once per group."""
    indexed: Dict[str, Dict[Tuple[str, int], Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        indexed[str(row["scenario"])][(str(row["arm"]), int(row["seed"]))] = row
    return dict(indexed)


def paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    contrast: Contrast,
    metrics: Sequence[Tuple[str, str]],
    scenario: str,
    index: Optional[Mapping[str, Mapping[Tuple[str, int], Mapping[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Seed-matched treatment-minus-reference deltas for one scenario."""
    from .evaluation import _sign_flip_test

    indexed = (index or index_rows(rows)).get(scenario, {})
    seeds = sorted({
        seed for (arm, seed) in indexed
        if arm == contrast.treatment and (contrast.reference, seed) in indexed
    })
    out: Dict[str, Any] = {}
    for metric, direction in metrics:
        deltas: List[float] = []
        for seed in seeds:
            high = _as_optional_float(indexed[(contrast.treatment, seed)].get(metric))
            low = _as_optional_float(indexed[(contrast.reference, seed)].get(metric))
            if high is None or low is None:
                continue
            delta = high - low
            if direction == "lower_is_better":
                delta = -delta
            if math.isfinite(delta):
                deltas.append(delta)
        if not deltas:
            continue
        one_sided, two_sided = _sign_flip_test(deltas)
        out[metric] = {
            "direction": direction,
            "n_paired_seeds": len(deltas),
            "mean_treatment_advantage": _finite_mean(deltas),
            "seeds_favouring_treatment": sum(1 for d in deltas if d > 0.0),
            "sign_flip_p_one_sided": one_sided,
            "sign_flip_p_two_sided": two_sided,
        }
    return {"scenario": scenario, "n_paired_seeds": len(seeds), "metrics": out}


def check_invariants(
    group: str, rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Read back whether the arms really were what the group claims."""
    out: List[Dict[str, Any]] = []
    by_arm: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    for invariant in INVARIANTS:
        if invariant.group != group:
            continue
        observed = {
            arm: {
                field: sorted({row.get(field) for row in by_arm[arm]}, key=repr)
                for field in invariant.fields
            }
            for arm in invariant.arms if by_arm.get(arm)
        }
        problems: List[str] = []
        for arm, fields in observed.items():
            for field, values in fields.items():
                if len(values) != 1:
                    problems.append(f"{arm}.{field} varies across cells: {values}")
                elif invariant.expected is not None:
                    want = invariant.expected[arm][field]
                    if values[0] != want:
                        problems.append(
                            f"{arm}.{field} is {values[0]!r}, expected {want!r}"
                        )
        if invariant.expected is None:
            for field in invariant.fields:
                distinct = {
                    tuple(fields[field]) for fields in observed.values()
                    if field in fields
                }
                if len(distinct) > 1:
                    problems.append(f"arms disagree on {field}: {sorted(distinct)}")
        out.append({
            "name": invariant.name,
            "arms": list(invariant.arms),
            "fields": list(invariant.fields),
            "observed": {arm: {f: v for f, v in fields.items()}
                         for arm, fields in observed.items()},
            "holds": not problems,
            "problems": problems,
        })
    return out


def summarize_group(
    group: str, rows: Sequence[Mapping[str, Any]], scenarios: Sequence[str],
) -> Dict[str, Any]:
    """Arm means, every declared contrast, and the group's invariants."""
    metrics = group_metrics(group)
    by_cell: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row["scenario"]), str(row["arm"]))].append(row)
    declared = arms_by_name()
    means = [
        {
            "scenario": scenario, "arm": arm, "n_seeds": len(group_rows),
            **{field: group_rows[0].get(field) for field in STATE_FIELDS},
            **(dict(declared[arm].facets) if arm in declared else {}),
            **{f"mean_{name}": _finite_mean(r.get(name) for r in group_rows)
               for name in (*(m for m, _d in metrics), *COST_FIELDS)},
        }
        for (scenario, arm), group_rows in sorted(by_cell.items())
    ]

    index = index_rows(rows)
    contrasts: List[Dict[str, Any]] = []
    for contrast in group_contrasts(group):
        for scenario in scenarios:
            entry = paired_deltas(rows, contrast, metrics, scenario, index)
            if not entry["metrics"]:
                continue
            contrasts.append({
                "name": contrast.name,
                "treatment": contrast.treatment,
                "reference": contrast.reference,
                "interpretation": contrast.interpretation,
                "primary": contrast.primary,
                "descriptive": contrast.descriptive,
                **entry,
            })
    # One family per group: the primary contrast carries the claim, the rest
    # are corrected together so a group's secondary table cannot be mined.
    secondary = [
        (entry, metric)
        for entry in contrasts if not entry["primary"]
        for metric in entry["metrics"]
    ]
    adjusted = _holm([
        entry["metrics"][metric]["sign_flip_p_two_sided"]
        for entry, metric in secondary
    ])
    for (entry, metric), value in zip(secondary, adjusted):
        entry["metrics"][metric]["sign_flip_p_two_sided_holm"] = value
    return {
        "group": group,
        "definition": GROUP_NOTES[group],
        "primary_contrast": next(
            c.name for c in group_contrasts(group) if c.primary
        ),
        "secondary_test_count": len(secondary),
        "multiplicity": (
            "The primary contrast carries the group's claim. Secondary "
            "contrasts share one Holm-Bonferroni family reported as "
            "sign_flip_p_two_sided_holm; contrasts marked descriptive are "
            "not component-matched and are not evidence for their name."
        ),
        "arms": [
            {"name": name,
             "label": declared[name].label if name in declared else name,
             "source": "ablation" if name in declared else "deployable roster",
             **(dict(declared[name].facets) if name in declared else {})}
            for name in group_arms(group)
        ],
        "mean_by_scenario_arm": means,
        "contrasts": contrasts,
        "invariants": check_invariants(group, rows),
    }


def run_ablations(
    selected: Sequence[str],
    config: Any,
    *,
    workers: int = DEFAULT_ABLATION_WORKERS,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run the selected groups and report them. Returns the full report."""
    chosen = tuple(dict.fromkeys(selected))
    unknown = sorted(set(chosen) - set(groups()))
    if unknown:
        raise ValueError(f"unknown ablation groups: {unknown}")
    run_dir, execution = run_arms(
        chosen, config, workers=workers, progress=progress,
    )
    scenarios = tuple(str(s) for s in config.scenarios)
    seeds = tuple(int(s) for s in config.seeds)
    rows = read_rows(run_dir, chosen, scenarios, seeds)
    report = {
        "design": (
            "An ablation is an arm plus a contrast. Arms are run by the "
            "evaluation's own per-cell runner into its run directory, so an "
            "arm shared by several groups -- or with the deployable roster "
            "-- is computed once and read by all of them."
        ),
        "execution": execution,
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "groups": {
            group: summarize_group(group, rows[group], scenarios)
            for group in chosen
        },
    }
    report["complete"] = all(
        len(rows[group]) == len(group_arms(group)) * len(scenarios) * len(seeds)
        for group in chosen
    )
    report["invariants_hold"] = all(
        check["holds"]
        for group in chosen
        for check in report["groups"][group]["invariants"]
    )
    return report


# ==========================================================================
# SECTION 3 -- COMMAND LINE
# ==========================================================================
def _parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablations. '--groups matcher' runs the offline matcher stress "
            "suite; every other group is a set of contrasts over arms that "
            "are contributed to the standard evaluation run."
        ),
    )
    parser.add_argument(
        "--groups", default="all",
        help="Comma-separated: all, matcher, " + ", ".join(groups()),
    )
    parser.add_argument(
        "--run",
        help="Contribute arms to this run instead of the newest complete one.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_ABLATION_WORKERS)
    parser.add_argument("--output", help="Write the report here instead of stdout.")
    parser.add_argument("--results-root", default=DEFAULT_ABLATION_RESULTS_ROOT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    # Matcher-only knobs.
    parser.add_argument("--seed", type=int, default=MatcherSettings.seed)
    parser.add_argument("--recipes", type=int, default=MatcherSettings.recipe_count)
    parser.add_argument("--case-limit", type=int, default=MatcherSettings.case_limit)
    parser.add_argument("--no-prefixes", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    return parser.parse_args(argv)


def _emit(payload: Mapping[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(payload, indent=args.indent, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if not args.quiet:
        print(text, end="")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    selected = _parse_csv(args.groups)
    if selected == ("all",):
        selected = ("matcher", *groups())

    report: Dict[str, Any] = {}
    longitudinal = tuple(g for g in selected if g != "matcher")
    if "matcher" in selected:
        report["matcher"] = run_matcher_ablation(MatcherSettings(
            seed=int(args.seed),
            recipe_count=int(args.recipes),
            case_limit=int(args.case_limit),
            include_prefixes=not bool(args.no_prefixes),
            calibrate=not bool(args.no_calibration),
        ))
    if longitudinal:
        # The grid is the run's; only the arm selection is ours.
        _run_dir, config = find_run(args.results_root, args.run)
        config = replace(
            config,
            baselines=("full",),
            include_oracle=False,
            show_eta=False,
            experiment="ablation",
            workers=max(0, int(args.workers)),
        )
        report.update(run_ablations(
            longitudinal, config,
            workers=int(args.workers), progress=not args.quiet,
        ))
    _emit(report, args)


if __name__ == "__main__":
    main()
