"""Variant default, adaptive rehearsal, pruning, and reentry."""
from __future__ import annotations
import contextlib
import hashlib
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple, TypeAlias
from .models import Settings, DEFAULT_SETTINGS
VariantKey: TypeAlias = Tuple[str, str]
StateTransition: TypeAlias = Tuple[Tuple[int, ...], str, Tuple[int, ...]]

@dataclass
class MatchResult:
    kind: str                  # known | preference_shift | new_recipe
    recipe_id: Optional[str]   # None until a new recipe is registered
    variant_id: Optional[str]
    jaccard: float
    order_distance: float

def _jaccard_counters(first_counts: Counter, second_counts: Counter) -> float:
    intersection = sum((first_counts & second_counts).values())
    union = sum((first_counts | second_counts).values())
    return intersection / union if union > 0 else 0.0

def jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    """Multiset overlap over complete semantic actions."""
    return _jaccard_counters(Counter(first), Counter(second))

def kendall_tau_distance(
    first: Sequence[str],
    second: Sequence[str],
    unmatched_penalty: float = 0.5,
) -> float:
    """Return normalized Kendall tau with a bounded unmatched-action penalty."""
    def indexed_tokens(sequence: Sequence[str]) -> List[Tuple[str, int]]:
        counts: Counter = Counter()
        indexed: List[Tuple[str, int]] = []
        for token in sequence:
            counts[token] += 1
            indexed.append((token, int(counts[token])))
        return indexed

    first_tokens = indexed_tokens(first)
    second_tokens = indexed_tokens(second)
    second_set = set(second_tokens)
    common_tokens = [token for token in first_tokens if token in second_set]
    second_positions = {
        token: index for index, token in enumerate(second_tokens)
    }
    first_counts = Counter(first)
    second_counts = Counter(second)
    unmatched = sum((first_counts - second_counts).values()) + sum(
        (second_counts - first_counts).values()
    )
    denominator = max(len(first), len(second), 1)
    penalty = max(0.0, float(unmatched_penalty)) * unmatched / denominator
    if len(common_tokens) < 2:
        return min(1.0, penalty)
    count = len(common_tokens)
    inversions = 0
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            if (
                second_positions[common_tokens[first_index]]
                > second_positions[common_tokens[second_index]]
            ):
                inversions += 1
    tau = (
        inversions / (count * (count - 1) / 2)
        if count > 1 else 0.0
    )
    return min(1.0, tau + penalty)

@lru_cache(maxsize=65536)
def _lcs_aligned_subsequence_cached(prefix: Tuple[str, ...], ordering: Tuple[str, ...]) -> Tuple[str, ...]:
    prefix_length = len(prefix)
    ordering_length = len(ordering)
    if prefix_length <= 0 or ordering_length <= 0:
        return ()
    lengths = [
        [0] * (ordering_length + 1)
        for _ in range(prefix_length + 1)
    ]
    for prefix_index in range(1, prefix_length + 1):
        for ordering_index in range(1, ordering_length + 1):
            if prefix[prefix_index - 1] == ordering[ordering_index - 1]:
                lengths[prefix_index][ordering_index] = (
                    lengths[prefix_index - 1][ordering_index - 1] + 1
                )
            else:
                lengths[prefix_index][ordering_index] = max(
                    lengths[prefix_index - 1][ordering_index],
                    lengths[prefix_index][ordering_index - 1],
                )
    aligned: List[str] = []
    prefix_index = prefix_length
    ordering_index = ordering_length
    while prefix_index > 0 and ordering_index > 0:
        if prefix[prefix_index - 1] == ordering[ordering_index - 1]:
            aligned.append(ordering[ordering_index - 1])
            prefix_index -= 1
            ordering_index -= 1
        elif (
            lengths[prefix_index - 1][ordering_index]
            >= lengths[prefix_index][ordering_index - 1]
        ):
            prefix_index -= 1
        else:
            ordering_index -= 1
    aligned.reverse()
    return tuple(aligned)

def _lcs_aligned_subsequence(prefix: Sequence[str], ordering: Sequence[str]) -> List[str]:
    """Return the best ordering-side subsequence aligned to `prefix` by LCS."""
    return list(_lcs_aligned_subsequence_cached(tuple(prefix), tuple(ordering)))

def make_variant_id(sequence: Sequence[str]) -> str:
    """Stable order-preserving hash for a demonstrated recipe-preference variant."""
    parts = [f"{i}:{a}" for i, a in enumerate(sequence)]
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class KnownVariant:
    recipe_id: str
    variant_id: str
    ordering: Tuple[str, ...]
    # Lazy classification caches excluded from equality.
    _counter_full: Optional[Counter] = field(default=None, repr=False, compare=False)
    def counts(self) -> Counter:
        if self._counter_full is None: self._counter_full = Counter(self.ordering)
        return self._counter_full


class RecipeMatcher:
    """Classifies new demonstrations against a library of known variants."""
    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        # Diagnostic profile: event -> (n_calls, total_wall_s). Populated only when settings.profile is True. Off path is a single bool check per call.
        self.profile: Dict[str, Tuple[int, float]] = {}
        self.variants_scored: int = 0

    @contextlib.contextmanager
    def _profile(self, event: str):
        if not self.settings.profile:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            calls, wall_seconds = self.profile.get(event, (0, 0.0))
            self.profile[event] = (
                calls + 1,
                wall_seconds + (time.perf_counter() - start_time),
            )

    def classify(
        self,
        sequence: Sequence[str],
        library: List[KnownVariant],
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Classify a sequence of canonical semantic actions."""
        match_threshold = float(
            self.settings.match_threshold if threshold is None else threshold
        )
        with self._profile("classify"):
            if not library: return MatchResult("new_recipe", None, None, 0.0, 0.0)
            actions = tuple(sequence)
            variant_id = make_variant_id(actions)
            for variant in library:
                if (
                    variant.variant_id == variant_id
                    and variant.ordering == actions
                ):
                    return MatchResult(
                        "known", variant.recipe_id, variant.variant_id,
                        1.0, 0.0,
                    )
            best_per_recipe = self.score(actions, library)
            best_recipe_id, (
                _best_variant,
                best_score,
                best_distance,
            ) = max(
                best_per_recipe.items(),
                key=lambda item: (item[1][1], -item[1][2]),
            )
            runner_up = max(
                (
                    score
                    for recipe_id, (_variant, score, _distance)
                    in best_per_recipe.items()
                    if recipe_id != best_recipe_id
                ),
                default=None,
            )
            margin = (
                float("inf")
                if runner_up is None else best_score - runner_up
            )
            if best_score < match_threshold or (
                runner_up is not None and margin < self.settings.match_margin
            ):
                return MatchResult(
                    "new_recipe", None, None, best_score, best_distance,
                )
            return MatchResult(
                "preference_shift", best_recipe_id, None,
                best_score, best_distance,
            )

    def match_known(
        self,
        sequence: Sequence[str],
        library: Sequence[KnownVariant],
    ) -> MatchResult:
        """Return the best known recipe without applying open-set rejection.

        Assist mode is an externally declared known-task protocol. Low evidence is
        handled by the commit gate, not by allocating or requesting a new recipe.
        """
        if not library:
            return MatchResult("assist_unavailable", None, None, 0.0, 0.0)
        actions = tuple(sequence)
        variant_id = make_variant_id(actions)
        for candidate in library:
            if (
                candidate.variant_id == variant_id
                and candidate.ordering == actions
            ):
                return MatchResult("known", candidate.recipe_id, candidate.variant_id, 1.0, 0.0)
        best_per_recipe = self.score(actions, library)
        best_recipe_id, (best_variant, best_score, best_distance) = max(
            best_per_recipe.items(),
            key=lambda item: (item[1][1], -item[1][2]),
        )
        return MatchResult(
            "preference_shift",
            best_recipe_id,
            best_variant.variant_id,
            best_score,
            best_distance,
        )

    def score(
        self,
        sequence: Sequence[str],
        library: Sequence[KnownVariant],
    ) -> Dict[str, Tuple[KnownVariant, float, float]]:
        """Score a library in one pass and retain its best variant per recipe."""
        with self._profile("score"):
            self.variants_scored += len(library)
            actions = tuple(sequence)
            variant_id = make_variant_id(actions)
            action_counts = Counter(actions)
            best: Dict[str, Tuple[KnownVariant, float, float]] = {}
            for variant in library:
                if (
                    variant.variant_id == variant_id
                    and variant.ordering == actions
                ):
                    score, distance = 1.0, 0.0
                else:
                    score = _jaccard_counters(
                        action_counts, variant.counts(),
                    )
                    distance = kendall_tau_distance(
                        actions, variant.ordering,
                        unmatched_penalty=self.settings.unmatched_penalty,
                    )
                previous = best.get(variant.recipe_id)
                if previous is None or score > previous[1]:
                    best[variant.recipe_id] = (variant, float(score), float(distance))
            return best

    def score_prefix(self, prefix: Sequence[str], variants: List[KnownVariant]) -> List[Tuple[KnownVariant, float]]:
        """Rank variants by prefix overlap and shared-action order."""
        prefix_counter = Counter(prefix)
        ranked = []
        unmatched_penalty = self.settings.unmatched_penalty
        for variant in variants:
            ordering = variant.ordering
            if not prefix or not ordering:
                ranked.append((variant, 0.0))
                continue
            aligned = _lcs_aligned_subsequence(prefix, ordering)
            if not aligned:
                ranked.append((variant, 0.0))
                continue
            variant_counter = Counter(aligned)
            intersection = sum((prefix_counter & variant_counter).values())
            union = sum((prefix_counter | variant_counter).values())
            set_overlap = intersection / max(union, 1)
            tau = 1.0 - kendall_tau_distance(prefix, aligned, unmatched_penalty=unmatched_penalty)
            ranked.append((
                variant,
                self.settings.overlap_weight * set_overlap
                + self.settings.order_weight * tau,
            ))
        ranked.sort(key=lambda item: -item[1])
        return ranked


@dataclass
class Variant:
    recipe_id: str
    variant_id: str
    ordering: Tuple[str, ...]
    first_seen_step: int
    last_seen_step: int


class VariantLibrary:
    """Stores demonstrated preference variants without contributing a prior."""
    def __init__(self):
        self.variants: Dict[str, Dict[str, Variant]] = defaultdict(dict)    # recipe_id -> insertion-ordered dict of variant_id -> Variant
        self.latest: Dict[str, str] = {}                                    # recipe_id -> variant_id of most-recently-observed variant.

    def register(
        self,
        recipe_id: str,
        ordering: Sequence[str],
        step: int,
    ) -> Variant:
        variant_id = make_variant_id(ordering)
        slot = self.variants[recipe_id]
        if variant_id in slot:
            variant = slot[variant_id]
            variant.last_seen_step = step
            variant.ordering = tuple(ordering)
        else:
            variant = Variant(recipe_id=recipe_id, variant_id=variant_id, ordering=tuple(ordering), first_seen_step=step, last_seen_step=step)
            slot[variant_id] = variant
        self.latest[recipe_id] = variant_id
        return variant

    def promote_latest(self, recipe_id: str, variant_id: str, step: int) -> None:
        if variant_id in self.variants.get(recipe_id, {}):
            self.latest[recipe_id] = variant_id
            self.variants[recipe_id][variant_id].last_seen_step = step

    def known_variants(self, allowed_keys: Optional[Set[VariantKey]] = None) -> List[KnownVariant]:
        return [
            KnownVariant(recipe_id, variant_id, variant.ordering)
            for recipe_id, slot in self.variants.items()
            for variant_id, variant in slot.items()
            if allowed_keys is None or (recipe_id, variant_id) in allowed_keys
        ]

    def latest_variant(self, recipe_id: str, allowed_keys: Optional[Set[VariantKey]] = None) -> Optional[Variant]:
        variant_id = self.latest.get(recipe_id)
        slot = self.variants.get(recipe_id, {})
        if variant_id and variant_id in slot and (allowed_keys is None or (recipe_id, variant_id) in allowed_keys): return slot[variant_id]
        if allowed_keys is None: return None
        candidates = [
            variant for variant in slot.values()
            if (variant.recipe_id, variant.variant_id) in allowed_keys
        ]
        return (
            max(candidates, key=lambda variant: variant.last_seen_step)
            if candidates else None
        )

def clear_caches() -> None:
    """Clear module-level caches that can otherwise bleed across seed jobs."""
    _lcs_aligned_subsequence_cached.cache_clear()


@dataclass
class MemoryItem:
    recipe_id: str
    variant_id: str
    ordering: Tuple[str, ...]
    weight: float
    added_step: int
    added_cycle: int
    last_seen_step: int
    source_mode: str = ""
    transitions: Tuple[StateTransition, ...] = ()

    @property
    def key(self) -> VariantKey:    return (self.recipe_id, self.variant_id)

@dataclass
class RemovedItem:
    recipe_id: str
    variant_id: str
    ordering: Tuple[str, ...]
    added_step: int
    removed_step: int
    added_cycle: int
    removed_cycle: int
    last_seen_step: int
    transitions: Tuple[StateTransition, ...] = ()

    @property
    def key(self) -> VariantKey:    return (self.recipe_id, self.variant_id)


class ReplayMemory:
    """Track replay weights and robust, hierarchically pooled retention horizons."""
    def __init__(self, settings: Settings = DEFAULT_SETTINGS, policy: str = "adaptive"):
        if policy not in {"adaptive", "fixed", "none"}:
            raise ValueError(f"unknown decay policy {policy!r}")
        self.settings = settings
        self.policy = policy
        self.default_grace_horizon: int = max(0, int(getattr(settings, "initial_grace", 50)))
        self.min_grace: int = max(0, int(getattr(settings, "min_grace", 6)))
        self.prune_delay: int = max(1, int(getattr(settings, "prune_delay", 3)))
        self.reuse_window: int = max(1, int(getattr(settings, "pair_gap_window", 12)))
        self.reuse_min_samples: int = min(
            self.reuse_window,
            max(1, int(getattr(settings, "parent_weight_samples", 5))),
        )
        self.pair_downward_half_life: float = float(
            getattr(settings, "pair_prior_half_life", 3.0),
        )
        if not math.isfinite(self.pair_downward_half_life) or self.pair_downward_half_life <= 0.0:
            raise ValueError("pair_prior_half_life must be finite and positive")
        self.reuse_quantile: float = min(
            1.0, max(0.0, float(getattr(settings, "gap_quantile", 0.90))),
        )
        self.reuse_iqr_multiplier: float = max(
            0.0, float(getattr(settings, "gap_iqr_scale", 1.50)),
        )
        self.recipe_reuse_window: int = max(
            self.reuse_window,
            int(getattr(settings, "recipe_gap_window", 24)),
        )
        self.global_reuse_window: int = max(
            self.recipe_reuse_window,
            int(getattr(settings, "global_gap_window", 60)),
        )
        self.post_grace_decay_rate = (
            0.0 if policy == "none"
            else float(settings.fixed_decay) if policy == "fixed"
            else 1.0 / float(self.prune_delay)
        )
        self.active: Dict[VariantKey, MemoryItem] = {}
        self.pruned: Dict[VariantKey, RemovedItem] = {}
        self.latest_by_recipe: Dict[str, str] = {}
        self.latest_keys: Set[VariantKey] = set()
        # Diagnostic trace; adaptation uses an exact-pair window.
        self._reuse_gap_window: Deque[int] = deque(maxlen=settings.diagnostic_gap_window)
        self._pair_gap_window: Dict[VariantKey, Deque[int]] = {}
        self._pair_last_seen_step: Dict[VariantKey, int] = {}
        # Bounded disjoint events supply recipe and global priors.
        self._recipe_gap_events: Dict[str, Deque[Tuple[VariantKey, int]]] = {}
        self._global_gap_events: Deque[Tuple[VariantKey, int]] = deque(
            maxlen=self.global_reuse_window,
        )
        self.reuse_gap_events: List[Tuple[int, VariantKey, int]] = []  # positive reuse gaps
        self.reentry_events: List[Tuple[int, VariantKey, int]] = []

    def register(
        self,
        recipe_id: str,
        variant_id: str,
        ordering: Tuple[str, ...],
        now: int,
        cycle: int,
        *,
        weight: float = 1.0,
        pin_latest: Optional[bool] = None,
        transitions: Optional[Sequence[StateTransition]] = None,
    ) -> MemoryItem:
        """Insert, reset an active variant, or restore a pruned variant."""
        key = (recipe_id, variant_id)
        should_pin_latest = self.settings.pin_latest if pin_latest is None else bool(pin_latest)
        initial_weight = max(0.0, min(1.0, float(weight)))
        transition_trace = tuple(transitions or ())
        reuse_gap = self._record_pair_reuse(key, now, step=now)
        if key in self.active:
            entry = self.active[key]
            entry.weight = (
                1.0 if key in self.latest_keys else initial_weight
            )
            entry.last_seen_step = now
            entry.ordering = ordering
            if transitions is not None:
                entry.transitions = transition_trace
            if should_pin_latest: self.mark_latest(recipe_id, variant_id)
            return entry

        reentering = key in self.pruned
        if reentering:
            removed_item = self.pruned.pop(key)
            if transitions is None:
                transition_trace = tuple(removed_item.transitions)

        entry = MemoryItem(recipe_id=recipe_id, variant_id=variant_id, ordering=ordering, weight=initial_weight, added_step=now, added_cycle=cycle, last_seen_step=now, transitions=transition_trace)
        self.active[key] = entry
        if reentering:
            self.reentry_events.append((int(now), key, int(reuse_gap or 0)))
        if should_pin_latest: self.mark_latest(recipe_id, variant_id)
        return entry

    def step(self, now: int, cycle: int, protected_keys: Optional[Sequence[VariantKey]] = None) -> List[VariantKey]:
        """Advance one demonstration, decay overdue entries, and prune expirations."""
        if self.policy == "none":
            return []
        protected = set(protected_keys or ())
        decay_rate = self.post_grace_decay_rate
        pruned: List[VariantKey] = []
        for key, entry in list(self.active.items()):
            if key in self.latest_keys or key in protected:
                entry.weight = 1.0
                continue
            age = int(now) - int(entry.last_seen_step)
            if self.policy == "adaptive" and age <= self.horizon(key):
                continue
            entry.weight -= decay_rate
            if entry.weight <= self.settings.prune_threshold:
                self._prune_entry(key, entry, now, cycle)
                pruned.append(key)
        return pruned

    def weights(self) -> Dict[VariantKey, float]:
        return {key: entry.weight for key, entry in self.active.items()}

    def active_items(self) -> List[MemoryItem]:
        return list(self.active.values())

    def recipe_items(self, recipe_id: str) -> List[MemoryItem]:
        return [
            entry for entry in self.active.values()
            if entry.recipe_id == recipe_id
        ]
    def _robust_upper_gap(self, gaps: Sequence[int]) -> float:
        ordered = sorted(float(gap) for gap in gaps)
        if not ordered:
            return float(self.default_grace_horizon)
        lower_quartile = self._linear_quantile(ordered, 0.25)
        upper_quartile = self._linear_quantile(ordered, 0.75)
        upper_quantile = self._linear_quantile(ordered, self.reuse_quantile)
        tukey_upper_fence = upper_quartile + self.reuse_iqr_multiplier * (
            upper_quartile - lower_quartile
        )
        return min(upper_quantile, tukey_upper_fence)

    def _linear_evidence_weight(self, sample_count: int) -> float:
        return min(1.0, int(sample_count) / float(self.reuse_min_samples))

    def _pool_parent_prior(
        self,
        prior: float,
        gaps: Sequence[int],
    ) -> Tuple[float, float]:
        """Let recipe/global evidence raise, but never shorten, its parent prior."""
        if not gaps:
            return float(prior), 0.0
        evidence_weight = self._linear_evidence_weight(len(gaps))
        estimate = self._robust_upper_gap(gaps)
        pooled = (
            (1.0 - evidence_weight) * float(prior)
            + evidence_weight * estimate
        )
        return max(float(prior), pooled), evidence_weight

    def _pool_exact_pair(
        self,
        prior: float,
        gaps: Sequence[int],
    ) -> Tuple[float, float, str]:
        """Adapt down asymptotically and preserve the fast linear upward response."""
        if not gaps:
            return float(prior), 0.0, "prior_only"
        estimate = self._robust_upper_gap(gaps)
        if estimate < float(prior):
            evidence_weight = 1.0 - math.pow(
                2.0,
                -len(gaps) / self.pair_downward_half_life,
            )
            pooling_mode = "exponential_downward"
        else:
            evidence_weight = self._linear_evidence_weight(len(gaps))
            pooling_mode = "linear_upward"
        pooled = (
            (1.0 - evidence_weight) * float(prior)
            + evidence_weight * estimate
        )
        return pooled, evidence_weight, pooling_mode

    def horizon_stats(self, key: VariantKey) -> Dict[str, Any]:
        """Return the pair horizon and its disjoint hierarchical evidence."""
        recipe_id = key[0]
        global_gaps = [
            gap for event_key, gap in self._global_gap_events
            if event_key[0] != recipe_id
        ]
        recipe_gaps = [
            gap for event_key, gap in self._recipe_gap_events.get(recipe_id, ())
            if event_key != key
        ]
        pair_gaps = list(self._pair_gap_window.get(key, ()))

        global_estimate, global_weight = self._pool_parent_prior(
            float(self.default_grace_horizon), global_gaps,
        )
        recipe_estimate, recipe_weight = self._pool_parent_prior(
            global_estimate, recipe_gaps,
        )
        pair_estimate, pair_weight, pair_pooling_mode = self._pool_exact_pair(
            recipe_estimate, pair_gaps,
        )
        horizon = max(
            float(math.ceil(pair_estimate)),
            float(self.min_grace),
        )
        return {
            "horizon_demos": horizon,
            "pair_gap_samples": int(len(pair_gaps)),
            "recipe_prior_gap_samples": int(len(recipe_gaps)),
            "global_prior_gap_samples": int(len(global_gaps)),
            "pair_evidence_weight": float(pair_weight),
            "pair_pooling_mode": pair_pooling_mode,
            "pair_downward_half_life_samples": float(self.pair_downward_half_life),
            "recipe_evidence_weight": float(recipe_weight),
            "global_evidence_weight": float(global_weight),
            "pair_robust_upper_demos": (
                float(self._robust_upper_gap(pair_gaps)) if pair_gaps else 0.0
            ),
            "recipe_prior_horizon_demos": float(recipe_estimate),
            "global_prior_horizon_demos": float(global_estimate),
        }

    def horizon(self, key: VariantKey) -> float:
        return self.horizon_stats(key)["horizon_demos"]

    def horizons(self) -> Dict[str, float]:
        keys = set(self.active) | set(self.pruned) | set(self._pair_gap_window)
        return {
            f"{recipe_id}/{variant_id}": self.horizon((recipe_id, variant_id))
            for recipe_id, variant_id in sorted(keys)
        }

    def horizon_evidence(self) -> Dict[str, Dict[str, Any]]:
        keys = set(self.active) | set(self.pruned) | set(self._pair_gap_window)
        return {
            f"{recipe_id}/{variant_id}": self.horizon_stats(
                (recipe_id, variant_id)
            )
            for recipe_id, variant_id in sorted(keys)
        }

    def mark_latest(self, recipe_id: str, variant_id: str) -> None:
        """Pin the latest variant and release the recipe's previous pin."""
        previous_variant_id = self.latest_by_recipe.get(recipe_id)
        if previous_variant_id is not None:
            previous_key = (recipe_id, previous_variant_id)
            self.latest_keys.discard(previous_key)
        self.latest_by_recipe[recipe_id] = variant_id
        key = (recipe_id, variant_id)
        self.latest_keys.add(key)
        if key in self.active: self.active[key].weight = 1.0

    def unmark_latest(self, recipe_id: str, variant_id: str) -> None:
        key = (recipe_id, variant_id)
        if self.latest_by_recipe.get(recipe_id) == variant_id: del self.latest_by_recipe[recipe_id]
        self.latest_keys.discard(key)

    def discard(self, recipe_id: str, variant_id: str, *, allow_latest: bool = False) -> None:
        """Remove an active or pruned variant without logging a prune event."""
        key = (recipe_id, variant_id)
        if key in self.latest_keys and not allow_latest: raise RuntimeError(f"discard() called on latest-pinned variant {key}. Use allow_latest=True only in controlled replacement paths.")
        was_latest = self.latest_by_recipe.get(recipe_id) == variant_id
        self.active.pop(key, None)
        self.pruned.pop(key, None)
        self._pair_gap_window.pop(key, None)
        self._pair_last_seen_step.pop(key, None)
        self.latest_keys.discard(key)
        if was_latest:
            self.latest_by_recipe.pop(recipe_id, None)
            replacement = max(
                self.recipe_items(recipe_id),
                key=lambda entry: (entry.last_seen_step, entry.variant_id),
                default=None,
            )
            if replacement is not None:
                self.mark_latest(recipe_id, replacement.variant_id)

    def gap_history(self) -> List[int]:
        """Return the bounded diagnostic trace of exact-pair reuse gaps."""
        return list(self._reuse_gap_window)

    def pair_history(self, key: VariantKey) -> List[int]:
        """Return the recurrence samples currently controlling one variant."""
        return list(self._pair_gap_window.get(key, ()))

    def _record_pair_reuse(self, key: VariantKey, now: int, step: Optional[int] = None) -> Optional[int]:
        last_seen = self._pair_last_seen_step.get(key)
        gap: Optional[int] = None
        if last_seen is not None:
            gap = int(now) - int(last_seen)
            self._record_reuse_gap(key, gap, step=step)
        self._pair_last_seen_step[key] = int(now)
        return gap if gap is not None and gap > 0 else None

    def _record_reuse_gap(self, key: VariantKey, gap: int, step: Optional[int] = None) -> None:
        """Record a positive exact-pair gap in that variant's rolling window."""
        if gap <= 0: return
        gap_i = int(gap)
        if key not in self._pair_gap_window:
            self._pair_gap_window[key] = deque(maxlen=self.reuse_window)
        self._pair_gap_window[key].append(gap_i)
        recipe_id = key[0]
        if recipe_id not in self._recipe_gap_events:
            self._recipe_gap_events[recipe_id] = deque(
                maxlen=self.recipe_reuse_window,
            )
        event = (key, gap_i)
        self._recipe_gap_events[recipe_id].append(event)
        self._global_gap_events.append(event)
        self.reuse_gap_events.append((step if step is not None else 0, key, gap_i))
        self._reuse_gap_window.append(gap_i)

    @staticmethod
    def _linear_quantile(ordered: Sequence[float], quantile: float) -> float:
        """Linearly interpolated quantile for an already-sorted sample."""
        if not ordered:
            raise ValueError("cannot estimate a quantile from an empty sample")
        if len(ordered) == 1:
            return float(ordered[0])
        position = min(1.0, max(0.0, float(quantile))) * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(ordered[lower])
        fraction = position - lower
        return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))

    def _prune_entry(self, key: VariantKey, entry: MemoryItem, now: int, cycle: int) -> None:
        self.latest_keys.discard(key)
        if self.latest_by_recipe.get(entry.recipe_id) == entry.variant_id: del self.latest_by_recipe[entry.recipe_id]
        self.pruned[key] = RemovedItem(recipe_id=entry.recipe_id, variant_id=entry.variant_id, ordering=entry.ordering, added_step=entry.added_step, removed_step=now, added_cycle=entry.added_cycle, removed_cycle=cycle, last_seen_step=entry.last_seen_step, transitions=tuple(entry.transitions))
        del self.active[key]
