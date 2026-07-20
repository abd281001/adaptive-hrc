"""Memory, novelty, and adaptive rehearsal state. This module groups the non-parametric memory subsystem: variant identity, sequence disambiguation, latest-preference bookkeeping, adaptive replay weights, pruning, pruned-variant metadata, and reentry. Keeping these together makes the memory contract visible to the agent and evaluation harness."""
from __future__ import annotations
import contextlib
import hashlib
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple, TypeAlias
from .models import Config, DEFAULT_CONFIG
# Sequence disambiguation
VariantKey: TypeAlias = Tuple[str, str]
DemoTransition: TypeAlias = Tuple[Tuple[int, ...], str, Tuple[int, ...]]

@dataclass
class Classification:
    kind: str                  # "known" | "preference_shift" | "new_recipe"
    recipe_id: Optional[str]   # disambiguator returns None for new_recipe; agent fills an internal ID after registration
    variant_hash: Optional[str]
    jaccard: float
    order_distance: float

def _action_type(action: str) -> str:
    return action.split("(")[0].strip()

def _jaccard_counters(ca: Counter, cb: Counter) -> float:
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union > 0 else 0.0

def jaccard(a: Sequence[str], b: Sequence[str], full_weight: float = 0.70, type_weight: float = 0.30) -> float:
    """Weighted multiset overlap over full actions and action types."""
    ca_full = Counter(a)
    cb_full = Counter(b)
    ca_type = Counter(_action_type(x) for x in a)
    cb_type = Counter(_action_type(x) for x in b)
    return float(full_weight) * _jaccard_counters(ca_full, cb_full) + float(type_weight) * _jaccard_counters(ca_type, cb_type)

def _jaccard_from_cached_counters(ca_full: Counter, ca_type: Counter, cb_full: Counter, cb_type: Counter, full_weight: float = 0.70, type_weight: float = 0.30) -> float:
    """Same as `jaccard` when both sides already have cached counters."""
    return float(full_weight) * _jaccard_counters(ca_full, cb_full) + float(type_weight) * _jaccard_counters(ca_type, cb_type)

def kendall_tau_distance(a: Sequence[str], b: Sequence[str], unmatched_penalty: float = 0.5) -> float:
    """Normalized Kendall tau over actions common to both sequences. Returns 0.0 for identical ordering and 1.0 for fully inverted ordering. Insertions/deletions add a bounded unmatched-action penalty so missing actions cannot masquerade as a clean ordering match."""
    def indexed_tokens(seq: Sequence[str]) -> List[Tuple[str, int]]:
        counts: Counter = Counter()
        out: List[Tuple[str, int]] = []
        for tok in seq:
            counts[tok] += 1
            out.append((tok, int(counts[tok])))
        return out
    ia = indexed_tokens(a)
    ib = indexed_tokens(b)
    b_set = set(ib)
    common_a = [tok for tok in ia if tok in b_set]
    pos_b = {tok: i for i, tok in enumerate(ib)}
    ca = Counter(a)
    cb = Counter(b)
    unmatched = sum((ca - cb).values()) + sum((cb - ca).values())
    denom = max(len(a), len(b), 1)
    penalty = max(0.0, float(unmatched_penalty)) * unmatched / denom
    if len(common_a) < 2: return min(1.0, penalty)
    n = len(common_a)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if common_a[i] in pos_b and common_a[j] in pos_b:
                if pos_b[common_a[i]] > pos_b[common_a[j]]: inv += 1
    tau = inv / (n * (n - 1) / 2) if n > 1 else 0.0
    return min(1.0, tau + penalty)

@lru_cache(maxsize=65536)
def _lcs_aligned_subsequence_cached(prefix: Tuple[str, ...], ordering: Tuple[str, ...]) -> Tuple[str, ...]:
    n_p, n_o = len(prefix), len(ordering)
    if n_p <= 0 or n_o <= 0:
        return ()
    dp = [[0] * (n_o + 1) for _ in range(n_p + 1)]
    for i in range(1, n_p + 1):
        for j in range(1, n_o + 1):
            if prefix[i - 1] == ordering[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    out: List[str] = []
    i, j = n_p, n_o
    while i > 0 and j > 0:
        if prefix[i - 1] == ordering[j - 1]:
            out.append(ordering[j - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return tuple(out)

def _lcs_aligned_subsequence(prefix: Sequence[str], ordering: Sequence[str]) -> List[str]:
    """Return the best ordering-side subsequence aligned to `prefix` by LCS."""
    return list(_lcs_aligned_subsequence_cached(tuple(prefix), tuple(ordering)))

def variant_hash(sequence: Sequence[str]) -> str:
    """Stable order-preserving hash for a demonstrated recipe-preference variant."""
    parts = [f"{i}:{a}" for i, a in enumerate(sequence)]
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class KnownVariant:
    recipe_id: str
    variant_hash: str
    ordering: Tuple[str, ...]
    # Context-normalized tokens used only to identify a recipe.  ``ordering``
    # remains the raw anonymous transition sequence used by replay/prediction.
    identity_ordering: Tuple[str, ...] = ()
    # Cached counters for fast classify(). They are computed on first access, stable for the lifetime of the instance, and excluded from equality.
    _counter_full: Optional[Counter] = field(default=None, repr=False, compare=False)
    _counter_type: Optional[Counter] = field(default=None, repr=False, compare=False)
    def counter_full(self) -> Counter:
        if self._counter_full is None: self._counter_full = Counter(self.ordering)
        return self._counter_full
    def counter_type(self) -> Counter:
        if self._counter_type is None: self._counter_type = Counter(_action_type(x) for x in self.ordering)
        return self._counter_type


class Disambiguator:
    """Classifies new demonstrations against a library of known variants."""
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        # Diagnostic profile: event -> (n_calls, total_wall_s). Populated only when cfg.profile is True. Off path is a single bool check per call.
        self.profile: Dict[str, Tuple[int, float]] = {}

    @contextlib.contextmanager
    def _profile(self, event: str):
        if not self.cfg.profile:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            n, w = self.profile.get(event, (0, 0.0))
            self.profile[event] = (n + 1, w + (time.perf_counter() - t0))

    def classify(
        self,
        sequence: Sequence[str],
        library: List[KnownVariant],
        threshold: Optional[float] = None,
        *,
        identity_sequence: Optional[Sequence[str]] = None,
    ) -> Classification:
        """Classify a raw sequence against known variants.

        Exact variants are always recognized from raw transition tokens.  When
        ``identity_sequence`` is supplied, recipe novelty is evaluated in the
        separate canonical identity space and additionally requires a margin
        over the runner-up recipe.  Legacy callers retain raw-Jaccard behavior.
        """
        use_identity = identity_sequence is not None
        thr = float(
            (getattr(self.cfg, "identity_jaccard_threshold", self.cfg.jaccard_threshold)
             if use_identity else self.cfg.jaccard_threshold)
            if threshold is None else threshold
        )
        with self._profile("classify"):
            if not library: return Classification("new_recipe", None, None, 0.0, 0.0)
            seq = tuple(sequence)
            seq_hash = variant_hash(seq)
            for v in library:
                if v.variant_hash == seq_hash and v.ordering == seq:
                    return Classification("known", v.recipe_id, v.variant_hash, 1.0, 0.0)
            identity_seq = tuple(identity_sequence) if use_identity else seq
            seq_counter_full = Counter(identity_seq)
            seq_counter_type = Counter(_action_type(x) for x in identity_seq)
            # Score every known variant. Variant-side counters are cached lazily on first access.
            scored: List[Tuple[KnownVariant, float, float]] = []
            for v in library:
                candidate = v.identity_ordering if use_identity and v.identity_ordering else v.ordering
                if use_identity:
                    j = _jaccard_counters(seq_counter_full, Counter(candidate))
                else:
                    j = _jaccard_from_cached_counters(
                        seq_counter_full,
                        seq_counter_type,
                        v.counter_full(),
                        v.counter_type(),
                        full_weight=float(getattr(self.cfg, "disambiguator_full_jaccard_weight", 0.70)),
                        type_weight=float(getattr(self.cfg, "disambiguator_type_jaccard_weight", 0.30)),
                    )
                o = kendall_tau_distance(identity_seq, candidate, unmatched_penalty=getattr(self.cfg, "ordering_unmatched_penalty", 0.5))
                scored.append((v, j, o))
            # Group by recipe, pick best-jaccard variant per recipe.
            best_per_recipe: Dict[str, Tuple[KnownVariant, float, float]] = {}
            for v, j, o in scored:
                if v.recipe_id not in best_per_recipe or j > best_per_recipe[v.recipe_id][1]:       best_per_recipe[v.recipe_id] = (v, j, o)
            # Pick the recipe whose best variant has highest Jaccard.
            best_recipe_id, (best_v, best_j, best_o) = max(best_per_recipe.items(), key=lambda kv: (kv[1][1], -kv[1][2]))   # highest jaccard; then lowest tau
            runner_up = max((score for rid, (_v, score, _o) in best_per_recipe.items() if rid != best_recipe_id), default=None)
            margin = float("inf") if runner_up is None else best_j - runner_up
            if best_j < thr or (use_identity and runner_up is not None and margin < float(getattr(self.cfg, "identity_score_margin", 0.0))):
                return Classification("new_recipe", None, None, best_j, best_o)
            # Same action set, different ordering: preference shift.
            return Classification("preference_shift", best_recipe_id, None, best_j, best_o)

    def score_partial(self, prefix: Sequence[str], variants: List[KnownVariant], *, identity: bool = False) -> List[Tuple[KnownVariant, float]]:
        """Rank variants by how well their prefix matches `prefix`. Score combines (a) prefix-identity fraction, (b) 1: Kendall-taudistance on the shared actions. Returned sorted descending."""
        prefix_counter = Counter(prefix)
        out = []
        unmatched_penalty = getattr(self.cfg, "ordering_unmatched_penalty", 0.5)
        for v in variants:
            ordering = v.identity_ordering if identity and v.identity_ordering else v.ordering
            if not prefix or not ordering: out.append((v, 0.0)); continue
            aligned = _lcs_aligned_subsequence(prefix, ordering)
            if not aligned: out.append((v, 0.0)); continue
            variant_counter = Counter(aligned)
            inter = sum((prefix_counter & variant_counter).values())
            union = sum((prefix_counter | variant_counter).values())
            set_overlap = inter / max(union, 1)
            tau = 1.0 - kendall_tau_distance(prefix, aligned, unmatched_penalty=unmatched_penalty)
            out.append((
                v,
                float(getattr(self.cfg, "disambiguator_partial_overlap_weight", 0.60)) * set_overlap
                + float(getattr(self.cfg, "disambiguator_partial_order_weight", 0.40)) * tau,
            ))
        out.sort(key=lambda kv: -kv[1])
        return out


# Variant registry
@dataclass
class Variant:
    recipe_id: str
    variant_hash: str
    ordering: Tuple[str, ...]
    first_seen_step: int
    last_seen_step: int
    identity_ordering: Tuple[str, ...] = ()


class VariantMemory:
    """Stores demonstrated preference variants without contributing a prior."""
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.variants: Dict[str, Dict[str, Variant]] = defaultdict(dict)    # recipe_id -> insertion-ordered dict of variant_hash -> Variant
        self.latest: Dict[str, str] = {}                                    # recipe_id -> variant_hash of most-recently-observed variant.

    def register(
        self,
        recipe_id: str,
        ordering: Sequence[str],
        step: int,
        *,
        identity_ordering: Optional[Sequence[str]] = None,
    ) -> Variant:
        h = variant_hash(ordering)
        identity = tuple(identity_ordering) if identity_ordering is not None else tuple(ordering)
        slot = self.variants[recipe_id]
        if h in slot:
            v = slot[h]
            v.last_seen_step = step
            v.ordering = tuple(ordering)
            v.identity_ordering = identity
        else:
            v = Variant(recipe_id=recipe_id, variant_hash=h, ordering=tuple(ordering), first_seen_step=step, last_seen_step=step, identity_ordering=identity)
            slot[h] = v
        self.latest[recipe_id] = h
        return v

    def promote_latest(self, recipe_id: str, variant_hash_: str, step: int) -> None:
        if variant_hash_ in self.variants.get(recipe_id, {}):
            self.latest[recipe_id] = variant_hash_
            self.variants[recipe_id][variant_hash_].last_seen_step = step

    def library(self, allowed_keys: Optional[Set[VariantKey]] = None) -> List[KnownVariant]:
        out: List[KnownVariant] = []
        for rid, slot in self.variants.items():
            for h, v in slot.items():
                if allowed_keys is None or (rid, h) in allowed_keys: out.append(KnownVariant(rid, h, v.ordering, v.identity_ordering))
        return out

    def variants_of(self, recipe_id: str, allowed_keys: Optional[Set[VariantKey]] = None) -> List[KnownVariant]:
        return [KnownVariant(v.recipe_id, v.variant_hash, v.ordering, v.identity_ordering) for v in self.variants.get(recipe_id, {}).values() if allowed_keys is None or (v.recipe_id, v.variant_hash) in allowed_keys]

    def latest_variant(self, recipe_id: str, allowed_keys: Optional[Set[VariantKey]] = None) -> Optional[Variant]:
        h = self.latest.get(recipe_id)
        slot = self.variants.get(recipe_id, {})
        if h and h in slot and (allowed_keys is None or (recipe_id, h) in allowed_keys): return slot[h]
        if allowed_keys is None: return None
        candidates = [v for v in slot.values() if (v.recipe_id, v.variant_hash) in allowed_keys]
        return max(candidates, key=lambda v: v.last_seen_step) if candidates else None


@lru_cache(maxsize=65536)
def _aligned_next_index_cached(prefix: Tuple[str, ...], ordering: Tuple[str, ...]) -> int:
    n_p, n_o = len(prefix), len(ordering)
    if n_p == 0: return 0
    # Build LCS table to find the longest common subsequence. dp[i][j] = length of LCS of prefix[:i] and ordering[:j]
    dp = [[0] * (n_o + 1) for _ in range(n_p + 1)]
    for i in range(1, n_p + 1):
        for j in range(1, n_o + 1):
            if prefix[i - 1] == ordering[j - 1]:    dp[i][j] = dp[i - 1][j - 1] + 1
            else:                                   dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # Backtrack to find the last matched position in ordering.
    i, j = n_p, n_o
    last_matched_j = -1
    while i > 0 and j > 0:
        if prefix[i - 1] == ordering[j - 1]:
            last_matched_j = max(last_matched_j, j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:  i -= 1
        else:                               j -= 1
    return last_matched_j + 1 if last_matched_j >= 0 else 0

def _aligned_next_index(prefix: Sequence[str], ordering: Sequence[str]) -> int:
    """Return the index of the next action in `ordering` after the LCS-aligned prefix. Uses an LCS pass that respects ordering direction so that preference-shifted prefixes still produce a sensible next-position estimate."""
    return _aligned_next_index_cached(tuple(prefix), tuple(ordering))


def clear_all_module_caches() -> None:
    """Clear module-level caches that can otherwise bleed across seed jobs."""
    _lcs_aligned_subsequence_cached.cache_clear()
    _aligned_next_index_cached.cache_clear()


# Adaptive rehearsal weighting
@dataclass
class Entry:
    recipe_id: str
    variant_hash: str
    ordering: Tuple[str, ...]
    weight: float
    added_step: int
    added_cycle: int
    last_seen_step: int
    seen_count: int = 1
    source_mode: str = ""
    transitions: Tuple[DemoTransition, ...] = ()
    identity_ordering: Tuple[str, ...] = ()

    @property
    def key(self) -> VariantKey:    return (self.recipe_id, self.variant_hash)

@dataclass
class PrunedEntry:
    recipe_id: str
    variant_hash: str
    ordering: Tuple[str, ...]
    added_step: int
    removed_step: int
    added_cycle: int
    removed_cycle: int
    last_seen_step: int
    transitions: Tuple[DemoTransition, ...] = ()
    identity_ordering: Tuple[str, ...] = ()

    @property
    def key(self) -> VariantKey:    return (self.recipe_id, self.variant_hash)


class DecayManager:
    """Tracks active replay weights and pruned-variant reentry metadata."""
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.default_grace_horizon: int = max(0, int(getattr(cfg, "decay_horizon_init", 15)))
        self.decay_horizon_floor: int = max(0, int(getattr(cfg, "decay_horizon_floor", 6)))
        self.decay_after_grace_steps: int = max(1, int(getattr(cfg, "decay_after_grace_steps", 3)))
        self.reuse_window: int = max(1, int(getattr(cfg, "decay_reuse_window", 3)))
        self.post_grace_decay_rate: float = 1.0 / float(self.decay_after_grace_steps)
        self.active: Dict[VariantKey, Entry] = {}
        self.pruned: Dict[VariantKey, PrunedEntry] = {}
        self.latest_by_recipe: Dict[str, str] = {}
        self.latest_keys: Set[VariantKey] = set()
        self.reuse_gaps: List[int] = []
        # Diagnostic trace across recipes. The adaptive controller uses per-recipe windows.
        self._reuse_gap_window: Deque[int] = deque(maxlen=cfg.mwr_window)
        self._recipe_gap_window: Dict[str, Deque[int]] = {}
        self._recipe_last_seen_step: Dict[str, int] = {}
        self.weight_history: List[Tuple[int, VariantKey, float]] = []
        self._last_logged_weight_by_key: Dict[VariantKey, float] = {}
        self.prune_events: List[Tuple[int, VariantKey]] = []
        # All positive recipe reuses: (step, triggering variant key, recipe reuse gap).
        self.reuse_gap_events: List[Tuple[int, VariantKey, int]] = []
        # Archive restorations only: (step, restored key, recipe reuse gap).
        self.reentry_events: List[Tuple[int, VariantKey, int]] = []

    # core ops
    def register(
        self,
        recipe_id: str,
        variant_hash_: str,
        ordering: Tuple[str, ...],
        now: int,
        cycle: int,
        *,
        weight: float = 1.0,
        pin_latest: Optional[bool] = None,
        transitions: Optional[Sequence[DemoTransition]] = None,
        identity_ordering: Optional[Sequence[str]] = None,
    ) -> Entry:
        """Insert, reset an active variant, or restore a pruned variant."""
        key = (recipe_id, variant_hash_)
        should_pin_latest = self.cfg.protect_latest_preference if pin_latest is None else bool(pin_latest)
        weight = max(0.0, min(1.0, float(weight)))
        transition_trace = tuple(transitions or ())
        identity = tuple(identity_ordering) if identity_ordering is not None else tuple(ordering)
        reuse_gap = self._record_recipe_reuse(recipe_id, key, now, step=now)
        if key in self.active:
            e = self.active[key]
            e.weight = 1.0 if key in self.latest_keys else weight
            e.last_seen_step = now
            e.ordering = ordering
            e.identity_ordering = identity
            if transitions is not None:
                e.transitions = transition_trace
            if should_pin_latest: self.mark_latest(recipe_id, variant_hash_)
            if key in self.active: self._log_weight(now, self.active[key])
            return e

        reentering = key in self.pruned
        if reentering:
            arch = self.pruned.pop(key)
            if transitions is None:
                transition_trace = tuple(arch.transitions)
            if identity_ordering is None:
                identity = tuple(arch.identity_ordering or ordering)

        e = Entry(recipe_id=recipe_id, variant_hash=variant_hash_, ordering=ordering, weight=weight, added_step=now, added_cycle=cycle, last_seen_step=now, transitions=transition_trace, identity_ordering=identity)
        self.active[key] = e
        if reentering:
            self.reentry_events.append((int(now), key, int(reuse_gap or 0)))
        if should_pin_latest: self.mark_latest(recipe_id, variant_hash_)
        if key in self.active: self._log_weight(now, self.active[key])
        return e

    def step(self, now: int, cycle: int, protected_keys: Optional[Sequence[VariantKey]] = None) -> List[VariantKey]:
        """Advance one completed session and prune expired unpinned entries.

        Each recipe owns its own grace horizon. Before its first positive reuse
        gap, the cold-start horizon applies. Afterwards, the horizon is the
        maximum of the current per-recipe reuse window, bounded below by the
        configured floor. While
        ``now - last_seen_step <= horizon_for(key)``, the weight is unchanged.
        Once the horizon is exceeded, the entry loses a fixed fraction each
        session and is pruned after ``decay_after_grace_steps`` overdue ticks.
        """
        protected = set(protected_keys or ())
        rate = self.post_grace_decay_rate
        pruned: List[VariantKey] = []
        for key, e in list(self.active.items()):
            if key in self.latest_keys or key in protected:
                e.weight = 1.0
                self._log_weight(now, e)
                continue
            age = int(now) - int(e.last_seen_step)
            horizon = self.horizon_for(key)
            if age <= horizon:
                self._log_weight(now, e)
                continue
            e.weight -= rate
            if e.weight <= self.cfg.prune_threshold:
                self._prune_entry(key, e, now, cycle)
                pruned.append(key)
            else:       self._log_weight(now, e)
        return pruned

    # introspection
    def weights(self) -> Dict[VariantKey, float]:                       return {k: e.weight for k, e in self.active.items()}
    def active_entries(self) -> List[Entry]:                            return list(self.active.values())
    def active_entries_for(self, recipe_id: str) -> List[Entry]:        return [e for e in self.active.values() if e.recipe_id == recipe_id]
    def variants_of(self, recipe_id: str) -> List[Entry]:               return [e for e in self.active.values() if e.recipe_id == recipe_id]
    def recipe_horizon_for(self, recipe_id: str) -> float:
        gaps = self._recipe_gap_window.get(recipe_id)
        if not gaps:
            return max(float(self.default_grace_horizon), float(self.decay_horizon_floor))
        return max(float(max(gaps)), float(self.decay_horizon_floor))
    def horizon_for(self, key: VariantKey) -> float:                    return self.recipe_horizon_for(key[0])
    def horizon_snapshot(self) -> Dict[str, float]:                     return {rid: self.recipe_horizon_for(rid) for rid in {k[0] for k in set(self.active) | set(self.pruned)} | set(self._recipe_gap_window)}
    def mark_latest(self, recipe_id: str, variant_hash_: str) -> None:
        """Protect the latest preference for a recipe from decay/pruning. The previously-pinned variant for this recipe is unpinned and becomes subject to its own grace horizon on future steps."""
        old_h = self.latest_by_recipe.get(recipe_id)
        if old_h is not None:
            old_key = (recipe_id, old_h)
            self.latest_keys.discard(old_key)
        self.latest_by_recipe[recipe_id] = variant_hash_
        key = (recipe_id, variant_hash_)
        self.latest_keys.add(key)
        if key in self.active: self.active[key].weight = 1.0

    def unmark_latest(self, recipe_id: str, variant_hash_: str) -> None:
        key = (recipe_id, variant_hash_)
        if self.latest_by_recipe.get(recipe_id) == variant_hash_: del self.latest_by_recipe[recipe_id]
        self.latest_keys.discard(key)

    def discard(self, recipe_id: str, variant_hash_: str, *, allow_latest: bool = False) -> None:
        """Remove an active or pruned variant without logging a prune event."""
        key = (recipe_id, variant_hash_)
        if key in self.latest_keys and not allow_latest: raise RuntimeError(f"discard() called on latest-pinned variant {key}. Use allow_latest=True only in controlled replacement paths.")
        self.active.pop(key, None)
        self.pruned.pop(key, None)
        self._last_logged_weight_by_key.pop(key, None)
        self.latest_keys.discard(key)
        if self.latest_by_recipe.get(recipe_id) == variant_hash_: del self.latest_by_recipe[recipe_id]

    # introspection helpers
    def window_snapshot(self) -> List[int]:
        """Return current stored recipe reuse-gap samples across all recipes."""
        return list(self._reuse_gap_window)

    def recipe_window_snapshot(self, recipe_id: str) -> List[int]:
        """Return the reuse-gap samples currently controlling one recipe."""
        return list(self._recipe_gap_window.get(recipe_id, ()))

    # rate adaptation
    def _record_recipe_reuse(self, recipe_id: str, key: VariantKey, now: int, step: Optional[int] = None) -> Optional[int]:
        last_seen = self._recipe_last_seen_step.get(recipe_id)
        gap: Optional[int] = None
        if last_seen is not None:
            gap = int(now) - int(last_seen)
            self._record_reuse_gap(key, gap, step=step)
        self._recipe_last_seen_step[recipe_id] = int(now)
        return gap if gap is not None and gap > 0 else None

    def _record_reuse_gap(self, key: VariantKey, gap: int, step: Optional[int] = None) -> None:
        """Record a positive recipe reuse gap in that recipe's moving horizon window."""
        if gap <= 0: return
        gap_i = int(gap)
        self.reuse_gaps.append(gap_i)
        recipe_id = key[0]
        if recipe_id not in self._recipe_gap_window: self._recipe_gap_window[recipe_id] = deque(maxlen=self.reuse_window)
        self._recipe_gap_window[recipe_id].append(gap_i)
        self.reuse_gap_events.append((step if step is not None else 0, key, gap_i))
        self._reuse_gap_window.append(gap_i)

    def _prune_entry(self, key: VariantKey, entry: Entry, now: int, cycle: int) -> None:
        self.latest_keys.discard(key)
        if self.latest_by_recipe.get(entry.recipe_id) == entry.variant_hash: del self.latest_by_recipe[entry.recipe_id]
        self.pruned[key] = PrunedEntry(recipe_id=entry.recipe_id, variant_hash=entry.variant_hash, ordering=entry.ordering, added_step=entry.added_step, removed_step=now, added_cycle=entry.added_cycle, removed_cycle=cycle, last_seen_step=entry.last_seen_step, transitions=tuple(entry.transitions), identity_ordering=tuple(entry.identity_ordering))
        del self.active[key]
        self._last_logged_weight_by_key.pop(key, None)
        self.prune_events.append((now, key))

    def _log_weight(self, step: int, e: Entry) -> None:
        previous = self._last_logged_weight_by_key.get(e.key)
        if previous == e.weight:
            return
        self._last_logged_weight_by_key[e.key] = e.weight
        self.weight_history.append((step, e.key, e.weight))
