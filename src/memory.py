"""Memory, novelty, and adaptive rehearsal state. This module groups the non-parametric memory subsystem: variant identity, sequence disambiguation, latest-preference bookkeeping, adaptive replay weights, pruning, pruned-variant metadata, and reentry. Keeping these together makes the memory contract visible to the agent and evaluation harness."""
from __future__ import annotations
import contextlib
import hashlib
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple, TypeAlias
from .models import Config, DEFAULT_CONFIG
# Sequence disambiguation
VariantKey: TypeAlias = Tuple[str, str]

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

def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Weighted multiset overlap. 70% full action-string overlap + 30% action-type overlap."""
    ca_full = Counter(a)
    cb_full = Counter(b)
    ca_type = Counter(_action_type(x) for x in a)
    cb_type = Counter(_action_type(x) for x in b)
    return 0.70 * _jaccard_counters(ca_full, cb_full) + 0.30 * _jaccard_counters(ca_type, cb_type)

def _jaccard_from_cached_counters(ca_full: Counter, ca_type: Counter, cb_full: Counter, cb_type: Counter) -> float:
    """Same as `jaccard` when both sides already have cached counters."""
    return 0.70 * _jaccard_counters(ca_full, cb_full) + 0.30 * _jaccard_counters(ca_type, cb_type)

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

    def classify(self, sequence: Sequence[str], library: List[KnownVariant], threshold: Optional[float] = None) -> Classification:
        """Classify `sequence` against the library. `threshold` overrides `self.cfg.jaccard_threshold` for this call only, useful for threshold-sweep tests that want to share an agent across threshold values without reconstructing the disambiguator."""
        thr = float(self.cfg.jaccard_threshold if threshold is None else threshold)
        with self._profile("classify"):
            if not library: return Classification("new_recipe", None, None, 0.0, 0.0)
            seq = tuple(sequence)
            seq_hash = variant_hash(seq)
            for v in library:
                if v.variant_hash == seq_hash and v.ordering == seq:
                    return Classification("known", v.recipe_id, v.variant_hash, 1.0, 0.0)
            seq_counter_full = Counter(seq)
            seq_counter_type = Counter(_action_type(x) for x in seq)
            # Score every known variant. Variant-side counters are cached lazily on first access.
            scored: List[Tuple[KnownVariant, float, float]] = []
            for v in library:
                j = _jaccard_from_cached_counters(seq_counter_full, seq_counter_type, v.counter_full(), v.counter_type())
                o = kendall_tau_distance(seq, v.ordering, unmatched_penalty=getattr(self.cfg, "ordering_unmatched_penalty", 0.5))
                scored.append((v, j, o))
            # Group by recipe, pick best-jaccard variant per recipe.
            best_per_recipe: Dict[str, Tuple[KnownVariant, float, float]] = {}
            for v, j, o in scored:
                if v.recipe_id not in best_per_recipe or j > best_per_recipe[v.recipe_id][1]:       best_per_recipe[v.recipe_id] = (v, j, o)
            # Pick the recipe whose best variant has highest Jaccard.
            best_recipe_id, (best_v, best_j, best_o) = max(best_per_recipe.items(), key=lambda kv: (kv[1][1], -kv[1][2]))   # highest jaccard; then lowest tau
            if best_j < thr:     return Classification("new_recipe", None, None, best_j, best_o)
            # Same action set, different ordering: preference shift.
            return Classification("preference_shift", best_recipe_id, None, best_j, best_o)

    def score_partial(self, prefix: Sequence[str], variants: List[KnownVariant]) -> List[Tuple[KnownVariant, float]]:
        """Rank variants by how well their prefix matches `prefix`. Score combines (a) prefix-identity fraction, (b) 1: Kendall-taudistance on the shared actions. Returned sorted descending."""
        prefix_counter = Counter(prefix)
        out = []
        for v in variants:
            m = min(len(prefix), len(v.ordering))
            if m == 0: out.append((v, 0.0)); continue
            # Set-based overlap: fraction of prefix tokens that appear in the variant's first m actions.
            variant_counter = Counter(v.ordering[:m])
            inter = sum((prefix_counter & variant_counter).values())
            union = sum((prefix_counter | variant_counter).values())
            set_overlap = inter / max(union, 1)
            tau = 1.0 - kendall_tau_distance(prefix, v.ordering[:m], unmatched_penalty=0.5)
            out.append((v, 0.6 * set_overlap + 0.4 * tau))
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


class VariantMemory:
    """Stores demonstrated preference variants without contributing a prior."""
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.variants: Dict[str, Dict[str, Variant]] = defaultdict(dict)    # recipe_id -> insertion-ordered dict of variant_hash -> Variant
        self.latest: Dict[str, str] = {}                                    # recipe_id -> variant_hash of most-recently-observed variant.

    def register(self, recipe_id: str, ordering: Sequence[str], step: int) -> Variant:
        h = variant_hash(ordering)
        slot = self.variants[recipe_id]
        if h in slot:
            v = slot[h]
            v.last_seen_step = step
            v.ordering = tuple(ordering)
        else:
            v = Variant(recipe_id=recipe_id, variant_hash=h, ordering=tuple(ordering), first_seen_step=step, last_seen_step=step)
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
                if allowed_keys is None or (rid, h) in allowed_keys: out.append(KnownVariant(rid, h, v.ordering))
        return out

    def variants_of(self, recipe_id: str, allowed_keys: Optional[Set[VariantKey]] = None) -> List[KnownVariant]:
        return [KnownVariant(v.recipe_id, v.variant_hash, v.ordering) for v in self.variants.get(recipe_id, {}).values() if allowed_keys is None or (v.recipe_id, v.variant_hash) in allowed_keys]

    def latest_variant(self, recipe_id: str, allowed_keys: Optional[Set[VariantKey]] = None) -> Optional[Variant]:
        h = self.latest.get(recipe_id)
        slot = self.variants.get(recipe_id, {})
        if h and h in slot and (allowed_keys is None or (recipe_id, h) in allowed_keys): return slot[h]
        if allowed_keys is None: return None
        candidates = [v for v in slot.values() if (v.recipe_id, v.variant_hash) in allowed_keys]
        return max(candidates, key=lambda v: v.last_seen_step) if candidates else None


def _aligned_next_index(prefix: Sequence[str], ordering: Sequence[str]) -> int:
    """Return the index of the next action in `ordering` after the LCS-aligned prefix. Uses a greedy LCS pass that respects ordering direction so that preference-shifted prefixes (where prefix order != ordering order) still produce a sensible next-position estimate."""
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
    prior_lifespan: Optional[int] = None

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

    @property
    def key(self) -> VariantKey:    return (self.recipe_id, self.variant_hash)


class DecayManager:
    """Tracks active replay weights and pruned-variant reentry metadata."""
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.base_rate: float = 1.0 / max(1, int(getattr(cfg, "decay_horizon_init", 10)))
        self.global_rate: float = self.base_rate
        self.active: Dict[VariantKey, Entry] = {}
        self.pruned: Dict[VariantKey, PrunedEntry] = {}
        self.latest_by_recipe: Dict[str, str] = {}
        self.latest_keys: Set[VariantKey] = set()
        self.reuse_gaps: List[int] = []
        # Bounded same-variant reuse cadence: active repeats and post-pruning restores both answer "how long until this recipe_pref is used again?"
        self._reuse_gap_window: Deque[int] = deque(maxlen=cfg.mwr_window)
        self._per_recipe_gap_window: Dict[str, deque] = {}   # recipe_id -> deque of gaps
        self._active_recipe_count_window: Deque[int] = deque(maxlen=cfg.mwr_window)
        self.rate_history: List[Tuple[int, float]] = [(0, self.global_rate)]
        self.weight_history: List[Tuple[int, VariantKey, float]] = []
        self.prune_events: List[Tuple[int, VariantKey]] = []
        # Entries are all same-variant reuses: (step, key, reuse_gap).
        self.reentry_events: List[Tuple[int, VariantKey, int]] = []
        self._last_count_append_step: int = -1   # guards _active_recipe_count_window deduplication
        self._recompute_effective(step=0)

    # core ops
    def register(self, recipe_id: str, variant_hash_: str, ordering: Tuple[str, ...], now: int, cycle: int, *, weight: float = 1.0, pin_latest: Optional[bool] = None) -> Entry:
        """Insert, reset an active variant, or restore a pruned variant."""
        key = (recipe_id, variant_hash_)
        should_pin_latest = self.cfg.protect_latest_preference if pin_latest is None else bool(pin_latest)
        weight = max(0.0, min(1.0, float(weight)))
        if key in self.active:
            e = self.active[key]
            gap = now - e.last_seen_step
            self._record_reuse_gap(key, gap, step=now)
            e.weight = 1.0 if key in self.latest_keys else weight
            e.last_seen_step = now
            e.ordering = ordering
            if should_pin_latest: self.mark_latest(recipe_id, variant_hash_)
            self._enforce_recipe_capacity(recipe_id, now, cycle)
            self._recompute_effective(step=now)
            if key in self.active: self._log_weight(now, self.active[key])
            return e

        prior_lifespan = None
        if key in self.pruned:
            arch = self.pruned.pop(key)
            gap = now - arch.last_seen_step
            prior_lifespan = arch.removed_step - arch.added_step
            self._record_reuse_gap(key, gap, step=now)

        e = Entry(recipe_id=recipe_id, variant_hash=variant_hash_, ordering=ordering, weight=weight, added_step=now, added_cycle=cycle, last_seen_step=now, prior_lifespan=prior_lifespan)
        self.active[key] = e
        if should_pin_latest: self.mark_latest(recipe_id, variant_hash_)
        self._enforce_recipe_capacity(recipe_id, now, cycle)
        self._recompute_effective(step=now)
        if key in self.active: self._log_weight(now, self.active[key])
        return e

    def step(self, now: int, cycle: int, protected_keys: Optional[Sequence[VariantKey]] = None) -> List[VariantKey]:
        """Advance one interaction step (one session). Returns pruned keys. Decay is deducted using the CURRENT `self.global_rate`, not a per-entry snapshot. This is the contract: when the cadence signal moves the rate, every existing entry's lifespan adapts. The optional `protected_keys` argument is no longer 
        needed in normal use; pinning is owned by `latest_keys`; and is kept only for defensive callers."""
        protected = set(protected_keys or ())
        # Refresh the global rate first so the deduction uses the current value.
        self._recompute_effective(step=now)
        rate = self.global_rate
        pruned: List[VariantKey] = []
        for key, e in list(self.active.items()):
            if key in self.latest_keys or key in protected:
                e.weight = 1.0
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
    def mark_latest(self, recipe_id: str, variant_hash_: str) -> None:
        """Protect the latest preference for a recipe from decay/pruning. The previously-pinned variant for this recipe is unpinned and starts decaying at the next `step()` (which uses `self.global_rate` global-rate changes therefore propagate to its lifespan)."""
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
        self.latest_keys.discard(key)
        if self.latest_by_recipe.get(recipe_id) == variant_hash_: del self.latest_by_recipe[recipe_id]

    def _enforce_recipe_capacity(self, recipe_id: str, now: int, cycle: int) -> List[VariantKey]:
        """Keep the active replay set bounded without deleting the full registry. The latest-pinned variant is never a victim. Older non-latest entries are moved to ``pruned`` so they remain recognizable for reentry."""
        cap = max(1, int(getattr(self.cfg, "max_variants_per_recipe", 0) or 0))
        active_keys = [key for key in self.active if key[0] == recipe_id]
        pruned: List[VariantKey] = []
        while len(active_keys) > cap:
            candidates = [self.active[key] for key in active_keys if key not in self.latest_keys]
            if not candidates: break
            victim = min(candidates, key=lambda e: (float(e.weight), int(e.last_seen_step), int(e.added_step), e.variant_hash))
            self._prune_entry(victim.key, victim, now, cycle)
            pruned.append(victim.key)
            active_keys = [key for key in self.active if key[0] == recipe_id]
        return pruned

    # introspection helpers
    def window_snapshot(self) -> List[int]:
        """Return current stored same-variant reuse-gap samples."""
        return list(self._reuse_gap_window)

    def active_recipe_count_window_snapshot(self) -> List[int]:
        """Return recent active-library sizes used for rate scaling."""
        return list(self._active_recipe_count_window)

    # rate adaptation
    def _record_reuse_gap(self, key: VariantKey, gap: int, step: Optional[int] = None) -> None:
        """Record any same-variant reuse gap for diagnostics."""
        if gap <= 0: return
        gap_i = int(gap)
        self.reuse_gaps.append(gap_i)
        recipe_id = key[0]
        if recipe_id not in self._per_recipe_gap_window: self._per_recipe_gap_window[recipe_id] = deque(maxlen=self.cfg.mwr_window)
        self._per_recipe_gap_window[recipe_id].append(gap_i)
        self.reentry_events.append((step if step is not None else 0, key, gap_i))
        self._reuse_gap_window.append(gap_i)
        self._refresh_base_from_reuse_window()

    def _refresh_base_from_reuse_window(self) -> None:
        horizon = max(1, int(getattr(self.cfg, "decay_horizon_init", 10)))
        # Use the median of per-recipe max-gaps, not a global max. A single long-cadence recipe should not slow decay for all others.
        if self._per_recipe_gap_window:
            per_recipe_maxes = [max(w) for w in self._per_recipe_gap_window.values() if w]
            if per_recipe_maxes:
                per_recipe_maxes.sort()
                mid = len(per_recipe_maxes) // 2
                horizon = max(horizon, per_recipe_maxes[mid])
        elif self._reuse_gap_window:
            horizon = max(horizon, max(self._reuse_gap_window))
        self.base_rate = 1.0 / float(horizon)

    def _recompute_effective(self, step: int) -> None:
        self._refresh_base_from_reuse_window()
        active_count = len(self.active)
        d_active = max(active_count, 1)
        # Append at most once per step to keep active-count history at session-level granularity (one entry per completed session, not one entry per internal recompute call within a session).
        if step != getattr(self, "_last_count_append_step", -1):
            self._active_recipe_count_window.append(active_count)
            self._last_count_append_step = step
        elif self._active_recipe_count_window:
            self._active_recipe_count_window[-1] = max(int(self._active_recipe_count_window[-1]), active_count)
        d_recent_max = max(max(self._active_recipe_count_window), d_active) if self._active_recipe_count_window else d_active
        exponent = float(self.cfg.size_rescale_exponent)

        # More active variants imply lower probability of any one variant being reused soon, so each entry decays more slowly. A recent-max term avoids abrupt rate jumps immediately after pruning.
        scale = (max(d_active, d_recent_max, 1) ** exponent)
        rate = float(self.base_rate / max(scale, 1e-9))
        max_horizon = int(getattr(self.cfg, "max_decay_horizon", 0) or 0)
        if max_horizon > 0: rate = max(rate, 1.0 / float(max_horizon))
        rate = max(rate, float(getattr(self.cfg, "min_global_rate", 0.0) or 0.0))
        max_rate = float(getattr(self.cfg, "max_global_rate", 0.0) or 0.0)
        if max_rate > 0.0: rate = min(rate, max_rate)
        self.global_rate = rate
        self.rate_history.append((step, rate))

    def _prune_entry(self, key: VariantKey, entry: Entry, now: int, cycle: int) -> None:
        self.latest_keys.discard(key)
        if self.latest_by_recipe.get(entry.recipe_id) == entry.variant_hash: del self.latest_by_recipe[entry.recipe_id]
        self.pruned[key] = PrunedEntry(recipe_id=entry.recipe_id, variant_hash=entry.variant_hash, ordering=entry.ordering, added_step=entry.added_step, removed_step=now, added_cycle=entry.added_cycle, removed_cycle=cycle, last_seen_step=entry.last_seen_step)
        del self.active[key]
        self.prune_events.append((now, key))

    def _log_weight(self, step: int, e: Entry) -> None: self.weight_history.append((step, e.key, e.weight))
