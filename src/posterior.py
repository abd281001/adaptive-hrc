"""Prototype learners and online preference posterior. Recipe prototypes estimate recipe compatibility/frontiers, preference prototypes cluster role-order patterns, and the online posterior combines recipe, preference, and memory evidence for assistive prediction. None of these classes consume simulator preference labels."""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .memory import _aligned_next_index
from .representations import (ROLE_ADD_TO_CONTAINER, ROLE_ACTIVATE_APPLIANCE, ROLE_CLEAN_CONTAINER, ROLE_COOK_OR_BLEND, ROLE_PREPARE_INGREDIENT, ROLE_RETRIEVE_CONTAINER, ROLE_RETRIEVE_INGREDIENT, ROLE_SERVE, ROLE_STAGE_SERVING_VESSEL, ROLES, role_bigrams, role_trigrams)


# Recipe prototypes
State = Tuple[int, ...]
MAX_TERMINAL_SIGNATURES = 5


@dataclass
class RecipePrototype:
    """Aggregated representation of one recipe identity."""
    recipe_id: str
    action_set: FrozenSet[str] = frozenset()
    action_counts: Counter = field(default_factory=Counter)
    n_demos: int = 0
    mass: float = 0.0
    precedence_counts: Counter = field(default_factory=Counter)
    terminal_signatures: List[State] = field(default_factory=list)

    def fold_demo(self, tokens: Sequence[str], terminal_state: Optional[State], weight: float = 1.0) -> "RecipePrototype":
        """Return a new prototype with this demo folded in. Pure (returns a new dataclass) so the learner can hold immutable snapshots; in practice the learner mutates in place via update_from_demo."""
        weight = max(float(weight), 0.0)
        new_action_set = frozenset(self.action_set | set(tokens))
        new_counts = Counter(self.action_counts)
        new_prec = Counter(self.precedence_counts)
        for t in tokens: new_counts[t] += weight
        for i, a in enumerate(tokens):
            for b in tokens[i + 1:]: new_prec[(a, b)] += weight
        new_signatures = list(self.terminal_signatures[-(MAX_TERMINAL_SIGNATURES - 1):])
        if terminal_state is not None: new_signatures.append(terminal_state)
        return RecipePrototype(recipe_id=self.recipe_id, action_set=new_action_set, action_counts=new_counts, n_demos=self.n_demos + 1, mass=self.mass + weight, precedence_counts=new_prec, terminal_signatures=new_signatures)


class RecipePrototypeLearner:
    """Registry of RecipePrototype objects keyed by recipe_id. Owns no variant data; it consumes whatever the agent passes in. Variant storage, active/pruned status, and replay weighting remain in VariantMemory and DecayManager."""

    def __init__(self) -> None:
        self.prototypes: Dict[str, RecipePrototype] = {}

    # updates
    def update_from_demo(self, recipe_id: str, tokens: Sequence[str], terminal_state: Optional[State] = None, weight: float = 1.0) -> RecipePrototype:
        """Record one observed demonstration of `recipe_id`."""
        existing = self.prototypes.get(recipe_id, RecipePrototype(recipe_id=recipe_id))
        updated = existing.fold_demo(tokens, terminal_state, weight=weight)
        self.prototypes[recipe_id] = updated
        return updated

    def get(self, recipe_id: str) -> Optional[RecipePrototype]:
        return self.prototypes.get(recipe_id)

    def all(self) -> List[RecipePrototype]:
        return list(self.prototypes.values())

    def remaining_action_mass(self, prefix_tokens: Sequence[str], recipe_id: str) -> Dict[str, float]:
        """Weighted remaining recipe actions after subtracting the observed prefix."""
        proto = self.prototypes.get(recipe_id)
        if proto is None: return {}
        mass = max(float(proto.mass), 1.0)
        # Expected occurrences per single demo execution.
        expected = {tok: float(cnt) / mass for tok, cnt in proto.action_counts.items()}
        used = Counter(prefix_tokens)
        remaining: Dict[str, float] = {}
        for tok, exp in expected.items():
            rem = max(exp - float(used.get(tok, 0)), 0.0)
            if rem > 0.0: remaining[tok] = rem
        return remaining

    # scoring
    def recipe_match(self, prefix_tokens: Sequence[str], recipe_id: str) -> float:
        """Compatibility between an observed prefix and this recipe's action set. Uses asymmetric overlap: the fraction of prefix tokens that appear in the recipe's known action set. Returns 0.0 if the recipe is unknown or the prefix is empty. This is intentionally coarse. 
        Richer features such as object-role and state-effect signatures live in the prototype but are not used here yet. """
        proto = self.prototypes.get(recipe_id)
        if proto is None or not prefix_tokens or not proto.action_set: return 0.0
        prefix_set = set(prefix_tokens)
        if not prefix_set: return 0.0
        overlap = len(prefix_set & proto.action_set) / max(len(prefix_set), 1)
        pairs = [(a, b) for i, a in enumerate(prefix_tokens) for b in prefix_tokens[i + 1:]]
        if not pairs: return overlap
        ok = 0.0
        for a, b in pairs:
            fwd = float(proto.precedence_counts.get((a, b), 0.0))
            rev = float(proto.precedence_counts.get((b, a), 0.0))
            ok += 1.0 if fwd >= rev and fwd > 0.0 else 0.0
        return 0.70 * overlap + 0.30 * (ok / max(len(pairs), 1))

    def frontier(self, prefix_tokens: Sequence[str], recipe_id: str, variants: Sequence[Tuple[Sequence[str], float]], align_weight: float = 0.45) -> Dict[str, float]:
        """Distribution over plausible next tokens for this recipe. Exact variant alignment is only part of the signal. The remaining prototype mass keeps recipe-valid but unseen-order actions available so a preference prototype learned on another recipe can transfer."""
        aligned = self._align_frontier(prefix_tokens, variants)
        proto = self.prototypes.get(recipe_id)
        remaining: Dict[str, float] = {}
        if proto is not None:
            remaining = self.remaining_action_mass(prefix_tokens, recipe_id)
            z = sum(remaining.values())
            if z > 0.0: remaining = {tok: v / z for tok, v in remaining.items()}
        if not aligned: return remaining
        if not remaining: return aligned
        aw = max(0.0, min(1.0, float(align_weight)))
        out: Dict[str, float] = {}
        for tok in set(aligned) | set(remaining): out[tok] = aw * aligned.get(tok, 0.0) + (1.0 - aw) * remaining.get(tok, 0.0)
        z = sum(out.values())
        return {tok: v / z for tok, v in out.items()} if z > 0 else {}

    @staticmethod
    def _align_frontier(prefix_tokens: Sequence[str], variants: Sequence[Tuple[Sequence[str], float]]) -> Dict[str, float]:
        next_tokens: Counter = Counter()
        for ordering, weight in variants:
            idx = _aligned_next_index(prefix_tokens, ordering)
            if idx < len(ordering): next_tokens[ordering[idx]] += float(weight) if weight is not None else 1.0
        total = sum(next_tokens.values())
        if total <= 0.0: return {}
        return {tok: c / total for tok, c in next_tokens.items()}


# Preference prototypes
MATCH_THRESHOLD        = 0.70
NOVELTY_MAX_SIMILARITY = 0.45
NOVELTY_ENTROPY_MIN    = 0.60
TAU_PREF_CLUSTER       = 0.5
AXIS_SPLIT_THRESHOLD   = 0.35


# Embedding
SCALAR_FEATURES: Tuple[str, ...] = ("retrieval_before_first_add", "prep_before_first_add", "prep_completion_before_assembly", "serving_vessel_staging_time", "cleanup_delay_after_last_use", "appliance_activation_position", "container_setup_lead_time")
CRITICAL_AXIS_FEATURES: Tuple[str, ...] = ("retrieval_before_first_add", "prep_before_first_add", "prep_completion_before_assembly", "cleanup_delay_after_last_use", "container_setup_lead_time")


def _safe_div(a: float, b: float) -> float: return a / b if b > 0 else 0.0


def role_order_features(roles: Sequence[str]) -> Dict[str, float]:
    """Compute the normalized scalar workflow features from a role sequence. All features land in [0, 1]. They capture the same intuition as the plan's list (retrieval-before-first-add, prep-before-first-add, etc.)."""
    feats: Dict[str, float] = {f: 0.0 for f in SCALAR_FEATURES}
    n = len(roles)
    if n == 0: return feats

    # Index of the first add_to_container, first cook_or_blend, first serve.
    def _first(name: str) -> int:
        for i, r in enumerate(roles):
            if r == name: return i
        return n  # past-the-end sentinel

    first_add  = _first(ROLE_ADD_TO_CONTAINER)
    first_cook = _first(ROLE_COOK_OR_BLEND)
    first_serve = _first(ROLE_SERVE)

    n_retrieve = sum(1 for r in roles if r == ROLE_RETRIEVE_INGREDIENT)
    n_prep     = sum(1 for r in roles if r == ROLE_PREPARE_INGREDIENT)
    n_clean    = sum(1 for r in roles if r == ROLE_CLEAN_CONTAINER)

    # 1. Retrievals before first add / total retrievals
    retr_before = sum(1 for r in roles[:first_add] if r == ROLE_RETRIEVE_INGREDIENT)
    feats["retrieval_before_first_add"] = _safe_div(retr_before, n_retrieve)
    # 2. Preparations before first add / total preps
    prep_before = sum(1 for r in roles[:first_add] if r == ROLE_PREPARE_INGREDIENT)
    feats["prep_before_first_add"] = _safe_div(prep_before, n_prep)
    # 3. Preps completed before first cook/blend / total preps
    prep_pre_cook = sum(1 for r in roles[:first_cook] if r == ROLE_PREPARE_INGREDIENT)
    feats["prep_completion_before_assembly"] = _safe_div(prep_pre_cook, n_prep)
    # 4. Position fraction of stage_serving_vessel (early=0, late=1).
    stage_idx = next((i for i, r in enumerate(roles) if r == ROLE_STAGE_SERVING_VESSEL), None)
    if stage_idx is not None:   feats["serving_vessel_staging_time"] = stage_idx / max(n - 1, 1)
    else:                       feats["serving_vessel_staging_time"] = 1.0  # never staged ~= "as late as possible"
    # 5. Mean cleanup delay (gap between clean_container and the prior add_to_container or move_container that referenced its container). Approximated here as: fraction of cleans that occur AFTER serve. This cleanly separates "as_soon_as_free" (early cleans) from "after_service" (late cleans).
    cleans_after_serve = sum(1 for i, r in enumerate(roles) if r == ROLE_CLEAN_CONTAINER and i > first_serve)
    feats["cleanup_delay_after_last_use"] = _safe_div(cleans_after_serve, n_clean)
    # 6. Mean position of activate_appliance steps (normalized by length).
    activates = [i for i, r in enumerate(roles) if r == ROLE_ACTIVATE_APPLIANCE]
    if activates:
        mean_pos = sum(activates) / len(activates) / max(n - 1, 1)
        feats["appliance_activation_position"] = mean_pos
    else:
        feats["appliance_activation_position"] = 0.0
    # 7. Mean position of container retrieval from storage (frontload vs just-in-time).
    setup = [i for i, r in enumerate(roles) if r == ROLE_RETRIEVE_CONTAINER]
    if setup:
        mean_pos = sum(setup) / len(setup) / max(n - 1, 1)
        feats["container_setup_lead_time"] = 1.0 - mean_pos  # 1 = frontloaded
    else:
        feats["container_setup_lead_time"] = 0.0
    return feats


def role_bigram_counts(roles: Sequence[str]) -> Counter:
    return Counter(role_bigrams(roles))


def role_trigram_counts(roles: Sequence[str]) -> Counter:
    return Counter(role_trigrams(roles))


def _normalize(counts: Counter, total: Optional[float] = None) -> Dict:
    if total is None:   total = sum(counts.values())
    if total <= 0:          return {}
    return {k: v / total for k, v in counts.items()}


def _l2_norm(vec: Dict) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def _l2_normalize(vec: Dict) -> Dict:
    norm = _l2_norm(vec)
    if norm <= 0:           return dict(vec)
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: Dict, b: Dict) -> float:
    if not a or not b:      return 0.0
    keys = set(a.keys()) & set(b.keys())
    if not keys:            return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na <= 0 or nb <= 0:  return 0.0
    return dot / (na * nb)


def build_embedding(roles: Sequence[str]) -> Dict[str, float]:
    """Build a single concatenated, L2-normalized embedding for a role sequence. Keys are namespaced: ``scalar:<feature>``, ``bi:<r1>|<r2>``, ``tri:<r1>|<r2>|<r3>``."""
    out: Dict[str, float] = {}
    for k, v in role_order_features(roles).items():             out[f"scalar:{k}"] = float(v)
    bigram_total = max(1, len(roles) - 1)
    for (a, b), c in role_bigram_counts(roles).items():         out[f"bi:{a}|{b}"] = c / bigram_total
    trigram_total = max(1, len(roles) - 2)
    for (a, b, c), cnt in role_trigram_counts(roles).items():   out[f"tri:{a}|{b}|{c}"] = cnt / trigram_total
    return _l2_normalize(out)


# PreferencePrototype + Learner
@dataclass
class PreferencePrototype:
    pref_id: str
    embedding: Dict[str, float] = field(default_factory=dict)
    n_demos: int = 0
    mass: float = 0.0
    scalar_totals: Counter = field(default_factory=Counter)
    start_role_counts: Counter = field(default_factory=Counter)
    bigram_counts: Counter = field(default_factory=Counter)
    trigram_counts: Counter = field(default_factory=Counter)
    recipes_seen: set = field(default_factory=set)

    def update_from(self, demo_embedding: Dict[str, float], roles: Sequence[str], recipe_id: str, weight: float = 1.0) -> None:
        """Fold a new demo into the prototype, weighted by ``weight``."""
        weight = max(float(weight), 0.0)
        if weight <= 0.0: return
        # Running-average update of the L2-normalized embedding using effective mass. n_demos remains a raw update count for diagnostics.
        old_mass = float(self.mass)
        new_mass = old_mass + weight
        self.n_demos += 1
        if not self.embedding or old_mass <= 0.0: new_emb = dict(demo_embedding)
        else:
            new_emb: Dict[str, float] = {}
            keys = set(self.embedding.keys()) | set(demo_embedding.keys())
            for k in keys:
                old = self.embedding.get(k, 0.0)
                add = demo_embedding.get(k, 0.0)
                new_emb[k] = (old_mass * old + weight * add) / new_mass
        self.embedding = _l2_normalize(new_emb)
        self.mass = new_mass
        for name, value in role_order_features(roles).items(): self.scalar_totals[name] += weight * float(value)
        if roles:                         self.start_role_counts[roles[0]] += weight
        for bg in role_bigrams(roles):  self.bigram_counts[bg] += weight
        for tg in role_trigrams(roles): self.trigram_counts[tg] += weight
        if recipe_id is not None:       self.recipes_seen.add(recipe_id)

    def bigram_distribution(self) -> Dict[Tuple[str, str], float]:
        return _normalize(self.bigram_counts)

    def scalar_profile(self) -> Dict[str, float]:
        if self.mass <= 0.0: return {name: 0.0 for name in SCALAR_FEATURES}
        return {name: float(self.scalar_totals.get(name, 0.0)) / float(self.mass) for name in SCALAR_FEATURES}

class PreferencePrototypeLearner:
    """Soft-clustering registry of latent preference prototypes.
    Decision rule per completed demo: material workflow-axis gaps create a new prototype; otherwise `max_sim >= MATCH_THRESHOLD` hard-updates the argmax; low-similarity/high-entropy cases create novelty; ambiguous cases soft-update the top prototypes."""

    def __init__(self, match_threshold: float = MATCH_THRESHOLD, novelty_max_similarity: float = NOVELTY_MAX_SIMILARITY, novelty_entropy_min: float = NOVELTY_ENTROPY_MIN, tau_pref_cluster: float = TAU_PREF_CLUSTER, axis_split_threshold: float = AXIS_SPLIT_THRESHOLD) -> None:
        self.prototypes: Dict[str, PreferencePrototype] = {}
        self._next_pref_id = 1
        self.match_threshold = float(match_threshold)
        self.novelty_max_similarity = float(novelty_max_similarity)
        self.novelty_entropy_min = float(novelty_entropy_min)
        self.tau_pref_cluster = float(tau_pref_cluster)
        self.axis_split_threshold = float(axis_split_threshold)

    def _new_pref_id(self) -> str:
        pid = f"P{self._next_pref_id}"
        self._next_pref_id += 1
        return pid

    def _soft_assignment(self, embedding: Dict[str, float]) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """Return softmax(similarities / tau) + sorted similarity list."""
        sims: List[Tuple[str, float]] = []
        for pid, proto in self.prototypes.items(): sims.append((pid, _cosine(embedding, proto.embedding)))
        if not sims: return {}, []
        # softmax with temperature
        scaled = [(pid, s / max(self.tau_pref_cluster, 1e-8)) for pid, s in sims]
        m = max(s for _, s in scaled)
        exps = [(pid, math.exp(s - m)) for pid, s in scaled]
        z = sum(e for _, e in exps)
        q = {pid: (e / z if z > 0 else 1.0 / len(exps)) for pid, e in exps}
        sims.sort(key=lambda kv: kv[1], reverse=True)
        return q, sims

    @staticmethod
    def _normalized_entropy(q: Dict[str, float]) -> float:
        if not q:   return 0.0
        n = len(q)
        if n <= 1:  return 0.0
        h = 0.0
        for p in q.values():
            if p > 0:   h -= p * math.log(p)
        return h / math.log(n)

    @staticmethod
    def _scalar_value(profile: Dict[str, float], name: str) -> float:
        return float(profile.get(name, 0.0))

    @classmethod
    def _critical_axis_gap(cls, a: Dict[str, float], b: Dict[str, float]) -> float:
        return max((abs(cls._scalar_value(a, f) - cls._scalar_value(b, f)) for f in CRITICAL_AXIS_FEATURES), default=0.0)

    def update_from_roles(self, roles: Sequence[str], recipe_id: Optional[str] = None, weight: float = 1.0) -> str:
        """Fold a completed demo's role sequence into the cluster registry. Returns the pref_id of the prototype that received (most of) the update."""
        weight = max(float(weight), 0.0)
        if weight <= 0.0 and self.prototypes: return next(iter(self.prototypes))
        if not roles:
            # Empty demos cannot inform clustering.
            if not self.prototypes:
                pid = self._new_pref_id()
                self.prototypes[pid] = PreferencePrototype(pref_id=pid)
                return pid
            return next(iter(self.prototypes))
        emb = build_embedding(roles)
        scalar_profile = role_order_features(roles)

        # Cold-start: first demo creates the seed prototype.
        if not self.prototypes:
            pid = self._new_pref_id()
            proto = PreferencePrototype(pref_id=pid)
            proto.update_from(emb, roles, recipe_id or "", weight=weight)
            self.prototypes[pid] = proto
            return pid

        q, sims = self._soft_assignment(emb)
        max_sim = sims[0][1] if sims else 0.0
        argmax_pid = sims[0][0] if sims else None
        h_norm = self._normalized_entropy(q)

        if argmax_pid is not None and self._critical_axis_gap(scalar_profile, self.prototypes[argmax_pid].scalar_profile()) >= self.axis_split_threshold:
            pid = self._new_pref_id()
            proto = PreferencePrototype(pref_id=pid)
            proto.update_from(emb, roles, recipe_id or "", weight=weight)
            self.prototypes[pid] = proto
            return pid

        if argmax_pid is not None and max_sim >= self.match_threshold:
            # Hard-update matched prototype.
            self.prototypes[argmax_pid].update_from(emb, roles, recipe_id or "", weight=weight)
            return argmax_pid

        if max_sim < self.novelty_max_similarity and h_norm >= self.novelty_entropy_min:
            pid = self._new_pref_id()
            proto = PreferencePrototype(pref_id=pid)
            proto.update_from(emb, roles, recipe_id or "", weight=weight)
            self.prototypes[pid] = proto
            return pid

        # Ambiguous: soft-update top-2 prototypes weighted by q.
        topk = sims[: min(2, len(sims))]
        for pid, _ in topk: self.prototypes[pid].update_from(emb, roles, recipe_id or "", weight=weight * q.get(pid, 0.0))
        return argmax_pid or next(iter(self.prototypes))

    # scoring
    def score_prefix(self, prefix_roles: Sequence[str], pref_id: str) -> float:
        """Log-probability proxy of an observed role prefix under prototype `pref_id`. Uses the first-role distribution plus smoothed bigrams, normalized by evidence count. Returns 0.0 (i.e. log 1.0) for an unknown prototype or empty prefix."""
        proto = self.prototypes.get(pref_id)
        if proto is None or not prefix_roles: return 0.0
        bigrams = proto.bigram_distribution()
        # Smoothing floor: avoid -inf for never-seen bigrams.
        floor = 1.0 / max(1, sum(proto.bigram_counts.values()) + 1)
        total = 0.0
        n = 0
        if proto.start_role_counts:
            den0 = sum(proto.start_role_counts.values())
            p0 = (proto.start_role_counts.get(prefix_roles[0], 0.0) + 0.1) / (den0 + 0.1 * len(ROLES))
            total += math.log(max(p0, floor))
            n += 1
        for bg in role_bigrams(prefix_roles):
            p = bigrams.get(bg, floor)
            total += math.log(max(p, floor))
            n += 1
        return total / max(n, 1)

    def score_action_role(self, candidate_role: str, prefix_roles: Sequence[str], pref_id: str, candidate_roles: Optional[Sequence[str]] = None) -> float:
        """Preference score for one candidate role under prototype `pref_id`. Local bigram/trigram evidence is blended with phase-level scalar cues (e.g., retrieve/prep before first add, early/late cleanup). The phase term is what makes a preference learned on one recipe transferable to a recipe with a different number of ingredients."""
        proto = self.prototypes.get(pref_id)
        if proto is None or candidate_role not in ROLES: return 0.0
        bg = proto.bigram_counts
        tg = proto.trigram_counts
        local = 1.0 / len(ROLES)
        if len(prefix_roles) >= 2 and tg:
            last_two = (prefix_roles[-2], prefix_roles[-1])
            num = tg.get((last_two[0], last_two[1], candidate_role), 0)
            den = sum(c for k, c in tg.items() if k[0] == last_two[0] and k[1] == last_two[1])
            if den > 0: local = (num + 0.1) / (den + 0.1 * len(ROLES))
        elif len(prefix_roles) >= 1 and bg:
            last = prefix_roles[-1]
            num = bg.get((last, candidate_role), 0)
            den = sum(c for k, c in bg.items() if k[0] == last)
            if den > 0: local = (num + 0.1) / (den + 0.1 * len(ROLES))
        phase = self._phase_action_score(candidate_role, prefix_roles, proto, candidate_roles)
        return max(1e-6, 0.20 * local + 0.80 * phase)

    def _phase_action_score(self, candidate_role: str, prefix_roles: Sequence[str], proto: PreferencePrototype, candidate_roles: Optional[Sequence[str]] = None) -> float:
        profile = proto.scalar_profile()
        target_retrieve = self._scalar_value(profile, "retrieval_before_first_add")
        target_prep = self._scalar_value(profile, "prep_before_first_add")
        target_prep_done = self._scalar_value(profile, "prep_completion_before_assembly")
        target_stage = self._scalar_value(profile, "serving_vessel_staging_time")
        target_cleanup = self._scalar_value(profile, "cleanup_delay_after_last_use")
        target_activate = self._scalar_value(profile, "appliance_activation_position")
        roles_left = list(candidate_roles or ())
        has_left = lambda r: r in roles_left
        first_add_seen = ROLE_ADD_TO_CONTAINER in prefix_roles
        first_cook_seen = ROLE_COOK_OR_BLEND in prefix_roles
        served_seen = ROLE_SERVE in prefix_roles
        progress = len(prefix_roles) / max(len(prefix_roles) + max(len(roles_left), 1), 1)
        score = 1.0
        if candidate_role == ROLE_RETRIEVE_CONTAINER:
            score *= 0.65 + 0.70 * self._scalar_value(profile, "container_setup_lead_time")
        if not first_add_seen:
            if candidate_role == ROLE_RETRIEVE_INGREDIENT and target_retrieve >= 0.65: score *= 5.0
            if candidate_role == ROLE_PREPARE_INGREDIENT and target_prep >= 0.65 and not has_left(ROLE_RETRIEVE_INGREDIENT): score *= 4.0
            if candidate_role == ROLE_ADD_TO_CONTAINER:
                if target_retrieve >= 0.65 and has_left(ROLE_RETRIEVE_INGREDIENT): score *= 0.18
                if target_prep >= 0.65 and has_left(ROLE_PREPARE_INGREDIENT): score *= 0.18
        if not first_cook_seen:
            if candidate_role == ROLE_PREPARE_INGREDIENT and target_prep_done >= 0.65: score *= 3.0
            if candidate_role == ROLE_COOK_OR_BLEND and target_prep_done >= 0.65 and has_left(ROLE_PREPARE_INGREDIENT): score *= 0.20
        if candidate_role == ROLE_STAGE_SERVING_VESSEL:
            score *= 0.85 + 0.30 * (1.0 - min(1.0, abs(progress - target_stage) * 2.0))
        if candidate_role == ROLE_ACTIVATE_APPLIANCE:
            score *= 0.85 + 0.30 * (1.0 - min(1.0, abs(progress - target_activate) * 2.0))
        if candidate_role == ROLE_CLEAN_CONTAINER:
            if target_cleanup >= 0.65 and not served_seen: score *= 0.20
            if target_cleanup <= 0.35 and not served_seen: score *= 2.0
            if target_cleanup >= 0.65 and served_seen: score *= 2.0
        return max(1e-6, min(score, 6.0))

    def all_pref_ids(self) -> List[str]: return list(self.prototypes.keys())


# Online posterior. Floor used to keep log-arguments away from zero.
_LOG_FLOOR = 1e-6


@dataclass
class PosteriorWeights:
    """Calibration knobs for the posterior log-linear factorization."""
    lambda_recipe: float = 1.0
    lambda_pref:   float = 1.0
    lambda_memory: float = 0.5
    tau_recipe:    float = 1.0
    tau_pref:      float = 1.0
    tau_memory:    float = 1.0


class OnlinePreferencePosterior:
    """Online posterior over (recipe_id, pref_id) hypotheses. The posterior is *read-only* with respect to the prototype learners and the variant memory: it queries them but never mutates them. This keeps it safe to update during evaluation as long as the agent's `frozen()` snapshot includes the posterior's 
    own state, which the agent layer ensures by holding the marginal vectors as instance attributes that are checked on freeze entry/exit."""

    UNSEEN_RECIPE = "<unseen_recipe>"
    UNSEEN_PREF   = "<unseen_pref>"

    def __init__(self, weights: Optional[PosteriorWeights] = None) -> None:
        self.weights = weights or PosteriorWeights()
        self._joint: Dict[Tuple[str, str], float] = {}

    def reset(self) -> None:
        self._joint = {}

    def update(self, prefix_tokens: Sequence[str], prefix_roles: Sequence[str], recipe_protos: RecipePrototypeLearner, pref_protos:   PreferencePrototypeLearner, memory_state_for_recipe) -> Dict[Tuple[str, str], float]:
        """Recompute the posterior given the current prefix. `memory_state_for_recipe` is a callable `rid -> str` returning one of {"active", "pruned", "absent"}; the posterior uses it to weight `log_p_memory` without ever consulting `pref_hash`."""
        recipe_ids = [p.recipe_id for p in recipe_protos.all()]
        if not recipe_ids:  recipe_ids = [self.UNSEEN_RECIPE]
        pref_ids = pref_protos.all_pref_ids()
        if not pref_ids:    pref_ids = [self.UNSEEN_PREF]

        log_scores: Dict[Tuple[str, str], float] = {}
        for rid in recipe_ids:
            log_pr = self._log_p_recipe(prefix_tokens, rid, recipe_protos)
            mem_state = memory_state_for_recipe(rid) if rid != self.UNSEEN_RECIPE else "absent"
            log_pm = self._log_p_memory(mem_state)
            for pid in pref_ids:
                log_pp = self._log_p_pref(prefix_roles, pid, pref_protos)
                score = (self.weights.lambda_recipe * log_pr / max(self.weights.tau_recipe, 1e-8) + self.weights.lambda_pref * log_pp / max(self.weights.tau_pref, 1e-8) + self.weights.lambda_memory * log_pm / max(self.weights.tau_memory, 1e-8))
                log_scores[(rid, pid)] = score

        # Softmax-normalize.
        if not log_scores:
            self._joint = {}
            return self._joint
        m = max(log_scores.values())
        exps = {k: math.exp(v - m) for k, v in log_scores.items()}
        z = sum(exps.values())
        if z <= 0:
            uniform = 1.0 / len(exps)
            self._joint = {k: uniform for k in exps}
        else:
            self._joint = {k: v / z for k, v in exps.items()}
        return self._joint

    @staticmethod
    def _log_p_recipe(prefix_tokens: Sequence[str], rid: str, recipe_protos: RecipePrototypeLearner) -> float:
        """log P(prefix | recipe). Uses the prototype's recipe_match score as a calibrated similarity in [0, 1]."""
        if rid == OnlinePreferencePosterior.UNSEEN_RECIPE:  return math.log(_LOG_FLOOR * 5)
        proto = recipe_protos.get(rid)
        match = recipe_protos.recipe_match(prefix_tokens, rid)
        mass = 1.0 if proto is None else max(float(proto.mass), 1.0)
        return math.log(max(match, _LOG_FLOOR)) + 0.10 * math.log(mass)

    @staticmethod
    def _log_p_pref(prefix_roles: Sequence[str], pid: str, pref_protos: PreferencePrototypeLearner) -> float:
        """log P(prefix | preference). The prototype's score_prefix already returns a length-normalized log-probability; we pass it through."""
        if pid == OnlinePreferencePosterior.UNSEEN_PREF:    return math.log(_LOG_FLOOR * 5)
        proto = pref_protos.prototypes.get(pid)
        mass = 1.0 if proto is None else max(float(proto.mass), 1.0)
        return pref_protos.score_prefix(prefix_roles, pid) + 0.10 * math.log(mass)

    @staticmethod
    def _log_p_memory(state: str) -> float:
        """log P(recipe, preference_state). The plan calls for high prior on active recipes, lower on pruned, lowest on absent. We use fixed bands; calibration adjusts the temperature, not these constants."""
        if state == "active":   return math.log(1.0)    #  0.0
        if state == "pruned":   return math.log(0.5)    # ≈ -0.693
        return math.log(0.1)                            # ≈ -2.30

    # queries
    def joint(self) -> Dict[Tuple[str, str], float]:
        return dict(self._joint)

    def marginal_recipe(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (rid, _pid), p in self._joint.items(): out[rid] = out.get(rid, 0.0) + p
        return out

    def marginal_preference(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (_rid, pid), p in self._joint.items(): out[pid] = out.get(pid, 0.0) + p
        return out

    def argmax_recipe(self) -> Optional[str]:
        m = self.marginal_recipe()
        if not m:                           return None
        rid = max(m, key=m.get)
        if rid == self.UNSEEN_RECIPE:       return None
        return rid

    def argmax_preference(self) -> Optional[str]:
        m = self.marginal_preference()
        if not m:                           return None
        pid = max(m, key=m.get)
        if pid == self.UNSEEN_PREF:         return None
        return pid

    # gating
    def normalized_entropy(self) -> float:
        if not self._joint:                 return 1.0
        n = len(self._joint)
        if n <= 1:                          return 0.0
        h = 0.0
        for p in self._joint.values():
            if p > 0:                       h -= p * math.log(p)
        return h / math.log(n)

    def confidence(self) -> float:
        """1 - normalized_entropy(joint)."""
        return 1.0 - self.normalized_entropy()

    # snapshot
    def freeze_snapshot(self) -> Tuple:
        """A hashable summary of posterior state used by the agent's freeze contract. Empty until the first update."""
        return tuple(sorted((k[0], k[1], round(float(v), 12)) for k, v in self._joint.items()))
