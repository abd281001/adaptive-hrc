"""Prototype learners and online preference posterior. Recipe prototypes estimate recipe compatibility/frontiers, preference prototypes cluster role-order patterns, and the online posterior combines recipe, preference, and memory evidence for assistive prediction. None of these classes consume simulator preference labels."""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

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
    start_token_counts: Counter = field(default_factory=Counter)
    position_token_counts: Dict[int, Counter] = field(default_factory=dict)
    adjacent_bigram_counts: Counter = field(default_factory=Counter)
    adjacent_trigram_counts: Counter = field(default_factory=Counter)
    terminal_signatures: List[State] = field(default_factory=list)

    def fold_demo(self, tokens: Sequence[str], terminal_state: Optional[State], weight: float = 1.0) -> "RecipePrototype":
        """Return a new prototype with this demo folded in. Pure (returns a new dataclass) so the learner can hold immutable snapshots; in practice the learner mutates in place via update_from_demo."""
        weight = max(float(weight), 0.0)
        new_action_set = frozenset(self.action_set | set(tokens))
        new_counts = Counter(self.action_counts)
        new_prec = Counter(self.precedence_counts)
        new_start = Counter(self.start_token_counts)
        new_position = {int(pos): Counter(counts) for pos, counts in self.position_token_counts.items()}
        new_bigram = Counter(self.adjacent_bigram_counts)
        new_trigram = Counter(self.adjacent_trigram_counts)
        for t in tokens: new_counts[t] += weight
        for i, a in enumerate(tokens):
            for b in tokens[i + 1:]: new_prec[(a, b)] += weight
        if tokens:
            new_start[tokens[0]] += weight
        for i, token in enumerate(tokens):
            new_position.setdefault(i, Counter())[token] += weight
        for i in range(max(0, len(tokens) - 1)):
            new_bigram[(tokens[i], tokens[i + 1])] += weight
        for i in range(max(0, len(tokens) - 2)):
            new_trigram[(tokens[i], tokens[i + 1], tokens[i + 2])] += weight
        new_signatures = list(self.terminal_signatures[-(MAX_TERMINAL_SIGNATURES - 1):])
        if terminal_state is not None: new_signatures.append(terminal_state)
        return RecipePrototype(recipe_id=self.recipe_id, action_set=new_action_set, action_counts=new_counts, n_demos=self.n_demos + 1, mass=self.mass + weight, precedence_counts=new_prec, start_token_counts=new_start, position_token_counts=new_position, adjacent_bigram_counts=new_bigram, adjacent_trigram_counts=new_trigram, terminal_signatures=new_signatures)


class RecipePrototypeLearner:
    """Registry of RecipePrototype objects keyed by recipe_id. Owns no variant data; it consumes whatever the agent passes in. Variant storage, active/pruned status, and replay weighting remain in VariantMemory and DecayManager."""

    def __init__(self, cfg: Optional[Any] = None) -> None:
        self.cfg = cfg
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

    @staticmethod
    def _remaining_from_weighted_orderings(prefix_tokens: Sequence[str], variants: Sequence[Tuple[Sequence[str], float]]) -> Dict[str, float]:
        counts: Counter = Counter()
        mass = 0.0
        for ordering, weight in variants:
            w = max(0.0, float(weight if weight is not None else 1.0))
            if w <= 0.0:
                continue
            mass += w
            for tok in ordering:
                counts[tok] += w
        if mass <= 0.0:
            return {}
        used = Counter(prefix_tokens)
        remaining: Dict[str, float] = {}
        for tok, cnt in counts.items():
            rem = max((float(cnt) / mass) - float(used.get(tok, 0)), 0.0)
            if rem > 0.0:
                remaining[tok] = rem
        return remaining

    def remaining_action_mass(self, prefix_tokens: Sequence[str], recipe_id: str, variants: Optional[Sequence[Tuple[Sequence[str], float]]] = None) -> Dict[str, float]:
        """Weighted remaining recipe actions after subtracting the observed prefix."""
        if variants is not None:
            return self._remaining_from_weighted_orderings(prefix_tokens, variants)
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
        """Preference-invariant compatibility between an observed prefix and a recipe.

        This deliberately ignores directional order. Recipe identity should
        answer "are these actions part of this recipe?", while preference
        prototypes own the order-sensitive evidence.
        """
        proto = self.prototypes.get(recipe_id)
        if proto is None or not prefix_tokens or not proto.action_set: return 0.0
        prefix_set = set(prefix_tokens)
        if not prefix_set: return 0.0
        fallback = max(0.0, min(1.0, float(getattr(self.cfg, "recipe_match_position_fallback", 0.0))))
        matched_unique = len(prefix_set & proto.action_set)
        precision = sum(1.0 for token in prefix_tokens if token in proto.action_set) / max(len(prefix_tokens), 1)
        recall = matched_unique / max(len(proto.action_set), 1)
        f1 = (2.0 * precision * recall / max(precision + recall, 1e-8)) if precision > 0.0 and recall > 0.0 else 0.0
        pairs = [(a, b) for i, a in enumerate(prefix_tokens) for b in prefix_tokens[i + 1:]]
        cooccurrence_score = None
        if pairs:
            ok = 0.0
            for a, b in pairs:
                fwd = float(proto.precedence_counts.get((a, b), 0.0))
                rev = float(proto.precedence_counts.get((b, a), 0.0))
                if fwd > 0.0 or rev > 0.0:
                    ok += 1.0
                elif a in proto.action_set and b in proto.action_set:
                    ok += fallback
            cooccurrence_score = ok / max(len(pairs), 1)
        token_w = max(0.0, float(getattr(self.cfg, "recipe_match_token_weight", 0.70)))
        precedence_w = max(0.0, float(getattr(self.cfg, "recipe_match_precedence_weight", 0.30)))
        if cooccurrence_score is None:
            precedence_w = 0.0
        total_w = token_w + precedence_w
        if total_w <= 0.0: return f1
        score = token_w * f1
        if cooccurrence_score is not None:
            score += precedence_w * cooccurrence_score
        return score / total_w

    def frontier(self, prefix_tokens: Sequence[str], recipe_id: str, variants: Sequence[Tuple[Sequence[str], float]], align_weight: float = 0.45, remaining_variants: Optional[Sequence[Tuple[Sequence[str], float]]] = None) -> Dict[str, float]:
        """Distribution over plausible next tokens for this recipe. Exact variant alignment is only part of the signal. The remaining prototype mass keeps recipe-valid but unseen-order actions available so a preference prototype learned on another recipe can transfer."""
        aligned = self._align_frontier(prefix_tokens, variants)
        proto = self.prototypes.get(recipe_id)
        remaining: Dict[str, float] = {}
        if proto is not None:
            remaining = self.remaining_action_mass(prefix_tokens, recipe_id, variants=remaining_variants)
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

# These are workflow relations whose order can plausibly express a user
# preference.  We deliberately exclude hard causal dependencies (for example,
# serving before an ingredient is prepared), because those would make every
# recipe look similar without representing a preference.  The role ontology is
# task-interface knowledge; no preference label or recipe name enters this
# representation.
PREFERENCE_PRECEDENCE_PAIRS: Tuple[Tuple[str, str], ...] = (
    (ROLE_RETRIEVE_CONTAINER, ROLE_RETRIEVE_INGREDIENT),
    (ROLE_RETRIEVE_CONTAINER, ROLE_PREPARE_INGREDIENT),
    (ROLE_RETRIEVE_INGREDIENT, ROLE_PREPARE_INGREDIENT),
    (ROLE_RETRIEVE_INGREDIENT, ROLE_ADD_TO_CONTAINER),
    (ROLE_PREPARE_INGREDIENT, ROLE_ADD_TO_CONTAINER),
    (ROLE_ACTIVATE_APPLIANCE, ROLE_PREPARE_INGREDIENT),
    (ROLE_ACTIVATE_APPLIANCE, ROLE_COOK_OR_BLEND),
    (ROLE_CLEAN_CONTAINER, ROLE_SERVE),
    (ROLE_STAGE_SERVING_VESSEL, ROLE_SERVE),
)


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


def role_precedence_profile(roles: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """Return observed, recipe-normalized preference precedence evidence.

    Each value is in ``[-1, 1]``.  ``+1`` means every instance of the first
    role precedes every instance of the second; ``-1`` is the reverse.  Missing
    roles are absent from the returned mapping rather than interpreted as an
    ordering choice.  This lets two recipes compare only the workflow axes
    they genuinely share.
    """
    positions: Dict[str, List[int]] = {}
    for idx, role in enumerate(roles):
        positions.setdefault(role, []).append(idx)
    profile: Dict[Tuple[str, str], float] = {}
    for first, second in PREFERENCE_PRECEDENCE_PAIRS:
        first_pos = positions.get(first, ())
        second_pos = positions.get(second, ())
        if not first_pos or not second_pos:
            continue
        total = len(first_pos) * len(second_pos)
        before = sum(1 for left in first_pos for right in second_pos if left < right)
        profile[(first, second)] = 2.0 * _safe_div(float(before), float(total)) - 1.0
    return profile


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
    """Build a preference embedding with explicit reusable precedence axes.

    Local role n-grams are retained as a sparse-data fallback, but the learner
    gives shared precedence evidence priority during clustering and scoring.
    """
    out: Dict[str, float] = {}
    for k, v in role_order_features(roles).items():             out[f"scalar:{k}"] = float(v)
    for (first, second), value in role_precedence_profile(roles).items():
        out[f"precedence:{first}|{second}"] = float(value)
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
    bigram_context_totals: Counter = field(default_factory=Counter)
    trigram_context_totals: Counter = field(default_factory=Counter)
    # Only observed role pairs contribute support.  A missing role is not
    # evidence for either ordering, which is essential when transferring
    # between recipes with different ingredient/action inventories.
    precedence_totals: Counter = field(default_factory=Counter)
    precedence_support: Counter = field(default_factory=Counter)
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
        for bg in role_bigrams(roles):
            self.bigram_counts[bg] += weight
            self.bigram_context_totals[bg[0]] += weight
        for tg in role_trigrams(roles):
            self.trigram_counts[tg] += weight
            self.trigram_context_totals[(tg[0], tg[1])] += weight
        for pair, value in role_precedence_profile(roles).items():
            self.precedence_totals[pair] += weight * float(value)
            self.precedence_support[pair] += weight
        if recipe_id is not None:       self.recipes_seen.add(recipe_id)

    def bigram_distribution(self) -> Dict[Tuple[str, str], float]:
        return _normalize(self.bigram_counts)

    def scalar_profile(self) -> Dict[str, float]:
        if self.mass <= 0.0: return {name: 0.0 for name in SCALAR_FEATURES}
        return {name: float(self.scalar_totals.get(name, 0.0)) / float(self.mass) for name in SCALAR_FEATURES}

    def precedence_profile(self) -> Dict[Tuple[str, str], float]:
        return {
            pair: float(self.precedence_totals[pair]) / float(support)
            for pair, support in self.precedence_support.items()
            if float(support) > 0.0
        }

class PreferencePrototypeLearner:
    """Soft-clustering registry of latent preference prototypes.
    Decision rule per completed demo: material workflow-axis gaps create a new prototype; otherwise `max_sim >= MATCH_THRESHOLD` hard-updates the argmax; low-similarity/high-entropy cases create novelty; ambiguous cases soft-update the top prototypes."""

    def __init__(self, match_threshold: Optional[float] = None, novelty_max_similarity: Optional[float] = None, novelty_entropy_min: Optional[float] = None, tau_pref_cluster: Optional[float] = None, axis_split_threshold: Optional[float] = None, cfg: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.prototypes: Dict[str, PreferencePrototype] = {}
        self._next_pref_id = 1
        self.match_threshold = float(getattr(cfg, "preference_match_threshold", MATCH_THRESHOLD) if match_threshold is None else match_threshold)
        self.novelty_max_similarity = float(getattr(cfg, "preference_novelty_max_similarity", NOVELTY_MAX_SIMILARITY) if novelty_max_similarity is None else novelty_max_similarity)
        self.novelty_entropy_min = float(getattr(cfg, "preference_novelty_entropy_min", NOVELTY_ENTROPY_MIN) if novelty_entropy_min is None else novelty_entropy_min)
        self.tau_pref_cluster = float(getattr(cfg, "preference_cluster_temperature", TAU_PREF_CLUSTER) if tau_pref_cluster is None else tau_pref_cluster)
        self.axis_split_threshold = float(getattr(cfg, "preference_axis_split_threshold", AXIS_SPLIT_THRESHOLD) if axis_split_threshold is None else axis_split_threshold)
        self._cached_role_unigrams: Optional[Counter] = None
        self._cached_role_unigram_total: float = 0.0

    def _new_pref_id(self) -> str:
        pid = f"P{self._next_pref_id}"
        self._next_pref_id += 1
        return pid

    def _invalidate_cache(self) -> None:
        self._cached_role_unigrams = None
        self._cached_role_unigram_total = 0.0

    def _precedence_similarity(
        self,
        profile: Mapping[Tuple[str, str], float],
        proto: PreferencePrototype,
    ) -> Tuple[Optional[float], int]:
        proto_profile = proto.precedence_profile()
        common = sorted(set(profile) & set(proto_profile))
        if not common:
            return None, 0
        # Values live in [-1, 1], so this converts absolute disagreement into
        # a bounded agreement score in [0, 1].
        agreement = [1.0 - min(2.0, abs(float(profile[pair]) - float(proto_profile[pair]))) / 2.0 for pair in common]
        return sum(agreement) / len(agreement), len(common)

    def _prototype_similarity(
        self,
        embedding: Dict[str, float],
        precedence_profile: Mapping[Tuple[str, str], float],
        proto: PreferencePrototype,
    ) -> float:
        legacy = _cosine(embedding, proto.embedding)
        precedence, n_common = self._precedence_similarity(precedence_profile, proto)
        if precedence is None or n_common <= 0:
            return legacy
        weight = max(0.0, min(1.0, float(getattr(self.cfg, "preference_precedence_similarity_weight", 0.75))))
        return weight * precedence + (1.0 - weight) * legacy

    def _soft_assignment(
        self,
        embedding: Dict[str, float],
        precedence_profile: Mapping[Tuple[str, str], float],
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """Return softmax(similarities / tau) + sorted similarity list."""
        sims: List[Tuple[str, float]] = []
        for pid, proto in self.prototypes.items():
            sims.append((pid, self._prototype_similarity(embedding, precedence_profile, proto)))
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
                self._invalidate_cache()
                return pid
            return next(iter(self.prototypes))
        self._invalidate_cache()
        emb = build_embedding(roles)
        scalar_profile = role_order_features(roles)
        precedence_profile = role_precedence_profile(roles)

        # Cold-start: first demo creates the seed prototype.
        if not self.prototypes:
            pid = self._new_pref_id()
            proto = PreferencePrototype(pref_id=pid)
            proto.update_from(emb, roles, recipe_id or "", weight=weight)
            self.prototypes[pid] = proto
            return pid

        q, sims = self._soft_assignment(emb, precedence_profile)
        max_sim = sims[0][1] if sims else 0.0
        argmax_pid = sims[0][0] if sims else None
        h_norm = self._normalized_entropy(q)

        precedence_gap = 0.0
        if argmax_pid is not None:
            proto_precedence = self.prototypes[argmax_pid].precedence_profile()
            common = set(precedence_profile) & set(proto_precedence)
            precedence_gap = max(
                (abs(float(precedence_profile[pair]) - float(proto_precedence[pair])) for pair in common),
                default=0.0,
            )
        split_gap = max(
            self._critical_axis_gap(scalar_profile, self.prototypes[argmax_pid].scalar_profile()) if argmax_pid is not None else 0.0,
            precedence_gap,
        )
        precedence_split = float(getattr(self.cfg, "preference_precedence_split_threshold", self.axis_split_threshold))
        if argmax_pid is not None and split_gap >= min(self.axis_split_threshold, precedence_split):
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

    def _aggregate_role_unigram_counts(self) -> Counter:
        if self._cached_role_unigrams is not None:
            return Counter(self._cached_role_unigrams)
        counts: Counter = Counter()
        for proto in self.prototypes.values():
            for role, count in proto.start_role_counts.items():
                counts[role] += count
            for (_a, b), count in proto.bigram_counts.items():
                counts[b] += count
        self._cached_role_unigrams = Counter(counts)
        self._cached_role_unigram_total = float(sum(counts.values()))
        return counts

    # scoring
    def score_prefix(self, prefix_roles: Sequence[str], pref_id: str) -> float:
        """Cumulative role-order evidence for a prefix under one preference.

        The score is a capped log-likelihood ratio against an active-support
        role-unigram baseline. Unlike a per-token average, repeated correct
        evidence can sharpen the posterior; the cap keeps this term comparable
        to recipe and memory evidence on long prefixes.
        """
        proto = self.prototypes.get(pref_id)
        if proto is None or not prefix_roles: return 0.0
        smooth = 0.1
        n_roles = max(1, len(ROLES))
        aggregate_roles = self._aggregate_role_unigram_counts()
        aggregate_total = self._cached_role_unigram_total
        proto_start_total = float(sum(proto.start_role_counts.values()))
        proto_bigram_total = float(sum(proto.bigram_counts.values()))

        def unigram_prob(role: str) -> float:
            return (float(aggregate_roles.get(role, 0.0)) + smooth) / (aggregate_total + smooth * n_roles)

        def start_prob(role: str) -> float:
            return (float(proto.start_role_counts.get(role, 0.0)) + smooth) / (proto_start_total + smooth * n_roles)

        def bigram_prob(bg: Tuple[str, str]) -> float:
            return (float(proto.bigram_counts.get(bg, 0.0)) + smooth) / (proto_bigram_total + smooth * n_roles * n_roles)

        total = 0.0
        p0 = max(start_prob(prefix_roles[0]), _LOG_FLOOR)
        b0 = max(unigram_prob(prefix_roles[0]), _LOG_FLOOR)
        total += math.log(p0) - math.log(b0)
        for bg in role_bigrams(prefix_roles):
            p = max(bigram_prob(bg), _LOG_FLOOR)
            baseline = max(unigram_prob(bg[0]) * unigram_prob(bg[1]), _LOG_FLOOR)
            total += math.log(p) - math.log(baseline)
        prefix_precedence = role_precedence_profile(prefix_roles)
        proto_precedence = proto.precedence_profile()
        common = set(prefix_precedence) & set(proto_precedence)
        if common:
            # Convert agreement in [0, 1] to centred evidence in [-1, 1].
            # Unlike an absolute position feature, this compares only pairs
            # available in the current recipe prefix.
            precedence_evidence = sum(
                1.0 - abs(float(prefix_precedence[pair]) - float(proto_precedence[pair]))
                for pair in common
            ) / len(common)
            total += float(getattr(self.cfg, "preference_precedence_prefix_weight", 1.20)) * precedence_evidence
        temp = max(float(getattr(self.cfg, "preference_prefix_llr_temperature", 1.0)), 1e-8)
        cap = max(0.0, float(getattr(self.cfg, "preference_prefix_llr_cap", 8.0)))
        total /= temp
        if cap > 0.0:
            total = max(-cap, min(cap, total))
        return total

    @staticmethod
    def _precedence_action_score(
        candidate_role: str,
        candidate_roles: Sequence[str],
        precedence_profile: Mapping[Tuple[str, str], float],
    ) -> Optional[float]:
        """Score whether executing ``candidate_role`` now obeys known axes.

        The candidate set is recipe-derived.  Consequently this is an ordering
        preference over available actions, not a Full-only feasibility mask.
        """
        remaining = Counter(candidate_roles)
        scores: List[float] = []
        for (first, second), preference in precedence_profile.items():
            if candidate_role == first and remaining.get(second, 0) > 0:
                scores.append((float(preference) + 1.0) / 2.0)
            elif candidate_role == second and remaining.get(first, 0) > 0:
                scores.append((1.0 - float(preference)) / 2.0)
        if not scores:
            return None
        # Preserve a nonzero floor so a noisy single correction cannot make
        # a recipe-supported action mathematically impossible.
        return max(0.05, min(1.0, sum(scores) / len(scores)))

    def score_action_role(
        self,
        candidate_role: str,
        prefix_roles: Sequence[str],
        pref_id: str,
        candidate_roles: Optional[Sequence[str]] = None,
        session_precedence: Optional[Mapping[Tuple[str, str], float]] = None,
    ) -> float:
        """Score an available role from reusable precedence first, local cues second.

        ``session_precedence`` is temporary evidence from human corrections in
        the current interaction.  It is supplied by the agent and never folded
        into a persistent prototype here.
        """
        proto = self.prototypes.get(pref_id)
        if proto is None or candidate_role not in ROLES: return 0.0
        bg = proto.bigram_counts
        tg = proto.trigram_counts
        local = 1.0 / len(ROLES)
        if len(prefix_roles) >= 2 and tg:
            last_two = (prefix_roles[-2], prefix_roles[-1])
            num = tg.get((last_two[0], last_two[1], candidate_role), 0)
            den = proto.trigram_context_totals.get(last_two, 0)
            if den > 0: local = (num + 0.1) / (den + 0.1 * len(ROLES))
        elif len(prefix_roles) >= 1 and bg:
            last = prefix_roles[-1]
            num = bg.get((last, candidate_role), 0)
            den = proto.bigram_context_totals.get(last, 0)
            if den > 0: local = (num + 0.1) / (den + 0.1 * len(ROLES))
        phase = self._phase_action_score(candidate_role, prefix_roles, proto, candidate_roles)
        candidates = list(candidate_roles or ())
        persistent_precedence = self._precedence_action_score(
            candidate_role,
            candidates,
            proto.precedence_profile(),
        )
        session_score = self._precedence_action_score(
            candidate_role,
            candidates,
            session_precedence or {},
        )
        if persistent_precedence is None and session_score is None:
            return max(1e-6, 0.20 * local + 0.80 * phase)
        if persistent_precedence is None:
            precedence = float(session_score)
        elif session_score is None:
            precedence = float(persistent_precedence)
        else:
            session_weight = max(0.0, min(1.0, float(getattr(self.cfg, "session_correction_precedence_weight", 0.75))))
            precedence = (1.0 - session_weight) * float(persistent_precedence) + session_weight * float(session_score)
        precedence_weight = max(0.0, min(1.0, float(getattr(self.cfg, "preference_precedence_action_weight", 0.65))))
        legacy = 0.25 * local + 0.75 * phase
        # Map [0, 1] precedence agreement into a positive ordering factor.  A
        # strong reusable ordering must be able to overcome recipe-frontier
        # frequency mass; otherwise it cannot transfer an unseen permutation.
        precedence_factor = 0.20 + 1.80 * precedence
        preference_score = precedence_weight * precedence_factor + (1.0 - precedence_weight) * legacy
        return max(1e-6, preference_score)

    def _phase_action_score(self, candidate_role: str, prefix_roles: Sequence[str], proto: PreferencePrototype, candidate_roles: Optional[Sequence[str]] = None) -> float:
        def cfg_float(name: str, default: float) -> float: return float(getattr(self.cfg, name, default))
        def phase_factor(factor: float) -> float:
            strength = max(0.0, cfg_float("phase_score_strength", 1.0))
            return max(float(factor), 1e-9) ** strength
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
        threshold = cfg_float("phase_score_threshold", 0.65)
        if candidate_role == ROLE_RETRIEVE_CONTAINER: score *= phase_factor(cfg_float("phase_score_container_base", 0.65) + cfg_float("phase_score_container_lead_weight", 0.70) * self._scalar_value(profile, "container_setup_lead_time"))
        if not first_add_seen:
            if candidate_role == ROLE_RETRIEVE_INGREDIENT and target_retrieve >= threshold:                                         score *= phase_factor(cfg_float("phase_score_retrieve_boost", 5.0))
            if candidate_role == ROLE_PREPARE_INGREDIENT and target_prep >= threshold and not has_left(ROLE_RETRIEVE_INGREDIENT):   score *= phase_factor(cfg_float("phase_score_prep_boost", 4.0))
            if candidate_role == ROLE_ADD_TO_CONTAINER:
                if target_retrieve >= threshold and has_left(ROLE_RETRIEVE_INGREDIENT):                                             score *= phase_factor(cfg_float("phase_score_early_add_suppress", 0.18))
                if target_prep >= threshold and has_left(ROLE_PREPARE_INGREDIENT):                                                  score *= phase_factor(cfg_float("phase_score_early_add_suppress", 0.18))
        if not first_cook_seen:
            if candidate_role == ROLE_PREPARE_INGREDIENT and target_prep_done >= threshold:                                         score *= phase_factor(cfg_float("phase_score_prep_done_boost", 3.0))
            if candidate_role == ROLE_COOK_OR_BLEND and target_prep_done >= threshold and has_left(ROLE_PREPARE_INGREDIENT):        score *= phase_factor(cfg_float("phase_score_early_cook_suppress", 0.20))
        if candidate_role == ROLE_STAGE_SERVING_VESSEL:                                                                             score *= phase_factor(cfg_float("phase_score_position_base", 0.85) + cfg_float("phase_score_position_match_weight", 0.30) * (1.0 - min(1.0, abs(progress - target_stage) * 2.0)))
        if candidate_role == ROLE_ACTIVATE_APPLIANCE:                                                                               score *= phase_factor(cfg_float("phase_score_position_base", 0.85) + cfg_float("phase_score_position_match_weight", 0.30) * (1.0 - min(1.0, abs(progress - target_activate) * 2.0)))
        if candidate_role == ROLE_CLEAN_CONTAINER:
            if target_cleanup >= threshold and not served_seen:                                                                     score *= phase_factor(cfg_float("phase_score_cleanup_early_supp", 0.20))
            if target_cleanup <= cfg_float("phase_score_cleanup_eager_threshold", 0.35) and not served_seen:                        score *= phase_factor(cfg_float("phase_score_cleanup_late_boost", 2.0))
            if target_cleanup >= threshold and served_seen:                                                                         score *= phase_factor(cfg_float("phase_score_cleanup_late_boost", 2.0))
        return max(1e-6, min(score, cfg_float("phase_score_max_clamp", 6.0)))

    def all_pref_ids(self) -> List[str]: return list(self.prototypes.keys())


# Online posterior. Floor used to keep log-arguments away from zero.
_LOG_FLOOR = 1e-6


@dataclass(frozen=True)
class MemoryPrior:
    state: str
    active_weight: float = 0.0


@dataclass
class PosteriorWeights:
    """Calibration knobs for raw-evidence posterior expert combination."""
    alpha_recipe: float = 1.0
    alpha_pref:   float = 1.5
    alpha_memory: float = 0.5
    alpha_compat: float = 0.5
    temperature_recipe: float = 1.0
    temperature_pref:   float = 1.0
    temperature_memory: float = 1.0
    temperature_compat: float = 1.0
    global_temperature: float = 1.0
    memory_prior_floor: float = 1e-6
    active_prior_floor: float = 0.05
    absent_prior: float = 0.10


def _softmax_log_values(log_values: Mapping[Any, float], temperature: float = 1.0) -> Dict[Any, float]:
    if not log_values: return {}
    temp = max(float(temperature), 1e-8)
    scaled = {k: float(v) / temp for k, v in log_values.items()}
    m = max(scaled.values())
    exps = {k: math.exp(v - m) for k, v in scaled.items()}
    z = sum(exps.values())
    if z <= 0.0 or not math.isfinite(z):
        uniform = 1.0 / max(1, len(exps))
        return {k: uniform for k in exps}
    return {k: v / z for k, v in exps.items()}


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

    def update(self, prefix_tokens: Sequence[str], prefix_roles: Sequence[str], recipe_protos: RecipePrototypeLearner, pref_protos:   PreferencePrototypeLearner, memory_state_for_recipe, memory_state_for_pair=None, compatibility_for_pair=None) -> Dict[Tuple[str, str], float]:
        """Recompute the posterior from calibrated raw expert evidence."""
        recipe_ids = [p.recipe_id for p in recipe_protos.all()]
        if not recipe_ids:  recipe_ids = [self.UNSEEN_RECIPE]
        pref_ids = pref_protos.all_pref_ids()
        if not pref_ids:    pref_ids = [self.UNSEEN_PREF]

        log_recipe = {rid: self._log_p_recipe(prefix_tokens, rid, recipe_protos) for rid in recipe_ids}
        log_pref = {pid: self._log_p_pref(prefix_roles, pid, pref_protos) for pid in pref_ids}
        memory_priors = {rid: (MemoryPrior("absent") if rid == self.UNSEEN_RECIPE else self._coerce_memory_prior(memory_state_for_recipe(rid))) for rid in recipe_ids}
        log_scores: Dict[Tuple[str, str], float] = {}
        for rid in recipe_ids:
            for pid in pref_ids:
                if callable(memory_state_for_pair) and rid != self.UNSEEN_RECIPE and pid != self.UNSEEN_PREF:
                    pair_prior = self._coerce_memory_prior(memory_state_for_pair(rid, pid))
                else:
                    pair_prior = memory_priors.get(rid, MemoryPrior("absent"))
                compat = 1.0
                if callable(compatibility_for_pair) and rid != self.UNSEEN_RECIPE and pid != self.UNSEEN_PREF:
                    try:
                        compat = float(compatibility_for_pair(rid, pid))
                    except Exception:
                        compat = 1.0
                log_compat = math.log(max(float(compat), _LOG_FLOOR))
                score = (
                    self.weights.alpha_recipe * self._temperature_scale(log_recipe.get(rid, math.log(_LOG_FLOOR)), self.weights.temperature_recipe)
                    + self.weights.alpha_pref * self._temperature_scale(log_pref.get(pid, 0.0), self.weights.temperature_pref)
                    + self.weights.alpha_memory * self._temperature_scale(self._log_p_memory(pair_prior, self.weights), self.weights.temperature_memory)
                    + self.weights.alpha_compat * self._temperature_scale(log_compat, self.weights.temperature_compat)
                )
                log_scores[(rid, pid)] = score

        # Softmax-normalize.
        if not log_scores:
            self._joint = {}
            return self._joint
        self._joint = _softmax_log_values(log_scores, self.weights.global_temperature)
        return self._joint

    @staticmethod
    def _temperature_scale(log_score: float, temperature: float) -> float:
        return float(log_score) / max(float(temperature), 1e-8)

    @staticmethod
    def _coerce_memory_prior(value: Any) -> MemoryPrior:
        if isinstance(value, MemoryPrior):  return value
        if isinstance(value, Mapping):      return MemoryPrior(state=str(value.get("state", "absent")), active_weight=float(value.get("active_weight", 0.0) or 0.0))
        state = str(value)
        if state == "active":               return MemoryPrior("active", active_weight=1.0)
        return MemoryPrior(state)

    @staticmethod
    def _log_p_recipe(prefix_tokens: Sequence[str], rid: str, recipe_protos: RecipePrototypeLearner) -> float:
        """log P(prefix | recipe). Uses the prototype's recipe_match score as a calibrated similarity in [0, 1]."""
        if rid == OnlinePreferencePosterior.UNSEEN_RECIPE:  return math.log(_LOG_FLOOR * 5)
        match = recipe_protos.recipe_match(prefix_tokens, rid)
        return math.log(max(match, _LOG_FLOOR))

    @staticmethod
    def _log_p_pref(prefix_roles: Sequence[str], pid: str, pref_protos: PreferencePrototypeLearner) -> float:
        """Preference prefix evidence as a capped role-order log-likelihood ratio."""
        if pid == OnlinePreferencePosterior.UNSEEN_PREF:    return math.log(_LOG_FLOOR * 5)
        return pref_protos.score_prefix(prefix_roles, pid)

    @staticmethod
    def _log_p_memory(prior: MemoryPrior, weights: PosteriorWeights) -> float:
        """Continuous memory prior derived only from active replay weight."""
        floor = max(float(weights.memory_prior_floor), _LOG_FLOOR)
        if prior.state == "active":
            active_weight = max(0.0, min(1.0, float(prior.active_weight)))
            p = float(weights.active_prior_floor) + (1.0 - float(weights.active_prior_floor)) * active_weight
            return math.log(max(p, floor))
        return math.log(max(float(weights.absent_prior), floor))

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
