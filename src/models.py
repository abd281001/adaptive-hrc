"""Semantic MaxEnt IRL for personalized HRC."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .environment import (_FEAT,CONTAINERS,COOKABLES,CUTTABLES,GRATABLE,INGREDIENTS,ITEMS,LOCATIONS,SEASONINGS,StateTracker)
from .latent_strategy import (LatentStrategyResidual, fuse_strategy_residual)
from .domain import DomainAdapter, default_domain

StateVector = Tuple[int, ...]
Demonstration = List[Tuple[StateVector, str]]


def _softmax_probs(values: Sequence[float], temp: float) -> List[float]:
    if not values: return []
    safe_temperature = max(float(temp), 1e-8)
    peak = max(float(value) for value in values)
    weights = [math.exp((float(value) - peak) / safe_temperature) for value in values]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total): return [1.0 / len(weights)] * len(weights)
    return [weight / total for weight in weights]

@dataclass
class Settings:
    """Single source of truth for every tunable knob."""
    seed: int = 1337

    # Replay memory.
    fixed_decay: float = 0.1
    initial_grace: int = 50
    min_grace: int = 6
    prune_delay: int = 3
    pair_gap_window: int = 12
    parent_weight_samples: int = 5
    pair_prior_half_life: float = 3.0
    gap_quantile: float = 0.90
    gap_iqr_scale: float = 1.50
    recipe_gap_window: int = 24
    global_gap_window: int = 60
    prune_threshold: float = 1e-9
    diagnostic_gap_window: int = 30
    pin_latest: bool = True

    # Recipe matching.
    match_threshold: float = 0.96
    match_margin: float = 0.03
    unmatched_penalty: float = 0.5
    overlap_weight: float = 0.60
    order_weight: float = 0.40

    # Predictor probability floor.
    min_probability: float = 1e-6

    # MaxEnt IRL
    irl_discount: float = 0.9
    irl_temperature: float = 0.5
    # 0.05 oscillated: the gradient norm plateaued at ~8.5 regardless of the
    # iteration budget, leaving the fit non-stationary and its weights
    # sensitive to perturbations below float32 epsilon. At 0.01 the norm falls
    # with the budget and the fit is stable.
    irl_learning_rate: float = 0.01
    irl_l2: float = 0.01
    # Engineered reward representation used by the MaxEnt policy.
    irl_features: str = "engineered"
    predictor: str = "maxent"
    irl_horizon: int = 45
    # 100 truncated the fit before its own stale-progress criterion fired.
    irl_cold_steps: int = 200
    irl_warm_steps: int = 80
    expand_actions: bool = False

    # Semantic value interpolation is a separate out-of-support fallback; it does not change the MaxEnt reward representation above.
    semantic_fallback_enabled: bool = True
    semantic_knn: int = 3
    # Raw 50-D semantic RMS: 0.20 admits at most two unit feature differences.
    semantic_fallback_max_rms_distance: float = 0.20

    # Lightweight episode-level strategy transfer.  This is a bounded residual on MaxEnt, not an independently fitted policy head.
    latent_strategy_enabled: bool = True
    latent_strategy_rank: int = 8
    latent_strategy_knn: int = 3
    latent_strategy_strength: float = 1.0
    # Optional heavier masked-role trajectory alignment (zero = timing only).
    latent_strategy_sequence_weight: float = 0.0

    # Frozen in-context LLM baseline.
    llm_model: str = ""
    llm_context_tokens: int = 32768
    llm_candidate_batch: int = 1
    # Spare VRAM left unclaimed for everything outside this process. On a
    # desktop the display server, compositor and browser share the GPU, their
    # demand is spiky and outside this run's control, and a GPU that cannot
    # serve them takes the whole session down rather than failing this process.
    # This is headroom for that spikiness only: memory those clients already
    # hold is measured and excluded separately, so the total reserve is their
    # current usage plus this. Set to 0 only on a GPU driving no display.
    llm_vram_headroom_gib: float = 1.25
    # Prefill the prompt this many positions at a time. Chunking bounds the
    # attention intermediates that peak alongside the key/value cache, which
    # raises the prompt budget on a GPU shared with a display by about half
    # (measured 8571 -> 12827 tokens). It costs prefill wall time roughly in
    # proportion to the chunk count, and it perturbs candidate probabilities by
    # up to a few percent relative, because a chunk boundary changes the order
    # of a bf16 reduction. Zero keeps the single-forward prefill and today's
    # exact scores.
    llm_prefill_chunk_tokens: int = 0
    # "auto" annotates demonstrations with per-step state and sheds that
    # annotation only when the prompt stops fitting, so a long run can change
    # encoding partway through. Pin "state_delta" or "action_only" to hold one
    # encoding for a whole run; "action_only" is about 2.7x smaller.
    llm_context_encoding: str = "auto"

    # Agent and retraining.
    top_k: int = 1
    # EWC
    ewc_strength: float = 0.4
    ewc_precision_floor: float = 1e-12
    fisher_cap: float = 100.0

    # Behavior-cloning baselines
    bc_learning_rate: float = 0.1
    bc_l2: float = 1e-4
    bc_history: int = 3
    bc_cold_epochs: int = 120
    bc_warm_epochs: int = 60
    bc_batch: int = 64

    # Experience replay.
    replay_capacity: int = 64
    replay_batch: int = 64
    verbose: bool = True
    profile: bool = False

    # Online classification and commit policy.
    # Full default and policy evidence promote directly.
    commit_threshold: float = 0.70
    tentative_threshold: float = 0.45
    provisional_weight: float = 0.20
    provisional_cap: float = 0.75
    confirm_window: int = 5
    confirm_accuracy: float = 0.60
    confirm_similarity: float = 0.95
    # Auditable online self-training score.
    match_weight: float = 0.46
    margin_weight: float = 0.18
    policy_weight: float = 0.04
    agreement_bonus: float = 0.06
    agreement_window: float = 0.40
    margin_scale: float = 0.25
    known_similarity: float = 0.99
    known_score_floor: float = 0.90
    empty_score_cap: float = 0.40

    def __post_init__(self) -> None:
        # Settings is a pure value object: it declares `seed` but owns no
        # generator. Randomness is owned by the consumer that draws from it
        # (see AdaptiveAgent._init_rng), so two agents can never share a
        # stream by accidentally sharing one Settings instance.
        unit_fields = (
            "match_threshold", "match_margin",
            "overlap_weight", "order_weight",
            "commit_threshold",
            "tentative_threshold", "provisional_weight",
            "provisional_cap", "confirm_accuracy",
            "confirm_similarity", "match_weight",
            "margin_weight", "policy_weight",
            "agreement_bonus", "agreement_window",
            "known_similarity", "known_score_floor",
            "empty_score_cap",
        )
        for name in unit_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:                 raise ValueError(f"{name} must be finite and within [0, 1]")
        mixtures = (("matcher partial", self.overlap_weight, self.order_weight),)
        for name, first, second in mixtures:
            if not math.isclose(float(first) + float(second), 1.0, abs_tol=1e-9):   raise ValueError(f"{name} weights must sum to 1")
        if self.tentative_threshold > self.commit_threshold:                        raise ValueError("online commit tentative threshold cannot exceed the full threshold")
        if self.known_score_floor < self.commit_threshold:                          raise ValueError("known-score floor cannot be below the full commit threshold")
        if not 0.0 < float(self.margin_scale):                                      raise ValueError("margin_scale must be positive")
        if (not math.isfinite(float(self.pair_prior_half_life)) or float(self.pair_prior_half_life) <= 0.0):                                raise ValueError("pair_prior_half_life must be finite and positive")
        if self.provisional_weight > self.provisional_cap:                          raise ValueError("provisional commit weight cannot exceed its cap")
        if self.irl_features not in {"semantic", "engineered", "raw_state"}:        raise ValueError("irl_features must be 'semantic', 'engineered', or 'raw_state'")
        if self.predictor not in {"maxent", "in_context_llm"}:                      raise ValueError("predictor must be 'maxent' or 'in_context_llm'")
        if int(self.llm_context_tokens) < 0:                                        raise ValueError("llm_context_tokens cannot be negative")
        if int(self.llm_candidate_batch) < 1:                                       raise ValueError("llm_candidate_batch must be positive")
        if float(self.llm_vram_headroom_gib) < 0.0:                                 raise ValueError("llm_vram_headroom_gib cannot be negative")
        if int(self.llm_prefill_chunk_tokens) < 0:                                  raise ValueError("llm_prefill_chunk_tokens cannot be negative")
        if self.llm_context_encoding not in {"auto", "state_delta", "action_only"}: raise ValueError("llm_context_encoding must be 'auto', 'state_delta', or 'action_only'")
        for name in ("irl_cold_steps", "irl_warm_steps"):
            if int(getattr(self, name)) < 1:                                        raise ValueError(f"{name} must be positive")
        if int(self.semantic_knn) < 1:                                              raise ValueError("semantic_knn must be positive")
        if int(self.latent_strategy_rank) < 1:                                      raise ValueError("latent_strategy_rank must be positive")
        if int(self.latent_strategy_knn) < 1:                                       raise ValueError("latent_strategy_knn must be positive")
        if (not math.isfinite(float(self.latent_strategy_strength)) or float(self.latent_strategy_strength) < 0.0):                         raise ValueError("latent_strategy_strength must be finite and non-negative")
        if not 0.0 <= float(self.latent_strategy_sequence_weight) <= 1.0:           raise ValueError("latent_strategy_sequence_weight must lie in [0, 1]")
        if (not math.isfinite(float(self.semantic_fallback_max_rms_distance)) or float(self.semantic_fallback_max_rms_distance) < 0.0):     raise ValueError("semantic_fallback_max_rms_distance must be finite and non-negative")
DEFAULT_SETTINGS = Settings()


def index_demos(demonstrations: Sequence[Demonstration], unique_actions: Optional[Sequence[str]] = None):
    """StateVector/action index maps shared by all predictors."""
    state_ids: Dict[StateVector, int] = {}
    for demo in demonstrations:
        for state, _ in demo:
            if state not in state_ids:   state_ids[state] = len(state_ids)
    state_vectors = {index: state for state, index in state_ids.items()}

    if unique_actions is None:              action_set = sorted({a for demo in demonstrations for _, a in demo})
    else:                                   action_set = list(unique_actions)
    action_ids = {action: index for index, action in enumerate(action_set)}
    action_labels = {index: action for action, index in action_ids.items()}
    return state_ids, state_vectors, action_ids, action_labels


def feasible_actions(state: StateVector, actions: Sequence[str]) -> Tuple[str, ...]:
    """Retain every effectful, precondition-valid action in the grounded task."""
    return tuple(sorted({str(action) for action in actions if action != "stop" and _apply_action(tuple(state), str(action)) is not None}))


@dataclass
class RunningScaler:
    """Streaming per-feature standardizer over observed symbolic states."""

    count: int = 0
    mean: Optional[np.ndarray] = None
    squared_deviation: Optional[np.ndarray] = None

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:     return
        x = np.asarray(batch, dtype=np.float32)
        if x.ndim != 2:         raise ValueError("feature batch must be 2-D")
        batch_count = int(x.shape[0])
        batch_mean = x.mean(axis=0)
        centered = x - batch_mean
        batch_squared_deviation = np.sum(centered * centered, axis=0)
        if (self.count == 0 or self.mean is None or self.squared_deviation is None):
            self.count = batch_count
            self.mean = batch_mean.astype(np.float32)
            self.squared_deviation = batch_squared_deviation.astype(np.float32)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = (self.mean + delta * (batch_count / max(total, 1))).astype(np.float32)
        self.squared_deviation = (self.squared_deviation + batch_squared_deviation + (delta * delta) * (self.count * batch_count / max(total, 1))).astype(np.float32)
        self.count = total

    @property
    def variance(self) -> np.ndarray:
        if self.mean is None or self.squared_deviation is None: return np.ones(0, dtype=np.float32)
        if self.count < 2: return np.ones_like(self.mean, dtype=np.float32)
        return np.maximum(self.squared_deviation / float(self.count - 1), 1e-8).astype(np.float32)

    @property
    def scale(self) -> np.ndarray:
        if self.mean is None:       return np.ones(0, dtype=np.float32)
        return np.sqrt(self.variance).astype(np.float32)

    def transform(self, batch: np.ndarray) -> np.ndarray:
        x = np.asarray(batch, dtype=np.float32)
        if x.size == 0:             return x.astype(np.float32)
        if self.mean is None:       return x.astype(np.float32)
        scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        return ((x - self.mean) / scale).astype(np.float32)


def _normalize_feature_rows(raw: np.ndarray, known_mean: Optional[np.ndarray], known_scale: Optional[np.ndarray], normalizer: Optional[RunningScaler], update_normalizer: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize either engineered or full-state features through one contract."""
    if raw.size == 0:
        return raw, np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    if normalizer is not None:
        if update_normalizer: normalizer.update(raw)
        features = normalizer.transform(raw)
        mean = normalizer.mean if normalizer.mean is not None else np.zeros(raw.shape[1], dtype=np.float32)
        scale = normalizer.scale if normalizer.mean is not None else np.ones(raw.shape[1], dtype=np.float32)
        return features.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)
    if known_mean is None or known_scale is None:
        local = RunningScaler()
        local.update(raw)
        features = local.transform(raw)
        mean = local.mean if local.mean is not None else np.zeros(raw.shape[1], dtype=np.float32)
        scale = local.scale if local.mean is not None else np.ones(raw.shape[1], dtype=np.float32)
    else:
        mean = np.asarray(known_mean, dtype=np.float32)
        scale = np.asarray(known_scale, dtype=np.float32)
        if mean.shape != (raw.shape[1],) or scale.shape != (raw.shape[1],):
            raise ValueError("known feature statistics do not match feature dimension")
        denom = np.where(np.abs(scale) > 1e-8, scale, 1.0)
        features = (raw - mean) / denom
    return features.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


KEY_INGREDIENTS = ("tomato", "onion", "mushroom", "rice", "meat", "chicken", "fish", "egg", "banana", "strawberries", "lettuce", "cheese", "garlic", "yoghurt", "milk", "oil")
KEY_CONTAINERS  = ("pot", "pan", "plate", "bowl")
KEY_LOCATIONS   = ("prep_station", "cooking_station", "plating_station", "blending_station")
CONTAINER_ANCHORS = (
    ("pot", "cooking_station"), ("pot", "washing_station"), ("pan", "cooking_station"), ("pan", "washing_station"),
    ("plate", "plating_station"), ("plate", "serving_station"), ("plate", "washing_station"),
    ("bowl", "prep_station"), ("bowl", "cooking_station"), ("bowl", "plating_station"), ("bowl", "washing_station"),
    ("glass", "blending_station"), ("glass", "serving_station"), ("glass", "washing_station"),
    ("measuring_cup", "blending_station"), ("measuring_cup", "washing_station"),
)


@dataclass(frozen=True)
class FeatureTerm:
    """One named column of the reward representation.

    ``op`` is ``"sum"`` (add the named state bits) or ``"product"`` (multiply the
    first group's single bit by the sum of the second group).  Making the
    representation declarative rather than implicit in nested loops means a
    given reward weight can be attributed to a named term when reporting which
    features drive the learned reward.
    """
    name: str
    op: str
    keys: Tuple[str, ...]
    factor_keys: Tuple[str, ...] = ()


def _feature_plan(feature_mode: str) -> Tuple[FeatureTerm, ...]:
    """Declare the engineered/semantic reward columns, in their fixed order."""
    engineered = feature_mode == "engineered"
    terms: List[FeatureTerm] = []
    add = terms.append
    # 1) Item density by location.
    for location in LOCATIONS: add(FeatureTerm(f"items_at_{location}", "sum", tuple(f"{item}_at_{location}" for item in ITEMS)))
    # 2) Container-location anchors + composite counts.
    for container, location in CONTAINER_ANCHORS: add(FeatureTerm(f"{container}_at_{location}", "sum", (f"{container}_at_{location}",)))
    add(FeatureTerm("cookware_at_cooking_station", "sum", ("pot_at_cooking_station", "pan_at_cooking_station")))
    add(FeatureTerm("items_at_prep_station_total", "sum", tuple(f"{item}_at_prep_station" for item in ITEMS)))
    add(FeatureTerm("items_at_blending_station_total", "sum", tuple(f"{item}_at_blending_station" for item in ITEMS)))
    # 3) Containment aggregates.
    add(FeatureTerm("contained_total", "sum", tuple(f"{c}_contains_{i}" for c in CONTAINERS for i in INGREDIENTS)))
    for container in CONTAINERS: add(FeatureTerm(f"{container}_contents_total", "sum", tuple(f"{container}_contains_{i}" for i in INGREDIENTS)))
    add(FeatureTerm("containers_holding_mixture", "sum", tuple(f"{c}_contains_mixture" for c in CONTAINERS)))
    # 4) Processing features.
    if engineered:
        for item in CUTTABLES:  add(FeatureTerm(f"{item}_cut", "sum", (f"{item}_cut",)))
        for item in COOKABLES:  add(FeatureTerm(f"{item}_cooked", "sum", (f"{item}_cooked",)))
    add(FeatureTerm("cut_total",      "sum", tuple(f"{i}_cut" for i in CUTTABLES)))
    add(FeatureTerm("grated_total",   "sum", tuple(f"{i}_grated" for i in GRATABLE)))
    add(FeatureTerm("cooked_total",   "sum", tuple(f"{i}_cooked" for i in COOKABLES)))
    add(FeatureTerm("seasoned_total", "sum", tuple(f"{i}_seasoned" for i in INGREDIENTS)))
    add(FeatureTerm("washed_total",   "sum", tuple(f"{i}_washed" for i in ITEMS)))
    if engineered:
        for ingredient in ("chicken", "fish", "mixture"): add(FeatureTerm(f"{ingredient}_seasoned", "sum", (f"{ingredient}_seasoned",)))
    # 5) Tool and serving status.
    for key in ("stove_on", "sink_on", "blender_on", "dish_served"): add(FeatureTerm(key, "sum", (key,)))
    # Literal identity is useful for the engineered representation but is deliberately absent from the transferable semantic reward.
    if engineered:
        # 6) Key ingredient x location grid.
        for ingredient in KEY_INGREDIENTS:
            for location in KEY_LOCATIONS: add(FeatureTerm(f"{ingredient}_at_{location}", "sum", (f"{ingredient}_at_{location}",)))
        # 7) Key ingredient containment in key containers.
        for container in KEY_CONTAINERS:
            for ingredient in KEY_INGREDIENTS: add(FeatureTerm(f"{container}_contains_{ingredient}", "sum", (f"{container}_contains_{ingredient}",)))
    # 8) Seasoning signatures.
    for seasoning in SEASONINGS: add(FeatureTerm(f"seasoned_with_{seasoning}_total", "sum", tuple(f"{i}_seasoned_with_{seasoning}" for i in KEY_INGREDIENTS)))
    # 9) Interaction composites.
    add(FeatureTerm("stove_on_x_cookware_present",  "product", ("stove_on",),   ("pot_at_cooking_station", "pan_at_cooking_station")))
    add(FeatureTerm("blender_on_x_glass_present",   "product", ("blender_on",), ("glass_at_blending_station",)))
    add(FeatureTerm("sink_on_x_dishes_present",     "product", ("sink_on",),    ("plate_at_washing_station", "bowl_at_washing_station")))
    return tuple(terms)


@dataclass(frozen=True)
class _CompiledPlan:
    """A feature plan reduced to one integer matmul plus the product columns.

    ``matrix`` accumulates every sum term, so the whole sum block is a single
    ``states @ matrix``.  Doing it as one operation rather than one call per
    column keeps the cost flat whether a caller builds features for a single
    candidate successor or for a whole fitted state graph.
    """
    names: Tuple[str, ...]
    matrix: np.ndarray                                        # (state_width, n_features), integer
    products: Tuple[Tuple[int, int, np.ndarray], ...]         # (column, left index, factor indices)


def _compiled_plan(feature_mode: str) -> Tuple[Tuple[FeatureTerm, ...], _CompiledPlan]:
    """Resolve a plan's feature names to state indices once, then memoize it.
    Every term is validated here rather than at build time, so a malformed
    column fails at import instead of quietly contributing a wrong value to
    the reward representation.
    """
    terms = _feature_plan(feature_mode)
    state_width = len(_FEAT)
    matrix = np.zeros((state_width, len(terms)), dtype=np.int64)
    products: List[Tuple[int, int, np.ndarray]] = []
    for column, term in enumerate(terms):
        if term.op not in {"sum", "product"}:               raise ValueError(f"feature term {term.name!r} has unknown op {term.op!r}")
        if not term.keys:                                   raise ValueError(f"feature term {term.name!r} has no state keys")
        if term.op == "sum" and term.factor_keys:           raise ValueError(f"sum term {term.name!r} must not declare factor keys")
        # A product multiplies one indicator by a group sum; more than one
        # left-hand key would be silently ignored by the vectorised form.
        if term.op == "product" and len(term.keys) != 1:    raise ValueError(f"product term {term.name!r} needs exactly one left-hand key")
        if term.op == "product" and not term.factor_keys:   raise ValueError(f"product term {term.name!r} has no factor keys")
        if term.op == "sum":
            for key in term.keys: matrix[_FEAT[key], column] += 1
        else:
            products.append((column, _FEAT[term.keys[0]], np.asarray([_FEAT[key] for key in term.factor_keys], dtype=np.intp)))
    names = tuple(term.name for term in terms)
    if len(set(names)) != len(names):                       raise ValueError("feature term names must be unique")
    return terms, _CompiledPlan(names=names, matrix=matrix, products=tuple(products))


_PLANS: Dict[str, Tuple[Tuple[FeatureTerm, ...], _CompiledPlan]] = {
    mode: _compiled_plan(mode) for mode in ("engineered", "semantic")
}


def feature_names(feature_mode: str = "engineered") -> Tuple[str, ...]:
    """Names of the reward columns, aligned with ``reward_weights`` positions."""
    if feature_mode == "raw_state": return tuple(sorted(_FEAT, key=_FEAT.__getitem__))
    if feature_mode not in _PLANS:  raise ValueError("feature_mode must be 'semantic', 'engineered', or 'raw_state'")
    return _PLANS[feature_mode][1].names


def build_features(state_vectors: Dict[int, StateVector], known_mean: Optional[np.ndarray] = None, known_scale: Optional[np.ndarray] = None, normalizer: Optional[RunningScaler] = None,
                   update_normalizer: bool = False, *, feature_mode: str = "engineered", normalize: bool = True):
    """Build identity-masked semantic, engineered, or raw state features."""
    if feature_mode not in {"semantic", "engineered", "raw_state"}: raise ValueError("feature_mode must be 'semantic', 'engineered', or 'raw_state'")
    if feature_mode == "raw_state":
        raw = np.asarray([state_vectors[index] for index in range(len(state_vectors))], dtype=np.float32)
        if not normalize:
            width = raw.shape[1] if raw.ndim == 2 else 0
            return raw, np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
        return _normalize_feature_rows(raw, known_mean, known_scale, normalizer, update_normalizer)

    # Integer accumulation, then a single float32 cast, so the result is
    # identical to evaluating each term independently.
    states = np.asarray([state_vectors[index] for index in range(len(state_vectors))], dtype=np.int64)
    if states.ndim != 2: raise ValueError("state vectors must all have the same width")
    _terms, plan = _PLANS[feature_mode]
    if states.shape[1] != plan.matrix.shape[0]: raise ValueError(f"state width {states.shape[1]} does not match the feature schema width {plan.matrix.shape[0]}")
    columns = states @ plan.matrix
    for column, left, factors in plan.products:
        columns[:, column] = states[:, left] * states[:, factors].sum(axis=1)
    raw = columns.astype(np.float32)
    if not normalize:
        width = raw.shape[1] if raw.ndim == 2 else 0
        return raw, np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    return _normalize_feature_rows(raw, known_mean, known_scale, normalizer, update_normalizer)


def fit_maxent_irl(demonstrations: Sequence[Demonstration], features: np.ndarray, state_ids: Dict[StateVector, int], action_ids: Dict[str, int], settings: Settings, init_weights: Optional[np.ndarray] = None,
                   demo_weights: Optional[Sequence[float]] = None, ewc_anchor: Optional[np.ndarray] = None, ewc_fisher: Optional[np.ndarray] = None, extra_transitions: Optional[Dict[Tuple[StateVector, str], StateVector]] = None,
                   rng: Optional[np.random.Generator] = None):
    """Importance-weighted MaxEnt IRL on the demonstrated state graph.

    The demonstrated graph is small and ragged (a few hundred states, a handful
    of actions each), so it is held as one padded ``(rows, actions)`` grid and
    the soft-Bellman sweep, the policy and the discounted occupancy are all
    array operations over that grid.

    ``rng`` supplies cold-start weight initialization. Callers pass the owning
    agent's generator so one agent is one reproducible stream; omitting it
    derives a fresh stream from ``settings.seed``.
    """
    if rng is None: rng = np.random.default_rng(int(settings.seed))
    n_states, n_features = features.shape
    n_actions = len(action_ids)
    if n_states == 0 or n_features == 0 or n_actions == 0:
        stats = {"model_family": "maxent_irl", "estimated_flops": 0.0, "n_states": float(n_states), "n_features": float(n_features), "n_actions": float(n_actions), "iterations_run": 0.0, "warm_start": 1.0 if init_weights is not None else 0.0}
        return (np.zeros(n_features, dtype=np.float32), np.zeros(n_states), np.zeros(n_states), {}, {}, stats)

    weights = (np.ones(len(demonstrations), dtype=np.float32) if demo_weights is None else np.asarray(demo_weights, dtype=np.float32))
    if float(weights.sum()) <= 0.0: weights = np.ones(len(demonstrations), dtype=np.float32)

    # One predicate decides the initialization, the iteration budget, and the
    # reported flag, so a warm start cannot be claimed while random weights are
    # actually used. `(n_features,)` is a 1-tuple: without the comma this test
    # compared a shape against an int and never held, which silently turned
    # every warm retrain into a truncated random restart.
    warm_start = init_weights is not None and init_weights.shape == (n_features,)
    if warm_start:  reward_weights = init_weights.astype(np.float32).copy()
    else:           reward_weights = rng.standard_normal(n_features).astype(np.float32) * 0.1

    state_action_pairs: set[Tuple[int, int]] = set()
    transitions: Dict[Tuple[int, int], int] = {}
    action_ids_by_state: Dict[int, List[int]] = {}
    for demo in demonstrations:
        for index, (state, action) in enumerate(demo):
            state_id = state_ids[state]
            action_id = action_ids[action]
            state_action_pairs.add((state_id, action_id))
            action_ids_by_state.setdefault(state_id, [])
            if action_id not in action_ids_by_state[state_id]: action_ids_by_state[state_id].append(action_id)
            if index + 1 < len(demo): transitions[(state_id, action_id)] = state_ids[demo[index + 1][0]]
    for (state, action), next_state in (extra_transitions or {}).items():
        if (state not in state_ids or next_state not in state_ids or action not in action_ids or action == "stop"): continue
        state_id = state_ids[state]
        action_id = action_ids[action]
        state_action_pairs.add((state_id, action_id))
        transitions[(state_id, action_id)] = state_ids[next_state]
        action_ids_by_state.setdefault(state_id, [])
        if action_id not in action_ids_by_state[state_id]: action_ids_by_state[state_id].append(action_id)

    deviation_action_id = n_actions
    sink_state_id = n_states
    features = np.vstack([features, np.zeros((1, n_features), dtype=np.float32)])
    for state_id in tuple(action_ids_by_state):
        transitions[(state_id, deviation_action_id)] = sink_state_id
        action_ids_by_state[state_id].append(deviation_action_id)

    empirical_features = np.zeros(n_features, dtype=np.float32)
    start_states = [state_ids[demo[0][0]] for demo in demonstrations]
    discount = float(settings.irl_discount)
    for index, demo in enumerate(demonstrations):
        for step, (state, _action) in enumerate(demo): empirical_features += (float(weights[index]) * (discount ** step) * features[state_ids[state]])
    empirical_features /= max(float(len(demonstrations)), 1.0)

    # --- padded (row, action) grid over the states that have a policy ---------
    policy_states = sorted(action_ids_by_state)
    n_rows = len(policy_states)
    grid_width = max(len(action_ids_by_state[state_id]) for state_id in policy_states)
    row_of_state = {state_id: row for row, state_id in enumerate(policy_states)}
    row_states = np.asarray(policy_states, dtype=np.intp)
    grid_actions = np.full((n_rows, grid_width), -1, dtype=np.intp)
    grid_next = np.zeros((n_rows, grid_width), dtype=np.intp)
    grid_valid = np.zeros((n_rows, grid_width), dtype=bool)
    for row, state_id in enumerate(policy_states):
        for column, action_id in enumerate(action_ids_by_state[state_id]):
            grid_actions[row, column] = action_id
            grid_valid[row, column] = True
            # A pair with no recorded successor contributes only its own reward,
            # which is what routing it to the zero-reward sink reproduces.
            grid_next[row, column] = transitions.get((state_id, action_id), sink_state_id)
    grid_column_of = {(policy_states[row], int(grid_actions[row, column])): (row, column)
                      for row in range(n_rows) for column in range(grid_width) if grid_valid[row, column]}
    # Expert (row, column) index arrays for the weighted log-loss.
    expert_rows: List[int] = []
    expert_columns: List[int] = []
    expert_weights: List[float] = []
    for index, demo in enumerate(demonstrations):
        weight = max(0.0, float(weights[index]))
        if weight <= 0.0: continue
        for state, action in demo:
            row, column = grid_column_of[(state_ids[state], action_ids[action])]
            expert_rows.append(row)
            expert_columns.append(column)
            expert_weights.append(weight)
    expert_row_index = np.asarray(expert_rows, dtype=np.intp)
    expert_column_index = np.asarray(expert_columns, dtype=np.intp)
    expert_weight = np.asarray(expert_weights, dtype=np.float64)
    expert_mass = float(expert_weight.sum())

    temperature = max(float(settings.irl_temperature), 1e-8)
    iterations = int(settings.irl_warm_steps if warm_start else settings.irl_cold_steps)
    horizon = int(settings.irl_horizon)
    demo_mass = np.maximum(weights.astype(np.float64), 0.0) / max(float(len(demonstrations)), 1.0)
    start_rows = np.asarray(start_states, dtype=np.intp)

    def soft_backup(rewards: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One soft-Bellman backup over the whole grid."""
        base = rewards[row_states].astype(np.float64)
        q_grid = base[:, None] + discount * values[grid_next].astype(np.float64)
        peak = np.where(grid_valid, q_grid, -np.inf).max(axis=1)
        mass = np.where(grid_valid, np.exp((q_grid - peak[:, None]) / temperature), 0.0).sum(axis=1)
        updated = rewards.copy()
        updated[row_states] = peak + temperature * np.log(np.maximum(mass, 1e-12))
        return updated, q_grid

    def converge_values(rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        values = rewards.copy()
        q_grid = np.zeros((n_rows, grid_width), dtype=np.float64)
        sweeps = 0
        for _sweep in range(50):
            sweeps += 1
            updated, q_grid = soft_backup(rewards, values)
            gap = float(np.max(np.abs(updated - values)))
            values = updated
            if gap < 1e-6: break
        return values, q_grid, sweeps

    def grid_policy(q_grid: np.ndarray) -> np.ndarray:
        peak = np.where(grid_valid, q_grid, -np.inf).max(axis=1)
        weight_grid = np.where(grid_valid, np.exp((q_grid - peak[:, None]) / temperature), 0.0)
        total = weight_grid.sum(axis=1, keepdims=True)
        return weight_grid / np.where(total > 0.0, total, 1.0)

    best_weights = reward_weights.copy()
    best_loss = float("inf")
    best_gradient_norm = float("inf")
    stale_steps = 0
    momentum = np.zeros(n_features, dtype=np.float32)
    iterations_run = value_sweeps = occupancy_updates = 0
    ewc_active = ewc_anchor is not None and ewc_fisher is not None

    for iteration in range(iterations):
        iterations_run = iteration + 1
        learning_rate = float(settings.irl_learning_rate) * (0.97 ** (stale_steps // 20))
        rewards = features @ reward_weights
        values, q_grid, sweeps = converge_values(rewards)
        value_sweeps += sweeps
        policy = grid_policy(q_grid)

        probabilities = policy[expert_row_index, expert_column_index]
        current_loss = float(-(expert_weight * np.log(np.maximum(probabilities, 1e-12))).sum() / max(expert_mass, 1e-9))
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = reward_weights.copy()
            stale_steps = 0
        else:
            stale_steps += 1

        # Discounted occupancy. Occupancy is linear in the start distribution
        # and the dynamics are shared, so the per-demonstration walks pool into
        # one weighted vector: sum_d w_d (occ_d @ T) == (sum_d w_d occ_d) @ T.
        # The deviation action absorbs into the zero-reward sink and therefore
        # leaks its mass, exactly as the per-state walk did.
        transition_mass = np.where(grid_valid & (grid_actions != deviation_action_id), policy, 0.0)
        pooled = np.zeros(n_states + 1, dtype=np.float64)
        np.add.at(pooled, start_rows, demo_mass)
        expected_features_vector = np.zeros(n_features, dtype=np.float32)
        step_discount = 1.0
        for _step in range(horizon):
            occupancy_updates += int(np.count_nonzero(pooled))
            expected_features_vector += (step_discount * pooled[:n_states] @ features[:n_states]).astype(np.float32)
            contribution = pooled[row_states][:, None] * transition_mass
            next_pooled = np.zeros_like(pooled)
            np.add.at(next_pooled, grid_next.reshape(-1), contribution.reshape(-1))
            if not next_pooled.any(): break
            pooled = next_pooled
            step_discount *= discount

        gradient = empirical_features - expected_features_vector - float(settings.irl_l2) * reward_weights
        if ewc_active:
            size = min(len(reward_weights), len(ewc_anchor), len(ewc_fisher))
            if size: gradient[:size] -= (float(settings.ewc_strength) * ewc_fisher[:size].astype(np.float32) * (reward_weights[:size] - ewc_anchor[:size].astype(np.float32)))
        best_gradient_norm = min(best_gradient_norm, float(np.linalg.norm(gradient)))
        momentum = 0.9 * momentum + 0.1 * gradient
        reward_weights = reward_weights + learning_rate * momentum
        if stale_steps > 60: break

    best_rewards = features @ best_weights
    best_values, best_q_grid, final_value_sweeps = converge_values(best_rewards)

    policy_actions = {state_id: [action for action in actions if action != deviation_action_id] for state_id, actions in action_ids_by_state.items()}
    policy_q_values: Dict[Tuple[int, int], float] = {}
    for (state_id, action_id), (row, column) in grid_column_of.items():
        if action_id == deviation_action_id: continue
        policy_q_values[(state_id, action_id)] = float(best_q_grid[row, column])
    deviation_column = {row: column for (state_id, action_id), (row, column) in grid_column_of.items() if action_id == deviation_action_id}
    deviation_margins: List[float] = []
    for demo in demonstrations:
        for state, action in demo:
            if action == "stop": continue
            row, column = grid_column_of[(state_ids[state], action_ids[action])]
            deviation = deviation_column.get(row)
            if deviation is None: continue
            deviation_margins.append(float(best_q_grid[row, column]) - float(best_q_grid[row, deviation]))
    if deviation_margins:
        deviation_margin_min = float(min(deviation_margins))
        deviation_margin_mean = float(sum(deviation_margins) / len(deviation_margins))
        deviation_margin_negative_frac = float(sum(margin < 0.0 for margin in deviation_margins) / len(deviation_margins))
    else:
        deviation_margin_min = 0.0
        deviation_margin_mean = 0.0
        deviation_margin_negative_frac = 0.0
    stats = {"model_family": "maxent_irl", "flop_accounting_scope": "maxent_fit_partial_arithmetic_only", "flop_cross_model_comparable": False, "n_demonstrations": float(len(demonstrations)),
            "n_demo_state_visits": float(sum(len(demo) for demo in demonstrations)), "n_states": float(n_states), "n_states_augmented": float(n_states + 1), "n_features": float(n_features),
            "parameter_count": float(n_features), "n_actions": float(n_actions), "n_state_action_pairs": float(len(state_action_pairs)), "n_policy_states": float(len(action_ids_by_state)),
            "n_policy_state_actions": float(sum(len(row) for row in action_ids_by_state.values())), "n_transition_edges": float(len(transitions)), "n_extra_transition_edges": float(len(extra_transitions or {})),
            "iterations_requested": float(iterations), "iterations_run": float(iterations_run), "value_sweeps": float(value_sweeps), "final_value_sweeps": float(final_value_sweeps), "occupancy_updates": float(occupancy_updates),
            "occupancy_method": "finite_horizon_dp", "best_demo_log_loss": float(best_loss), "best_gradient_norm": float(best_gradient_norm), "deviation_margin_n": float(len(deviation_margins)),
            "deviation_margin_min": deviation_margin_min, "deviation_margin_mean": deviation_margin_mean, "deviation_margin_negative_frac": deviation_margin_negative_frac, "warm_start": 1.0 if warm_start else 0.0,
            "ewc_enabled": 1.0 if ewc_active else 0.0}
    stats["estimated_flops"] = _estimate_maxent_flops(stats)
    return (best_weights.astype(np.float32), best_rewards[:n_states], best_values[:n_states], policy_q_values, policy_actions, stats)


def _normalize_probs(distribution: Dict[str, float], floor: float) -> Dict[str, float]:
    if not distribution: return {}
    clipped = {action: max(float(probability), floor) for action, probability in distribution.items() if action != "stop"}
    if not clipped: return {}
    total = sum(clipped.values())
    if total <= 0:
        uniform_probability = 1.0 / len(clipped)
        return {action: uniform_probability for action in clipped}
    return {action: probability / total for action, probability in clipped.items()}


_ENV_ACTION_PREFIXES = ("transfer", "load", "unload", "move_container", "cut", "grate", "cook", "cook_contents", "combine", "season_container", "season", "pour", "turn_on", "turn_off", "blend", "serve", "wash")


def _apply_action(state: StateVector, action: str) -> Optional[StateVector]:
    if action == "stop" or not str(action).startswith(_ENV_ACTION_PREFIXES): return None
    tracker = StateTracker()
    if len(state) != tracker.n_features: return None
    tracker.current_state = np.asarray(state, dtype=int).copy()
    before = tuple(tracker.get_state_vector().astype(int).tolist())
    try: tracker.apply_action(action, enforce_preconditions=True)
    except (IndexError, KeyError, ValueError): return None
    after = tuple(tracker.get_state_vector().astype(int).tolist())
    return after if after != before else None


def _estimate_maxent_flops(stats: Mapping[str, Any]) -> float:
    features = float(stats.get("n_features", 0.0))
    states = float(stats.get("n_states_augmented", 0.0))
    state_action_pairs = float(stats.get("n_state_action_pairs", 0.0))
    policy_edges = float(stats.get("n_policy_state_actions", 0.0))
    visits = float(stats.get("n_demo_state_visits", 0.0))
    iterations = float(stats.get("iterations_run", 0.0))
    sweeps = float(stats.get("value_sweeps", 0.0)) + float(stats.get("final_value_sweeps", 0.0))
    occupancy = float(stats.get("occupancy_updates", 0.0))
    empirical_work = 2.0 * visits * features
    reward_work = 2.0 * states * features * max(1.0, iterations + 1.0)
    bellman_work = sweeps * (4.0 * state_action_pairs + 8.0 * policy_edges)
    policy_work = iterations * 8.0 * policy_edges
    occupancy_work = 2.0 * occupancy * features
    gradient_work = iterations * 8.0 * features
    if bool(stats.get("ewc_enabled", 0.0)): gradient_work += iterations * 4.0 * features
    return float(empirical_work + reward_work + bellman_work + policy_work + occupancy_work + gradient_work)


class MaxEntIrl:
    """Tested replay-weighted MaxEnt IRL on demonstrated transitions."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, domain: Optional[DomainAdapter] = None, rng: Optional[np.random.Generator] = None):
        self.settings = settings
        self.domain = domain or default_domain()
        # Assigned before reset() so a re-fit after reset keeps the same stream.
        self.init_rng = rng if rng is not None else np.random.default_rng(int(settings.seed))
        self.reset()

    def reset(self) -> None:
        self.reward_weights:    Optional[np.ndarray] = None
        self.features:          Optional[np.ndarray] = None
        self.normalizer = RunningScaler()
        self.feature_mean:      Optional[np.ndarray] = None
        self.feature_scale:     Optional[np.ndarray] = None
        self.semantic_features: Optional[np.ndarray] = None
        self.state_ids:         Dict[StateVector, int] = {}
        self.state_vectors:     Dict[int, StateVector] = {}
        self.action_ids:        Dict[str, int] = {}
        self.values:            Optional[np.ndarray] = None
        self.q_values:          Dict[Tuple[int, int], float] = {}
        # Demonstrated (state, action) -> successor, mirroring the graph the fit
        # actually optimizes over. Consumed by the Fisher estimator.
        self.transitions:       Dict[Tuple[StateVector, str], StateVector] = {}
        self.last_prediction_stats: Dict[str, Any] = {}
        self.last_fisher_stats: Dict[str, Any] = {}
        self._fallback_counts = (0, 0, 0)  # attempted, accepted, rejected
        self._last_exact_learned_action_count = 0
        self.latent_strategy =  LatentStrategyResidual(self.settings, domain=self.domain)
        self._last_expansion_stats: Dict[str, float] = {}
        self.last_fit_stats:    Dict[str, Any] = {"model_family": "maxent_irl", "estimated_flops": 0.0, "flop_accounting_scope": "maxent_fit_partial_arithmetic_only", "flop_cross_model_comparable": False}

    def _expand_actions(self, demonstrations: Sequence[Demonstration]) -> Dict[Tuple[StateVector, str], StateVector]:
        if not bool(self.settings.expand_actions):
            self._last_expansion_stats = {"enabled": 0.0, "candidate_states": 0.0, "candidate_actions": 0.0, "attempted": 0.0, "accepted": 0.0, "rejected": 0.0}
            return {}
        states = {state for demo in demonstrations for state, _action in demo}
        actions = sorted({action for demo in demonstrations for _state, action in demo if action != "stop"})
        expanded: Dict[Tuple[StateVector, str], StateVector] = {}
        attempted = 0
        for state in states:
            for action in actions:
                attempted += 1
                next_state = self.domain.successor(state, action)
                if next_state is not None: expanded[(state, action)] = next_state
        self._last_expansion_stats = {"enabled": 1.0, "candidate_states": float(len(states)), "candidate_actions": float(len(actions)), "attempted": float(attempted), "accepted": float(len(expanded)), "rejected": float(max(0, attempted - len(expanded)))}
        return expanded

    def fit(self, demonstrations: Sequence[Demonstration], demo_weights: Optional[Sequence[float]] = None, ewc_anchor: Optional[np.ndarray] = None, ewc_fisher: Optional[np.ndarray] = None) -> None:
        if not demonstrations:
            self.reset()
            return
        actions = sorted({action for demo in demonstrations for _, action in demo})
        (self.state_ids, self.state_vectors, self.action_ids, _action_labels) = index_demos(
            demonstrations,
            unique_actions=actions,
        )
        extra_transitions = self._expand_actions(demonstrations)
        self.transitions = {
            (state, action): demo[index + 1][0]
            for demo in demonstrations
            for index, (state, action) in enumerate(demo)
            if index + 1 < len(demo)
        }
        self.transitions.update(extra_transitions)
        for next_state in extra_transitions.values():
            if next_state not in self.state_ids:
                index = len(self.state_ids)
                self.state_ids[next_state] = index
                self.state_vectors[index] = next_state

        self.normalizer = RunningScaler()
        features, mean, scale = self.domain.reward_features(self.state_vectors, normalizer=self.normalizer, update_normalizer=True, feature_mode=self.settings.irl_features)
        # Raw identity-masked features measure fallback similarity only; they never enter reward learning or its normalization.
        semantic_features: Optional[np.ndarray] = None
        if self.settings.semantic_fallback_enabled: semantic_features, _, _ = self.domain.semantic_features(self.state_vectors, normalize=False)
        initial_weights = (self.reward_weights if self.reward_weights is not None and self.reward_weights.shape == (features.shape[1],) else None)
        (reward_weights, _reward_values, values, q_values, _action_ids_by_state, stats) = fit_maxent_irl(
            demonstrations, features, self.state_ids, self.action_ids, self.settings, init_weights=initial_weights, demo_weights=demo_weights, ewc_anchor=ewc_anchor, ewc_fisher=ewc_fisher, extra_transitions=extra_transitions,
            rng=self.init_rng)
        self.reward_weights = reward_weights.astype(np.float32)
        self.features = features
        self.feature_mean = mean
        self.feature_scale = scale
        self.semantic_features = semantic_features
        self.values = values.astype(np.float32)
        self.q_values = q_values
        self.latent_strategy.fit(demonstrations, demo_weights)
        latent_stats = dict(self.latent_strategy.last_fit_stats)
        latent_flops = float(latent_stats.get("latent_strategy_fit_flops", 0.0))
        model_structure = ("maxent_irl_with_latent_strategy_residual" if self.settings.latent_strategy_enabled else "single_maxent_irl")
        self.last_fit_stats = {**dict(stats), **latent_stats, "model_family": "maxent_irl", "model_structure": model_structure, "feature_mode": self.settings.irl_features, "irl_features": self.settings.irl_features,
            "semantic_fallback_enabled": bool(self.settings.semantic_fallback_enabled),
            "semantic_fallback_similarity_features": ("semantic_raw_rms"if self.settings.semantic_fallback_enabled else "disabled"),
            "semantic_fallback_max_rms_distance": float(self.settings.semantic_fallback_max_rms_distance),
            "n_transitions": float(sum(max(0, len(demo) - 1) for demo in demonstrations)),
            "valid_action_expansion": dict(self._last_expansion_stats)}
        self.last_fit_stats["estimated_flops"] = float(self.last_fit_stats.get("estimated_flops", 0.0)) + latent_flops

    def _state_features(self, state: StateVector) -> Optional[np.ndarray]:
        if self.reward_weights is None: return None
        features, _mean, _scale = self.domain.reward_features({0: tuple(state)}, known_mean=self.feature_mean, known_scale=self.feature_scale, feature_mode=self.settings.irl_features)
        return features[0] if features.size else None

    def _semantic_neighbor_value(self, state: StateVector) -> Optional[float]:
        """Return a value only when a semantic neighbour passes the gate."""
        if (not self.settings.semantic_fallback_enabled or self.values is None or self.semantic_features is None or not len(self.values)): return None
        exact = self.state_ids.get(tuple(state))
        if exact is not None and exact < len(self.values): return float(self.values[exact])
        query, _, _ = self.domain.semantic_features({0: tuple(state)}, normalize=False)
        if query.size == 0 or self.semantic_features.size == 0: return None
        distances = np.sqrt(np.mean((self.semantic_features - query[0][None, :]) ** 2, axis=1))
        threshold = float(self.settings.semantic_fallback_max_rms_distance)
        eligible = np.flatnonzero(distances <= threshold)
        if eligible.size == 0: return None
        count = min(max(1, int(self.settings.semantic_knn)), len(eligible))
        eligible_distances = distances[eligible]
        selected = np.argpartition(eligible_distances, count - 1)[:count]
        neighbors = eligible[selected]
        neighbor_distances = distances[neighbors]
        exact_neighbors = neighbors[neighbor_distances <= 1e-12]
        if exact_neighbors.size: return float(np.mean(self.values[exact_neighbors]))
        weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
        return float(np.average(self.values[neighbors], weights=weights))

    def _feasible_distribution(self, state: StateVector, candidates: Sequence[str]) -> Dict[str, float]:
        if self.reward_weights is None: return {}
        state_id = self.state_ids.get(tuple(state))
        candidate_actions = list(self.domain.legal_actions(tuple(state), tuple(str(value) for value in candidates)))
        learned = [(action, float(self.q_values[(state_id, action_id)])) for action in candidate_actions for action_id in (self.action_ids.get(action),) if state_id is not None and action_id is not None and (state_id, action_id) in self.q_values]
        self._last_exact_learned_action_count = len(learned)
        if learned:
            self._fallback_counts = (0, 0, 0)
            probabilities = _softmax_probs([value for _action, value in learned], self.settings.irl_temperature)
            distribution = {action: probability for (action, _value), probability in zip(learned, probabilities)}
            # Preserve the complete state-valid distribution requested by the evaluation contract. Unlearned semantic estimates do not compete with an exact-state MaxEnt policy; they retain floor support for calibration and NLL.
            for action in candidate_actions:
                if action not in distribution: distribution[action] = 0.0
            return _normalize_probs(distribution, self.settings.min_probability)

        current_features = self._state_features(tuple(state))
        if current_features is None: return {}
        current_reward = float(current_features @ self.reward_weights)
        actions: List[str] = []
        q_values: List[float] = []
        semantic_fallback_enabled = bool(self.settings.semantic_fallback_enabled)
        fallback_counts = [0, 0, 0]
        for action in candidate_actions:
            successor = self.domain.successor(tuple(state), action)
            if successor is None: continue
            actions.append(action)
            if not semantic_fallback_enabled:
                # Pure MaxEnt controls retain the common feasible-action interface without receiving a semantic transfer signal.
                q_values.append(current_reward)
                continue
            fallback_counts[0] += 1
            neighbor_value = self._semantic_neighbor_value(successor)
            if neighbor_value is None:
                # Rejected transfer retains neutral probability support.
                fallback_counts[2] += 1
                q_values.append(current_reward)
                continue
            fallback_counts[1] += 1
            q_values.append(current_reward + float(self.settings.irl_discount) * neighbor_value)
        self._fallback_counts = tuple(fallback_counts)
        if not actions: return {}
        probabilities = _softmax_probs(q_values, self.settings.irl_temperature)
        return _normalize_probs(dict(zip(actions, probabilities)), self.settings.min_probability)

    def _record_prediction_stats(self, candidate_count: int) -> None:
        attempted, accepted, rejected = self._fallback_counts
        self.last_prediction_stats = {
            "model_structure": ("maxent_irl_with_latent_strategy_residual" if self.settings.latent_strategy_enabled else "single_maxent_irl"),
            "candidate_count": int(candidate_count),
            "exact_state_learned_action_count": int(self._last_exact_learned_action_count),
            "semantic_fallback_enabled": bool(self.settings.semantic_fallback_enabled),
            "semantic_fallback_attempted": bool(attempted),
            "semantic_fallback_used": bool(accepted),
            "semantic_fallback_accepted_actions": int(accepted),
            "semantic_fallback_rejected_actions": int(rejected)
        }

    def predict(self, state: StateVector, candidate_actions: Optional[Sequence[str]] = None, prefix: Optional[Sequence[str]] = None, allow_latent_strategy: bool = True) -> Dict[str, float]:
        if self.reward_weights is None or self.features is None: return {}
        candidates = (tuple(candidate_actions) if candidate_actions is not None else self.domain.legal_actions(tuple(state), tuple(self.action_ids)))
        distribution = self._feasible_distribution(tuple(state), candidates)
        attempted = bool(self._fallback_counts[0])
        conflict = self._last_exact_learned_action_count >= 2
        latent_score = self.latent_strategy.score(tuple(prefix or ()), candidates)
        alpha = 0.0
        latent_enabled = bool(self.settings.latent_strategy_enabled)
        if latent_enabled and allow_latent_strategy and (attempted or conflict): distribution, alpha = fuse_strategy_residual(distribution, latent_score, float(self.settings.latent_strategy_strength))
        self._record_prediction_stats(len(distribution))
        self.last_prediction_stats.update(dict(self.latent_strategy.last_score_stats))
        self.last_prediction_stats.update({"latent_strategy_eligible": bool(latent_enabled and allow_latent_strategy and (attempted or conflict)), "latent_strategy_used": bool(alpha > 0.0), "latent_strategy_alpha": float(alpha)})
        return distribution

    def latent_supports_correction(self, corrected_prefix: Sequence[str], candidates: Sequence[str], actual: str, predicted: str) -> bool:
        """Test a correction after incorporating it into the latent prefix."""
        if not self.settings.latent_strategy_enabled: return False
        score = self.latent_strategy.score(corrected_prefix, candidates, decision_prefix=corrected_prefix[:-1])
        return self.latent_strategy.supports_correction(actual, predicted)

    def _successor_features(self, state: StateVector, action: str) -> np.ndarray:
        """Return phi(s') for a state-action, or the zero-reward sink row.

        ``fit_maxent_irl`` routes any pair without a recorded successor to an
        absorbing zero-feature sink, so an unrecorded edge contributes a zero
        row here for exactly the same reason.
        """
        assert self.features is not None
        successor = self.transitions.get((tuple(state), str(action)))
        if successor is None: return np.zeros(self.features.shape[1], dtype=np.float32)
        successor_id = self.state_ids.get(successor)
        if successor_id is None or successor_id >= len(self.features): return np.zeros(self.features.shape[1], dtype=np.float32)
        return self.features[successor_id]

    def fisher(self, trajectories: Sequence[Demonstration], demo_weights: Optional[Sequence[float]] = None) -> Optional[np.ndarray]:
        """Diagonal empirical Fisher of the softmax policy's log-likelihood.

        The quantity EWC needs is the curvature of ``log pi(a|s)`` in the reward
        weights, not the magnitude of the state features.  Writing
        ``Q(s,a) = w.phi(s) + gamma V(s'_a)`` and linearizing one step
        (``dQ/dw ~= phi(s) + gamma phi(s'_a)``, dropping the recursive tail of
        ``dV/dw``), the score of the softmax policy is

            d log pi(a|s) / dw  ~=  (gamma / T) * [ phi(s'_a) - E_pi phi(s'_.) ]

        because the state's own ``phi(s)`` is common to every action and cancels
        in the centering.  The diagonal empirical Fisher is the replay-weighted
        mean of that score squared.  The candidate set includes the deviation
        action (``Q = w.phi(s)`` with a zero-feature successor) so that a state
        with a single demonstrated action still carries curvature, matching the
        action set the fit itself normalizes over.

        A state whose successor features coincide with the policy mean
        contributes nothing: EWC then protects only the directions in which the
        policy's action choice is actually sensitive to the reward weights.
        """
        if self.reward_weights is None or self.features is None: return None
        n_features = int(self.features.shape[1])
        sample_weights = ([1.0] * len(trajectories) if demo_weights is None else list(demo_weights))
        temperature = max(float(self.settings.irl_temperature), 1e-8)
        discount = float(self.settings.irl_discount)
        fisher = np.zeros(n_features, dtype=np.float64)
        total = 0.0
        n_visits = n_degenerate = 0
        # Candidate actions per state, cached across repeated visits.
        candidates_by_state: Dict[int, Tuple[str, ...]] = {}
        for trajectory, weight in zip(trajectories, sample_weights):
            weight = float(weight)
            if weight <= 0.0: continue
            for state, action in trajectory:
                state_id = self.state_ids.get(state)
                if state_id is None or state_id >= len(self.features): continue
                if state_id not in candidates_by_state:
                    candidates_by_state[state_id] = tuple(sorted(
                        name for name, action_id in self.action_ids.items()
                        if (state_id, action_id) in self.q_values
                    ))
                candidates = candidates_by_state[state_id]
                if action not in candidates: continue
                # Deviation action: Q = reward(s) because the sink is zero-valued.
                state_reward = float(self.features[state_id] @ self.reward_weights)
                q_values = [float(self.q_values[(state_id, self.action_ids[name])]) for name in candidates]
                q_values.append(state_reward)
                probabilities = np.asarray(_softmax_probs(q_values, temperature), dtype=np.float64)
                successors = np.vstack([
                    *(self._successor_features(state, name) for name in candidates),
                    np.zeros(n_features, dtype=np.float32),
                ]).astype(np.float64)
                mean_successor = probabilities @ successors
                taken = successors[candidates.index(action)]
                score = (discount / temperature) * (taken - mean_successor)
                fisher += weight * score * score
                total += weight
                n_visits += 1
                n_degenerate += int(not np.any(score))
        if total > 0.0: fisher /= total
        fisher_cap = float(self.settings.fisher_cap)
        capped = np.clip(fisher, 0.0, fisher_cap).astype(np.float32)
        self.last_fisher_stats = {
            "fisher_estimator": "diagonal_empirical_fisher_one_step_q_linearization",
            "fisher_scored_visits": float(n_visits),
            "fisher_zero_score_visits": float(n_degenerate),
            "fisher_nonzero_features": float(int(np.count_nonzero(capped))),
            "fisher_mean": float(capped.mean()) if capped.size else 0.0,
            "fisher_max": float(capped.max()) if capped.size else 0.0,
        }
        return capped


def top_probability_tie_size(distribution: Mapping[str, float]) -> int:
    """Return the number of actions sharing the exact maximum probability."""
    if not distribution:
        return 0
    maximum = max(float(probability) for probability in distribution.values())
    return sum(float(probability) == maximum for probability in distribution.values())


def top_actions(distribution: Mapping[str, float], k: int = 1, *, rng: np.random.Generator) -> List[str]:
    """Rank actions by probability and i.i.d. Gaussian draws within exact ties.
    The input distribution is never modified.  Gaussian draws are used only to order equal-probability actions, so non-tied probabilities retain their original ranking and every member of a tie is exchangeable.
    """
    limit = max(0, int(k))
    if not distribution or limit == 0: return []
    groups: Dict[float, List[str]] = {}
    for action, probability in distribution.items(): groups.setdefault(float(probability), []).append(str(action))
    ranked: List[str] = []
    for probability in sorted(groups, reverse=True):
        actions = groups[probability]
        if len(actions) > 1:
            draws = np.asarray(rng.normal(loc=0.0, scale=1.0, size=len(actions)), dtype=np.float64)
            order = np.argsort(-draws, kind="stable")
            ranked.extend(actions[int(index)] for index in order)
        else: ranked.extend(actions)
        if len(ranked) >= limit: break
    return ranked[:limit]
