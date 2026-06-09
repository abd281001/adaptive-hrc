"""State-feature learning models: MaxEnt IRL 2.0, n-gram Markov 2.0, and log-linear fusion.

This module uses the following state-aware learning design:
    * fixed-dimensional engineered feature matrix from symbolic states      * discounted feature expectations               * demo-augmented MDP with virtual deviate action -> null state
    * soft Bellman value iteration                                          * Monte Carlo expected feature rollouts         * momentum ascent with warm starts across retrains
"""
from __future__ import annotations

import random
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .environment import (_FEAT, CONTAINERS, COOKABLES, CUTTABLES, GRATABLE, INGREDIENTS, ITEMS, LOCATIONS, SEASONINGS)

State = Tuple[int, ...]
Trajectory = List[Tuple[State, str]]


def _sample_categorical(probs: Sequence[float], rng: Optional[random.Random] = None) -> int:
    """Sample an index from a categorical distribution in Python space.
    Pass ``rng`` (a ``random.Random`` instance, typically ``cfg.prng``) for
    seeded reproducibility. Falls back to the global ``random`` module only
    when no rng is supplied; the IRL fit loop should pass the configured RNG.
    """
    if len(probs) == 0: raise ValueError("cannot sample from an empty probability vector")

    total = 0.0
    clean: List[float] = []
    for p in probs:
        val = float(p)
        if val > 0.0 and np.isfinite(val):
            clean.append(val)
            total += val
        else:
            clean.append(0.0)

    r = rng if rng is not None else random
    if total <= 0.0: return r.randrange(len(clean))

    threshold = r.random() * total
    cumulative = 0.0
    for idx, p in enumerate(clean):
        cumulative += p
        if threshold <= cumulative: return idx
    return len(clean) - 1


def _soft_value(q_vals: Sequence[float], temperature) -> float:
    """Numerically stable soft value without per-state NumPy allocations."""
    if len(q_vals) == 0: return 0.0
    temp = max(float(temperature), 1e-8)
    vals = [float(q) for q in q_vals]
    max_q = max(vals)
    z = sum(math.exp((q - max_q) / temp) for q in vals)
    return max_q + temp * math.log(max(z, 1e-12))


def _softmax_probs(q_vals: Sequence[float], temperature) -> List[float]:
    """Numerically stable softmax without using NumPy's exp/reduction path."""
    if len(q_vals) == 0: return []
    temp = max(float(temperature), 1e-8)
    vals = [float(q) for q in q_vals]
    max_q = max(vals)
    weights = [math.exp((q - max_q) / temp) for q in vals]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        uniform = 1.0 / len(weights)
        return [uniform] * len(weights)
    return [w / total for w in weights]


@dataclass
class Config:
    """Single source of truth for every tunable knob."""
    # reproducibility
    seed: int = 1337
    # decay manager
    decay_init:         float = 0.1
    decay_horizon_init: int   = 10              # Initial retention horizon in completed sessions. The adaptive controller uses max(decay_horizon_init, recent reuse gaps).
    prune_threshold:    float = 1e-9
    max_global_rate:    float = 0.5              # Cap on session-level decay: when the adaptive controller calls for a global rate above this, it is clipped to this value. This prevents catastrophic forgetting under large libraries when the reuse gap signal is noisy.
    min_global_rate:    float = 0.02             # Floor on session-level decay: unpinned variants cannot linger forever under large libraries.
    max_decay_horizon:  int   = 50               # 0 disables the cap; default matches min_global_rate.
    mwr_window:         int   = 30               # mwr_window: number of recent exact-variant reuse gap samples retained by the adaptive base-rate controller. Active repeats and post-pruning restores are both cadence signals; the gap value itself may be larger than this window.
    size_rescale_exponent:      float = 0.5      # Sublinear active-count rescaling. Setting it to 1.0 gives strict active-count scaling; 0.0 disables active-set rescale.
    protect_latest_preference:  bool  = True     # Experimental condition: protect the latest preference variant of each recipe from decay/pruning. This is enabled for the main adaptive agent only; baselines explicitly disable it.
    pref_transfer_alpha:        float = 0.6      # blend weight for preference signal in cross-recipe transfer
    posterior_switch_margin:     float = 0.30   # min log-ratio for recipe identity switch
    posterior_switch_agreement:  int   = 3      # consecutive steps new argmax must win
    posterior_switch_min_confidence: float = 0.55
    posterior_switch_min_gap:        float = 0.05
    # disambiguator
    jaccard_threshold:          float = 0.95
    ordering_unmatched_penalty: float = 0.5
    # ensemble
    ensemble_alpha: float = 0.42
    ensemble_beta:  float = 0.24
    markov_order:   int   = 3
    prob_floor:     float = 1e-6
    # MaxEnt IRL 2.0
    maxent_gamma:           float = 0.9
    maxent_temperature:     float = 0.5
    maxent_learning_rate:   float = 0.05
    maxent_l2:              float = 0.01
    maxent_mc_rollouts:     int   = 150
    maxent_mc_horizon:      int   = 45   # max steps per MC rollout; should exceed longest recipe
    maxent_iters_cold:      int   = 100
    maxent_iters_warm:      int   = 40
    maxent_valid_action_expansion: bool = True  # Add cached state-action candidates from observed action effects, so prediction is over plausible valid frontiers rather than only demo edges.
    # variant memory
    max_variants_per_recipe: int = 7
    # agent / retraining
    topk:       int     = 1
    irl_weight: float   = 1.0
    # EWC (model Fisher-diagonal) regularisation weight
    ewc_lambda: float   = 0.4
    ewc_fisher_clip: float = 100.0
    # Behavior-cloning baselines
    bc_learning_rate:   float = 0.1
    bc_l2:              float = 1e-4
    bc_history_len:     int   = 3
    bc_history_bins:    int   = 64
    bc_epochs_cold:     int   = 120
    bc_epochs_warm:     int   = 60
    # Experience replay baselines. er_buffer_size is the reservoir capacity;
    # er_batch_size is the number of replay trajectories drawn at each retrain.
    # Defaults replay the full reservoir so settled evaluation is not dominated
    # by vocabulary dropout. Recency-prioritized ER uses the same capacity/head
    # with a blended recency/uniform sampler.
    er_buffer_size: int = 256
    er_batch_size:  int = 256
    er_recency_alpha: float = 1.0
    er_uniform_mix: float = 0.05
    # narration
    verbose: bool = True
    # diagnostic profiling: when True, the agent and disambiguator accumulate per-event call counts + wall_s into self.profile / self.disambig.profile. Off by default; negligible overhead when off (one bool check per wrap).
    profile: bool = False
    # Posterior expert-combination weights and temperatures. Each factor is
    # normalized into an expert distribution before the alpha weights combine
    # them, so alpha values are interpretable across differently-scaled factors.
    posterior_alpha_recipe:      float = 1.0
    posterior_alpha_pref:        float = 1.5
    posterior_alpha_memory:      float = 0.5
    posterior_temperature_recipe: float = 1.0
    posterior_temperature_pref:   float = 1.0
    posterior_temperature_memory: float = 1.0
    posterior_global_temperature: float = 1.0
    memory_prior_floor:          float = 1e-6
    active_prior_floor:          float = 0.05
    pruned_prior_initial:        float = 0.50
    pruned_prior_half_life:      float = 10.0
    absent_prior:                float = 0.10
    recipe_match_token_weight:      float = 0.70
    recipe_match_precedence_weight: float = 0.30
    phase_score_retrieve_boost:      float = 5.0
    phase_score_prep_boost:          float = 4.0
    phase_score_early_add_suppress:  float = 0.18
    phase_score_prep_done_boost:     float = 3.0
    phase_score_early_cook_suppress: float = 0.20
    phase_score_cleanup_early_supp:  float = 0.20
    phase_score_cleanup_late_boost:  float = 2.0
    phase_score_threshold:           float = 0.65
    phase_score_cleanup_eager_threshold: float = 0.35
    phase_score_position_base:       float = 0.85
    phase_score_position_match_weight: float = 0.30
    phase_score_container_base:      float = 0.65
    phase_score_container_lead_weight: float = 0.70
    phase_score_max_clamp:           float = 6.0
    phase_score_strength:            float = 1.0
    recipe_frontier_align_weight: float = 0.45  # Remaining mass goes to recipe-level candidates, enabling cross-recipe preference transfer beyond exact replay.
    recipe_frontier_transfer_align_weight: float = 0.0  # When a pref prototype has not seen this recipe, suppress exact-variant alignment so the recipe prototype exposes reorderable candidates.
    # Preference-conditioned next-action gate.
    posterior_action_confidence_threshold: float = 0.20
    posterior_action_topk:          int     = 16
    posterior_assist_strength:      float   = 0.70
    posterior_assist_strength_min:  float   = 0.15
    posterior_assist_strength_max:  float   = 0.90
    posterior_assist_agreement_bonus: float = 0.15
    posterior_assist_disagreement_penalty: float = 0.45
    ensemble_fallback_assist_threshold: float = 0.35
    locked_variant_action_boost: float = 0.70
    # The action gate operates on a calibrated score, not just raw max action
    # probability. Temperature < 1 compensates for posterior-marginal dilution;
    # margin and entropy terms reward decisive action marginals.
    posterior_action_gate_temperature: float = 0.85
    posterior_action_gate_margin_weight: float = 0.15
    posterior_action_gate_entropy_weight: float = 0.10
    posterior_action_margin_threshold: float = 0.06
    posterior_action_margin_min_confidence: float = 0.10
    posterior_action_entropy_threshold: float = 0.85
    online_new_recipe_min_prefix:   int     = 3
    online_new_recipe_partial_threshold: float = 0.30
    min_classify_length: int = 6
    online_unknown_confirm_streak: int = 3
    online_commit_full_threshold: float = 0.75
    online_commit_tentative_threshold: float = 0.45
    provisional_commit_weight: float = 0.20
    provisional_confirm_window: int = 5
    provisional_confirm_top1: float = 0.60
    provisional_min_recipe_jaccard: float = 0.95
    # Ablation switches. Each disables one component of the preference-conditioned pathway so the experiment matrix can attribute gains to specific components. All default to False (full system).
    ablation_disable_preference_head:       bool = False
    ablation_disable_recipe_prototype:      bool = False
    ablation_disable_posterior:             bool = False
    ablation_disable_pruned_memory_prior:   bool = False
    # Oracle leakage upper bounds. Clearly labelled, not deployable. These only take effect when the harness routes through ``predict_with_oracle``; they do NOT change the unconditioned signature of predict_next_tokens.
    ablation_oracle_preference_label:       bool = False
    ablation_oracle_recipe_and_preference_label: bool = False

    def __post_init__(self) -> None:
        # Per-instance RNGs derived from `seed`. Using these (instead of the `np.random` and `random` module globals) is the contract that makes parallel-seed runs reproducible: each Config gets its own bit-stream. Field-set bypass keeps the dataclass usable without `frozen=False` semantics.
        object.__setattr__(self, "rng",  np.random.default_rng(int(self.seed)))
        object.__setattr__(self, "prng", random.Random(int(self.seed)))


DEFAULT_CONFIG = Config()


def create_state_action_mappings(demonstrations: Sequence[Trajectory], unique_actions: Optional[Sequence[str]] = None):
    """State/action index maps shared by all heads."""
    state_to_idx: Dict[State, int] = {}
    for traj in demonstrations:
        for state, _ in traj:
            if state not in state_to_idx:   state_to_idx[state] = len(state_to_idx)
    idx_to_state = {idx: state for state, idx in state_to_idx.items()}

    if unique_actions is None:              action_set = sorted({a for traj in demonstrations for _, a in traj})
    else:                                   action_set = list(unique_actions)
    action_to_idx = {action: idx for idx, action in enumerate(action_set)}
    idx_to_action = {idx: action for action, idx in action_to_idx.items()}
    return state_to_idx, idx_to_state, action_to_idx, idx_to_action


def _safe_feat(state: np.ndarray, key: str) -> int:
    return int(state[_FEAT[key]]) if key in _FEAT else 0


@dataclass
class WelfordFeatureNormalizer:
    """Streaming per-feature standardizer over observed symbolic states."""

    count: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:     return
        x = np.asarray(batch, dtype=np.float32)
        if x.ndim != 2:         raise ValueError("feature batch must be 2-D")
        batch_count = int(x.shape[0])
        batch_mean = x.mean(axis=0)
        centered = x - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0 or self.mean is None or self.m2 is None:
            self.count = batch_count
            self.mean = batch_mean.astype(np.float32)
            self.m2 = batch_m2.astype(np.float32)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = (self.mean + delta * (batch_count / max(total, 1))).astype(np.float32)
        self.m2 = (self.m2 + batch_m2 + (delta * delta) * (self.count * batch_count / max(total, 1))).astype(np.float32)
        self.count = total

    @property
    def variance(self) -> np.ndarray:
        if self.mean is None or self.m2 is None:    return np.ones(0, dtype=np.float32)
        if self.count < 2:                          return np.ones_like(self.mean, dtype=np.float32)
        return np.maximum(self.m2 / float(self.count - 1), 1e-8).astype(np.float32)

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


def create_feature_matrix_2_0(idx_to_state: Dict[int, State], col_min_known: Optional[np.ndarray] = None, col_max_known: Optional[np.ndarray] = None, normalizer: Optional[WelfordFeatureNormalizer] = None, update_normalizer = False):
    """Fixed-dimensional engineered feature matrix over symbolic states. `normalizer` uses streaming Welford statistics. When no normalizer is supplied, `col_min_known` and `col_max_known` are interpreted as mean/scale statistics."""
    key_ingredients = ["tomato", "onion", "mushroom", "rice", "meat", "chicken", "fish", "egg", "banana", "strawberries", "lettuce", "cheese", "garlic", "yoghurt", "milk"]
    key_containers  = ["pot", "pan", "plate", "bowl"]
    key_locs        = ["prep_station", "cooking_station", "plating_station", "blending_station"]

    feats = []
    for idx in range(len(idx_to_state)):
        s = np.asarray(idx_to_state[idx], dtype=np.int32)
        f: List[float] = []

        # 1) Item density by location.
        for loc in LOCATIONS:
            f.append(sum(_safe_feat(s, f"{item}_at_{loc}") for item in ITEMS))
        # 2) Container-location anchors + composite counts.
        for container, loc in [
            ("pot", "cooking_station"), ("pot", "washing_station"), ("pan", "cooking_station"), ("pan", "washing_station"),
            ("plate", "plating_station"), ("plate", "serving_station"), ("plate", "washing_station"),
            ("bowl", "prep_station"), ("bowl", "cooking_station"), ("bowl", "plating_station"), ("bowl", "washing_station"),
            ("glass", "blending_station"), ("glass", "serving_station"), ("glass", "washing_station"),
            ("measuring_cup", "blending_station"), ("measuring_cup", "washing_station")]:  f.append(_safe_feat(s, f"{container}_at_{loc}"))
        f.append(_safe_feat(s, "pot_at_cooking_station") + _safe_feat(s, "pan_at_cooking_station"))
        f.append(sum(_safe_feat(s, f"{item}_at_prep_station") for item in ITEMS))
        f.append(sum(_safe_feat(s, f"{item}_at_blending_station") for item in ITEMS))
        # 3) Containment aggregates.
        total_contained = sum(_safe_feat(s, f"{c}_contains_{ing}") for c in CONTAINERS for ing in INGREDIENTS)
        f.append(total_contained)
        for c in CONTAINERS: f.append(sum(_safe_feat(s, f"{c}_contains_{ing}") for ing in INGREDIENTS))
        f.append(sum(_safe_feat(s, f"{c}_contains_mixture") for c in CONTAINERS))
        # 4) Processing features.
        cut_vals = [_safe_feat(s, f"{item}_cut") for item in CUTTABLES]
        cooked_vals = [_safe_feat(s, f"{item}_cooked") for item in COOKABLES]
        f.extend(cut_vals)
        f.extend(cooked_vals)
        total_cut = sum(cut_vals)
        total_grated = sum(_safe_feat(s, f"{item}_grated") for item in GRATABLE)
        total_cooked = sum(cooked_vals)
        total_seasoned = sum(_safe_feat(s, f"{item}_seasoned") for item in INGREDIENTS)
        total_washed = sum(_safe_feat(s, f"{item}_washed") for item in ITEMS)
        f.extend([total_cut, total_grated, total_cooked, total_seasoned, total_washed])
        for ing in ["chicken", "fish", "mixture"]:
            f.append(_safe_feat(s, f"{ing}_seasoned"))
        # 5) Tool and serving status.
        for key in ["stove_on", "sink_on", "blender_on", "plate_served"]: f.append(_safe_feat(s, key))
        # 6) Key ingredient x location grid.
        for ing in key_ingredients:
            for loc in key_locs: f.append(_safe_feat(s, f"{ing}_at_{loc}"))
        # 7) Key ingredient containment in key containers.
        for c in key_containers:
            for ing in key_ingredients: f.append(_safe_feat(s, f"{c}_contains_{ing}"))
        # 8) Seasoning signatures.
        for seasoning in SEASONINGS: f.append(sum(_safe_feat(s, f"{ing}_seasoned_with_{seasoning}") for ing in key_ingredients))
        # 9) Interaction composites.
        f.append((_safe_feat(s, "stove_on")) * (_safe_feat(s, "pot_at_cooking_station") + _safe_feat(s, "pan_at_cooking_station")))
        f.append((_safe_feat(s, "blender_on")) * _safe_feat(s, "glass_at_blending_station"))
        f.append((_safe_feat(s, "sink_on")) * (_safe_feat(s, "plate_at_washing_station") + _safe_feat(s, "bowl_at_washing_station")))
        feats.append(f)

    raw = np.asarray(feats, dtype=np.float32)
    if raw.size == 0: return raw, np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)

    if normalizer is not None:
        if update_normalizer: normalizer.update(raw)
        fm = normalizer.transform(raw)
        mean = normalizer.mean if normalizer.mean is not None else np.zeros(raw.shape[1], dtype=np.float32)
        scale = normalizer.scale if normalizer.mean is not None else np.ones(raw.shape[1], dtype=np.float32)
        return fm.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)

    if col_min_known is None or col_max_known is None:
        local = WelfordFeatureNormalizer()
        local.update(raw)
        fm = local.transform(raw)
        mean = local.mean if local.mean is not None else np.zeros(raw.shape[1], dtype=np.float32)
        scale = local.scale if local.mean is not None else np.ones(raw.shape[1], dtype=np.float32)
    else:
        mean = np.asarray(col_min_known, dtype=np.float32)
        scale = np.asarray(col_max_known, dtype=np.float32)
        denom = np.where(np.abs(scale) > 1e-8, scale, 1.0)
        fm = (raw - mean) / denom
    return fm.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def max_ent_irl_2(demonstrations: Sequence[Trajectory], feature_matrix: np.ndarray, state_to_idx: Dict[State, int], action_to_idx: Dict[str, int], cfg: Config, init_weights: Optional[np.ndarray] = None, demo_weights: Optional[Sequence[float]] = None, 
                  ewc_theta_star: Optional[np.ndarray] = None, ewc_fisher: Optional[np.ndarray] = None, extra_transitions: Optional[Dict[Tuple[State, str], State]] = None):
    """Importance-weighted MaxEnt IRL with demo-augmented MDP and warm starts.
    When `ewc_theta_star` and `ewc_fisher` are provided, an in-loop Fisher-weighted quadratic penalty `(ewc_lambda/2) * Σ_i F_i (θ_i - θ*_i)^2` is added to the ascent gradient at every iteration (Kirkpatrick et al.). The `maxent_l2` term is kept as a separate isotropic prior."""
    n_states, n_features = feature_matrix.shape
    n_actions = len(action_to_idx)
    if n_states == 0 or n_features == 0 or n_actions == 0:
        stats = {
            "model_family": "maxent_irl",
            "estimated_flops": 0.0,
            "n_states": float(n_states),
            "n_features": float(n_features),
            "n_actions": float(n_actions),
            "iterations_run": 0.0,
            "warm_start": 1.0 if init_weights is not None else 0.0,
        }
        return np.zeros(n_features, dtype=np.float32), np.zeros(n_states), np.zeros(n_states), {}, {}, stats

    weights = np.ones(len(demonstrations), dtype=np.float32) if demo_weights is None else np.asarray(demo_weights, dtype=np.float32)
    w_sum = float(weights.sum())
    if w_sum <= 0:
        weights = np.ones(len(demonstrations), dtype=np.float32)
        w_sum = float(weights.sum())
    demo_probs = weights / max(w_sum, 1e-9)

    if init_weights is not None and init_weights.shape[0] == n_features:    reward_weights = init_weights.astype(np.float32).copy()
    else:                                                                   reward_weights = (cfg.rng.standard_normal(n_features).astype(np.float32) * 0.1)

    # Build observed transition dynamics.
    state_action_pairs = set()
    transition_model: Dict[Tuple[int, int], int] = {}
    s_to_actions: Dict[int, List[int]] = {}
    for traj in demonstrations:
        for i, (state, action) in enumerate(traj):
            s_idx = state_to_idx[state]
            a_idx = action_to_idx[action]
            state_action_pairs.add((s_idx, a_idx))
            s_to_actions.setdefault(s_idx, [])
            if a_idx not in s_to_actions[s_idx]:    s_to_actions[s_idx].append(a_idx)
            if i < len(traj) - 1:                   transition_model[(s_idx, a_idx)] = state_to_idx[traj[i + 1][0]]
    for (state, action), ns in (extra_transitions or {}).items():
        if state not in state_to_idx or ns not in state_to_idx or action not in action_to_idx or action == "stop": continue
        s_idx = state_to_idx[state]
        a_idx = action_to_idx[action]
        ns_idx = state_to_idx[ns]
        state_action_pairs.add((s_idx, a_idx))
        transition_model[(s_idx, a_idx)] = ns_idx
        s_to_actions.setdefault(s_idx, [])
        if a_idx not in s_to_actions[s_idx]: s_to_actions[s_idx].append(a_idx)

    # Demo-augmented MDP: virtual deviate action to null state.
    deviate_a = n_actions
    null_s = n_states
    feat_aug = np.vstack([feature_matrix, np.zeros((1, n_features), dtype=np.float32)])
    n_states_aug = n_states + 1
    for s_idx in list(s_to_actions.keys()):
        transition_model[(s_idx, deviate_a)] = null_s
        s_to_actions[s_idx] = s_to_actions[s_idx] + [deviate_a]

    # Weighted empirical discounted feature expectations. Normalising by the number of demonstrations (NOT by w_sum) means the gradient magnitude scales with the *absolute* weight scale: a demo at weight 0.5 contributes half as much to the gradient as one at weight 1.0.
    # That is the "true loss weighting" contract; without this normalization change a uniform multiplicative rescaling of weights cancels out.
    empirical_fe = np.zeros(n_features, dtype=np.float32)
    demo_starts = [state_to_idx[traj[0][0]] for traj in demonstrations]
    gamma = cfg.maxent_gamma
    for i, traj in enumerate(demonstrations):
        traj_fe = np.zeros(n_features, dtype=np.float32)
        for t, (state, _) in enumerate(traj):   traj_fe += (gamma ** t) * feature_matrix[state_to_idx[state]]
        empirical_fe += weights[i] * traj_fe
    empirical_fe /= max(float(len(demonstrations)), 1.0)

    temperature = cfg.maxent_temperature
    learning_rate = cfg.maxent_learning_rate
    l2 = cfg.maxent_l2
    mc_count = max(1, int(cfg.maxent_mc_rollouts))
    n_iterations = cfg.maxent_iters_warm if init_weights is not None else cfg.maxent_iters_cold

    q_values: Dict[Tuple[int, int], float] = {}
    rewards     = feat_aug @ reward_weights
    values      = rewards.copy()
    best_diff   = float("inf")
    no_improve  = 0
    best_weights    = reward_weights.copy()
    momentum        = np.zeros(n_features, dtype=np.float32)
    momentum_beta   = 0.9
    iterations_run = 0
    total_vi_sweeps = 0
    mc_rollouts_run = 0
    mc_steps = 0
    

    for iteration in range(n_iterations):
        iterations_run = iteration + 1
        lr = learning_rate * (0.97 ** (no_improve // 20))
        rewards = feat_aug @ reward_weights

        # Soft Bellman VI.
        for _ in range(50):
            total_vi_sweeps += 1
            new_values = rewards.copy()
            for (s_idx, a_idx) in state_action_pairs:
                ns = transition_model.get((s_idx, a_idx))
                q_values[(s_idx, a_idx)] = rewards[s_idx] + (gamma * values[ns] if ns is not None else 0.0)
            for s_idx in s_to_actions:
                if (s_idx, deviate_a) not in q_values:      q_values[(s_idx, deviate_a)] = rewards[s_idx] + gamma * values[null_s]
            for s_idx, a_list in s_to_actions.items():
                avail = [q_values.get((s_idx, a), rewards[s_idx]) for a in a_list]
                new_values[s_idx] = _soft_value(avail, temperature)
            max_delta = max(abs(float(new_values[i]) - float(values[i])) for i in range(len(new_values)))
            if max_delta < 1e-6:
                values = new_values
                break
            values = new_values

        # Policy table.
        policy_table: Dict[int, Tuple[List[int], List[float]]] = {}
        for s_idx, a_list in s_to_actions.items():
            sq = [q_values.get((s_idx, a), rewards[s_idx]) for a in a_list]
            probs = _softmax_probs(sq, temperature)
            policy_table[s_idx] = (a_list, probs)

        # Expected discounted features via weighted MC rollouts.
        expected_fc = np.zeros(n_features, dtype=np.float32)
        for _ in range(mc_count):
            mc_rollouts_run += 1
            demo_idx = _sample_categorical(demo_probs, rng=cfg.prng)
            s_idx = demo_starts[demo_idx]
            traj_feat = np.zeros(n_features, dtype=np.float32)
            for t in range(cfg.maxent_mc_horizon):
                mc_steps += 1
                if s_idx < n_states:                            traj_feat += (gamma ** t) * feature_matrix[s_idx]
                if s_idx not in policy_table:                   break
                a_list, probs = policy_table[s_idx]
                a_idx = a_list[_sample_categorical(probs, rng=cfg.prng)]
                if a_idx == deviate_a:                          break
                if (s_idx, a_idx) not in transition_model:      break
                s_idx = transition_model[(s_idx, a_idx)]
            expected_fc += traj_feat
        expected_fc /= max(float(mc_count), 1.0)

        gradient = empirical_fe - expected_fc
        gradient -= l2 * reward_weights
        if ewc_theta_star is not None and ewc_fisher is not None:
            n = min(reward_weights.shape[0], int(ewc_theta_star.shape[0]), int(ewc_fisher.shape[0]))
            if n > 0:
                lam = float(cfg.ewc_lambda)
                penalty = np.zeros_like(reward_weights)
                penalty[:n] = (lam * ewc_fisher[:n].astype(np.float32) * (reward_weights[:n] - ewc_theta_star[:n].astype(np.float32)))
                gradient -= penalty
        grad_norm = float(np.linalg.norm(gradient))
        momentum = momentum_beta * momentum + (1.0 - momentum_beta) * gradient
        reward_weights += lr * momentum

        if grad_norm < best_diff:
            best_diff = grad_norm
            best_weights = reward_weights.copy()
            no_improve = 0
        else:                   no_improve += 1
        if no_improve > 60:     break

    # Final VI with best weights for prediction.
    best_rewards = feat_aug @ best_weights
    best_values = best_rewards.copy()
    best_q: Dict[Tuple[int, int], float] = {}
    final_vi_sweeps = 0
    for _ in range(50):
        final_vi_sweeps += 1
        new_vals = best_rewards.copy()
        for (s_idx, a_idx) in state_action_pairs:
            ns = transition_model.get((s_idx, a_idx))
            best_q[(s_idx, a_idx)] = best_rewards[s_idx] + (gamma * best_values[ns] if ns is not None else best_rewards[s_idx])
        for s_idx, a_list in s_to_actions.items():
            avail = [best_q.get((s_idx, a), best_rewards[s_idx]) for a in a_list]
            new_vals[s_idx] = _soft_value(avail, temperature)
        max_delta = max(abs(float(new_vals[i]) - float(best_values[i])) for i in range(len(new_vals)))
        if max_delta < 1e-6:
            best_values = new_vals
            break
        best_values = new_vals

    clean_s_to_actions = {s: [a for a in a_list if a != deviate_a] for s, a_list in s_to_actions.items()}
    best_q_clean = {(s, a): v for (s, a), v in best_q.items() if a != deviate_a}
    fit_stats = {
        "model_family": "maxent_irl",
        "n_demonstrations": float(len(demonstrations)),
        "n_demo_state_visits": float(sum(len(traj) for traj in demonstrations)),
        "n_states": float(n_states),
        "n_states_augmented": float(n_states_aug),
        "n_features": float(n_features),
        "n_actions": float(n_actions),
        "n_state_action_pairs": float(len(state_action_pairs)),
        "n_policy_states": float(len(s_to_actions)),
        "n_policy_state_actions": float(sum(len(v) for v in s_to_actions.values())),
        "n_transition_edges": float(len(transition_model)),
        "iterations_requested": float(n_iterations),
        "iterations_run": float(iterations_run),
        "vi_sweeps": float(total_vi_sweeps),
        "final_vi_sweeps": float(final_vi_sweeps),
        "mc_rollouts_requested_per_iteration": float(mc_count),
        "mc_rollouts_run": float(mc_rollouts_run),
        "mc_steps": float(mc_steps),
        "warm_start": 1.0 if init_weights is not None else 0.0,
        "ewc_enabled": 1.0 if ewc_theta_star is not None and ewc_fisher is not None else 0.0,
    }
    fit_stats["estimated_flops"] = _maxent_fit_flop_estimate(fit_stats)
    return best_weights, best_rewards[:n_states], best_values[:n_states], best_q_clean, clean_s_to_actions, fit_stats


def predict_action_2(state: State, reward_weights: np.ndarray, feature_matrix: np.ndarray, state_to_idx: Dict[State, int], action_to_idx: Dict[str, int], idx_to_action: Dict[int, str], transition_model: Dict[Tuple[int, int], int], col_min: Optional[np.ndarray] = None, 
                     col_max: Optional[np.ndarray] = None, normalizer: Optional[WelfordFeatureNormalizer] = None, temperature = 0.5, gamma = 0.9, q_table: Optional[Dict[Tuple[int, int], float]] = None, s_to_actions: Optional[Dict[int, List[int]]] = None):
    """Predict next action and distribution from current symbolic state."""
    if state not in state_to_idx:
        query_fm, _, _ = create_feature_matrix_2_0({0: tuple(state)}, col_min_known=col_min, col_max_known=col_max, normalizer=normalizer)
        if query_fm.size == 0 or feature_matrix.size == 0: return None, {}
        query = query_fm[0].astype(np.float32)
        norms = np.linalg.norm(feature_matrix, axis=1)
        q_norm = float(np.linalg.norm(query))
        if q_norm > 0 and np.any(norms > 0):
            sims = feature_matrix @ query / (norms * q_norm + 1e-10)
            s_idx = int(np.argmax(sims))
        else:   s_idx = 0
    else:       s_idx = state_to_idx[state]

    if q_table is not None and s_to_actions is not None and s_idx in s_to_actions:      candidates = [(a, q_table[(s_idx, a)]) for a in s_to_actions[s_idx] if (s_idx, a) in q_table]
    elif q_table is not None:                                                           candidates = [(a, q_table[(s_idx, a)]) for a in range(len(action_to_idx)) if (s_idx, a) in q_table]
    else: rewards = feature_matrix @ reward_weights;                                    candidates = [(a, float(rewards[s_idx]) + gamma * float(rewards[transition_model[(s_idx, a)]])) for a in range(len(action_to_idx)) if (s_idx, a) in transition_model]
    if not candidates: return None, {}

    valid_actions, q_vals = zip(*candidates)
    probs = _softmax_probs(q_vals, temperature)
    dist = {idx_to_action[a]: p for a, p in zip(valid_actions, probs)}
    dist.pop("stop", None)
    if not dist: return None, {}
    pred = max(dist.items(), key=lambda kv: kv[1])[0]
    return pred, dist


def _normalise_dist(dist: Dict[str, float], floor) -> Dict[str, float]:
    if not dist:        return {}
    clipped = {a: max(float(p), floor) for a, p in dist.items() if a != "stop"}
    if not clipped:     return {}
    s = sum(clipped.values())
    if s <= 0:
        u = 1.0 / len(clipped)
        return {a: u for a in clipped}
    return {a: p / s for a, p in clipped.items()}


def _state_delta(before: State, after: State) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    b = np.asarray(before, dtype=np.int8)
    a = np.asarray(after, dtype=np.int8)
    return tuple(np.where((b == 0) & (a == 1))[0].astype(int).tolist()), tuple(np.where((b == 1) & (a == 0))[0].astype(int).tolist())


def _apply_delta(state: State, delta: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> State:
    s = np.asarray(state, dtype=np.int8).copy()
    on, off = delta
    for i in off:
        if 0 <= i < len(s): s[i] = 0
    for i in on:
        if 0 <= i < len(s): s[i] = 1
    return tuple(s.astype(int).tolist())


def _maxent_fit_flop_estimate(stats: Dict[str, Any]) -> float:
    """Estimate scalar floating-point work from actual MaxEnt fit loop counts."""
    n_features = float(stats.get("n_features", 0.0))
    n_states_aug = float(stats.get("n_states_augmented", 0.0))
    n_state_actions = float(stats.get("n_state_action_pairs", 0.0))
    n_policy_edges = float(stats.get("n_policy_state_actions", 0.0))
    n_demo_visits = float(stats.get("n_demo_state_visits", 0.0))
    iterations = float(stats.get("iterations_run", 0.0))
    vi_sweeps = float(stats.get("vi_sweeps", 0.0) + stats.get("final_vi_sweeps", 0.0))
    mc_steps = float(stats.get("mc_steps", 0.0))
    ewc_enabled = bool(stats.get("ewc_enabled", 0.0))

    empirical_feature_work = 2.0 * n_demo_visits * n_features
    reward_dot_work = 2.0 * n_states_aug * n_features * max(1.0, iterations + 1.0)
    # Bellman and policy work is mostly scalar arithmetic plus exp/log calls.
    # Count it separately from feature-vector work so the estimate is not
    # multiplied by nonexistent dense action-feature products.
    vi_scalar_work = vi_sweeps * ((4.0 * n_state_actions) + (8.0 * n_policy_edges))
    policy_scalar_work = iterations * 8.0 * n_policy_edges
    rollout_feature_work = 2.0 * mc_steps * n_features
    gradient_work = iterations * (8.0 * n_features + (4.0 * n_features if ewc_enabled else 0.0))
    return float(empirical_feature_work + reward_dot_work + vi_scalar_work + policy_scalar_work + rollout_feature_work + gradient_work)


class MaxEntIRL2:
    """State-feature MaxEnt IRL with warm-started retrains."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.theta: Optional[np.ndarray] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.normalizer = WelfordFeatureNormalizer()
        self.col_min: Optional[np.ndarray] = None  # Welford mean
        self.col_max: Optional[np.ndarray] = None  # Welford scale
        self.state_to_idx: Dict[State, int] = {}
        self.idx_to_state: Dict[int, State] = {}
        self.action_to_idx: Dict[str, int] = {}
        self.idx_to_action: Dict[int, str] = {}
        self.transition_model: Dict[Tuple[int, int], int] = {}
        self.values: Optional[np.ndarray] = None
        self.q_table: Dict[Tuple[int, int], float] = {}
        self.s_to_actions: Dict[int, List[int]] = {}
        self.last_fit_stats: Dict[str, Any] = {}

    def reset(self) -> None:
        self.theta = None
        self.feature_matrix = None
        self.normalizer = WelfordFeatureNormalizer()
        self.col_min = None
        self.col_max = None
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.action_to_idx = {}
        self.idx_to_action = {}
        self.transition_model = {}
        self.values = None
        self.q_table = {}
        self.s_to_actions = {}
        self.last_fit_stats = {"model_family": "maxent_irl", "estimated_flops": 0.0}

    def _expanded_state_actions(self, demonstrations: Sequence[Trajectory]) -> Dict[Tuple[State, str], State]:
        """Cached valid-action approximation from observed action effects."""
        if not getattr(self.cfg, "maxent_valid_action_expansion", True): return {}
        deltas: Dict[str, Counter] = defaultdict(Counter)
        states: set = set()
        for traj in demonstrations:
            for i in range(len(traj) - 1):
                s, a = traj[i]
                ns = traj[i + 1][0]
                states.add(s)
                if a != "stop": deltas[a][_state_delta(s, ns)] += 1
            if traj: states.add(traj[-1][0])
        action_delta = {a: c.most_common(1)[0][0] for a, c in deltas.items() if c}
        out: Dict[Tuple[State, str], State] = {}
        for s in states:
            for a, delta in action_delta.items():
                ns = _apply_delta(s, delta)
                if ns != s: out[(s, a)] = ns
        return out

    def fit(self, demonstrations: Sequence[Trajectory], demo_weights: Optional[Sequence[float]] = None, ewc_theta_star: Optional[np.ndarray] = None, ewc_fisher: Optional[np.ndarray] = None) -> None:
        if not demonstrations: 
            self.reset()
            return
        unique_actions = sorted({a for traj in demonstrations for _, a in traj})
        self.state_to_idx, self.idx_to_state, self.action_to_idx, self.idx_to_action = create_state_action_mappings(demonstrations, unique_actions=unique_actions)
        extra_transitions = self._expanded_state_actions(demonstrations)
        for ns in extra_transitions.values():
            if ns not in self.state_to_idx:
                idx = len(self.state_to_idx)
                self.state_to_idx[ns] = idx
                self.idx_to_state[idx] = ns
        # Welford normalizer must be reset every fit. Cumulative means/scales across retrains let pruned-recipe states permanently bias the feature normalisation, which makes `pruned_influence_audit` a tautology (the active-only fit shares the same normalizer).
        self.normalizer = WelfordFeatureNormalizer()
        fm, col_min, col_max = create_feature_matrix_2_0(self.idx_to_state, normalizer=self.normalizer, update_normalizer=True)
        init = self.theta if self.theta is not None and self.theta.shape[0] == fm.shape[1] else None
        weights, _, values, q_table, s_to_actions, fit_stats = max_ent_irl_2(demonstrations, fm, self.state_to_idx, self.action_to_idx, cfg=self.cfg, init_weights=init, demo_weights=demo_weights, ewc_theta_star=ewc_theta_star, ewc_fisher=ewc_fisher, extra_transitions=extra_transitions)
        # Deterministic transition table.
        transition: Dict[Tuple[int, int], int] = {}
        for traj in demonstrations:
            for i in range(len(traj) - 1):
                s, a = traj[i]
                ns = traj[i + 1][0]
                if s in self.state_to_idx and ns in self.state_to_idx and a in self.action_to_idx: transition[(self.state_to_idx[s], self.action_to_idx[a])] = self.state_to_idx[ns]
        for (s, a), ns in extra_transitions.items():
            if s in self.state_to_idx and ns in self.state_to_idx and a in self.action_to_idx: transition[(self.state_to_idx[s], self.action_to_idx[a])] = self.state_to_idx[ns]

        self.theta = weights.astype(np.float32)
        self.feature_matrix = fm
        self.col_min = col_min
        self.col_max = col_max
        self.transition_model = transition
        self.values = values.astype(np.float32)
        self.q_table = q_table
        self.s_to_actions = s_to_actions
        fit_stats = dict(fit_stats)
        fit_stats["model_family"] = "maxent_irl"
        fit_stats["n_transitions"] = float(sum(max(0, len(traj) - 1) for traj in demonstrations))
        self.last_fit_stats = fit_stats

    def predict(self, state: State) -> Dict[str, float]:
        if self.theta is None or self.feature_matrix is None: return {}
        _, dist = predict_action_2(state=tuple(state),  reward_weights=self.theta,  feature_matrix=self.feature_matrix, state_to_idx=self.state_to_idx, action_to_idx=self.action_to_idx,   idx_to_action=self.idx_to_action,   transition_model=self.transition_model, 
                                   col_min=self.col_min, col_max=self.col_max,      normalizer=self.normalizer,         temperature=self.cfg.maxent_temperature,    gamma=self.cfg.maxent_gamma,    q_table=self.q_table,       s_to_actions=self.s_to_actions)
        return _normalise_dist(dist, self.cfg.prob_floor)

    def fisher_diagonal(self, trajectories: Sequence[Trajectory], weights: Optional[Sequence[float]] = None) -> Optional[np.ndarray]:
        """Diagonal Fisher information for the linear-reward MaxEnt policy. For a softmax policy parameterised as `pi(a|s) ∝ exp(Q(s,a)/T)` with linear reward `R(s) = theta . phi(s)`, the per-state log-likelihood gradient is dominated by `phi(s)` (the reward gradient term carries through the soft Bellman backup). 
        The diagonal Fisher under the data distribution is therefore `E_{s~p}[phi_i(s)^2]`, a standard linear softmax approximation that is cheap to compute and meaningful for EWC's elastic anchor.
        Returns None if the model has not been fit. Per-trajectory weighting respects the same convention as ``fit`` (absolute scale matters: a downweighted trajectory contributes proportionally less to the accumulator).
        """
        if self.theta is None or self.feature_matrix is None:
            return None
        n_features = int(self.theta.shape[0])
        fisher = np.zeros(n_features, dtype=np.float32)
        if weights is None:
            weights = [1.0] * len(trajectories)
        total = 0.0
        for traj, w in zip(trajectories, weights):
            wf = float(w)
            if wf <= 0.0: continue
            for state, _ in traj:
                s_idx = self.state_to_idx.get(state)
                if s_idx is None or s_idx >= self.feature_matrix.shape[0]:  continue
                phi = self.feature_matrix[s_idx]
                fisher += wf * (phi * phi).astype(np.float32)
                total += wf
        if total > 0.0:     fisher /= total
        clip = float(getattr(self.cfg, "ewc_fisher_clip", 0.0))
        if clip > 0.0:
            fisher = np.clip(fisher, 0.0, clip).astype(np.float32)
        return fisher


class _StateNNMixin:
    state_to_idx: Dict[State, int]
    feature_matrix: Optional[np.ndarray]
    col_min: Optional[np.ndarray]
    col_max: Optional[np.ndarray]
    normalizer: Optional[WelfordFeatureNormalizer]
    idx_to_state: Dict[int, State]

    def _nearest_state_idx(self, state: State) -> Optional[int]:
        if state in self.state_to_idx:                                      return self.state_to_idx[state]
        if self.feature_matrix is None or self.feature_matrix.size == 0:    return None
        query_fm, _, _ = create_feature_matrix_2_0({0: tuple(state)}, col_min_known=self.col_min, col_max_known=self.col_max, normalizer=self.normalizer)
        if query_fm.size == 0: return None
        query = query_fm[0]
        norms = np.linalg.norm(self.feature_matrix, axis=1)
        q_norm = float(np.linalg.norm(query))
        if q_norm <= 0 or not np.any(norms > 0): return None
        sims = self.feature_matrix @ query / (norms * q_norm + 1e-10)
        return int(np.argmax(sims))


class NGramMarkov(_StateNNMixin):
    """State-aware weighted n-gram predictor with nearest-state fallback."""

    def __init__(self, order = 3, prob_floor = 1e-6):
        self.order = order
        self.prob_floor = prob_floor
        self.counts_ngram: List[Dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(order + 1)]
        self.counts_state: Dict[int, Counter] = defaultdict(Counter)
        self.vocab: set = set()
        self.state_to_idx: Dict[State, int] = {}
        self.idx_to_state: Dict[int, State] = {}
        self.feature_matrix: Optional[np.ndarray] = None
        self.col_min: Optional[np.ndarray] = None
        self.col_max: Optional[np.ndarray] = None
        self.normalizer: Optional[WelfordFeatureNormalizer] = None

    def fit(self,  demonstrations: Sequence[Trajectory], weights: Optional[Sequence[float]] = None, state_to_idx: Optional[Dict[State, int]] = None, idx_to_state: Optional[Dict[int, State]] = None, feature_matrix: Optional[np.ndarray] = None, col_min: Optional[np.ndarray] = None, col_max: Optional[np.ndarray] = None, normalizer: Optional[WelfordFeatureNormalizer] = None) -> None:
        self.counts_ngram = [defaultdict(Counter) for _ in range(self.order + 1)]
        self.counts_state = defaultdict(Counter)
        self.vocab = set()
        self.state_to_idx = dict(state_to_idx or {})
        self.idx_to_state = dict(idx_to_state or {})
        self.feature_matrix = feature_matrix
        self.col_min = col_min
        self.col_max = col_max
        self.normalizer = normalizer

        if weights is None: weights = [1.0] * len(demonstrations)
        # Relative rehearsal weights are real counts here. Uniform scaling still cancels at prediction, but decayed demos exert proportionally less influence than fresh or pinned demos.
        for traj, w in zip(demonstrations, weights):
            wf = float(w)
            if wf <= 0: continue
            seq = [a for _, a in traj if a != "stop"]
            for i, action in enumerate(seq):
                self.vocab.add(action)
                state = traj[i][0]
                s_idx = self.state_to_idx.get(state)
                if s_idx is not None: self.counts_state[s_idx][action] += wf
                for k in range(self.order + 1):
                    ctx = tuple(seq[max(0, i - k):i])
                    if len(ctx) == k: self.counts_ngram[k][ctx][action] += wf

    def predict(self, state: State, prefix: Sequence[str]) -> Dict[str, float]:
        if not self.vocab:  return {}

        # State-local distribution.
        s_idx = self._nearest_state_idx(tuple(state))
        state_dist: Dict[str, float] = {}
        if s_idx is not None and self.counts_state.get(s_idx):
            c = self.counts_state[s_idx]
            z = float(sum(c.values()))
            state_dist = {a: c[a] / max(z, 1e-9) for a in c}

        # n-gram backoff distribution.
        ngram_dist: Dict[str, float] = {}
        for k in range(min(self.order, len(prefix)), -1, -1):
            ctx = tuple(prefix[-k:]) if k > 0 else ()
            c = self.counts_ngram[k].get(ctx)
            if c:
                z = float(sum(c.values()))
                ngram_dist = {a: c[a] / max(z, 1e-9) for a in c}
                break

        if state_dist and ngram_dist:
            vocab = set(state_dist) | set(ngram_dist)
            mix = { a: (max(state_dist.get(a, self.prob_floor), self.prob_floor) ** 0.6) * (max(ngram_dist.get(a, self.prob_floor), self.prob_floor) ** 0.4) for a in vocab}
            return _normalise_dist(mix, self.prob_floor)
        if state_dist:  return _normalise_dist(state_dist, self.prob_floor)
        if ngram_dist:  return _normalise_dist(ngram_dist, self.prob_floor)
        u = 1.0 / len(self.vocab)
        return {a: u for a in self.vocab}


def ensemble_predict(p_irl: Dict[str, float], p_markov: Dict[str, float], cfg: Config = DEFAULT_CONFIG) -> Dict[str, float]:
    """Fuse the MaxEnt IRL and state-aware n-gram heads as p ∝ p_irl^a · p_markov^b.

    ``a = cfg.ensemble_alpha * cfg.irl_weight`` and ``b = cfg.ensemble_beta`` are
    used **without** normalising to a+b=1.  This preserves the absolute scale of
    each exponent so that:
      - higher |a|+|b| sharpens the combined distribution toward argmax;
      - the ratio a/b sets the relative IRL vs. Markov influence;
      - ``irl_weight`` is a genuine multiplicative scaling of the IRL contribution,
        not a no-op masked by subsequent renormalisation.
    The output distribution is normalised to sum to 1 via the final division by s.
    """
    vocab = set(p_irl) | set(p_markov)
    if not vocab:
        return {}
    floor = cfg.prob_floor
    a = max(0.0, float(cfg.ensemble_alpha) * float(cfg.irl_weight))
    b = max(0.0, float(cfg.ensemble_beta))
    if a <= 0.0 and b <= 0.0:
        a = b = 0.5  # degenerate config — fall back to equal weighting
    out: Dict[str, float] = {}
    for act in vocab:
        pi = max(p_irl.get(act, floor), floor) if p_irl else 1.0
        pm = max(p_markov.get(act, floor), floor) if p_markov else 1.0
        out[act] = (pi ** a) * (pm ** b)
    s = sum(out.values())
    if s <= 0:
        u = 1.0 / len(vocab)
        return {a_: u for a_ in vocab}
    return {a_: p / s for a_, p in out.items()}

def top_k(distribution: Dict[str, float], k = 1) -> List[str]: return [a for a, _ in sorted(distribution.items(), key=lambda kv: -kv[1])[:k]]
