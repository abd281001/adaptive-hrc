"""Ablation baselines and comparison agents for the publication benchmark."""
from __future__ import annotations
from dataclasses import replace
import math
import random
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .adaptive_agent import AdaptiveHRCAgent, MODE_ONLINE
from .memory import Classification, variant_hash
from .models import (Config,    DEFAULT_CONFIG,     Trajectory,     WelfordFeatureNormalizer,       create_feature_matrix_2_0,      create_state_action_mappings)
from .posterior import PreferencePrototypeLearner, RecipePrototypeLearner
State = Tuple[int, ...]


def _symbolic_fit_stats(model_family: str, trajectories: Sequence[Trajectory], estimated_flops: float = 0.0) -> Dict[str, object]:
    n_examples = sum(1 for traj in trajectories for _state, action in traj if action != "stop")
    n_actions = len({action for traj in trajectories for _state, action in traj if action != "stop"})
    return {"model_family": model_family, "estimated_flops": float(estimated_flops), "n_demonstrations": float(len(trajectories)), "n_examples": float(n_examples), "n_actions": float(n_actions)}


# Behavior-cloning support used by BC and replay baselines.
def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:    return logits.astype(np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    return (exp / denom).astype(np.float32)


class BehaviorCloningHead:
    """Linear softmax imitation model over state and hashed prefix features."""
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.normalizer = WelfordFeatureNormalizer()
        self.state_feature_dim: int = 0
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.action_to_idx: Dict[str, int] = {}
        self.idx_to_action: Dict[int, str] = {}
        self.last_fit_stats: Dict[str, object] = {}

    def reset(self) -> None:
        self.weights = None
        self.bias = None
        self.action_to_idx = {}
        self.idx_to_action = {}
        self.last_fit_stats = {"model_family": "behavior_cloning", "estimated_flops": 0.0}

    def _history_dim(self) -> int:
        return max(8, int(self.cfg.bc_history_bins)) * max(1, int(self.cfg.bc_history_len)) + 1

    def _prefix_features(self, prefix: Sequence[str]) -> np.ndarray:
        bins = max(8, int(self.cfg.bc_history_bins))
        history = max(1, int(self.cfg.bc_history_len))
        out = np.zeros(history * bins + 1, dtype=np.float32)
        out[-1] = float(len(prefix))
        for lag in range(1, history + 1):
            if len(prefix) < lag: continue
            token = prefix[-lag]
            key = f"{lag}:{token}".encode("utf-8")
            idx = zlib.crc32(key) % bins
            out[(lag - 1) * bins + idx] += 1.0 / float(lag)
        return out

    def _compose_features(self, state_features: np.ndarray, prefix: Sequence[str]) -> np.ndarray:
        return np.concatenate([np.asarray(state_features, dtype=np.float32), self._prefix_features(prefix),], axis=0).astype(np.float32)

    def _prepare_examples(self, demonstrations: Sequence[Trajectory], demo_weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
        unique_actions = sorted({a for traj in demonstrations for _, a in traj if a != "stop"})
        if not unique_actions: return (np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), {}, {})
        state_to_idx, idx_to_state, action_to_idx, idx_to_action = create_state_action_mappings(demonstrations, unique_actions=unique_actions)
        state_matrix, _, _ = create_feature_matrix_2_0(idx_to_state, normalizer=self.normalizer, update_normalizer=True)
        self.state_feature_dim = int(state_matrix.shape[1]) if state_matrix.ndim == 2 else 0
        xs: List[np.ndarray] = []
        ys: List[int] = []
        sw: List[float] = []
        for traj, demo_weight in zip(demonstrations, demo_weights):
            prefix: List[str] = []
            weight = max(float(demo_weight), 0.0)
            for state, action in traj:
                if action == "stop":    break
                s_idx = state_to_idx.get(state)
                if s_idx is None:       continue
                xs.append(self._compose_features(state_matrix[s_idx], prefix))
                ys.append(action_to_idx[action])
                sw.append(weight)
                prefix.append(action)
        if not xs:
            return (np.zeros((0, self.state_feature_dim + self._history_dim()), dtype=np.float32),  np.zeros(0, dtype=np.int64),    np.zeros(0, dtype=np.float32),      action_to_idx,  idx_to_action)
        return (np.vstack(xs).astype(np.float32),                               np.asarray(ys,      dtype=np.int64),                np.asarray(sw, dtype=np.float32),   action_to_idx,  idx_to_action)

    def _warm_start(self, feature_dim: int, action_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((feature_dim, len(action_to_idx)), dtype=np.float32)
        bias = np.zeros(len(action_to_idx), dtype=np.float32)
        if self.weights is None or self.bias is None: return weights, bias
        for action, new_idx in action_to_idx.items():
            old_idx = self.action_to_idx.get(action)
            if old_idx is None: continue
            if old_idx >= self.weights.shape[1] or new_idx >= weights.shape[1]: continue
            weights[:, new_idx] = self.weights[:, old_idx]
            bias[new_idx] = self.bias[old_idx]
        return weights, bias

    def fit(self, demonstrations: Sequence[Trajectory], demo_weights: Optional[Sequence[float]] = None) -> None:
        if not demonstrations:
            self.reset()
            return
        self.normalizer = WelfordFeatureNormalizer()
        weights = [1.0] * len(demonstrations) if demo_weights is None else [float(w) for w in demo_weights]
        xs, ys, sample_weights, action_to_idx, idx_to_action = self._prepare_examples(demonstrations, weights)
        if xs.size == 0 or not action_to_idx:
            self.reset()
            return

        feature_dim = int(xs.shape[1])
        warm_start = self.weights is not None and self.bias is not None
        if warm_start:
            model_w, model_b = self._warm_start(feature_dim, action_to_idx)
            epochs = int(self.cfg.bc_epochs_warm)
        else:
            model_w = np.zeros((feature_dim, len(action_to_idx)), dtype=np.float32)
            model_b = np.zeros(len(action_to_idx), dtype=np.float32)
            epochs = int(self.cfg.bc_epochs_cold)

        total_weight = max(float(np.sum(sample_weights)), 1e-8)
        lr = float(self.cfg.bc_learning_rate)
        l2 = float(self.cfg.bc_l2)

        for _ in range(max(1, epochs)):
            logits = xs @ model_w + model_b[None, :]
            probs = _softmax(logits)
            delta = probs.copy()
            delta[np.arange(len(ys)), ys] -= 1.0
            delta *= (sample_weights[:, None] / total_weight).astype(np.float32)
            grad_w = (xs.T @ delta).astype(np.float32) + l2 * model_w
            grad_b = np.sum(delta, axis=0).astype(np.float32)
            model_w -= lr * grad_w
            model_b -= lr * grad_b

        self.weights = model_w.astype(np.float32)
        self.bias = model_b.astype(np.float32)
        self.action_to_idx = dict(action_to_idx)
        self.idx_to_action = dict(idx_to_action)
        n_examples = int(xs.shape[0])
        n_actions = int(len(action_to_idx))
        param_count = int(feature_dim * n_actions + n_actions)
        # Dense softmax regression accounting from the actual fit.
        forward = 2.0 * n_examples * feature_dim * n_actions
        softmax_delta = 7.0 * n_examples * n_actions
        grad = 2.0 * feature_dim * n_examples * n_actions + float(n_examples * n_actions)
        regularize_update = 3.0 * param_count
        estimated_flops = float(max(1, epochs) * (forward + softmax_delta + grad + regularize_update))
        self.last_fit_stats = {"model_family": "behavior_cloning", "estimated_flops": estimated_flops, "n_demonstrations": float(len(demonstrations)), "n_examples": float(n_examples), "n_actions": float(n_actions), "feature_dim": float(feature_dim), "state_feature_dim": float(self.state_feature_dim),
            "history_dim": float(self._history_dim()), "epochs": float(max(1, epochs)), "warm_start": 1.0 if warm_start else 0.0}

    def predict(self, state: State, prefix: Sequence[str]) -> Dict[str, float]:
        if self.weights is None or self.bias is None or not self.idx_to_action: return {}
        feature_matrix, _, _ = create_feature_matrix_2_0({0: tuple(state)}, normalizer=self.normalizer)
        if feature_matrix.size == 0: return {}
        x = self._compose_features(feature_matrix[0], prefix)[None, :]
        probs = _softmax(x @ self.weights + self.bias[None, :])[0]
        return {self.idx_to_action[idx]: float(prob) for idx, prob in enumerate(probs) if float(prob) > 0.0}


def _without_latest_preference_protection(cfg: Config) -> Config:
    return replace(cfg, protect_latest_preference=False)


class AdaptiveDecayAgent(AdaptiveHRCAgent):
    """Adaptive decay without latest-preference protection."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)


class NoReplayAgent(AdaptiveHRCAgent):
    """Strict no-replay baseline: only the newest committed demo survives."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)

    def _clear_memory(self) -> None:
        self.memory.variants.clear()
        self.memory.latest.clear()
        self.decay.active.clear()
        self.decay.pruned.clear()
        self.decay.latest_by_recipe.clear()
        self.decay.latest_keys.clear()
        # Clear action codebooks so the IRL feature matrix cannot see tokens from cleared recipes. Without this, states from pruned recipes remain in the state_to_idx table and influence feature normalization on the next retrain.
        self.action_vector_to_token.clear()
        self.token_to_action_vector.clear()
        self.token_to_role.clear()
        
        self.decay.reuse_gaps.clear()
        self.decay._reuse_gap_window.clear()
        self.decay._active_recipe_count_window.clear()
        self.decay.base_rate = 1.0 / max(1, int(getattr(self.cfg, "decay_horizon_init", 10)))
        self.decay.global_rate = self.decay.base_rate
        self.decay.rate_history.append((self.session_counter, self.decay.global_rate))
        self.recipe_prototypes = RecipePrototypeLearner(self.cfg)
        self.preference_prototypes = PreferencePrototypeLearner(cfg=self.cfg)
        self.task_signatures.clear()
        self.preference_signatures.clear()
        self.last_pref_id = None
        self.inferred_recipe = None
        self.inferred_variant_hash = None
        self.inferred_latent_pref_id = None
        self.posterior.reset()
        self._last_fit_fingerprint = None

    def _replace_memory(self, recipe_id: str, seq: List[str]) -> None:
        self._clear_memory()
        self._register_if_live(recipe_id, seq, self.step_counter)

    def _end_observe_demo(self, apply_decay: bool = False) -> Classification:
        seq = list(self.pending_demo)
        active_keys = self._active_keys()
        active_lib = self.memory.library(allowed_keys=active_keys)
        cls = self.disambig.classify(seq, active_lib)
        self.classification_events.append((self.step_counter, cls))
        if cls.kind == "new_recipe":
            rid = f"R{self._next_recipe_idx}"
            self._next_recipe_idx += 1
            cls.recipe_id = rid
        else:
            if cls.recipe_id is None: raise RuntimeError(f"disambiguator returned {cls.kind} without a recipe_id")
            rid = cls.recipe_id
        self._replace_memory(rid, seq)
        self.mode = MODE_ONLINE
        self.pending_demo = []
        if apply_decay: self.decay.step(self.session_counter, self.retrain_cycle)
        self._retrain()
        return cls

    def _end_online_session(self, commit_cls: Classification, reentry_from_pruned: bool = False, apply_decay: bool = False) -> Classification:
        prefix = list(self.current_prefix)
        if not prefix:
            self.current_prefix = []
            self.inferred_recipe = None
            self.inferred_variant_hash = None
            self.inferred_latent_pref_id = None
            return Classification("known", None, None, 0.0, 0.0)
        if commit_cls.kind == "new_recipe" or commit_cls.recipe_id is None: raise RuntimeError("online new-recipe commit reached mutating path")

        rid = commit_cls.recipe_id
        self._replace_memory(rid, prefix)
        cls_kind = "known" if commit_cls.kind == "known" else "preference_shift"
        cls = Classification(cls_kind, rid, variant_hash(prefix), commit_cls.jaccard, commit_cls.order_distance)
        if apply_decay: self.decay.step(self.session_counter, self.retrain_cycle)
        self._retrain()
        self.classification_events.append((self.step_counter, cls))
        self.current_prefix = []
        self.inferred_recipe = None
        self.inferred_variant_hash = None
        self.inferred_latent_pref_id = None
        return cls

    def _retrain(self) -> None:
        if self._frozen: return
        retrain_t0 = time.perf_counter()
        self.retrain_cycle += 1
        entries = self.decay.active_entries()
        if not entries:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return
        latest = max(entries, key=lambda e: e.last_seen_step)
        build_t0 = time.perf_counter()
        trajectories, dropped_total = self._build_trajectories([list(latest.ordering)])
        build_wall_s = time.perf_counter() - build_t0
        self._prepare_retrain_fit()
        fit_t0 = time.perf_counter()
        self._fit_heads(trajectories, [1.0])
        fit_wall_s = time.perf_counter() - fit_t0
        self._record_retrain_event(dropped_actions=dropped_total, active_demos=1, total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_retrain_flops(trajectories))


class UniformWeightAgent(AdaptiveHRCAgent):
    """Rehearsal weights are ignored at training time (all demos weight 1)."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)

    def _retrain(self) -> None:
        if self._frozen: return
        retrain_t0 = time.perf_counter()
        self.retrain_cycle += 1
        entries = self.decay.active_entries()
        if not entries:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return
        demos = [list(e.ordering) for e in entries]
        build_t0 = time.perf_counter()
        trajectories, dropped_total = self._build_trajectories(demos)
        build_wall_s = time.perf_counter() - build_t0
        self._prepare_retrain_fit()
        fit_t0 = time.perf_counter()
        self._fit_heads(trajectories, [1.0] * len(trajectories))
        fit_wall_s = time.perf_counter() - fit_t0
        self._record_retrain_event(dropped_actions=dropped_total, active_demos=len(demos), total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_retrain_flops(trajectories))


class FixedDecayAgent(AdaptiveHRCAgent):
    """Global decay rate is frozen to decay_init; adaptive rate disabled."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self.decay._record_reuse_gap = lambda *args, **kwargs: None       # type: ignore
        self.decay._recompute_effective = lambda *args, **kwargs: None    # type: ignore
        self.decay.base_rate = cfg.decay_init
        self.decay.global_rate = cfg.decay_init


class NoDecayAgent(AdaptiveHRCAgent):
    """Decay is fully disabled: weights never decrease, nothing is ever pruned. Keeps registration and retraining active so the IRL and Markov heads still learn. This is the complement to FixedDecayAgent for ablations where we want to see what happens with infinite perfect memory."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        # Disable decay while keeping registration and retraining active.
        self.decay.step = lambda now, cycle, protected_keys=None: []  # type: ignore[assignment]
        self.decay._record_reuse_gap = lambda *args, **kwargs: None   # type: ignore
        self.decay._recompute_effective = lambda *args, **kwargs: None # type: ignore
        # Keep base/global rate at 0 to make it explicit.
        self.decay.base_rate = 0.0
        self.decay.global_rate = 0.0


class LatestOnlyPreferenceAgent(AdaptiveDecayAgent):
    """Keeps only the most recently observed preference variant per recipe."""

    def _register_if_live(self, rid: str, seq: List[str], step: int):
        variant = super()._register_if_live(rid, seq, step)
        slot = self.memory.variants.get(rid, {})
        for variant_hash in list(slot.keys()):
            if variant_hash == variant.variant_hash: continue
            del slot[variant_hash]
            self.decay.discard(rid, variant_hash)
        self.memory.latest[rid] = variant.variant_hash
        return variant


class L2AnchorAgent(AdaptiveHRCAgent):
    """True L2 anchor on IRL theta: at each retrain, penalise drift from the snapshot of the last successful theta. Implemented as EWC with a uniform Fisher (i.e. a quadratic penalty `lambda * ||theta - theta_star||^2`) routed through `MaxEntIRL2.fit`'s existing EWC plumbing."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, anchor_lambda: Optional[float] = None, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        # If not specified, reuse the EWC lambda knob — treating L2 anchor as a constant-curvature special case of EWC.
        self._anchor_lambda = float(cfg.ewc_lambda if anchor_lambda is None else anchor_lambda)
        self._prev_theta: Optional[np.ndarray] = None

    def _fit_heads(self, trajectories, weights) -> None:
        ewc_theta_star = self._prev_theta
        ewc_fisher = (np.ones_like(self._prev_theta) if self._prev_theta is not None else None)
        self.irl.fit(trajectories, weights, ewc_theta_star=ewc_theta_star, ewc_fisher=ewc_fisher)
        self.markov.fit(trajectories, weights, state_to_idx=self.irl.state_to_idx, idx_to_state=self.irl.idx_to_state, feature_matrix=self.irl.feature_matrix, col_min=self.irl.col_min, col_max=self.irl.col_max, normalizer=self.irl.normalizer)
        # Snapshot the new theta for next anchor.
        self._prev_theta = None if self.irl.theta is None else self.irl.theta.copy()


class BehaviorCloningAgent(AdaptiveHRCAgent):
    """Supervised next-action imitation baseline with adaptive rehearsal weights."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self.bc = BehaviorCloningHead(cfg=self.cfg)

    def _fit_heads(self, trajectories, weights) -> None:
        self.bc.fit(trajectories, weights)

    def _estimate_retrain_flops(self, trajectories) -> float:
        """Approximate BC fit cost from the fit that actually just ran."""
        stats = getattr(self.bc, "last_fit_stats", {}) or {}
        estimate = stats.get("estimated_flops") if isinstance(stats, dict) else None
        if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)):
            return float(estimate)
        # Fallback for unfitted heads only. Current retrain events should use
        # BehaviorCloningHead.fit statistics.
        n_examples = sum(1 for traj in trajectories for _state, action in traj if action != "stop")
        n_actions = max(1, len({action for traj in trajectories for _state, action in traj if action != "stop"}))
        state_dim = len(trajectories[0][0][0]) if trajectories and trajectories[0] else 1
        feature_dim = max(1, state_dim + self.bc._history_dim())
        epochs = int(self.cfg.bc_epochs_cold)
        return float(2 * n_examples * n_actions * feature_dim * max(1, epochs))

    def _prepare_retrain_fit(self) -> None:
        self._reset_heads()

    def _reset_heads(self) -> None:
        super()._reset_heads()
        self.bc = BehaviorCloningHead(cfg=self.cfg)

    def pruned_influence_audit(self, max_prefixes: int = 24, tolerance: float = 5e-2) -> Dict[str, object]:
        entries = self.decay.active_entries()
        if not entries: return {"max_l1": 0.0, "mean_l1": 0.0, "n_prefixes": 0, "passed": True, "tolerance": float(tolerance), "model_family": "behavior_cloning"}
        demos = [list(e.ordering) for e in entries]
        weights = [float(e.weight) for e in entries]
        trajectories, _ = self._build_trajectories(demos)
        fresh_bc = BehaviorCloningHead(cfg=self.cfg)
        fresh_bc.fit(trajectories, weights)
        prefixes: List[Tuple[str, ...]] = []
        seen = set()
        for e in entries:
            seq = tuple(e.ordering)
            for k in (0, min(len(seq), 1), len(seq) // 2, max(0, len(seq) - 1)):
                pref = tuple(seq[:k])
                if pref not in seen:
                    prefixes.append(pref)
                    seen.add(pref)
                if len(prefixes) >= max(1, int(max_prefixes)):  break
            if len(prefixes) >= max(1, int(max_prefixes)):      break
        diffs: List[float] = []
        for pref in prefixes:
            state = self._state_from_prefix(pref)
            cur = self.bc.predict(state, pref)
            ref = fresh_bc.predict(state, pref)
            keys = set(cur.keys()) | set(ref.keys())
            diffs.append(sum(abs(float(cur.get(k, 0.0)) - float(ref.get(k, 0.0))) for k in keys))
        max_l1 = max(diffs) if diffs else 0.0
        mean_l1 = sum(diffs) / max(len(diffs), 1)
        return {"max_l1": float(max_l1), "mean_l1": float(mean_l1), "n_prefixes": len(diffs), "passed": bool(max_l1 <= tolerance), "tolerance": float(tolerance), "model_family": "behavior_cloning"}

    def predict_next_tokens(self, prefix=None) -> Dict[str, float]:
        prefix_tokens = list(prefix) if prefix is not None else list(self.current_prefix)
        prefix_tokens = self._coerce_prefix_tokens(prefix_tokens)
        state = self._state_from_prefix(prefix_tokens)
        dist = self.bc.predict(state, prefix_tokens)
        if dist:
            conf = max(float(v) for v in dist.values())
            ent = 0.0
            for p in dist.values(): 
                if p > 0: ent -= float(p) * math.log(float(p))
            if len(dist) > 1: ent /= math.log(len(dist))
            threshold = self._action_confidence_threshold()
            self._set_assist_gate(conf >= threshold, conf, ent, "baseline_policy" if conf >= threshold else "low_action_confidence")
        else: self._set_assist_gate(False, None, None, "baseline_empty")
        return dist


class ProgressiveAnchorEWCAgent(AdaptiveHRCAgent):
    """Single-task-anchor EWC (sliding variant).

    After each refit the consolidation point is overwritten with the most
    recently completed task's theta and Fisher.  Only the LAST task boundary
    is protected — earlier tasks receive no penalty once a new snapshot is
    taken.  Fisher is replaced (not accumulated), so this is NOT the original
    Kirkpatrick et al. 2017 formulation.

    Retained as a lightweight ablation: it answers "how much does the proposed
    agent benefit from protecting the latest task alone?"  For the full
    multi-task Kirkpatrick formulation see EWCAgent.
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self._ewc_theta_star: Optional[np.ndarray] = None
        self._ewc_fisher: Optional[np.ndarray] = None

    def _fit_heads(self, trajectories, weights) -> None:
        self.irl.fit(trajectories, weights, ewc_theta_star=self._ewc_theta_star, ewc_fisher=self._ewc_fisher)
        self.markov.fit(trajectories, weights, state_to_idx=self.irl.state_to_idx, idx_to_state=self.irl.idx_to_state, feature_matrix=self.irl.feature_matrix, col_min=self.irl.col_min, col_max=self.irl.col_max, normalizer=self.irl.normalizer)
        # Overwrite anchor with THIS task's optimum.
        if self.irl.theta is not None:
            self._ewc_theta_star = self.irl.theta.copy()
            self._ewc_fisher = self.irl.fisher_diagonal(trajectories, weights)


class EWCAgent(AdaptiveHRCAgent):
    """Kirkpatrick et al. (2017) diagonal-Fisher EWC on the IRL head.

    Maintains precision-weighted running accumulators across all observed task
    boundaries so that the effective EWC penalty is equivalent to summing
    per-task quadratic terms:

        penalty = sum_i  F_i * (theta - theta_i*)^2

    Because the per-task terms share the same quadratic form, they can be
    collapsed into a single consolidated penalty:

        F_total        = sum_i F_i
        theta_star_total = (sum_i F_i * theta_i*) / F_total   [precision-weighted anchor]
        penalty        = F_total * (theta - theta_star_total)^2

    This single-anchor form is what MaxEntIRL2.fit receives via its
    `ewc_theta_star` / `ewc_fisher` arguments, keeping the fit interface
    unchanged while correctly representing all past tasks.

    When feature dimensionality increases (new recipes introduce new states),
    accumulators are zero-padded so old-task Fisher mass is preserved and
    new dimensions start unconstrained.
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        # Running accumulators (feature-space vectors)
        self._ewc_fisher_total: Optional[np.ndarray] = None        # sum_i F_i
        self._ewc_fisher_theta_total: Optional[np.ndarray] = None  # sum_i F_i * theta_i*
        # Consolidated anchor passed to irl.fit — recomputed after every task.
        self._ewc_theta_star_consolidated: Optional[np.ndarray] = None
        self._ewc_fisher_consolidated: Optional[np.ndarray] = None

    @staticmethod
    def _resize_to(arr: Optional[np.ndarray], n: int) -> np.ndarray:
        """Return a float32 vector of length n, zero-padding or truncating arr."""
        if arr is None:
            return np.zeros(n, dtype=np.float32)
        if arr.shape[0] == n:
            return arr
        out = np.zeros(n, dtype=np.float32)
        k = min(arr.shape[0], n)
        out[:k] = arr[:k]
        return out

    def _fit_heads(self, trajectories, weights) -> None:
        self.irl.fit(
            trajectories, weights,
            ewc_theta_star=self._ewc_theta_star_consolidated,
            ewc_fisher=self._ewc_fisher_consolidated,
        )
        self.markov.fit(
            trajectories, weights,
            state_to_idx=self.irl.state_to_idx,
            idx_to_state=self.irl.idx_to_state,
            feature_matrix=self.irl.feature_matrix,
            col_min=self.irl.col_min,
            col_max=self.irl.col_max,
            normalizer=self.irl.normalizer,
        )
        if self.irl.theta is None:
            return

        new_theta = self.irl.theta.astype(np.float32)   # theta_i*
        new_fisher = self.irl.fisher_diagonal(trajectories, weights)
        if new_fisher is None:
            return
        new_fisher = new_fisher.astype(np.float32)       # F_i

        n = new_theta.shape[0]
        # Resize accumulators to current feature dimension, preserving history.
        self._ewc_fisher_total = self._resize_to(self._ewc_fisher_total, n)
        self._ewc_fisher_theta_total = self._resize_to(self._ewc_fisher_theta_total, n)

        # Accumulate: F_total += F_i,  (F*theta)_total += F_i * theta_i*
        self._ewc_fisher_total += new_fisher
        self._ewc_fisher_theta_total += new_fisher * new_theta

        # Consolidated precision-weighted anchor.
        # theta_star_total = (sum_i F_i * theta_i*) / sum_i F_i
        eps = float(getattr(self.cfg, "ewc_precision_eps", 1e-12))
        fisher_clip = float(getattr(self.cfg, "ewc_fisher_clip", 1e3))
        fisher_safe = np.maximum(self._ewc_fisher_total, eps)
        self._ewc_theta_star_consolidated = (
            self._ewc_fisher_theta_total / fisher_safe
        ).astype(np.float32)
        self._ewc_fisher_consolidated = np.clip(
            self._ewc_fisher_total, 0.0, fisher_clip
        ).astype(np.float32)


class ExperienceReplayAgent(BehaviorCloningAgent):
    """Reservoir replay over ALL ingested demonstrations (no `decay.active` filter; that would leak the proposed system's pruning into the baseline).
    On each retrain, a uniform sample from the reservoir is unioned with the most recent demo and used to refit a BC next-action head. Sample weights are uniform (1.0): this baseline does not consume the proposed decay weighting."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self._buffer: List[List[str]] = []
        self._buffered_keys: set = set()
        self._seen: int = 0
        self._rng = random.Random(getattr(cfg, "seed", 1337))
        self._last_replay_metadata: Dict[str, Any] = self.replay_buffer_metadata()

    def _buffer_add(self, ordering: List[str]) -> None:
        cap = max(1, int(self.cfg.er_buffer_size))
        self._seen += 1
        if len(self._buffer) < cap:
            self._buffer.append(list(ordering))
            return
        j = self._rng.randrange(self._seen)
        if j < cap: self._buffer[j] = list(ordering)

    def replay_buffer_metadata(self) -> Dict[str, Any]:
        n_buffered = len(getattr(self, "_buffer", []))
        footprint = sum(len(ordering) for ordering in getattr(self, "_buffer", []))
        return {"policy": "uniform_reservoir", "er_buffer_size": int(self.cfg.er_buffer_size), "er_batch_size": int(self.cfg.er_batch_size), "n_buffered": int(n_buffered), "replay_memory_footprint": int(footprint), "replay_memory_steps": int(footprint)}

    def _retrain(self) -> None:
        if self._frozen: return
        retrain_t0 = time.perf_counter()
        self.retrain_cycle += 1
        # Reservoir-add newly committed active entries. This baseline disables latest-preference protection, so `decay.latest_keys` may be empty; using it here silently starves replay and yields an empty predictor.
        entries = self.decay.active_entries()
        for entry in entries:
            if entry.key in self._buffered_keys: continue
            self._buffered_keys.add(entry.key)
            self._buffer_add(list(entry.ordering))

        if not self._buffer:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return

        # Uniform reservoir sample at every refit, then dedup.
        batch_size = max(1, int(self.cfg.er_batch_size))
        k = min(batch_size, len(self._buffer))
        sampled = self._rng.sample(self._buffer, k) if k > 0 else []
        demos: List[List[str]] = []
        seen_keys: set = set()
        for s in sampled:
            key = tuple(s)
            if key in seen_keys: continue
            seen_keys.add(key)
            demos.append(list(s))
        if not demos:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return
        self._last_replay_metadata = {**self.replay_buffer_metadata(), "n_sampled": int(len(demos)), "sampler": "uniform_without_replacement"}

        build_t0 = time.perf_counter()
        trajectories, dropped_total = self._build_trajectories(demos)
        build_wall_s = time.perf_counter() - build_t0
        weights = [1.0] * len(demos)  # uniform, by design (no decay leakage).
        self._reset_heads()
        fit_t0 = time.perf_counter()
        self._fit_heads(trajectories, weights)
        fit_wall_s = time.perf_counter() - fit_t0
        self._record_retrain_event(dropped_actions=dropped_total, active_demos=len(demos), total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_retrain_flops(trajectories))


class RecencyPrioritizedReplayAgent(ExperienceReplayAgent):
    """BC replay baseline with the same nominal buffer as ER, but sampled by recency. The buffer stores one record per observed variant: `(ordering, last_seen_session, seen_count)`. Sampling uses p_i proportional to `(sessions_since_seen_i + 1)^(-alpha)` mixed with a
    small uniform component. This is a stronger memory-policy comparator than a uniform reservoir because it answers whether adaptive selective forgetting still helps against a fair recency replay policy."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=cfg, **kw)
        self._recency_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._buffer = []
        self._last_replay_metadata = self.replay_buffer_metadata()

    def _refresh_buffer_view(self) -> None:
        ordered = sorted(self._recency_records.values(), key=lambda r: (int(r.get("last_seen_session", 0)), tuple(r.get("ordering", ()))), reverse=True)
        self._buffer = [list(r.get("ordering", ())) for r in ordered]

    def _upsert_record(self, entry) -> None:
        cap = max(1, int(self.cfg.er_buffer_size))
        key = entry.key
        existing = self._recency_records.get(key)
        if existing is None and len(self._recency_records) >= cap:
            evict_key = min(
                self._recency_records,
                key=lambda k: (int(self._recency_records[k].get("last_seen_session", 0)), int(self._recency_records[k].get("seen_count", 0)), k))
            self._recency_records.pop(evict_key, None)
        rec = self._recency_records.setdefault(key, {"ordering": list(entry.ordering), "last_seen_session": int(entry.last_seen_step), "seen_count": 0})
        rec["ordering"] = list(entry.ordering)
        rec["last_seen_session"] = int(entry.last_seen_step)
        rec["seen_count"] = int(rec.get("seen_count", 0)) + 1

    def _sample_records(self, k: int) -> List[Dict[str, Any]]:
        records = list(self._recency_records.values())
        if not records or k <= 0: return []
        now = max([int(r.get("last_seen_session", 0)) for r in records] + [int(self.session_counter)])
        alpha = max(0.0, float(self.cfg.er_recency_alpha))
        mix = min(max(float(self.cfg.er_uniform_mix), 0.0), 1.0)
        remaining = list(records)
        sampled: List[Dict[str, Any]] = []
        while remaining and len(sampled) < min(k, len(records)):
            raw = [(max(0, now - int(r.get("last_seen_session", 0))) + 1.0) ** (-alpha) for r in remaining]
            total_raw = max(float(sum(raw)), 1e-12)
            uniform = 1.0 / len(remaining)
            weights = [(1.0 - mix) * (w / total_raw) + mix * uniform for w in raw]
            choice = self._rng.random()
            acc = 0.0
            picked = len(remaining) - 1
            for idx, weight in enumerate(weights):
                acc += float(weight)
                if choice <= acc:
                    picked = idx
                    break
            sampled.append(remaining.pop(picked))
        return sampled

    def replay_buffer_metadata(self) -> Dict[str, Any]:
        records = list(getattr(self, "_recency_records", {}).values())
        footprint = sum(len(r.get("ordering", ())) for r in records)
        return { "policy": "recency_prioritized",       "alpha": float(self.cfg.er_recency_alpha),      "uniform_mix": float(self.cfg.er_uniform_mix),      "er_buffer_size": int(self.cfg.er_buffer_size),     "er_batch_size": int(self.cfg.er_batch_size),
                "n_buffered": int(len(records)),        "replay_memory_footprint": int(footprint),      "replay_memory_steps": int(footprint)}

    def _retrain(self) -> None:
        if self._frozen: return
        retrain_t0 = time.perf_counter()
        self.retrain_cycle += 1
        entries = self.decay.active_entries()
        for entry in entries: self._upsert_record(entry)
        self._refresh_buffer_view()
        if not self._recency_records:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return
        batch_size = max(1, int(self.cfg.er_batch_size))
        sampled = self._sample_records(batch_size)
        demos: List[List[str]] = []
        seen_keys: set = set()
        for record in sampled:
            demo = list(record.get("ordering", ()))
            key = tuple(demo)
            if not demo or key in seen_keys: continue
            seen_keys.add(key)
            demos.append(demo)
        if not demos:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return
        self._last_replay_metadata = {**self.replay_buffer_metadata(), "n_sampled": int(len(demos)), "sampler": "recency_prioritized_without_replacement"}
        build_t0 = time.perf_counter()
        trajectories, dropped_total = self._build_trajectories(demos)
        build_wall_s = time.perf_counter() - build_t0
        weights = [1.0] * len(demos)
        self._reset_heads()
        fit_t0 = time.perf_counter()
        self._fit_heads(trajectories, weights)
        fit_wall_s = time.perf_counter() - fit_t0
        self._record_retrain_event(dropped_actions=dropped_total, active_demos=len(demos), total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_retrain_flops(trajectories))


class BigramOnlyAgent(AdaptiveHRCAgent):
    """True bigram floor: a fresh `Counter[(prev_token, next_token)] -> Categorical`.
    Does NOT inherit the parent's state-conditional N-gram head. Predict reads from this agent's own bigram counter, populated on each retrain from the active set with unit counts. The predict path completely ignores IRL and posterior signals; the baseline answers "what does a vanilla last-token Markov chain give?"."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self._bigram: Dict[str, Dict[str, int]] = {}
        self._unigram: Dict[str, int] = {}

    def _fit_heads(self, trajectories, weights) -> None:
        # Replace the parent's IRL/Markov fits with a flat bigram pass.
        self._bigram = {}
        self._unigram = {}
        scalar_updates = 0
        for traj in trajectories:
            seq = [a for _, a in traj if a != "stop"]
            for i, a in enumerate(seq):
                self._unigram[a] = self._unigram.get(a, 0) + 1
                scalar_updates += 1
                if i == 0: continue
                prev = seq[i - 1]
                row = self._bigram.setdefault(prev, {})
                row[a] = row.get(a, 0) + 1
                scalar_updates += 1
        # Wipe the parent heads so any accidental call returns nothing.
        self._reset_heads()
        self._custom_fit_stats = {**_symbolic_fit_stats("bigram", trajectories, estimated_flops=0.0), "scalar_counter_updates": float(scalar_updates)}

    def predict_next_tokens(self, prefix=None) -> Dict[str, float]:
        prefix = list(prefix) if prefix is not None else list(self.current_prefix)
        prefix = self._coerce_prefix_tokens(prefix)
        if not prefix:
            total = sum(self._unigram.values())
            dist = {a: c / total for a, c in self._unigram.items()} if total > 0 else {}
            self._set_bigram_gate(dist)
            return dist
        prev = prefix[-1]
        row = self._bigram.get(prev)
        if not row:
            total = sum(self._unigram.values())
            dist = {a: c / total for a, c in self._unigram.items()} if total > 0 else {}
            self._set_bigram_gate(dist)
            return dist
        z = float(sum(row.values()))
        dist = {a: c / z for a, c in row.items()}
        self._set_bigram_gate(dist)
        return dist

    def _set_bigram_gate(self, dist: Dict[str, float]) -> None:
        if not dist:
            self._set_assist_gate(False, None, None, "baseline_empty")
            return
        conf = max(float(v) for v in dist.values())
        ent = 0.0
        for p in dist.values():
            if p > 0:
                ent -= float(p) * math.log(float(p))
        if len(dist) > 1:
            ent /= math.log(len(dist))
        threshold = self._action_confidence_threshold()
        self._set_assist_gate(conf >= threshold, conf, ent, "baseline_policy" if conf >= threshold else "low_action_confidence")


class FrequencyConditionedBigramAgent(AdaptiveHRCAgent):
    """Weighted active-memory next-action frequency conditioned on the current token."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)

    def _fit_heads(self, trajectories, weights) -> None:
        self._reset_heads()
        self._custom_fit_stats = _symbolic_fit_stats("frequency_conditioned_bigram", trajectories, estimated_flops=0.0)

    def predict_next_tokens(self, prefix=None) -> Dict[str, float]:
        prefix = list(prefix) if prefix is not None else list(self.current_prefix)
        prefix = self._coerce_prefix_tokens(prefix)
        counts: Dict[str, float] = {}
        for entry in self.decay.active_entries():
            ordering = list(entry.ordering)
            weight = float(entry.weight)
            if not prefix:
                if ordering: counts[ordering[0]] = counts.get(ordering[0], 0.0) + weight
                continue
            prev = prefix[-1]
            for i in range(len(ordering) - 1):
                if ordering[i] == prev:
                    nxt = ordering[i + 1]
                    counts[nxt] = counts.get(nxt, 0.0) + weight
        if not counts:
            for entry in self.decay.active_entries():
                for action in entry.ordering: counts[action] = counts.get(action, 0.0) + float(entry.weight)
        total = float(sum(counts.values()))
        if total <= 0:
            self._set_assist_gate(False, None, None, "baseline_empty")
            return {}
        dist = {action: count / total for action, count in counts.items()}
        conf = max(dist.values())
        ent = 0.0
        for p in dist.values():
            if p > 0: ent -= float(p) * math.log(float(p))
        if len(dist) > 1: ent /= math.log(len(dist))
        threshold = self._action_confidence_threshold()
        self._set_assist_gate(conf >= threshold, conf, ent, "baseline_policy" if conf >= threshold else "low_action_confidence")
        return dist


class UniformValidActionAgent(AdaptiveHRCAgent):
    """Uniform over the tokens this agent has seen anywhere in any active demo.
    The cheapest possible floor: no model, no learning, no preference signal; just "I've seen these tokens, so any of them could be next." Treats every candidate as equally likely. Useful as a sanity floor on every accuracy figure: any baseline below this is degenerate."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)

    def _fit_heads(self, trajectories, weights) -> None:
        # Vocabulary is built lazily from active demo tokens at predict time.
        self._reset_heads()
        self._custom_fit_stats = _symbolic_fit_stats("uniform_valid", trajectories, estimated_flops=0.0)

    def predict_next_tokens(self, prefix=None) -> Dict[str, float]:
        vocab: set = set()
        for e in self.decay.active_entries(): vocab.update(e.ordering)
        if not vocab: return {}
        u = 1.0 / len(vocab)
        return {a: u for a in vocab}


class OracleCeilingAgent(AdaptiveHRCAgent):
    """Oracle ceiling: at predict time, reads the next action directly from the simulator's ground-truth ordering. NOT deployable.
    The harness sets the current target via `set_oracle_target(actions)` immediately before each evaluation/online demo. The agent tracks its own step counter via the StepResult-emitting predict path: each call to `predict_next_tokens` consumes one position. On `start_demo`/`end_demo` the counter resets.
    This is the only "baseline" that has access to the ground-truth pair; it bounds what any (recipe, pref)-conditioned policy could achieve."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self._oracle_target: Optional[List[str]] = None
        self._oracle_step: int = 0

    def _fit_heads(self, trajectories, weights) -> None:
        self._reset_heads()
        self._custom_fit_stats = _symbolic_fit_stats("oracle_ceiling", trajectories, estimated_flops=0.0)

    def set_oracle_target(self, actions: Sequence[str]) -> None:
        """Harness hook: tell the oracle which (recipe, pref) ordering is next."""
        self._oracle_target = list(actions)
        self._oracle_step = 0

    def start_demo(self) -> None:
        super().start_demo()
        self._oracle_step = 0

    def predict_next_tokens(self, prefix=None) -> Dict[str, float]:
        if self._oracle_target is None: return {}
        prefix = list(prefix) if prefix is not None else list(self.current_prefix)
        idx = len(prefix)  # position to predict
        if idx >= len(self._oracle_target): return {}
        return {self._oracle_target[idx]: 1.0}


class OnlineEWCAgent(ProgressiveAnchorEWCAgent):
    """Online EWC (Schwarz et al. 2018 Progress & Compress, EWC++ variant).

    Inherits from ProgressiveAnchorEWCAgent (single sliding anchor) and
    replaces its hard-overwrite update with exponential moving averages:

        theta_star ← gamma_theta * theta_star_prev + (1-gamma_theta) * theta_i*
        fisher     ← gamma_fisher * fisher_prev     + (1-gamma_fisher) * F_i

    With gamma → 1 the history is long; gamma = 0 reduces to ProgressiveAnchorEWCAgent.
    Unlike the cumulative-Fisher EWCAgent, old tasks are gradually down-weighted,
    making this suitable for non-stationary streams where old recipes become irrelevant.

    Note: parent class field names (_ewc_theta_star, _ewc_fisher) are reused;
    EMA is applied in-place after super()._fit_heads() updates them.
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG, gamma_fisher: float = 0.95, gamma_theta: float = 0.95, **kw):
        super().__init__(cfg=cfg, **kw)
        self._gamma_fisher = float(gamma_fisher)
        self._gamma_theta = float(gamma_theta)

    def _fit_heads(self, trajectories, weights) -> None:
        # Capture previous snapshots before parent overwrites them.
        prev_theta = None if self._ewc_theta_star is None else self._ewc_theta_star.copy()
        prev_fisher = None if self._ewc_fisher is None else self._ewc_fisher.copy()
        # Parent writes this task's theta_star and fisher to self._ewc_*.
        super()._fit_heads(trajectories, weights)
        # EMA-blend: move slowly toward the new task's anchor.
        if (prev_theta is not None and self._ewc_theta_star is not None
                and prev_theta.shape == self._ewc_theta_star.shape):
            self._ewc_theta_star = (
                self._gamma_theta * prev_theta
                + (1.0 - self._gamma_theta) * self._ewc_theta_star
            ).astype(np.float32)
        if (prev_fisher is not None and self._ewc_fisher is not None
                and prev_fisher.shape == self._ewc_fisher.shape):
            self._ewc_fisher = (
                self._gamma_fisher * prev_fisher
                + (1.0 - self._gamma_fisher) * self._ewc_fisher
            ).astype(np.float32)


class MostFrequentNextAgent(AdaptiveHRCAgent):
    """Predicts the marginally most frequent next action across all active demos. No state conditioning; this is a simple frequency baseline / floor."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)

    def _fit_heads(self, trajectories, weights) -> None:
        self._reset_heads()
        self._custom_fit_stats = _symbolic_fit_stats("most_frequent", trajectories, estimated_flops=0.0)

    def predict_next_tokens(self, prefix=None) -> Dict[str, float]:
        prefix = list(prefix) if prefix is not None else list(self.current_prefix)
        counts: Dict[str, float] = {}
        for e in self.decay.active_entries():
            ordering = list(e.ordering)
            # Count each action as a "next action" at every position
            for a in ordering: counts[a] = counts.get(a, 0.0) + e.weight
        total = sum(counts.values())
        if not total: return {}
        return {a: c / total for a, c in counts.items()}


BASELINE_AGENTS = {
    "latest_only": LatestOnlyPreferenceAgent,
    "no_replay": NoReplayAgent,
    "uniform_weight": UniformWeightAgent,
    "fixed_decay": FixedDecayAgent,
    "no_decay": NoDecayAgent,
    "l2_anchor": L2AnchorAgent,
    "bc": BehaviorCloningAgent,
    # Kirkpatrick et al. 2017: cumulative-Fisher, precision-weighted anchor.
    "ewc": EWCAgent,
    # Sliding single-anchor variant (ablation: only last task protected).
    "progressive_anchor_ewc": ProgressiveAnchorEWCAgent,
    # Schwarz et al. 2018 EWC++: EMA of single-anchor Fisher and theta.
    "online_ewc": OnlineEWCAgent,
    "experience_replay_bc": ExperienceReplayAgent,
    "recency_prioritized_replay": RecencyPrioritizedReplayAgent,
    "bigram": BigramOnlyAgent,
    "frequency_conditioned_bigram": FrequencyConditionedBigramAgent,
    "most_frequent": MostFrequentNextAgent,
    "uniform_valid": UniformValidActionAgent,
    "oracle_ceiling": OracleCeilingAgent,
}
