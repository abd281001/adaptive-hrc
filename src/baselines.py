"""Runnable comparison baselines for the publication benchmark."""
from __future__ import annotations
from dataclasses import replace
import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .adaptive_agent import AdaptiveHRCAgent, MODE_ONLINE, StepResult
from .memory import Classification, DecayManager, DemoTransition, Entry, VariantKey
from .models import (Config,    DEFAULT_CONFIG,     Trajectory,     WelfordFeatureNormalizer,       create_feature_matrix_2_0,      create_state_action_mappings)
from .representations import ActionObservation, ActionVector
State = Tuple[int, ...]


def _symbolic_fit_stats(model_family: str, trajectories: Sequence[Trajectory], estimated_flops: float = 0.0) -> Dict[str, object]:
    n_examples = sum(1 for traj in trajectories for _state, action in traj if action != "stop")
    n_actions = len({action for traj in trajectories for _state, action in traj if action != "stop"})
    return {"model_family": model_family, "estimated_flops": float(estimated_flops), "n_demonstrations": float(len(trajectories)), "n_examples": float(n_examples), "n_actions": float(n_actions), "flop_accounting_scope": f"{model_family}_fit_symbolic_counter_only", "flop_cross_model_comparable": False}


# Behavior-cloning support used by BC and replay baselines.
def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:    return logits.astype(np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    return (exp / denom).astype(np.float32)


class BehaviorCloningHead:
    """Linear softmax imitation model over state and one-hot prefix features."""
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.normalizer = WelfordFeatureNormalizer()
        self.state_feature_dim: int = 0
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.action_to_idx: Dict[str, int] = {}
        self.idx_to_action: Dict[int, str] = {}
        self.prefix_token_to_idx: Dict[str, int] = {}
        self.idx_to_prefix_token: Dict[int, str] = {}
        self.prefix_history_len: int = max(1, int(self.cfg.bc_history_len))
        self.last_fit_stats: Dict[str, object] = {}

    def reset(self) -> None:
        self.weights = None
        self.bias = None
        self.action_to_idx = {}
        self.idx_to_action = {}
        self.prefix_token_to_idx = {}
        self.idx_to_prefix_token = {}
        self.prefix_history_len = max(1, int(self.cfg.bc_history_len))
        self.last_fit_stats = {"model_family": "behavior_cloning", "estimated_flops": 0.0, "flop_accounting_scope": "behavior_cloning_fit_dense_numeric_only", "flop_cross_model_comparable": False}

    def _history_dim(self) -> int:
        return max(1, int(self.cfg.bc_history_len)) * len(self.prefix_token_to_idx) + 1

    def _prefix_features(self, prefix: Sequence[str]) -> np.ndarray:
        history = max(1, int(self.cfg.bc_history_len))
        vocab_size = len(self.prefix_token_to_idx)
        out = np.zeros(history * vocab_size + 1, dtype=np.float32)
        out[-1] = float(len(prefix))
        for lag in range(1, history + 1):
            if len(prefix) < lag: continue
            token = prefix[-lag]
            token_idx = self.prefix_token_to_idx.get(token)
            if token_idx is None: continue
            out[(lag - 1) * vocab_size + token_idx] = 1.0
        return out

    def _compose_features(self, state_features: np.ndarray, prefix: Sequence[str]) -> np.ndarray:
        return np.concatenate([np.asarray(state_features, dtype=np.float32), self._prefix_features(prefix),], axis=0).astype(np.float32)

    def _prepare_examples(self, demonstrations: Sequence[Trajectory], demo_weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
        unique_actions = sorted({a for traj in demonstrations for _, a in traj if a != "stop"})
        if not unique_actions: return (np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), {}, {})
        self.prefix_token_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
        self.idx_to_prefix_token = {idx: action for action, idx in self.prefix_token_to_idx.items()}
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

    def _warm_start(
        self,
        feature_dim: int,
        action_to_idx: Dict[str, int],
        old_weights: Optional[np.ndarray],
        old_bias: Optional[np.ndarray],
        old_action_to_idx: Dict[str, int],
        old_state_feature_dim: int,
        old_prefix_token_to_idx: Dict[str, int],
        old_history_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((feature_dim, len(action_to_idx)), dtype=np.float32)
        bias = np.zeros(len(action_to_idx), dtype=np.float32)
        if old_weights is None or old_bias is None: return weights, bias
        new_history_len = max(1, int(self.cfg.bc_history_len))
        state_rows = min(old_state_feature_dim, self.state_feature_dim, old_weights.shape[0], weights.shape[0])
        old_prefix_offset = max(0, old_state_feature_dim)
        new_prefix_offset = max(0, self.state_feature_dim)
        old_prefix_vocab_size = len(old_prefix_token_to_idx)
        new_prefix_vocab_size = len(self.prefix_token_to_idx)
        shared_history_len = min(max(1, int(old_history_len)), new_history_len)
        old_len_row = old_prefix_offset + max(1, int(old_history_len)) * old_prefix_vocab_size
        new_len_row = new_prefix_offset + new_history_len * new_prefix_vocab_size
        for action, new_idx in action_to_idx.items():
            old_idx = old_action_to_idx.get(action)
            if old_idx is None: continue
            if old_idx >= old_weights.shape[1] or new_idx >= weights.shape[1]: continue
            if state_rows > 0:
                weights[:state_rows, new_idx] = old_weights[:state_rows, old_idx]
            for lag_idx in range(shared_history_len):
                old_base = old_prefix_offset + lag_idx * old_prefix_vocab_size
                new_base = new_prefix_offset + lag_idx * new_prefix_vocab_size
                for token, new_token_idx in self.prefix_token_to_idx.items():
                    old_token_idx = old_prefix_token_to_idx.get(token)
                    if old_token_idx is None: continue
                    old_row = old_base + old_token_idx
                    new_row = new_base + new_token_idx
                    if old_row < old_weights.shape[0] and new_row < weights.shape[0]:
                        weights[new_row, new_idx] = old_weights[old_row, old_idx]
            if old_len_row < old_weights.shape[0] and new_len_row < weights.shape[0]:
                weights[new_len_row, new_idx] = old_weights[old_len_row, old_idx]
            bias[new_idx] = old_bias[old_idx]
        return weights, bias

    def fit(self, demonstrations: Sequence[Trajectory], demo_weights: Optional[Sequence[float]] = None) -> None:
        if not demonstrations:
            self.reset()
            return
        old_weights = self.weights
        old_bias = self.bias
        old_action_to_idx = dict(self.action_to_idx)
        old_state_feature_dim = int(self.state_feature_dim)
        old_prefix_token_to_idx = dict(self.prefix_token_to_idx)
        old_history_len = int(self.prefix_history_len)
        self.normalizer = WelfordFeatureNormalizer()
        weights = [1.0] * len(demonstrations) if demo_weights is None else [float(w) for w in demo_weights]
        xs, ys, sample_weights, action_to_idx, idx_to_action = self._prepare_examples(demonstrations, weights)
        if xs.size == 0 or not action_to_idx:
            self.reset()
            return

        feature_dim = int(xs.shape[1])
        warm_start = old_weights is not None and old_bias is not None
        if warm_start:
            model_w, model_b = self._warm_start(
                feature_dim,
                action_to_idx,
                old_weights,
                old_bias,
                old_action_to_idx,
                old_state_feature_dim,
                old_prefix_token_to_idx,
                old_history_len,
            )
            epochs = int(self.cfg.bc_epochs_warm)
        else:
            model_w = np.zeros((feature_dim, len(action_to_idx)), dtype=np.float32)
            model_b = np.zeros(len(action_to_idx), dtype=np.float32)
            epochs = int(self.cfg.bc_epochs_cold)

        lr = float(self.cfg.bc_learning_rate)
        l2 = float(self.cfg.bc_l2)
        batch_size = max(1, int(getattr(self.cfg, "bc_batch_size", 64)))
        rng = np.random.default_rng(int(getattr(self.cfg, "seed", 1337)))
        indices = np.arange(len(ys), dtype=np.int64)

        for _ in range(max(1, epochs)):
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                xb = xs[batch]
                yb = ys[batch]
                wb = sample_weights[batch].astype(np.float32)
                batch_weight = max(float(np.sum(wb)), 1e-8)
                logits = xb @ model_w + model_b[None, :]
                probs = _softmax(logits)
                delta = probs.copy()
                delta[np.arange(len(yb)), yb] -= 1.0
                delta *= (wb[:, None] / batch_weight).astype(np.float32)
                grad_w = (xb.T @ delta).astype(np.float32) + l2 * model_w
                grad_b = np.sum(delta, axis=0).astype(np.float32)
                model_w -= lr * grad_w
                model_b -= lr * grad_b

        self.weights = model_w.astype(np.float32)
        self.bias = model_b.astype(np.float32)
        self.action_to_idx = dict(action_to_idx)
        self.idx_to_action = dict(idx_to_action)
        self.prefix_history_len = max(1, int(self.cfg.bc_history_len))
        n_examples = int(xs.shape[0])
        n_actions = int(len(action_to_idx))
        param_count = int(feature_dim * n_actions + n_actions)
        # Dense softmax-regression accounting from the actual fit.  The two
        # GEMMs scale with the number of examples, while regularisation and
        # parameter updates occur once *per mini-batch*.  Keeping these
        # components explicit makes the estimate auditable and prevents a
        # silent under-count when the batch size changes.
        n_batches = int(math.ceil(n_examples / float(batch_size)))
        forward = 2.0 * n_examples * feature_dim * n_actions
        softmax_delta = 7.0 * n_examples * n_actions
        grad = 2.0 * feature_dim * n_examples * n_actions + float(n_examples * n_actions)
        optimizer_update = float(n_batches * (4 * feature_dim * n_actions + 2 * n_actions))
        estimated_flops = float(max(1, epochs) * (forward + softmax_delta + grad + optimizer_update))
        self.last_fit_stats = {"model_family": "behavior_cloning", "estimated_flops": estimated_flops, "n_demonstrations": float(len(demonstrations)), "n_examples": float(n_examples), "n_actions": float(n_actions), "feature_dim": float(feature_dim), "state_feature_dim": float(self.state_feature_dim),
            "history_dim": float(self._history_dim()), "parameter_count": float(param_count), "bc_prefix_encoding": "one_hot_active_vocab", "bc_prefix_vocab_size": float(len(self.prefix_token_to_idx)), "bc_prefix_feature_dim": float(self._history_dim()), "epochs": float(max(1, epochs)), "batch_size": float(batch_size), "n_batches_per_epoch": float(n_batches), "flop_forward": float(forward), "flop_softmax_delta": float(softmax_delta), "flop_gradient": float(grad), "flop_optimizer_update": float(optimizer_update), "flop_accounting_scope": "behavior_cloning_fit_dense_numeric_only", "flop_cross_model_comparable": False, "warm_start": 1.0 if warm_start else 0.0}

    def predict(self, state: State, prefix: Sequence[str]) -> Dict[str, float]:
        if self.weights is None or self.bias is None or not self.idx_to_action: return {}
        feature_matrix, _, _ = create_feature_matrix_2_0({0: tuple(state)}, normalizer=self.normalizer)
        if feature_matrix.size == 0: return {}
        x = self._compose_features(feature_matrix[0], prefix)[None, :]
        probs = _softmax(x @ self.weights + self.bias[None, :])[0]
        return {self.idx_to_action[idx]: float(prob) for idx, prob in enumerate(probs) if float(prob) > 0.0}


def _without_latest_preference_protection(cfg: Config) -> Config:
    return replace(cfg, protect_latest_preference=False)


class FixedRateDecayManager(DecayManager):
    """Picklable fixed-rate decay manager for the fixed-decay ablation."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        super().__init__(cfg)
        self.fixed_rate = float(cfg.decay_init)
        self.post_grace_decay_rate = self.fixed_rate

    def step(self, now: int, cycle: int, protected_keys: Optional[Sequence[VariantKey]] = None) -> List[VariantKey]:
        protected = set(protected_keys or ())
        pruned: List[VariantKey] = []
        for key, entry in list(self.active.items()):
            if key in self.latest_keys or key in protected:
                entry.weight = 1.0
                self._log_weight(now, entry)
                continue
            entry.weight -= self.fixed_rate
            if entry.weight <= self.cfg.prune_threshold:
                self._prune_entry(key, entry, now, cycle)
                pruned.append(key)
            else:
                self._log_weight(now, entry)
        return pruned


class NoDecayManager(DecayManager):
    """Unbounded, no-forgetting manager for the no-decay ablation.

    This comparator retains every registered recipe-preference variant at
    weight 1.0.  No variant-count capacity exists anywhere in the shared
    memory manager; this class disables the remaining temporal pruning path.
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        super().__init__(cfg)
        self.post_grace_decay_rate = 0.0

    def step(self, now: int, cycle: int, protected_keys: Optional[Sequence[VariantKey]] = None) -> List[VariantKey]:
        return []


class AdaptiveDecayAgent(AdaptiveHRCAgent):
    """Adaptive decay without latest-preference protection."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)






class OfflinePretrainedFrozenAgent(AdaptiveHRCAgent):
    """Offline-trained reference with a fixed deployment policy.

    This is a training-regime control inspired by the offline/deployment split
    used in TALENTS, not a reproduction of that method.  It deliberately uses
    the same IRL+n-gram learner as ``full`` during its evaluator-supplied
    offline phase, then permits only episode-local prediction state at
    deployment.  In particular, deployment observations cannot change the
    codebook, memory, decay weights, model heads, or retraining statistics.
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=cfg, **kw)
        self._deployment_locked = False
        self._deployment_prefix_vectors: Dict[str, ActionVector] = {}
        self._deployment_vector_tokens: Dict[ActionVector, str] = {}
        self._deployment_step = 0
        self._deployment_action_policy: Dict[str, Any] = {}
        self._offline_pretraining_metadata: Dict[str, Any] = {}

    def _clear_deployment_episode_state(self) -> None:
        """Discard only transient deployment state; learned state stays fixed."""
        self.mode = MODE_ONLINE
        self.pending_demo = []
        self.pending_demo_transitions = []
        self.pending_demo_identity = []
        self.current_prefix = []
        self.current_transition_trace = []
        self.current_identity_prefix = []
        self._online_policy_history = []
        self._deployment_prefix_vectors = {}
        self._deployment_vector_tokens = {}
        self._deployment_step = 0
        self._deployment_action_policy = {}

    def lock_deployment(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Freeze the learned offline policy before the deployment stream."""
        if self._deployment_locked:
            return self.offline_pretraining_metadata()
        fit_times = [float(value) for value in self.retrain_fit_wall_times]
        total_times = [float(value) for value in self.retrain_total_wall_times]
        build_times = [float(value) for value in self.retrain_build_wall_times]
        flops = [float(value) for value in self.retrain_flop_estimates]
        self._deployment_locked = True
        self._offline_pretraining_metadata = {
            "training_regime": "offline_pretraining_then_frozen_deployment",
            "talents_comparison_scope": "training_regime_reference_not_method_replication",
            "deployment_updates_allowed": False,
            "offline_training_retrain_count": int(len(total_times)),
            "offline_training_total_retrain_wall_s": float(sum(total_times)),
            "offline_training_fit_wall_s": float(sum(fit_times)),
            "offline_training_build_wall_s": float(sum(build_times)),
            "offline_training_estimated_fit_flops": float(sum(flops)),
            "offline_training_active_variants": int(len(self.decay.active)),
            "offline_training_active_action_steps": int(
                sum(len(entry.ordering) for entry in self.decay.active.values())
            ),
            **dict(metadata or {}),
        }
        self._clear_deployment_episode_state()
        return self.offline_pretraining_metadata()

    def offline_pretraining_metadata(self) -> Dict[str, Any]:
        return dict(self._offline_pretraining_metadata)

    def _deployment_token_for_vector(self, vector: ActionVector) -> str:
        known = self.action_vector_to_token.get(vector)
        if known is not None:
            return known
        token = self._deployment_vector_tokens.get(vector)
        if token is None:
            # Deterministic anonymous OOV token.  It lives only until the
            # current episode ends and is never added to the learned codebook.
            token = "deployment_oov_" + "_".join(str(int(value)) for value in vector)
            self._deployment_vector_tokens[vector] = token
            self._deployment_prefix_vectors[token] = vector
        return token

    def _vector_for_token(self, token: str) -> Optional[ActionVector]:
        vector = super()._vector_for_token(token)
        if vector is not None:
            return vector
        return self._deployment_prefix_vectors.get(token)

    def _set_action_policy_stats(
        self,
        confidence: Optional[float],
        support_normalized_entropy: Optional[float],
        reason: str,
        *,
        margin: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        if not self._deployment_locked:
            super()._set_action_policy_stats(
                confidence,
                support_normalized_entropy,
                reason,
                margin=margin,
                source=source,
            )
            return
        # The evaluator reads these fields while an episode is running.  Keep
        # them episode-local so deployment telemetry never becomes model state.
        self._deployment_action_policy = {
            "robot_action_mandatory": True,
            "action_confidence": confidence,
            "raw_action_confidence": confidence,
            "action_margin": margin,
            "action_support_normalized_entropy": support_normalized_entropy,
            "action_entropy_normalization": "prediction_support",
            "policy_source": source or "irl_markov_ensemble",
            "final_action_confidence": confidence,
            "final_action_margin": margin,
            "final_action_support_normalized_entropy": support_normalized_entropy,
            "reason": reason,
        }

    def action_policy_stats(self) -> Dict[str, Any]:
        if self._deployment_locked:
            return dict(self._deployment_action_policy)
        return super().action_policy_stats()

    def start_demo(self) -> None:
        if not self._deployment_locked:
            super().start_demo()
            return
        self._clear_deployment_episode_state()

    def observe_observation(
        self,
        observation: ActionObservation,
        ground_truth_recipe: Optional[str] = None,
        *,
        precomputed_distribution: Optional[Dict[str, float]] = None,
    ) -> StepResult:
        if not self._deployment_locked:
            return super().observe_observation(
                observation,
                ground_truth_recipe=ground_truth_recipe,
                precomputed_distribution=precomputed_distribution,
            )
        self._deployment_step += 1
        first_online_action = not self.current_prefix and precomputed_distribution is None
        distribution = (
            {} if first_online_action else
            (dict(precomputed_distribution) if precomputed_distribution is not None else self.predict_next_tokens(self.current_prefix))
        )
        predicted = max(distribution, key=distribution.get) if distribution else None
        action = self._deployment_token_for_vector(observation.action_vector)
        self.current_prefix.append(action)
        return StepResult(
            step=self._deployment_step,
            predicted=self._display_token(predicted),
            actual=self._display_token(action) or action,
            correct=predicted == action,
            mode=MODE_ONLINE,
            event="offline_frozen_deployment",
        )

    def end_demo(self) -> Classification:
        if not self._deployment_locked:
            return super().end_demo()
        self._clear_deployment_episode_state()
        return Classification("offline_frozen", None, None, 0.0, 0.0)

    def _retrain(self) -> None:
        if self._deployment_locked:
            return
        super()._retrain()

    def refresh_model_from_memory(self) -> None:
        if self._deployment_locked:
            return
        super().refresh_model_from_memory()


class FixedDecayAgent(AdaptiveHRCAgent):
    """Immediate fixed-rate forgetting ablation; adaptive grace horizons disabled."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self.decay = FixedRateDecayManager(self.cfg)


class NoDecayAgent(AdaptiveHRCAgent):
    """Unlimited-memory control: all variants remain active at weight 1.0.

    Keeps registration and retraining active so the IRL and Markov heads still
    learn, while disabling temporal decay.
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG, **kw):
        super().__init__(cfg=_without_latest_preference_protection(cfg), **kw)
        self.decay = NoDecayManager(self.cfg)


class LatestOnlyPreferenceAgent(AdaptiveDecayAgent):
    """Keeps only the most recently observed preference variant per recipe."""

    def _register_if_live(self, rid: str, seq: List[str], step: int, transitions: Optional[Sequence[DemoTransition]] = None, identity_ordering: Optional[Sequence[str]] = None):
        variant = super()._register_if_live(
            rid,
            seq,
            step,
            transitions=transitions,
            identity_ordering=identity_ordering,
        )
        slot = self.memory.variants.get(rid, {})
        for variant_hash in list(slot.keys()):
            if variant_hash == variant.variant_hash: continue
            del slot[variant_hash]
            self.decay.discard(rid, variant_hash)
        self.memory.latest[rid] = variant.variant_hash
        return variant




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
        weights = self._length_normalized_demo_weights(entries, [float(e.weight) for e in entries])
        trajectories, _ = self._build_trajectories(entries)
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
            self._set_action_policy_stats(conf, ent, "baseline_policy")
        else: self._set_action_policy_stats(None, None, "baseline_empty")
        return dist




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
        self._ewc_pending_task_demos: List[Any] = []
        self._custom_fit_stats: Dict[str, Any] = {}

    @property
    def _ewc_theta_star(self) -> Optional[np.ndarray]:
        return self._ewc_theta_star_consolidated

    @property
    def _ewc_fisher(self) -> Optional[np.ndarray]:
        return self._ewc_fisher_consolidated

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

    def _record_committed_replay_demo(
        self,
        recipe_id: str,
        variant_hash_: str,
        ordering: Tuple[str, ...],
        *,
        transitions: Tuple[DemoTransition, ...] = (),
        session_step: int,
        action_step: int,
        source_mode: str,
        entry: Optional[Entry] = None,
    ) -> None:
        if entry is not None:
            self._ewc_pending_task_demos.append(entry)
        else:
            self._ewc_pending_task_demos.append({"ordering": tuple(ordering), "transitions": tuple(transitions)})

    @staticmethod
    def _estimate_fisher_flops(trajectories: Sequence[Trajectory], n_features: int) -> float:
        state_visits = sum(len(traj) for traj in trajectories)
        return float(3.0 * max(0, state_visits) * max(1, int(n_features)))

    def _record_ewc_accounting(self, aux_fit_stats: Dict[str, Any], fisher_flops: float) -> None:
        primary_stats = dict(getattr(self.irl, "last_fit_stats", {}) or {})
        primary_est = primary_stats.get("estimated_flops", 0.0)
        aux_est = aux_fit_stats.get("estimated_flops", 0.0)
        try:
            primary_flops = float(primary_est)
        except (TypeError, ValueError):
            primary_flops = 0.0
        try:
            aux_flops = float(aux_est)
        except (TypeError, ValueError):
            aux_flops = 0.0
        if not math.isfinite(primary_flops):
            primary_flops = 0.0
        if not math.isfinite(aux_flops):
            aux_flops = 0.0
        try:
            fisher_flops = float(fisher_flops)
        except (TypeError, ValueError):
            fisher_flops = 0.0
        if not math.isfinite(fisher_flops):
            fisher_flops = 0.0
        self._custom_fit_stats = {
            "estimated_flops": float(primary_flops + aux_flops + fisher_flops),
            "ewc_primary_estimated_flops": float(primary_flops),
            "ewc_aux_estimated_flops": float(aux_flops),
            "ewc_fisher_estimated_flops": float(fisher_flops),
            "ewc_aux_iterations_run": float(aux_fit_stats.get("iterations_run", 0.0) or 0.0),
            "ewc_aux_n_states": float(aux_fit_stats.get("n_states", 0.0) or 0.0),
            "ewc_aux_n_features": float(aux_fit_stats.get("n_features", 0.0) or 0.0),
        }

    def _fit_heads(self, trajectories, weights) -> None:
        self._custom_fit_stats = {}
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
            self._record_ewc_accounting({}, 0.0)
            return

        pending = list(self._ewc_pending_task_demos)
        self._ewc_pending_task_demos = []
        if pending:
            task_trajectories, _ = self._build_trajectories(pending)
            task_weights = self._length_normalized_demo_weights(pending, [1.0] * len(pending))
        else:
            task_trajectories = list(trajectories)
            task_weights = [float(w) for w in weights]
        task_irl = type(self.irl)(cfg=self.cfg)
        task_irl.fit(
            task_trajectories,
            task_weights,
            ewc_theta_star=self._ewc_theta_star_consolidated,
            ewc_fisher=self._ewc_fisher_consolidated,
        )
        aux_fit_stats = dict(getattr(task_irl, "last_fit_stats", {}) or {})
        if task_irl.theta is None:
            self._record_ewc_accounting(aux_fit_stats, 0.0)
            return

        new_theta = task_irl.theta.astype(np.float32)   # theta_i*
        fisher_flops = self._estimate_fisher_flops(task_trajectories, int(new_theta.shape[0]))
        new_fisher = task_irl.fisher_diagonal(task_trajectories, task_weights)
        self._record_ewc_accounting(aux_fit_stats, fisher_flops)
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
        self._buffer: List[Any] = []
        self._all_seen_orderings: List[List[str]] = []
        self._seen: int = 0
        self._rng = random.Random(getattr(cfg, "seed", 1337))
        self._last_replay_metadata: Dict[str, Any] = self.replay_buffer_metadata()

    def _replay_record(self, ordering: Sequence[str], transitions: Sequence[DemoTransition] = ()) -> Dict[str, Any]:
        return {"ordering": list(ordering), "transitions": tuple(transitions)}

    def _record_ordering(self, record: Any) -> List[str]:
        if isinstance(record, dict):
            return list(record.get("ordering", ()))
        return list(record)

    def _buffer_add(self, record: Any) -> None:
        cap = max(1, int(self.cfg.er_buffer_size))
        self._seen += 1
        stored = self._replay_record(self._record_ordering(record), record.get("transitions", ()) if isinstance(record, dict) else ())
        if len(self._buffer) < cap:
            self._buffer.append(stored)
            return
        j = self._rng.randrange(self._seen)
        if j < cap: self._buffer[j] = stored

    def replay_buffer_metadata(self) -> Dict[str, Any]:
        n_buffered = len(getattr(self, "_buffer", []))
        footprint = sum(len(self._record_ordering(record)) for record in getattr(self, "_buffer", []))
        return {"policy": "uniform_reservoir", "er_buffer_size": int(self.cfg.er_buffer_size), "er_batch_size": int(self.cfg.er_batch_size), "n_buffered": int(n_buffered), "n_seen": int(getattr(self, "_seen", 0)), "replay_memory_footprint": int(footprint), "replay_memory_steps": int(footprint)}

    def _record_committed_replay_demo(
        self,
        recipe_id: str,
        variant_hash_: str,
        ordering: Tuple[str, ...],
        *,
        transitions: Tuple[DemoTransition, ...] = (),
        session_step: int,
        action_step: int,
        source_mode: str,
        entry: Optional[Entry] = None,
    ) -> None:
        demo = list(ordering)
        self._all_seen_orderings.append(demo)
        self._buffer_add(self._replay_record(demo, transitions))

    def _retrain(self) -> None:
        if self._frozen: return
        retrain_t0 = time.perf_counter()
        self.retrain_cycle += 1

        if not self._buffer:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return

        # Uniform reservoir sample at every refit, then dedup.
        batch_size = max(1, int(self.cfg.er_batch_size))
        k = min(batch_size, len(self._buffer))
        sampled = self._rng.sample(self._buffer, k) if k > 0 else []
        demos: List[Any] = []
        for s in sampled:
            demos.append(s)
        if not demos:
            self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_heads()
            return
        self._last_replay_metadata = {**self.replay_buffer_metadata(), "n_sampled": int(len(demos)), "sampler": "uniform_without_replacement"}

        build_t0 = time.perf_counter()
        trajectories, dropped_total = self._build_trajectories(demos)
        build_wall_s = time.perf_counter() - build_t0
        weights = self._length_normalized_demo_weights(demos, [1.0] * len(demos))  # uniform demo mass; no decay leakage.
        self._reset_heads()
        fit_t0 = time.perf_counter()
        self._fit_heads(trajectories, weights)
        fit_wall_s = time.perf_counter() - fit_t0
        self._record_retrain_event(dropped_actions=dropped_total, active_demos=len(demos), total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_retrain_flops(trajectories))






class BigramOnlyAgent(AdaptiveHRCAgent):
    """True bigram floor: a fresh `Counter[(prev_token, next_token)] -> Categorical`.
    Does NOT inherit the parent's state-conditional N-gram head. Predict reads from this agent's own bigram counter, populated on each retrain from the active set with unit counts. The predict path completely ignores IRL; the baseline answers "what does a vanilla last-token Markov chain give?"."""

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
        # Each counter update performs at least a lookup, addition, and store.
        estimated_flops = float(3.0 * max(1, scalar_updates))
        self._custom_fit_stats = {**_symbolic_fit_stats("bigram", trajectories, estimated_flops=estimated_flops), "scalar_counter_updates": float(scalar_updates)}

    def baseline_memory_metadata(self) -> Dict[str, Any]:
        bigram_edges = sum(len(row) for row in self._bigram.values())
        unigram_types = len(self._unigram)
        counter_mass = sum(self._unigram.values()) + sum(sum(row.values()) for row in self._bigram.values())
        return {
            "policy": "token_bigram",
            "bigram_contexts": int(len(self._bigram)),
            "bigram_edges": int(bigram_edges),
            "unigram_types": int(unigram_types),
            "counter_entries": int(unigram_types + bigram_edges),
            "counter_mass": int(counter_mass),
            "baseline_memory_footprint": int(unigram_types + bigram_edges),
            "baseline_memory_steps": int(counter_mass),
        }

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
            self._set_action_policy_stats(None, None, "baseline_empty")
            return
        conf = max(float(v) for v in dist.values())
        ent = 0.0
        for p in dist.values():
            if p > 0:
                ent -= float(p) * math.log(float(p))
        if len(dist) > 1:
            ent /= math.log(len(dist))
        self._set_action_policy_stats(conf, ent, "baseline_policy")




BASELINE_AGENTS = {
    "offline_pretrained_frozen": OfflinePretrainedFrozenAgent,
    "offline_all_recipes_identity_frozen": OfflinePretrainedFrozenAgent,
    "adaptive_decay": AdaptiveDecayAgent,
    "latest_only": LatestOnlyPreferenceAgent,
    "fixed_decay": FixedDecayAgent,
    "no_decay": NoDecayAgent,
    "bc": BehaviorCloningAgent,
    # Kirkpatrick et al. 2017: cumulative-Fisher, precision-weighted anchor.
    "ewc": EWCAgent,
    "experience_replay_bc": ExperienceReplayAgent,
    "bigram": BigramOnlyAgent,
}
