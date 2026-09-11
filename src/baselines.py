"""Runnable comparison baselines for the publication benchmark."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .adaptive_agent import ACTION_MASK_CONTRACT, AdaptiveAgent, BASELINE_TRAIN_POLICY, MODE_ONLINE, StepResult
from .domain import DomainAdapter, default_domain
from .memory import MatchResult, ReplayMemory, StateTransition, MemoryItem, VariantKey
from .models import (Settings, DEFAULT_SETTINGS, Demonstration, RunningScaler, index_demos)
from .representations import Observation
from .llm_baseline import InContextLlmAgent
StateVector = Tuple[int, ...]


@dataclass(frozen=True)
class ReplayItem:
    ordering: Tuple[str, ...]
    transitions: Tuple[StateTransition, ...] = ()
    key: Optional[VariantKey] = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:    return logits.astype(np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    return (exp / denom).astype(np.float32)


class BehaviorCloner:
    """Linear softmax imitation model over state and one-hot prefix features."""
    def __init__(self, settings: Settings = DEFAULT_SETTINGS, domain: Optional[DomainAdapter] = None):
        self.settings = settings
        self.domain = domain or default_domain()
        self.normalizer = RunningScaler()
        self.state_dim: int = 0
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.action_ids: Dict[str, int] = {}
        self.action_labels: Dict[int, str] = {}
        self.history_action_ids: Dict[str, int] = {}
        self.history_size: int = max(0, int(self.settings.bc_history))
        self.last_fit_stats: Dict[str, object] = {}

    def reset(self) -> None:
        self.weights = None
        self.bias = None
        self.action_ids = {}
        self.action_labels = {}
        self.history_action_ids = {}
        self.history_size = self._history_lags()
        self.last_fit_stats = {"model_family": "behavior_cloning", "estimated_flops": 0.0, "flop_accounting_scope": "behavior_cloning_fit_dense_numeric_only", "flop_cross_model_comparable": False}

    def _history_lags(self) -> int:
        """Action lags the cloner conditions on; zero is state-only.

        This used to clamp to one, so a configuration asking for no history
        still received a lag of it, and the step counter below was
        unconditional. Both are now honoured, because "no within-episode
        history" is the condition that separates this model family from a
        state-indexed policy rather than a setting nobody sets.
        """
        return max(0, int(self.settings.bc_history))

    def _history_dim(self) -> int:
        counter = 1 if bool(self.settings.bc_prefix_length_feature) else 0
        return self._history_lags() * len(self.history_action_ids) + counter

    def _prefix_features(self, prefix: Sequence[str]) -> np.ndarray:
        history = self._history_lags()
        vocab_size = len(self.history_action_ids)
        counter = 1 if bool(self.settings.bc_prefix_length_feature) else 0
        out = np.zeros(history * vocab_size + counter, dtype=np.float32)
        if counter: out[-1] = float(len(prefix))
        for lag in range(1, history + 1):
            if len(prefix) < lag: continue
            action = prefix[-lag]
            action_index = self.history_action_ids.get(action)
            if action_index is None: continue
            out[(lag - 1) * vocab_size + action_index] = 1.0
        return out

    def _compose_features(self, state_features: np.ndarray, prefix: Sequence[str]) -> np.ndarray:
        return np.concatenate([np.asarray(state_features, dtype=np.float32), self._prefix_features(prefix)])

    def _prepare_examples(self, demonstrations: Sequence[Demonstration], demo_weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
        unique_actions = sorted({a for demo in demonstrations for _, a in demo if a != "stop"})
        if not unique_actions: return (np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), {}, {})
        self.history_action_ids = {action: index for index, action in enumerate(unique_actions)}
        state_ids, state_vectors, action_ids, action_labels = index_demos(demonstrations, unique_actions=unique_actions)
        state_matrix, _, _ = self.domain.build_features(state_vectors, normalizer=self.normalizer, update_normalizer=True)
        self.state_dim = int(state_matrix.shape[1]) if state_matrix.ndim == 2 else 0
        feature_rows: List[np.ndarray] = []
        targets: List[int] = []
        example_weights: List[float] = []
        for demo, demo_weight in zip(demonstrations, demo_weights):
            prefix: List[str] = []
            weight = max(float(demo_weight), 0.0)
            for state, action in demo:
                if action == "stop":    break
                state_id = state_ids.get(state)
                if state_id is None:       continue
                feature_rows.append(self._compose_features(state_matrix[state_id], prefix))
                targets.append(action_ids[action])
                example_weights.append(weight)
                prefix.append(action)
        if not feature_rows: return (np.zeros((0, self.state_dim + self._history_dim()), dtype=np.float32),  np.zeros(0, dtype=np.int64),    np.zeros(0, dtype=np.float32),      action_ids,  action_labels)
        return (np.vstack(feature_rows).astype(np.float32), np.asarray(targets, dtype=np.int64), np.asarray(example_weights, dtype=np.float32), action_ids, action_labels)

    def _warm_start(self, feature_dim: int, action_ids: Dict[str, int], previous_weights: Optional[np.ndarray], previous_bias: Optional[np.ndarray], previous_action_ids: Dict[str, int],
        previous_state_dim: int, previous_history_ids: Dict[str, int], previous_history_size: int) -> Tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((feature_dim, len(action_ids)), dtype=np.float32)
        bias = np.zeros(len(action_ids), dtype=np.float32)
        if previous_weights is None or previous_bias is None: return weights, bias
        current_history_size = self._history_lags()
        state_rows = min(previous_state_dim, self.state_dim, previous_weights.shape[0], weights.shape[0])
        previous_history_offset = max(0, previous_state_dim)
        history_offset = max(0, self.state_dim)
        previous_history_vocab_size = len(previous_history_ids)
        history_vocab_size = len(self.history_action_ids)
        shared_history_size = min(max(0, int(previous_history_size)), current_history_size)
        previous_length_row = previous_history_offset + max(0, int(previous_history_size)) * previous_history_vocab_size
        length_row = history_offset + current_history_size * history_vocab_size
        for action, current_index in action_ids.items():
            previous_index = previous_action_ids.get(action)
            if previous_index is None: continue
            if previous_index >= previous_weights.shape[1] or current_index >= weights.shape[1]: continue
            if state_rows > 0: weights[:state_rows, current_index] = previous_weights[:state_rows, previous_index]
            for lag_index in range(shared_history_size):
                previous_base = previous_history_offset + lag_index * previous_history_vocab_size
                current_base = history_offset + lag_index * history_vocab_size
                for action, current_action_index in self.history_action_ids.items():
                    previous_action_index = previous_history_ids.get(action)
                    if previous_action_index is None: continue
                    previous_row = previous_base + previous_action_index
                    current_row = current_base + current_action_index
                    if previous_row < previous_weights.shape[0] and current_row < weights.shape[0]: weights[current_row, current_index] = previous_weights[previous_row, previous_index]
            # Absent when the step counter is disabled, so the row exists in
            # neither layout and there is nothing to carry across.
            if bool(self.settings.bc_prefix_length_feature) and previous_length_row < previous_weights.shape[0] and length_row < weights.shape[0]: weights[length_row, current_index] = previous_weights[previous_length_row, previous_index]
            bias[current_index] = previous_bias[previous_index]
        return weights, bias

    def fit(self, demonstrations: Sequence[Demonstration], demo_weights: Optional[Sequence[float]] = None) -> None:
        if not demonstrations:
            self.reset()
            return
        previous_weights = self.weights
        previous_bias = self.bias
        previous_action_ids = dict(self.action_ids)
        previous_state_dim = int(self.state_dim)
        previous_history_ids = dict(self.history_action_ids)
        previous_history_size = int(self.history_size)
        self.normalizer = RunningScaler()
        weights = [1.0] * len(demonstrations) if demo_weights is None else [float(w) for w in demo_weights]
        feature_rows, targets, sample_weights, action_ids, action_labels = (self._prepare_examples(demonstrations, weights))
        if feature_rows.size == 0 or not action_ids:
            self.reset()
            return

        feature_dim = int(feature_rows.shape[1])
        warm_start = previous_weights is not None and previous_bias is not None
        if warm_start:
            model_weights, model_bias = self._warm_start(feature_dim, action_ids, previous_weights, previous_bias, previous_action_ids, previous_state_dim, previous_history_ids, previous_history_size)
            epochs = int(self.settings.bc_warm_epochs)
        else:
            model_weights = np.zeros((feature_dim, len(action_ids)), dtype=np.float32)
            model_bias = np.zeros(len(action_ids), dtype=np.float32)
            epochs = int(self.settings.bc_cold_epochs)

        learning_rate = float(self.settings.bc_learning_rate)
        l2_penalty = float(self.settings.bc_l2)
        batch_size = max(1, int(self.settings.bc_batch))
        rng = np.random.default_rng(int(self.settings.seed))
        indices = np.arange(len(targets), dtype=np.int64)

        for _ in range(max(1, epochs)):
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                batch_features = feature_rows[batch]
                batch_targets = targets[batch]
                batch_weights = sample_weights[batch].astype(np.float32)
                batch_weight = max(float(np.sum(batch_weights)), 1e-8)
                logits = batch_features @ model_weights + model_bias[None, :]
                probs = _softmax(logits)
                delta = probs.copy()
                delta[np.arange(len(batch_targets)), batch_targets] -= 1.0
                delta *= (batch_weights[:, None] / batch_weight).astype(np.float32)
                weight_gradient = (batch_features.T @ delta).astype(np.float32) + l2_penalty * model_weights
                bias_gradient = np.sum(delta, axis=0).astype(np.float32)
                model_weights -= learning_rate * weight_gradient
                model_bias -= learning_rate * bias_gradient

        self.weights = model_weights.astype(np.float32)
        self.bias = model_bias.astype(np.float32)
        self.action_ids = dict(action_ids)
        self.action_labels = dict(action_labels)
        self.history_size = self._history_lags()
        n_examples = int(feature_rows.shape[0])
        n_actions = int(len(action_ids))
        param_count = int(feature_dim * n_actions + n_actions)
        # Dense fit accounting keeps per-example GEMMs and per-batch updates separate.
        n_batches = int(math.ceil(n_examples / float(batch_size)))
        forward = 2.0 * n_examples * feature_dim * n_actions
        softmax_delta = 7.0 * n_examples * n_actions
        grad = 2.0 * feature_dim * n_examples * n_actions + float(n_examples * n_actions)
        optimizer_update = float(n_batches * (4 * feature_dim * n_actions + 2 * n_actions))
        estimated_flops = float(max(1, epochs) * (forward + softmax_delta + grad + optimizer_update))
        self.last_fit_stats = {"model_family": "behavior_cloning", "estimated_flops": estimated_flops, "n_demonstrations": float(len(demonstrations)), "n_examples": float(n_examples), "n_actions": float(n_actions), "feature_dim": float(feature_dim), "state_dim": float(self.state_dim),
            "history_dim": float(self._history_dim()), "parameter_count": float(param_count), "history_encoding": "one_hot_active_action_vocab", "history_vocab_size": float(len(self.history_action_ids)), "history_feature_dim": float(self._history_dim()), "epochs": float(max(1, epochs)), "batch_size": float(batch_size), "n_batches_per_epoch": float(n_batches), "flop_forward": float(forward), "flop_softmax_delta": float(softmax_delta), "flop_gradient": float(grad), "flop_optimizer_update": float(optimizer_update), "flop_accounting_scope": "behavior_cloning_fit_dense_numeric_only", "flop_cross_model_comparable": False, "warm_start": 1.0 if warm_start else 0.0}

    def predict(self, state: StateVector, prefix: Sequence[str]) -> Dict[str, float]:
        if self.weights is None or self.bias is None or not self.action_labels: return {}
        features, _, _ = self.domain.build_features({0: tuple(state)}, normalizer=self.normalizer)
        if features.size == 0: return {}
        x = self._compose_features(features[0], prefix)[None, :]
        probs = _softmax(x @ self.weights + self.bias[None, :])[0]
        return {self.action_labels[index]: float(prob) for index, prob in enumerate(probs) if float(prob) > 0.0}


def _without_proposed_components(settings: Settings) -> Settings:
    """Remove Full's memory pin and both semantic predictor components."""
    return replace(settings, pin_latest=False, semantic_fallback_enabled=False, latent_strategy_enabled=False)


class BaselineAgent(AdaptiveAgent):
    """Shared continual-fit policy for comparison systems."""
    RETRAIN_POLICY = BASELINE_TRAIN_POLICY


class UnpinnedAgent(BaselineAgent):
    """IRL-only adaptive decay without latest-preference protection."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, **kwargs):
        super().__init__(settings=_without_proposed_components(settings), **kwargs)

class FrozenAgent(BaselineAgent):
    """Train offline, then freeze all learned state during deployment."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self._deployment_locked = False
        self._deployment_step = 0
        self._deployment_action_policy: Dict[str, Any] = {}
        self._offline_pretraining_metadata: Dict[str, Any] = {}

    def _clear_episode(self) -> None:
        """Discard only transient deployment state; learned state stays fixed."""
        self.mode = MODE_ONLINE
        self.pending_demo = []
        self.pending_trace = []
        self.current_prefix = []
        self.current_trace = []
        self._policy_history = []
        self._prediction_mismatch_count = 0
        self._latent_strategy_confirmed = False
        self._deployment_step = 0
        self._deployment_action_policy = {}

    def lock_deployment(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Freeze the learned offline policy before the deployment stream."""
        if self._deployment_locked:
            return self.training_stats()
        fit_times = [float(value) for value in self.retrain_fit_wall_times]
        total_times = [float(value) for value in self.retrain_total_wall_times]
        build_times = [float(value) for value in self.retrain_build_wall_times]
        flops = [float(value) for value in self.retrain_flop_estimates]
        self._deployment_locked = True
        self._offline_pretraining_metadata = {"training_regime": "offline_pretraining_then_frozen_deployment", "comparison_scope": "training_regime_reference_not_method_replication", "updates_allowed": False,
            "train_count": int(len(total_times)), "train_wall_s": float(sum(total_times)), "fit_wall_s": float(sum(fit_times)), "build_wall_s": float(sum(build_times)), "train_flops": float(sum(flops)),
            "train_variants": int(len(self.replay.active)), "train_steps": int(sum(len(entry.ordering) for entry in self.replay.active.values())), **dict(metadata or {})}
        self._clear_episode()
        return self.training_stats()

    def training_stats(self) -> Dict[str, Any]:
        return dict(self._offline_pretraining_metadata)

    def _set_policy_stats(self, confidence: Optional[float], entropy: Optional[float], reason: str, *, margin: Optional[float] = None, source: Optional[str] = None) -> None:
        if not self._deployment_locked:
            super()._set_policy_stats(confidence, entropy, reason, margin=margin, source=source)
            return
        # Keep evaluator telemetry episode-local.
        self._deployment_action_policy = {"action_required": True, "confidence": confidence, "raw_confidence": confidence, "margin": margin, "entropy": entropy,
            "entropy_basis": "prediction_support", "predictor": source or "maxent", "final_confidence": confidence, "final_margin": margin, "final_entropy": entropy, "reason": reason}

    def policy_stats(self) -> Dict[str, Any]:
        if self._deployment_locked: return {**ACTION_MASK_CONTRACT, **dict(self._deployment_action_policy), **dict(self._last_action_mask_stats), **dict(getattr(self.maxent, "last_prediction_stats", {}) or {})}
        return super().policy_stats()

    def start_demo(self) -> None:
        if not self._deployment_locked:
            super().start_demo()
            return
        self._clear_episode()

    def observe(self, observation: Observation, ground_truth_recipe: Optional[str] = None, *, precomputed_distribution: Optional[Dict[str, float]] = None, precomputed_prediction: Optional[str] = None) -> StepResult:
        if not self._deployment_locked:
            return super().observe(observation, ground_truth_recipe=ground_truth_recipe, precomputed_distribution=precomputed_distribution, precomputed_prediction=precomputed_prediction)
        self._deployment_step += 1
        first_online_action = not self.current_prefix
        distribution = ({} if first_online_action else (dict(precomputed_distribution) if precomputed_distribution is not None else self.predict_actions(self.current_prefix)))
        predicted = (None if first_online_action else (precomputed_prediction if precomputed_distribution is not None else (self.rank_actions(distribution, k=1)[0] if distribution else None)))
        action = observation.action
        self.current_prefix.append(action)
        return StepResult(step=self._deployment_step, predicted=predicted, actual=action, correct=predicted == action, mode=MODE_ONLINE, event="offline_frozen_deployment")

    def end_demo(self) -> MatchResult:
        if not self._deployment_locked: return super().end_demo()
        self._clear_episode()
        return MatchResult("offline_frozen", None, None, 0.0, 0.0)

    def _retrain(self) -> None:
        if self._deployment_locked: return
        super()._retrain()

    def refresh(self) -> None:
        if self._deployment_locked: return
        super().refresh()


class FixedDecayAgent(BaselineAgent):
    """IRL-only fixed-rate forgetting; adaptive grace horizons disabled."""
    def __init__(self, settings: Settings = DEFAULT_SETTINGS, **kwargs):
        super().__init__(settings=_without_proposed_components(settings), **kwargs)
        self.replay = ReplayMemory(self.settings, policy="fixed")


class NoDecayAgent(BaselineAgent):
    """IRL-only control retaining every variant at unit weight."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, **kwargs):
        super().__init__(settings=_without_proposed_components(settings), **kwargs)
        self.replay = ReplayMemory(self.settings, policy="none")


class LatestAgent(UnpinnedAgent):
    """Keeps only the most recently observed preference variant per recipe."""

    def _register_if_live(self, recipe_id: str, actions: List[str], step: int, transitions: Optional[Sequence[StateTransition]] = None):
        variant = super()._register_if_live(recipe_id, actions, step, transitions=transitions)
        slot = self.library.variants.get(recipe_id, {})
        for variant_id in list(slot.keys()):
            if variant_id == variant.variant_id: continue
            del slot[variant_id]
            self.replay.discard(recipe_id, variant_id)
        self.library.latest[recipe_id] = variant.variant_id
        return variant

class BehaviorCloningAgent(BaselineAgent):
    """Supervised next-action imitation over unweighted, nondecaying storage."""

    # Storage is a class attribute so a subclass can hold the predictor fixed
    # and vary only the memory policy; see MemoryMatchedBcAgent.
    MEMORY_POLICY = "none"
    PIN_LATEST = False

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, **kwargs):
        super().__init__(settings=replace(
            _without_proposed_components(settings), pin_latest=self.PIN_LATEST,
        ), **kwargs)
        self.replay = ReplayMemory(self.settings, policy=self.MEMORY_POLICY)
        self.cloner = BehaviorCloner(settings=self.settings, domain=self.domain)

    def predictor_name(self) -> str:
        return "behavior_cloning"

    def irl_feature_name(self) -> str:
        return "not_applicable"

    def _fit_predictors(self, trajectories, weights, *, warm_start: bool, records=None) -> None:
        self.cloner.fit(trajectories, weights)

    def _estimate_flops(self, trajectories) -> float:
        """Approximate BC fit cost from the fit that actually just ran."""
        stats = getattr(self.cloner, "last_fit_stats", {}) or {}
        estimate = stats.get("estimated_flops") if isinstance(stats, dict) else None
        if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)):
            return float(estimate)
        # Fallback for external or unfitted models without fit statistics.
        n_examples = sum(1 for demo in trajectories for _state, action in demo if action != "stop")
        n_actions = max(1, len({action for demo in trajectories for _state, action in demo if action != "stop"}))
        state_dim = len(trajectories[0][0][0]) if trajectories and trajectories[0] else 1
        feature_dim = max(1, state_dim + self.cloner._history_dim())
        epochs = int(self.settings.bc_cold_epochs)
        return float(2 * n_examples * n_actions * feature_dim * max(1, epochs))

    def _reset_predictors(self) -> None:
        super()._reset_predictors()
        self.cloner = BehaviorCloner(settings=self.settings, domain=self.domain)

    def audit_pruning(self, max_prefixes: int = 24, tolerance: float = 5e-2) -> Dict[str, object]:
        """Behaviour-cloning form of the audit in AdaptiveAgent.

        Same three quantities and the same key names, so the arms are directly
        comparable. Cloner fits are deterministic, so ``deployed_path_dependence``
        is structurally zero here rather than merely small.
        """
        entries = self.replay.active_items()
        pruned_records = list(self.replay.pruned.values())
        active_keys = {entry.key for entry in entries}
        pruned_keys = {record.key for record in pruned_records}
        inputs_verified = not (active_keys & pruned_keys)
        base: Dict[str, object] = {"passed": bool(inputs_verified), "active_only_training_inputs_verified": bool(inputs_verified),
                "comparison": "cold_active_vs_cold_active_plus_pruned", "model_family": "behavior_cloning",
                "n_active_variants": len(entries), "n_pruned_variants": len(pruned_records), "tolerance": float(tolerance)}
        if not entries:
            return {**base, "max_l1": 0.0, "mean_l1": 0.0, "n_prefixes": 0, "pruned_available": False,
                    "redundancy_max_l1": 0.0, "redundancy_mean_l1": 0.0,
                    "deployed_path_dependence_max_l1": 0.0, "deployed_path_dependence_mean_l1": 0.0}

        def cold_fit(records, record_weights) -> BehaviorCloner:
            trajectories, _dropped = self._build_demos(records)
            cloner = BehaviorCloner(settings=self.settings, domain=self.domain)
            cloner.fit(trajectories, self._demo_weights(records, record_weights))
            return cloner

        active_reference = cold_fit(entries, [float(entry.weight) for entry in entries])
        path_dependence = self._compare_policies(entries, lambda state, prefix: self.cloner.predict(state, prefix), lambda state, prefix: active_reference.predict(state, prefix),
            max_prefixes=max_prefixes, tolerance=tolerance)
        deployed = {"deployed_path_dependence_max_l1": float(path_dependence["max_l1"]),
                    "deployed_path_dependence_mean_l1": float(path_dependence["mean_l1"])}

        if not pruned_records:
            return {**base, **deployed, "max_l1": 0.0, "mean_l1": 0.0, "n_prefixes": int(path_dependence["n_prefixes"]),
                    "pruned_available": False, "redundancy_max_l1": 0.0, "redundancy_mean_l1": 0.0}

        combined = list(entries) + pruned_records
        restored_reference = cold_fit(combined, [float(entry.weight) for entry in entries] + [1.0] * len(pruned_records))
        redundancy = self._compare_policies(entries, lambda state, prefix: active_reference.predict(state, prefix), lambda state, prefix: restored_reference.predict(state, prefix),
            max_prefixes=max_prefixes, tolerance=tolerance)
        return {**base, **deployed, **redundancy, "pruned_available": True,
                "redundancy_max_l1": float(redundancy["max_l1"]), "redundancy_mean_l1": float(redundancy["mean_l1"])}

    def predict_actions(
        self,
        prefix=None,
        *,
        state=None,
        actor_id: int = 0,
        action_universe=None,
    ) -> Dict[str, float]:
        """Use the same state and action mask interface as every other arm."""
        prefix_actions = list(prefix) if prefix is not None else list(self.current_prefix)
        encoded_state = (
            self._replay_prefix(prefix_actions)
            if state is None else self.domain.state_key(state, actor_id=actor_id)
        )
        distribution = self.cloner.predict(encoded_state, prefix_actions)
        if action_universe is not None:
            allowed = set(map(str, action_universe))
            distribution = {
                action: probability for action, probability in distribution.items()
                if action in allowed
            }
        distribution = self._apply_shared_action_mask(encoded_state, distribution)
        if distribution:
            confidence, entropy, _margin = self._prediction_stats(distribution)
            self._set_policy_stats(confidence, entropy, "baseline_policy")
        else: self._set_policy_stats(None, None, "baseline_empty")
        return distribution


class MemoryMatchedBcAgent(BehaviorCloningAgent):
    """BC under Full's adaptive-decay and latest-pin memory policy.

    ``bc`` differs from Full in two places at once: it swaps MaxEnt IRL for a
    linear cloner and it retains every variant at unit weight.  Its margin
    therefore cannot be attributed to either change.  This arm moves only the
    predictor, so the ``full``/``bc_adaptive``/``bc``/``no_decay`` quartet
    separates the model family from the retention policy.

    Full's two semantic components stay disabled because both are reached
    through the MaxEnt predictor, which this arm does not use.  ``pin_latest``
    is memory-side, so it is restored here.
    """

    MEMORY_POLICY = "adaptive"
    PIN_LATEST = True

    def predictor_name(self) -> str:
        return "behavior_cloning_adaptive_memory"


class EwcAgent(BaselineAgent):
    """IRL plus diagonal-Fisher EWC over nondecaying storage."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, **kwargs):
        super().__init__(settings=_without_proposed_components(settings), **kwargs)
        self.replay = ReplayMemory(self.settings, policy="none")
        self._fisher_total: Optional[np.ndarray] = None
        self._weighted_anchor: Optional[np.ndarray] = None
        self._anchor: Optional[np.ndarray] = None
        self._fisher: Optional[np.ndarray] = None
        self._pending_demos: List[Any] = []
        self._custom_fit_stats: Dict[str, Any] = {}

    def predictor_name(self) -> str:
        return "ewc_maxent"

    @staticmethod
    def _resize_to(vector: Optional[np.ndarray], size: int) -> np.ndarray:
        """Return a float32 vector of the requested size."""
        if vector is None: return np.zeros(size, dtype=np.float32)
        if vector.shape[0] == size: return vector
        resized = np.zeros(size, dtype=np.float32)
        shared_size = min(vector.shape[0], size)
        resized[:shared_size] = vector[:shared_size]
        return resized

    def _record_demo(self, recipe_id: str, variant_id: str, ordering: Tuple[str, ...], *, transitions: Tuple[StateTransition, ...] = (), demo_step: int, action_step: int, source_mode: str, entry: Optional[MemoryItem] = None) -> None:
        if entry is not None: self._pending_demos.append(entry)
        else: self._pending_demos.append({"ordering": tuple(ordering), "transitions": tuple(transitions)})

    @staticmethod
    def _estimate_fisher_flops(trajectories: Sequence[Demonstration], feature_count: int, mean_candidates: float = 2.0) -> float:
        """Cost of the diagonal empirical Fisher.

        Per demonstrated visit the estimator forms a policy-weighted mean over
        ``mean_candidates`` successor feature rows, subtracts it from the taken
        row, then squares and accumulates: roughly ``2*k + 4`` feature-width
        operations rather than the 3 the feature-second-moment version needed.
        """
        state_visits = sum(len(demo) for demo in trajectories)
        per_visit = 2.0 * max(1.0, float(mean_candidates)) + 4.0
        return float(per_visit * max(0, state_visits) * max(1, int(feature_count)))

    def _record_costs(self, auxiliary_stats: Dict[str, Any], fisher_flops: float) -> None:
        primary_stats = dict(getattr(self.maxent, "last_fit_stats", {}) or {})
        primary_estimate = primary_stats.get("estimated_flops", 0.0)
        auxiliary_estimate = auxiliary_stats.get("estimated_flops", 0.0)
        try:                                primary_flops = float(primary_estimate)
        except (TypeError, ValueError):     primary_flops = 0.0
        try:                                auxiliary_flops = float(auxiliary_estimate)
        except (TypeError, ValueError):     auxiliary_flops = 0.0
        if not math.isfinite(primary_flops):    primary_flops = 0.0
        if not math.isfinite(auxiliary_flops):  auxiliary_flops = 0.0
        try:                                measured_fisher_flops = float(fisher_flops)
        except (TypeError, ValueError):     measured_fisher_flops = 0.0
        if not math.isfinite(measured_fisher_flops):    measured_fisher_flops = 0.0
        self._custom_fit_stats = {"estimated_flops": float(primary_flops + auxiliary_flops + measured_fisher_flops),    "ewc_primary_flops": float(primary_flops),      "ewc_auxiliary_flops": float(auxiliary_flops),
            "ewc_fisher_flops": float(measured_fisher_flops),   "ewc_aux_iterations_run": float(auxiliary_stats.get("iterations_run", 0.0) or 0.0),     "ewc_aux_n_states": float(auxiliary_stats.get("n_states", 0.0) or 0.0), "ewc_aux_n_features": float(auxiliary_stats.get("n_features", 0.0) or 0.0)}

    def _fit_predictors(self, trajectories, weights, *, warm_start: bool, records=None) -> None:
        self._custom_fit_stats = {}
        self.maxent.fit(trajectories, weights, ewc_anchor=self._anchor, ewc_fisher=self._fisher)
        if self.maxent.reward_weights is None:
            self._record_costs({}, 0.0)
            return

        pending = list(self._pending_demos)
        self._pending_demos = []
        if pending:
            task_trajectories, _ = self._build_demos(pending)
            task_weights = self._demo_weights(pending, [1.0] * len(pending))
        else:
            task_trajectories = list(trajectories)
            task_weights = [float(w) for w in weights]
        # Same stream as the primary model, so the auxiliary fit consumes
        # initializations in the agent's single reproducible sequence.
        task_maxent = type(self.maxent)(settings=self.settings, domain=self.domain, rng=self._init_rng)
        task_maxent.fit(task_trajectories,task_weights, ewc_anchor=self._anchor, ewc_fisher=self._fisher)
        auxiliary_stats = dict(getattr(task_maxent, "last_fit_stats", {}) or {})
        if task_maxent.reward_weights is None:
            self._record_costs(auxiliary_stats, 0.0)
            return

        new_reward_weights = task_maxent.reward_weights.astype(np.float32)
        fisher_flops = self._estimate_fisher_flops(task_trajectories, int(new_reward_weights.shape[0]))
        new_fisher = task_maxent.fisher(task_trajectories, task_weights)
        self._record_costs(auxiliary_stats, fisher_flops)
        self._custom_fit_stats.update(dict(getattr(task_maxent, "last_fisher_stats", {}) or {}))
        if new_fisher is None: return
        new_fisher = new_fisher.astype(np.float32)

        feature_count = new_reward_weights.shape[0]
        self._fisher_total = self._resize_to(self._fisher_total, feature_count)
        self._weighted_anchor = self._resize_to(self._weighted_anchor, feature_count)

        self._fisher_total += new_fisher
        self._weighted_anchor += new_fisher * new_reward_weights

        # Precision-weighted consolidated anchor.
        precision_floor = float(self.settings.ewc_precision_floor)
        fisher_cap = float(self.settings.fisher_cap)
        safe_fisher = np.maximum(self._fisher_total, precision_floor)
        self._anchor = (self._weighted_anchor / safe_fisher).astype(np.float32)
        self._fisher = np.clip(self._fisher_total, 0.0, fisher_cap).astype(np.float32)


class ReplayBcAgent(BehaviorCloningAgent):
    """BC with a bounded uniform reservoir and nondecaying storage."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self._buffer: List[ReplayItem] = []
        self._seen: int = 0
        self._rng = random.Random(int(self.settings.seed))

    def predictor_name(self) -> str:
        return "experience_replay_behavior_cloning"

    def _buffer_item(self, ordering: Sequence[str], transitions: Sequence[StateTransition] = (), key: Optional[VariantKey] = None) -> ReplayItem:
        return ReplayItem(tuple(ordering), tuple(transitions), key)

    def _store_demo(self, record: Any) -> List[str]:
        return list(getattr(record, "ordering", record))

    def _add_to_buffer(self, record: Any) -> None:
        cap = max(1, int(self.settings.replay_capacity))
        self._seen += 1
        stored = self._buffer_item(self._store_demo(record), getattr(record, "transitions", ()), getattr(record, "key", None))
        if len(self._buffer) < cap:
            self._buffer.append(stored)
            return
        j = self._rng.randrange(self._seen)
        if j < cap: self._buffer[j] = stored

    def replay_stats(self) -> Dict[str, Any]:
        n_buffered = len(getattr(self, "_buffer", []))
        footprint = sum(len(self._store_demo(record)) for record in getattr(self, "_buffer", []))
        return {"policy": "uniform_reservoir", "replay_capacity": int(self.settings.replay_capacity), "replay_batch": int(self.settings.replay_batch), "n_buffered": int(n_buffered), "n_seen": int(getattr(self, "_seen", 0)), "buffer_items": int(footprint), "buffer_steps": int(footprint)}

    def _record_demo(self, recipe_id: str, variant_id: str, ordering: Tuple[str, ...], *, transitions: Tuple[StateTransition, ...] = (), demo_step: int, action_step: int, source_mode: str, entry: Optional[MemoryItem] = None) -> None:
        self._add_to_buffer(ReplayItem(ordering, transitions, (recipe_id, variant_id)))

    def _retrain(self) -> None:
        if self._frozen: return
        retrain_t0 = time.perf_counter()
        self.retrain_cycle += 1

        active = set(self.replay.active)
        eligible = [record for record in self._buffer if record.key is None or record.key in active]
        if not eligible:
            self._log_training(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            self._reset_predictors()
            return

        batch_size = max(1, int(self.settings.replay_batch))
        k = min(batch_size, len(eligible))
        demos = self._rng.sample(eligible, k)
        build_t0 = time.perf_counter()
        trajectories, dropped_total = self._build_demos(demos)
        # This arm fits from its own sampled buffer rather than active replay,
        # so it must index the decision regimes from the same trajectories it
        # actually trained on, or its decisions would look uniformly unseen.
        self._index_observed_actions(trajectories)
        build_wall_s = time.perf_counter() - build_t0
        base_weights = [float(self.replay.active[record.key].weight) if record.key in self.replay.active else 1.0 for record in demos]
        weights = self._demo_weights(demos, base_weights)
        self._reset_predictors()
        fit_t0 = time.perf_counter()
        self._fit_predictors(trajectories, weights, warm_start=False, records=demos)
        fit_wall_s = time.perf_counter() - fit_t0
        self._log_training(dropped_actions=dropped_total, active_demos=len(demos), total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_flops(trajectories))


BASELINE_AGENTS = {
    "frozen": FrozenAgent,
    "offline_default": FrozenAgent,
    "offline_all": FrozenAgent,
    "unpinned": UnpinnedAgent,
    "latest": LatestAgent,
    "fixed": FixedDecayAgent,
    "no_decay": NoDecayAgent,
    "bc": BehaviorCloningAgent,
    "bc_adaptive": MemoryMatchedBcAgent,
    "ewc": EwcAgent,
    "replay_bc": ReplayBcAgent,
    "in_context_llm": InContextLlmAgent
}
