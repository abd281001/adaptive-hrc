"""Domain boundary shared by symbolic Adaptive-HRC and external HRC tasks.
The learning and memory code consumes this protocol instead of reconstructing domain state directly.  ``SymbolicDomainAdapter`` preserves the original semantic-action environment exactly and remains the default implementation.
"""
from __future__ import annotations
from typing import Any, Hashable, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable
import numpy as np


StateVector = Tuple[int, ...]


@runtime_checkable
class DomainAdapter(Protocol):
    """Minimal contract required by the existing learning stack.
    External environments may keep rich simulator states internally, but must expose a stable numeric key for learning and logged transition traces.
    """

    name: str
    action_representation: str

    def initial_state(self) -> StateVector:
        """Return the encoded state at the start of an episode."""

    def state_key(self, state: Any, *, actor_id: int = 0) -> StateVector:
        """Encode a raw or already encoded state as a stable numeric tuple."""

    def replay_transition(self, state: StateVector, action: str) -> StateVector:
        """Replay a recorded action without applying an action-selection mask."""

    def state_from_actions(self, actions: Sequence[str]) -> StateVector:
        """Reconstruct an encoded state from an action prefix."""

    def successor(self, state: StateVector, action: str) -> Optional[StateVector]:
        """Return the effectful legal successor, or ``None`` when unavailable."""

    def legal_actions(self, state: StateVector, actions: Sequence[str]) -> Tuple[str, ...]:
        """Apply the domain's preference-neutral action mask."""

    def canonical_action(self, action: str) -> str:
        """Return the stable action token used by learning and manifests."""

    def reward_features(self, state_vectors: Mapping[int, StateVector], known_mean: Optional[np.ndarray] = None, known_scale: Optional[np.ndarray] = None, normalizer: Any = None, update_normalizer: bool = False,
        *, feature_mode: str = "engineered", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct features used by reward learning."""

    def semantic_features(self, state_vectors: Mapping[int, StateVector], *, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct identity-masked features used only for fallback."""

    def build_features(self, state_vectors: Mapping[int, StateVector], known_mean: Optional[np.ndarray] = None, known_scale: Optional[np.ndarray] = None, normalizer: Any = None, update_normalizer: bool = False,
        *, feature_mode: str = "engineered", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct reward or similarity features for encoded states."""

    def action_role(self, action: str) -> str:
        """Map a grounded action to a domain-invariant workflow role."""

    def goal_signature(self, actions: Sequence[str]) -> Hashable:
        """Return a preference-neutral task signature for routing/auditing."""

class SymbolicDomainAdapter:
    """Compatibility adapter for the project's semantic kitchen simulator."""

    name = "symbolic_kitchen"
    action_representation = "semantic_actions"

    def initial_state(self) -> StateVector:
        from .environment import StateTracker
        tracker = StateTracker()
        return tuple(tracker.get_state_vector().astype(int).tolist())

    def state_key(self, state: Any, *, actor_id: int = 0) -> StateVector:
        del actor_id  # The symbolic simulator has one task-level state.
        if hasattr(state, "get_state_vector"): state = state.get_state_vector()
        array = np.asarray(state, dtype=int)
        if array.ndim != 1: raise ValueError("symbolic state must be a one-dimensional vector")
        return tuple(array.astype(int).tolist())

    def replay_transition(self, state: StateVector, action: str) -> StateVector:
        from .environment import StateTracker
        tracker = StateTracker()
        if len(state) != tracker.n_features: raise ValueError("symbolic state has the wrong feature dimension")
        tracker.current_state = np.asarray(state, dtype=int).copy()
        tracker.apply_action(str(action))
        return tuple(tracker.get_state_vector().astype(int).tolist())

    def state_from_actions(self, actions: Sequence[str]) -> StateVector:
        state = self.initial_state()
        for action in actions: state = self.replay_transition(state, str(action))
        return state

    def successor(self, state: StateVector, action: str) -> Optional[StateVector]:
        # Lazy import avoids a module cycle while preserving the tested legacy transition function as the single symbolic source of truth.
        from .models import _apply_action
        return _apply_action(tuple(state), str(action))

    def legal_actions( self, state: StateVector, actions: Sequence[str]) -> Tuple[str, ...]:
        return tuple(sorted({str(action) for action in actions if action != "stop" and self.successor(tuple(state), str(action)) is not None}))

    def canonical_action(self, action: str) -> str:
        # Symbolic actions are already emitted by the canonical environment formatter. Keeping this identity mapping preserves the frozen token space byte-for-byte.
        return str(action)

    def reward_features(self, state_vectors: Mapping[int, StateVector], known_mean: Optional[np.ndarray] = None, known_scale: Optional[np.ndarray] = None, normalizer: Any = None, update_normalizer: bool = False,
        *, feature_mode: str = "engineered", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_features(state_vectors, known_mean=known_mean, known_scale=known_scale, normalizer=normalizer, update_normalizer=update_normalizer, feature_mode=feature_mode, normalize=normalize)

    def semantic_features(  self, state_vectors: Mapping[int, StateVector], *, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_features(state_vectors, feature_mode="semantic", normalize=normalize)

    def build_features(     self, state_vectors: Mapping[int, StateVector], known_mean: Optional[np.ndarray] = None, known_scale: Optional[np.ndarray] = None, normalizer: Any = None,
        update_normalizer: bool = False, *, feature_mode: str = "engineered", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from .models import build_features
        return build_features(dict(state_vectors), known_mean=known_mean, known_scale=known_scale, normalizer=normalizer,update_normalizer=update_normalizer, feature_mode=feature_mode, normalize=normalize)

    def action_role(self, action: str) -> str:
        from .latent_strategy import action_role
        return action_role(str(action))

    def goal_signature(self, actions: Sequence[str]) -> Hashable:
        # The current symbolic matcher defines a recipe by its action set; the ordering remains preference evidence and is intentionally excluded.
        return tuple(sorted({str(action) for action in actions if action != "stop"}))


_DEFAULT_DOMAIN = SymbolicDomainAdapter()


def default_domain() -> DomainAdapter:
    """Return the stateless default adapter used by all existing experiments."""
    return _DEFAULT_DOMAIN
