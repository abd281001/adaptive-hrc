"""Learner-facing transition representations for demonstrations.

This module is the leakage boundary between symbolic kitchen actions and the
learning stack.  Learners consume observed state transitions and anonymous
action tokens; simulator-side recipe and preference labels never enter here.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import List, Sequence, Tuple
import numpy as np

from .environment import StateTracker
from .models import State

# Transition-vector encoding
ActionVector = Tuple[int, ...]

@dataclass(frozen=True)
class ActionObservation:
    """One observed transition from a human demonstration."""
    state: State
    action_vector: ActionVector
    next_state: State

def transition_vector(before: Sequence[int], after: Sequence[int]) -> ActionVector:
    """Binary delta: on-bits followed by off-bits."""
    b = np.asarray(before, dtype=np.int8)
    a = np.asarray(after, dtype=np.int8)
    turned_on = ((b == 0) & (a == 1)).astype(np.int8)
    turned_off = ((b == 1) & (a == 0)).astype(np.int8)
    return tuple(np.concatenate([turned_on, turned_off]).astype(int).tolist())

def apply_transition_vector(state: Sequence[int], vector: ActionVector) -> State:
    """Apply a binary transition vector to a state without symbolic semantics.

    A malformed vector is treated as an identity transition.  This keeps the
    representation helper total for defensive callers; valid observed
    transitions produced by :func:`transition_vector` always have exactly
    twice the state dimension.
    """
    s = np.asarray(state, dtype=np.int8).copy()
    if len(vector) != 2 * len(s): return tuple(s.astype(int).tolist())
    half = len(s)
    on = np.asarray(vector[:half], dtype=np.int8)
    off = np.asarray(vector[half:], dtype=np.int8)
    s[off == 1] = 0
    s[on == 1] = 1
    return tuple(s.astype(int).tolist())

def observations_from_actions(actions: Sequence[str]) -> List[ActionObservation]:
    """Convert a string-driven demonstration into state/delta observations."""
    tracker = StateTracker()
    tracker.reset()
    out: List[ActionObservation] = []
    for action in actions:
        before = tuple(tracker.get_state_vector().astype(int).tolist())
        tracker.apply_action(action)
        after = tuple(tracker.get_state_vector().astype(int).tolist())
        vec = transition_vector(before, after)
        out.append(ActionObservation(before, vec, after))
    return out

def identity_transition_vector(before: Sequence[int], after: Sequence[int]) -> ActionVector:
    """Return the observed transition delta used for anonymous identity tokens."""
    return transition_vector(before, after)


def identity_token_from_observation(observation: ActionObservation) -> str:
    """Stable anonymous recipe-identity token for one observed transition."""
    vector = identity_transition_vector(observation.state, observation.next_state)
    payload = bytes(vector)
    return "identity_" + hashlib.sha256(payload).hexdigest()[:16]
