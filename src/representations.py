"""Semantic action observations and transition-effect validation helpers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np

from .environment import StateTracker, parse_action_label

StateVector = Tuple[int, ...]
TransitionDelta = Tuple[int, ...]
ACTION_REPRESENTATION = "semantic_actions"

@dataclass(frozen=True)
class Observation:
    """One semantic action and its observed state transition."""
    state: StateVector
    action: str
    next_state: StateVector

    def __post_init__(self) -> None:
        parse_action_label(self.action)

def state_delta(before: Sequence[int], after: Sequence[int]) -> TransitionDelta:
    """Return an effect signature used only to validate state changes."""
    before_array = np.asarray(before, dtype=np.int8)
    after_array = np.asarray(after, dtype=np.int8)
    turned_on = ((before_array == 0) & (after_array == 1)).astype(np.int8)
    turned_off = ((before_array == 1) & (after_array == 0)).astype(np.int8)
    return tuple(np.concatenate([turned_on, turned_off]).astype(int).tolist())

def observe_actions(actions: Sequence[str]) -> List[Observation]:
    """Execute semantic actions and retain each observed state transition."""
    tracker = StateTracker()
    tracker.reset()
    out: List[Observation] = []
    for action in actions:
        before = tuple(tracker.get_state_vector().astype(int).tolist())
        tracker.apply_action(action)
        after = tuple(tracker.get_state_vector().astype(int).tolist())
        out.append(Observation(before, str(action), after))
    return out
