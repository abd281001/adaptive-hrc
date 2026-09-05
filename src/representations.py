"""Semantic action observations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .environment import StateTracker, parse_action_label

StateVector = Tuple[int, ...]
ACTION_REPRESENTATION = "semantic_actions"

@dataclass(frozen=True)
class Observation:
    """One semantic action and its observed state transition."""
    state: StateVector
    action: str
    next_state: StateVector

    def __post_init__(self) -> None:
        parse_action_label(self.action)

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
