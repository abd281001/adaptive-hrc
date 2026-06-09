"""Learner-facing representations for demonstrations. This module is the leakage boundary between symbolic kitchen actions and the learning stack. It contains both transition-vector encoding and role-level workflow abstraction, so downstream models consume state deltas and roles rather than simulator-side preference labels."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .environment import (_FEAT,    CONTAINERS,    INGREDIENTS,    LOCATIONS,    StateTracker)
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
    """Apply a binary transition vector to a state without symbolic semantics. Length-mismatched vectors are a no-op (return the unchanged state). This is by design: synthetic task-signature builders pass short mock vectors against the full state and rely on the no-op contract; production callers always feed full-length transition vectors via observations_from_actions."""
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
        try:                tracker.apply_action(action)
        except ValueError:  pass        # Corruption/OOD stress tests deliberately inject impossible actions.  The observed state does not change; the binary action vector therefore becomes the zero transition.
        after = tuple(tracker.get_state_vector().astype(int).tolist())
        vec = transition_vector(before, after)
        out.append(ActionObservation(before, vec, after))
    return out


# Role abstraction
ROLE_RETRIEVE_INGREDIENT   = "retrieve_ingredient"
ROLE_RETRIEVE_CONTAINER    = "retrieve_container"
ROLE_PREPARE_INGREDIENT    = "prepare_ingredient"
ROLE_ADD_TO_CONTAINER      = "add_to_container"
ROLE_MOVE_CONTAINER        = "move_container"
ROLE_ACTIVATE_APPLIANCE    = "activate_appliance"
ROLE_DEACTIVATE_APPLIANCE  = "deactivate_appliance"
ROLE_COOK_OR_BLEND         = "cook_or_blend"
ROLE_STAGE_SERVING_VESSEL  = "stage_serving_vessel"
ROLE_SERVE                 = "serve"
ROLE_CLEAN_CONTAINER       = "clean_container"
ROLE_OTHER_VALID           = "other_valid_transition"
ROLE_UNKNOWN_OR_NOOP       = "unknown_or_noop"

ROLES: Tuple[str, ...] = (ROLE_RETRIEVE_INGREDIENT,     ROLE_RETRIEVE_CONTAINER,     ROLE_PREPARE_INGREDIENT,    ROLE_ADD_TO_CONTAINER,      ROLE_MOVE_CONTAINER,        ROLE_ACTIVATE_APPLIANCE,    ROLE_DEACTIVATE_APPLIANCE,
                            ROLE_COOK_OR_BLEND,         ROLE_STAGE_SERVING_VESSEL,   ROLE_SERVE,                 ROLE_CLEAN_CONTAINER,       ROLE_OTHER_VALID,           ROLE_UNKNOWN_OR_NOOP)


# Precomputed feature-index sets used by the role decoder
def _idx(name: str) -> int: return _FEAT[name]


def _build_index_sets():
    """Compile feature-index sets so role decoding is O(active-bits) per call."""
    appliance = {_idx("stove_on"): "stove",     _idx("sink_on"): "sink",    _idx("blender_on"): "blender"}
    plate_served = _idx("plate_served")
    washed = {}
    for item in CONTAINERS + INGREDIENTS:
        key = f"{item}_washed"
        if key in _FEAT:    washed[_idx(key)] = item
    cut = {}
    grated = {}
    cooked = {}
    seasoned = {}
    in_mixture = {}
    for ing in INGREDIENTS:
        for suffix, bucket in (("_cut", cut), ("_grated", grated), ("_cooked", cooked), ("_seasoned", seasoned), ("_in_mixture", in_mixture)):
            key = f"{ing}{suffix}"
            if key in _FEAT:    bucket[_idx(key)] = ing
    contains = {}
    for c in CONTAINERS:
        for ing in INGREDIENTS:
            key = f"{c}_contains_{ing}"
            if key in _FEAT:    contains[_idx(key)] = (c, ing)
    at_location: dict = {}
    for item in CONTAINERS + INGREDIENTS:
        for loc in LOCATIONS:
            key = f"{item}_at_{loc}"
            if key in _FEAT:    at_location[_idx(key)] = (item, loc)
    return {"appliance": appliance, "plate_served": plate_served, "washed": washed, "cut": cut, "grated": grated, "cooked": cooked, "seasoned": seasoned, "in_mixture": in_mixture, "contains": contains, "at_location": at_location}


_INDEX_SETS = _build_index_sets()
_CONTAINER_SET = set(CONTAINERS)

def _diff_indices(before: Sequence[int], after: Sequence[int]):
    """Return (turned_on, turned_off) as lists of feature indices."""
    on: List[int] = []
    off: List[int] = []
    # Both vectors are equal-length feature snapshots. We do not need action_vector for the differential; it is taken as input for API symmetry with action_encoding.transition_vector.
    n = min(len(before), len(after))
    for i in range(n):
        b = int(before[i])
        a = int(after[i])
        if a > b:   on.append(i)
        elif a < b: off.append(i)
    return on, off


def role_from_transition(state_before: Sequence[int], action_vector: Sequence[int], state_after: Sequence[int]) -> str:
    """Decode a single role tag from an observed state transition. Deterministic. Reads only state deltas and feature-index metadata; never consults preference labels, modifier identity, or any side channel."""
    on, off = _diff_indices(state_before, state_after)
    if not on and not off:              return ROLE_UNKNOWN_OR_NOOP

    sets = _INDEX_SETS

    # 1. Appliance toggle is the strongest, single-bit signal.
    for i in on:
        if i in sets["appliance"]:                                                                  return ROLE_ACTIVATE_APPLIANCE
    for i in off:
        if i in sets["appliance"]:                                                                  return ROLE_DEACTIVATE_APPLIANCE
    # 2. Serve flips the plate-served bit.
    if sets["plate_served"] in on:                                                                  return ROLE_SERVE
    # 3. Cleaning a container/item raises a *_washed flag.
    for i in on:
        if i in sets["washed"]:                                                                     return ROLE_CLEAN_CONTAINER
    # 4. Preparation produces _cut / _grated / _seasoned.
    for i in on:
        if i in sets["cut"] or i in sets["grated"] or i in sets["seasoned"]:                        return ROLE_PREPARE_INGREDIENT
    # 5. Cooking / blending raise _cooked or _in_mixture.
    for i in on:
        if i in sets["cooked"] or i in sets["in_mixture"]:                                          return ROLE_COOK_OR_BLEND
    # 6. Loading raises a *_contains_* flag (count delta > 0).
    contains_on = [sets["contains"][i] for i in on if i in sets["contains"]]
    contains_off = [sets["contains"][i] for i in off if i in sets["contains"]]
    if contains_on and not contains_off:    return ROLE_ADD_TO_CONTAINER
    # 7. Position-only deltas (no contains change). Distinguish: ** plate moved out of storage to plating/serving station -> stage  ** ingredient moved out of storage -> retrieve  ** container moved (with multiple paired flips for its contents) -> move_container
    at_on = [sets["at_location"][i] for i in on if i in sets["at_location"]]
    at_off = [sets["at_location"][i] for i in off if i in sets["at_location"]]
    if at_on and at_off:
        # Items leaving storage.
        ingredient_from_storage = any((item in INGREDIENTS) and (loc == "storage") for (item, loc) in at_off)
        container_from_storage = any((item in _CONTAINER_SET) and (item != "plate") and (loc == "storage") for (item, loc) in at_off)
        plate_to_serving_or_plating = any(item == "plate" and loc in ("plating_station", "serving_station") for (item, loc) in at_on)
        plate_from_storage = any(item == "plate" and loc == "storage" for (item, loc) in at_off)
        if plate_to_serving_or_plating and plate_from_storage:                                      return ROLE_STAGE_SERVING_VESSEL
        if ingredient_from_storage and not any(item in _CONTAINER_SET for (item, _) in at_off):     return ROLE_RETRIEVE_INGREDIENT
        if container_from_storage:                                                                  return ROLE_RETRIEVE_CONTAINER
        if any(item in _CONTAINER_SET for (item, _) in at_off):                                     return ROLE_MOVE_CONTAINER  # Otherwise it is a container move (which drags ingredients with it).
        # Single-item ingredient move not from storage, e.g. between stations for plating staging. Treat it as move_container as a coarse bucket.
        return ROLE_MOVE_CONTAINER
    # 8. Contains turned off (unload / combine / blend). The cook/blend branch handled the cook/blend case already; remaining is unload-style.
    if contains_off and not contains_on and not at_on:                                              return ROLE_OTHER_VALID

    return ROLE_OTHER_VALID


# Role-order feature extraction
def roles_from_actions(actions: Sequence[str]) -> List[str]:
    """Replay `actions` through a fresh StateTracker and emit one role per step. This is the canonical way to derive a role sequence from a recipe demonstration in tests and in the preference-prototype learner. It does NOT use any preference label. """
    tracker = StateTracker()
    out: List[str] = []
    state_before = tuple(tracker.get_state_vector().astype(int).tolist())
    for action in actions:
        tracker.apply_action(action)
        state_after = tuple(tracker.get_state_vector().astype(int).tolist())
        # The action_vector is the bit delta; for the differential decoder we do not need it explicitly, but we pass it through for API symmetry.
        action_vector = tuple(int(a) - int(b) for a, b in zip(state_after, state_before))
        out.append(role_from_transition(state_before, action_vector, state_after))
        state_before = state_after
    return out

def role_bigrams(roles: Sequence[str]) -> List[Tuple[str, str]]:
    return [(roles[i], roles[i + 1]) for i in range(len(roles) - 1)]

def role_trigrams(roles: Sequence[str]) -> List[Tuple[str, str, str]]:
    return [(roles[i], roles[i + 1], roles[i + 2]) for i in range(len(roles) - 2)]


# Task/preference signatures
@dataclass(frozen=True)
class TaskSignature:
    """Recipe identity evidence derived from demonstrations. The signature is deliberately sequence-light: it stores action-token support, role/effect counts, terminal states, and coarse ordering constraints.  It is intended to be rebuilt from the active replay set after pruning."""
    action_tokens: FrozenSet[str]
    role_multiset: Tuple[Tuple[str, int], ...]
    state_effect_multiset: Tuple[Tuple[ActionVector, int], ...]
    terminal_state_signature: Tuple[State, ...]
    partial_order_constraints: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class PreferenceSignature:
    """Transferable workflow-style evidence independent of a recipe id."""
    scalar_features: Tuple[Tuple[str, float], ...]
    role_bigrams: Tuple[Tuple[Tuple[str, str], int], ...]
    role_trigrams: Tuple[Tuple[Tuple[str, str, str], int], ...]


def _sorted_counts(counter: Counter) -> Tuple[Tuple, ...]:
    return tuple(sorted(counter.items(), key=lambda kv: (str(kv[0]), kv[1])))


def _role_order_scalar_features(roles: Sequence[str]) -> Dict[str, float]:
    """Small, interpretable workflow-style feature set. These features mirror the preference-prototype cues used by the posterior without importing posterior.py and creating a representation/model cycle."""
    n = max(1, len(roles))
    def first_pos(role: str) -> float:
        try:                return roles.index(role) / n
        except ValueError:  return 1.0

    def mean_pos(role: str) -> float:
        idxs = [i for i, r in enumerate(roles) if r == role]
        return (sum(idxs) / len(idxs) / n) if idxs else 1.0

    cleanup_positions = [i for i, r in enumerate(roles) if r == ROLE_CLEAN_CONTAINER]
    serve_first = first_pos(ROLE_SERVE)
    if cleanup_positions:   before_serve = sum(1 for i in cleanup_positions if i / n < serve_first) / max(1, len(cleanup_positions))
    else:                   before_serve = 0.0
    return {"first_retrieve": first_pos(ROLE_RETRIEVE_INGREDIENT),  "first_retrieve_container": first_pos(ROLE_RETRIEVE_CONTAINER),  "first_prepare": first_pos(ROLE_PREPARE_INGREDIENT),  "first_add_to_container": first_pos(ROLE_ADD_TO_CONTAINER),  "first_move_container": first_pos(ROLE_MOVE_CONTAINER),  "first_activate_appliance": first_pos(ROLE_ACTIVATE_APPLIANCE),  "first_clean": first_pos(ROLE_CLEAN_CONTAINER),  "first_serve": first_pos(ROLE_SERVE),   "mean_cleanup": mean_pos(ROLE_CLEAN_CONTAINER),     "cleanup_before_serve_fraction": before_serve,  "prep_lead_time": first_pos(ROLE_ADD_TO_CONTAINER) - first_pos(ROLE_PREPARE_INGREDIENT),"retrieval_lead_time": first_pos(ROLE_PREPARE_INGREDIENT) - first_pos(ROLE_RETRIEVE_INGREDIENT),  "appliance_lead_time": first_pos(ROLE_COOK_OR_BLEND) - first_pos(ROLE_ACTIVATE_APPLIANCE), "container_setup_lead_time": first_pos(ROLE_ADD_TO_CONTAINER) - first_pos(ROLE_RETRIEVE_CONTAINER), "container_move_lead_time": first_pos(ROLE_ADD_TO_CONTAINER) - first_pos(ROLE_MOVE_CONTAINER)    }


def task_signature_from_tokens(tokens: Sequence[str], token_to_action_vector: Mapping[str, ActionVector], token_to_role: Optional[Mapping[str, str]] = None) -> TaskSignature:
    """Build a recipe/task signature from tokenized active demonstrations."""
    role_counter: Counter = Counter()
    effect_counter: Counter = Counter()
    partial_order = set()
    state = tuple(StateTracker().get_state_vector().astype(int).tolist())
    terminal_states: List[State] = []
    token_list = [str(t) for t in tokens]
    roles = token_to_role or {}
    for i, token in enumerate(token_list):
        role_counter[roles.get(token, ROLE_UNKNOWN_OR_NOOP)] += 1
        vector = token_to_action_vector.get(token)
        if vector is not None:
            effect_counter[vector] += 1
            state = apply_transition_vector(state, vector)
        # Coarse precedence constraints are enough for compatibility scoring and avoid storing every full ordering as a hidden replay buffer.
        for prev in token_list[max(0, i - 4):i]:
            if prev != token: partial_order.add((prev, token))
    terminal_states.append(state)
    return TaskSignature(action_tokens=frozenset(token_list),   role_multiset=_sorted_counts(role_counter),     state_effect_multiset=_sorted_counts(effect_counter),       terminal_state_signature=tuple(sorted(set(terminal_states))),   partial_order_constraints=tuple(sorted(partial_order)))


def preference_signature_from_roles(roles: Sequence[str]) -> PreferenceSignature:
    """Build a transferable preference signature from role order only."""
    features = _role_order_scalar_features(list(roles))
    return PreferenceSignature(scalar_features=tuple(sorted((k, float(v)) for k, v in features.items())),       role_bigrams=_sorted_counts(Counter(role_bigrams(roles))),      role_trigrams=_sorted_counts(Counter(role_trigrams(roles))))
