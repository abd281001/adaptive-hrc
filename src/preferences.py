"""Goal-preserving workflow-preference transformations for HRC recipes."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .environment import (ACTION_LABEL_ARGUMENTS, Action, CONTAINERS, INGREDIENTS, parse_action_label, task_goal_signature, validate_ordering)

PREP_VALUES = ("serial", "prep_first")
EQUIPMENT_VALUES = ("early", "just_in_time")
SERVING_VALUES = ("early", "just_in_time")
CLEANUP_VALUES = ("after_serving", "when_free")
SHUTDOWN_VALUES = ("immediate", "delayed")
LOADING_VALUES = ("incremental", "just_in_time")
COOK_START_VALUES = ("immediate", "delayed")
SERVE_ORDER_VALUES = ("serve_first", "cleanup_first")
PREFERENCE_AXES: Tuple[str, ...] =              ("prep",              "equipment",                   "serving",                 "loading",                 "shutdown",                  "cook_start",                    "cleanup",                 "serve_order")
PREFERENCE_VALUES: Dict[str, Tuple[str, ...]] = {"prep": PREP_VALUES, "equipment": EQUIPMENT_VALUES, "serving": SERVING_VALUES, "loading": LOADING_VALUES, "shutdown": SHUTDOWN_VALUES, "cook_start": COOK_START_VALUES, "cleanup": CLEANUP_VALUES, "serve_order": SERVE_ORDER_VALUES}

@dataclass(frozen=True)
class Preference:
    prep: str = "serial"
    equipment: str = "early"
    serving: str = "just_in_time"
    loading: str = "incremental"
    shutdown: str = "immediate"
    cook_start: str = "immediate"
    cleanup: str = "after_serving"
    serve_order: str = "serve_first"

    def __post_init__(self) -> None:
        for axis, value in self.as_dict().items():
            if value not in PREFERENCE_VALUES[axis]:
                raise ValueError(f"invalid {axis} value: {value!r}")

    @property
    def label(self) -> str:
        return "_".join(f"{axis}-{value}" for axis, value in self.as_dict().items())

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)

@dataclass(frozen=True)
class PreferenceResult:
    actions: List[str]
    applied: List[str]
    failed: List[str]
    unchanged: List[str]
    values: Dict[str, str]

# Simulation-only preference labels never enter the learner.
PREFERENCES: Dict[str, Preference] = {
    "default": Preference(),
    # Isolated primary axes in schema order.
    "prep_first":                       Preference(prep="prep_first"),
    "equipment_jit":                    Preference(                                                                                     equipment="just_in_time"),
    "serving_early":                    Preference(                   serving="early"),
    "loading_jit":                      Preference(                                     loading="just_in_time"),
    "shutdown_late":                    Preference(                                                                                                                                 shutdown="delayed"),
    "cook_start_late":                  Preference(                                                                                                                                             cook_start="delayed"),
    "cleanup_when_free":                Preference(                                                             cleanup="when_free"),
    "cleanup_first":                    Preference(                                                                                                                 serve_order="cleanup_first"),
    # Composition probes outside the primary ladder.
    "equipment_jit_serving_early":      Preference(                   serving="early",                                                  equipment="just_in_time"),
    "prep_first_cleanup":               Preference(prep="prep_first",                                           cleanup="when_free"),
    "prep_first_serving_cleanup":       Preference(prep="prep_first", serving="early",                          cleanup="when_free"),
    "prep_loading_serving_cleanup":     Preference(prep="prep_first", serving="early",  loading="just_in_time", cleanup="when_free"),
    "equipment_jit_cleanup":            Preference(                                                             cleanup="when_free",    equipment="just_in_time"),
    "prep_loading_cleanup":             Preference(prep="prep_first",                   loading="just_in_time", cleanup="when_free"),
}
PREFERENCE_IDS: Tuple[str, ...] = tuple(PREFERENCES)
DEFAULT_PREFERENCE = PREFERENCES["default"].as_dict()


@lru_cache(maxsize=None)
def _action(action: str) -> Action:
    return parse_action_label(action)

def _verb(action: str) -> str:
    return _action(action).verb

def _args(action: str) -> List[str]:
    parsed = _action(action)
    return [parsed.args[key] for key in ACTION_LABEL_ARGUMENTS[parsed.verb] if key in parsed.args]

def _first_argument(action: str) -> str:
    args = _args(action)
    return args[0] if args else ""

def _argument(action: str, key: str) -> Optional[str]:
    return _action(action).get(key)

def _source(action: str) -> Optional[str]:
    return _argument(action, "from")

def _destination(action: str) -> Optional[str]:
    return _argument(action, "to")


def _stable_partition(actions, predicate):
    picked, remaining = [], []
    for action in actions:
        (picked if predicate(action) else remaining).append(action)
    return picked, remaining


def _try_validate(candidate: Sequence[str]) -> bool:
    return validate_ordering(list(candidate))


def _move_matching_block(actions: Sequence[str], predicate, *, earliest: bool) -> List[str]:
    """Move all matching actions as one stable block to the earliest/latest valid position."""
    block, rest = _stable_partition(actions, predicate)
    if not block:
        return list(actions)
    positions = range(0, len(rest) + 1) if earliest else range(len(rest), -1, -1)
    for position in positions:
        candidate = rest[:position] + block + rest[position:]
        if _try_validate(candidate):
            return candidate
    return list(actions)


def _move_actions(actions: Sequence[str], predicate, *, earliest: bool) -> List[str]:
    """Move each matching occurrence to a valid, goal-equivalent extreme."""
    base = list(actions)
    base_goal = task_goal_signature(base)
    tagged = list(enumerate(base))
    target_ids = [item_id for item_id, action in tagged if predicate(action)]
    ordered_ids = target_ids if earliest else list(reversed(target_ids))

    for target_id in ordered_ids:
        current_index = next(index for index, (item_id, _action) in enumerate(tagged) if item_id == target_id)
        item = tagged.pop(current_index)
        positions = range(0, len(tagged) + 1) if earliest else range(len(tagged), -1, -1)
        for position in positions:
            candidate = tagged[:position] + [item] + tagged[position:]
            candidate_actions = [action for _uid, action in candidate]
            if validate_ordering(candidate_actions, expected_goal=base_goal):
                tagged = candidate
                break
        else:
            # Retain the original position if no valid extreme exists.
            tagged.insert(current_index, item)

    result = [action for _uid, action in tagged]
    return result if validate_ordering(result, expected_goal=base_goal) else base


def _is_equipment_setup(action: str) -> bool:               return (_verb(action) in ("transfer", "move_container") and _first_argument(action) in CONTAINERS and _first_argument(action) != "plate" and _source(action) == "storage")

def _is_cleanup_transport(action: str, item: str) -> bool:  return (_verb(action) in ("transfer", "move_container") and _first_argument(action) == item and _destination(action) == "washing_station")

def _is_ingredient_retrieval(action: str) -> bool:          return _verb(action) == "transfer" and _source(action) == "storage" and _first_argument(action) in INGREDIENTS

def _is_productive_load(action: str) -> bool:
    args = _args(action)
    return _verb(action) == "load" and len(args) >= 2 and args[1] not in ("plate", "glass")

def _is_cook_block_action(action: str) -> bool:             return _verb(action) in ("turn_on", "turn_off", "cook", "cook_contents", "blend")

def _is_serving_action(action: str) -> bool:
    if _verb(action) == "serve":
        return True
    return _verb(action) == "move_container" and _first_argument(action) in ("plate", "glass") and _destination(action) == "serving_station"

def _cleanup_blocks(actions: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    """Split actions into productive actions and transport-plus-wash blocks."""
    consumed = set()
    blocks: List[List[str]] = []
    for action_index, action in enumerate(actions):
        if _verb(action) != "wash":
            continue
        item = _first_argument(action)
        # Find the latest unpaired transport to the washing station.
        transport_index: Optional[int] = None
        for search_index in range(action_index - 1, -1, -1):
            if search_index in consumed:
                continue
            if _is_cleanup_transport(actions[search_index], item):
                transport_index = search_index
                break
            # A different wash starts another cleanup block.
            if (_verb(actions[search_index]) == "wash" and _first_argument(actions[search_index]) != item): break
        start = (transport_index if transport_index is not None else action_index)
        block = list(actions[start: action_index + 1])
        consumed.update(range(start, action_index + 1))
        blocks.append(block)
    rest = [action for index, action in enumerate(actions) if index not in consumed]
    return rest, blocks


def _apply_prep(actions: Sequence[str], _value: str) -> List[str]:
    """Apply the non-default mise-en-place policy."""
    return _move_actions(actions, lambda action: (_is_ingredient_retrieval(action) or _verb(action) in ("cut", "grate") or _is_productive_load(action)), earliest=True)


def _apply_equipment(actions: Sequence[str], _value: str) -> List[str]:
    """Move equipment setup to its last valid position."""
    return _move_actions(actions, _is_equipment_setup, earliest=False)


def _apply_serving(actions: Sequence[str], _value: str) -> List[str]:
    """Frontload plate staging."""
    plate_stage, rest = _stable_partition(actions, lambda a: (_verb(a) == "transfer" and _first_argument(a) == "plate" and _argument(a, "to") == "plating_station"))
    if not plate_stage:                 return list(actions)
    candidate = list(plate_stage) + list(rest)
    if _try_validate(candidate):        return candidate
    raise ValueError("early serving setup cannot preserve the recipe goal")


def _apply_cleanup(actions: Sequence[str], _value: str) -> List[str]:
    """Place cleanup blocks at their earliest valid points."""
    rest, blocks = _cleanup_blocks(actions)
    if not blocks: return list(actions)
    candidate = list(rest)
    remaining = [list(block) for block in blocks]
    while remaining:
        best = None
        for block_index, block in enumerate(remaining):
            for position in range(0, len(candidate) + 1):
                trial = candidate[:position] + block + candidate[position:]
                if _try_validate(trial):
                    if best is None or position < best[0]:
                        best = (position, block_index, trial)
                    break
        if best is None:
            # Preserve original order when no earlier valid placement exists.
            for block in remaining: candidate.extend(block)
            break
        _position, block_index, candidate = best
        remaining.pop(block_index)
    return candidate if _try_validate(candidate) else list(actions)


def _apply_shutdown(actions: Sequence[str], _value: str) -> List[str]:
    """Delay appliance shutdown until the last goal-preserving opportunity."""
    return _move_actions(actions, lambda action: _verb(action) == "turn_off", earliest=False)


def _apply_loading(actions: Sequence[str], _value: str) -> List[str]:
    """Use raw incremental loading or defer each load to its last valid point."""
    return _move_actions(actions, _is_productive_load, earliest=False)


def _apply_cook_start(actions: Sequence[str], _value: str) -> List[str]:
    """`delayed` pushes stove/blender activation and cooking/blending actions later as one block."""
    return _move_matching_block(actions, _is_cook_block_action, earliest=False)


def _apply_serve_order(actions: Sequence[str], _value: str) -> List[str]:
    """`cleanup_first` delays the serving move/serve block so cleanup can occur first when physically valid."""
    return _move_matching_block(actions, _is_serving_action, earliest=False)

PREFERENCE_RULES = {
    "prep": _apply_prep,
    "equipment": _apply_equipment,
    "serving": _apply_serving,
    "loading": _apply_loading,
    "shutdown": _apply_shutdown,
    "cook_start": _apply_cook_start,
    "cleanup": _apply_cleanup,
    "serve_order": _apply_serve_order,
}
def apply_preference(actions: Sequence[str],    preference: Preference) -> PreferenceResult:
    """Apply one goal-preserving workflow preference."""
    if not _try_validate(actions):
        raise ValueError("input recipe is invalid")
    goal = task_goal_signature(actions)
    modified = list(actions)
    applied: List[str] = []
    failed: List[str] = []
    unchanged: List[str] = []
    values = preference.as_dict()
    for axis in PREFERENCE_AXES:
        value = values[axis]
        if value == DEFAULT_PREFERENCE[axis]:
            unchanged.append(axis)
            continue
        before = tuple(modified)
        candidate = PREFERENCE_RULES[axis](modified, value)
        if not validate_ordering(candidate, expected_goal=goal):
            failed.append(axis)
            continue
        modified = list(candidate)
        (applied if tuple(modified) != before else failed).append(axis)
    if not validate_ordering(modified, expected_goal=goal):     raise ValueError("preference changed the recipe goal")
    return PreferenceResult(actions=modified,   applied=applied,    failed=failed,      unchanged=unchanged,        values=dict(values))


def _get_preference(name: str) -> Preference:
    try:                        return PREFERENCES[name]
    except KeyError as exc:     raise KeyError(f"unknown preset {name!r}; known: {sorted(PREFERENCES)}") from exc


def apply_preset(actions: Sequence[str], preference_id: str) -> PreferenceResult:
    preference = _get_preference(preference_id)
    if preference_id == "default":      return PreferenceResult(actions=list(actions),      applied=[],     failed=[],   unchanged=list(PREFERENCE_AXES),       values=preference.as_dict())
    return apply_preference(actions, preference)


def apply_preset_actions(actions: Sequence[str], preference_id: str) -> List[str]:
    return list(apply_preset(actions, preference_id).actions)
