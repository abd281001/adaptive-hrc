"""Main workflow-preference generator for adaptive HRC experiments. This module is the canonical preference generator. It uses orthogonal workflow axes:
    ingredient_flow:  serial | prep_first               equipment_setup:  frontloaded | just_in_time                serving_setup:    frontloaded | just_in_time                cleanup_timing:   after_service | as_soon_as_free
Key invariants:
    * No action is inserted, deleted, or rewritten — every transform is a reordering of the base recipe's existing actions.
    * Every emitted candidate is validated against the kitchen environment; invalid/no-effect axes are reportable so experiments can filter or stratify no-op preferences.
    * Workflow labels are simulation-only metadata. They never enter the learner; ``_make_pair`` only stores them on the ``RecipePrefPair``so evaluation code can stratify episodes.
Public surface:     WorkflowPreference          - one (axis-value, axis-value, ...) tuple.      
                    PRESET_PREFERENCES          - named presets used by experimental conditions.     
                    WorkflowPreferenceModifier  - reorders a base recipe by workflow preference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .environment import CONTAINERS, INGREDIENTS, validate_ordering

# Axes & presets
INGREDIENT_FLOW_VALUES   = ("serial", "prep_first")
EQUIPMENT_SETUP_VALUES   = ("frontloaded", "just_in_time")
SERVING_SETUP_VALUES     = ("frontloaded", "just_in_time")
CLEANUP_TIMING_VALUES    = ("after_service", "as_soon_as_free")
AXES: Tuple[str, ...] = (                   "ingredient_flow",                              "equipment_setup",                              "serving_setup",                                "cleanup_timing")
AXIS_VALUES: Dict[str, Tuple[str, ...]] = {"ingredient_flow":  INGREDIENT_FLOW_VALUES,      "equipment_setup":  EQUIPMENT_SETUP_VALUES,    "serving_setup":    SERVING_SETUP_VALUES,        "cleanup_timing":   CLEANUP_TIMING_VALUES,}

@dataclass(frozen=True)
class WorkflowPreference:
    ingredient_flow: str = "serial"
    equipment_setup: str = "just_in_time"
    serving_setup: str = "just_in_time"
    cleanup_timing: str = "after_service"
    def __post_init__(self):
        for axis, value in ((               "ingredient_flow", self.ingredient_flow),      ("equipment_setup", self.equipment_setup),      ("serving_setup",   self.serving_setup),        ("cleanup_timing",  self.cleanup_timing)):
            if value not in AXIS_VALUES[axis]: raise ValueError(f"invalid {axis} value: {value!r}")
    @property
    def label(self) -> str: return (f"if-{self.ingredient_flow}" f"_eq-{self.equipment_setup}" f"_sv-{self.serving_setup}" f"_cu-{self.cleanup_timing}")
    def as_dict(self) -> Dict[str, str]:
        return {                            "ingredient_flow": self.ingredient_flow,         "equipment_setup": self.equipment_setup,       "serving_setup":   self.serving_setup,          "cleanup_timing":  self.cleanup_timing}

@dataclass(frozen=True)
class ModificationResult:
    actions: List[str]
    applied_axes: List[str]
    failed_axes: List[str]
    unchanged_axes: List[str]
    axis_values: Dict[str, str]

# Preset library used by the experimental conditions. Names are short tags that label one (recipe, preference) row in test schedules; the simulator uses them only to stratify episodes — they never enter the learner.
PRESET_PREFERENCES: Dict[str, WorkflowPreference] = {
    # Identity preset = the recipe's natural ordering with no axis transforms.
    "identity":        WorkflowPreference(  ingredient_flow="serial",                       equipment_setup="just_in_time",                 serving_setup="just_in_time",                   cleanup_timing="after_service"),
    # P1: prep-first batching, just-in-time equipment/serving, defer cleanup.
    "p1_prep_first":   WorkflowPreference(  ingredient_flow="prep_first",                   equipment_setup="just_in_time",                 serving_setup="just_in_time",                   cleanup_timing="after_service"),
    # P2: frontload equipment + serving (e.g. mise-en-place style).
    "p2_frontload":    WorkflowPreference(  ingredient_flow="serial",                       equipment_setup="frontloaded",                  serving_setup="frontloaded",                    cleanup_timing="after_service"),
    # P3: serial ingredients, just-in-time equipment/serving, eager cleanup.
    "p3_clean_eager":  WorkflowPreference(  ingredient_flow="serial",                       equipment_setup="just_in_time",                 serving_setup="just_in_time",                   cleanup_timing="as_soon_as_free"),
    # P4: combine prep_first + clean_eager under just-in-time setup.
    "p4_prep_clean":   WorkflowPreference(  ingredient_flow="prep_first",                   equipment_setup="just_in_time",                 serving_setup="just_in_time",                   cleanup_timing="as_soon_as_free"),
    # P5: prep-first mise-en-place with eager cleanup.
    "p5_prep_stage_clean": WorkflowPreference(ingredient_flow="prep_first",                 equipment_setup="just_in_time",                 serving_setup="frontloaded",                    cleanup_timing="as_soon_as_free"),
    # P6: all non-default workflow axes for maximal restructuring.
    "p6_full_restructure": WorkflowPreference(ingredient_flow="prep_first",                 equipment_setup="frontloaded",                  serving_setup="frontloaded",                    cleanup_timing="as_soon_as_free"),}
PREFERENCE_NAMES: Tuple[str, ...] = tuple(PRESET_PREFERENCES.keys())


# Action parsing helpers (lightweight; format is "verb (arg1, arg2, ...)")
def _verb(action: str) -> str:
    return action.split("(", 1)[0].strip()

def _args(action: str) -> List[str]:
    if "(" not in action: return []
    body = action[action.find("(") + 1 : action.rfind(")")]
    return [p.strip() for p in body.split(",")]

def _arg0(action: str) -> str:
    args = _args(action)
    return args[0] if args else ""

def _arg_kw(action: str, key: str) -> Optional[str]:
    for a in _args(action):
        if "=" in a:
            k, v = a.split("=", 1)
            if k.strip() == key: return v.strip()
    return None

def _source(action: str) -> Optional[str]:
    return _arg_kw(action, "from")

def _destination(action: str) -> Optional[str]:
    return _arg_kw(action, "to")


# Reordering primitives
def _stable_partition(actions, predicate):
    picked, remaining = [], []
    for a in actions: (picked if predicate(a) else remaining).append(a)
    return picked, remaining


def _try_validate(candidate: Sequence[str]) -> bool:
    try:                return validate_ordering(list(candidate))
    except Exception:   return False


def _move_one_action(actions: Sequence[str], action: str, *, earliest: bool, lower_bound: int = 0, upper_bound: Optional[int] = None) -> Tuple[List[str], int]:
    """Move one existing action to an earliest/latest valid index. Optional bounds let callers constrain a family of moved actions when that is semantically useful. We validate the full sequence at every candidate position, so dependencies not encoded in these lightweight string predicates still come from the environment model"""
    current = list(actions)
    try:                idx = current.index(action)
    except ValueError:  return current, lower_bound

    rest = current[:idx] + current[idx + 1 :]
    hi = len(rest) if upper_bound is None else min(upper_bound, len(rest))
    lo = max(0, min(lower_bound, len(rest)))
    indices = range(lo, hi + 1) if earliest else range(hi, lo - 1, -1)
    for k in indices:
        candidate = rest[:k] + [action] + rest[k:]
        if _try_validate(candidate):    return candidate, k
    return current, idx


def _is_equipment_setup(action: str) -> bool:               return (_verb(action) in ("transfer", "move_container") and _arg0(action) in CONTAINERS and _arg0(action) != "plate" and _source(action) == "storage")

def _is_cleanup_transport(action: str, item: str) -> bool:  return (_verb(action) in ("transfer", "move_container") and _arg0(action) == item and _destination(action) == "washing_station")

def _cleanup_blocks(actions: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    """Split actions into non-cleanup actions and cleanup transport+wash blocks. The old transform moved only ``wash`` actions, which made eager cleanup a no-op because the item was not at the washing station yet. Treating the immediately preceding transport to washing_station as part of the same block keeps cleanup physically meaningful while preserving the exact action set."""
    consumed = set()
    blocks: List[List[str]] = []
    for i, action in enumerate(actions):
        if _verb(action) != "wash":
            continue
        item = _arg0(action)
        # Search backward for the most recent unconsumed transport of this item to washing_station. It may not be immediately adjacent.
        transport_idx: Optional[int] = None
        for j in range(i - 1, -1, -1):
            if j in consumed:
                continue
            if _is_cleanup_transport(actions[j], item):
                transport_idx = j
                break
            # Stop the backward search at another wash action (different item): that wash and its paired transport belong to a separate block.
            if _verb(actions[j]) == "wash" and _arg0(actions[j]) != item:
                break
        start = transport_idx if transport_idx is not None else i
        block = list(actions[start: i + 1])
        consumed.update(range(start, i + 1))
        blocks.append(block)
    rest = [a for idx, a in enumerate(actions) if idx not in consumed]
    return rest, blocks


# Axis transforms
def _apply_ingredient_flow(actions: Sequence[str], value: str) -> List[str]:
    """`serial` is per-ingredient pipeline; ``prep_first`` batches retrievals, then preps, then loads. For a serial recipe (the natural ordering of all RecipeGenerator outputs) the prep-first reorder pulls all retrieve→storage transfers to the front, then all cut/grate actions, then everything else. We validate; if invalid we fall back to the input unchanged."""
    if value == "serial":       return list(actions)
    if value != "prep_first":   return list(actions)

    # Phase A: retrievals from storage of *ingredients* go first.
    retrievals, rest =  _stable_partition(actions, lambda a: (_verb(a) == "transfer" and _arg_kw(a, "from") == "storage" and _arg0(a) in INGREDIENTS))
    # Phase B: prep actions (cut/grate) collected from `rest`.
    preps, rest2 =      _stable_partition(rest, lambda a: _verb(a) in ("cut", "grate"))
    candidate = retrievals + preps + rest2
    if _try_validate(candidate): return candidate
    # Phase A only.
    candidate = retrievals + rest
    if _try_validate(candidate): return candidate
    return list(actions)


def _apply_equipment_setup(actions: Sequence[str], value: str) -> List[str]:
    """Move non-serving container setup to early or late valid positions. `frontloaded` pulls setup actions as early as the environment permits. `just_in_time` pushes them as late as possible while preserving the full ordering's validity. Some storage-loaded containers cannot move
      before the storage load; validation keeps those cases at the first physically possible point."""
    if value not in ("frontloaded", "just_in_time"): return list(actions)

    setup_actions = [a for a in actions if _is_equipment_setup(a)]
    if not setup_actions: return list(actions)

    candidate = list(actions)
    if value == "frontloaded":
        lower = 0
        for setup in setup_actions:
            candidate, pos = _move_one_action(candidate, setup, earliest=True, lower_bound=lower)
            lower = pos + 1
    else:
        for setup in setup_actions:
            candidate, _pos = _move_one_action(candidate, setup, earliest=False)

    if _try_validate(candidate): return candidate
    return list(actions)


def _apply_serving_setup(actions: Sequence[str], value: str) -> List[str]:
    """`frontloaded` stages the plate at plating_station before stove ignites; `just_in_time` defers it until just before serve."""
    if value == "just_in_time":
        # Defer: try moving plate-staging block to just before the first serve.
        plate_stage, rest = _stable_partition(actions, lambda a: (_verb(a) == "transfer" and _arg0(a) == "plate" and _arg_kw(a, "to") == "plating_station"))
        if not plate_stage:             return list(actions)
        # Insert before first serve action; if no serve, leave at end.
        idx = next((i for i, a in enumerate(rest) if _verb(a) == "serve"), len(rest),)
        candidate = list(rest[:idx]) + list(plate_stage) + list(rest[idx:])
        if _try_validate(candidate):    return candidate
        return list(actions)
    if value != "frontloaded":          return list(actions)

    plate_stage, rest = _stable_partition(actions, lambda a: (_verb(a) == "transfer" and _arg0(a) == "plate" and _arg_kw(a, "to") == "plating_station"))
    if not plate_stage:                 return list(actions)
    candidate = list(plate_stage) + list(rest)
    if _try_validate(candidate):        return candidate
    return list(actions)


def _apply_cleanup_timing(actions: Sequence[str], value: str) -> List[str]:
    """Move cleanup transport+wash blocks. `after_service` places all cleanup blocks after productive work. `as_soon_as_free` repeatedly inserts each remaining block at its earliest currently valid point, which lets a freed prep bowl be cleaned before a still-needed pot or serving vessel."""
    rest, blocks = _cleanup_blocks(actions)
    if not blocks: return list(actions)

    if value == "after_service":
        candidate = list(rest)
        for block in blocks:            candidate.extend(block)
        if _try_validate(candidate):    return candidate
        return list(actions)

    if value == "as_soon_as_free":
        candidate = list(rest)
        remaining = [list(block) for block in blocks]
        while remaining:
            best = None
            for block_idx, block in enumerate(remaining):
                for k in range(0, len(candidate) + 1):
                    trial = candidate[:k] + block + candidate[k:]
                    if _try_validate(trial):
                        if best is None or k < best[0]: best = (k, block_idx, trial)
                        break
            if best is None:
                # Preserve safety: append the remaining blocks in their original order and let the final validator decide.
                for block in remaining: candidate.extend(block)
                break
            _k, block_idx, candidate = best
            remaining.pop(block_idx)
        if _try_validate(candidate):    return candidate
        return list(actions)

    return list(actions)

# Public modifier
class WorkflowPreferenceModifier:
    """Apply a workflow preference to a base recipe action list."""
    AXIS_ORDER: Tuple[str, ...] = ("ingredient_flow",                           "equipment_setup",                              "serving_setup",                            "cleanup_timing")
    AXIS_FN = {                     "ingredient_flow": _apply_ingredient_flow,  "equipment_setup": _apply_equipment_setup,      "serving_setup":   _apply_serving_setup,    "cleanup_timing":  _apply_cleanup_timing}
    DEFAULT_VALUES = WorkflowPreference().as_dict()

    def __init__(self) -> None:
        self.last_result: Optional[ModificationResult] = None

    def modify_recipe_with_report(self, actions: Sequence[str], preference: WorkflowPreference) -> ModificationResult:
        if not _try_validate(actions):  raise ValueError("modify_recipe: input action list is already invalid")
        modified = list(actions)
        applied: List[str] = []
        failed: List[str] = []
        unchanged: List[str] = []
        prefs_dict = preference.as_dict()
        for axis in self.AXIS_ORDER:
            value = prefs_dict[axis]
            before = list(modified)
            candidate = self.AXIS_FN[axis](modified, value)
            if not _try_validate(candidate):
                failed.append(axis)
                continue
            modified = list(candidate)
            if tuple(modified) != tuple(before): applied.append(axis)
            elif value != self.DEFAULT_VALUES[axis]: failed.append(axis)
            else: unchanged.append(axis)
        if not _try_validate(modified):     raise ValueError("modify_recipe: composite preference reordering is invalid")
        result = ModificationResult(actions=modified, applied_axes=applied, failed_axes=failed, unchanged_axes=unchanged, axis_values=dict(prefs_dict))
        self.last_result = result
        return result

    def modify_recipe(self,     actions: Sequence[str],     preference: WorkflowPreference) -> List[str]:
        return list(self.modify_recipe_with_report(actions, preference).actions)

# Convenience: construct the action list for a named preset.
def materialize(actions: Sequence[str], preset_name: str) -> List[str]:
    if preset_name not in PRESET_PREFERENCES:       raise KeyError(f"unknown preset {preset_name!r}; known: {sorted(PRESET_PREFERENCES)}")
    pref = PRESET_PREFERENCES[preset_name]
    if preset_name == "identity":                   return list(actions)
    return WorkflowPreferenceModifier().modify_recipe(actions, pref)

def materialize_with_report(actions: Sequence[str], preset_name: str) -> ModificationResult:
    if preset_name not in PRESET_PREFERENCES:       raise KeyError(f"unknown preset {preset_name!r}; known: {sorted(PRESET_PREFERENCES)}")
    pref = PRESET_PREFERENCES[preset_name]
    return WorkflowPreferenceModifier().modify_recipe_with_report(actions, pref)
