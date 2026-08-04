"""Main workflow-preference generator for adaptive HRC experiments. This module is the canonical preference generator. It uses orthogonal workflow axes:
    ingredient_flow:          serial | mise_en_place
    equipment_setup:          pre_staged | just_in_time
    serving_setup:            frontloaded | just_in_time
    container_loading_style:  incremental | just_in_time
    appliance_shutdown_timing: immediate | deferred
    cook_start_timing:        as_soon_as_ready | deferred
    cleanup_timing:           after_service | as_soon_as_free
    serving_priority:         serve_when_ready | cleanup_before_serve
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

from .environment import CONTAINERS, INGREDIENTS, task_goal_signature, validate_ordering

# Axes & presets
INGREDIENT_FLOW_VALUES          = ("serial", "mise_en_place")
EQUIPMENT_SETUP_VALUES          = ("pre_staged", "just_in_time")
SERVING_SETUP_VALUES            = ("frontloaded", "just_in_time")
CLEANUP_TIMING_VALUES           = ("after_service", "as_soon_as_free")
APPLIANCE_SHUTDOWN_TIMING_VALUES = ("immediate", "deferred")
CONTAINER_LOADING_STYLE_VALUES  = ("incremental", "just_in_time")
COOK_START_TIMING_VALUES        = ("as_soon_as_ready", "deferred")
SERVING_PRIORITY_VALUES         = ("serve_when_ready", "cleanup_before_serve")
AXES: Tuple[str, ...] = (
    "ingredient_flow",
    "equipment_setup",
    "serving_setup",
    "container_loading_style",
    "appliance_shutdown_timing",
    "cook_start_timing",
    "cleanup_timing",
    "serving_priority",
)
AXIS_VALUES: Dict[str, Tuple[str, ...]] = {
    "ingredient_flow":            INGREDIENT_FLOW_VALUES,
    "equipment_setup":            EQUIPMENT_SETUP_VALUES,
    "serving_setup":              SERVING_SETUP_VALUES,
    "container_loading_style":    CONTAINER_LOADING_STYLE_VALUES,
    "appliance_shutdown_timing":  APPLIANCE_SHUTDOWN_TIMING_VALUES,
    "cook_start_timing":          COOK_START_TIMING_VALUES,
    "cleanup_timing":             CLEANUP_TIMING_VALUES,
    "serving_priority":           SERVING_PRIORITY_VALUES,
}

@dataclass(frozen=True)
class WorkflowPreference:
    ingredient_flow: str = "serial"
    equipment_setup: str = "pre_staged"
    serving_setup: str = "just_in_time"
    container_loading_style: str = "incremental"
    appliance_shutdown_timing: str = "immediate"
    cook_start_timing: str = "as_soon_as_ready"
    cleanup_timing: str = "after_service"
    serving_priority: str = "serve_when_ready"
    def __post_init__(self):
        for axis, value in self.as_dict().items():
            if value not in AXIS_VALUES[axis]: raise ValueError(f"invalid {axis} value: {value!r}")
    @property
    def label(self) -> str:
        return (
            f"if-{self.ingredient_flow}"
            f"_eq-{self.equipment_setup}"
            f"_sv-{self.serving_setup}"
            f"_cl-{self.container_loading_style}"
            f"_sh-{self.appliance_shutdown_timing}"
            f"_co-{self.cook_start_timing}"
            f"_cu-{self.cleanup_timing}"
            f"_sp-{self.serving_priority}"
        )
    def as_dict(self) -> Dict[str, str]:
        return {
            "ingredient_flow": self.ingredient_flow,
            "equipment_setup": self.equipment_setup,
            "serving_setup": self.serving_setup,
            "container_loading_style": self.container_loading_style,
            "appliance_shutdown_timing": self.appliance_shutdown_timing,
            "cook_start_timing": self.cook_start_timing,
            "cleanup_timing": self.cleanup_timing,
            "serving_priority": self.serving_priority,
        }

@dataclass(frozen=True)
class ModificationResult:
    actions: List[str]
    applied_axes: List[str]
    failed_axes: List[str]
    unchanged_axes: List[str]
    axis_values: Dict[str, str]

# Preset library used by the experimental conditions. Names are short tags that label one (recipe, preference) row in test schedules; the simulator uses them only to stratify episodes — they never enter the learner.
PRESET_PREFERENCES: Dict[str, WorkflowPreference] = {
    "identity": WorkflowPreference(),
    # Isolated, goal-preserving primary axes in WorkflowPreference axis order.
    "p1_mise_en_place": WorkflowPreference(ingredient_flow="mise_en_place"),
    "p2_equipment_just_in_time": WorkflowPreference(equipment_setup="just_in_time"),
    "p3_frontload_serving_setup": WorkflowPreference(serving_setup="frontloaded"),
    "p4_load_just_in_time": WorkflowPreference(container_loading_style="just_in_time"),
    "p5_shutdown_late": WorkflowPreference(appliance_shutdown_timing="deferred"),
    "p6_deferred_cook_start": WorkflowPreference(cook_start_timing="deferred"),
    "p7_clean_eager": WorkflowPreference(cleanup_timing="as_soon_as_free"),
    "p8_cleanup_before_serve": WorkflowPreference(serving_priority="cleanup_before_serve"),
    # Explicit composition probes; they are not part of the primary axis ladder.
    "p9_equipment_jit_frontload_serving": WorkflowPreference(
        equipment_setup="just_in_time", serving_setup="frontloaded",
    ),
    "p10_mise_en_place_clean": WorkflowPreference(
        ingredient_flow="mise_en_place", cleanup_timing="as_soon_as_free",
    ),
    "p11_mise_en_place_serving_clean": WorkflowPreference(
        ingredient_flow="mise_en_place", serving_setup="frontloaded",
        cleanup_timing="as_soon_as_free",
    ),
    "p12_multi_stage_reorganization": WorkflowPreference(
        ingredient_flow="mise_en_place", serving_setup="frontloaded",
        container_loading_style="just_in_time", cleanup_timing="as_soon_as_free",
    ),
    "p13_equipment_jit_clean": WorkflowPreference(
        equipment_setup="just_in_time", cleanup_timing="as_soon_as_free",
    ),
    "p14_mise_load_clean": WorkflowPreference(
        ingredient_flow="mise_en_place", container_loading_style="just_in_time",
        cleanup_timing="as_soon_as_free",
    ),
}
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


def _move_matching_block(actions: Sequence[str], predicate, *, earliest: bool) -> List[str]:
    """Move all matching actions as one stable block to the earliest/latest valid position."""
    indexed = [(i, a) for i, a in enumerate(actions) if predicate(a)]
    if not indexed:
        return list(actions)
    picked_indices = {i for i, _a in indexed}
    block = [a for _i, a in indexed]
    rest = [a for i, a in enumerate(actions) if i not in picked_indices]
    positions = range(0, len(rest) + 1) if earliest else range(len(rest), -1, -1)
    for k in positions:
        candidate = rest[:k] + block + rest[k:]
        if _try_validate(candidate):
            return candidate
    return list(actions)


def _move_matching_actions_goal_preserving(
    actions: Sequence[str],
    predicate,
    *,
    earliest: bool,
) -> List[str]:
    """Move each matching action to a valid extreme without changing the goal.

    Block moves are only sound when all selected actions share one contiguous
    dependency window.  Recipe loads and equipment actions do not: a single
    recipe can contain a storage load, an intermediate transfer, and a final
    mixture load.  This scheduler moves one *identified occurrence* at a time
    and accepts an insertion only when the complete candidate remains both
    executable and goal-equivalent to the input recipe.
    """
    base = list(actions)
    base_goal = task_goal_signature(base)
    tagged = list(enumerate(base))
    target_ids = [uid for uid, action in tagged if predicate(action)]
    ordered_ids = target_ids if earliest else list(reversed(target_ids))

    for target_id in ordered_ids:
        current_idx = next(idx for idx, (uid, _action) in enumerate(tagged) if uid == target_id)
        item = tagged.pop(current_idx)
        positions = range(0, len(tagged) + 1) if earliest else range(len(tagged), -1, -1)
        for position in positions:
            candidate = tagged[:position] + [item] + tagged[position:]
            candidate_actions = [action for _uid, action in candidate]
            if validate_ordering(candidate_actions, expected_goal=base_goal):
                tagged = candidate
                break
        else:
            # The original placement is valid by construction; retain it when
            # no more extreme goal-preserving placement exists.
            tagged.insert(current_idx, item)

    result = [action for _uid, action in tagged]
    return result if validate_ordering(result, expected_goal=base_goal) else base


def _is_equipment_setup(action: str) -> bool:               return (_verb(action) in ("transfer", "move_container") and _arg0(action) in CONTAINERS and _arg0(action) != "plate" and _source(action) == "storage")

def _is_cleanup_transport(action: str, item: str) -> bool:  return (_verb(action) in ("transfer", "move_container") and _arg0(action) == item and _destination(action) == "washing_station")

def _is_ingredient_retrieval(action: str) -> bool:          return _verb(action) == "transfer" and _source(action) == "storage" and _arg0(action) in INGREDIENTS

def _is_productive_load(action: str) -> bool:
    args = _args(action)
    return _verb(action) == "load" and len(args) >= 2 and args[1] not in ("plate", "glass")

def _is_cook_block_action(action: str) -> bool:             return _verb(action) in ("turn_on", "turn_off", "cook", "cook_contents", "blend")

def _is_serving_action(action: str) -> bool:
    if _verb(action) == "serve":
        return True
    return _verb(action) == "move_container" and _arg0(action) in ("plate", "glass") and _destination(action) == "serving_station"

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
    """Use raw serial order or a full, goal-preserving mise-en-place policy."""
    if value == "serial":       return list(actions)
    if value != "mise_en_place": return list(actions)
    return _move_matching_actions_goal_preserving(
        actions,
        lambda action: (
            _is_ingredient_retrieval(action)
            or _verb(action) in ("cut", "grate")
            or _is_productive_load(action)
        ),
        earliest=True,
    )


def _apply_equipment_setup(actions: Sequence[str], value: str) -> List[str]:
    """Contrast raw pre-staging with last-valid, just-in-time equipment setup."""
    if value == "pre_staged": return list(actions)
    if value != "just_in_time": return list(actions)
    return _move_matching_actions_goal_preserving(actions, _is_equipment_setup, earliest=False)


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


def _apply_appliance_shutdown_timing(actions: Sequence[str], value: str) -> List[str]:
    """Delay appliance shutdown until the last goal-preserving opportunity."""
    if value == "immediate":
        return list(actions)
    if value == "deferred":
        return _move_matching_actions_goal_preserving(actions, lambda action: _verb(action) == "turn_off", earliest=False)
    return list(actions)


def _apply_container_loading_style(actions: Sequence[str], value: str) -> List[str]:
    """Use raw incremental loading or defer each load to its last valid point."""
    if value == "incremental":
        return list(actions)
    if value == "just_in_time":
        return _move_matching_actions_goal_preserving(actions, _is_productive_load, earliest=False)
    return list(actions)


def _apply_cook_start_timing(actions: Sequence[str], value: str) -> List[str]:
    """`deferred` pushes stove/blender activation and cooking/blending actions later as one block."""
    if value == "as_soon_as_ready":
        return list(actions)
    if value == "deferred":
        return _move_matching_block(actions, _is_cook_block_action, earliest=False)
    return list(actions)


def _apply_serving_priority(actions: Sequence[str], value: str) -> List[str]:
    """`cleanup_before_serve` delays the serving move/serve block so cleanup can occur first when physically valid."""
    if value == "serve_when_ready":
        return list(actions)
    if value == "cleanup_before_serve":
        return _move_matching_block(actions, _is_serving_action, earliest=False)
    return list(actions)

# Public modifier
class WorkflowPreferenceModifier:
    """Apply a workflow preference to a base recipe action list."""
    # ``AXES`` is the single canonical schema and transform order. Keeping
    # this alias prevents documentation, metadata, and composition order from
    # drifting apart again.
    AXIS_ORDER: Tuple[str, ...] = AXES
    AXIS_FN = {
        "ingredient_flow": _apply_ingredient_flow,
        "equipment_setup": _apply_equipment_setup,
        "serving_setup": _apply_serving_setup,
        "container_loading_style": _apply_container_loading_style,
        "appliance_shutdown_timing": _apply_appliance_shutdown_timing,
        "cook_start_timing": _apply_cook_start_timing,
        "cleanup_timing": _apply_cleanup_timing,
        "serving_priority": _apply_serving_priority,
    }
    DEFAULT_VALUES = WorkflowPreference().as_dict()

    def __init__(self) -> None:
        self.last_result: Optional[ModificationResult] = None

    def modify_recipe_with_report(self, actions: Sequence[str], preference: WorkflowPreference) -> ModificationResult:
        if not _try_validate(actions):  raise ValueError("modify_recipe: input action list is already invalid")
        base_goal = task_goal_signature(actions)
        modified = list(actions)
        applied: List[str] = []
        failed: List[str] = []
        unchanged: List[str] = []
        prefs_dict = preference.as_dict()
        for axis in self.AXIS_ORDER:
            value = prefs_dict[axis]
            if value == self.DEFAULT_VALUES[axis]:
                # A default value denotes the raw generator policy, not an
                # instruction to run a second hidden transform.  This makes
                # one-axis presets genuinely one-axis interventions.
                unchanged.append(axis)
                continue
            before = list(modified)
            candidate = self.AXIS_FN[axis](modified, value)
            if not validate_ordering(candidate, expected_goal=base_goal):
                failed.append(axis)
                continue
            modified = list(candidate)
            if tuple(modified) != tuple(before): applied.append(axis)
            elif value != self.DEFAULT_VALUES[axis]: failed.append(axis)
            else: unchanged.append(axis)
        if not validate_ordering(modified, expected_goal=base_goal):
            raise ValueError("modify_recipe: composite preference reordering is invalid or changes the task goal")
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
    # Identity is the experimental control: it must preserve the generator's
    # raw ordering exactly, rather than passing through a normalising modifier.
    if preset_name == "identity":
        return ModificationResult(
            actions=list(actions),
            applied_axes=[],
            failed_axes=[],
            unchanged_axes=list(WorkflowPreferenceModifier.AXIS_ORDER),
            axis_values=dict(pref.as_dict()),
        )
    return WorkflowPreferenceModifier().modify_recipe_with_report(actions, pref)
