"""Symbolic kitchen state, action grammar, semantics, and reference recipes."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Mapping
import numpy as np


CONTAINERS = ("pot", "pan", "plate", "bowl", "glass", "measuring_cup")
LIQUID_INGREDIENTS = ("milk", "oil")
SOLID_INGREDIENTS = ("tomato", "garlic", "onion", "mushroom", "lettuce", "cheese", "rice", "yoghurt", "strawberries", "banana", "egg", "fish", "chicken", "meat", "salt", "spice1", "spice2", "mixture")
INGREDIENTS = LIQUID_INGREDIENTS + SOLID_INGREDIENTS
ITEMS = CONTAINERS + INGREDIENTS
CUTTABLES = ("tomato", "onion", "mushroom", "lettuce", "banana", "strawberries", "chicken", "fish", "cheese")
GRATABLE = ("cheese",)
COOKABLES = ("meat", "egg", "rice", "tomato", "onion", "mushroom", "chicken", "fish", "mixture")
SEASONINGS = ("salt", "spice1", "spice2", "garlic")
LOCATIONS = ("storage", "prep_station", "cooking_station", "plating_station", "serving_station", "washing_station", "blending_station")
TOOLS = ("stove", "sink", "blender")

ACTION_ARGUMENTS: Mapping[str, tuple[str, ...]] = {
    "transfer": ("item", "from", "to"),
    "load": ("item", "container", "location"),
    "unload": ("item", "container", "location"),
    "move_container": ("container", "from", "to"),
    "cut": ("item", "location"),
    "grate": ("item", "location"),
    "cook": ("item", "container", "location"),
    "cook_contents": ("container", "location"),
    "combine": ("container", "location"),
    "season_container": ("container", "seasoning", "location"),
    "season": ("item", "seasoning", "location"),
    "pour": ("liquid", "from_container", "to_container", "location"),
    "turn_on": ("tool",),
    "turn_off": ("tool",),
    "blend": ("container", "location"),
    "serve": ("vessel", "location"),
    "wash": ("item", "location"),
}

# Reference trajectories ground appliance actions at a location, whereas the structured LLM interface needs only the tool name. The simulator accepts both.
ACTION_LABEL_ARGUMENTS: Mapping[str, tuple[str, ...]] = {**ACTION_ARGUMENTS, "turn_on": ("tool", "location"), "turn_off": ("tool", "location")}
_ACTION_PATTERN = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*$")


class ActionSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    verb: str
    args: Mapping[str, str]
    def get(self, key: str, default: str | None = None) -> str | None: return self.args.get(key, default)


def parse_action_label(label: str) -> Action:
    match = _ACTION_PATTERN.fullmatch(label)
    if match is None:               raise ActionSyntaxError(f"invalid action syntax: {label!r}")
    verb, body = match.groups()
    names = ACTION_LABEL_ARGUMENTS.get(verb)
    if names is None:               raise ActionSyntaxError(f"unknown action: {verb}")
    tokens = [part.strip() for part in body.split(",") if part.strip()]
    if len(tokens) > len(names):    raise ActionSyntaxError(f"too many arguments for {verb}")
    values: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if "=" in token:
            key, value = (part.strip() for part in token.split("=", 1))
            if key not in names:    raise ActionSyntaxError(f"unknown {verb} argument: {key}")
        else: key, value = names[index], token
        if key in values:           raise ActionSyntaxError(f"duplicate {verb} argument: {key}")
        values[key] = value
    return Action(verb, values)


def _validate_argument(action: str, key: str, value: object) -> str:
    if not isinstance(value, str): raise ActionSyntaxError(f"argument_{key}_must_be_string")
    domains = {"from": LOCATIONS, "to": LOCATIONS, "location": LOCATIONS,"container": CONTAINERS, "from_container": CONTAINERS, "to_container": CONTAINERS, "tool": TOOLS, "seasoning": SEASONINGS, "liquid": LIQUID_INGREDIENTS, "vessel": ("plate", "glass"), "item": ITEMS}
    if key in domains and value not in domains[key]: raise ActionSyntaxError(f"invalid_{key}:{value}")
    if action in {"load", "unload", "season"} and key == "item" and value not in INGREDIENTS: raise ActionSyntaxError(f"{action}_requires_ingredient:{value}")
    return value


def canonical_action(action: str, args: Mapping[str, object]) -> str:
    names = ACTION_ARGUMENTS.get(action)
    if names is None:           raise ActionSyntaxError(f"unknown_action:{action}")
    if set(args) != set(names): raise ActionSyntaxError(f"wrong_argument_keys_for:{action}")
    values = {key: _validate_argument(action, key, args[key]) for key in names}
    if action in {"transfer", "move_container"}:
        first = "item" if action == "transfer" else "container"
        return f"{action} ({values[first]}, from={values['from']}, to={values['to']})"
    return f"{action} ({', '.join(values[key] for key in names)})"


_TOOL_FEATURE = {tool: f"{tool}_on" for tool in TOOLS}
class StateTracker:
    """PDDL-style kitchen simulator with a shared fixed-width feature map."""

    _CLASS_FEATURE_MAP = None

    @classmethod
    def _build_feature_map(cls):
        """Build the systematic feature layout once and memoize it."""
        if cls._CLASS_FEATURE_MAP is not None:
            return cls._CLASS_FEATURE_MAP

        feature_map, index = {}, 0
        for item in ITEMS:
            for location in LOCATIONS:
                feature_map[f"{item}_at_{location}"] = index
                index += 1
        for container in CONTAINERS:
            for ingredient in INGREDIENTS:
                feature_map[f"{container}_contains_{ingredient}"] = index
                index += 1
        for item in CUTTABLES:
            feature_map[f"{item}_cut"] = index
            index += 1
        for item in GRATABLE:
            feature_map[f"{item}_grated"] = index
            index += 1
        for item in COOKABLES:
            feature_map[f"{item}_cooked"] = index
            index += 1
        for item in INGREDIENTS:
            feature_map[f"{item}_seasoned"] = index
            index += 1
        for item in ITEMS:
            feature_map[f"{item}_washed"] = index
            index += 1
        # Serving is vessel-agnostic because plates and glasses are valid.
        for feature in ("stove_on", "sink_on", "blender_on", "dish_served"):
            feature_map[feature] = index
            index += 1
        for ingredient in INGREDIENTS:
            if ingredient != "mixture":
                feature_map[f"{ingredient}_in_mixture"] = index
                index += 1
        for seasoning in SEASONINGS:
            for item in INGREDIENTS:
                feature_map[f"{item}_seasoned_with_{seasoning}"] = index
                index += 1

        cls._CLASS_FEATURE_MAP = feature_map
        return feature_map

    def __init__(self):
        self.feature_map = self.__class__._build_feature_map()
        self.n_features  = len(self.feature_map)
        self.reset()

    def reset(self):
        """Initial state: every real item lives in storage; no tool is on."""
        self.current_state = np.zeros(self.n_features, dtype=int)
        for item in ITEMS:
            if item != "mixture": self.set_feature(f"{item}_at_storage", 1)  # "mixture" only exists once ingredients are combined


    def set_feature(self, key, value):
        if key in self.feature_map:
            self.current_state[self.feature_map[key]] = value

    def get_feature(self, key):
        return self.current_state[self.feature_map[key]] if key in self.feature_map else 0

    def get_state_vector(self):
        return self.current_state.copy()

    def get_item_location(self, item):
        """Return the location an item is at, or None if none/contained."""
        for loc in LOCATIONS:
            if self.get_feature(f"{item}_at_{loc}") == 1: return loc
        return None

    def is_contained(self, item):
        """Return the container holding `item`, or None if it is free."""
        for container in CONTAINERS:
            if self.get_feature(f"{container}_contains_{item}") == 1: return container
        return None

    def apply_action(self, action_str, *, enforce_preconditions: bool = False):
        """Apply an action, optionally enforcing symbolic preconditions."""
        action = parse_action_label(action_str)
        verb, args = action.verb, action.args

        def _require(condition, msg):
            if enforce_preconditions and not condition: raise ValueError(msg)

        if verb == "transfer":
            item, from_loc, to_loc = args["item"], args["from"], args["to"]
            _require(self.get_feature(f"{item}_at_{from_loc}") == 1, f"Precondition failed: {item} not at {from_loc}")
            _require(not self.is_contained(item), f"Precondition failed: {item} is contained")
            self.set_feature(f"{item}_at_{from_loc}", 0)
            self.set_feature(f"{item}_at_{to_loc}",   1)

        elif verb == "load":
            item, container, location = args["item"], args["container"], args["location"]
            _require(self.get_feature(f"{item}_at_{location}") == 1, f"Precondition failed: {item} not at {location}")
            _require(self.get_feature(f"{container}_at_{location}") == 1, f"Precondition failed: {container} not at {location}")
            _require(not self.is_contained(item), f"Precondition failed: {item} already contained")
            self.set_feature(f"{item}_at_{location}", 0)
            self.set_feature(f"{container}_contains_{item}", 1)

        elif verb == "unload":
            item, container, location = args["item"], args["container"], args["location"]
            _require(self.get_feature(f"{container}_contains_{item}") == 1, f"Precondition failed: {item} not in {container}")
            _require(self.get_feature(f"{container}_at_{location}") == 1, f"Precondition failed: {container} not at {location}")
            self.set_feature(f"{container}_contains_{item}", 0)
            self.set_feature(f"{item}_at_{location}", 1)

        elif verb == "move_container":
            container, from_loc, to_loc = args["container"], args["from"], args["to"]
            _require(self.get_feature(f"{container}_at_{from_loc}") == 1, f"Precondition failed: {container} not at {from_loc}")
            self.set_feature(f"{container}_at_{from_loc}", 0)
            self.set_feature(f"{container}_at_{to_loc}", 1)

        elif verb == "cut":
            item, location = args["item"], args.get("location", "prep_station")
            _require(item in CUTTABLES, f"Precondition failed: {item} is not cuttable")
            _require(self.get_feature(f"{item}_at_{location}") == 1, f"Precondition failed: {item} not at {location}")
            self.set_feature(f"{item}_cut", 1)

        elif verb == "grate":
            item, location = args["item"], args.get("location", "prep_station")
            _require(item in GRATABLE, f"Precondition failed: {item} is not gratable")
            _require(self.get_feature(f"{item}_at_{location}") == 1, f"Precondition failed: {item} not at {location}")
            self.set_feature(f"{item}_grated", 1)

        elif verb == "cook":
            item = args["item"]
            container = args.get("container", "pot")
            location = args.get("location", "cooking_station")
            _require(item in COOKABLES, f"Precondition failed: {item} is not cookable")
            _require(container in {"pot", "pan"}, f"Precondition failed: {container} is not cook-safe")
            _require(self.get_feature(f"{container}_contains_{item}") == 1, f"Precondition failed: {item} not in {container}")
            _require(self.get_feature(f"{container}_at_{location}") == 1, f"Precondition failed: {container} not at {location}")
            _require(location != "cooking_station" or self.get_feature("stove_on") == 1, "Precondition failed: stove not on")
            self.set_feature(f"{item}_cooked", 1)

        elif verb == "cook_contents":
            container = args["container"]
            location = args.get("location", "cooking_station")
            _require(container in {"pot", "pan"}, f"Precondition failed: {container} is not cook-safe")
            _require(self.get_feature(f"{container}_at_{location}") == 1, f"Precondition failed: {container} not at {location}")
            _require(location != "cooking_station" or self.get_feature("stove_on") == 1, "Precondition failed: stove not on")
            _require(any(self.get_feature(f"{container}_contains_{ingredient}") == 1 and ingredient in COOKABLES for ingredient in INGREDIENTS), f"Precondition failed: {container} contains no cookable ingredients")
            for ingredient in INGREDIENTS:
                if (self.get_feature(f"{container}_contains_{ingredient}") == 1 and ingredient in COOKABLES): self.set_feature(f"{ingredient}_cooked", 1)

        elif verb == "combine":
            container, location = args["container"], args.get("location")
            _require(not location or self.get_feature(f"{container}_at_{location}") == 1, f"Precondition failed: {container} not at {location}")
            contained = [ing for ing in INGREDIENTS if ing != "mixture" and self.get_feature(f"{container}_contains_{ing}") == 1]
            _require(len(contained) >= 2, f"combine requires >=2 ingredients in {container}")
            for ing in contained:
                self.set_feature(f"{container}_contains_{ing}", 0)
                self.set_feature(f"{ing}_in_mixture", 1)
            self.set_feature(f"{container}_contains_mixture", 1)

        elif verb == "season_container":
            container, seasoning = args["container"], args["seasoning"]
            location = args.get("location")
            _require(seasoning in SEASONINGS, f"{seasoning} is not a seasoning")
            _require(location is not None, "season_container requires an explicit location")
            _require(self.get_feature(f"{container}_at_{location}") == 1, f"{container} not at {location}")
            _require(any(self.get_feature(f"{container}_contains_{ingredient}") == 1 for ingredient in INGREDIENTS), f"Precondition failed: {container} contains no ingredients")
            for ing in INGREDIENTS:
                if self.get_feature(f"{container}_contains_{ing}") == 1:
                    self.set_feature(f"{ing}_seasoned", 1)
                    self.set_feature(f"{ing}_seasoned_with_{seasoning}", 1)

        elif verb == "season":
            target, seasoning = args["item"], args["seasoning"]
            location = args.get("location")
            _require(target in INGREDIENTS, f"{target} is not an ingredient")
            _require(seasoning in SEASONINGS, f"{seasoning} is not a seasoning")
            _require(location is not None, "season requires an explicit location")
            _require(self.get_feature(f"{target}_at_{location}") == 1, f"{target} not at {location}")
            self.set_feature(f"{target}_seasoned", 1)
            self.set_feature(f"{target}_seasoned_with_{seasoning}", 1)

        elif verb == "pour":
            liquid = args["liquid"]
            from_c, to_c = args["from_container"], args["to_container"]
            location = args.get("location")
            _require(liquid in LIQUID_INGREDIENTS, f"{liquid} is not a liquid")
            _require(self.get_feature(f"{from_c}_contains_{liquid}") == 1, f"{liquid} not in {from_c}")
            if location:
                _require(self.get_feature(f"{from_c}_at_{location}") == 1, f"{from_c} not at {location}")
                _require(self.get_feature(f"{to_c}_at_{location}") == 1, f"{to_c} not at {location}")
            self.set_feature(f"{from_c}_contains_{liquid}", 0)
            self.set_feature(f"{to_c}_contains_{liquid}", 1)

        elif verb in {"turn_on", "turn_off"}:
            tool = args["tool"]
            _require(tool in TOOLS, f"Precondition failed: unknown tool {tool}")
            if tool in _TOOL_FEATURE:
                self.set_feature(_TOOL_FEATURE[tool], int(verb == "turn_on"))

        elif verb == "blend":
            container = args["container"]
            location = args.get("location", "blending_station")
            _require(self.get_feature(f"{container}_at_{location}") == 1, f"{container} not at {location}")
            _require(self.get_feature("blender_on") == 1, "blender not on")
            contained = [ing for ing in INGREDIENTS if ing != "mixture" and self.get_feature(f"{container}_contains_{ing}") == 1]
            _require(len(contained) >= 1, f"blend requires >=1 ingredient in {container}")
            for ing in contained:
                self.set_feature(f"{container}_contains_{ing}", 0)
                self.set_feature(f"{ing}_in_mixture", 1)
            self.set_feature(f"{container}_contains_mixture", 1)

        elif verb == "serve":
            vessel = args["vessel"]
            location = args.get("location", "serving_station")
            _require(vessel in {"plate", "glass"}, f"Precondition failed: {vessel} is not a serving vessel")
            _require(location == "serving_station", "serve requires serving_station")
            _require(self.get_feature(f"{vessel}_at_{location}") == 1, f"{vessel} not at {location}")
            _require(any(self.get_feature(f"{vessel}_contains_{ingredient}") == 1 for ingredient in INGREDIENTS), f"{vessel} contains no dish")
            self.set_feature("dish_served", 1)

        elif verb == "wash":
            item = args["item"]
            location = args.get("location", "washing_station")
            _require(self.get_feature(f"{item}_at_{location}") == 1, f"{item} not at {location}")
            self.set_feature(f"{item}_washed", 1)


# Shared feature lookup avoids per-access tracker allocation.
_FEAT = StateTracker._build_feature_map()


_GOAL_FEATURE_INDICES = tuple(index for name, index in _FEAT.items() if "_at_" not in name and not name.endswith("_washed") and name not in {"stove_on", "sink_on", "blender_on"})


def replay_validated_actions(actions):
    """Replay a task strictly, rejecting invalid or no-effect actions."""
    tracker = StateTracker()
    for action in actions:
        before = tuple(tracker.get_state_vector().astype(int).tolist())
        tracker.apply_action(action, enforce_preconditions=True)
        after = tuple(tracker.get_state_vector().astype(int).tolist())
        if after == before: raise ValueError(f"No-effect action in validated ordering: {action}")
    return tuple(tracker.get_state_vector().astype(int).tolist())


def task_goal_signature(actions):
    """Return task outcomes after excluding workflow-dependent state."""
    final_state = replay_validated_actions(actions)
    return tuple(final_state[index] for index in _GOAL_FEATURE_INDICES)


def validate_ordering(actions, *, expected_goal=None):
    """Return whether an ordering is executable, effectful, and goal-preserving."""
    try: goal = task_goal_signature(actions)
    except (ValueError, IndexError, KeyError): return False
    return expected_goal is None or tuple(goal) == tuple(expected_goal)

RECIPES = {'tomato_onion_soup': ('transfer (pot, from=storage, to=cooking_station)',
                          'transfer (bowl, from=storage, to=prep_station)',
                          'transfer (tomato, from=storage, to=prep_station)',
                          'cut (tomato, prep_station)',
                          'load (tomato, bowl, prep_station)',
                          'transfer (onion, from=storage, to=prep_station)',
                          'cut (onion, prep_station)',
                          'load (onion, bowl, prep_station)',
                          'combine (bowl, prep_station)',
                          'move_container (bowl, from=prep_station, to=cooking_station)',
                          'unload (mixture, bowl, cooking_station)',
                          'load (mixture, pot, cooking_station)',
                          'turn_on (stove, cooking_station)',
                          'cook_contents (pot, cooking_station)',
                          'turn_off (stove, cooking_station)',
                          'transfer (plate, from=storage, to=plating_station)',
                          'move_container (pot, from=cooking_station, to=plating_station)',
                          'unload (mixture, pot, plating_station)',
                          'load (mixture, plate, plating_station)',
                          'move_container (plate, from=plating_station, to=serving_station)',
                          'serve (plate, serving_station)',
                          'move_container (pot, from=plating_station, to=washing_station)',
                          'wash (pot, washing_station)',
                          'transfer (bowl, from=cooking_station, to=washing_station)',
                          'wash (bowl, washing_station)'),
 'tomato_soup': ('transfer (pot, from=storage, to=cooking_station)',
                 'transfer (bowl, from=storage, to=prep_station)',
                 'turn_on (stove, cooking_station)',
                 'transfer (tomato, from=storage, to=prep_station)',
                 'cut (tomato, prep_station)',
                 'load (tomato, bowl, prep_station)',
                 'move_container (bowl, from=prep_station, to=cooking_station)',
                 'unload (tomato, bowl, cooking_station)',
                 'load (tomato, pot, cooking_station)',
                 'cook_contents (pot, cooking_station)',
                 'turn_off (stove, cooking_station)',
                 'transfer (plate, from=storage, to=plating_station)',
                 'unload (tomato, pot, cooking_station)',
                 'load (tomato, bowl, cooking_station)',
                 'move_container (bowl, from=cooking_station, to=plating_station)',
                 'unload (tomato, bowl, plating_station)',
                 'load (tomato, plate, plating_station)',
                 'move_container (plate, from=plating_station, to=serving_station)',
                 'serve (plate, serving_station)',
                 'transfer (pot, from=cooking_station, to=washing_station)',
                 'wash (pot, washing_station)',
                 'transfer (bowl, from=plating_station, to=washing_station)',
                 'wash (bowl, washing_station)'),
 'mushroom_soup': ('transfer (pot, from=storage, to=cooking_station)',
                   'transfer (bowl, from=storage, to=prep_station)',
                   'transfer (mushroom, from=storage, to=prep_station)',
                   'cut (mushroom, prep_station)',
                   'load (mushroom, bowl, prep_station)',
                   'transfer (onion, from=storage, to=prep_station)',
                   'cut (onion, prep_station)',
                   'load (onion, bowl, prep_station)',
                   'combine (bowl, prep_station)',
                   'move_container (bowl, from=prep_station, to=cooking_station)',
                   'unload (mixture, bowl, cooking_station)',
                   'load (mixture, pot, cooking_station)',
                   'turn_on (stove, cooking_station)',
                   'cook_contents (pot, cooking_station)',
                   'turn_off (stove, cooking_station)',
                   'transfer (plate, from=storage, to=plating_station)',
                   'move_container (pot, from=cooking_station, to=plating_station)',
                   'unload (mixture, pot, plating_station)',
                   'load (mixture, plate, plating_station)',
                   'move_container (plate, from=plating_station, to=serving_station)',
                   'serve (plate, serving_station)',
                   'move_container (pot, from=plating_station, to=washing_station)',
                   'wash (pot, washing_station)',
                   'transfer (bowl, from=cooking_station, to=washing_station)',
                   'wash (bowl, washing_station)'),
 'seasoned_mixture_soup': ('transfer (pot, from=storage, to=cooking_station)',
                           'transfer (bowl, from=storage, to=prep_station)',
                           'transfer (tomato, from=storage, to=prep_station)',
                           'cut (tomato, prep_station)',
                           'load (tomato, bowl, prep_station)',
                           'transfer (onion, from=storage, to=prep_station)',
                           'cut (onion, prep_station)',
                           'load (onion, bowl, prep_station)',
                           'combine (bowl, prep_station)',
                           'season_container (bowl, salt, prep_station)',
                           'season_container (bowl, spice1, prep_station)',
                           'move_container (bowl, from=prep_station, to=cooking_station)',
                           'unload (mixture, bowl, cooking_station)',
                           'load (mixture, pot, cooking_station)',
                           'turn_on (stove, cooking_station)',
                           'cook_contents (pot, cooking_station)',
                           'turn_off (stove, cooking_station)',
                           'transfer (plate, from=storage, to=plating_station)',
                           'move_container (pot, from=cooking_station, to=plating_station)',
                           'unload (mixture, pot, plating_station)',
                           'load (mixture, plate, plating_station)',
                           'move_container (plate, from=plating_station, to=serving_station)',
                           'serve (plate, serving_station)',
                           'move_container (pot, from=plating_station, to=washing_station)',
                           'wash (pot, washing_station)',
                           'transfer (bowl, from=cooking_station, to=washing_station)',
                           'wash (bowl, washing_station)'),
 'grilled_steak': ('transfer (pan, from=storage, to=cooking_station)',
                   'load (meat, bowl, storage)',
                   'move_container (bowl, from=storage, to=cooking_station)',
                   'unload (meat, bowl, cooking_station)',
                   'load (meat, pan, cooking_station)',
                   'turn_on (stove, cooking_station)',
                   'cook_contents (pan, cooking_station)',
                   'turn_off (stove, cooking_station)',
                   'transfer (plate, from=storage, to=plating_station)',
                   'unload (meat, pan, cooking_station)',
                   'load (meat, bowl, cooking_station)',
                   'move_container (bowl, from=cooking_station, to=plating_station)',
                   'unload (meat, bowl, plating_station)',
                   'load (meat, plate, plating_station)',
                   'move_container (plate, from=plating_station, to=serving_station)',
                   'serve (plate, serving_station)',
                   'transfer (pan, from=cooking_station, to=washing_station)',
                   'wash (pan, washing_station)',
                   'transfer (bowl, from=plating_station, to=washing_station)',
                   'wash (bowl, washing_station)'),
 'burger': ('transfer (pan, from=storage, to=cooking_station)',
            'load (meat, bowl, storage)',
            'move_container (bowl, from=storage, to=cooking_station)',
            'unload (meat, bowl, cooking_station)',
            'load (meat, pan, cooking_station)',
            'turn_on (stove, cooking_station)',
            'cook_contents (pan, cooking_station)',
            'turn_off (stove, cooking_station)',
            'transfer (plate, from=storage, to=plating_station)',
            'unload (meat, pan, cooking_station)',
            'load (meat, bowl, cooking_station)',
            'move_container (bowl, from=cooking_station, to=plating_station)',
            'unload (meat, bowl, plating_station)',
            'load (meat, plate, plating_station)',
            'transfer (lettuce, from=storage, to=prep_station)',
            'move_container (bowl, from=plating_station, to=prep_station)',
            'cut (lettuce, prep_station)',
            'load (lettuce, bowl, prep_station)',
            'move_container (bowl, from=prep_station, to=plating_station)',
            'unload (lettuce, bowl, plating_station)',
            'load (lettuce, plate, plating_station)',
            'move_container (plate, from=plating_station, to=serving_station)',
            'serve (plate, serving_station)',
            'transfer (pan, from=cooking_station, to=washing_station)',
            'wash (pan, washing_station)',
            'transfer (bowl, from=plating_station, to=washing_station)',
            'wash (bowl, washing_station)'),
 'seasoned_chicken': ('transfer (pan, from=storage, to=cooking_station)',
                      'transfer (bowl, from=storage, to=prep_station)',
                      'transfer (chicken, from=storage, to=prep_station)',
                      'season (chicken, salt, prep_station)',
                      'season (chicken, spice1, prep_station)',
                      'load (chicken, bowl, prep_station)',
                      'move_container (bowl, from=prep_station, to=cooking_station)',
                      'unload (chicken, bowl, cooking_station)',
                      'load (chicken, pan, cooking_station)',
                      'turn_on (stove, cooking_station)',
                      'cook_contents (pan, cooking_station)',
                      'turn_off (stove, cooking_station)',
                      'transfer (plate, from=storage, to=plating_station)',
                      'unload (chicken, pan, cooking_station)',
                      'load (chicken, bowl, cooking_station)',
                      'move_container (bowl, from=cooking_station, to=plating_station)',
                      'unload (chicken, bowl, plating_station)',
                      'load (chicken, plate, plating_station)',
                      'move_container (plate, from=plating_station, to=serving_station)',
                      'serve (plate, serving_station)',
                      'transfer (pan, from=cooking_station, to=washing_station)',
                      'wash (pan, washing_station)',
                      'transfer (bowl, from=plating_station, to=washing_station)',
                      'wash (bowl, washing_station)'),
 'garlic_fish': ('transfer (pan, from=storage, to=cooking_station)',
                 'transfer (bowl, from=storage, to=prep_station)',
                 'transfer (fish, from=storage, to=prep_station)',
                 'season (fish, garlic, prep_station)',
                 'season (fish, spice2, prep_station)',
                 'load (fish, bowl, prep_station)',
                 'move_container (bowl, from=prep_station, to=cooking_station)',
                 'unload (fish, bowl, cooking_station)',
                 'load (fish, pan, cooking_station)',
                 'turn_on (stove, cooking_station)',
                 'cook_contents (pan, cooking_station)',
                 'turn_off (stove, cooking_station)',
                 'transfer (plate, from=storage, to=plating_station)',
                 'unload (fish, pan, cooking_station)',
                 'load (fish, bowl, cooking_station)',
                 'move_container (bowl, from=cooking_station, to=plating_station)',
                 'unload (fish, bowl, plating_station)',
                 'load (fish, plate, plating_station)',
                 'move_container (plate, from=plating_station, to=serving_station)',
                 'serve (plate, serving_station)',
                 'transfer (pan, from=cooking_station, to=washing_station)',
                 'wash (pan, washing_station)',
                 'transfer (bowl, from=plating_station, to=washing_station)',
                 'wash (bowl, washing_station)'),
 'simple_salad': ('transfer (bowl, from=storage, to=prep_station)',
                  'transfer (lettuce, from=storage, to=prep_station)',
                  'transfer (onion, from=storage, to=prep_station)',
                  'cut (lettuce, prep_station)',
                  'load (lettuce, bowl, prep_station)',
                  'cut (onion, prep_station)',
                  'load (onion, bowl, prep_station)',
                  'combine (bowl, prep_station)',
                  'transfer (plate, from=storage, to=plating_station)',
                  'move_container (bowl, from=prep_station, to=plating_station)',
                  'unload (mixture, bowl, plating_station)',
                  'load (mixture, plate, plating_station)',
                  'move_container (plate, from=plating_station, to=serving_station)',
                  'serve (plate, serving_station)',
                  'move_container (plate, from=serving_station, to=washing_station)',
                  'wash (plate, washing_station)',
                  'transfer (bowl, from=plating_station, to=washing_station)',
                  'wash (bowl, washing_station)'),
 'grated_cheese_salad': ('transfer (bowl, from=storage, to=prep_station)',
                         'transfer (lettuce, from=storage, to=prep_station)',
                         'transfer (cheese, from=storage, to=prep_station)',
                         'cut (lettuce, prep_station)',
                         'load (lettuce, bowl, prep_station)',
                         'grate (cheese, prep_station)',
                         'load (cheese, bowl, prep_station)',
                         'combine (bowl, prep_station)',
                         'transfer (plate, from=storage, to=plating_station)',
                         'move_container (bowl, from=prep_station, to=plating_station)',
                         'unload (mixture, bowl, plating_station)',
                         'load (mixture, plate, plating_station)',
                         'move_container (plate, from=plating_station, to=serving_station)',
                         'serve (plate, serving_station)',
                         'move_container (plate, from=serving_station, to=washing_station)',
                         'wash (plate, washing_station)',
                         'transfer (bowl, from=plating_station, to=washing_station)',
                         'wash (bowl, washing_station)'),
 'smoothie': ('transfer (banana, from=storage, to=prep_station)',
              'transfer (strawberries, from=storage, to=prep_station)',
              'cut (banana, prep_station)',
              'transfer (banana, from=prep_station, to=blending_station)',
              'cut (strawberries, prep_station)',
              'transfer (strawberries, from=prep_station, to=blending_station)',
              'transfer (milk, from=storage, to=blending_station)',
              'transfer (glass, from=storage, to=blending_station)',
              'transfer (measuring_cup, from=storage, to=blending_station)',
              'load (milk, measuring_cup, blending_station)',
              'pour (milk, measuring_cup, glass, blending_station)',
              'load (banana, glass, blending_station)',
              'load (strawberries, glass, blending_station)',
              'turn_on (blender, blending_station)',
              'blend (glass, blending_station)',
              'turn_off (blender, blending_station)',
              'transfer (glass, from=blending_station, to=serving_station)',
              'serve (glass, serving_station)',
              'transfer (measuring_cup, from=blending_station, to=washing_station)',
              'wash (measuring_cup, washing_station)'),
 'yoghurt_smoothie': ('transfer (banana, from=storage, to=prep_station)',
                      'transfer (yoghurt, from=storage, to=blending_station)',
                      'cut (banana, prep_station)',
                      'transfer (banana, from=prep_station, to=blending_station)',
                      'transfer (glass, from=storage, to=blending_station)',
                      'load (yoghurt, glass, blending_station)',
                      'transfer (milk, from=storage, to=blending_station)',
                      'transfer (measuring_cup, from=storage, to=blending_station)',
                      'load (milk, measuring_cup, blending_station)',
                      'pour (milk, measuring_cup, glass, blending_station)',
                      'load (banana, glass, blending_station)',
                      'turn_on (blender, blending_station)',
                      'blend (glass, blending_station)',
                      'turn_off (blender, blending_station)',
                      'transfer (glass, from=blending_station, to=serving_station)',
                      'serve (glass, serving_station)',
                      'transfer (measuring_cup, from=blending_station, to=washing_station)',
                      'wash (measuring_cup, washing_station)'),
 'boiled_eggs': ('transfer (pot, from=storage, to=cooking_station)',
                 'load (egg, bowl, storage)',
                 'move_container (bowl, from=storage, to=cooking_station)',
                 'unload (egg, bowl, cooking_station)',
                 'load (egg, pot, cooking_station)',
                 'turn_on (stove, cooking_station)',
                 'cook_contents (pot, cooking_station)',
                 'turn_off (stove, cooking_station)',
                 'transfer (plate, from=storage, to=plating_station)',
                 'move_container (pot, from=cooking_station, to=plating_station)',
                 'unload (egg, pot, plating_station)',
                 'load (egg, plate, plating_station)',
                 'move_container (plate, from=plating_station, to=serving_station)',
                 'serve (plate, serving_station)',
                 'transfer (pot, from=plating_station, to=washing_station)',
                 'wash (pot, washing_station)',
                 'transfer (bowl, from=cooking_station, to=washing_station)',
                 'wash (bowl, washing_station)'),
 'boiled_rice': ('transfer (pot, from=storage, to=cooking_station)',
                 'load (rice, bowl, storage)',
                 'move_container (bowl, from=storage, to=cooking_station)',
                 'unload (rice, bowl, cooking_station)',
                 'turn_on (stove, cooking_station)',
                 'load (rice, pot, cooking_station)',
                 'cook_contents (pot, cooking_station)',
                 'turn_off (stove, cooking_station)',
                 'transfer (plate, from=storage, to=plating_station)',
                 'unload (rice, pot, cooking_station)',
                 'load (rice, bowl, cooking_station)',
                 'move_container (bowl, from=cooking_station, to=plating_station)',
                 'unload (rice, bowl, plating_station)',
                 'load (rice, plate, plating_station)',
                 'move_container (plate, from=plating_station, to=serving_station)',
                 'serve (plate, serving_station)',
                 'transfer (pot, from=cooking_station, to=washing_station)',
                 'wash (pot, washing_station)',
                 'transfer (bowl, from=plating_station, to=washing_station)',
                 'wash (bowl, washing_station)'),
 'tomato_garlic_soup': ('transfer (pot, from=storage, to=cooking_station)',
                        'transfer (bowl, from=storage, to=prep_station)',
                        'transfer (tomato, from=storage, to=prep_station)',
                        'cut (tomato, prep_station)',
                        'season (tomato, garlic, prep_station)',
                        'load (tomato, bowl, prep_station)',
                        'move_container (bowl, from=prep_station, to=cooking_station)',
                        'unload (tomato, bowl, cooking_station)',
                        'load (tomato, pot, cooking_station)',
                        'turn_on (stove, cooking_station)',
                        'cook_contents (pot, cooking_station)',
                        'turn_off (stove, cooking_station)',
                        'transfer (plate, from=storage, to=plating_station)',
                        'unload (tomato, pot, cooking_station)',
                        'load (tomato, bowl, cooking_station)',
                        'move_container (bowl, from=cooking_station, to=plating_station)',
                        'unload (tomato, bowl, plating_station)',
                        'load (tomato, plate, plating_station)',
                        'move_container (plate, from=plating_station, to=serving_station)',
                        'serve (plate, serving_station)',
                        'transfer (pot, from=cooking_station, to=washing_station)',
                        'wash (pot, washing_station)',
                        'transfer (bowl, from=plating_station, to=washing_station)',
                        'wash (bowl, washing_station)'),
 'mushroom_garlic_soup': ('transfer (pot, from=storage, to=cooking_station)',
                          'transfer (bowl, from=storage, to=prep_station)',
                          'transfer (mushroom, from=storage, to=prep_station)',
                          'cut (mushroom, prep_station)',
                          'season (mushroom, garlic, prep_station)',
                          'load (mushroom, bowl, prep_station)',
                          'move_container (bowl, from=prep_station, to=cooking_station)',
                          'unload (mushroom, bowl, cooking_station)',
                          'load (mushroom, pot, cooking_station)',
                          'turn_on (stove, cooking_station)',
                          'cook_contents (pot, cooking_station)',
                          'turn_off (stove, cooking_station)',
                          'transfer (plate, from=storage, to=plating_station)',
                          'unload (mushroom, pot, cooking_station)',
                          'load (mushroom, bowl, cooking_station)',
                          'move_container (bowl, from=cooking_station, to=plating_station)',
                          'unload (mushroom, bowl, plating_station)',
                          'load (mushroom, plate, plating_station)',
                          'move_container (plate, from=plating_station, to=serving_station)',
                          'serve (plate, serving_station)',
                          'transfer (pot, from=cooking_station, to=washing_station)',
                          'wash (pot, washing_station)',
                          'transfer (bowl, from=plating_station, to=washing_station)',
                          'wash (bowl, washing_station)'),
 'tomato_mushroom_soup': ('transfer (pot, from=storage, to=cooking_station)',
                          'transfer (bowl, from=storage, to=prep_station)',
                          'transfer (tomato, from=storage, to=prep_station)',
                          'cut (tomato, prep_station)',
                          'load (tomato, bowl, prep_station)',
                          'transfer (mushroom, from=storage, to=prep_station)',
                          'cut (mushroom, prep_station)',
                          'load (mushroom, bowl, prep_station)',
                          'combine (bowl, prep_station)',
                          'move_container (bowl, from=prep_station, to=cooking_station)',
                          'unload (mixture, bowl, cooking_station)',
                          'load (mixture, pot, cooking_station)',
                          'turn_on (stove, cooking_station)',
                          'cook_contents (pot, cooking_station)',
                          'turn_off (stove, cooking_station)',
                          'transfer (plate, from=storage, to=plating_station)',
                          'move_container (pot, from=cooking_station, to=plating_station)',
                          'unload (mixture, pot, plating_station)',
                          'load (mixture, plate, plating_station)',
                          'move_container (plate, from=plating_station, to=serving_station)',
                          'serve (plate, serving_station)',
                          'move_container (pot, from=plating_station, to=washing_station)',
                          'wash (pot, washing_station)',
                          'transfer (bowl, from=cooking_station, to=washing_station)',
                          'wash (bowl, washing_station)'),
 'onion_rice_pot': ('transfer (pot, from=storage, to=cooking_station)',
                    'transfer (bowl, from=storage, to=prep_station)',
                    'transfer (rice, from=storage, to=prep_station)',
                    'load (rice, bowl, prep_station)',
                    'transfer (onion, from=storage, to=prep_station)',
                    'cut (onion, prep_station)',
                    'load (onion, bowl, prep_station)',
                    'combine (bowl, prep_station)',
                    'move_container (bowl, from=prep_station, to=cooking_station)',
                    'unload (mixture, bowl, cooking_station)',
                    'load (mixture, pot, cooking_station)',
                    'turn_on (stove, cooking_station)',
                    'cook_contents (pot, cooking_station)',
                    'turn_off (stove, cooking_station)',
                    'transfer (plate, from=storage, to=plating_station)',
                    'move_container (pot, from=cooking_station, to=plating_station)',
                    'unload (mixture, pot, plating_station)',
                    'load (mixture, plate, plating_station)',
                    'move_container (plate, from=plating_station, to=serving_station)',
                    'serve (plate, serving_station)',
                    'move_container (pot, from=plating_station, to=washing_station)',
                    'wash (pot, washing_station)',
                    'transfer (bowl, from=cooking_station, to=washing_station)',
                    'wash (bowl, washing_station)'),
 'tomato_cheese_salad': ('transfer (bowl, from=storage, to=prep_station)',
                         'transfer (tomato, from=storage, to=prep_station)',
                         'transfer (cheese, from=storage, to=prep_station)',
                         'cut (tomato, prep_station)',
                         'load (tomato, bowl, prep_station)',
                         'grate (cheese, prep_station)',
                         'load (cheese, bowl, prep_station)',
                         'combine (bowl, prep_station)',
                         'transfer (plate, from=storage, to=plating_station)',
                         'move_container (bowl, from=prep_station, to=plating_station)',
                         'unload (mixture, bowl, plating_station)',
                         'load (mixture, plate, plating_station)',
                         'move_container (plate, from=plating_station, to=serving_station)',
                         'serve (plate, serving_station)',
                         'move_container (plate, from=serving_station, to=washing_station)',
                         'wash (plate, washing_station)',
                         'transfer (bowl, from=plating_station, to=washing_station)',
                         'wash (bowl, washing_station)'),
 'tomato_lettuce_salad': ('transfer (bowl, from=storage, to=prep_station)',
                          'transfer (tomato, from=storage, to=prep_station)',
                          'transfer (lettuce, from=storage, to=prep_station)',
                          'cut (tomato, prep_station)',
                          'load (tomato, bowl, prep_station)',
                          'cut (lettuce, prep_station)',
                          'load (lettuce, bowl, prep_station)',
                          'combine (bowl, prep_station)',
                          'transfer (plate, from=storage, to=plating_station)',
                          'move_container (bowl, from=prep_station, to=plating_station)',
                          'unload (mixture, bowl, plating_station)',
                          'load (mixture, plate, plating_station)',
                          'move_container (plate, from=plating_station, to=serving_station)',
                          'serve (plate, serving_station)',
                          'move_container (plate, from=serving_station, to=washing_station)',
                          'wash (plate, washing_station)',
                          'transfer (bowl, from=plating_station, to=washing_station)',
                          'wash (bowl, washing_station)'),
 'banana_strawberry_fruit_bowl': ('transfer (bowl, from=storage, to=prep_station)',
                                  'transfer (banana, from=storage, to=prep_station)',
                                  'transfer (strawberries, from=storage, to=prep_station)',
                                  'cut (banana, prep_station)',
                                  'load (banana, bowl, prep_station)',
                                  'cut (strawberries, prep_station)',
                                  'load (strawberries, bowl, prep_station)',
                                  'combine (bowl, prep_station)',
                                  'transfer (plate, from=storage, to=plating_station)',
                                  'move_container (bowl, from=prep_station, to=plating_station)',
                                  'unload (mixture, bowl, plating_station)',
                                  'load (mixture, plate, plating_station)',
                                  'move_container (plate, from=plating_station, '
                                  'to=serving_station)',
                                  'serve (plate, serving_station)',
                                  'move_container (plate, from=serving_station, '
                                  'to=washing_station)',
                                  'wash (plate, washing_station)',
                                  'transfer (bowl, from=plating_station, to=washing_station)',
                                  'wash (bowl, washing_station)'),
 'oil_tomato_salad': ('transfer (bowl, from=storage, to=prep_station)',
                      'transfer (measuring_cup, from=storage, to=prep_station)',
                      'transfer (oil, from=storage, to=prep_station)',
                      'load (oil, measuring_cup, prep_station)',
                      'pour (oil, measuring_cup, bowl, prep_station)',
                      'transfer (tomato, from=storage, to=prep_station)',
                      'cut (tomato, prep_station)',
                      'load (tomato, bowl, prep_station)',
                      'transfer (lettuce, from=storage, to=prep_station)',
                      'cut (lettuce, prep_station)',
                      'load (lettuce, bowl, prep_station)',
                      'combine (bowl, prep_station)',
                      'transfer (plate, from=storage, to=plating_station)',
                      'move_container (bowl, from=prep_station, to=plating_station)',
                      'unload (mixture, bowl, plating_station)',
                      'load (mixture, plate, plating_station)',
                      'move_container (plate, from=plating_station, to=serving_station)',
                      'serve (plate, serving_station)',
                      'transfer (measuring_cup, from=prep_station, to=washing_station)',
                      'wash (measuring_cup, washing_station)',
                      'transfer (bowl, from=plating_station, to=washing_station)',
                      'wash (bowl, washing_station)'),
 'scrambled_eggs_pan': ('transfer (pan, from=storage, to=cooking_station)',
                        'transfer (bowl, from=storage, to=prep_station)',
                        'transfer (egg, from=storage, to=prep_station)',
                        'season (egg, salt, prep_station)',
                        'load (egg, bowl, prep_station)',
                        'move_container (bowl, from=prep_station, to=cooking_station)',
                        'unload (egg, bowl, cooking_station)',
                        'load (egg, pan, cooking_station)',
                        'turn_on (stove, cooking_station)',
                        'cook_contents (pan, cooking_station)',
                        'turn_off (stove, cooking_station)',
                        'transfer (plate, from=storage, to=plating_station)',
                        'unload (egg, pan, cooking_station)',
                        'load (egg, bowl, cooking_station)',
                        'move_container (bowl, from=cooking_station, to=plating_station)',
                        'unload (egg, bowl, plating_station)',
                        'load (egg, plate, plating_station)',
                        'move_container (plate, from=plating_station, to=serving_station)',
                        'serve (plate, serving_station)',
                        'transfer (pan, from=cooking_station, to=washing_station)',
                        'wash (pan, washing_station)',
                        'transfer (bowl, from=plating_station, to=washing_station)',
                        'wash (bowl, washing_station)'),
 'mushroom_omelette': ('transfer (pan, from=storage, to=cooking_station)',
                       'transfer (bowl, from=storage, to=prep_station)',
                       'transfer (egg, from=storage, to=prep_station)',
                       'load (egg, bowl, prep_station)',
                       'transfer (mushroom, from=storage, to=prep_station)',
                       'cut (mushroom, prep_station)',
                       'load (mushroom, bowl, prep_station)',
                       'combine (bowl, prep_station)',
                       'move_container (bowl, from=prep_station, to=cooking_station)',
                       'unload (mixture, bowl, cooking_station)',
                       'load (mixture, pan, cooking_station)',
                       'turn_on (stove, cooking_station)',
                       'cook_contents (pan, cooking_station)',
                       'turn_off (stove, cooking_station)',
                       'transfer (plate, from=storage, to=plating_station)',
                       'unload (mixture, pan, cooking_station)',
                       'load (mixture, bowl, cooking_station)',
                       'move_container (bowl, from=cooking_station, to=plating_station)',
                       'unload (mixture, bowl, plating_station)',
                       'load (mixture, plate, plating_station)',
                       'move_container (plate, from=plating_station, to=serving_station)',
                       'serve (plate, serving_station)',
                       'transfer (pan, from=cooking_station, to=washing_station)',
                       'wash (pan, washing_station)',
                       'transfer (bowl, from=plating_station, to=washing_station)',
                       'wash (bowl, washing_station)'),
 'meat_mushroom_skillet': ('transfer (pan, from=storage, to=cooking_station)',
                           'load (meat, bowl, storage)',
                           'move_container (bowl, from=storage, to=cooking_station)',
                           'unload (meat, bowl, cooking_station)',
                           'load (meat, pan, cooking_station)',
                           'transfer (mushroom, from=storage, to=prep_station)',
                           'cut (mushroom, prep_station)',
                           'move_container (bowl, from=cooking_station, to=prep_station)',
                           'load (mushroom, bowl, prep_station)',
                           'move_container (bowl, from=prep_station, to=cooking_station)',
                           'unload (mushroom, bowl, cooking_station)',
                           'load (mushroom, pan, cooking_station)',
                           'season_container (pan, spice1, cooking_station)',
                           'turn_on (stove, cooking_station)',
                           'cook_contents (pan, cooking_station)',
                           'turn_off (stove, cooking_station)',
                           'transfer (plate, from=storage, to=plating_station)',
                           'unload (meat, pan, cooking_station)',
                           'load (meat, bowl, cooking_station)',
                           'move_container (bowl, from=cooking_station, to=plating_station)',
                           'unload (meat, bowl, plating_station)',
                           'load (meat, plate, plating_station)',
                           'move_container (bowl, from=plating_station, to=cooking_station)',
                           'unload (mushroom, pan, cooking_station)',
                           'load (mushroom, bowl, cooking_station)',
                           'move_container (bowl, from=cooking_station, to=plating_station)',
                           'unload (mushroom, bowl, plating_station)',
                           'load (mushroom, plate, plating_station)',
                           'move_container (plate, from=plating_station, to=serving_station)',
                           'serve (plate, serving_station)',
                           'transfer (pan, from=cooking_station, to=washing_station)',
                           'wash (pan, washing_station)',
                           'transfer (bowl, from=plating_station, to=washing_station)',
                           'wash (bowl, washing_station)'),
 'garlic_chicken_salad': ('transfer (pan, from=storage, to=cooking_station)',
                          'transfer (bowl, from=storage, to=prep_station)',
                          'transfer (chicken, from=storage, to=prep_station)',
                          'season (chicken, garlic, prep_station)',
                          'season (chicken, spice1, prep_station)',
                          'load (chicken, bowl, prep_station)',
                          'move_container (bowl, from=prep_station, to=cooking_station)',
                          'unload (chicken, bowl, cooking_station)',
                          'load (chicken, pan, cooking_station)',
                          'turn_on (stove, cooking_station)',
                          'cook_contents (pan, cooking_station)',
                          'turn_off (stove, cooking_station)',
                          'transfer (plate, from=storage, to=plating_station)',
                          'unload (chicken, pan, cooking_station)',
                          'load (chicken, bowl, cooking_station)',
                          'move_container (bowl, from=cooking_station, to=plating_station)',
                          'unload (chicken, bowl, plating_station)',
                          'load (chicken, plate, plating_station)',
                          'transfer (lettuce, from=storage, to=prep_station)',
                          'move_container (bowl, from=plating_station, to=prep_station)',
                          'cut (lettuce, prep_station)',
                          'load (lettuce, bowl, prep_station)',
                          'move_container (bowl, from=prep_station, to=plating_station)',
                          'unload (lettuce, bowl, plating_station)',
                          'load (lettuce, plate, plating_station)',
                          'move_container (plate, from=plating_station, to=serving_station)',
                          'serve (plate, serving_station)',
                          'transfer (pan, from=cooking_station, to=washing_station)',
                          'wash (pan, washing_station)',
                          'transfer (bowl, from=plating_station, to=washing_station)',
                          'wash (bowl, washing_station)'),
 'chicken_rice_plate': ('transfer (pot, from=storage, to=cooking_station)',
                        'transfer (pan, from=storage, to=cooking_station)',
                        'load (rice, bowl, storage)',
                        'move_container (bowl, from=storage, to=cooking_station)',
                        'unload (rice, bowl, cooking_station)',
                        'load (rice, pot, cooking_station)',
                        'transfer (chicken, from=storage, to=prep_station)',
                        'season (chicken, salt, prep_station)',
                        'transfer (chicken, from=prep_station, to=cooking_station)',
                        'load (chicken, pan, cooking_station)',
                        'turn_on (stove, cooking_station)',
                        'cook_contents (pot, cooking_station)',
                        'cook_contents (pan, cooking_station)',
                        'turn_off (stove, cooking_station)',
                        'transfer (plate, from=storage, to=plating_station)',
                        'unload (rice, pot, cooking_station)',
                        'load (rice, bowl, cooking_station)',
                        'move_container (bowl, from=cooking_station, to=plating_station)',
                        'unload (rice, bowl, plating_station)',
                        'load (rice, plate, plating_station)',
                        'move_container (bowl, from=plating_station, to=cooking_station)',
                        'unload (chicken, pan, cooking_station)',
                        'load (chicken, bowl, cooking_station)',
                        'move_container (bowl, from=cooking_station, to=plating_station)',
                        'unload (chicken, bowl, plating_station)',
                        'load (chicken, plate, plating_station)',
                        'move_container (plate, from=plating_station, to=serving_station)',
                        'serve (plate, serving_station)',
                        'transfer (pot, from=cooking_station, to=washing_station)',
                        'wash (pot, washing_station)',
                        'transfer (pan, from=cooking_station, to=washing_station)',
                        'wash (pan, washing_station)',
                        'transfer (bowl, from=plating_station, to=washing_station)',
                        'wash (bowl, washing_station)'),
 'fish_rice_plate': ('transfer (pot, from=storage, to=cooking_station)',
                     'transfer (pan, from=storage, to=cooking_station)',
                     'load (rice, bowl, storage)',
                     'move_container (bowl, from=storage, to=cooking_station)',
                     'unload (rice, bowl, cooking_station)',
                     'load (rice, pot, cooking_station)',
                     'transfer (fish, from=storage, to=prep_station)',
                     'season (fish, garlic, prep_station)',
                     'transfer (fish, from=prep_station, to=cooking_station)',
                     'load (fish, pan, cooking_station)',
                     'turn_on (stove, cooking_station)',
                     'cook_contents (pot, cooking_station)',
                     'cook_contents (pan, cooking_station)',
                     'turn_off (stove, cooking_station)',
                     'transfer (plate, from=storage, to=plating_station)',
                     'unload (rice, pot, cooking_station)',
                     'load (rice, bowl, cooking_station)',
                     'move_container (bowl, from=cooking_station, to=plating_station)',
                     'unload (rice, bowl, plating_station)',
                     'load (rice, plate, plating_station)',
                     'move_container (bowl, from=plating_station, to=cooking_station)',
                     'unload (fish, pan, cooking_station)',
                     'load (fish, bowl, cooking_station)',
                     'move_container (bowl, from=cooking_station, to=plating_station)',
                     'unload (fish, bowl, plating_station)',
                     'load (fish, plate, plating_station)',
                     'move_container (plate, from=plating_station, to=serving_station)',
                     'serve (plate, serving_station)',
                     'transfer (pot, from=cooking_station, to=washing_station)',
                     'wash (pot, washing_station)',
                     'transfer (pan, from=cooking_station, to=washing_station)',
                     'wash (pan, washing_station)',
                     'transfer (bowl, from=plating_station, to=washing_station)',
                     'wash (bowl, washing_station)'),
 'yoghurt_fruit_bowl': ('transfer (bowl, from=storage, to=prep_station)',
                        'transfer (banana, from=storage, to=prep_station)',
                        'transfer (strawberries, from=storage, to=prep_station)',
                        'transfer (yoghurt, from=storage, to=prep_station)',
                        'cut (banana, prep_station)',
                        'load (banana, bowl, prep_station)',
                        'cut (strawberries, prep_station)',
                        'load (strawberries, bowl, prep_station)',
                        'load (yoghurt, bowl, prep_station)',
                        'combine (bowl, prep_station)',
                        'transfer (plate, from=storage, to=plating_station)',
                        'move_container (bowl, from=prep_station, to=plating_station)',
                        'unload (mixture, bowl, plating_station)',
                        'load (mixture, plate, plating_station)',
                        'move_container (plate, from=plating_station, to=serving_station)',
                        'serve (plate, serving_station)',
                        'move_container (plate, from=serving_station, to=washing_station)',
                        'wash (plate, washing_station)',
                        'transfer (bowl, from=plating_station, to=washing_station)',
                        'wash (bowl, washing_station)'),
 'rice_mushroom_bowl': ('transfer (pot, from=storage, to=cooking_station)',
                        'transfer (bowl, from=storage, to=prep_station)',
                        'transfer (rice, from=storage, to=prep_station)',
                        'load (rice, bowl, prep_station)',
                        'transfer (mushroom, from=storage, to=prep_station)',
                        'cut (mushroom, prep_station)',
                        'load (mushroom, bowl, prep_station)',
                        'combine (bowl, prep_station)',
                        'season_container (bowl, spice2, prep_station)',
                        'move_container (bowl, from=prep_station, to=cooking_station)',
                        'unload (mixture, bowl, cooking_station)',
                        'load (mixture, pot, cooking_station)',
                        'turn_on (stove, cooking_station)',
                        'cook_contents (pot, cooking_station)',
                        'turn_off (stove, cooking_station)',
                        'transfer (plate, from=storage, to=plating_station)',
                        'move_container (pot, from=cooking_station, to=plating_station)',
                        'unload (mixture, pot, plating_station)',
                        'load (mixture, plate, plating_station)',
                        'move_container (plate, from=plating_station, to=serving_station)',
                        'serve (plate, serving_station)',
                        'move_container (pot, from=plating_station, to=washing_station)',
                        'wash (pot, washing_station)',
                        'transfer (bowl, from=cooking_station, to=washing_station)',
                        'wash (bowl, washing_station)')}


def recipe_builders():
    """Return ordered builders that create fresh action lists."""
    return {name: (lambda actions=actions: list(actions)) for name, actions in RECIPES.items()}
