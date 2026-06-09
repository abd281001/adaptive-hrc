"""Kitchen-domain environment: vocabulary, state tracker, and recipe library.

This module owns the *symbolic* side of the HRC benchmark:
  * A fixed vocabulary (containers, ingredients, locations, appliances, etc.) from which every feature name and every action symbol is derived.
  * `StateTracker` — a PDDL-style world model that maintains a fixed-width binary feature vector and applies parsed action strings with precondition checks.
  * `RecipeGenerator` — hand-written reference recipes used as teacher demonstrations.

The feature-vector layout is built exactly once at class level (see `StateTracker._build_feature_map`) and exported as the module-level `_FEAT` dict so state-feature learners can index raw states without reconstructing a tracker.
"""
import numpy as np

#  Kitchen-domain vocabulary
# Every feature name, every action argument, and every preference hook in the whole codebase is derived from these lists. Adding a new ingredient or location here automatically enlarges the raw state vector (and therefore the IRL feature space) in a consistent way.

CONTAINERS         = ["pot", "pan", "plate", "bowl", "glass", "measuring_cup"]
LIQUID_INGREDIENTS = ["milk", "oil"]
SOLID_INGREDIENTS  = [
    "tomato", "garlic", "onion", "mushroom", "lettuce",
    "cheese", "rice", "yoghurt", "strawberries", "banana",
    "egg", "fish", "chicken", "meat",
    "salt", "spice1", "spice2", "mixture",
]
INGREDIENTS = LIQUID_INGREDIENTS + SOLID_INGREDIENTS
ITEMS       = CONTAINERS + INGREDIENTS

# Per-operation subsets — used both for precondition checks in `apply_action` and for engineered-feature construction in `irl.py`.
CUTTABLES   = ["tomato", "onion", "mushroom", "lettuce", "banana", "strawberries", "chicken", "fish", "cheese"]
GRATABLE    = ["cheese"]
COOKABLES   = ["meat", "egg", "rice", "tomato", "onion", "mushroom", "chicken", "fish", "mixture"]
SEASONINGS  = ["salt", "spice1", "spice2", "garlic"]

LOCATIONS   = ["storage", "prep_station", "cooking_station", "plating_station", "serving_station", "washing_station", "blending_station"]


# ═════════════════════════════════════════════════════════════════════════════
#  StateTracker — fixed-width binary world state + action semantics
# ═════════════════════════════════════════════════════════════════════════════
class StateTracker:
    """PDDL-style kitchen simulator with a class-shared feature map. 
    Feature names are systematic e.g. `tomato_at_prep_station`, `pot_contains_rice`, `chicken_cooked`, `stove_on`, .. so the feature dimension is fully determined by the vocabulary constants above. This is what makes IRL warm-starting valid across retrains: the weight vector's length never changes.
    """

    # Shared across all instances — built lazily on first __init__.
    _CLASS_FEATURE_MAP = None

    @classmethod
    def _build_feature_map(cls):
        """Build the systematic feature layout once and memoize it."""
        if cls._CLASS_FEATURE_MAP is not None: return cls._CLASS_FEATURE_MAP

        fm, idx = {}, 0
        # Location features:   <item>_at_<location>
        for item in ITEMS:
            for loc in LOCATIONS:           fm[f"{item}_at_{loc}"] = idx; idx += 1
        # Containment:         <container>_contains_<ingredient>
        for container in CONTAINERS:
            for ingredient in INGREDIENTS:  fm[f"{container}_contains_{ingredient}"] = idx; idx += 1
        # Processing flags
        for item in CUTTABLES:              fm[f"{item}_cut"]     = idx; idx += 1
        for item in GRATABLE:               fm[f"{item}_grated"]  = idx; idx += 1
        for item in COOKABLES:              fm[f"{item}_cooked"]  = idx; idx += 1
        for item in INGREDIENTS:            fm[f"{item}_seasoned"]= idx; idx += 1
        for item in ITEMS:                  fm[f"{item}_washed"]  = idx; idx += 1
        # Global tool state and dish-delivered flag
        fm["stove_on"]     = idx; idx += 1
        fm["sink_on"]      = idx; idx += 1
        fm["blender_on"]   = idx; idx += 1
        fm["plate_served"] = idx; idx += 1
        # Mixture membership (excluding "mixture" itself)
        for ingredient in INGREDIENTS:
            if ingredient != "mixture":     fm[f"{ingredient}_in_mixture"] = idx; idx += 1
        # Seasoning crossed with ingredient (preserves which spice went where)
        for seasoning in SEASONINGS:
            for item in INGREDIENTS:        fm[f"{item}_seasoned_with_{seasoning}"] = idx; idx += 1

        cls._CLASS_FEATURE_MAP = fm
        return fm

    def __init__(self):
        self.feature_map = self.__class__._build_feature_map()
        self.n_features  = len(self.feature_map)
        self.reset()

    def reset(self):
        """Initial state: every real item lives in storage; no tool is on."""
        self.current_state = np.zeros(self.n_features, dtype=int)
        for item in ITEMS:
            if item != "mixture":   # "mixture" only exists once ingredients are combined
                self.set_feature(f"{item}_at_storage", 1)

    # raw feature accessors
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
            if self.get_feature(f"{item}_at_{loc}") == 1:
                return loc
        return None

    def is_contained(self, item):
        """Return the container holding `item`, or None if it is free."""
        for container in CONTAINERS:
            if self.get_feature(f"{container}_contains_{item}") == 1:
                return container
        return None

    # ─── action dispatch ─────────────────────────────────────────────────────
    def apply_action(self, action_str):
        """Parse and apply an action like ``transfer (tomato, from=storage, to=prep_station)``.

        Each branch enforces its own preconditions and raises ValueError on violation. Action strings come verbatim from `RecipeGenerator` demos, so precondition failures indicate a recipe bug, not a runtime user error.
        """
        action_str = action_str.strip()

        # Local parsers: `parts` = comma-split args inside `(...)`; `_loc` strips an optional "from="/"to=" prefix.
        def _parse(s):
            content = s[s.find("(") + 1 : s.find(")")]
            return [p.strip() for p in content.split(",")]

        def _loc(part):
            return part.split("=")[1] if "=" in part else part

        # ── transfer (item, from=L1, to=L2) — move a free item between locations
        if action_str.startswith("transfer"):
            parts    = _parse(action_str)
            item     = parts[0]
            from_loc = _loc(parts[1])
            to_loc   = _loc(parts[2])
            if self.get_feature(f"{item}_at_{from_loc}") != 1:          raise ValueError(f"Precondition failed: {item} not at {from_loc}")
            if self.is_contained(item):                                 raise ValueError(f"Precondition failed: {item} is contained")
            self.set_feature(f"{item}_at_{from_loc}", 0)
            self.set_feature(f"{item}_at_{to_loc}",   1)

        # ── load (item, container, location) — put item into a co-located container
        elif action_str.startswith("load"):
            parts     = _parse(action_str)
            item      = parts[0]; container = parts[1]; location = parts[2]
            if self.get_feature(f"{item}_at_{location}") != 1:          raise ValueError(f"Precondition failed: {item} not at {location}")
            if self.get_feature(f"{container}_at_{location}") != 1:     raise ValueError(f"Precondition failed: {container} not at {location}")
            if self.is_contained(item):                                 raise ValueError(f"Precondition failed: {item} already contained")
            self.set_feature(f"{item}_at_{location}", 0)
            self.set_feature(f"{container}_contains_{item}", 1)

        # ── unload (item, container, location) — take item out of container at L
        elif action_str.startswith("unload"):
            parts     = _parse(action_str)
            item      = parts[0]; container = parts[1]; location = parts[2]
            if self.get_feature(f"{container}_contains_{item}") != 1:   raise ValueError(f"Precondition failed: {item} not in {container}")
            if self.get_feature(f"{container}_at_{location}") != 1:     raise ValueError(f"Precondition failed: {container} not at {location}")
            self.set_feature(f"{container}_contains_{item}", 0)
            self.set_feature(f"{item}_at_{location}", 1)

        # ── move_container (container, from=L1, to=L2) — container carries its contents
        elif action_str.startswith("move_container"):
            parts     = _parse(action_str)
            container = parts[0]; from_loc = _loc(parts[1]); to_loc = _loc(parts[2])
            if self.get_feature(f"{container}_at_{from_loc}") != 1:     raise ValueError(f"Precondition failed: {container} not at {from_loc}")
            self.set_feature(f"{container}_at_{from_loc}", 0)
            self.set_feature(f"{container}_at_{to_loc}", 1)
            # Drag along every ingredient currently held by this container.
            for ingredient in INGREDIENTS:
                if self.get_feature(f"{container}_contains_{ingredient}") == 1:
                    self.set_feature(f"{ingredient}_at_{from_loc}", 0)
                    self.set_feature(f"{ingredient}_at_{to_loc}", 1)

        # ── cut (item, location=prep_station)
        elif action_str.startswith("cut"):
            parts    = _parse(action_str)
            item     = parts[0]
            location = parts[1] if len(parts) > 1 else "prep_station"
            if item not in CUTTABLES:                                   raise ValueError(f"Precondition failed: {item} is not cuttable")
            if self.get_feature(f"{item}_at_{location}") != 1:          raise ValueError(f"Precondition failed: {item} not at {location}")
            self.set_feature(f"{item}_cut", 1)

        # ── grate (item, location=prep_station)
        elif action_str.startswith("grate"):
            parts    = _parse(action_str)
            item     = parts[0]
            location = parts[1] if len(parts) > 1 else "prep_station"
            if item not in GRATABLE:                                    raise ValueError(f"Precondition failed: {item} is not gratable")
            if self.get_feature(f"{item}_at_{location}") != 1:          raise ValueError(f"Precondition failed: {item} not at {location}")
            self.set_feature(f"{item}_grated", 1)

        # ── cook (item, container=pot, location=cooking_station) — single-item cook. Both "cook " and "cook(" prefixes accepted so we don't collide with cook_contents.
        elif action_str.startswith("cook ") or action_str.startswith("cook("):
            parts     = _parse(action_str)
            item      = parts[0]
            container = parts[1] if len(parts) > 1 else "pot"
            location  = parts[2] if len(parts) > 2 else "cooking_station"
            if item not in COOKABLES:                                   raise ValueError(f"Precondition failed: {item} is not cookable")
            if self.get_feature(f"{container}_contains_{item}") != 1:   raise ValueError(f"Precondition failed: {item} not in {container}")
            if self.get_feature(f"{container}_at_{location}") != 1:     raise ValueError(f"Precondition failed: {container} not at {location}")
            if location == "cooking_station" and self.get_feature("stove_on") != 1:     raise ValueError("Precondition failed: stove not on")
            self.set_feature(f"{item}_cooked", 1)

        # ── cook_contents (container, location) — cook every cookable ingredient inside
        elif action_str.startswith("cook_contents"):
            parts     = _parse(action_str)
            container = parts[0]
            location  = parts[1] if len(parts) > 1 else "cooking_station"
            if self.get_feature(f"{container}_at_{location}") != 1:                     raise ValueError(f"Precondition failed: {container} not at {location}")
            if location == "cooking_station" and self.get_feature("stove_on") != 1:     raise ValueError("Precondition failed: stove not on")
            for ingredient in INGREDIENTS:
                if (self.get_feature(f"{container}_contains_{ingredient}") == 1 and ingredient in COOKABLES): self.set_feature(f"{ingredient}_cooked", 1)

        # ── combine (container, location?) — merge ≥2 ingredients into "mixture"
        elif action_str.startswith("combine"):
            parts     = _parse(action_str)
            container = parts[0]
            location  = parts[1] if len(parts) > 1 else None
            if location and self.get_feature(f"{container}_at_{location}") != 1:        raise ValueError(f"Precondition failed: {container} not at {location}")
            contained = [ing for ing in INGREDIENTS if ing != "mixture" and self.get_feature(f"{container}_contains_{ing}") == 1]
            if len(contained) < 2:                                                      raise ValueError(f"combine requires >=2 ingredients in {container}")
            for ing in contained:
                self.set_feature(f"{container}_contains_{ing}", 0)
                self.set_feature(f"{ing}_in_mixture", 1)
            self.set_feature(f"{container}_contains_mixture", 1)
            if location: self.set_feature(f"mixture_at_{location}", 1)

        # ── season_container (container, seasoning, location?) — apply seasoning to every ingredient currently inside the container.
        elif action_str.startswith("season_container"):
            parts     = _parse(action_str)
            container = parts[0]; seasoning = parts[1]
            location  = parts[2] if len(parts) > 2 else None
            if seasoning not in SEASONINGS:                                             raise ValueError(f"{seasoning} is not a seasoning")
            if location and self.get_feature(f"{container}_at_{location}") != 1:        raise ValueError(f"{container} not at {location}")
            for ing in INGREDIENTS:
                if self.get_feature(f"{container}_contains_{ing}") == 1:
                    self.set_feature(f"{ing}_seasoned", 1)
                    self.set_feature(f"{ing}_seasoned_with_{seasoning}", 1)

        # ── season (target, seasoning, location?) — single-item seasoning
        elif action_str.startswith("season"):
            parts     = _parse(action_str)
            target    = parts[0]; seasoning = parts[1]
            location  = parts[2] if len(parts) > 2 else None
            if seasoning not in SEASONINGS:                                             raise ValueError(f"{seasoning} is not a seasoning")
            if location and self.get_feature(f"{target}_at_{location}") != 1:           raise ValueError(f"{target} not at {location}")
            self.set_feature(f"{target}_seasoned", 1)
            self.set_feature(f"{target}_seasoned_with_{seasoning}", 1)

        # ── pour (liquid, from_container, to_container, location?)
        elif action_str.startswith("pour"):
            parts    = _parse(action_str)
            liquid   = parts[0]; from_c = parts[1]; to_c = parts[2]
            location = parts[3] if len(parts) > 3 else None
            if liquid not in LIQUID_INGREDIENTS:                                        raise ValueError(f"{liquid} is not a liquid")
            if self.get_feature(f"{from_c}_contains_{liquid}") != 1:                    raise ValueError(f"{liquid} not in {from_c}")
            if location:
                if self.get_feature(f"{from_c}_at_{location}") != 1:                    raise ValueError(f"{from_c} not at {location}")
                if self.get_feature(f"{to_c}_at_{location}") != 1:                      raise ValueError(f"{to_c} not at {location}")
            self.set_feature(f"{from_c}_contains_{liquid}", 0)
            self.set_feature(f"{to_c}_contains_{liquid}", 1)
            if location:    self.set_feature(f"{liquid}_at_{location}", 1)

        # ── turn_on / turn_off (tool)
        elif action_str.startswith("turn_on"):
            tool = _parse(action_str)[0]
            if   tool == "stove":   self.set_feature("stove_on",   1)
            elif tool == "sink":    self.set_feature("sink_on",    1)
            elif tool == "blender": self.set_feature("blender_on", 1)
        elif action_str.startswith("turn_off"):
            tool = _parse(action_str)[0]
            if   tool == "stove":   self.set_feature("stove_on",   0)
            elif tool == "sink":    self.set_feature("sink_on",    0)
            elif tool == "blender": self.set_feature("blender_on", 0)

        # ── blend (container, location=blending_station) — liquefy ≥1 ingredient
        elif action_str.startswith("blend"):
            parts     = _parse(action_str)
            container = parts[0]
            location  = parts[1] if len(parts) > 1 else "blending_station"
            if self.get_feature(f"{container}_at_{location}") != 1:                     raise ValueError(f"{container} not at {location}")
            if self.get_feature("blender_on") != 1:                                     raise ValueError("blender not on")
            contained = [ing for ing in INGREDIENTS if ing != "mixture" and self.get_feature(f"{container}_contains_{ing}") == 1]
            if len(contained) < 1:                                                      raise ValueError(f"blend requires >=1 ingredient in {container}")
            for ing in contained:
                self.set_feature(f"{container}_contains_{ing}", 0)
                self.set_feature(f"{ing}_in_mixture", 1)
            self.set_feature(f"{container}_contains_mixture", 1)
            self.set_feature(f"mixture_at_{location}", 1)

        # ── serve (plate, location=serving_station) — flips the dish-delivered flag
        elif action_str.startswith("serve"):
            parts    = _parse(action_str)
            plate    = parts[0]
            location = parts[1] if len(parts) > 1 else "serving_station"
            if self.get_feature(f"{plate}_at_{location}") != 1:                         raise ValueError(f"{plate} not at {location}")
            self.set_feature("plate_served", 1)

        # ── wash (item, location=washing_station)
        elif action_str.startswith("wash"):
            parts    = _parse(action_str)
            item     = parts[0]
            location = parts[1] if len(parts) > 1 else "washing_station"
            if self.get_feature(f"{item}_at_{location}") != 1:                          raise ValueError(f"{item} not at {location}")
            self.set_feature(f"{item}_washed", 1)

        else:                                                                           raise ValueError(f"Unknown action: {action_str}")


# Exported module-level feature map `irl.py` reads columns through this dict, avoiding a throw-away StateTracker allocation per lookup.
_FEAT = StateTracker._build_feature_map()


def validate_ordering(actions):
    """Return True if every action in `actions` applies cleanly to a fresh StateTracker. Used by the preference modifier to reject reorderings that violate kitchen preconditions before they ever enter the dataset preserving the invariant that no variant with a broken precondition chain can reach the learners.
    """
    tracker = StateTracker()
    for a in actions:
        try:            tracker.apply_action(a)
        except ValueError:  return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  RecipeGenerator — hand-written "teacher" demonstrations
# ═════════════════════════════════════════════════════════════════════════════
class RecipeGenerator:
    """Library of reference recipes."""

    # Hand-written recipe generators. These action sequences are the "ground-truth" demonstrations the IRL learner sees; preferences can transform them into reordered variants.
    def generate_grilled_steak(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "load (meat, bowl, storage)",
        "move_container (bowl, from=storage, to=cooking_station)",
        "unload (meat, bowl, cooking_station)",
        "load (meat, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (meat, pan, cooking_station)",
        "load (meat, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (meat, bowl, plating_station)",
        "load (meat, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_boiled_eggs(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "load (egg, bowl, storage)",
        "move_container (bowl, from=storage, to=cooking_station)",
        "unload (egg, bowl, cooking_station)",
        "load (egg, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (pot, from=cooking_station, to=plating_station)",
        "unload (egg, pot, plating_station)",
        "load (egg, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pot, from=plating_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=cooking_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_boiled_rice(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "load (rice, bowl, storage)",
        "move_container (bowl, from=storage, to=cooking_station)",
        "unload (rice, bowl, cooking_station)",
        "turn_on (stove, cooking_station)",
        "load (rice, pot, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (rice, pot, cooking_station)",
        "load (rice, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (rice, bowl, plating_station)",
        "load (rice, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pot, from=cooking_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_simple_salad(self): return [
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (lettuce, from=storage, to=prep_station)",
        "transfer (onion, from=storage, to=prep_station)",
        "cut (lettuce, prep_station)",
        "load (lettuce, bowl, prep_station)",
        "cut (onion, prep_station)",
        "load (onion, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (plate, from=serving_station, to=washing_station)",
        "wash (plate, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_burger(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "load (meat, bowl, storage)",
        "move_container (bowl, from=storage, to=cooking_station)",
        "unload (meat, bowl, cooking_station)",
        "load (meat, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (meat, pan, cooking_station)",
        "load (meat, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (meat, bowl, plating_station)",
        "load (meat, plate, plating_station)",
        "transfer (lettuce, from=storage, to=prep_station)",
        "move_container (bowl, from=plating_station, to=prep_station)",
        "cut (lettuce, prep_station)",
        "load (lettuce, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (lettuce, bowl, plating_station)",
        "load (lettuce, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_tomato_soup(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "turn_on (stove, cooking_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "load (tomato, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (tomato, bowl, cooking_station)",
        "load (tomato, pot, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (tomato, pot, cooking_station)",
        "load (tomato, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (tomato, bowl, plating_station)",
        "load (tomato, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pot, from=cooking_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_tomato_onion_soup_v1(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "load (tomato, bowl, prep_station)",
        "transfer (onion, from=storage, to=prep_station)",
        "cut (onion, prep_station)",
        "load (onion, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mixture, bowl, cooking_station)",
        "load (mixture, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (pot, from=cooking_station, to=plating_station)",
        "unload (mixture, pot, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (pot, from=plating_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=cooking_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_mushroom_soup(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (mushroom, from=storage, to=prep_station)",
        "cut (mushroom, prep_station)",
        "load (mushroom, bowl, prep_station)",
        "transfer (onion, from=storage, to=prep_station)",
        "cut (onion, prep_station)",
        "load (onion, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mixture, bowl, cooking_station)",
        "load (mixture, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (pot, from=cooking_station, to=plating_station)",
        "unload (mixture, pot, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (pot, from=plating_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=cooking_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_seasoned_chicken(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (chicken, from=storage, to=prep_station)",
        "season (chicken, salt, prep_station)",
        "season (chicken, spice1, prep_station)",
        "load (chicken, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (chicken, bowl, cooking_station)",
        "load (chicken, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (chicken, pan, cooking_station)",
        "load (chicken, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (chicken, bowl, plating_station)",
        "load (chicken, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_garlic_fish(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (fish, from=storage, to=prep_station)",
        "season (fish, garlic, prep_station)",
        "season (fish, spice2, prep_station)",
        "load (fish, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (fish, bowl, cooking_station)",
        "load (fish, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (fish, pan, cooking_station)",
        "load (fish, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (fish, bowl, plating_station)",
        "load (fish, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_seasoned_mixture_soup(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "load (tomato, bowl, prep_station)",
        "transfer (onion, from=storage, to=prep_station)",
        "cut (onion, prep_station)",
        "load (onion, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "season_container (bowl, salt, prep_station)",
        "season_container (bowl, spice1, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mixture, bowl, cooking_station)",
        "load (mixture, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (pot, from=cooking_station, to=plating_station)",
        "unload (mixture, pot, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (pot, from=plating_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=cooking_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_grated_cheese_salad(self): return [
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (lettuce, from=storage, to=prep_station)",
        "transfer (cheese, from=storage, to=prep_station)",
        "cut (lettuce, prep_station)",
        "load (lettuce, bowl, prep_station)",
        "grate (cheese, prep_station)",
        "load (cheese, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (plate, from=serving_station, to=washing_station)",
        "wash (plate, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_smoothie(self): return [
        "transfer (banana, from=storage, to=prep_station)",
        "transfer (strawberries, from=storage, to=prep_station)",
        "cut (banana, prep_station)",
        "transfer (banana, from=prep_station, to=blending_station)",
        "cut (strawberries, prep_station)",
        "transfer (strawberries, from=prep_station, to=blending_station)",
        "transfer (milk, from=storage, to=blending_station)",
        "transfer (glass, from=storage, to=blending_station)",
        "transfer (measuring_cup, from=storage, to=blending_station)",
        "load (milk, measuring_cup, blending_station)",
        "pour (milk, measuring_cup, glass, blending_station)",
        "load (banana, glass, blending_station)",
        "load (strawberries, glass, blending_station)",
        "turn_on (blender, blending_station)",
        "blend (glass, blending_station)",
        "turn_off (blender, blending_station)",
        "transfer (glass, from=blending_station, to=serving_station)",
        "serve (glass, serving_station)",
        "transfer (measuring_cup, from=blending_station, to=washing_station)",
        "wash (measuring_cup, washing_station)"]

    def generate_yoghurt_smoothie(self): return [
        "transfer (banana, from=storage, to=prep_station)",
        "transfer (yoghurt, from=storage, to=blending_station)",
        "cut (banana, prep_station)",
        "transfer (banana, from=prep_station, to=blending_station)",
        "transfer (glass, from=storage, to=blending_station)",
        "load (yoghurt, glass, blending_station)",
        "transfer (milk, from=storage, to=blending_station)",
        "transfer (measuring_cup, from=storage, to=blending_station)",
        "load (milk, measuring_cup, blending_station)",
        "pour (milk, measuring_cup, glass, blending_station)",
        "load (banana, glass, blending_station)",
        "turn_on (blender, blending_station)",
        "blend (glass, blending_station)",
        "turn_off (blender, blending_station)",
        "transfer (glass, from=blending_station, to=serving_station)",
        "serve (glass, serving_station)",
        "transfer (measuring_cup, from=blending_station, to=washing_station)",
        "wash (measuring_cup, washing_station)"]



    def generate_tomato_garlic_soup(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "season (tomato, garlic, prep_station)",
        "load (tomato, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (tomato, bowl, cooking_station)",
        "load (tomato, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (tomato, pot, cooking_station)",
        "load (tomato, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (tomato, bowl, plating_station)",
        "load (tomato, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pot, from=cooking_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_mushroom_garlic_soup(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (mushroom, from=storage, to=prep_station)",
        "cut (mushroom, prep_station)",
        "season (mushroom, garlic, prep_station)",
        "load (mushroom, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mushroom, bowl, cooking_station)",
        "load (mushroom, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (mushroom, pot, cooking_station)",
        "load (mushroom, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (mushroom, bowl, plating_station)",
        "load (mushroom, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pot, from=cooking_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_tomato_mushroom_soup(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "load (tomato, bowl, prep_station)",
        "transfer (mushroom, from=storage, to=prep_station)",
        "cut (mushroom, prep_station)",
        "load (mushroom, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mixture, bowl, cooking_station)",
        "load (mixture, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (pot, from=cooking_station, to=plating_station)",
        "unload (mixture, pot, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (pot, from=plating_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=cooking_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_onion_rice_pot(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (rice, from=storage, to=prep_station)",
        "load (rice, bowl, prep_station)",
        "transfer (onion, from=storage, to=prep_station)",
        "cut (onion, prep_station)",
        "load (onion, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mixture, bowl, cooking_station)",
        "load (mixture, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (pot, from=cooking_station, to=plating_station)",
        "unload (mixture, pot, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (pot, from=plating_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=cooking_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_tomato_cheese_salad(self): return [
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "transfer (cheese, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "load (tomato, bowl, prep_station)",
        "grate (cheese, prep_station)",
        "load (cheese, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (plate, from=serving_station, to=washing_station)",
        "wash (plate, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_tomato_lettuce_salad(self): return [
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "transfer (lettuce, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "load (tomato, bowl, prep_station)",
        "cut (lettuce, prep_station)",
        "load (lettuce, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (plate, from=serving_station, to=washing_station)",
        "wash (plate, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_banana_strawberry_fruit_bowl(self): return [
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (banana, from=storage, to=prep_station)",
        "transfer (strawberries, from=storage, to=prep_station)",
        "cut (banana, prep_station)",
        "load (banana, bowl, prep_station)",
        "cut (strawberries, prep_station)",
        "load (strawberries, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (plate, from=serving_station, to=washing_station)",
        "wash (plate, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_oil_tomato_salad(self): return [
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (measuring_cup, from=storage, to=prep_station)",
        "transfer (oil, from=storage, to=prep_station)",
        "load (oil, measuring_cup, prep_station)",
        "pour (oil, measuring_cup, bowl, prep_station)",
        "transfer (tomato, from=storage, to=prep_station)",
        "cut (tomato, prep_station)",
        "load (tomato, bowl, prep_station)",
        "transfer (lettuce, from=storage, to=prep_station)",
        "cut (lettuce, prep_station)",
        "load (lettuce, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (measuring_cup, from=prep_station, to=washing_station)",
        "wash (measuring_cup, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_scrambled_eggs_pan(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (egg, from=storage, to=prep_station)",
        "season (egg, salt, prep_station)",
        "load (egg, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (egg, bowl, cooking_station)",
        "load (egg, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (egg, pan, cooking_station)",
        "load (egg, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (egg, bowl, plating_station)",
        "load (egg, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_mushroom_omelette(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (egg, from=storage, to=prep_station)",
        "load (egg, bowl, prep_station)",
        "transfer (mushroom, from=storage, to=prep_station)",
        "cut (mushroom, prep_station)",
        "load (mushroom, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mixture, bowl, cooking_station)",
        "load (mixture, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (mixture, pan, cooking_station)",
        "load (mixture, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_meat_mushroom_skillet(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "load (meat, bowl, storage)",
        "move_container (bowl, from=storage, to=cooking_station)",
        "unload (meat, bowl, cooking_station)",
        "load (meat, pan, cooking_station)",
        "transfer (mushroom, from=storage, to=prep_station)",
        "cut (mushroom, prep_station)",
        "move_container (bowl, from=cooking_station, to=prep_station)",
        "load (mushroom, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mushroom, bowl, cooking_station)",
        "load (mushroom, pan, cooking_station)",
        "season_container (pan, spice1, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (meat, pan, cooking_station)",
        "load (meat, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (meat, bowl, plating_station)",
        "load (meat, plate, plating_station)",
        "move_container (bowl, from=plating_station, to=cooking_station)",
        "unload (mushroom, pan, cooking_station)",
        "load (mushroom, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (mushroom, bowl, plating_station)",
        "load (mushroom, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_garlic_chicken_salad(self): return [
        "transfer (pan, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (chicken, from=storage, to=prep_station)",
        "season (chicken, garlic, prep_station)",
        "season (chicken, spice1, prep_station)",
        "load (chicken, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (chicken, bowl, cooking_station)",
        "load (chicken, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (chicken, pan, cooking_station)",
        "load (chicken, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (chicken, bowl, plating_station)",
        "load (chicken, plate, plating_station)",
        "transfer (lettuce, from=storage, to=prep_station)",
        "move_container (bowl, from=plating_station, to=prep_station)",
        "cut (lettuce, prep_station)",
        "load (lettuce, bowl, prep_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (lettuce, bowl, plating_station)",
        "load (lettuce, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_chicken_rice_plate(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (pan, from=storage, to=cooking_station)",
        "load (rice, bowl, storage)",
        "move_container (bowl, from=storage, to=cooking_station)",
        "unload (rice, bowl, cooking_station)",
        "load (rice, pot, cooking_station)",
        "transfer (chicken, from=storage, to=prep_station)",
        "season (chicken, salt, prep_station)",
        "transfer (chicken, from=prep_station, to=cooking_station)",
        "load (chicken, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (rice, pot, cooking_station)",
        "load (rice, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (rice, bowl, plating_station)",
        "load (rice, plate, plating_station)",
        "move_container (bowl, from=plating_station, to=cooking_station)",
        "unload (chicken, pan, cooking_station)",
        "load (chicken, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (chicken, bowl, plating_station)",
        "load (chicken, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pot, from=cooking_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_fish_rice_plate(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (pan, from=storage, to=cooking_station)",
        "load (rice, bowl, storage)",
        "move_container (bowl, from=storage, to=cooking_station)",
        "unload (rice, bowl, cooking_station)",
        "load (rice, pot, cooking_station)",
        "transfer (fish, from=storage, to=prep_station)",
        "season (fish, garlic, prep_station)",
        "transfer (fish, from=prep_station, to=cooking_station)",
        "load (fish, pan, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "cook_contents (pan, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "unload (rice, pot, cooking_station)",
        "load (rice, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (rice, bowl, plating_station)",
        "load (rice, plate, plating_station)",
        "move_container (bowl, from=plating_station, to=cooking_station)",
        "unload (fish, pan, cooking_station)",
        "load (fish, bowl, cooking_station)",
        "move_container (bowl, from=cooking_station, to=plating_station)",
        "unload (fish, bowl, plating_station)",
        "load (fish, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "transfer (pot, from=cooking_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (pan, from=cooking_station, to=washing_station)",
        "wash (pan, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_yoghurt_fruit_bowl(self): return [
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (banana, from=storage, to=prep_station)",
        "transfer (strawberries, from=storage, to=prep_station)",
        "transfer (yoghurt, from=storage, to=prep_station)",
        "cut (banana, prep_station)",
        "load (banana, bowl, prep_station)",
        "cut (strawberries, prep_station)",
        "load (strawberries, bowl, prep_station)",
        "load (yoghurt, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (bowl, from=prep_station, to=plating_station)",
        "unload (mixture, bowl, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (plate, from=serving_station, to=washing_station)",
        "wash (plate, washing_station)",
        "transfer (bowl, from=plating_station, to=washing_station)",
        "wash (bowl, washing_station)"]

    def generate_rice_mushroom_bowl(self): return [
        "transfer (pot, from=storage, to=cooking_station)",
        "transfer (bowl, from=storage, to=prep_station)",
        "transfer (rice, from=storage, to=prep_station)",
        "load (rice, bowl, prep_station)",
        "transfer (mushroom, from=storage, to=prep_station)",
        "cut (mushroom, prep_station)",
        "load (mushroom, bowl, prep_station)",
        "combine (bowl, prep_station)",
        "season_container (bowl, spice2, prep_station)",
        "move_container (bowl, from=prep_station, to=cooking_station)",
        "unload (mixture, bowl, cooking_station)",
        "load (mixture, pot, cooking_station)",
        "turn_on (stove, cooking_station)",
        "cook_contents (pot, cooking_station)",
        "turn_off (stove, cooking_station)",
        "transfer (plate, from=storage, to=plating_station)",
        "move_container (pot, from=cooking_station, to=plating_station)",
        "unload (mixture, pot, plating_station)",
        "load (mixture, plate, plating_station)",
        "move_container (plate, from=plating_station, to=serving_station)",
        "serve (plate, serving_station)",
        "move_container (pot, from=plating_station, to=washing_station)",
        "wash (pot, washing_station)",
        "transfer (bowl, from=cooking_station, to=washing_station)",
        "wash (bowl, washing_station)"]



    def recipe_library(self):
        """Ordered mapping of canonical recipe names to generator methods. Single source of truth for canonical recipe names used by the harness, evaluation code, and tests.
        """
        return {
            "tomato_onion_soup_v1": self.generate_tomato_onion_soup_v1,
            "tomato_soup": self.generate_tomato_soup,
            "mushroom_soup": self.generate_mushroom_soup,
            "seasoned_mixture_soup": self.generate_seasoned_mixture_soup,
            "grilled_steak": self.generate_grilled_steak,
            "burger": self.generate_burger,
            "seasoned_chicken": self.generate_seasoned_chicken,
            "garlic_fish": self.generate_garlic_fish,
            "simple_salad": self.generate_simple_salad,
            "grated_cheese_salad": self.generate_grated_cheese_salad,
            "smoothie": self.generate_smoothie,
            "yoghurt_smoothie": self.generate_yoghurt_smoothie,
            "boiled_eggs": self.generate_boiled_eggs,
            "boiled_rice": self.generate_boiled_rice,

            "tomato_garlic_soup": self.generate_tomato_garlic_soup,
            "mushroom_garlic_soup": self.generate_mushroom_garlic_soup,
            "tomato_mushroom_soup": self.generate_tomato_mushroom_soup,
            "onion_rice_pot": self.generate_onion_rice_pot,
            "tomato_cheese_salad": self.generate_tomato_cheese_salad,
            "tomato_lettuce_salad": self.generate_tomato_lettuce_salad,
            "banana_strawberry_fruit_bowl": self.generate_banana_strawberry_fruit_bowl,
            "oil_tomato_salad": self.generate_oil_tomato_salad,
            "scrambled_eggs_pan": self.generate_scrambled_eggs_pan,
            "mushroom_omelette": self.generate_mushroom_omelette,
            "meat_mushroom_skillet": self.generate_meat_mushroom_skillet,
            "garlic_chicken_salad": self.generate_garlic_chicken_salad,
            "chicken_rice_plate": self.generate_chicken_rice_plate,
            "fish_rice_plate": self.generate_fish_rice_plate,
            "yoghurt_fruit_bowl": self.generate_yoghurt_fruit_bowl,
            "rice_mushroom_bowl": self.generate_rice_mushroom_bowl,
        }
gen = RecipeGenerator()
