"""Authoritative recipes, task options, and event preferences.

The catalog is intentionally simulator-independent.  Physical executors map
each option to navigation and interaction; the learner only sees completion-
checked task progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, Dict, FrozenSet, Mapping, Tuple


@dataclass(frozen=True)
class ActionSpec:
    token: str
    role: str
    events: FrozenSet[str]
    requires: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    environment: str
    layout: str
    ingredients: Tuple[str, ...]
    actions: Tuple[ActionSpec, ...]
    upstream_dish: str
    compatibility_dynamics: bool = False

    @property
    def expected_deliveries(self) -> int:
        """Dishes an episode of this recipe must deliver."""
        return sum(
            1 for action in self.actions if "deliver_dish" in action.events
        )

    @property
    def action_tokens(self) -> Tuple[str, ...]:
        return tuple(action.token for action in self.actions)

    @property
    def action_by_token(self) -> Mapping[str, ActionSpec]:
        return {action.token: action for action in self.actions}

    @property
    def stratum(self) -> str:
        """Reporting stratum.

        Compatibility recipes are physically executed by wrapper-restored
        station transitions rather than by the pinned interaction handler, so
        they are never pooled with natively executed Burrito recipes.
        """
        if self.environment != "burrito":
            return self.environment
        return "burrito_compat" if self.compatibility_dynamics else "burrito_native"

    @property
    def structure_signature(self) -> Tuple[Any, ...]:
        """Role-level DAG shape, invariant to ingredient renaming.

        Two recipes with the same signature are isomorphic task graphs, so a
        transfer between them is a relabelling rather than a generalisation.
        The holdout generator uses this to refuse an isomorphic source/target.
        """
        by_token = self.action_by_token
        return tuple(sorted(
            (
                action.role,
                tuple(sorted(by_token[token].role for token in action.requires)),
            )
            for action in self.actions
        ))


@dataclass(frozen=True)
class PreferenceSpec:
    name: str
    target_event: str
    early: bool = True
    source: str = "talents_like"


def _action(
    token: str,
    role: str,
    *events: str,
    requires: Tuple[str, ...] = (),
) -> ActionSpec:
    return ActionSpec(
        token=token,
        role=role,
        events=frozenset(events),
        requires=frozenset(requires),
    )


def _overcooked_recipe(ingredients: Tuple[str, ...]) -> RecipeSpec:
    adds = []
    previous: Dict[str, str] = {}
    for ingredient in ingredients:
        occurrence = len([item for item in adds if item.token.startswith(f"ADD_{ingredient.upper()}_")]) + 1
        token = f"ADD_{ingredient.upper()}_{occurrence}"
        requirement = (previous[ingredient],) if ingredient in previous else ()
        adds.append(_action(
            token,
            "cook_ingredient",
            f"pickup_{ingredient}",
            f"pot_{ingredient}",
            requires=requirement,
        ))
        previous[ingredient] = token
    # Declaration order is the canonical strategy and the tie-break for every
    # preference policy.  STAGE_DISH precedes START_COOKING_SOUP so that both
    # "container first" and "container just in time" are expressible: the
    # former hoists the dish above the ingredients, the latter sinks it below
    # the pot.  Bundling the pot-start into the final ADD (as the first
    # integration did) hid this decision from the learner entirely.
    stage = _action(
        "STAGE_DISH", "stage_container", "stage_dish", "stage_container"
    )
    cook = _action(
        "START_COOKING_SOUP",
        "cook_soup",
        "start_cooking",
        requires=tuple(action.token for action in adds),
    )
    plate = _action(
        "PLATE_SOUP",
        "assemble",
        "plate_ingredients",
        requires=(cook.token, stage.token),
    )
    serve = _action(
        "SERVE_SOUP", "serve", "deliver_dish", requires=(plate.token,)
    )
    suffix = "_".join(ingredients)
    return RecipeSpec(
        recipe_id=f"overcooked_{suffix}",
        environment="overcooked",
        layout="cramped_room_tomato",
        ingredients=ingredients,
        actions=tuple(adds) + (stage, cook, plate, serve),
        upstream_dish="soup",
    )


def _burrito_recipe(
    recipe_id: str,
    ingredients: Tuple[str, ...],
    actions: Tuple[ActionSpec, ...],
    upstream_dish: str,
    *,
    layout: str,
    compatibility: bool = False,
) -> RecipeSpec:
    return RecipeSpec(
        recipe_id=recipe_id,
        environment="burrito",
        layout=layout,
        ingredients=ingredients,
        actions=actions,
        upstream_dish=upstream_dish,
        compatibility_dynamics=compatibility,
    )


def _catalog() -> Dict[str, RecipeSpec]:
    recipes: Dict[str, RecipeSpec] = {}
    # Publication tasks must contain at least two ingredients.  Single-item
    # onion/tomato orders are trivial chains with no meaningful preference
    # choice and substantially inflate action accuracy.
    for size in (2, 3):
        for ingredients in combinations_with_replacement(("onion", "tomato"), size):
            recipe = _overcooked_recipe(tuple(ingredients))
            recipes[recipe.recipe_id] = recipe

    fetch_meat = _action("FETCH_MEAT", "retrieve", "pickup_meat")
    cook_steak = _action(
        "COOK_STEAK", "cook_protein", "grill_protein", requires=(fetch_meat.token,)
    )
    fetch_chicken = _action("FETCH_CHICKEN", "retrieve", "pickup_chicken")
    boil_chicken = _action(
        "BOIL_CHICKEN", "cook_protein", "pot_chicken", requires=(fetch_chicken.token,)
    )
    fetch_onion = _action("FETCH_ONION", "retrieve", "pickup_onion")
    chop_onion = _action(
        "CHOP_ONION", "prepare", "chop_ingredients", requires=(fetch_onion.token,)
    )
    recipes["burrito_steak_onion"] = _burrito_recipe(
        "burrito_steak_onion", ("meat", "onion"),
        (fetch_meat, cook_steak, fetch_onion, chop_onion, _action(
            "ASSEMBLE_STEAK_ONION", "assemble", "plate_ingredients",
            requires=(cook_steak.token, chop_onion.token),
        ), _action(
            "SERVE_STEAK_ONION", "serve", "deliver_dish",
            requires=("ASSEMBLE_STEAK_ONION",),
        )),
        "steak_onion_dish", layout="free_roam", compatibility=True,
    )
    recipes["burrito_chicken_onion"] = _burrito_recipe(
        "burrito_chicken_onion", ("chicken", "onion"),
        (fetch_chicken, boil_chicken, fetch_onion, chop_onion, _action(
            "ASSEMBLE_CHICKEN_ONION", "assemble", "plate_ingredients",
            requires=(boil_chicken.token, chop_onion.token),
        ), _action(
            "SERVE_CHICKEN_ONION", "serve", "deliver_dish",
            requires=("ASSEMBLE_CHICKEN_ONION",),
        )),
        "boiled_chicken_onion_dish", layout="free_roam", compatibility=True,
    )

    for protein, raw, cooked, pickup_event in (
        ("steak", "meat", "chopped_steak", "pickup_meat"),
        ("mushroom", "mushroom", "fried_mushroom", "pickup_mushroom"),
    ):
        fetch = _action(
            f"FETCH_AND_STAGE_{protein.upper()}", "retrieve", pickup_event
        )
        prepare = _action(
            f"PREPARE_AND_STAGE_{protein.upper()}", "prepare",
            "chop_ingredients", requires=(fetch.token,),
        )
        grill = _action(
            f"START_COOKING_{protein.upper()}", "cook_protein",
            "grill_protein", requires=(prepare.token,),
        )
        rice = _action(
            "START_BOILING_RICE", "cook_starch", "pickup_rice", "pot_rice"
        )
        plate = _action(
            "STAGE_CLEAN_PLATE", "stage_container", "wash_plates",
            "stage_container",
        )
        # The pinned environment accepts rice, tortilla and the cooked protein
        # onto a plate in any of the six orders (every GRAB_TORTILLA /
        # GET_BOILED_RICE / GET_*_FROM_GRILLER ``name_in_hand`` list contains
        # each partial-plate name).  The first integration bundled that into
        # COLLECT_AND_STAGE_COOKED_RICE + ASSEMBLE_*, which discarded a real
        # three-way branch.  Keeping the three additions separate is what makes
        # the assembly-order preferences observable at all, and it moves
        # preference-discriminating decisions into the second half of the
        # episode instead of concentrating them on the opening move.
        plate_rice = _action(
            "PLATE_RICE", "collect", "plate_ingredients", "plate_rice",
            requires=(rice.token, plate.token),
        )
        plate_tortilla = _action(
            "PLATE_TORTILLA", "wrap", "plate_ingredients", "plate_tortilla",
            requires=(plate.token,),
        )
        plate_protein = _action(
            f"PLATE_{protein.upper()}", "assemble",
            "plate_ingredients", "plate_protein",
            requires=(grill.token, plate.token),
        )
        serve = _action(
            f"SERVE_{protein.upper()}_BURRITO", "serve", "deliver_dish",
            requires=(
                plate_rice.token, plate_tortilla.token, plate_protein.token,
            ),
        )
        recipe = _burrito_recipe(
            f"burrito_{protein}_burrito", (raw, "rice", "tortilla"),
            (
                fetch, prepare, grill, rice, plate,
                plate_rice, plate_tortilla, plate_protein, serve,
            ),
            f"{protein}_burrito_dish", layout="burrito_1-2_2p",
        )
        recipes[recipe.recipe_id] = recipe

    recipes["burrito_combo"] = _combo_recipe()
    return recipes


def _combo_recipe() -> RecipeSpec:
    """One steak burrito and one mushroom burrito in a single episode.

    ``burrito_1-2_2p`` ships this order list and has two pots and two grills.
    Two constraints from the pinned environment shape the graph:

    * The sink does not yield a second addressable ``clean_plate``, and the
      upstream grab primitives select their target by object name, so two
      same-named plates could not be told apart.  Assembly is serialised.
    * Cooked proteins and boiled rice burn (``warn_time`` 60 on an 80-tick
      cook).  A perishable started for the second burrito while the first is
      still being assembled is charcoal by the time it is needed, so the
      second burrito's grill and pot start only once the first is served.

    Non-perishable preparation -- fetching and chopping the second protein --
    stays concurrent throughout.  The result is a task graph that is not
    isomorphic to either single-burrito recipe, which is what lets the
    container-axis holdout test generalisation on the Burrito side rather than
    relabelling, and it spreads preference-discriminating decisions across both
    halves of an 18-step episode.
    """
    first, second = "steak", "mushroom"
    serve_first = f"SERVE_{first.upper()}_BURRITO"
    actions: list[ActionSpec] = [
        _action(f"FETCH_AND_STAGE_{first.upper()}", "retrieve", "pickup_meat"),
        _action(
            f"PREPARE_AND_STAGE_{first.upper()}", "prepare", "chop_ingredients",
            requires=(f"FETCH_AND_STAGE_{first.upper()}",),
        ),
        _action(
            f"START_COOKING_{first.upper()}", "cook_protein", "grill_protein",
            requires=(f"PREPARE_AND_STAGE_{first.upper()}",),
        ),
        _action(
            f"FETCH_AND_STAGE_{second.upper()}", "retrieve", "pickup_mushroom",
        ),
        _action(
            f"PREPARE_AND_STAGE_{second.upper()}", "prepare", "chop_ingredients",
            requires=(f"FETCH_AND_STAGE_{second.upper()}",),
        ),
        _action("START_BOILING_RICE", "cook_starch", "pickup_rice", "pot_rice"),
        _action(
            "STAGE_CLEAN_PLATE", "stage_container", "wash_plates",
            "stage_container",
        ),
        _action(
            "PLATE_RICE", "collect", "plate_ingredients", "plate_rice",
            requires=("START_BOILING_RICE", "STAGE_CLEAN_PLATE"),
        ),
        _action(
            "PLATE_TORTILLA", "wrap", "plate_ingredients", "plate_tortilla",
            requires=("STAGE_CLEAN_PLATE",),
        ),
        _action(
            f"PLATE_{first.upper()}", "assemble",
            "plate_ingredients", "plate_protein",
            requires=(f"START_COOKING_{first.upper()}", "STAGE_CLEAN_PLATE"),
        ),
        _action(
            serve_first, "serve", "deliver_dish",
            requires=("PLATE_RICE", "PLATE_TORTILLA", f"PLATE_{first.upper()}"),
        ),
        _action(
            f"START_COOKING_{second.upper()}", "cook_protein", "grill_protein",
            requires=(f"PREPARE_AND_STAGE_{second.upper()}", serve_first),
        ),
        _action(
            "START_BOILING_RICE_2", "cook_starch", "pickup_rice", "pot_rice",
            requires=(serve_first,),
        ),
        _action(
            "STAGE_CLEAN_PLATE_2", "stage_container", "wash_plates",
            "stage_container", requires=(serve_first,),
        ),
        _action(
            "PLATE_RICE_2", "collect", "plate_ingredients", "plate_rice",
            requires=("START_BOILING_RICE_2", "STAGE_CLEAN_PLATE_2"),
        ),
        _action(
            "PLATE_TORTILLA_2", "wrap", "plate_ingredients", "plate_tortilla",
            requires=("STAGE_CLEAN_PLATE_2",),
        ),
        _action(
            f"PLATE_{second.upper()}", "assemble",
            "plate_ingredients", "plate_protein",
            requires=(
                f"START_COOKING_{second.upper()}", "STAGE_CLEAN_PLATE_2",
            ),
        ),
        _action(
            f"SERVE_{second.upper()}_BURRITO", "serve", "deliver_dish",
            requires=(
                "PLATE_RICE_2", "PLATE_TORTILLA_2", f"PLATE_{second.upper()}",
            ),
        ),
    ]
    return _burrito_recipe(
        "burrito_combo", ("meat", "mushroom", "rice", "tortilla"),
        tuple(actions), "steak_and_mushroom_burrito_dishes",
        layout="burrito_1-2_2p",
    )


RECIPES: Mapping[str, RecipeSpec] = _catalog()
OVERCOOKED_RECIPE_IDS = tuple(
    recipe_id for recipe_id, recipe in RECIPES.items()
    if recipe.environment == "overcooked"
)
BURRITO_RECIPE_IDS = tuple(
    recipe_id for recipe_id, recipe in RECIPES.items()
    if recipe.environment == "burrito"
)

# Every preference below must produce an ordering that differs from the
# canonical (declaration-order) strategy in at least one recipe; the module
# asserts this at import.  Preferences whose target event is structurally
# pinned by the task DAG -- ``deliver_dish`` is always last, ``chop`` and
# ``grill`` always follow their own fetch -- are not declared, because a
# preference that can never change behaviour silently inflates the apparent
# size of the strategy space.
PREFERENCES: Mapping[str, PreferenceSpec] = {
    item.name: item for item in (
        PreferenceSpec("plate_ingredients_early", "plate_ingredients"),
        PreferenceSpec("wash_plates_early", "stage_container"),
        PreferenceSpec("pot_rice_early", "pot_rice"),
        PreferenceSpec("pickup_mushroom_early", "pickup_mushroom"),
        PreferenceSpec(
            "plate_tortilla_early", "plate_tortilla", source="assembly_order",
        ),
        PreferenceSpec(
            "plate_protein_early", "plate_protein", source="assembly_order",
        ),
        PreferenceSpec("tomato_early", "pot_tomato", source="overcooked_relevant"),
        PreferenceSpec(
            "container_just_in_time", "stage_container", early=False,
            source="cross_environment_relevant",
        ),
        PreferenceSpec(
            "pickup_onion_early", "pickup_onion", source="burrito_relevant",
        ),
    )
}
TALENTS_LIKE_PREFERENCES = tuple(
    name for name, item in PREFERENCES.items() if item.source == "talents_like"
)
OVERCOOKED_PREFERENCES = tuple(
    name for name, item in PREFERENCES.items()
    if item.source != "talents_like"
)
MAX_TASK_ACTIONS = max(len(recipe.actions) for recipe in RECIPES.values())
FUNCTIONAL_ROLES = tuple(sorted({
    action.role for recipe in RECIPES.values() for action in recipe.actions
}))


def get_recipe(recipe_id: str) -> RecipeSpec:
    try:
        return RECIPES[str(recipe_id)]
    except KeyError as error:
        raise ValueError(f"unknown cooking recipe {recipe_id!r}") from error


def get_preference(name: str) -> PreferenceSpec:
    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return PREFERENCES[normalized]
    except KeyError as error:
        raise ValueError(f"unknown cooking preference {name!r}") from error


def preference_order(recipe: RecipeSpec, preference: PreferenceSpec) -> Tuple[str, ...]:
    completed: list[str] = []
    by_token = recipe.action_by_token
    order_index = {action.token: index for index, action in enumerate(recipe.actions)}
    while len(completed) < len(recipe.actions):
        done = frozenset(completed)
        legal = [
            action for action in recipe.actions
            if action.token not in done and action.requires <= done
        ]
        if not legal:
            raise RuntimeError(f"cyclic task graph for {recipe.recipe_id}")
        if preference.early:
            key = lambda action: (
                preference.target_event not in action.events,
                order_index[action.token],
            )
        else:
            key = lambda action: (
                preference.target_event in action.events,
                order_index[action.token],
            )
        completed.append(min(legal, key=key).token)
    assert set(completed) == set(by_token)
    return tuple(completed)


def applicable_preferences(recipe_id: str) -> Tuple[str, ...]:
    """One representative for every distinct behavior in this recipe."""
    recipe = get_recipe(recipe_id)
    seen = set()
    applicable = []
    for name, preference in PREFERENCES.items():
        ordering = preference_order(recipe, preference)
        if ordering not in seen:
            seen.add(ordering)
            applicable.append(name)
    return tuple(applicable)


def realized_preferences() -> Tuple[str, ...]:
    """Declared preferences that change the ordering of at least one recipe."""
    realized: set[str] = set()
    for recipe_id in RECIPES:
        realized.update(applicable_preferences(recipe_id))
    return tuple(name for name in PREFERENCES if name in realized)


def _assert_every_preference_is_behavioural() -> None:
    """Refuse to import a catalog that advertises unreachable strategies.

    A preference whose greedy ordering coincides with the canonical ordering in
    every recipe is indistinguishable from the canonical strategy.  Declaring
    such a preference makes the strategy space look larger than it is, which is
    exactly how the first integration came to advertise nine TALENTS-like
    preferences while only three of them could ever be observed.
    """
    inert = tuple(
        name for name in PREFERENCES if name not in set(realized_preferences())
    )
    if inert:
        raise RuntimeError(
            "these declared preferences never change any recipe ordering and "
            f"would overstate the strategy space: {sorted(inert)}"
        )


_assert_every_preference_is_behavioural()

STRATA: Tuple[str, ...] = ("overcooked", "burrito_native", "burrito_compat")
RECIPE_IDS_BY_STRATUM: Mapping[str, Tuple[str, ...]] = {
    stratum: tuple(
        recipe_id for recipe_id, recipe in RECIPES.items()
        if recipe.stratum == stratum
    )
    for stratum in STRATA
}


def get_stratum(recipe_id: str) -> str:
    return get_recipe(recipe_id).stratum


__all__ = [
    "ActionSpec", "BURRITO_RECIPE_IDS", "FUNCTIONAL_ROLES",
    "RECIPE_IDS_BY_STRATUM", "STRATA", "get_stratum", "realized_preferences",
    "MAX_TASK_ACTIONS", "OVERCOOKED_PREFERENCES", "OVERCOOKED_RECIPE_IDS",
    "PREFERENCES", "RECIPES", "RecipeSpec", "TALENTS_LIKE_PREFERENCES",
    "applicable_preferences", "get_preference", "get_recipe",
    "preference_order",
]
