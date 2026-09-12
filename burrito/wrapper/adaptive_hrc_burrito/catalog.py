"""Authoritative recipes, task options, and event preferences.

The catalog is intentionally simulator-independent.  Physical executors map
each option to navigation and interaction; the learner only sees completion-
checked task progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import (
    AbstractSet, Any, Callable, Dict, FrozenSet, Mapping, Optional, Tuple,
)


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
    """One workflow preference over a recipe's task graph.

    Two shapes. A *priority* preference names one event and pulls it to the
    front (``early``) or pushes it to the back; it resolves the first time
    that event is legal, after which the ordering falls back to declaration
    order. A *precedence* preference names an ordered pair of events and only
    acts where both compete, so it cannot be identified from the opening move
    and stays unresolved deeper into the episode.
    """

    name: str
    target_event: Optional[str] = None
    early: bool = True
    before: Optional[Tuple[str, str]] = None
    source: str = "talents_like"

    def __post_init__(self) -> None:
        if (self.target_event is None) == (self.before is None):
            raise ValueError(
                f"{self.name}: give exactly one of target_event or before"
            )
        if self.before is not None and self.before[0] == self.before[1]:
            raise ValueError(f"{self.name}: before must name two distinct events")


def preference_sort_key(
    preference: PreferenceSpec,
    events_of: Callable[[str], AbstractSet[str]],
    order_index: Mapping[str, int],
) -> Callable[[str], Tuple[int, int]]:
    """The single definition of what a preference does to a legal frontier.

    Both the whole-ordering builder and the single-frontier policy call this,
    so they cannot drift apart: a disagreement between them would make the
    reference ordering and the discriminating-decision test describe
    different preferences.
    """
    if preference.before is not None:
        first, second = preference.before

        def rank(token: str) -> int:
            events = events_of(token)
            if first in events:
                return 0
            if second in events:
                return 2
            return 1

        return lambda token: (rank(token), order_index[token])
    target = preference.target_event
    if preference.early:
        return lambda token: (int(target not in events_of(token)), order_index[token])
    return lambda token: (int(target in events_of(token)), order_index[token])


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

    # The catalog's one multi-order episode, and the only task graph that is
    # not a relabelling of a single-dish one. Longer versions were built and
    # measured and are deliberately absent: a third order adds 50% more
    # decisions at a 13.6% contested rate against this recipe's 20.4%, because
    # what generates preference conflict is orderings per action, not episode
    # length. See _multi_order_recipe for what the builder supports.
    recipes["burrito_combo"] = _multi_order_recipe(
        "burrito_combo", ("steak", "mushroom"),
        "steak_and_mushroom_burrito_dishes",
    )
    return recipes


_PROTEIN_PICKUP = {"steak": "pickup_meat", "mushroom": "pickup_mushroom"}


def _multi_order_recipe(
    recipe_id: str,
    orders: Tuple[str, ...],
    dish: str,
) -> RecipeSpec:
    """One episode that fills several burrito orders in sequence.

    ``burrito_1-2_2p`` has two pots, two grills, four plates and an order list
    that alternates steak and mushroom burritos, so several orders in one
    episode are within what the pinned environment supplies. Two constraints
    from that environment shape every graph this builds:

    * The sink does not yield a second addressable ``clean_plate``, and the
      upstream grab primitives select their target by object name, so two
      same-named plates could not be told apart. Assembly is serialised: an
      order's plate is staged only once the previous order is served.
    * Cooked proteins and boiled rice burn (``warn_time`` 60 on an 80-tick
      cook). A perishable started for a later order while an earlier one is
      still being assembled is charcoal by the time it is needed, so grills
      and pots start only after the previous serve.

    Non-perishable preparation -- fetching and chopping a protein -- stays
    concurrent, but only while that protein is distinguishable: the fetch
    macro is illegal whenever a chopped or cooked instance of the same
    protein already exists, so a repeated protein's preparation serialises
    behind the previous order's serve. A repeated-protein episode is
    therefore a different task graph from an alternating one, not a
    relabelling of it.
    """
    # Protein tokens are suffixed by that protein's own occurrence; shared
    # resources (rice, plate) by the order they belong to. That keeps a
    # two-order alternating episode's tokens identical to the single-dish
    # recipes it is built from.
    protein_seen: Dict[str, int] = {}
    token: list[Dict[str, str]] = []
    for index, protein in enumerate(orders):
        upper = protein.upper()
        occurrence = protein_seen.get(protein, 0) + 1
        protein_seen[protein] = occurrence
        psuf = "" if occurrence == 1 else f"_{occurrence}"
        osuf = "" if index == 0 else f"_{index + 1}"
        token.append({
            "fetch": f"FETCH_AND_STAGE_{upper}{psuf}",
            "prepare": f"PREPARE_AND_STAGE_{upper}{psuf}",
            "cook": f"START_COOKING_{upper}{psuf}",
            "rice": f"START_BOILING_RICE{osuf}",
            "stage": f"STAGE_CLEAN_PLATE{osuf}",
            "plate_rice": f"PLATE_RICE{osuf}",
            "plate_tortilla": f"PLATE_TORTILLA{osuf}",
            "plate_protein": f"PLATE_{upper}{psuf}",
            "serve": f"SERVE_{upper}_BURRITO{psuf}",
        })

    # A later order's protein can be fetched and chopped concurrently only
    # while it is distinguishable: the fetch macro is illegal whenever a
    # chopped or cooked instance of the same protein already exists.
    concurrent = [
        index for index, protein in enumerate(orders)
        if index > 0 and protein not in orders[:index]
    ]

    def prep(index: int, gate: Tuple[str, ...]) -> list[ActionSpec]:
        n = token[index]
        return [
            _action(n["fetch"], "retrieve", _PROTEIN_PICKUP[orders[index]],
                    requires=gate),
            _action(n["prepare"], "prepare", "chop_ingredients",
                    requires=(n["fetch"],)),
        ]

    def cook(index: int, gate: Tuple[str, ...]) -> ActionSpec:
        n = token[index]
        return _action(n["cook"], "cook_protein", "grill_protein",
                       requires=(n["prepare"],) + gate)

    def assemble(index: int, gate: Tuple[str, ...]) -> list[ActionSpec]:
        n = token[index]
        return [
            _action(n["rice"], "cook_starch", "pickup_rice", "pot_rice",
                    requires=gate),
            _action(n["stage"], "stage_container", "wash_plates",
                    "stage_container", requires=gate),
            _action(n["plate_rice"], "collect", "plate_ingredients",
                    "plate_rice", requires=(n["rice"], n["stage"])),
            _action(n["plate_tortilla"], "wrap", "plate_ingredients",
                    "plate_tortilla", requires=(n["stage"],)),
            _action(n["plate_protein"], "assemble", "plate_ingredients",
                    "plate_protein", requires=(n["cook"], n["stage"])),
            _action(n["serve"], "serve", "deliver_dish",
                    requires=(n["plate_rice"], n["plate_tortilla"],
                              n["plate_protein"])),
        ]

    # Declaration order is the canonical strategy and every preference's
    # tie-break, so it is built deliberately: the first order's protein chain,
    # then the non-perishable preparation that can genuinely run alongside it,
    # then the first order's assembly, then each later order in turn.
    actions: list[ActionSpec] = [*prep(0, ()), cook(0, ())]
    for index in concurrent:
        actions.extend(prep(index, ()))
    actions.extend(assemble(0, ()))
    for index in range(1, len(orders)):
        gate = (token[index - 1]["serve"],)
        if index not in concurrent:
            actions.extend(prep(index, gate))
        actions.append(cook(index, gate))
        actions.extend(assemble(index, gate))
    ingredients = ("meat", "mushroom", "rice", "tortilla")
    return _burrito_recipe(
        recipe_id, ingredients, tuple(actions), dish, layout="burrito_1-2_2p",
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
        # Deferral counterparts. The language already supported `early=False`
        # and only one preference used it, which left every other preference
        # resolving at the first frontier that carries its event.
        PreferenceSpec("pot_rice_late", "pot_rice", early=False, source="deferral"),
        PreferenceSpec(
            "pickup_mushroom_late", "pickup_mushroom", early=False,
            source="deferral",
        ),
        PreferenceSpec(
            "plate_tortilla_late", "plate_tortilla", early=False, source="deferral",
        ),
        PreferenceSpec("tomato_late", "pot_tomato", early=False, source="deferral"),
        PreferenceSpec(
            "pickup_onion_late", "pickup_onion", early=False, source="deferral",
        ),
        # Precedence preferences. These only act where both named events
        # compete for the same frontier, so they are invisible in the opening
        # move and survive further into the episode than a priority can.
        PreferenceSpec(
            "prep_before_cooking", before=("chop_ingredients", "grill_protein"),
            source="workflow_precedence",
        ),
        PreferenceSpec(
            "protein_before_rice", before=("plate_protein", "plate_rice"),
            source="workflow_precedence",
        ),
        PreferenceSpec(
            "gather_before_staging", before=("pickup_tomato", "stage_container"),
            source="workflow_precedence",
        ),
        PreferenceSpec(
            "slow_items_first", before=("pot_rice", "chop_ingredients"),
            source="workflow_precedence",
        ),
        PreferenceSpec(
            "container_before_gathering", before=("stage_container", "pickup_onion"),
            source="workflow_precedence",
        ),
        # The native burrito recipes are the only ones with headroom left:
        # their protein, rice and plate chains are genuinely independent, so
        # 9 actions admit 337 orderings where the Overcooked recipes admit
        # 4-15. Every preference below discriminates between those chains,
        # which is what raises orderings-per-action -- the quantity that
        # drives how often the stored variants can disagree.
        PreferenceSpec(
            "chop_late", "chop_ingredients", early=False,
            source="burrito_workflow",
        ),
        PreferenceSpec(
            "plate_rice_late", "plate_rice", early=False,
            source="burrito_workflow",
        ),
        PreferenceSpec(
            "protein_before_starting_rice",
            before=("plate_protein", "pickup_rice"),
            source="burrito_workflow",
        ),
        PreferenceSpec(
            "tortilla_before_grilling",
            before=("plate_tortilla", "grill_protein"),
            source="burrito_workflow",
        ),
        PreferenceSpec(
            "rice_before_grilling", before=("pickup_rice", "grill_protein"),
            source="burrito_workflow",
        ),
        PreferenceSpec(
            "rice_before_tortilla", before=("pickup_rice", "plate_tortilla"),
            source="burrito_workflow",
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
        key = preference_sort_key(
            preference, lambda token: by_token[token].events, order_index,
        )
        completed.append(min((a.token for a in legal), key=key))
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
