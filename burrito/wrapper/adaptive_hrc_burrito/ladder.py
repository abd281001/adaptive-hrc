"""Publication longitudinal schedules for the cooking-domain evaluation.

The timing contract mirrors :mod:`src.evaluation`: seven macro phases, three
demonstrations per gap unit, a fixed 210-demonstration homogeneous budget, and
serialized heavy-tailed heterogeneous climbs. Cooking has fewer effective
preferences than the symbolic benchmark, so infeasible lifecycle operations
are sampled from the feasible subset and this conditioning is reported by the
schedule audit rather than silently retrying until an easy schedule appears.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .catalog import RECIPES, applicable_preferences
from .protocol import CookingTask


SCENARIOS = ("homogeneous", "heterogeneous", "holdout")
CANONICAL_STRATEGY = "canonical"
SUPPORT_FIRST_STRATEGY = "support_branch_first"
CONTAINER_FIRST_STRATEGY = "container_first"
# The preference that *is* the held-out container axis.  The ladder's own
# invariant checker keys on this name, and so do the evaluator's transfer
# groups, so it is named once here rather than spelled out in three places.
CONTAINER_FIRST_PREFERENCE = "wash_plates_early"
DEMOS_PER_GAP = 3
_SHARED_STRATEGIES = (
    CANONICAL_STRATEGY,
    SUPPORT_FIRST_STRATEGY,
    CONTAINER_FIRST_STRATEGY,
)
HOLDOUT_SOURCE_PHASES = 8
HOLDOUT_TARGET_PHASES = 6
HOLDOUT_PHASES = HOLDOUT_SOURCE_PHASES + HOLDOUT_TARGET_PHASES


@dataclass(frozen=True)
class LadderSettings:
    """The Adaptive-HRC publication schedule settings, without smoke knobs."""

    panel_size: int = 20
    phases: int = 7
    demos: int = 210
    min_recipes: int = 5
    max_recipes: int = 8
    transition_min: float = 0.20
    transition_max: float = 0.25
    gap_allocation: float = 0.80
    hetero_gap_mean: int = 15
    hetero_gap_min: int = 3
    hetero_gap_max: int = 40
    hetero_gap_shape: float = 0.35
    climb_decay: float = 0.50
    active_size_weights: Tuple[float, float, float] = (0.90, 0.08, 0.02)
    lifecycle_weights: Tuple[float, float, float, float] = (
        0.40, 0.20, 0.20, 0.20,
    )
    reentry_rate: float = 0.05
    recipe_skew: float = 0.70
    pair_skew: float = 0.70
    holdout_start: float = 0.60
    holdout_demos: int = 3

    def validate(self, recipe_count: int = len(RECIPES)) -> None:
        if self.phases < 3:
            raise ValueError("the publication ladder requires at least three phases")
        if self.demos <= 0 or self.demos % DEMOS_PER_GAP:
            raise ValueError("demos must be positive and divisible by three")
        if self.demos // DEMOS_PER_GAP < self.phases:
            raise ValueError("the fixed gap budget is smaller than the phase count")
        if not 1 <= self.min_recipes <= recipe_count:
            raise ValueError("invalid minimum recipes per phase")
        if self.max_recipes < self.min_recipes:
            raise ValueError("maximum recipes is smaller than minimum recipes")
        if not 0.20 <= self.transition_min <= self.transition_max <= 0.25:
            raise ValueError("transition fractions must lie in [0.20, 0.25]")
        if not (
            1 <= self.hetero_gap_min
            <= self.hetero_gap_mean
            <= self.hetero_gap_max
        ):
            raise ValueError("invalid heterogeneous gap bounds")
        if self.hetero_gap_shape <= 0.0 or self.gap_allocation <= 0.0:
            raise ValueError("gap allocation shapes must be positive")
        if not 0.0 < self.climb_decay < 1.0:
            raise ValueError("climb_decay must lie in (0, 1)")
        if len(self.active_size_weights) != 3 or any(
            value < 0.0 for value in self.active_size_weights
        ) or sum(self.active_size_weights) <= 0.0:
            raise ValueError("active_size_weights must contain positive mass")
        if len(self.lifecycle_weights) != 4 or any(
            value < 0.0 for value in self.lifecycle_weights
        ) or not math.isclose(sum(self.lifecycle_weights), 1.0, abs_tol=1e-9):
            raise ValueError("lifecycle weights must sum to one")
        if not 0.0 <= self.reentry_rate <= 1.0:
            raise ValueError("reentry_rate must lie in [0, 1]")
        if not 0.0 <= self.holdout_start <= 1.0:
            raise ValueError("holdout_start must lie in [0, 1]")
        if self.holdout_demos < 1:
            raise ValueError("holdout_demos must be positive")
        if self.recipe_skew <= 0.0 or self.pair_skew <= 0.0:
            raise ValueError("recipe and pair skew must be positive")


@dataclass
class _Lifecycle:
    active: set[str] = field(default_factory=set)
    ever: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)


@dataclass
class _Segment:
    index: int
    stage: int
    active: Dict[str, Tuple[str, ...]]
    climb_recipes: Tuple[str, ...]
    operation_by_recipe: Dict[str, str]
    role_by_pair: Dict[Tuple[str, str], str]
    removed_pairs: Tuple[Tuple[str, str], ...]
    strategy: str
    gap: int = 0
    climb_pairs: Tuple[Tuple[str, str], ...] = ()
    settled_pairs: Tuple[Tuple[str, str], ...] = ()


def _validate_recipes(recipe_ids: Sequence[str]) -> Tuple[str, ...]:
    declared = tuple(dict.fromkeys(map(str, recipe_ids)))
    unknown = set(declared) - set(RECIPES)
    if unknown:
        raise ValueError(f"unknown ladder recipes: {sorted(unknown)}")
    trivial = [
        recipe_id for recipe_id in declared
        if len(RECIPES[recipe_id].ingredients) < 2
    ]
    if trivial:
        raise ValueError(f"single-ingredient recipes are prohibited: {trivial}")
    return declared


def _weighted_sample(
    population: Sequence[str], weights: Mapping[str, float], count: int,
    rng: random.Random,
) -> list[str]:
    remaining = list(population)
    selected: list[str] = []
    while remaining and len(selected) < count:
        choice = rng.choices(
            remaining, weights=[weights[item] for item in remaining], k=1,
        )[0]
        selected.append(choice)
        remaining.remove(choice)
    return selected


def _panel(
    recipe_ids: Sequence[str], settings: LadderSettings, seed: int,
) -> Tuple[Tuple[str, ...], Dict[str, float]]:
    recipes = _validate_recipes(recipe_ids)
    profile_rng = random.Random(f"longitudinal|user_profile|seed={int(seed)}")
    raw = {
        recipe_id: max(1e-12, profile_rng.gammavariate(settings.recipe_skew, 1.0))
        for recipe_id in recipes
    }
    total = sum(raw.values())
    popularity = {recipe_id: value / total for recipe_id, value in raw.items()}
    chosen = _weighted_sample(
        recipes,
        popularity,
        min(len(recipes), max(1, settings.panel_size)),
        random.Random(f"longitudinal|recipe_panel|seed={int(seed)}"),
    )
    if len(chosen) >= 2:
        environments = {RECIPES[item].environment for item in chosen}
        for environment in {"overcooked", "burrito"} - environments:
            replacement = next(
                item for item in recipes if RECIPES[item].environment == environment
            )
            chosen[-1] = replacement
            chosen = list(dict.fromkeys(chosen))
    return tuple(chosen), popularity


def _canonical_preference(recipe_id: str) -> str:
    return applicable_preferences(recipe_id)[0]


def _strategy_preference(recipe_id: str, strategy: str) -> str:
    """Map a shared homogeneous strategy to one effective recipe ordering.

    A non-canonical strategy must ground to a non-canonical ordering.  Falling
    back to the canonical preference (as ``container_first`` previously did for
    every recipe without a plate-washing step) produces a phase that is
    labelled a strategy change while changing nothing, and inflates
    ``covered_strategy_count`` with labels that never altered behaviour.
    """
    choices = applicable_preferences(recipe_id)
    canonical = choices[0]
    if strategy == CANONICAL_STRATEGY:
        return canonical
    if strategy == CONTAINER_FIRST_STRATEGY:
        preferred = "wash_plates_early"
    elif strategy == SUPPORT_FIRST_STRATEGY:
        recipe = RECIPES[recipe_id]
        if recipe.environment == "overcooked":
            preferred = "tomato_early"
        elif recipe.compatibility_dynamics:
            preferred = "pickup_onion_early"
        else:
            preferred = "pot_rice_early"
    else:
        raise ValueError(f"unknown homogeneous strategy {strategy!r}")
    if preferred in choices:
        return preferred
    fallback = next((item for item in choices if item != canonical), None)
    if fallback is None:
        raise ValueError(
            f"{recipe_id} cannot express any non-canonical strategy; it must "
            "not participate in a strategy-varying ladder"
        )
    return fallback


def strategy_grounding(recipe_ids: Sequence[str]) -> Dict[str, Any]:
    """How each abstract strategy grounds, and where the grounding is not exact.

    ``exact`` groundings hit the strategy's own target event.  ``substituted``
    ones fall back to another non-canonical ordering because the recipe has no
    such event -- the phase still changes behaviour, but the label must not be
    read as evidence about that event.  ``aliased`` pairs are two strategies
    that ground to one ordering for a recipe, so the phases are behaviourally
    indistinguishable for it.
    """
    grounding: Dict[str, Dict[str, str]] = {}
    substituted: list[Tuple[str, str, str]] = []
    aliased: list[Tuple[str, str, str]] = []
    for recipe_id in recipe_ids:
        choices = applicable_preferences(recipe_id)
        by_strategy = {
            strategy: _strategy_preference(recipe_id, strategy)
            for strategy in _SHARED_STRATEGIES
        }
        grounding[recipe_id] = by_strategy
        for strategy, preference in by_strategy.items():
            if strategy == CANONICAL_STRATEGY:
                continue
            target = {
                CONTAINER_FIRST_STRATEGY: CONTAINER_FIRST_PREFERENCE,
            }.get(strategy)
            if target is not None and target not in choices:
                substituted.append((recipe_id, strategy, preference))
        for left in _SHARED_STRATEGIES:
            for right in _SHARED_STRATEGIES:
                if left < right and by_strategy[left] == by_strategy[right]:
                    aliased.append((recipe_id, left, right))
    return {
        "strategy_grounding": grounding,
        "substituted_strategy_groundings": sorted(substituted),
        "aliased_strategy_pairs": sorted(aliased),
        "degenerate_strategy_groundings": sorted(
            (recipe_id, strategy)
            for recipe_id, by_strategy in grounding.items()
            for strategy, preference in by_strategy.items()
            if strategy != CANONICAL_STRATEGY
            and preference == by_strategy[CANONICAL_STRATEGY]
        ),
    }


def _can_space(
    counts: Mapping[Tuple[str, str], int], *, minimum_gap: int = 4,
) -> bool:
    """Whether repeats can be held ``minimum_gap`` apart at all.

    A phase supporting only three distinct pairs cannot separate a repeat by
    four episodes however it is shuffled, so extra climb support must not be
    allocated there.  The shortest arrangement of the most frequent pair is
    ``(m - 1) * gap + k`` slots, where ``k`` is how many pairs share that
    frequency.
    """
    values = [count for count in counts.values() if count > 0]
    if not values:
        return True
    most = max(values)
    return (most - 1) * minimum_gap + sum(
        1 for count in values if count == most
    ) <= sum(values)


def _space_repeats(
    counts: Mapping[Tuple[str, str], int],
    rng: random.Random,
    *,
    minimum_gap: int = 4,
) -> list[Tuple[str, str]]:
    """Order repeated climb support so one pair never recurs within a gap.

    Shuffle-and-reject was used here, which fails to find a valid arrangement
    for feasible-but-tight inputs (it did so for one of the five fixed seeds as
    soon as the catalog grew).  Placing the pair with the most remaining
    support that is off cooldown is the standard optimal strategy for this
    constraint, so it succeeds whenever any arrangement exists, and randomness
    is retained in the tie-break.
    """
    remaining = Counter(dict(counts))
    ordered: list[Tuple[str, str]] = []
    last_position: Dict[Tuple[str, str], int] = {}
    while sum(remaining.values()):
        ready = [
            pair for pair, count in remaining.items()
            if count > 0
            and len(ordered) - last_position.get(pair, -minimum_gap) >= minimum_gap
        ]
        if not ready:
            raise RuntimeError(
                "homogeneous climb support cannot be separated by "
                f"{minimum_gap} episodes: {dict(remaining)}"
            )
        most = max(remaining[pair] for pair in ready)
        choice = rng.choice(sorted(pair for pair in ready if remaining[pair] == most))
        ordered.append(choice)
        remaining[choice] -= 1
        last_position[choice] = len(ordered) - 1
    return ordered


def _cardinality(
    settings: LadderSettings, maximum: int, rng: random.Random,
) -> int:
    sizes = list(range(1, min(3, maximum) + 1))
    return rng.choices(
        sizes, weights=settings.active_size_weights[:len(sizes)], k=1,
    )[0]


def _choose_operation(
    feasible: Sequence[str], settings: LadderSettings, rng: random.Random,
) -> str:
    names = ("retention", "addition", "removal", "swap")
    weights = dict(zip(names, settings.lifecycle_weights))
    return rng.choices(feasible, weights=[weights[name] for name in feasible], k=1)[0]


def _evolve(
    state: _Lifecycle,
    candidates: Sequence[str],
    settings: LadderSettings,
    rng: random.Random,
    *,
    ordered: bool = False,
) -> Tuple[set[str], Dict[str, set[str]], str, Tuple[str, ...]]:
    """Evolve an active set, conditioning only on operations that are possible."""
    candidate_order = list(dict.fromkeys(candidates))
    candidate_set = set(candidate_order)
    previous = set(state.active) & candidate_set
    if not previous:
        desired = _cardinality(settings, len(candidate_order), rng)
        active = set(
            candidate_order[:desired]
            if ordered else rng.sample(candidate_order, desired)
        )
        operation = "initialization"
        feasible = ("initialization",)
    else:
        inactive = candidate_set - previous
        fresh = inactive - state.ever
        returning = inactive & state.removed
        feasible_list = ["retention"]
        if inactive and len(previous) < min(3, len(candidate_set)):
            feasible_list.append("addition")
        if len(previous) > 1:
            feasible_list.append("removal")
        if inactive:
            feasible_list.append("swap")
        feasible = tuple(feasible_list)
        operation = _choose_operation(feasible, settings, rng)
        active = set(previous)

        def incoming() -> str:
            if ordered:
                return next(item for item in candidate_order if item in inactive)
            if returning and (not fresh or rng.random() < settings.reentry_rate):
                return rng.choice(sorted(returning))
            return rng.choice(sorted(fresh or inactive))

        if operation == "addition":
            active.add(incoming())
        elif operation == "removal":
            removable = [item for item in candidate_order if item in active]
            active.remove(removable[-1] if ordered else rng.choice(removable))
        elif operation == "swap":
            entrant = incoming()
            removable = [item for item in candidate_order if item in active]
            active.remove(removable[0] if ordered else rng.choice(removable))
            active.add(entrant)

    added = active - previous - state.removed
    reintroduced = (active - previous) & state.removed
    removed = previous - active
    retained = active & previous
    state.active = set(active)
    state.ever.update(active)
    state.removed.update(removed)
    state.removed.difference_update(active)
    return active, {
        "added": added,
        "reintroduced": reintroduced,
        "removed": removed,
        "retained": retained,
    }, operation, feasible


def _phase_recipes(
    panel: Sequence[str], popularity: Mapping[str, float], count: int,
    covered: set[str], rng: random.Random,
) -> Tuple[str, ...]:
    count = min(len(panel), count)
    uncovered = [item for item in panel if item not in covered]
    selected = _weighted_sample(uncovered, popularity, min(count, len(uncovered)), rng)
    selected.extend(_weighted_sample(
        [item for item in panel if item not in selected],
        popularity,
        count - len(selected),
        rng,
    ))
    rng.shuffle(selected)
    covered.update(selected)
    return tuple(selected)


def _allocate_budget(
    total: int, minimum: Sequence[int], shape: float, rng: random.Random,
    *, maximum: int | None = None,
) -> Tuple[int, ...]:
    allocation = [max(1, int(value)) for value in minimum]
    remaining = int(total) - sum(allocation)
    capacity = math.inf if maximum is None else sum(maximum - item for item in allocation)
    if remaining < 0 or remaining > capacity:
        raise ValueError("gap budget cannot satisfy realized active-pair support")
    weights = [max(1e-12, rng.gammavariate(shape, 1.0)) for _ in allocation]
    while remaining:
        eligible = [
            index for index, value in enumerate(allocation)
            if maximum is None or value < maximum
        ]
        selected = rng.choices(
            eligible, weights=[weights[index] for index in eligible], k=1,
        )[0]
        allocation[selected] += 1
        remaining -= 1
    return tuple(allocation)


def _settled_pairs(
    active_pairs: Sequence[Tuple[str, str]],
    count: int,
    popularity: Mapping[str, float],
    settings: LadderSettings,
    rng: random.Random,
    *,
    priority_pairs: Sequence[Tuple[str, str]] = (),
) -> Tuple[Tuple[str, str], ...]:
    if count < len(active_pairs):
        raise ValueError("settled support cannot cover every active pair")
    rows = list(active_pairs)
    priority = list(priority_pairs)
    while priority and sum(pair in priority for pair in rows) < settings.holdout_demos:
        if len(rows) >= count:
            raise ValueError("holdout support does not fit the settled budget")
        rows.append(rng.choice(priority))
    weights = {
        pair: max(
            1e-12,
            popularity[pair[0]] * rng.gammavariate(settings.pair_skew, 1.0),
        )
        for pair in active_pairs
    }
    rows.extend(rng.choices(
        list(active_pairs),
        weights=[weights[pair] for pair in active_pairs],
        k=count - len(rows),
    ))
    rng.shuffle(rows)
    return tuple(rows)


def _defer_climb_recurrence(draft: _Segment) -> None:
    """Keep acquisition and first reuse separated by ordinary episodes."""
    last_climb = {
        pair: index for index, pair in enumerate(draft.climb_pairs)
    }
    remaining = list(draft.settled_pairs)
    ordered: list[Tuple[str, str]] = []
    first_reuse: set[Tuple[str, str]] = set()
    while remaining:
        event_index = len(draft.climb_pairs) + len(ordered)
        allowed = [
            pair for pair in remaining
            if pair in first_reuse
            or pair not in last_climb
            or event_index - last_climb[pair] >= 4
        ]
        if not allowed:
            # This can only occur in the first serialized heterogeneous step,
            # before its sole recipe is known; no adaptation interval exists
            # yet, so spacing that introduction is unnecessary.
            allowed = [remaining[0]]
        pair = allowed[0]
        remaining.remove(pair)
        ordered.append(pair)
        first_reuse.add(pair)
    draft.settled_pairs = tuple(ordered)


def _role_map(
    active: Mapping[str, Sequence[str]],
    deltas: Mapping[str, Mapping[str, set[str]]],
) -> Dict[Tuple[str, str], str]:
    return {
        (recipe, preference): next((
            role for role in ("added", "reintroduced", "retained")
            if preference in deltas[recipe][role]
        ), "retained")
        for recipe, preferences in active.items()
        for preference in preferences
    }


def _build_homogeneous(
    panel: Sequence[str], popularity: Mapping[str, float],
    settings: LadderSettings, seed: int,
    _attempt: int = 0,
) -> Tuple[list[_Segment], Dict[str, Any]]:
    if _attempt >= 1000:
        raise RuntimeError("could not construct homogeneous climb budget")
    rng = random.Random(
        f"cooking|homogeneous|seed={int(seed)}|attempt={int(_attempt)}"
    )
    shared = _Lifecycle()
    state_by_recipe: Dict[str, set[str]] = defaultdict(set)
    covered: set[str] = set()
    drafts: list[_Segment] = []
    feasible_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for phase in range(settings.phases):
        strategies, _delta, operation, feasible = _evolve(
            shared, _SHARED_STRATEGIES, settings, rng, ordered=True,
        )
        feasible_counts.update(feasible)
        operation_counts[operation] += 1
        requested = rng.randint(
            min(settings.min_recipes, len(panel)),
            min(settings.max_recipes, len(panel)),
        )
        recipes = _phase_recipes(panel, popularity, requested, covered, rng)
        active: Dict[str, Tuple[str, ...]] = {}
        deltas: Dict[str, Dict[str, set[str]]] = {}
        operations: Dict[str, str] = {}
        removed_pairs: list[Tuple[str, str]] = []
        for recipe in recipes:
            preferences = tuple(dict.fromkeys(
                _strategy_preference(recipe, strategy)
                for strategy in _SHARED_STRATEGIES if strategy in strategies
            ))
            previous = state_by_recipe[recipe]
            current = set(preferences)
            entered = current - previous
            removed = previous - current
            deltas[recipe] = {
                "added": entered,
                "reintroduced": set(),
                "retained": current & previous,
                "removed": removed,
            }
            state_by_recipe[recipe] = current
            active[recipe] = preferences
            operations[recipe] = "initialization" if not previous else operation
            removed_pairs.extend((recipe, item) for item in removed)
        strategy_label = "homogeneous[" + "+".join(
            item for item in _SHARED_STRATEGIES if item in strategies
        ) + "]"
        drafts.append(_Segment(
            index=phase,
            stage=phase,
            active=active,
            climb_recipes=recipes if operation != "retention" else tuple(
                recipe for recipe in recipes if deltas[recipe]["added"]
            ),
            operation_by_recipe=operations,
            role_by_pair=_role_map(active, deltas),
            removed_pairs=tuple(removed_pairs),
            strategy=strategy_label,
        ))

    minimum = []
    for draft in drafts:
        active_pairs = [
            (recipe, preference)
            for recipe, preferences in draft.active.items()
            for preference in preferences
        ]
        climb = [pair for pair in active_pairs if pair[0] in draft.climb_recipes]
        draft.climb_pairs = tuple(climb)
    mandatory = sum(len(draft.climb_pairs) for draft in drafts)
    climb_minimum = math.ceil(settings.transition_min * settings.demos)
    climb_maximum = math.floor(settings.transition_max * settings.demos)
    if mandatory > climb_maximum:
        return _build_homogeneous(
            panel, popularity, settings, seed, _attempt=_attempt + 1,
        )
    target = rng.randint(max(climb_minimum, mandatory), climb_maximum)
    candidates = [
        (draft.index, pair)
        for draft in drafts for pair in draft.climb_pairs
    ]
    extras: Dict[int, list[Tuple[str, str]]] = defaultdict(list)
    support_counts = Counter(candidates)
    phase_counts = {
        draft.index: Counter(draft.climb_pairs) for draft in drafts
    }
    for _ in range(target - mandatory):
        # Only place extra support where the phase can still hold its repeats
        # apart; otherwise the schedule is unrealisable rather than merely
        # unlucky, and no amount of reshuffling recovers it.
        placeable = []
        for phase, pair in candidates:
            trial = Counter(phase_counts[phase])
            trial[pair] += 1
            if _can_space(trial):
                placeable.append((phase, pair))
        if not placeable:
            return _build_homogeneous(
                panel, popularity, settings, seed, _attempt=_attempt + 1,
            )
        least = min(support_counts[item] for item in placeable)
        eligible = [item for item in placeable if support_counts[item] == least]
        phase, pair = rng.choices(
            eligible,
            weights=[popularity[pair[0]] for _phase, pair in eligible],
            k=1,
        )[0]
        extras[phase].append(pair)
        support_counts[(phase, pair)] += 1
        phase_counts[phase][pair] += 1
    for draft in drafts:
        counts = Counter(draft.climb_pairs)
        counts.update(extras.get(draft.index, ()))
        draft.climb_pairs = tuple(_space_repeats(counts, rng))
    minimum = []
    for draft in drafts:
        active_pairs = [
            (recipe, preference)
            for recipe, preferences in draft.active.items()
            for preference in preferences
        ]
        minimum.append(math.ceil(
            (len(draft.climb_pairs) + len(active_pairs)) / DEMOS_PER_GAP
        ))
    gaps = _allocate_budget(
        settings.demos // DEMOS_PER_GAP,
        minimum,
        settings.gap_allocation,
        random.Random(f"cooking|gap|homogeneous|seed={int(seed)}"),
    )
    for draft, gap in zip(drafts, gaps):
        draft.gap = gap
        active_pairs = tuple(
            (recipe, preference)
            for recipe, preferences in draft.active.items()
            for preference in preferences
        )
        draft.settled_pairs = _settled_pairs(
            active_pairs,
            DEMOS_PER_GAP * gap - len(draft.climb_pairs),
            popularity, settings, rng,
        )
        _defer_climb_recurrence(draft)
    return drafts, {
        "attempts": _attempt + 1,
        "climb_demo_count": target,
        "transition_fraction": target / settings.demos,
        "conditioned_feasible_operation_counts": dict(feasible_counts),
        "lifecycle_operation_counts": dict(operation_counts),
    }


def _climb_groups(
    recipes: Sequence[str], decay: float, rng: random.Random,
) -> Tuple[Tuple[str, ...], ...]:
    remaining = list(recipes)
    groups = []
    while remaining:
        sizes = list(range(1, len(remaining) + 1))
        size = rng.choices(
            sizes,
            weights=[float(decay) ** (value - 1) for value in sizes],
            k=1,
        )[0]
        groups.append(tuple(remaining[:size]))
        del remaining[:size]
    return tuple(groups)


def _build_heterogeneous(
    panel: Sequence[str], popularity: Mapping[str, float],
    settings: LadderSettings, seed: int,
) -> Tuple[list[_Segment], Dict[str, Any]]:
    rng = random.Random(f"cooking|heterogeneous|seed={int(seed)}")
    states = {recipe: _Lifecycle() for recipe in panel}
    covered: set[str] = set()
    macros = []
    feasible_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for phase in range(settings.phases):
        requested = rng.randint(
            min(settings.min_recipes, len(panel)),
            min(settings.max_recipes, len(panel)),
        )
        recipes = _phase_recipes(panel, popularity, requested, covered, rng)
        active: Dict[str, Tuple[str, ...]] = {}
        deltas: Dict[str, Dict[str, set[str]]] = {}
        operations: Dict[str, str] = {}
        removed: list[Tuple[str, str]] = []
        for recipe in recipes:
            selected, delta, operation, feasible = _evolve(
                states[recipe], applicable_preferences(recipe), settings, rng,
            )
            feasible_counts.update(feasible)
            operation_counts[operation] += 1
            active[recipe] = tuple(
                item for item in applicable_preferences(recipe) if item in selected
            )
            deltas[recipe] = delta
            operations[recipe] = operation
            removed.extend((recipe, item) for item in delta["removed"])
        macros.append((phase, recipes, active, deltas, operations, tuple(removed)))

    group_rng = random.Random(f"longitudinal|climb_groups|none|seed={int(seed)}")
    current: Dict[str, Tuple[str, ...]] = {}
    drafts: list[_Segment] = []
    for phase, recipes, update, deltas, operations, removed in macros:
        for group in _climb_groups(recipes, settings.climb_decay, group_rng):
            for recipe in group:
                current[recipe] = update[recipe]
            active = {recipe: current[recipe] for recipe in panel if recipe in current}
            role_by_pair = {
                (recipe, preference): (
                    next((
                        role for role in ("added", "reintroduced", "retained")
                        if preference in deltas[recipe][role]
                    ), "retained") if recipe in group else "retained"
                )
                for recipe, preferences in active.items()
                for preference in preferences
            }
            drafts.append(_Segment(
                index=len(drafts),
                stage=phase,
                active=dict(active),
                climb_recipes=group,
                operation_by_recipe={recipe: operations[recipe] for recipe in group},
                role_by_pair=role_by_pair,
                removed_pairs=tuple(pair for pair in removed if pair[0] in group),
                strategy="heterogeneous_recipe_specific",
                climb_pairs=tuple(
                    (recipe, preference)
                    for recipe in group for preference in active[recipe]
                ),
            ))

    minimum = [
        max(
            settings.hetero_gap_min,
            math.ceil((
                len(draft.climb_pairs)
                + sum(len(items) for items in draft.active.values())
            ) / DEMOS_PER_GAP),
        )
        for draft in drafts
    ]
    gaps = _allocate_budget(
        settings.hetero_gap_mean * len(drafts),
        minimum,
        settings.hetero_gap_shape,
        random.Random(f"longitudinal|gap|heterogeneous|none|seed={int(seed)}"),
        maximum=settings.hetero_gap_max,
    )
    for draft, gap in zip(drafts, gaps):
        draft.gap = gap
        active_pairs = tuple(
            (recipe, preference)
            for recipe, preferences in draft.active.items()
            for preference in preferences
        )
        draft.settled_pairs = _settled_pairs(
            active_pairs,
            DEMOS_PER_GAP * gap - len(draft.climb_pairs),
            popularity, settings, rng,
        )
        _defer_climb_recurrence(draft)
    return drafts, {
        "conditioned_feasible_operation_counts": dict(feasible_counts),
        "lifecycle_operation_counts": dict(operation_counts),
        "serialized_phase_count": len(drafts),
    }


def _holdout_recipes(panel: Sequence[str]) -> Tuple[str, ...]:
    eligible = [
        recipe for recipe in panel
        if "wash_plates_early" in applicable_preferences(recipe)
        and len(applicable_preferences(recipe)) >= 2
    ]
    environments = {RECIPES[recipe].environment for recipe in eligible}
    if environments != {"overcooked", "burrito"}:
        raise ValueError("container-axis holdout needs eligible recipes in both environments")
    return tuple(eligible)


def _build_holdout(
    panel: Sequence[str], popularity: Mapping[str, float],
    settings: LadderSettings, seed: int,
) -> Tuple[list[_Segment], Dict[str, Any]]:
    """Controlled source-to-target container-axis transfer."""
    rng = random.Random(f"cooking|holdout|seed={int(seed)}")
    eligible = list(_holdout_recipes(panel))
    count = rng.randint(
        min(settings.min_recipes, len(eligible)),
        min(settings.max_recipes, len(eligible)),
    )
    probes: list[str] = []
    for stratum in ("overcooked", "burrito_native"):
        candidates = [
            recipe for recipe in eligible if RECIPES[recipe].stratum == stratum
        ]
        if not candidates:
            raise ValueError(f"container-axis holdout needs a {stratum} recipe")
        first = _weighted_sample(candidates, popularity, 1, rng)[0]
        probes.append(first)
        # The second draw must come from a different role-level DAG when the
        # stratum has one, otherwise this stratum's whole transfer is a
        # relabelling.  Sampling blind left one of the five fixed seeds with an
        # isomorphic-only Burrito holdout.
        distinct = [
            recipe for recipe in candidates
            if recipe != first
            and RECIPES[recipe].structure_signature
            != RECIPES[first].structure_signature
        ]
        remainder = [recipe for recipe in candidates if recipe != first]
        pool = distinct or remainder
        if pool:
            probes.extend(_weighted_sample(pool, popularity, 1, rng))
    probes.extend(_weighted_sample(
        [recipe for recipe in eligible if recipe not in probes],
        popularity,
        count - len(probes),
        rng,
    ))
    # Pick each environment's source to maximise the number of structurally
    # distinct targets it can transfer to.  Transferring between two recipes
    # with the same role-level DAG is a relabelling, not a generalisation, and
    # the learner sees only the task graph -- so an isomorphic pair proves
    # nothing about the container axis even when the physical layouts differ.
    sources = []
    for stratum in ("overcooked", "burrito_native"):
        candidates = [
            recipe for recipe in probes if RECIPES[recipe].stratum == stratum
        ]
        if not candidates:
            raise ValueError(f"holdout has no {stratum} recipe to use as a source")
        sources.append(max(
            candidates,
            key=lambda recipe: (
                sum(
                    RECIPES[other].structure_signature
                    != RECIPES[recipe].structure_signature
                    for other in probes
                    if other != recipe
                    and RECIPES[other].stratum == RECIPES[recipe].stratum
                ),
                recipe,
            ),
        ))
    targets = tuple(recipe for recipe in probes if recipe not in sources)
    if not targets:
        raise ValueError("holdout requires recipes distinct from its source recipes")
    source_by_stratum = {RECIPES[recipe].stratum: recipe for recipe in sources}
    isomorphic_targets = tuple(sorted(
        target for target in targets
        if RECIPES[target].structure_signature
        == RECIPES[source_by_stratum[RECIPES[target].stratum]].structure_signature
    ))
    if len(isomorphic_targets) == len(targets):
        raise ValueError(
            "every holdout target is structurally isomorphic to its source; "
            "the container axis would only be relabelled, never generalised"
        )
    isomorphic_strata = tuple(sorted({
        RECIPES[target].stratum for target in isomorphic_targets
    }))
    intro = HOLDOUT_SOURCE_PHASES
    current = {recipe: _canonical_preference(recipe) for recipe in probes}
    drafts: list[_Segment] = []
    group_count = HOLDOUT_TARGET_PHASES - 1
    target_groups = [targets[index::group_count] for index in range(group_count)]
    for phase in range(HOLDOUT_PHASES):
        previous = dict(current)
        changed: Tuple[str, ...] = ()
        if phase == HOLDOUT_SOURCE_PHASES // 2:
            support_changes = []
            for recipe in probes:
                support = _strategy_preference(recipe, SUPPORT_FIRST_STRATEGY)
                if support == "wash_plates_early":
                    support = _canonical_preference(recipe)
                if support != current[recipe]:
                    current[recipe] = support
                    support_changes.append(recipe)
            changed = tuple(support_changes)
        elif phase == intro:
            changed = tuple(sources)
        elif phase > intro:
            changed = tuple(target_groups[min(phase - intro - 1, len(target_groups) - 1)])
        if phase >= intro:
            for recipe in changed:
                current[recipe] = "wash_plates_early"
        active = {recipe: (preference,) for recipe, preference in current.items()}
        role = {
            (recipe, preference): ("added" if recipe in changed else "retained")
            for recipe, preference in current.items()
        }
        removed = tuple(
            (recipe, previous[recipe]) for recipe in changed
            if previous[recipe] != current[recipe]
        )
        drafts.append(_Segment(
            index=phase,
            stage=phase,
            active=active,
            climb_recipes=(tuple(probes) if phase == 0 else changed),
            operation_by_recipe={
                recipe: (
                    "initialization" if phase == 0 else
                    "swap" if recipe in changed else "retention"
                ) for recipe in probes
            },
            role_by_pair=role,
            removed_pairs=removed,
            strategy=(
                "holdout_source_training" if phase < intro else
                "holdout_axis_source_introduction" if phase == intro else
                "holdout_axis_target_composition"
            ),
        ))

    minimum = []
    for draft in drafts:
        active_pairs = tuple(
            (recipe, preference)
            for recipe, preferences in draft.active.items()
            for preference in preferences
        )
        draft.climb_pairs = tuple(
            pair for pair in active_pairs if pair[0] in draft.climb_recipes
        )
        priority = [
            pair for pair in draft.climb_pairs if pair[1] == "wash_plates_early"
        ]
        minimum.append(math.ceil((
            len(draft.climb_pairs) + len(active_pairs)
            + max(0, settings.holdout_demos - len(priority))
        ) / DEMOS_PER_GAP))
    gaps = (settings.hetero_gap_mean,) * HOLDOUT_PHASES
    if any(required > settings.hetero_gap_mean for required in minimum):
        raise ValueError("holdout active support exceeds its 45-demo phase")
    for draft, gap in zip(drafts, gaps):
        draft.gap = gap
        active_pairs = tuple(
            (recipe, preference)
            for recipe, preferences in draft.active.items()
            for preference in preferences
        )
        priority = tuple(
            pair for pair in draft.climb_pairs if pair[1] == "wash_plates_early"
        )
        draft.settled_pairs = _settled_pairs(
            active_pairs,
            DEMOS_PER_GAP * gap - len(draft.climb_pairs),
            popularity, settings, rng,
            priority_pairs=priority,
        )
        _defer_climb_recurrence(draft)
    return drafts, {
        "holdout_axis": "container_timing",
        "holdout_value": "container_first",
        "holdout_introduction_phase": intro,
        "source_phase_count": HOLDOUT_SOURCE_PHASES,
        "target_phase_count": HOLDOUT_TARGET_PHASES,
        "holdout_source_recipe_ids": sorted(sources),
        "holdout_target_recipe_ids": sorted(targets),
        # Targets whose role-level DAG matches their source: the container axis
        # is only relabelled across these, so they must not be pooled into a
        # cross-environment generalisation claim.
        "holdout_isomorphic_target_recipe_ids": list(isomorphic_targets),
        "holdout_isomorphic_transfer_strata": list(isomorphic_strata),
        "holdout_non_isomorphic_target_recipe_ids": sorted(
            set(targets) - set(isomorphic_targets)
        ),
    }


def _materialize(
    drafts: Sequence[_Segment], scenario: str, seed: int,
    holdout_targets: Iterable[str] = (),
) -> Tuple[CookingTask, ...]:
    tasks: list[CookingTask] = []
    known_recipes: set[str] = set()
    interval_by_pair: Dict[Tuple[str, str], str] = {}
    exposure_by_interval: Counter[str] = Counter()
    shift_count: Counter[str] = Counter()
    holdout_target_set = set(holdout_targets)
    for draft in drafts:
        first_climb = set(draft.climb_pairs)
        for phase_role, pairs in (
            ("climb", draft.climb_pairs),
            ("settled", draft.settled_pairs),
        ):
            for recipe, preference in pairs:
                acquisition = False
                first_pair_support = (
                    phase_role == "climb"
                    and (recipe, preference) in first_climb
                )
                if first_pair_support:
                    first_climb.remove((recipe, preference))
                    if (
                        draft.role_by_pair.get((recipe, preference))
                        in {"added", "reintroduced"}
                        and recipe in known_recipes
                    ):
                        shift_count[recipe] += 1
                        interval_by_pair[(recipe, preference)] = (
                            f"{scenario}|{int(seed)}|{recipe}|shift_{shift_count[recipe]}"
                        )
                        acquisition = True
                adaptation_id = interval_by_pair.get((recipe, preference))
                exposure = 0
                if adaptation_id is not None:
                    exposure_by_interval[adaptation_id] += 1
                    exposure = exposure_by_interval[adaptation_id]
                first_recipe = recipe not in known_recipes
                lifecycle = (
                    "introduce_recipe" if first_recipe else
                    "acquire_shift" if acquisition else
                    "post_update_recurrence" if exposure >= 2 else
                    "retain"
                )
                tasks.append(CookingTask.create(
                    recipe,
                    preference,
                    scenario=scenario,
                    phase=draft.stage,
                    schedule_step=draft.index,
                    phase_role=phase_role,
                    lifecycle=lifecycle,
                    preference_changed=acquisition,
                    exposure_after_change=exposure,
                    strategy=draft.strategy,
                    adaptation_id=adaptation_id,
                    holdout_target=recipe in holdout_target_set,
                ))
                known_recipes.add(recipe)
    return tuple(tasks)


def generate_ladder(
    *,
    seed: int,
    scenario: str,
    recipe_ids: Sequence[str] = tuple(RECIPES),
    settings: LadderSettings = LadderSettings(),
    return_audit: bool = False,
) -> Tuple[CookingTask, ...] | Tuple[Tuple[CookingTask, ...], Mapping[str, Any]]:
    normalized = str(scenario).strip().lower()
    if normalized not in SCENARIOS:
        raise ValueError(f"unknown ladder scenario {scenario!r}")
    recipes = _validate_recipes(recipe_ids)
    settings.validate(len(recipes))
    panel, popularity = _panel(recipes, settings, int(seed))
    if normalized == "homogeneous":
        drafts, metadata = _build_homogeneous(panel, popularity, settings, int(seed))
        targets: Tuple[str, ...] = ()
    elif normalized == "heterogeneous":
        drafts, metadata = _build_heterogeneous(panel, popularity, settings, int(seed))
        targets = ()
    else:
        drafts, metadata = _build_holdout(panel, popularity, settings, int(seed))
        targets = tuple(metadata["holdout_target_recipe_ids"])
    tasks = _materialize(drafts, normalized, int(seed), targets)
    audit = ladder_audit(tasks, settings=settings, metadata=metadata)
    return (tasks, audit) if return_audit else tasks


def ladder_audit(
    tasks: Sequence[CookingTask],
    *,
    settings: LadderSettings | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate the longitudinal semantics and expose schedule difficulty."""
    if not tasks:
        raise ValueError("ladder is empty")
    scenario_set = {task.scenario for task in tasks}
    if len(scenario_set) != 1:
        raise ValueError("a ladder must contain exactly one scenario")
    scenario = next(iter(scenario_set))
    if any(len(RECIPES[task.recipe_id].ingredients) < 2 for task in tasks):
        raise ValueError("single-ingredient recipe leaked into the ladder")
    if settings is not None:
        expected = settings.demos
        if scenario == "homogeneous" and len(tasks) != expected:
            raise ValueError(f"{scenario} has {len(tasks)} events, expected {expected}")
        holdout_expected = DEMOS_PER_GAP * settings.hetero_gap_mean * HOLDOUT_PHASES
        if scenario == "holdout" and len(tasks) != holdout_expected:
            raise ValueError(
                f"holdout has {len(tasks)} events, expected {holdout_expected}"
            )
        if scenario == "heterogeneous" and len(tasks) <= expected:
            raise ValueError("heterogeneous serialization did not exceed the fixed ladder")

    by_adaptation: Dict[str, list[Tuple[int, CookingTask]]] = defaultdict(list)
    first_recipe_index: Dict[str, int] = {}
    for index, task in enumerate(tasks):
        first_recipe_index.setdefault(task.recipe_id, index)
        if task.adaptation_id is not None:
            by_adaptation[task.adaptation_id].append((index, task))
    for adaptation_id, indexed in by_adaptation.items():
        rows = [task for _index, task in indexed]
        exposures = [task.exposure_after_change for task in rows]
        if exposures != list(range(1, len(rows) + 1)):
            raise ValueError(f"non-contiguous exposures for {adaptation_id}")
        acquisitions = [task for task in rows if task.preference_changed]
        if len(acquisitions) != 1 or acquisitions[0].exposure_after_change != 1:
            raise ValueError(f"invalid acquisition marker for {adaptation_id}")
        acquisition_index = next(index for index, task in indexed if task.preference_changed)
        if first_recipe_index[acquisitions[0].recipe_id] >= acquisition_index:
            raise ValueError("a preference shift was not applied to a known recipe")
        if len(rows) < 2:
            raise ValueError(f"adaptation {adaptation_id} has no later recurrence")

    phases = sorted({task.phase for task in tasks})
    strategies_by_phase = {
        str(phase): sorted({task.strategy for task in tasks if task.phase == phase})
        for phase in phases
    }
    if scenario == "homogeneous" and any(
        len(values) != 1 for values in strategies_by_phase.values()
    ):
        raise ValueError("homogeneous recipes did not share one ordered phase strategy")

    grounding = strategy_grounding(sorted({task.recipe_id for task in tasks}))
    if grounding["degenerate_strategy_groundings"]:
        raise ValueError(
            "a non-canonical strategy grounded to the canonical ordering: "
            f"{grounding['degenerate_strategy_groundings']}"
        )
    holdout_targets = {task.recipe_id for task in tasks if task.holdout_target}
    holdout_intro = None
    if scenario == "holdout":
        acquisitions = [
            (index, task) for index, task in enumerate(tasks)
            if task.preference == CONTAINER_FIRST_PREFERENCE
            and task.preference_changed
        ]
        if not acquisitions:
            raise ValueError("held-out container axis was never introduced")
        holdout_intro = min(task.phase for _index, task in acquisitions)
        first_holdout_index = min(index for index, _task in acquisitions)
        if any(
            task.preference == CONTAINER_FIRST_PREFERENCE
            for task in tasks[:first_holdout_index]
        ):
            raise ValueError("container-first value leaked before introduction")
        for recipe in holdout_targets:
            target_first = min(
                index for index, task in enumerate(tasks)
                if task.recipe_id == recipe
                and task.preference == CONTAINER_FIRST_PREFERENCE
            )
            if not any(
                task.recipe_id == recipe
                and task.preference != CONTAINER_FIRST_PREFERENCE
                for task in tasks[:target_first]
            ):
                raise ValueError("holdout target recipe was not known before composition")

    adaptation_gaps = []
    for indexed in by_adaptation.values():
        if len(indexed) >= 2:
            adaptation_gaps.append(indexed[1][0] - indexed[0][0] - 1)
    active_preferences_by_phase_recipe: Dict[Tuple[int, str], set[str]] = defaultdict(set)
    for task in tasks:
        active_preferences_by_phase_recipe[(task.phase, task.recipe_id)].add(task.preference)
    climb_demo_count = sum(task.phase_role == "climb" for task in tasks)
    transition_fraction = climb_demo_count / len(tasks)
    if (
        settings is not None
        and scenario == "homogeneous"
        and not settings.transition_min <= transition_fraction <= settings.transition_max
    ):
        raise ValueError("homogeneous climb fraction is outside the publication band")
    return {
        "episodes": len(tasks),
        "recipes": len({task.recipe_id for task in tasks}),
        "recipe_ids": sorted({task.recipe_id for task in tasks}),
        "macro_phases": len(phases),
        "climb_demo_count": climb_demo_count,
        "transition_fraction": transition_fraction,
        "preference_acquisitions": sum(task.preference_changed for task in tasks),
        "post_update_recurrences": sum(
            task.exposure_after_change >= 2 for task in tasks
        ),
        "adaptation_intervals": len(by_adaptation),
        "mean_intervening_episodes_before_first_recurrence": (
            sum(adaptation_gaps) / len(adaptation_gaps) if adaptation_gaps else None
        ),
        "max_active_preferences_per_recipe_phase": max(
            map(len, active_preferences_by_phase_recipe.values())
        ),
        "strategies_by_phase": strategies_by_phase,
        "holdout_introduction_phase": holdout_intro,
        "holdout_target_recipe_ids": sorted(holdout_targets),
        "holdout_target_environments": sorted({
            RECIPES[recipe].stratum for recipe in holdout_targets
        }),
        "holdout_isomorphic_target_recipe_ids": list(
            (metadata or {}).get("holdout_isomorphic_target_recipe_ids", ())
        ),
        "holdout_isomorphic_transfer_strata": list(
            (metadata or {}).get("holdout_isomorphic_transfer_strata", ())
        ),
        "holdout_non_isomorphic_target_recipe_ids": list(
            (metadata or {}).get("holdout_non_isomorphic_target_recipe_ids", ())
        ),
        **grounding,
        "scenario_invariants_passed": True,
        "schedule_metadata": dict(metadata or {}),
    }


__all__ = [
    "CANONICAL_STRATEGY", "CONTAINER_FIRST_PREFERENCE",
    "CONTAINER_FIRST_STRATEGY", "DEMOS_PER_GAP",
    "LadderSettings", "SCENARIOS", "SUPPORT_FIRST_STRATEGY",
    "generate_ladder", "ladder_audit", "strategy_grounding",
]
