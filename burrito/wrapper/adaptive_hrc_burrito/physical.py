"""Small physical-executor factory for the unified recipe catalog."""
from __future__ import annotations

from typing import Any, Optional

from .catalog import get_recipe
from .legacy_options import LegacyBurritoOptionExecutor
from .options import BurritoOptionExecutor
from .overcooked_options import OvercookedOptionExecutor
from .runtime import BurritoRuntime


def create_executor(
    runtime: BurritoRuntime,
    recipe_id: str,
    *,
    horizon: int,
    seed: int,
    layout: Optional[str] = None,
) -> Any:
    """Build the physical executor for one recipe.

    ``layout`` overrides where an Overcooked recipe is carried out. The
    Burrito executors bind to their own layouts, so an override is rejected
    there rather than silently ignored.
    """
    recipe = get_recipe(recipe_id)
    if recipe.environment == "overcooked":
        return OvercookedOptionExecutor(
            runtime, recipe_id, horizon=horizon, seed=seed, layout=layout,
        )
    if layout is not None:
        raise ValueError(
            f"layout override is only supported for Overcooked recipes, "
            f"not {recipe_id!r} ({recipe.environment})"
        )
    if recipe.compatibility_dynamics:
        return LegacyBurritoOptionExecutor(
            runtime, recipe_id, horizon=horizon, seed=seed,
        )
    return BurritoOptionExecutor(
        runtime, recipe_id, horizon=horizon, seed=seed,
    )


__all__ = ["create_executor"]
