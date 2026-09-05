"""Small physical-executor factory for the unified recipe catalog."""
from __future__ import annotations

from typing import Any

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
) -> Any:
    recipe = get_recipe(recipe_id)
    if recipe.environment == "overcooked":
        return OvercookedOptionExecutor(
            runtime, recipe_id, horizon=horizon, seed=seed,
        )
    if recipe.compatibility_dynamics:
        return LegacyBurritoOptionExecutor(
            runtime, recipe_id, horizon=horizon, seed=seed,
        )
    return BurritoOptionExecutor(
        runtime, recipe_id, horizon=horizon, seed=seed,
    )


__all__ = ["create_executor"]
