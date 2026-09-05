"""Adaptive-HRC integration for standard Overcooked and Burrito."""

from .catalog import (
    BURRITO_RECIPE_IDS,
    OVERCOOKED_RECIPE_IDS,
    PREFERENCES,
    RECIPES,
    TALENTS_LIKE_PREFERENCES,
    applicable_preferences,
    get_recipe,
)
from .domain import (
    BurritoDomainAdapter,
    CookingDomainAdapter,
    REWARD_FEATURE_VERSION,
    SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
    SEMANTIC_FEATURE_VERSION,
    STRATEGY_ROLE_VERSION,
    TaskState,
)
from .ladder import LadderSettings, SCENARIOS, generate_ladder, ladder_audit
from .legacy_options import LegacyBurritoOptionExecutor
from .options import BurritoOptionExecutor, OptionExecution, OptionExecutionError
from .overcooked_options import OvercookedOptionExecutor
from .protocol import (
    ASSIST,
    OBSERVE,
    BurritoDecision,
    BurritoEpisodeResult,
    BurritoHrcRunner,
    BurritoObservation,
    BurritoTask,
    CookingDecision,
    CookingEpisodeResult,
    CookingHrcRunner,
    CookingObservation,
    CookingTask,
)
from .runtime import (
    BurritoRuntime,
    PinMismatchError,
    SubmoduleMissingError,
    UpstreamPaths,
    verify_pins,
)
from .task_graph import (
    BurritoPreferencePolicy,
    BurritoTaskGraph,
    CookingPreferencePolicy,
    CookingTaskGraph,
    PREFERENCE_NAMES,
)


__all__ = [name for name in globals() if not name.startswith("_")]
