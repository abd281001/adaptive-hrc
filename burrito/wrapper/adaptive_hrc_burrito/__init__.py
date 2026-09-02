"""Adaptive-HRC's isolated runtime boundary for the Burrito simulator."""

from .runtime import (
    BurritoRuntime,
    PinMismatchError,
    SubmoduleMissingError,
    UpstreamPaths,
    verify_pins,
)
from .domain import (
    BurritoDomainAdapter,
    REWARD_FEATURE_VERSION,
    SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
    SEMANTIC_FEATURE_VERSION,
)
from .macros import (
    COLLECT_AND_STAGE_COOKED_RICE,
    STAGE_CLEAN_PLATE,
    START_BOILING_RICE,
    assemble_action,
    fetch_and_stage_action,
    macro_actions,
    prepare_and_stage_action,
    serve_action,
    start_cooking_action,
)
from .options import (
    CONTROLLED_PARKING_POSITIONS,
    BurritoOptionExecutor,
    OptionExecution,
    OptionExecutionError,
)
from .protocol import (
    ASSIST,
    OBSERVE,
    BurritoDecision,
    BurritoEpisodeResult,
    BurritoHrcRunner,
    BurritoObservation,
    BurritoTask,
)
from .task_graph import (
    BurritoPreferencePolicy,
    BurritoTaskGraph,
    PREFERENCE_NAMES,
    choose_acceptable_action,
)

__all__ = [
    "BurritoRuntime",
    "PinMismatchError",
    "SubmoduleMissingError",
    "UpstreamPaths",
    "verify_pins",
    "ASSIST",
    "OBSERVE",
    "BurritoDecision",
    "BurritoDomainAdapter",
    "BurritoEpisodeResult",
    "BurritoHrcRunner",
    "BurritoObservation",
    "BurritoOptionExecutor",
    "BurritoPreferencePolicy",
    "BurritoTask",
    "BurritoTaskGraph",
    "COLLECT_AND_STAGE_COOKED_RICE",
    "CONTROLLED_PARKING_POSITIONS",
    "OptionExecution",
    "OptionExecutionError",
    "PREFERENCE_NAMES",
    "REWARD_FEATURE_VERSION",
    "STAGE_CLEAN_PLATE",
    "START_BOILING_RICE",
    "SEMANTIC_FALLBACK_MAX_RMS_DISTANCE",
    "SEMANTIC_FEATURE_VERSION",
    "assemble_action",
    "choose_acceptable_action",
    "fetch_and_stage_action",
    "macro_actions",
    "prepare_and_stage_action",
    "serve_action",
    "start_cooking_action",
]
