"""Canonical handoff-safe decision macros for the Burrito task graph."""
from __future__ import annotations

from typing import Tuple


START_BOILING_RICE = "START_BOILING_RICE"
STAGE_CLEAN_PLATE = "STAGE_CLEAN_PLATE"
COLLECT_AND_STAGE_COOKED_RICE = "COLLECT_AND_STAGE_COOKED_RICE"


def protein_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"steak", "mushroom"}:
        raise ValueError("protein must be 'steak' or 'mushroom'")
    return normalized


def fetch_and_stage_action(protein: str) -> str:
    return f"FETCH_AND_STAGE_{protein_name(protein).upper()}"


def prepare_and_stage_action(protein: str) -> str:
    return f"PREPARE_AND_STAGE_{protein_name(protein).upper()}"


def start_cooking_action(protein: str) -> str:
    return f"START_COOKING_{protein_name(protein).upper()}"


def assemble_action(protein: str) -> str:
    return f"ASSEMBLE_{protein_name(protein).upper()}_BURRITO"


def serve_action(protein: str) -> str:
    return f"SERVE_{protein_name(protein).upper()}_BURRITO"


def action_protein(action: str) -> str | None:
    label = str(action).upper()
    if "STEAK" in label:
        return "steak"
    if "MUSHROOM" in label:
        return "mushroom"
    return None


def macro_actions(protein: str) -> Tuple[str, ...]:
    """The recipe action set; preference policies only change its order."""
    normalized = protein_name(protein)
    return (
        fetch_and_stage_action(normalized),
        prepare_and_stage_action(normalized),
        start_cooking_action(normalized),
        START_BOILING_RICE,
        STAGE_CLEAN_PLATE,
        COLLECT_AND_STAGE_COOKED_RICE,
        assemble_action(normalized),
        serve_action(normalized),
    )


__all__ = [
    "COLLECT_AND_STAGE_COOKED_RICE",
    "STAGE_CLEAN_PLATE",
    "START_BOILING_RICE",
    "action_protein",
    "assemble_action",
    "fetch_and_stage_action",
    "macro_actions",
    "prepare_and_stage_action",
    "protein_name",
    "serve_action",
    "start_cooking_action",
]
