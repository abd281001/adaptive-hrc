"""Frozen, participant-specific episode schedules for robot studies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Tuple

from .config import ConfigurationError, LabConfig


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _identifier(value: Any, label: str) -> str:
    text = str(value).strip()
    if not _SAFE_ID.fullmatch(text):
        raise ConfigurationError(
            f"{label} must be 1-64 safe identifier characters"
        )
    return text


@dataclass(frozen=True)
class ScheduledEpisode:
    trial_id: str
    recipe_id: str
    preference_id: str
    condition: str
    expected_mode: str

    @property
    def trial_metadata(self) -> Mapping[str, str]:
        return {
            "trial_id": self.trial_id,
            "preference_id": self.preference_id,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class StudySchedule:
    schema_version: int
    schedule_id: str
    participant_id: str
    counterbalance_id: str
    episodes: Tuple[ScheduledEpisode, ...]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "participant_id": self.participant_id,
            "counterbalance_id": self.counterbalance_id,
            "episodes": [
                {
                    "trial_id": row.trial_id,
                    "recipe_id": row.recipe_id,
                    "preference_id": row.preference_id,
                    "condition": row.condition,
                    "expected_mode": row.expected_mode,
                }
                for row in self.episodes
            ],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], config: LabConfig,
    ) -> "StudySchedule":
        unknown = set(raw) - {
            "schema_version", "schedule_id", "participant_id",
            "counterbalance_id", "episodes",
        }
        if unknown:
            raise ConfigurationError(
                f"study schedule has unknown keys: {sorted(unknown)}"
            )
        if int(raw.get("schema_version", 0)) != 1:
            raise ConfigurationError("study schedule schema_version must be 1")
        schedule_id = _identifier(raw.get("schedule_id", ""), "schedule_id")
        participant_id = _identifier(
            raw.get("participant_id", ""), "participant_id",
        )
        counterbalance_id = _identifier(
            raw.get("counterbalance_id", ""), "counterbalance_id",
        )
        episode_rows = raw.get("episodes")
        if not isinstance(episode_rows, list) or not episode_rows:
            raise ConfigurationError("study schedule episodes must be a non-empty array")

        episodes = []
        seen_trials: set[str] = set()
        observed_recipes: set[str] = set()
        assisted_recipes: set[str] = set()
        assist_started = False
        for index, value in enumerate(episode_rows):
            if not isinstance(value, Mapping):
                raise ConfigurationError(f"study schedule episodes[{index}] must be an object")
            extra = set(value) - {
                "trial_id", "recipe_id", "preference_id", "condition",
                "expected_mode",
            }
            if extra:
                raise ConfigurationError(
                    f"study schedule episodes[{index}] has unknown keys: {sorted(extra)}"
                )
            trial_id = _identifier(value.get("trial_id", ""), f"episodes[{index}].trial_id")
            if trial_id in seen_trials:
                raise ConfigurationError(f"duplicate scheduled trial_id {trial_id!r}")
            seen_trials.add(trial_id)
            recipe_id = _identifier(value.get("recipe_id", ""), f"episodes[{index}].recipe_id")
            if recipe_id not in config.recipes:
                raise ConfigurationError(f"scheduled recipe {recipe_id!r} is not configured")
            preference_id = _identifier(
                value.get("preference_id", ""),
                f"episodes[{index}].preference_id",
            )
            condition = _identifier(
                value.get("condition", ""), f"episodes[{index}].condition",
            )
            expected_mode = str(value.get("expected_mode", "")).strip()
            if expected_mode not in {"observe", "assist"}:
                raise ConfigurationError(
                    f"episodes[{index}].expected_mode must be observe or assist"
                )
            if expected_mode == "observe":
                if assist_started:
                    raise ConfigurationError(
                        "all scheduled observation episodes must precede every assist episode"
                    )
                if recipe_id in observed_recipes:
                    raise ConfigurationError(
                        f"recipe {recipe_id!r} has more than one scheduled observation episode"
                    )
                if condition != "observation":
                    raise ConfigurationError(
                        "observation episodes must use condition='observation'"
                    )
                observed_recipes.add(recipe_id)
            else:
                assist_started = True
                if recipe_id not in observed_recipes:
                    raise ConfigurationError(
                        f"recipe {recipe_id!r} is scheduled for assist before observation"
                    )
                if condition == "observation":
                    raise ConfigurationError(
                        "assist episodes may not use condition='observation'"
                    )
                assisted_recipes.add(recipe_id)
            episodes.append(ScheduledEpisode(
                trial_id, recipe_id, preference_id, condition, expected_mode,
            ))
        if not 3 <= len(observed_recipes) <= 4:
            raise ConfigurationError(
                "a publication schedule must observe exactly three or four recipes"
            )
        missing_assist = observed_recipes - assisted_recipes
        if missing_assist:
            raise ConfigurationError(
                f"every observed recipe needs a later assist episode; missing {sorted(missing_assist)}"
            )
        return cls(
            1, schedule_id, participant_id, counterbalance_id, tuple(episodes),
        )


def load_study_schedule(path: str | Path, config: LabConfig) -> StudySchedule:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError("study schedule must contain valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError("study schedule root must be an object")
    return StudySchedule.from_mapping(raw, config)
