"""Generate a deterministic integrity, outcome, and comparator report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from .config import LabConfig, validate_calibration_artifacts
from .schedule import StudySchedule
from .shadow_baselines import score_shadow_baselines


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _describe(values: Iterable[float]) -> Mapping[str, Any]:
    rows = [float(value) for value in values]
    return {
        "n": len(rows), "mean": mean(rows) if rows else None,
        "median": median(rows) if rows else None,
        "p95": _quantile(rows, 0.95), "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
    }


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _read_events(path: Path) -> list[Mapping[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid events.jsonl line {line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"events.jsonl line {line_number} is not an object")
            events.append(row)
    return events


def _summarize_decisions(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    robot = [row for row in rows if row.get("executed_by") == "robot"]
    corrections = [row for row in rows if row.get("executed_by") == "human_correction"]
    robot_turns = robot + corrections
    human = [row for row in rows if row.get("executed_by") in {"human", "human_correction"}]
    predicted = [row for row in rows if row.get("predicted") is not None]
    durations = [
        float(row["execution"]["elapsed_s"])
        for row in robot
        if isinstance(row.get("execution"), Mapping)
        and row["execution"].get("elapsed_s") is not None
    ]
    return {
        "actions_committed": len(rows),
        "robot_turns": len(robot_turns),
        "robot_completed": len(robot),
        "robot_top_1_hits": len(robot),
        "robot_top_1": len(robot) / len(robot_turns) if robot_turns else None,
        "human_corrections": len(corrections),
        "human_actions": len(human),
        "normalized_human_action_load": len(human) / len(rows) if rows else None,
        "teacher_forced_predictions": len(predicted),
        "teacher_forced_top_1": (
            sum(bool(row.get("correct")) for row in predicted) / len(predicted)
            if predicted else None
        ),
        "robot_action_duration_s": _describe(durations),
    }


def _grouped(episodes: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    values = sorted({
        str(episode.get("trial_metadata", {}).get(key, ""))
        for episode in episodes
        if episode.get("trial_metadata", {}).get(key) not in {None, ""}
    })
    return {
        value: {
            "episodes": sum(
                episode.get("trial_metadata", {}).get(key) == value
                for episode in episodes
            ),
            **_summarize_decisions([
                decision
                for episode in episodes
                if episode.get("trial_metadata", {}).get(key) == value
                for decision in episode.get("decisions", ())
            ]),
        }
        for value in values
    }


def _metadata_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        isinstance(row.get("trial_metadata", {}).get(name), str)
        and bool(row["trial_metadata"][name].strip())
        for row in rows
        for name in ("trial_id", "preference_id", "condition")
    )


def _episode_rows(
    events: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_trial: dict[str, list[Mapping[str, Any]]] = {}
    for decision in decisions:
        trial_id = str(decision.get("trial_metadata", {}).get("trial_id", ""))
        by_trial.setdefault(trial_id, []).append(decision)
    output = []
    for event in events:
        if event.get("event") != "episode_completed":
            continue
        episode = event.get("payload", {}).get("episode")
        if not isinstance(episode, Mapping):
            continue
        metadata = dict(episode.get("trial_metadata", {}))
        trial_id = str(metadata.get("trial_id", ""))
        output.append({
            "episode_index": len(output),
            "external_recipe_id": episode.get("external_recipe_id"),
            "mode": episode.get("mode"),
            "trial_metadata": metadata,
            "actions": list(episode.get("actions", ())),
            "summary": dict(episode.get("summary", {})),
            "decisions": list(by_trial.get(trial_id, ())),
        })
    return output


def build_report(run_dir: Path) -> Mapping[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot_path = run_dir / "config.snapshot.json"
    snapshot_raw = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else None
    config = LabConfig.from_mapping(snapshot_raw) if isinstance(snapshot_raw, Mapping) else None
    snapshot_digest = config.digest if config is not None else None
    events = _read_events(run_dir / "events.jsonl")
    indices = [int(row["event_index"]) for row in events]
    contiguous = indices == list(range(indices[0], indices[0] + len(indices))) if indices else True
    starts_at_one = bool(indices and indices[0] == 1)
    decisions = [
        row["payload"] for row in events
        if row.get("event") in {"learner_transition_committed", "episode_completed"}
        and isinstance(row.get("payload"), Mapping)
    ]
    episodes = _episode_rows(events, decisions)
    failures = [row for row in events if row.get("event") in {
        "robot_execution_failed", "robot_execution_ambiguous", "learner_commit_failed",
    }]
    emergency_stops = [row for row in events if row.get("event") == "emergency_stop"]
    aborted = [row for row in events if row.get("event") == "episode_aborted"]
    started_ids = {
        str(row.get("payload", {}).get("execution_id"))
        for row in events if row.get("event") == "robot_execution_started"
    }
    resolved_ids = {
        str(row.get("payload", {}).get("metadata", {}).get("execution_id"))
        for row in events if row.get("event") in {
            "robot_execution_succeeded", "robot_execution_failed",
            "robot_execution_ambiguous",
        }
    }
    unresolved_ids = sorted(started_ids - resolved_ids)

    metadata_complete = _metadata_complete(decisions) and _metadata_complete(episodes)
    completed_trial_ids = [
        str(row.get("trial_metadata", {}).get("trial_id", "")) for row in episodes
    ]
    trial_ids_unique = len(completed_trial_ids) == len(set(completed_trial_ids))

    schedule_path = run_dir / "schedule.snapshot.json"
    schedule = None
    schedule_error = None
    if schedule_path.is_file() and config is not None:
        try:
            schedule_raw = json.loads(schedule_path.read_text(encoding="utf-8"))
            schedule = StudySchedule.from_mapping(schedule_raw, config)
        except Exception as exc:
            schedule_error = f"{type(exc).__name__}: {exc}"
    manifest_schedule = manifest.get("study_schedule")
    schedule_digest_matches = bool(
        schedule is not None and isinstance(manifest_schedule, Mapping)
        and schedule.digest == manifest_schedule.get("digest")
    )
    schedule_manifest_matches = bool(
        schedule_digest_matches
        and manifest_schedule.get("schedule_id") == schedule.schedule_id
        and manifest_schedule.get("participant_id") == schedule.participant_id
        and manifest_schedule.get("counterbalance_id") == schedule.counterbalance_id
        and manifest_schedule.get("episode_count") == len(schedule.episodes)
    ) if schedule is not None and isinstance(manifest_schedule, Mapping) else False
    expected_trials = [row.trial_id for row in schedule.episodes] if schedule is not None else []
    schedule_complete = bool(
        schedule is not None
        and completed_trial_ids == expected_trials
        and len(episodes) == len(schedule.episodes)
        and all(
            episode.get("external_recipe_id") == expected.recipe_id
            and episode.get("mode") == expected.expected_mode
            and all(
                episode.get("trial_metadata", {}).get(key) == value
                for key, value in expected.trial_metadata.items()
            )
            and episode.get("trial_metadata", {}).get("schedule_id") == schedule.schedule_id
            and episode.get("trial_metadata", {}).get("participant_id") == schedule.participant_id
            and episode.get("trial_metadata", {}).get("counterbalance_id") == schedule.counterbalance_id
            and episode.get("trial_metadata", {}).get("schedule_index") == index
            for index, (episode, expected) in enumerate(zip(episodes, schedule.episodes))
        )
    )
    episode_records_valid = bool(config is not None and episodes) and all(
        episode.get("external_recipe_id") in config.recipes
        and len(episode.get("actions", ())) == len(episode.get("decisions", ()))
        and list(episode.get("actions", ())) == [
            row.get("action") for row in episode.get("decisions", ())
        ]
        and len(episode.get("actions", ()))
        == len(config.recipes[str(episode["external_recipe_id"])].actions)
        and set(episode.get("actions", ()))
        == set(config.recipes[str(episode["external_recipe_id"])].actions)
        for episode in episodes
    )
    robot_postconditions_complete = all(
        isinstance(row.get("execution"), Mapping)
        and row["execution"].get("success") is True
        and isinstance(row["execution"].get("metadata"), Mapping)
        and isinstance(row["execution"]["metadata"].get("postcondition"), Mapping)
        for row in decisions if row.get("executed_by") == "robot"
    )

    artifact_checks: dict[str, Any] = {
        "calibration_record_present": False,
        "calibration_record_digest_matches": False,
        "runtime_lock_present": False,
        "runtime_lock_digest_matches": False,
        "semantic_validation": False,
        "error": None,
    }
    if config is not None and config.motion.calibrated:
        calibration_path = run_dir / "calibration.record.json"
        runtime_path = run_dir / "runtime.lock.json"
        artifact_checks.update({
            "calibration_record_present": calibration_path.is_file(),
            "calibration_record_digest_matches": _sha256(calibration_path) == config.motion.calibration_record_sha256,
            "runtime_lock_present": runtime_path.is_file(),
            "runtime_lock_digest_matches": _sha256(runtime_path) == config.motion.runtime_lock_sha256,
        })
        try:
            calibration_record = json.loads(calibration_path.read_text(encoding="utf-8"))
            runtime_lock = json.loads(runtime_path.read_text(encoding="utf-8"))
            validate_calibration_artifacts(config, calibration_record, runtime_lock)
            artifact_checks["semantic_validation"] = True
        except Exception as exc:
            artifact_checks["error"] = f"{type(exc).__name__}: {exc}"

    shadow = score_shadow_baselines(config, episodes) if config is not None and episodes else {}
    shadow_complete = bool(shadow) and all(
        row.get("status") == "complete"
        and row.get("overall", {}).get("teacher_forced_predictions", 0) > 0
        for row in shadow.values()
    )
    hardware = manifest.get("hardware_preflight", {})
    software = manifest.get("software", {})
    reasons: list[str] = []

    def require(ok: bool, reason: str) -> None:
        if not ok:
            reasons.append(reason)

    require(manifest.get("schema_version") == 3, "manifest schema is not the publication schema")
    require(bool(manifest.get("publication_run")), "run was not started with --publication-run")
    require(bool(manifest.get("require_motion")), "live motion was not required")
    require(manifest.get("hardware_backend") == "stretch_http_bridge", "run did not use the Stretch HTTP bridge")
    require(config is not None and config.motion.calibrated, "configuration is not calibrated")
    require(contiguous, "event indices are not contiguous")
    require(starts_at_one, "event log does not start at index 1")
    require(snapshot_digest == manifest.get("config_digest"), "configuration snapshot digest mismatch")
    require(not unresolved_ids, "execution IDs remain unresolved")
    require(not failures, "execution or learner-commit failures occurred")
    require(not emergency_stops, "an emergency stop occurred")
    require(not aborted, "an episode was aborted")
    require(metadata_complete and trial_ids_unique, "trial metadata is missing or duplicated")
    require(
        schedule_manifest_matches and schedule_complete,
        "frozen schedule is missing, mismatched, or incomplete",
    )
    require(episode_records_valid, "completed episode records do not match their configured recipes")
    require(robot_postconditions_complete, "a committed robot action lacks a successful physical postcondition")
    require(
        bool(
            artifact_checks["calibration_record_present"]
            and artifact_checks["calibration_record_digest_matches"]
            and artifact_checks["runtime_lock_present"]
            and artifact_checks["runtime_lock_digest_matches"]
            and artifact_checks["semantic_validation"]
        ),
        "calibration/runtime artifacts failed validation",
    )
    require(bool(hardware.get("motion_enabled")), "hardware preflight was not motion-enabled")
    require(bool(hardware.get("ready")), "hardware preflight was not ready")
    require(hardware.get("config_digest") == manifest.get("config_digest"), "hardware config digest mismatch")
    require(hardware.get("calibration_id") == manifest.get("calibration_id"), "hardware calibration ID mismatch")
    require(bool(hardware.get("runtime_qualification", {}).get("qualified")), "hardware runtime was not qualified")
    require(not software.get("git_dirty", True), "software worktree was dirty")
    require(software.get("git_commit") not in {None, "", "unavailable"}, "Git revision is unidentified")
    require(manifest.get("seed") == int(config.agent_settings.get("seed", 1337)) if config is not None else False, "manifest seed does not match configuration")
    require(manifest.get("status") == "stopped", "run did not stop from a clean terminal phase")
    require(shadow_complete, "shadow baselines were not scored successfully")

    return {
        "schema_version": 2,
        "run_directory": str(run_dir.resolve()),
        "config_digest": manifest.get("config_digest"),
        "manifest_status": manifest.get("status"),
        "integrity": {
            "event_count": len(events),
            "event_indices_contiguous": contiguous,
            "event_indices_start_at_one": starts_at_one,
            "checkpoint_present": (run_dir / "checkpoint.pkl").is_file(),
            "config_snapshot_present": snapshot_path.is_file(),
            "config_digest_matches_snapshot": snapshot_digest == manifest.get("config_digest"),
            "unresolved_execution_ids": unresolved_ids,
            "emergency_stop_count": len(emergency_stops),
            "aborted_episode_count": len(aborted),
            "trial_metadata_complete": metadata_complete,
            "trial_ids_unique": trial_ids_unique,
            "schedule_snapshot_present": schedule_path.is_file(),
            "schedule_digest_matches": schedule_digest_matches,
            "schedule_manifest_matches": schedule_manifest_matches,
            "schedule_complete": schedule_complete,
            "episode_records_valid": episode_records_valid,
            "robot_postconditions_complete": robot_postconditions_complete,
            "schedule_error": schedule_error,
            "calibration_artifacts": artifact_checks,
            "publication_eligible": not reasons,
            "publication_ineligibility_reasons": reasons,
        },
        "outcomes": {
            "episodes_completed": len(episodes),
            "execution_or_commit_failures": len(failures),
            **_summarize_decisions(decisions),
            "by_condition": _grouped(episodes, "condition"),
            "by_preference": _grouped(episodes, "preference_id"),
            "by_recipe": {
                recipe: {
                    "episodes": sum(row.get("external_recipe_id") == recipe for row in episodes),
                    **_summarize_decisions([
                        decision for row in episodes
                        if row.get("external_recipe_id") == recipe
                        for decision in row.get("decisions", ())
                    ]),
                }
                for recipe in sorted({str(row.get("external_recipe_id")) for row in episodes})
            },
        },
        "episodes": [
            {key: value for key, value in row.items() if key != "decisions"}
            for row in episodes
        ],
        "shadow_baselines": shadow,
        "shadow_baseline_interpretation": (
            "Teacher-forced prediction scores on the realized Full-system trace; "
            "not baseline-controlled motion or causal human-effort estimates."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize and integrity-check one real-robot run")
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--require-publication-eligible", action="store_true",
        help="Exit nonzero unless every publication integrity gate passes",
    )
    args = parser.parse_args(argv)
    report = build_report(Path(args.run_dir))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["integrity"]["event_indices_contiguous"]:
        return 2
    if args.require_publication_eligible and not report["integrity"]["publication_eligible"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
