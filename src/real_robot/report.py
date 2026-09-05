"""Generate a deterministic integrity and outcome report for one robot run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

from .config import load_lab_config


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
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


def build_report(run_dir: Path) -> Mapping[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot_path = run_dir / "config.snapshot.json"
    snapshot_digest = load_lab_config(snapshot_path).digest if snapshot_path.is_file() else None
    events = []
    with (run_dir / "events.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid events.jsonl line {line_number}") from exc
            events.append(row)
    indices = [int(row["event_index"]) for row in events]
    contiguous = indices == list(range(indices[0], indices[0] + len(indices))) if indices else True
    completed = [row for row in events if row.get("event") == "episode_completed"]
    decisions = [
        row["payload"] for row in events
        if row.get("event") in {"learner_transition_committed", "episode_completed"}
    ]
    robot_rows = [row for row in decisions if row.get("executed_by") == "robot"]
    correction_rows = [row for row in decisions if row.get("executed_by") == "human_correction"]
    robot_turns = len(robot_rows) + len(correction_rows)
    failures = [row for row in events if row.get("event") in {
        "robot_execution_failed", "robot_execution_ambiguous", "learner_commit_failed",
    }]
    emergency_stops = [row for row in events if row.get("event") == "emergency_stop"]
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
    durations = [
        float(row["execution"]["elapsed_s"])
        for row in robot_rows
        if isinstance(row.get("execution"), Mapping) and row["execution"].get("elapsed_s") is not None
    ]
    return {
        "schema_version": 1, "run_directory": str(run_dir.resolve()),
        "config_digest": manifest.get("config_digest"),
        "manifest_status": manifest.get("status"),
        "integrity": {
            "event_count": len(events), "event_indices_contiguous": contiguous,
            "checkpoint_present": (run_dir / "checkpoint.pkl").is_file(),
            "config_snapshot_present": snapshot_path.is_file(),
            "config_digest_matches_snapshot": snapshot_digest == manifest.get("config_digest"),
            "unresolved_execution_ids": unresolved_ids,
            "emergency_stop_count": len(emergency_stops),
            "publication_eligible": bool(
                contiguous and snapshot_digest == manifest.get("config_digest")
                and not unresolved_ids and not failures and not emergency_stops
                and manifest.get("require_motion")
                and manifest.get("calibration_id")
                and not manifest.get("software", {}).get("git_dirty", True)
                and manifest.get("status") == "stopped"
            ),
        },
        "outcomes": {
            "episodes_completed": len(completed), "actions_committed": len(decisions),
            "robot_turns": robot_turns, "robot_completed": len(robot_rows),
            "human_corrections": len(correction_rows),
            "robot_top_1": len(robot_rows) / robot_turns if robot_turns else None,
            "execution_or_commit_failures": len(failures),
            "robot_action_duration_s": _describe(durations),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize and integrity-check one real-robot run")
    parser.add_argument("run_dir")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = build_report(Path(args.run_dir))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["integrity"]["event_indices_contiguous"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
