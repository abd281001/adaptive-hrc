"""Dependency-free operator web UI for live Adaptive-HRC sessions."""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pickle
import platform
import secrets
import signal
import subprocess
import sys
import threading
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from src.adaptive_agent import AdaptiveAgent
from src.models import Settings

from .config import LabConfig, load_lab_config
from .domain import PhysicalTaskDomain
from .hardware import DryRunExecutor, HttpStretchExecutor
from .schedule import StudySchedule, load_study_schedule
from .session import LiveHrcSession, SessionStateError


class EventJournal:
    """Append-only event log with a run manifest."""

    def __init__(
        self, root: Path, config: LabConfig, *, hardware_backend: str,
        resumed_from: str | None = None,
        hardware_preflight: Mapping[str, Any] | None = None,
        require_motion: bool = False,
        schedule: StudySchedule | None = None,
        publication_run: bool = False,
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.events_path = self.root / "events.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.Lock()
        self._agent: Any = None
        self._session: LiveHrcSession | None = None
        self._config = config
        self._closed = False
        _atomic_write_text(
            self.root / "config.snapshot.json",
            json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n",
        )
        if schedule is not None:
            _atomic_write_text(
                self.root / "schedule.snapshot.json",
                json.dumps(schedule.as_dict(), indent=2, sort_keys=True) + "\n",
            )
        calibration_artifacts: dict[str, Any] = {}
        if config.motion.calibrated:
            artifacts = (
                ("calibration_record", config.calibration_record_path, config.motion.calibration_record_sha256, "calibration.record.json"),
                ("runtime_lock", config.runtime_lock_path, config.motion.runtime_lock_sha256, "runtime.lock.json"),
            )
            for label, source_path, digest, filename in artifacts:
                _atomic_write_bytes(self.root / filename, Path(source_path).read_bytes())
                calibration_artifacts[label] = {"file": filename, "sha256": digest}
        software = _software_provenance()
        manifest = {
            "schema_version": 3,
            "run_id": self.root.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "config_digest": config.digest,
            "calibration_id": config.motion.calibration_id,
            "hardware_backend": hardware_backend,
            "resumed_from": resumed_from,
            "resumed_from_sha256": (
                __import__("hashlib").sha256(Path(resumed_from).read_bytes()).hexdigest()
                if resumed_from else None
            ),
            "parent_run_id": Path(resumed_from).resolve().parent.name if resumed_from else None,
            "require_motion": bool(require_motion),
            "publication_run": bool(publication_run),
            "seed": int(config.agent_settings.get("seed", 1337)),
            "study_schedule": (
                None if schedule is None else {
                    "schedule_id": schedule.schedule_id,
                    "participant_id": schedule.participant_id,
                    "counterbalance_id": schedule.counterbalance_id,
                    "digest": schedule.digest,
                    "episode_count": len(schedule.episodes),
                }
            ),
            "calibration_artifacts": calibration_artifacts,
            "hardware_preflight": dict(hardware_preflight or {}),
            "software": software,
            "pid": os.getpid(),
        }
        _atomic_write_text(
            self.manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    def append(self, event: Mapping[str, Any]) -> None:
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        if event.get("event") in {
            "robot_execution_succeeded", "learner_transition_committed",
            "episode_completed", "execution_reconciled_completed",
            "execution_reconciled_not_completed",
        }:
            self.checkpoint()

    def bind(self, agent: Any, session: LiveHrcSession) -> None:
        self._agent = agent
        self._session = session

    def checkpoint(self) -> None:
        if self._agent is None or self._session is None:
            return
        payload = {
            "schema_version": 2,
            "config_digest": self._config.digest,
            "agent": self._agent,
            "observed_recipes": self._session.observed_recipes,
            "session": self._session.checkpoint_state(),
        }
        target = self.root / "checkpoint.pkl"
        temporary = self.root / f".checkpoint-{os.getpid()}.tmp"
        with self._lock:
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _fsync_directory(self.root)

    def close(self, *, status: str = "stopped") -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = status
            manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_text(
                self.manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}-{os.getpid()}-{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}-{os.getpid()}-{threading.get_ident()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _software_provenance() -> Mapping[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], check=True, capture_output=True, text=True,
                timeout=5.0,
            ).stdout.strip()
        except Exception:
            return "unavailable"

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status and status != "unavailable"),
        "git_status_sha256": __import__("hashlib").sha256(status.encode()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def _load_checkpoint_full(path: Path, config: LabConfig) -> tuple[Any, tuple[str, ...], Mapping[str, Any] | None]:
    # Pickle is intentionally restricted to an explicit local experiment file.
    # Never load checkpoints received from another person or untrusted source.
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping) or payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported real-robot checkpoint")
    if payload.get("config_digest") != config.digest:
        raise ValueError("checkpoint configuration does not match --config")
    agent = payload.get("agent")
    observed = tuple(map(str, payload.get("observed_recipes", ())))
    if agent is None or getattr(agent, "domain", None) is None:
        raise ValueError("checkpoint does not contain an Adaptive-HRC agent")
    session_state = payload.get("session")
    return agent, observed, session_state if isinstance(session_state, Mapping) else None


def _load_checkpoint(path: Path, config: LabConfig) -> tuple[Any, tuple[str, ...]]:
    agent, observed, _session_state = _load_checkpoint_full(path, config)
    return agent, observed


def _settings(config: LabConfig) -> Settings:
    known = {field.name for field in fields(Settings) if field.init}
    unknown = set(config.agent_settings) - known
    if unknown:
        raise ValueError(f"unknown agent_settings: {sorted(unknown)}")
    return Settings(**dict(config.agent_settings))


class OperatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        session: LiveHrcSession,
        config: LabConfig,
        operator_token: str,
    ):
        super().__init__(address, OperatorHandler)
        self.session = session
        self.lab_config = config
        self.operator_token = operator_token


class OperatorHandler(BaseHTTPRequestHandler):
    server: OperatorServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Mapping[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise SessionStateError("invalid Content-Length") from exc
        if length < 0 or length > 64_000:
            raise SessionStateError("request body is too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionStateError("request body must be a JSON object") from exc
        if not isinstance(value, Mapping):
            raise SessionStateError("request body must be a JSON object")
        return value

    def _authorized(self) -> bool:
        header_token = self.headers.get("X-HRC-Operator-Token", "")
        query_token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
        return any(
            secrets.compare_digest(candidate, self.server.operator_token)
            for candidate in (header_token, query_token) if candidate
        )

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = Path(__file__).with_name("operator.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/") and not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid operator token"})
            return
        if path == "/api/state":
            self._json(HTTPStatus.OK, self.server.session.snapshot())
            return
        if path == "/api/camera.jpg":
            body = self.server.session.executor.camera_jpeg()
            if body is None:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "camera frame unavailable"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/config":
            config = self.server.lab_config
            self._json(HTTPStatus.OK, {
                "recipes": {
                    key: {"label": recipe.label, "actions": list(recipe.actions)}
                    for key, recipe in config.recipes.items()
                },
                "actions": {
                    key: {"label": action.label, "role": action.role}
                    for key, action in config.actions.items()
                },
                "stations": {
                    key: {"label": station.label, "heading_deg": station.heading_deg}
                    for key, station in config.stations.items()
                },
                "placement_slots": {
                    key: {"label": slot.label}
                    for key, slot in config.placement_slots.items()
                },
                "config_digest": config.digest,
            })
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid operator token"})
            return
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
            if path == "/api/episode/start":
                if not isinstance(body.get("props_reset"), bool):
                    raise SessionStateError("props_reset must be a JSON boolean")
                result = self.server.session.start_episode(
                    str(body.get("recipe_id", "")),
                    props_reset=body["props_reset"],
                    trial_metadata=(body.get("trial_metadata") if isinstance(body.get("trial_metadata"), Mapping) else {}),
                )
            elif path == "/api/human-action":
                if not isinstance(body.get("physical_completed"), bool):
                    raise SessionStateError("physical_completed must be a JSON boolean")
                result = self.server.session.record_human_action(
                    str(body.get("action", "")), physical_completed=body["physical_completed"],
                )
            elif path == "/api/robot/approve":
                if not isinstance(body.get("human_clear"), bool):
                    raise SessionStateError("human_clear must be a JSON boolean")
                result = self.server.session.approve_robot_proposal(human_clear=body["human_clear"])
            elif path == "/api/robot/correct":
                if not isinstance(body.get("physical_completed"), bool):
                    raise SessionStateError("physical_completed must be a JSON boolean")
                result = self.server.session.reject_robot_proposal(
                    str(body.get("action", "")), physical_completed=body["physical_completed"],
                )
            elif path == "/api/robot/reconcile":
                if not isinstance(body.get("execution_completed"), bool) or not isinstance(body.get("physical_scene_verified"), bool):
                    raise SessionStateError("reconciliation confirmations must be JSON booleans")
                result = self.server.session.reconcile_robot_execution(
                    execution_completed=body["execution_completed"],
                    physical_scene_verified=body["physical_scene_verified"],
                )
            elif path == "/api/episode/abort":
                result = self.server.session.abort_episode()
            elif path == "/api/emergency-stop":
                result = self.server.session.emergency_stop()
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._json(HTTPStatus.ACCEPTED, result)
        except SessionStateError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc), "state": self.server.session.snapshot()})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def _run_directory(root: Path, requested: str | None) -> Path:
    if requested:
        return root / requested
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"real-robot__{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive-HRC real-robot operator UI")
    parser.add_argument("--config", default="robot_configs/stretch3_lab.json")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hardware-url", default="", help="Local Stretch bridge URL; empty uses dry-run")
    parser.add_argument("--hardware-timeout", type=float, default=180.0)
    parser.add_argument("--require-motion", action="store_true", help="Refuse startup unless the bridge is live, ready, and digest-matched")
    parser.add_argument("--dry-run-delay", type=float, default=0.25)
    parser.add_argument("--operator-token", default="")
    parser.add_argument("--output", default="real_robot_runs")
    parser.add_argument("--run", default="")
    parser.add_argument("--resume-checkpoint", default="", help="Trusted local action-level checkpoint.pkl")
    parser.add_argument("--schedule", default="", help="Frozen participant-specific study schedule JSON")
    parser.add_argument("--publication-run", action="store_true", help="Require live calibrated hardware, a frozen schedule, and a clean identified revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_lab_config(args.config)
    schedule = load_study_schedule(args.schedule, config) if args.schedule else None
    if args.publication_run:
        if not args.require_motion:
            raise SystemExit("--publication-run requires --require-motion")
        if schedule is None:
            raise SystemExit("--publication-run requires --schedule")
        if args.resume_checkpoint:
            raise SystemExit(
                "--publication-run requires a fresh continuous run; preserve resumed runs as failure/recovery evidence"
            )
        if not config.motion.calibrated:
            raise SystemExit("--publication-run requires a calibrated configuration")
        software = _software_provenance()
        if software["git_commit"] == "unavailable" or software["git_dirty"]:
            raise SystemExit("--publication-run requires a clean, identified Git revision")
    observed: tuple[str, ...] = ()
    session_state: Mapping[str, Any] | None = None
    if args.resume_checkpoint:
        agent, observed, session_state = _load_checkpoint_full(Path(args.resume_checkpoint), config)
        domain = agent.domain
    else:
        domain = PhysicalTaskDomain(config)
        agent = AdaptiveAgent(settings=_settings(config), domain=domain)
    executor = (
        HttpStretchExecutor(
            args.hardware_url,
            timeout_s=max(args.hardware_timeout, config.motion.action_timeout_s + 10.0),
        )
        if args.hardware_url else DryRunExecutor(delay_s=args.dry_run_delay)
    )
    if args.require_motion and not args.hardware_url:
        raise SystemExit("--require-motion requires --hardware-url")
    hardware_preflight: Mapping[str, Any]
    if isinstance(executor, HttpStretchExecutor):
        hardware_preflight = executor.preflight(config.digest, require_motion=args.require_motion)
    else:
        hardware_preflight = executor.status()
    run_dir = _run_directory(Path(args.output), args.run or None)
    journal = EventJournal(
        run_dir, config, hardware_backend=executor.name,
        resumed_from=args.resume_checkpoint or None,
        hardware_preflight=hardware_preflight,
        require_motion=args.require_motion,
        schedule=schedule,
        publication_run=args.publication_run,
    )
    session = LiveHrcSession(
        agent, domain, config, executor, event_sink=journal.append,
        schedule=schedule,
    )
    session.restore_observed_recipes(observed)
    journal.bind(agent, session)
    if session_state is not None:
        session.restore_checkpoint_state(session_state)
    token = args.operator_token or secrets.token_urlsafe(18)
    server = OperatorServer((args.bind, args.port), session, config, token)
    print(f"Operator UI: http://{args.bind}:{args.port}")
    print(f"Operator token: {token}")
    print(f"Hardware backend: {executor.name}; motion_enabled={executor.motion_enabled}")
    print(f"Run directory: {run_dir}")
    if args.bind not in {"127.0.0.1", "localhost", "::1"}:
        print("Lab-network binding enabled. Share the operator token only with the experimenter.")
    shutdown_started = threading.Event()

    def request_shutdown(_signum=None, _frame=None):
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        if session.snapshot()["phase"] == "robot_executing":
            session.emergency_stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        session.shutdown(timeout_s=config.motion.command_timeout_s + 5.0)
        final_phase = session.snapshot()["phase"]
        journal.close(status="stopped" if final_phase in {"idle", "complete"} else f"stopped_with_{final_phase}")
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
