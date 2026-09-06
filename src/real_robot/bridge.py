"""Local, idempotent HTTP bridge for the Stretch SDK process."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from .config import ActionSpec, LabConfig, load_lab_config
from .stretch_hardware import BridgeDryRunController, StretchHardwareController


TERMINAL_STATUSES = {"completed", "failed", "stopped", "reconciled_after_restart"}
EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,127}$")


class BridgeConflict(RuntimeError):
    pass


class ProcessLock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a+")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            raise RuntimeError(f"another robot bridge owns {path}") from exc
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(f"{os.getpid()}\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()


class ExecutionLedger:
    """Append-only bridge truth for idempotent physical action execution."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._active_id: str | None = None
        self._active_worker: threading.Thread | None = None
        if path.exists():
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"corrupt execution ledger line {line_number}") from exc
                    self._records[str(row["execution_id"])] = dict(row)

    @property
    def unresolved_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(
                execution_id for execution_id, row in self._records.items()
                if row.get("status") not in TERMINAL_STATUSES
            ))

    def _append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(row)
        record["updated_at"] = time.time()
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._records[str(record["execution_id"])] = record
        return dict(record)

    def reconcile_after_restart(self, execution_id: str) -> None:
        with self._lock:
            row = self._records.get(execution_id)
            if row is None or row.get("status") in TERMINAL_STATUSES:
                raise RuntimeError(f"{execution_id!r} is not an unresolved execution")
            self._append({
                **row, "status": "reconciled_after_restart", "success": False,
                "message": "operator inspected scene and reconciled robot pose after bridge restart",
                "restart_required": False,
            })

    def get(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._records.get(execution_id)
            return None if row is None else dict(row)

    @staticmethod
    def _request_hash(request: Mapping[str, Any]) -> str:
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def submit(
        self, request: Mapping[str, Any], action: ActionSpec, controller: Any,
    ) -> tuple[dict[str, Any], bool]:
        execution_id = str(request["execution_id"])
        request_hash = self._request_hash(request)
        with self._lock:
            previous = self._records.get(execution_id)
            if previous is not None:
                if previous.get("request_hash") != request_hash:
                    raise BridgeConflict("execution_id was already used for a different request")
                return dict(previous), False
            if self._active_id is not None:
                raise BridgeConflict(f"robot is busy with execution {self._active_id}")
            controller_status = dict(controller.status())
            if not controller_status.get("ready", False):
                raise BridgeConflict(
                    f"robot controller is not ready: {controller_status.get('last_error') or controller_status.get('current_phase') or 'unknown reason'}"
                )
            record = self._append({
                "execution_id": execution_id, "request_hash": request_hash,
                "request": dict(request), "status": "accepted", "success": False,
                "message": "execution accepted", "accepted_at": time.time(),
            })
            self._active_id = execution_id
            worker = threading.Thread(
                target=self._run, args=(record, action, controller), daemon=True,
                name=f"stretch-execution-{execution_id[:12]}",
            )
            self._active_worker = worker
            worker.start()
            return record, True

    def _run(self, record: Mapping[str, Any], action: ActionSpec, controller: Any) -> None:
        execution_id = str(record["execution_id"])
        request = dict(record["request"])
        with self._lock:
            self._append({**record, "status": "running", "message": "physical execution running", "started_at": time.time()})
        try:
            result = dict(controller.execute(
                action, execution_id=execution_id,
                placement_slot_id=str(request["placement_slot_id"]),
            ))
        except Exception as exc:
            result = {
                "success": False, "status": "failed", "message": str(exc),
                "restart_required": True,
            }
        status = str(result.get("status", "failed"))
        if status not in TERMINAL_STATUSES:
            status = "completed" if result.get("success") else "failed"
        terminal = {
            **record, **result, "execution_id": execution_id, "status": status,
            "success": bool(result.get("success", False)), "finished_at": time.time(),
        }
        with self._lock:
            self._append(terminal)
            if self._active_id == execution_id:
                self._active_id = None
            if self._active_worker is threading.current_thread():
                self._active_worker = None

    def wait_for_idle(self, timeout_s: float) -> bool:
        with self._lock:
            worker = self._active_worker
        if worker is None:
            return True
        worker.join(timeout=max(0.0, float(timeout_s)))
        return not worker.is_alive()

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "active_execution_id": self._active_id,
                "unresolved_execution_ids": list(self.unresolved_ids),
                "execution_count": len(self._records),
            }


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], controller: Any, config: LabConfig, ledger: ExecutionLedger):
        super().__init__(address, BridgeHandler)
        self.controller = controller
        self.lab_config = config
        self.ledger = ledger


class BridgeHandler(BaseHTTPRequestHandler):
    server: BridgeServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Mapping[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > 32_000:
            raise ValueError("request too large")
        value = json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")
        if not isinstance(value, Mapping):
            raise ValueError("JSON object required")
        return value

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/v1/status":
            payload = {**dict(self.server.controller.status()), **dict(self.server.ledger.status())}
            payload["config_digest"] = self.server.lab_config.digest
            if payload.get("active_execution_id") is not None:
                payload["ready"] = False
            self._json(HTTPStatus.OK, payload)
        elif path.startswith("/v1/executions/"):
            execution_id = unquote(path.removeprefix("/v1/executions/"))
            row = self.server.ledger.get(execution_id)
            self._json(HTTPStatus.OK, row) if row is not None else self._json(HTTPStatus.NOT_FOUND, {"status": "not_found", "message": "unknown execution_id"})
        elif path == "/v1/camera.jpg":
            body = self.server.controller.camera_jpeg()
            if body is None:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "camera_unavailable", "message": "camera frame unavailable"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"status": "not_found", "message": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path == "/v1/emergency-stop":
                self._json(HTTPStatus.OK, dict(self.server.controller.emergency_stop()))
                return
            if path != "/v1/executions":
                self._json(HTTPStatus.NOT_FOUND, {"status": "not_found", "message": "not found"})
                return
            body = self._body()
            execution_id = str(body.get("execution_id", ""))
            if not EXECUTION_ID_RE.fullmatch(execution_id):
                raise ValueError("execution_id must be 8-128 safe identifier characters")
            if str(body.get("config_digest", "")) != self.server.lab_config.digest:
                raise BridgeConflict("client and bridge configuration digests differ")
            token = str(body.get("token", "")).strip().upper()
            action = self.server.lab_config.actions.get(token)
            if action is None:
                raise ValueError(f"unknown action {token!r}")
            expected = {
                "object_id": action.object_id, "source_station": action.source_station,
                "destination_station": action.destination_station,
            }
            for key, value in expected.items():
                if str(body.get(key, "")) != value:
                    raise BridgeConflict(f"client {key} disagrees with bridge configuration")
            slot_id = str(body.get("placement_slot_id", ""))
            if slot_id not in self.server.lab_config.placement_slots:
                raise ValueError(f"unknown placement slot {slot_id!r}")
            canonical = {
                "execution_id": execution_id, "config_digest": self.server.lab_config.digest,
                "token": action.token, **expected, "placement_slot_id": slot_id,
            }
            result, created = self.server.ledger.submit(canonical, action, self.server.controller)
            self._json(HTTPStatus.ACCEPTED if created else HTTPStatus.OK, result)
        except BridgeConflict as exc:
            self._json(HTTPStatus.CONFLICT, {"success": False, "status": "conflict", "message": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"success": False, "status": "invalid_request", "message": str(exc)})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "status": "bridge_error", "message": str(exc)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Stretch 3 hardware bridge")
    parser.add_argument("--config", default="src/real_robot/robot_configs/stretch3_lab.json")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--state-dir", default="src/real_robot/real_robot_bridge_state")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--camera-preview", action="store_true", help="Open the D405 in dry-run mode; robot actions remain simulated")
    parser.add_argument("--calibration-mode", action="store_true", help="Permit supervised single-action probes with an uncalibrated config")
    parser.add_argument("--confirm-start-station", default="")
    parser.add_argument("--acknowledge-reconciled", action="append", default=[])
    parser.add_argument("--allow-remote-hardware-api", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.camera_preview and args.enable_motion:
        raise SystemExit("--camera-preview is for dry runs only; live mode already owns the camera")
    if args.bind not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote_hardware_api:
        raise SystemExit("refusing a non-loopback hardware API without --allow-remote-hardware-api")
    config = load_lab_config(args.config)
    if args.calibration_mode and not args.enable_motion:
        raise SystemExit("--calibration-mode requires --enable-motion")
    state_dir = Path(args.state_dir)
    # A path cleanup must not hide an existing execution ledger or its lock.
    relocated_root = Path("src/real_robot/real_robot_bridge_state").resolve()
    if state_dir.resolve().is_relative_to(relocated_root):
        legacy_root = Path("real_robot_bridge_state")
        if legacy_root.exists():
            raise SystemExit(
                f"existing bridge state at {legacy_root}; stop the bridge and move "
                "that directory to src/real_robot/real_robot_bridge_state before restarting. "
                "See src/real_robot/README.md"
            )
    process_lock = ProcessLock(state_dir / "bridge.lock")
    ledger = ExecutionLedger(state_dir / "executions.jsonl")
    try:
        for execution_id in args.acknowledge_reconciled:
            ledger.reconcile_after_restart(execution_id)
        if args.enable_motion and ledger.unresolved_ids:
            unresolved = ", ".join(ledger.unresolved_ids)
            raise RuntimeError(
                f"unresolved executions require scene/pose inspection and --acknowledge-reconciled for each ID: {unresolved}"
            )
        controller = (
            StretchHardwareController(
                config, confirmed_start_station=args.confirm_start_station,
                calibration_mode=args.calibration_mode,
            )
            if args.enable_motion else BridgeDryRunController(config, camera_preview=args.camera_preview)
        )
        try:
            server = BridgeServer((args.bind, args.port), controller, config, ledger)
        except BaseException:
            controller.close()
            raise
        print(f"Stretch bridge: http://{args.bind}:{args.port}")
        print(f"Motion enabled: {bool(args.enable_motion)}")
        print(f"Dry-run camera preview: {bool(args.camera_preview)}")
        print(f"Calibration mode: {bool(args.calibration_mode)}")
        print(f"Config digest: {config.digest}")
        print(f"State directory: {state_dir}")

        shutdown_started = threading.Event()

        def request_shutdown(_signum=None, _frame=None):
            if shutdown_started.is_set():
                return
            shutdown_started.set()
            controller.emergency_stop()
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            server.server_close()
            ledger.wait_for_idle(config.motion.command_timeout_s + 5.0)
            controller.close()
    finally:
        process_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
