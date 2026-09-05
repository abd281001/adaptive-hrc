"""Hardware transport boundary used by the live HRC session."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import threading
import time
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import ActionSpec


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    status: str
    message: str
    started_at: float
    finished_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    monotonic_elapsed_s: float | None = None

    @property
    def elapsed_s(self) -> float:
        if self.monotonic_elapsed_s is not None:
            return max(0.0, float(self.monotonic_elapsed_s))
        return max(0.0, self.finished_at - self.started_at)

    def as_dict(self) -> Mapping[str, Any]:
        return {**asdict(self), "elapsed_s": self.elapsed_s}


class HardwareExecutor(Protocol):
    name: str
    motion_enabled: bool

    def execute(self, action: ActionSpec, *, execution_id: str, placement_slot_id: str, config_digest: str) -> ExecutionResult:
        """Execute one idempotently identified action and report its terminal state."""

    def status(self) -> Mapping[str, Any]: ...
    def emergency_stop(self) -> Mapping[str, Any]: ...
    def camera_jpeg(self) -> bytes | None: ...
    def close(self) -> None: ...


class DryRunExecutor:
    """Deterministic backend for end-to-end UI and protocol validation."""

    name = "dry_run"
    motion_enabled = False

    def __init__(self, *, delay_s: float = 0.0):
        self.delay_s = max(0.0, float(delay_s))
        self.executed: list[str] = []
        self._results: dict[str, ExecutionResult] = {}
        self._stopped = False
        self._lock = threading.Lock()

    def execute(self, action: ActionSpec, *, execution_id: str = "dry-run-execution", placement_slot_id: str = "slot_1", config_digest: str = "") -> ExecutionResult:
        started = time.time()
        started_monotonic = time.monotonic()
        with self._lock:
            previous = self._results.get(execution_id)
            if previous is not None:
                return previous
        if self.delay_s:
            time.sleep(self.delay_s)
        with self._lock:
            if self._stopped:
                result = ExecutionResult(False, "stopped", "dry-run executor is emergency-stopped", started, time.time(), {"execution_id": execution_id, "action": action.token}, time.monotonic() - started_monotonic)
            else:
                self.executed.append(action.token)
                result = ExecutionResult(True, "completed", f"dry-run completed {action.token}", started, time.time(), {
                    "execution_id": execution_id, "action": action.token,
                    "placement_slot_id": placement_slot_id,
                    "config_digest": config_digest, "postcondition": {"dry_run": True},
                }, time.monotonic() - started_monotonic)
            self._results[execution_id] = result
            return result

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "backend": self.name, "motion_enabled": False,
                "ready": not self._stopped, "reachable": True,
                "stopped": self._stopped, "executed_actions": list(self.executed),
            }

    def emergency_stop(self) -> Mapping[str, Any]:
        with self._lock:
            self._stopped = True
        return {"ok": True, "status": "stopped", "motion_enabled": False}

    def camera_jpeg(self) -> bytes | None:
        return None

    def close(self) -> None:
        return None


class BridgeHttpError(RuntimeError):
    def __init__(self, code: int, payload: Mapping[str, Any]):
        self.code = int(code)
        self.payload = dict(payload)
        super().__init__(str(payload.get("message") or payload.get("error") or f"Stretch bridge HTTP {code}"))


class HttpStretchExecutor:
    """Idempotent client for the local asynchronous Stretch bridge."""

    name = "stretch_http_bridge"

    def __init__(self, base_url: str = "http://127.0.0.1:9100", *, timeout_s: float = 180.0):
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = max(5.0, float(timeout_s))
        self.motion_enabled = False
        self._status_lock = threading.Lock()
        self._status_cache: dict[str, Any] = {
            "backend": self.name, "motion_enabled": False,
            "reachable": False, "ready": False, "error": "status not checked",
        }
        self._stop_refresh = threading.Event()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True, name="stretch-status-cache")
        self._refresh_thread.start()

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None, *, timeout_s: float = 2.0) -> Mapping[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=body, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=max(0.1, float(timeout_s))) as response:
                raw = response.read().decode("utf-8")
                result = json.loads(raw)
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload_value = json.loads(raw)
            except json.JSONDecodeError:
                payload_value = {"status": "http_error", "message": raw or str(exc)}
            if not isinstance(payload_value, Mapping):
                payload_value = {"status": "http_error", "message": str(payload_value)}
            raise BridgeHttpError(exc.code, payload_value) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Stretch bridge unavailable at {self.base_url}: {exc}") from exc
        if not isinstance(result, Mapping):
            raise RuntimeError("Stretch bridge returned a non-object response")
        return result

    def _set_status(self, status: Mapping[str, Any]) -> None:
        row = dict(status)
        row["reachable"] = True
        with self._status_lock:
            self._status_cache = row
            self.motion_enabled = bool(row.get("motion_enabled", False))

    def _refresh_once(self) -> Mapping[str, Any]:
        try:
            status = self._request("GET", "/v1/status", timeout_s=1.5)
            self._set_status(status)
            return status
        except Exception as exc:
            with self._status_lock:
                previous = dict(self._status_cache)
                previous.update({"reachable": False, "ready": False, "error": str(exc)})
                self._status_cache = previous
            return previous

    def _refresh_loop(self) -> None:
        while not self._stop_refresh.wait(0.5):
            self._refresh_once()

    def preflight(self, config_digest: str, *, require_motion: bool, allow_calibration_mode: bool = False) -> Mapping[str, Any]:
        status = dict(self._refresh_once())
        if not status.get("reachable", True):
            raise RuntimeError(str(status.get("error", "Stretch bridge is unreachable")))
        if status.get("config_digest") != config_digest:
            raise RuntimeError("operator and Stretch bridge configuration digests differ")
        if not status.get("ready"):
            raise RuntimeError(f"Stretch bridge is not ready: {status.get('last_error') or status.get('error') or status.get('current_phase')}")
        if require_motion and not status.get("motion_enabled"):
            raise RuntimeError("--require-motion was requested but the bridge is dry-run")
        if status.get("motion_enabled") and not require_motion:
            raise RuntimeError("bridge has live motion enabled; pass --require-motion to authorize a live UI")
        if status.get("calibration_mode") and not allow_calibration_mode:
            raise RuntimeError("bridge is in calibration mode and cannot run the experiment UI")
        return status

    def execute(self, action: ActionSpec, *, execution_id: str, placement_slot_id: str, config_digest: str) -> ExecutionResult:
        started_wall = time.time()
        started_monotonic = time.monotonic()
        deadline = started_monotonic + self.timeout_s
        payload = {
            "execution_id": execution_id, "config_digest": config_digest,
            "token": action.token, "object_id": action.object_id,
            "source_station": action.source_station,
            "destination_station": action.destination_station,
            "placement_slot_id": placement_slot_id,
        }
        structured_failure: Mapping[str, Any] | None = None
        try:
            response = self._request("POST", "/v1/executions", payload, timeout_s=min(3.0, max(0.1, deadline - time.monotonic())))
            if response.get("status") in {"conflict", "invalid_request", "bridge_error"}:
                structured_failure = response
        except BridgeHttpError as exc:
            structured_failure = exc.payload
        except Exception:
            # Submission may have reached the bridge.  Never resubmit with a
            # different ID; query this execution ID until the action deadline.
            pass
        if structured_failure is not None:
            return ExecutionResult(False, str(structured_failure.get("status", "bridge_error")), str(structured_failure.get("message", "bridge rejected execution")), started_wall, time.time(), dict(structured_failure), time.monotonic() - started_monotonic)

        last_error = "execution status unavailable"
        path = f"/v1/executions/{quote(execution_id, safe='')}"
        while time.monotonic() < deadline:
            try:
                row = self._request("GET", path, timeout_s=min(2.0, max(0.1, deadline - time.monotonic())))
                status = str(row.get("status", "unknown"))
                if status in {"completed", "failed", "stopped", "reconciled_after_restart"}:
                    success = bool(row.get("success", False)) and status == "completed"
                    duration = row.get("elapsed_s")
                    return ExecutionResult(success, status, str(row.get("message", status)), started_wall, time.time(), dict(row), float(duration) if duration is not None else time.monotonic() - started_monotonic)
                last_error = f"bridge reports {status}"
            except BridgeHttpError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)
            time.sleep(min(0.20, max(0.0, deadline - time.monotonic())))
        return ExecutionResult(False, "execution_unknown", f"action deadline expired; {last_error}. Inspect bridge ledger and physical scene; do not execute again.", started_wall, time.time(), {
            "execution_id": execution_id, "action": action.token,
            "bridge": self.base_url, "restart_required": True,
        }, time.monotonic() - started_monotonic)

    def status(self) -> Mapping[str, Any]:
        with self._status_lock:
            status = dict(self._status_cache)
        status["client_status_thread_alive"] = self._refresh_thread.is_alive()
        return status

    def emergency_stop(self) -> Mapping[str, Any]:
        try:
            return dict(self._request("POST", "/v1/emergency-stop", {}, timeout_s=2.0))
        except BridgeHttpError as exc:
            return {"ok": False, **exc.payload}
        except Exception as exc:
            return {"ok": False, "status": "stop_request_failed", "error": str(exc)}

    def camera_jpeg(self) -> bytes | None:
        request = Request(f"{self.base_url}/v1/camera.jpg", method="GET")
        try:
            with urlopen(request, timeout=1.5) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError):
            return None

    def close(self) -> None:
        self._stop_refresh.set()
        self._refresh_thread.join(timeout=2.0)
