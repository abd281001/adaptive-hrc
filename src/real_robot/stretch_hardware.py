"""Direct Stretch 3 backend using the packaged adaptive-HRC runtime.

Robot-only imports remain lazy so the project test environment does not need
the Stretch SDK or RealSense libraries.
"""
from __future__ import annotations

import importlib
import math
import os
import platform
import sys
import threading
import time
from typing import Any, Mapping

from .config import ActionSpec, LabConfig, PlacementSlotSpec
from .runtime_qualification import require_qualified_runtime


class StretchHardwareController:
    """Fail-closed marker pickup, rotation-only routing, and checked placement."""

    def __init__(self, config: LabConfig, *, confirmed_start_station: str, calibration_mode: bool = False):
        if not config.motion.calibrated and not calibration_mode:
            raise RuntimeError(
                "motion.calibrated is false; finish and identify a physical calibration before enabling motion"
            )
        if confirmed_start_station != config.home_station:
            raise RuntimeError(
                f"--confirm-start-station must equal configured home station {config.home_station!r}"
            )
        self.config = config
        self._calibration_mode = bool(calibration_mode)
        self._sdk_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._busy = False
        self._stopped = False
        self._stop_event = threading.Event()
        self._last_error: str | None = None
        self._last_action: str | None = None
        self._execution_id: str | None = None
        self._phase = "startup"
        self._last_successful_phase: str | None = None
        self._last_reference_evidence: Mapping[str, Any] | None = None
        self._velocity_controller_status: Mapping[str, Any] = {
            "thread_alive": False, "stop_requested": True,
            "error": None, "state": "not_started",
        }
        self._pose_confident = False
        self.current_station: str | None = None
        self.robot: Any = None
        self.camera_service: Any = None
        self._modules: dict[str, Any] = {}
        self._runtime_provenance: dict[str, Any] = {
            "python": sys.version, "platform": platform.platform(),
        }
        self._runtime_qualification: Mapping[str, Any] = {
            "qualified": False,
            "reason": "uncalibrated calibration probe" if calibration_mode else "not checked",
        }
        self._startup(confirmed_start_station)

    def _set_phase(self, phase: str, *, successful: bool = False) -> None:
        with self._state_lock:
            self._phase = phase
            if successful:
                self._last_successful_phase = phase

    def _set_velocity_controller_status(self, status: Mapping[str, Any]) -> None:
        with self._state_lock:
            self._velocity_controller_status = dict(status)

    def _startup(self, confirmed_start_station: str) -> None:
        try:
            for name in ("camera_service", "grasping", "routes", "delivery", "perception"):
                self._modules[name] = importlib.import_module(
                    f".stretch_runtime.{name}", package=__package__
                )
            robot_module = importlib.import_module("stretch_body.robot")
            stretch_package = importlib.import_module("stretch_body")
            self._runtime_provenance["stretch_body_version"] = getattr(stretch_package, "__version__", "unknown")
            self.robot = robot_module.Robot()
            if not self.robot.startup():
                raise RuntimeError("could not connect to Stretch hardware")
            try:
                homed = self.robot.is_homed()
            except AttributeError:
                homed = self.robot.is_calibrated()
            if not homed:
                raise RuntimeError(
                    "Stretch is not homed; home it explicitly with the standard robot procedure"
                )
            runstop = getattr(getattr(self.robot, "pimu", None), "status", {}).get("runstop_event")
            if runstop:
                raise RuntimeError("Stretch runstop is active")
            robot_params = getattr(self.robot, "params", {})
            if not isinstance(robot_params, Mapping):
                robot_params = {}
            nested_robot_params = robot_params.get("robot", {})
            if not isinstance(nested_robot_params, Mapping):
                nested_robot_params = {}
            robot_id = (
                os.environ.get("HELLO_FLEET_ID", "").strip()
                or str(robot_params.get("serial_no", "")).strip()
                or str(nested_robot_params.get("serial_no", "")).strip()
                or "unknown"
            )
            self._runtime_provenance.update({
                "robot_id": robot_id,
                "robot_serial": robot_params.get("serial_no", robot_id),
                "robot_batch": robot_params.get("batch_name", "unknown"),
                "robot_tool": robot_params.get("tool", "unknown"),
            })
            p = self.config.perception
            camera_type = self._modules["camera_service"].D405CameraService
            self.camera_service = camera_type(
                exposure="medium", capture_fps=15, stream_fps=10,
                startup_timeout_s=p.camera_startup_timeout_s,
                frame_timeout_ms=p.camera_frame_timeout_ms,
                stale_after_s=p.max_frame_age_s,
            )
            self.camera_service.start()
            camera_status = self.camera_service.get_status()
            if not camera_status.get("ready"):
                raise RuntimeError("D405 did not become ready")
            if self.config.motion.calibrated:
                self._runtime_qualification = require_qualified_runtime(
                    self.config.runtime_lock_data,
                    robot_id=robot_id,
                    camera_identity=dict(camera_status.get("device_identity", {})),
                )
                self._runtime_provenance.update({
                    "runtime_lock_sha256": self.config.motion.runtime_lock_sha256,
                    "calibration_record_sha256": self.config.motion.calibration_record_sha256,
                })
            self.current_station = confirmed_start_station
            if self.config.motion.calibrated and not self._calibration_mode:
                self._verify_station_reference(
                    confirmed_start_station,
                    time.monotonic() + self.config.perception.postcondition_timeout_s,
                )
            self._pose_confident = True
            self._set_phase("ready", successful=True)
        except Exception:
            self._safe_close_resources()
            raise

    @staticmethod
    def _shortest_rotation(target_deg: float, current_deg: float) -> float:
        delta = (float(target_deg) - float(current_deg) + 180.0) % 360.0 - 180.0
        return 180.0 if math.isclose(delta, -180.0) else delta

    @staticmethod
    def _angle_delta_deg(after_rad: float, before_rad: float) -> float:
        raw = math.degrees(float(after_rad) - float(before_rad))
        return (raw + 180.0) % 360.0 - 180.0

    def _check_cancelled(self, deadline: float) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("stop requested")
        if time.monotonic() >= deadline:
            raise RuntimeError("action-wide deadline exceeded")

    def _rotate_checked(self, delta_deg: float, label: str, deadline: float) -> None:
        self._check_cancelled(deadline)
        if abs(delta_deg) <= 0.25:
            return
        before = float(self.robot.base.status["theta"])
        ok = self._modules["routes"].rotate_deg(
            self.robot, delta_deg,
            timeout=self.config.motion.command_timeout_s,
            deadline=deadline, cancel_event=self._stop_event,
            velocity_scale=(min(self.config.motion.velocity_scale, 0.25) if self._calibration_mode else self.config.motion.velocity_scale),
        )
        after = float(self.robot.base.status["theta"])
        actual = self._angle_delta_deg(after, before)
        error = abs(self._shortest_rotation(delta_deg, actual))
        if not ok:
            self._pose_confident = False
            raise RuntimeError(f"base rotation {label} timed out")
        if error > self.config.motion.rotation_tolerance_deg:
            self._pose_confident = False
            raise RuntimeError(
                f"base rotation {label} error {error:.2f} deg exceeds calibrated tolerance"
            )

    def _move_to_station(self, station_id: str, deadline: float) -> None:
        if not self._pose_confident or self.current_station is None:
            raise RuntimeError("base pose is not reconciled")
        if station_id not in self.config.stations:
            raise RuntimeError(f"unknown station {station_id!r}")
        current = self.config.stations[self.current_station]
        target = self.config.stations[station_id]
        delta = self._shortest_rotation(target.heading_deg, current.heading_deg)
        self._rotate_checked(delta, f"to station {station_id}", deadline)
        self._verify_station_reference(station_id, deadline)
        # Logical pose advances only after odometry and the table-fixed
        # reference marker both agree with the commanded station.
        self.current_station = station_id

    def _verify_station_reference(self, station_id: str, deadline: float) -> Mapping[str, Any] | None:
        if self._calibration_mode or not self.config.motion.calibrated:
            return None
        marker = self.config.reference_marker
        p = self.config.perception
        try:
            evidence = self._modules["perception"].wait_for_stable_marker(
                camera_service=self.camera_service,
                marker_info=self._marker_info(),
                marker_id=marker.marker_id,
                expected_xyz_m=marker.expected_position_by_station_m[station_id],
                tolerance_m=marker.tolerance_m,
                stable_frames=p.stable_frames,
                max_position_jump_m=p.max_position_jump_m,
                max_frame_age_s=p.max_frame_age_s,
                deadline=min(deadline, time.monotonic() + p.postcondition_timeout_s),
                cancel_event=self._stop_event,
            )
            evidence = {**dict(evidence), "station_id": station_id}
            with self._state_lock:
                self._last_reference_evidence = evidence
            return evidence
        except Exception:
            self._pose_confident = False
            raise

    def _marker_info(self) -> Mapping[str, Any]:
        p = self.config.perception
        quality = {
            "min_marker_pixels": p.min_marker_pixels,
            "max_reprojection_error_px": p.max_reprojection_error_px,
            "min_marker_depth_m": p.min_marker_depth_m,
            "max_marker_depth_m": p.max_marker_depth_m,
        }
        left, right = self.config.fingertip_marker_ids
        rows: dict[str, Any] = {
            str(left): {"length_mm": p.fingertip_marker_length_mm, "use_rgb_only": True, "name": "finger_left", "link": "link_finger_left", "type": "aruco", **quality},
            str(right): {"length_mm": p.fingertip_marker_length_mm, "use_rgb_only": True, "name": "finger_right", "link": "link_finger_right", "type": "aruco", **quality},
            "default": {"length_mm": 24.0, "use_rgb_only": True, "name": "unknown", "link": "None", "type": "aruco", **quality},
        }
        for item in self.config.objects.values():
            rows[str(item.marker_id)] = {
                "length_mm": item.marker_length_mm, "use_rgb_only": True,
                "name": item.object_id, "link": "None", "type": "aruco", **quality,
            }
        reference = self.config.reference_marker
        rows[str(reference.marker_id)] = {
            "length_mm": reference.marker_length_mm, "use_rgb_only": True,
            "name": "table_reference", "link": "None", "type": "aruco", **quality,
        }
        return rows

    def execute(
        self, action: ActionSpec, *, execution_id: str,
        placement_slot_id: str,
    ) -> Mapping[str, Any]:
        started_wall = time.time()
        started_monotonic = time.monotonic()
        deadline = started_monotonic + self.config.motion.action_timeout_s
        slot = self.config.placement_slots.get(placement_slot_id)
        if slot is None:
            return {"success": False, "status": "invalid_request", "message": f"unknown placement slot {placement_slot_id!r}"}
        with self._sdk_lock:
            with self._state_lock:
                if self._stopped:
                    return {"success": False, "status": "stopped", "message": "hardware restart and pose reconciliation required"}
                self._busy = True
                self._execution_id = execution_id
                self._last_action = action.token
                self._last_error = None
            item = self.config.objects[action.object_id]
            slot_rotated = False
            evidence = None
            try:
                self._set_phase("rotate_to_source")
                self._move_to_station(action.source_station, deadline)
                self._set_phase("rotate_to_source", successful=True)

                self._set_phase("visual_servo_grasp")
                p = self.config.perception
                result, _snapshot = self._modules["grasping"].execute_grasp_and_return_base(
                    robot=self.robot, camera_service=self.camera_service,
                    target_tag_id=item.marker_id, target_tag_name=item.label,
                    marker_info=self._marker_info(),
                    max_duration_s=min(self.config.motion.grasp_timeout_s, max(0.1, deadline - time.monotonic())),
                    deadline=deadline,
                    command_timeout_s=self.config.motion.command_timeout_s,
                    command_watchdog_s=min(0.30, p.max_frame_age_s),
                    stable_frames_required=p.stable_frames,
                    max_position_jump_m=p.max_position_jump_m,
                    max_frame_age_s=p.max_frame_age_s,
                    max_translation_drift_m=self.config.motion.max_translation_drift_m,
                    max_heading_error_deg=self.config.motion.rotation_tolerance_deg,
                    velocity_scale=(min(self.config.motion.velocity_scale, 0.25) if self._calibration_mode else self.config.motion.velocity_scale),
                    cancel_event=self._stop_event,
                    controller_status_callback=self._set_velocity_controller_status,
                    show_visualization=False,
                )
                if result != 0:
                    raise RuntimeError(f"grasp failed for marker {item.marker_id}")
                self._set_phase("visual_servo_grasp", successful=True)

                self._set_phase("verify_source_return")
                self._verify_station_reference(action.source_station, deadline)
                self._set_phase("verify_source_return", successful=True)

                self._set_phase("rotate_to_destination")
                self._move_to_station(action.destination_station, deadline)
                self._set_phase("rotate_to_destination", successful=True)

                self._set_phase("rotate_to_placement_slot")
                self._rotate_checked(slot.heading_offset_deg, f"to placement slot {slot.slot_id}", deadline)
                slot_rotated = abs(slot.heading_offset_deg) > 0.25
                self._set_phase("rotate_to_placement_slot", successful=True)

                self._set_phase("place_object")
                result = self._modules["delivery"].hand_object_on_workspace(
                    self.robot, arm_extension_delta=slot.arm_extension_m,
                    lift_down_pos=slot.lift_height_m,
                    command_timeout_s=self.config.motion.command_timeout_s,
                    deadline=deadline, cancel_event=self._stop_event,
                )
                if result != 0:
                    raise RuntimeError("placement routine failed")
                self._set_phase("place_object", successful=True)

                settle_until = min(deadline, time.monotonic() + self.config.motion.placement_settle_s)
                while time.monotonic() < settle_until:
                    self._check_cancelled(deadline)
                    time.sleep(min(0.05, settle_until - time.monotonic()))

                self._set_phase("verify_placement")
                evidence = self._modules["perception"].wait_for_stable_marker(
                    camera_service=self.camera_service, marker_info=self._marker_info(),
                    marker_id=item.marker_id,
                    expected_xyz_m=slot.verify_marker_xyz_m,
                    tolerance_m=slot.verify_tolerance_m,
                    stable_frames=p.stable_frames,
                    max_position_jump_m=p.max_position_jump_m,
                    max_frame_age_s=p.max_frame_age_s,
                    deadline=min(deadline, time.monotonic() + p.postcondition_timeout_s),
                    cancel_event=self._stop_event,
                )
                self._set_phase("verify_placement", successful=True)

                self._set_phase("restore_canonical_heading")
                if slot_rotated:
                    self._rotate_checked(-slot.heading_offset_deg, f"from placement slot {slot.slot_id}", deadline)
                    slot_rotated = False
                self._verify_station_reference(action.destination_station, deadline)
                self._set_phase("completed", successful=True)
                return {
                    "success": True, "status": "completed", "message": f"completed {action.token}",
                    "execution_id": execution_id, "action": action.token,
                    "marker_id": item.marker_id, "source_station": action.source_station,
                    "destination_station": action.destination_station,
                    "placement_slot_id": slot.slot_id, "postcondition": evidence,
                    "started_at": started_wall, "finished_at": time.time(),
                    "elapsed_s": time.monotonic() - started_monotonic,
                    "pose_confident": self._pose_confident,
                }
            except Exception as exc:
                if slot_rotated and not self._stop_event.is_set():
                    try:
                        self._rotate_checked(-slot.heading_offset_deg, f"failure recovery from slot {slot.slot_id}", deadline)
                        slot_rotated = False
                    except Exception:
                        self._pose_confident = False
                with self._state_lock:
                    self._last_error = str(exc)
                    self._stopped = True
                self._stop_event.set()
                stop_error = None
                try:
                    self.robot.stop()
                except Exception as stop_exc:
                    stop_error = str(stop_exc)
                self._set_phase("failed")
                return {
                    "success": False, "status": "failed", "message": str(exc),
                    "execution_id": execution_id, "action": action.token,
                    "current_station": self.current_station,
                    "started_at": started_wall, "finished_at": time.time(),
                    "elapsed_s": time.monotonic() - started_monotonic,
                    "restart_required": True, "pose_confident": self._pose_confident,
                    "stop_error": stop_error,
                }
            finally:
                with self._state_lock:
                    self._busy = False
                    self._execution_id = None

    def status(self) -> Mapping[str, Any]:
        with self._state_lock:
            state = {
                "backend": "stretch3_packaged_runtime", "motion_enabled": True,
                "ready": not self._stopped and self._pose_confident,
                "busy": self._busy, "stopped": self._stopped,
                "current_station": self.current_station,
                "pose_confident": self._pose_confident,
                "execution_id": self._execution_id,
                "current_phase": self._phase,
                "last_successful_phase": self._last_successful_phase,
                "cancel_requested": self._stop_event.is_set(),
                "last_action": self._last_action, "last_error": self._last_error,
                "calibration_id": self.config.motion.calibration_id,
                "calibration_mode": self._calibration_mode,
                "config_digest": self.config.digest,
                "runtime": dict(self._runtime_provenance),
                "runtime_qualification": dict(self._runtime_qualification),
                "velocity_controller": dict(self._velocity_controller_status),
                "last_reference_evidence": None if self._last_reference_evidence is None else dict(self._last_reference_evidence),
            }
        state["camera"] = self.camera_service.get_status() if self.camera_service is not None else None
        if state["camera"] is not None and not state["camera"].get("ready"):
            state["ready"] = False
        if self.config.motion.calibrated and not state["runtime_qualification"].get("qualified"):
            state["ready"] = False
        velocity = state["velocity_controller"]
        if velocity.get("error") or (
            not velocity.get("stop_requested", True)
            and not velocity.get("thread_alive", False)
        ):
            state["ready"] = False
        return state

    def emergency_stop(self) -> Mapping[str, Any]:
        with self._state_lock:
            self._stopped = True
            self._last_error = "emergency stop requested"
            self._pose_confident = False
        self._stop_event.set()
        self._set_phase("stop_requested")
        # Never call the SDK concurrently with an executing action.  The action
        # worker observes the cancellation event at bounded checkpoints.  The
        # physical runstop remains the immediate stop mechanism.
        if self._sdk_lock.acquire(blocking=False):
            try:
                if self.robot is not None:
                    self.robot.stop()
            finally:
                self._sdk_lock.release()
        return {"ok": True, "status": "stop_requested", "restart_required": True}

    def camera_jpeg(self) -> bytes | None:
        if self.camera_service is None:
            return None
        frame = self.camera_service.get_latest_color(apply_annotations=True)
        if frame is None:
            return None
        cv2 = importlib.import_module("cv2")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return encoded.tobytes() if ok else None

    def _safe_close_resources(self) -> None:
        if self.camera_service is not None:
            try:
                self.camera_service.stop()
            except Exception:
                pass
        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception:
                pass

    def close(self) -> None:
        self._stop_event.set()
        acquired = self._sdk_lock.acquire(timeout=self.config.motion.command_timeout_s + 5.0)
        if acquired:
            try:
                self._safe_close_resources()
            finally:
                self._sdk_lock.release()
        with self._state_lock:
            self._stopped = True


class BridgeDryRunController:
    """Simulated actions with an optional D405 preview; never starts the SDK."""

    def __init__(self, config: LabConfig, *, camera_preview: bool = False):
        self.config = config
        self.current_station = config.home_station
        self._stopped = False
        self._executed: list[str] = []
        self._phase = "ready"
        self.camera_service: Any = None
        if camera_preview:
            camera_type = importlib.import_module(
                "src.real_robot.stretch_runtime.camera_service"
            ).D405CameraService
            p = config.perception
            self.camera_service = camera_type(
                exposure="medium", capture_fps=15, stream_fps=10,
                startup_timeout_s=p.camera_startup_timeout_s,
                frame_timeout_ms=p.camera_frame_timeout_ms,
                stale_after_s=p.max_frame_age_s,
            )
            try:
                self.camera_service.start()
            except BaseException:
                self.close()
                raise

    def execute(self, action: ActionSpec, *, execution_id: str, placement_slot_id: str) -> Mapping[str, Any]:
        if self._stopped:
            return {"success": False, "status": "stopped", "message": "bridge is stopped"}
        if placement_slot_id not in self.config.placement_slots:
            return {"success": False, "status": "invalid_request", "message": "unknown placement slot"}
        self._phase = "dry_run_execute"
        self.current_station = action.destination_station
        self._executed.append(action.token)
        self._phase = "completed"
        now = time.time()
        return {
            "success": True, "status": "completed",
            "message": f"bridge dry-run completed {action.token}",
            "execution_id": execution_id, "action": action.token,
            "placement_slot_id": placement_slot_id,
            "current_station": self.current_station,
            "started_at": now, "finished_at": now,
            "elapsed_s": 0.0,
            "postcondition": {"dry_run": True}, "pose_confident": True,
        }

    def status(self) -> Mapping[str, Any]:
        return {
            "backend": "stretch_bridge_dry_run", "motion_enabled": False,
            "calibration_mode": False,
            "ready": not self._stopped, "stopped": self._stopped,
            "current_station": self.current_station, "pose_confident": True,
            "current_phase": self._phase, "executed_actions": list(self._executed),
            "config_digest": self.config.digest,
            "camera": self.camera_service.get_status() if self.camera_service is not None else None,
        }

    def emergency_stop(self) -> Mapping[str, Any]:
        self._stopped = True
        self._phase = "stopped"
        return {"ok": True, "status": "stopped", "restart_required": True}

    def camera_jpeg(self) -> bytes | None:
        if self.camera_service is None:
            return None
        bundle = self.camera_service.get_latest_bundle()
        age_s = bundle["age_s"]
        frame = bundle["color"]
        if frame is None or age_s is None or age_s > self.config.perception.max_frame_age_s:
            return None
        cv2 = importlib.import_module("cv2")
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return encoded.tobytes() if ok else None

    def close(self) -> None:
        self._stopped = True
        camera, self.camera_service = self.camera_service, None
        if camera is not None:
            camera.stop()
