"""Calibrated fixed-layout controller with marker-observed dynamic identity."""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Mapping
from urllib.request import urlopen

from .motion_console import MotionConsole, request_stop


LOCATION_HEADINGS = {
    "S0_left": -12.0,
    "S0_center": -24.0,
    "S0_right": -36.0,
    "S2": 12.0,
    "S3": 24.0,
    "S4": 36.0,
}

SOURCE_LOCATIONS = (
    "S0_left",
    "S0_center",
    "S0_right",
)

TOKEN_TO_LOCATION = {
    "S0_LEFT": "S0_left",
    "S0_CENTER": "S0_center",
    "S0_RIGHT": "S0_right",
    "S2": "S2",
    "S3": "S3",
    "S4": "S4",
}

LOCATION_TO_TOKEN = {
    value: key
    for key, value in TOKEN_TO_LOCATION.items()
}


class FixedLayoutDemoController:
    """AprilTag identity + calibrated physical location execution."""

    def __init__(
        self,
        config,
        *,
        confirmed_start_station: str,
    ):
        if confirmed_start_station != config.home_station:
            raise RuntimeError(
                "--confirm-start-station must be "
                f"{config.home_station!r}"
            )

        self.config = config
        self._lock = threading.RLock()
        self._busy = False
        self._stopped = False
        self._phase = "startup"
        self._last_error = None
        self._execution_id = None
        self._records = []

        from stretch_body.robot import Robot

        self.robot = Robot()
        if not self.robot.startup():
            raise RuntimeError(
                "could not connect to Stretch hardware"
            )

        try:
            if not self.robot.is_homed():
                raise RuntimeError(
                    "Stretch is not homed"
                )

            if self.robot.pimu.status.get(
                "runstop_event",
                True,
            ):
                raise RuntimeError(
                    "Stretch runstop is active"
                )

            self.console = MotionConsole(
                self.robot,
                self._motion_record,
                read_only=False,
            )
            self.console.ready()

            self._bootstrap_box_view()
            self._phase = "ready"

        except Exception:
            try:
                self.robot.stop()
            finally:
                raise

    def _motion_record(
        self,
        event,
        **data,
    ):
        self._records.append({
            "event": event,
            "time": time.time(),
            **data,
        })
        if len(self._records) > 200:
            self._records = self._records[-200:]

    def _tag_to_object(self):
        return {
            int(spec.marker_id): object_id
            for object_id, spec in self.config.objects.items()
        }

    def _by_tag(self, scene):
        result = {}
        for location, row in scene.items():
            tag_id = row.get("tag_id")
            if tag_id is not None:
                result[int(tag_id)] = location
        return result

    def _bootstrap_box_view(self):
        print("")
        print("FIXED-LAYOUT HRC STARTUP")
        print("Base must be physically on taped HOME.")
        print("Workspace must be clear; gripper empty.")
        print("")

        try:
            self.console.goto_calibrated("box_view")
        except ValueError as exc:
            if "clearance" not in str(exc).lower():
                raise

            answer = input(
                "Inspect high/retracted pose and type "
                "CLEARANCE to continue: "
            ).strip().upper()

            if answer != "CLEARANCE":
                raise RuntimeError(
                    "operator did not approve startup clearance"
                )

            self.console.execute("clearance")
            self.console.goto_calibrated("box_view")

        answer = input(
            "Inspect final box_view sweep and type "
            "CLEARANCE to use it: "
        ).strip().upper()

        if answer != "CLEARANCE":
            raise RuntimeError(
                "operator did not approve calibrated box_view"
            )

        self.console.execute("clearance")

        if abs(
            self.console.pose()["heading_deg"]
        ) > 1.0:
            self.console.execute("heading 0")

    def reset_scene_baseline(self):
        with self._lock:
            if self._stopped:
                raise RuntimeError(
                    "controller is stopped"
                )

            self._busy = True
            self._phase = "scan_reset"

            try:
                self.console.last_scene = None
                scene = self.console.scan_scene()

                by_tag = self._by_tag(scene)
                expected_tags = set(
                    self._tag_to_object()
                )

                if set(by_tag) != expected_tags:
                    raise RuntimeError(
                        "reset scene must contain each configured "
                        f"box exactly once; got {by_tag}"
                    )

                occupied_sources = set(
                    by_tag.values()
                )

                if occupied_sources != set(
                    SOURCE_LOCATIONS
                ):
                    raise RuntimeError(
                        "episode reset requires the three boxes "
                        "in the three S0 source locations; "
                        f"observed {occupied_sources}"
                    )

                for destination in (
                    "S2",
                    "S3",
                    "S4",
                ):
                    if (
                        scene[destination]["status"]
                        != "empty"
                    ):
                        raise RuntimeError(
                            f"reset requires {destination} empty"
                        )

                self._phase = "ready"

                return {
                    "ok": True,
                    "status": "baseline_captured",
                    "scene": scene,
                    "tag_locations": by_tag,
                }

            finally:
                self._busy = False

    def observe_human_move(self):
        with self._lock:
            if self._stopped:
                raise RuntimeError(
                    "controller is stopped"
                )

            before = self.console.last_scene
            if before is None:
                raise RuntimeError(
                    "capture episode baseline first"
                )

            self._busy = True
            self._phase = "observe_human"

            try:
                after = self.console.scan_scene()

                before_by = self._by_tag(before)
                after_by = self._by_tag(after)

                expected_tags = set(
                    self._tag_to_object()
                )

                if (
                    set(before_by) != expected_tags
                    or set(after_by) != expected_tags
                ):
                    self.console.last_scene = before
                    raise RuntimeError(
                        "scene did not contain every configured box"
                    )

                moves = []
                for tag_id in sorted(expected_tags):
                    if (
                        before_by[tag_id]
                        != after_by[tag_id]
                    ):
                        moves.append({
                            "tag_id": tag_id,
                            "from": before_by[tag_id],
                            "to": after_by[tag_id],
                        })

                if len(moves) != 1:
                    self.console.last_scene = before
                    raise RuntimeError(
                        "expected exactly one human-moved box; "
                        f"observed {moves}"
                    )

                move = moves[0]
                tag_to_object = self._tag_to_object()
                object_id = tag_to_object[
                    move["tag_id"]
                ]

                destination_token = (
                    LOCATION_TO_TOKEN[
                        move["to"]
                    ]
                )

                freeform_token = (
                    f"MOVE_{object_id.upper()}_TO_"
                    f"{destination_token}"
                )

                configured_matches = []

                if move["to"] in {
                    "S2",
                    "S3",
                    "S4",
                }:
                    configured_matches = [
                        token
                        for token, action
                        in self.config.actions.items()
                        if (
                            action.object_id == object_id
                            and
                            action.destination_station
                            == move["to"]
                        )
                    ]

                configured_token = (
                    configured_matches[0]
                    if len(configured_matches) == 1
                    else None
                )

                self._phase = "ready"

                return {
                    "ok": True,
                    "status": "human_move_observed",
                    "action_token": (
                        configured_token
                        or freeform_token
                    ),
                    "configured_action_token": (
                        configured_token
                    ),
                    "freeform_action_token": (
                        freeform_token
                    ),
                    "object_id": object_id,
                    "move": move,
                    "scene": after,
                }

            finally:
                self._busy = False

    def _route_to(self, heading):
        self.console.goto_calibrated(
            "box_view"
        )

        current = self.console.pose()[
            "heading_deg"
        ]

        if abs(
            float(heading) - current
        ) > 48.0:
            self.console.execute("heading 0")

        if abs(float(heading)) > 0.1:
            self.console.execute(
                f"heading {float(heading):.3f}"
            )
        elif abs(
            self.console.pose()["heading_deg"]
        ) > 1.0:
            self.console.execute("heading 0")

    def _decode_action(self, action):
        token = str(action.token).strip().upper()

        if token.startswith("MOVE_"):
            for destination_token, physical_location in (
                TOKEN_TO_LOCATION.items()
            ):
                suffix = (
                    f"_TO_{destination_token}"
                )

                if token.endswith(suffix):
                    object_token = token[
                        len("MOVE_"):
                        -len(suffix)
                    ]
                    object_id = object_token.lower()

                    if object_id in self.config.objects:
                        return (
                            object_id,
                            physical_location,
                        )

        object_id = str(action.object_id)

        if (
            action.destination_station
            in LOCATION_HEADINGS
        ):
            return (
                object_id,
                action.destination_station,
            )

        raise RuntimeError(
            f"unsupported physical action {token}"
        )

    def execute(
        self,
        action,
        *,
        execution_id: str,
        placement_slot_id: str,
    ) -> Mapping[str, Any]:
        del placement_slot_id

        started = time.time()

        with self._lock:
            if self._stopped:
                return {
                    "success": False,
                    "status": "stopped",
                    "message": (
                        "controller is stopped; "
                        "restart and reconcile"
                    ),
                }

            if self._busy:
                return {
                    "success": False,
                    "status": "failed",
                    "message": (
                        "fixed-layout controller is busy"
                    ),
                }

            self._busy = True
            self._execution_id = execution_id
            self._phase = "execute"
            self._last_error = None

            try:
                scene_before = (
                    self.console.last_scene
                )
                if scene_before is None:
                    raise RuntimeError(
                        "no validated scene baseline "
                        "is available"
                    )

                object_id, target_location = (
                    self._decode_action(action)
                )

                item = self.config.objects[
                    object_id
                ]
                tag_id = int(item.marker_id)

                before_by = self._by_tag(
                    scene_before
                )

                if tag_id not in before_by:
                    raise RuntimeError(
                        f"tag {tag_id} for "
                        f"{object_id} is not in scene"
                    )

                source_location = before_by[
                    tag_id
                ]

                if (
                    source_location
                    == target_location
                ):
                    raise RuntimeError(
                        f"{object_id} is already at "
                        f"{target_location}"
                    )

                target_row = scene_before[
                    target_location
                ]

                if (
                    target_row.get("tag_id")
                    is not None
                ):
                    raise RuntimeError(
                        f"target {target_location} "
                        "is occupied"
                    )

                source_heading = (
                    LOCATION_HEADINGS[
                        source_location
                    ]
                )
                destination_heading = (
                    LOCATION_HEADINGS[
                        target_location
                    ]
                )

                self._phase = "pick"

                self.console.goto_calibrated(
                    "box_view"
                )
                self.console.move_box_gripper(
                    132.0
                )

                self._route_to(
                    source_heading
                )

                self.console.goto_calibrated(
                    "box_pregrasp"
                )

                self.console.move_box_gripper(
                    103.0
                )

                self.console.goto_calibrated(
                    "box_carry"
                )

                self._phase = "transfer"

                self._route_to(
                    destination_heading
                )

                self._phase = "place"

                self.console.goto_calibrated(
                    "box_return"
                )

                if abs(
                    self.console.pose()[
                        "heading_deg"
                    ]
                ) > 1.0:
                    self._route_to(0.0)

                self._phase = "verify"

                scene_after = (
                    self.console.scan_scene()
                )
                after_by = self._by_tag(
                    scene_after
                )

                if (
                    after_by.get(tag_id)
                    != target_location
                ):
                    raise RuntimeError(
                        "postcondition failed: "
                        f"tag {tag_id} expected at "
                        f"{target_location}, observed "
                        f"{after_by.get(tag_id)}"
                    )

                for other_tag in before_by:
                    if other_tag == tag_id:
                        continue

                    if (
                        after_by.get(other_tag)
                        != before_by.get(other_tag)
                    ):
                        raise RuntimeError(
                            "postcondition failed: "
                            f"tag {other_tag} moved "
                            "unexpectedly"
                        )

                self._phase = "ready"

                return {
                    "success": True,
                    "status": "completed",
                    "message": (
                        f"completed {action.token}"
                    ),
                    "execution_id": execution_id,
                    "action": action.token,
                    "object_id": object_id,
                    "tag_id": tag_id,
                    "source_location": (
                        source_location
                    ),
                    "destination_location": (
                        target_location
                    ),
                    "scene_after": scene_after,
                    "elapsed_s": (
                        time.time() - started
                    ),
                }

            except Exception as exc:
                self._last_error = str(exc)
                self._phase = "fault"
                self._stopped = True

                try:
                    request_stop(self.robot)
                except Exception:
                    pass

                return {
                    "success": False,
                    "status": "failed",
                    "message": str(exc),
                    "execution_id": execution_id,
                    "action": action.token,
                    "restart_required": True,
                    "elapsed_s": (
                        time.time() - started
                    ),
                }

            finally:
                self._busy = False
                self._execution_id = None

    def status(self):
        try:
            homed = bool(
                self.robot.is_homed()
            )
        except Exception:
            homed = False

        try:
            runstop = bool(
                self.robot.pimu.status.get(
                    "runstop_event",
                    True,
                )
            )
        except Exception:
            runstop = True

        return {
            "backend": "fixed_layout_demo",
            "motion_enabled": True,
            "fixed_layout_demo": True,
            "freeform_supported": True,
            "calibration_mode": False,
            "ready": bool(
                homed
                and not runstop
                and not self._stopped
                and not self._busy
            ),
            "homed": homed,
            "runstop": runstop,
            "stopped": self._stopped,
            "busy": self._busy,
            "current_phase": self._phase,
            "last_error": self._last_error,
            "calibration_id": (
                "fixed-layout-3box-v1"
            ),
        }

    def camera_jpeg(self):
        camera_url = os.environ.get(
            "HRC_CAMERA_URL",
            "http://127.0.0.1:9101/v1/camera.jpg",
        )
        try:
            with urlopen(
                camera_url,
                timeout=1.5,
            ) as response:
                return response.read()
        except Exception:
            return None

    def emergency_stop(self):
        self._stopped = True
        self._phase = "stopped"

        try:
            request_stop(self.robot)
            return {
                "ok": True,
                "status": "stopped",
                "motion_enabled": True,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "stop_request_failed",
                "error": str(exc),
            }

    def close(self):
        try:
            self.robot.stop()
        except Exception:
            pass
