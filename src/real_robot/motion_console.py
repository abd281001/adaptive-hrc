"""Supervised joint checks before station or grasp calibration.

This connects directly to Stretch Body. It never starts a camera, homes,
recenters, translates the base, or runs a recipe. See MOTION_CHECK.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import signal
import sys
import time


HELP = """Enter ONE command at a time and watch the robot:
  status          measured positions and changes from startup
  clearance       record current RETRACTED pose after checking swept clearance
  heading DEG     heading relative to startup, -48..+48 (not a relative turn)
                  requires recorded clearance height and wrist orientation
  lift CM         signed change, at most 2 cm per command (+ up, - down)
  arm CM          signed change, at most 2 cm per command (+ extend, - retract)
  pitch DEG       signed wrist pitch change, at most 5 deg (+ tilt up)
                  requires retracted arm and recorded clearance height
  open            open to this robot's configured gripper open position
  close           EMPTY gripper only: close to fingertips touching (0 units)
  grip UNITS      signed gripper change, at most 5 units (+ open, - close)
  note LABEL      record the current measured pose under a label
  scene           scan all 6 calibrated locations using box AprilTags
  quit            stop SDK; no automatic return or stow (support any object)
  Ctrl+C          request runstop and exit; physical runstop is authoritative
"""

GRIPPER_SPEED = 0.2  # SDK angular units: rad/s, not gripper units/s
GRIPPER_ACCEL = 0.4
WRIST_JOINTS = ("wrist_yaw", "wrist_pitch", "wrist_roll")


class MotionFault(RuntimeError):
    """Stop the session; do not retry or recover automatically."""


class PreflightFault(MotionFault, ValueError):
    """Refuse a command before dispatch without shutting down the SDK."""


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return result


def measured(value):
    try:
        return finite(value)
    except (ValueError, TypeError) as exc:
        raise MotionFault("Invalid numeric feedback from hardware") from exc


def angle_delta(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def gripper_timing(gripper, current, target):
    """Budget travel in SDK radians, with ramp time and quantization margin."""
    distance = abs(measured(gripper.pct_to_world_rad(target))
                   - measured(gripper.pct_to_world_rad(current)))
    maximum = gripper.params["motion"]["max"]
    speed = min(GRIPPER_SPEED, measured(maximum["vel"]))
    accel = min(GRIPPER_ACCEL, measured(maximum["accel"]))
    if speed <= 0 or accel <= 0:
        raise MotionFault("Invalid gripper motion limits")
    if distance <= speed * speed / accel:
        travel_s = 2 * math.sqrt(distance / accel)
    else:
        travel_s = distance / speed + speed / accel
    timeout_s = max(3.0, 1.25 * travel_s + 2.0)
    if timeout_s > 90:
        raise ValueError("Gripper travel exceeds the 90-second check budget; use smaller grip steps")
    return speed, accel, travel_s, timeout_s


class MotionConsole:
    def __init__(self, robot, record, *, clock=time.monotonic, sleep=time.sleep, read_only=False):
        self.robot, self.record = robot, record
        self.clock, self.sleep = clock, sleep
        self.read_only = read_only
        self.gripper = robot.end_of_arm.get_joint("stretch_gripper")
        if self.gripper is None:
            raise MotionFault("stretch_gripper is not present")
        self.wrist = {name: robot.end_of_arm.get_joint(name) for name in WRIST_JOINTS}
        self.wrist = {name: joint for name, joint in self.wrist.items() if joint is not None}
        self.clear_pose = None
        self.last_scene = None
        self.start = self.pose()
        self.stamps = {}
        self.assert_health()
        self.record("start", pose=self.start, limits=self.limits())

    def limits(self):
        return {
            "arm_m": [measured(x) for x in self.robot.arm.soft_motion_limits["current"]],
            "lift_m": [measured(x) for x in self.robot.lift.soft_motion_limits["current"]],
            "gripper_open_units": measured(self.gripper.poses["open"]),
            "wrist_limits_deg": {
                name: [math.degrees(measured(x)) for x in joint.soft_motion_limits["current"]]
                for name, joint in self.wrist.items()
            },
        }

    def pose(self):
        r = self.robot
        pose = {
            "x_m": measured(r.base.status["x"]),
            "y_m": measured(r.base.status["y"]),
            "theta_rad": measured(r.base.status["theta"]),
            "lift_m": measured(r.lift.status["pos"]),
            "arm_m": measured(r.arm.status["pos"]),
            "gripper_units": measured(self.gripper.status["pos_pct"]),
            "gripper_velocity_rad_s": measured(self.gripper.status["vel"]),
            "gripper_effort": measured(self.gripper.status["effort"]),
            **{f"{name}_rad": measured(joint.status["pos"]) for name, joint in self.wrist.items()},
        }
        start_theta = self.start["theta_rad"] if hasattr(self, "start") else pose["theta_rad"]
        pose["heading_deg"] = math.degrees(angle_delta(pose["theta_rad"], start_theta))
        return pose

    def assert_health(self):
        r = self.robot
        if not self.read_only and not r.is_homed():
            raise MotionFault("Robot is not homed; use the standard supervised homing procedure separately")
        if not self.read_only and r.pimu.status.get("runstop_event", True):
            raise MotionFault("Runstop active; this console never clears it")
        devices = {"pimu": (r.pimu.status, "timestamp")}
        for name, motor in (("arm", r.arm.motor), ("lift", r.lift.motor),
                            ("left_wheel", r.base.left_wheel), ("right_wheel", r.base.right_wheel)):
            if not self.read_only and (motor.status.get("in_guarded_event") or motor.status.get("runstop_on")):
                raise MotionFault(f"{name}: guarded contact or runstop")
            devices[name] = (motor.status, "timestamp")
        if self.gripper.status.get("hardware_error"):
            raise MotionFault("Gripper reports a hardware error")
        devices["gripper"] = (self.gripper.status, "timestamp_pc")
        for name, joint in self.wrist.items():
            if joint.status.get("hardware_error"):
                raise MotionFault(f"{name}: hardware error")
            devices[name] = (joint.status, "timestamp_pc")
        now = self.clock()
        for name, (status, key) in devices.items():
            stamp = measured(status[key])
            if stamp <= 0:
                raise MotionFault(f"{name}: no valid feedback timestamp")
            previous, changed = self.stamps.get(name, (None, now))
            if stamp != previous:
                self.stamps[name] = (stamp, now)
            elif now - changed > 1.0:
                raise MotionFault(f"{name}: feedback stopped updating")
        p = self.pose()
        if math.hypot(p["x_m"] - self.start["x_m"], p["y_m"] - self.start["y_m"]) > 0.025:
            raise MotionFault("Base odometry translated >2.5 cm; reconcile the floor position")

    def ready(self):
        # Require advancing feedback immediately before every command, including
        # after the operator has been idle. Cached setpoints are not completion.
        self.assert_health()
        before = {name: stamp for name, (stamp, _) in self.stamps.items()}
        deadline = self.clock() + 1.1
        while True:
            self.sleep(0.05)
            self.assert_health()
            if all(self.stamps[name][0] != stamp for name, stamp in before.items()):
                return
            if self.clock() >= deadline:
                raise MotionFault("Feedback did not advance before command")

    def preflight(self):
        """Health-check before dispatch; tolerate only a brief transient PIMU timestamp gap."""
        deadline = self.clock() + 0.75
        while True:
            try:
                self.ready()
                return
            except MotionFault as exc:
                # This fault has occurred transiently on the real Stretch PIMU.
                # No actuator command has been dispatched yet, so briefly retry
                # only this exact condition. Every other health fault refuses
                # immediately.
                if str(exc) != "pimu: no valid feedback timestamp" or self.clock() >= deadline:
                    raise PreflightFault(
                        f"Pre-command health check failed: {exc}"
                    ) from exc
                self.sleep(0.05)

    def status(self):
        p = self.pose()
        heading = math.degrees(angle_delta(p["theta_rad"], self.start["theta_rad"]))
        print(f"Heading {heading:+.2f} deg | lift {p['lift_m']:.3f} m "
              f"(change {100*(p['lift_m']-self.start['lift_m']):+.1f} cm) | "
              f"arm {p['arm_m']:.3f} m | gripper {p['gripper_units']:.1f} units")
        print(f"Startup lift {self.start['lift_m']:.3f} m; "
              f"SDK limits: {self.limits()}")
        print("Wrist degrees: " + ", ".join(
            f"{name}={math.degrees(p[f'{name}_rad']):+.2f}" for name in self.wrist))
        print(f"Homed: {bool(self.robot.is_homed())}; "
              f"runstop: {bool(self.robot.pimu.status.get('runstop_event', True))}; "
              f"clearance pose recorded: {self.clear_pose is not None}")
        return p

    def check_clearance(self, p, *, turning):
        if p["arm_m"] > 0.02:
            raise ValueError("Retract arm to <=0.020 m before turning or pitching")
        if self.clear_pose is None:
            raise ValueError("Raise to a clear pose, inspect the sweep, then enter clearance")
        if p["lift_m"] < self.clear_pose["lift_m"] - 0.003:
            raise ValueError("Raise lift to recorded clearance height before turning or pitching")
        if turning and any(abs(p[f"{name}_rad"] - self.clear_pose[f"{name}_rad"])
                           > math.radians(1) for name in self.wrist):
            raise ValueError("Restore recorded wrist angles, or inspect the new sweep and record clearance again")

    def wait_target(self, key, target, *, tolerance, timeout):
        deadline, settled = self.clock() + timeout, None
        best_error = abs(target - self.pose()[key])
        last_progress = self.clock()
        while self.clock() < deadline:
            self.sleep(0.05)
            self.assert_health()
            current = self.pose()[key]
            error = angle_delta(target, current) if key == "theta_rad" else target - current
            if key == "gripper_units":
                if best_error - abs(error) >= 0.2:
                    best_error, last_progress = abs(error), self.clock()
                if abs(error) > tolerance and self.clock() - last_progress > 2.0:
                    raise MotionFault("Gripper made no progress toward target for 2 seconds; inspect for obstruction")
            if key == "theta_rad":
                velocity = abs(measured(self.robot.base.status["theta_vel"]))
                still = velocity < math.radians(0.5)
            elif key in ("arm_m", "lift_m"):
                joint = getattr(self.robot, key[:-2])
                still = abs(measured(joint.status["vel"])) < 0.002
            elif key == "wrist_pitch_rad":
                still = abs(measured(self.wrist["wrist_pitch"].status["vel"])) < 0.02
            else:
                still = abs(measured(self.gripper.status["vel"])) < 0.03
            if abs(error) <= tolerance and still:
                settled = self.clock() if settled is None else settled
                if self.clock() - settled >= 0.3:
                    return
            else:
                settled = None
        raise MotionFault(f"{key}: target {target:.4f} not reached and settled; "
                          f"measured {self.pose()[key]:.4f}. Inspect before retrying")

    def macro_move_linear(self, name, key, target):
        """Continuous arm/lift move for already-qualified macro trajectories."""
        if name not in ("arm", "lift"):
            raise ValueError("macro_move_linear supports only arm/lift")

        self.preflight()
        p = self.pose()
        current = p[key]

        if abs(target - current) <= 0.001:
            return

        low, high = self.limits()[key]

        if name == "arm":
            high = min(high, 0.35)
        else:
            low = max(low, self.start[key] - 0.15)
            high = min(high, self.start[key] + 0.15)

        if not low <= target <= high:
            raise ValueError(
                f"Macro target {target:.3f} m outside check limits "
                f"[{low:.3f}, {high:.3f}]"
            )

        joint = getattr(self.robot, name)

        # Roughly twice the old calibration-console speed.
        speed = 0.020
        accel = 0.040
        timeout = max(10.0, abs(target - current) / speed + 6.0)

        command_name = f"macro_{name}_to_{target:.3f}"

        self.record(
            "command",
            command=command_name,
            target=target,
            coordinate=key,
            before=p,
            timeout_s=timeout,
        )

        print(
            f"Continuous {name} move {current:.3f} -> {target:.3f} m; "
            "watch clearance and keep runstop at hand.",
            flush=True,
        )

        try:
            joint.move_to(target, v_m=speed, a_m=accel)
            self.robot.push_command()
            self.wait_target(
                key,
                target,
                tolerance=0.003,
                timeout=timeout,
            )
        except Exception as exc:
            try:
                self.record(
                    "motion_failed",
                    command=command_name,
                    error=str(exc),
                    pose=self.pose(),
                )
            except Exception:
                pass
            raise MotionFault(str(exc)) from exc

        self.record(
            "target_reached",
            command=command_name,
            target=target,
            pose=self.status(),
        )

    def goto_calibrated(self, label):
        """Restore known same-size-box geometry using bounded console moves."""
        valid = {
            "box_view",
            "box_scan",
            "box_pregrasp",
            "box_carry",
            "box_return",
        }
        if label not in valid:
            raise ValueError(
                "Use: goto box_view | box_scan | box_pregrasp | "
                "box_carry | box_return"
            )

        ARM_RETRACTED = 0.003
        ARM_PREGRASP = 0.223
        LIFT_VIEW = 0.878
        LIFT_PREGRASP = 0.758
        LIFT_CLEAR = 0.858
        PITCH_BOX_DEG = -16.61

        def move_linear(command, key, target):
            self.macro_move_linear(command, key, target)

        def move_pitch(target_deg):
            while True:
                current_deg = math.degrees(self.pose()["wrist_pitch_rad"])
                error_deg = target_deg - current_deg
                if abs(error_deg) <= 1.0:
                    return
                step = max(-5.0, min(5.0, error_deg))
                self.execute(f"pitch {step:.3f}")

        def require_box_pitch():
            current_deg = math.degrees(self.pose()["wrist_pitch_rad"])
            if abs(current_deg - PITCH_BOX_DEG) > 2.0:
                raise ValueError(
                    f"Expected box wrist pitch near {PITCH_BOX_DEG:.2f} deg; "
                    f"measured {current_deg:.2f} deg"
                )

        if label in {"box_view", "box_scan", "box_pregrasp"}:
            # Always establish the known high/retracted geometry first.
            move_linear("arm", "arm_m", ARM_RETRACTED)
            move_linear("lift", "lift_m", LIFT_VIEW)
            move_pitch(PITCH_BOX_DEG)

            if label == "box_view":
                print("Reached calibrated box_view geometry.")
                return

            move_linear("arm", "arm_m", ARM_PREGRASP)

            if label == "box_scan":
                print("Reached calibrated box_scan geometry.")
                return

            move_linear("lift", "lift_m", LIFT_PREGRASP)
            print("Reached calibrated box_pregrasp geometry.")
            return

        if label == "box_carry":
            # Do not change wrist orientation while a box may be held.
            require_box_pitch()
            move_linear("lift", "lift_m", LIFT_CLEAR)
            move_linear("arm", "arm_m", ARM_RETRACTED)
            move_linear("lift", "lift_m", LIFT_VIEW)
            print("Reached calibrated box_carry geometry.")
            return

        if label == "box_return":
            # Assumes the base is still aimed at the same source slot.
            require_box_pitch()
            move_linear("lift", "lift_m", LIFT_VIEW)
            move_linear("arm", "arm_m", ARM_PREGRASP)
            move_linear("lift", "lift_m", LIFT_PREGRASP)

            # Release at the previously calibrated source pose.
            self.move_box_gripper(132.0)

            move_linear("lift", "lift_m", LIFT_VIEW)
            move_linear("arm", "arm_m", ARM_RETRACTED)
            print("Returned box and restored box_view geometry.")
            return


    def move_box_gripper(self, target):
        """Move gripper to a calibrated box aperture in bounded <=5-unit steps."""
        target = float(target)

        for _ in range(30):
            current = self.pose()["gripper_units"]
            error = target - current

            if abs(error) <= 1.5:
                print(
                    f"Reached box gripper target {target:.1f}; "
                    f"measured {current:.1f} units."
                )
                return

            step = max(-5.0, min(5.0, error))
            self.execute(f"grip {step:.3f}")

        raise MotionFault(
            f"Gripper did not converge to calibrated target {target:.1f}"
        )

    def cycle_box(self, box):
        """Pick a same-size box from its fixed Station-1 slot and return it."""
        headings = {
            "b1": -12.0,
            "b5": -24.0,
            "b3": -36.0,
        }
        box = box.lower()
        if box not in headings:
            raise ValueError("Use: cycle b1 | cycle b5 | cycle b3")

        heading = headings[box]

        # Establish turn-safe high/retracted geometry.
        self.goto_calibrated("box_view")
        self.move_box_gripper(132.0)

        print(f"Rotating to {box.upper()} at heading {heading:+.1f} deg.")
        self.execute(f"heading {heading}")

        # Same-size boxes share the calibrated radial grasp geometry.
        self.goto_calibrated("box_pregrasp")

        answer = input(
            f"Inspect {box.upper()} alignment. "
            "Press ENTER to grasp, or type NO to abort: "
        ).strip().lower()
        if answer:
            print("Cycle aborted before grasp.")
            return

        self.move_box_gripper(103.0)

        answer = input(
            f"Confirm {box.upper()} is securely held. "
            "Press ENTER to lift, or type NO to abort: "
        ).strip().lower()
        if answer:
            print("Cycle stopped with robot at grasp pose; support/reconcile box manually.")
            return

        self.goto_calibrated("box_carry")

        answer = input(
            f"{box.upper()} lifted. Press ENTER to return it to its source mark, "
            "or type NO to stop here: "
        ).strip().lower()
        if answer:
            print("Cycle stopped in carry pose; support/reconcile box manually.")
            return

        self.goto_calibrated("box_return")
        print(f"{box.upper()} pickup-and-return cycle complete.")


    def mark_stations(self):
        """Use B1 to establish and mark S2, S3, and S4."""

        headings = {
            "S1_B1": -12.0,
            "S2": 12.0,
            "S3": 24.0,
            "S4": 36.0,
        }

        LIFT_VIEW = 0.878
        ARM_RETRACTED = 0.003

        def confirm(message):
            answer = input(
                message + " Press ENTER to continue, or type NO to abort: "
            ).strip().lower()
            if answer:
                raise ValueError("Station-marking sequence aborted by operator")

        def move_linear(command, key, target):
            while True:
                current = self.pose()[key]
                error_cm = 100.0 * (target - current)

                if abs(error_cm) <= 0.1:
                    return

                step = max(-2.0, min(2.0, error_cm))
                self.execute(f"{command} {step:.3f}")

        def pickup(label, heading):
            # Always begin turn-safe.
            self.goto_calibrated("box_view")
            self.move_box_gripper(132.0)

            self.execute(f"heading {heading}")

            self.goto_calibrated("box_pregrasp")

            confirm(
                f"Inspect B1 alignment at {label}; fingers should straddle "
                "the box without touching the table."
            )

            self.move_box_gripper(103.0)

            confirm(f"Confirm B1 is securely gripped at {label}.")

            self.goto_calibrated("box_carry")

            print(f"B1 picked from {label} and is in carry pose.")

        def place(station, heading):
            # Carry pose is high/retracted before this base turn.
            self.execute(f"heading {heading}")

            confirm(
                f"Check the table area for {station}. It must be flat, clear, "
                "supported, and free of obstacles."
            )

            self.goto_calibrated("box_pregrasp")

            # This pose becomes the repeatable placement/pickup location.
            self.execute(f"note {station}_place")

            # Release B1 without going to the huge full-open aperture.
            self.move_box_gripper(132.0)

            # IMPORTANT: raise away from the released box before retracting.
            move_linear("lift", "lift_m", LIFT_VIEW)
            move_linear("arm", "arm_m", ARM_RETRACTED)

            self.execute(f"note {station}_view")

            print(f"B1 placed at {station}.")
            confirm(
                f"MARK B1's footprint on the table now as {station}. "
                "Do not move B1 after marking."
            )

        print("")
        print("=== STATION MARKING WITH B1 ===")
        print("S1/B1=-12 deg, S2=+12, S3=+24, S4=+36")
        print("B1 will finish at S4.")
        print("")

        confirm(
            "Verify the robot base is on the taped HOME position, "
            "B1 is on its marked S1 spot, and S2/S3/S4 table areas are clear."
        )

        pickup("S1/B1", headings["S1_B1"])

        place("S2", headings["S2"])
        pickup("S2", headings["S2"])

        place("S3", headings["S3"])
        pickup("S3", headings["S3"])

        place("S4", headings["S4"])

        print("")
        print("=== STATION MARKING COMPLETE ===")
        print("S2 = +12 deg")
        print("S3 = +24 deg")
        print("S4 = +36 deg")
        print("B1 remains at S4.")

    def verify_box_tour(self, box):
        """Move one box S0 -> S2 -> S3 -> S4 -> its original S0 slot."""
        source_headings = {
            "b1": -12.0,
            "b5": -24.0,
            "b3": -36.0,
        }
        destinations = [
            ("S2", 12.0),
            ("S3", 24.0),
            ("S4", 36.0),
        ]

        box = box.lower()
        if box not in source_headings:
            raise ValueError("Use: tour b1 | tour b5 | tour b3")

        source_heading = source_headings[box]

        def confirm(message):
            answer = input(
                message +
                "\nPress the ENTER KEY to continue; type NO to abort: "
            ).strip().lower()
            if answer:
                raise ValueError("Verification tour aborted by operator")

        def route_to(target):
            # Always route through heading 0 while carrying.
            # This avoids a direct +36 <-> -36 sweep.
            self.goto_calibrated("box_view")
            self.execute("heading 0")
            if abs(target) > 0.1:
                self.execute(f"heading {target}")

        def pickup_here(label):
            self.move_box_gripper(132.0)
            self.goto_calibrated("box_pregrasp")

            confirm(
                f"Inspect {box.upper()} at {label}. "
                "Gripper should be centered around the marked box."
            )

            self.move_box_gripper(103.0)

            confirm(
                f"Confirm {box.upper()} is securely gripped at {label}."
            )

            self.goto_calibrated("box_carry")

        def place_here(label):
            # box_return lowers to calibrated placement pose,
            # releases to 132 units, raises, then retracts.
            self.goto_calibrated("box_return")

            confirm(
                f"Verify {box.upper()} is sitting correctly on the "
                f"{label} taped mark."
            )

        print("")
        print(f"=== VERIFY {box.upper()} ===")
        print(
            f"S0/{box.upper()} {source_heading:+.0f} -> "
            "S2 +12 -> S3 +24 -> S4 +36 -> S0"
        )

        # Initial source pickup.
        self.goto_calibrated("box_view")
        self.move_box_gripper(132.0)
        self.execute(f"heading {source_heading}")
        pickup_here(f"S0/{box.upper()}")

        # S2, S3, S4.
        for station, heading in destinations:
            route_to(heading)
            place_here(station)
            pickup_here(station)

        # Return to this box's original source slot.
        route_to(source_heading)
        place_here(f"S0/{box.upper()}")

        # Finish every tour in the canonical taped HOME orientation.
        self.execute("heading 0")

        print(f"=== {box.upper()} TOUR PASSED ===")
        print(f"{box.upper()} is back at its source mark.")


    def verify_all_boxes(self):
        """Run full station tour for B1, B5, then B3."""
        for box in ("b1", "b5", "b3"):
            self.verify_box_tour(box)

            if box != "b3":
                answer = input(
                    f"{box.upper()} completed and returned to S0.\n"
                    "Check all stations are clear. Press ENTER KEY for the "
                    "next box; type NO to stop: "
                ).strip().lower()
                if answer:
                    raise ValueError("Verification stopped between boxes")

        print("")
        print("================================")
        print("ALL THREE BOX TOURS COMPLETED")
        print("B1, B5 and B3 returned to S0.")
        print("================================")

    def scan_scene(self):
        """Scan all calibrated physical locations using AprilTag identity."""
        from collections import Counter
        import hashlib
        from urllib.error import URLError
        from urllib.request import Request, urlopen

        import cv2
        import numpy as np
        from pupil_apriltags import Detector

        if self.clear_pose is None:
            raise ValueError(
                "Record the high/retracted box_view clearance before scene scanning"
            )

        # Identity is fixed. Location is NOT.
        tag_to_box = {
            1: "B1",
            3: "B3",
            5: "B5",
        }

        # Fixed calibrated geometry.
        locations = [
            ("S0_left",   -12.0),
            ("S0_center", -24.0),
            ("S0_right",  -36.0),
            ("S2",         12.0),
            ("S3",         24.0),
            ("S4",         36.0),
        ]

        CAMERA_URL = "http://127.0.0.1:9100/v1/camera.jpg"
        SAMPLES = 9
        CONSENSUS = 5

        # Only a tag reasonably close to the horizontal camera center belongs
        # to the location currently being inspected. Adjacent stations may
        # still be visible in the wide camera image.
        CENTER_GATE_FRAC = 0.13

        april = Detector(
            families="tagStandard41h12",
            nthreads=2,
            quad_decimate=1.0,
        )

        def one_observation():
            url = f"{CAMERA_URL}?t={time.time_ns()}"
            request = Request(
                url,
                headers={
                    "Cache-Control": "no-cache, no-store",
                    "Pragma": "no-cache",
                },
            )

            with urlopen(request, timeout=2.0) as response:
                raw = response.read()

            digest = hashlib.sha1(raw).hexdigest()
            frame = cv2.imdecode(
                np.frombuffer(raw, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                raise ValueError("camera response was not a decodable JPEG")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            width = float(frame.shape[1])
            center_x = width / 2.0

            candidates = []
            for tag in april.detect(gray, estimate_tag_pose=False):
                tag_id = int(tag.tag_id)
                hamming = int(tag.hamming)

                if tag_id not in tag_to_box:
                    continue

                # B3 was empirically observed at h=0,1,2. Consensus across
                # fresh images protects against accepting one weak detection.
                if hamming > 2:
                    continue

                corners = np.asarray(tag.corners, dtype=float)
                cx = float(corners[:, 0].mean())
                dx_frac = abs(cx - center_x) / width

                candidates.append(
                    (dx_frac, tag_id, hamming, cx)
                )

            if not candidates:
                return digest, None, None

            candidates.sort(key=lambda row: row[0])
            dx_frac, tag_id, hamming, cx = candidates[0]

            detail = {
                "nearest_tag_id": tag_id,
                "nearest_box": tag_to_box[tag_id],
                "nearest_hamming": hamming,
                "nearest_center_error_frac": dx_frac,
                "nearest_center_x_px": cx,
            }

            if dx_frac > CENTER_GATE_FRAC:
                # We can see a known box, but it is too far off-axis to
                # claim it occupies this calibrated location.
                return digest, None, detail

            return digest, tag_id, detail

        print("")
        print("=== APRILTAG SCENE SCAN ===")
        print("tag1=B1, tag3=B3, tag5=B5")
        print(
            f"{SAMPLES} unique frames/location; "
            f"{CONSENSUS} matching votes required."
        )
        print("")

        # Scanner always uses the qualified high/retracted viewing geometry.
        self.goto_calibrated("box_view")

        scene = {}
        scan_errors = []

        for location, heading in locations:
            print(f"Scanning {location} at {heading:+.0f} deg ...")

            self.execute(f"heading {heading}")

            # Allow the base/camera image to settle after the completed turn.
            self.sleep(0.40)

            votes = []
            seen_images = set()
            nearest_errors = []
            capture_failures = []
            deadline = self.clock() + 5.0

            while (
                len(votes) < SAMPLES
                and self.clock() < deadline
            ):
                try:
                    digest, candidate, detail = one_observation()
                except (URLError, TimeoutError, OSError, ValueError) as exc:
                    capture_failures.append(str(exc))
                    self.sleep(0.10)
                    continue

                # Never count the same bridge JPEG twice.
                if digest in seen_images:
                    self.sleep(0.08)
                    continue

                seen_images.add(digest)
                votes.append(candidate)

                if detail is not None:
                    nearest_errors.append(
                        float(detail["nearest_center_error_frac"])
                    )

                self.sleep(0.08)

            vote_counts = Counter(votes)
            if vote_counts:
                winner, winner_count = vote_counts.most_common(1)[0]
            else:
                winner, winner_count = None, 0

            vote_summary = {
                ("empty" if key is None else f"tag{key}"): int(count)
                for key, count in vote_counts.items()
            }

            median_center_error = None
            if nearest_errors:
                ordered = sorted(nearest_errors)
                median_center_error = ordered[len(ordered) // 2]

            if len(votes) < CONSENSUS or winner_count < CONSENSUS:
                row = {
                    "status": "uncertain",
                    "heading_deg": heading,
                    "tag_id": None,
                    "box": None,
                    "samples": len(votes),
                    "winning_votes": winner_count,
                    "votes": vote_summary,
                    "median_nearest_center_error_frac": median_center_error,
                    "capture_failures": capture_failures[-3:],
                }
                scene[location] = row
                scan_errors.append(
                    f"{location}: no {CONSENSUS}-frame consensus "
                    f"({vote_summary})"
                )
                print(
                    f"  {location}: UNCERTAIN "
                    f"{vote_summary}"
                )
                continue

            if winner is None:
                row = {
                    "status": "empty",
                    "heading_deg": heading,
                    "tag_id": None,
                    "box": None,
                    "samples": len(votes),
                    "winning_votes": winner_count,
                    "votes": vote_summary,
                    "median_nearest_center_error_frac": median_center_error,
                }
                scene[location] = row
                print(
                    f"  {location}: EMPTY "
                    f"({winner_count}/{len(votes)} votes)"
                )
                continue

            row = {
                "status": "occupied",
                "heading_deg": heading,
                "tag_id": int(winner),
                "box": tag_to_box[int(winner)],
                "samples": len(votes),
                "winning_votes": winner_count,
                "votes": vote_summary,
                "median_nearest_center_error_frac": median_center_error,
            }
            scene[location] = row

            print(
                f"  {location}: {row['box']} / tag {winner} "
                f"({winner_count}/{len(votes)} votes)"
            )

        # Finish every successful scanning traversal at canonical HOME.
        self.execute("heading 0")

        # A valid scene must contain every physical demo box exactly once.
        locations_by_tag = {tag_id: [] for tag_id in tag_to_box}
        for location, row in scene.items():
            tag_id = row.get("tag_id")
            if tag_id in locations_by_tag:
                locations_by_tag[tag_id].append(location)

        for tag_id, occupied_locations in locations_by_tag.items():
            box = tag_to_box[tag_id]
            if len(occupied_locations) == 0:
                scan_errors.append(
                    f"{box}/tag{tag_id} was not assigned to any location"
                )
            elif len(occupied_locations) > 1:
                scan_errors.append(
                    f"{box}/tag{tag_id} was assigned to multiple locations: "
                    f"{occupied_locations}"
                )

        valid = not scan_errors

        self.record(
            "scene_scan",
            valid=valid,
            scene=scene,
            errors=scan_errors,
        )

        print("")
        print("SCENE")
        for location, _heading in locations:
            row = scene[location]
            if row["status"] == "occupied":
                value = f"{row['box']} / tag {row['tag_id']}"
            else:
                value = row["status"].upper()
            print(f"  {location:10s}: {value}")

        if not valid:
            print("")
            print("SCAN REJECTED:")
            for error in scan_errors:
                print(f"  - {error}")
            raise ValueError(
                "Scene scan ambiguous; previous valid scene was not changed"
            )

        # Compare only two fully validated scenes.
        previous = self.last_scene
        if previous is not None:
            def by_tag(value):
                result = {}
                for location, row in value.items():
                    tag_id = row.get("tag_id")
                    if tag_id in tag_to_box:
                        result[int(tag_id)] = location
                return result

            before = by_tag(previous)
            after = by_tag(scene)

            moves = []
            for tag_id in sorted(tag_to_box):
                if before[tag_id] != after[tag_id]:
                    moves.append({
                        "tag_id": tag_id,
                        "box": tag_to_box[tag_id],
                        "from": before[tag_id],
                        "to": after[tag_id],
                    })

            print("")
            if len(moves) == 1:
                move = moves[0]
                print("DETECTED HUMAN MOVE")
                print(
                    f"  {move['box']} / tag {move['tag_id']}: "
                    f"{move['from']} -> {move['to']}"
                )
            elif len(moves) == 0:
                print("SCENE DELTA: no box moved.")
            else:
                print("SCENE DELTA: multiple boxes changed:")
                for move in moves:
                    print(
                        f"  {move['box']} / tag {move['tag_id']}: "
                        f"{move['from']} -> {move['to']}"
                    )

            self.record(
                "scene_delta",
                moves=moves,
            )
        else:
            print("")
            print("Baseline scene stored.")

        self.last_scene = scene
        return scene


    def execute(self, raw):
        parts = raw.split()
        if not parts:
            return
        name = parts[0].lower()
        if name == "help":
            print(HELP)
            return
        if name == "status" and len(parts) == 1:
            self.preflight()
            self.status()
            return
        if name == "note" and len(parts) > 1:
            self.preflight()
            self.record("note", label=" ".join(parts[1:]), pose=self.status())
            return
        if self.read_only:
            raise ValueError("This connection is read-only")
        if name == "scene" and len(parts) == 1:
            self.scan_scene()
            return
        if name == "goto" and len(parts) == 2:
            self.goto_calibrated(parts[1].lower())
            return
        if name == "markstations" and len(parts) == 1:
            self.mark_stations()
            return
        if name == "tour" and len(parts) == 2:
            self.verify_box_tour(parts[1])
            return
        if name == "verifyall" and len(parts) == 1:
            self.verify_all_boxes()
            return
        if name == "cycle" and len(parts) == 2:
            self.cycle_box(parts[1])
            return
        if name == "boxopen" and len(parts) == 1:
            self.move_box_gripper(132.0)
            return
        if name == "boxgrip" and len(parts) == 1:
            self.move_box_gripper(103.0)
            return
        if name == "clearance" and len(parts) == 1:
            self.preflight()
            p = self.pose()
            if p["arm_m"] > 0.02:
                raise ValueError("Retract arm to <=0.020 m before recording clearance")
            self.clear_pose = p
            self.record("clearance", pose=p)
            print("Recorded operator-checked sweep clearance:")
            self.status()
            return
        if (name in ("heading", "lift", "arm", "grip", "pitch") and len(parts) != 2
                or name in ("open", "close") and len(parts) != 1
                or name not in ("heading", "lift", "arm", "grip", "open", "close", "pitch")):
            raise ValueError("Use help for commands")
        value = finite(parts[1]) if len(parts) == 2 else None
        self.preflight()
        p = self.pose()
        if name == "heading":
            if abs(value) > 48:
                raise ValueError("Heading must be within -48..+48 degrees from startup")
            self.check_clearance(p, turning=True)
            target = self.start["theta_rad"] + math.radians(value)
            delta = angle_delta(target, p["theta_rad"])
            if abs(delta) > math.radians(49):
                raise ValueError("Turn <=48 degrees per command; visit heading 0 between extremes")
            key, tolerance, timeout = "theta_rad", math.radians(1), 25
            command = lambda: self.robot.base.rotate_by(delta, v_r=0.15, a_r=0.15)
        elif name in ("lift", "arm"):
            if not 0 < abs(value) <= 2:
                raise ValueError("Use a nonzero change of at most 2 cm")
            key = f"{name}_m"
            joint = getattr(self.robot, name)
            target = p[key] + value / 100
            low, high = self.limits()[key]
            if name == "arm":
                high = min(high, 0.35)
            else:
                low = max(low, self.start[key] - 0.15)
                high = min(high, self.start[key] + 0.15)
            if not low <= target <= high:
                raise ValueError(f"Target {target:.3f} m outside check limits [{low:.3f}, {high:.3f}]")
            tolerance, timeout = 0.003, 10
            command = lambda: joint.move_to(target, v_m=0.01, a_m=0.02)
        elif name == "pitch":
            if "wrist_pitch" not in self.wrist:
                raise ValueError("No wrist_pitch joint available")
            if not 0 < abs(value) <= 5:
                raise ValueError("Use a nonzero pitch change of at most 5 degrees")
            self.check_clearance(p, turning=False)
            joint = self.wrist["wrist_pitch"]
            key = "wrist_pitch_rad"
            target = p[key] + math.radians(value)
            low, high = [measured(x) for x in joint.soft_motion_limits["current"]]
            if not low <= target <= high:
                raise ValueError("Pitch target exceeds the SDK's current joint limits")
            tolerance, timeout = math.radians(2.0), 8
            command = lambda: joint.move_to(target, v_des=0.08, a_des=0.16)
        else:
            maximum = self.limits()["gripper_open_units"]
            if name == "grip" and not 0 < abs(value) <= 5:
                raise ValueError("Use a nonzero gripper change of at most 5 units")
            target = maximum if name == "open" else 0.0 if name == "close" else p["gripper_units"] + value
            if not 0 <= target <= maximum:
                raise ValueError(f"Gripper target must be 0..{maximum:.1f}; negative squeeze positions disabled")
            target_rad = measured(self.gripper.pct_to_world_rad(target))
            low, high = [measured(x) for x in self.gripper.soft_motion_limits["current"]]
            if not low - 1e-9 <= target_rad <= high + 1e-9:
                raise ValueError("Gripper target exceeds SDK current limits; inspect before retrying")
            speed, accel, travel, timeout = gripper_timing(self.gripper, p["gripper_units"], target)
            print(f"Gripper profile estimate {travel:.1f} s; deadline {timeout:.1f} s "
                  "(no-progress stop remains active).")
            key, tolerance = "gripper_units", 1.5
            command = lambda: self.gripper.move_to(target, v_r=speed, a_r=accel)
        self.record("command", command=raw, target=target, coordinate=key, before=p, timeout_s=timeout)
        print(f"Moving {key} to {target:.4f}; watch clearance and keep runstop at hand.", flush=True)
        try:
            command()  # Gripper commands take effect immediately in Stretch Body.
            if key not in ("gripper_units", "wrist_pitch_rad"):
                self.robot.push_command()
            self.wait_target(key, target, tolerance=tolerance, timeout=timeout)
        except Exception as exc:
            # After dispatch even a ValueError must stop the session, not be
            # mistaken for a harmless console syntax error.
            try:
                self.record("motion_failed", command=raw, error=str(exc), pose=self.pose())
            except Exception:
                pass
            raise MotionFault(str(exc)) from exc
        self.record("target_reached", command=raw, target=target, pose=self.status())
        print("Joint feedback reached target. This does not certify clearance or a successful grasp.")


def request_stop(robot):
    """Latch runstop before SDK shutdown; never attempt a return trajectory."""
    try:
        robot.pimu.runstop_event_trigger()
        robot.pimu.push_command()
    except Exception as exc:
        print(f"Software stop failed ({exc}); use the physical runstop.", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status-only", action="store_true", help="Connect and read; send no movement commands")
    mode.add_argument("--enable-motion", action="store_true", help="Enable the attended, interactive console")
    parser.add_argument("--report", type=Path, help="New JSONL file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.enable_motion and not sys.stdin.isatty():
        parser.error("Live mode requires an interactive terminal; do not pipe a sequence of movements")
    if args.enable_motion:
        print("Stop other robot controllers first. Robot must already be homed.\n"
              "Use an empty gripper. Check the current posture and table clearance.\n"
              "Raise/retract as needed; record clearance before base turns or wrist pitching.\n"
              "This console's stop control is the PHYSICAL runstop, not the browser.")
        if input("Type READY after checking the starting posture: ").strip().upper() != "READY":
            return 1
    # Deferred so --help works in the app venv and on non-robot machines.
    from stretch_body.robot import Robot

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.report or Path(f"eval_results/real_robot_runs/motion-check-{timestamp}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        def record(event, **data):
            stream.write(json.dumps({"event": event, "time_utc": datetime.now(timezone.utc).isoformat(),
                                     **data}, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()

        record("runtime", python=sys.version, sdk=version("hello-robot-stretch-body"),
               mode="live" if args.enable_motion else "status_only")
        robot, started, fault = Robot(), False, False
        try:
            started = bool(robot.startup())
            if not started:
                raise MotionFault("SDK startup failed; stop the existing controller or inspect SDK errors")
            # SDK 0.7.31 installs its own signal handler. Handle interruption here
            # so a command cannot outlive our error path without a stop request.
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGINT, interrupted)
            signal.signal(signal.SIGTERM, interrupted)
            time.sleep(0.3)
            console = MotionConsole(robot, record, read_only=args.status_only)
            console.ready()
            console.status()
            print(f"Report: {path}")
            if args.enable_motion:
                print(HELP)
                while True:
                    raw = input("motion> ").strip()
                    if raw.lower() == "quit":
                        break
                    try:
                        console.execute(raw)
                    except ValueError as exc:
                        print(f"Refused: {exc}")
            record("finished", pose=console.pose())
        except (Exception, KeyboardInterrupt, EOFError) as exc:
            fault = True
            print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
            if started and args.enable_motion:
                request_stop(robot)
            record("fault", error=f"{type(exc).__name__}: {exc}")
        finally:
            # A failed startup can still have opened devices after acquiring
            # the SDK lock. Never stop another process that owns that lock.
            owns_lock = bool(getattr(getattr(robot, "_file_lock", None), "is_locked", False))
            if started or owns_lock:
                robot.stop()
    return 2 if fault else 0


if __name__ == "__main__":
    raise SystemExit(main())
