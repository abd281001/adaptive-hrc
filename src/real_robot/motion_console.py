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
            while True:
                current = self.pose()[key]
                error_cm = 100.0 * (target - current)
                if abs(error_cm) <= 0.1:
                    return
                step = max(-2.0, min(2.0, error_cm))
                self.execute(f"{command} {step:.3f}")

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
        if name == "goto" and len(parts) == 2:
            self.goto_calibrated(parts[1].lower())
            return
        if name == "markstations" and len(parts) == 1:
            self.mark_stations()
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
            command = lambda: self.robot.base.rotate_by(delta, v_r=0.10, a_r=0.10)
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
