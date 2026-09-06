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
  heading DEG     heading relative to startup, -48..+48 (not a relative turn)
                  arm must be <= 2 cm; lift must be at least startup height
  lift CM         signed change, at most 2 cm per command (+ up, - down)
  arm CM          signed change, at most 2 cm per command (+ extend, - retract)
  open            open to this robot's configured gripper open position
  close           EMPTY gripper only: close to fingertips touching (0 units)
  grip UNITS      signed gripper change, at most 5 units (+ open, - close)
  note LABEL      record the current measured pose under a label
  quit            stop SDK; no automatic return or stow (support any object)
  Ctrl+C          request runstop and exit; physical runstop is authoritative
"""


class MotionFault(RuntimeError):
    """Stop the session; do not retry or recover automatically."""


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


class MotionConsole:
    def __init__(self, robot, record, *, clock=time.monotonic, sleep=time.sleep):
        self.robot, self.record = robot, record
        self.clock, self.sleep = clock, sleep
        self.gripper = robot.end_of_arm.get_joint("stretch_gripper")
        if self.gripper is None:
            raise MotionFault("stretch_gripper is not present")
        self.start = self.pose()
        self.stamps = {}
        self.assert_health()
        self.record("start", pose=self.start, limits=self.limits())

    def limits(self):
        return {
            "arm_m": [measured(x) for x in self.robot.arm.soft_motion_limits["current"]],
            "lift_m": [measured(x) for x in self.robot.lift.soft_motion_limits["current"]],
            "gripper_open_units": measured(self.gripper.poses["open"]),
        }

    def pose(self):
        r = self.robot
        return {
            "x_m": measured(r.base.status["x"]),
            "y_m": measured(r.base.status["y"]),
            "theta_rad": measured(r.base.status["theta"]),
            "lift_m": measured(r.lift.status["pos"]),
            "arm_m": measured(r.arm.status["pos"]),
            "gripper_units": measured(self.gripper.status["pos_pct"]),
        }

    def assert_health(self):
        r = self.robot
        if not r.is_homed():
            raise MotionFault("Robot is not homed; use the standard supervised homing procedure separately")
        if r.pimu.status.get("runstop_event", True):
            raise MotionFault("Runstop active; this console never clears it")
        devices = {"pimu": (r.pimu.status, "timestamp")}
        for name, motor in (("arm", r.arm.motor), ("lift", r.lift.motor),
                            ("left_wheel", r.base.left_wheel), ("right_wheel", r.base.right_wheel)):
            if motor.status.get("in_guarded_event") or motor.status.get("runstop_on"):
                raise MotionFault(f"{name}: guarded contact or runstop")
            devices[name] = (motor.status, "timestamp")
        if self.gripper.status.get("hardware_error"):
            raise MotionFault("Gripper reports a hardware error")
        devices["gripper"] = (self.gripper.status, "timestamp_pc")
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

    def status(self):
        p = self.pose()
        heading = math.degrees(angle_delta(p["theta_rad"], self.start["theta_rad"]))
        print(f"Heading {heading:+.2f} deg | lift {p['lift_m']:.3f} m "
              f"(change {100*(p['lift_m']-self.start['lift_m']):+.1f} cm) | "
              f"arm {p['arm_m']:.3f} m | gripper {p['gripper_units']:.1f} units")
        print(f"Startup lift {self.start['lift_m']:.3f} m; "
              f"SDK limits: {self.limits()}")
        return p

    def wait_target(self, key, target, *, tolerance, timeout):
        deadline, settled = self.clock() + timeout, None
        while self.clock() < deadline:
            self.sleep(0.05)
            self.assert_health()
            current = self.pose()[key]
            error = angle_delta(target, current) if key == "theta_rad" else target - current
            if key == "theta_rad":
                velocity = abs(measured(self.robot.base.status["theta_vel"]))
                still = velocity < math.radians(0.5)
            elif key in ("arm_m", "lift_m"):
                joint = getattr(self.robot, key[:-2])
                still = abs(measured(joint.status["vel"])) < 0.002
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

    def execute(self, raw):
        parts = raw.split()
        if not parts:
            return
        name = parts[0].lower()
        if name == "help":
            print(HELP)
            return
        if name == "status" and len(parts) == 1:
            self.ready()
            self.status()
            return
        if name == "note" and len(parts) > 1:
            self.ready()
            self.record("note", label=" ".join(parts[1:]), pose=self.status())
            return
        if (name in ("heading", "lift", "arm", "grip") and len(parts) != 2
                or name in ("open", "close") and len(parts) != 1
                or name not in ("heading", "lift", "arm", "grip", "open", "close")):
            raise ValueError("Use help for commands")
        value = finite(parts[1]) if len(parts) == 2 else None
        self.ready()
        p = self.pose()
        if name == "heading":
            if abs(value) > 48:
                raise ValueError("Heading must be within -48..+48 degrees from startup")
            if p["arm_m"] > 0.02:
                raise ValueError("Retract arm to <=0.020 m before turning; use small arm steps")
            if p["lift_m"] < self.start["lift_m"] - 0.005:
                raise ValueError("Raise lift back to startup height before turning")
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
        else:
            maximum = self.limits()["gripper_open_units"]
            if name == "grip" and not 0 < abs(value) <= 5:
                raise ValueError("Use a nonzero gripper change of at most 5 units")
            target = maximum if name == "open" else 0.0 if name == "close" else p["gripper_units"] + value
            if not 0 <= target <= maximum:
                raise ValueError(f"Gripper target must be 0..{maximum:.1f}; negative squeeze positions disabled")
            key, tolerance, timeout = "gripper_units", 1.5, 3 if name == "grip" else 30
            command = lambda: self.gripper.move_to(target, v_r=0.2, a_r=0.4)
        self.record("command", command=raw, target=target, coordinate=key, before=p)
        print(f"Moving {key} to {target:.4f}; watch clearance and keep runstop at hand.", flush=True)
        try:
            command()  # Gripper commands take effect immediately in Stretch Body.
            if key != "gripper_units":
                self.robot.push_command()
            self.wait_target(key, target, tolerance=tolerance, timeout=timeout)
        except Exception as exc:
            # After dispatch even a ValueError must stop the session, not be
            # mistaken for a harmless console syntax error.
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
        print("Stop the bridge and other robot controllers first. Robot must already be homed.\n"
              "Use an empty gripper, a retracted arm, and a starting height whose entire\n"
              "rotation sweep clears the table. Check wrist orientation and cables.\n"
              "This console's stop control is the PHYSICAL runstop, not the browser.")
        if input("Type READY after checking the starting posture and clear sweep: ").strip() != "READY":
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
            console = MotionConsole(robot, record)
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
            if started:
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
