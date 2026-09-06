#!/usr/bin/env python
"""Manual base-rotation aid for surveying table layout before a config exists.

Run with the Stretch SDK Python, at the physical E-stop, robot already homed:

    python -m src.real_robot.layout_survey --i-understand-this-moves-the-robot

Prompt commands:
    rotate <deg>   rotate from current heading (signed, CCW positive);
                   prints the heading to record as the station's heading_deg.
    reach [m]      extend the arm by <m> (default 0.20) from carry pose, then
                   retract, to check table reach at the current heading.
    home           rotate back to the pose marked as home at startup.
    quit           recenter the arm/wrist and stop.

Mark the floor under the base before running: wherever the robot is at start
becomes home (0 deg) for the session.
"""
from __future__ import annotations

import argparse
import math

from .stretch_runtime.base_motion import (
    recenter_robot,
    return_base_to_saved_pose,
    save_base_pose,
)
from .stretch_runtime.config import joint_state_center, max_joint_state
from .stretch_runtime.routes import rotate_deg


def _checked_homed(robot) -> bool:
    try:
        return bool(robot.is_homed())
    except AttributeError:
        return bool(robot.is_calibrated())


def _runstop_active(robot) -> bool:
    pimu = getattr(robot, "pimu", None)
    status = getattr(pimu, "status", {}) or {}
    return bool(status.get("runstop_event"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--i-understand-this-moves-the-robot", action="store_true")
    parser.add_argument("--velocity-scale", type=float, default=0.25, help="Fraction of normal rotation speed (default 0.25, matches calibration-mode cap).")
    args = parser.parse_args(argv)
    if not args.i_understand_this_moves_the_robot:
        raise SystemExit("pass --i-understand-this-moves-the-robot to run this")
    if not 0.0 < args.velocity_scale <= 1.0:
        raise SystemExit("--velocity-scale must be in (0, 1]")

    from stretch_body.robot import Robot  # deferred: only present in the Stretch SDK env

    robot = Robot()
    if not robot.startup():
        raise SystemExit("could not connect to Stretch hardware")
    if not _checked_homed(robot):
        raise SystemExit("Stretch is not homed; home it with the standard supervised procedure first")
    if _runstop_active(robot):
        raise SystemExit("Stretch runstop is active; release it before surveying")

    print("Recentering wrist/arm/lift/gripper to the carry pose...")
    recenter_robot(robot, joint_state_center)
    home_pose = save_base_pose(robot)
    max_reach_m = max_joint_state["arm_pos"]
    print(f"Marked current base pose as home (0.0 deg). Arm's physical extension limit is {max_reach_m:.2f} m.")
    print("Commands: rotate <deg> | reach [m] | home | quit\n")

    try:
        while True:
            try:
                raw = input("layout> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            parts = raw.split()
            command = parts[0].lower()

            if command == "quit":
                break

            elif command == "rotate" and len(parts) == 2:
                try:
                    delta_deg = float(parts[1])
                except ValueError:
                    print("  usage: rotate <degrees>")
                    continue
                if not rotate_deg(robot, delta_deg, velocity_scale=args.velocity_scale):
                    print("  WARNING: rotation wait timed out; inspect the base before continuing")
                    continue
                theta_now = float(robot.base.status["theta"])
                heading_from_home = math.degrees(theta_now - home_pose["theta"])
                heading_from_home = (heading_from_home + 180.0) % 360.0 - 180.0
                print(f"  now at {heading_from_home:+.2f} deg from marked home -- record this as heading_deg")

            elif command == "reach":
                extension_m = 0.20
                if len(parts) == 2:
                    try:
                        extension_m = float(parts[1])
                    except ValueError:
                        print("  usage: reach [meters]")
                        continue
                if not 0.0 < extension_m <= max_reach_m - 0.05:
                    print(f"  refusing: extension must be in (0, {max_reach_m - 0.05:.2f}] m (0.05 m safety margin below the joint limit)")
                    continue
                print(f"  extending arm {extension_m:.2f} m -- look at where the gripper sits over the table")
                robot.arm.move_by(extension_m)
                robot.push_command()
                robot.wait_command(timeout=15.0)
                input("  press Enter once you've noted the reach, to retract...")
                robot.arm.move_to(joint_state_center["arm_pos"])
                robot.push_command()
                robot.wait_command(timeout=15.0)
                print("  retracted to carry pose")

            elif command == "home":
                try:
                    return_base_to_saved_pose(robot, home_pose)
                    print("  returned to marked home heading")
                except RuntimeError as exc:
                    print(f"  REFUSED: {exc}")
                    print("  the base has drifted; reconcile its position by hand before continuing")

            else:
                print("  unknown command. Use: rotate <deg> | reach [m] | home | quit")

    except KeyboardInterrupt:
        print("\ninterrupted -- stopping")
    finally:
        try:
            recenter_robot(robot, joint_state_center)
        except Exception as exc:
            print(f"  (could not recenter cleanly: {exc})")
        robot.stop()
        print("Robot stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
