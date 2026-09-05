# Author: Gabriel Armas Aranibar

from .routes import push_and_wait
from .config import joint_state_center
from .base_motion import (
    recenter_robot,
)


def hand_object_on_workspace(
    robot,
    arm_extension_delta=0.15,
    lift_down_pos=0.91,
    lift_up_after_release_pos=None,
    retract_after_release=True,
    retract_delta=0.13,
    command_timeout_s=15.0,
    deadline=None,
    cancel_event=None,
):

    if lift_up_after_release_pos is None:
        lift_up_after_release_pos = joint_state_center["lift_pos"]

    print("\n=== Releasing object on table ===")

    # 1. Extend arm forward
    print(f"Extending arm forward by {arm_extension_delta} m...")
    robot.arm.move_by(arm_extension_delta)
    if not push_and_wait(robot, "extend arm", timeout=command_timeout_s, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out extending arm for placement")

    # 2. Lower lift
    print(f"Moving lift to {lift_down_pos} m...")
    robot.lift.move_to(lift_down_pos)
    if not push_and_wait(robot, "lower lift", timeout=command_timeout_s, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out lowering lift for placement")

    # 3. Open gripper
    print("Opening gripper fully to release object...")
    gripper = robot.end_of_arm.get_joint("stretch_gripper")
    gripper.pose("open")
    if not push_and_wait(robot, "open gripper", timeout=command_timeout_s, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out opening gripper for placement")

    # 4. Raise lift
    print(f"Raising lift back to {lift_up_after_release_pos} m...")
    robot.lift.move_to(lift_up_after_release_pos)
    if not push_and_wait(robot, "raise lift", timeout=command_timeout_s, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out raising lift after placement")

    recenter_robot(
        robot, joint_state_center, timeout=command_timeout_s,
        deadline=deadline, cancel_event=cancel_event,
    )

    return 0
