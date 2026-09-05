# base_motion.py - Base movement and pose management functions

import math
from .config import joint_state_center
from .routes import push_and_wait

def normalize_angle_rad(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def save_base_pose(robot):
    return {
        "x": float(robot.base.status["x"]),
        "y": float(robot.base.status["y"]),
        "theta": float(robot.base.status["theta"]),
    }

def return_base_to_saved_pose(
    robot, saved_pose, *, timeout=15.0, deadline=None, cancel_event=None,
    heading_tolerance_rad=math.radians(3.0), translation_tolerance_m=0.025,
):
    current_x = float(robot.base.status["x"])
    current_y = float(robot.base.status["y"])
    current_theta = float(robot.base.status["theta"])

    dx = saved_pose["x"] - current_x
    dy = saved_pose["y"] - current_y

    distance = math.sqrt(dx * dx + dy * dy)

    print("\n=== Returning base to saved TABLE 1 pre-grasp pose ===")
    print(f"Current pose: x={current_x:.3f}, y={current_y:.3f}, theta={math.degrees(current_theta):.2f} deg")
    print(f"Saved pose:   x={saved_pose['x']:.3f}, y={saved_pose['y']:.3f}, theta={math.degrees(saved_pose['theta']):.2f} deg")

    # The publication layout is rotation-only.  A meaningful x/y displacement
    # means the fixed-base assumption was violated; do not improvise a
    # translation through a participant workspace.
    if distance > translation_tolerance_m:
        raise RuntimeError(
            f"visual servo displaced base by {distance:.3f} m; manual pose reconciliation required"
        )

    final_theta = float(robot.base.status["theta"])
    final_theta_error = normalize_angle_rad(saved_pose["theta"] - final_theta)

    if abs(final_theta_error) > math.radians(0.5):
        print(f"Restore final orientation by {math.degrees(final_theta_error):.2f} deg")
        robot.base.rotate_by(
            final_theta_error,
            v_r=0.8,  # BASE_ROTATION_SPEED_RADPS
            a_r=1.0,  # BASE_ROTATION_ACCEL_RADPS2
        )
        if not push_and_wait(
            robot, "return_to_saved_pose: final orientation", timeout=timeout,
            deadline=deadline, cancel_event=cancel_event,
        ):
            raise RuntimeError("timed out restoring the pre-grasp base heading")

    restored_theta = float(robot.base.status["theta"])
    residual = abs(normalize_angle_rad(saved_pose["theta"] - restored_theta))
    if residual > heading_tolerance_rad:
        raise RuntimeError(
            f"base heading residual {math.degrees(residual):.2f} deg exceeds tolerance"
        )

    print("=== Base returned to saved pre-grasp pose ===")

def recenter_robot(robot, joint_state_center, *, timeout=15.0, deadline=None, cancel_event=None):
    robot.end_of_arm.get_joint('wrist_yaw').move_to(joint_state_center['wrist_yaw_pos'])
    robot.end_of_arm.get_joint('wrist_pitch').move_to(joint_state_center['wrist_pitch_pos'])
    if not push_and_wait(robot, "center wrist", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out centering wrist")

    robot.arm.move_to(joint_state_center['arm_pos'])
    if not push_and_wait(robot, "retract arm", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out retracting arm")

    robot.lift.move_to(joint_state_center['lift_pos'])
    if not push_and_wait(robot, "raise lift to carry pose", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out moving lift to carry pose")

    robot.end_of_arm.get_joint('stretch_gripper').move_to(joint_state_center['gripper_pos'])
    if not push_and_wait(robot, "set gripper center pose", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out setting gripper center pose")

def return_arm_to_carry_with_object(robot, joint_state_center, closed_gripper_pos, *, timeout=15.0, deadline=None, cancel_event=None):
    robot.end_of_arm.get_joint('wrist_yaw').move_to(joint_state_center['wrist_yaw_pos'])
    robot.end_of_arm.get_joint('wrist_pitch').move_to(joint_state_center['wrist_pitch_pos'])
    if not push_and_wait(robot, "carry: center wrist", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out centering wrist for carry")

    robot.arm.move_to(joint_state_center['arm_pos'])
    if not push_and_wait(robot, "carry: retract arm", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out retracting arm for carry")

    robot.lift.move_to(joint_state_center['lift_pos'])
    if not push_and_wait(robot, "carry: raise lift", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out raising lift for carry")

    robot.end_of_arm.get_joint('stretch_gripper').move_to(closed_gripper_pos)
    if not push_and_wait(robot, "carry: hold gripper", timeout=timeout, deadline=deadline, cancel_event=cancel_event):
        raise RuntimeError("timed out setting carry gripper pose")
