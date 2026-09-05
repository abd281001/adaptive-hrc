"""Motion constants retained from the lab's prior Stretch 3 demo."""
from __future__ import annotations

from stretch_body import hello_utils as hu
from stretch_body import robot_params

BASE_ROTATION_SPEED_RADPS = 0.8
BASE_ROTATION_ACCEL_RADPS2 = 1.0

print_timing = False
stop_if_target_not_detected_this_many_frames = 10
stop_if_fingers_not_detected_this_many_frames = 10
max_retract_state_count = 60
min_base_speed = 0.05
grasp_target_width_m = 0.0542
grasp_if_error_below_this = 0.05
gripper_open_speed = 1.0
gripper_close_speed = 1.0
lost_target_error_too_large = 0.18
lost_target_fingertips_too_close = 0.038
successful_grasp_effort = -1.5
successful_grasp_max_fingertip_distance = 0.13
successful_grasp_min_fingertip_distance = 0.05
distance_between_fully_open_fingertips = 0.16
max_distance_for_attempted_reach = 1.0
arm_retraction_speedup = 5.0
overall_visual_servoing_velocity_scale = 1.0

joint_visual_servoing_velocity_scale = {
    "base_counterclockwise": 4.0,
    "lift_up": 6.0,
    "arm_out": 6.0,
    "wrist_yaw_counterclockwise": 4.0,
    "wrist_pitch_up": 6.0,
    "wrist_roll_counterclockwise": 1.0,
    "gripper_open": 1.0,
}

joint_state_center = {
    "lift_pos": 1.10,
    "arm_pos": 0.01,
    "wrist_yaw_pos": 0.0,
    "wrist_pitch_pos": -0.65,
    "wrist_roll_pos": 0.0,
    "gripper_pos": 10.46,
}

min_joint_state = {
    "base_odom_theta": -0.8,
    "lift_pos": 0.1,
    "arm_pos": 0.01,
    "wrist_yaw_pos": -0.20,
    "wrist_pitch_pos": -1.2,
    "wrist_roll_pos": -0.1,
    "gripper_pos": 3.0,
}


def get_dxl_joint_limits(joint: str) -> list[float]:
    params = robot_params.RobotParams().get_params()[1][joint]
    polarity = -1.0 if params["flip_encoder_polarity"] else 1.0
    return [
        polarity
        * hu.deg_to_rad(360.0 * (tick - params["zero_t"]) / 4096.0)
        / params["gr"]
        for tick in params["range_t"]
    ]


max_joint_state = {
    "base_odom_theta": 0.8,
    "lift_pos": 1.10,
    "arm_pos": 0.45,
    "wrist_yaw_pos": 1.0,
    "wrist_pitch_pos": 0.2,
    "wrist_roll_pos": 0.1,
    "gripper_pos": get_dxl_joint_limits("stretch_gripper")[1],
}

zero_vel = {
    "base_counterclockwise": 0.0,
    "lift_up": 0.0,
    "arm_out": 0.0,
    "wrist_yaw_counterclockwise": 0.0,
    "wrist_pitch_up": 0.0,
    "wrist_roll_counterclockwise": 0.0,
    "gripper_open": 0.0,
}

pos_to_vel_cmd = {
    "base_odom_theta": "base_counterclockwise",
    "lift_pos": "lift_up",
    "arm_pos": "arm_out",
    "wrist_yaw_pos": "wrist_yaw_counterclockwise",
    "wrist_pitch_pos": "wrist_pitch_up",
    "wrist_roll_pos": "wrist_roll_counterclockwise",
    "gripper_pos": "gripper_open",
}
vel_cmd_to_pos = {velocity: position for position, velocity in pos_to_vel_cmd.items()}
