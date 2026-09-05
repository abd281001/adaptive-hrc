import time
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import d405_helpers as dh
from . import normalized_velocity_control as nvc
from . import aruco_detector as ad
from . import aruco_to_fingertips as af
from . import loop_timer as lt

from .config import *
from .base_motion import (
    recenter_robot,
    return_arm_to_carry_with_object,
    save_base_pose,
    return_base_to_saved_pose,
)
from .vision import clamp_cmd_by_joint_limits, draw_origin, draw_text


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return value


def capture_grasp_snapshot(robot, joint_state):
    base_pose = save_base_pose(robot)
    return {
        "base_pose": {
            "x": _to_float(base_pose["x"]),
            "y": _to_float(base_pose["y"]),
            "theta": _to_float(base_pose["theta"]),
        },
        "arm_pos": _to_float(joint_state["arm_pos"]),
        "lift_pos": _to_float(joint_state["lift_pos"]),
        "wrist_yaw_pos": _to_float(joint_state["wrist_yaw_pos"]),
        "wrist_pitch_pos": _to_float(joint_state["wrist_pitch_pos"]),
        "wrist_roll_pos": _to_float(joint_state["wrist_roll_pos"]),
        "gripper_pos": _to_float(joint_state["gripper_pos"]),
        "table1_pregrasp_pose": None,
    }





def execute_grasp_and_return_base(
    robot,
    camera_service,
    target_tag_id,
    target_tag_name,
    marker_info,
    max_duration_s=45.0,
    deadline=None,
    command_timeout_s=15.0,
    command_watchdog_s=0.30,
    stable_frames_required=3,
    max_position_jump_m=0.035,
    max_frame_age_s=0.30,
    velocity_scale=1.0,
    cancel_event=None,
    show_visualization=False,
    verbose=False,
):
    controller = None
    closed_gripper_pos = None
    grasp_success = False
    grasp_snapshot = None
    table1_pregrasp_pose = None

    try:
        if not marker_info:
            raise ValueError("marker_info must define the target and fingertip ArUco markers")
        if max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive")
        log = print if verbose else (lambda *_args, **_kwargs: None)
        log("\n=== Starting ArUco grasping ===")

        if deadline is None:
            deadline = time.monotonic() + max_duration_s
        else:
            deadline = min(float(deadline), time.monotonic() + max_duration_s)

        recenter_robot(
            robot, joint_state_center, timeout=command_timeout_s,
            deadline=deadline, cancel_event=cancel_event,
        )

        controller = nvc.NormalizedVelocityControl(
            robot, command_watchdog_s=command_watchdog_s
        )
        controller.reset_base_odometry()
        time.sleep(0.2)

        table1_pregrasp_pose = save_base_pose(robot)
        log("Saved table1_pregrasp_pose:", table1_pregrasp_pose)

        aruco_detector = ad.ArucoDetector(
            marker_info=marker_info,
            show_debug_images=show_visualization,
            use_apriltag_refinement=False,
            brighten_images=True,
        )

        aruco_to_fingertips = af.ArucoToFingertips(
            default_height_above_mounting_surface=af.suctioncup_height["cup_bottom"]
        )

        first_frame = True
        state = "reach"
        grasping_the_target = False
        pre_reach = True

        distance_between_fingertips = distance_between_fully_open_fingertips

        frames_since_target_detected = 0
        frames_since_fingers_detected = 0
        retract_state_count = 0

        loop_timer = lt.LoopTimer()
        fingertips = {}

        camera_info = None
        last_frame_id = -1
        started_at = time.monotonic()
        stable_marker_frames = 0
        last_positions = None

        while True:
            loop_timer.start_of_iteration()

            if cancel_event is not None and cancel_event.is_set():
                log("Grasp cancelled by stop request.")
                if controller is not None:
                    controller.set_command(zero_vel.copy())
                break
            if time.monotonic() >= deadline:
                log(f"Grasp timed out after {max_duration_s:.1f} seconds.")
                if controller is not None:
                    controller.set_command(zero_vel.copy())
                break

            target_xyz = None
            fingertip_left_pos = None
            fingertip_right_pos = None
            between_fingertips = None
            distance_between_fingertips = None

            bundle = camera_service.get_latest_bundle()
            color_image = bundle["color"]
            depth_image = bundle["depth"]
            depth_camera_info = bundle["depth_camera_info"]
            color_camera_info = bundle["color_camera_info"]
            frame_id = bundle["frame_id"]
            frame_age_s = bundle.get("age_s")

            if (
                color_image is None or depth_image is None
                or frame_age_s is None or frame_age_s > max_frame_age_s
            ):
                if controller is not None:
                    controller.set_command(zero_vel.copy())
                time.sleep(0.01)
                continue

            if frame_id == last_frame_id:
                if controller is not None:
                    controller.set_command(zero_vel.copy())
                time.sleep(0.002)
                continue
            last_frame_id = frame_id

            if first_frame:
                # Detection runs on the color image, so its pose estimate must
                # use color intrinsics.  Depth intrinsics are not interchangeable.
                camera_info = color_camera_info
                log("depth camera_info:", depth_camera_info)
                log("color camera_info:", color_camera_info)
                first_frame = False

            if camera_info is None:
                time.sleep(0.01)
                continue

            image = np.copy(color_image)
            aruco_detector.update(color_image, camera_info)
            markers = aruco_detector.get_detected_marker_dict()
            fingertips = aruco_to_fingertips.get_fingertips(markers)

            target_corners = None
            target_center = None
            target_marker = markers.get(target_tag_id)
            if target_marker is not None:
                target_xyz = np.array(target_marker["pos"]).reshape(3)
                target_corners = np.array(target_marker["corners"])
                target_center = np.array(target_marker["center"])

            if target_corners is not None:
                corners = target_corners.astype(int)
                ptA, ptB, ptC, ptD = corners
                cv2.line(image, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
                cv2.line(image, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
                cv2.line(image, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
                cv2.line(image, tuple(ptD), tuple(ptA), (0, 255, 0), 2)
                cv2.circle(image, tuple(target_center.astype(int)), 5, (0, 0, 255), -1)
                cv2.putText(
                    image,
                    f"tagId: {target_tag_id}, {target_tag_name}",
                    (ptA[0], ptA[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            log("ArUco Target Detection:", "SUCCEEDED" if target_xyz is not None else "FAILED")

            f = fingertips.get("left", None)
            if f is not None:
                fingertip_left_pos = f["pos"]
                log("Left Finger ArUco Marker Detection: SUCCEEDED")
            else:
                log("Left Finger ArUco Marker Detection: FAILED")

            f = fingertips.get("right", None)
            if f is not None:
                fingertip_right_pos = f["pos"]
                log("Right Finger ArUco Marker Detection: SUCCEEDED")
            else:
                log("Right Finger ArUco Marker Detection: FAILED")

            marker_positions = (
                target_xyz, fingertip_left_pos, fingertip_right_pos
            )
            if all(value is not None for value in marker_positions):
                if last_positions is None or all(
                    np.linalg.norm(current - previous) <= max_position_jump_m
                    for current, previous in zip(marker_positions, last_positions)
                ):
                    stable_marker_frames += 1
                else:
                    stable_marker_frames = 1
                    controller.set_command(zero_vel.copy())
                last_positions = tuple(np.array(value) for value in marker_positions)
            else:
                stable_marker_frames = 0
                last_positions = None
                controller.set_command(zero_vel.copy())

            markers_stable = stable_marker_frames >= stable_frames_required
            if markers_stable and (fingertip_left_pos is not None) and (fingertip_right_pos is not None):
                between_fingertips = (fingertip_left_pos + fingertip_right_pos) / 2.0
                distance_between_fingertips = np.linalg.norm(fingertip_left_pos - fingertip_right_pos)
            else:
                # Live runs require both observed fingertip tags.  The prior
                # fixed synthetic midpoint could drive the arm from an
                # unobserved gripper pose and is intentionally forbidden.
                between_fingertips = None
                distance_between_fingertips = None

            joint_state = controller.get_joint_state()
            theta = joint_state["base_odom_theta"]
            joint_state["base_odom_theta"] = math.atan2(math.sin(theta), math.cos(theta))

            log("gripper effort = {:.2f}".format(joint_state["gripper_eff"]))
            if distance_between_fingertips is not None:
                log("distance_between_fingertips = {:.2f} cm".format(100.0 * distance_between_fingertips))

            if target_xyz is not None:
                frames_since_target_detected = 0
            else:
                frames_since_target_detected += 1

            if between_fingertips is not None:
                frames_since_fingers_detected = 0
            else:
                frames_since_fingers_detected += 1

            log("state =", state)
            log("pre_reach =", pre_reach)
            log("grasping_the_target =", grasping_the_target)

            if distance_between_fingertips is not None:
                if distance_between_fingertips < lost_target_fingertips_too_close and grasping_the_target:
                    log("lost target: fingertip distance")
                    grasping_the_target = False

            position_error = None
            target_error = None
            if (between_fingertips is not None) and (target_xyz is not None):
                position_error = target_xyz - between_fingertips
                target_error = np.linalg.norm(position_error)
                log("target_error = {:.2f} cm".format(100.0 * target_error))

                if target_error > lost_target_error_too_large and grasping_the_target:
                    log("lost target: pose error")
                    grasping_the_target = False

            if state == "retract":
                cmd = {
                    "lift_up": 0.7,
                    "arm_out": -1.0,
                }

                cmd = clamp_cmd_by_joint_limits(
                    cmd, joint_state, min_joint_state, max_joint_state, vel_cmd_to_pos
                )
                controller.set_command(cmd)
                retract_state_count += 1

                retract_done = (
                    (not grasping_the_target)
                    or (retract_state_count > max_retract_state_count)
                    or (joint_state["arm_pos"] < 0.02)
                )

                if retract_done:
                    controller.set_command(zero_vel.copy())
                    time.sleep(0.2)

                    if grasping_the_target:
                        grasp_success = True
                        closed_gripper_pos = joint_state["gripper_pos"]
                        log("Retract finished with stable grasp.")
                    else:
                        log("Retract finished without a stable grasp.")

                    break

            elif state == "reach":
                if pre_reach:
                    cmd = {}

                    gripper_ready = False
                    if joint_state["gripper_pos"] >= (0.9 * max_joint_state["gripper_pos"]):
                        gripper_ready = True
                        cmd["gripper_open"] = 0.0
                    elif not grasping_the_target:
                        cmd["gripper_open"] = gripper_open_speed

                    cmd["wrist_pitch_up"] = 0.0

                    if gripper_ready:
                        pre_reach = False
                        cmd = zero_vel.copy()

                    if cmd:
                        cmd = {k: overall_visual_servoing_velocity_scale * velocity_scale * v for (k, v) in cmd.items()}
                        cmd = {k: joint_visual_servoing_velocity_scale[k] * v for (k, v) in cmd.items()}
                        cmd = clamp_cmd_by_joint_limits(
                            cmd,
                            joint_state,
                            min_joint_state,
                            max_joint_state,
                            vel_cmd_to_pos,
                        )
                        controller.set_command(cmd)

                elif (
                    (between_fingertips is not None)
                    and (target_xyz is not None)
                    and (target_error is not None)
                    and (target_error <= max_distance_for_attempted_reach)
                ):
                    x_error, y_error, z_error = position_error

                    yaw_velocity = -x_error
                    pitch_velocity = -y_error
                    roll_velocity = 0.0 - joint_state["wrist_roll_pos"]

                    yaw = joint_state["wrist_yaw_pos"]
                    pitch = -joint_state["wrist_pitch_pos"]
                    roll = -joint_state["wrist_roll_pos"]
                    r = Rotation.from_euler("yxz", [yaw, pitch, roll]).as_matrix()

                    rotated_lift = np.matmul(r, np.array([0.0, -1.0, 0.0]))
                    rotated_arm = np.matmul(r, np.array([0.0, 0.0, 1.0]))
                    rotated_base = np.matmul(r, np.array([-1.0, 0.0, 0.0]))

                    lift_velocity = np.dot(rotated_lift, position_error)
                    arm_velocity = np.dot(rotated_arm, position_error)
                    base_rotational_velocity = np.dot(rotated_base, position_error)

                    if abs(base_rotational_velocity) < min_base_speed:
                        base_rotational_velocity = 0.0

                    if arm_velocity < 0.0:
                        arm_velocity = arm_retraction_speedup * arm_velocity

                    cmd = {
                        "lift_up": lift_velocity,
                        "arm_out": arm_velocity,
                        "wrist_yaw_counterclockwise": yaw_velocity,
                        "wrist_pitch_up": pitch_velocity,
                        "wrist_roll_counterclockwise": roll_velocity,
                        "base_counterclockwise": base_rotational_velocity,
                    }

                    if target_error < grasp_if_error_below_this:
                        cmd["gripper_open"] = -gripper_close_speed

                        if (
                            (not grasping_the_target)
                            and (joint_state["gripper_eff"] < successful_grasp_effort)
                            and (distance_between_fingertips < successful_grasp_max_fingertip_distance)
                            and (distance_between_fingertips > successful_grasp_min_fingertip_distance)
                        ):
                            log("stable grasp detected")
                            grasping_the_target = True

                            grasp_snapshot = capture_grasp_snapshot(robot, joint_state)
                            grasp_snapshot["table1_pregrasp_pose"] = {
                                "x": _to_float(table1_pregrasp_pose["x"]),
                                "y": _to_float(table1_pregrasp_pose["y"]),
                                "theta": _to_float(table1_pregrasp_pose["theta"]),
                            }
                            log("Captured grasp_snapshot:", grasp_snapshot)

                            state = "retract"
                            retract_state_count = 0
                    else:
                        cmd["gripper_open"] = gripper_open_speed

                    cmd = {k: overall_visual_servoing_velocity_scale * velocity_scale * v for (k, v) in cmd.items()}
                    cmd = {k: joint_visual_servoing_velocity_scale[k] * v for (k, v) in cmd.items()}

                    cmd = clamp_cmd_by_joint_limits(
                        cmd,
                        joint_state,
                        min_joint_state,
                        max_joint_state,
                        vel_cmd_to_pos,
                    )
                    controller.set_command(cmd)

                else:
                    stop_joints = zero_vel.copy()

                    if frames_since_target_detected >= stop_if_target_not_detected_this_many_frames:
                        cmd = stop_joints
                        cmd["gripper_open"] = gripper_open_speed
                    elif frames_since_fingers_detected >= stop_if_fingers_not_detected_this_many_frames:
                        cmd = stop_joints
                    else:
                        cmd = stop_joints

                    cmd = clamp_cmd_by_joint_limits(
                        cmd,
                        joint_state,
                        min_joint_state,
                        max_joint_state,
                        vel_cmd_to_pos,
                    )
                    controller.set_command(cmd)

            annotations = {
                "apriltags": [],
                "fingertips": fingertips,
                "camera_info": camera_info,
                "aruco_to_fingertips": aruco_to_fingertips,
                "origins": [],
                "texts": [],
            }

            if target_xyz is not None and target_corners is not None:
                tag_annotation = {
                    "id": target_tag_id,
                    "name": target_tag_name,
                    "corners": target_corners,
                    "center": target_center,
                    "position": target_xyz,
                }
                annotations["apriltags"].append(tag_annotation)

            if target_xyz is not None:
                annotations["origins"].append({
                    "position": target_xyz,
                    "color": (255, 0, 0),
                })
                x, y, z = target_xyz * 100.0
                text_lines = [
                    "{:.1f} cm wide".format(grasp_target_width_m * 100.0),
                    "{:.1f}, {:.1f}, {:.1f} cm".format(x, y, z),
                ]
                center = np.round(dh.pixel_from_3d(target_xyz, camera_info)).astype(np.int32)
                annotations["texts"].append({
                    "position": center,
                    "lines": text_lines,
                    "color": (255, 255, 255),
                })

            if between_fingertips is not None:
                annotations["origins"].append({
                    "position": between_fingertips,
                    "color": (255, 255, 255),
                })

            camera_service.set_color_annotations(annotations)

            if show_visualization:
                if target_xyz is not None:
                    draw_origin(image, camera_info, target_xyz, (255, 0, 0))
                    x, y, z = target_xyz * 100.0
                    text_lines = [
                        "{:.1f} cm wide".format(grasp_target_width_m * 100.0),
                        "{:.1f}, {:.1f}, {:.1f} cm".format(x, y, z),
                    ]
                    center = np.round(dh.pixel_from_3d(target_xyz, camera_info)).astype(np.int32)
                    draw_text(image, center, text_lines)

                if between_fingertips is not None:
                    draw_origin(image, camera_info, between_fingertips, (255, 255, 255))

                aruco_to_fingertips.draw_fingertip_frames(
                    fingertips,
                    image,
                    camera_info,
                    axis_length_in_m=0.02,
                    draw_origins=True,
                    write_coordinates=True,
                )

                cv2.imshow("Core Grasping", image)
                cv2.waitKey(1)

            loop_timer.end_of_iteration()
            if print_timing:
                loop_timer.pretty_print(minimum=True)

        if controller is not None:
            controller.stop()
            time.sleep(0.3)
            controller = None

        if grasp_success:
            return_arm_to_carry_with_object(
                robot, joint_state_center, closed_gripper_pos,
                timeout=command_timeout_s, deadline=deadline,
                cancel_event=cancel_event,
            )
            return_base_to_saved_pose(
                robot, table1_pregrasp_pose, timeout=command_timeout_s,
                deadline=deadline, cancel_event=cancel_event,
            )
            return 0, grasp_snapshot

        return 1, None

    finally:
        camera_service.set_color_annotations(None)

        if show_visualization:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if controller is not None:
            controller.stop()
