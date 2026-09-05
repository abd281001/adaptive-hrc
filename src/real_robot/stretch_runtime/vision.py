"""Vision helpers required by the ArUco grasp controller."""
from __future__ import annotations

import cv2
import numpy as np

from . import d405_helpers as dh


def draw_origin(image, camera_info, origin_xyz, color):
    center = np.round(dh.pixel_from_3d(origin_xyz, camera_info)).astype(np.int32)
    cv2.circle(image, center, 6, color, -1, lineType=cv2.LINE_AA)


def draw_text(image, origin, text_lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    location = (origin + np.array([0, -55])).astype(np.int32)
    for index, line in enumerate(text_lines):
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_size, 4)
        offset = np.array([-int(text_width / 2), index * (1.7 * text_height)]).astype(np.int32)
        cv2.putText(image, line, location + offset, font, font_size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, location + offset, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)


def clamp_cmd_by_joint_limits(cmd, joint_state, min_joint_state, max_joint_state, vel_cmd_to_pos):
    bounded = {
        key: 0.0
        if value < 0.0 and joint_state[vel_cmd_to_pos[key]] < min_joint_state[vel_cmd_to_pos[key]]
        else value
        for key, value in cmd.items()
    }
    return {
        key: 0.0
        if value > 0.0 and joint_state[vel_cmd_to_pos[key]] > max_joint_state[vel_cmd_to_pos[key]]
        else value
        for key, value in bounded.items()
    }

