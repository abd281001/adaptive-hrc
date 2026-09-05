"""Fail-closed, stable-marker checks used for physical postconditions."""
from __future__ import annotations

import time

import numpy as np

def wait_for_stable_marker(
    *, camera_service, marker_info, marker_id, expected_xyz_m, tolerance_m,
    stable_frames, max_position_jump_m, max_frame_age_s, deadline,
    cancel_event=None,
    _detector_type=None,
):
    """Return measured marker evidence only after a stable in-region detection."""
    if _detector_type is None:
        from .aruco_detector import ArucoDetector
        _detector_type = ArucoDetector
    detector = _detector_type(
        marker_info=marker_info, show_debug_images=False,
        use_apriltag_refinement=False, brighten_images=True,
    )
    expected = np.asarray(expected_xyz_m, dtype=float).reshape(3)
    previous = None
    accepted = 0
    last_frame_id = -1
    last_reason = "no frame"
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("postcondition cancelled")
        bundle = camera_service.get_latest_bundle()
        frame_id = bundle.get("frame_id", -1)
        age_s = bundle.get("age_s")
        color = bundle.get("color")
        camera_info = bundle.get("color_camera_info")
        if (
            color is None or camera_info is None or age_s is None
            or age_s > max_frame_age_s or frame_id == last_frame_id
        ):
            last_reason = "missing, stale, or repeated camera frame"
            time.sleep(0.01)
            continue
        last_frame_id = frame_id
        detector.update(color, camera_info)
        marker = detector.get_detected_marker_dict().get(marker_id)
        if marker is None:
            accepted = 0
            previous = None
            last_reason = "marker not detected with acceptable quality"
            continue
        position = np.asarray(marker["pos"], dtype=float).reshape(3)
        jump = 0.0 if previous is None else float(np.linalg.norm(position - previous))
        region_error = float(np.linalg.norm(position - expected))
        previous = position
        if jump > max_position_jump_m:
            accepted = 1
            last_reason = f"marker jump {jump:.4f} m"
            continue
        if region_error > tolerance_m:
            accepted = 0
            last_reason = f"marker is {region_error:.4f} m outside expected slot pose"
            continue
        accepted += 1
        if accepted >= stable_frames:
            return {
                "marker_id": int(marker_id),
                "position_m": [float(value) for value in position],
                "expected_position_m": [float(value) for value in expected],
                "position_error_m": region_error,
                "stable_frames": accepted,
                "min_corner_distance_px": float(marker["min_dist_between_corners"]),
                "reprojection_error_px": float(marker["reprojection_error_px"]),
                "frame_id": int(frame_id),
                "frame_age_s": float(age_s),
            }
    raise RuntimeError(f"placement postcondition timed out: {last_reason}")
