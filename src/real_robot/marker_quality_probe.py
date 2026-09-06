#!/usr/bin/env python
"""Live ArUco marker preview -- no grasp, no bridge, no robot motion.

Prints a marker's pixel footprint, solvePnP reprojection error, depth and
measured position for calibration_record.json's marker_quality_trials, which
nothing in the production path otherwise reports outside an active grasp.

Starts only the D405 camera service directly, so it can run alongside
layout_survey.py once the base is at the station to sample. Reuses
ArucoDetector exactly as the production grasp loop does.

Usage (Stretch SDK python, D405 mounted, robot facing the target station):

    python -m src.real_robot.marker_quality_probe \
        --config robot_configs/lab_site.json \
        --marker-id 12 --station-id produce

Streams a live readout every ~0.3 s. Press Enter to freeze the current
reading and print a ready-to-paste marker_quality_trials JSON row; press
Enter again to keep sampling, or type quit.
"""
from __future__ import annotations

import argparse
import time

from .config import load_lab_config


def _marker_info(config) -> dict:
    """Reproduce stretch_hardware._marker_info() without a Robot() connection."""
    p = config.perception
    quality = {
        "min_marker_pixels": p.min_marker_pixels,
        "max_reprojection_error_px": p.max_reprojection_error_px,
        "min_marker_depth_m": p.min_marker_depth_m,
        "max_marker_depth_m": p.max_marker_depth_m,
    }
    left, right = config.fingertip_marker_ids
    rows: dict = {
        str(left): {"length_mm": p.fingertip_marker_length_mm, "use_rgb_only": True, "name": "finger_left", "link": "link_finger_left", "type": "aruco", **quality},
        str(right): {"length_mm": p.fingertip_marker_length_mm, "use_rgb_only": True, "name": "finger_right", "link": "link_finger_right", "type": "aruco", **quality},
        "default": {"length_mm": 24.0, "use_rgb_only": True, "name": "unknown", "link": "None", "type": "aruco", **quality},
    }
    for item in config.objects.values():
        rows[str(item.marker_id)] = {
            "length_mm": item.marker_length_mm, "use_rgb_only": True,
            "name": item.object_id, "link": "None", "type": "aruco", **quality,
        }
    reference = config.reference_marker
    rows[str(reference.marker_id)] = {
        "length_mm": reference.marker_length_mm, "use_rgb_only": True,
        "name": "table_reference", "link": "None", "type": "aruco", **quality,
    }
    return rows


def _required_stations(config, marker_id: int) -> set[str]:
    if marker_id == config.reference_marker.marker_id or marker_id in config.fingertip_marker_ids:
        return set(config.stations)
    return {
        item.source_station for item in config.objects.values()
        if item.marker_id == marker_id
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument("--station-id", required=True, help="Written into the printed row's station_id; must match where the robot is actually pointed.")
    args = parser.parse_args(argv)

    config = load_lab_config(args.config)
    marker_info = _marker_info(config)
    if str(args.marker_id) not in marker_info:
        raise SystemExit(f"marker id {args.marker_id} is not configured (known ids: {sorted(int(k) for k in marker_info if k != 'default')})")
    required = _required_stations(config, args.marker_id)
    if args.station_id not in config.stations:
        raise SystemExit(f"unknown station {args.station_id!r}; configured stations are {sorted(config.stations)}")
    if args.station_id not in required:
        print(f"NOTE: marker {args.marker_id} is not expected to be sampled at {args.station_id!r} "
              f"(expected at {sorted(required)}); continuing anyway since you know your physical setup.")

    from .stretch_runtime.camera_service import D405CameraService  # deferred: pyrealsense2 only in the Stretch SDK env
    from .stretch_runtime.aruco_detector import ArucoDetector

    p = config.perception
    camera = D405CameraService(
        exposure="medium", capture_fps=15, stream_fps=10,
        startup_timeout_s=p.camera_startup_timeout_s,
        frame_timeout_ms=p.camera_frame_timeout_ms,
        stale_after_s=p.max_frame_age_s,
    )
    camera.start()
    status = camera.get_status()
    if not status.get("ready"):
        camera.stop()
        raise SystemExit(f"D405 did not become ready: {status}")
    print(f"Camera ready. Sampling marker {args.marker_id} at station {args.station_id!r}. "
          "Press Enter to freeze a reading, or type 'quit'.\n")

    detector = ArucoDetector(marker_info=marker_info, show_debug_images=False, use_apriltag_refinement=False, brighten_images=True)
    last_frame_id = -1

    try:
        while True:
            bundle = camera.get_latest_bundle()
            color = bundle.get("color")
            camera_info = bundle.get("color_camera_info")
            frame_id = bundle.get("frame_id", -1)
            if color is None or camera_info is None or frame_id == last_frame_id:
                time.sleep(0.02)
                continue
            last_frame_id = frame_id
            detector.update(color, camera_info)
            marker = detector.get_detected_marker_dict().get(args.marker_id)
            if marker is None:
                print("\r  not detected" + " " * 60, end="", flush=True)
                reading = None
            else:
                position_m = [float(value) for value in marker["pos"]]
                min_marker_pixels = float(marker["min_dist_between_corners"])
                reprojection_error_px = float(marker["reprojection_error_px"])
                depth_m = position_m[2]
                print(
                    f"\r  pos=[{position_m[0]:+.3f}, {position_m[1]:+.3f}, {position_m[2]:+.3f}] m  "
                    f"min_marker_pixels={min_marker_pixels:7.2f}  "
                    f"reprojection_error_px={reprojection_error_px:6.3f}   [Enter to capture]",
                    end="", flush=True,
                )
                reading = {
                    "quality_row": {
                        "station_id": args.station_id,
                        "min_marker_pixels": round(min_marker_pixels, 3),
                        "reprojection_error_px": round(reprojection_error_px, 3),
                        "depth_m": round(depth_m, 4),
                    },
                    "position_m": [round(value, 4) for value in position_m],
                }

            # Non-blocking-ish: only check stdin between frames, don't stall the loop.
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                raw = sys.stdin.readline().strip().lower()
                if raw == "quit":
                    break
                if reading is None:
                    print("\n  no stable reading to capture yet -- keep the marker in view\n")
                    continue
                import json
                print(f"\n\n  marker_quality_trials.\"{args.marker_id}\" row:\n")
                print("  " + json.dumps(reading["quality_row"], sort_keys=True) + ",\n")
                print(f"  measured position_m (x, y, z) for reference_marker.expected_position_by_station_m,")
                print(f"  reference_marker_pose_trials_by_station_m, or a slot's verify_marker_xyz_m:\n")
                print("  " + json.dumps(reading["position_m"]) + "\n")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        camera.stop()
        print("Camera stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
