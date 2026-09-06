#!/usr/bin/env python3
"""Identify box AprilTags and gripper ArUco tags from the bridge's JPEG feed.

Robot installation (Python 3.10 with the existing NumPy and OpenCV):
  /usr/bin/python3 -E -m pip install --user --no-deps --only-binary=:all: \
      pupil-apriltags==1.0.4.post11

Run from the project root, with the dry-run camera bridge already running:
  /usr/bin/python3 -E tools/mixed_marker_probe.py --seconds 300 \
      --report real_robot_runs/five_boxes_marker_probe.json

This is an ID diagnostic. It does not estimate metric poses, control the robot,
open a RealSense pipeline, or change the production detector/configuration.
Image counts are observations, not proof of fresh camera frames or calibration.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
from pupil_apriltags import Detector


APRIL_FAMILY = "tagStandard41h12"
ARUCO_FAMILY = "DICT_6X6_250"


def find_markers(frame, april_detector, aruco_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = []
    for tag in april_detector.detect(gray, estimate_tag_pose=False):
        rows.append({
            "family": APRIL_FAMILY, "id": int(tag.tag_id),
            "hamming": int(tag.hamming),
            "corners_px": np.asarray(tag.corners).tolist(),
        })
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is not None:
        for marker_id, quad in zip(ids.flatten(), corners):
            rows.append({
                "family": ARUCO_FAMILY, "id": int(marker_id),
                "hamming": None,
                "corners_px": np.asarray(quad).reshape(4, 2).tolist(),
            })
    counts = Counter((row["family"], row["id"]) for row in rows)
    for row in rows:
        row["duplicate_in_image"] = counts[row["family"], row["id"]] > 1
    return sorted(rows, key=lambda row: (row["family"], row["id"]))


def draw_markers(frame, rows):
    canvas = frame.copy()
    for row in rows:
        corners = np.rint(row["corners_px"]).astype(np.int32)
        if row["family"] == APRIL_FAMILY:
            label = f"April41h12 ID {row['id']} h={row['hamming']}"
            color = (0, 220, 0) if row["hamming"] == 0 else (0, 170, 255)
        else:
            label = f"ArUco6x6 ID {row['id']}"
            color = (255, 170, 0)
        if row["duplicate_in_image"]:
            label += " DUPLICATE"
            color = (0, 0, 255)
        cv2.polylines(canvas, [corners], True, color, 2)
        x = max(0, min(int(corners[:, 0].min()), canvas.shape[1] - 80))
        y = max(18, int(corners[:, 1].min()) - 6)
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 1, cv2.LINE_AA)
    return canvas


def describe(rows):
    april = [f"{r['id']}(h={r['hamming']})" for r in rows
             if r["family"] == APRIL_FAMILY]
    aruco = [str(r["id"]) for r in rows if r["family"] == ARUCO_FAMILY]
    warning = " | DUPLICATE IDs IN IMAGE" if any(
        r["duplicate_in_image"] for r in rows) else ""
    return f"April41h12: [{', '.join(april)}] | ArUco6x6: [{', '.join(aruco)}]{warning}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:9100/v1/camera.jpg")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--image", type=Path, help="Analyze one saved image instead of HTTP")
    parser.add_argument("--overlay", type=Path, help="Save an annotated image (requires --image)")
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error("--seconds must be between 0 and 3600")
    if args.overlay and not args.image:
        parser.error("--overlay requires --image")

    april = Detector(families=APRIL_FAMILY, nthreads=2, quad_decimate=1.0)
    aruco = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
        cv2.aruco.DetectorParameters())
    report = {
        "diagnostic_only": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.image) if args.image else args.url,
        "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                     "numpy": np.__version__, "pupil_apriltags": version("pupil-apriltags")},
        "image_observations": 0, "markers": [], "error": None,
    }
    seen = {}
    started = time.monotonic()
    last_print = float("-inf")
    failures = 0
    window_open = False
    exit_code = 0
    print("ID-only diagnostic: box April41h12 + gripper ArUco6x6.", flush=True)
    print("The bridge retains camera ownership. Click the image and press q/Esc to finish.", flush=True)
    try:
        while time.monotonic() - started < args.seconds:
            if args.image:
                frame = cv2.imread(str(args.image))
                if frame is None:
                    raise RuntimeError(f"Cannot read image: {args.image}")
            else:
                try:
                    with urlopen(args.url, timeout=3) as response:
                        raw = response.read()
                    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Bridge response was not a decodable JPEG")
                    failures = 0
                except (URLError, TimeoutError, OSError, ValueError) as exc:
                    failures += 1
                    print(f"Camera read failed ({failures}/5): {exc}", flush=True)
                    if failures >= 5:
                        raise RuntimeError("Check that the camera-preview bridge is running and serving JPEGs") from exc
                    time.sleep(0.25)
                    continue
            rows = find_markers(frame, april, aruco)
            report["image_observations"] += 1
            for row in rows:
                key = (row["family"], row["id"])
                item = seen.setdefault(key, {
                    "family": row["family"], "id": row["id"],
                    "detections": 0, "hamming_counts": {},
                    "duplicate_seen": False,
                })
                item["detections"] += 1
                item["duplicate_seen"] |= row["duplicate_in_image"]
                if row["hamming"] is not None:
                    hamming = str(row["hamming"])
                    item["hamming_counts"][hamming] = item["hamming_counts"].get(hamming, 0) + 1
            if time.monotonic() - last_print >= 1 or args.image:
                print(describe(rows), flush=True)
                last_print = time.monotonic()
            if not args.no_window or args.overlay:
                canvas = draw_markers(frame, rows)
                if args.overlay:
                    if not cv2.imwrite(str(args.overlay), canvas):
                        raise RuntimeError(f"Could not write overlay: {args.overlay}")
                if not args.no_window:
                    cv2.imshow("Five boxes - AprilTag and ArUco IDs", canvas)
                    window_open = True
                    if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                        break
            if args.image:
                break
            time.sleep(0.10)
    except KeyboardInterrupt:
        print("Stopped by operator.", flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if window_open:
            cv2.destroyAllWindows()
        report["markers"] = [seen[key] for key in sorted(seen)]
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        print("Seen across the run:", [(r["family"], r["id"]) for r in report["markers"]])
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Report saved: {args.report}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
