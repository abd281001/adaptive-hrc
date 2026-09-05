"""Deployment preflight for the Stretch demo; never commands robot motion."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .config import load_lab_config
from .hardware import HttpStretchExecutor


ROBOT_MODULES = ("stretch_body.robot", "pyrealsense2", "cv2", "numpy", "scipy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only real-robot deployment checks")
    parser.add_argument("--config", default="robot_configs/stretch3_lab.json")
    parser.add_argument("--hardware-url", default="")
    parser.add_argument("--expect-motion", action="store_true")
    parser.add_argument("--check-robot-runtime", action="store_true")
    parser.add_argument("--json-output", default="")
    return parser


def run_checks(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        config = load_lab_config(args.config)
        record("configuration", True, {
            "digest": config.digest, "schema_version": config.schema_version,
            "stations": len(config.stations), "placement_slots": len(config.placement_slots),
            "objects": len(config.objects), "recipes": len(config.recipes),
            "reference_marker_id": config.reference_marker.marker_id,
            "calibrated": config.motion.calibrated,
            "calibration_id": config.motion.calibration_id,
            "calibration_record": config.motion.calibration_record,
            "calibration_record_sha256": config.motion.calibration_record_sha256,
        })
    except Exception as exc:
        record("configuration", False, str(exc))
        return {"ok": False, "checks": checks}, False

    if args.expect_motion:
        record("calibration_gate", bool(config.motion.calibrated and config.motion.calibration_id),
               "calibration is identified" if config.motion.calibrated else "motion.calibrated is false")

    if args.check_robot_runtime:
        versions = {}
        loaded = {}
        for module_name in ROBOT_MODULES:
            try:
                module = importlib.import_module(module_name)
                loaded[module_name] = module
                versions[module_name] = getattr(module, "__version__", "available")
                record(f"import:{module_name}", True, versions[module_name])
            except Exception as exc:
                record(f"import:{module_name}", False, str(exc))
        cv2 = loaded.get("cv2")
        aruco_ok = bool(cv2 is not None and hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_6X6_250"))
        record("opencv_aruco_dictionary", aruco_ok, "DICT_6X6_250 available" if aruco_ok else "OpenCV ArUco DICT_6X6_250 unavailable")
        rs = loaded.get("pyrealsense2")
        if rs is not None:
            try:
                devices = [device.get_info(rs.camera_info.name) for device in rs.context().devices]
                record("d405_connected", any(name.endswith("D405") for name in devices), devices)
            except Exception as exc:
                record("d405_connected", False, str(exc))

    if args.hardware_url:
        executor = HttpStretchExecutor(args.hardware_url, timeout_s=10.0)
        try:
            status = executor.preflight(config.digest, require_motion=args.expect_motion)
            record("bridge", True, status)
        except Exception as exc:
            record("bridge", False, str(exc))
        finally:
            executor.close()
    elif args.expect_motion:
        record("bridge", False, "--hardware-url is required with --expect-motion")

    report = {
        "ok": all(row["ok"] for row in checks),
        "python": sys.version, "platform": platform.platform(),
        "config_path": str(Path(args.config).resolve()), "checks": checks,
    }
    return report, bool(report["ok"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, ok = run_checks(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        Path(args.json_output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
