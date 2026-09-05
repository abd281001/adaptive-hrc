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
from .runtime_qualification import (
    RUNTIME_MODULES, inspect_runtime, module_version, qualify_runtime,
)


ROBOT_MODULES = ("stretch_body.robot", *RUNTIME_MODULES)


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
            "runtime_lock": config.motion.runtime_lock,
            "runtime_lock_sha256": config.motion.runtime_lock_sha256,
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
                versions[module_name] = (
                    module_version(module, module_name)
                    if module_name in RUNTIME_MODULES else "available"
                )
                record(f"import:{module_name}", True, versions[module_name])
            except Exception as exc:
                record(f"import:{module_name}", False, str(exc))
        cv2 = loaded.get("cv2")
        aruco_ok = bool(cv2 is not None and hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_6X6_250"))
        record("opencv_aruco_dictionary", aruco_ok, "DICT_6X6_250 available" if aruco_ok else "OpenCV ArUco DICT_6X6_250 unavailable")
        rs = loaded.get("pyrealsense2")
        d405_devices = []
        if rs is not None:
            try:
                for device in rs.context().devices:
                    identity = {
                        "name": device.get_info(rs.camera_info.name),
                        "serial_number": device.get_info(rs.camera_info.serial_number),
                        "firmware_version": device.get_info(rs.camera_info.firmware_version),
                    }
                    if identity["name"].endswith("D405"):
                        d405_devices.append(identity)
                record("d405_connected", bool(d405_devices), d405_devices)
            except Exception as exc:
                record("d405_connected", False, str(exc))
        runtime = inspect_runtime()
        record("runtime_imports", not runtime["import_errors"], runtime)
        if config.motion.calibrated:
            locked_serial = str(config.runtime_lock_data.get("d405_serial", ""))
            locked_camera = next(
                (
                    item for item in d405_devices
                    if item.get("serial_number") == locked_serial
                ),
                d405_devices[0] if len(d405_devices) == 1 else None,
            )
            qualification = qualify_runtime(
                config.runtime_lock_data, camera_identity=locked_camera,
            )
            record(
                "runtime_lock_versions_and_camera", qualification["qualified"],
                qualification,
            )

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
