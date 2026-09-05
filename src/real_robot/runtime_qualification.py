"""Exact runtime and hardware identity checks for calibrated Stretch runs."""
from __future__ import annotations

import importlib
from importlib import metadata
import platform
from typing import Any, Mapping


RUNTIME_MODULES = ("stretch_body", "pyrealsense2", "cv2", "numpy", "scipy")
_DISTRIBUTION_CANDIDATES = {
    "stretch_body": ("hello-robot-stretch-body", "stretch-body", "stretch_body"),
    "pyrealsense2": ("pyrealsense2",),
    "cv2": (
        "opencv-contrib-python", "opencv-contrib-python-headless",
        "opencv-python", "opencv-python-headless",
    ),
    "numpy": ("numpy",),
    "scipy": ("scipy",),
}


def module_version(module: Any, module_name: str) -> str:
    for attribute in ("__version__", "version"):
        value = getattr(module, attribute, None)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    for distribution in _DISTRIBUTION_CANDIDATES.get(module_name, (module_name,)):
        try:
            value = metadata.version(distribution).strip()
        except metadata.PackageNotFoundError:
            continue
        if value:
            return value
    return "unavailable"


def inspect_runtime() -> Mapping[str, Any]:
    versions: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name in RUNTIME_MODULES:
        try:
            versions[name] = module_version(importlib.import_module(name), name)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return {
        "python_version": platform.python_version(),
        "modules": versions,
        "import_errors": errors,
    }


def qualify_runtime(
    lock: Mapping[str, Any], *, robot_id: str | None = None,
    camera_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    observed = dict(inspect_runtime())
    mismatches: list[str] = []
    if observed["python_version"] != str(lock.get("python_version", "")):
        mismatches.append(
            f"python_version expected {lock.get('python_version')!r}, observed {observed['python_version']!r}"
        )
    expected_modules = dict(lock.get("modules", {}))
    for name in RUNTIME_MODULES:
        actual = observed["modules"].get(name, "unavailable")
        expected = str(expected_modules.get(name, ""))
        if actual != expected:
            mismatches.append(f"module {name} expected {expected!r}, observed {actual!r}")
    for name, error in observed["import_errors"].items():
        mismatches.append(f"module {name} failed import: {error}")

    if robot_id is not None and str(robot_id) != str(lock.get("robot_id", "")):
        mismatches.append(
            f"robot_id expected {lock.get('robot_id')!r}, observed {robot_id!r}"
        )
    if camera_identity is not None:
        serial = str(camera_identity.get("serial_number", ""))
        firmware = str(camera_identity.get("firmware_version", ""))
        if serial != str(lock.get("d405_serial", "")):
            mismatches.append(
                f"D405 serial expected {lock.get('d405_serial')!r}, observed {serial!r}"
            )
        if firmware != str(lock.get("d405_firmware", "")):
            mismatches.append(
                f"D405 firmware expected {lock.get('d405_firmware')!r}, observed {firmware!r}"
            )
    return {
        **observed,
        "robot_id": robot_id,
        "camera_identity": dict(camera_identity or {}),
        "qualified": not mismatches,
        "mismatches": mismatches,
    }


def require_qualified_runtime(
    lock: Mapping[str, Any], *, robot_id: str,
    camera_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    result = qualify_runtime(
        lock, robot_id=robot_id, camera_identity=camera_identity,
    )
    if not result["qualified"]:
        raise RuntimeError("runtime qualification failed: " + "; ".join(result["mismatches"]))
    return result
