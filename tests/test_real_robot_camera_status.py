"""Exercise the actual D405 status producer with SDK-shaped intrinsics."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.real_robot.bridge import BridgeServer, ExecutionLedger
from src.real_robot.config import load_lab_config
from src.real_robot.hardware import HttpStretchExecutor
from src.real_robot.stretch_hardware import BridgeDryRunController


ROOT = Path(__file__).resolve().parents[1]


class DistortionModel:
    """A non-JSON SDK enum, with the same public string form as RealSense."""
    def __str__(self):
        return "distortion.brown_conrady"


class CameraStatusTests(unittest.TestCase):
    def setUp(self):
        # Load the real service and helper under an isolated package name. Only
        # external camera/GUI bindings are stubbed; no physical device is opened.
        namespace = "_hrc_camera_status_test_runtime"
        package = ModuleType(namespace)
        package.__path__ = [str(ROOT / "src/real_robot/stretch_runtime")]
        rs = ModuleType("pyrealsense2")
        rs.video_stream_profile = lambda profile: profile
        modules = patch.dict(sys.modules, {
            namespace: package, "cv2": ModuleType("cv2"), "pyrealsense2": rs,
        })
        modules.start()
        self.addCleanup(modules.stop)
        runtime = importlib.import_module(namespace + ".camera_service")
        intrinsics = SimpleNamespace(
            fx=610.5, fy=611.25, ppx=320.0, ppy=240.0,
            model=DistortionModel(), coeffs=[0.1, -0.02, 0.0, 0.0, 0.003],
        )
        frame = SimpleNamespace(profile=SimpleNamespace(get_intrinsics=lambda: intrinsics))
        self.camera = runtime.D405CameraService(exposure="medium", capture_fps=15)
        self.camera.color_camera_info = runtime.dh.get_camera_info(frame)
        self.camera.depth_camera_info = runtime.dh.get_camera_info(frame)
        self.camera.running = True
        self.camera.capture_thread = SimpleNamespace(is_alive=lambda: True)
        self.camera.latest_color = np.zeros((8, 8, 3), dtype=np.uint8)
        self.camera.latest_depth = np.ones((8, 8), dtype=np.uint16)
        self.camera.latest_monotonic = time.monotonic()
        self.camera.latest_timestamp = time.time()
        self.camera.latest_frame_id = 7
        self.camera.device_identity = {"name": "Intel RealSense D405", "serial_number": "test-d405"}

    def test_status_serializes_real_intrinsics_arrays_and_sdk_enum(self):
        status = json.loads(json.dumps(self.camera.get_status(), allow_nan=False))
        self.assertTrue(status["ready"])
        for key in ("color_camera_info", "depth_camera_info"):
            self.assertEqual(status[key]["camera_matrix"], [
                [610.5, 0.0, 320.0], [0.0, 611.25, 240.0], [0.0, 0.0, 1.0],
            ])
            self.assertEqual(status[key]["distortion_coefficients"], [0.1, -0.02, 0.0, 0.0, 0.003])
            self.assertEqual(status[key]["distortion_model"], "distortion.brown_conrady")

    def test_status_snapshot_does_not_change_numeric_perception_data(self):
        status = self.camera.get_status()
        status["color_camera_info"]["camera_matrix"][0][0] = -1.0
        bundle = self.camera.get_latest_bundle()
        for key in ("color_camera_info", "depth_camera_info"):
            self.assertIsInstance(bundle[key]["camera_matrix"], np.ndarray)
            self.assertIsInstance(bundle[key]["distortion_coefficients"], np.ndarray)
            self.assertEqual(bundle[key]["camera_matrix"][0, 0], 610.5)
            self.assertIsInstance(bundle[key]["distortion_model"], DistortionModel)
        self.assertEqual(self.camera.get_color_camera_info()["camera_matrix"][0, 0], 610.5)

    def test_status_before_intrinsics_arrive_preserves_null(self):
        self.camera.color_camera_info = None
        self.camera.depth_camera_info = None
        status = json.loads(json.dumps(self.camera.get_status()))
        self.assertFalse(status["ready"])
        self.assertIsNone(status["color_camera_info"])
        self.assertIsNone(status["depth_camera_info"])

    def test_ui_preflight_round_trip_accepts_actual_camera_status(self):
        config = load_lab_config(ROOT / "robot_configs/stretch3_lab.json")
        controller = BridgeDryRunController(config)
        controller.camera_service = self.camera
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = ExecutionLedger(Path(temp_dir) / "ledger.jsonl")
            server = BridgeServer(("127.0.0.1", 0), controller, config, ledger)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            executor = HttpStretchExecutor(f"http://127.0.0.1:{server.server_port}")
            try:
                self.camera.latest_monotonic = time.monotonic()
                status = executor.preflight(config.digest, require_motion=False)
                self.assertTrue(status["ready"])
                self.assertFalse(status["motion_enabled"])
                self.assertEqual(status["camera"]["color_camera_info"]["camera_matrix"][0][0], 610.5)
                json.dumps(status, allow_nan=False)
                with patch.object(controller, "camera_jpeg", return_value=b"test-jpeg"):
                    self.assertEqual(executor.camera_jpeg(), b"test-jpeg")
            finally:
                executor.close()
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
