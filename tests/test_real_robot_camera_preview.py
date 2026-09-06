"""Camera-only dry-run regression checks; no physical robot or camera required."""
from __future__ import annotations

from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from src.real_robot import bridge
from src.real_robot.config import load_lab_config
from src.real_robot.hardware import HttpStretchExecutor
from src.real_robot.stretch_hardware import BridgeDryRunController


CONFIG_PATH = Path(__file__).resolve().parents[1] / "robot_configs/stretch3_lab.json"


class CameraPreviewTests(unittest.TestCase):
    def setUp(self):
        self.config = load_lab_config(CONFIG_PATH)
        self.started = 0
        self.stopped = 0
        self.start_error = None
        self.bundle = {"color": object(), "age_s": 0.01}
        self.imports = []
        self.camera = SimpleNamespace(
            start=self.start_camera, stop=self.stop_camera,
            get_latest_bundle=lambda: self.bundle,
            get_status=lambda: {"ready": True, "device_identity": {"serial_number": "test-camera"}},
        )
        self.encoder = SimpleNamespace(
            IMWRITE_JPEG_QUALITY=1,
            imencode=lambda *_: (True, SimpleNamespace(tobytes=lambda: b"test-jpeg")),
        )
        self.import_patch = patch(
            "src.real_robot.stretch_hardware.importlib.import_module",
            side_effect=self.fake_import,
        )
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)

    def fake_import(self, name, *args, **kwargs):
        self.imports.append(name)
        if name == "src.real_robot.stretch_runtime.camera_service":
            return SimpleNamespace(D405CameraService=lambda **_: self.camera)
        if name == "cv2":
            return self.encoder
        raise AssertionError(f"Preview unexpectedly imported {name}")

    def start_camera(self):
        self.started += 1
        if self.start_error is not None:
            raise self.start_error

    def stop_camera(self):
        self.stopped += 1

    def preview(self):
        controller = BridgeDryRunController(self.config, camera_preview=True)
        self.addCleanup(controller.close)
        return controller

    def test_default_dry_run_requires_no_camera_dependencies(self):
        controller = BridgeDryRunController(self.config)
        self.assertIsNone(controller.camera_jpeg())
        controller.close()
        self.assertEqual(self.imports, [])
        self.assertEqual(self.started, 0)

    def test_preview_executes_only_simulated_actions_and_keeps_stop_latched(self):
        controller = self.preview()
        result = controller.execute(
            self.config.actions["STAGE_BOWL"], execution_id="preview-test",
            placement_slot_id="slot_1",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["postcondition"], {"dry_run": True})
        self.assertFalse(controller.status()["motion_enabled"])
        self.assertFalse(controller.status()["calibration_mode"])
        self.assertEqual(controller.camera_jpeg(), b"test-jpeg")
        controller.emergency_stop()
        self.assertFalse(controller.status()["ready"])
        self.assertFalse(controller.execute(
            self.config.actions["STAGE_BOWL"], execution_id="stopped-test",
            placement_slot_id="slot_1",
        )["success"])
        self.assertFalse(any(name.startswith("stretch_body") for name in self.imports))

    def test_missing_and_stale_frames_are_not_served(self):
        controller = self.preview()
        for bundle in (
            {"color": None, "age_s": 0.01},
            {"color": object(), "age_s": None},
            {"color": object(), "age_s": self.config.perception.max_frame_age_s + 0.01},
        ):
            self.bundle = bundle
            self.assertIsNone(controller.camera_jpeg())

    def test_encoder_failure_reports_no_frame(self):
        controller = self.preview()
        self.encoder.imencode = lambda *_: (False, None)
        self.assertIsNone(controller.camera_jpeg())

    def test_failed_camera_start_releases_camera(self):
        self.start_error = RuntimeError("camera unavailable")
        with self.assertRaisesRegex(RuntimeError, "camera unavailable"):
            BridgeDryRunController(self.config, camera_preview=True)
        self.assertEqual(self.stopped, 1)

    def test_close_releases_camera_once(self):
        controller = self.preview()
        controller.close()
        controller.close()
        self.assertEqual(self.stopped, 1)
        self.assertIsNone(controller.camera_jpeg())

    def test_preview_and_live_mode_cannot_be_combined(self):
        with self.assertRaisesRegex(SystemExit, "dry runs only"):
            bridge.main(["--camera-preview", "--enable-motion"])
        self.assertEqual(self.started, 0)

    def test_failed_http_bind_releases_camera(self):
        with tempfile.TemporaryDirectory() as state_dir:
            with patch.object(bridge, "BridgeServer", side_effect=OSError("port busy")):
                with self.assertRaisesRegex(OSError, "port busy"):
                    bridge.main([
                        "--config", str(CONFIG_PATH), "--state-dir", state_dir,
                        "--camera-preview",
                    ])
        self.assertEqual(self.started, 1)
        self.assertEqual(self.stopped, 1)

    def test_bridge_http_and_ui_executor_deliver_preview_then_reject_stale_frame(self):
        controller = self.preview()
        with tempfile.TemporaryDirectory() as state_dir:
            ledger = bridge.ExecutionLedger(Path(state_dir) / "ledger.jsonl")
            server = bridge.BridgeServer(("127.0.0.1", 0), controller, self.config, ledger)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            executor = HttpStretchExecutor(f"http://127.0.0.1:{server.server_port}")
            try:
                self.assertEqual(executor.camera_jpeg(), b"test-jpeg")
                self.bundle["age_s"] = 10.0
                self.assertIsNone(executor.camera_jpeg())
                with self.assertRaises(HTTPError) as error:
                    urlopen(f"http://127.0.0.1:{server.server_port}/v1/camera.jpg", timeout=2)
                self.assertEqual(error.exception.code, 503)
            finally:
                executor.close()
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
