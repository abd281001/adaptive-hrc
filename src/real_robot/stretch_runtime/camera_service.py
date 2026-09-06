# Author: Gabriel Armas Aranibar
# camera_service.py
#
# Shared always-on D405 camera service.
#
# Purpose:
# - Own the D405 pipeline exactly once
# - Continuously capture color/depth frames in a background thread
# - Keep the latest frames in memory for low-latency robot use
# - Provide getters for robot code and HTTP stream endpoints
#
# Important:
# - Only ONE D405CameraService instance should own the D405 at a time
# - Do not also call dh.start_d405(...) elsewhere for the same camera

import threading
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np

from . import d405_helpers as dh


class D405CameraService:
    def __init__(
        self,
        exposure="low",
        capture_fps=20,
        stream_fps=15,
        jpeg_quality=80,
        depth_alpha=0.03,
        startup_timeout_s=12.0,
        frame_timeout_ms=1000,
        stale_after_s=0.30,
    ):
        self.exposure = exposure
        self.capture_fps = capture_fps
        self.stream_fps = stream_fps
        self.jpeg_quality = jpeg_quality
        self.depth_alpha = depth_alpha
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.frame_timeout_ms = max(100, int(frame_timeout_ms))
        self.stale_after_s = max(0.05, float(stale_after_s))

        self.pipeline = None
        self.profile = None

        self.running = False
        self._capture_requested = False
        self.capture_thread = None
        self.lock = threading.Lock()

        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_vis: Optional[np.ndarray] = None

        # Annotation overlays for streaming
        self.color_annotations: Optional[Dict[str, Any]] = None
        self.depth_annotations: Optional[Dict[str, Any]] = None

        self.latest_timestamp: Optional[float] = None
        self.latest_monotonic: Optional[float] = None
        self.latest_frame_id: int = 0

        self.depth_camera_info: Optional[Dict[str, Any]] = None
        self.color_camera_info: Optional[Dict[str, Any]] = None

        self.last_error: Optional[str] = None
        self.device_identity: Dict[str, str] = {}

    def start(self):
        if self.running:
            return

        self.pipeline, self.profile = dh.start_d405(self.exposure)
        try:
            device = self.profile.get_device()
            fields = {
                "name": dh.rs.camera_info.name,
                "serial_number": dh.rs.camera_info.serial_number,
                "firmware_version": dh.rs.camera_info.firmware_version,
                "product_line": dh.rs.camera_info.product_line,
            }
            self.device_identity = {
                label: device.get_info(field)
                for label, field in fields.items()
                if device.supports(field)
            }
        except Exception as exc:
            self.device_identity = {"identity_error": str(exc)}
        self.running = True
        self._capture_requested = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            with self.lock:
                ready = (
                    self.latest_color is not None and self.latest_depth is not None
                    and self.color_camera_info is not None and self.depth_camera_info is not None
                )
                error = self.last_error
            if ready:
                break
            if self.capture_thread is None or not self.capture_thread.is_alive():
                self.stop()
                raise RuntimeError(f"D405 capture thread exited during startup: {error or 'unknown error'}")
            time.sleep(0.02)
        else:
            self.stop()
            raise RuntimeError(f"D405 did not provide color, depth, and intrinsics within {self.startup_timeout_s:.1f}s")
        print("D405 camera service started.")

    def stop(self):
        self._capture_requested = False
        self.running = False
        # Stop the pipeline first so a blocked wait_for_frames call is released.
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Warning while stopping D405 pipeline: {e}")

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                raise RuntimeError("D405 capture thread did not stop")
            self.capture_thread = None

        self.pipeline = None
        self.profile = None
        print("D405 camera service stopped.")

    def restart(self):
        self.stop()
        time.sleep(0.2)
        self.start()

    def _capture_loop(self):
        frame_interval = 1.0 / max(1, self.capture_fps)
        last_loop_time = 0.0

        while self._capture_requested:
            try:
                if self.pipeline is None:
                    raise RuntimeError("D405 pipeline is not initialized.")

                frames = self.pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    time.sleep(0.005)
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                if self.depth_camera_info is None:
                    self.depth_camera_info = dh.get_camera_info(depth_frame)

                if self.color_camera_info is None:
                    self.color_camera_info = dh.get_camera_info(color_frame)

                depth_vis = cv2.convertScaleAbs(depth_image, alpha=self.depth_alpha)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                now = time.time()
                monotonic_now = time.monotonic()

                with self.lock:
                    self.latest_color = color_image.copy()
                    self.latest_depth = depth_image.copy()
                    self.latest_depth_vis = depth_vis.copy()
                    self.latest_timestamp = now
                    self.latest_monotonic = monotonic_now
                    self.latest_frame_id += 1
                    self.last_error = None

                elapsed = monotonic_now - last_loop_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_loop_time = time.monotonic()

            except Exception as e:
                err = str(e)
                print(f"D405 capture loop error: {err}")
                with self.lock:
                    if self._capture_requested:
                        self.last_error = err
                time.sleep(0.2)
        self.running = False

    def get_latest_color(self, apply_annotations: bool = True) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_color is None:
                return None
            frame = self.latest_color.copy()
            if apply_annotations:
                return self._apply_annotations(frame, self.color_annotations)
            return frame

    def get_latest_depth(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_depth is None:
                return None
            return self.latest_depth.copy()

    def get_latest_depth_vis(self, apply_annotations: bool = True) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_depth_vis is None:
                return None
            frame = self.latest_depth_vis.copy()
            if apply_annotations:
                return self._apply_annotations(frame, self.depth_annotations)
            return frame

    def get_latest_bundle(self):
        with self.lock:
            return {
                "color": None if self.latest_color is None else self.latest_color.copy(),
                "depth": None if self.latest_depth is None else self.latest_depth.copy(),
                "depth_vis": None if self.latest_depth_vis is None else self.latest_depth_vis.copy(),
                "timestamp": self.latest_timestamp,
                "age_s": None if self.latest_monotonic is None else max(0.0, time.monotonic() - self.latest_monotonic),
                "frame_id": self.latest_frame_id,
                "depth_camera_info": self.depth_camera_info,
                "color_camera_info": self.color_camera_info,
            }

    @staticmethod
    def _camera_info_for_status(camera_info):
        """Copy intrinsics into JSON types without changing perception arrays."""
        if camera_info is None:
            return None
        return {
            "camera_matrix": np.asarray(camera_info["camera_matrix"]).tolist(),
            "distortion_coefficients": np.asarray(camera_info["distortion_coefficients"]).tolist(),
            "distortion_model": str(camera_info["distortion_model"]),
        }

    def get_status(self):
        with self.lock:
            age_s = None if self.latest_monotonic is None else max(0.0, time.monotonic() - self.latest_monotonic)
            thread_alive = self.capture_thread is not None and self.capture_thread.is_alive()
            return {
                "running": self.running,
                "thread_alive": thread_alive,
                "ready": bool(
                    self.running and thread_alive and age_s is not None
                    and age_s <= self.stale_after_s
                    and self.latest_color is not None and self.latest_depth is not None
                    and self.color_camera_info is not None and self.depth_camera_info is not None
                ),
                "stale": age_s is None or age_s > self.stale_after_s,
                "age_s": age_s,
                "has_color": self.latest_color is not None,
                "has_depth": self.latest_depth is not None,
                "frame_id": self.latest_frame_id,
                "timestamp": self.latest_timestamp,
                "capture_fps": self.capture_fps,
                "stream_fps": self.stream_fps,
                "jpeg_quality": self.jpeg_quality,
                "exposure": self.exposure,
                "last_error": self.last_error,
                "device_identity": dict(self.device_identity),
                "depth_camera_info": self._camera_info_for_status(self.depth_camera_info),
                "color_camera_info": self._camera_info_for_status(self.color_camera_info),
            }

    def get_color_camera_info(self):
        with self.lock:
            return self.color_camera_info

    def set_color_annotations(self, annotations: Optional[Dict[str, Any]]):
        """Set annotations to overlay on color frames for streaming"""
        with self.lock:
            self.color_annotations = annotations

    def set_depth_annotations(self, annotations: Optional[Dict[str, Any]]):
        """Set annotations to overlay on depth frames for streaming"""
        with self.lock:
            self.depth_annotations = annotations

    def get_color_annotations(self) -> Optional[Dict[str, Any]]:
        """Get current color annotations"""
        with self.lock:
            return self.color_annotations.copy() if self.color_annotations else None

    def get_depth_annotations(self) -> Optional[Dict[str, Any]]:
        """Get current depth annotations"""
        with self.lock:
            return self.depth_annotations.copy() if self.depth_annotations else None

    def _apply_annotations(self, frame: np.ndarray, annotations: Optional[Dict[str, Any]]) -> np.ndarray:
        """Apply annotations to a frame copy"""
        if annotations is None:
            return frame

        annotated_frame = frame.copy()

        # Draw AprilTag annotations (fast)
        if 'apriltags' in annotations:
            for tag in annotations['apriltags']:
                if 'corners' in tag:
                    corners = tag['corners'].astype(int)
                    ptA, ptB, ptC, ptD = corners
                    cv2.line(annotated_frame, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
                    cv2.line(annotated_frame, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
                    cv2.line(annotated_frame, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
                    cv2.line(annotated_frame, tuple(ptD), tuple(ptA), (0, 255, 0), 2)

                if 'center' in tag:
                    center = tuple(tag['center'].astype(int))
                    cv2.circle(annotated_frame, center, 5, (0, 0, 255), -1)

                if 'id' in tag and 'corners' in tag:
                    ptA = tag['corners'][0].astype(int)
                    tag_name = tag.get('name')
                    label = f'tagId: {tag["id"]}'
                    if tag_name:
                        label += f', {tag_name}'

                    cv2.putText(
                        annotated_frame,
                        label,
                        (ptA[0], ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

        # Draw simplified MediaPipe hand annotations (much faster)
        if 'hands' in annotations and 'mp_hands' in annotations:
            mp_hands = annotations['mp_hands']
            for hand_result in annotations['hands']:
                if hasattr(hand_result, 'multi_hand_landmarks') and hand_result.multi_hand_landmarks:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        key_points = [0, 4, 8, 12, 16, 20]
                        for idx in key_points:
                            landmark = hand_landmarks.landmark[idx]
                            h, w, _ = annotated_frame.shape
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(annotated_frame, (cx, cy), 3, (255, 0, 255), -1)

        # Draw fingertip frames (simplified)
        if 'fingertips' in annotations and 'camera_info' in annotations:
            camera_info = annotations['camera_info']
            fingertips = annotations['fingertips']

            for finger_name, finger_data in fingertips.items():
                if finger_data and 'pos' in finger_data:
                    try:
                        pixel = dh.pixel_from_3d(finger_data['pos'], camera_info)
                        center = tuple(pixel.astype(int))
                        cv2.circle(annotated_frame, center, 4, (255, 255, 0), 2)
                    except:
                        pass

        # Draw origin markers (fast)
        if 'origins' in annotations and 'camera_info' in annotations:
            camera_info = annotations['camera_info']
            for origin in annotations['origins']:
                pos = origin['position']
                color = origin.get('color', (255, 255, 255))
                try:
                    pixel = dh.pixel_from_3d(pos, camera_info)
                    center = tuple(pixel.astype(int))
                    cv2.circle(annotated_frame, center, 8, color, 2)
                except:
                    pass

        # Draw text overlays (fast)
        if 'texts' in annotations:
            for text_item in annotations['texts']:
                pos = text_item['position']
                lines = text_item['lines']
                color = text_item.get('color', (255, 255, 255))
                y_offset = 0
                for line in lines:
                    cv2.putText(
                        annotated_frame,
                        line,
                        (pos[0] + 10, pos[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )
                    y_offset += 20

        return annotated_frame

    def _draw_origin(self, image, camera_info, position_3d, color=(255, 255, 255)):
        """Draw coordinate frame origin on image"""
        try:
            pixel = dh.pixel_from_3d(position_3d, camera_info)
            center = tuple(pixel.astype(int))
            cv2.circle(image, center, 8, color, 2)
            cv2.line(image, center, (center[0] + 20, center[1]), (0, 0, 255), 2)
            cv2.line(image, center, (center[0], center[1] - 20), (0, 255, 0), 2)
        except:
            pass

    def _draw_text(self, image, center, text_lines, color=(255, 255, 255)):
        """Draw multi-line text on image"""
        y_offset = 0
        for line in text_lines:
            cv2.putText(
                image,
                line,
                (center[0] + 10, center[1] + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
            y_offset += 20
