# Runtime provenance

The initial implementations of the camera, marker, fingertip, normalized
velocity, grasp, base-motion, and placement routines were copied from the
lab's prior Stretch 3 demo in `Gabriel/Gabriel/salad-robot` on 2026-09-03.
They are maintained here as an independent runtime for adaptive HRC.

Local changes include package-relative imports, an ArUco-only target path,
color-camera intrinsics for color-image pose estimation, action cancellation
and timeouts, portable resource paths, and removal of MediaPipe/AprilTag code
that the proposed experiment does not use.

The source directory was used as a read-only reference and is not a runtime
dependency.

