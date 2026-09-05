# routes.py - Route navigation functions

import math
import time
from ..cancellable_wait import wait_command_cancellable
from .config import (
    BASE_ROTATION_SPEED_RADPS, BASE_ROTATION_ACCEL_RADPS2
)

def print_base_status(robot, label="status"):
    x = robot.base.status["x"]
    y = robot.base.status["y"]
    th = math.degrees(robot.base.status["theta"])
    print(f"[{label}] x={x:.3f} m | y={y:.3f} m | theta={th:.1f} deg")

def _remaining_timeout(deadline=None, timeout=60.0):
    if deadline is None:
        return float(timeout)
    return max(0.0, min(float(timeout), float(deadline) - time.monotonic()))


def push_and_wait(
    robot, label="", timeout=60.0, deadline=None, cancel_event=None,
    cancel_poll_s=0.10,
):
    """Push one command and wait in bounded slices so cancellation is observed.

    Stretch ``wait_command`` is allowed to time out while the command remains
    active, so repeated short waits are used instead of one long uninterruptible
    wait.  The action thread remains the sole SDK owner throughout.
    """
    if cancel_event is not None and cancel_event.is_set():
        return False
    wait_timeout = _remaining_timeout(deadline, timeout)
    if wait_timeout <= 0.0:
        return False
    robot.push_command()
    ok = wait_command_cancellable(
        robot, timeout_s=wait_timeout,
        cancel_event=cancel_event, poll_s=cancel_poll_s,
    )
    print_base_status(robot, label)
    if not ok:
        print(f"WARNING: command wait timed out during: {label}")
    return ok

def rotate_deg(robot, angle_deg, *, timeout=60.0, deadline=None, cancel_event=None, velocity_scale=1.0):
    print(f"rotate {angle_deg:.2f} deg")
    robot.base.rotate_by(
        math.radians(angle_deg),
        v_r=BASE_ROTATION_SPEED_RADPS * velocity_scale,
        a_r=BASE_ROTATION_ACCEL_RADPS2 * velocity_scale,
    )
    return push_and_wait(
        robot, f"rotate {angle_deg:.2f} deg", timeout=timeout,
        deadline=deadline, cancel_event=cancel_event,
    )
