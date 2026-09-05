"""Dependency-free bounded polling for Stretch command completion."""
from __future__ import annotations

import time


def wait_command_cancellable(
    robot, *, timeout_s: float, cancel_event=None, poll_s: float = 0.10,
) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    interval = max(0.02, min(float(poll_s), 0.25))
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            return False
        remaining = deadline - time.monotonic()
        if bool(robot.wait_command(timeout=min(interval, remaining))):
            return True
    return False
