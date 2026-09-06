"""Motion commissioning contracts, using feedback-controlled fake hardware."""
import math
import json
import sys
from types import SimpleNamespace

import pytest

from src.real_robot.motion_console import MotionConsole, MotionFault, main, request_stop
from src.real_robot import motion_console


class Joint:
    def __init__(self, robot, name, position, limits):
        self.robot, self.name = robot, name
        self.status = {"pos": position, "vel": 0.0}
        self.soft_motion_limits = {"current": limits}
        self.motor = SimpleNamespace(status={"timestamp": 1.0})

    def move_to(self, target, **kwargs):
        self.robot.calls.append((self.name, target, kwargs))
        self.robot.pending = lambda: self.status.update(pos=target)


class Robot:
    def __init__(self):
        self.calls, self.pending = [], None
        self.homed, self.stale, self.blocked = True, False, False
        self.arm = Joint(self, "arm", 0.0, [0.0, 0.52])
        self.lift = Joint(self, "lift", 0.9, [0.0, 1.1])
        self.base = SimpleNamespace(
            status={"x": 0.0, "y": 0.0, "theta": 0.4, "theta_vel": 0.0},
            left_wheel=SimpleNamespace(status={"timestamp": 1.0}),
            right_wheel=SimpleNamespace(status={"timestamp": 1.0}),
            rotate_by=self.rotate_by,
        )
        self.gripper = SimpleNamespace(
            status={"pos_pct": 70.0, "vel": 0.0, "timestamp_pc": 1.0},
            poses={"open": 70.0, "close": -100}, move_to=self.gripper_move,
        )
        self.end_of_arm = SimpleNamespace(get_joint=lambda name: self.gripper)
        self.pimu = SimpleNamespace(
            status={"runstop_event": False, "timestamp": 1.0},
            runstop_event_trigger=lambda: self.calls.append(("runstop",)),
            push_command=lambda: self.calls.append(("pimu_push",)),
        )

    def is_homed(self):
        return self.homed

    def rotate_by(self, delta, **kwargs):
        self.calls.append(("base", delta, kwargs))
        target = (self.base.status["theta"] + delta + math.pi) % (2 * math.pi) - math.pi
        self.pending = lambda: self.base.status.update(theta=target)

    def gripper_move(self, target, **kwargs):
        self.calls.append(("gripper", target, kwargs))
        if not self.blocked:
            self.gripper.status["pos_pct"] = target

    def push_command(self):
        self.calls.append(("push",))
        if not self.blocked:
            self.pending()
        self.pending = None

    def tick(self):
        if not self.stale:
            for motor in (self.arm.motor, self.lift.motor, self.base.left_wheel, self.base.right_wheel, self.pimu):
                motor.status["timestamp"] += 0.05
            self.gripper.status["timestamp_pc"] += 0.05


@pytest.fixture
def rig():
    robot, events, now = Robot(), [], [0.0]

    def sleep(dt):
        now[0] += dt
        robot.tick()

    console = MotionConsole(robot, lambda event, **data: events.append((event, data)),
                            clock=lambda: now[0], sleep=sleep)
    return robot, console, events


def test_startup_never_repositions(rig):
    robot, console, _ = rig
    console.ready()
    assert robot.calls == []


def test_station_headings_are_absolute_from_start_even_across_wrap(rig):
    robot, console, events = rig
    robot.base.status["theta"] = math.radians(175)
    console.start["theta_rad"] = math.radians(175)
    for heading in (-48, -16, 16, 48, 0):
        console.execute(f"heading {heading}")
    turns = [call[1] for call in robot.calls if call[0] == "base"]
    assert turns == pytest.approx([math.radians(x) for x in (-48, 32, 32, 32, -48)])
    assert sum(e == "target_reached" for e, _ in events) == 5


def test_rotation_requires_retracted_arm_and_starting_height(rig):
    robot, console, _ = rig
    robot.arm.status["pos"] = 0.1
    with pytest.raises(ValueError, match="Retract"):
        console.execute("heading 16")
    robot.arm.status["pos"] = 0.0
    robot.lift.status["pos"] = 0.8
    with pytest.raises(ValueError, match="Raise"):
        console.execute("heading 16")
    assert robot.calls == []


def test_ten_centimetres_means_relative_lift_drop(rig):
    robot, console, _ = rig
    for _ in range(5):
        console.execute("lift -2")
    assert robot.lift.status["pos"] == pytest.approx(0.8)
    assert all(call[0] in ("lift", "push") for call in robot.calls)


@pytest.mark.parametrize("command", ["heading nan", "heading 90", "arm inf", "arm 10", "lift -10", "grip -50"])
def test_invalid_or_excessive_commands_send_nothing(rig, command):
    robot, console, _ = rig
    with pytest.raises(ValueError):
        console.execute(command)
    assert robot.calls == []


def test_joint_and_extra_commissioning_limits(rig):
    robot, console, _ = rig
    robot.arm.status["pos"] = 0.345
    with pytest.raises(ValueError, match="outside"):
        console.execute("arm 1")
    robot.lift.soft_motion_limits["current"] = [0.895, 1.1]
    with pytest.raises(ValueError, match="outside"):
        console.execute("lift -1")
    assert robot.calls == []


@pytest.mark.parametrize("problem", ["unhomed", "runstop", "drift", "stale", "nan", "guarded"])
def test_health_faults_prevent_dispatch(rig, problem):
    robot, console, _ = rig
    if problem == "unhomed":
        robot.homed = False
    elif problem == "runstop":
        robot.pimu.status["runstop_event"] = True
    elif problem == "drift":
        robot.base.status["x"] = 0.03
    elif problem == "stale":
        robot.stale = True
    elif problem == "nan":
        robot.arm.status["pos"] = float("nan")
    else:
        robot.lift.motor.status["in_guarded_event"] = True
    with pytest.raises(MotionFault):
        console.execute("lift -1")
    assert robot.calls == []


def test_unreached_target_is_never_logged_as_success(rig):
    robot, console, events = rig
    robot.blocked = True
    with pytest.raises(MotionFault, match="not reached"):
        console.execute("arm 1")
    assert not any(event == "target_reached" for event, _ in events)


def test_gripper_open_uses_robot_pose_close_stops_at_touching(rig):
    robot, console, _ = rig
    console.execute("close")
    console.execute("open")
    console.execute("grip -5")
    assert [call[1] for call in robot.calls] == [0.0, 70.0, 65.0]
    assert all(call[0] == "gripper" for call in robot.calls)


def test_sdk_error_after_dispatch_is_fatal_not_syntax_error(rig):
    robot, console, _ = rig
    def error(*args, **kwargs):
        raise ValueError("servo error")
    robot.gripper.move_to = error
    with pytest.raises(MotionFault, match="servo error"):
        console.execute("open")


def test_stop_latches_runstop_without_recovery_motion(rig):
    robot, _, _ = rig
    request_stop(robot)
    assert robot.calls == [("runstop",), ("pimu_push",)]


def test_no_mode_or_piped_motion_never_imports_sdk(monkeypatch):
    with pytest.raises(SystemExit):
        main([])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        main(["--enable-motion"])


@pytest.mark.parametrize("ending", ["quit", "interrupt", "blocked", "status_only"])
def test_session_shutdown_and_reports(rig, tmp_path, monkeypatch, ending):
    robot, console, _ = rig
    robot.startup = lambda: True
    robot.stop = lambda: robot.calls.append(("sdk_stop",))
    monkeypatch.setitem(sys.modules, "stretch_body", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "stretch_body.robot", SimpleNamespace(Robot=lambda: robot))
    monkeypatch.setattr(motion_console, "version", lambda name: "0.7.31")
    monkeypatch.setattr(motion_console.signal, "signal", lambda *args: None)
    monkeypatch.setattr(motion_console.time, "sleep", lambda dt: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    def create(r, record):
        console.record = record
        return console
    monkeypatch.setattr(motion_console, "MotionConsole", create)
    answers = iter(["READY", "arm 1", "quit"])
    def answer(prompt):
        value = next(answers)
        if value == "quit" and ending == "interrupt":
            raise KeyboardInterrupt
        return value
    monkeypatch.setattr("builtins.input", answer)
    robot.blocked = ending == "blocked"
    report = tmp_path / "motion.jsonl"
    mode = "--status-only" if ending == "status_only" else "--enable-motion"
    result = main([mode, "--report", str(report)])
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    assert robot.calls[-1] == ("sdk_stop",)
    if ending in ("interrupt", "blocked"):
        assert result == 2
        assert robot.calls[-3:] == [("runstop",), ("pimu_push",), ("sdk_stop",)]
        assert rows[-1]["event"] == "fault"
    else:
        assert result == 0
        assert rows[-1]["event"] == "finished"
        assert ("runstop",) not in robot.calls
    if ending == "status_only":
        assert robot.calls == [("sdk_stop",)]
