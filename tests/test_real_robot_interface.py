"""Tests for the completion-checked Stretch interface and live protocol."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.adaptive_agent import AdaptiveAgent
from src.models import Settings
from src.real_robot.cancellable_wait import wait_command_cancellable
from src.real_robot.config import ConfigurationError, LabConfig, load_lab_config
from src.real_robot.app import EventJournal, _load_checkpoint, _load_checkpoint_full
from src.real_robot.bridge import (
    BridgeConflict,
    ExecutionLedger,
    build_parser as build_bridge_parser,
)
from src.real_robot.domain import PhysicalObservation, PhysicalTaskDomain
from src.real_robot.hardware import DryRunExecutor, ExecutionResult, HttpStretchExecutor
from src.real_robot.report import build_report
from src.real_robot.runtime_qualification import qualify_runtime
from src.real_robot.schedule import StudySchedule, load_study_schedule
from src.real_robot.stretch_runtime.perception import wait_for_stable_marker
from src.real_robot.stretch_runtime.fingertip_geometry import fingertip_transforms
from src.real_robot.stretch_hardware import StretchHardwareController
from src.real_robot.session import (
    ASSIST,
    COMPLETE,
    CORRECTION_REQUIRED,
    EXECUTION_FAILED,
    HUMAN_TURN,
    ROBOT_PROPOSAL,
    LiveHrcSession,
    SessionStateError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src/real_robot/robot_configs" / "stretch3_lab.json"


@pytest.mark.parametrize("subdirectory", ["", "dry-run"])
def test_relocated_bridge_does_not_bypass_existing_ledger(tmp_path, monkeypatch, subdirectory):
    from src.real_robot import bridge

    monkeypatch.chdir(tmp_path)
    legacy_root = tmp_path / "real_robot_bridge_state"
    legacy_state = legacy_root / subdirectory
    legacy_state.mkdir(parents=True)
    record = '{"execution_id":"unfinished-001","status":"running"}\n'
    (legacy_state / "executions.jsonl").write_text(record, encoding="utf-8")
    (legacy_state / "bridge.lock").write_text("previous-process\n", encoding="utf-8")
    relocated_root = tmp_path / "src/real_robot/real_robot_bridge_state"

    def unexpected_start(*args, **kwargs):
        pytest.fail("Bridge acquired a new lock before migrating the old ledger")

    monkeypatch.setattr(bridge, "ProcessLock", unexpected_start)
    arguments = ["--config", str(CONFIG_PATH), "--camera-preview"]
    if subdirectory:
        arguments.extend(["--state-dir", str(relocated_root / subdirectory)])
    with pytest.raises(SystemExit, match="existing bridge state"):
        bridge.main(arguments)

    assert not relocated_root.exists()
    assert (legacy_state / "executions.jsonl").read_text(encoding="utf-8") == record
    relocated_root.parent.mkdir(parents=True)
    legacy_root.rename(relocated_root)
    ledger = ExecutionLedger(relocated_root / subdirectory / "executions.jsonl")
    assert ledger.unresolved_ids == ("unfinished-001",)


class FakeAgent:
    def __init__(self, domain):
        self.domain = domain
        self.current_prefix = []
        self.mode = "online"
        self.step_counter = 0
        self.predictions = []
        self.observations = []

    def snapshot(self):
        return copy.deepcopy({
            "current_prefix": self.current_prefix,
            "mode": self.mode,
            "step_counter": self.step_counter,
            "predictions": self.predictions,
            "observations": self.observations,
        })

    def restore_from(self, snapshot):
        self.current_prefix = snapshot["current_prefix"]
        self.mode = snapshot["mode"]
        self.step_counter = snapshot["step_counter"]
        self.predictions = snapshot["predictions"]
        self.observations = snapshot["observations"]

    def start_demo(self):
        self.mode = "observe"
        self.current_prefix = []

    def end_demo(self):
        self.mode = "online"
        self.current_prefix = []
        return SimpleNamespace(
            kind="known", recipe_id="R1", variant_id="V1",
            jaccard=1.0, order_distance=0.0,
        )

    def predict_actions(self, *_args, **_kwargs):
        return dict(self.predictions.pop(0)) if self.predictions else {}

    def rank_actions(self, distribution, k=1):
        return sorted(distribution, key=distribution.get, reverse=True)[:k]

    def policy_stats(self):
        return {"predictor": "fake"}

    def observe(self, observation, **kwargs):
        self.step_counter += 1
        self.current_prefix.append(observation.action)
        self.observations.append((observation, dict(kwargs)))
        return SimpleNamespace(step=self.step_counter)


class FailingExecutor(DryRunExecutor):
    def execute(self, action, **kwargs):
        now = time.time()
        return ExecutionResult(False, "failed", "injected failure", now, now, {"action": action.token, **kwargs})


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def trial_metadata(trial_id, preference_id="A", condition="test"):
    return {
        "trial_id": trial_id,
        "preference_id": preference_id,
        "condition": condition,
    }


def calibrated_config(tmp_path, base_config):
    raw = copy.deepcopy(base_config.as_dict())
    calibration_id = "test-calibration-v1"
    minimum_trials = 3
    marker_stations = {
        str(base_config.reference_marker.marker_id): tuple(base_config.stations),
        **{
            str(value): tuple(base_config.stations)
            for value in base_config.fingertip_marker_ids
        },
        **{
            str(value.marker_id): (value.source_station,)
            for value in base_config.objects.values()
        },
    }
    quality = {
        "min_marker_pixels": base_config.perception.min_marker_pixels + 1.0,
        "reprojection_error_px": base_config.perception.max_reprojection_error_px - 0.1,
        "depth_m": (
            base_config.perception.min_marker_depth_m
            + base_config.perception.max_marker_depth_m
        ) / 2.0,
    }
    record = {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "date_utc": "2026-09-05T12:00:00+00:00",
        "robot_id": "stretch-test-1",
        "d405_serial": "d405-test-1",
        "operator": "test-operator",
        "apparatus_revision": "table-v1",
        "config_file": "lab.json",
        "station_heading_trials_deg": {
            key: [row.heading_deg] * minimum_trials
            for key, row in base_config.stations.items()
        },
        "reference_marker_pose_trials_by_station_m": {
            key: [list(value)] * minimum_trials
            for key, value in base_config.reference_marker.expected_position_by_station_m.items()
        },
        "slot_pose_trials": {
            key: [
                {"success": True, "marker_xyz_m": list(value.verify_marker_xyz_m)}
                for _ in range(minimum_trials)
            ]
            for key, value in base_config.placement_slots.items()
        },
        "marker_quality_trials": {
            marker_id: [
                {**quality, "station_id": station_id}
                for station_id in station_ids
                for _ in range(minimum_trials)
            ]
            for marker_id, station_ids in marker_stations.items()
        },
        "object_grasp_trials": {
            key: [{"success": True} for _ in range(minimum_trials)]
            for key in base_config.objects
        },
        "canonical_return_trials_by_station": {
            station_id: [
                {"heading_error_deg": 0.5, "translation_drift_m": 0.005}
                for _ in range(minimum_trials)
            ]
            for station_id in base_config.stations
        },
        "fault_injection_results": {
            "stale_velocity_command": {"passed": True},
            "camera_disconnect": {"passed": True},
            "command_timeout": {"passed": True},
            "emergency_stop": {"passed": True, "latency_s": 0.1},
        },
        "acceptance_thresholds_frozen_before_test": {
            "min_trials_per_pose": minimum_trials,
            "max_heading_error_deg": 3.0,
            "max_translation_drift_m": 0.025,
            "min_grasp_success_rate": 0.9,
            "min_placement_success_rate": 0.9,
            "max_software_stop_latency_s": 0.25,
        },
        "notes": "synthetic unit-test evidence",
    }
    runtime_lock = {
        "schema_version": 1,
        "python_version": "3.10.0",
        "modules": {
            "stretch_body": "1", "pyrealsense2": "2", "cv2": "3",
            "numpy": "4", "scipy": "5",
        },
        "robot_id": record["robot_id"],
        "d405_serial": record["d405_serial"],
        "d405_firmware": "firmware-test-1",
    }
    record_path = tmp_path / "calibration.json"
    runtime_path = tmp_path / "runtime.json"
    record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_lock, sort_keys=True), encoding="utf-8")
    raw["motion"].update({
        "calibrated": True,
        "rotation_tolerance_deg": 3.0,
        "calibration_id": calibration_id,
        "calibration_record": record_path.name,
        "calibration_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "runtime_lock": runtime_path.name,
        "runtime_lock_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
    })
    config_path = tmp_path / "lab.json"
    config_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    return load_lab_config(config_path), config_path, record_path, runtime_path


@pytest.fixture
def config():
    return load_lab_config(CONFIG_PATH)


@pytest.fixture
def domain(config):
    return PhysicalTaskDomain(config)


def test_sample_config_has_five_unique_station_headings_and_reserved_fingertip_ids(config):
    assert len(config.stations) == 5
    assert len({station.heading_deg for station in config.stations.values()}) == 5
    assert min(station.heading_deg for station in config.stations.values()) == -48.0
    assert max(station.heading_deg for station in config.stations.values()) == 48.0
    object_markers = {item.marker_id for item in config.objects.values()}
    assert not object_markers.intersection(config.fingertip_marker_ids)
    assert config.reference_marker.marker_id not in object_markers
    assert config.reference_marker.marker_id not in config.fingertip_marker_ids
    assert set(config.reference_marker.expected_position_by_station_m) == set(config.stations)
    assert config.tag_dictionary == "DICT_6X6_250"
    assert not config.motion.calibrated
    assert len(config.placement_slots) == 6
    assert all(len(recipe.actions) == 6 for recipe in config.recipes.values())
    assert config.perception.require_both_fingertips


def test_stretch_runtime_is_packaged_and_bridge_has_no_external_source_path():
    parser = build_bridge_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--gabriel-dir" not in options

    hardware_source = (PROJECT_ROOT / "src" / "real_robot" / "stretch_hardware.py").read_text()
    bridge_source = (PROJECT_ROOT / "src" / "real_robot" / "bridge.py").read_text()
    assert "Gabriel/" not in hardware_source
    assert "Gabriel/" not in bridge_source
    assert "sys.path" not in hardware_source

    runtime = PROJECT_ROOT / "src" / "real_robot" / "stretch_runtime"
    for filename in (
        "aruco_detector.py", "aruco_to_fingertips.py", "camera_service.py",
        "d405_helpers.py", "grasping.py", "normalized_velocity_control.py",
        "routes.py", "delivery.py",
    ):
        assert (runtime / filename).is_file()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in runtime.glob("*.py"))
    assert "wait_command()" not in combined
    assert "default_between_fingertips" not in combined
    assert "wait_for_frames(timeout_ms=" in (runtime / "camera_service.py").read_text(encoding="utf-8")
    operator_source = (PROJECT_ROOT / "src" / "real_robot" / "operator.html").read_text(encoding="utf-8")
    assert "/api/robot/retry" not in operator_source


def test_embedded_fingertip_transforms_are_rigid_and_independent():
    for transform in fingertip_transforms().values():
        rotation = transform[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    transforms = fingertip_transforms()
    transforms["left"][0, 0] = 99.0
    assert fingertip_transforms()["left"][0, 0] != 99.0


def test_live_startup_refuses_to_home_robot_implicitly(monkeypatch, config):
    class Robot:
        home_called = False
        stop_called = False

        def startup(self):
            return True

        def is_homed(self):
            return False

        def home(self):
            self.home_called = True

        def stop(self):
            self.stop_called = True

    robot = Robot()

    def fake_import(name, package=None):
        if name == "stretch_body.robot":
            return SimpleNamespace(Robot=lambda: robot)
        return SimpleNamespace()

    monkeypatch.setattr("src.real_robot.stretch_hardware.importlib.import_module", fake_import)
    live_config = replace(config, motion=replace(config.motion, calibrated=True, calibration_id="test-calibration"))
    with pytest.raises(RuntimeError, match="not homed"):
        StretchHardwareController(live_config, confirmed_start_station=config.home_station)
    assert not robot.home_called
    assert robot.stop_called


def test_live_execution_failure_stops_and_latches_controller(config):
    class Robot:
        stop_called = False

        def stop(self):
            self.stop_called = True

    def fail_grasp(**_kwargs):
        raise RuntimeError("injected grasp failure")

    action = config.actions["STAGE_BOWL"]
    controller = StretchHardwareController.__new__(StretchHardwareController)
    controller.config = config
    controller._sdk_lock = threading.Lock()
    controller._state_lock = threading.Lock()
    controller._busy = False
    controller._stopped = False
    controller._stop_event = threading.Event()
    controller._calibration_mode = False
    controller._last_error = None
    controller._last_action = None
    controller._execution_id = None
    controller._phase = "ready"
    controller._last_successful_phase = "ready"
    controller._pose_confident = True
    controller.current_station = action.source_station
    controller.robot = Robot()
    controller.camera_service = object()
    controller._modules = {
        "grasping": SimpleNamespace(execute_grasp_and_return_base=fail_grasp),
    }

    result = controller.execute(action, execution_id="test-execution-1", placement_slot_id="slot_1")
    assert not result["success"]
    assert result["restart_required"]
    assert controller._stopped
    assert controller._stop_event.is_set()
    assert controller.robot.stop_called


def test_config_rejects_duplicate_marker_ids(config):
    raw = copy.deepcopy(config.as_dict())
    raw["objects"]["tomato"]["marker_id"] = raw["objects"]["lettuce"]["marker_id"]
    with pytest.raises(ConfigurationError, match="duplicate"):
        LabConfig.from_mapping(raw)


def test_config_requires_reference_pose_for_every_station(config):
    raw = copy.deepcopy(config.as_dict())
    raw["reference_marker"]["expected_position_by_station_m"].pop("produce")
    with pytest.raises(ConfigurationError, match="every station"):
        LabConfig.from_mapping(raw)

    raw = copy.deepcopy(config.as_dict())
    raw["reference_marker"]["marker_id"] = raw["objects"]["lettuce"]["marker_id"]
    with pytest.raises(ConfigurationError, match="duplicate or reserved"):
        LabConfig.from_mapping(raw)


def test_domain_state_contains_completion_only_not_recipe_identity(config, domain):
    assert domain.initial_state() == (0,) * len(config.actions)
    first = config.recipes["garden_salad"].actions[0]
    after = domain.successor(domain.initial_state(), first)
    assert after is not None
    assert sum(after) == 1
    assert domain.successor(after, first) is None
    assert domain.goal_signature(config.recipes["garden_salad"].actions) != domain.goal_signature(config.recipes["fruit_bowl"].actions)


def test_real_adaptive_agent_trains_and_predicts_with_physical_domain(config, domain):
    settings = Settings(
        seed=19, verbose=False,
        irl_cold_steps=2, irl_warm_steps=1, irl_horizon=8,
        latent_strategy_enabled=False,
    )
    agent = AdaptiveAgent(settings=settings, domain=domain)
    actions = config.recipes["garden_salad"].actions
    state = domain.initial_state()
    agent.start_demo()
    for action in actions:
        after = domain.replay_transition(state, action)
        agent.observe(PhysicalObservation(state, action, after))
        state = after
    match = agent.end_demo()
    assert match.recipe_id == "R1"

    first = actions[0]
    after_first = domain.replay_transition(domain.initial_state(), first)
    distribution = agent.predict_actions(
        [first], state=after_first, action_universe=domain.actions,
    )
    assert distribution
    assert first not in distribution
    assert set(distribution) <= set(actions)


def _complete_first_observation(session, config, recipe_id="garden_salad"):
    session.start_episode(
        recipe_id, props_reset=True,
        trial_metadata=trial_metadata(f"{recipe_id}-observation", condition="observation"),
    )
    for action in config.recipes[recipe_id].actions:
        session.record_human_action(action)
    assert session.snapshot()["phase"] == COMPLETE


def test_live_protocol_vetoes_wrong_prediction_and_keeps_robot_turn(config, domain):
    agent = FakeAgent(domain)
    executor = DryRunExecutor()
    session = LiveHrcSession(agent, domain, config, executor)
    _complete_first_observation(session, config)

    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("garden-assist"))
    assert session.snapshot()["mode"] == ASSIST
    # First human action -> wrong/out-of-recipe robot proposal.  After the
    # correction, the second queued prediction must be proposed immediately.
    agent.predictions = [
        {"STAGE_BOWL": 1.0},
        {"FETCH_BANANA": 1.0},
        {"FETCH_TOMATO": 1.0},
    ]
    session.record_human_action("STAGE_BOWL")
    assert session.snapshot()["phase"] == ROBOT_PROPOSAL
    assert session.snapshot()["pending_prediction"] == "FETCH_BANANA"
    with pytest.raises(SessionStateError, match="outside"):
        session.approve_robot_proposal(human_clear=True)

    session.reject_robot_proposal("FETCH_LETTUCE")
    assert session.snapshot()["pending_prediction"] == "FETCH_TOMATO"
    session.approve_robot_proposal(human_clear=True)
    wait_until(lambda: session.snapshot()["phase"] == HUMAN_TURN)
    assert executor.executed == ["FETCH_TOMATO"]
    session.record_human_action("FETCH_DRESSING")
    assert session.snapshot()["phase"] == CORRECTION_REQUIRED
    session.reject_robot_proposal("FETCH_CUCUMBER")
    session.reject_robot_proposal("FETCH_CHEESE")
    assert session.snapshot()["phase"] == COMPLETE
    assert [row[1]["precomputed_prediction"] for row in agent.observations[-6:]] == [
        "STAGE_BOWL", "FETCH_BANANA", "FETCH_TOMATO", None, None, None,
    ]


def test_failed_hardware_action_is_not_observed_by_learner(config, domain):
    agent = FakeAgent(domain)
    session = LiveHrcSession(agent, domain, config, FailingExecutor())
    _complete_first_observation(session, config)
    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("failed-assist"))
    agent.predictions = [{"STAGE_BOWL": 1.0}, {"FETCH_LETTUCE": 1.0}]
    session.record_human_action("STAGE_BOWL")
    observations_before = len(agent.observations)
    session.approve_robot_proposal(human_clear=True)
    wait_until(lambda: session.snapshot()["phase"] == EXECUTION_FAILED)
    assert len(agent.observations) == observations_before
    assert session.snapshot()["completed"] == ["STAGE_BOWL"]


def test_episode_requires_prop_reset_confirmation_and_abort_restores_agent(config, domain):
    agent = FakeAgent(domain)
    session = LiveHrcSession(agent, domain, config, DryRunExecutor())
    with pytest.raises(SessionStateError, match="proxy objects"):
        session.start_episode("garden_salad", props_reset=False)
    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("aborted-observation"))
    session.record_human_action("STAGE_BOWL")
    assert agent.current_prefix == ["STAGE_BOWL"]
    session.abort_episode()
    assert agent.current_prefix == []
    assert session.snapshot()["phase"] == "idle"


def test_completed_episode_writes_resumable_checkpoint(tmp_path, config, domain):
    agent = FakeAgent(domain)
    journal = EventJournal(tmp_path / "run", config, hardware_backend="dry_run")
    session = LiveHrcSession(agent, domain, config, DryRunExecutor(), event_sink=journal.append)
    journal.bind(agent, session)
    _complete_first_observation(session, config)
    checkpoint = journal.root / "checkpoint.pkl"
    assert checkpoint.is_file()
    restored_agent, observed = _load_checkpoint(checkpoint, config)
    assert restored_agent.step_counter == len(config.recipes["garden_salad"].actions)
    assert observed == ("garden_salad",)


def test_action_level_checkpoint_resumes_observation_without_double_commit(tmp_path, config, domain):
    agent = FakeAgent(domain)
    journal = EventJournal(tmp_path / "run", config, hardware_backend="dry_run")
    session = LiveHrcSession(agent, domain, config, DryRunExecutor(), event_sink=journal.append)
    journal.bind(agent, session)
    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("checkpoint-observation"))
    session.record_human_action("STAGE_BOWL")

    restored_agent, observed, session_state = _load_checkpoint_full(journal.root / "checkpoint.pkl", config)
    restored = LiveHrcSession(restored_agent, restored_agent.domain, config, DryRunExecutor())
    restored.restore_observed_recipes(observed)
    restored.restore_checkpoint_state(session_state)
    assert restored.snapshot()["completed"] == ["STAGE_BOWL"]
    assert restored_agent.current_prefix == ["STAGE_BOWL"]
    for action in config.recipes["garden_salad"].actions[1:]:
        restored.record_human_action(action)
    assert restored.snapshot()["phase"] == COMPLETE
    assert restored_agent.step_counter == len(config.recipes["garden_salad"].actions)


def test_config_is_strict_and_requires_separated_headings(config):
    raw = copy.deepcopy(config.as_dict())
    raw["unexpected"] = True
    with pytest.raises(ConfigurationError, match="unknown keys"):
        LabConfig.from_mapping(raw)

    raw = copy.deepcopy(config.as_dict())
    raw["stations"]["produce"]["heading_deg"] = 1.0
    with pytest.raises(ConfigurationError, match="only 1.00 deg apart"):
        LabConfig.from_mapping(raw)

    raw = copy.deepcopy(config.as_dict())
    raw["motion"]["calibrated"] = True
    with pytest.raises(ConfigurationError, match="calibration_id"):
        LabConfig.from_mapping(raw)


def test_human_completion_and_clearance_are_explicit_gates(config, domain):
    agent = FakeAgent(domain)
    session = LiveHrcSession(agent, domain, config, DryRunExecutor())
    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("clearance-observation"))
    with pytest.raises(SessionStateError, match="physical completion"):
        session.record_human_action("STAGE_BOWL", physical_completed=False)
    session.abort_episode()
    _complete_first_observation(session, config)
    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("clearance-assist"))
    agent.predictions = [{"STAGE_BOWL": 1.0}, {"FETCH_LETTUCE": 1.0}]
    session.record_human_action("STAGE_BOWL")
    with pytest.raises(SessionStateError, match="bystanders are clear"):
        session.approve_robot_proposal(human_clear=False)


def test_execution_ledger_is_idempotent_and_detects_mismatched_reuse(tmp_path, config):
    class Controller:
        def __init__(self):
            self.calls = 0

        def execute(self, action, **kwargs):
            self.calls += 1
            return {"success": True, "status": "completed", "message": "ok", **kwargs}

        def status(self):
            return {"ready": True}

    controller = Controller()
    ledger = ExecutionLedger(tmp_path / "executions.jsonl")
    action = config.actions["STAGE_BOWL"]
    request = {
        "execution_id": "execution-0001", "config_digest": config.digest,
        "token": action.token, "object_id": action.object_id,
        "source_station": action.source_station,
        "destination_station": action.destination_station,
        "placement_slot_id": "slot_1",
    }
    _accepted, created = ledger.submit(request, action, controller)
    assert created
    wait_until(lambda: ledger.get("execution-0001")["status"] == "completed")
    terminal, created = ledger.submit(request, action, controller)
    assert not created and terminal["status"] == "completed"
    assert controller.calls == 1
    conflicting = {**request, "placement_slot_id": "slot_2"}
    with pytest.raises(BridgeConflict, match="different request"):
        ledger.submit(conflicting, action, controller)


def test_execution_ledger_requires_explicit_restart_reconciliation(tmp_path):
    path = tmp_path / "executions.jsonl"
    row = {
        "execution_id": "execution-unknown", "request_hash": "abc",
        "request": {}, "status": "running", "success": False,
    }
    path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
    ledger = ExecutionLedger(path)
    assert ledger.unresolved_ids == ("execution-unknown",)
    ledger.reconcile_after_restart("execution-unknown")
    assert ledger.unresolved_ids == ()
    assert ledger.get("execution-unknown")["status"] == "reconciled_after_restart"


def test_http_bridge_digest_handshake_and_async_execution(tmp_path, config):
    from src.real_robot.stretch_hardware import BridgeDryRunController

    ledger = ExecutionLedger(tmp_path / "executions.jsonl")
    controller = BridgeDryRunController(config)

    class InProcessExecutor(HttpStretchExecutor):
        def _refresh_loop(self):
            return

        def _request(self, method, path, payload=None, **_kwargs):
            if method == "GET" and path == "/v1/status":
                return {**controller.status(), **ledger.status()}
            if method == "POST" and path == "/v1/executions":
                action = config.actions[payload["token"]]
                return ledger.submit(dict(payload), action, controller)[0]
            if method == "GET" and path.startswith("/v1/executions/"):
                execution_id = path.rsplit("/", 1)[-1]
                row = ledger.get(execution_id)
                return row or {"status": "not_found", "message": "unknown"}
            raise AssertionError((method, path))

    executor = InProcessExecutor("http://in-process", timeout_s=5.0)
    try:
        status = executor.preflight(config.digest, require_motion=False)
        assert status["config_digest"] == config.digest
        with pytest.raises(RuntimeError, match="bridge is dry-run"):
            executor.preflight(config.digest, require_motion=True)
        with pytest.raises(RuntimeError, match="digests differ"):
            executor.preflight("wrong-digest", require_motion=False)
        result = executor.execute(
            config.actions["STAGE_BOWL"], execution_id="execution-http-1",
            placement_slot_id="slot_1", config_digest=config.digest,
        )
        assert result.success
        assert result.metadata["postcondition"] == {"dry_run": True}
        controller.status = lambda: {
            "backend": "fake_live", "motion_enabled": True, "ready": True,
            "config_digest": config.digest, "calibration_mode": False,
        }
        with pytest.raises(RuntimeError, match="pass --require-motion"):
            executor.preflight(config.digest, require_motion=False)
        controller.status = lambda: {
            "backend": "fake_calibration", "motion_enabled": True, "ready": True,
            "config_digest": config.digest, "calibration_mode": True,
        }
        with pytest.raises(RuntimeError, match="calibration mode"):
            executor.preflight(config.digest, require_motion=True)
    finally:
        executor.close()


def test_successful_motion_with_commit_failure_requires_no_second_motion(config, domain):
    class FailOnceAgent(FakeAgent):
        fail_next = False

        def observe(self, observation, **kwargs):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("injected learner failure")
            return super().observe(observation, **kwargs)

    agent = FailOnceAgent(domain)
    executor = DryRunExecutor()
    session = LiveHrcSession(agent, domain, config, executor)
    _complete_first_observation(session, config)
    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("commit-failure-assist"))
    agent.predictions = [{"STAGE_BOWL": 1.0}, {"FETCH_LETTUCE": 1.0}]
    session.record_human_action("STAGE_BOWL")
    agent.fail_next = True
    session.approve_robot_proposal(human_clear=True)
    wait_until(lambda: session.snapshot()["phase"] == "reconciliation_required")
    assert executor.executed == ["FETCH_LETTUCE"]
    session.reconcile_robot_execution(execution_completed=True, physical_scene_verified=True)
    assert executor.executed == ["FETCH_LETTUCE"]
    assert session.snapshot()["phase"] == HUMAN_TURN


def test_journal_sink_failure_after_physical_success_uses_commit_only_recovery(config, domain):
    class FailSuccessEventOnce:
        failed = False

        def __call__(self, row):
            if row["event"] == "robot_execution_succeeded" and not self.failed:
                self.failed = True
                raise OSError("injected fsync failure")

    agent = FakeAgent(domain)
    executor = DryRunExecutor()
    session = LiveHrcSession(agent, domain, config, executor, event_sink=FailSuccessEventOnce())
    _complete_first_observation(session, config)
    session.start_episode("garden_salad", props_reset=True, trial_metadata=trial_metadata("journal-failure-assist"))
    agent.predictions = [{"STAGE_BOWL": 1.0}, {"FETCH_LETTUCE": 1.0}]
    session.record_human_action("STAGE_BOWL")
    session.approve_robot_proposal(human_clear=True)
    wait_until(lambda: session.snapshot()["phase"] == "reconciliation_required")
    session.reconcile_robot_execution(execution_completed=True, physical_scene_verified=True)
    assert executor.executed == ["FETCH_LETTUCE"]
    assert session.snapshot()["completed"] == ["STAGE_BOWL", "FETCH_LETTUCE"]


def test_checked_station_rotation_does_not_advance_logical_pose_on_large_error(config):
    class Base:
        status = {"theta": 0.0}

    class Robot:
        base = Base()

    def inaccurate_rotate(robot, angle_deg, **_kwargs):
        robot.base.status["theta"] += __import__("math").radians(angle_deg / 2.0)
        return True

    controller = StretchHardwareController.__new__(StretchHardwareController)
    controller.config = config
    controller.robot = Robot()
    controller._modules = {"routes": SimpleNamespace(rotate_deg=inaccurate_rotate)}
    controller._stop_event = threading.Event()
    controller._calibration_mode = False
    controller._pose_confident = True
    controller.current_station = "workspace"
    with pytest.raises(RuntimeError, match="exceeds calibrated tolerance"):
        controller._move_to_station("produce", time.monotonic() + 1.0)
    assert controller.current_station == "workspace"
    assert not controller._pose_confident


def test_reference_marker_failure_does_not_advance_logical_pose(config):
    class Base:
        status = {"theta": 0.0}

    class Robot:
        base = Base()

    def accurate_rotate(robot, angle_deg, **_kwargs):
        robot.base.status["theta"] += __import__("math").radians(angle_deg)
        return True

    def missing_reference(**_kwargs):
        raise RuntimeError("reference marker not found")

    controller = StretchHardwareController.__new__(StretchHardwareController)
    controller.config = replace(
        config, motion=replace(config.motion, calibrated=True, calibration_id="test")
    )
    controller.robot = Robot()
    controller.camera_service = object()
    controller._modules = {
        "routes": SimpleNamespace(rotate_deg=accurate_rotate),
        "perception": SimpleNamespace(wait_for_stable_marker=missing_reference),
    }
    controller._state_lock = threading.Lock()
    controller._last_reference_evidence = None
    controller._stop_event = threading.Event()
    controller._calibration_mode = False
    controller._pose_confident = True
    controller.current_station = "workspace"
    with pytest.raises(RuntimeError, match="reference marker not found"):
        controller._move_to_station("produce", time.monotonic() + 1.0)
    assert controller.current_station == "workspace"
    assert not controller._pose_confident


def test_stable_marker_postcondition_rejects_jump_then_accepts_consecutive_frames():
    positions = [
        np.array([0.0, 0.0, 0.30]),
        np.array([0.10, 0.0, 0.30]),
        np.array([0.101, 0.0, 0.30]),
        np.array([0.102, 0.0, 0.30]),
    ]

    class Detector:
        def __init__(self, **_kwargs):
            self.position = positions[0]

        def update(self, _color, _camera_info):
            self.position = positions.pop(0) if positions else self.position

        def get_detected_marker_dict(self):
            return {10: {
                "pos": self.position, "min_dist_between_corners": 30.0,
                "reprojection_error_px": 0.5,
            }}

    class Camera:
        frame = 0

        def get_latest_bundle(self):
            self.frame += 1
            return {
                "frame_id": self.frame, "age_s": 0.0,
                "color": np.zeros((4, 4, 3), dtype=np.uint8),
                "color_camera_info": {},
            }

    evidence = wait_for_stable_marker(
        camera_service=Camera(), marker_info={}, marker_id=10,
        expected_xyz_m=[0.101, 0.0, 0.30], tolerance_m=0.02,
        stable_frames=3, max_position_jump_m=0.02,
        max_frame_age_s=0.3, deadline=time.monotonic() + 1.0,
        _detector_type=Detector,
    )
    assert evidence["stable_frames"] == 3
    assert evidence["position_error_m"] < 0.01


def test_run_report_checks_integrity_and_outcomes(tmp_path, config, domain):
    agent = FakeAgent(domain)
    journal = EventJournal(tmp_path / "run", config, hardware_backend="dry_run")
    session = LiveHrcSession(agent, domain, config, DryRunExecutor(), event_sink=journal.append)
    journal.bind(agent, session)
    _complete_first_observation(session, config)
    journal.close()
    report = build_report(journal.root)
    assert report["integrity"]["event_indices_contiguous"]
    assert report["outcomes"]["episodes_completed"] == 1
    assert report["outcomes"]["actions_committed"] == len(config.recipes["garden_salad"].actions)
    assert report["outcomes"]["by_condition"]["observation"]["episodes"] == 1
    assert len(report["episodes"]) == 1
    assert set(report["shadow_baselines"]) == {"latest", "no_decay", "bc"}
    assert not report["integrity"]["publication_eligible"]


def test_publication_report_requires_exact_schedule_and_physical_evidence(
    tmp_path, config,
):
    live_config, _config_path, _record_path, _runtime_path = calibrated_config(
        tmp_path, config,
    )
    recipe_ids = tuple(live_config.recipes)[:3]
    schedule_rows = []
    for index, recipe_id in enumerate(recipe_ids, 1):
        schedule_rows.append({
            "trial_id": f"P900-{index:03d}", "recipe_id": recipe_id,
            "preference_id": "A", "condition": "observation",
            "expected_mode": "observe",
        })
    for index, recipe_id in enumerate(recipe_ids, len(recipe_ids) + 1):
        schedule_rows.append({
            "trial_id": f"P900-{index:03d}", "recipe_id": recipe_id,
            "preference_id": "B", "condition": "drift",
            "expected_mode": "assist",
        })
    schedule = StudySchedule.from_mapping({
        "schema_version": 1, "schedule_id": "publication-test",
        "participant_id": "P900", "counterbalance_id": "order-test",
        "episodes": schedule_rows,
    }, live_config)
    hardware = {
        "backend": "stretch3_packaged_runtime", "motion_enabled": True,
        "ready": True, "config_digest": live_config.digest,
        "calibration_id": live_config.motion.calibration_id,
        "runtime_qualification": {"qualified": True},
    }
    journal = EventJournal(
        tmp_path / "publication-run", live_config,
        hardware_backend="stretch_http_bridge", hardware_preflight=hardware,
        require_motion=True, schedule=schedule, publication_run=True,
    )
    domain = PhysicalTaskDomain(live_config)
    agent = FakeAgent(domain)
    session = LiveHrcSession(
        agent, domain, live_config, DryRunExecutor(),
        event_sink=journal.append, schedule=schedule,
    )
    journal.bind(agent, session)
    for scheduled in schedule.episodes:
        actions = live_config.recipes[scheduled.recipe_id].actions
        session.start_episode(scheduled.recipe_id, props_reset=True)
        if scheduled.expected_mode == "observe":
            for action in actions:
                session.record_human_action(action)
        else:
            agent.predictions = [{action: 1.0} for action in actions]
            for index, action in enumerate(actions):
                if index % 2 == 0:
                    session.record_human_action(action)
                else:
                    assert session.snapshot()["pending_prediction"] == action
                    session.approve_robot_proposal(human_clear=True)
                    wait_until(lambda: session.snapshot()["phase"] != "robot_executing")
        assert session.snapshot()["phase"] == COMPLETE
    journal.close()
    manifest = json.loads(journal.manifest_path.read_text(encoding="utf-8"))
    manifest["software"].update({"git_commit": "a" * 40, "git_dirty": False})
    journal.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8",
    )

    report = build_report(journal.root)
    assert report["integrity"]["publication_eligible"]
    assert report["integrity"]["schedule_complete"]
    assert report["integrity"]["robot_postconditions_complete"]
    assert all(
        row["overall"]["teacher_forced_predictions"] > 0
        for row in report["shadow_baselines"].values()
    )

    event_lines = journal.events_path.read_text(encoding="utf-8").splitlines()
    final_event = json.loads(event_lines[-1])
    final_event["payload"]["episode"]["trial_metadata"]["preference_id"] = "tampered"
    event_lines[-1] = json.dumps(final_event, sort_keys=True)
    journal.events_path.write_text("\n".join(event_lines) + "\n", encoding="utf-8")
    tampered = build_report(journal.root)
    assert not tampered["integrity"]["schedule_complete"]
    assert not tampered["integrity"]["publication_eligible"]


def test_trial_metadata_is_required_nonempty_and_unique(config, domain):
    session = LiveHrcSession(FakeAgent(domain), domain, config, DryRunExecutor())
    with pytest.raises(SessionStateError, match="trial_metadata"):
        session.start_episode("garden_salad", props_reset=True)
    with pytest.raises(SessionStateError, match="preference_id"):
        session.start_episode(
            "garden_salad", props_reset=True,
            trial_metadata={"trial_id": "t1", "preference_id": "", "condition": "settled"},
        )
    _complete_first_observation(session, config)
    with pytest.raises(SessionStateError, match="already been used"):
        session.start_episode(
            "garden_salad", props_reset=True,
            trial_metadata=trial_metadata("garden_salad-observation"),
        )


def test_frozen_schedule_enforces_observation_block_and_next_trial(config, domain):
    schedule = load_study_schedule(
        PROJECT_ROOT / "src/real_robot/robot_configs" / "study_schedule.template.json", config,
    )
    assert len(schedule.episodes) == 16
    session = LiveHrcSession(
        FakeAgent(domain), domain, config, DryRunExecutor(), schedule=schedule,
    )
    with pytest.raises(SessionStateError, match="next scheduled recipe"):
        session.start_episode("fruit_bowl", props_reset=True)
    session.start_episode("garden_salad", props_reset=True)
    assert session.snapshot()["trial_metadata"]["trial_id"] == "P001-001"
    for action in config.recipes["garden_salad"].actions:
        session.record_human_action(action)
    state = session.snapshot()
    assert state["schedule"]["index"] == 1
    assert state["schedule"]["next"]["recipe_id"] == "fruit_bowl"

    bad = copy.deepcopy(schedule.as_dict())
    bad["episodes"][1]["expected_mode"] = "assist"
    with pytest.raises(ConfigurationError, match="assist before observation"):
        StudySchedule.from_mapping(bad, config)


def test_cancellable_wait_checks_stop_between_short_sdk_waits():
    stop = threading.Event()

    class Robot:
        timeouts = []

        def wait_command(self, timeout):
            self.timeouts.append(timeout)
            stop.set()
            return False

    robot = Robot()
    started = time.monotonic()
    assert not wait_command_cancellable(
        robot, timeout_s=10.0, cancel_event=stop, poll_s=0.05,
    )
    assert time.monotonic() - started < 0.5
    assert len(robot.timeouts) == 1
    assert robot.timeouts[0] <= 0.05


def test_stable_marker_sequence_restarts_after_stale_frame():
    bundles = [
        {"frame_id": 1, "age_s": 0.0},
        {"frame_id": 2, "age_s": 1.0},
        {"frame_id": 3, "age_s": 0.0},
        {"frame_id": 4, "age_s": 0.0},
        {"frame_id": 5, "age_s": 0.0},
    ]

    class Camera:
        def get_latest_bundle(self):
            row = bundles.pop(0) if len(bundles) > 1 else bundles[0]
            return {
                **row,
                "color": np.zeros((2, 2, 3), dtype=np.uint8),
                "color_camera_info": {},
            }

    class Detector:
        def __init__(self, **_kwargs):
            pass

        def update(self, _color, _camera_info):
            pass

        def get_detected_marker_dict(self):
            return {10: {
                "pos": np.array([0.0, 0.0, 0.3]),
                "min_dist_between_corners": 30.0,
                "reprojection_error_px": 0.2,
            }}

    evidence = wait_for_stable_marker(
        camera_service=Camera(), marker_info={}, marker_id=10,
        expected_xyz_m=[0.0, 0.0, 0.3], tolerance_m=0.02,
        stable_frames=3, max_position_jump_m=0.02,
        max_frame_age_s=0.3, deadline=time.monotonic() + 1.0,
        _detector_type=Detector,
    )
    assert evidence["frame_id"] == 5


def test_calibrated_config_requires_semantic_evidence_and_runtime_lock(tmp_path, config):
    loaded, config_path, record_path, _runtime_path = calibrated_config(tmp_path, config)
    assert loaded.motion.calibrated
    assert loaded.calibration_record_data["robot_id"] == "stretch-test-1"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["station_heading_trials_deg"]["produce"] = []
    record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["motion"]["calibration_record_sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="recorded trials"):
        load_lab_config(config_path)


def test_runtime_qualification_checks_versions_and_hardware(monkeypatch):
    observed = {
        "python_version": "3.10.0",
        "modules": {
            "stretch_body": "1", "pyrealsense2": "2", "cv2": "3",
            "numpy": "4", "scipy": "5",
        },
        "import_errors": {},
    }
    monkeypatch.setattr(
        "src.real_robot.runtime_qualification.inspect_runtime", lambda: observed,
    )
    lock = {
        **observed,
        "schema_version": 1,
        "robot_id": "stretch-1", "d405_serial": "camera-1",
        "d405_firmware": "firmware-1",
    }
    lock.pop("import_errors")
    result = qualify_runtime(
        lock, robot_id="stretch-1",
        camera_identity={"serial_number": "camera-1", "firmware_version": "firmware-1"},
    )
    assert result["qualified"]
    mismatch = qualify_runtime(
        lock, robot_id="stretch-2",
        camera_identity={"serial_number": "camera-1", "firmware_version": "firmware-1"},
    )
    assert not mismatch["qualified"]
    assert "robot_id" in mismatch["mismatches"][0]


def test_velocity_controller_thread_failure_is_latched(monkeypatch):
    stretch = ModuleType("stretch_body")
    stretch.__path__ = []
    robot_module = ModuleType("stretch_body.robot")
    hello_module = ModuleType("stretch_body.hello_utils")
    params_module = ModuleType("stretch_body.robot_params")
    params_module.RobotParams = type("RobotParams", (), {})
    monkeypatch.setitem(sys.modules, "stretch_body", stretch)
    monkeypatch.setitem(sys.modules, "stretch_body.robot", robot_module)
    monkeypatch.setitem(sys.modules, "stretch_body.hello_utils", hello_module)
    monkeypatch.setitem(sys.modules, "stretch_body.robot_params", params_module)
    module_name = "src.real_robot.stretch_runtime.normalized_velocity_control"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    control = module.NormalizedVelocityControl.__new__(module.NormalizedVelocityControl)
    control.lock = threading.Lock()
    control.stop_loop = False
    control.new_command_received = True
    control.command = {"num": 1, "time": time.monotonic(), "cmd": {"arm_out": 1.0}}
    control.command_watchdog_s = 0.1
    control.wait_between_executions = 0.001
    control._last_executed_nonzero = False
    control.watchdog_stop_count = 0
    control._controller_error = None
    control._controller_safety_stop_error = None
    control.controller_thread = None

    def fail_execute(_command):
        raise RuntimeError("injected SDK failure")

    control._execute = fail_execute
    control._start_controller()
    wait_until(lambda: not control.controller_thread.is_alive())
    status = control.status()
    assert "injected SDK failure" in status["error"]
    assert status["safety_stop_error"] is not None
    with pytest.raises(RuntimeError, match="thread failed"):
        control.set_command({"arm_out": 0.0})


def test_abort_rebinds_real_agent_to_session_domain(config, domain):
    settings = Settings(
        seed=7, verbose=False, irl_cold_steps=1, irl_warm_steps=1,
        irl_horizon=6, latent_strategy_enabled=False,
    )
    agent = AdaptiveAgent(settings=settings, domain=domain)
    session = LiveHrcSession(agent, domain, config, DryRunExecutor())
    session.start_episode(
        "garden_salad", props_reset=True,
        trial_metadata=trial_metadata("domain-rebind", condition="observation"),
    )
    session.record_human_action("STAGE_BOWL")
    session.abort_episode()
    assert agent.domain is session.domain
    assert agent.maxent.domain is session.domain
