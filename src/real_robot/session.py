"""Operator-mediated, transactional alternating HRC protocol."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional
import uuid

from .config import LabConfig
from .domain import PhysicalObservation, PhysicalTaskDomain, StateVector
from .hardware import ExecutionResult, HardwareExecutor


IDLE = "idle"
HUMAN_TURN = "human_turn"
ROBOT_PROPOSAL = "robot_proposal"
ROBOT_EXECUTING = "robot_executing"
CORRECTION_REQUIRED = "correction_required"
EXECUTION_FAILED = "execution_failed"
RECONCILIATION_REQUIRED = "reconciliation_required"
COMPLETE = "complete"
EMERGENCY_STOPPED = "emergency_stopped"

OBSERVE = "observe"
ASSIST = "assist"


class SessionStateError(RuntimeError):
    """Raised when an operator command is invalid in the current phase."""


class LiveHrcSession:
    """Transactional live protocol around one AdaptiveAgent.

    The external recipe and trial metadata are used only by the operator and
    journal.  They are never part of the learner state.  A robot transition is
    committed only after a durable, terminal execution result exists.
    """

    def __init__(
        self, agent: Any, domain: PhysicalTaskDomain, config: LabConfig,
        executor: HardwareExecutor, *,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ):
        if getattr(agent, "domain", None) is not domain:
            raise ValueError("agent and live session must share one domain adapter")
        self.agent = agent
        self.domain = domain
        self.config = config
        self.executor = executor
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._event_index = 0
        self._observed_recipes: set[str] = set()
        self._agent_snapshot: Any = None
        self._worker: Optional[threading.Thread] = None
        self._generation = 0
        self._pending_commit: Optional[Dict[str, Any]] = None
        self.phase = IDLE
        self.mode: Optional[str] = None
        self.recipe_id: Optional[str] = None
        self.trial_metadata: Dict[str, Any] = {}
        self.state: StateVector = domain.initial_state()
        self.completed: list[str] = []
        self.decisions: list[Dict[str, Any]] = []
        self.pending_distribution: Dict[str, float] = {}
        self.pending_prediction: Optional[str] = None
        self.last_execution: Optional[ExecutionResult] = None
        self.last_match: Optional[Mapping[str, Any]] = None
        self.error: Optional[str] = None

    @property
    def observed_recipes(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._observed_recipes))

    def restore_observed_recipes(self, recipe_ids: tuple[str, ...]) -> None:
        with self._lock:
            if self.phase != IDLE:
                raise SessionStateError("observed recipes can only be restored while idle")
            unknown = set(recipe_ids) - set(self.config.recipes)
            if unknown:
                raise SessionStateError(f"checkpoint has unknown recipes: {sorted(unknown)}")
            self._observed_recipes = set(recipe_ids)

    def start_episode(self, recipe_id: str, *, props_reset: bool, trial_metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        with self._lock:
            if self.phase not in {IDLE, COMPLETE}:
                raise SessionStateError("finish or abort the active episode first")
            if not props_reset:
                raise SessionStateError("operator must confirm that all proxy objects and placement slots were reset")
            if recipe_id not in self.config.recipes:
                raise SessionStateError(f"unknown recipe {recipe_id!r}")
            if self.agent.current_prefix:
                raise SessionStateError("agent has an unfinished online prefix")
            hardware = dict(self.executor.status())
            if not hardware.get("ready", True):
                raise SessionStateError(f"hardware is not ready: {hardware.get('error') or hardware.get('last_error') or 'unknown reason'}")
            self._generation += 1
            self._agent_snapshot = self.agent.snapshot()
            self.recipe_id = recipe_id
            self.mode = OBSERVE if recipe_id not in self._observed_recipes else ASSIST
            self.trial_metadata = self._validate_trial_metadata(trial_metadata or {})
            self.phase = HUMAN_TURN
            self.state = self.domain.initial_state()
            self.completed = []
            self.decisions = []
            self.pending_distribution = {}
            self.pending_prediction = None
            self.last_execution = None
            self.last_match = None
            self._pending_commit = None
            self.error = None
            if self.mode == OBSERVE:
                self.agent.start_demo()
            try:
                self._emit("episode_started", {
                    "recipe_id": recipe_id, "mode": self.mode,
                    "trial_metadata": dict(self.trial_metadata),
                    "config_digest": self.config.digest,
                })
            except Exception:
                self.agent.restore_from(self._agent_snapshot)
                self._reset_runtime_state()
                raise
        return self.snapshot()

    def record_human_action(self, action: str, *, physical_completed: bool = True) -> Mapping[str, Any]:
        with self._lock:
            if self.phase != HUMAN_TURN:
                raise SessionStateError("a human action is not expected in the current phase")
            if not physical_completed:
                raise SessionStateError("confirm physical completion before recording a human action")
            token = self._validate_planned_action(action)
            distribution: Optional[Mapping[str, float]] = None
            predicted: Optional[str] = None
            if self.mode == ASSIST:
                distribution, predicted = self._predict_current_state()
            self._commit_observation(token, executed_by="human", distribution=distribution, predicted=predicted)
            if self.phase != COMPLETE and self.mode == ASSIST:
                self._prepare_robot_proposal()
        return self.snapshot()

    def reject_robot_proposal(self, human_action: str, *, physical_completed: bool = True) -> Mapping[str, Any]:
        with self._lock:
            if self.phase not in {ROBOT_PROPOSAL, CORRECTION_REQUIRED}:
                raise SessionStateError("there is no robot proposal to correct")
            if not physical_completed:
                raise SessionStateError("confirm the participant completed the correction")
            token = self._validate_planned_action(human_action)
            if self.pending_prediction is not None and token == self.pending_prediction:
                raise SessionStateError("the correction equals the proposal; approve it instead")
            distribution = dict(self.pending_distribution)
            predicted = self.pending_prediction
            self._commit_observation(token, executed_by="human_correction", distribution=distribution, predicted=predicted)
            if self.phase != COMPLETE:
                # A correction consumes the human action but not the scheduled
                # robot turn, exactly matching the simulation protocol.
                self._prepare_robot_proposal()
        return self.snapshot()

    def approve_robot_proposal(self, *, human_clear: bool = False) -> Mapping[str, Any]:
        with self._lock:
            if self.phase != ROBOT_PROPOSAL or self.pending_prediction is None:
                raise SessionStateError("there is no executable robot proposal")
            if not human_clear:
                raise SessionStateError("operator must confirm the participant and bystanders are clear")
            if self.pending_prediction not in self._remaining_recipe_actions():
                raise SessionStateError("proposal is outside the selected recipe; correct it")
            self._start_robot_worker()
        return self.snapshot()

    def reconcile_robot_execution(self, *, execution_completed: bool, physical_scene_verified: bool) -> Mapping[str, Any]:
        with self._lock:
            if self.phase != RECONCILIATION_REQUIRED or self._pending_commit is None:
                raise SessionStateError("there is no execution awaiting reconciliation")
            if not physical_scene_verified:
                raise SessionStateError("inspect the bridge ledger, robot pose, gripper, object, and placement slot first")
            pending = self._pending_commit
            result: ExecutionResult = pending["execution"]
            if not execution_completed:
                self._pending_commit = None
                self.phase = EXECUTION_FAILED
                self.error = "operator verified that the ambiguous physical action did not complete; abort and reset before continuing"
                self._emit("execution_reconciled_not_completed", {
                    "execution_id": result.metadata.get("execution_id"),
                    "operator_verified": True,
                })
            else:
                if not result.success:
                    result = replace(
                        result, success=True, status="operator_reconciled_completed",
                        message="operator and bridge ledger verified physical completion",
                        metadata={**dict(result.metadata), "operator_reconciled": True},
                    )
                    pending["execution"] = result
                self._emit("execution_reconciled_completed", {
                    "execution_id": result.metadata.get("execution_id"),
                    "operator_verified": True,
                })
                try:
                    self._commit_pending_robot_observation()
                except Exception as exc:
                    self.phase = RECONCILIATION_REQUIRED
                    self.error = f"commit-only reconciliation failed: {exc}"
                    raise
        return self.snapshot()

    def abort_episode(self) -> Mapping[str, Any]:
        with self._lock:
            if self.phase == ROBOT_EXECUTING:
                raise SessionStateError("cannot abort during motion; emergency-stop first")
            if self.phase in {IDLE, COMPLETE}:
                self.phase = IDLE
            else:
                if self._agent_snapshot is not None:
                    self.agent.restore_from(self._agent_snapshot)
                aborted = self.recipe_id
                self._generation += 1
                self._reset_runtime_state()
                self._emit("episode_aborted", {"recipe_id": aborted})
        return self.snapshot()

    def emergency_stop(self) -> Mapping[str, Any]:
        result = dict(self.executor.emergency_stop())
        with self._lock:
            self._generation += 1
            self.phase = EMERGENCY_STOPPED
            self.error = str(result.get("error") or "emergency stop requested")
            self._emit("emergency_stop", result)
        return self.snapshot()

    def shutdown(self, timeout_s: float = 20.0) -> bool:
        """Request cancellation and wait boundedly for the session worker."""
        with self._lock:
            worker = self._worker
            executing = self.phase == ROBOT_EXECUTING
        if executing:
            self.emergency_stop()
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, float(timeout_s)))
        return worker is None or not worker.is_alive()

    def _prepare_robot_proposal(self) -> None:
        distribution, predicted = self._predict_current_state()
        self.pending_distribution = dict(distribution)
        self.pending_prediction = predicted
        self.phase = ROBOT_PROPOSAL if predicted is not None else CORRECTION_REQUIRED
        self._emit("robot_proposed", {
            "prediction": predicted, "distribution": distribution,
            "policy_stats": self.agent.policy_stats(),
        })

    def _predict_current_state(self) -> tuple[Dict[str, float], Optional[str]]:
        distribution = dict(self.agent.predict_actions(
            tuple(self.agent.current_prefix), state=self.state,
            action_universe=self.domain.actions,
        ))
        ranked = self.agent.rank_actions(distribution, k=1) if distribution else []
        return distribution, ranked[0] if ranked else None

    def _start_robot_worker(self) -> None:
        token = self.pending_prediction
        assert token is not None
        generation = self._generation
        execution_id = str(uuid.uuid4())
        slot_id = tuple(self.config.placement_slots)[len(self.completed)]
        self.phase = ROBOT_EXECUTING
        self.last_execution = None
        self._pending_commit = None
        self.error = None
        try:
            self._emit("robot_execution_started", {
                "action": token, "execution_id": execution_id,
                "placement_slot_id": slot_id,
            })
            self._worker = threading.Thread(
                target=self._execute_robot_action,
                args=(generation, token, dict(self.pending_distribution), execution_id, slot_id),
                daemon=True, name=f"hrc-robot-{token.lower()}",
            )
            self._worker.start()
        except Exception as exc:
            self.phase = ROBOT_PROPOSAL
            self.error = f"robot worker did not start: {exc}"
            self._worker = None
            raise

    def _execute_robot_action(self, generation: int, token: str, distribution: Mapping[str, float], execution_id: str, slot_id: str) -> None:
        try:
            result = self.executor.execute(
                self.config.actions[token], execution_id=execution_id,
                placement_slot_id=slot_id, config_digest=self.config.digest,
            )
        except Exception as exc:
            now = time.time()
            result = ExecutionResult(False, "backend_exception", str(exc), now, now, {"action": token, "execution_id": execution_id})
        with self._lock:
            if generation != self._generation or self.phase != ROBOT_EXECUTING:
                return
            self.last_execution = result
            if not result.success:
                if result.status == "execution_unknown":
                    self._pending_commit = {"token": token, "distribution": dict(distribution), "execution": result}
                    self.phase = RECONCILIATION_REQUIRED
                    self.error = result.message
                    self._emit("robot_execution_ambiguous", result.as_dict())
                else:
                    self.phase = EXECUTION_FAILED
                    self.error = result.message
                    self._emit("robot_execution_failed", result.as_dict())
                return
            self._pending_commit = {"token": token, "distribution": dict(distribution), "execution": result}
            try:
                # This event is fsynced before learner mutation and causes an
                # action-level recovery checkpoint in EventJournal.
                self._emit("robot_execution_succeeded", result.as_dict())
                self._commit_pending_robot_observation()
            except Exception as exc:
                self.phase = RECONCILIATION_REQUIRED
                self.error = f"physical execution succeeded but learner/journal commit failed: {exc}"
                try:
                    self._emit("learner_commit_failed", {"execution_id": execution_id, "error": str(exc)})
                except Exception:
                    pass

    def _commit_pending_robot_observation(self) -> None:
        pending = self._pending_commit
        if pending is None:
            raise SessionStateError("no physical execution is pending learner commit")
        result: ExecutionResult = pending["execution"]
        self._pending_commit = None
        try:
            self._commit_observation(
                str(pending["token"]), executed_by="robot",
                distribution=dict(pending["distribution"]),
                predicted=str(pending["token"]), execution=result,
            )
        except Exception:
            self._pending_commit = pending
            raise

    def _commit_observation(self, token: str, *, executed_by: str, distribution: Optional[Mapping[str, float]], predicted: Optional[str], execution: Optional[ExecutionResult] = None) -> None:
        before = self.state
        after = self.domain.successor(before, token)
        if after is None:
            raise SessionStateError(f"action {token} is not legal in the current task state")
        agent_before = self.agent.snapshot()
        state_before = self.state
        completed_before = list(self.completed)
        decisions_before = list(self.decisions)
        pending_distribution_before = dict(self.pending_distribution)
        pending_prediction_before = self.pending_prediction
        phase_before = self.phase
        match_before = self.last_match
        observed_before = set(self._observed_recipes)
        episode_agent_snapshot_before = self._agent_snapshot
        try:
            observation = PhysicalObservation(before, token, after)
            observed = self.agent.observe(
                observation,
                precomputed_distribution=None if distribution is None else dict(distribution),
                precomputed_prediction=predicted,
            )
            self.state = after
            self.completed.append(token)
            record = {
                "step": len(self.completed) - 1, "action": token,
                "intended_action": token, "executed_by": executed_by,
                "placement_slot_id": tuple(self.config.placement_slots)[len(self.completed) - 1],
                "predicted": predicted,
                "correct": predicted == token if predicted is not None else None,
                "prediction_kind": "robot_control" if executed_by == "robot" else (
                    "robot_proposal" if executed_by == "human_correction" else
                    ("human_turn_shadow" if distribution is not None else None)
                ),
                "distribution": dict(distribution or {}), "agent_step": observed.step,
                "execution": execution.as_dict() if execution is not None else None,
                "trial_metadata": dict(self.trial_metadata),
            }
            self.decisions.append(record)
            self.pending_distribution = {}
            self.pending_prediction = None
            episode_summary = None
            if not self._remaining_recipe_actions():
                episode_summary = self._complete_episode()
            elif self.mode == OBSERVE:
                self.phase = HUMAN_TURN
            elif executed_by == "robot":
                self.phase = HUMAN_TURN
            self._emit("episode_completed" if episode_summary is not None else "learner_transition_committed", {
                **record, "episode": episode_summary,
            })
        except Exception:
            self.agent.restore_from(agent_before)
            self.state = state_before
            self.completed = completed_before
            self.decisions = decisions_before
            self.pending_distribution = pending_distribution_before
            self.pending_prediction = pending_prediction_before
            self.phase = phase_before
            self.last_match = match_before
            self._observed_recipes = observed_before
            self._agent_snapshot = episode_agent_snapshot_before
            raise

    def _complete_episode(self) -> Mapping[str, Any]:
        recipe_id = self.recipe_id
        mode = self.mode
        match = self.agent.end_demo()
        self.last_match = {
            "kind": match.kind, "recipe_id": match.recipe_id,
            "variant_id": match.variant_id, "jaccard": float(match.jaccard),
            "order_distance": float(match.order_distance),
        }
        if mode == OBSERVE and recipe_id is not None:
            self._observed_recipes.add(recipe_id)
        robot_turns = sum(row["executed_by"] in {"robot", "human_correction"} for row in self.decisions)
        robot_hits = sum(row["executed_by"] == "robot" for row in self.decisions)
        corrections = sum(row["executed_by"] == "human_correction" for row in self.decisions)
        human_actions = sum(row["executed_by"] in {"human", "human_correction"} for row in self.decisions)
        shadow_rows = [row for row in self.decisions if row.get("prediction_kind") == "human_turn_shadow" and row.get("predicted") is not None]
        self.phase = COMPLETE
        self._agent_snapshot = None
        return {
            "external_recipe_id": recipe_id, "mode": mode,
            "trial_metadata": dict(self.trial_metadata),
            "learner_match": dict(self.last_match), "actions": list(self.completed),
            "summary": {
                "recipe_steps": len(self.completed), "robot_turn_count": robot_turns,
                "robot_top_1_hits": robot_hits,
                "robot_top_1": robot_hits / robot_turns if robot_turns else None,
                "human_correction_count": corrections, "human_action_count": human_actions,
                "normalized_human_action_load": human_actions / max(1, len(self.completed)),
                "human_shadow_prediction_count": len(shadow_rows),
                "human_shadow_top_1": sum(bool(row["correct"]) for row in shadow_rows) / len(shadow_rows) if shadow_rows else None,
            },
        }

    def _validate_planned_action(self, action: str) -> str:
        try:
            token = self.domain.canonical_action(action)
        except ValueError as exc:
            raise SessionStateError(str(exc)) from exc
        if token not in self._remaining_recipe_actions():
            raise SessionStateError(f"{token} is not an unfinished action in this recipe")
        if self.domain.successor(self.state, token) is None:
            raise SessionStateError(f"{token} is blocked by task prerequisites")
        return token

    def _remaining_recipe_actions(self) -> tuple[str, ...]:
        if self.recipe_id is None:
            return ()
        complete = set(self.completed)
        return tuple(action for action in self.config.recipes[self.recipe_id].actions if action not in complete)

    @staticmethod
    def _validate_trial_metadata(value: Mapping[str, Any]) -> Dict[str, Any]:
        if len(value) > 16:
            raise SessionStateError("trial_metadata has too many fields")
        output: Dict[str, Any] = {}
        for key, item in value.items():
            name = str(key).strip()
            if not name or len(name) > 64 or not isinstance(item, (str, int, float, bool, type(None))):
                raise SessionStateError("trial_metadata must contain short names and scalar JSON values")
            if isinstance(item, str) and len(item) > 256:
                raise SessionStateError("trial_metadata string is too long")
            output[name] = item
        return output

    def _reset_runtime_state(self) -> None:
        self.phase = IDLE
        self.mode = None
        self.recipe_id = None
        self.trial_metadata = {}
        self.state = self.domain.initial_state()
        self.completed = []
        self.decisions = []
        self.pending_distribution = {}
        self.pending_prediction = None
        self.last_execution = None
        self.last_match = None
        self.error = None
        self._pending_commit = None
        self._agent_snapshot = None

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        self._event_index += 1
        row = {
            "event_index": self._event_index,
            "timestamp": time.time(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_s": time.monotonic(),
            "event": event, "phase": self.phase, "mode": self.mode,
            "recipe_id": self.recipe_id, "payload": dict(payload),
        }
        if self.event_sink is not None:
            self.event_sink(row)

    def checkpoint_state(self) -> Mapping[str, Any]:
        with self._lock:
            pending = None
            if self._pending_commit is not None:
                pending = {
                    **self._pending_commit,
                    "execution": self._pending_commit["execution"].as_dict(),
                }
            return {
                "schema_version": 1, "event_index": self._event_index,
                "observed_recipes": list(sorted(self._observed_recipes)),
                "phase": self.phase, "mode": self.mode, "recipe_id": self.recipe_id,
                "trial_metadata": dict(self.trial_metadata), "state": list(self.state),
                "completed": list(self.completed), "decisions": list(self.decisions),
                "pending_distribution": dict(self.pending_distribution),
                "pending_prediction": self.pending_prediction,
                "last_execution": None if self.last_execution is None else self.last_execution.as_dict(),
                "last_match": None if self.last_match is None else dict(self.last_match),
                "error": self.error, "pending_commit": pending,
                "agent_start_snapshot": self._agent_snapshot,
            }

    def restore_checkpoint_state(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            if int(row.get("schema_version", 0)) != 1:
                raise SessionStateError("unsupported live-session checkpoint")
            recipe_id = row.get("recipe_id")
            if recipe_id is not None and recipe_id not in self.config.recipes:
                raise SessionStateError("checkpoint references an unknown recipe")
            self._event_index = int(row.get("event_index", 0))
            self._observed_recipes = set(map(str, row.get("observed_recipes", ())))
            self.phase = str(row.get("phase", IDLE))
            if self.phase == ROBOT_EXECUTING:
                self.phase = RECONCILIATION_REQUIRED
            self.mode = row.get("mode")
            self.recipe_id = recipe_id
            self.trial_metadata = dict(row.get("trial_metadata", {}))
            self.state = tuple(int(value) for value in row.get("state", self.domain.initial_state()))
            self.completed = list(map(str, row.get("completed", ())))
            self.decisions = list(row.get("decisions", ()))
            self.pending_distribution = {str(key): float(value) for key, value in dict(row.get("pending_distribution", {})).items()}
            self.pending_prediction = row.get("pending_prediction")
            self.last_execution = self._execution_from_mapping(row.get("last_execution"))
            self.last_match = row.get("last_match")
            self.error = row.get("error")
            pending = row.get("pending_commit")
            if isinstance(pending, Mapping):
                self._pending_commit = {
                    "token": str(pending["token"]),
                    "distribution": dict(pending.get("distribution", {})),
                    "execution": self._execution_from_mapping(pending.get("execution")),
                }
                self.phase = RECONCILIATION_REQUIRED
            self._agent_snapshot = row.get("agent_start_snapshot")
            if self._agent_snapshot is None and self.phase not in {IDLE, COMPLETE}:
                self._agent_snapshot = self.agent.snapshot()

    @staticmethod
    def _execution_from_mapping(value: Any) -> Optional[ExecutionResult]:
        if not isinstance(value, Mapping):
            return None
        return ExecutionResult(
            bool(value.get("success")), str(value.get("status", "unknown")),
            str(value.get("message", "")), float(value.get("started_at", 0.0)),
            float(value.get("finished_at", 0.0)), dict(value.get("metadata", {})),
            (None if value.get("monotonic_elapsed_s") is None else float(value["monotonic_elapsed_s"])),
        )

    def snapshot(self) -> Mapping[str, Any]:
        # Hardware status is cached by the HTTP executor and deliberately read
        # outside the session lock so UI polling cannot block protocol commits.
        hardware = dict(self.executor.status())
        with self._lock:
            recipe = self.config.recipes.get(self.recipe_id or "")
            remaining = self._remaining_recipe_actions()
            state = {
                "phase": self.phase, "mode": self.mode,
                "recipe_id": self.recipe_id,
                "recipe_label": recipe.label if recipe is not None else None,
                "trial_metadata": dict(self.trial_metadata),
                "next_placement_slot": (
                    tuple(self.config.placement_slots)[len(self.completed)]
                    if recipe is not None and len(self.completed) < len(self.config.placement_slots)
                    else None
                ),
                "completed": list(self.completed), "remaining_actions": list(remaining),
                "available_human_actions": [action for action in remaining if self.domain.successor(self.state, action) is not None],
                "pending_prediction": self.pending_prediction,
                "pending_prediction_in_recipe": self.pending_prediction in remaining,
                "pending_distribution": dict(self.pending_distribution),
                "decisions": list(self.decisions),
                "last_execution": self.last_execution.as_dict() if self.last_execution else None,
                "last_match": dict(self.last_match) if self.last_match else None,
                "error": self.error, "observed_recipes": list(sorted(self._observed_recipes)),
                "reconciliation_execution_id": None if self._pending_commit is None else self._pending_commit["execution"].metadata.get("execution_id"),
                "session_worker_alive": self._worker is not None and self._worker.is_alive(),
                "config_digest": self.config.digest,
            }
        state["hardware"] = hardware
        return state
