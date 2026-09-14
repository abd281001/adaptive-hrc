"""Open-ended, label-free physical Adaptive-HRC protocol."""
from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional
import uuid

from .config import LabConfig
from .domain import PhysicalObservation
from .hardware import ExecutionResult, HardwareExecutor
from .session import (
    ASSIST,
    COMPLETE,
    CORRECTION_REQUIRED,
    EMERGENCY_STOPPED,
    EXECUTION_FAILED,
    HUMAN_TURN,
    IDLE,
    OBSERVE,
    RECONCILIATION_REQUIRED,
    ROBOT_EXECUTING,
    ROBOT_PROPOSAL,
    SessionStateError,
)


class FreeformLiveHrcSession:
    """Physical HRC session with no externally supplied recipe/preference label."""

    protocol = "freeform_physical"

    def __init__(
        self,
        agent: Any,
        domain: Any,
        config: LabConfig,
        executor: HardwareExecutor,
        *,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ):
        if getattr(agent, "domain", None) is not domain:
            raise ValueError(
                "agent and freeform session must share one domain adapter"
            )

        self.agent = agent
        self.domain = domain
        self.config = config
        self.executor = executor
        self.event_sink = event_sink

        self._lock = threading.RLock()
        self._event_index = 0
        self._generation = 0
        self._worker: Optional[threading.Thread] = None
        self._pending_commit: Optional[Dict[str, Any]] = None
        self._agent_snapshot: Any = None

        self.phase = IDLE
        self.mode: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.state = domain.initial_state()
        self.completed: list[str] = []
        self.decisions: list[Dict[str, Any]] = []
        self.pending_distribution: Dict[str, float] = {}
        self.pending_prediction: Optional[str] = None
        self.last_execution: Optional[ExecutionResult] = None
        self.last_match: Optional[Mapping[str, Any]] = None
        self.error: Optional[str] = None

    @property
    def observed_recipes(self) -> tuple[str, ...]:
        variants = getattr(
            getattr(self.agent, "library", None),
            "variants",
            {},
        )
        return tuple(sorted(str(key) for key in variants))

    def start_episode(
        self,
        mode: str,
        *,
        props_reset: bool,
    ) -> Mapping[str, Any]:
        with self._lock:
            if self.phase not in {IDLE, COMPLETE}:
                raise SessionStateError(
                    "finish or abort the active freeform episode first"
                )

            if not props_reset:
                raise SessionStateError(
                    "confirm the physical scene was reset before starting"
                )

            mode = str(mode).strip().lower()
            if mode not in {OBSERVE, ASSIST}:
                raise SessionStateError(
                    "freeform mode must be 'observe' or 'assist'"
                )

            if mode == ASSIST and not self.observed_recipes:
                raise SessionStateError(
                    "assist requires at least one completed demonstration"
                )

            if getattr(self.agent, "current_prefix", ()):
                raise SessionStateError(
                    "agent has an unfinished online prefix"
                )

            hardware = dict(self.executor.status())
            if not hardware.get("ready", True):
                raise SessionStateError(
                    "hardware is not ready: "
                    + str(
                        hardware.get("error")
                        or hardware.get("last_error")
                        or hardware.get("current_phase")
                        or "unknown reason"
                    )
                )

            if not hardware.get("fixed_layout_demo"):
                raise SessionStateError(
                    "freeform physical mode requires the fixed-layout live bridge"
                )

            reset_scene = getattr(
                self.executor,
                "reset_scene_baseline",
                None,
            )
            if not callable(reset_scene):
                raise SessionStateError(
                    "hardware backend cannot capture a scene baseline"
                )

            baseline = dict(reset_scene())
            scene = baseline.get("scene")
            if not isinstance(scene, Mapping):
                raise SessionStateError(
                    "scene baseline did not return a physical scene"
                )

            self.domain.set_initial_scene(scene)

            self._generation += 1
            self._agent_snapshot = self.agent.snapshot()

            self.phase = HUMAN_TURN
            self.mode = mode
            self.episode_id = uuid.uuid4().hex
            self.state = self.domain.initial_state()
            self.completed = []
            self.decisions = []
            self.pending_distribution = {}
            self.pending_prediction = None
            self.last_execution = None
            self.last_match = None
            self._pending_commit = None
            self.error = None

            if mode == OBSERVE:
                self.agent.start_demo()

            self._emit(
                "episode_started",
                {
                    "protocol": self.protocol,
                    "episode_id": self.episode_id,
                    "mode": self.mode,
                    "scene_baseline": baseline,
                    "initial_locations": dict(
                        self.domain.describe_state(self.state)
                    ),
                    "config_digest": self.config.digest,
                },
            )

        return self.snapshot()

    def observe_human_move(self) -> Mapping[str, Any]:
        with self._lock:
            phase_before = self.phase

            if phase_before not in {
                HUMAN_TURN,
                ROBOT_PROPOSAL,
                CORRECTION_REQUIRED,
            }:
                raise SessionStateError(
                    "a human physical move is not expected now"
                )

            observe = getattr(
                self.executor,
                "observe_human_move",
                None,
            )
            if not callable(observe):
                raise SessionStateError(
                    "hardware backend cannot observe marker scene changes"
                )

            evidence = dict(observe())
            token = str(
                evidence.get("freeform_action_token")
                or evidence.get("action_token")
                or ""
            ).strip().upper()

            try:
                token = self.domain.canonical_action(token)
            except ValueError as exc:
                self.phase = EXECUTION_FAILED
                self.error = (
                    "physical move was observed but could not be represented: "
                    f"{exc}; abort and reset before continuing"
                )
                self._emit(
                    "human_move_unrepresentable",
                    {
                        "evidence": evidence,
                        "error": str(exc),
                    },
                )
                raise SessionStateError(self.error) from exc

            if self.domain.successor(self.state, token) is None:
                self.phase = EXECUTION_FAILED
                self.error = (
                    f"observed physical action {token} is illegal from the "
                    "learner's current physical state; abort and reset"
                )
                self._emit(
                    "human_move_state_mismatch",
                    {
                        "action": token,
                        "evidence": evidence,
                        "state": list(self.state),
                    },
                )
                raise SessionStateError(self.error)

            self._emit(
                "human_move_perceived",
                {
                    "action": token,
                    "evidence": evidence,
                    "phase_before": phase_before,
                },
            )

            if phase_before == HUMAN_TURN:
                distribution = None
                predicted = None

                if self.mode == ASSIST:
                    distribution, predicted = self._predict_current_state()

                self._commit_observation(
                    token,
                    executed_by="human",
                    distribution=distribution,
                    predicted=predicted,
                    evidence=evidence,
                )

                if self.mode == ASSIST:
                    self._prepare_robot_proposal()

            else:
                distribution = dict(self.pending_distribution)
                predicted = self.pending_prediction

                self._commit_observation(
                    token,
                    executed_by="human_correction",
                    distribution=distribution,
                    predicted=predicted,
                    evidence=evidence,
                )

                self._prepare_robot_proposal()

        return self.snapshot()

    def _predict_current_state(
        self,
    ) -> tuple[Dict[str, float], Optional[str]]:
        legal = self.domain.legal_actions(
            self.state,
            self.domain.actions,
        )

        if not legal:
            return {}, None

        distribution = dict(
            self.agent.predict_actions(
                tuple(self.agent.current_prefix),
                state=self.state,
                action_universe=legal,
            )
        )

        ranked = (
            self.agent.rank_actions(distribution, k=1)
            if distribution
            else []
        )
        predicted = ranked[0] if ranked else None

        if (
            predicted is not None
            and self.domain.successor(self.state, predicted) is None
        ):
            predicted = None

        return distribution, predicted

    def _prepare_robot_proposal(self) -> None:
        distribution, predicted = self._predict_current_state()

        self.pending_distribution = dict(distribution)
        self.pending_prediction = predicted
        self.phase = (
            ROBOT_PROPOSAL
            if predicted is not None
            else CORRECTION_REQUIRED
        )

        self._emit(
            "robot_proposed",
            {
                "prediction": predicted,
                "distribution": distribution,
                "policy_stats": self.agent.policy_stats(),
            },
        )

    def approve_robot_proposal(
        self,
        *,
        human_clear: bool = False,
    ) -> Mapping[str, Any]:
        with self._lock:
            if (
                self.phase != ROBOT_PROPOSAL
                or self.pending_prediction is None
            ):
                raise SessionStateError(
                    "there is no executable robot proposal"
                )

            if not human_clear:
                raise SessionStateError(
                    "confirm participant/operator/bystanders are clear"
                )

            if (
                self.domain.successor(
                    self.state,
                    self.pending_prediction,
                )
                is None
            ):
                raise SessionStateError(
                    "the proposed action is no longer physically legal"
                )

            self._start_robot_worker()

        return self.snapshot()

    def _start_robot_worker(self) -> None:
        token = self.pending_prediction
        assert token is not None

        generation = self._generation
        execution_id = str(uuid.uuid4())

        self.phase = ROBOT_EXECUTING
        self.last_execution = None
        self._pending_commit = None
        self.error = None

        self._emit(
            "robot_execution_started",
            {
                "action": token,
                "execution_id": execution_id,
            },
        )

        self._worker = threading.Thread(
            target=self._execute_robot_action,
            args=(
                generation,
                token,
                dict(self.pending_distribution),
                execution_id,
            ),
            daemon=True,
            name=f"hrc-freeform-{token.lower()}",
        )
        self._worker.start()

    def _execute_robot_action(
        self,
        generation: int,
        token: str,
        distribution: Mapping[str, float],
        execution_id: str,
    ) -> None:
        try:
            execute = getattr(
                self.executor,
                "execute_freeform",
                None,
            )
            if not callable(execute):
                raise RuntimeError(
                    "hardware backend does not support freeform execution"
                )

            result = execute(
                token,
                execution_id=execution_id,
                config_digest=self.config.digest,
            )

        except Exception as exc:
            now = time.time()
            result = ExecutionResult(
                False,
                "backend_exception",
                str(exc),
                now,
                now,
                {
                    "action": token,
                    "execution_id": execution_id,
                },
            )

        with self._lock:
            if (
                generation != self._generation
                or self.phase != ROBOT_EXECUTING
            ):
                return

            self.last_execution = result

            if not result.success:
                if result.status == "execution_unknown":
                    self._pending_commit = {
                        "token": token,
                        "distribution": dict(distribution),
                        "execution": result,
                    }
                    self.phase = RECONCILIATION_REQUIRED
                    self.error = result.message
                    self._emit(
                        "robot_execution_ambiguous",
                        result.as_dict(),
                    )
                else:
                    self.phase = EXECUTION_FAILED
                    self.error = result.message
                    self._emit(
                        "robot_execution_failed",
                        result.as_dict(),
                    )
                return

            self._pending_commit = {
                "token": token,
                "distribution": dict(distribution),
                "execution": result,
            }

            try:
                self._emit(
                    "robot_execution_succeeded",
                    result.as_dict(),
                )
                self._commit_pending_robot_observation()

            except Exception as exc:
                self.phase = RECONCILIATION_REQUIRED
                self.error = (
                    "physical execution succeeded but learner/journal "
                    f"commit failed: {exc}"
                )
                self._emit(
                    "learner_commit_failed",
                    {
                        "execution_id": execution_id,
                        "error": str(exc),
                    },
                )

    def _commit_pending_robot_observation(self) -> None:
        pending = self._pending_commit
        if pending is None:
            raise SessionStateError(
                "no physical execution is pending learner commit"
            )

        result = pending["execution"]
        self._pending_commit = None

        try:
            self._commit_observation(
                str(pending["token"]),
                executed_by="robot",
                distribution=dict(pending["distribution"]),
                predicted=str(pending["token"]),
                execution=result,
                evidence={
                    "execution": result.as_dict(),
                },
            )
        except Exception:
            self._pending_commit = pending
            raise

    def _commit_observation(
        self,
        token: str,
        *,
        executed_by: str,
        distribution: Optional[Mapping[str, float]],
        predicted: Optional[str],
        execution: Optional[ExecutionResult] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        before = self.state
        after = self.domain.successor(before, token)

        if after is None:
            raise SessionStateError(
                f"action {token} is illegal in the current physical state"
            )

        agent_before = self.agent.snapshot()
        state_before = self.state
        completed_before = list(self.completed)
        decisions_before = list(self.decisions)
        pending_distribution_before = dict(self.pending_distribution)
        pending_prediction_before = self.pending_prediction
        phase_before = self.phase

        try:
            observation = PhysicalObservation(
                before,
                token,
                after,
            )

            observed = self.agent.observe(
                observation,
                precomputed_distribution=(
                    None
                    if distribution is None
                    else dict(distribution)
                ),
                precomputed_prediction=predicted,
            )

            self.state = after
            self.completed.append(token)

            record = {
                "step": len(self.completed) - 1,
                "action": token,
                "executed_by": executed_by,
                "predicted": predicted,
                "correct": (
                    predicted == token
                    if predicted is not None
                    else None
                ),
                "prediction_kind": (
                    "robot_control"
                    if executed_by == "robot"
                    else (
                        "robot_proposal"
                        if executed_by == "human_correction"
                        else (
                            "human_turn_shadow"
                            if distribution is not None
                            else None
                        )
                    )
                ),
                "distribution": dict(distribution or {}),
                "agent_step": observed.step,
                "execution": (
                    execution.as_dict()
                    if execution is not None
                    else None
                ),
                "evidence": dict(evidence or {}),
                "locations_after": dict(
                    self.domain.describe_state(after)
                ),
            }

            self.decisions.append(record)
            self.pending_distribution = {}
            self.pending_prediction = None

            if self.mode == OBSERVE or executed_by == "robot":
                self.phase = HUMAN_TURN

            self._emit(
                "learner_transition_committed",
                record,
            )

        except Exception:
            self._restore_agent(agent_before)
            self.state = state_before
            self.completed = completed_before
            self.decisions = decisions_before
            self.pending_distribution = pending_distribution_before
            self.pending_prediction = pending_prediction_before
            self.phase = phase_before
            raise

    def end_episode(self) -> Mapping[str, Any]:
        with self._lock:
            if self.phase in {
                IDLE,
                COMPLETE,
                ROBOT_EXECUTING,
                RECONCILIATION_REQUIRED,
                EXECUTION_FAILED,
                EMERGENCY_STOPPED,
            }:
                raise SessionStateError(
                    f"cannot end the episode while phase is {self.phase!r}"
                )

            if not self.completed:
                raise SessionStateError(
                    "observe at least one physical action before ending"
                )

            match = self.agent.end_demo()

            self.last_match = {
                "kind": match.kind,
                "recipe_id": match.recipe_id,
                "variant_id": match.variant_id,
                "jaccard": float(match.jaccard),
                "order_distance": float(match.order_distance),
            }

            robot_turns = sum(
                row["executed_by"]
                in {"robot", "human_correction"}
                for row in self.decisions
            )
            robot_hits = sum(
                row["executed_by"] == "robot"
                for row in self.decisions
            )
            corrections = sum(
                row["executed_by"] == "human_correction"
                for row in self.decisions
            )
            human_actions = sum(
                row["executed_by"]
                in {"human", "human_correction"}
                for row in self.decisions
            )

            summary = {
                "protocol": self.protocol,
                "episode_id": self.episode_id,
                "mode": self.mode,
                "actions": list(self.completed),
                "learner_match": dict(self.last_match),
                "summary": {
                    "steps": len(self.completed),
                    "robot_turn_count": robot_turns,
                    "robot_top_1_hits": robot_hits,
                    "robot_top_1": (
                        robot_hits / robot_turns
                        if robot_turns
                        else None
                    ),
                    "human_correction_count": corrections,
                    "human_action_count": human_actions,
                },
            }

            self.pending_distribution = {}
            self.pending_prediction = None
            self.phase = COMPLETE
            self._agent_snapshot = None

            self._emit(
                "episode_completed",
                summary,
            )

        return self.snapshot()

    def reconcile_robot_execution(
        self,
        *,
        execution_completed: bool,
        physical_scene_verified: bool,
    ) -> Mapping[str, Any]:
        with self._lock:
            if (
                self.phase != RECONCILIATION_REQUIRED
                or self._pending_commit is None
            ):
                raise SessionStateError(
                    "there is no execution awaiting reconciliation"
                )

            if not physical_scene_verified:
                raise SessionStateError(
                    "inspect the bridge ledger, robot pose, gripper, "
                    "box and destination first"
                )

            pending = self._pending_commit
            result = pending["execution"]

            if not execution_completed:
                self._pending_commit = None
                self.phase = EXECUTION_FAILED
                self.error = (
                    "operator verified ambiguous physical action did not "
                    "complete; abort/reset before continuing"
                )
                self._emit(
                    "execution_reconciled_not_completed",
                    {
                        "execution_id": result.metadata.get(
                            "execution_id"
                        ),
                        "operator_verified": True,
                    },
                )
            else:
                if not result.success:
                    result = ExecutionResult(
                        True,
                        "operator_reconciled_completed",
                        "operator verified physical completion",
                        result.started_at,
                        time.time(),
                        {
                            **dict(result.metadata),
                            "operator_reconciled": True,
                        },
                        result.monotonic_elapsed_s,
                    )
                    pending["execution"] = result

                self._emit(
                    "execution_reconciled_completed",
                    {
                        "execution_id": result.metadata.get(
                            "execution_id"
                        ),
                        "operator_verified": True,
                    },
                )
                self._commit_pending_robot_observation()

        return self.snapshot()

    def abort_episode(self) -> Mapping[str, Any]:
        with self._lock:
            if self.phase == ROBOT_EXECUTING:
                raise SessionStateError(
                    "cannot abort during robot motion; emergency-stop first"
                )

            if self.phase not in {IDLE, COMPLETE}:
                if self._agent_snapshot is not None:
                    self._restore_agent(self._agent_snapshot)

                aborted = self.episode_id
                self._generation += 1
                self._reset_runtime_state()

                self._emit(
                    "episode_aborted",
                    {
                        "episode_id": aborted,
                    },
                )
            else:
                self.phase = IDLE

        return self.snapshot()

    def emergency_stop(self) -> Mapping[str, Any]:
        result = dict(self.executor.emergency_stop())

        with self._lock:
            self._generation += 1
            self.phase = EMERGENCY_STOPPED
            self.error = str(
                result.get("error")
                or "emergency stop requested"
            )
            self._emit(
                "emergency_stop",
                result,
            )

        return self.snapshot()

    def shutdown(self, timeout_s: float = 20.0) -> bool:
        with self._lock:
            worker = self._worker
            executing = self.phase == ROBOT_EXECUTING

        if executing:
            self.emergency_stop()

        if worker is not None and worker.is_alive():
            worker.join(
                timeout=max(0.0, float(timeout_s))
            )

        return worker is None or not worker.is_alive()

    def _restore_agent(self, snapshot: Any) -> None:
        self.agent.restore_from(snapshot)

        self.agent.domain = self.domain

        maxent = getattr(self.agent, "maxent", None)
        if maxent is not None:
            maxent.domain = self.domain

    def _reset_runtime_state(self) -> None:
        self.phase = IDLE
        self.mode = None
        self.episode_id = None
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

    def _emit(
        self,
        event: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._event_index += 1

        row = {
            "event_index": self._event_index,
            "timestamp": time.time(),
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "monotonic_s": time.monotonic(),
            "event": event,
            "protocol": self.protocol,
            "phase": self.phase,
            "mode": self.mode,
            "recipe_id": None,
            "episode_id": self.episode_id,
            "payload": dict(payload),
        }

        if self.event_sink is not None:
            self.event_sink(row)

    def checkpoint_state(self) -> Mapping[str, Any]:
        with self._lock:
            pending = None

            if self._pending_commit is not None:
                pending = {
                    **self._pending_commit,
                    "execution": self._pending_commit[
                        "execution"
                    ].as_dict(),
                }

            return {
                "schema_version": 1,
                "protocol": self.protocol,
                "event_index": self._event_index,
                "phase": self.phase,
                "mode": self.mode,
                "episode_id": self.episode_id,
                "state": list(self.state),
                "completed": list(self.completed),
                "decisions": list(self.decisions),
                "pending_distribution": dict(
                    self.pending_distribution
                ),
                "pending_prediction": self.pending_prediction,
                "last_execution": (
                    self.last_execution.as_dict()
                    if self.last_execution
                    else None
                ),
                "last_match": (
                    dict(self.last_match)
                    if self.last_match
                    else None
                ),
                "error": self.error,
                "pending_commit": pending,
                "agent_start_snapshot": self._agent_snapshot,
            }

    def snapshot(self) -> Mapping[str, Any]:
        hardware = dict(self.executor.status())

        with self._lock:
            legal = self.domain.legal_actions(
                self.state,
                self.domain.actions,
            )

            known_recipes = self.observed_recipes
            variants = getattr(
                getattr(self.agent, "library", None),
                "known_variants",
                None,
            )
            known_variant_count = (
                len(variants())
                if callable(variants)
                else 0
            )

            state = {
                "protocol": self.protocol,
                "phase": self.phase,
                "mode": self.mode,
                "episode_id": self.episode_id,
                "recipe_id": None,
                "recipe_label": None,
                "trial_metadata": {},
                "completed": list(self.completed),
                "remaining_actions": [],
                "available_human_actions": list(legal),
                "pending_prediction": self.pending_prediction,
                "pending_prediction_in_recipe": (
                    self.pending_prediction in legal
                    if self.pending_prediction
                    else False
                ),
                "pending_distribution": dict(
                    self.pending_distribution
                ),
                "decisions": list(self.decisions),
                "last_execution": (
                    self.last_execution.as_dict()
                    if self.last_execution
                    else None
                ),
                "last_match": (
                    dict(self.last_match)
                    if self.last_match
                    else None
                ),
                "error": self.error,
                "known_recipes": list(known_recipes),
                "known_recipe_count": len(known_recipes),
                "known_variant_count": known_variant_count,
                "can_start_assist": bool(known_recipes),
                "can_end": bool(
                    self.completed
                    and self.phase
                    in {
                        HUMAN_TURN,
                        ROBOT_PROPOSAL,
                        CORRECTION_REQUIRED,
                    }
                ),
                "locations": dict(
                    self.domain.describe_state(self.state)
                ),
                "next_placement_slot": None,
                "schedule": None,
                "reconciliation_execution_id": (
                    None
                    if self._pending_commit is None
                    else self._pending_commit[
                        "execution"
                    ].metadata.get("execution_id")
                ),
                "session_worker_alive": (
                    self._worker is not None
                    and self._worker.is_alive()
                ),
                "config_digest": self.config.digest,
            }

        state["hardware"] = hardware
        return state
