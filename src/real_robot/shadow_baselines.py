"""Teacher-forced baseline scoring on the immutable physical event trace."""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Sequence

from src.baselines import BehaviorCloningAgent, LatestAgent, NoDecayAgent
from src.models import Settings

from .config import LabConfig
from .domain import PhysicalObservation, PhysicalTaskDomain


SHADOW_BASELINES = {
    "latest": LatestAgent,
    "no_decay": NoDecayAgent,
    "bc": BehaviorCloningAgent,
}


def _settings(config: LabConfig) -> Settings:
    known = {row.name for row in fields(Settings) if row.init}
    unknown = set(config.agent_settings) - known
    if unknown:
        raise ValueError(f"unknown agent_settings: {sorted(unknown)}")
    return Settings(**dict(config.agent_settings))


def _score_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    predicted = [row for row in rows if row.get("predicted") is not None]
    robot_turns = [
        row for row in predicted
        if row.get("executed_by") in {"robot", "human_correction"}
    ]
    return {
        "teacher_forced_predictions": len(predicted),
        "teacher_forced_top_1_hits": sum(bool(row.get("correct")) for row in predicted),
        "teacher_forced_top_1": (
            sum(bool(row.get("correct")) for row in predicted) / len(predicted)
            if predicted else None
        ),
        "scheduled_robot_turn_predictions": len(robot_turns),
        "scheduled_robot_turn_top_1_hits": sum(bool(row.get("correct")) for row in robot_turns),
        "scheduled_robot_turn_top_1": (
            sum(bool(row.get("correct")) for row in robot_turns) / len(robot_turns)
            if robot_turns else None
        ),
    }


def _group_scores(
    rows: Sequence[Mapping[str, Any]], key: str,
) -> Mapping[str, Mapping[str, Any]]:
    values = sorted({str(row[key]) for row in rows if row.get(key) not in {None, ""}})
    return {
        value: _score_rows([row for row in rows if row.get(key) == value])
        for value in values
    }


def score_shadow_baselines(
    config: LabConfig, episodes: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Replay observed actions without allowing a baseline to control motion.

    These are counterfactual teacher-forced scores on the realized trace. They
    are valid prediction comparators, not estimates of baseline-controlled task
    success or human effort.
    """
    output: dict[str, Any] = {}
    for name, agent_type in SHADOW_BASELINES.items():
        domain = PhysicalTaskDomain(config)
        agent = agent_type(settings=_settings(config), domain=domain)
        scored: list[dict[str, Any]] = []
        try:
            for episode in episodes:
                mode = str(episode.get("mode", ""))
                decisions = list(episode.get("decisions", ()))
                if mode == "observe":
                    agent.start_demo()
                state = domain.initial_state()
                prefix: list[str] = []
                for decision in decisions:
                    action = domain.canonical_action(str(decision["action"]))
                    after = domain.replay_transition(state, action)
                    distribution: Mapping[str, float] | None = None
                    predicted = None
                    if mode == "assist":
                        distribution = dict(agent.predict_actions(
                            tuple(prefix), state=state,
                            action_universe=domain.actions,
                        ))
                        ranked = agent.rank_actions(distribution, k=1) if distribution else []
                        predicted = ranked[0] if ranked else None
                        scored.append({
                            "trial_id": episode.get("trial_metadata", {}).get("trial_id"),
                            "condition": episode.get("trial_metadata", {}).get("condition"),
                            "preference_id": episode.get("trial_metadata", {}).get("preference_id"),
                            "recipe_id": episode.get("external_recipe_id"),
                            "step": decision.get("step"),
                            "executed_by": decision.get("executed_by"),
                            "actual": action,
                            "predicted": predicted,
                            "correct": predicted == action if predicted is not None else None,
                        })
                    agent.observe(
                        PhysicalObservation(state, action, after),
                        precomputed_distribution=(
                            None if distribution is None else dict(distribution)
                        ),
                        precomputed_prediction=predicted,
                    )
                    state = after
                    prefix.append(action)
                agent.end_demo()
            output[name] = {
                "status": "complete",
                "scope": "teacher_forced_counterfactual_on_realized_trace",
                "overall": _score_rows(scored),
                "by_condition": _group_scores(scored, "condition"),
                "by_preference": _group_scores(scored, "preference_id"),
                "by_recipe": _group_scores(scored, "recipe_id"),
            }
        except Exception as exc:
            output[name] = {
                "status": "failed",
                "scope": "teacher_forced_counterfactual_on_realized_trace",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return output
