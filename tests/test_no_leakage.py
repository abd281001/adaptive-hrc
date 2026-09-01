"""Static and behavioral checks for simulator-label leakage."""
from __future__ import annotations

import inspect
import re
import unittest

from src.adaptive_agent import AdaptiveAgent
from src.environment import recipe_builders
from src.evaluation import EvalSettings, assist_demo, build_task, observe_demo
from src.memory import make_variant_id
from src.models import Settings
from src.representations import observe_actions


_PREF_LABEL_PATTERN = re.compile(r"^pref(erence)?_(name|label|id_human|set)$", re.IGNORECASE)
_VARIANT_HANDLE_NAMES = {"variant_id"}


def _is_preference_label_param(name: str) -> bool:
    """Identify simulator preference labels, excluding exact-memory handles."""
    if name in _VARIANT_HANDLE_NAMES:
        return False
    return bool(_PREF_LABEL_PATTERN.match(name))


class NoPreferenceLabelInLearnerSignatures(unittest.TestCase):
    """Reject simulator-label parameters on learner-facing methods."""

    LEARNER_METHODS = (
        "start_demo",
        "end_demo",
        "observe",
        "predict_actions",
        "evaluate_actions",
        "refresh",
    )

    def test_no_preference_label_param(self):
        offenders = []
        for name in self.LEARNER_METHODS:
            method = getattr(AdaptiveAgent, name, None)
            if method is None:
                continue
            sig = inspect.signature(method)
            for pname in sig.parameters:
                if pname == "self":
                    continue
                if _is_preference_label_param(pname):
                    offenders.append((name, pname))
        self.assertEqual(
            offenders, [],
            f"preference labels reached learner-facing signatures: {offenders}",
        )


class NoPrecomputedTargetVariantAtDeploy(unittest.TestCase):
    """Ensure deploy targets enter memory only after their online episode."""

    def test_deploy_target_not_in_memory_before_episode(self):
        agent = AdaptiveAgent(settings=Settings(verbose=False))
        library = list(recipe_builders().items())
        rname, fn = library[0]
        train = build_task(rname, "default", fn)
        target = build_task(rname, "prep_first", fn)
        observe_demo(agent, train, None)
        target_h = make_variant_id(target.actions)
        recipe_id = next(iter(agent.library.variants))
        self.assertNotIn(target_h, agent.library.variants[recipe_id],
                         "deploy target preference variant was precomputed in memory")

class SemanticActionObservationTests(unittest.TestCase):
    def test_observations_and_memory_carry_actions_but_not_task_labels(self):
        agent = AdaptiveAgent(
            settings=Settings(verbose=False, irl_cold_steps=2, irl_warm_steps=1)
        )
        recipe_name, builder = next(iter(recipe_builders().items()))
        pair = build_task(recipe_name, "default", builder)
        observations = observe_actions(pair.actions)
        self.assertTrue(observations)
        self.assertEqual(
            tuple(observation.action for observation in observations),
            pair.actions,
        )
        self.assertTrue(all(not hasattr(obs, "recipe_name") for obs in observations))
        self.assertTrue(all(not hasattr(obs, "preference_name") for obs in observations))

        agent.start_demo()
        for obs in observations:
            agent.observe(obs)
        agent.end_demo()

        entry = next(iter(agent.replay.active_items()))
        self.assertTrue(entry.transitions)
        self.assertEqual(tuple(t[1] for t in entry.transitions), entry.ordering)

    def test_validated_preference_trace_has_no_observed_self_loops(self):
        agent = AdaptiveAgent(
            settings=Settings(verbose=False, irl_cold_steps=2, irl_warm_steps=1)
        )
        builder = recipe_builders()["tomato_soup"]
        pair = build_task("tomato_soup", "prep_loading_serving_cleanup", builder)
        agent.start_demo()
        for obs in observe_actions(pair.actions):
            agent.observe(obs)
        agent.end_demo()

        entry = max(agent.replay.active_items(), key=lambda e: e.last_seen_step)
        trajectories, dropped = agent._build_demos([entry])
        self.assertEqual(dropped, 0)
        self.assertFalse([
            (index, action)
            for index, ((before, action), (after, _next_action)) in enumerate(zip(trajectories[0], trajectories[0][1:]))
            if action != "stop" and before == after
        ])


class AssistiveEpisodeLeakageTests(unittest.TestCase):
    """Verify the deployed assistance path completes without test-label input."""

    def _train(self, agent):
        rname, fn = next(iter(recipe_builders().items()))
        a = build_task(rname, "default", fn)
        b = build_task(rname, "prep_first", fn)
        recipe_ids = {}
        observe_demo(agent, a, recipe_ids)
        return rname, a, b, recipe_ids

    def test_assist_episode_emits_robot_turns(self):
        full = AdaptiveAgent(settings=Settings(verbose=False))
        rname, a, b, recipe_ids = self._train(full)
        metrics = assist_demo(
            full,
            b,
            recipe_ids,
            config=EvalSettings(top_k=3),
            observed_pairs={a.label},
            observed_recipes={rname},
        )
        robot_turns = [turn for turn in metrics["_turn_records"] if turn["turn_kind"] == "robot"]
        self.assertEqual(len(robot_turns), int(metrics["n_steps"]))
        self.assertGreater(len(robot_turns), 0)

if __name__ == "__main__":
    unittest.main()
