"""Phase 0B static leakage tests + Phase 6 behavioral leakage tests.

Static checks prevent preference labels, modifier identity, or precomputed
target variants from entering learner-facing call paths.  Behavioral checks
exercise the deployed assistance path without exposing test labels.
"""
from __future__ import annotations

import inspect
import re
import unittest

from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.evaluation import EvaluationConfig, assist_episode, make_agent, materialize_pair, observe_episode
from src.memory import variant_hash
from src.models import Config


_PREF_LABEL_PATTERN = re.compile(r"^pref(erence)?_(name|label|id_human|set)$", re.IGNORECASE)
_VARIANT_HANDLE_NAMES = {"variant_hash"}


def _is_preference_label_param(name: str) -> bool:
    """Return True only for parameter names that look like SIMULATOR-side labels.

    `variant_hash` is an exact-memory handle and is legitimately exposed on
    agent APIs. What is forbidden is the simulator-side preference name.
    """
    if name in _VARIANT_HANDLE_NAMES:
        return False
    return bool(_PREF_LABEL_PATTERN.match(name))


class NoPreferenceLabelInLearnerSignatures(unittest.TestCase):
    """test_no_preference_label_in_learner_signatures.

    Inspect AdaptiveHRCAgent's public, learner-facing methods. None of their
    parameter names may match the simulator-side preference label pattern.
    """

    LEARNER_METHODS = (
        "start_demo",
        "end_demo",
        "observe_observation",
        "predict_next_tokens",
        "predict_next",
        "evaluate_autonomous_tokens",
        "refresh_model_from_memory",
    )

    def test_no_preference_label_param(self):
        offenders = []
        for name in self.LEARNER_METHODS:
            method = getattr(AdaptiveHRCAgent, name, None)
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
    """test_no_precomputed_target_variants_at_deploy.

    At the start of an online assistive episode, the agent's variant memory
    must NOT already contain the (recipe, preference) tuple that the deploy
    target represents — otherwise the apparent "online assistance" is just
    replay of a precomputed variant.

    For the Phase 0A baseline regime we relax this to a different rule that
    captures the same intent: the test target's (recipe_id, variant_hash) must
    not appear in memory until the test demo is actually run.
    """

    def test_deploy_target_not_in_memory_before_episode(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False))
        library = list(gen.recipe_library().items())
        rname, fn = library[0]
        train = materialize_pair(rname, "identity", fn)
        target = materialize_pair(rname, "p1_mise_en_place", fn)
        observe_episode(agent, train, None)
        # Now the agent has seen rname/identity. Verify rname/wash_asap is NOT
        # in memory yet — the exact deploy target.
        target_tokens = agent._tokens_from_action_labels(list(target.actions))
        target_h = variant_hash(target_tokens)
        # Find the rid (one cluster).
        rid = next(iter(agent.memory.variants))
        self.assertNotIn(target_h, agent.memory.variants[rid],
                         "deploy target preference variant was precomputed in memory")




class ObservedTransitionTraceNoLeakage(unittest.TestCase):
    def test_observations_and_codebook_do_not_carry_action_strings(self):
        from src.representations import observations_from_actions

        agent = AdaptiveHRCAgent(
            cfg=Config(verbose=False, maxent_iters_cold=2, maxent_iters_warm=1)
        )
        recipe_name, builder = next(iter(gen.recipe_library().items()))
        pair = materialize_pair(recipe_name, "identity", builder)
        observations = observations_from_actions(pair.actions)
        self.assertTrue(observations)
        self.assertTrue(all(not hasattr(obs, "action_str") for obs in observations))

        agent.start_demo()
        for obs in observations:
            agent.observe_observation(obs)
        agent.end_demo()

        codebook = agent.save_codebook()
        self.assertNotIn("token_to_action_string", codebook)
        self.assertEqual(set(codebook), {"token_to_vector", "vector_to_token", "n_tokens"})
        entry = next(iter(agent.decay.active_entries()))
        self.assertTrue(entry.transitions)
        self.assertEqual(tuple(t[1] for t in entry.transitions), entry.ordering)

    def test_validated_preference_trace_has_no_observed_self_loops(self):
        from src.representations import observations_from_actions

        agent = AdaptiveHRCAgent(
            cfg=Config(verbose=False, maxent_iters_cold=2, maxent_iters_warm=1)
        )
        builder = gen.recipe_library()["tomato_soup"]
        pair = materialize_pair("tomato_soup", "p12_multi_stage_reorganization", builder)
        agent.start_demo()
        for obs in observations_from_actions(pair.actions):
            agent.observe_observation(obs)
        agent.end_demo()

        entry = max(agent.decay.active_entries(), key=lambda e: e.last_seen_step)
        trajectories, dropped = agent._build_trajectories([entry])
        self.assertEqual(dropped, 0)
        self.assertFalse([
            (idx, action)
            for idx, ((before, action), (after, _next_action)) in enumerate(zip(trajectories[0], trajectories[0][1:]))
            if action != "stop" and before == after
        ])


class AssistiveEpisodeLeakageTests(unittest.TestCase):
    """Verify the deployed assistance path completes without test-label input."""

    def _train(self, agent):
        from src.environment import gen
        library = list(gen.recipe_library().items())[:1]
        rname, fn = library[0]
        a = materialize_pair(rname, "identity", fn)
        b = materialize_pair(rname, "p1_mise_en_place", fn)
        name_to_rid = {}
        observe_episode(agent, a, name_to_rid)
        return rname, a, b, name_to_rid

    def test_assist_episode_emits_robot_turns(self):
        full = AdaptiveHRCAgent(cfg=Config(verbose=False))
        rname, a, b, name_to_rid = self._train(full)
        metrics = assist_episode(
            full,
            b,
            name_to_rid,
            config=EvaluationConfig(topk=3),
            observed_pairs={a.label},
            observed_recipes={rname},
        )
        # The deployed system must complete and emit robot-turn records.
        robot_turns = [turn for turn in metrics["_turn_records"] if turn["turn_kind"] == "robot"]
        self.assertEqual(len(robot_turns), int(metrics["n_steps"]))
        self.assertGreater(len(robot_turns), 0)






if __name__ == "__main__":
    unittest.main()
