"""Phase 4: tests for the calibrated online posterior."""
from __future__ import annotations

import unittest

from src import experiments
from src.adaptive_agent import AdaptiveHRCAgent
from src.environment import gen
from src.models import Config
from src.posterior import (
    MemoryPrior,
    OnlinePreferencePosterior,
    PreferencePrototypeLearner,
    PosteriorWeights,
    RecipePrototypeLearner,
)


class PosteriorUnitTests(unittest.TestCase):
    def test_empty_state_is_uniform(self):
        post = OnlinePreferencePosterior()
        recipe = RecipePrototypeLearner()
        pref = PreferencePrototypeLearner()
        joint = post.update(
            prefix_tokens=[],
            prefix_roles=[],
            recipe_protos=recipe,
            pref_protos=pref,
            memory_state_for_recipe=lambda r: "absent",
        )
        # No prototypes -> single sentinel hypothesis.
        self.assertGreaterEqual(len(joint), 1)
        self.assertAlmostEqual(sum(joint.values()), 1.0, places=5)

    def test_active_recipe_outranks_pruned(self):
        recipe = RecipePrototypeLearner()
        recipe.update_from_demo("R_active",  ["a", "b", "c"])
        recipe.update_from_demo("R_pruned",  ["a", "b", "c"])  # identical action set
        pref = PreferencePrototypeLearner()
        post = OnlinePreferencePosterior()
        post.update(
            prefix_tokens=["a", "b"],
            prefix_roles=["retrieve_ingredient", "prepare_ingredient"],
            recipe_protos=recipe,
            pref_protos=pref,
            memory_state_for_recipe=lambda r: "active" if r == "R_active" else "pruned",
        )
        m = post.marginal_recipe()
        self.assertGreater(m["R_active"], m["R_pruned"])

    def test_pruned_memory_prior_decays_with_reuse_gap(self):
        weights = PosteriorWeights(pruned_prior_initial=0.5, pruned_prior_half_life=10.0)
        recent = OnlinePreferencePosterior._log_p_memory(
            MemoryPrior("pruned", sessions_since_pruned=1),
            weights,
        )
        stale = OnlinePreferencePosterior._log_p_memory(
            MemoryPrior("pruned", sessions_since_pruned=50),
            weights,
        )
        self.assertGreater(recent, stale)

    def test_normalized_factor_combination_keeps_joint_distribution_valid(self):
        recipe = RecipePrototypeLearner()
        recipe.update_from_demo("R1", ["a", "b", "c"])
        recipe.update_from_demo("R2", ["x", "y", "z"])
        pref = PreferencePrototypeLearner()
        pref.update_from_roles(["retrieve_ingredient", "prepare_ingredient"], recipe_id="R1")
        pref.update_from_roles(["stage_serving_vessel", "serve"], recipe_id="R2")
        post = OnlinePreferencePosterior(
            PosteriorWeights(
                alpha_recipe=1.0,
                alpha_pref=1.0,
                alpha_memory=1.0,
                temperature_recipe=0.5,
                temperature_pref=2.0,
                temperature_memory=1.5,
            )
        )
        joint = post.update(
            prefix_tokens=["a", "b"],
            prefix_roles=["retrieve_ingredient", "prepare_ingredient"],
            recipe_protos=recipe,
            pref_protos=pref,
            memory_state_for_recipe=lambda r: MemoryPrior("active", active_weight=1.0)
            if r == "R1" else MemoryPrior("pruned", sessions_since_pruned=20),
        )
        self.assertAlmostEqual(sum(joint.values()), 1.0, places=6)
        for p in joint.values():
            self.assertGreaterEqual(p, 0.0)

    def test_normalized_entropy_in_unit_interval(self):
        post = OnlinePreferencePosterior()
        recipe = RecipePrototypeLearner()
        recipe.update_from_demo("R1", ["a", "b"])
        recipe.update_from_demo("R2", ["a", "c"])
        pref = PreferencePrototypeLearner()
        post.update(
            prefix_tokens=["a"],
            prefix_roles=["retrieve_ingredient"],
            recipe_protos=recipe,
            pref_protos=pref,
            memory_state_for_recipe=lambda r: "active",
        )
        h = post.normalized_entropy()
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, 1.0 + 1e-9)
        c = post.confidence()
        self.assertAlmostEqual(c + h, 1.0, places=6)

    def test_unseen_sentinel_does_not_pollute_argmax(self):
        post = OnlinePreferencePosterior()
        recipe = RecipePrototypeLearner()
        recipe.update_from_demo("R1", ["a", "b"])
        pref = PreferencePrototypeLearner()
        post.update(
            prefix_tokens=["a", "b"],
            prefix_roles=["retrieve_ingredient", "prepare_ingredient"],
            recipe_protos=recipe,
            pref_protos=pref,
            memory_state_for_recipe=lambda r: "active",
        )
        # Argmax should be the real recipe, not a sentinel.
        self.assertEqual(post.argmax_recipe(), "R1")


class PosteriorAgentIntegrationTests(unittest.TestCase):
    def test_posterior_resets_at_session_boundary(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False))
        library = list(gen.recipe_library().items())
        rname, fn = library[0]
        a = experiments.materialize_pair(rname, "identity", fn)
        experiments.observe_episode(agent, a)
        # The posterior may have been touched during refresh paths; force a
        # second observation demo, then verify the posterior is empty after
        # session commit.
        b = experiments.materialize_pair(rname, "identity", fn)
        experiments.observe_episode(agent, b)
        # After end_demo() the agent resets the posterior.
        self.assertEqual(agent.posterior.joint(), {})

    def test_inferred_recipe_via_posterior_during_online(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False))
        library = list(gen.recipe_library().items())
        rname, fn = library[0]
        a = experiments.materialize_pair(rname, "identity", fn)
        name_to_rid = {}
        experiments.observe_episode(agent, a, name_to_rid)
        _metrics, steps = experiments.assist_episode(
            agent,
            a,
            name_to_rid,
            run_config=experiments.RunConfig(topk=3),
        )
        # During the online episode, posterior should have identified the
        # recipe (we capture inferred_recipe at each step).
        recs = [s.get("inferred_recipe") for s in steps if s.get("inferred_recipe") is not None]
        self.assertGreater(len(recs), 0)

    def test_posterior_does_not_mutate_under_frozen(self):
        agent = AdaptiveHRCAgent(cfg=Config(verbose=False))
        library = list(gen.recipe_library().items())
        rname, fn = library[0]
        a = experiments.materialize_pair(rname, "identity", fn)
        experiments.observe_episode(agent, a)
        # frozen() snapshot includes the posterior; any mutation during
        # frozen would assert on exit. Run a frozen prediction loop.
        with agent.frozen():
            tokens = agent._tokens_from_action_labels(list(a.actions))
            for k in range(len(tokens)):
                _ = agent.predict_next_tokens(tokens[:k])


if __name__ == "__main__":
    unittest.main()
