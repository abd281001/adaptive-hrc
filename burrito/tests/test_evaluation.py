import json
from pathlib import Path
import tempfile
import unittest

from adaptive_hrc_burrito import (
    BURRITO_RECIPE_IDS,
    OVERCOOKED_RECIPE_IDS,
    RECIPES,
    generate_ladder,
    ladder_audit,
)
from adaptive_hrc_burrito.domain import CookingDomainAdapter
from adaptive_hrc_burrito.protocol import CookingObservation
from adaptive_hrc_burrito.task_graph import (
    CookingPreferencePolicy,
    CookingTaskGraph,
)
from adaptive_hrc_burrito.evaluation import (
    ARM_NAMES,
    NULL_ARM_NAMES,
    _aggregate,
    _arm_dir,
    _build_agent,
    _cell_checkpoint_path,
    cell_execution_order,
    load_config,
)
from src.evaluation import PAPER_SEEDS
from src.models import Settings


BURRITO_ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_only_full_config_matches_adaptive_hrc_contract(self):
        configs = sorted(path.name for path in (BURRITO_ROOT / "configs").glob("*.json"))
        self.assertEqual(configs, ["full.json"])
        config = load_config(BURRITO_ROOT / "configs" / "full.json")
        self.assertNotIn("require_clean_git", config)
        self.assertEqual(tuple(config["arms"]), ARM_NAMES + NULL_ARM_NAMES)
        # Derived, not restated: the config is required to run the same
        # paired grid as the symbolic evaluation, and restating it here is
        # how the two fell out of step at five seeds against eight.
        self.assertEqual(config["seeds"], [int(seed) for seed in PAPER_SEEDS])
        self.assertEqual(
            config["scenarios"], ["homogeneous", "heterogeneous", "holdout"],
        )
        self.assertEqual(
            set(config["recipe_ids"]),
            set(OVERCOOKED_RECIPE_IDS) | set(BURRITO_RECIPE_IDS),
        )
        self.assertEqual((len(OVERCOOKED_RECIPE_IDS), len(BURRITO_RECIPE_IDS)), (7, 5))
        self.assertTrue(all(len(recipe.ingredients) >= 2 for recipe in RECIPES.values()))
        self.assertEqual(config["settings"]["irl_cold_steps"], 100)
        self.assertEqual(config["settings"]["irl_warm_steps"], 40)
        self.assertEqual(config["settings"]["initial_grace"], 50)
        self.assertEqual(config["settings"]["min_grace"], 6)

    def test_old_or_reduced_config_is_rejected(self):
        old = {
            "schema_version": 3,
            "experiment": "smoke",
            "seeds": [1],
            "scenarios": ["homogeneous"],
            "recipe_ids": ["overcooked_onion_onion"],
            "arms": ["full"],
            "ladder": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema must be 4"):
                load_config(path)

    def test_paper_seed_schedule_lengths_and_longitudinal_invariants(self):
        for seed in PAPER_SEEDS:
            homogeneous = generate_ladder(seed=seed, scenario="homogeneous")
            heterogeneous = generate_ladder(seed=seed, scenario="heterogeneous")
            holdout = generate_ladder(seed=seed, scenario="holdout")
            self.assertEqual(len(homogeneous), 210)
            self.assertEqual(len(holdout), 630)
            self.assertGreater(len(heterogeneous), 210)
            for tasks in (homogeneous, heterogeneous, holdout):
                audit = ladder_audit(tasks)
                self.assertTrue(audit["scenario_invariants_passed"])
                self.assertGreater(audit["post_update_recurrences"], 0)
                self.assertTrue(all(
                    len(RECIPES[task.recipe_id].ingredients) >= 2 for task in tasks
                ))
            self.assertGreaterEqual(
                ladder_audit(heterogeneous)["max_active_preferences_per_recipe_phase"],
                2,
            )
            self.assertEqual(
                set(ladder_audit(holdout)["holdout_target_environments"]),
                {"overcooked", "burrito_native"},
            )

    def test_baseline_roster_uses_adaptive_hrc_implementations(self):
        expected_types = {
            "full": "AdaptiveAgent",
            "frozen": "FrozenAgent",
            "offline_default": "FrozenAgent",
            "offline_all": "FrozenAgent",
            "unpinned": "UnpinnedAgent",
            "latest": "LatestAgent",
            "fixed": "FixedDecayAgent",
            "no_decay": "NoDecayAgent",
            "bc": "BehaviorCloningAgent",
            "ewc": "EwcAgent",
            "replay_bc": "ReplayBcAgent",
            "memory_oracle": "AdaptiveAgent",
            "canonical_order": "CanonicalOrderAgent",
        }
        for arm in ARM_NAMES + NULL_ARM_NAMES:
            domain = CookingDomainAdapter()
            agent = _build_agent(arm, Settings(verbose=False), domain)
            self.assertEqual(type(agent).__name__, expected_types[arm])
            self.assertIs(agent.domain, domain)
        oracle = _build_agent(
            "memory_oracle", Settings(verbose=False), CookingDomainAdapter(),
        )
        self.assertEqual(oracle.replay.policy, "none")

    def test_primary_accuracy_is_the_symbolic_parity_metric(self):
        """The headline must be a metric the symbolic evaluation also reports.

        ``preference_discriminating_top_1`` has no counterpart in
        ``src.evaluation`` -- the word does not appear there -- so promoting it
        left the replication with no number that could be placed beside the
        symbolic result.  ``teacher_forced_top_1`` is what both sides report.
        """
        row = {
            "mode": "assist",
            "arm": "full",
            "scenario": "homogeneous",
            "environment": "overcooked",
            "recipe_id": "overcooked_onion_tomato",
            "preference": "tomato_early",
            "strategy": "homogeneous[support_branch_first]",
            "task_completed": True,
            "exposure_after_change": 0,
            "robot_decision_count": 100,
            "human_action_count": 2,
            "recipe_steps": 102,
            "robot_top_1_hits": 99,
            "teacher_forced_decision_count": 200,
            "teacher_forced_top_1_hits": 180,
            # 1 of the 100 robot turns is a real choice point under the recipe's
            # whole preference set; none of them is still ambiguous once the
            # prefix is taken into account.
            "preference_discriminating_robot_decisions": 1,
            "preference_discriminating_top_1_hits": 0,
            "prefix_conditioned_robot_decisions": 0,
            "prefix_conditioned_top_1_hits": 0,
            "single_legal_action_decision_count": 99,
            "human_corrections": 1,
            "correction_free": False,
            "corrections_per_robot_decision": 0.01,
            "adaptation_id": None,
            "seed": 1337,
            "cell_complete": True,
            "lead_actor_policy": "human_first",
        }
        summary = _aggregate([row], [])
        self.assertEqual(summary["primary_accuracy_metric"], "teacher_forced_top_1")
        self.assertEqual(
            summary["cross_environment_parity_metric"], "teacher_forced_top_1",
        )
        self.assertEqual(summary["teacher_forced_top_1"], 0.9)
        self.assertEqual(summary["robot_top_1"], 0.99)
        self.assertEqual(summary["preference_discriminating_top_1"], 0.0)
        self.assertEqual(summary["single_legal_action_fraction"], 0.99)
        # An all-forced, all-settled cell reports no conditioned accuracy at
        # all rather than a spuriously perfect one.
        self.assertIsNone(summary["prefix_conditioned_top_1"])
        self.assertEqual(summary["prefix_conditioned_robot_decisions"], 0)

    def test_zero_learning_reference_never_learns(self):
        """The null arm must be structurally, not nominally, zero-learning.

        It carries the delegate's replay bookkeeping so routing, open-set
        separation and the active-only audits stay comparable across arms, but
        no model may be fitted and no learned component consulted -- otherwise
        it is not a reference for "what does a fixed rule score here".
        """
        from adaptive_hrc_burrito.evaluation import _model_storage_metrics

        domain = CookingDomainAdapter()
        agent = _build_agent("canonical_order", Settings(verbose=False), domain)
        graph = CookingTaskGraph.create("overcooked_onion_onion_tomato")
        legal = graph.frontier([])
        state = domain.state_from_completed("overcooked_onion_onion_tomato", [])
        before = agent.predict_actions((), state=state, action_universe=legal)

        # Feed it a full episode of the *non*-canonical behaviour.  A learner
        # would move; this must not.
        policy = CookingPreferencePolicy.create("wash_plates_early")
        agent.start_demo()
        completed: list[str] = []
        while not graph.is_complete(completed):
            frontier = graph.frontier(completed)
            action = policy.choose_action(frontier, graph)
            observation = CookingObservation(
                domain.state_from_completed(graph.recipe.recipe_id, completed),
                action,
                domain.state_from_completed(
                    graph.recipe.recipe_id, completed + [action],
                ),
            )
            agent.observe(observation, ground_truth_recipe=graph.recipe.recipe_id)
            completed.append(action)
        agent.end_demo()

        after = agent.predict_actions((), state=state, action_universe=legal)
        self.assertEqual(before, after)
        self.assertEqual(
            max(after, key=after.get), graph.recipe.action_tokens[0],
        )
        self.assertEqual(
            _model_storage_metrics(agent)["learner_model_structure"], "unfitted",
        )
        stats = agent.policy_stats()
        self.assertFalse(stats["semantic_fallback_used"])
        self.assertFalse(stats["latent_strategy_used"])
        # One adapter, so a frozen probe cannot leave the two halves pointing
        # at different copies.
        self.assertIs(agent.domain, agent._delegate.domain)

    def test_dropped_metrics_stay_out_of_the_summary(self):
        """Three reported metrics were at a structural ceiling or redundant.

        ``robot_top_k`` at k=3 sits on a frontier of at most four options;
        ``nontrivial_choice_top_1`` tracked
        ``preference_discriminating_top_1`` to within a thousandth; and the
        normalized-human-action-load family is an affine restatement of
        ``robot_top_1`` bounded below by the alternation floor.  The raw
        per-episode counts are kept so any of them can be recomputed.
        """
        summary = _aggregate([], [])
        for name in (
            "robot_top_k",
            "nontrivial_choice_top_1",
            "normalized_human_action_load",
            "normalized_human_action_load_floor",
            "human_action_load_excess",
            "primary_metric",
        ):
            self.assertNotIn(name, summary)


if __name__ == "__main__":
    unittest.main()


class ArmMajorCellSchedulingTests(unittest.TestCase):
    """One arm finishes the whole grid before the next starts."""

    CONFIG = {
        "arms": ["full", "no_decay", "bc"],
        "seeds": [1337, 2024],
        "scenarios": ["homogeneous", "holdout"],
    }

    def test_cells_are_ordered_arm_major(self):
        order = cell_execution_order(self.CONFIG)
        self.assertEqual(len(order), 3 * 2 * 2)
        arms = [arm for _seed, _scenario, arm in order]
        # Each arm appears as one uninterrupted block, in roster order.
        self.assertEqual(
            [arm for index, arm in enumerate(arms)
             if index == 0 or arms[index - 1] != arm],
            self.CONFIG["arms"],
        )
        for arm in self.CONFIG["arms"]:
            block = {
                (seed, scenario)
                for seed, scenario, name in order if name == arm
            }
            self.assertEqual(block, {
                (seed, scenario)
                for seed in self.CONFIG["seeds"]
                for scenario in self.CONFIG["scenarios"]
            })

    def test_every_cell_is_scheduled_exactly_once(self):
        order = cell_execution_order(self.CONFIG)
        self.assertEqual(len(set(order)), len(order))

    def test_each_arm_keeps_its_cells_in_its_own_folder(self):
        root = Path("/tmp/does-not-need-to-exist")
        first = _cell_checkpoint_path(root, 1337, "homogeneous", "full")
        second = _cell_checkpoint_path(root, 1337, "homogeneous", "bc")
        self.assertEqual(first.parent, _arm_dir(root, "full"))
        self.assertEqual(second.parent, _arm_dir(root, "bc"))
        self.assertNotEqual(first.parent, second.parent)
        self.assertEqual(first.name, "1337__homogeneous.json")

    def test_an_arm_name_that_escapes_the_run_directory_is_rejected(self):
        for bad in ("../escape", "with/slash", ".hidden", ""):
            with self.subTest(arm=bad):
                with self.assertRaises(ValueError):
                    _arm_dir(Path("/tmp/run"), bad)
