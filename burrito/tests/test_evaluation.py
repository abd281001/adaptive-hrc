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
from adaptive_hrc_burrito.evaluation import (
    ARM_NAMES,
    _aggregate,
    _build_agent,
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
        self.assertEqual(tuple(config["arms"]), ARM_NAMES)
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
            "unpinned": "UnpinnedAgent",
            "latest": "LatestAgent",
            "fixed": "FixedDecayAgent",
            "no_decay": "NoDecayAgent",
            "bc": "BehaviorCloningAgent",
            "ewc": "EwcAgent",
            "replay_bc": "ReplayBcAgent",
            "memory_oracle": "AdaptiveAgent",
        }
        for arm in ARM_NAMES:
            domain = CookingDomainAdapter()
            agent = _build_agent(arm, Settings(verbose=False), domain)
            self.assertEqual(type(agent).__name__, expected_types[arm])
            self.assertIs(agent.domain, domain)
        oracle = _build_agent(
            "memory_oracle", Settings(verbose=False), CookingDomainAdapter(),
        )
        self.assertEqual(oracle.replay.policy, "none")

    def test_primary_accuracy_excludes_forced_decisions(self):
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
            "preference_discriminating_robot_decisions": 1,
            "preference_discriminating_top_1_hits": 0,
            "nontrivial_choice_decision_count": 1,
            "nontrivial_choice_top_1_hits": 0,
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
        self.assertEqual(summary["robot_top_1"], 0.99)
        self.assertEqual(summary["preference_discriminating_top_1"], 0.0)
        self.assertEqual(summary["single_legal_action_fraction"], 0.99)
        self.assertEqual(summary["primary_metric"], "normalized_human_action_load")
        self.assertEqual(
            summary["primary_accuracy_metric"], "preference_discriminating_top_1",
        )


if __name__ == "__main__":
    unittest.main()
