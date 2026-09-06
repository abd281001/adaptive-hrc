"""Regressions for defects found auditing the first Overcooked/Burrito wrapper.

Each test pins one behaviour that was previously wrong in a way no reported
metric could reveal.
"""
import json
import tempfile
import unittest
from pathlib import Path

from adaptive_hrc_burrito import (
    RECIPES,
    CookingDomainAdapter,
    CookingPreferencePolicy,
    CookingTask,
    CookingTaskGraph,
    LadderSettings,
    applicable_preferences,
    generate_ladder,
)
from adaptive_hrc_burrito.catalog import (
    PREFERENCES,
    STRATA,
    realized_preferences,
)
from adaptive_hrc_burrito.evaluation import (
    VALIDATION_REQUIREMENTS,
    _aggregate,
    _frozen_probe,
    _human_action_load,
    _load_checkpoint,
    load_config,
)
from adaptive_hrc_burrito.ladder import ladder_audit
from adaptive_hrc_burrito.task_graph import is_preference_discriminating


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "full.json"


class CatalogRegressions(unittest.TestCase):
    def test_every_declared_preference_changes_some_recipe(self):
        """Nine 'TALENTS-like' preferences were declared; six were unreachable."""
        self.assertEqual(set(realized_preferences()), set(PREFERENCES))

    def test_goal_signature_separates_every_recipe(self):
        """A role multiset mapped eleven recipes onto four signatures."""
        domain = CookingDomainAdapter()
        signatures = {
            domain.goal_signature(recipe.action_tokens): recipe_id
            for recipe_id, recipe in RECIPES.items()
        }
        self.assertEqual(len(signatures), len(RECIPES))

    def test_compatibility_recipes_are_a_separate_stratum(self):
        """Wrapper-restored dynamics must not pool with native execution."""
        strata = {recipe.stratum for recipe in RECIPES.values()}
        self.assertEqual(strata, set(STRATA))
        for recipe in RECIPES.values():
            if recipe.compatibility_dynamics:
                self.assertEqual(recipe.stratum, "burrito_compat")

    def test_frozen_probes_use_the_same_reporting_strata_as_episodes(self):
        class PredictableAgent:
            def set_frozen(self, _value):
                pass

            def predict_actions(self, _completed, *, state, action_universe):
                del state
                return {action: 1.0 / len(action_universe) for action in action_universe}

            def rank_actions(self, distribution, *, k):
                return tuple(distribution)[:k]

        recipe_id = "burrito_steak_onion"
        task = CookingTask.create(
            recipe_id, applicable_preferences(recipe_id)[0],
        )
        probe = _frozen_probe(
            PredictableAgent(), CookingDomainAdapter(), task,
            metadata={}, probe_kind="unit",
        )

        self.assertEqual(probe["environment"], RECIPES[recipe_id].stratum)
        self.assertEqual(probe["environment"], "burrito_compat")

    def test_assembly_order_is_a_real_branch(self):
        """The three plate additions must be mutually independent."""
        recipe = RECIPES["burrito_steak_burrito"]
        by_token = recipe.action_by_token
        additions = ("PLATE_RICE", "PLATE_TORTILLA", "PLATE_STEAK")
        for token in additions:
            self.assertIn(token, by_token)
            for other in additions:
                if other != token:
                    self.assertNotIn(other, by_token[token].requires)

    def test_discrimination_reaches_the_second_half_of_episodes(self):
        """Bundled assembly concentrated every real choice on the opening move."""
        graph = CookingTaskGraph.create("burrito_steak_burrito")
        policy = CookingPreferencePolicy.create("wash_plates_early")
        completed, late = [], 0
        while not graph.is_complete(completed):
            legal = graph.frontier(completed)
            if len(completed) >= len(graph.actions) // 2 and (
                is_preference_discriminating(legal, graph)
            ):
                late += 1
            completed.append(policy.choose_action(legal, graph))
        self.assertGreater(late, 0)


class MetricRegressions(unittest.TestCase):
    def test_human_action_load_reports_its_protocol_floor(self):
        """Strict alternation floors the raw ratio near 0.5 regardless of skill."""
        rows = [{"recipe_steps": 9, "human_action_count": 5}]
        perfect = _human_action_load(rows)
        self.assertAlmostEqual(perfect["normalized_human_action_load"], 5 / 9)
        self.assertAlmostEqual(perfect["normalized_human_action_load_floor"], 5 / 9)
        self.assertEqual(perfect["human_action_load_excess"], 0.0)
        useless = _human_action_load(
            [{"recipe_steps": 9, "human_action_count": 9}]
        )
        self.assertEqual(useless["human_action_load_excess"], 1.0)

    def test_truncated_cells_are_not_pooled(self):
        """A cell that died at episode 2 used to be summed with complete cells."""
        def row(complete, hits, decisions):
            return {
                "mode": "assist", "arm": "full", "scenario": "homogeneous",
                "environment": "overcooked", "seed": 1, "recipe_id": "r",
                "preference": "p", "strategy": "s", "exposure_after_change": 0,
                "cell_complete": complete, "recipe_steps": 6,
                "human_action_count": 3, "robot_decision_count": decisions,
                "robot_top_1_hits": hits, "robot_top_k_hits": hits,
                "preference_discriminating_robot_decisions": decisions,
                "preference_discriminating_top_1_hits": hits,
                "teacher_forced_preference_discriminating_decisions": decisions,
                "teacher_forced_preference_discriminating_top_1_hits": hits,
                "nontrivial_choice_decision_count": decisions,
                "nontrivial_choice_top_1_hits": hits,
                "single_legal_action_decision_count": 0,
                "human_corrections": 0, "correction_free": True,
                "task_completed": True, "adaptation_id": None, "phase": 0,
                "lifecycle": "retain", "preference_changed": False,
            }
        summary = _aggregate([row(True, 1, 2), row(False, 2, 2)], [])
        self.assertEqual(summary["preference_discriminating_top_1"], 0.5)
        self.assertEqual(summary["excluded_incomplete_cell_episode_count"], 1)
        self.assertEqual(summary["pooled_episode_count"], 1)


class ConfigRegressions(unittest.TestCase):
    def test_config_must_declare_every_validation_requirement(self):
        """Subset checking silently disabled the preference-coverage check."""
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            set(config["validation_requirements"]), set(VALIDATION_REQUIREMENTS)
        )
        config["validation_requirements"] = [
            name for name in config["validation_requirements"]
            if name != "behavioral_preference_coverage"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reduced.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_frozen_panel_size_matches_the_real_pair_count(self):
        """The panel advertised 48 pairs the catalog could never supply."""
        config = load_config(CONFIG)
        actual = sum(
            len(applicable_preferences(recipe)) for recipe in config["recipe_ids"]
        )
        self.assertEqual(config["frozen_pairs"], actual)

    def test_resume_rejects_a_mismatched_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config.json").write_text('{"experiment": "other"}')
            self.assertIsNone(_load_checkpoint(run_dir, 1, "homogeneous", "full"))


class HoldoutRegressions(unittest.TestCase):
    def test_holdout_labels_isomorphic_transfers(self):
        """A source and target with one DAG relabel the axis, never generalise."""
        for seed in (1337, 2024, 7, 9001, 31415):
            tasks, audit = generate_ladder(
                seed=seed, scenario="holdout", recipe_ids=tuple(RECIPES),
                settings=LadderSettings(), return_audit=True,
            )
            targets = set(audit["holdout_target_recipe_ids"])
            isomorphic = set(audit["holdout_isomorphic_target_recipe_ids"])
            non_isomorphic = set(audit["holdout_non_isomorphic_target_recipe_ids"])
            self.assertEqual(isomorphic | non_isomorphic, targets)
            self.assertTrue(non_isomorphic, "holdout proved nothing but relabelling")
            sources = audit["schedule_metadata"]["holdout_source_recipe_ids"]
            for target in isomorphic:
                self.assertTrue(any(
                    RECIPES[target].structure_signature
                    == RECIPES[source].structure_signature
                    for source in sources
                ))

    def test_holdout_excludes_compatibility_recipes(self):
        _tasks, audit = generate_ladder(
            seed=1337, scenario="holdout", recipe_ids=tuple(RECIPES),
            settings=LadderSettings(), return_audit=True,
        )
        for recipe_id in audit["recipe_ids"]:
            self.assertFalse(RECIPES[recipe_id].compatibility_dynamics)


if __name__ == "__main__":
    unittest.main()


class StrategyGroundingRegressions(unittest.TestCase):
    def test_no_strategy_label_grounds_to_the_canonical_ordering(self):
        """`container_first` silently became a no-op for compat recipes."""
        from adaptive_hrc_burrito.ladder import strategy_grounding
        grounding = strategy_grounding(sorted(RECIPES))
        self.assertEqual(grounding["degenerate_strategy_groundings"], [])

    def test_substituted_and_aliased_groundings_are_reported(self):
        """A label that could not be grounded exactly must say so."""
        from adaptive_hrc_burrito.ladder import strategy_grounding
        grounding = strategy_grounding(sorted(RECIPES))
        for recipe_id, strategy, preference in grounding[
            "substituted_strategy_groundings"
        ]:
            self.assertNotIn(
                "wash_plates_early", applicable_preferences(recipe_id)
            )
            self.assertNotEqual(
                preference, applicable_preferences(recipe_id)[0]
            )
        _tasks, audit = generate_ladder(
            seed=1337, scenario="homogeneous", recipe_ids=tuple(RECIPES),
            settings=LadderSettings(), return_audit=True,
        )
        for key in (
            "strategy_grounding",
            "substituted_strategy_groundings",
            "aliased_strategy_pairs",
        ):
            self.assertIn(key, audit)

    def test_summary_counts_recipes_per_stratum(self):
        """The stratum rename left the Burrito recipe counter matching nothing."""
        rows = [{
            "mode": "assist", "arm": "full", "scenario": "homogeneous",
            "environment": stratum, "seed": 1, "recipe_id": f"r-{stratum}",
            "preference": "p", "strategy": "s", "exposure_after_change": 0,
            "cell_complete": True, "recipe_steps": 6, "human_action_count": 3,
            "robot_decision_count": 1, "robot_top_1_hits": 1,
            "robot_top_k_hits": 1, "task_completed": True,
            "preference_discriminating_robot_decisions": 1,
            "preference_discriminating_top_1_hits": 1,
            "teacher_forced_preference_discriminating_decisions": 1,
            "teacher_forced_preference_discriminating_top_1_hits": 1,
            "nontrivial_choice_decision_count": 1,
            "nontrivial_choice_top_1_hits": 1,
            "single_legal_action_decision_count": 0, "human_corrections": 0,
            "correction_free": True, "adaptation_id": None, "phase": 0,
            "lifecycle": "retain", "preference_changed": False,
        } for stratum in STRATA]
        summary = _aggregate(rows, [])
        self.assertEqual(
            summary["covered_recipe_count_by_stratum"],
            {stratum: 1 for stratum in STRATA},
        )


class RealizedBehaviourRegressions(unittest.TestCase):
    """`applicable_preferences` dedupes ABSTRACT orderings, but the learner
    observes the REALIZED one.  Physics used to collapse distinct preferences:
    rice and protein readiness, not preference, decided the assembly order."""

    @classmethod
    def setUpClass(cls):
        from adaptive_hrc_burrito import BurritoRuntime
        cls.runtime = BurritoRuntime.discover()

    def test_every_preference_realizes_a_distinct_behaviour(self):
        from adaptive_hrc_burrito.catalog import preference_order, get_preference
        from adaptive_hrc_burrito.physical import create_executor
        from adaptive_hrc_burrito.task_graph import (
            CookingPreferencePolicy, CookingTaskGraph,
        )
        for recipe_id, recipe in RECIPES.items():
            executor = create_executor(
                self.runtime, recipe_id, horizon=9000, seed=11
            )
            graph = CookingTaskGraph.create(recipe_id)
            realized = {}
            for preference in applicable_preferences(recipe_id):
                policy = CookingPreferencePolicy.create(preference)
                executor.reset()
                completed, waits = [], 0
                while not graph.is_complete(completed):
                    structural = graph.frontier(completed)
                    action = policy.choose_action(structural, graph)
                    while action not in graph.available_actions(
                        completed,
                        executor.legal_actions(
                            structural, actor_id=len(completed) % 2
                        ),
                    ):
                        executor.advance_environment()
                        waits += 1
                        self.assertLess(waits, 900, f"{recipe_id}/{preference}")
                    executor.execute(action, actor_id=len(completed) % 2)
                    completed.append(action)
                with self.subTest(recipe=recipe_id, preference=preference):
                    # The realized ordering must equal the abstract one, so the
                    # variant ids the learner commits describe real behaviour.
                    self.assertEqual(
                        tuple(completed),
                        preference_order(recipe, get_preference(preference)),
                    )
                    self.assertNotIn(
                        tuple(completed), realized,
                        f"collapses onto {realized.get(tuple(completed))}",
                    )
                realized[tuple(completed)] = preference

    def test_multi_order_recipe_is_structurally_novel(self):
        combo = RECIPES["burrito_combo"]
        self.assertEqual(combo.expected_deliveries, 2)
        for recipe_id, recipe in RECIPES.items():
            if recipe_id != "burrito_combo":
                self.assertNotEqual(
                    combo.structure_signature, recipe.structure_signature
                )

    def test_holdout_generalises_within_every_stratum(self):
        for seed in (1337, 2024, 7, 9001, 31415):
            _tasks, audit = generate_ladder(
                seed=seed, scenario="holdout", recipe_ids=tuple(RECIPES),
                settings=LadderSettings(), return_audit=True,
            )
            non_isomorphic = set(audit["holdout_non_isomorphic_target_recipe_ids"])
            for stratum in ("overcooked", "burrito_native"):
                with self.subTest(seed=seed, stratum=stratum):
                    self.assertTrue(
                        any(
                            RECIPES[recipe].stratum == stratum
                            for recipe in non_isomorphic
                        ),
                        f"{stratum} holdout transfer is only a relabelling",
                    )


class ProtocolAndFeatureRegressions(unittest.TestCase):
    def test_lead_actor_policy_is_declared_and_switchable(self):
        from adaptive_hrc_burrito.protocol import (
            COUNTERBALANCED, HUMAN_FIRST, LEAD_ACTOR_POLICIES,
        )
        self.assertEqual(LEAD_ACTOR_POLICIES, (HUMAN_FIRST, COUNTERBALANCED))
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["lead_actor_policy"], HUMAN_FIRST)

    def test_raw_state_features_do_not_order_recipes(self):
        """The recipe code is an enumeration index, not a magnitude."""
        from adaptive_hrc_burrito.domain import TaskState
        domain = CookingDomainAdapter()
        ids = list(RECIPES)
        rows = {index: TaskState(rid).encode() for index, rid in enumerate(ids)}
        raw, _mean, _scale = domain.build_features(
            rows, feature_mode="raw_state", normalize=False
        )
        # Each recipe contributes exactly one identity bit.
        for index in range(len(ids)):
            self.assertEqual(raw[index][: len(ids)].sum(), 1.0)

    def test_seed_macro_average_is_reported(self):
        rows = []
        for seed, hits in ((1, 1), (2, 0)):
            rows.extend([{
                "mode": "assist", "arm": "full", "scenario": "homogeneous",
                "environment": "overcooked", "seed": seed, "recipe_id": "r",
                "preference": "p", "strategy": "s", "exposure_after_change": 0,
                "cell_complete": True, "recipe_steps": 6,
                "human_action_count": 3, "robot_decision_count": 1,
                "robot_top_1_hits": hits, "robot_top_k_hits": hits,
                "preference_discriminating_robot_decisions": 1,
                "preference_discriminating_top_1_hits": hits,
                "teacher_forced_preference_discriminating_decisions": 1,
                "teacher_forced_preference_discriminating_top_1_hits": hits,
                "nontrivial_choice_decision_count": 1,
                "nontrivial_choice_top_1_hits": hits,
                "single_legal_action_decision_count": 0, "human_corrections": 0,
                "correction_free": True, "task_completed": True,
                "adaptation_id": None, "phase": 0, "lifecycle": "retain",
                "preference_changed": False, "lead_actor_policy": "human_first",
            }] * (1 if seed == 1 else 3))
        summary = _aggregate(rows, [])
        # Pooled weights seed 2 three times as heavily; the seed mean does not.
        self.assertAlmostEqual(summary["preference_discriminating_top_1"], 0.25)
        self.assertAlmostEqual(
            summary["by_seed"]["preference_discriminating_top_1_seed_mean"], 0.5
        )
        self.assertEqual(len(summary["by_seed"]["per_seed"]), 2)


class ParallelismRegressions(unittest.TestCase):
    def test_cells_are_independent_and_ordering_is_stable(self):
        from adaptive_hrc_burrito.evaluation import _cell_sort_key, _worker_count
        rows = [
            {"seed": 2, "scenario": "b", "arm": "z", "event_index": 0},
            {"seed": 1, "scenario": "a", "arm": "y", "event_index": 3},
            {"seed": 1, "scenario": "a", "arm": "y", "event_index": 1},
        ]
        # Worker completion order must not reach the artefacts.
        self.assertEqual(
            [row["event_index"] for row in sorted(rows, key=_cell_sort_key)],
            [1, 3, 0],
        )
        self.assertEqual(_worker_count({"workers": 4}, 165), 4)
        self.assertEqual(_worker_count({"workers": 99}, 3), 3)
        self.assertGreaterEqual(_worker_count({}, 165), 1)
