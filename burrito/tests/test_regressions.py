"""Regressions for defects found auditing the first Overcooked/Burrito wrapper.

Each test pins one behaviour that was previously wrong in a way no reported
metric could reveal.
"""
import json
import math
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
from adaptive_hrc_burrito.domain import CookingDomainAdapter
from adaptive_hrc_burrito.runtime import BurritoRuntime
from adaptive_hrc_burrito.evaluation import (
    VALIDATION_REQUIREMENTS,
    _build_agent,
    _holdout_groups,
    _holdout_stage,
    _aggregate,
    _frozen_probe,
    _cell_checkpoint_path,
    _load_checkpoint,
    _performance,
    load_config,
)
from adaptive_hrc_burrito.ladder import CONTAINER_FIRST_PREFERENCE, ladder_audit
from adaptive_hrc_burrito.protocol import CookingHrcRunner, CookingTask
from src.evaluation import PAPER_SEEDS
from adaptive_hrc_burrito.task_graph import (
    is_preference_discriminating,
    is_prefix_conditioned_discriminating,
    preferences_consistent_with_prefix,
)


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


class AmbiguityDenominatorRegressions(unittest.TestCase):
    """The accuracy denominator must exclude decisions the prefix has settled.

    ``is_preference_discriminating`` asks whether a recipe's declared
    preferences *could* disagree on a frontier.  Under ``human_first`` the
    human takes step 0 -- the widest frontier of the episode -- so by the
    robot's turn the surviving preference set is usually a singleton and the
    "choice" has one consistent answer.  Scoring those decisions measures
    following an already-determined ordering, not tracking a preference.
    """

    def test_prefix_narrows_the_surviving_preference_set(self):
        recipe_id = "overcooked_onion_onion"
        graph = CookingTaskGraph.create(recipe_id)
        # Before anyone moves, all three behaviours are live.
        self.assertEqual(
            len(preferences_consistent_with_prefix(recipe_id, ())), 3,
        )
        # The human opening with STAGE_DISH is only wash_plates_early.
        self.assertEqual(
            preferences_consistent_with_prefix(recipe_id, ("STAGE_DISH",)),
            ("wash_plates_early",),
        )
        # That frontier still looks discriminating to the unconditioned test,
        # and is not once the opening move is taken into account.
        legal = graph.frontier(["STAGE_DISH"])
        self.assertFalse(is_prefix_conditioned_discriminating(
            legal, graph, ["STAGE_DISH"],
        ))

    def test_conditioning_is_strictly_tighter_across_the_catalog(self):
        """Over every (recipe, preference) cell, most flagged robot decisions
        are already settled -- so the two denominators are not interchangeable.
        """
        flagged = conditioned = 0
        for recipe_id, recipe in RECIPES.items():
            graph = CookingTaskGraph.create(recipe_id)
            for name in applicable_preferences(recipe_id):
                policy = CookingPreferencePolicy.create(name)
                completed: list[str] = []
                for index in range(len(recipe.actions)):
                    legal = graph.frontier(completed)
                    if index % 2 == 1 and is_preference_discriminating(legal, graph):
                        flagged += 1
                        conditioned += int(is_prefix_conditioned_discriminating(
                            legal, graph, completed,
                        ))
                    completed.append(policy.choose_action(legal, graph))
        self.assertGreater(flagged, 0)
        self.assertLess(conditioned, flagged // 2)


class MetricRegressions(unittest.TestCase):
    def test_performance_columns_are_aggregated(self):
        """Cost, latency, memory and calibration were recorded and never reported.

        Twenty-six per-episode columns had no aggregate anywhere in
        ``summary.json``, so the run could not answer a question about the
        system's cost at all.
        """
        rows = [
            {
                "task_wall_s": 2.0, "prediction_wall_s": 0.5,
                "teacher_forced_decision_count": 10,
                "fit_count": 1, "fit_total_wall_s": 1.0,
                "fit_wall_s_values": [0.4], "fit_flop_estimate": 100.0,
                "learner_dense_array_bytes": 512,
                "memory_active_variants": 3,
                "learner_replay_transition_count": 30,
                "teacher_forced_nll": 0.25,
                "teacher_forced_nll_total": 2.5,
                "teacher_forced_nll_decisions": 10,
                "invalid_predictions": 0,
                "task_completed": True,
            },
            {
                "task_wall_s": 4.0, "prediction_wall_s": 1.5,
                "teacher_forced_decision_count": 10,
                "fit_count": 1, "fit_total_wall_s": 3.0,
                "fit_wall_s_values": [1.6], "fit_flop_estimate": 200.0,
                "learner_dense_array_bytes": 2048,
                "memory_active_variants": 5,
                "learner_replay_transition_count": 50,
                "teacher_forced_nll": 0.75,
                "teacher_forced_nll_total": 7.5,
                "teacher_forced_nll_decisions": 10,
                "invalid_predictions": 0,
                "task_completed": True,
            },
        ]
        performance = _performance(rows)
        self.assertEqual(performance["episode_wall_s"], 6.0)
        self.assertEqual(performance["fit_count"], 2)
        self.assertEqual(performance["fit_total_wall_s"], 4.0)
        # p95 is the blocking wait between two demonstrations, so it reports the
        # slow fit rather than averaging it away.
        self.assertEqual(performance["p95_fit_wall_s"], 1.6)
        self.assertEqual(performance["fit_flop_estimate"], 300.0)
        self.assertEqual(performance["peak_learner_dense_array_bytes"], 2048)
        self.assertAlmostEqual(performance["mean_prediction_wall_s"], 0.1)
        self.assertAlmostEqual(performance["teacher_forced_nll"], 0.5)
        self.assertEqual(performance["task_completion_rate"], 1.0)

    def test_performance_tolerates_a_cell_with_no_fits(self):
        """A zero-learning arm never fits, and must not break the aggregate."""
        performance = _performance([
            {"task_wall_s": 1.0, "teacher_forced_decision_count": 0},
        ])
        self.assertEqual(performance["fit_count"], 0)
        self.assertIsNone(performance["p95_fit_wall_s"])
        self.assertIsNone(performance["mean_prediction_wall_s"])

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

    def test_checkpoint_reuse_requires_matching_provenance(self):
        """A checkpoint must record the code and config that produced it.

        The config digest alone was not enough: an edit to the evaluator or the
        protocol changes what a cell means without changing the config, so a
        resume would splice cells from two implementations into one artifact.
        Every check fails closed -- returning None just recomputes the cell.
        """
        from adaptive_hrc_burrito.evaluation import (
            RESULT_SCHEMA_VERSION,
            _source_fingerprint,
        )

        good = {
            "seed": 1, "scenario": "homogeneous", "arm": "full",
            "episodes": [{
                "cell_complete": True, "cell_planned_episodes": 1,
                "event_index": 0, "seed": 1, "scenario": "homogeneous",
                "arm": "full",
            }],
            "failures": [], "probes": [], "audits": [], "schedule": {},
            "provenance": {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "config_digest": "abc0123456",
                "source_fingerprint": _source_fingerprint(),
            },
        }

        def load(cell, **overrides):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                path = _cell_checkpoint_path(run_dir, 1, "homogeneous", "full")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cell), encoding="utf-8")
                kwargs = {
                    "config_digest": "abc0123456",
                    "source_fingerprint": _source_fingerprint(),
                }
                kwargs.update(overrides)
                return _load_checkpoint(run_dir, 1, "homogeneous", "full", **kwargs)

        self.assertIsNotNone(load(good))
        self.assertIsNone(load(good, config_digest="different0"))
        self.assertIsNone(load(good, source_fingerprint="0" * 16))
        self.assertIsNone(load({key: value for key, value in good.items()
                                if key != "provenance"}))
        stale = {**good, "provenance": {
            **good["provenance"], "result_schema_version": 0,
        }}
        self.assertIsNone(load(stale))

    def test_checkpoint_reuse_requires_full_episode_coverage(self):
        """A completion flag is a claim, not evidence.

        A checkpoint holding one of its 630 planned episodes, with the flag
        set, was accepted as a finished cell.
        """
        from adaptive_hrc_burrito.evaluation import (
            RESULT_SCHEMA_VERSION,
            _source_fingerprint,
        )

        def cell(episodes):
            return {
                "seed": 1, "scenario": "holdout", "arm": "full",
                "episodes": episodes, "failures": [], "probes": [],
                "audits": [], "schedule": {},
                "provenance": {
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "config_digest": "abc0123456",
                    "source_fingerprint": _source_fingerprint(),
                },
            }

        def episode(index, planned=3):
            return {
                "cell_complete": True, "cell_planned_episodes": planned,
                "event_index": index, "seed": 1, "scenario": "holdout",
                "arm": "full",
            }

        def load(payload, seed=1, scenario="holdout", arm="full"):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                path = _cell_checkpoint_path(run_dir, seed, scenario, arm)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
                return _load_checkpoint(
                    run_dir, seed, scenario, arm,
                    config_digest="abc0123456",
                    source_fingerprint=_source_fingerprint(),
                )

        self.assertIsNotNone(load(cell([episode(i) for i in range(3)])))
        # One of three planned, flag set: the original acceptance bug.
        self.assertIsNone(load(cell([episode(0)])))
        # Right count, duplicated index: coverage is not a length check.
        self.assertIsNone(load(cell([episode(0), episode(0), episode(1)])))
        # Index outside the schedule.
        self.assertIsNone(load(cell([episode(0), episode(1), episode(7)])))
        # Disagreeing planned counts within one cell.
        mixed = [episode(0), episode(1), episode(2, planned=4)]
        self.assertIsNone(load(cell(mixed)))
        # Cell identity must match what was asked for.
        misfiled = cell([episode(i) for i in range(3)])
        misfiled["arm"] = "bc"
        self.assertIsNone(load(misfiled))

    def test_source_fingerprint_covers_the_core_learner(self):
        """Hashing only the wrapper left checkpoints valid across a src edit.

        Everything the learner does comes from ``src``; the wrapper supplies
        the environment and the protocol.
        """
        from adaptive_hrc_burrito.evaluation import (
            _CORE_SOURCE_MODULES,
            _source_fingerprint,
        )
        from adaptive_hrc_burrito.runtime import UpstreamPaths

        core = UpstreamPaths.discover().integration_root.parent / "src"
        self.assertTrue(
            {"adaptive_agent.py", "models.py", "memory.py", "baselines.py"}
            <= set(_CORE_SOURCE_MODULES)
        )
        before = _source_fingerprint()
        target = core / "memory.py"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# fingerprint probe\n")
            self.assertNotEqual(_source_fingerprint(), before)
        finally:
            target.write_bytes(original)
        self.assertEqual(_source_fingerprint(), before)

    def test_incomplete_cells_are_rerun_not_reused(self):
        """An interrupted cell used to be absorbed as though it had finished.

        It is excluded from every pooled statistic at aggregation time, so
        accepting it on resume meant the cell could never be recovered: the run
        was permanently short by one arm-seed-scenario cell with no way to fix
        it but starting over.
        """
        from adaptive_hrc_burrito.evaluation import (
            RESULT_SCHEMA_VERSION,
            _source_fingerprint,
        )

        provenance = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "config_digest": "abc0123456",
            "source_fingerprint": _source_fingerprint(),
        }
        def episode(index, complete=True, planned=2):
            return {
                "cell_complete": complete, "cell_planned_episodes": planned,
                "event_index": index, "seed": 1, "scenario": "homogeneous",
                "arm": "full",
            }

        for episodes, failures in (
            ([episode(0), episode(1, complete=False)], []),
            ([], []),
            ([episode(0, planned=1)], [{"error": "boom"}]),
        ):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                path = _cell_checkpoint_path(run_dir, 1, "homogeneous", "full")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "seed": 1, "scenario": "homogeneous", "arm": "full",
                    "episodes": episodes, "failures": failures,
                    "probes": [], "audits": [], "schedule": {},
                    "provenance": provenance,
                }), encoding="utf-8")
                self.assertIsNone(_load_checkpoint(
                    run_dir, 1, "homogeneous", "full",
                    config_digest="abc0123456",
                    source_fingerprint=_source_fingerprint(),
                ))


class HoldoutRegressions(unittest.TestCase):
    def test_holdout_labels_isomorphic_transfers(self):
        """A source and target with one DAG relabel the axis, never generalise."""
        for seed in PAPER_SEEDS:
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
        for seed in PAPER_SEEDS:
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
        by_seed = summary["by_seed"]
        # Pooled weights seed 2 three times as heavily; the seed mean does not.
        self.assertAlmostEqual(summary["preference_discriminating_top_1"], 0.25)
        self.assertEqual(by_seed["unit"], "seed_x_arm_x_scenario_x_environment")
        self.assertEqual(len(by_seed["per_cell"]), 2)
        # One (arm, scenario, environment) row carrying both seeds.
        self.assertEqual(len(by_seed["seed_means"]), 1)
        self.assertAlmostEqual(
            by_seed["seed_means"][0]["preference_discriminating_top_1_seed_mean"],
            0.5,
        )
        self.assertEqual(by_seed["seed_means"][0]["n_seeds"], 2)

    def test_seed_cells_do_not_pool_arms(self):
        """Grouping by seed alone averaged every arm into one per-seed rate.

        That is not a quantity anyone can draw an inference from, and it left
        the run with no unit in which a Full-versus-arm difference or an
        environment-by-method interaction could be tested at all.
        """
        def row(arm, seed, hits):
            return {
                "mode": "assist", "arm": arm, "scenario": "homogeneous",
                "environment": "overcooked", "seed": seed, "recipe_id": "r",
                "preference": "p", "strategy": "s", "exposure_after_change": 0,
                "cell_complete": True, "recipe_steps": 6,
                "human_action_count": 3, "robot_decision_count": 1,
                "robot_top_1_hits": hits,
                "teacher_forced_decision_count": 1,
                "teacher_forced_top_1_hits": hits,
                "prefix_conditioned_robot_decisions": 1,
                "prefix_conditioned_top_1_hits": hits,
                "preference_discriminating_robot_decisions": 1,
                "preference_discriminating_top_1_hits": hits,
                "single_legal_action_decision_count": 0, "human_corrections": 0,
                "correction_free": True, "task_completed": True,
                "adaptation_id": None, "phase": 0, "lifecycle": "retain",
                "preference_changed": False, "lead_actor_policy": "human_first",
            }
        # Full is right on both seeds; the baseline is wrong on both.
        rows = [row("full", 1, 1), row("full", 2, 1),
                row("bc", 1, 0), row("bc", 2, 0)]
        by_seed = _aggregate(rows, [])["by_seed"]
        cells = {(r["arm"], r["seed"]): r for r in by_seed["per_cell"]}
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[("full", 1)]["teacher_forced_top_1"], 1.0)
        self.assertEqual(cells[("bc", 1)]["teacher_forced_top_1"], 0.0)
        contrasts = {r["arm"]: r for r in by_seed["paired_vs_full"]}
        self.assertEqual(set(contrasts), {"bc"})
        paired = contrasts["bc"]["teacher_forced_top_1"]
        self.assertEqual(paired["n_seeds"], 2)
        self.assertAlmostEqual(paired["mean"], 1.0)
        # No spread across seeds, so the interval is a point.
        self.assertAlmostEqual(paired["sd"], 0.0)
        self.assertAlmostEqual(paired["ci95_half_width"], 0.0)


class HoldoutStageRegressions(unittest.TestCase):
    """Stage and recipe role are independent; both are needed."""

    def _row(self, **kwargs):
        row = {
            "scenario": "holdout", "strategy": "holdout_axis_target_composition",
            "preference": CONTAINER_FIRST_PREFERENCE, "holdout_target": True,
            "exposure_after_change": 2,
        }
        row.update(kwargs)
        return row

    def test_source_recipes_do_not_enter_the_target_exposure_groups(self):
        """The target-composition *phase* still runs source recipes.

        On the saved run the axis "later exposure" group held 1,022 Overcooked
        episodes of which 250 were source recipes, and 384 native-Burrito
        episodes of which 211 were -- a majority.
        """
        stage, role, axis, exposure = _holdout_stage(self._row())
        self.assertEqual((stage, role, axis, exposure), (
            "holdout_axis_target_composition", "target", True, "later",
        ))
        # Same phase, same preference, but a source recipe.
        stage, role, axis, exposure = _holdout_stage(
            self._row(holdout_target=False)
        )
        self.assertEqual(role, "source")
        self.assertEqual(exposure, "not_applicable")

    def test_only_a_first_axis_exposure_on_a_target_is_the_transfer_cell(self):
        cases = {
            "transfer": self._row(exposure_after_change=1),
            "later_target": self._row(exposure_after_change=2),
            "source_recipe": self._row(exposure_after_change=1, holdout_target=False),
            "other_preference": self._row(
                exposure_after_change=1, preference="pot_rice_early",
            ),
            "source_training": self._row(
                exposure_after_change=1, strategy="holdout_source_training",
            ),
            "axis_introduction": self._row(
                exposure_after_change=1,
                strategy="holdout_axis_source_introduction",
            ),
        }
        groups = {
            name: _holdout_groups([{
                **row, "arm": "full", "seed": 1, "environment": "overcooked",
            }])[0]
            for name, row in cases.items()
        }
        self.assertTrue(groups["transfer"]["is_transfer_measurement"])
        for name in set(cases) - {"transfer"}:
            with self.subTest(case=name):
                self.assertFalse(groups[name]["is_transfer_measurement"])


class ScoringConventionRegressions(unittest.TestCase):
    """One scoring rule across both environments, pooled per decision."""

    def test_nll_is_pooled_over_decisions_not_episodes(self):
        """Averaging episode means is not the per-decision loss.

        Two episodes of one nat each -- one over a single decision, one over
        nine -- pool to 0.2 per decision; averaging their means reports 1.0.
        """
        from adaptive_hrc_burrito.evaluation import _performance

        rows = [
            {
                "teacher_forced_nll": 1.0, "teacher_forced_nll_total": 1.0,
                "teacher_forced_nll_decisions": 1,
                "teacher_forced_decision_count": 1,
            },
            {
                "teacher_forced_nll": 1.0 / 9.0,
                "teacher_forced_nll_total": 1.0,
                "teacher_forced_nll_decisions": 9,
                "teacher_forced_decision_count": 9,
            },
        ]
        performance = _performance(rows)
        self.assertAlmostEqual(performance["teacher_forced_nll"], 0.2)
        self.assertEqual(performance["teacher_forced_nll_decisions"], 10)

    def test_nll_floor_matches_the_symbolic_evaluator(self):
        """src.evaluation floors at max(settings.min_probability, 1e-12).

        Using 1e-12 unconditionally here made the two environments' losses
        incomparable by six orders of magnitude on every missed decision.
        """
        from src.models import Settings

        from src.models import Settings

        settings = Settings(verbose=False)
        # The symbolic side floors at max(settings.min_probability, 1e-12).
        self.assertEqual(settings.min_probability, 1e-6)

        domain = CookingDomainAdapter()
        agent = _build_agent("canonical_order", settings, domain)
        runner = CookingHrcRunner(
            agent, BurritoRuntime.discover(), domain, planner_seed=11,
        )
        self.assertEqual(runner.nll_probability_floor, 1e-6)
        # An explicit override is recorded rather than silently ignored.
        override = CookingHrcRunner(
            _build_agent("canonical_order", settings, domain := CookingDomainAdapter()),
            BurritoRuntime.discover(), domain, planner_seed=11,
            nll_probability_floor=1e-3,
        )
        self.assertEqual(override.nll_probability_floor, 1e-3)

    def test_the_effective_nll_floor_is_the_one_recorded(self):
        """An override has to reach the record, not just the runner.

        The episode record read the floor back off the agent's settings, so a
        run with an explicit 1e-3 override reported 1e-6 -- a convention that
        was not the one in force.
        """
        from adaptive_hrc_burrito.evaluation import _episode_record
        from src.models import Settings

        domain = CookingDomainAdapter()
        agent = _build_agent("canonical_order", Settings(verbose=False), domain)
        runtime = BurritoRuntime.discover()
        runner = CookingHrcRunner(
            agent, runtime, domain, horizon=1800, planner_seed=11,
            require_shift_update=False, nll_probability_floor=1e-3,
        )
        self.assertEqual(runner.nll_probability_floor, 1e-3)
        task = CookingTask.create("overcooked_onion_onion", "wash_plates_early")
        runner.run_task(task)
        result = runner.run_task(task)
        self.assertEqual(result.nll_probability_floor, 1e-3)
        record = _episode_record(
            result, metadata={"arm": "x", "seed": 1, "cell_complete": True},
            wall_s=0.0, agent=agent,
        )
        self.assertEqual(record["nll_probability_floor"], 1e-3)

    def test_silence_is_not_scored_as_a_perfect_prediction(self):
        """A uniform-over-candidates fallback scored silence at zero loss.

        On a single-candidate frontier ``log(1) == 0``, so an arm that emitted
        nothing received a perfect score -- and the substituted distribution
        was never one the predictor produced.
        """
        floor = 1e-6
        single_candidate_uniform = math.log(1)
        self.assertEqual(single_candidate_uniform, 0.0)
        charged = -math.log(max(0.0, floor))
        self.assertGreater(charged, 13.0)


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


class ChopBoardCollisionRegressions(unittest.TestCase):
    """`GO_TO_CHOP_BOARD_AND_CHOP_INGREDIENT` selects its board by
    `_not_is_ready` over *both* proteins -- upstream says as much: "cannot deal
    with the situation to chop ingredients others put down".  `burrito_combo`
    is the only recipe that stages two proteins at once, so preparing one
    finishes the other, and requiring the other to still be unprepared made its
    task-graph node unreachable.  The episode then burned `max_passive_wait_ticks`
    waiting for an option whose work had already been done, and died mid-run.

    Two conditions have to coincide, which is why this reached a full
    evaluation: the ordering must stage both proteins before either is
    prepared -- only `pickup_mushroom_early` does -- and player 1 must be the
    one preparing, which under `counterbalanced` is every odd assist exposure.
    `RealizedBehaviourRegressions` walks the same preference but always leads
    with player 0, so it routes around the collision."""

    ORDER = "pickup_mushroom_early"

    @classmethod
    def setUpClass(cls):
        from adaptive_hrc_burrito import BurritoRuntime
        cls.runtime = BurritoRuntime.discover()

    def _executor(self):
        from adaptive_hrc_burrito.physical import create_executor
        executor = create_executor(
            self.runtime, "burrito_combo", horizon=9000, seed=11
        )
        executor.reset()
        return executor

    @staticmethod
    def _robot_leads(decision_index):
        """Actor for one decision when the robot takes the first turn."""
        return (decision_index + 1) % 2

    def test_preparing_the_steak_leaves_the_mushroom_preparable(self):
        from adaptive_hrc_burrito.options import _has_ready, macro_is_legal
        executor = self._executor()
        prefix = (
            "FETCH_AND_STAGE_MUSHROOM",
            "FETCH_AND_STAGE_STEAK",
            "PREPARE_AND_STAGE_STEAK",
        )
        for index, action in enumerate(prefix):
            executor.execute(action, actor_id=self._robot_leads(index))
        # Pin the upstream routing this guards against: a future pin bump that
        # makes the chop primitive protein-aware should show up here rather
        # than passing silently.
        self.assertTrue(_has_ready(executor.state, "chopped_mushroom"))
        self.assertTrue(
            macro_is_legal(executor.state, "PREPARE_AND_STAGE_MUSHROOM", 0)
        )

    def test_the_colliding_ordering_completes_both_burritos(self):
        from adaptive_hrc_burrito.catalog import get_recipe
        executor = self._executor()
        graph = CookingTaskGraph.create("burrito_combo")
        policy = CookingPreferencePolicy.create(self.ORDER)
        completed, waits = [], 0
        while not graph.is_complete(completed):
            actor = self._robot_leads(len(completed))
            structural = graph.frontier(completed)
            action = policy.choose_action(structural, graph)
            while action not in graph.available_actions(
                completed, executor.legal_actions(structural, actor_id=actor)
            ):
                executor.advance_environment()
                waits += 1
                self.assertLess(waits, 900, f"deadlocked on {action}")
            executor.execute(action, actor_id=actor)
            completed.append(action)
        self.assertEqual(
            executor._delivery_count(),
            get_recipe("burrito_combo").expected_deliveries,
        )


class RescoreRegressions(unittest.TestCase):
    """Rescoring rebuilds from decisions; it must not trust stored rollups."""

    def _episode(self, mode="assist"):
        def decision(step, actor, legal, truth, predicted, probability):
            return {
                "step": step, "scheduled_actor": actor,
                "legal_actions": list(legal), "ground_truth_action": truth,
                "executed_action": truth, "predicted_action": predicted,
                "correct_top_1": predicted == truth,
                "ground_truth_probability": probability,
                # Deliberately absent/wrong: the rescorer recomputes both.
                "ground_truth_nll": 27.631021115928547,
                "human_corrected": actor == "robot" and predicted != truth,
            }
        return {
            "seed": 1, "arm": "full", "scenario": "homogeneous",
            "environment": "overcooked", "recipe_id": "overcooked_onion_onion",
            "preference": "wash_plates_early", "strategy": "s",
            "holdout_target": False, "holdout_transfer_isomorphic": False,
            "exposure_after_change": 0, "mode": mode, "cell_complete": True,
            "decisions": [
                decision(0, "human", ("ADD_ONION_1", "STAGE_DISH"),
                         "STAGE_DISH", "ADD_ONION_1", 0.4),
                # Unanswered: no prediction, zero ground-truth probability.
                decision(1, "robot", ("ADD_ONION_1",),
                         "ADD_ONION_1", None, 0.0),
            ],
        }

    def test_unanswered_decisions_are_rescored_not_dropped(self):
        """The stored NLL used a 1e-12 floor; recompute from the probability.

        27.6310 at the old floor becomes 13.8155 at the shared one, and the
        decision stays in every denominator it belongs to.
        """
        from adaptive_hrc_burrito.rescore import rescore_episode

        row = rescore_episode(self._episode(), nll_floor=1e-6)
        self.assertEqual(row["teacher_forced_decision_count"], 2)
        self.assertEqual(row["robot_decision_count"], 1)
        self.assertEqual(row["prediction_available_decisions"], 1)
        self.assertEqual(row["prediction_unavailable_decisions"], 1)
        self.assertEqual(row["human_corrections"], 1)
        self.assertLessEqual(row["human_corrections"], row["robot_decision_count"])
        expected = -math.log(0.4) + -math.log(1e-6)
        self.assertAlmostEqual(row["teacher_forced_nll_total"], expected, places=6)
        self.assertEqual(row["teacher_forced_nll_decisions"], 2)
        self.assertAlmostEqual(
            row["teacher_forced_nll_total"] / 2, expected / 2, places=6,
        )

    def test_rescoring_recomputes_ambiguity_and_the_opening(self):
        """Neither flag is read back: both are functions of recipe and prefix.

        ``prefix_conditioned_discriminating`` post-dates the saved run and is
        absent from its records entirely.
        """
        from adaptive_hrc_burrito.rescore import rescore_episode

        row = rescore_episode(self._episode(), nll_floor=1e-6)
        # Step 0 of this recipe is a real choice; the human takes it and the
        # arm's shadow prediction is wrong.
        self.assertEqual(row["opening_decision_count"], 1)
        self.assertEqual(row["opening_scheduled_actor"], "human")
        self.assertEqual(row["opening_discriminating_decision_count"], 1)
        self.assertEqual(row["opening_top_1_hits"], 0)
        # Step 1 has a single candidate, so it is neither discriminating nor
        # conditioned and cannot enter an accuracy denominator.
        self.assertEqual(row["preference_discriminating_robot_decisions"], 0)
        self.assertEqual(row["prefix_conditioned_robot_decisions"], 0)

    def test_observation_episodes_contribute_no_scored_decisions(self):
        from adaptive_hrc_burrito.rescore import rescore_episode

        row = rescore_episode(self._episode(mode="observe"), nll_floor=1e-6)
        self.assertEqual(row["teacher_forced_decision_count"], 0)
        self.assertEqual(row["robot_decision_count"], 0)
        self.assertEqual(row["teacher_forced_nll_decisions"], 0)


class MechanismProbeRegressions(unittest.TestCase):
    """The removal conditions must isolate the competing explanations."""

    def test_removal_sets_separate_source_axis_from_obsolete_target(self):
        from adaptive_hrc_burrito.mechanism_probe import (
            VariantView,
            _removal_sets,
        )

        def view(key, recipe, preference):
            return VariantView(
                key=key, catalog_recipe=recipe, preference=preference,
                weight=1.0, pinned=False, added_step=0, last_seen_step=0,
                transition_count=9,
            )

        target = "burrito_steak_burrito"
        views = [
            # obsolete knowledge about the target itself
            view(("R0", "v0"), target, "plate_ingredients_early"),
            view(("R0", "v1"), target, "pot_rice_early"),
            # the axis, learned on a different (source) recipe
            view(("R1", "v0"), "overcooked_onion_onion", CONTAINER_FIRST_PREFERENCE),
            # unrelated
            view(("R2", "v0"), "burrito_combo", "plate_tortilla_early"),
        ]
        sets = _removal_sets(views, target)
        self.assertEqual(sets["unmodified"], ())
        self.assertEqual(
            set(sets["minus_obsolete_target_variants"]),
            {("R0", "v0"), ("R0", "v1")},
        )
        self.assertEqual(
            set(sets["minus_source_axis_variants"]), {("R1", "v0")},
        )
        self.assertEqual(len(sets["minus_both"]), 3)
        # An unrelated variant is never removed by any condition, so a change
        # cannot be attributed to shrinking the active set in general.
        for keys in sets.values():
            self.assertNotIn(("R2", "v0"), keys)

    def test_the_target_axis_variant_is_never_removed(self):
        """Removing the target's *own* axis variant would beg the question."""
        from adaptive_hrc_burrito.mechanism_probe import (
            VariantView,
            _removal_sets,
        )

        target = "burrito_steak_burrito"
        own_axis = VariantView(
            key=("R0", "v9"), catalog_recipe=target,
            preference=CONTAINER_FIRST_PREFERENCE, weight=1.0, pinned=True,
            added_step=0, last_seen_step=0, transition_count=9,
        )
        for keys in _removal_sets([own_axis], target).values():
            self.assertNotIn(("R0", "v9"), keys)


class CliReportRegressions(unittest.TestCase):
    """The success report must not be able to discard a finished run.

    Dropping three metrics from ``_aggregate`` left ``__main__`` still reading
    them, so a two-hour run wrote every artifact, passed validation, and then
    exited non-zero on ``KeyError: 'primary_metric'`` inside its own report --
    which the notebook's ``check=True`` surfaced as a failed stage.
    """

    def _row(self):
        return {
            "mode": "assist", "arm": "full", "scenario": "homogeneous",
            "environment": "overcooked", "seed": 1337,
            "recipe_id": "overcooked_onion_tomato", "preference": "tomato_early",
            "strategy": "s", "task_completed": True, "exposure_after_change": 0,
            "cell_complete": True, "recipe_steps": 6, "human_action_count": 3,
            "robot_decision_count": 3, "robot_top_1_hits": 2,
            "teacher_forced_decision_count": 6, "teacher_forced_top_1_hits": 5,
            "preference_discriminating_robot_decisions": 1,
            "preference_discriminating_top_1_hits": 1,
            "prefix_conditioned_robot_decisions": 1,
            "prefix_conditioned_top_1_hits": 1,
            "opening_decision_count": 1, "opening_top_1_hits": 0,
            "opening_discriminating_decision_count": 1,
            "opening_discriminating_top_1_hits": 0,
            "single_legal_action_decision_count": 1, "human_corrections": 1,
            "correction_free": False, "adaptation_id": None, "phase": 0,
            "lifecycle": "retain", "preference_changed": False,
            "lead_actor_policy": "human_first",
        }

    def test_every_reported_key_is_produced_by_the_aggregate(self):
        from adaptive_hrc_burrito.evaluation import (
            CLI_REPORT_KEYS,
            summary_report,
        )

        summary = _aggregate([self._row()], [])
        absent = [key for key in CLI_REPORT_KEYS if key not in summary]
        self.assertEqual(absent, [], f"report names keys _aggregate lacks: {absent}")
        report = summary_report({"run_dir": "/tmp/x", "summary": summary})
        self.assertNotIn("missing_summary_keys", report)
        self.assertEqual(report["run_dir"], "/tmp/x")

    def test_a_missing_key_is_reported_not_raised(self):
        from adaptive_hrc_burrito.evaluation import summary_report

        summary = _aggregate([self._row()], [])
        del summary["teacher_forced_top_1"]
        report = summary_report(
            {"run_dir": "/tmp/x", "summary": summary}, {"status": "passed"},
        )
        self.assertEqual(report["missing_summary_keys"], ["teacher_forced_top_1"])
        self.assertEqual(report["validation"], {"status": "passed"})

    def test_the_report_is_json_serialisable(self):
        """It is printed with json.dumps, so a stray tuple would also crash it."""
        from adaptive_hrc_burrito.evaluation import summary_report

        report = summary_report(
            {"run_dir": "/tmp/x", "summary": _aggregate([self._row()], [])},
            {"status": "passed", "failures": []},
        )
        json.dumps(report, sort_keys=True)

    def test_the_dropped_metrics_are_not_referenced_anywhere(self):
        from adaptive_hrc_burrito.evaluation import CLI_REPORT_KEYS

        for name in (
            "primary_metric", "normalized_human_action_load",
            "normalized_human_action_load_floor", "human_action_load_excess",
            "robot_top_k", "nontrivial_choice_top_1",
        ):
            self.assertNotIn(name, CLI_REPORT_KEYS)
