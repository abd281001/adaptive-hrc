import math
import unittest

import numpy as np

from adaptive_hrc_burrito import (
    ASSIST,
    BURRITO_RECIPE_IDS,
    OVERCOOKED_RECIPE_IDS,
    RECIPES,
    BurritoRuntime,
    CookingDomainAdapter,
    CookingHrcRunner,
    CookingPreferencePolicy,
    CookingTask,
    CookingTaskGraph,
    LadderSettings,
    SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
    TaskState,
    applicable_preferences,
    generate_ladder,
)
from adaptive_hrc_burrito.ladder import (
    CANONICAL_STRATEGY,
    CONTAINER_FIRST_STRATEGY,
    SUPPORT_FIRST_STRATEGY,
    ladder_audit,
)
from adaptive_hrc_burrito.physical import create_executor


class CatalogAndDomainTests(unittest.TestCase):
    def test_catalog_contains_only_nontrivial_multi_ingredient_recipes(self):
        self.assertEqual(len(RECIPES), 12)
        self.assertEqual(len(OVERCOOKED_RECIPE_IDS), 7)
        self.assertEqual(len(BURRITO_RECIPE_IDS), 5)
        self.assertEqual(RECIPES["burrito_combo"].expected_deliveries, 2)
        self.assertTrue(all(len(recipe.ingredients) >= 2 for recipe in RECIPES.values()))
        ingredients = {
            ingredient for recipe in RECIPES.values()
            for ingredient in recipe.ingredients
        }
        self.assertTrue({
            "onion", "tomato", "meat", "chicken", "mushroom", "rice", "tortilla"
        } <= ingredients)

    def test_task_state_is_actor_and_navigation_independent(self):
        domain = CookingDomainAdapter("overcooked_onion_tomato")
        state = domain.state_from_completed(
            "overcooked_onion_tomato", ("ADD_ONION_1",)
        )
        self.assertEqual(domain.state_key(state, actor_id=0), state)
        self.assertEqual(domain.state_key(state, actor_id=1), state)
        decoded = TaskState.decode(state)
        self.assertEqual(decoded.recipe_id, "overcooked_onion_tomato")
        self.assertEqual(decoded.completed, frozenset(("ADD_ONION_1",)))
        with self.assertRaisesRegex(TypeError, "raw simulator geometry"):
            domain.state_key(object())

    def test_semantic_features_overlap_across_recipe_identity(self):
        domain = CookingDomainAdapter()
        left = domain.state_from_completed("overcooked_onion_onion", ())
        right = domain.state_from_completed("overcooked_tomato_tomato", ())
        semantic, _, _ = domain.semantic_features(
            {0: left, 1: right}, normalize=False
        )
        engineered, _, _ = domain.build_features(
            {0: left, 1: right}, feature_mode="engineered", normalize=False
        )
        np.testing.assert_array_equal(semantic[0], semantic[1])
        self.assertFalse(np.array_equal(engineered[0], engineered[1]))

    def test_ladders_use_natural_acquisition_and_later_recurrence(self):
        settings = LadderSettings()
        for scenario in ("homogeneous", "heterogeneous", "holdout"):
            tasks = generate_ladder(seed=17, scenario=scenario, settings=settings)
            acquisitions = [task for task in tasks if task.lifecycle == "acquire_shift"]
            self.assertTrue(acquisitions)
            self.assertTrue(all(
                task.preference_changed and task.exposure_after_change == 1
                for task in acquisitions
            ))
            for acquisition in acquisitions:
                index = tasks.index(acquisition)
                later = [
                    task for task in tasks[index + 1:]
                    if task.adaptation_id == acquisition.adaptation_id
                    and task.exposure_after_change >= 2
                ]
                self.assertTrue(later)
                self.assertGreaterEqual(tasks.index(later[0]) - index - 1, 3)

    def test_ladder_scenarios_enforce_their_claimed_structure(self):
        settings = LadderSettings()
        generated = {
            scenario: generate_ladder(
                seed=1337, scenario=scenario, settings=settings,
            )
            for scenario in ("homogeneous", "heterogeneous", "holdout")
        }
        self.assertEqual(
            generated["homogeneous"],
            generate_ladder(
                seed=1337, scenario="homogeneous", settings=settings,
            ),
        )
        first_phase = {
            task.recipe_id for task in generated["homogeneous"] if task.phase == 0
        }
        self.assertEqual(
            {RECIPES[recipe_id].environment for recipe_id in first_phase},
            {"overcooked", "burrito"},
        )
        homogeneous_audit = ladder_audit(generated["homogeneous"])
        self.assertEqual(len(generated["homogeneous"]), 210)
        self.assertTrue(all(
            len(strategies) == 1
            and strategies[0].startswith("homogeneous[")
            for strategies in homogeneous_audit["strategies_by_phase"].values()
        ))
        heterogeneous = generated["heterogeneous"]
        heterogeneous_audit = ladder_audit(heterogeneous)
        self.assertGreater(len(heterogeneous), settings.demos)
        self.assertGreaterEqual(
            heterogeneous_audit["max_active_preferences_per_recipe_phase"], 2,
        )
        holdout = generated["holdout"]
        holdout_audit = ladder_audit(holdout)
        introduction = holdout_audit["holdout_introduction_phase"]
        targets = set(holdout_audit["holdout_target_recipe_ids"])
        self.assertEqual(
            set(holdout_audit["holdout_target_environments"]),
            {"overcooked", "burrito_native"},
        )
        # Compatibility recipes are executed by wrapper-restored transitions
        # and must never enter the holdout's transfer claim.
        self.assertNotIn("burrito_compat", holdout_audit["holdout_target_environments"])
        self.assertFalse(any(
            task.preference == "wash_plates_early" and task.phase < introduction
            for task in holdout
        ))
        self.assertTrue(all(any(
            task.recipe_id == recipe_id and task.phase < introduction
            for task in holdout
        ) for recipe_id in targets))
        self.assertTrue(all(any(
            task.recipe_id == recipe_id
            and task.preference == "wash_plates_early"
            and task.phase > introduction
            for task in holdout
        ) for recipe_id in targets))


class PhysicalAndProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = BurritoRuntime.discover()

    def test_every_recipe_and_distinct_preference_completes_physically(self):
        for recipe_id in RECIPES:
            for preference in applicable_preferences(recipe_id):
                with self.subTest(recipe=recipe_id, preference=preference):
                    executor = create_executor(
                        self.runtime, recipe_id, horizon=1800, seed=11
                    )
                    graph = CookingTaskGraph.create(recipe_id)
                    policy = CookingPreferencePolicy.create(preference)
                    completed = []
                    actor = 0
                    waits = 0
                    while not graph.is_complete(completed):
                        structural = graph.frontier(completed)
                        # Mirror CookingHrcRunner: the preference selects over
                        # the task-graph frontier and waits for its choice.
                        action = policy.choose_action(structural, graph)
                        while action not in graph.available_actions(
                            completed,
                            executor.legal_actions(structural, actor_id=actor),
                        ):
                            executor.advance_environment()
                            waits += 1
                            self.assertLess(waits, 900)
                        executor.execute(action, actor_id=actor)
                        completed.append(action)
                        actor = 1 - actor
                    self.assertEqual(
                        executor._delivery_count(),
                        RECIPES[recipe_id].expected_deliveries,
                    )
                    self.assertEqual(set(completed), set(graph.actions))
                    self.assertEqual(
                        executor.compatibility_dynamics,
                        RECIPES[recipe_id].compatibility_dynamics,
                    )
                    if executor.compatibility_dynamics:
                        self.assertTrue(executor.compatibility_calls)

    def _runner(self, seed=1337):
        from src.adaptive_agent import AdaptiveAgent
        from src.models import Settings

        domain = CookingDomainAdapter()
        agent = AdaptiveAgent(Settings(
            verbose=False,
            seed=seed,
            irl_cold_steps=4,
            irl_warm_steps=2,
            irl_horizon=12,
            initial_grace=2,
            min_grace=1,
            semantic_fallback_max_rms_distance=SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
        ), domain=domain)
        return CookingHrcRunner(
            agent, self.runtime, domain, horizon=1800, planner_seed=11
        ), agent

    def test_candidate_lists_are_the_structural_frontier(self):
        """The candidate list must not be a function of the ground truth.

        It used to be the physically-legal subset at the first tick where the
        *preferred* option became executable, so competing options that were
        not ready yet silently vanished from the list the predictor was scored
        on.  Readiness is now waited out after the decision, against whichever
        option is actually executed.
        """
        runner, _agent = self._runner()
        graph = CookingTaskGraph.create("burrito_steak_burrito")
        results = runner.run_stream((
            CookingTask.create("burrito_steak_burrito", "plate_protein_early"),
            CookingTask.create(
                "burrito_steak_burrito", "plate_protein_early", phase=1,
            ),
        ))
        for result in results:
            completed = []
            for decision in result.decisions:
                self.assertEqual(
                    tuple(decision.legal_actions), graph.frontier(completed),
                )
                completed.append(decision.actual)
        # Waiting still happens -- it just happens after the decision.
        self.assertGreater(sum(r.passive_wait_ticks for r in results), 0)

    def test_candidate_lists_match_across_methods(self):
        """Two arms must be scored on identical states, prefixes and candidates.

        The executed prefix is the human's ordering either way, so the
        structural frontier is method-independent by construction; when the
        candidate list was the physically-legal subset it was not, because the
        two arms reach a given step with different pot timers.
        """
        stream = (
            CookingTask.create("burrito_steak_burrito", "pot_rice_early"),
            CookingTask.create("burrito_steak_burrito", "pot_rice_early", phase=1),
        )
        traces = []
        for arm in ("full", "canonical_order"):
            from adaptive_hrc_burrito.evaluation import _build_agent
            from src.models import Settings

            domain = CookingDomainAdapter()
            agent = _build_agent(arm, Settings(
                verbose=False, seed=1337, irl_cold_steps=4, irl_warm_steps=2,
                irl_horizon=12, initial_grace=2, min_grace=1,
                semantic_fallback_max_rms_distance=(
                    SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
                ),
            ), domain=domain)
            runner = CookingHrcRunner(
                agent, self.runtime, domain, horizon=1800, planner_seed=11,
                require_shift_update=False,
            )
            traces.append([
                (d.recipe_step, tuple(d.legal_actions), d.ground_truth_action)
                for result in runner.run_stream(stream)
                for d in result.decisions
            ])
        self.assertEqual(traces[0], traces[1])

    def test_an_arm_that_emits_nothing_is_scored_not_dropped(self):
        """Unanswered decisions belong in the denominator, not outside it.

        Filtering them out removed the decisions an arm failed to answer from
        every rate, which inflated precisely the arms that fail to answer.
        """
        from adaptive_hrc_burrito.evaluation import _build_agent, _episode_record
        from src.models import Settings

        domain = CookingDomainAdapter()
        agent = _build_agent("canonical_order", Settings(verbose=False), domain)
        runner = CookingHrcRunner(
            agent, self.runtime, domain, horizon=1800, planner_seed=11,
            require_shift_update=False,
        )
        task = CookingTask.create("overcooked_onion_tomato", "tomato_early")
        runner.run_task(task)                                  # observation
        agent.predict_actions = lambda *a, **k: {}              # goes mute
        result = runner.run_task(task)                          # assist
        record = _episode_record(
            result, metadata={"arm": "mute", "seed": 1, "cell_complete": True},
            wall_s=0.0, agent=agent,
        )

        steps = len(result.decisions)
        self.assertEqual(record["teacher_forced_decision_count"], steps)
        self.assertEqual(record["teacher_forced_top_1_hits"], 0)
        self.assertEqual(record["prediction_available_decisions"], 0)
        self.assertEqual(record["prediction_unavailable_decisions"], steps)
        # The robot denominator is the schedule, not the answers.
        self.assertEqual(record["robot_decision_count"], result.robot_turns)
        self.assertGreater(record["robot_decision_count"], 0)
        self.assertEqual(record["robot_top_1_hits"], 0)
        # Every robot turn it could not answer became a human correction.
        self.assertEqual(record["human_corrections"], result.robot_turns)
        self.assertLessEqual(
            record["human_corrections"], record["robot_decision_count"],
        )
        self.assertEqual(record["corrections_per_robot_decision"], 1.0)
        # Silence and "emitted a distribution that excluded the truth" are the
        # same event for a scoring rule, so both are charged the shared floor
        # -- the one src.evaluation uses, not a uniform distribution the
        # predictor never produced.
        self.assertEqual(record["nll_probability_floor"], 1e-6)
        self.assertAlmostEqual(
            record["teacher_forced_nll"], -math.log(1e-6), places=6,
        )
        self.assertEqual(record["teacher_forced_nll_decisions"], steps)
        self.assertAlmostEqual(
            record["teacher_forced_nll_total"], steps * -math.log(1e-6),
            places=4,
        )

    def test_the_opening_move_is_scored_and_separable(self):
        """Under human_first no robot-turn metric can see the opening move.

        It is the most preference-informative decision in these task graphs,
        and on the container-axis transfer cell it is discriminating in every
        episode -- which makes it *the* transfer measurement, invisible to
        every metric that only counts robot turns.
        """
        from adaptive_hrc_burrito.evaluation import _episode_record

        runner, agent = self._runner()
        task = CookingTask.create("overcooked_onion_onion", "wash_plates_early")
        runner.run_task(task)
        result = runner.run_task(CookingTask.create(
            "overcooked_onion_onion", "wash_plates_early", phase=1,
        ))
        record = _episode_record(
            result, metadata={"arm": "full", "seed": 1337, "cell_complete": True},
            wall_s=0.0, agent=agent,
        )
        self.assertTrue(record["opening_scored"])
        self.assertEqual(record["opening_scheduled_actor"], "human")
        self.assertEqual(record["opening_decision_count"], 1)
        self.assertTrue(record["opening_preference_discriminating"])
        self.assertEqual(record["opening_discriminating_decision_count"], 1)
        self.assertIn(record["opening_top_1_hits"], (0, 1))
        # This cell has no scorable robot decision at all, so the opening is
        # the only place its preference can be measured.
        self.assertEqual(record["preference_discriminating_robot_decisions"], 0)

    def test_first_shift_updates_then_natural_recurrence_assists(self):
        runner, agent = self._runner()
        observation, acquisition, recurrence = runner.run_stream((
            CookingTask.create("overcooked_onion_tomato", "plate_ingredients_early"),
            CookingTask.create(
                "overcooked_onion_tomato", "wash_plates_early",
                phase=1, lifecycle="acquire_shift", preference_changed=True,
                exposure_after_change=1,
            ),
            CookingTask.create(
                "overcooked_onion_tomato", "wash_plates_early",
                phase=1, lifecycle="post_update_recurrence",
                exposure_after_change=2,
            ),
        ))
        self.assertNotEqual(observation.mode, ASSIST)
        self.assertEqual(acquisition.mode, ASSIST)
        self.assertTrue(acquisition.commit_applied)
        self.assertTrue(acquisition.active_rehearsal)
        self.assertTrue(acquisition.retrain_executed)
        self.assertTrue(recurrence.post_update_recurrence)
        self.assertEqual(recurrence.corrections, 0)
        internal_recipe = runner._learner_recipe_by_task[
            "overcooked_onion_tomato"
        ]
        latest = agent.library.latest[internal_recipe]
        self.assertEqual(
            agent.library.variants[internal_recipe][latest].ordering,
            recurrence.preference_sequence,
        )

    def test_human_leads_then_robot_and_corrections_train_ground_truth(self):
        runner, agent = self._runner()
        recipe_id = "overcooked_onion_tomato"
        runner.run_task(CookingTask.create(recipe_id, "plate_ingredients_early"))
        result = runner.run_task(CookingTask.create(
            recipe_id,
            "wash_plates_early",
            phase=1,
            lifecycle="acquire_shift",
            preference_changed=True,
            exposure_after_change=1,
            adaptation_id="unit-shift",
        ))
        self.assertEqual(result.decisions[0].scheduled_actor, "human")
        # The opening move is scored as a teacher-forced shadow prediction
        # (matching src.hrc_simulation.simulate_episode) but never controls
        # execution: the human still performs the ground-truth option.  In
        # these task graphs step 0 is often the only preference-discriminating
        # decision, so leaving it unscored hid the strategy being measured.
        opening = result.decisions[0]
        self.assertIsNotNone(opening.predicted)
        self.assertEqual(opening.executed_by, "human")
        self.assertEqual(opening.actual, opening.ground_truth_action)
        self.assertFalse(opening.proposal_executed)
        self.assertGreaterEqual(result.scored_turns, len(result.decisions))
        self.assertGreaterEqual(
            result.scored_discriminating_decisions,
            sum(
                decision.preference_discriminating
                and decision.scheduled_actor == "robot"
                for decision in result.decisions
            ),
        )
        for previous, current in zip(result.decisions, result.decisions[1:]):
            if previous.scheduled_actor == "robot" and previous.human_corrected:
                self.assertEqual(current.scheduled_actor, "robot")
        internal_recipe = runner._learner_recipe_by_task[recipe_id]
        latest = agent.library.latest[internal_recipe]
        self.assertEqual(
            agent.library.variants[internal_recipe][latest].ordering,
            tuple(decision.ground_truth_action for decision in result.decisions),
        )

    def test_wrong_robot_proposal_is_vetoed_and_robot_keeps_turn(self):
        runner, agent = self._runner()
        recipe_id = "overcooked_onion_tomato"
        runner.run_task(CookingTask.create(recipe_id, "plate_ingredients_early"))
        wrong = "FETCH_MEAT"
        agent.predict_actions = lambda *args, **kwargs: {wrong: 1.0}
        result = runner.run_task(CookingTask.create(recipe_id, "wash_plates_early"))
        robot_turns = [
            decision for decision in result.decisions
            if decision.scheduled_actor == "robot"
        ]
        self.assertTrue(robot_turns)
        self.assertTrue(all(
            decision.human_corrected
            and decision.executed_by == "human_correction"
            and not decision.proposal_executed
            for decision in robot_turns
        ))
        for first, second in zip(result.decisions, result.decisions[1:]):
            if first.scheduled_actor == "robot" and first.human_corrected:
                self.assertEqual(second.scheduled_actor, "robot")
        executed = tuple(
            row.action for row in runner._executors[recipe_id].execution_log
        )
        self.assertNotIn(wrong, executed)


if __name__ == "__main__":
    unittest.main()
