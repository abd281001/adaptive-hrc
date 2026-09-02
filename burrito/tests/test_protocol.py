import unittest

import numpy as np

from adaptive_hrc_burrito.domain import SCHEMA
from adaptive_hrc_burrito.domain import SEMANTIC_FALLBACK_MAX_RMS_DISTANCE

from adaptive_hrc_burrito import (
    ASSIST,
    OBSERVE,
    BurritoDomainAdapter,
    BurritoHrcRunner,
    BurritoOptionExecutor,
    BurritoPreferencePolicy,
    BurritoRuntime,
    BurritoTask,
    BurritoTaskGraph,
    COLLECT_AND_STAGE_COOKED_RICE,
    CONTROLLED_PARKING_POSITIONS,
    PREFERENCE_NAMES,
    STAGE_CLEAN_PLATE,
    START_BOILING_RICE,
    assemble_action,
    fetch_and_stage_action,
    macro_actions,
    prepare_and_stage_action,
    serve_action,
    start_cooking_action,
)


class BurritoDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = BurritoRuntime.discover()

    def test_semantic_features_mask_protein_identity(self):
        executor = BurritoOptionExecutor(self.runtime, seed=1)
        domain = BurritoDomainAdapter(
            executor.state,
            terrain_positions=executor.env.mdp.terrain_pos_dict,
        )

        def projected(protein):
            state = domain.initial_state()
            for action in (
                STAGE_CLEAN_PLATE,
                fetch_and_stage_action(protein),
                prepare_and_stage_action(protein),
                start_cooking_action(protein),
                START_BOILING_RICE,
            ):
                state = domain.replay_transition(state, action)
            return state

        steak = projected("steak")
        mushroom = projected("mushroom")
        semantic, _, _ = domain.build_features(
            {0: steak, 1: mushroom},
            feature_mode="semantic",
            normalize=False,
        )
        engineered, _, _ = domain.build_features(
            {0: steak, 1: mushroom},
            feature_mode="engineered",
            normalize=False,
        )
        np.testing.assert_array_equal(semantic[0], semantic[1])
        self.assertFalse(np.array_equal(engineered[0], engineered[1]))
        self.assertEqual(domain.action_role(
            fetch_and_stage_action("steak"),
        ), "retrieve")
        self.assertEqual(domain.action_role(
            fetch_and_stage_action("mushroom"),
        ), "retrieve")
        self.assertEqual(
            domain.canonical_action(" start_boiling_rice "),
            START_BOILING_RICE,
        )
        self.assertEqual(
            tuple(domain.strategy_roles),
            (
                "retrieve", "prepare", "start_cook", "stage",
                "collect", "assemble", "serve",
            ),
        )

    def test_declared_strategy_roles_exactly_cover_recipe_actions(self):
        executor = BurritoOptionExecutor(self.runtime, seed=2)
        domain = BurritoDomainAdapter(
            executor.state,
            terrain_positions=executor.env.mdp.terrain_pos_dict,
        )
        emitted_roles = {
            domain.action_role(action)
            for protein in ("steak", "mushroom")
            for action in macro_actions(protein)
        }
        self.assertNotIn("other", emitted_roles)
        self.assertEqual(emitted_roles, set(domain.strategy_roles))

    def test_all_four_policies_are_state_conditioned_partial_orders(self):
        executor = BurritoOptionExecutor(self.runtime, seed=3)
        domain = BurritoDomainAdapter(
            executor.state,
            terrain_positions=executor.env.mdp.terrain_pos_dict,
        )
        state = domain.initial_state()
        graph = BurritoTaskGraph.create("steak")

        self.assertEqual(
            BurritoPreferencePolicy.create("plate_early").acceptable_actions(
                graph, state, (), domain,
            ),
            (STAGE_CLEAN_PLATE,),
        )
        self.assertNotIn(
            STAGE_CLEAN_PLATE,
            BurritoPreferencePolicy.create("plate_jit").acceptable_actions(
                graph, state, (), domain,
            ),
        )
        protein_first = BurritoPreferencePolicy.create(
            "protein_first",
        ).acceptable_actions(graph, state, (), domain)
        self.assertIn(fetch_and_stage_action("steak"), protein_first)
        self.assertNotIn(START_BOILING_RICE, protein_first)

        after_fetch = domain.replay_transition(
            state, fetch_and_stage_action("steak"),
        )
        rice_first = BurritoPreferencePolicy.create(
            "rice_first",
        ).acceptable_actions(
            graph, after_fetch, (fetch_and_stage_action("steak"),), domain,
        )
        self.assertIn(START_BOILING_RICE, rice_first)
        self.assertNotIn(prepare_and_stage_action("steak"), rice_first)

        after_prepare = domain.replay_transition(
            after_fetch, prepare_and_stage_action("steak"),
        )
        self.assertNotIn(
            start_cooking_action("steak"),
            graph.available_actions(
                after_prepare,
                (
                    fetch_and_stage_action("steak"),
                    prepare_and_stage_action("steak"),
                ),
                domain,
            ),
        )
        after_rice = domain.replay_transition(
            after_prepare, START_BOILING_RICE,
        )
        after_both_started = domain.replay_transition(
            after_rice, start_cooking_action("steak"),
        )
        self.assertEqual(
            BurritoPreferencePolicy.create("plate_jit").acceptable_actions(
                graph,
                after_both_started,
                (
                    fetch_and_stage_action("steak"),
                    prepare_and_stage_action("steak"),
                    START_BOILING_RICE,
                    start_cooking_action("steak"),
                ),
                domain,
            ),
            (STAGE_CLEAN_PLATE,),
        )

    def test_all_policies_deliver_using_eight_physical_macros(self):
        for protein in ("steak", "mushroom"):
            for preference in PREFERENCE_NAMES:
                with self.subTest(protein=protein, preference=preference):
                    runner, _executor, _agent = self._runner(seed=11)
                    result = runner.run_task(BurritoTask.create(
                        protein, preference,
                    ))
                    self.assertEqual(result.deliveries, 1)
                    self.assertEqual(result.memory_age_delta, 1)
                    self.assertEqual(len(result.decisions), 8)
                    self.assertEqual(
                        set(result.preference_sequence),
                        set(result.task.action_space),
                    )
                    self.assertTrue(all(
                        decision.actual in decision.acceptable_actions
                        for decision in result.decisions
                    ))

    def test_reset_detaches_upstream_shared_completed_order_list(self):
        first = BurritoOptionExecutor(self.runtime, seed=17, horizon=1800)
        self._execute_direct_recipe(first, "steak")
        self.assertTrue(first.state._complete_orders)
        second = BurritoOptionExecutor(self.runtime, seed=17, horizon=1800)
        self.assertEqual(second.state._complete_orders, [])

    def test_second_layout_and_distinct_model_seed_complete(self):
        from src.adaptive_agent import AdaptiveAgent
        from src.models import Settings

        executor = BurritoOptionExecutor(
            self.runtime, layout="burrito", seed=11, horizon=2400,
        )
        self.assertEqual(
            executor.parking_positions,
            dict(CONTROLLED_PARKING_POSITIONS["burrito"]),
        )
        domain = BurritoDomainAdapter(
            executor.state,
            terrain_positions=executor.env.mdp.terrain_pos_dict,
        )
        agent = AdaptiveAgent(Settings(
            verbose=False,
            seed=2024,
            irl_cold_steps=2,
            irl_warm_steps=1,
            irl_horizon=12,
            semantic_fallback_max_rms_distance=(
                SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
            ),
        ), domain=domain)
        result = BurritoHrcRunner(
            agent, executor, domain, seed=2024,
        ).run_task(BurritoTask.create("mushroom", "plate_early"))
        self.assertEqual(result.deliveries, 1)
        self.assertEqual(len(result.decisions), 8)
        self.assertTrue(all(
            executor.state.players[actor_id].held_object is None
            for actor_id in (0, 1)
        ))

    def _runner(self, seed=1337):
        from src.adaptive_agent import AdaptiveAgent
        from src.models import Settings

        executor = BurritoOptionExecutor(
            self.runtime, seed=seed, horizon=1800,
        )
        domain = BurritoDomainAdapter(
            executor.state,
            terrain_positions=executor.env.mdp.terrain_pos_dict,
        )
        agent = AdaptiveAgent(Settings(
            verbose=False,
            seed=seed,
            irl_cold_steps=2,
            irl_warm_steps=1,
            irl_horizon=12,
            semantic_fallback_max_rms_distance=(
                SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
            ),
        ), domain=domain)
        return BurritoHrcRunner(
            agent, executor, domain, seed=seed,
        ), executor, agent

    @staticmethod
    def _execute_direct_recipe(executor, protein):
        actions = (
            STAGE_CLEAN_PLATE,
            fetch_and_stage_action(protein),
            prepare_and_stage_action(protein),
            start_cooking_action(protein),
            START_BOILING_RICE,
            COLLECT_AND_STAGE_COOKED_RICE,
            assemble_action(protein),
            serve_action(protein),
        )
        for index, action in enumerate(actions):
            waited = 0
            while action not in executor.legal_actions(
                (action,), actor_id=index % 2,
            ):
                executor.advance_environment(1)
                waited += 1
                if waited > 400:
                    raise AssertionError(f"{action} never became legal")
            executor.execute(action, actor_id=index % 2)
            if executor.state.players[index % 2].held_object is not None:
                raise AssertionError(
                    f"{action} did not end at a handoff-safe boundary"
                )


class BurritoProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = BurritoRuntime.discover()

    def _runner(self):
        from src.adaptive_agent import AdaptiveAgent
        from src.models import Settings

        executor = BurritoOptionExecutor(
            self.runtime, seed=1337, horizon=1800,
        )
        domain = BurritoDomainAdapter(
            executor.state,
            terrain_positions=executor.env.mdp.terrain_pos_dict,
        )
        agent = AdaptiveAgent(Settings(
            verbose=False,
            seed=1337,
            irl_cold_steps=4,
            irl_warm_steps=2,
            irl_horizon=12,
            semantic_fallback_max_rms_distance=(
                SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
            ),
        ), domain=domain)
        return BurritoHrcRunner(
            agent, executor, domain, seed=1337,
        ), executor, agent, domain

    def test_first_exposure_observes_then_later_exposures_assist(self):
        runner, _executor, agent, _domain = self._runner()
        first = runner.run_task(BurritoTask.create("steak", "plate_early"))
        second = runner.run_task(BurritoTask.create("steak", "protein_first"))
        self.assertEqual(first.mode, OBSERVE)
        self.assertEqual(second.mode, ASSIST)
        self.assertEqual((first.deliveries, second.deliveries), (1, 1))
        self.assertEqual((first.memory_age_delta, second.memory_age_delta), (1, 1))
        self.assertEqual(agent.demo_counter, 2)
        self.assertEqual(second.decisions[0].scheduled_actor, "human")
        self.assertEqual(len(second.observations), 8)
        self.assertGreater(
            second.low_level_ticks,
            len(second.observations),
        )
        with self.assertRaisesRegex(ValueError, "second observation"):
            runner.run_task(
                BurritoTask.create("steak", "rice_first"),
                force_mode=OBSERVE,
            )

    def test_wrong_robot_proposal_is_vetoed_and_robot_retries(self):
        runner, executor, agent, _domain = self._runner()
        runner.run_task(BurritoTask.create("steak", "plate_early"))

        wrong = fetch_and_stage_action("mushroom")
        agent.predict_actions = lambda *args, **kwargs: {wrong: 1.0}
        result = runner.run_task(BurritoTask.create(
            "steak", "protein_first",
        ))
        wrong_turns = [
            decision for decision in result.decisions
            if decision.scheduled_actor == "robot"
        ]
        self.assertTrue(wrong_turns)
        self.assertTrue(all(
            decision.human_corrected
            and decision.executed_by == "human_correction"
            and not decision.proposal_executed
            and decision.actual in decision.acceptable_actions
            for decision in wrong_turns
        ))
        for first, second in zip(result.decisions, result.decisions[1:]):
            if first.scheduled_actor == "robot" and first.human_corrected:
                self.assertEqual(second.scheduled_actor, "robot")
        self.assertNotIn(wrong, tuple(
            execution.action for execution in executor.execution_log
        ))
        self.assertEqual(result.deliveries, 1)

    def test_native_frames_and_actor_context_do_not_become_demo_steps(self):
        runner, _executor, agent, domain = self._runner()
        result = runner.run_task(BurritoTask.create("steak", "plate_jit"))
        self.assertEqual(len(result.observations), 8)
        self.assertEqual(agent.step_counter, 8)
        self.assertEqual(result.memory_age_delta, 1)
        self.assertGreater(result.passive_wait_ticks, 0)
        self.assertTrue(any(
            decision.passive_wait_ticks_before > 0
            for decision in result.decisions
        ))
        self.assertTrue(all(
            len(decision.primitive_ticks) == len(decision.primitive_calls)
            for decision in result.decisions
        ))
        self.assertTrue(all(
            observation.state[SCHEMA.task_offset + 8] == 0
            for observation in result.observations
        ))

    def test_cross_recipe_preference_transfer_is_grounded_and_exercised(self):
        runner, _executor, agent, _domain = self._runner()
        first, second, transfer = runner.run_stream((
            BurritoTask.create("steak", "protein_first"),
            BurritoTask.create("mushroom", "rice_first"),
            BurritoTask.create("steak", "rice_first"),
        ))
        self.assertEqual((first.mode, second.mode, transfer.mode), (
            OBSERVE, OBSERVE, ASSIST,
        ))
        self.assertEqual(transfer.invalid_predictions, 0)
        self.assertTrue(all(
            decision.predicted in decision.legal_actions
            for decision in transfer.decisions
            if decision.predicted is not None
        ))
        self.assertTrue(any(
            decision.prediction_stats.get("semantic_fallback_used", False)
            for decision in transfer.decisions
        ))
        self.assertEqual(agent.demo_counter, 3)


if __name__ == "__main__":
    unittest.main()
