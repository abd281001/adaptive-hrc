"""Tests for the retention/component ablation knobs added to Settings.

Every knob defaults to the deployed configuration, so the first thing these
tests establish is that an unmodified ``Settings`` still produces exactly the
policy the main evaluation already reported. The rest check that each mode
actually moves the mechanism it names, since an ablation arm that silently
failed to apply would produce a plausible-looking null result.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import replace

from src.adaptive_agent import AdaptiveAgent, FULL_TRAIN_POLICY, TrainPolicy
from src.memory import ReplayMemory
from src.models import DEFAULT_SETTINGS, Settings

KEY_A = ("R1", "variant_a")
KEY_B = ("R1", "variant_b")
KEY_C = ("R2", "variant_a")


def _memory(**overrides) -> ReplayMemory:
    return ReplayMemory(replace(DEFAULT_SETTINGS, **overrides))


def _with_gaps(memory: ReplayMemory, key, gaps) -> ReplayMemory:
    for step, gap in enumerate(gaps):
        memory._record_reuse_gap(key, gap, step=step)
    return memory


class DefaultsAreUnchangedTests(unittest.TestCase):
    """The knobs must be inert until an arm sets them."""

    def test_default_settings_select_the_deployed_policy(self):
        self.assertEqual(DEFAULT_SETTINGS.retention_policy, "adaptive")
        self.assertEqual(DEFAULT_SETTINGS.horizon_estimator, "hierarchical")
        self.assertEqual(DEFAULT_SETTINGS.pair_adaptation, "asymmetric")
        self.assertEqual(DEFAULT_SETTINGS.pin_mode, "latest")

    def test_default_memory_matches_the_explicit_adaptive_construction(self):
        derived = ReplayMemory(DEFAULT_SETTINGS)
        explicit = ReplayMemory(DEFAULT_SETTINGS, policy="adaptive")
        self.assertEqual(derived.policy, explicit.policy)
        gaps = [4, 9, 30, 7]
        self.assertEqual(
            _with_gaps(derived, KEY_A, gaps).horizon(KEY_A),
            _with_gaps(explicit, KEY_A, gaps).horizon(KEY_A),
        )

    def test_default_scheduler_is_the_class_policy_object(self):
        self.assertEqual(TrainPolicy.from_settings(DEFAULT_SETTINGS), FULL_TRAIN_POLICY)
        # Identity, not just equality: a subclass that pins its own scheduler
        # must keep that object under default settings.
        self.assertIs(
            AdaptiveAgent._resolve_train_policy(DEFAULT_SETTINGS),
            AdaptiveAgent.RETRAIN_POLICY,
        )

    def test_agent_builds_adaptive_memory_by_default(self):
        self.assertEqual(AdaptiveAgent(settings=DEFAULT_SETTINGS).replay.policy, "adaptive")


class RetentionPolicyTests(unittest.TestCase):
    def test_retention_policy_reaches_a_plain_agent(self):
        agent = AdaptiveAgent(settings=replace(DEFAULT_SETTINGS, retention_policy="none"))
        self.assertEqual(agent.replay.policy, "none")

    def test_retain_all_never_prunes(self):
        memory = _memory(retention_policy="none")
        memory.register("R1", "variant_a", ("a",), now=0, cycle=0, pin_latest=False)
        # Far beyond any horizon the adaptive policy would grant.
        self.assertEqual(memory.step(now=500, cycle=1), [])
        self.assertIn(KEY_A, memory.active)


class HorizonEstimatorTests(unittest.TestCase):
    def test_constant_ignores_recurrence_evidence(self):
        memory = _memory(horizon_estimator="constant", constant_grace_horizon=40)
        _with_gaps(memory, KEY_A, [2, 3, 2])
        _with_gaps(memory, KEY_C, [70, 80, 90])
        self.assertEqual(memory.horizon(KEY_A), 40.0)
        self.assertEqual(memory.horizon(KEY_C), 40.0)

    def test_constant_respects_the_minimum_grace_floor(self):
        memory = _memory(horizon_estimator="constant", constant_grace_horizon=1, min_grace=6)
        self.assertEqual(memory.horizon(KEY_A), 6.0)

    def test_hierarchical_uses_parents_and_pair_only_does_not(self):
        # KEY_B has no evidence of its own, so only the hierarchical estimator
        # can borrow its sibling's and its recipe's long gaps.
        hierarchical = _memory(horizon_estimator="hierarchical", initial_grace=10)
        pair_only = _memory(horizon_estimator="pair_only", initial_grace=10)
        for memory in (hierarchical, pair_only):
            _with_gaps(memory, KEY_A, [40, 45, 50, 44, 47])
            _with_gaps(memory, KEY_C, [41, 46, 49, 43, 48])
        self.assertGreater(hierarchical.horizon(KEY_B), pair_only.horizon(KEY_B))
        self.assertEqual(pair_only.horizon(KEY_B), 10.0)

    def test_shuffled_preserves_the_horizon_population(self):
        gaps = {KEY_A: [2, 3, 2, 3, 2], KEY_B: [30, 33, 31, 32, 30], KEY_C: [12, 14, 13, 12, 15]}
        hierarchical = _memory(horizon_estimator="hierarchical")
        shuffled = _memory(horizon_estimator="shuffled")
        for memory in (hierarchical, shuffled):
            for key, values in gaps.items():
                _with_gaps(memory, key, values)
        reference = sorted(hierarchical.horizon(key) for key in gaps)
        reassigned = sorted(shuffled.horizon(key) for key in gaps)
        # Same set of horizons; a different pair holds each one.
        self.assertEqual(reference, reassigned)
        self.assertNotEqual(
            [hierarchical.horizon(key) for key in gaps],
            [shuffled.horizon(key) for key in gaps],
        )

    def test_shuffled_never_reads_a_pair_its_own_window(self):
        memory = _memory(horizon_estimator="shuffled")
        for key in (KEY_A, KEY_B, KEY_C):
            _with_gaps(memory, key, [5, 6, 7])
        for key in (KEY_A, KEY_B, KEY_C):
            self.assertNotEqual(memory._shuffled_evidence_key(key), key)

    def test_shuffled_is_a_no_op_with_a_single_tracked_pair(self):
        memory = _memory(horizon_estimator="shuffled")
        _with_gaps(memory, KEY_A, [5, 6, 7])
        self.assertEqual(memory._shuffled_evidence_key(KEY_A), KEY_A)

    def test_estimator_is_reported_on_every_horizon(self):
        for estimator in ("hierarchical", "pair_only", "constant", "shuffled"):
            memory = _memory(horizon_estimator=estimator)
            _with_gaps(memory, KEY_A, [4, 5])
            self.assertEqual(memory.horizon_stats(KEY_A)["horizon_estimator"], estimator)


class PairAdaptationTests(unittest.TestCase):
    def test_symmetric_shortens_faster_than_asymmetric(self):
        # Short gaps against a long prior: the deployed policy resists
        # shortening, the symmetric control does not.
        short_gaps = [3, 3, 4]
        asymmetric = _with_gaps(_memory(initial_grace=60), KEY_A, short_gaps)
        symmetric = _with_gaps(
            _memory(initial_grace=60, pair_adaptation="symmetric"), KEY_A, short_gaps,
        )
        self.assertLess(symmetric.horizon(KEY_A), asymmetric.horizon(KEY_A))

    def test_symmetric_matches_asymmetric_when_lengthening(self):
        # Both use the linear upward response, so an arm that only ever sees
        # lengthening evidence is not perturbed by this factor.
        long_gaps = [70, 75, 80]
        asymmetric = _with_gaps(_memory(initial_grace=10), KEY_A, long_gaps)
        symmetric = _with_gaps(
            _memory(initial_grace=10, pair_adaptation="symmetric"), KEY_A, long_gaps,
        )
        self.assertEqual(symmetric.horizon(KEY_A), asymmetric.horizon(KEY_A))

    def test_pooling_mode_names_the_active_response(self):
        symmetric = _with_gaps(
            _memory(initial_grace=60, pair_adaptation="symmetric"), KEY_A, [3, 3, 4],
        )
        self.assertEqual(
            symmetric.horizon_stats(KEY_A)["pair_pooling_mode"], "linear_symmetric",
        )


class PinModeTests(unittest.TestCase):
    def _two_variants(self, **overrides) -> ReplayMemory:
        memory = _memory(**overrides)
        memory.register("R1", "variant_a", ("a",), now=0, cycle=0)
        memory.register("R1", "variant_b", ("b",), now=1, cycle=0)
        return memory

    def test_latest_pins_exactly_one_variant_per_recipe(self):
        memory = self._two_variants()
        self.assertEqual(memory.latest_keys, {KEY_B})
        self.assertEqual(memory.latest_by_recipe["R1"], "variant_b")

    def test_recent_set_keeps_a_recent_sibling_pinned(self):
        memory = self._two_variants(pin_mode="recent_set", pin_window=8)
        self.assertEqual(memory.latest_keys, {KEY_A, KEY_B})
        # The newest variant is still the recipe's latest for every other
        # invariant that reads it.
        self.assertEqual(memory.latest_by_recipe["R1"], "variant_b")
        self.assertEqual(memory.active[KEY_A].weight, 1.0)

    def test_recent_set_releases_a_sibling_outside_the_window(self):
        memory = _memory(pin_mode="recent_set", pin_window=2)
        memory.register("R1", "variant_a", ("a",), now=0, cycle=0)
        memory.register("R1", "variant_b", ("b",), now=10, cycle=0)
        self.assertEqual(memory.latest_keys, {KEY_B})

    @staticmethod
    def _age_out(memory: ReplayMemory) -> None:
        """Advance past the horizon and run the full post-grace decay ramp.

        One ``step`` decrements an overdue entry by ``1 / prune_delay``, so a
        variant only leaves active fitting after that many demonstrations.
        """
        for offset in range(int(DEFAULT_SETTINGS.prune_delay) + 1):
            memory.step(now=500 + offset, cycle=1)

    def test_recent_set_protects_a_sibling_from_decay(self):
        memory = self._two_variants(pin_mode="recent_set", pin_window=8)
        self._age_out(memory)
        self.assertIn(KEY_A, memory.active)
        self.assertEqual(memory.active[KEY_A].weight, 1.0)

    def test_latest_mode_lets_a_superseded_variant_decay(self):
        memory = self._two_variants()
        self._age_out(memory)
        self.assertNotIn(KEY_A, memory.active)
        self.assertIn(KEY_A, memory.pruned)


class SchedulerTests(unittest.TestCase):
    def test_removal_only_warm_fits_by_default(self):
        self.assertEqual(FULL_TRAIN_POLICY.decide(0, 2, False, 1)[0], "warm")

    def test_removals_can_be_made_to_advance_the_cold_counter(self):
        policy = TrainPolicy.from_settings(
            replace(DEFAULT_SETTINGS, retrain_cold_counts_removals=True),
        )
        self.assertEqual(policy.cold_threshold_basis, "additions_and_removals")
        self.assertEqual(policy.decide(0, 2, False, 1)[0], "cold")

    def test_weight_only_change_skips_by_default_and_can_warm_fit(self):
        self.assertEqual(FULL_TRAIN_POLICY.decide(0, 0, True, 0)[0], "skip")
        policy = TrainPolicy.from_settings(
            replace(DEFAULT_SETTINGS, retrain_warm_on_weight_change=True),
        )
        self.assertEqual(policy.decide(0, 0, True, 0)[0], "warm")

    def test_non_default_scheduler_overrides_the_class_policy(self):
        settings = replace(DEFAULT_SETTINGS, retrain_warm_on_weight_change=True)
        self.assertEqual(
            AdaptiveAgent._resolve_train_policy(settings),
            TrainPolicy.from_settings(settings),
        )


class ValidationTests(unittest.TestCase):
    def test_unknown_modes_are_rejected(self):
        for field, value in (
            ("retention_policy", "sometimes"),
            ("horizon_estimator", "magic"),
            ("pair_adaptation", "lopsided"),
            ("pin_mode", "everything"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(DEFAULT_SETTINGS, **{field: value})

    def test_negative_horizon_and_window_are_rejected(self):
        for field in ("constant_grace_horizon", "pin_window"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(DEFAULT_SETTINGS, **{field: -1})
        with self.assertRaises(ValueError):
            replace(DEFAULT_SETTINGS, retrain_cold_after=0)


class AblationArmTests(unittest.TestCase):
    """The arm registry must still express the designs it replaced."""

    def _settings(self, arm):
        return replace(DEFAULT_SETTINGS, **dict(arm.overrides))

    def test_component_arms_move_only_their_declared_component(self):
        from src.ablations import arms_by_name

        expected = {
            "full_no_pin": (False, True, True),
            "full_no_semantic_fallback": (True, False, True),
            "full_no_latent_residual": (True, True, False),
        }
        arms = arms_by_name()
        for name, want in expected.items():
            with self.subTest(arm=name):
                settings = self._settings(arms[name])
                self.assertEqual(
                    (settings.pin_latest, settings.semantic_fallback_enabled,
                     settings.latent_strategy_enabled), want,
                )
                # Retention must not move; that is the other group's factor.
                self.assertEqual(settings.retention_policy, "adaptive")
                self.assertEqual(settings.horizon_estimator, "hierarchical")

    def test_retention_arms_hold_predictor_support_at_full(self):
        from src.ablations import ARMS

        for arm in ARMS:
            if "retention" not in arm.groups:
                continue
            with self.subTest(arm=arm.name):
                settings = self._settings(arm)
                self.assertTrue(settings.semantic_fallback_enabled)
                self.assertTrue(settings.latent_strategy_enabled)

    def test_every_arm_differs_from_the_reference_it_is_compared_against(self):
        """No arm may be Full under another name; Full comes from the roster."""
        from src.ablations import ARMS

        for arm in ARMS:
            with self.subTest(arm=arm.name):
                if arm.agent == "full" and arm.route != "local":
                    self.assertNotEqual(dict(arm.overrides), {})
                    self.assertNotEqual(self._settings(arm), DEFAULT_SETTINGS)

    def test_memory_group_forms_a_factorial_with_matched_maxent_support(self):
        from src.ablations import arms_by_name, group_arms

        arms = arms_by_name()
        # Two cells come from the deployable roster: Full and BC.
        self.assertEqual(
            set(group_arms("memory")),
            {"full", "bc", "bc_adaptive", "maxent_retain_all"},
        )
        levels = {
            (arms[name].facets["predictor"], arms[name].facets["memory"])
            for name in ("bc_adaptive", "maxent_retain_all")
        }
        self.assertEqual(
            levels,
            {("behavior_cloning", "adaptive_pinned"), ("maxent", "retain_all")},
        )
        # The repair: the MaxEnt retain-all cell keeps Full's predictor
        # support, so the memory factor is not confounded with it.
        settings = self._settings(arms["maxent_retain_all"])
        self.assertTrue(settings.semantic_fallback_enabled)
        self.assertTrue(settings.latent_strategy_enabled)
        self.assertEqual(settings.retention_policy, "none")
        self.assertFalse(settings.pin_latest)


if __name__ == "__main__":
    unittest.main()
