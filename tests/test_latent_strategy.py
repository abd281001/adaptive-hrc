import unittest

import numpy as np

from src.latent_strategy import (
    LatentStrategyResidual,
    StrategyScore,
    action_role,
    fuse_strategy_residual,
    workflow_fingerprint,
)
from src.models import Settings


TOMATO = (
    "transfer (pot, from=storage, to=cooking_station)",
    "transfer (tomato, from=storage, to=prep_station)",
    "cut (tomato, prep_station)",
    "load (tomato, pot, cooking_station)",
    "turn_on (stove, cooking_station)",
)
MUSHROOM = tuple(action.replace("tomato", "mushroom") for action in TOMATO)
JIT = (TOMATO[1], TOMATO[2], TOMATO[0], TOMATO[3], TOMATO[4])


def _demo(actions):
    return [((), action) for action in actions] + [((), "stop")]


class LatentStrategyTests(unittest.TestCase):
    def setUp(self):
        self.model = LatentStrategyResidual(Settings(verbose=False))
        self.model.fit([_demo(TOMATO), _demo(MUSHROOM), _demo(JIT)])

    def test_representation_masks_ingredient_identity(self):
        self.assertEqual(action_role(TOMATO[1]), action_role(MUSHROOM[1]))
        np.testing.assert_allclose(
            workflow_fingerprint(TOMATO),
            workflow_fingerprint(MUSHROOM),
        )

    def test_fit_is_low_rank_and_has_no_recipe_or_preference_labels(self):
        stats = self.model.last_fit_stats
        self.assertEqual(stats["latent_strategy_prototypes"], 3)
        self.assertLessEqual(stats["latent_strategy_rank"], 3)
        self.assertEqual(self.model.codes.shape[0], 3)
        self.assertFalse(hasattr(self.model, "recipe_labels"))
        self.assertFalse(hasattr(self.model, "preference_labels"))

    def test_sparse_prefix_cannot_change_policy(self):
        score = self.model.score(TOMATO[:2], (TOMATO[2], TOMATO[3]))
        self.assertFalse(score.utilities)
        base = {TOMATO[2]: 0.6, TOMATO[3]: 0.4}
        fused, alpha = fuse_strategy_residual(base, score, 1.0)
        self.assertEqual(fused, base)
        self.assertEqual(alpha, 0.0)

    def test_residual_preserves_support_and_probability_mass(self):
        base = {"a": 0.55, "b": 0.30, "c": 0.15}
        score = StrategyScore(
            {"a": -1.0, "b": 1.0, "c": 0.0},
            confidence=0.8,
            coverage=1.0,
            observed_roles=4,
            observed_relations=6,
            neighbor_count=2,
        )
        fused, alpha = fuse_strategy_residual(base, score, 1.0)
        self.assertGreater(alpha, 0.0)
        self.assertEqual(set(fused), set(base))
        self.assertTrue(all(value > 0.0 for value in fused.values()))
        self.assertAlmostEqual(sum(fused.values()), 1.0)

    def test_correction_is_scored_at_pre_correction_progress(self):
        corrected = (*TOMATO[:3], TOMATO[3])
        score = self.model.score(
            corrected,
            (TOMATO[3], TOMATO[4]),
            decision_prefix=corrected[:-1],
        )
        self.assertEqual(score.observed_roles, 4)
        self.assertTrue(np.isfinite(score.confidence))
        self.assertGreaterEqual(score.coverage, 0.0)

    def test_role_alignment_never_returns_a_grounded_action(self):
        next_role, confidence, flops = self.model._aligned_next_role(
            (0, 2), (0, 2, 3, 4),
        )
        self.assertEqual(next_role, 3)
        self.assertGreater(confidence, 0.0)
        self.assertGreater(flops, 0.0)

    def test_heavy_decoder_reports_alignment_work(self):
        model = LatentStrategyResidual(Settings(
            verbose=False,
            latent_strategy_sequence_weight=0.5,
        ))
        model.fit([_demo(TOMATO), _demo(MUSHROOM), _demo(JIT)])
        model.score(TOMATO[:3], (TOMATO[3], TOMATO[4]))
        self.assertEqual(
            model.last_score_stats["latent_strategy_sequence_weight"], 0.5,
        )
        self.assertGreater(
            model.last_score_stats["latent_strategy_alignment_flops"], 0.0,
        )


if __name__ == "__main__":
    unittest.main()
