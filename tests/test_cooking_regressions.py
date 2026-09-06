"""Regressions for two defects that only surfaced in a full cooking run.

Both live in code the project suite can import without the pinned Burrito
simulator, so they belong here rather than in ``burrito/tests`` -- which
``./hrc test`` never runs, and which is why neither defect was caught before a
165-cell evaluation had already spent hours reaching them.
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_ROOT = PROJECT_ROOT / "burrito" / "wrapper"
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))

from adaptive_hrc_burrito.catalog import get_recipe
from adaptive_hrc_burrito.domain import CookingDomainAdapter
from adaptive_hrc_burrito.options import macro_is_legal

from src.adaptive_agent import AdaptiveAgent
from src.memory import MemoryItem
from src.models import Settings


class _Object:
    def __init__(self, name, is_ready):
        self.name = name
        self.is_ready = is_ready


class _Player:
    held_object = None


class _State:
    """The three attributes ``macro_is_legal`` reads off a pinned state."""

    def __init__(self, objects):
        self.players = (_Player(), _Player())
        self.objects = {index: obj for index, obj in enumerate(objects)}


class ChopBoardCollisionRegression(unittest.TestCase):
    """``PREPARE_AND_STAGE_X`` must not be illegal once X is already chopped.

    The upstream chop primitive picks its board by ``_not_is_ready`` across
    *both* proteins, so in ``burrito_combo`` -- the one recipe that stages two
    at once -- chopping the steak also finishes the mushroom. Requiring the
    mushroom to still be unprepared made its task-graph node unreachable and
    deadlocked the episode after the option had, physically, already happened.
    """

    def test_already_chopped_protein_can_still_be_prepared(self):
        state = _State([_Object("chopped_mushroom", is_ready=True)])
        self.assertTrue(macro_is_legal(state, "PREPARE_AND_STAGE_MUSHROOM"))

    def test_preparing_one_protein_leaves_the_other_preparable(self):
        # The exact state the collision produces: both boards chopped through,
        # only the steak's task-graph node marked complete.
        state = _State([
            _Object("chopped_meat", is_ready=True),
            _Object("chopped_mushroom", is_ready=True),
        ])
        self.assertTrue(macro_is_legal(state, "PREPARE_AND_STAGE_MUSHROOM"))

    def test_unstaged_protein_still_cannot_be_prepared(self):
        """The relaxation drops readiness, not the staging requirement."""
        self.assertFalse(
            macro_is_legal(_State([]), "PREPARE_AND_STAGE_MUSHROOM")
        )
        self.assertFalse(
            macro_is_legal(
                _State([_Object("chopped_meat", is_ready=False)]),
                "PREPARE_AND_STAGE_MUSHROOM",
            )
        )


class AuditPrefixReplayRegression(unittest.TestCase):
    """Audit prefixes must replay under the recipe they were recorded from.

    ``state_from_actions`` starts at ``domain.initial_state()``, which on a
    multi-recipe domain is whichever recipe was played last. Replaying another
    recipe's ordering there raises on its first action, which left the
    active-only pruning audit unavailable -- and so recorded as *failed* -- for
    every heterogeneous and holdout cell.
    """

    @staticmethod
    def _entry(recipe_id, step):
        return MemoryItem(
            recipe_id=recipe_id,
            variant_id=f"{recipe_id}_v1",
            ordering=get_recipe(recipe_id).action_tokens,
            weight=1.0,
            added_step=step,
            added_cycle=0,
            last_seen_step=step,
        )

    def _agent_on(self, recipe_id):
        domain = CookingDomainAdapter()
        domain.begin_task(recipe_id)
        return AdaptiveAgent(Settings(seed=0), domain=domain)

    def test_prefixes_from_another_recipe_replay(self):
        agent = self._agent_on("burrito_combo")
        entries = [
            self._entry("overcooked_onion_onion_onion", 1),
            self._entry("burrito_steak_burrito", 2),
        ]
        seen = []
        result = agent._compare_policies(
            entries,
            lambda state, prefix: seen.append(state) or {},
            lambda state, prefix: {},
            max_prefixes=16,
            tolerance=0.05,
        )
        self.assertGreaterEqual(result["n_prefixes"], 4)
        self.assertEqual(len(seen), result["n_prefixes"])

    def test_every_recipe_contributes_a_prefix(self):
        """Deduplicating on the prefix alone dropped every recipe but the first.

        The empty prefix is shared by every ordering but denotes a different
        state under each recipe, so the key has to carry the recipe.
        """
        agent = self._agent_on("burrito_combo")
        recipes = ("overcooked_onion_onion_onion", "burrito_steak_burrito")
        states = []
        agent._compare_policies(
            [self._entry(recipe_id, index)
             for index, recipe_id in enumerate(recipes)],
            lambda state, prefix: states.append(state) or {},
            lambda state, prefix: {},
            max_prefixes=64,
            tolerance=0.05,
        )
        initial = {
            agent.domain.state_from_completed(recipe_id, ())
            for recipe_id in recipes
        }
        self.assertTrue(initial <= set(states))

    def test_the_domain_is_restored_after_an_audit(self):
        agent = self._agent_on("burrito_combo")
        agent._compare_policies(
            [self._entry("overcooked_onion_onion_onion", 1)],
            lambda state, prefix: {},
            lambda state, prefix: {},
            max_prefixes=8,
            tolerance=0.05,
        )
        self.assertEqual(agent.domain.recipe_id, "burrito_combo")


if __name__ == "__main__":
    unittest.main()
