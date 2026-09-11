"""Zero-learning null arms for the cooking replication.

These are not comparison baselines in the Adaptive-HRC sense -- they are not
plausible deployable systems, and they are deliberately excluded from the
baseline-parity roster.  They exist to answer one question the deployable
roster cannot: how much of a reported accuracy is attributable to learning at
all, rather than to the task graph.

Roughly half of the robot's turns in this catalog have exactly one legal task
option, and under ``human_first`` the human resolves the widest frontier of
every episode before the robot chooses.  A fixed rule that never looks at the
human therefore scores far above chance.  Reporting that number alongside the
learned arms is what makes a Full-versus-baseline margin interpretable.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .catalog import RECIPES
from .domain import CookingDomainAdapter, TaskState


def _declaration_index(recipe_id: str) -> Mapping[str, int]:
    return {
        action.token: index
        for index, action in enumerate(RECIPES[recipe_id].actions)
    }


class CanonicalOrderAgent:
    """Always take the legal option that comes first in declaration order.

    Declaration order is the catalog's canonical strategy, so this arm is the
    "assume every user is the default user" policy: no memory, no model, no
    training, and no dependence on anything the human has ever done.

    It is built by composition rather than by subclassing ``AdaptiveAgent`` so
    that the claim "this arm does not learn" is structural rather than a matter
    of trusting an override.  The learner-facing surface below is exactly the
    surface ``CookingHrcRunner`` and the evaluator's record builders touch; the
    delegate supplies replay bookkeeping, recipe identity and commit stats so
    that routing, open-set separation and active-only audits stay comparable
    across arms, while ``predict_actions`` ignores all of it.
    """

    def __init__(self, settings: Any, *, domain: CookingDomainAdapter):
        from src.baselines import BehaviorCloningAgent

        # The delegate is a memory-only carrier: its predictor is never
        # consulted, and _fit_predictors below is a no-op, so no model is ever
        # fitted for this arm.
        self._delegate = BehaviorCloningAgent(settings, domain=domain)
        self._delegate._fit_predictors = self._no_fit  # type: ignore[assignment]
        # _estimate_flops falls back to a trajectory-shape formula whenever a
        # fit reported no statistics, which for this arm would bill it for
        # arithmetic it never performed.  Its replay bookkeeping still runs and
        # its build wall time is still real; only the fit is empty.
        self._delegate._estimate_flops = self._no_flops  # type: ignore[assignment]
        # AdaptiveAgent.observe falls back to its own predictor when a caller
        # hands it no precomputed distribution.  Routing that back here is what
        # makes "this arm never consults a learned model" true on every path
        # rather than only on the runner's.
        self._delegate.predict_actions = self.predict_actions  # type: ignore[assignment]
        self.domain = domain

    @property
    def domain(self) -> CookingDomainAdapter:
        """One adapter, shared with the delegate.

        ``_frozen_probe`` freezes the agent (which deep-copies its adapter) and
        then restores ``agent.domain``, ``agent.maxent.domain`` and
        ``agent.cloner.domain``.  Holding a separate ``domain`` attribute here
        would leave the delegate pointing at the probe's copy afterwards while
        this object pointed at the real one, so the two are the same slot.
        """
        return self._delegate.domain

    @domain.setter
    def domain(self, value: CookingDomainAdapter) -> None:
        self._delegate.domain = value

    @staticmethod
    def _no_fit(*args: Any, **kwargs: Any) -> None:
        """Never fit. Retrain bookkeeping still runs; no weights are learned."""
        return None

    @staticmethod
    def _no_flops(*args: Any, **kwargs: Any) -> float:
        """No fit arithmetic happened, so none is billed."""
        return 0.0

    # -- learner surface consumed by the runner ---------------------------
    def __getattr__(self, name: str) -> Any:
        # Everything not overridden below (replay, library, demo_counter,
        # retrain_events, last_commit_stats, start_demo, end_demo, observe,
        # maxent, ...) is the delegate's.
        if name == "_delegate":
            # Reached only before __init__ binds it; recursing here instead
            # would turn any early attribute access into a RecursionError.
            raise AttributeError(name)
        return getattr(self._delegate, name)

    def predictor_name(self) -> str:
        return "canonical_declaration_order"

    def irl_feature_name(self) -> str:
        return "not_applicable"

    def predict_actions(
        self,
        prefix: Optional[Sequence[str]] = None,
        *,
        state: Any = None,
        actor_id: int = 0,
        action_universe: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Rank the legal options by declaration index, highest mass first.

        A full ranking rather than a point mass, so top-k is defined the same
        way it is for every other arm.
        """
        encoded = (
            self.domain.state_key(state, actor_id=actor_id)
            if state is not None
            else self.domain.state_from_actions(
                list(prefix) if prefix is not None else list(self.current_prefix)
            )
        )
        recipe_id = TaskState.decode(encoded).recipe_id
        order = _declaration_index(recipe_id)
        # The runner always supplies the shared mask.  Without one, derive the
        # same mask from the domain rather than guessing: this arm has no model
        # to fall back on.
        universe = (
            action_universe if action_universe is not None
            else self.domain.legal_actions(
                encoded, RECIPES[recipe_id].action_tokens,
            )
        )
        legal = [
            action for action in dict.fromkeys(map(str, universe))
            if action in order
        ]
        if not legal:
            self._delegate._set_policy_stats(None, None, "canonical_order_empty")
            return {}
        legal.sort(key=lambda action: order[action])
        # Strictly decreasing, normalized: argmax is the first-declared legal
        # option and the tail preserves declaration order under rank_actions.
        scores = [1.0 / (rank + 1) for rank in range(len(legal))]
        total = sum(scores)
        distribution = {
            action: score / total for action, score in zip(legal, scores)
        }
        confidence, entropy, _margin = self._delegate._prediction_stats(distribution)
        self._delegate._set_policy_stats(confidence, entropy, "canonical_order")
        return distribution

    def rank_actions(
        self, distribution: Mapping[str, float], k: int = 1,
    ) -> list[str]:
        return self._delegate.rank_actions(distribution, k=k)

    def policy_stats(self) -> Dict[str, Any]:
        stats = dict(self._delegate.policy_stats())
        stats.update({
            "predictor": self.predictor_name(),
            "semantic_fallback_used": False,
            "latent_strategy_used": False,
        })
        return stats


NULL_AGENTS: Mapping[str, Any] = {"canonical_order": CanonicalOrderAgent}


__all__ = ["NULL_AGENTS", "CanonicalOrderAgent"]
