"""Bounded mechanism diagnostic for the container-axis transfer failure.

The oracle comparison shows that a future-informed replay selection answers the
first container-first target opening (29/36) where the deployed system does not
(0/36).  It does not say *why*, and there are at least three incompatible
explanations:

* **forgetting** -- the source's container-first variant has decayed or been
  pruned by the time the target's first exposure arrives;
* **interference** -- it is still there, but obsolete target-specific variants
  outvote it in the fit;
* **fallback suppression** -- it is still there and the semantic-distance
  fallback that would surface it is never consulted, because
  ``MaxEntIrl.action_distribution`` returns ``exact_policy`` and floors every
  unlearned candidate as soon as *any* candidate has a learned Q-value at that
  exact state (``src/models.py``).  A single retained variant that visited the
  opening state is enough to do that.

This replays one seed's holdout schedule up to the first container-first target
exposure, dumps the active set with weights, pins and learned-action support at
the opening state, and then re-predicts that one decision under controlled
removals with a refit after each.  The removals are chosen so the three
explanations predict different outcomes:

The four conditions separate the explanations by what they move.  ``gate``
(``exact_policy`` versus ``fallback_used``) says whether the fallback was
consulted at all; ``expected_action_probability`` says how much mass the
container-first action received once it was.  Forgetting predicts the source
axis variant is absent or decayed and that removing it changes nothing;
suppression predicts removing the obsolete target variant flips the gate and
lifts that probability off zero; and whether either flips the *argmax* is a
separate question from whether it flips the gate -- reading only the top-1
would have missed a large probability shift that did not change the decision.

It changes no artifact and replays one seed's schedule once, rather than the
264 cells of a full evaluation.  Its numbers describe the original execution:
candidate lists are still the ground-truth-conditioned ones, so it diagnoses a
mechanism rather than producing a corrected result, and one seed is one
decision -- run every seed before generalising.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .catalog import (
    STRATA,
    applicable_preferences,
    get_preference,
    get_recipe,
    get_stratum,
    preference_order,
)
from .domain import CookingDomainAdapter
from .evaluation import _atomic_json, _build_agent, _utc_now, load_config
from .ladder import CONTAINER_FIRST_PREFERENCE, generate_ladder
from .protocol import ASSIST, CookingHrcRunner, CookingTask
from .runtime import BurritoRuntime
from .task_graph import CookingTaskGraph


PROBE_SCHEMA_VERSION = 1


def _preference_of_ordering(
    recipe_id: str, ordering: Sequence[str],
) -> Optional[str]:
    """Name the preference a stored variant encodes, by its realized ordering.

    Variant ids are the learner's own labels, so the ordering is the only
    stable way back to a catalog preference.
    """
    recipe = get_recipe(recipe_id)
    target = tuple(map(str, ordering))
    for name in applicable_preferences(recipe_id):
        if preference_order(recipe, get_preference(name)) == target:
            return name
    return None


@dataclass(frozen=True)
class VariantView:
    key: Tuple[str, str]
    catalog_recipe: Optional[str]
    preference: Optional[str]
    weight: float
    pinned: bool
    added_step: int
    last_seen_step: int
    transition_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "learner_key": list(self.key),
            "catalog_recipe": self.catalog_recipe,
            "preference": self.preference,
            "weight": self.weight,
            "latest_pinned": self.pinned,
            "added_step": self.added_step,
            "last_seen_step": self.last_seen_step,
            "transition_count": self.transition_count,
        }


def _view_active(agent: Any, recipe_by_learner: Mapping[str, str]) -> List[VariantView]:
    views = []
    for key, item in agent.replay.active.items():
        catalog = recipe_by_learner.get(key[0])
        views.append(VariantView(
            key=(str(key[0]), str(key[1])),
            catalog_recipe=catalog,
            preference=(
                _preference_of_ordering(catalog, item.ordering)
                if catalog else None
            ),
            weight=float(item.weight),
            pinned=key in agent.replay.latest_keys,
            added_step=int(item.added_step),
            last_seen_step=int(item.last_seen_step),
            transition_count=len(item.transitions),
        ))
    return sorted(views, key=lambda view: (str(view.catalog_recipe), str(view.preference)))


def _probe_opening(
    agent: Any,
    domain: CookingDomainAdapter,
    task: CookingTask,
) -> Dict[str, Any]:
    """Score the opening decision without mutating the agent."""
    graph = CookingTaskGraph.create(task.recipe_id)
    legal = graph.frontier([])
    truth = get_preference(task.preference)
    from .task_graph import CookingPreferencePolicy

    expected = CookingPreferencePolicy.create(task.preference).choose_action(
        legal, graph,
    )
    domain.begin_task(task.recipe_id)
    state = domain.state_from_completed(task.recipe_id, [])
    agent.set_frozen(True)
    try:
        distribution = dict(agent.predict_actions(
            (), state=state, action_universe=legal,
        ))
        ranked = tuple(agent.rank_actions(distribution, k=3))
        stats = dict(agent.policy_stats())
    finally:
        agent.set_frozen(False)
        agent.domain = domain
        if hasattr(agent, "maxent"):
            agent.maxent.domain = domain
        if hasattr(agent, "cloner"):
            agent.cloner.domain = domain
    predicted = ranked[0] if ranked else None
    return {
        "candidates": list(legal),
        "expected_action": expected,
        "predicted_action": predicted,
        "correct": predicted == expected,
        "expected_action_probability": float(distribution.get(expected, 0.0)),
        "distribution": {
            action: float(probability)
            for action, probability in sorted(distribution.items())
        },
        # These four separate the hypotheses.  They come from
        # ``MaxEntIrl.last_prediction_stats``, which the agent's freeze flag
        # does not gate, unlike the ``reason``/``predictor`` fields.
        "exact_state_learned_action_count": stats.get(
            "exact_state_learned_action_count"
        ),
        "semantic_fallback_attempted": bool(
            stats.get("semantic_fallback_attempted", False)
        ),
        "semantic_fallback_used": bool(stats.get("semantic_fallback_used", False)),
        "semantic_gate_outcome": stats.get("semantic_gate_outcome"),
        "policy_stats": {
            key: value for key, value in sorted(stats.items())
            if isinstance(value, (int, float, str, bool)) or value is None
        },
    }


def _removal_sets(
    views: Sequence[VariantView], target_recipe: str,
) -> Dict[str, Tuple[Tuple[str, str], ...]]:
    """The four conditions, defined over the active set at the decision."""
    obsolete_target = tuple(
        view.key for view in views
        if view.catalog_recipe == target_recipe
        and view.preference != CONTAINER_FIRST_PREFERENCE
    )
    source_axis = tuple(
        view.key for view in views
        if view.catalog_recipe != target_recipe
        and view.preference == CONTAINER_FIRST_PREFERENCE
    )
    return {
        "unmodified": (),
        "minus_obsolete_target_variants": obsolete_target,
        "minus_source_axis_variants": source_axis,
        "minus_both": obsolete_target + source_axis,
    }


def _apply_removal(agent: Any, keys: Sequence[Tuple[str, str]]) -> int:
    removed = 0
    for recipe_id, variant_id in keys:
        if (recipe_id, variant_id) in agent.replay.active:
            agent.replay.discard(recipe_id, variant_id, allow_latest=True)
            removed += 1
    if removed:
        # Controlled refit: the same path a removal takes during deployment, so
        # the counterfactual differs from the baseline only in its active set.
        agent._retrain()
    return removed


def run_probe(
    *,
    config_path: str | Path,
    seed: int,
    scenario: str = "holdout",
    arm: str = "full",
    stratum: Optional[str] = None,
    stop_after: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    settings_kwargs = dict(config.get("settings", {}))
    from src.models import Settings

    recipe_ids = tuple(map(str, config["recipe_ids"]))
    tasks, audit = generate_ladder(
        seed=seed, scenario=scenario, recipe_ids=recipe_ids, return_audit=True,
    )
    # Without a stratum this finds the earliest first-exposure in the schedule,
    # which on every paper seed is an Overcooked target.  The natively executed
    # Burrito targets come later and are a different decision, so they have to
    # be asked for explicitly rather than assumed covered.
    if stratum is not None and stratum not in STRATA:
        raise SystemExit(f"unknown stratum {stratum!r}; expected one of {STRATA}")
    first = next(
        (
            index for index, task in enumerate(tasks)
            if task.holdout_target
            and task.preference == CONTAINER_FIRST_PREFERENCE
            and task.exposure_after_change == 1
            and (stratum is None or get_stratum(task.recipe_id) == stratum)
        ),
        None,
    )
    if first is None:
        raise SystemExit(
            f"seed {seed} {scenario} has no first container-first target "
            f"exposure" + (f" in stratum {stratum}" if stratum else "")
        )
    limit = first if stop_after is None else min(first, int(stop_after))

    runtime = BurritoRuntime.discover()
    domain = CookingDomainAdapter()
    agent = _build_agent(arm, Settings(seed=seed, verbose=False, **settings_kwargs), domain)
    runner = CookingHrcRunner(
        agent, runtime, domain,
        horizon=int(config.get("horizon", 1800)),
        planner_seed=int(config.get("planner_seed", 11)),
        top_k=int(config.get("top_k", 3)),
        lead_actor_policy=str(config.get("lead_actor_policy", "human_first")),
    )
    for task in tasks[:limit]:
        runner.run_task(task)

    target = tasks[first]
    recipe_by_learner = {
        learner: catalog
        for catalog, learner in runner.learner_recipe_by_task.items()
    }
    views = _view_active(agent, recipe_by_learner)
    baseline = _probe_opening(agent, domain, target)
    conditions = _removal_sets(views, target.recipe_id)

    results = {}
    for name, keys in conditions.items():
        if name == "unmodified":
            results[name] = {
                "removed_variant_count": 0,
                "removed": [],
                **baseline,
            }
            continue
        # Deep-copy the replayed learner rather than replaying the schedule
        # again.  Four replays meant four pinned Overcooked runtimes and about
        # two thousand physical episodes, which exhausted the process; the
        # learner state is what the conditions differ in, and the agent holds
        # no simulator handle.
        probe_agent = copy.deepcopy(agent)
        probe_agent.domain = domain
        if hasattr(probe_agent, "maxent"):
            probe_agent.maxent.domain = domain
        if hasattr(probe_agent, "cloner"):
            probe_agent.cloner.domain = domain
        removed = _apply_removal(probe_agent, keys)
        results[name] = {
            "removed_variant_count": removed,
            "removed": [list(key) for key in keys],
            **_probe_opening(probe_agent, domain, target),
        }

    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "kind": "mechanism_diagnostic_on_original_execution",
        "generated_at": _utc_now(),
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "stratum_requested": stratum,
        "target_stratum": get_stratum(tasks[first].recipe_id),
        "config_path": str(config["_config_path"]),
        "first_transfer_event_index": first,
        "episodes_replayed": limit,
        "target": {
            "recipe_id": target.recipe_id,
            "preference": target.preference,
            "strategy": target.strategy,
            "phase": target.phase,
            "exposure_after_change": target.exposure_after_change,
        },
        "grace": {
            "initial_grace": int(settings_kwargs.get("initial_grace", 0)),
            "min_grace": int(settings_kwargs.get("min_grace", 0)),
            "episodes_since_previous_axis_episode": next(
                (
                    first - index for index in range(first - 1, -1, -1)
                    if tasks[index].preference == CONTAINER_FIRST_PREFERENCE
                ),
                None,
            ),
        },
        "active_variants_before_first_target_exposure": [
            view.as_dict() for view in views
        ],
        "active_variant_count": len(views),
        "pruned_variant_count": len(agent.replay.pruned),
        "source_axis_variant_present": any(
            view.catalog_recipe != target.recipe_id
            and view.preference == CONTAINER_FIRST_PREFERENCE
            for view in views
        ),
        "obsolete_target_variant_count": sum(
            1 for view in views
            if view.catalog_recipe == target.recipe_id
            and view.preference != CONTAINER_FIRST_PREFERENCE
        ),
        "conditions": results,
        "caveat": (
            "Diagnostic on the original execution. Candidate lists are the "
            "ground-truth-conditioned ones the saved run produced, and this is "
            "one seed and one scenario. It separates forgetting from "
            "interference and fallback suppression; it is not a corrected "
            "result and does not estimate an effect size."
        ),
        "ladder_audit": {
            key: audit[key] for key in (
                "holdout_target_recipe_ids", "holdout_introduction_phase",
            ) if key in audit
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m adaptive_hrc_burrito.mechanism_probe",
        description=(
            "Replay one holdout schedule to its first container-first target "
            "exposure and diagnose why the opening prediction fails."
        ),
    )
    parser.add_argument("--config", default="burrito/configs/full.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--scenario", default="holdout")
    parser.add_argument("--arm", default="full")
    parser.add_argument(
        "--stratum", default=None, choices=list(STRATA),
        help=(
            "diagnose the first axis-target exposure in this stratum; without "
            "it the earliest in the schedule is used, which is always an "
            "Overcooked target"
        ),
    )
    parser.add_argument(
        "--stop-after", type=int, default=None,
        help="replay at most this many episodes (debugging aid)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    faulthandler.enable()

    report = run_probe(
        config_path=args.config, seed=args.seed, scenario=args.scenario,
        arm=args.arm, stratum=args.stratum, stop_after=args.stop_after,
    )
    suffix = f"__{args.stratum}" if args.stratum else ""
    out = Path(args.out) if args.out else Path(
        f"eval_results/cooking/mechanism_probe__{args.arm}__{args.scenario}"
        f"{suffix}__seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(out, report)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
