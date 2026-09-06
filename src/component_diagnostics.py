"""Summarize permanent one-step semantic/latent counterfactual diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .evaluation import _paired_bootstrap_ci, _write_json


TRANSFER_CELLS = {
    "seen_recipe_preference_seen_elsewhere",
    "seen_recipe_new_preference",
}
SCOPES = {
    "all": None,
    "direct_retrieval": {"direct_retrieval"},
    "transfer_combined": TRANSFER_CELLS,
    "preference_seen_elsewhere": {"seen_recipe_preference_seen_elsewhere"},
    "new_preference": {"seen_recipe_new_preference"},
}
CONTRASTS: Mapping[str, Tuple[str, str, str]] = {
    # name: (candidate, reference, interpretation)
    "current_semantic_attribution": (
        "semantic_current", "maxent_only", "current semantic minus MaxEnt",
    ),
    "current_latent_attribution": (
        "deployed_current", "semantic_current", "current latent minus current semantic",
    ),
    "current_components_combined": (
        "deployed_current", "maxent_only", "deployed semantic and latent minus MaxEnt",
    ),
    "semantic_completion_component": (
        "semantic_completion", "semantic_current", "semantic completion with latent off",
    ),
    "semantic_completion_intervention": (
        "semantic_completion_current_gate", "deployed_current", "semantic completion versus deployed current",
    ),
    "latent_relaxed_attribution": (
        "latent_relaxed", "semantic_current", "relaxed latent minus current semantic",
    ),
    "latent_relaxed_intervention": (
        "latent_relaxed", "deployed_current", "relaxed latent versus deployed current",
    ),
    "latent_two_role_attribution": (
        "latent_two_role", "semantic_current", "two-role latent minus current semantic",
    ),
    "latent_two_role_intervention": (
        "latent_two_role", "deployed_current", "two-role latent versus deployed current",
    ),
    "two_role_floor_increment": (
        "latent_two_role", "latent_relaxed", "two-role floor beyond other latent gate relaxation",
    ),
}


def _turn_rows(run_dir: Path) -> Iterable[Mapping[str, Any]]:
    paths = sorted(run_dir.glob("scenarios/*/seeds/*/tables/turns.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no turn tables found under {run_dir}")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("baseline") == "full":
                    yield row


def _scope_matches(row: Mapping[str, Any], cells: set[str] | None) -> bool:
    return cells is None or row.get("transfer_cell_before") in cells


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _cluster_interval(values: Sequence[float], *, key: str) -> Dict[str, float]:
    interval = _paired_bootstrap_ci(values, key=key)
    interval["bootstrap_positive_fraction"] = interval.pop(
        "bootstrap_full_better_fraction"
    )
    return interval


def summarize(run_dir: Path) -> Dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not bool((manifest.get("config") or {}).get("component_counterfactuals")):
        raise ValueError(f"{run_dir} is not a component-counterfactual run")
    by_seed: Dict[Tuple[str, str, str, int], list[float]] = defaultdict(list)
    absolute: Dict[Tuple[str, str, str, int], list[float]] = defaultdict(list)
    scope_counts: Counter[str] = Counter()
    latent_gates: Counter[str] = Counter()
    semantic_gates: Counter[str] = Counter()
    relation_rows: Dict[int, list[Tuple[int, float, float]]] = defaultdict(list)
    conditional: Dict[Tuple[str, str, str, int], list[float]] = defaultdict(list)
    conditional_counts: Counter[str] = Counter()
    total_rows = 0

    for row in _turn_rows(run_dir):
        policies = row.get("counterfactual_policies")
        if not isinstance(policies, Mapping):
            raise RuntimeError("counterfactual run contains a turn without policies")
        seed = int(row["seed"])
        total_rows += 1
        latent_gates[str(row.get("latent_gate_outcome"))] += 1
        semantic_gates[str(row.get("semantic_gate_outcome"))] += 1
        for scope, cells in SCOPES.items():
            if not _scope_matches(row, cells):
                continue
            scope_counts[scope] += 1
            for branch, values in policies.items():
                absolute[(scope, str(branch), "expected_top_1", seed)].append(float(values["expected_top_1"]))
                absolute[(scope, str(branch), "nll", seed)].append(float(values["actual_action_nll"]))
            for name, (candidate, reference, _description) in CONTRASTS.items():
                candidate_values = policies[candidate]
                reference_values = policies[reference]
                by_seed[(scope, name, "expected_top_1", seed)].append(
                    float(candidate_values["expected_top_1"])
                    - float(reference_values["expected_top_1"])
                )
                # Positive always means the candidate is better.
                by_seed[(scope, name, "nll_improvement", seed)].append(
                    float(reference_values["actual_action_nll"])
                    - float(candidate_values["actual_action_nll"])
                )
        if (
            row.get("transfer_cell_before") in TRANSFER_CELLS
            and row.get("latent_relaxed_score_outcome") == "scored"
        ):
            relations = int(row.get("latent_strategy_observed_relations", 0))
            relaxed = policies["latent_relaxed"]
            reference = policies["semantic_current"]
            relation_rows[seed].append((
                relations,
                float(reference["actual_action_nll"]) - float(relaxed["actual_action_nll"]),
                float(relaxed["expected_top_1"]) - float(reference["expected_top_1"]),
            ))
        if row.get("transfer_cell_before") in TRANSFER_CELLS:
            conditions = [f"current_gate:{row.get('latent_gate_outcome')}" ]
            if (
                row.get("latent_two_role_score_outcome") == "scored"
                and row.get("latent_relaxed_score_outcome") != "scored"
            ):
                conditions.append("new_two_role_scored")
            conditional_specs = {
                "relaxed_vs_deployed": ("latent_relaxed", "deployed_current"),
                "two_role_increment": ("latent_two_role", "latent_relaxed"),
            }
            for condition in conditions:
                conditional_counts[condition] += 1
                for name, (candidate, reference) in conditional_specs.items():
                    candidate_values = policies[candidate]
                    reference_values = policies[reference]
                    conditional[(condition, name, "expected_top_1", seed)].append(
                        float(candidate_values["expected_top_1"])
                        - float(reference_values["expected_top_1"])
                    )
                    conditional[(condition, name, "nll_improvement", seed)].append(
                        float(reference_values["actual_action_nll"])
                        - float(candidate_values["actual_action_nll"])
                    )

    contrast_summary: Dict[str, Any] = {}
    for scope in SCOPES:
        contrast_summary[scope] = {}
        for name, (_candidate, _reference, description) in CONTRASTS.items():
            metrics: Dict[str, Any] = {}
            for metric in ("expected_top_1", "nll_improvement"):
                seed_values = {
                    seed: _mean(values)
                    for (stored_scope, stored_name, stored_metric, seed), values in by_seed.items()
                    if (stored_scope, stored_name, stored_metric) == (scope, name, metric)
                }
                interval = _cluster_interval(
                    list(seed_values.values()),
                    key=f"component|{scope}|{name}|{metric}",
                )
                metrics[metric] = {
                    "positive_favors_candidate": True,
                    "n_seed_clusters": len(seed_values),
                    "per_seed": {str(seed): value for seed, value in sorted(seed_values.items())},
                    **interval,
                }
            contrast_summary[scope][name] = {
                "description": description,
                **metrics,
            }

    absolute_summary: Dict[str, Any] = {}
    for scope in SCOPES:
        branches = sorted({branch for stored_scope, branch, _metric, _seed in absolute if stored_scope == scope})
        absolute_summary[scope] = {}
        for branch in branches:
            absolute_summary[scope][branch] = {}
            for metric in ("expected_top_1", "nll"):
                seed_values = {
                    seed: _mean(values)
                    for (stored_scope, stored_branch, stored_metric, seed), values in absolute.items()
                    if (stored_scope, stored_branch, stored_metric) == (scope, branch, metric)
                }
                absolute_summary[scope][branch][metric] = {
                    "mean_of_seed_means": _mean(list(seed_values.values())),
                    "per_seed": {str(seed): value for seed, value in sorted(seed_values.items())},
                }

    slopes_nll: Dict[int, float] = {}
    slopes_top1: Dict[int, float] = {}
    relation_bins: Dict[int, Dict[str, Any]] = {}
    pooled_bins: Dict[int, list[Tuple[float, float]]] = defaultdict(list)
    for seed, values in relation_rows.items():
        xs = [float(value[0]) for value in values]
        mean_x = _mean(xs)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator > 0.0:
            for output, column in ((slopes_nll, 1), (slopes_top1, 2)):
                ys = [float(value[column]) for value in values]
                mean_y = _mean(ys)
                output[seed] = sum(
                    (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
                ) / denominator
        for relations, nll_delta, top1_delta in values:
            pooled_bins[int(relations)].append((nll_delta, top1_delta))
    for relations, values in sorted(pooled_bins.items()):
        relation_bins[relations] = {
            "n_turns": len(values),
            "mean_nll_improvement": _mean([value[0] for value in values]),
            "mean_expected_top_1_delta": _mean([value[1] for value in values]),
        }

    conditional_summary: Dict[str, Any] = {}
    for condition in sorted(conditional_counts):
        conditional_summary[condition] = {
            "n_turns": int(conditional_counts[condition]),
            "contrasts": {},
        }
        for name in ("relaxed_vs_deployed", "two_role_increment"):
            conditional_summary[condition]["contrasts"][name] = {}
            for metric in ("expected_top_1", "nll_improvement"):
                seed_values = {
                    seed: _mean(values)
                    for (stored_condition, stored_name, stored_metric, seed), values in conditional.items()
                    if (stored_condition, stored_name, stored_metric) == (condition, name, metric)
                }
                conditional_summary[condition]["contrasts"][name][metric] = {
                    "positive_favors_candidate": True,
                    "n_seed_clusters": len(seed_values),
                    "per_seed": {str(seed): value for seed, value in sorted(seed_values.items())},
                    **_cluster_interval(
                        list(seed_values.values()),
                        key=f"component|conditional|{condition}|{name}|{metric}",
                    ),
                }

    return {
        "analysis_scope": "one-step shadow counterfactuals; no policy-learning feedback",
        "experimental_unit": "seed-level mean paired turn difference",
        "seed_weighting": "equal",
        "total_turns": total_rows,
        "scope_turn_counts": dict(scope_counts),
        "latent_gate_counts": dict(latent_gates),
        "semantic_gate_counts": dict(semantic_gates),
        "absolute": absolute_summary,
        "contrasts": contrast_summary,
        "conditional_transfer_contrasts": conditional_summary,
        "sparsity_prediction": {
            "population": "transfer turns where the three-role relaxed residual was scored",
            "primary_metric": "within-seed slope of NLL improvement versus observed precedence relations",
            "nll_slope": {
                "n_seed_clusters": len(slopes_nll),
                "per_seed": {str(seed): value for seed, value in sorted(slopes_nll.items())},
                **_cluster_interval(
                    list(slopes_nll.values()), key="component|sparsity|nll_slope",
                ),
            },
            "expected_top_1_slope": {
                "n_seed_clusters": len(slopes_top1),
                "per_seed": {str(seed): value for seed, value in sorted(slopes_top1.items())},
                **_cluster_interval(
                    list(slopes_top1.values()), key="component|sparsity|top1_slope",
                ),
            },
            "pooled_descriptive_bins": {
                str(relations): values for relations, values in relation_bins.items()
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a completed component-counterfactual run.",
    )
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    result = summarize(run_dir)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else run_dir / "aggregate" / "component_counterfactuals.json"
    )
    _write_json(output, result)
    print(json.dumps({"output": str(output), "total_turns": result["total_turns"]}, indent=2))


if __name__ == "__main__":
    main()
