"""Lightweight, ingredient-invariant workflow-strategy residual."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .environment import CONTAINERS, INGREDIENTS, parse_action_label


ROLES: Tuple[str, ...] = (
    "ingredient_retrieval", "equipment_setup", "prep", "load",
    "activate", "process", "shutdown", "serving_setup", "serve",
    "cleanup_move", "wash", "other",
)
_ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
_PAIR_OFFSET = 2 * len(ROLES)
_PAIR_INDEX = {
    (first, second): _PAIR_OFFSET + index
    for index, (first, second) in enumerate(
        (a, b)
        for a in range(len(ROLES))
        for b in range(a + 1, len(ROLES))
    )
}
FINGERPRINT_DIM = _PAIR_OFFSET + len(_PAIR_INDEX)


def action_role(label: str) -> str:
    """Map a grounded action to a recipe/ingredient-invariant workflow role."""
    action = parse_action_label(str(label))
    verb = action.verb
    item = (
        action.get("item") or action.get("container")
        or action.get("vessel") or ""
    )
    source = action.get("from")
    destination = action.get("to")
    if verb == "transfer" and source == "storage" and item in INGREDIENTS:
        return "ingredient_retrieval"
    if (
        verb in {"transfer", "move_container"}
        and source == "storage"
        and item in CONTAINERS
        and item not in {"plate", "glass"}
    ):
        return "equipment_setup"
    if verb in {"cut", "grate"}:
        return "prep"
    if verb in {"load", "unload"}:
        return "load"
    if verb == "turn_on":
        return "activate"
    if verb in {"cook", "cook_contents", "blend", "combine"}:
        return "process"
    if verb == "turn_off":
        return "shutdown"
    if (
        verb == "transfer"
        and item in {"plate", "glass"}
        and destination in {"plating_station", "serving_station"}
    ):
        return "serving_setup"
    if (
        verb == "serve"
        or (
            verb == "move_container"
            and item in {"plate", "glass"}
            and destination == "serving_station"
        )
    ):
        return "serve"
    if verb in {"transfer", "move_container"} and destination == "washing_station":
        return "cleanup_move"
    if verb == "wash":
        return "wash"
    return "other"


def _role_positions(actions: Sequence[str]) -> Tuple[Tuple[str, ...], Dict[int, list[int]]]:
    roles = tuple(action_role(action) for action in actions if action != "stop")
    positions: Dict[int, list[int]] = {}
    for position, role in enumerate(roles):
        positions.setdefault(_ROLE_INDEX[role], []).append(position)
    return roles, positions


def workflow_fingerprint(actions: Sequence[str]) -> np.ndarray:
    """Encode role timing/counts and pairwise precedence without identities."""
    roles, positions = _role_positions(actions)
    denominator = max(1, len(roles) - 1)
    out = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    means: Dict[int, float] = {}
    for role_index, indices in positions.items():
        means[role_index] = float(np.mean(indices))
        out[role_index] = means[role_index] / denominator
        out[len(ROLES) + role_index] = min(1.0, len(indices) / 3.0)
    for (first, second), feature_index in _PAIR_INDEX.items():
        if first in means and second in means:
            out[feature_index] = 1.0 if means[first] < means[second] else -1.0
    return out


@dataclass(frozen=True)
class StrategyScore:
    utilities: Mapping[str, float]
    confidence: float
    coverage: float
    observed_roles: int
    observed_relations: int
    neighbor_count: int


class LatentStrategyResidual:
    """Weighted PCA retrieval with timing and optional trajectory residuals."""

    def __init__(self, settings: Any):
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self.mean = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        self.scale = np.ones(FINGERPRINT_DIM, dtype=np.float32)
        self.components = np.zeros((0, FINGERPRINT_DIM), dtype=np.float32)
        self.codes = np.zeros((0, 0), dtype=np.float32)
        self.code_scale = np.ones(0, dtype=np.float32)
        self.fingerprints = np.zeros((0, FINGERPRINT_DIM), dtype=np.float32)
        self.role_counts = np.zeros((0, len(ROLES)), dtype=np.int16)
        self.role_sequences: Tuple[Tuple[int, ...], ...] = ()
        self.weights = np.zeros(0, dtype=np.float32)
        self.mean_demo_length = 1.0
        self.last_score = StrategyScore({}, 0.0, 0.0, 0, 0, 0)
        self.last_score_stats: Dict[str, Any] = {}
        self.last_fit_stats: Dict[str, Any] = {
            "latent_strategy_enabled": bool(
                getattr(self.settings, "latent_strategy_enabled", True)
            ),
            "latent_strategy_prototypes": 0,
            "latent_strategy_fit_flops": 0.0,
        }

    def fit(
        self,
        demonstrations: Sequence[Sequence[Tuple[Tuple[int, ...], str]]],
        demo_weights: Sequence[float] | None = None,
    ) -> None:
        self.reset()
        if not bool(getattr(self.settings, "latent_strategy_enabled", True)):
            return
        indexed_rows = [
            (index, tuple(action for _state, action in demo if action != "stop"))
            for index, demo in enumerate(demonstrations)
        ]
        indexed_rows = [(index, row) for index, row in indexed_rows if row]
        action_rows = [row for _index, row in indexed_rows]
        if not action_rows:
            return
        weights = np.asarray(
            [float(demo_weights[index]) for index, _row in indexed_rows]
            if demo_weights is not None and len(demo_weights) == len(demonstrations)
            else [1.0] * len(action_rows),
            dtype=np.float64,
        )
        if weights.shape != (len(action_rows),) or float(weights.sum()) <= 0.0:
            weights = np.ones(len(action_rows), dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        normalized = weights / max(float(weights.sum()), 1e-12)
        matrix = np.stack([workflow_fingerprint(row) for row in action_rows])
        self.mean = np.average(matrix, axis=0, weights=normalized).astype(np.float32)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=normalized,
        )
        self.scale = np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
        standardized = (matrix - self.mean) / self.scale
        rank = min(
            max(1, int(getattr(self.settings, "latent_strategy_rank", 8))),
            len(action_rows), FINGERPRINT_DIM,
        )
        _u, _singular, right = np.linalg.svd(
            np.sqrt(normalized)[:, None] * standardized,
            full_matrices=False,
        )
        self.components = right[:rank].astype(np.float32)
        self.codes = (standardized @ self.components.T).astype(np.float32)
        code_variance = np.average(
            self.codes ** 2, axis=0, weights=normalized,
        )
        self.code_scale = np.sqrt(np.maximum(code_variance, 1e-6)).astype(np.float32)
        self.fingerprints = matrix.astype(np.float32)
        self.role_counts = np.asarray([
            [sum(action_role(action) == role for action in row) for role in ROLES]
            for row in action_rows
        ], dtype=np.int16)
        self.role_sequences = tuple(
            tuple(_ROLE_INDEX[action_role(action)] for action in row)
            for row in action_rows
        ) if float(getattr(
            self.settings, "latent_strategy_sequence_weight", 0.0,
        )) > 0.0 else ()
        self.weights = weights.astype(np.float32)
        self.mean_demo_length = float(np.average(
            [len(row) for row in action_rows], weights=normalized,
        ))
        flop_estimate = float(
            6 * matrix.size
            + 4 * len(action_rows) * FINGERPRINT_DIM * rank
        )
        self.last_fit_stats = {
            "latent_strategy_enabled": True,
            "latent_strategy_representation": "ingredient_masked_workflow_precedence",
            "latent_strategy_fingerprint_dim": FINGERPRINT_DIM,
            "latent_strategy_rank": rank,
            "latent_strategy_prototypes": len(action_rows),
            "latent_strategy_parameter_count": int(
                self.components.size + self.mean.size + self.scale.size
            ),
            "latent_strategy_prototype_storage_values": int(
                self.fingerprints.size + self.codes.size + self.role_counts.size
                + sum(len(sequence) for sequence in self.role_sequences)
            ),
            "latent_strategy_fit_flops": flop_estimate,
            "latent_strategy_flop_scope": "weighted_pca_estimate_only",
        }

    def _partial_observation(
        self, prefix: Sequence[str],
    ) -> Tuple[np.ndarray | None, Tuple[int, ...], int, int]:
        roles, positions = _role_positions(prefix)
        observed = sorted(positions)
        if not observed or not len(self.components):
            return None, (), len(observed), 0
        query = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        mask: list[int] = []
        denominator = max(1.0, self.mean_demo_length - 1.0)
        means = {index: float(np.mean(values)) for index, values in positions.items()}
        for role_index in observed:
            query[role_index] = means[role_index] / denominator
        relations = 0
        for (first, second), feature_index in _PAIR_INDEX.items():
            if first in means and second in means:
                query[feature_index] = (
                    1.0 if means[first] < means[second] else -1.0
                )
                mask.append(feature_index)
                relations += 1
        target = query[mask].astype(np.float32)
        return target, tuple(mask), len(observed), relations

    def score(
        self,
        prefix: Sequence[str],
        candidates: Sequence[str],
        *,
        decision_prefix: Sequence[str] | None = None,
    ) -> StrategyScore:
        empty = StrategyScore({}, 0.0, 0.0, 0, 0, 0)
        target, mask, observed_roles, relations = self._partial_observation(prefix)
        # Three distinct roles expose three pairwise relations.  The residual
        # still cannot act until a human correction confirms its hypothesis.
        # partial prefix admits too many incompatible strategies and the
        # residual must remain neutral.
        if target is None or observed_roles < 3 or not len(self.codes):
            self.last_score_stats = {
                "latent_strategy_attempted": False,
                "latent_strategy_observed_roles": observed_roles,
                "latent_strategy_observed_relations": relations,
            }
            self.last_score = empty
            return empty
        # Strategy identity is inferred only from precedence relations already
        # exposed by the prefix.  The 8-D code remains the compact stored
        # representation; raw relation bits avoid inventing unobserved future
        # coordinates when the prefix is sparse.
        distances = np.sqrt(np.mean(
            (self.fingerprints[:, mask] - target[None, :]) ** 2, axis=1,
        ))
        finite = np.flatnonzero(np.isfinite(distances))
        if not len(finite):
            self.last_score = empty
            return empty
        count = min(
            max(1, int(getattr(self.settings, "latent_strategy_knn", 3))),
            len(finite),
        )
        kth_distance = float(np.partition(distances[finite], count - 1)[count - 1])
        indices = finite[distances[finite] <= kth_distance + 1e-8]
        count = int(len(indices))
        selected_distances = distances[indices]
        bandwidth = max(float(np.median(selected_distances)), 1e-3)
        logits = -selected_distances / bandwidth + np.log(
            np.maximum(self.weights[indices], 1e-8)
        )
        logits -= float(np.max(logits))
        neighbor_weights = np.exp(logits)
        neighbor_weights /= max(float(neighbor_weights.sum()), 1e-12)
        normalized_codes = self.codes[indices] / self.code_scale
        code_center = np.average(
            normalized_codes, axis=0, weights=neighbor_weights,
        )
        code_dispersion = float(np.average(
            np.mean((normalized_codes - code_center) ** 2, axis=1),
            weights=neighbor_weights,
        ))
        latent_agreement = 1.0 / (1.0 + code_dispersion)

        roles_by_action = {
            str(action): _ROLE_INDEX[action_role(str(action))]
            for action in candidates
        }
        prefix_roles, prefix_positions = _role_positions(
            prefix if decision_prefix is None else decision_prefix
        )
        progress = min(1.0, len(prefix_roles) / max(1.0, self.mean_demo_length - 1.0))
        utilities: Dict[str, float] = {}
        agreements: list[float] = []
        covered = possible = 0
        for action, role_index in roles_by_action.items():
            per_neighbor: list[float] = []
            for neighbor_index, fingerprint in zip(
                indices, self.fingerprints[indices],
            ):
                expected_count = int(self.role_counts[neighbor_index, role_index])
                observed_count = len(prefix_positions.get(role_index, ()))
                possible += 1
                if expected_count <= 0:
                    per_neighbor.append(-1.0)
                    continue
                covered += 1
                expected_position = float(fingerprint[role_index])
                timing = 1.0 - 2.0 * abs(expected_position - progress)
                per_neighbor.append(
                    timing if observed_count < expected_count else -1.0
                )
            values = np.asarray(per_neighbor, dtype=np.float64)
            utilities[action] = float(np.dot(neighbor_weights, values))
            agreements.append(1.0 - min(1.0, float(
                np.dot(neighbor_weights, (values - utilities[action]) ** 2)
            )))
        sequence_weight = float(getattr(
            self.settings, "latent_strategy_sequence_weight", 0.0,
        ))
        alignment_confidence = 1.0
        alignment_flops = 0.0
        if sequence_weight > 0.0 and prefix_roles:
            role_mass = np.zeros(len(ROLES), dtype=np.float64)
            confidence_mass = 0.0
            for neighbor_index, neighbor_weight in zip(indices, neighbor_weights):
                next_role, alignment, flops = self._aligned_next_role(
                    tuple(_ROLE_INDEX[role] for role in prefix_roles),
                    self.role_sequences[neighbor_index],
                )
                alignment_flops += flops
                if next_role is not None:
                    mass = float(neighbor_weight) * alignment
                    role_mass[next_role] += mass
                    confidence_mass += mass
            if confidence_mass > 0.0:
                role_mass /= confidence_mass
                for action, role_index in roles_by_action.items():
                    sequence_utility = 2.0 * float(role_mass[role_index]) - 1.0
                    utilities[action] = (
                        (1.0 - sequence_weight) * utilities[action]
                        + sequence_weight * sequence_utility
                    )
                alignment_confidence = min(1.0, confidence_mass)
        if not utilities:
            self.last_score = empty
            return empty
        center = float(np.mean(list(utilities.values())))
        scale = max(abs(value - center) for value in utilities.values())
        if scale <= 1e-8:
            self.last_score = empty
            return empty
        utilities = {
            action: (value - center) / scale
            for action, value in utilities.items()
        }
        coverage = min(1.0, covered / max(1, possible))
        evidence = min(1.0, (observed_roles - 1) / 3.0)
        confidence = (
            evidence * latent_agreement * float(np.mean(agreements))
            * ((1.0 - sequence_weight) + sequence_weight * alignment_confidence)
        )
        result = StrategyScore(
            utilities, confidence, coverage, observed_roles, relations, count,
        )
        self.last_score = result
        self.last_score_stats = {
            "latent_strategy_attempted": True,
            "latent_strategy_observed_roles": observed_roles,
            "latent_strategy_observed_relations": relations,
            "latent_strategy_neighbor_count": count,
            "latent_strategy_confidence": confidence,
            "latent_strategy_coverage": coverage,
            "latent_strategy_code_dispersion": code_dispersion,
            "latent_strategy_sequence_weight": sequence_weight,
            "latent_strategy_alignment_flops": alignment_flops,
        }
        return result

    @staticmethod
    def _aligned_next_role(
        observed: Tuple[int, ...], prototype: Tuple[int, ...],
    ) -> Tuple[int | None, float, float]:
        """Align an observed role prefix and return the prototype frontier."""
        if not prototype:
            return None, 0.0, 0.0
        gap = 0.35
        previous = np.arange(len(prototype) + 1, dtype=np.float64) * gap
        for observed_role in observed:
            current = np.empty_like(previous)
            current[0] = previous[0] + gap
            for column, prototype_role in enumerate(prototype, start=1):
                current[column] = min(
                    previous[column] + gap,
                    current[column - 1] + gap,
                    previous[column - 1] + (observed_role != prototype_role),
                )
            previous = current
        frontier = int(np.argmin(previous))
        cost = float(previous[frontier])
        confidence = math.exp(-cost / max(1, len(observed)))
        next_role = prototype[frontier] if frontier < len(prototype) else None
        flops = float(6 * len(observed) * len(prototype))
        return next_role, confidence, flops

    def supports_correction(self, actual: str, predicted: str) -> bool:
        """Return whether the last latent hypothesis favored the correction."""
        utilities = self.last_score.utilities
        return bool(
            utilities
            and str(actual) in utilities
            and str(predicted) in utilities
            and float(utilities[str(actual)]) > float(utilities[str(predicted)])
            and self.last_score.confidence > 0.0
            and self.last_score.coverage > 0.0
        )


def fuse_strategy_residual(
    base: Mapping[str, float], score: StrategyScore, strength: float,
) -> Tuple[Dict[str, float], float]:
    """Apply a bounded log-linear residual while preserving full support."""
    if not base or not score.utilities:
        return dict(base), 0.0
    probabilities = sorted((float(value) for value in base.values()), reverse=True)
    margin = probabilities[0] - probabilities[1] if len(probabilities) > 1 else 1.0
    alpha = (
        max(0.0, float(strength))
        * score.confidence * score.coverage * max(0.0, 1.0 - margin)
    )
    if alpha <= 1e-8:
        return dict(base), 0.0
    adjusted = {
        action: max(float(probability), 1e-12)
        * math.exp(alpha * float(score.utilities.get(action, 0.0)))
        for action, probability in base.items()
    }
    total = sum(adjusted.values())
    if total <= 0.0 or not math.isfinite(total):
        return dict(base), 0.0
    return {action: value / total for action, value in adjusted.items()}, alpha
