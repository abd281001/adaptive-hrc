"""Adaptive HRC state machine with observation, online, and frozen modes."""
from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np

from .domain import DomainAdapter, default_domain
from .memory import (MatchResult, ReplayMemory, StateTransition, RecipeMatcher, MemoryItem, Variant, VariantKey, VariantLibrary, make_variant_id)
from .models import (Settings, DEFAULT_SETTINGS, MaxEntIrl, top_actions)
from .representations import (Observation)

# Design invariants of the shared action mask. These are properties of the
# implementation, not per-decision observations, so they are asserted in the
# test suite and recorded once in a run manifest rather than written onto every
# turn row -- a value that cannot come out False is not evidence for itself.
ACTION_MASK_CONTRACT: Dict[str, Any] = {
    "action_mask": "shared_state_preconditions_only",
    "action_mask_shared_across_predictors": True,
    "action_mask_preference_neutral": True,
    "action_mask_uses_recipe_hypothesis": False,
}

MODE_OBSERVE = "observe"
MODE_ONLINE = "online"


def _silent_narrate(_msg: str) -> None:
    """Picklable no-op narrator."""
    return None


@dataclass
class StepResult:
    step: int
    predicted: Optional[str]
    actual: str
    correct: bool
    mode: str
    event: str = ""   # free-form tag for the narrator


@dataclass(frozen=True)
class TrainPolicy:
    """Select cold starts from cumulative replay *additions*.

    Only an addition introduces demonstrations the current reward weights have
    never been fit against, so only an addition can justify paying for a fresh
    optimum.  A removal strictly shrinks the training set: the incumbent
    weights remain a valid starting point for the surviving subset, and forcing
    a cold restart there is what made decay-driven churn cost more compute than
    retaining everything.  Removals therefore warm-start and do not advance the
    cold-start counter.
    """
    cold_after: int = 3
    warm_on_weight_change: bool = False
    cold_threshold_basis: str = "additions_only"

    def decide(self, additions: int, removals: int, weight_changed: bool, since_cold: int) -> Tuple[str, str, int]:
        additions = int(additions)
        removals = int(removals)
        projected = int(since_cold) + additions
        if additions > 0:
            if projected >= max(1, int(self.cold_after)):
                return "cold", "cumulative_addition_threshold_reached", projected
            return "warm", "cumulative_additions_below_threshold", projected
        if removals > 0: return "warm", "removal_only_membership_change", projected
        if weight_changed: return (("warm", "weight_only_change", projected) if self.warm_on_weight_change else ("skip", "weight_only_change_skipped", projected))
        return "skip", "replay_unchanged", projected


FULL_TRAIN_POLICY = TrainPolicy()
# Shared-agent comparisons use the same warm/cold schedule so their memory or predictor policy remains the experimental difference.
BASELINE_TRAIN_POLICY = FULL_TRAIN_POLICY


class AdaptiveAgent:
    """End-to-end continual-learning HRC agent."""

    RETRAIN_POLICY = FULL_TRAIN_POLICY

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, narrate: Optional[Callable[[str], None]] = None, retrain_policy: Optional[TrainPolicy] = None, domain: Optional[DomainAdapter] = None):
        self.settings = settings
        self.domain = domain or default_domain()
        self.retrain_policy = retrain_policy or self.RETRAIN_POLICY
        self.replay = ReplayMemory(settings)
        self.library = VariantLibrary()
        self.matcher = RecipeMatcher(settings)
        self.commit_events: List[Dict[str, Any]] = []
        self.last_commit_stats: Dict[str, Any] = {}

        # This agent owns its cold-start initialization stream. Two agents built
        # from the same Settings instance therefore stay independent, so one arm
        # can never perturb another's model initialization.
        self._init_rng = np.random.default_rng(int(settings.seed))
        self.maxent = MaxEntIrl(settings=settings, domain=self.domain, rng=self._init_rng)
        # Keep action-selection randomness separate from MaxEnt optimization so a prediction tie cannot alter future model initialization or fitting.
        tie_seed = np.random.SeedSequence([int(settings.seed) & 0xFFFFFFFF, 0x544945])
        self._tie_break_rng = np.random.default_rng(tie_seed)
        self._last_action_mask_stats: Dict[str, Any] = {}

        self.mode = MODE_ONLINE
        self.pending_demo: List[str] = []
        self.pending_trace: List[StateTransition] = []
        self.current_prefix: List[str] = []
        self.current_trace: List[StateTransition] = []
        self._needs_observation: bool = False
        self._policy_history: List[Dict[str, Any]] = []
        self._prediction_mismatch_count: int = 0
        self._latent_strategy_confirmed: bool = False
        self.provisional: Dict[VariantKey, Dict[str, Any]] = {}
        self._policy_stats: Dict[str, Any] = {
            "action_required": True,
            "confidence": None,
            "raw_confidence": None,
            "margin": None,
            "entropy": None,
            "entropy_basis": "prediction_support",
            "predictor": None,
            "final_confidence": None,
            "final_margin": None,
            "final_entropy": None,
            "reason": "cold_start",
        }
        self._next_recipe_index = 1
        self.step_counter = 0       # actions observed
        self.demo_counter = 0       # completed non-empty demonstrations
        self.retrain_cycle = 0
        self.narrate = narrate or _silent_narrate
        self.step_log: List[StepResult] = []
        self.accuracy_events: List[Tuple[int, str, bool]] = []  # (step,recipe,correct)
        # Retrain timing and no-op transitions dropped during trajectory conversion.
        self.retrain_events: List[Dict[str, Any]] = []
        self.retrain_fit_wall_times: List[float] = []
        self.retrain_total_wall_times: List[float] = []
        self.retrain_build_wall_times: List[float] = []
        self.retrain_flop_estimates: List[float] = []
        self.classification_events: List[Tuple[int, MatchResult]] = []

        self._frozen: bool = False
        self._freeze_snapshot: Optional[Tuple] = None

        # Optional event -> (calls, wall seconds) profile.
        self.profile: Dict[str, Tuple[int, float]] = {}

        # Encoded state -> distinct actions demonstrated there in active replay.
        # Rebuilt whenever replay is refitted; the basis of the decision-regime split.
        self._observed_actions: Dict[Tuple[int, ...], Set[str]] = {}
        self._last_observed_replay_weights: Dict[Tuple[str, str], float] = {}
        self.cold_change_count = 0
        # Public so reports distinguish retrain requests from executed fits.
        self.skipped_trains: int = 0

    @contextlib.contextmanager
    def _profile(self, event: str):
        """No-cost context manager when settings.profile is False."""
        if not self.settings.profile:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            calls, wall_seconds = self.profile.get(event, (0, 0.0))
            self.profile[event] = (calls + 1, wall_seconds + (time.perf_counter() - start_time))

    # Fields that carry the freeze bookkeeping itself and must never be part of
    # a captured state.
    _CAPTURE_EXCLUDED = ("_freeze_snapshot", "_frozen")

    def _capture_state(self) -> Dict[str, Any]:
        """Capture everything a probe could mutate, as one independent copy.

        This is the single definition of "restorable agent state", used by both
        the freeze context and snapshot/restore.  It deliberately captures the
        whole instance dictionary rather than an enumerated subset: the field
        that is easiest to forget is `_tie_break_rng`, whose stream position
        decides every prediction tie, and a probe that leaves it advanced
        silently changes later predictions without failing anything.
        """
        captured = copy.deepcopy({key: item for key, item in self.__dict__.items() if key not in self._CAPTURE_EXCLUDED})
        # Capture must be independent, not aliased: a captured generator that is
        # the live object would make the restore check below vacuously pass.
        for key, item in captured.items():
            if isinstance(item, np.random.Generator) and item is self.__dict__.get(key): raise RuntimeError(f"captured state aliases the live generator {key!r}; a restore from it could not undo a probe")
        return captured

    def _apply_state(self, state: Mapping[str, Any]) -> None:
        """Restore a captured state in place, then verify the RNG came back."""
        expected = self._rng_positions(state)
        for key, item in state.items(): self.__dict__[key] = item
        restored = self._rng_positions(self.__dict__)
        if restored != expected: raise RuntimeError(f"state restore left a random stream advanced: {sorted(set(expected) ^ set(restored)) or 'position mismatch'}; predictions after a probe would diverge")

    @staticmethod
    def _rng_positions(state: Mapping[str, Any]) -> Dict[str, Any]:
        """Bit-generator positions of every Generator in a captured state."""
        positions: Dict[str, Any] = {}
        for key, item in state.items():
            if isinstance(item, np.random.Generator): positions[key] = repr(item.bit_generator.state)
        return positions

    def snapshot(self) -> "AdaptiveAgent":
        """Deep-copy the agent for phase-A sweep reuse. Lossless and pure (no leakage between branches)."""
        if self._frozen: raise RuntimeError("snapshot called while frozen; release frozen first")
        return copy.deepcopy(self)

    def restore_from(self, snapshot: "AdaptiveAgent") -> None:
        """Overwrite this agent's state from a previously-taken snapshot, in place."""
        if type(self) is not type(snapshot): raise TypeError(f"restore_from type mismatch: {type(self).__name__} vs {type(snapshot).__name__}")
        state = copy.deepcopy({key: item for key, item in snapshot.__dict__.items() if key not in self._CAPTURE_EXCLUDED})
        frozen_bookkeeping = {key: snapshot.__dict__[key] for key in self._CAPTURE_EXCLUDED if key in snapshot.__dict__}
        self.__dict__.clear()
        self.__dict__.update(frozen_bookkeeping)
        self._apply_state(state)

    def set_frozen(self, value: bool) -> None:
        """Freeze or restore all mutable state so evaluation cannot leak."""
        if value and not self._frozen:
            with self._profile("freeze_snapshot_enter"): self._freeze_snapshot = self._capture_state()
            self._frozen = True
        elif not value and self._frozen:
            frozen_state = self._freeze_snapshot
            if frozen_state is not None:
                with self._profile("freeze_snapshot_exit"): self._apply_state(frozen_state)
            self._freeze_snapshot = None
            self._frozen = False

    def _frozen_structural_digest(self) -> Dict[str, Any]:
        """Small diagnostic digest for state that must not change in frozen eval."""
        active = tuple(sorted((recipe_id, variant_id, tuple(entry.ordering), round(float(entry.weight), 12), int(entry.added_step), int(entry.added_cycle), int(entry.last_seen_step), tuple(entry.transitions))
            for (recipe_id, variant_id), entry in self.replay.active.items()))
        pruned = tuple(sorted((recipe_id, variant_id, tuple(entry.ordering), int(entry.added_step), int(entry.removed_step), int(entry.added_cycle), int(entry.removed_cycle), int(entry.last_seen_step), tuple(entry.transitions),)
            for (recipe_id, variant_id), entry in self.replay.pruned.items()))
        variant_keys = tuple(sorted((recipe_id, variant_id) for recipe_id, slot in self.library.variants.items() for variant_id in slot))
        return {"mode": self.mode, "step_counter": int(self.step_counter), "demo_counter": int(self.demo_counter), "retrain_cycle": int(self.retrain_cycle), "current_prefix": tuple(self.current_prefix),
            "current_trace": tuple(self.current_trace), "pending_demo": tuple(self.pending_demo), "pending_trace": tuple(self.pending_trace),
            "step_log_len": len(self.step_log), "classification_events_len": len(self.classification_events), "accuracy_events_len": len(self.accuracy_events), "retrain_events_len": len(self.retrain_events),
            "online_commit_events_len": len(self.commit_events), "prediction_mismatch_count": int(self._prediction_mismatch_count), "latent_strategy_confirmed": bool(self._latent_strategy_confirmed),
            "active": active, "pruned": pruned, "variant_keys": variant_keys, "memory_latest": tuple(sorted(self.library.latest.items())), "latest_by_recipe": tuple(sorted(self.replay.latest_by_recipe.items())),
            "latest_keys": tuple(sorted(self.replay.latest_keys)), "pair_gap_windows": tuple(sorted((key, tuple(gaps)) for key, gaps in self.replay._pair_gap_window.items())),
            "pair_last_seen_steps": tuple(sorted(self.replay._pair_last_seen_step.items())), "recipe_gap_events": tuple(sorted((recipe_id, tuple(events)) for recipe_id, events in self.replay._recipe_gap_events.items())),
            "global_gap_events": tuple(self.replay._global_gap_events), "last_observed_replay_weights": tuple(sorted(self._last_observed_replay_weights.items())), "additions_since_cold": int(self.cold_change_count)}

    @contextlib.contextmanager
    def frozen(self):
        self.set_frozen(True)
        state_before = self._frozen_structural_digest()
        try:
            yield self
        finally:
            # Digest before rollback so frozen mutations remain detectable.
            state_after = self._frozen_structural_digest()
            self.set_frozen(False)
            if state_before != state_after:
                changed = sorted(key for key in state_before if state_before.get(key) != state_after.get(key))
                raise RuntimeError(f"frozen() invariant violated: structural state changed ({', '.join(changed)})")

    def start_demo(self) -> None:
        """Human signals 'I am about to show a new recipe.'"""
        if self._frozen: return
        self.mode = MODE_OBSERVE
        self.pending_demo = []
        self.pending_trace = []
        self._needs_observation = False
        self._policy_history = []
        self._prediction_mismatch_count = 0
        self._latent_strategy_confirmed = False
        self.narrate(f"[step {self.step_counter}] MODE -> OBSERVE (new-demo gate ON)")

    def _clear_online_session(self, *, needs_observation: bool = False) -> None:
        self.current_prefix = []
        self.current_trace = []
        self._needs_observation = needs_observation
        self._policy_history = []
        self._prediction_mismatch_count = 0
        self._latent_strategy_confirmed = False

    def _log_commit(self, **fields: Any) -> None:
        """Persist one auditable decision for the completed assist episode."""
        event = {"decision_id": f"assist_{self.demo_counter}_{self.step_counter}", "demo_index": int(self.demo_counter), "step": int(self.step_counter), **fields}
        self.last_commit_stats.update(event)
        self.commit_events.append(dict(event))

    def end_demo(self) -> MatchResult:
        """End one demonstration, advance its clock, and commit when justified."""
        if self._frozen: return MatchResult("frozen", None, None, 0.0, 0.0)

        # Empty observation sessions reset without aging memory.
        if self.mode == MODE_OBSERVE and not self.pending_demo:
            self.narrate(f"[step {self.step_counter}] end_demo called in OBSERVE with empty buffer - no-op")
            self.mode = MODE_ONLINE
            self.pending_trace = []
            return MatchResult("known", None, None, 0.0, 0.0)
        if self.mode == MODE_ONLINE and not self.current_prefix:
            self._clear_online_session()
            return MatchResult("known", None, None, 0.0, 0.0)

        # Every non-empty demonstration ages memory, regardless of its outcome.
        self.demo_counter += 1
        if self.mode == MODE_OBSERVE: return self._finish_observation(apply_decay=True)

        self.last_commit_stats = {}
        scored_before = self.matcher.variants_scored
        classification_t0 = time.perf_counter()
        prefix_match, reentry_from_pruned = self._match_prefix(self.current_prefix)
        classification_wall_s = time.perf_counter() - classification_t0
        active_keys = self._active_variants()
        registry_variants = sum(len(slot) for slot in self.library.variants.values())
        active_variants = sum((recipe_id, variant_id) in active_keys for recipe_id, slot in self.library.variants.items() for variant_id in slot)
        self.last_commit_stats = {"registry_size": registry_variants, "active_variants": active_variants, "archived_variants": registry_variants - active_variants, "registry_recipes": sum(bool(slot) for slot in self.library.variants.values()),
            "variants_scored": self.matcher.variants_scored - scored_before, "classification_wall_s": float(classification_wall_s), "scoring_wall_s": float(classification_wall_s)}
        # Assist is closed-set. Registration remains confidence-gated below.
        return self._finish_session(prefix_match, reentry_from_pruned, apply_decay=True)

    def _transition(self, action: str, observation: Observation) -> StateTransition:
        state = tuple(int(x) for x in observation.state)
        next_state = tuple(int(x) for x in observation.next_state)
        return (state, action, next_state)

    def _normalize_trace(self, transitions: Optional[Sequence[StateTransition]]) -> Tuple[StateTransition, ...]:
        if not transitions: return ()
        return tuple((tuple(int(x) for x in state), str(action), tuple(int(x) for x in next_state)) for state, action, next_state in transitions)

    def _store_trace(self, ordering: Sequence[str], transitions: Optional[Sequence[StateTransition]]) -> Tuple[StateTransition, ...]:
        """Validate an observed trace against its ordering.
        The replay item is the single owner of a demonstration's transitions. A
        trace whose actions disagree with the ordering it was recorded against
        is a defect, not a condition to recover from: silently substituting a
        different trace would train the model on something the human never did.
        """
        trace = self._normalize_trace(transitions)
        if not trace: return ()
        ordering_tuple = tuple(str(t) for t in ordering)
        if tuple(t[1] for t in trace) != ordering_tuple:
            raise ValueError(f"observed trace does not match its ordering: {len(trace)} transitions vs {len(ordering_tuple)} actions")
        return trace

    def _get_trace(self, demo: Any, actions: Sequence[str]) -> Tuple[StateTransition, ...]:
        if isinstance(demo, Mapping):   trace = self._normalize_trace(demo.get("transitions", ()))
        else:                           trace = self._normalize_trace(getattr(demo, "transitions", ()))
        if trace and tuple(t[1] for t in trace) == tuple(actions):
            return trace
        return ()

    def _register_if_live(self, recipe_id: str, actions: List[str], step: int, transitions: Optional[Sequence[StateTransition]] = None) -> Variant:
        """Register a completed demonstration in variant memory and decay state."""
        variant = self.library.register(recipe_id, actions, step)
        if not self._frozen:
            transition_trace = self._store_trace(actions, transitions)
            entry = self.replay.register(recipe_id, variant.variant_id, tuple(actions), self.demo_counter, self.retrain_cycle, transitions=transition_trace)
            self._record_demo(recipe_id, variant.variant_id, tuple(actions), transitions=transition_trace, demo_step=self.demo_counter, action_step=step, source_mode=self.mode, entry=entry)
            self._check_latest(recipe_id)
        return variant

    def _record_demo(self, recipe_id: str, variant_id: str, ordering: Tuple[str, ...], *, transitions: Tuple[StateTransition, ...] = (), demo_step: int, action_step: int, source_mode: str, entry: Optional[MemoryItem] = None) -> None:
        """Hook for replay baselines; the full agent trains from active memory."""
        return None

    def _check_latest(self, recipe_id: Optional[str] = None) -> None:
        if self._frozen or not self.settings.pin_latest: return
        recipe_ids = ([recipe_id] if recipe_id is not None else [known_recipe_id for known_recipe_id, variants in self.library.variants.items() if variants])
        for recipe_id in recipe_ids:
            slot = self.library.variants.get(recipe_id, {})
            if not slot: continue
            latest_variant_id = self.library.latest.get(recipe_id)
            latest_key = ((recipe_id, latest_variant_id)if latest_variant_id is not None else None)
            if latest_variant_id is None or latest_variant_id not in slot:
                raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: memory.latest missing from registry")
            if self.replay.latest_by_recipe.get(recipe_id) != latest_variant_id:
                raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: replay latest does not match variant library")
            if latest_key not in self.replay.latest_keys:
                raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest key is not pinned")
            if latest_key not in self.replay.active:
                raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest key is not active")
            if latest_key in self.replay.pruned:
                raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest key is pruned")
            if self.replay.active[latest_key].weight != 1.0:
                raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest weight is not 1.0")

    def _finish_observation(self, apply_decay: bool = False) -> MatchResult:
        actions = list(self.pending_demo)
        transition_trace = list(self.pending_trace)
        active_keys = self._active_variants()
        active_variants = self.library.known_variants(allowed_keys=active_keys)
        match = self.matcher.classify(actions, active_variants)
        # Check archived variants before allocating a new recipe ID.
        if match.kind == "new_recipe" and active_keys:
            all_variants = self.library.known_variants()
            if len(all_variants) > len(active_variants):
                archived_match = self.matcher.classify(actions, all_variants)
                if archived_match.kind != "new_recipe":
                    match = archived_match
                    self.narrate(f"[step {self.step_counter}] observe-mode: restoring pruned variant of '{match.recipe_id}'")

        self.classification_events.append((self.step_counter, match))
        if match.kind == "new_recipe":
            recipe_id = f"R{self._next_recipe_index}"
            self._next_recipe_index += 1
            self._register_if_live(recipe_id, actions, self.step_counter, transitions=transition_trace)
            match.recipe_id = recipe_id
            self.narrate(f"[step {self.step_counter}] classified NEW RECIPE '{recipe_id}' (jaccard={match.jaccard:.2f})")
        else:
            if match.recipe_id is None: raise RuntimeError(f"matcher returned {match.kind} without a recipe_id")
            recipe_id = match.recipe_id
            self._register_if_live(recipe_id, actions, self.step_counter, transitions=transition_trace)
            kind_msg = "KNOWN (re-demo)" if match.kind == "known" else "PREFERENCE VARIANT"
            self.narrate(f"[step {self.step_counter}] classified {kind_msg} of '{recipe_id}' (jaccard={match.jaccard:.2f}, tau={match.order_distance:.2f})")
        self.mode = MODE_ONLINE
        self.pending_demo = []
        self.pending_trace = []
        self._needs_observation = False
        self._policy_history = []
        self._update(apply_decay)
        return match

    def _match_prefix(self, prefix: Sequence[str]) -> Tuple[MatchResult, bool]:
        active_keys = self._active_variants()
        all_variants = self.library.known_variants()
        prefix_match = self.matcher.match_known(prefix, all_variants)
        if prefix_match.recipe_id is None: return prefix_match, False
        variant_id = make_variant_id(prefix)
        exact_archived = (variant_id in self.library.variants.get(prefix_match.recipe_id, {}) and (prefix_match.recipe_id, variant_id) not in active_keys)
        recipe_has_active_variant = any(recipe_id == prefix_match.recipe_id for recipe_id, _variant_id in active_keys)
        reentry_from_pruned = bool(exact_archived or not recipe_has_active_variant)
        return prefix_match, reentry_from_pruned

    def _finish_session(self, prefix_match: MatchResult, reentry_from_pruned: bool = False, apply_decay: bool = False) -> MatchResult:
        """Commit a complete known online sequence using matcher evidence."""
        prefix = list(self.current_prefix)
        transition_trace = list(self.current_trace)

        if not prefix:
            self._clear_online_session()
            return MatchResult("known", None, None, 0.0, 0.0)
        if prefix_match.recipe_id is None:
            match = MatchResult("assist_unavailable", None, None, prefix_match.jaccard, prefix_match.order_distance)
            self.classification_events.append((self.step_counter, match))
            self._log_commit(event="assist_unavailable", decision="none",   commit_applied=False, reason="no_known_recipe_candidates", confidence=None, candidate_recipe_id=None, candidate_variant_id=None, latest_pinned=False, promoted_from_tentative=False)
            self._clear_online_session(needs_observation=False)
            if apply_decay: self._update(apply_decay=True)
            return match

        recipe_id = prefix_match.recipe_id
        variant_id = make_variant_id(prefix)
        confidence, confidence_parts = self._commit_confidence(recipe_id, prefix, prefix_match)
        provisional_before = self.provisional.get((recipe_id, variant_id))
        full_threshold = self.settings.commit_threshold
        tentative_threshold = self.settings.tentative_threshold
        promotion_reasons = self._promotion_reasons(recipe_id, variant_id, confidence, confidence_parts)
        if promotion_reasons:
            confidence = max(confidence, full_threshold)
            confidence_parts["provisional_promotion"] = 1.0

        if confidence < tentative_threshold:
            match = MatchResult("known_recipe_uncertain", recipe_id, None, prefix_match.jaccard, prefix_match.order_distance)
            self.classification_events.append((self.step_counter, match))
            self._log_commit(event="commit_abstained", decision="none", commit_applied=False, reason="below_tentative_threshold", confidence=float(confidence), confidence_parts=dict(confidence_parts), candidate_recipe_id=recipe_id, candidate_variant_id=variant_id, latest_pinned=False, promoted_from_tentative=False)
            self._clear_online_session(needs_observation=False)
            if apply_decay: self._update(apply_decay=True)
            return match

        known_variant = variant_id in self.library.variants.get(recipe_id, {})
        if confidence < full_threshold and not known_variant:
            weight = max(self.settings.provisional_weight, min(self.settings.provisional_cap, confidence))
            trace = self._store_trace(prefix, transition_trace)
            self.replay.register(recipe_id, variant_id, tuple(prefix), self.demo_counter, self.retrain_cycle, weight=weight, pin_latest=False, transitions=trace)
            self.provisional[(recipe_id, variant_id)] = {"first_demo": self.demo_counter, "last_demo": self.demo_counter, "weight": weight, "confidence": confidence, "parts": confidence_parts, "promotion_eligible_until_demo": self.demo_counter + int(self.settings.confirm_window)}
            match = MatchResult("tentative_preference_shift", recipe_id, variant_id, prefix_match.jaccard, prefix_match.order_distance)
            decision = "tentative"
            latest_pinned = False
            self.narrate(f"[step {self.step_counter}] tentative online variant for '{recipe_id}' (confidence={confidence:.2f})")
        else:
            variant = self._register_if_live(recipe_id, prefix, self.step_counter, transitions=transition_trace)
            kind = "reentry_from_pruned" if reentry_from_pruned else ("known" if known_variant else "preference_shift")
            match = MatchResult(kind, recipe_id, variant.variant_id, prefix_match.jaccard, prefix_match.order_distance)
            self.provisional.pop((recipe_id, variant.variant_id), None)
            decision = "promotion" if promotion_reasons else ("known_refresh" if known_variant else "full")
            latest_pinned = True

        self._update(apply_decay)
        first_tentative_demo = (int(provisional_before.get("first_demo")) if provisional_before is not None else None)
        self._log_commit(
            event=("provisional_commit_promoted" if promotion_reasons else "online_commit"),
            decision=decision,
            commit_applied=True,
            reason=(";".join(promotion_reasons) if promotion_reasons else "confidence_gate_passed"),
            confidence=float(confidence),
            confidence_parts=dict(confidence_parts),
            candidate_recipe_id=recipe_id,
            candidate_variant_id=variant_id,
            known_variant=bool(known_variant),
            new_variant=not bool(known_variant),
            latest_pinned=bool(latest_pinned),
            promoted_from_tentative=bool(promotion_reasons),
            tentative_first_demo=first_tentative_demo,
            promotion_delay_demos=(int(self.demo_counter - first_tentative_demo) if promotion_reasons and first_tentative_demo is not None else None),
            promotion_reasons=tuple(promotion_reasons))
        self.classification_events.append((self.step_counter, match))
        self._clear_online_session()
        return match

    def _late_agreement(self, prefix: Sequence[str]) -> float:
        n = len(prefix)
        if n <= 0 or not self.step_log: return 0.0
        session_rows = self.step_log[-n:]
        if not session_rows:            return 0.0
        width = max(1, int(math.ceil(len(session_rows) * self.settings.agreement_window)))
        late_rows = session_rows[-width:]
        usable = [row for row in late_rows if row.predicted is not None]
        if not usable:                  return 0.0
        return sum(1 for row in usable if row.predicted == row.actual) / max(1, len(usable))

    def _promotion_reasons(self, recipe_id: str, variant_id: str, confidence: float, confidence_parts: Dict[str, float]) -> List[str]:
        provisional = self.provisional.get((recipe_id, variant_id))
        if provisional is None:
            return []
        window = int(self.settings.confirm_window)
        if self.demo_counter - int(provisional.get("first_demo", self.demo_counter)) > window:
            self.provisional.pop((recipe_id, variant_id), None)
            return []
        reasons = ["same_variant_recurred"]
        if confidence >= self.settings.commit_threshold:
            reasons.append("commit_confidence_full_threshold")
        if (float(confidence_parts.get("late_window_prediction_agreement", 0.0)) >= self.settings.confirm_accuracy and float(confidence_parts.get("recipe_jaccard", 0.0)) >= self.settings.confirm_similarity): reasons.append("late_window_prediction_alignment")
        provisional.update(last_demo=self.demo_counter, last_confidence=confidence, last_parts=dict(confidence_parts))
        return reasons

    def _commit_confidence(self, recipe_id: str, prefix: Sequence[str], prefix_match: MatchResult) -> Tuple[float, Dict[str, float]]:
        """Score a completed online sequence with deployable evidence only."""
        scoring_t0 = time.perf_counter()
        scored_before = self.matcher.variants_scored
        all_variants = self.library.known_variants()
        active_keys = self._active_variants()
        active_variants = [variant for variant in all_variants if (variant.recipe_id, variant.variant_id) in active_keys]
        scored = self.matcher.score(prefix, all_variants)
        recipe_score = float(scored.get(recipe_id, (None, float(prefix_match.jaccard or 0.0), 0.0))[1])
        second_jaccard = max((score for other_id, (_variant, score, _distance) in scored.items() if other_id != recipe_id), default=0.0)
        confidence_wall_s = time.perf_counter() - scoring_t0
        classification_wall_s = float(self.last_commit_stats.get("classification_wall_s", 0.0))
        self.last_commit_stats.update({"registry_size": len(all_variants), "active_variants": len(active_variants), "archived_variants": max(0, len(all_variants) - len(active_variants)), "registry_recipes": len(scored),
            "variants_scored": int(self.last_commit_stats.get("variants_scored", 0)) + self.matcher.variants_scored - scored_before, "confidence_scoring_wall_s": float(confidence_wall_s), "scoring_wall_s": classification_wall_s + float(confidence_wall_s)})
        jaccard_margin = max(0.0, recipe_score - second_jaccard)
        history = list(self._policy_history)
        policy_confidence = (sum(float(row.get("final_confidence") or row.get("confidence") or 0.0) for row in history) / len(history) if history else 0.0)
        late_agreement = self._late_agreement(prefix)
        margin_scale = self.settings.margin_scale
        score = (self.settings.match_weight * min(1.0, recipe_score) + self.settings.margin_weight * min(1.0, jaccard_margin / margin_scale) + self.settings.policy_weight * policy_confidence)
        score = min(1.0, score + self.settings.agreement_bonus * late_agreement)
        if (prefix_match.kind == "known" and recipe_score >= self.settings.known_similarity): score = max(score, self.settings.known_score_floor)
        # Archived-only fuzzy matches cannot self-commit on an artificial margin.
        if (all_variants and not active_variants and not (prefix_match.kind == "known" and recipe_score >= self.settings.known_similarity)): score = min(score, self.settings.empty_score_cap)
        # One key per distinct piece of evidence. Aliased duplicates were removed
        # so a reader cannot mistake the same quantity for independent support.
        parts = {"recipe_jaccard": recipe_score, "recipe_jaccard_margin": jaccard_margin, "online_policy_confidence": policy_confidence, "late_window_prediction_agreement": late_agreement}
        return max(0.0, min(1.0, float(score))), parts
    def _active_variants(self) -> Set[VariantKey]:
        """Current trainable memory keys. Pruned variants are excluded from online prediction."""
        return {entry.key for entry in self.replay.active_items()}

    def policy_stats(self) -> Dict[str, Any]:
        """Live policy telemetry, including the mask contract for assertions.
        Callers that persist per-turn rows should drop ACTION_MASK_CONTRACT keys
        via `varying_policy_stats`; they belong in the run manifest.
        """
        return {**ACTION_MASK_CONTRACT, **dict(self._policy_stats), **dict(self._last_action_mask_stats), **dict(getattr(self.maxent, "last_prediction_stats", {}) or {})}

    def varying_policy_stats(self) -> Dict[str, Any]:
        """`policy_stats` without the invariant keys, for per-turn persistence."""
        return {key: value for key, value in self.policy_stats().items() if key not in ACTION_MASK_CONTRACT}

    def _set_policy_stats(self, confidence: Optional[float], entropy: Optional[float], reason: str, *, margin: Optional[float] = None, source: Optional[str] = None) -> None:
        if self._frozen: return
        self._policy_stats = {"action_required": True, "confidence": confidence, "raw_confidence": confidence, "margin": margin, "entropy": entropy, "entropy_basis": "prediction_support", "predictor": source or "configured_policy", "final_confidence": confidence, "final_margin": margin, "final_entropy": entropy, "reason": reason}

    def _prediction_stats(self, distribution: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not distribution: return None, None, None
        probabilities = sorted((float(value) for value in distribution.values()), reverse=True)
        confidence = probabilities[0] if probabilities else None
        margin = probabilities[0] - probabilities[1] if len(probabilities) >= 2 else (probabilities[0] if probabilities else None)
        entropy = 0.0
        for probability in probabilities:
            if probability > 0: entropy -= probability * math.log(probability)
        if len(probabilities) > 1: entropy = entropy / math.log(len(probabilities))
        return confidence, entropy, margin

    def _replay_prefix(self, prefix: Sequence[str]) -> Tuple[int, ...]:
        with self._profile("state_from_prefix"):
            return self.domain.state_from_actions(prefix)

    def _entry_domain_task(self, entry: Any) -> str:
        """Name a replay entry's task the way the *domain* names it.

        ``entry.recipe_id`` is the label this agent allocated when the recipe
        was first observed (``R0``, ``R1``, ...). A multi-task domain has its
        own task identifiers and need not recognize ours, so scoping by our
        label raised on the domain's very first lookup and silenced the
        pruning audit for every entry. A domain that can recover the task from
        a state is asked to do so from the entry's own recorded transition;
        otherwise the agent's label is all there is, which is correct for the
        single-task and shared-namespace cases.
        """
        recover = getattr(self.domain, "recipe_of_state", None)
        transitions = getattr(entry, "transitions", ())
        if recover is not None and transitions:
            return str(recover(transitions[0][0]))
        return str(getattr(entry, "recipe_id", "") or "")

    @contextlib.contextmanager
    def _domain_scoped_to(self, recipe_id: Optional[str]):
        """Point a multi-task domain at the recipe a recording was made under.

        ``state_from_actions`` replays from ``domain.initial_state()``, which
        on a multi-recipe domain is the recipe the domain currently holds --
        the one most recently played, not the one the memory entry came from.
        Replaying a recorded ordering under the wrong recipe raises on its
        very first action, which silenced the pruning audit for every
        multi-recipe ladder. Single-task domains have no ``begin_task`` and
        are left untouched.
        """
        begin_task = getattr(self.domain, "begin_task", None)
        restore_to = getattr(self.domain, "recipe_id", None)
        if (
            begin_task is None
            or restore_to is None
            or not recipe_id
            or str(recipe_id) == str(restore_to)
        ):
            yield
            return
        begin_task(str(recipe_id))
        try:
            yield
        finally:
            begin_task(str(restore_to))

    def _known_action_universe(self) -> Tuple[str, ...]:
        """Union of grounded actions in active replay, without task routing."""
        actions = {self.domain.canonical_action(action) for entry in self.replay.active_items() for action in entry.ordering if action != "stop"}
        return tuple(sorted(actions))

    def _conditioned_actions(self, state: Tuple[int, ...], action_universe: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
        """Apply the shared mask using current-state preconditions only."""
        known_actions = self._known_action_universe()
        if action_universe is None: candidates = known_actions
        else:
            universe = {str(action) for action in action_universe}
            candidates = tuple(action for action in known_actions if action in universe)
        conditioned = self.domain.legal_actions(tuple(state), candidates)
        # Only the measured counts vary per decision. The mask's design
        # invariants live in ACTION_MASK_CONTRACT and are recorded once per run,
        # not restamped on every turn where they could not come out otherwise.
        self._last_action_mask_stats = {
            "action_universe_count": len(candidates),
            "feasible_action_count": len(conditioned),
            # How much choice this decision actually offered, defined on the
            # demonstrations rather than on any predictor's internals so every
            # arm is measured the same way.
            "observed_actions_at_state": len(self._observed_actions.get(tuple(state), ()))}
        return conditioned

    def _apply_shared_action_mask(self, state: Tuple[int, ...], distribution: Mapping[str, float]) -> Dict[str, float]:
        """Apply and renormalize the common state-only legality interface."""
        legal = set(self._conditioned_actions(state, tuple(distribution)))
        masked = {str(action): max(0.0, float(probability)) for action, probability in distribution.items() if str(action) in legal and math.isfinite(float(probability))}
        if not masked: return {}
        total = sum(masked.values())
        if total <= 0.0:
            probability = 1.0 / len(masked)
            return {action: probability for action in masked}
        return {action: probability / total for action, probability in masked.items()}

    def predict_actions(self, prefix: Optional[Sequence[str]] = None, *, state: Optional[Any] = None, actor_id: int = 0, action_universe: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Return the MaxEnt IRL policy over feasible task actions."""
        with self._profile("predict_actions"):
            prefix_actions = list(prefix) if prefix is not None else list(self.current_prefix)
            encoded_state = (self._replay_prefix(prefix_actions) if state is None else self.domain.state_key(state, actor_id=actor_id))
            candidates = self._conditioned_actions(encoded_state, action_universe)
            distribution = self.maxent.predict(encoded_state, candidates, prefix=prefix_actions, allow_latent_strategy=self._latent_strategy_confirmed)
            confidence, entropy, margin = self._prediction_stats(distribution)
            semantic_enabled = bool(self.settings.semantic_fallback_enabled)
            latent_enabled = bool(self.settings.latent_strategy_enabled)
            if semantic_enabled and latent_enabled: source = "maxent_with_semantic_fallback_and_latent_strategy"
            elif semantic_enabled:                  source = "maxent_with_semantic_fallback"
            elif latent_enabled:                    source = "maxent_with_latent_strategy"
            else:                                   source = "maxent_irl_only"
            self._set_policy_stats(confidence, entropy, source, margin=margin, source=source)
            return distribution

    def rank_actions(self, distribution: Mapping[str, float], k: int = 1) -> List[str]:
        """Apply seeded Gaussian tie-breaking without changing probabilities."""
        return top_actions(distribution, k=k, rng=self._tie_break_rng)

    def observe(self, observation: Observation, ground_truth_recipe: Optional[str] = None, *, precomputed_distribution: Optional[Mapping[str, float]] = None, precomputed_prediction: Optional[str] = None) -> StepResult:
        """Consume one observed semantic action and its state transition."""
        if self._frozen:
            action = observation.action
            distribution = (dict(precomputed_distribution) if precomputed_distribution is not None else self.predict_actions(self.current_prefix))
            predicted = (precomputed_prediction if precomputed_distribution is not None else (self.rank_actions(distribution, k=1)[0] if distribution else None))
            return StepResult(step=self.step_counter, predicted=predicted, actual=action, correct=predicted == action, mode=self.mode, event="frozen_prediction")

        self.step_counter += 1
        if self.mode == MODE_OBSERVE:
            action = observation.action
            self.pending_demo.append(action)
            self.pending_trace.append(self._transition(action, observation))
            result = StepResult(step=self.step_counter, predicted=None, actual=action, correct=False, mode=self.mode, event="observing")
            self.step_log.append(result)
            return result

        first_online_action = not self.current_prefix
        if first_online_action:
            distribution: Dict[str, float] = {}
        else:
            distribution = (dict(precomputed_distribution) if precomputed_distribution is not None else self.predict_actions(self.current_prefix))
            self._policy_history.append(dict(self._policy_stats))
        predicted = (None if first_online_action else (precomputed_prediction if precomputed_distribution is not None else (self.rank_actions(distribution, k=1)[0] if distribution else None)))
        action = observation.action
        correct = predicted == action
        if predicted is not None and not correct:
            self._prediction_mismatch_count += 1
            if self.maxent.latent_supports_correction((*self.current_prefix, action), tuple(distribution), action, predicted): self._latent_strategy_confirmed = True
        self.current_prefix.append(action)
        self.current_trace.append(self._transition(action, observation))
        result = StepResult(step=self.step_counter, predicted=predicted, actual=action, correct=correct, mode=self.mode)
        self.step_log.append(result)
        self.accuracy_events.append((self.step_counter, ground_truth_recipe or "?", correct))
        return result

    def evaluate_actions(self, ordering: Sequence[str], top_k: int = 3) -> Dict[str, float]:
        if not self._frozen:
            with self.frozen(): return self.evaluate_actions(ordering, top_k=top_k)
        actions = list(ordering)
        total = len(actions)
        rank_count = max(1, int(top_k))
        if total == 0:
            result = {"top_1": 0.0, "top_k": 0.0, "prediction_available_rate": 0.0, "empty_prediction_rate": 0.0, "cross_entropy": 0.0}
            result[f"top{rank_count}"] = 0.0
            return result

        prefix: List[str] = []
        top_1_hits = top_k_hits = available = empty_predictions = 0
        log_loss = 0.0
        floor = max(float(self.settings.min_probability), 1e-12)
        for actual in actions:
            distribution = self.predict_actions(prefix)
            ranked = (self.rank_actions(distribution, k=rank_count) if distribution else [])
            if distribution:
                available += 1
                top_1_hits += int(ranked[0] == actual)
                top_k_hits += int(actual in ranked)
                log_loss -= math.log(max(float(distribution.get(actual, floor)), floor))
            else:
                empty_predictions += 1
                log_loss -= math.log(floor)
            prefix.append(actual)

        result = {"top_1": top_1_hits / total, "top_k": top_k_hits / total, "prediction_available_rate": available / total, "empty_prediction_rate": empty_predictions / total, "cross_entropy": log_loss / total}
        result[f"top{rank_count}"] = top_k_hits / total
        return result

    def _build_demos(self, records):
        """Build O(L) trajectories from the observed transition traces.
        Training states come only from what was actually observed during the
        demonstration. A record without its trace is a defect rather than a
        case to approximate: reconstructing states from the action labels alone
        would fit the reward on a trajectory the human never performed, and the
        substitution would be invisible in the results.
        """
        trajectories = []
        dropped_total = 0
        for record in records:
            if isinstance(record, Mapping): actions = list(record.get("ordering", ()))
            else:                           actions = list(getattr(record, "ordering", record))
            trace = self._get_trace(record, actions)
            if not trace:
                raise ValueError(f"replay record for {len(actions)} actions carries no observed transition trace; refusing to train on reconstructed states")
            trajectory = []
            state = trace[0][0]
            for before, token, after in trace:
                if after == before:
                    dropped_total += 1
                    state = after
                    continue
                trajectory.append((before, token))
                state = after
            trajectory.append((state, "stop"))
            trajectories.append(trajectory)
        return trajectories, dropped_total

    def _index_observed_actions(self, trajectories: Sequence[List[Tuple[Tuple[int, ...], str]]]) -> None:
        """Record which actions were demonstrated at each encoded state.

        This is a property of the replayed demonstrations, not of a fitted
        model, so it classifies a decision identically for every predictor:
        one action means memorisation suffices, two or more is a real ranking
        problem, and an absent state means the answer must be generalised.
        """
        observed: Dict[Tuple[int, ...], Set[str]] = {}
        for trajectory in trajectories:
            for state, action in trajectory:
                if action == "stop": continue
                observed.setdefault(tuple(state), set()).add(str(action))
        self._observed_actions = observed

    def _log_training(self, dropped_actions: int, active_demos: int, *, total_wall_s: float = 0.0, build_wall_s: float = 0.0, fit_wall_s: float = 0.0, flop_estimate: float = 0.0, skipped: bool = False) -> None:
        """Append a structured retrain event. All agent variants funnel through this."""
        fit_stats = self._fit_stats() if not skipped else {}
        recorded_flops = float(flop_estimate)
        if not skipped and float(flop_estimate) <= 0.0:
            estimate = fit_stats.get("estimated_flops") if isinstance(fit_stats, dict) else None
            if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)): recorded_flops = float(estimate)
        event = {"step": self.step_counter, "cycle": int(self.retrain_cycle), "dropped_actions": int(dropped_actions), "active_demos": int(active_demos), "total_wall_s": float(total_wall_s), "build_wall_s": float(build_wall_s),
            "fit_wall_s": float(fit_wall_s), "flop_estimate": recorded_flops, "skipped": bool(skipped)}
        if fit_stats: event["fit_stats"] = fit_stats
        self.retrain_events.append(event)
        if not skipped:
            self.retrain_total_wall_times.append(float(total_wall_s))
            self.retrain_build_wall_times.append(float(build_wall_s))
            self.retrain_fit_wall_times.append(float(fit_wall_s))
            self.retrain_flop_estimates.append(recorded_flops)

    def _fit_stats(self) -> Dict[str, Any]:
        """Return structured accounting from the model fit that just ran."""
        stats: Dict[str, Any] = {}
        maxent_stats = getattr(getattr(self, "maxent", None), "last_fit_stats", None)
        if (self.settings.predictor == "maxent" and isinstance(maxent_stats, dict) and maxent_stats): stats.update(maxent_stats)
        cloning_stats = getattr(getattr(self, "cloner", None), "last_fit_stats", None)
        if isinstance(cloning_stats, dict) and cloning_stats:         stats.update(cloning_stats)
        custom_stats = getattr(self, "_custom_fit_stats", None)
        if isinstance(custom_stats, dict) and custom_stats: stats.update(custom_stats)
        if stats:
            stats["predictor"] = self.predictor_name()
            stats["irl_features"] = self.irl_feature_name()
        return stats

    def replay_stats(self) -> Dict[str, Any]:
        return {}

    def predictor_name(self) -> str:
        """Return the predictor that actually produces action probabilities."""
        return str(self.settings.predictor)

    def irl_feature_name(self) -> str:
        return (str(self.settings.irl_features) if self.settings.predictor == "maxent" else "not_applicable")

    def baseline_stats(self) -> Dict[str, Any]:
        return {}

    def training_stats(self) -> Dict[str, Any]:
        return {}

    def diagnostics(self) -> Dict[str, Any]:
        """Return the model-specific sections required by evaluation."""
        sections = {"fit_stats": self._fit_stats(), "replay_buffer": self.replay_stats(), "baseline_model_memory": self.baseline_stats(), "offline_pretraining": self.training_stats()}
        return {name: value for name, value in sections.items() if value}

    def _estimate_flops(self, trajectories: Sequence[List[Tuple[Tuple[int, ...], str]]]) -> float:
        """Estimate fit FLOPs from the fitted model's actual accounting."""
        stats = self._fit_stats()
        estimate = stats.get("estimated_flops") if isinstance(stats, dict) else None
        if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)): return float(estimate)
        # Fallback for external predictors without fit statistics.
        n_transitions = sum(max(0, len(demo) - 1) for demo in trajectories)
        n_actions = max(1, len({action for demo in trajectories for _state, action in demo}))
        feature_dim = len(trajectories[0][0][0]) if trajectories and trajectories[0] else 1
        iters = int(self.settings.irl_warm_steps if self.retrain_cycle > 1 else self.settings.irl_cold_steps)
        return float(n_transitions * n_actions * feature_dim * max(1, iters))

    def _fit_models(self, maxent: MaxEntIrl, trajectories: Sequence[List[Tuple[Tuple[int, ...], str]]], weights: Sequence[float], *, warm_start: bool, records: Optional[Sequence[Any]] = None) -> None:
        maxent.fit(trajectories, weights)

    def _fit_predictors(self, trajectories: Sequence[List[Tuple[Tuple[int, ...], str]]], weights: Sequence[float], *, warm_start: bool, records: Optional[Sequence[Any]] = None) -> None:
        self._fit_models(self.maxent, trajectories, weights, warm_start=warm_start, records=records)

    def _predictors_ready(self) -> bool:
        return self.maxent.reward_weights is not None

    def _reset_predictors(self) -> None:
        # Reuse the agent's stream so a cold restart continues one reproducible
        # sequence of initializations rather than restarting it.
        self.maxent = MaxEntIrl(settings=self.settings, domain=self.domain, rng=self._init_rng)

    def _demo_length(self, demo: Any) -> int:
        if isinstance(demo, Mapping):
            ordering = demo.get("ordering", ())
            return max(1, len(ordering))
        ordering = getattr(demo, "ordering", None)
        if ordering is not None:            return max(1, len(ordering))
        if isinstance(demo, (list, tuple)): return max(1, len(demo))
        return 1

    def _demo_weights(self, demos: Sequence[Any], base_weights: Sequence[float]) -> List[float]:
        """Equalize episode mass by length while preserving mean replay weight."""
        replay_weights = [float(weight) for weight in base_weights]
        if not demos or not replay_weights: return replay_weights
        count = min(len(demos), len(replay_weights))
        selected_demos      = demos[:count]
        replay_weights      = replay_weights[:count]
        normalized_weights  = [float(weight) / float(self._demo_length(demo)) for demo, weight in zip(selected_demos, replay_weights)]
        normalized_mean     = sum(normalized_weights) / max(1, len(normalized_weights))
        replay_mean         = sum(replay_weights) / max(1, len(replay_weights))
        if normalized_mean <= 0.0 or not math.isfinite(normalized_mean): return normalized_weights
        scale = replay_mean / normalized_mean
        return [float(weight * scale) for weight in normalized_weights]

    def _update(self, apply_decay: bool) -> None:
        if apply_decay: self.replay.step(self.demo_counter, self.retrain_cycle)
        self._retrain()

    def _retrain(self) -> None:
        if self._frozen: return

        retrain_t0  = time.perf_counter()
        current     = {key: float(entry.weight) for key, entry in self.replay.active.items()}
        previous    = self._last_observed_replay_weights
        added   = set(current) - set(previous)
        removed = set(previous) - set(current)
        weight_changed = {key for key in set(current) & set(previous) if not math.isclose(current[key], previous[key], rel_tol=0.0, abs_tol=1e-12)}
        self._last_observed_replay_weights = dict(current)

        counter_before = self.cold_change_count
        requested_start, trigger, projected = self.retrain_policy.decide(len(added), len(removed), bool(weight_changed), counter_before)

        def audit(effective_start: str, counter_after: int) -> None:
            self.retrain_events[-1].update({
                "retrain_requested_start": requested_start,             "retrain_effective_start": effective_start,             "retrain_trigger": trigger,
                "active_added_count": len(added),                       "active_removed_count": len(removed),                   "active_weight_changed_count": len(weight_changed),
                "additions_since_cold_before": counter_before,          "additions_since_cold_projected": projected,            "additions_since_cold_after": counter_after,
                "cold_threshold_basis": self.retrain_policy.cold_threshold_basis,})

        self.retrain_cycle += 1
        if requested_start == "skip":
            self.skipped_trains += 1
            self._log_training(dropped_actions=0, active_demos=len(self.replay.active), total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            audit("skipped", counter_before)
            return

        entries = self.replay.active_items()
        if not entries:
            self._observed_actions = {}
            self._reset_predictors()
            self.cold_change_count = 0
            self.skipped_trains += 1
            self._log_training(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            audit("clear_empty", 0)
            return

        effective_start = requested_start
        if requested_start == "cold":
            self._reset_predictors()
        elif not self._predictors_ready():
            self._reset_predictors()
            effective_start = "cold_fallback_no_model"

        build_t0 = time.perf_counter()
        with self._profile("retrain_build_trajectories"): trajectories, dropped_total = self._build_demos(entries)
        self._index_observed_actions(trajectories)
        build_wall_s = time.perf_counter() - build_t0
        weights = self._demo_weights(entries, [entry.weight for entry in entries])
        fit_t0 = time.perf_counter()
        with self._profile("retrain_fit_predictors"): self._fit_predictors(trajectories, weights, warm_start=(effective_start == "warm"), records=entries)
        fit_wall_s = time.perf_counter() - fit_t0
        self._log_training(dropped_actions=dropped_total, active_demos=len(entries), total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_flops(trajectories))
        if effective_start == "warm": self.cold_change_count = projected
        else: self.cold_change_count = 0
        audit(effective_start, self.cold_change_count)
        if self.settings.verbose:
            self.narrate(
                f"[step {self.step_counter}] {effective_start} retrained on "
                f"{len(trajectories)} weighted demos (cycle {self.retrain_cycle}, "
                f"post_grace_decay_rate={self.replay.post_grace_decay_rate:.6f}, "
                f"dropped_actions={dropped_total})"
            )

    def refresh(self) -> None:
        """Re-fit predictors against the current active memory without adding a demo."""
        self._retrain()

    def discard(self, keys: Sequence[VariantKey]) -> None:
        """Remove replay variants and repair both latest-variant indexes."""
        recipe_ids = {recipe_id for recipe_id, _variant_id in keys}
        for recipe_id, variant_id in keys: self.replay.discard(recipe_id, variant_id, allow_latest=True)
        for recipe_id in recipe_ids:
            latest = self.replay.latest_by_recipe.get(recipe_id)
            if latest is None:  self.library.latest.pop(recipe_id, None)
            else:               self.library.latest[recipe_id] = latest

    def _compare_policies(self, entries: Sequence[MemoryItem],  current: Callable[[Tuple[int, ...], Tuple[str, ...]], Mapping[str, float]], reference: Callable[[Tuple[int, ...], Tuple[str, ...]], Mapping[str, float]],
                        *, max_prefixes: int,                   tolerance: float,                                                           **metadata: Any) -> Dict[str, Any]:
        """Return the L1 policy gap between two predictors on shared prefixes.

        Magnitude only: which gaps constitute a contract violation is the
        caller's decision, because a warm-start gap and a pruned-data gap mean
        opposite things.
        """
        # Keyed by (recipe, prefix), not by prefix alone: the same token
        # sequence -- the empty prefix above all -- denotes a different state
        # under a different recipe, so deduplicating on the prefix would drop
        # every recipe after the first.
        prefixes: List[Tuple[str, Tuple[str, ...]]] = []
        seen: Set[Tuple[str, Tuple[str, ...]]] = set()
        for entry in entries:
            sequence = tuple(entry.ordering)
            recipe_id = self._entry_domain_task(entry)
            for length in (0, min(len(sequence), 1), len(sequence) // 2, max(0, len(sequence) - 1)):
                key = (recipe_id, sequence[:length])
                if key not in seen:
                    prefixes.append(key)
                    seen.add(key)
                if len(prefixes) >= max(1, int(max_prefixes)): break
            if len(prefixes) >= max(1, int(max_prefixes)): break
        differences = []
        for recipe_id, prefix in prefixes:
            # The predictors stay inside the scope: they read the domain too.
            with self._domain_scoped_to(recipe_id):
                state = self._replay_prefix(prefix)
                deployed = current(state, prefix)
                refit = reference(state, prefix)
            tokens = sorted(set(deployed) | set(refit))
            differences.append(sum(abs(float(deployed.get(token, 0.0)) - float(refit.get(token, 0.0))) for token in tokens))
        maximum = max(differences, default=0.0)
        return {"max_l1": float(maximum), "mean_l1": float(sum(differences) / max(1, len(differences))), "n_prefixes": len(prefixes), "tolerance": float(tolerance), **metadata}

    def audit_pruning(self, max_prefixes: int = 24, tolerance: float = 5e-2) -> Dict[str, Any]:
        """Audit what pruned replay does, and does not, do to the fitted policy.

        Three separate quantities, because the earlier single pass/fail
        conflated them and could never hold:

        ``active_only_training_inputs_verified``
            The contract. Exact set check that the records handed to the fit
            contain no pruned variant. This is what "trains on active memory
            only" asserts, and it is the only term that can fail.
        ``redundancy_*``
            Cold fit on active memory versus a cold fit on active plus pruned
            memory, same initialization stream. A large gap means the decayed
            variants carried information the survivors do not; a small one
            means the retention decision cost nothing.
        ``deployed_path_dependence_*``
            The deployed model versus a fresh cold fit on the same active set.
            Non-zero whenever the deployed weights were warm-started, which is
            ordinary optimizer path dependence and not a leak: a warm start
            carries weights, never demonstrations. Comparing the deployed model
            against a cold refit was the previous test, which is why it
            reported a violation at every checkpoint on every agent, including
            agents that prune nothing at all.
        """
        entries = self.replay.active_items()
        pruned_records = list(self.replay.pruned.values())
        active_keys = {entry.key for entry in entries}
        pruned_keys = {record.key for record in pruned_records}
        inputs_verified = not (active_keys & pruned_keys)
        base = {"passed": bool(inputs_verified), "active_only_training_inputs_verified": bool(inputs_verified),
                "comparison": "cold_active_vs_cold_active_plus_pruned", "model_family": "maxent_irl",
                "n_active_variants": len(entries), "n_pruned_variants": len(pruned_records),
                "tolerance": float(tolerance), "audit_reference_seed": int(self.settings.seed)}
        if not entries:
            return {**base, "max_l1": 0.0, "mean_l1": 0.0, "n_prefixes": 0, "pruned_available": False,
                    "redundancy_max_l1": 0.0, "redundancy_mean_l1": 0.0,
                    "deployed_path_dependence_max_l1": 0.0, "deployed_path_dependence_mean_l1": 0.0}

        # Every reference fit draws from its own fresh stream seeded identically,
        # so the two sides differ only in replay membership.
        audit_seed = int(self.settings.seed)

        def cold_fit(records: Sequence[Any], record_weights: Sequence[float]) -> MaxEntIrl:
            trajectories, _dropped = self._build_demos(records)
            model = MaxEntIrl(settings=self.settings, domain=self.domain, rng=np.random.default_rng(audit_seed))
            self._fit_models(model, trajectories, self._demo_weights(records, record_weights), warm_start=False, records=records)
            return model

        def predict_with(model: MaxEntIrl, state: Tuple[int, ...], prefix: Tuple[str, ...]) -> Mapping[str, float]:
            return model.predict(state, self._conditioned_actions(state), prefix=prefix)

        active_reference = cold_fit(entries, [float(entry.weight) for entry in entries])
        path_dependence = self._compare_policies(
            entries,
            lambda state, prefix: predict_with(self.maxent, state, prefix),
            lambda state, prefix: predict_with(active_reference, state, prefix),
            max_prefixes=max_prefixes, tolerance=tolerance)
        deployed = {"deployed_path_dependence_max_l1": float(path_dependence["max_l1"]),
                    "deployed_path_dependence_mean_l1": float(path_dependence["mean_l1"])}

        if not pruned_records:
            return {**base, **deployed, "max_l1": 0.0, "mean_l1": 0.0, "n_prefixes": int(path_dependence["n_prefixes"]),
                    "pruned_available": False, "redundancy_max_l1": 0.0, "redundancy_mean_l1": 0.0}

        combined = list(entries) + pruned_records
        restored_reference = cold_fit(combined, [float(entry.weight) for entry in entries] + [1.0] * len(pruned_records))
        redundancy = self._compare_policies(
            entries,
            lambda state, prefix: predict_with(active_reference, state, prefix),
            lambda state, prefix: predict_with(restored_reference, state, prefix),
            max_prefixes=max_prefixes, tolerance=tolerance)
        return {**base, **deployed, **redundancy, "pruned_available": True,
                "redundancy_max_l1": float(redundancy["max_l1"]), "redundancy_mean_l1": float(redundancy["mean_l1"])}

    def evaluate(self, ordering: Sequence[str]) -> float:
        """Prefix-conditioned accuracy of the configured predictor on one ordering."""
        if not ordering: return 0.0
        return float(self.evaluate_actions(ordering, top_k=1)["top_1"])
