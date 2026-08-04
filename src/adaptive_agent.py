"""Adaptive HRC agent state machine.

Modes:
    MODE_OBSERVE:   The human is showing a brand-new recipe. The agent buffers actions silently until ``end_demo()`` is called, then classifies the buffer against existing variants. The disambiguator decides whether to create a new recipe or treat the demo as a new preference variant.

    MODE_ONLINE:    Default mode. The agent predicts at every step. If the human action disagrees with top-1, the agent uses partial disambiguation to track the best-matching known variant for session-boundary commit. Prediction remains in the IRL and state-aware n-gram heads. On `end_demo()`, the matched variant is
    promoted to latest and its rehearsal weight is reset to 1.0. Retraining is requested after each end_demo() or online preference commit, but the expensive fit is gated on active replay membership changes. Each executed ordinary retrain rebuilds predictors from the active replay set so pruned demos cannot survive through stale fitted parameters or feature-normalizer statistics.

Terminology:         "Adaptive rehearsal weighting" operates on per-demo replay weights `w_i`. The full agent refits the IRL head from active memory rather than decaying or carrying forward `theta`. Online fine-tuning is supervised next-action imitation over weighted active rehearsal; observed human actions are the ground-truth labels.
Freeze mode (set_frozen / frozen context manager):      When frozen, all mutating paths are no-ops. Predictions and logging still work so probes produce per-step data. `frozen()` raises AssertionError on exit if any state was accidentally mutated.
"""
from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import StateTracker
from .memory import (Classification,    DecayManager,       DemoTransition, Disambiguator,  Entry,          Variant,        VariantKey,                 VariantMemory,      variant_hash)
from .models import (Config,            DEFAULT_CONFIG,     MaxEntIRL2,     NGramMarkov,       ensemble_predict,   top_k)
from .representations import (ActionObservation,    ActionVector,   apply_transition_vector,    identity_token_from_observation, observations_from_actions)

MODE_OBSERVE = "observe"
MODE_ONLINE = "online"


def _silent_narrate(_msg: str) -> None:
    """Module-level no-op narrator. Picklable; replaces the previous lambda default so agents can survive ProcessPoolExecutor (spawn) and copy.deepcopy."""
    return None


@dataclass
class StepResult:
    step: int
    predicted: Optional[str]
    actual: str
    correct: bool
    mode: str
    event: str = ""   # free-form tag for the narrator


class AdaptiveHRCAgent:
    """End-to-end continual-learning HRC agent."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, narrate: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.decay = DecayManager(cfg)
        self.memory = VariantMemory(cfg)
        self.disambig = Disambiguator(cfg)
        self.online_commit_events: List[Dict[str, Any]] = []

        self.irl = MaxEntIRL2(cfg=cfg)
        self.markov = NGramMarkov(
            order=cfg.markov_order,
            prob_floor=cfg.prob_floor,
            state_log_weight=cfg.ngram_state_log_weight,
            prefix_log_weight=cfg.ngram_prefix_log_weight,
        )

        self.mode = MODE_ONLINE
        self.pending_demo: List[str] = []
        self.pending_demo_transitions: List[DemoTransition] = []
        self.pending_demo_identity: List[str] = []
        # Online state
        self.current_prefix: List[str] = []
        self.current_transition_trace: List[DemoTransition] = []
        self.current_identity_prefix: List[str] = []
        self._needs_observation: bool = False
        self._online_unknown_streak: int = 0
        self._online_step_status: str = "known_confident"
        self._online_policy_history: List[Dict[str, Any]] = []
        self.provisional_commits: Dict[VariantKey, Dict[str, Any]] = {}
        self._last_action_policy: Dict[str, Any] = {
            "robot_action_mandatory": True,
            "action_confidence": None,
            "raw_action_confidence": None,
            "action_margin": None,
            "action_support_normalized_entropy": None,
            "action_entropy_normalization": "prediction_support",
            "policy_source": None,
            "final_action_confidence": None,
            "final_action_margin": None,
            "final_action_support_normalized_entropy": None,
            "reason": "cold_start",
        }
        # Anonymous action codebook.  The learner sees action vectors; the strings here are stable internal IDs, not symbolic kitchen actions.
        self.action_vector_to_token: Dict[ActionVector, str] = {}
        self.token_to_action_vector: Dict[str, ActionVector] = {}
        # Exact observed transition traces keyed by anonymous token sequence. Values are (state_before, token, state_after) tuples, never symbolic simulator action strings.
        self.demo_transition_traces: Dict[Tuple[str, ...], Tuple[DemoTransition, ...]] = {}
        self._next_recipe_idx = 1
        # Bookkeeping
        self.step_counter = 0       # actions observed
        self.session_counter = 0    # completed demos (decay tick rate)
        self.retrain_cycle = 0
        self.observation_mode_entries = 0
        self.narrate = narrate or _silent_narrate
        # Metrics for the paper
        self.step_log: List[StepResult] = []
        self.accuracy_events: List[Tuple[int, str, bool]] = []  # (step,recipe,correct)
        # Each entry records retrain timing and the number of no-op action transitions detected while converting active demos into trajectories.
        self.retrain_events: List[Dict[str, Any]] = []
        self.retrain_fit_wall_times: List[float] = []
        self.retrain_total_wall_times: List[float] = []
        self.retrain_build_wall_times: List[float] = []
        self.retrain_flop_estimates: List[float] = []
        self.adaptation_latencies: List[Tuple[int, str, int]] = []  # (step,recipe,latency)
        self.classification_events: List[Tuple[int, Classification]] = []

        # Freeze state: gates all mutating paths when True.
        self._frozen: bool = False
        self._freeze_snapshot: Optional[Tuple] = None

        # Diagnostic profile: populated only when cfg.profile is True. Maps event name -> (n_calls, total_wall_s). Negligible overhead when off.
        self.profile: Dict[str, Tuple[int, float]] = {}

        # Fingerprint of active replay membership at the last successful fit. Weight-only decay updates intentionally do not invalidate the fit; additions and removals do.
        self._last_fit_fingerprint: Optional[frozenset] = None
        # Public counter of how many _retrain calls hit the skip-fingerprint gate. Always tracked (not conditional on cfg.profile). Surfaced in memory_stats so reports can distinguish "n_retrains called" from "n_fits actually executed", which makes `retrain_count` interpretable.
        self.retrain_skipped_count: int = 0

    @contextlib.contextmanager
    def _profile(self, event: str):
        """No-cost context manager when cfg.profile is False."""
        if not self.cfg.profile:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            n, w = self.profile.get(event, (0, 0.0))
            self.profile[event] = (n + 1, w + (time.perf_counter() - t0))

    ############### snapshot
    def snapshot(self) -> "AdaptiveHRCAgent":
        """Deep-copy the agent for phase-A sweep reuse. Lossless and pure (no leakage between branches)."""
        if self._frozen: raise RuntimeError("snapshot called while frozen; release frozen first")
        return copy.deepcopy(self)

    def restore_from(self, snapshot: "AdaptiveHRCAgent") -> None:
        """Overwrite this agent's state from a previously-taken snapshot, in place."""
        if type(self) is not type(snapshot): raise TypeError(f"restore_from type mismatch: {type(self).__name__} vs {type(snapshot).__name__}")
        fresh = copy.deepcopy(snapshot)
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)

    ############### freeze (deepcopy-and-restore contract)
    def set_frozen(self, value: bool) -> None:
        """Freeze or unfreeze the agent.
        When freezing, deep-copy all instance state (excluding the snapshot pointer and the frozen flag) so that any mutation during the frozen window is rolled back at unfreeze. This is the safety contract for evaluation: the agent's externally-visible state at exit is guaranteed bit-identical to entry, 
        so eval probes cannot leak into training.
        """
        if value and not self._frozen:
            with self._profile("freeze_snapshot_enter"): self._freeze_snapshot = copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in ("_freeze_snapshot", "_frozen")})
            self._frozen = True
        elif not value and self._frozen:
            snap = self._freeze_snapshot
            if snap is not None:
                with self._profile("freeze_snapshot_exit"):
                    for k, v in snap.items(): self.__dict__[k] = v
            self._freeze_snapshot = None
            self._frozen = False

    def _frozen_structural_digest(self) -> Dict[str, Any]:
        """Small diagnostic digest for state that must not change in frozen eval."""
        active = tuple(sorted(
            (
                rid,
                h,
                tuple(e.ordering),
                round(float(e.weight), 12),
                int(e.added_step),
                int(e.added_cycle),
                int(e.last_seen_step),
                tuple(e.transitions),
                tuple(e.identity_ordering),
            )
            for (rid, h), e in self.decay.active.items()
        ))
        pruned = tuple(sorted(
            (
                rid,
                h,
                tuple(e.ordering),
                int(e.added_step),
                int(e.removed_step),
                int(e.added_cycle),
                int(e.removed_cycle),
                int(e.last_seen_step),
                tuple(e.transitions),
                tuple(e.identity_ordering),
            )
            for (rid, h), e in self.decay.pruned.items()
        ))
        variant_keys = tuple(sorted((rid, h) for rid, slot in self.memory.variants.items() for h in slot))
        return {
            "mode": self.mode,
            "step_counter": int(self.step_counter),
            "session_counter": int(self.session_counter),
            "retrain_cycle": int(self.retrain_cycle),
            "current_prefix": tuple(self.current_prefix),
            "current_transition_trace": tuple(self.current_transition_trace),
            "current_identity_prefix": tuple(self.current_identity_prefix),
            "pending_demo": tuple(self.pending_demo),
            "pending_demo_transitions": tuple(self.pending_demo_transitions),
            "pending_demo_identity": tuple(self.pending_demo_identity),
            "demo_transition_traces": tuple(sorted(self.demo_transition_traces.items())),
            "step_log_len": len(self.step_log),
            "classification_events_len": len(self.classification_events),
            "accuracy_events_len": len(self.accuracy_events),
            "retrain_events_len": len(self.retrain_events),
            "online_commit_events_len": len(self.online_commit_events),
            "active": active,
            "pruned": pruned,
            "variant_keys": variant_keys,
            "memory_latest": tuple(sorted(self.memory.latest.items())),
            "latest_by_recipe": tuple(sorted(self.decay.latest_by_recipe.items())),
            "latest_keys": tuple(sorted(self.decay.latest_keys)),
            "last_fit_fingerprint": tuple(sorted(self._last_fit_fingerprint)) if self._last_fit_fingerprint is not None else None,
        }

    @contextlib.contextmanager
    def frozen(self):
        self.set_frozen(True)
        _before = self._frozen_structural_digest()
        try:
            yield self
        finally:
            # Capture BEFORE restore so mutated state is still visible.
            _after = self._frozen_structural_digest()
            self.set_frozen(False)   # restores snapshot
            if _before != _after:
                changed = sorted(k for k in _before if _before.get(k) != _after.get(k))
                raise RuntimeError(f"frozen() invariant violated: structural state changed ({', '.join(changed)})")

    # codebook serialisation
    def save_codebook(self) -> Dict[str, Any]:
        """Serialise the action codebook for checkpoint reproducibility.

        The codebook maps anonymous token IDs (e.g. ``'act_0001'``) to the action
        vectors seen during training.  Without it, per-action metrics saved to
        ``result.json`` cannot be decoded back to symbolic kitchen actions after the
        fact.  Include this alongside the agent state in any experiment checkpoint.

        Returns a JSON-serialisable dict with keys:
          ``token_to_vector``  – token_id -> list(action_vector)
          ``vector_to_token``  – str(action_vector) -> token_id
          ``n_tokens``         – total number of registered tokens
        """
        return {
            "token_to_vector": {tok: list(vec) for tok, vec in self.token_to_action_vector.items()},
            "vector_to_token": {str(list(vec)): tok for vec, tok in self.action_vector_to_token.items()},
            "n_tokens": len(self.action_vector_to_token),
        }

    def load_codebook(self, codebook: Dict[str, Any]) -> None:
        """Restore the action codebook from a previously saved dict.

        Call **before** any ``observe_observation`` or ``step`` calls when loading
        a checkpoint; otherwise the agent will allocate fresh token IDs that conflict
        with those embedded in the saved memory/decay state.

        Args:
            codebook: A dict produced by a prior call to ``save_codebook()``.

        Raises:
            ValueError: If the codebook format is invalid.
        """
        if "token_to_vector" not in codebook:
            raise ValueError("codebook must contain 'token_to_vector'; got: " + str(list(codebook)))
        self.action_vector_to_token = {}
        self.token_to_action_vector = {}
        for tok, vec_list in codebook["token_to_vector"].items():
            vec = tuple(vec_list)
            self.token_to_action_vector[str(tok)] = vec
            self.action_vector_to_token[vec] = str(tok)

    # mode control
    def start_demo(self) -> None:
        """Human signals 'I am about to show a new recipe.'"""
        if self._frozen: return
        self.mode = MODE_OBSERVE
        self.pending_demo = []
        self.pending_demo_transitions = []
        self.pending_demo_identity = []
        self._needs_observation = False
        self._online_unknown_streak = 0
        self._online_step_status = "known_confident"
        self._online_policy_history = []
        self.observation_mode_entries += 1
        self.narrate(f"[step {self.step_counter}] MODE -> OBSERVE (new-demo gate ON)")

    def end_demo(self) -> Classification:
        """End a demonstration or collaboration session and commit a known sequence."""
        # When frozen, reset transient state and return without mutating anything.
        if self._frozen: return Classification("frozen", None, None, 0.0, 0.0)

        # Guard against an empty-session end_demo: clicking end-demo without observing any actions should reset state cleanly, not register an empty variant or advance decay (which would erode memory on a noop).
        if self.mode == MODE_OBSERVE and not self.pending_demo:
            self.narrate(f"[step {self.step_counter}] end_demo called in OBSERVE with empty buffer - no-op")
            self.mode = MODE_ONLINE
            self.pending_demo_transitions = []
            self.pending_demo_identity = []
            return Classification("known", None, None, 0.0, 0.0)
        if self.mode == MODE_ONLINE and not self.current_prefix:
            self.current_prefix = []
            self.current_transition_trace = []
            self.current_identity_prefix = []
            self._needs_observation = False
            self._online_unknown_streak = 0
            self._online_step_status = "known_confident"
            self._online_policy_history = []
            return Classification("known", None, None, 0.0, 0.0)
        if self.mode == MODE_OBSERVE:
            self.session_counter += 1
            return self._end_observe_demo(apply_decay=True)

        commit_cls, reentry_from_pruned = self._classify_online_prefix(self.current_prefix)
        if commit_cls.kind == "new_recipe":
            cls = Classification("needs_observation", None, None, commit_cls.jaccard, commit_cls.order_distance)
            self.classification_events.append((self.step_counter, cls))
            self.current_prefix = []
            self.current_identity_prefix = []
            self._needs_observation = True
            self._online_unknown_streak = int(getattr(self.cfg, "online_unknown_confirm_streak", 3))
            self._online_step_status = "unknown_confirmed"
            self._online_policy_history = []
            self.narrate(f"[step {self.step_counter}] online sequence did not match known recipes; observation mode required")
            return cls

        # Only known recipe/preference sessions advance decay and mutate memory. The currently-demonstrated variant is protected by the latest-pin in `decay.latest_keys` (set by `mark_latest`), so no extra protected_keys plumbing is needed here.
        self.session_counter += 1
        return self._end_online_session(commit_cls, reentry_from_pruned, apply_decay=True)

    def _transition_from_observation(self, token: str, observation: ActionObservation) -> DemoTransition:
        state = tuple(int(x) for x in observation.state)
        next_state = tuple(int(x) for x in observation.next_state)
        return (state, token, next_state)

    def _normalize_transition_trace(
        self,
        transitions: Optional[Sequence[DemoTransition]],
    ) -> Tuple[DemoTransition, ...]:
        if not transitions:
            return ()
        out: List[DemoTransition] = []
        for state, token, next_state in transitions:
            out.append((
                tuple(int(x) for x in state),
                str(token),
                tuple(int(x) for x in next_state),
            ))
        return tuple(out)

    def _remember_transition_trace(
        self,
        ordering: Sequence[str],
        transitions: Optional[Sequence[DemoTransition]],
    ) -> Tuple[DemoTransition, ...]:
        trace = self._normalize_transition_trace(transitions)
        ordering_tuple = tuple(str(t) for t in ordering)
        if trace and tuple(t[1] for t in trace) == ordering_tuple:
            self.demo_transition_traces[ordering_tuple] = trace
            return trace
        return self.demo_transition_traces.get(ordering_tuple, ())

    def _trace_for_demo(self, demo: Any, tokens: Sequence[str]) -> Tuple[DemoTransition, ...]:
        trace: Tuple[DemoTransition, ...] = ()
        if isinstance(demo, Entry):
            trace = self._normalize_transition_trace(getattr(demo, "transitions", ()))
        elif isinstance(demo, Mapping):
            trace = self._normalize_transition_trace(demo.get("transitions", ()))
        if not trace:
            trace = self.demo_transition_traces.get(tuple(tokens), ())
        if trace and tuple(t[1] for t in trace) == tuple(tokens):
            return trace
        return ()

    def _register_if_live(
        self,
        rid: str,
        seq: List[str],
        step: int,
        transitions: Optional[Sequence[DemoTransition]] = None,
        identity_ordering: Optional[Sequence[str]] = None,
    ) -> Variant:
        """Register a completed demonstration in variant memory and decay state."""
        identity = tuple(identity_ordering) if identity_ordering is not None else tuple(seq)
        v = self.memory.register(rid, seq, step, identity_ordering=identity)
        if not self._frozen:
            transition_trace = self._remember_transition_trace(seq, transitions)
            entry = self.decay.register(
                rid,
                v.variant_hash,
                tuple(seq),
                self.session_counter,
                self.retrain_cycle,
                transitions=transition_trace,
                identity_ordering=identity,
            )
            self._record_committed_replay_demo(
                rid,
                v.variant_hash,
                tuple(seq),
                transitions=transition_trace,
                session_step=self.session_counter,
                action_step=step,
                source_mode=self.mode,
                entry=entry,
            )
            self._assert_latest_pin_invariant(rid)
        return v

    def _record_committed_replay_demo(
        self,
        recipe_id: str,
        variant_hash_: str,
        ordering: Tuple[str, ...],
        *,
        transitions: Tuple[DemoTransition, ...] = (),
        session_step: int,
        action_step: int,
        source_mode: str,
        entry: Optional[Entry] = None,
    ) -> None:
        """Hook for replay baselines; the full agent trains from active memory."""
        return None

    def _assert_latest_pin_invariant(self, rid: Optional[str] = None) -> None:
        if self._frozen or not getattr(self.cfg, "protect_latest_preference", True): return
        recipe_ids = [rid] if rid is not None else [r for r, slot in self.memory.variants.items() if slot]
        for recipe_id in recipe_ids:
            slot = self.memory.variants.get(recipe_id, {})
            if not slot: continue
            latest = self.memory.latest.get(recipe_id)
            key = (recipe_id, latest) if latest is not None else None
            if latest is None or latest not in slot:                    raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: memory.latest missing from registry")
            if self.decay.latest_by_recipe.get(recipe_id) != latest:    raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: decay latest does not match memory latest")
            if key not in self.decay.latest_keys:       raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest key is not pinned")
            if key not in self.decay.active:            raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest key is not active")
            if key in self.decay.pruned:                raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest key is pruned")
            if self.decay.active[key].weight != 1.0:    raise RuntimeError(f"latest-pin invariant failed for {recipe_id}: latest weight is not 1.0")

    def _end_observe_demo(self, apply_decay: bool = False) -> Classification:
        seq = list(self.pending_demo)
        transition_trace = list(self.pending_demo_transitions)
        identity_seq = list(self.pending_demo_identity)
        active_keys = self._active_keys()
        active_lib = self.memory.library(allowed_keys=active_keys)
        cls = self.disambig.classify(seq, active_lib, identity_sequence=identity_seq)
        # Pruned-reentry detection: when active-only says new_recipe, check the full library. If a pruned variant matches, restore it rather than creating a new recipe ID.
        if cls.kind == "new_recipe" and active_keys:
            full_lib = self.memory.library()
            if len(full_lib) > len(active_lib):
                archived_cls = self.disambig.classify(seq, full_lib, identity_sequence=identity_seq)
                if archived_cls.kind != "new_recipe":
                    cls = archived_cls          # restore under original recipe_id
                    self.narrate(f"[step {self.step_counter}] observe-mode: restoring pruned variant of '{cls.recipe_id}'")

        self.classification_events.append((self.step_counter, cls))
        # Register in memory + decay.
        if cls.kind == "new_recipe":
            rid = f"R{self._next_recipe_idx}"
            self._next_recipe_idx += 1
            self._register_if_live(rid, seq, self.step_counter, transitions=transition_trace, identity_ordering=identity_seq)
            cls.recipe_id = rid
            self.narrate(f"[step {self.step_counter}] classified NEW RECIPE '{rid}' (jaccard={cls.jaccard:.2f})")
        else:
            if cls.recipe_id is None: raise RuntimeError(f"disambiguator returned {cls.kind} without a recipe_id")
            rid = cls.recipe_id
            self._register_if_live(rid, seq, self.step_counter, transitions=transition_trace, identity_ordering=identity_seq)
            kind_msg = "KNOWN (re-demo)" if cls.kind == "known" else "PREFERENCE VARIANT"
            self.narrate(f"[step {self.step_counter}] classified {kind_msg} of '{rid}' (jaccard={cls.jaccard:.2f}, tau={cls.order_distance:.2f})")
        self.mode = MODE_ONLINE
        self.pending_demo = []
        self.pending_demo_transitions = []
        self.pending_demo_identity = []
        self._needs_observation = False
        self._online_unknown_streak = 0
        self._online_step_status = "known_confident"
        self._online_policy_history = []
        if apply_decay: self.decay.step(self.session_counter, self.retrain_cycle)
        self._retrain()
        return cls

    def _classify_online_prefix(self, prefix: Sequence[str]) -> Tuple[Classification, bool]:
        active_keys = self._active_keys()
        active_lib = self.memory.library(allowed_keys=active_keys)
        identity_prefix = (
            list(self.current_identity_prefix)
            if tuple(prefix) == tuple(self.current_prefix)
            and len(self.current_identity_prefix) == len(prefix)
            else list(prefix)
        )
        # For short prefixes, score_partial is more reliable than full Jaccard because Jaccard(short_prefix, full_variant) is always near 0. The full classify() is used only when the prefix is long enough for Jaccard to be meaningful (controlled by min_classify_length in Config).
        min_len = int(getattr(self.cfg, "min_classify_length", 6))
        if len(prefix) < min_len:
            # Use score_partial to get a ranked recipe list but wrap it as a Classification for the rest of the method to consume.
            ranked = self.disambig.score_partial(identity_prefix, active_lib, identity=True)
            if ranked and ranked[0][1] >= float(getattr(self.cfg, "online_new_recipe_partial_threshold", 0.30)):
                best_v, best_score = ranked[0]
                commit_cls = Classification("preference_shift", best_v.recipe_id, None, best_score, 0.0)
            else: commit_cls = Classification("new_recipe", None, None, 0.0, 0.0)
        else: commit_cls = self.disambig.classify(prefix, active_lib, identity_sequence=identity_prefix)
        reentry_from_pruned = False
        if prefix:
            h = variant_hash(prefix)
            for rid, slot in self.memory.variants.items():
                if h in slot and (rid, h) not in active_keys: return Classification("known", rid, h, 1.0, 0.0), True
        if commit_cls.kind == "new_recipe":
            # No active match reached threshold. Consult the pruned-variant metadata/full registry so a forgotten known variant can be restored at commit time.
            full_lib = self.memory.library()
            if len(full_lib) > len(active_lib):
                full_cls = self.disambig.classify(prefix, full_lib, identity_sequence=identity_prefix)
                if full_cls.kind != "new_recipe":
                    commit_cls = full_cls
                    reentry_from_pruned = True
        return commit_cls, reentry_from_pruned

    def _end_online_session(self, commit_cls: Classification, reentry_from_pruned: bool = False, apply_decay: bool = False) -> Classification:
        """Commit a complete known online sequence using disambiguator evidence."""
        prefix = list(self.current_prefix)
        transition_trace = list(self.current_transition_trace)
        identity_prefix = list(self.current_identity_prefix)

        def clear_session(*, needs_observation: bool = False) -> None:
            self.current_prefix = []
            self.current_transition_trace = []
            self.current_identity_prefix = []
            self._needs_observation = needs_observation
            self._online_unknown_streak = (
                int(getattr(self.cfg, "online_unknown_confirm_streak", 3))
                if needs_observation else 0
            )
            self._online_step_status = "unknown_confirmed" if needs_observation else "known_confident"
            self._online_policy_history = []

        if not prefix:
            clear_session()
            return Classification("known", None, None, 0.0, 0.0)
        if commit_cls.kind == "new_recipe" or commit_cls.recipe_id is None:
            if apply_decay and self.session_counter > 0:
                self.session_counter -= 1
            cls = Classification("needs_observation", None, None, commit_cls.jaccard, commit_cls.order_distance)
            self.classification_events.append((self.step_counter, cls))
            clear_session(needs_observation=True)
            return cls

        rid = commit_cls.recipe_id
        h = variant_hash(prefix)
        confidence, confidence_parts = self._online_commit_confidence(rid, prefix, commit_cls)
        full_threshold = float(getattr(self.cfg, "online_commit_full_threshold", 0.75))
        tentative_threshold = float(getattr(self.cfg, "online_commit_tentative_threshold", 0.45))
        promotion_reasons = self._provisional_promotion_reasons(rid, h, confidence, confidence_parts)
        if promotion_reasons:
            confidence = max(confidence, full_threshold)
            confidence_parts["provisional_promotion"] = 1.0

        if confidence < tentative_threshold:
            if apply_decay and self.session_counter > 0:
                self.session_counter -= 1
            cls = Classification("needs_observation", rid, None, commit_cls.jaccard, commit_cls.order_distance)
            self.classification_events.append((self.step_counter, cls))
            clear_session(needs_observation=True)
            return cls

        known_variant = h in self.memory.variants.get(rid, {})
        if confidence < full_threshold and not known_variant:
            weight = max(float(getattr(self.cfg, "provisional_commit_weight", 0.20)), min(0.75, confidence))
            trace = self._remember_transition_trace(prefix, transition_trace)
            self.decay.register(
                rid, h, tuple(prefix), self.session_counter, self.retrain_cycle,
                weight=weight, pin_latest=False, transitions=trace,
                identity_ordering=identity_prefix,
            )
            self.provisional_commits[(rid, h)] = {
                "first_session": self.session_counter,
                "last_session": self.session_counter,
                "weight": weight,
                "confidence": confidence,
                "parts": confidence_parts,
                "promotion_eligible_until": self.session_counter + int(getattr(self.cfg, "provisional_confirm_window", 5)),
            }
            cls = Classification("tentative_preference_shift", rid, h, commit_cls.jaccard, commit_cls.order_distance)
            self.narrate(f"[step {self.step_counter}] tentative online variant for '{rid}' (confidence={confidence:.2f})")
        else:
            variant = self._register_if_live(
                rid, prefix, self.step_counter, transitions=transition_trace,
                identity_ordering=identity_prefix,
            )
            kind = "reentry_from_pruned" if reentry_from_pruned else (
                "known" if known_variant else "preference_shift"
            )
            cls = Classification(kind, rid, variant.variant_hash, commit_cls.jaccard, commit_cls.order_distance)
            self.provisional_commits.pop((rid, variant.variant_hash), None)
            if promotion_reasons:
                self.online_commit_events.append({
                    "step": self.step_counter,
                    "event": "provisional_commit_promoted",
                    "recipe_id": rid,
                    "variant_hash": variant.variant_hash,
                    "promotion_reasons": promotion_reasons,
                    "commit_confidence": confidence,
                    "commit_confidence_parts": dict(confidence_parts),
                })

        if apply_decay:
            self.decay.step(self.session_counter, self.retrain_cycle)
        self._retrain()
        self.classification_events.append((self.step_counter, cls))
        clear_session()
        return cls

    def _late_window_prediction_agreement(self, prefix: Sequence[str]) -> float:
        n = len(prefix)
        if n <= 0 or not self.step_log: return 0.0
        session_rows = self.step_log[-n:]
        if not session_rows:            return 0.0
        width = max(1, int(math.ceil(len(session_rows) * 0.40)))
        late_rows = session_rows[-width:]
        usable = [row for row in late_rows if row.predicted is not None]
        if not usable:                  return 0.0
        return sum(1 for row in usable if row.predicted == row.actual) / max(1, len(usable))

    def _provisional_promotion_reasons(
        self,
        rid: str,
        variant_hash_: str,
        confidence: float,
        confidence_parts: Dict[str, float],
    ) -> List[str]:
        provisional = self.provisional_commits.get((rid, variant_hash_))
        if provisional is None:
            return []
        window = int(getattr(self.cfg, "provisional_confirm_window", 5))
        if self.session_counter - int(provisional.get("first_session", self.session_counter)) > window:
            self.provisional_commits.pop((rid, variant_hash_), None)
            return []
        reasons = ["same_variant_hash_recurred"]
        if confidence >= float(getattr(self.cfg, "online_commit_full_threshold", 0.75)):
            reasons.append("commit_confidence_full_threshold")
        if (
            float(confidence_parts.get("late_window_prediction_agreement", 0.0))
            >= float(getattr(self.cfg, "provisional_confirm_top1", 0.60))
            and float(confidence_parts.get("recipe_jaccard", 0.0))
            >= float(getattr(self.cfg, "provisional_min_recipe_jaccard", 0.95))
        ):
            reasons.append("late_window_prediction_alignment")
        provisional.update(
            last_session=self.session_counter,
            last_confidence=confidence,
            last_parts=dict(confidence_parts),
        )
        return reasons

    def _online_commit_confidence(
        self,
        rid: str,
        prefix: Sequence[str],
        commit_cls: Classification,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a completed online sequence with deployable evidence only."""
        active_lib = self.memory.library(allowed_keys=self._active_keys())
        full_lib = self.memory.library()
        identity_prefix = (
            list(self.current_identity_prefix)
            if tuple(prefix) == tuple(self.current_prefix)
            and len(self.current_identity_prefix) == len(prefix)
            else list(prefix)
        )
        scored = [
            (
                variant.recipe_id,
                self.disambig.classify(prefix, [variant], identity_sequence=identity_prefix).jaccard,
            )
            for variant in full_lib
        ]
        best_for_rid = max(
            (score for recipe_id, score in scored if recipe_id == rid),
            default=float(commit_cls.jaccard or 0.0),
        )
        second_jaccard = max(
            (score for recipe_id, score in scored if recipe_id != rid),
            default=0.0,
        )
        jaccard_margin = max(0.0, best_for_rid - second_jaccard)
        history = list(self._online_policy_history)
        policy_confidence = (
            sum(float(row.get("final_action_confidence") or row.get("action_confidence") or 0.0) for row in history)
            / len(history)
            if history else 0.0
        )
        late_agreement = self._late_window_prediction_agreement(prefix)
        margin_scale = max(float(getattr(self.cfg, "online_commit_identity_margin_scale", 0.25)), 1e-9)
        score = (
            float(getattr(self.cfg, "online_commit_identity_weight", 0.46)) * min(1.0, best_for_rid)
            + float(getattr(self.cfg, "online_commit_identity_margin_weight", 0.18))
            * min(1.0, jaccard_margin / margin_scale)
            + float(getattr(self.cfg, "online_commit_policy_confidence_weight", 0.04))
            * policy_confidence
        )
        score = min(
            1.0,
            score + float(getattr(self.cfg, "online_commit_late_agreement_bonus", 0.06)) * late_agreement,
        )
        if (
            commit_cls.kind == "known"
            and best_for_rid >= float(getattr(self.cfg, "online_commit_known_identity_threshold", 0.99))
        ):
            score = max(score, float(getattr(self.cfg, "online_commit_known_score_floor", 0.90)))
        # An archived-only fuzzy match has no active evidence and should not
        # self-commit merely because a one-item full library yields a large
        # apparent margin. An exact known re-entry is independently decisive.
        if (
            full_lib
            and not active_lib
            and not (
                commit_cls.kind == "known"
                and best_for_rid >= float(getattr(self.cfg, "online_commit_known_identity_threshold", 0.99))
            )
        ):
            score = min(score, float(getattr(self.cfg, "online_commit_empty_library_score_ceiling", 0.40)))
        parts = {
            "recipe_jaccard": best_for_rid,
            "recipe_jaccard_margin": jaccard_margin,
            "recipe_identity_score": best_for_rid,
            "recipe_identity_margin": jaccard_margin,
            "online_policy_confidence": policy_confidence,
            "late_window_prediction_agreement": late_agreement,
        }
        return max(0.0, min(1.0, float(score))), parts
    def _active_keys(self) -> Set[VariantKey]:
        """Current trainable memory keys. Pruned variants are excluded from online prediction."""
        return {e.key for e in self.decay.active_entries()}

    def _token_for_vector(self, vector: ActionVector) -> str:
        token = self.action_vector_to_token.get(vector)
        if token is not None: return token
        if self._frozen:
            # Frozen prediction must not mutate the codebook. Unseen vectors map to a sentinel token that downstream predictors treat as OOV.
            return "act_unseen"
        token = f"act_{len(self.action_vector_to_token) + 1:04d}"
        self.action_vector_to_token[vector] = token
        self.token_to_action_vector[token] = vector
        return token

    def _vector_for_token(self, token: str) -> Optional[ActionVector]:
        return self.token_to_action_vector.get(token)

    def _display_token(self, token: Optional[str]) -> Optional[str]:
        # No more registry indirection: tokens are stable internal ids and the ground-truth string travels with the StepResult separately.
        return token


    def _decode_distribution(self, dist: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for token, prob in dist.items():
            label = self._display_token(token) or token
            out[label] = out.get(label, 0.0) + float(prob)
        total = sum(out.values())
        return {a: p / total for a, p in out.items()} if total > 0 else out

    def _tokens_from_action_labels(self, actions: Sequence[str]) -> List[str]:
        tokens: List[str] = []
        for obs in observations_from_actions(actions): tokens.append(self.action_vector_to_token.get(obs.action_vector, "act_unseen"))
        return tokens

    def _identity_tokens_from_action_labels(self, actions: Sequence[str]) -> List[str]:
        """Evaluator-only convenience mirror of the observed identity stream.

        Production identity tokens are appended inside ``observe_observation``.
        This helper lets offline audits reproduce that same representation from
        their simulated observations without exposing action strings to the
        classifier itself.
        """
        return [identity_token_from_observation(obs) for obs in observations_from_actions(actions)]

    def _coerce_prefix_tokens(self, prefix: Sequence[str]) -> List[str]:
        if not prefix: return []
        if all(p in self.token_to_action_vector for p in prefix): return list(prefix)
        if all("(" not in str(p) for p in prefix): return [str(p) for p in prefix]
        return self._tokens_from_action_labels(prefix)

    def action_policy_stats(self) -> Dict[str, Any]:
        return dict(self._last_action_policy)

    def _set_action_policy_stats(
        self,
        confidence: Optional[float],
        support_normalized_entropy: Optional[float],
        reason: str,
        *,
        margin: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        if self._frozen:
            return
        self._last_action_policy = {
            "robot_action_mandatory": True,
            "action_confidence": confidence,
            "raw_action_confidence": confidence,
            "action_margin": margin,
            "action_support_normalized_entropy": support_normalized_entropy,
            "action_entropy_normalization": "prediction_support",
            "policy_source": source or "irl_markov_ensemble",
            "final_action_confidence": confidence,
            "final_action_margin": margin,
            "final_action_support_normalized_entropy": support_normalized_entropy,
            "reason": reason,
        }

    def _distribution_stats(self, dist: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not dist: return None, None, None
        vals = sorted((float(v) for v in dist.values()), reverse=True)
        confidence = vals[0] if vals else None
        margin = vals[0] - vals[1] if len(vals) >= 2 else (vals[0] if vals else None)
        support_normalized_entropy = 0.0
        for p in vals: 
            if p > 0: support_normalized_entropy -= p * math.log(p)
        if len(vals) > 1: support_normalized_entropy = support_normalized_entropy / math.log(len(vals))
        return confidence, support_normalized_entropy, margin



    def _state_from_prefix(self, prefix: Sequence[str]) -> Tuple[int, ...]:
        with self._profile("state_from_prefix"):
            tracker = StateTracker()
            state = tuple(tracker.get_state_vector().astype(int).tolist())
            for token in self._coerce_prefix_tokens(prefix):
                vector = self._vector_for_token(token)
                if vector is None: continue
                state = apply_transition_vector(state, vector)
            return state

    def predict_next_tokens(self, prefix: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Return the deployable IRL and state-aware n-gram ensemble policy."""
        with self._profile("predict_next_tokens"):
            prefix_tokens = self._coerce_prefix_tokens(prefix) if prefix is not None else list(self.current_prefix)
            state = self._state_from_prefix(prefix_tokens)
            p_irl = self.irl.predict(state)
            p_ngram = self.markov.predict(state, prefix_tokens)
            distribution = ensemble_predict(p_irl, p_ngram, cfg=self.cfg)
            confidence, support_normalized_entropy, margin = self._distribution_stats(distribution)
            self._set_action_policy_stats(
                confidence,
                support_normalized_entropy,
                "irl_markov_ensemble",
                margin=margin,
                source="irl_markov_ensemble",
            )
            return distribution


    def predict_next(self, prefix: Optional[Sequence[str]] = None) -> Dict[str, float]:
        return self._decode_distribution(self.predict_next_tokens(prefix))

    def observe_observation(
        self,
        observation: ActionObservation,
        ground_truth_recipe: Optional[str] = None,
        *,
        precomputed_distribution: Optional[Mapping[str, float]] = None,
    ) -> StepResult:
        """Consume one observed human transition using anonymous action tokens."""
        if self._frozen:
            action = self.action_vector_to_token.get(observation.action_vector, "act_unseen")
            distribution = (
                dict(precomputed_distribution)
                if precomputed_distribution is not None
                else self.predict_next_tokens(self.current_prefix)
            )
            predicted = top_k(distribution, k=1)[0] if distribution else None
            return StepResult(
                step=self.step_counter,
                predicted=self._display_token(predicted),
                actual=action,
                correct=predicted == action,
                mode=self.mode,
                event="frozen_prediction",
            )

        self.step_counter += 1
        if self.mode == MODE_OBSERVE:
            action = self._token_for_vector(observation.action_vector)
            self.pending_demo.append(action)
            self.pending_demo_transitions.append(self._transition_from_observation(action, observation))
            self.pending_demo_identity.append(identity_token_from_observation(observation))
            result = StepResult(
                step=self.step_counter,
                predicted=None,
                actual=self._display_token(action) or action,
                correct=False,
                mode=self.mode,
                event="observing",
            )
            self.step_log.append(result)
            return result

        first_online_action = not self.current_prefix and precomputed_distribution is None
        if first_online_action:
            distribution: Dict[str, float] = {}
        else:
            distribution = (
                dict(precomputed_distribution)
                if precomputed_distribution is not None
                else self.predict_next_tokens(self.current_prefix)
            )
            self._online_policy_history.append(dict(self._last_action_policy))
        predicted = top_k(distribution, k=1)[0] if distribution else None
        action = self.action_vector_to_token.get(observation.action_vector)
        if action is None:
            action = self._token_for_vector(observation.action_vector)
        correct = predicted == action
        self.current_prefix.append(action)
        self.current_transition_trace.append(self._transition_from_observation(action, observation))
        self.current_identity_prefix.append(identity_token_from_observation(observation))
        result = StepResult(
            step=self.step_counter,
            predicted=self._display_token(predicted),
            actual=self._display_token(action) or action,
            correct=correct,
            mode=self.mode,
        )
        self.step_log.append(result)
        self.accuracy_events.append((self.step_counter, ground_truth_recipe or "?", correct))
        return result

    def evaluate_autonomous_tokens(
        self,
        ordering: Sequence[str],
        topn: int = 3,
    ) -> Dict[str, float]:
        if not self._frozen:
            with self.frozen():
                return self.evaluate_autonomous_tokens(ordering, topn=topn)
        tokens = self._coerce_prefix_tokens(ordering)
        total = len(tokens)
        topn = max(1, int(topn))
        if total == 0:
            result = {
                "top1": 0.0,
                "topk": 0.0,
                "prediction_available_rate": 0.0,
                "empty_prediction_rate": 0.0,
                "cross_entropy": 0.0,
            }
            result[f"top{topn}"] = 0.0
            return result

        prefix: List[str] = []
        top1_hits = topk_hits = available = empty_predictions = 0
        nll = 0.0
        floor = max(float(self.cfg.prob_floor), 1e-12)
        for actual in tokens:
            distribution = self.predict_next_tokens(prefix)
            ranked = top_k(distribution, k=topn) if distribution else []
            if distribution:
                available += 1
                top1_hits += int(ranked[0] == actual)
                topk_hits += int(actual in ranked)
                nll -= math.log(max(float(distribution.get(actual, floor)), floor))
            else:
                empty_predictions += 1
                nll -= math.log(floor)
            prefix.append(actual)

        result = {
            "top1": top1_hits / total,
            "topk": topk_hits / total,
            "prediction_available_rate": available / total,
            "empty_prediction_rate": empty_predictions / total,
            "cross_entropy": nll / total,
        }
        result[f"top{topn}"] = topk_hits / total
        return result

    def _build_trajectories(self, demos):
        """Convert each demo into a trajectory in O(L) per demo.

        Production demos carry observed transition traces:
        ``(state_before, anonymous_token, state_after)``. These traces are the
        learner's legitimate sensor stream and avoid replaying hidden symbolic
        simulator action strings. Anonymous synthetic tests or old records that
        lack traces fall back to transition-vector replay.
        """
        trajs = []
        dropped_total = 0
        for demo in demos:
            if isinstance(demo, Entry):
                raw_tokens = list(demo.ordering)
            elif isinstance(demo, Mapping):
                raw_tokens = list(demo.get("ordering", ()))
            else:
                raw_tokens = list(demo)
            tokens = self._coerce_prefix_tokens(raw_tokens)
            traj = []
            trace = self._trace_for_demo(demo, tokens)
            if trace:
                state = trace[0][0]
                for before, token, after in trace:
                    if after == before:
                        dropped_total += 1
                        state = after
                        continue
                    traj.append((before, token))
                    state = after
            else:
                # Fallback: no stored transition trace (anonymous test-fixture
                # tokens).  Re-apply pre-computed vectors incrementally.
                # Mirror the trace path exactly: skip (state, token) pairs where
                # the vector produces no state change so the IRL never trains on
                # self-loop transitions.  Unknown or mock length-mismatched
                # vectors are kept because they represent anonymous test
                # actions whose full environment effects are not catalogued.
                tracker = StateTracker()
                state = tuple(tracker.get_state_vector().astype(int).tolist())
                for token in tokens:
                    vector = self._vector_for_token(token)
                    if vector is None or len(vector) != 2 * len(state):
                        traj.append((state, token))
                        continue
                    new_state = apply_transition_vector(state, vector)
                    if new_state == state:
                        dropped_total += 1
                        state = new_state
                        continue
                    traj.append((state, token))
                    state = new_state
            traj.append((state, "stop"))
            trajs.append(traj)
        return trajs, dropped_total

    def _record_retrain_event(self, dropped_actions: int, active_demos: int, *, total_wall_s: float = 0.0, build_wall_s: float = 0.0, fit_wall_s: float = 0.0, flop_estimate: float = 0.0, skipped: bool = False) -> None:
        """Append a structured retrain event. All agent variants funnel through this."""
        fit_stats = self._latest_fit_stats() if not skipped else {}
        if not skipped and float(flop_estimate) <= 0.0:
            estimate = fit_stats.get("estimated_flops") if isinstance(fit_stats, dict) else None
            if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)): flop_estimate = float(estimate)
        event = {"step": self.step_counter, "cycle": int(self.retrain_cycle), "dropped_actions": int(dropped_actions), "active_demos": int(active_demos), "total_wall_s": float(total_wall_s), "build_wall_s": float(build_wall_s),
            "fit_wall_s": float(fit_wall_s), "flop_estimate": float(flop_estimate), "skipped": bool(skipped)}
        if fit_stats: event["fit_stats"] = fit_stats
        self.retrain_events.append(event)
        if not skipped:
            self.retrain_total_wall_times.append(float(total_wall_s))
            self.retrain_build_wall_times.append(float(build_wall_s))
            self.retrain_fit_wall_times.append(float(fit_wall_s))
            self.retrain_flop_estimates.append(float(flop_estimate))

    def _latest_fit_stats(self) -> Dict[str, Any]:
        """Return structured accounting from the model head fit that just ran."""
        stats: Dict[str, Any] = {}
        irl_stats = getattr(getattr(self, "irl", None), "last_fit_stats", None)
        if isinstance(irl_stats, dict) and irl_stats:       stats.update(irl_stats)
        bc_stats = getattr(getattr(self, "bc", None), "last_fit_stats", None)
        if isinstance(bc_stats, dict) and bc_stats:         stats.update(bc_stats)
        custom_stats = getattr(self, "_custom_fit_stats", None)
        if isinstance(custom_stats, dict) and custom_stats: stats.update(custom_stats)
        return stats

    def _estimate_retrain_flops(self, trajectories: Sequence[List[Tuple[Tuple[int, ...], str]]]) -> float:
        """Estimate fit FLOPs from the fitted head's actual accounting."""
        stats = self._latest_fit_stats()
        estimate = stats.get("estimated_flops") if isinstance(stats, dict) else None
        if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)):
            if stats.get("model_family") != "maxent_irl": return float(estimate)
            n_transitions = sum(max(0, len(traj) - 1) for traj in trajectories)
            count_head_flops = float(n_transitions * max(1, int(getattr(self.cfg, "markov_order", 1)) + 2))
            return float(estimate) + count_head_flops
        # Custom heads should expose last_fit_stats; this estimate is retained
        # only for externally supplied heads that do not.
        n_transitions = sum(max(0, len(traj) - 1) for traj in trajectories)
        n_actions = max(1, len({action for traj in trajectories for _state, action in traj}))
        feature_dim = len(trajectories[0][0][0]) if trajectories and trajectories[0] else 1
        iters = int(getattr(self.cfg, "maxent_iters_warm", 1) if self.retrain_cycle > 1 else getattr(self.cfg, "maxent_iters_cold", 1))
        return float(n_transitions * n_actions * feature_dim * max(1, iters))

    # offline retrain on weighted active set
    def _fit_heads(self, trajectories: Sequence[List[Tuple[Tuple[int, ...], str]]], weights: Sequence[float]) -> None:
        self.irl.fit(trajectories, weights)
        self.markov.fit(trajectories, weights, state_to_idx=self.irl.state_to_idx, idx_to_state=self.irl.idx_to_state, feature_matrix=self.irl.feature_matrix, col_min=self.irl.col_min, col_max=self.irl.col_max, normalizer=self.irl.normalizer)

    def _reset_heads(self) -> None:
        self.irl = MaxEntIRL2(cfg=self.cfg)
        self.markov = NGramMarkov(
            order=self.cfg.markov_order,
            prob_floor=self.cfg.prob_floor,
            state_log_weight=self.cfg.ngram_state_log_weight,
            prefix_log_weight=self.cfg.ngram_prefix_log_weight,
        )

    def _prepare_retrain_fit(self) -> None:
        """Drop fitted predictor state before fitting the current active set."""
        self._reset_heads()

    def _fit_fingerprint(self) -> Optional[frozenset]:
        """Identity of active replay membership.

        None means the active set is empty. The fingerprint deliberately ignores
        weights, timestamps, and transition traces so short decay ramps do not
        trigger one fit per decay tick; only additions/removals do.
        """
        if not self.decay.active: return None
        return frozenset(self.decay.active.keys())

    def _demo_weighting_length(self, demo: Any) -> int:
        if isinstance(demo, Entry):
            return max(1, len(demo.ordering))
        if isinstance(demo, Mapping):
            ordering = demo.get("ordering", ())
            return max(1, len(ordering))
        if isinstance(demo, (list, tuple)):
            return max(1, len(demo))
        return 1

    def _length_normalized_demo_weights(self, demos: Sequence[Any], base_weights: Sequence[float]) -> List[float]:
        """Normalize per-demo training influence so longer recipes do not dominate.

        The learner consumes per-demo weights but expands each demo into a
        variable number of transitions. Dividing by demo length makes each
        recipe/preference episode contribute comparable total mass, then
        rescaling preserves the average decay/replay weight used by the
        baseline.
        """
        bases = [float(w) for w in base_weights]
        if not demos or not bases:
            return bases
        n = min(len(demos), len(bases))
        demos = list(demos)[:n]
        bases = bases[:n]
        raw = [
            float(weight) / float(self._demo_weighting_length(demo))
            for demo, weight in zip(demos, bases)
        ]
        raw_mean = sum(raw) / max(1, len(raw))
        base_mean = sum(bases) / max(1, len(bases))
        if raw_mean <= 0.0 or not math.isfinite(raw_mean):
            return raw
        scale = base_mean / raw_mean
        return [float(w * scale) for w in raw]

    def _retrain(self) -> None:
        if self._frozen:
            return
        retrain_t0 = time.perf_counter()
        fp = self._fit_fingerprint()
        if fp is not None and fp == self._last_fit_fingerprint:
            # Active membership is identical to the last successful fit. Weight-only decay changes are deliberately absorbed without refitting; additions/removals still invalidate this gate.
            with self._profile("retrain_skipped"):
                self.retrain_cycle += 1
                self.retrain_skipped_count += 1
                self._record_retrain_event(dropped_actions=0, active_demos=len(self.decay.active), total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
            return
        with self._profile("retrain_total"):
            self.retrain_cycle += 1
            entries = self.decay.active_entries()
            if not entries:
                self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
                self._reset_heads()
                self._last_fit_fingerprint = None
                if self.cfg.verbose: self.narrate(f"[step {self.step_counter}] cleared predictors because active memory is empty (cycle {self.retrain_cycle})")
                return
            build_t0 = time.perf_counter()
            with self._profile("retrain_build_trajectories"):   trajectories, dropped_total = self._build_trajectories(entries)
            build_wall_s = time.perf_counter() - build_t0
            weights = self._length_normalized_demo_weights(entries, [e.weight for e in entries])
            self._prepare_retrain_fit()
            fit_t0 = time.perf_counter()
            with self._profile("retrain_fit_heads"):            self._fit_heads(trajectories, weights)
            fit_wall_s = time.perf_counter() - fit_t0
            self._record_retrain_event(dropped_actions=dropped_total, active_demos=len(entries), total_wall_s=time.perf_counter() - retrain_t0, build_wall_s=build_wall_s, fit_wall_s=fit_wall_s, flop_estimate=self._estimate_retrain_flops(trajectories))
            self._last_fit_fingerprint = fp
            if self.cfg.verbose:                                self.narrate(f"[step {self.step_counter}] retrained on {len(trajectories)} weighted demos (cycle {self.retrain_cycle}, post_grace_decay_rate={self.decay.post_grace_decay_rate:.6f}, dropped_actions={dropped_total})")

    def refresh_model_from_memory(self) -> None:
        """Re-fit predictors against the current active memory without adding a demo."""
        self._retrain()

    def pruned_influence_audit(self, max_prefixes: int = 24, tolerance: float = 5e-2) -> Dict[str, Any]:
        """Verify that the fitted deployable heads use active replay only."""
        entries = self.decay.active_entries()
        if not entries:
            return {
                "max_l1": 0.0,
                "mean_l1": 0.0,
                "n_prefixes": 0,
                "passed": True,
                "tolerance": float(tolerance),
                "audit_reference_seed": int(self.cfg.seed) + 1_000_003,
            }

        trajectories, _ = self._build_trajectories(entries)
        weights = self._length_normalized_demo_weights(entries, [float(entry.weight) for entry in entries])
        audit_seed = int(self.cfg.seed) + 1_000_003
        audit_cfg = replace(self.cfg, seed=audit_seed)
        fresh_irl = MaxEntIRL2(cfg=audit_cfg)
        fresh_markov = NGramMarkov(
            order=self.cfg.markov_order,
            prob_floor=self.cfg.prob_floor,
            state_log_weight=self.cfg.ngram_state_log_weight,
            prefix_log_weight=self.cfg.ngram_prefix_log_weight,
        )
        fresh_irl.fit(trajectories, weights)
        fresh_markov.fit(
            trajectories,
            weights,
            state_to_idx=fresh_irl.state_to_idx,
            idx_to_state=fresh_irl.idx_to_state,
            feature_matrix=fresh_irl.feature_matrix,
            col_min=fresh_irl.col_min,
            col_max=fresh_irl.col_max,
            normalizer=fresh_irl.normalizer,
        )

        prefixes: List[Tuple[str, ...]] = []
        seen: Set[Tuple[str, ...]] = set()
        for entry in entries:
            sequence = tuple(entry.ordering)
            for k in (0, min(len(sequence), 1), len(sequence) // 2, max(0, len(sequence) - 1)):
                prefix = sequence[:k]
                if prefix not in seen:
                    prefixes.append(prefix)
                    seen.add(prefix)
                if len(prefixes) >= max(1, int(max_prefixes)):
                    break
            if len(prefixes) >= max(1, int(max_prefixes)):
                break

        differences: List[float] = []
        for prefix in prefixes:
            state = self._state_from_prefix(prefix)
            current_ngram = self.markov.predict(state, prefix)
            reference_ngram = fresh_markov.predict(state, prefix)
            current = ensemble_predict(self.irl.predict(state), current_ngram, cfg=self.cfg)
            reference = ensemble_predict(fresh_irl.predict(state), reference_ngram, cfg=self.cfg)
            tokens = set(current) | set(reference)
            differences.append(sum(abs(float(current.get(token, 0.0)) - float(reference.get(token, 0.0))) for token in tokens))

        max_l1 = max(differences, default=0.0)
        mean_l1 = sum(differences) / max(1, len(differences))
        return {
            "max_l1": float(max_l1),
            "mean_l1": float(mean_l1),
            "n_prefixes": len(prefixes),
            "passed": bool(max_l1 <= tolerance),
            "tolerance": float(tolerance),
            "audit_reference_seed": audit_seed,
        }
    def evaluate_sequence(self, ordering: Sequence[str]) -> float:
        """Prefix-conditioned accuracy of the current ensemble on a reference ordering. Used for the held-out forgetting metric."""
        if not ordering: return 0.0
        return float(self.evaluate_autonomous_tokens(ordering, topn=1)["top1"])
