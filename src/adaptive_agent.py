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
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import StateTracker
from .memory import (Classification,    DecayManager,       DemoTransition, Disambiguator,  Entry,          Variant,        VariantKey,                 VariantMemory,      _aligned_next_index, kendall_tau_distance, variant_hash)
from .models import (Config,            DEFAULT_CONFIG,     MaxEntIRL2,     NGramMarkov,       ensemble_predict,   top_k)
from .posterior import (MemoryPrior, OnlinePreferencePosterior,      PosteriorWeights,   PreferencePrototypeLearner, RecipePrototypeLearner)
from .representations import (ActionObservation,    ActionVector,   PreferenceSignature,    ROLE_UNKNOWN_OR_NOOP,   TaskSignature,  apply_transition_vector,    identity_token_from_observation, observations_from_actions,  preference_signature_from_roles,    role_from_transition,   task_signature_from_tokens)

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
    inferred_recipe: Optional[str]
    inferred_variant_hash: Optional[str]
    inferred_latent_pref_id: Optional[str] = None
    event: str = ""   # free-form tag for the narrator


class AdaptiveHRCAgent:
    """End-to-end continual-learning HRC agent."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, narrate: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.decay = DecayManager(cfg)
        self.memory = VariantMemory(cfg)
        self.disambig = Disambiguator(cfg)
        self.recipe_prototypes = RecipePrototypeLearner(cfg)
        self.preference_prototypes = PreferencePrototypeLearner(cfg=cfg)
        self.task_signatures: Dict[str, TaskSignature] = {}
        self.preference_signatures: Dict[str, PreferenceSignature] = {}
        # Online posterior over (recipe_id, pref_id). Owned by the agent so the freeze snapshot can include it.
        posterior_weights = PosteriorWeights(
            alpha_recipe=getattr(  cfg,    "posterior_alpha_recipe",  1.0),
            alpha_pref=getattr(    cfg,    "posterior_alpha_pref",    1.5),
            alpha_memory=getattr(  cfg,    "posterior_alpha_memory",  0.5),
            alpha_compat=getattr(  cfg,    "posterior_alpha_compat",  0.5),
            temperature_recipe=getattr(     cfg,    "posterior_temperature_recipe",     1.0),
            temperature_pref=getattr(       cfg,    "posterior_temperature_pref",       1.0),
            temperature_memory=getattr(     cfg,    "posterior_temperature_memory",     1.0),
            temperature_compat=getattr(     cfg,    "posterior_temperature_compat",     1.0),
            global_temperature=getattr(     cfg,    "posterior_global_temperature",     1.0),
            memory_prior_floor=getattr(     cfg,    "memory_prior_floor",     1e-6),
            active_prior_floor=getattr(     cfg,    "active_prior_floor",     0.05),
            absent_prior=getattr(           cfg,    "absent_prior",           0.10))
        self.posterior = OnlinePreferencePosterior(weights=posterior_weights)
        # token -> abstract role tag, populated as observations stream through observe_observation. This lets the agent build role sequences for preference prototypes without storing raw action labels there.
        self.token_to_role: Dict[str, str] = {}
        # The latest pref_id assignment from the most recent end_demo update. Used as a default conditioning when the posterior is not yet active.
        self.last_pref_id: Optional[str] = None
        # Active variant -> latent preference prototype assigned during the last active-only rebuild. This is the bridge between concrete preference variants and the posterior's latent pref_ids.
        self.variant_pref_ids: Dict[VariantKey, str] = {}
        self.locked_variant_key: Optional[VariantKey] = None
        self.locked_pref_id: Optional[str] = None
        self.preference_id_registry: Dict[str, str] = {}
        self.prototype_events: List[Dict[str, Any]] = []
        self.online_commit_events: List[Dict[str, Any]] = []
        self._active_pref_clusters: Dict[str, Set[VariantKey]] = {}
        self._active_pref_cluster_roles: Dict[str, Dict[VariantKey, Tuple[str, ...]]] = {}

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
        self.inferred_recipe: Optional[str] = None
        self.inferred_variant_hash: Optional[str] = None
        self.inferred_latent_pref_id: Optional[str] = None
        self._needs_observation: bool = False
        self._online_unknown_streak: int = 0
        self._online_step_status: str = "known_confident"
        self._online_policy_history: List[Dict[str, Any]] = []
        # Ephemeral evidence extracted only from human corrections during the
        # current assisted session.  It is deliberately separate from the
        # active-memory prototypes and is cleared at every session boundary.
        self._session_precedence_evidence: Dict[Tuple[str, str], float] = {}
        self._session_correction_count: int = 0
        self.provisional_commits: Dict[VariantKey, Dict[str, Any]] = {}
        self._last_action_policy: Dict[str, Any] = {
            "robot_action_mandatory": True,
            "action_confidence": None,
            "raw_action_confidence": None,
            "action_margin": None,
            "action_entropy": None,
            "policy_source": None,
            "final_action_confidence": None,
            "final_action_margin": None,
            "final_action_entropy": None,
            "conditioned_action_confidence": None,
            "ensemble_action_confidence": None,
            "conditioned_action_margin": None,
            "conditioned_action_entropy": None,
            "ensemble_action_margin": None,
            "ensemble_action_entropy": None,
            "conditioned_top_token": None,
            "ensemble_top_token": None,
            "posterior_confidence": None,
            "posterior_margin": None,
            "session_correction_count": 0,
            "session_precedence_edge_count": 0,
            "policy_agreement": None,
            "blend_strength": None,
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
        # Hysteresis state for online identity switching: the next argmax has to win the log-ratio margin AND repeat for K consecutive steps before `inferred_recipe` flips. Reset on every confirmed switch.
        self._pending_argmax: Optional[str] = None
        self._pending_count: int = 0

        # Metrics for the paper
        self.step_log: List[StepResult] = []
        self.accuracy_events: List[Tuple[int, str, bool]] = []  # (step,recipe,correct)
        self.posterior_switch_events: List[Dict[str, Any]] = []
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
            "prototype_events_len": len(self.prototype_events),
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
        self.inferred_recipe = None
        self.inferred_variant_hash = None
        self.inferred_latent_pref_id = None
        self._needs_observation = False
        self._online_unknown_streak = 0
        self._online_step_status = "known_confident"
        self._online_policy_history = []
        self._reset_session_adaptation()
        self._clear_preference_lock()
        self.observation_mode_entries += 1
        self.narrate(f"[step {self.step_counter}] MODE -> OBSERVE (new-demo gate ON)")

    def end_demo(self) -> Classification:
        """Human signals the end of a demonstration or collaboration session. In observe mode, the buffered sequence is classified against the existing library. In online mode, a locked preference shift is committed by promoting the inferred recipe/variant."""
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
            self.inferred_recipe = None
            self.inferred_variant_hash = None
            self.inferred_latent_pref_id = None
            self._needs_observation = False
            self._online_unknown_streak = 0
            self._online_step_status = "known_confident"
            self._online_policy_history = []
            self._reset_session_adaptation()
            self._clear_preference_lock()
            return Classification("known", None, None, 0.0, 0.0)
        if self.mode == MODE_OBSERVE:
            self.session_counter += 1
            return self._end_observe_demo(apply_decay=True)

        commit_cls, reentry_from_pruned = self._classify_online_prefix(self.current_prefix)
        # Diagnostic: check if posterior's live inference agrees with commit-time Jaccard. Log the mismatch rate so it can be tracked in evaluation.
        if (self.inferred_recipe is not None and commit_cls.recipe_id is not None and self.inferred_recipe != commit_cls.recipe_id):
            self.narrate(f"[step {self.step_counter}] COMMIT MISMATCH: posterior tracked '{self.inferred_recipe}' but commit resolved to '{commit_cls.recipe_id}'")
            # Track as a metric for the paper.
            if not hasattr(self, "_commit_mismatches"): self._commit_mismatches: List[Tuple[int, str, str]] = []
            self._commit_mismatches.append((self.step_counter, self.inferred_recipe, commit_cls.recipe_id))
        if commit_cls.kind == "new_recipe":
            cls = Classification("needs_observation", None, None, commit_cls.jaccard, commit_cls.order_distance)
            self.classification_events.append((self.step_counter, cls))
            self.current_prefix = []
            self.current_identity_prefix = []
            self.inferred_recipe = None
            self.inferred_variant_hash = None
            self.inferred_latent_pref_id = None
            self._needs_observation = True
            self._online_unknown_streak = int(getattr(self.cfg, "online_unknown_confirm_streak", 3))
            self._online_step_status = "unknown_confirmed"
            self._online_policy_history = []
            self._reset_session_adaptation()
            self._clear_preference_lock()
            self.narrate(f"[step {self.step_counter}] online sequence did not match known recipes; observation mode required")
            return cls

        # Only known recipe/preference sessions advance decay and mutate memory. The currently-demonstrated variant is protected by the latest-pin in `decay.latest_keys` (set by `mark_latest`), so no extra protected_keys plumbing is needed here.
        self.session_counter += 1
        return self._end_online_session(commit_cls, reentry_from_pruned, apply_decay=True)

    def _cache_token_role(self, token: str, observation: ActionObservation) -> None:
        """Cache the abstract role for a token on first sight.

        Roles are decoded only from observed state deltas (no preference labels
        or simulator action strings). The cache survives across sessions because
        the action vector -> role mapping is deterministic for this domain.
        """
        if token in self.token_to_role:
            return
        try:                                role = role_from_transition(observation.state, observation.action_vector, observation.next_state)
        except Exception:                   role = ROLE_UNKNOWN_OR_NOOP
        self.token_to_role[token] = role

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

    def _roles_for_tokens(self, tokens: Sequence[str]) -> List[str]:
        return [self.token_to_role.get(t, ROLE_UNKNOWN_OR_NOOP) for t in tokens]

    def _clear_preference_lock(self) -> None:
        if self._frozen: return
        locked_key = self.locked_variant_key
        self.locked_variant_key = None
        self.locked_pref_id = None
        if locked_key is not None and self.inferred_variant_hash == locked_key[1]: self.inferred_variant_hash = None

    def _reset_session_adaptation(self) -> None:
        """Discard nonpersistent evidence accumulated from this session's corrections."""
        if self._frozen:
            return
        self._session_precedence_evidence = {}
        self._session_correction_count = 0

    def record_robot_feedback(
        self,
        *,
        prefix: Sequence[str],
        predicted: Optional[str],
        actual: str,
        correct_top1: bool,
    ) -> None:
        """Use a human correction as temporary ordering evidence.

        The callback receives opaque action tokens already used by the policy;
        it never receives recipe or simulator preference labels.  A correction
        that selects role ``a`` while the robot selected role ``b`` indicates a
        local preference for ``a`` before ``b`` whenever those roles differ.
        This evidence affects only later actions in this session.
        """
        if self._frozen or not bool(getattr(self.cfg, "session_correction_adaptation", True)):
            return
        if correct_top1 or predicted is None:
            return
        predicted_role = self.token_to_role.get(predicted, ROLE_UNKNOWN_OR_NOOP)
        actual_role = self.token_to_role.get(actual, ROLE_UNKNOWN_OR_NOOP)
        if (
            predicted_role == actual_role
            or predicted_role == ROLE_UNKNOWN_OR_NOOP
            or actual_role == ROLE_UNKNOWN_OR_NOOP
        ):
            return
        key = (actual_role, predicted_role)
        self._session_precedence_evidence[key] = (
            float(self._session_precedence_evidence.get(key, 0.0))
            + max(0.0, float(getattr(self.cfg, "session_correction_precedence_weight", 0.75)))
        )
        self._session_correction_count += 1

    def _session_precedence_profile(self) -> Dict[Tuple[str, str], float]:
        """Normalise correction evidence into the prototype score domain [-1, 1]."""
        if not self._session_precedence_evidence:
            return {}
        # Each directed correction is already a direct ordering preference;
        # cap repeated corrections to prevent one early mistake from becoming a
        # hard constraint.
        return {
            pair: max(-1.0, min(1.0, float(weight)))
            for pair, weight in self._session_precedence_evidence.items()
        }

    def _latest_active_variant_key(self) -> Optional[VariantKey]:
        candidates: List[Tuple[int, str, str]] = []
        for rid, variant_hash_ in self.memory.latest.items():
            key = (rid, variant_hash_)
            if key not in self.variant_pref_ids:    continue
            variant = self.memory.variants.get(rid, {}).get(variant_hash_)
            if variant is None:                     continue
            candidates.append((int(variant.last_seen_step), rid, variant_hash_))
        if not candidates:      return None
        _step, rid, variant_hash_ = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        return (rid, variant_hash_)

    def _remap_preference_ids_after_rebuild(self) -> None:
        if self.locked_variant_key is not None:
            remapped = self.variant_pref_ids.get(self.locked_variant_key)
            if remapped is None:
                self.prototype_events.append({"step": self.step_counter, "event": "stale_preference_lock_cleared", "locked_variant_key": list(self.locked_variant_key)})
                self._clear_preference_lock()
            else:
                self.locked_pref_id = remapped
        latest_key = self._latest_active_variant_key()
        self.last_pref_id = self.variant_pref_ids.get(latest_key) if latest_key is not None else None

    def _register_if_live(
        self,
        rid: str,
        seq: List[str],
        step: int,
        transitions: Optional[Sequence[DemoTransition]] = None,
        identity_ordering: Optional[Sequence[str]] = None,
    ) -> Variant:
        """Register in memory + decay only when not frozen. Returns the Variant. Also folds the demo into the recipe and latent preference prototypes. Both are read-only structural summaries used by the posterior and the preference-conditioned policy; they are independent of the decay/replay path."""
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

    def _sync_preference_signature(self, pref_id: str, fallback_roles: Optional[Sequence[str]] = None) -> None:
        """Expose an aggregate PreferenceSignature for the latent prototype."""
        proto = self.preference_prototypes.prototypes.get(pref_id)
        if proto is None:
            if fallback_roles is not None:  self.preference_signatures[pref_id] = preference_signature_from_roles(fallback_roles)
            return
        scalar = tuple(sorted((k, float(v)) for k, v in proto.scalar_profile().items()))
        self.preference_signatures[pref_id] = PreferenceSignature(
            scalar_features=scalar,
            role_bigrams=tuple(sorted(proto.bigram_counts.items(), key=lambda kv: (str(kv[0]), kv[1]))),
            role_trigrams=tuple(sorted(proto.trigram_counts.items(), key=lambda kv: (str(kv[0]), kv[1]))),
            precedence_features=tuple(sorted(proto.precedence_profile().items())),
        )

    def _rebuild_active_prototypes(self, entries: Optional[Sequence[Any]] = None) -> None:
        """Rebuild all learner-facing prototypes from active weighted memory."""
        active_entries = sorted(list(self.decay.active_entries() if entries is None else entries), key=lambda e: (e.recipe_id, e.variant_hash))
        recipe_learner = RecipePrototypeLearner(self.cfg)
        pref_learner = PreferencePrototypeLearner(cfg=self.cfg)
        tokens_by_recipe: Dict[str, List[str]] = {}
        variant_pref_ids: Dict[VariantKey, str] = {}
        members_by_pid: Dict[str, List[Tuple[VariantKey, Tuple[str, ...]]]] = {}
        last_pid: Optional[str] = None
        for entry in active_entries:
            seq = list(entry.ordering)
            w = float(entry.weight)
            tokens_by_recipe.setdefault(entry.recipe_id, []).extend(seq)
            terminal_state = entry.transitions[-1][2] if getattr(entry, "transitions", ()) else (self._state_from_prefix(seq) if seq else None)
            if not getattr(self.cfg, "ablation_disable_recipe_prototype", False): recipe_learner.update_from_demo(entry.recipe_id, seq, terminal_state=terminal_state, weight=w)
            if not getattr(self.cfg, "ablation_disable_preference_head", False):
                last_pid = pref_learner.update_from_roles(self._roles_for_tokens(seq), recipe_id=entry.recipe_id, weight=w)
                variant_pref_ids[entry.key] = last_pid
                members_by_pid.setdefault(last_pid, []).append((entry.key, tuple(self._roles_for_tokens(seq))))
        if members_by_pid:
            id_map, semantic_key_by_pid = self._stable_preference_id_map(members_by_pid)
            stable_members: Dict[str, Set[VariantKey]] = {}
            stable_roles: Dict[str, Dict[VariantKey, Tuple[str, ...]]] = {}
            for old_pid, members in members_by_pid.items():
                stable_pid = id_map.get(old_pid, old_pid)
                for key, roles in members:
                    stable_members.setdefault(stable_pid, set()).add(key)
                    stable_roles.setdefault(stable_pid, {})[key] = tuple(roles)
            continuity_map = self._prototype_continuity_id_map(stable_members, stable_roles)
            if continuity_map:
                for old_pid, stable_pid in list(id_map.items()):
                    final_pid = continuity_map.get(stable_pid, stable_pid)
                    id_map[old_pid] = final_pid
                    if final_pid != stable_pid and old_pid in semantic_key_by_pid: self.preference_id_registry[semantic_key_by_pid[old_pid]] = final_pid
            remapped = PreferencePrototypeLearner(cfg=self.cfg)
            remapped.match_threshold = pref_learner.match_threshold
            remapped.novelty_max_similarity = pref_learner.novelty_max_similarity
            remapped.novelty_entropy_min = pref_learner.novelty_entropy_min
            remapped.tau_pref_cluster = pref_learner.tau_pref_cluster
            remapped.axis_split_threshold = pref_learner.axis_split_threshold
            remapped.prototypes = {}
            for old_pid, proto in pref_learner.prototypes.items():
                new_pid = id_map.get(old_pid, old_pid)
                proto.pref_id = new_pid
                remapped.prototypes[new_pid] = proto
            pref_learner = remapped
            variant_pref_ids = {key: id_map.get(pid, pid) for key, pid in variant_pref_ids.items()}
            last_pid = id_map.get(last_pid, last_pid) if last_pid is not None else None
        self.recipe_prototypes = recipe_learner
        self.preference_prototypes = pref_learner
        self.variant_pref_ids = variant_pref_ids
        old_ids = set(getattr(self, "preference_signatures", {}).keys())
        new_ids = set(self.preference_prototypes.all_pref_ids())
        new_clusters: Dict[str, Set[VariantKey]] = {}
        new_cluster_roles: Dict[str, Dict[VariantKey, Tuple[str, ...]]] = {}
        for key, pid in self.variant_pref_ids.items():
            new_clusters.setdefault(pid, set()).add(key)
        for _raw_pid, members in members_by_pid.items():
            final_pid = id_map.get(_raw_pid, _raw_pid) if members_by_pid else _raw_pid
            for key, roles in members: new_cluster_roles.setdefault(final_pid, {})[key] = tuple(roles)
        stability = self._prototype_stability_report(self._active_pref_clusters, new_clusters)
        if old_ids or new_ids: self.prototype_events.append({"step": self.step_counter, "event": "prototype_rebuild", "prototype_new": sorted(new_ids - old_ids), "prototype_retired": sorted(old_ids - new_ids), "prototype_persisted": sorted(old_ids & new_ids), **stability})
        self._active_pref_clusters = {pid: set(keys) for pid, keys in new_clusters.items()}
        self._active_pref_cluster_roles = {pid: dict(rows) for pid, rows in new_cluster_roles.items()}
        self.task_signatures = {rid: task_signature_from_tokens(tokens, self.token_to_action_vector, self.token_to_role) for rid, tokens in tokens_by_recipe.items()}
        self.preference_signatures = {}
        for pid in self.preference_prototypes.all_pref_ids(): self._sync_preference_signature(pid)
        self._remap_preference_ids_after_rebuild()
        # Hard active-only invariant: stable IDs may survive as metadata, but
        # no learner-facing prototype or variant-to-prototype assignment may
        # retain a pruned entry's statistics or membership.
        expected_keys = {entry.key for entry in active_entries}
        if not getattr(self.cfg, "ablation_disable_preference_head", False) and set(self.variant_pref_ids) != expected_keys:
            raise RuntimeError("active-only prototype rebuild produced a variant assignment outside the supplied active entries")
        expected_recipes = {entry.recipe_id for entry in active_entries}
        if not getattr(self.cfg, "ablation_disable_recipe_prototype", False) and set(self.recipe_prototypes.prototypes) != expected_recipes:
            raise RuntimeError("active-only prototype rebuild retained a recipe outside the supplied active entries")
        if not self.current_prefix: self.posterior.reset()

    def _stable_preference_id_map(self, members_by_pid: Dict[str, List[Tuple[VariantKey, Tuple[str, ...]]]]) -> Tuple[Dict[str, str], Dict[str, str]]:
        id_map: Dict[str, str] = {}
        semantic_key_by_pid: Dict[str, str] = {}
        used: Set[str] = set()
        for pid, members in sorted(members_by_pid.items(), key=lambda kv: kv[0]):
            medoid_key, medoid_roles = min(members, key=lambda item: (sum(self._role_sequence_distance(item[1], other_roles) for _other_key, other_roles in members), item[0][0], item[0][1]))
            semantic_key = json.dumps({"roles": list(medoid_roles)}, sort_keys=True)
            semantic_key_by_pid[pid] = semantic_key
            payload = semantic_key.encode("utf-8")
            base = f"P_{hashlib.sha256(payload).hexdigest()[:8]}"
            stable = self.preference_id_registry.get(semantic_key, base)
            if stable in used:
                semantic_key = json.dumps({"roles": list(medoid_roles), "medoid": list(medoid_key)}, sort_keys=True)
                semantic_key_by_pid[pid] = semantic_key
                payload = semantic_key.encode("utf-8")
                stable = self.preference_id_registry.get(semantic_key, f"P_{hashlib.sha256(payload).hexdigest()[:8]}")
            suffix = 2
            while stable in used:
                suffix_payload = json.dumps({"roles": list(medoid_roles), "tie": list(medoid_key), "suffix": suffix}, sort_keys=True).encode("utf-8")
                stable = f"P_{hashlib.sha256(suffix_payload).hexdigest()[:8]}"
                suffix += 1
            self.preference_id_registry[semantic_key] = stable
            used.add(stable)
            id_map[pid] = stable
        return id_map, semantic_key_by_pid

    def _prototype_continuity_id_map(self, new_clusters: Dict[str, Set[VariantKey]], new_roles: Dict[str, Dict[VariantKey, Tuple[str, ...]]]) -> Dict[str, str]:
        """Reuse a prior stable ID when the active-member cluster clearly persists."""
        old_clusters = getattr(self, "_active_pref_clusters", {}) or {}
        if not old_clusters or not new_clusters: return {}
        intersection = set().union(*old_clusters.values(), set()) & set().union(*new_clusters.values(), set())
        candidates: List[Tuple[float, float, str, str]] = []
        for new_pid, new_keys in new_clusters.items():
            for old_pid, old_keys in old_clusters.items():
                overlap = len((new_keys & old_keys) & intersection)
                overlap_score = 0.0
                if overlap > 0:
                    denom = max(1, min(len(new_keys & intersection), len(old_keys & intersection)))
                    overlap_score = overlap / denom
                role_score = self._cluster_role_similarity(new_roles.get(new_pid, {}), self._active_pref_cluster_roles.get(old_pid, {}))
                score = max(overlap_score, role_score)
                if overlap_score >= 0.50 or role_score >= 0.95: candidates.append((score, overlap_score, new_pid, old_pid))
        if not candidates: return {}
        candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
        used_new: Set[str] = set()
        used_old: Set[str] = set()
        occupied_ids = set(new_clusters)
        out: Dict[str, str] = {}
        for _score, _overlap, new_pid, old_pid in candidates:
            if new_pid in used_new or old_pid in used_old:      continue
            if new_pid != old_pid and old_pid in occupied_ids:  continue
            out[new_pid] = old_pid
            used_new.add(new_pid)
            used_old.add(old_pid)
        return {new_pid: old_pid for new_pid, old_pid in out.items() if new_pid != old_pid}

    def _cluster_role_similarity(self, a: Dict[VariantKey, Tuple[str, ...]], b: Dict[VariantKey, Tuple[str, ...]]) -> float:
        if not a or not b: return 0.0
        shared = sorted(set(a) & set(b))
        if shared:
            vals = [1.0 - self._role_sequence_distance(a[key], b[key]) for key in shared]
            return max(0.0, min(1.0, sum(vals) / max(1, len(vals))))
        best = 0.0
        for roles_a in a.values():
            for roles_b in b.values(): best = max(best, 1.0 - self._role_sequence_distance(roles_a, roles_b))
        return max(0.0, min(1.0, best))

    def _prototype_stability_report(self, old_clusters: Dict[str, Set[VariantKey]], new_clusters: Dict[str, Set[VariantKey]]) -> Dict[str, Any]:
        if not old_clusters and not new_clusters: return {"prototype_split": {}, "prototype_merge": {}, "prototype_split_count": 0, "prototype_merge_count": 0, "prototype_stability_ari": None, "prototype_active_intersection": 0}
        old_assign = {key: pid for pid, keys in old_clusters.items() for key in keys}
        new_assign = {key: pid for pid, keys in new_clusters.items() for key in keys}
        intersection = sorted(set(old_assign) & set(new_assign))
        old_to_new: Dict[str, Set[str]] = {}
        new_to_old: Dict[str, Set[str]] = {}
        for key in intersection:
            old_pid = old_assign[key]
            new_pid = new_assign[key]
            old_to_new.setdefault(old_pid, set()).add(new_pid)
            new_to_old.setdefault(new_pid, set()).add(old_pid)
        splits = {pid: sorted(vals) for pid, vals in old_to_new.items() if len(vals) > 1}
        merges = {pid: sorted(vals) for pid, vals in new_to_old.items() if len(vals) > 1}
        ari = self._adjusted_rand_index([old_assign[key] for key in intersection], [new_assign[key] for key in intersection]) if intersection else None
        return {"prototype_split": splits, "prototype_merge": merges, "prototype_split_count": len(splits), "prototype_merge_count": len(merges), "prototype_stability_ari": ari, "prototype_active_intersection": len(intersection)}

    @staticmethod
    def _adjusted_rand_index(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
        n = len(labels_a)
        if n != len(labels_b):  raise ValueError("ARI inputs must have equal length")
        if n < 2:               return None

        def comb2(x: int) -> float: return float(x * (x - 1) / 2)

        counts_a: Dict[str, int] = {}
        counts_b: Dict[str, int] = {}
        table: Dict[Tuple[str, str], int] = {}
        for a, b in zip(labels_a, labels_b):
            counts_a[a] = counts_a.get(a, 0) + 1
            counts_b[b] = counts_b.get(b, 0) + 1
            table[(a, b)] = table.get((a, b), 0) + 1
        sum_table = sum(comb2(v) for v in table.values())
        sum_a = sum(comb2(v) for v in counts_a.values())
        sum_b = sum(comb2(v) for v in counts_b.values())
        total = comb2(n)
        if total <= 0: return None
        expected = (sum_a * sum_b) / total
        max_index = 0.5 * (sum_a + sum_b)
        denom = max_index - expected
        if abs(denom) <= 1e-12: return 1.0
        return max(-1.0, min(1.0, (sum_table - expected) / denom))

    @staticmethod
    def _role_sequence_distance(a: Sequence[str], b: Sequence[str]) -> float:
        return kendall_tau_distance(a, b)

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
        self._reset_session_adaptation()
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
        """End of an ONLINE session: promote the currently inferred variant if it differs from the recipe's latest variant."""
        rid = self.inferred_recipe
        prefix = self.current_prefix
        transition_trace = list(self.current_transition_trace)
        identity_prefix = list(self.current_identity_prefix)
        if not prefix:
            # Nothing to commit.
            self.current_prefix = []
            self.current_transition_trace = []
            self.current_identity_prefix = []
            self.inferred_recipe = None
            self.inferred_variant_hash = None
            self.inferred_latent_pref_id = None
            self._online_unknown_streak = 0
            self._online_step_status = "known_confident"
            self._online_policy_history = []
            self._reset_session_adaptation()
            self._clear_preference_lock()
            return Classification("known", None, None, 0.0, 0.0)

        if commit_cls.kind == "new_recipe":     raise RuntimeError("online new-recipe commit reached mutating path")
        else:                                   rid = commit_cls.recipe_id
        if rid is None:
            if apply_decay and self.session_counter > 0: self.session_counter -= 1
            cls = Classification("needs_observation", None, None, commit_cls.jaccard, commit_cls.order_distance)
            self.classification_events.append((self.step_counter, cls))
            self.current_prefix = []
            self.current_transition_trace = []
            self.current_identity_prefix = []
            self.inferred_recipe = None
            self.inferred_variant_hash = None
            self.inferred_latent_pref_id = None
            self._needs_observation = True
            self._online_step_status = "unknown_confirmed"
            self._online_policy_history = []
            self._reset_session_adaptation()
            self._clear_preference_lock()
            self.posterior.reset()
            return cls

        h = variant_hash(prefix)
        confidence, confidence_parts = self._online_commit_confidence(rid, prefix, commit_cls)
        key = (rid, h)
        full_threshold = float(getattr(self.cfg, "online_commit_full_threshold", 0.75))
        tentative_threshold = float(getattr(self.cfg, "online_commit_tentative_threshold", 0.45))
        promotion_reasons = self._provisional_promotion_reasons(rid, h, confidence, confidence_parts)
        if promotion_reasons:
            confidence = max(confidence, full_threshold)
            confidence_parts["provisional_promotion"] = 1.0
        if confidence < tentative_threshold:
            if apply_decay and self.session_counter > 0: self.session_counter -= 1
            cls = Classification("needs_observation", rid, None, commit_cls.jaccard, commit_cls.order_distance)
            self.classification_events.append((self.step_counter, cls))
            self.current_prefix = []
            self.current_transition_trace = []
            self.current_identity_prefix = []
            self.inferred_recipe = None
            self.inferred_variant_hash = None
            self.inferred_latent_pref_id = None
            self._needs_observation = True
            self._online_step_status = "unknown_confirmed"
            self._online_policy_history = []
            self._reset_session_adaptation()
            self._clear_preference_lock()
            self.posterior.reset()
            return cls
        latest = self.memory.latest_variant(rid)
        known_variant = h in self.memory.variants.get(rid, {})
        if confidence < full_threshold and not known_variant:
            weight = max(float(getattr(self.cfg, "provisional_commit_weight", 0.20)), min(0.75, confidence))
            trace = self._remember_transition_trace(prefix, transition_trace)
            self.decay.register(rid, h, tuple(prefix), self.session_counter, self.retrain_cycle, weight=weight, pin_latest=False, transitions=trace, identity_ordering=identity_prefix)
            self.provisional_commits[key] = {"first_session": self.session_counter, "last_session": self.session_counter, "weight": weight, "confidence": confidence, "parts": confidence_parts, "promotion_eligible_until": self.session_counter + int(getattr(self.cfg, "provisional_confirm_window", 5))}
            self._rebuild_active_prototypes(self.decay.active_entries())
            cls = Classification("tentative_preference_shift", rid, h, commit_cls.jaccard, commit_cls.order_distance)
            self.narrate(f"[step {self.step_counter}] tentative online variant for '{rid}' (confidence={confidence:.2f})")
        elif known_variant:
            # Known replay: promote it at the session boundary, update timestamps/weights, and retrain.
            v = self._register_if_live(rid, list(prefix), self.step_counter, transitions=transition_trace, identity_ordering=identity_prefix)
            cls = commit_cls or Classification("known", rid, h, 1.0, 0.0)
            if reentry_from_pruned: cls.kind = "reentry_from_pruned"
        else:
            v = self._register_if_live(rid, list(prefix), self.step_counter, transitions=transition_trace, identity_ordering=identity_prefix)
            cls_kind = "preference_shift"
            if reentry_from_pruned: cls_kind = "reentry_from_pruned"
            cls = Classification(cls_kind, rid, v.variant_hash, commit_cls.jaccard, commit_cls.order_distance if commit_cls.kind != "new_recipe" else (0.0 if latest is None else 1.0))
            cls.recipe_id = rid
            cls.variant_hash = v.variant_hash
            self.provisional_commits.pop((rid, v.variant_hash), None)
            if promotion_reasons: self.online_commit_events.append({"step": self.step_counter, "event": "provisional_commit_promoted", "recipe_id": rid, "variant_hash": v.variant_hash, "promotion_reasons": promotion_reasons, "commit_confidence": confidence, "commit_confidence_parts": dict(confidence_parts)})
            self.narrate(f"[step {self.step_counter}] committing preference variant of '{rid}' from online session")
        if apply_decay: self.decay.step(self.session_counter, self.retrain_cycle)
        # Retrain after every completed demo because weights are updated by the decay/register path on every session.
        self._retrain()
        self.classification_events.append((self.step_counter, cls))
        self.current_prefix = []
        self.current_transition_trace = []
        self.current_identity_prefix = []
        self.inferred_recipe = None
        self.inferred_variant_hash = None
        self.inferred_latent_pref_id = None
        self._needs_observation = False
        self._online_unknown_streak = 0
        self._online_step_status = "known_confident"
        self._online_policy_history = []
        self._reset_session_adaptation()
        self._clear_preference_lock()
        self.posterior.reset()
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

    def _provisional_promotion_reasons(self, rid: str, variant_hash_: str, confidence: float, confidence_parts: Dict[str, float]) -> List[str]:
        key = (rid, variant_hash_)
        provisional = self.provisional_commits.get(key)
        if provisional is None: return []
        window = int(getattr(self.cfg, "provisional_confirm_window", 5))
        first_session = int(provisional.get("first_session", self.session_counter))
        if self.session_counter - first_session > window:
            self.provisional_commits.pop(key, None)
            return []
        reasons = ["same_variant_hash_recurred"]
        full_threshold = float(getattr(self.cfg, "online_commit_full_threshold", 0.75))
        if confidence >= full_threshold:                                                                                                                                            reasons.append("commit_confidence_full_threshold")
        if float(confidence_parts.get("posterior_recipe_confidence", 0.0)) >= float(getattr(self.cfg, "provisional_confirm_top1", 0.60)):                                           reasons.append("posterior_recipe_confidence_full_threshold")
        late_agreement = float(confidence_parts.get("late_window_prediction_agreement", 0.0))
        recipe_jaccard = float(confidence_parts.get("recipe_jaccard", 0.0))
        if (late_agreement >= float(getattr(self.cfg, "provisional_confirm_top1", 0.60)) and recipe_jaccard >= float(getattr(self.cfg, "provisional_min_recipe_jaccard", 0.95))):   reasons.append("late_window_prediction_alignment")
        provisional["last_session"] = self.session_counter
        provisional["last_confidence"] = confidence
        provisional["last_parts"] = dict(confidence_parts)
        return reasons

    def _online_commit_confidence(self, rid: str, prefix: Sequence[str], commit_cls: Classification) -> Tuple[float, Dict[str, float]]:
        marginals = self.posterior.marginal_recipe()
        p_recipe = float(marginals.get(rid, 0.0))
        second_recipe = max((float(p) for r, p in marginals.items() if r != rid), default=0.0)
        recipe_gap = max(0.0, p_recipe - second_recipe)
        active_lib = self.memory.library(allowed_keys=self._active_keys())
        full_lib = self.memory.library()
        scored: List[Tuple[str, float]] = []
        identity_prefix = (
            list(self.current_identity_prefix)
            if tuple(prefix) == tuple(self.current_prefix)
            and len(self.current_identity_prefix) == len(prefix)
            else list(prefix)
        )
        for v in full_lib:
            scored.append((v.recipe_id, self.disambig.classify(prefix, [v], identity_sequence=identity_prefix).jaccard))
        best_for_rid = max((score for recipe_id, score in scored if recipe_id == rid), default=float(commit_cls.jaccard or 0.0))
        second_jaccard = max((score for recipe_id, score in scored if recipe_id != rid), default=0.0)
        jaccard_margin = max(0.0, best_for_rid - second_jaccard)
        policy_history = list(self._online_policy_history)
        online_policy_confidence = sum(float(g.get("final_action_confidence") or g.get("action_confidence") or 0.0) for g in policy_history) / max(1, len(policy_history)) if policy_history else 0.0
        posterior_agrees = 1.0 if self.inferred_recipe is None or self.inferred_recipe == rid else 0.0
        late_agreement = self._late_window_prediction_agreement(prefix)
        # Known recipe preference shifts often have strong full-sequence materiality before the online posterior has fully recovered. Keep the Jaccard term dominant and use posterior/policy-confidence evidence as safeguards.
        identity_margin_scale = max(float(getattr(self.cfg, "online_commit_identity_margin_scale", 0.25)), 1e-9)
        recipe_gap_scale = max(float(getattr(self.cfg, "online_commit_recipe_gap_scale", 0.10)), 1e-9)
        score = (
            float(getattr(self.cfg, "online_commit_identity_weight", 0.46)) * min(1.0, best_for_rid)
            + float(getattr(self.cfg, "online_commit_identity_margin_weight", 0.18)) * min(1.0, jaccard_margin / identity_margin_scale if jaccard_margin > 0 else 0.0)
            + float(getattr(self.cfg, "online_commit_recipe_posterior_weight", 0.20)) * min(1.0, p_recipe)
            + float(getattr(self.cfg, "online_commit_recipe_gap_weight", 0.08)) * min(1.0, recipe_gap / recipe_gap_scale if recipe_gap > 0 else 0.0)
            + float(getattr(self.cfg, "online_commit_policy_confidence_weight", 0.04)) * online_policy_confidence
            + float(getattr(self.cfg, "online_commit_posterior_agreement_weight", 0.04)) * posterior_agrees
        )
        score = min(1.0, score + float(getattr(self.cfg, "online_commit_late_agreement_bonus", 0.06)) * late_agreement)
        if commit_cls.kind == "known" and best_for_rid >= float(getattr(self.cfg, "online_commit_known_identity_threshold", 0.99)):
            score = max(score, float(getattr(self.cfg, "online_commit_known_score_floor", 0.90)))
        if active_lib and not full_lib:
            score = min(score, float(getattr(self.cfg, "online_commit_empty_library_score_ceiling", 0.40)))
        return max(0.0, min(1.0, float(score))), {"posterior_recipe_confidence": p_recipe, "posterior_recipe_gap": recipe_gap, "recipe_jaccard": best_for_rid, "recipe_jaccard_margin": jaccard_margin, "recipe_identity_score": best_for_rid, "recipe_identity_margin": jaccard_margin, "online_policy_confidence": online_policy_confidence, "posterior_commit_recipe_agreement": posterior_agrees, "late_window_prediction_agreement": late_agreement}

    # interaction
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

    def _preference_ordered_transfer_frontier(self, prefix_tokens: Sequence[str], recipe_id: str, pref_id: str) -> Dict[str, float]:
        """Synthesize an unseen (recipe, preference) frontier from active recipe mass and the latent preference role policy."""
        remaining = self.recipe_prototypes.remaining_action_mass(prefix_tokens, recipe_id)
        if not remaining: return {}
        prefix_roles = self._roles_for_tokens(prefix_tokens)
        roles_by_token = {token: self.token_to_role.get(token, ROLE_UNKNOWN_OR_NOOP) for token in remaining}
        candidate_roles = list(roles_by_token.values())
        # Linear interpolation between recipe mass and preference score. pref_transfer_alpha: 0 = pure recipe mass, 1 = pure preference signal. Default 0.6 gives the preference signal meaningful weight even when weak.
        alpha = max(0.0, min(1.0, float(getattr(self.cfg, "pref_transfer_alpha", 0.6))))
        session_precedence = self._session_precedence_profile()
        scored: Dict[str, float] = {}
        for token, mass in remaining.items():
            role = roles_by_token.get(token, ROLE_UNKNOWN_OR_NOOP)
            pref_score = self.preference_prototypes.score_action_role(
                role,
                prefix_roles,
                pref_id,
                candidate_roles,
                session_precedence=session_precedence,
            )
            # Clamp both to [prob_floor, 1] before interpolating.
            m = max(float(mass), self.cfg.prob_floor)
            p = max(float(pref_score), self.cfg.prob_floor)
            scored[token] = (1.0 - alpha) * m + alpha * p
        z = sum(scored.values())
        return {tok: val / z for tok, val in scored.items()} if z > 0 else {}

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

    def _set_action_policy_stats(self, confidence: Optional[float], entropy: Optional[float], reason: str, *, raw_confidence: Optional[float] = None, margin: Optional[float] = None, source: Optional[str] = None,
        final_confidence: Optional[float] = None, final_entropy: Optional[float] = None, final_margin: Optional[float] = None, conditioned_confidence: Optional[float] = None, conditioned_entropy: Optional[float] = None, conditioned_margin: Optional[float] = None,
        conditioned_top_token: Optional[str] = None, ensemble_confidence: Optional[float] = None, ensemble_entropy: Optional[float] = None, ensemble_margin: Optional[float] = None, ensemble_top_token: Optional[str] = None,
        posterior_confidence: Optional[float] = None, posterior_margin: Optional[float] = None, agreement: Optional[bool] = None, blend_strength: Optional[float] = None,) -> None:
        if self._frozen: return
        raw = confidence if raw_confidence is None else raw_confidence
        final_conf = confidence if final_confidence is None else final_confidence
        final_ent = entropy if final_entropy is None else final_entropy
        final_marg = margin if final_margin is None else final_margin
        self._last_action_policy = {
            "robot_action_mandatory": True,
            "action_confidence": confidence,
            "raw_action_confidence": raw,
            "action_margin": margin,
            "action_entropy": entropy,
            "policy_source": source,
            "final_action_confidence": final_conf,
            "final_action_margin": final_marg,
            "final_action_entropy": final_ent,
            "conditioned_action_confidence": conditioned_confidence,
            "conditioned_action_entropy": conditioned_entropy,
            "conditioned_action_margin": conditioned_margin,
            "conditioned_top_token": conditioned_top_token,
            "ensemble_action_confidence": ensemble_confidence,
            "ensemble_action_entropy": ensemble_entropy,
            "ensemble_action_margin": ensemble_margin,
            "ensemble_top_token": ensemble_top_token,
            "posterior_confidence": posterior_confidence,
            "posterior_margin": posterior_margin,
            "session_correction_count": int(self._session_correction_count),
            "session_precedence_edge_count": len(self._session_precedence_evidence),
            "policy_agreement": agreement,
            "blend_strength": blend_strength,
            "reason": reason,
        }

    def _distribution_stats(self, dist: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not dist: return None, None, None
        vals = sorted((float(v) for v in dist.values()), reverse=True)
        confidence = vals[0] if vals else None
        margin = vals[0] - vals[1] if len(vals) >= 2 else (vals[0] if vals else None)
        entropy = 0.0
        for p in vals: 
            if p > 0: entropy -= p * math.log(p)
        if len(vals) > 1: entropy = entropy / math.log(len(vals))
        return confidence, entropy, margin

    def _top_token(self, dist: Dict[str, float]) -> Optional[str]:
        if not dist: return None
        return max(dist.items(), key=lambda kv: float(kv[1]))[0]

    def _posterior_decision_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """Return pre-action posterior concentration and top-two margin."""
        joint = self.posterior.joint()
        if not joint:
            return None, None
        values = sorted((float(value) for value in joint.values()), reverse=True)
        concentration = float(self.posterior.confidence())
        margin = values[0] - values[1] if len(values) > 1 else values[0]
        return concentration, margin

    def _update_posterior_for_prefix(self, prefix_tokens: Sequence[str]) -> None:
        if not self.recipe_prototypes.all(): return
        memory_for_recipe, memory_for_pair, compatibility_for_pair = self._posterior_callbacks()
        self.posterior.update(prefix_tokens=prefix_tokens, prefix_roles=self._roles_for_tokens(prefix_tokens), recipe_protos=self.recipe_prototypes, pref_protos=self.preference_prototypes, memory_state_for_recipe=memory_for_recipe,
            memory_state_for_pair=memory_for_pair, compatibility_for_pair=compatibility_for_pair)

    def _ensure_posterior_for_prefix(self, prefix_tokens: Sequence[str]) -> None:
        self._update_posterior_for_prefix(prefix_tokens)

    def _prefix_needs_observation(self, prefix_tokens: Sequence[str]) -> bool:
        # _needs_observation is a session-end outcome flag. Prefix-time gating must remain step-local so one bad path cannot silence the rest of an active online session.
        min_prefix = int(getattr(self.cfg, "online_new_recipe_min_prefix", 3))
        if len(prefix_tokens) < max(1, min_prefix):
            if not self._frozen:
                self._online_step_status = "known_ambiguous" if prefix_tokens else "known_confident"
                self._online_unknown_streak = 0
            return False
        active_keys = self._active_keys()
        active_lib = self.memory.library(allowed_keys=active_keys)
        if not active_lib:
            needs_step_observation = bool(self.memory.library())
            if not self._frozen:
                self._online_unknown_streak = self._online_unknown_streak + 1 if needs_step_observation else 0
                self._online_step_status = "unknown_candidate" if needs_step_observation else "known_ambiguous"
            return needs_step_observation
        threshold = float(getattr(self.cfg, "online_new_recipe_partial_threshold", 0.30))
        identity_prefix = (
            list(self.current_identity_prefix)
            if tuple(prefix_tokens) == tuple(self.current_prefix)
            and len(self.current_identity_prefix) == len(prefix_tokens)
            else list(prefix_tokens)
        )
        best_active = max((s for _v, s in self.disambig.score_partial(identity_prefix, active_lib, identity=True)), default=0.0)
        active_recipes = {v.recipe_id for v in active_lib}
        best_recipe = max((self.recipe_prototypes.recipe_match(prefix_tokens, rid) for rid in active_recipes), default=0.0)
        if max(best_active, best_recipe) >= threshold:
            if not self._frozen:
                self._online_unknown_streak = 0
                self._online_step_status = "known_confident" if best_active >= threshold else "known_ambiguous"
            return False
        if not self._frozen:
            self._online_unknown_streak += 1
            self._online_step_status = "unknown_candidate"
        return True

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
        """Posterior- and preference-conditioned next-action distribution.

        Fusion order:
            1. Compute the unconditioned ensemble distribution.
            2. If the posterior is empty, fall back to the ensemble.
            3. Otherwise, build a posterior-conditioned distribution from the active recipe frontier + preference-head re-weighting under the top latent pref_id, then mix with the ensemble at strength `cfg.posterior_assist_strength`.
        """
        with self._profile("predict_next_tokens"):
            prefix_tokens = self._coerce_prefix_tokens(prefix) if prefix is not None else list(self.current_prefix)
            state = self._state_from_prefix(prefix_tokens)
            p_irl = self.irl.predict(state)
            p_mk = self.markov.predict(state, prefix_tokens)
            ensemble_dist = ensemble_predict(p_irl, p_mk, cfg=self.cfg)
            ensemble_confidence, ensemble_entropy, ensemble_margin = self._distribution_stats(ensemble_dist)

            if getattr(self.cfg, "ablation_disable_posterior", False):
                self._set_action_policy_stats(ensemble_confidence, ensemble_entropy, "posterior_disabled", margin=ensemble_margin, source="posterior_disabled", final_confidence=ensemble_confidence, final_entropy=ensemble_entropy, final_margin=ensemble_margin, ensemble_confidence=ensemble_confidence, ensemble_entropy=ensemble_entropy, ensemble_margin=ensemble_margin, ensemble_top_token=self._top_token(ensemble_dist))
                return ensemble_dist
            if self._prefix_needs_observation(prefix_tokens):
                self._set_action_policy_stats(ensemble_confidence, ensemble_entropy, "needs_observation", margin=ensemble_margin, source="needs_observation", final_confidence=ensemble_confidence, final_entropy=ensemble_entropy, final_margin=ensemble_margin, ensemble_confidence=ensemble_confidence, ensemble_entropy=ensemble_entropy, ensemble_margin=ensemble_margin, ensemble_top_token=self._top_token(ensemble_dist))
                return ensemble_dist
            self._update_posterior_for_prefix(prefix_tokens)
            if not self.posterior.joint():
                final_confidence, final_entropy, final_margin = self._distribution_stats(ensemble_dist)
                self._set_action_policy_stats(final_confidence, final_entropy, "posterior_empty", raw_confidence=final_confidence, margin=final_margin, source="ensemble_posterior_empty",
                    final_confidence=final_confidence, final_entropy=final_entropy, final_margin=final_margin, ensemble_confidence=ensemble_confidence, ensemble_entropy=ensemble_entropy, ensemble_margin=ensemble_margin, ensemble_top_token=self._top_token(ensemble_dist))
                return ensemble_dist

            conditioned, confidence, entropy, margin = self._action_marginal_distribution(prefix_tokens)
            if not conditioned:
                final_confidence, final_entropy, final_margin = self._distribution_stats(ensemble_dist)
                self._set_action_policy_stats(final_confidence, final_entropy, "no_conditioned_frontier", raw_confidence=final_confidence, margin=final_margin, source="ensemble_no_conditioned_frontier",
                    final_confidence=final_confidence, final_entropy=final_entropy, final_margin=final_margin, conditioned_confidence=confidence, conditioned_entropy=entropy, conditioned_margin=margin,
                    ensemble_confidence=ensemble_confidence, ensemble_entropy=ensemble_entropy, ensemble_margin=ensemble_margin, ensemble_top_token=self._top_token(ensemble_dist))
                return ensemble_dist

            agreement = self._top_token(conditioned) == self._top_token(ensemble_dist)
            posterior_confidence, posterior_margin = self._posterior_decision_stats()
            blend_strength = self._posterior_blend_strength(
                conditioned,
                ensemble_dist,
                confidence,
                entropy,
                margin,
                ensemble_confidence,
                ensemble_entropy,
                ensemble_margin,
                posterior_confidence=posterior_confidence,
                posterior_margin=posterior_margin,
            )
            final_dist = self._mix_distributions(conditioned, ensemble_dist, strength=blend_strength)
            final_confidence, final_entropy, final_margin = self._distribution_stats(final_dist)
            self._set_action_policy_stats(final_confidence, final_entropy, "conditioned_blend", raw_confidence=final_confidence, margin=final_margin, source="conditioned_blend", final_confidence=final_confidence, final_entropy=final_entropy,
                final_margin=final_margin, conditioned_confidence=confidence, conditioned_entropy=entropy, conditioned_margin=margin, conditioned_top_token=self._top_token(conditioned),
                ensemble_confidence=ensemble_confidence, ensemble_entropy=ensemble_entropy, ensemble_margin=ensemble_margin, ensemble_top_token=self._top_token(ensemble_dist),
                posterior_confidence=posterior_confidence, posterior_margin=posterior_margin, agreement=agreement, blend_strength=blend_strength)
            return final_dist

    def predict_with_oracle(self, prefix: Optional[Sequence[str]], oracle_recipe_id: Optional[str], oracle_pref_id: Optional[str]) -> Dict[str, float]:
        """Oracle upper-bound prediction. Bypass the posterior and condition prediction directly on the supplied ground-truth labels. This is a leakage baseline; it is only ever called from explicitly-labelled oracle agents in the ablation matrix (the harness gates this on cfg.ablation_oracle_*). NOT deployable."""
        with self._profile("predict_with_oracle"):
            prefix_tokens = self._coerce_prefix_tokens(prefix) if prefix is not None else list(self.current_prefix)
            state = self._state_from_prefix(prefix_tokens)
            ensemble_dist = ensemble_predict(self.irl.predict(state), self.markov.predict(state, prefix_tokens), cfg=self.cfg)
            if oracle_recipe_id is None: return ensemble_dist
            # `_conditioned_distribution` already applies preference rescoring internally when a pref_id is supplied (see its docstring); the previous implementation re-applied the same rescore here, which squared the preference score and double-shrunk the distribution.
            conditioned = self._conditioned_distribution(prefix_tokens, oracle_recipe_id, oracle_pref_id)
            if not conditioned:     return ensemble_dist
            return self._mix_distributions(conditioned, ensemble_dist)

    def _same_role_order_score(self, token: str, role: str, roles_by_token: Dict[str, str], recipe_proto: Any) -> float:
        if recipe_proto is None:    return 1.0
        same_role = [other for other, other_role in roles_by_token.items() if other != token and other_role == role]
        if not same_role:           return 1.0
        blockers = sum(float(recipe_proto.precedence_counts.get((other, token), 0.0)) for other in same_role)
        leads = sum(float(recipe_proto.precedence_counts.get((token, other), 0.0)) for other in same_role)
        total = blockers + leads
        if total <= 0.0:            return 1.0
        order_score = (1.0 + leads) / (2.0 + total)
        return max(0.35, min(1.45, 0.60 + 0.80 * order_score))

    def _conditioned_distribution(self, prefix_tokens: Sequence[str], top_rid: str, pref_id: Optional[str] = None) -> Dict[str, float]:
        """Conditioned distribution under posterior argmax (recipe, pref). Builds the recipe-prototype frontier from active variant orderings only. Pruned variants are intentionally invisible during live assistance; they are consulted only at completed-demo reentry time. Then re-weights candidate tokens using the 
        latent preference head's role-conditioned bigram/trigram score under the posterior's argmax pref_id."""
        active_keys = self._active_keys()
        active_variants = self.memory.variants_of(top_rid, allowed_keys=active_keys)
        weights_map = {entry.key: float(entry.weight) for entry in self.decay.active_entries()}
        variant_orderings: List[Tuple[Sequence[str], float]] = []
        pref_variant_orderings: List[Tuple[Sequence[str], float]] = []
        for v in active_variants:
            w = weights_map.get((top_rid, v.variant_hash), 1.0)
            row = (v.ordering, w)
            variant_orderings.append(row)
            if pref_id is not None and self.variant_pref_ids.get((top_rid, v.variant_hash)) == pref_id:
                pref_variant_orderings.append(row)
        if not variant_orderings:
            return {}
        conditioned_orderings = pref_variant_orderings if pref_id is not None and pref_variant_orderings else variant_orderings

        pref_proto = self.preference_prototypes.prototypes.get(pref_id) if pref_id is not None else None
        transfer_pair = pref_proto is not None and top_rid not in pref_proto.recipes_seen and pref_id is not None
        # The -recipe_prototype ablation skips the prototype frontier and uses only the exact-memory active variant alignment.
        if getattr(self.cfg, "ablation_disable_recipe_prototype", False):
            frontier = self.recipe_prototypes._align_frontier(prefix_tokens, conditioned_orderings)
        elif transfer_pair and not getattr(self.cfg, "ablation_disable_preference_head", False):
            frontier = self._preference_ordered_transfer_frontier(prefix_tokens, top_rid, pref_id)
            if frontier: return frontier
        else:
            align_weight = float(getattr(self.cfg, "recipe_frontier_align_weight", 0.45))
            if transfer_pair: align_weight = min(align_weight, float(getattr(self.cfg, "recipe_frontier_transfer_align_weight", 0.0)))
            frontier = self.recipe_prototypes.frontier(prefix_tokens, top_rid, conditioned_orderings, align_weight=align_weight, remaining_variants=conditioned_orderings)
        if not frontier:
            return {}

        # The -preference_head ablation skips preference-head re-weighting.
        if getattr(self.cfg, "ablation_disable_preference_head", False):    return frontier

        top_pid = pref_id if pref_id is not None else self.posterior.argmax_preference()
        if top_pid is None:                                                 return frontier

        prefix_roles = self._roles_for_tokens(prefix_tokens)
        roles_by_token = {token: self.token_to_role.get(token, ROLE_UNKNOWN_OR_NOOP) for token in frontier}
        candidate_roles = list(roles_by_token.values())
        recipe_proto = self.recipe_prototypes.get(top_rid)
        session_precedence = self._session_precedence_profile()
        rescored: Dict[str, float] = {}
        for token, p in frontier.items():
            role = roles_by_token.get(token, ROLE_UNKNOWN_OR_NOOP)
            pref_score = self.preference_prototypes.score_action_role(
                role,
                prefix_roles,
                top_pid,
                candidate_roles,
                session_precedence=session_precedence,
            )
            pref_score *= self._same_role_order_score(token, role, roles_by_token, recipe_proto)
            rescored[token] = p * pref_score
        z = sum(rescored.values())
        if z <= 0: return frontier
        return {t: v / z for t, v in rescored.items()}

    def _locked_variant_frontier(self, prefix_tokens: Sequence[str]) -> Dict[str, float]:
        key = self.locked_variant_key
        if key is None: 
            return {}
        if key not in self._active_keys():
            self._clear_preference_lock()
            return {}
        rid, variant_hash_ = key
        variant = self.memory.variants.get(rid, {}).get(variant_hash_)
        if variant is None:
            self._clear_preference_lock()
            return {}
        if prefix_tokens:
            ranked = self.disambig.score_partial(prefix_tokens, [variant])
            score = ranked[0][1] if ranked else 0.0
            if score < float(getattr(self.cfg, "online_new_recipe_partial_threshold", 0.30)):
                self._clear_preference_lock()
                return {}
        idx = _aligned_next_index(prefix_tokens, variant.ordering)
        if idx >= len(variant.ordering):
            return {}
        return {variant.ordering[idx]: 1.0}

    def _action_marginal_distribution(self, prefix_tokens: Sequence[str]) -> Tuple[Dict[str, float], Optional[float], Optional[float], Optional[float]]:
        joint = self.posterior.joint()
        if not joint: return {}, None, None, None
        k = max(1, int(getattr(self.cfg, "posterior_action_topk", 16)))
        ranked = sorted(joint.items(), key=lambda kv: -float(kv[1]))[:k]
        out: Dict[str, float] = {}
        mass = 0.0
        for (rid, pid), jp in ranked:
            if rid == OnlinePreferencePosterior.UNSEEN_RECIPE: continue
            pref_id = None if pid == OnlinePreferencePosterior.UNSEEN_PREF else pid
            dist = self._conditioned_distribution(prefix_tokens, rid, pref_id)
            if not dist: continue
            w = float(jp)
            mass += w
            for token, p in dist.items(): out[token] = out.get(token, 0.0) + w * float(p)
        if mass <= 0.0 or not out:  return {}, None, None, None
        z = sum(out.values())
        if z <= 0.0:                return {}, None, None, None
        out = {t: v / z for t, v in out.items()}
        locked_frontier = self._locked_variant_frontier(prefix_tokens)
        if locked_frontier:
            boost = max(0.0, min(1.0, float(getattr(self.cfg, "locked_variant_action_boost", 0.70))))
            boosted: Dict[str, float] = {}
            for token in set(out) | set(locked_frontier):
                boosted[token] = (1.0 - boost) * out.get(token, 0.0) + boost * locked_frontier.get(token, 0.0)
            bz = sum(boosted.values())
            if bz > 0.0:
                out = {t: v / bz for t, v in boosted.items()}
        confidence, entropy, margin = self._distribution_stats(out)
        return out, confidence, entropy, margin

    def _posterior_blend_strength(self, conditioned: Dict[str, float], ensemble: Dict[str, float], conditioned_confidence: Optional[float] = None, conditioned_entropy: Optional[float] = None, conditioned_margin: Optional[float] = None, ensemble_confidence: Optional[float] = None,
        ensemble_entropy: Optional[float] = None, ensemble_margin: Optional[float] = None, *, posterior_confidence: Optional[float] = None, posterior_margin: Optional[float] = None) -> float:
        base = float(getattr(self.cfg, "posterior_assist_strength", 0.70))
        lo = float(getattr(self.cfg, "posterior_assist_strength_min", 0.15))
        hi = float(getattr(self.cfg, "posterior_assist_strength_max", 0.90))
        if hi < lo: lo, hi = hi, lo
        strength = base
        if posterior_confidence is not None:
            # Posterior confidence is normalized entropy complement, hence the
            # centred correction is invariant to the number of hypotheses.
            strength += float(getattr(self.cfg, "posterior_assist_posterior_confidence_weight", 0.20)) * (float(posterior_confidence) - 0.50)
        if posterior_margin is not None:
            strength += float(getattr(self.cfg, "posterior_assist_posterior_margin_bonus", 0.10)) * max(0.0, float(posterior_margin))
        if conditioned and ensemble and self._top_token(conditioned) == self._top_token(ensemble):
            strength += float(getattr(self.cfg, "posterior_assist_agreement_bonus", 0.15))
        elif conditioned and ensemble:
            c = float(conditioned_confidence or 0.0)
            e = float(ensemble_confidence or 0.0)
            strength -= float(getattr(self.cfg, "posterior_assist_disagreement_penalty", 0.45)) * max(0.0, e - c)
            strength += float(getattr(self.cfg, "posterior_assist_conditioned_advantage_bonus", 0.15)) * max(0.0, c - e)
            if conditioned_margin is not None and ensemble_margin is not None:
                strength += float(getattr(self.cfg, "posterior_assist_margin_advantage_bonus", 0.10)) * max(0.0, float(conditioned_margin) - float(ensemble_margin))
            if conditioned_entropy is not None and ensemble_entropy is not None:
                strength += float(getattr(self.cfg, "posterior_assist_entropy_advantage_bonus", 0.10)) * max(0.0, float(ensemble_entropy) - float(conditioned_entropy))
        return max(0.0, min(1.0, max(lo, min(hi, strength))))

    def _mix_distributions(self, conditioned: Dict[str, float], ensemble: Dict[str, float], strength: Optional[float] = None) -> Dict[str, float]:
        """Convex-combine conditioned and ensemble distributions."""
        if strength is None: strength = self._posterior_blend_strength(conditioned, ensemble)
        strength = max(0.0, min(1.0, strength))
        out: Dict[str, float] = {}
        keys = set(conditioned.keys()) | set(ensemble.keys())
        for tok in keys: out[tok] = strength * conditioned.get(tok, 0.0) + (1.0 - strength) * ensemble.get(tok, 0.0)
        z = sum(out.values())
        if z <= 0:  return ensemble
        return {t: v / z for t, v in out.items()}

    def predict_next(self, prefix: Optional[Sequence[str]] = None) -> Dict[str, float]:
        return self._decode_distribution(self.predict_next_tokens(prefix))

    def observe_observation(
        self,
        observation: ActionObservation,
        ground_truth_recipe: Optional[str] = None,
        *,
        precomputed_distribution: Optional[Mapping[str, float]] = None,
    ) -> StepResult:
        """Consume one state/action-vector observation. `ground_truth_recipe` is optional and used only for metrics.  The behavior policy only sees the anonymous action token derived from the binary transition vector. """
        if self._frozen:
            action = self.action_vector_to_token.get(observation.action_vector, "act_unseen")
            prefix = list(self.current_prefix)
            dist = dict(precomputed_distribution) if precomputed_distribution is not None else self.predict_next_tokens(prefix)
            predicted_token = top_k(dist, k=1)[0] if dist else None
            return StepResult(step=self.step_counter, predicted=self._display_token(predicted_token), actual=action, correct=predicted_token == action, mode=self.mode, inferred_recipe=self.inferred_recipe, inferred_variant_hash=self.inferred_variant_hash, inferred_latent_pref_id=self.inferred_latent_pref_id, event="frozen_prediction",)
        self.step_counter += 1
        # NOTE: decay advances once per session (see end_demo), not per action; otherwise weights collapse within one demo.

        if self.mode == MODE_OBSERVE:
            action = self._token_for_vector(observation.action_vector)
            self._cache_token_role(action, observation)
            self.pending_demo.append(action)
            self.pending_demo_transitions.append(self._transition_from_observation(action, observation))
            self.pending_demo_identity.append(identity_token_from_observation(observation))
            result = StepResult(step=self.step_counter, predicted=None, actual=self._display_token(action) or action, correct=False, mode=self.mode, inferred_recipe=None, inferred_variant_hash=None, inferred_latent_pref_id=None, event="observing")
            self.step_log.append(result)
            return result
        # ONLINE mode.
        first_online_action = not self.current_prefix and precomputed_distribution is None
        if first_online_action:
            # There is no prefix yet, so treating the first human action as a
            # meaningful prediction pollutes online confidence/switch evidence.
            # Subsequent human turns are predicted from the observed prefix.
            dist = {}
        else:
            dist = dict(precomputed_distribution) if precomputed_distribution is not None else self.predict_next_tokens(self.current_prefix)
            # With a precomputed distribution, the caller must have just produced it
            # through the policy path that set _last_action_policy for this prefix.
            self._online_policy_history.append(dict(self._last_action_policy))
        predicted_token = top_k(dist, k=1)[0] if dist else None
        action = self.action_vector_to_token.get(observation.action_vector)
        if action is None: action = self._token_for_vector(observation.action_vector)
        self._cache_token_role(action, observation)
        correct = predicted_token == action
        # Commit the action to prefix.
        self.current_prefix.append(action)
        self.current_transition_trace.append(self._transition_from_observation(action, observation))
        self.current_identity_prefix.append(identity_token_from_observation(observation))
        # Re-evaluate recipe identity each step. Evidence accumulates with the prefix, so a wrong first-action commit can be corrected when later actions make another recipe clearly more likely.
        switch_evidence_ok = bool(precomputed_distribution is not None or (not first_online_action and correct))
        self._refresh_recipe_inference(self.current_prefix, switch_evidence_ok=switch_evidence_ok)

        event = ""
        if not correct and self.inferred_recipe is not None:
            # Possibly a preference shift: rank variants by prefix match.
            variants = self.memory.variants_of(self.inferred_recipe, allowed_keys=self._active_keys())
            ranked = self.disambig.score_partial(self.current_prefix, variants)
            if ranked:
                top_variant, _ = ranked[0]
                if top_variant.variant_hash != self.inferred_variant_hash:
                    # Do not mutate inferred_variant_hash when frozen; probes should not alter persistent inference state across calls.
                    if not self._frozen:
                        self.inferred_variant_hash = top_variant.variant_hash
                        self.locked_variant_key = (top_variant.recipe_id, top_variant.variant_hash)
                        self.locked_pref_id = self.variant_pref_ids.get(self.locked_variant_key)
                    event = "preference_shift_locked"
                    self.narrate(f"[step {self.step_counter}] MISMATCH (predicted '{self._display_token(predicted_token)}', got '{self._display_token(action)}') -> locking onto variant {top_variant.variant_hash[:24]}...")

        result = StepResult(step=self.step_counter, predicted=self._display_token(predicted_token), actual=self._display_token(action) or action, correct=correct, mode=self.mode, inferred_recipe=self.inferred_recipe, inferred_variant_hash=self.inferred_variant_hash, inferred_latent_pref_id=self.inferred_latent_pref_id, event=event)
        self.step_log.append(result)
        self.accuracy_events.append((self.step_counter, ground_truth_recipe or (self.inferred_recipe or "?"), correct))
        return result

    # recipe identification (online, running posterior)
    def _memory_state_for_recipe(self, rid: str) -> MemoryPrior:
        """Continuous memory prior for live posterior use."""
        active_entries = [e for (recipe_id, _h), e in self.decay.active.items() if recipe_id == rid]
        if active_entries:
            return MemoryPrior(state="active", active_weight=max(float(e.weight) for e in active_entries))
        return MemoryPrior("absent")

    def _active_pair_masses(self) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], Dict[str, float]]:
        pair_mass: Dict[Tuple[str, str], float] = {}
        recipe_mass: Dict[str, float] = {}
        pref_mass: Dict[str, float] = {}
        for entry in self.decay.active_entries():
            pid = self.variant_pref_ids.get(entry.key)
            if pid is None:
                continue
            w = max(0.0, float(entry.weight))
            recipe_mass[entry.recipe_id] = recipe_mass.get(entry.recipe_id, 0.0) + w
            pref_mass[pid] = pref_mass.get(pid, 0.0) + w
            key = (entry.recipe_id, pid)
            pair_mass[key] = pair_mass.get(key, 0.0) + w
        return pair_mass, recipe_mass, pref_mass

    def _posterior_callbacks(self):
        pair_mass, recipe_mass, pref_mass = self._active_pair_masses()

        def memory_for_pair(rid: str, pref_id: str) -> MemoryPrior:
            return self._memory_state_for_pair(rid, pref_id, pair_mass=pair_mass, recipe_mass=recipe_mass)

        def compatibility_for_pair(rid: str, pref_id: str) -> float:
            return self._compatibility_for_pair(rid, pref_id, pair_mass=pair_mass, recipe_mass=recipe_mass, pref_mass=pref_mass)

        return self._memory_state_for_recipe, memory_for_pair, compatibility_for_pair

    def _memory_state_for_pair(self, rid: str, pref_id: str, *, pair_mass: Optional[Dict[Tuple[str, str], float]] = None, recipe_mass: Optional[Dict[str, float]] = None) -> MemoryPrior:
        """Pair-level memory prior used by the joint posterior.

        Latest pinning protects replay retention; it should not collapse every
        active recipe to unit posterior memory evidence for every preference.
        """
        if pair_mass is None or recipe_mass is None:
            pair_mass, recipe_mass, _pref_mass = self._active_pair_masses()
        r_mass = float(recipe_mass.get(rid, 0.0))
        if r_mass > 0.0:
            p_mass = float(pair_mass.get((rid, pref_id), 0.0))
            return MemoryPrior(state="active", active_weight=max(0.0, min(1.0, p_mass / max(r_mass, 1e-9))))
        return self._memory_state_for_recipe(rid)

    def _compatibility_for_pair(self, rid: str, pref_id: str, *, pair_mass: Optional[Dict[Tuple[str, str], float]] = None, recipe_mass: Optional[Dict[str, float]] = None, pref_mass: Optional[Dict[str, float]] = None) -> float:
        """Smoothed active-memory estimate of P(pref | recipe)."""
        if pair_mass is None or recipe_mass is None or pref_mass is None:
            pair_mass, recipe_mass, pref_mass = self._active_pair_masses()
        r_mass = float(recipe_mass.get(rid, 0.0))
        floor = max(float(getattr(self.cfg, "posterior_compat_floor", 1e-4)), 1e-9)
        smooth = max(0.0, float(getattr(self.cfg, "posterior_compat_smoothing", 0.25)))
        total_pref_mass = float(sum(pref_mass.values()))
        pref_ids = set(self.preference_prototypes.all_pref_ids()) | set(pref_mass)
        if total_pref_mass > 0.0:
            global_prior = float(pref_mass.get(pref_id, 0.0)) / total_pref_mass
        else:
            global_prior = 1.0 / max(1, len(pref_ids))
        if global_prior <= 0.0 and pref_ids:
            global_prior = floor
        if r_mass <= 0.0:
            return max(floor, min(1.0, global_prior))
        p_mass = float(pair_mass.get((rid, pref_id), 0.0))
        compat = (p_mass + smooth * global_prior) / max(r_mass + smooth, 1e-9)
        return max(floor, min(1.0, compat))

    def _refresh_recipe_inference(self, prefix: Sequence[str], *, switch_evidence_ok: bool = True) -> None:
        """Posterior-driven recipe inference. Single online identity head. The posterior owns the running belief. The disambiguator is no longer consulted on the per-step path; it runs only at observation-mode entry (`_end_observe_demo` / `_classify_online_prefix`). The `ablation_disable_posterior` flag now means: 
        do not commit any recipe identity, leaving `inferred_recipe` at its previous value (None until the agent has seen a clean exact-variant match)."""
        if getattr(self.cfg, "ablation_disable_posterior", False): return
        prefix_tokens = self._coerce_prefix_tokens(prefix)
        prefix_roles = self._roles_for_tokens(prefix_tokens)
        memory_for_recipe, memory_for_pair, compatibility_for_pair = self._posterior_callbacks()
        self.posterior.update(prefix_tokens=prefix_tokens, prefix_roles=prefix_roles, recipe_protos=self.recipe_prototypes, pref_protos=self.preference_prototypes, memory_state_for_recipe=memory_for_recipe,
            memory_state_for_pair=memory_for_pair, compatibility_for_pair=compatibility_for_pair)
        self._prefix_needs_observation(prefix_tokens)
        new_rid = self.posterior.argmax_recipe()
        new_pid = self.posterior.argmax_preference()
        if not self._frozen: self.inferred_latent_pref_id = new_pid
        if self._frozen: return
        old_rid = self.inferred_recipe

        # Cold commit: no current identity, accept the first non-None argmax.
        if old_rid is None:
            if new_rid is not None and self._cold_recipe_evidence_ok(new_rid):
                self.inferred_recipe = new_rid
                self.inferred_variant_hash = None
                if self.locked_variant_key is not None and self.locked_variant_key[0] != new_rid: self._clear_preference_lock()
                self._pending_argmax = None
                self._pending_count = 0
                self.narrate(f"[step {self.step_counter}] posterior set recipe = '{new_rid}' (confidence={self.posterior.confidence():.2f})")
            return

        # Already committed and posterior agrees: reset any pending switch.
        if new_rid is None or new_rid == old_rid:
            self._pending_argmax = None
            self._pending_count = 0
            return

        # Disagreement. Apply the hysteresis gate: log-ratio margin AND K consecutive steps of agreement on the new argmax.
        if self._fast_switch_recipe_evidence_ok(old_rid, new_rid):
            self.inferred_recipe = new_rid
            self.inferred_variant_hash = None
            if self.locked_variant_key is not None and self.locked_variant_key[0] != new_rid: self._clear_preference_lock()
            self._pending_argmax = None
            self._pending_count = 0
            self._record_posterior_switch_event("fast_path", old_rid, new_rid, committed=True)
            self.narrate(f"[step {self.step_counter}] posterior fast-switched recipe '{old_rid}' -> '{new_rid}' (confidence={self.posterior.confidence():.2f})")
            return
        if not self._switch_recipe_evidence_ok(old_rid, new_rid):
            self._record_posterior_switch_event("none", old_rid, new_rid, committed=False, reason="evidence_gate_failed")
            return
        if new_rid != self._pending_argmax:
            if not switch_evidence_ok:
                self._record_posterior_switch_event("none", old_rid, new_rid, committed=False, reason="pending_evidence_blocked")
                return
            self._pending_argmax = new_rid
            self._pending_count = 1
            self._record_posterior_switch_event("none", old_rid, new_rid, committed=False, reason="pending_started")
            return
        if not switch_evidence_ok:
            self._record_posterior_switch_event("none", old_rid, new_rid, committed=False, reason="pending_evidence_blocked")
            return
        self._pending_count += 1
        if self._pending_count < int(getattr(self.cfg, "posterior_switch_agreement", 2)):
            self._record_posterior_switch_event("none", old_rid, new_rid, committed=False, reason="pending_wait")
            return

        # Commit the switch.
        self.inferred_recipe = new_rid
        self.inferred_variant_hash = None
        if self.locked_variant_key is not None and self.locked_variant_key[0] != new_rid: self._clear_preference_lock()
        self._pending_argmax = None
        self._pending_count = 0
        self._record_posterior_switch_event("consecutive_agreement", old_rid, new_rid, committed=True)
        self.narrate(f"[step {self.step_counter}] posterior switched recipe '{old_rid}' -> '{new_rid}' (confidence={self.posterior.confidence():.2f})")

    def _record_posterior_switch_event(self, switch_path: str, old_rid: Optional[str], new_rid: Optional[str], *, committed: bool, reason: str = "") -> None:
        marginals = self.posterior.marginal_recipe()
        self.posterior_switch_events.append({
            "step": int(self.step_counter),
            "switch_path": switch_path,
            "committed": bool(committed),
            "reason": reason,
            "old_recipe": old_rid,
            "new_recipe": new_rid,
            "confidence": float(self.posterior.confidence()),
            "pending_argmax": self._pending_argmax,
            "pending_count": int(self._pending_count),
            "recipe_marginals": dict(marginals),
        })

    def _cold_recipe_evidence_ok(self, new_rid: str) -> bool:
        marginals = self.posterior.marginal_recipe()
        p_new = float(marginals.get(new_rid, 0.0))
        second = max((float(p) for rid, p in marginals.items() if rid != new_rid), default=0.0)
        min_conf = max(0.35, float(getattr(self.cfg, "posterior_switch_min_confidence", 0.55)) - 0.10)
        min_gap = float(getattr(self.cfg, "posterior_switch_min_gap", 0.05))
        return p_new >= min_conf and (p_new - second) >= min_gap

    def _fast_switch_recipe_evidence_ok(self, old_rid: str, new_rid: str) -> bool:
        marginals = self.posterior.marginal_recipe()
        p_new = float(marginals.get(new_rid, 0.0))
        p_old = float(marginals.get(old_rid, 0.0))
        second = max((float(p) for rid, p in marginals.items() if rid != new_rid), default=0.0)
        if p_new < float(getattr(self.cfg, "posterior_switch_fast_confidence", 0.95)): return False
        if p_new - second < float(getattr(self.cfg, "posterior_switch_fast_gap", 0.35)): return False
        log_ratio = math.log(max(p_new, 1e-9)) - math.log(max(p_old, 1e-9))
        return log_ratio >= float(getattr(self.cfg, "posterior_switch_fast_margin", 1.25))

    def _switch_recipe_evidence_ok(self, old_rid: str, new_rid: str) -> bool:
        marginals = self.posterior.marginal_recipe()
        p_new = float(marginals.get(new_rid, 0.0))
        p_old = float(marginals.get(old_rid, 0.0))
        if p_new < float(getattr(self.cfg, "posterior_switch_min_confidence", 0.55)): return False
        second = max((float(p) for rid, p in marginals.items() if rid != new_rid), default=0.0)
        if p_new - second < float(getattr(self.cfg, "posterior_switch_min_gap", 0.05)): return False
        log_ratio = math.log(max(p_new, 1e-9)) - math.log(max(p_old, 1e-9))
        return log_ratio >= float(getattr(self.cfg, "posterior_switch_margin", 0.30))

    def evaluate_autonomous_tokens(self, ordering: Sequence[str], topn: int = 3, ground_truth_recipe: Optional[str] = None, ground_truth_pref: Optional[str] = None) -> Dict[str, float]:
        if not self._frozen:
            with self.frozen():
                return self.evaluate_autonomous_tokens(
                    ordering,
                    topn=topn,
                    ground_truth_recipe=ground_truth_recipe,
                    ground_truth_pref=ground_truth_pref,
                )
        tokens = self._coerce_prefix_tokens(ordering)
        total = len(tokens)
        topn = max(1, int(topn))
        if total == 0:
            out = {"top1": 0.0, "topk": 0.0, "prediction_available_rate": 0.0, "empty_prediction_rate": 0.0, "cross_entropy": 0.0}
            out[f"top{topn}"] = 0.0
            return out
        prefix: List[str] = []
        inferred_recipe: Optional[str] = None
        inferred_latent_pref_id: Optional[str] = None
        top1_hits = 0
        topk_hits = 0
        available = 0
        empty_predictions = 0
        nll = 0.0
        recipe_hits = 0
        pref_hits = 0
        floor = max(float(self.cfg.prob_floor), 1e-12)

        for actual in tokens:
            dist = self.predict_next_tokens(prefix)
            ranked = top_k(dist, k=topn) if dist else []
            predicted = ranked[0] if ranked else None
            if dist:
                available += 1
                top1_hits += int(predicted == actual)
                topk_hits += int(actual in ranked)
                nll -= math.log(max(float(dist.get(actual, floor)), floor))
            else:
                empty_predictions += 1
                nll -= math.log(floor)

            prefix.append(actual)
            # Single identity head: refresh the posterior and read its argmaxes. This is the same path predict_next_tokens uses in deployment, so eval and online inference cannot diverge. The deepcopy frozen() contract rolls back any posterior mutation on context exit.
            self._refresh_recipe_inference(prefix)
            inferred_recipe = self.posterior.argmax_recipe()
            inferred_latent_pref_id = self.posterior.argmax_preference()
            if ground_truth_recipe is not None:     recipe_hits += int(inferred_recipe == ground_truth_recipe)
            if ground_truth_pref is not None:       pref_hits += int(inferred_latent_pref_id == ground_truth_pref)

        out = {
            "top1": top1_hits / total,
            "topk": topk_hits / total,
            "prediction_available_rate": available / total,
            "empty_prediction_rate": empty_predictions / total,
            "cross_entropy": nll / total,
        }
        out[f"top{topn}"] = topk_hits / total
        if ground_truth_recipe is not None: out["recipe_accuracy"] = recipe_hits / total
        if ground_truth_pref is not None:   out["preference_accuracy"] = pref_hits / total
        return out

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
        rollouts = int(getattr(self.cfg, "maxent_mc_rollouts", 1))
        return float(n_transitions * n_actions * feature_dim * max(1, iters) * max(1, rollouts))

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
                self._rebuild_active_prototypes(entries)
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
            self._rebuild_active_prototypes(entries)
            self._last_fit_fingerprint = fp
            if self.cfg.verbose:                                self.narrate(f"[step {self.step_counter}] retrained on {len(trajectories)} weighted demos (cycle {self.retrain_cycle}, post_grace_decay_rate={self.decay.post_grace_decay_rate:.6f}, dropped_actions={dropped_total})")

    def refresh_model_from_memory(self) -> None:
        """Re-fit predictors against the current active memory without adding a demo."""
        self._retrain()

    def pruned_influence_audit(self, max_prefixes: int = 24, tolerance: float = 5e-2) -> Dict[str, Any]:
        """Audit the active-only contract.
        Returns two diagnostics:
        - ``active_head_*`` compares the current fitted heads to a fresh fit from active replay entries only.
        - ``live_prediction_*`` verifies that pruned variants cannot change live conditioned frontiers.
        The second diagnostic is the hard behavioral contract for the primary full system. The first is still reported separately because some baseline classes intentionally retain parameter information.
        """
        entries = self.decay.active_entries()
        if not entries:
            return {"max_l1": 0.0,      "mean_l1": 0.0, "active_head_max_l1": 0.0,  "active_head_mean_l1": 0.0, "live_prediction_max_l1": 0.0,  "live_prediction_mean_l1": 0.0, "n_prefixes": 0, "passed": True, "active_head_passed": True, "live_prediction_passed": True, "tolerance": float(tolerance)}
        trajectories, _ = self._build_trajectories(entries)
        weights = self._length_normalized_demo_weights(entries, [float(e.weight) for e in entries])
        # The audit reference uses its own deterministic RNG stream.  This
        # prevents an unrelated number of prior fits from changing the audit
        # result while preserving an independently initialised reference fit.
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
        fresh_markov.fit(trajectories, weights, state_to_idx=fresh_irl.state_to_idx, idx_to_state=fresh_irl.idx_to_state, feature_matrix=fresh_irl.feature_matrix, col_min=fresh_irl.col_min, col_max=fresh_irl.col_max, normalizer=fresh_irl.normalizer)
        prefixes: List[Tuple[str, ...]] = []
        seen: Set[Tuple[str, ...]] = set()
        for e in entries:
            seq = tuple(e.ordering)
            for k in (0, min(len(seq), 1), len(seq) // 2, max(0, len(seq) - 1)):
                pref = tuple(seq[:k])
                if pref not in seen:
                    prefixes.append(pref)
                    seen.add(pref)
                if len(prefixes) >= max(1, int(max_prefixes)):  break
            if len(prefixes) >= max(1, int(max_prefixes)):      break
        head_diffs: List[float] = []
        live_diffs: List[float] = []
        active_keys = self._active_keys()
        active_memory = {rid: {h: v for h, v in slot.items() if (rid, h) in active_keys} for rid, slot in self.memory.variants.items()}
        for pref in prefixes:
            state = self._state_from_prefix(pref)
            cur = ensemble_predict(self.irl.predict(state), self.markov.predict(state, pref), cfg=self.cfg)
            ref = ensemble_predict(fresh_irl.predict(state), fresh_markov.predict(state, pref), cfg=self.cfg)
            keys = set(cur.keys()) | set(ref.keys())
            head_diffs.append(sum(abs(float(cur.get(k, 0.0)) - float(ref.get(k, 0.0))) for k in keys))
            for rid in sorted({e.recipe_id for e in entries}):
                live = self._conditioned_distribution(pref, rid, None)
                # Recompute the same frontier against an active-only view. The production code should already be equivalent; this guard makes that contract explicit and regression-testable.
                old_variants = self.memory.variants
                try:
                    self.memory.variants = active_memory
                    active_only = self._conditioned_distribution(pref, rid, None)
                finally:
                    self.memory.variants = old_variants
                lkeys = set(live.keys()) | set(active_only.keys())
                live_diffs.append(sum(abs(float(live.get(k, 0.0)) - float(active_only.get(k, 0.0))) for k in lkeys))
        active_head_max = max(head_diffs) if head_diffs else 0.0
        active_head_mean = sum(head_diffs) / max(len(head_diffs), 1)
        live_max = max(live_diffs) if live_diffs else 0.0
        live_mean = sum(live_diffs) / max(len(live_diffs), 1)
        max_l1 = max(active_head_max, live_max)
        mean_l1 = max(active_head_mean, live_mean)
        return {
            "max_l1": float(max_l1),
            "mean_l1": float(mean_l1),
            "active_head_max_l1": float(active_head_max),
            "active_head_mean_l1": float(active_head_mean),
            "live_prediction_max_l1": float(live_max),
            "live_prediction_mean_l1": float(live_mean),
            "n_prefixes": len(prefixes),
            "passed": bool(active_head_max <= tolerance and live_max <= tolerance),
            "active_head_passed": bool(active_head_max <= tolerance),
            "live_prediction_passed": bool(live_max <= tolerance),
            "tolerance": float(tolerance),
            "audit_reference_seed": audit_seed,
        }

    # evaluation helpers (for tests) 
    def evaluate_sequence(self, ordering: Sequence[str]) -> float:
        """Prefix-conditioned accuracy of the current ensemble on a reference ordering. Used for the held-out forgetting metric."""
        if not ordering: return 0.0
        return float(self.evaluate_autonomous_tokens(ordering, topn=1)["top1"])
