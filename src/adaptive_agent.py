"""Adaptive HRC agent state machine.

Modes:
    MODE_OBSERVE:   The human is showing a brand-new recipe. The agent buffers actions silently until ``end_demo()`` is called, then classifies the buffer against existing variants. The disambiguator decides whether to create a new recipe or treat the demo as a new preference variant.

    MODE_ONLINE:    Default mode. The agent predicts at every step. If the human action disagrees with top-1, the agent uses partial disambiguation to track the best-matching known variant for session-boundary commit. Prediction remains in the IRL and state-aware n-gram heads. On `end_demo()`, the matched variant is 
    promoted to latest and its rehearsal weight is reset to 1.0. Retraining runs synchronously after each end_demo() or online preference commit. Each ordinary retrain rebuilds predictors from the active replay set so pruned demos cannot survive through warm-started model parameters or feature-normalizer statistics.

Terminology:         "Adaptive rehearsal weighting" operates on per-demo replay weights `w_i`. It does not decay the IRL parameters `theta`. Online fine-tuning is supervised next-action imitation over weighted active rehearsal; observed human actions are the ground-truth labels.
Freeze mode (set_frozen / frozen context manager):      When frozen, all mutating paths are no-ops. Predictions and logging still work so probes produce per-step data. `frozen()` raises AssertionError on exit if any state was accidentally mutated.
"""
from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .environment import StateTracker
from .memory import (Classification,    DecayManager,       Disambiguator,  Variant,        VariantKey,                 VariantMemory,      _aligned_next_index, variant_hash)
from .models import (Config,            DEFAULT_CONFIG,     MaxEntIRL2,     NGramMarkov,       ensemble_predict,   top_k)
from .posterior import (OnlinePreferencePosterior,      PosteriorWeights,   PreferencePrototypeLearner, RecipePrototypeLearner)
from .representations import (ActionObservation,    ActionVector,   PreferenceSignature,    ROLE_UNKNOWN_OR_NOOP,   TaskSignature,  apply_transition_vector,    observations_from_actions,  preference_signature_from_roles,    role_from_transition,   task_signature_from_tokens)

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
    inferred_pref: Optional[str]
    event: str = ""   # free-form tag for the narrator


class AdaptiveHRCAgent:
    """End-to-end continual-learning HRC agent."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, narrate: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.decay = DecayManager(cfg)
        self.memory = VariantMemory(cfg)
        self.disambig = Disambiguator(cfg)
        self.recipe_prototypes = RecipePrototypeLearner()
        self.preference_prototypes = PreferencePrototypeLearner()
        self.task_signatures: Dict[str, TaskSignature] = {}
        self.preference_signatures: Dict[str, PreferenceSignature] = {}
        # Online posterior over (recipe_id, pref_id). Owned by the agent so the freeze snapshot can include it.
        posterior_weights = PosteriorWeights(
            lambda_recipe=getattr(  cfg,    "posterior_lambda_recipe",  1.0),
            lambda_pref=getattr(    cfg,    "posterior_lambda_pref",    1.0),
            lambda_memory=getattr(  cfg,    "posterior_lambda_memory",  0.5),
            tau_recipe=getattr(     cfg,    "posterior_tau_recipe",     1.0),
            tau_pref=getattr(       cfg,    "posterior_tau_pref",       1.0),
            tau_memory=getattr(     cfg,    "posterior_tau_memory",     1.0))
        self.posterior = OnlinePreferencePosterior(weights=posterior_weights)
        # token -> abstract role tag, populated as observations stream through observe_observation. This lets the agent build role sequences for preference prototypes without storing raw action labels there.
        self.token_to_role: Dict[str, str] = {}
        # The latest pref_id assignment from the most recent end_demo update. Used as a default conditioning when the posterior is not yet active.
        self.last_pref_id: Optional[str] = None
        # Active variant -> latent preference prototype assigned during the last active-only rebuild. This is the bridge between concrete preference variants and the posterior's latent pref_ids.
        self.variant_pref_ids: Dict[VariantKey, str] = {}
        self.locked_variant_key: Optional[VariantKey] = None
        self.locked_pref_id: Optional[str] = None

        self.irl = MaxEntIRL2(cfg=cfg)
        self.markov = NGramMarkov(order=cfg.markov_order, prob_floor=cfg.prob_floor)

        self.mode = MODE_ONLINE
        self.pending_demo: List[str] = []
        # Online state
        self.current_prefix: List[str] = []
        self.inferred_recipe: Optional[str] = None
        self.inferred_pref: Optional[str] = None
        self._needs_observation: bool = False
        self._last_assist_gate: Dict[str, Any] = {
            "assist_used": False,
            "action_confidence": None,
            "raw_action_confidence": None,
            "action_gate_score": None,
            "action_margin": None,
            "action_entropy": None,
            "action_gate_threshold": None,
            "assist_source": None,
            "final_action_confidence": None,
            "final_action_margin": None,
            "final_action_entropy": None,
            "conditioned_action_confidence": None,
            "ensemble_action_confidence": None,
            "policy_agreement": None,
            "blend_strength": None,
            "reason": "cold_start",
        }
        # Anonymous action codebook.  The learner sees action vectors; the strings here are stable internal IDs, not symbolic kitchen actions.
        self.action_vector_to_token: Dict[ActionVector, str] = {}
        self.token_to_action_vector: Dict[str, ActionVector] = {}
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
        # Each entry: {"step": int, "dropped_actions": int, "active_demos": int} where dropped_actions is the total count of actions whose preconditions failed during trajectory construction across all demos fit in this retrain (see build_demo_trajectory).
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

        # Fingerprint of the active-set + weights at the last successful fit. When set, _retrain checks whether the current active set + weights are identical and skips the (re)fit if so. The skipped path is provably identical-output to the full path: same demos, same weights, deterministic init -> identical theta. 
        # No leakage: the gate uses ONLY active-set state.
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

    @contextlib.contextmanager
    def frozen(self):
        self.set_frozen(True)
        # Capture a lightweight structural digest for the mutation check.
        _before_active = frozenset(self.decay.active.keys())
        _before_variants = frozenset((rid, h) for rid, slot in self.memory.variants.items() for h in slot)
        try:
            yield self
        finally:
            # Capture BEFORE restore so mutated state is still visible.
            _after_active = frozenset(self.decay.active.keys())
            _after_variants = frozenset((rid, h) for rid, slot in self.memory.variants.items() for h in slot)
            self.set_frozen(False)   # restores snapshot
            if __debug__:
                assert _before_active == _after_active, (
                    f"frozen() invariant violated: decay.active keys changed "
                    f"(added={_after_active - _before_active}, "
                    f"removed={_before_active - _after_active})")
                assert _before_variants == _after_variants, ("frozen() invariant violated: memory.variants keys changed")

    # mode control
    def start_demo(self) -> None:
        """Human signals 'I am about to show a new recipe.'"""
        if self._frozen: return
        self.mode = MODE_OBSERVE
        self.pending_demo = []
        self._needs_observation = False
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
            return Classification("known", None, None, 0.0, 0.0)
        if self.mode == MODE_ONLINE and not self.current_prefix:
            self.current_prefix = []
            self.inferred_recipe = None
            self.inferred_pref = None
            self._needs_observation = False
            self._clear_preference_lock()
            return Classification("known", None, None, 0.0, 0.0)
        if self.mode == MODE_OBSERVE:
            self.session_counter += 1
            return self._end_observe_demo(apply_decay=True)

        commit_cls, reentry_from_pruned = self._classify_online_prefix(self.current_prefix)
        # Diagnostic: check if posterior's live inference agrees with commit-time Jaccard. Log the mismatch rate so it can be tracked in evaluation.
        if (self.inferred_recipe is not None
                and commit_cls.recipe_id is not None
                and self.inferred_recipe != commit_cls.recipe_id):
            self.narrate(f"[step {self.step_counter}] COMMIT MISMATCH: posterior tracked '{self.inferred_recipe}' but commit resolved to '{commit_cls.recipe_id}'")
            # Track as a metric for the paper.
            if not hasattr(self, "_commit_mismatches"): self._commit_mismatches: List[Tuple[int, str, str]] = []
            self._commit_mismatches.append((self.step_counter, self.inferred_recipe, commit_cls.recipe_id))
        if commit_cls.kind == "new_recipe":
            cls = Classification("needs_observation", None, None, commit_cls.jaccard, commit_cls.order_distance)
            self.classification_events.append((self.step_counter, cls))
            self.current_prefix = []
            self.inferred_recipe = None
            self.inferred_pref = None
            self._needs_observation = True
            self._clear_preference_lock()
            self.narrate(f"[step {self.step_counter}] online sequence did not match known recipes; observation mode required")
            return cls

        # Only known recipe/preference sessions advance decay and mutate memory. The currently-demonstrated variant is protected by the latest-pin in `decay.latest_keys` (set by `mark_latest`), so no extra protected_keys plumbing is needed here.
        self.session_counter += 1
        return self._end_online_session(commit_cls, reentry_from_pruned, apply_decay=True)

    def _cache_token_role(self, token: str, observation: ActionObservation) -> None:
        """Cache the abstract role for a token on first sight. Roles are decoded only from observed state deltas (no preference labels). The cache survives across sessions because the action vector -> role mapping is deterministic for this domain."""
        if token in self.token_to_role:     return
        try:                                role = role_from_transition(observation.state, observation.action_vector, observation.next_state)
        except Exception:                   role = ROLE_UNKNOWN_OR_NOOP
        self.token_to_role[token] = role

    def _roles_for_tokens(self, tokens: Sequence[str]) -> List[str]:
        return [self.token_to_role.get(t, ROLE_UNKNOWN_OR_NOOP) for t in tokens]

    def _clear_preference_lock(self) -> None:
        if self._frozen: return
        self.locked_variant_key = None
        self.locked_pref_id = None

    def _register_if_live(self, rid: str, seq: List[str], step: int) -> Variant:
        """Register in memory + decay only when not frozen. Returns the Variant. Also folds the demo into the recipe and latent preference prototypes. Both are read-only structural summaries used by the posterior and the preference-conditioned policy; they are independent of the decay/replay path."""
        v = self.memory.register(rid, seq, step)
        if not self._frozen:
            evicted = getattr(self.memory, "last_evicted_key", None)
            if evicted is not None: self.decay.discard(*evicted)
            self.decay.register(rid, v.pref_hash, tuple(seq), self.session_counter, self.retrain_cycle)
            self._rebuild_active_prototypes(self.decay.active_entries())
        return v

    def _sync_preference_signature(self, pref_id: str, fallback_roles: Optional[Sequence[str]] = None) -> None:
        """Expose an aggregate PreferenceSignature for the latent prototype."""
        proto = self.preference_prototypes.prototypes.get(pref_id)
        if proto is None:
            if fallback_roles is not None:  self.preference_signatures[pref_id] = preference_signature_from_roles(fallback_roles)
            return
        scalar = tuple(sorted((k, float(v)) for k, v in proto.scalar_profile().items()))
        self.preference_signatures[pref_id] = PreferenceSignature( scalar_features=scalar,      role_bigrams=tuple(sorted(proto.bigram_counts.items(), key=lambda kv: (str(kv[0]), kv[1]))),    role_trigrams=tuple(sorted(proto.trigram_counts.items(), key=lambda kv: (str(kv[0]), kv[1]))),)

    def _rebuild_active_prototypes(self, entries: Optional[Sequence[Any]] = None) -> None:
        """Rebuild all learner-facing prototypes from active weighted memory."""
        active_entries = list(self.decay.active_entries() if entries is None else entries)
        recipe_learner = RecipePrototypeLearner()
        pref_learner = PreferencePrototypeLearner()
        tokens_by_recipe: Dict[str, List[str]] = {}
        variant_pref_ids: Dict[VariantKey, str] = {}
        last_pid: Optional[str] = None
        for entry in active_entries:
            seq = list(entry.ordering)
            w = float(entry.weight)
            tokens_by_recipe.setdefault(entry.recipe_id, []).extend(seq)
            if not getattr(self.cfg, "ablation_disable_recipe_prototype", False): recipe_learner.update_from_demo(entry.recipe_id, seq, terminal_state=self._state_from_prefix(seq) if seq else None, weight=w)
            if not getattr(self.cfg, "ablation_disable_preference_head", False):
                last_pid = pref_learner.update_from_roles(self._roles_for_tokens(seq), recipe_id=entry.recipe_id, weight=w)
                variant_pref_ids[entry.key] = last_pid
        self.recipe_prototypes = recipe_learner
        self.preference_prototypes = pref_learner
        self.variant_pref_ids = variant_pref_ids
        if self.locked_variant_key is not None and self.locked_variant_key not in self.variant_pref_ids:
            self._clear_preference_lock()
        self.task_signatures = {rid: task_signature_from_tokens(tokens, self.token_to_action_vector, self.token_to_role) for rid, tokens in tokens_by_recipe.items()}
        self.preference_signatures = {}
        for pid in self.preference_prototypes.all_pref_ids(): self._sync_preference_signature(pid)
        self.last_pref_id = last_pid

    def _rebuild_active_recipe_prototypes(self, entries: Optional[Sequence[Any]] = None) -> None:
        """Backwards-compatible wrapper for callers/tests."""
        self._rebuild_active_prototypes(entries)

    def _end_observe_demo(self, apply_decay: bool = False) -> Classification:
        seq = list(self.pending_demo)
        active_keys = self._active_keys()
        active_lib = self.memory.library(allowed_keys=active_keys)
        cls = self.disambig.classify(seq, active_lib)
        # Pruned-reentry detection: when active-only says new_recipe, check the
        # full library. If a pruned variant matches, restore it rather than
        # creating a new recipe ID.
        if cls.kind == "new_recipe" and active_keys:
            full_lib = self.memory.library()
            if len(full_lib) > len(active_lib):
                archived_cls = self.disambig.classify(seq, full_lib)
                if archived_cls.kind != "new_recipe":
                    cls = archived_cls          # restore under original recipe_id
                    self.narrate(
                        f"[step {self.step_counter}] observe-mode: "
                        f"restoring pruned variant of '{cls.recipe_id}'")

        self.classification_events.append((self.step_counter, cls))
        # Register in memory + decay.
        if cls.kind == "new_recipe":
            rid = f"R{self._next_recipe_idx}"
            self._next_recipe_idx += 1
            self._register_if_live(rid, seq, self.step_counter)
            cls.recipe_id = rid
            self.narrate(f"[step {self.step_counter}] classified NEW RECIPE '{rid}' (jaccard={cls.jaccard:.2f})")
        else:
            if cls.recipe_id is None: raise RuntimeError(f"disambiguator returned {cls.kind} without a recipe_id")
            rid = cls.recipe_id
            self._register_if_live(rid, seq, self.step_counter)
            kind_msg = "KNOWN (re-demo)" if cls.kind == "known" else "PREFERENCE VARIANT"
            self.narrate(f"[step {self.step_counter}] classified {kind_msg} of '{rid}' (jaccard={cls.jaccard:.2f}, tau={cls.order_distance:.2f})")
        self.mode = MODE_ONLINE
        self.pending_demo = []
        if apply_decay: self.decay.step(self.session_counter, self.retrain_cycle)
        self._retrain()
        return cls

    def _classify_online_prefix(self, prefix: Sequence[str]) -> Tuple[Classification, bool]:
        active_keys = self._active_keys()
        active_lib = self.memory.library(allowed_keys=active_keys)
        # For short prefixes, score_partial is more reliable than full Jaccard because Jaccard(short_prefix, full_variant) is always near 0. The full classify() is used only when the prefix is long enough for Jaccard to be meaningful (controlled by min_classify_length in Config).
        min_len = int(getattr(self.cfg, "min_classify_length", 6))
        if len(prefix) < min_len:
            # Use score_partial to get a ranked recipe list but wrap it as a Classification for the rest of the method to consume.
            ranked = self.disambig.score_partial(prefix, active_lib)
            if ranked and ranked[0][1] >= float(getattr(self.cfg, "online_new_recipe_partial_threshold", 0.30)):
                best_v, best_score = ranked[0]
                commit_cls = Classification("preference_shift", best_v.recipe_id, None, best_score, 0.0)
            else: commit_cls = Classification("new_recipe", None, None, 0.0, 0.0)
        else: commit_cls = self.disambig.classify(prefix, active_lib)
        reentry_from_pruned = False
        if prefix:
            h = variant_hash(prefix)
            for rid, slot in self.memory.variants.items():
                if h in slot and (rid, h) not in active_keys: return Classification("known", rid, h, 1.0, 0.0), True
        if commit_cls.kind == "new_recipe":
            # No active match reached threshold. Consult the pruned-variant metadata/full registry so a forgotten known variant can be restored at commit time.
            full_lib = self.memory.library()
            if len(full_lib) > len(active_lib):
                full_cls = self.disambig.classify(prefix, full_lib)
                if full_cls.kind != "new_recipe":
                    commit_cls = full_cls
                    reentry_from_pruned = True
        return commit_cls, reentry_from_pruned

    def _end_online_session(self, commit_cls: Classification, reentry_from_pruned: bool = False, apply_decay: bool = False) -> Classification:
        """End of an ONLINE session: promote the currently inferred variant if it differs from the recipe's latest variant."""
        rid = self.inferred_recipe
        prefix = self.current_prefix
        if not prefix:
            # Nothing to commit.
            self.current_prefix = []
            self.inferred_recipe = None
            self.inferred_pref = None
            self._clear_preference_lock()
            return Classification("known", None, None, 0.0, 0.0)

        if commit_cls.kind == "new_recipe":     raise RuntimeError("online new-recipe commit reached mutating path")
        else:                                   rid = commit_cls.recipe_id

        h = variant_hash(prefix)
        latest = self.memory.latest_variant(rid)
        known_variant = h in self.memory.variants.get(rid, {})
        if known_variant:
            # Known replay: promote it at the session boundary, update timestamps/weights, and retrain.
            v = self._register_if_live(rid, list(prefix), self.step_counter)
            cls = commit_cls or Classification("known", rid, h, 1.0, 0.0)
            if reentry_from_pruned: cls.kind = "reentry_from_pruned"
        else:
            v = self._register_if_live(rid, list(prefix), self.step_counter)
            cls_kind = "preference_shift"
            if reentry_from_pruned: cls_kind = "reentry_from_pruned"
            cls = Classification(cls_kind, rid, v.pref_hash, commit_cls.jaccard, commit_cls.order_distance if commit_cls.kind != "new_recipe" else (0.0 if latest is None else 1.0))
            cls.recipe_id = rid
            cls.pref_hash = v.pref_hash
            self.narrate(f"[step {self.step_counter}] committing preference variant of '{rid}' from online session")
        if apply_decay: self.decay.step(self.session_counter, self.retrain_cycle)
        # Retrain after every completed demo because weights are updated by the decay/register path on every session.
        self._retrain()
        self.classification_events.append((self.step_counter, cls))
        self.current_prefix = []
        self.inferred_recipe = None
        self.inferred_pref = None
        self._needs_observation = False
        self._clear_preference_lock()
        self.posterior.reset()
        return cls

    # interaction
    def _active_keys(self) -> Set[VariantKey]:
        """Current trainable memory keys. Pruned variants are excluded from online prediction."""
        return {e.key for e in self.decay.active_entries()}

    def _token_for_vector(self, vector: ActionVector) -> str:
        token = self.action_vector_to_token.get(vector)
        if token is not None:
            return token
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
        scored: Dict[str, float] = {}
        for token, mass in remaining.items():
            role = roles_by_token.get(token, ROLE_UNKNOWN_OR_NOOP)
            pref_score = self.preference_prototypes.score_action_role(role, prefix_roles, pref_id, candidate_roles)
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

    def _coerce_prefix_tokens(self, prefix: Sequence[str]) -> List[str]:
        if not prefix: return []
        if all(p in self.token_to_action_vector for p in prefix): return list(prefix)
        return self._tokens_from_action_labels(prefix)

    def assist_gate_stats(self) -> Dict[str, Any]:
        return dict(self._last_assist_gate)

    def _set_assist_gate(
        self,
        used: bool,
        confidence: Optional[float],
        entropy: Optional[float],
        reason: str,
        *,
        raw_confidence: Optional[float] = None,
        gate_score: Optional[float] = None,
        margin: Optional[float] = None,
        threshold: Optional[float] = None,
        source: Optional[str] = None,
        final_confidence: Optional[float] = None,
        final_entropy: Optional[float] = None,
        final_margin: Optional[float] = None,
        conditioned_confidence: Optional[float] = None,
        ensemble_confidence: Optional[float] = None,
        agreement: Optional[bool] = None,
        blend_strength: Optional[float] = None,
    ) -> None:
        if self._frozen: return
        score = confidence if gate_score is None else gate_score
        raw = confidence if raw_confidence is None else raw_confidence
        self._last_assist_gate = {
            "assist_used": bool(used),
            "action_confidence": score,
            "raw_action_confidence": raw,
            "action_gate_score": score,
            "action_margin": margin,
            "action_entropy": entropy,
            "action_gate_threshold": threshold,
            "assist_source": source,
            "final_action_confidence": final_confidence,
            "final_action_margin": final_margin,
            "final_action_entropy": final_entropy,
            "conditioned_action_confidence": conditioned_confidence,
            "ensemble_action_confidence": ensemble_confidence,
            "policy_agreement": agreement,
            "blend_strength": blend_strength,
            "reason": reason,
        }

    def _action_confidence_threshold(self) -> float:
        return float(getattr(self.cfg, "posterior_action_confidence_threshold", 0.5))

    def _calibrated_action_gate_score(self, confidence: Optional[float], entropy: Optional[float], margin: Optional[float]) -> float:
        if confidence is None:
            return 0.0
        conf = max(0.0, min(1.0, float(confidence)))
        temp = max(1e-6, float(getattr(self.cfg, "posterior_action_gate_temperature", 1.0)))
        margin_weight = max(0.0, float(getattr(self.cfg, "posterior_action_gate_margin_weight", 0.0)))
        entropy_weight = max(0.0, float(getattr(self.cfg, "posterior_action_gate_entropy_weight", 0.0)))
        calibrated = conf ** temp
        calibrated += margin_weight * max(0.0, min(1.0, float(margin or 0.0)))
        if entropy is not None:
            calibrated += entropy_weight * max(0.0, min(1.0, 1.0 - float(entropy)))
        return max(0.0, min(1.0, float(calibrated)))

    def _distribution_stats(self, dist: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not dist:
            return None, None, None
        vals = sorted((float(v) for v in dist.values()), reverse=True)
        confidence = vals[0] if vals else None
        margin = vals[0] - vals[1] if len(vals) >= 2 else (vals[0] if vals else None)
        entropy = 0.0
        for p in vals:
            if p > 0: entropy -= p * math.log(p)
        if len(vals) > 1: entropy = entropy / math.log(len(vals))
        return confidence, entropy, margin

    def _action_gate_allows(self, confidence: Optional[float], entropy: Optional[float], margin: Optional[float], threshold: Optional[float] = None) -> Tuple[bool, float, str]:
        threshold = self._action_confidence_threshold() if threshold is None else float(threshold)
        score = self._calibrated_action_gate_score(confidence, entropy, margin)
        raw_score = 0.0
        if confidence is not None:
            conf = max(0.0, min(1.0, float(confidence)))
            temp = max(1e-6, float(getattr(self.cfg, "posterior_action_gate_temperature", 1.0)))
            raw_score = conf ** temp
        min_conf = float(getattr(self.cfg, "posterior_action_margin_min_confidence", 0.10))
        margin_thr = float(getattr(self.cfg, "posterior_action_margin_threshold", 0.06))
        entropy_thr = float(getattr(self.cfg, "posterior_action_entropy_threshold", 0.85))
        margin_entropy_signal = (
            confidence is not None
            and margin is not None
            and entropy is not None
            and float(confidence) >= min_conf
            and float(margin) >= margin_thr
            and float(entropy) <= entropy_thr
        )
        if score >= threshold:
            reason = "margin_entropy_gate" if margin_entropy_signal and raw_score < threshold else "action_gate_score"
            return True, score, reason
        return False, score, "low_action_gate_score" if margin_entropy_signal else "low_action_confidence"

    def _top_token(self, dist: Dict[str, float]) -> Optional[str]:
        if not dist: return None
        return max(dist.items(), key=lambda kv: float(kv[1]))[0]

    def _ensure_posterior_for_prefix(self, prefix_tokens: Sequence[str]) -> None:
        if self.posterior.joint(): return
        if not self.recipe_prototypes.all(): return
        self.posterior.update(
            prefix_tokens=prefix_tokens,
            prefix_roles=self._roles_for_tokens(prefix_tokens),
            recipe_protos=self.recipe_prototypes,
            pref_protos=self.preference_prototypes,
            memory_state_for_recipe=self._memory_state_for_recipe,
        )

    def _final_gate_threshold(self, source: str) -> float:
        threshold = self._action_confidence_threshold()
        if source.startswith("ensemble"):
            threshold = max(threshold, float(getattr(self.cfg, "ensemble_fallback_assist_threshold", 0.35)))
        return threshold

    def _final_gate_allows(self, dist: Dict[str, float], source: str) -> Tuple[bool, float, str, Optional[float], Optional[float], Optional[float], float]:
        confidence, entropy, margin = self._distribution_stats(dist)
        threshold = self._final_gate_threshold(source)
        allowed, score, reason = self._action_gate_allows(confidence, entropy, margin, threshold=threshold)
        return allowed, score, reason, confidence, entropy, margin, threshold

    def _prefix_needs_observation(self, prefix_tokens: Sequence[str]) -> bool:
        if self._needs_observation: return True
        min_prefix = int(getattr(self.cfg, "online_new_recipe_min_prefix", 3))
        if len(prefix_tokens) < max(1, min_prefix): return False
        active_keys = self._active_keys()
        active_lib = self.memory.library(allowed_keys=active_keys)
        if not active_lib:
            if not self._frozen:    self._needs_observation = bool(self.memory.library())
            return self._needs_observation
        threshold = float(getattr(self.cfg, "online_new_recipe_partial_threshold", 0.30))
        best_active = max((s for _v, s in self.disambig.score_partial(prefix_tokens, active_lib)), default=0.0)
        active_recipes = {v.recipe_id for v in active_lib}
        best_recipe = max((self.recipe_prototypes.recipe_match(prefix_tokens, rid) for rid in active_recipes), default=0.0)
        if max(best_active, best_recipe) >= threshold: return False
        if not self._frozen:        self._needs_observation = True
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

    def predict_next_tokens(self, prefix: Optional[Sequence[str]] = None, recipe_id: Optional[str] = None, pref_hash: Optional[str] = None) -> Dict[str, float]:
        """Posterior- and preference-conditioned next-action distribution.

        Fusion order:
            1. Compute the unconditioned ensemble distribution (legacy).
            2. If the posterior is empty or action confidence is too low, fall back to the ensemble.
            3. Otherwise, build a posterior-conditioned distribution from the active recipe frontier + preference-head re-weighting under the top latent pref_id, then mix with the ensemble at strength `cfg.posterior_assist_strength`.

        `recipe_id` and `pref_hash` parameters are retained for backwards-compatible call sites; the conditioned path uses the posterior's own marginals, not these arguments. ``pref_hash`` is treated only as an exact-memory frontier handle (deprecation boundary, plan / Component 0); it is not exposed to the preference head.
        """
        with self._profile("predict_next_tokens"):
            prefix_tokens = self._coerce_prefix_tokens(prefix) if prefix is not None else list(self.current_prefix)
            state = self._state_from_prefix(prefix_tokens)
            p_irl = self.irl.predict(state)
            p_mk = self.markov.predict(state, prefix_tokens)
            ensemble_dist = ensemble_predict(p_irl, p_mk, cfg=self.cfg)
            ensemble_confidence, ensemble_entropy, ensemble_margin = self._distribution_stats(ensemble_dist)

            if getattr(self.cfg, "ablation_disable_posterior", False):
                self._set_assist_gate(False, None, None, "posterior_disabled", source="posterior_disabled", final_confidence=ensemble_confidence, final_entropy=ensemble_entropy, final_margin=ensemble_margin, ensemble_confidence=ensemble_confidence)
                return ensemble_dist
            if self._prefix_needs_observation(prefix_tokens):
                self._set_assist_gate(False, None, None, "needs_observation", source="needs_observation", final_confidence=ensemble_confidence, final_entropy=ensemble_entropy, final_margin=ensemble_margin, ensemble_confidence=ensemble_confidence)
                return ensemble_dist
            self._ensure_posterior_for_prefix(prefix_tokens)
            if not self.posterior.joint():
                allowed, score, reason, final_confidence, final_entropy, final_margin, threshold = self._final_gate_allows(ensemble_dist, "ensemble_posterior_empty")
                self._set_assist_gate(
                    allowed,
                    score,
                    final_entropy,
                    reason if allowed else "posterior_empty",
                    raw_confidence=final_confidence,
                    gate_score=score,
                    margin=final_margin,
                    threshold=threshold,
                    source="ensemble_posterior_empty",
                    final_confidence=final_confidence,
                    final_entropy=final_entropy,
                    final_margin=final_margin,
                    ensemble_confidence=ensemble_confidence,
                )
                return ensemble_dist

            conditioned, confidence, entropy, margin = self._action_marginal_distribution(prefix_tokens)
            if not conditioned:
                allowed, score, reason, final_confidence, final_entropy, final_margin, threshold = self._final_gate_allows(ensemble_dist, "ensemble_no_conditioned_frontier")
                self._set_assist_gate(
                    allowed,
                    score,
                    final_entropy,
                    reason if allowed else "no_conditioned_frontier",
                    raw_confidence=final_confidence,
                    gate_score=score,
                    margin=final_margin,
                    threshold=threshold,
                    source="ensemble_no_conditioned_frontier",
                    final_confidence=final_confidence,
                    final_entropy=final_entropy,
                    final_margin=final_margin,
                    conditioned_confidence=confidence,
                    ensemble_confidence=ensemble_confidence,
                )
                return ensemble_dist

            agreement = self._top_token(conditioned) == self._top_token(ensemble_dist)
            blend_strength = self._posterior_blend_strength(conditioned, ensemble_dist, confidence, entropy, margin, ensemble_confidence, ensemble_entropy, ensemble_margin)
            final_dist = self._mix_distributions(conditioned, ensemble_dist, strength=blend_strength)
            allowed, score, reason, final_confidence, final_entropy, final_margin, threshold = self._final_gate_allows(final_dist, "conditioned_blend")
            self._set_assist_gate(
                allowed,
                score,
                final_entropy,
                reason,
                raw_confidence=final_confidence,
                gate_score=score,
                margin=final_margin,
                threshold=threshold,
                source="conditioned_blend",
                final_confidence=final_confidence,
                final_entropy=final_entropy,
                final_margin=final_margin,
                conditioned_confidence=confidence,
                ensemble_confidence=ensemble_confidence,
                agreement=agreement,
                blend_strength=blend_strength,
            )
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
        if recipe_proto is None:
            return 1.0
        same_role = [other for other, other_role in roles_by_token.items() if other != token and other_role == role]
        if not same_role:
            return 1.0
        blockers = sum(float(recipe_proto.precedence_counts.get((other, token), 0.0)) for other in same_role)
        leads = sum(float(recipe_proto.precedence_counts.get((token, other), 0.0)) for other in same_role)
        total = blockers + leads
        if total <= 0.0:
            return 1.0
        order_score = (1.0 + leads) / (2.0 + total)
        return max(0.35, min(1.45, 0.60 + 0.80 * order_score))

    def _conditioned_distribution(self, prefix_tokens: Sequence[str], top_rid: str, pref_id: Optional[str] = None) -> Dict[str, float]:
        """Conditioned distribution under posterior argmax (recipe, pref). Builds the recipe-prototype frontier from active variant orderings only. Pruned variants are intentionally invisible during live assistance; they are consulted only at completed-demo reentry time. Then re-weights candidate tokens using the 
        latent preference head's role-conditioned bigram/trigram score under the posterior's argmax pref_id."""
        active_keys = self._active_keys()
        active_variants = self.memory.variants_of(top_rid, allowed_keys=active_keys)
        weights_map = {entry.key: float(entry.weight) for entry in self.decay.active_entries()}
        variant_orderings: List[Tuple[Sequence[str], float]] = []
        for v in active_variants:
            w = weights_map.get((top_rid, v.pref_hash), 1.0)
            variant_orderings.append((v.ordering, w))
        if not variant_orderings:
            return {}

        pref_proto = self.preference_prototypes.prototypes.get(pref_id) if pref_id is not None else None
        transfer_pair = pref_proto is not None and top_rid not in pref_proto.recipes_seen and pref_id is not None
        # The -recipe_prototype ablation skips the prototype frontier and uses only the exact-memory active variant alignment.
        if getattr(self.cfg, "ablation_disable_recipe_prototype", False):
            frontier = self.recipe_prototypes._align_frontier(prefix_tokens, variant_orderings)
        elif transfer_pair and not getattr(self.cfg, "ablation_disable_preference_head", False):
            frontier = self._preference_ordered_transfer_frontier(prefix_tokens, top_rid, pref_id)
            if frontier: return frontier
        else:
            align_weight = float(getattr(self.cfg, "recipe_frontier_align_weight", 0.45))
            if transfer_pair: align_weight = min(align_weight, float(getattr(self.cfg, "recipe_frontier_transfer_align_weight", 0.0)))
            frontier = self.recipe_prototypes.frontier(prefix_tokens, top_rid, variant_orderings, align_weight=align_weight)
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
        rescored: Dict[str, float] = {}
        for token, p in frontier.items():
            role = roles_by_token.get(token, ROLE_UNKNOWN_OR_NOOP)
            pref_score = self.preference_prototypes.score_action_role(role, prefix_roles, top_pid, candidate_roles)
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
        rid, pref_hash_ = key
        variant = self.memory.variants.get(rid, {}).get(pref_hash_)
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

    def _posterior_blend_strength(
        self,
        conditioned: Dict[str, float],
        ensemble: Dict[str, float],
        conditioned_confidence: Optional[float] = None,
        conditioned_entropy: Optional[float] = None,
        conditioned_margin: Optional[float] = None,
        ensemble_confidence: Optional[float] = None,
        ensemble_entropy: Optional[float] = None,
        ensemble_margin: Optional[float] = None,
    ) -> float:
        base = float(getattr(self.cfg, "posterior_assist_strength", 0.70))
        lo = float(getattr(self.cfg, "posterior_assist_strength_min", 0.15))
        hi = float(getattr(self.cfg, "posterior_assist_strength_max", 0.90))
        if hi < lo: lo, hi = hi, lo
        strength = base
        if conditioned and ensemble and self._top_token(conditioned) == self._top_token(ensemble):
            strength += float(getattr(self.cfg, "posterior_assist_agreement_bonus", 0.15))
        elif conditioned and ensemble:
            c = float(conditioned_confidence or 0.0)
            e = float(ensemble_confidence or 0.0)
            strength -= float(getattr(self.cfg, "posterior_assist_disagreement_penalty", 0.45)) * max(0.0, e - c)
            strength += 0.15 * max(0.0, c - e)
            if conditioned_margin is not None and ensemble_margin is not None:
                strength += 0.10 * max(0.0, float(conditioned_margin) - float(ensemble_margin))
            if conditioned_entropy is not None and ensemble_entropy is not None:
                strength += 0.10 * max(0.0, float(ensemble_entropy) - float(conditioned_entropy))
        return max(0.0, min(1.0, max(lo, min(hi, strength))))

    def _mix_distributions(self, conditioned: Dict[str, float], ensemble: Dict[str, float], strength: Optional[float] = None) -> Dict[str, float]:
        """Convex-combine conditioned and ensemble distributions."""
        if strength is None:
            strength = self._posterior_blend_strength(conditioned, ensemble)
        strength = max(0.0, min(1.0, strength))
        out: Dict[str, float] = {}
        keys = set(conditioned.keys()) | set(ensemble.keys())
        for tok in keys: out[tok] = strength * conditioned.get(tok, 0.0) + (1.0 - strength) * ensemble.get(tok, 0.0)
        z = sum(out.values())
        if z <= 0:  return ensemble
        return {t: v / z for t, v in out.items()}

    def predict_next(self, prefix: Optional[Sequence[str]] = None) -> Dict[str, float]:
        return self._decode_distribution(self.predict_next_tokens(prefix))

    def observe_observation(self, observation: ActionObservation, ground_truth_recipe: Optional[str] = None) -> StepResult:
        """Consume one state/action-vector observation. `ground_truth_recipe` is optional and used only for metrics.  The behavior policy only sees the anonymous action token derived from the binary transition vector. """
        if self._frozen:
            action = self.action_vector_to_token.get(observation.action_vector, "act_unseen")
            prefix = list(self.current_prefix)
            dist = self.predict_next_tokens(prefix)
            predicted_token = top_k(dist, k=1)[0] if dist else None
            return StepResult(step=self.step_counter, predicted=self._display_token(predicted_token), actual=action, correct=predicted_token == action, mode=self.mode, inferred_recipe=self.inferred_recipe, inferred_pref=self.inferred_pref, event="frozen_prediction",)
        self.step_counter += 1
        # NOTE: decay advances once per session (see end_demo), not per action; otherwise weights collapse within one demo.

        if self.mode == MODE_OBSERVE:
            action = self._token_for_vector(observation.action_vector)
            self._cache_token_role(action, observation)
            self.pending_demo.append(action)
            result = StepResult(step=self.step_counter, predicted=None, actual=self._display_token(action) or action, correct=False, mode=self.mode, inferred_recipe=None, inferred_pref=None, event="observing")
            self.step_log.append(result)
            return result
        # ONLINE mode.
        dist = self.predict_next_tokens(self.current_prefix)
        predicted_token = top_k(dist, k=1)[0] if dist else None
        action = self.action_vector_to_token.get(observation.action_vector)
        if action is None: action = self._token_for_vector(observation.action_vector)
        self._cache_token_role(action, observation)
        correct = predicted_token == action
        # Commit the action to prefix.
        self.current_prefix.append(action)
        # Re-evaluate recipe identity each step. Evidence accumulates with the prefix, so a wrong first-action commit can be corrected when later actions make another recipe clearly more likely.
        self._refresh_recipe_inference(self.current_prefix)

        event = ""
        if not correct and self.inferred_recipe is not None:
            # Possibly a preference shift: rank variants by prefix match.
            variants = self.memory.variants_of(self.inferred_recipe, allowed_keys=self._active_keys())
            ranked = self.disambig.score_partial(self.current_prefix, variants)
            if ranked:
                top_variant, _ = ranked[0]
                if top_variant.pref_hash != self.inferred_pref:
                    # Do not mutate inferred_pref when frozen; probes should not alter persistent inference state across calls.
                    if not self._frozen:
                        self.inferred_pref = top_variant.pref_hash
                        self.locked_variant_key = (top_variant.recipe_id, top_variant.pref_hash)
                        self.locked_pref_id = self.variant_pref_ids.get(self.locked_variant_key)
                    event = "preference_shift_locked"
                    self.narrate(f"[step {self.step_counter}] MISMATCH (predicted '{self._display_token(predicted_token)}', got '{self._display_token(action)}') -> locking onto variant {top_variant.pref_hash[:24]}...")

        result = StepResult(step=self.step_counter, predicted=self._display_token(predicted_token), actual=self._display_token(action) or action, correct=correct, mode=self.mode, inferred_recipe=self.inferred_recipe, inferred_pref=self.inferred_pref, event=event)
        self.step_log.append(result)
        self.accuracy_events.append((self.step_counter, ground_truth_recipe or (self.inferred_recipe or "?"), correct))
        return result

    # recipe identification (online, running posterior)
    def _memory_state_for_recipe(self, rid: str) -> str:
        """active | pruned | absent for live posterior use. The posterior consumes this without ever seeing pref_hash directly (the deprecation boundary in plan / Component 0)."""
        active_recipes = {r for (r, _h) in self.decay.active.keys()}
        if rid in active_recipes:
            return "active"
        pruned_recipes = {r for (r, _h) in self.decay.pruned.keys()}
        if rid in pruned_recipes and not getattr(self.cfg, "ablation_disable_pruned_memory_prior", False):
            return "pruned"
        return "absent"

    def _refresh_recipe_inference(self, prefix: Sequence[str]) -> None:
        """Posterior-driven recipe inference. Single online identity head. The posterior owns the running belief. The disambiguator is no longer consulted on the per-step path; it runs only at observation-mode entry (`_end_observe_demo` / `_classify_online_prefix`). The `ablation_disable_posterior` flag now means: 
        do not commit any recipe identity, leaving `inferred_recipe` at its previous value (None until the agent has seen a clean exact-variant match)."""
        if getattr(self.cfg, "ablation_disable_posterior", False): return
        prefix_tokens = self._coerce_prefix_tokens(prefix)
        prefix_roles = self._roles_for_tokens(prefix_tokens)
        self.posterior.update(prefix_tokens=prefix_tokens, prefix_roles=prefix_roles, recipe_protos=self.recipe_prototypes, pref_protos=self.preference_prototypes, memory_state_for_recipe=self._memory_state_for_recipe)
        self._prefix_needs_observation(prefix_tokens)
        new_rid = self.posterior.argmax_recipe()
        if self._frozen: return
        old_rid = self.inferred_recipe

        # Cold commit: no current identity, accept the first non-None argmax.
        if old_rid is None:
            if new_rid is not None:
                self.inferred_recipe = new_rid
                self.inferred_pref = None
                if self.locked_variant_key is not None and self.locked_variant_key[0] != new_rid:
                    self._clear_preference_lock()
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
        margin = self.posterior.marginal_recipe()
        log_p_new = math.log(max(margin.get(new_rid, 1e-9), 1e-9))
        log_p_cur = math.log(max(margin.get(old_rid, 1e-9), 1e-9))
        if log_p_new - log_p_cur < float(getattr(self.cfg, "posterior_switch_margin", 0.30)): return
        if new_rid != self._pending_argmax:
            self._pending_argmax = new_rid
            self._pending_count = 1
            return
        self._pending_count += 1
        if self._pending_count < int(getattr(self.cfg, "posterior_switch_agreement", 2)): return

        # Commit the switch.
        self.inferred_recipe = new_rid
        self.inferred_pref = None
        if self.locked_variant_key is not None and self.locked_variant_key[0] != new_rid:
            self._clear_preference_lock()
        self._pending_argmax = None
        self._pending_count = 0
        self.narrate(f"[step {self.step_counter}] posterior switched recipe '{old_rid}' -> '{new_rid}' (confidence={self.posterior.confidence():.2f})")

    def evaluate_autonomous_tokens(self, ordering: Sequence[str], topn: int = 3, ground_truth_recipe: Optional[str] = None, ground_truth_pref: Optional[str] = None) -> Dict[str, float]:
        tokens = self._coerce_prefix_tokens(ordering)
        total = len(tokens)
        if total == 0: return {"top1": 0.0, "top3": 0.0, "coverage": 0.0, "abstain_rate": 0.0, "cross_entropy": 0.0}
        prefix: List[str] = []
        inferred_recipe: Optional[str] = None
        inferred_pref: Optional[str] = None
        top1_hits = 0
        topk_hits = 0
        covered = 0
        abstentions = 0
        nll = 0.0
        recipe_hits = 0
        pref_hits = 0
        floor = max(float(self.cfg.prob_floor), 1e-12)

        for actual in tokens:
            dist = self.predict_next_tokens(prefix, recipe_id=inferred_recipe, pref_hash=inferred_pref)
            ranked = top_k(dist, k=topn) if dist else []
            predicted = ranked[0] if ranked else None
            if dist:
                covered += 1
                top1_hits += int(predicted == actual)
                topk_hits += int(actual in ranked)
                nll -= math.log(max(float(dist.get(actual, floor)), floor))
            else:
                abstentions += 1
                nll -= math.log(floor)

            prefix.append(actual)
            # Single identity head: refresh the posterior and read its argmaxes. This is the same path predict_next_tokens uses in deployment, so eval and online inference cannot diverge. The deepcopy frozen() contract rolls back any posterior mutation on context exit.
            self._refresh_recipe_inference(prefix)
            inferred_recipe = self.posterior.argmax_recipe()
            inferred_pref = self.posterior.argmax_preference()
            if ground_truth_recipe is not None:     recipe_hits += int(inferred_recipe == ground_truth_recipe)
            if ground_truth_pref is not None:       pref_hits += int(inferred_pref == ground_truth_pref)

        out = {"top1": top1_hits / total,   "top3": topk_hits / total,  "coverage": covered / total,    "abstain_rate": abstentions / total,    "cross_entropy": nll / total}
        if ground_truth_recipe is not None: out["recipe_accuracy"] = recipe_hits / total
        if ground_truth_pref is not None:   out["preference_accuracy"] = pref_hits / total
        return out

    def _build_trajectories(self, demos):
        """Convert each demo into a trajectory in O(L) per demo. The previous implementation called `_state_from_prefix(prefix[:k])` for each position k in 0..L, replaying the StateTracker from scratch every time (O(L^2) per demo). Equivalent to walking each demo once with an incremental
        state update, which is what we do here. Strictly lossless: same (state, action) pairs in the same order, identical numeric values."""
        trajs = []
        dropped_total = 0
        for demo in demos:
            tokens = self._coerce_prefix_tokens(demo)
            traj = []
            tracker = StateTracker()
            state = tuple(tracker.get_state_vector().astype(int).tolist())
            for token in tokens:
                traj.append((state, token))
                vector = self._vector_for_token(token)
                if vector is not None:
                    new_state = apply_transition_vector(state, vector)
                    if new_state == state:
                        dropped_total += 1   # no-op transition = failed precondition
                    state = new_state
            traj.append((state, "stop"))
            trajs.append(traj)
        return trajs, dropped_total

    def _record_retrain_event(
        self,
        dropped_actions: int,
        active_demos: int,
        *,
        total_wall_s: float = 0.0,
        build_wall_s: float = 0.0,
        fit_wall_s: float = 0.0,
        flop_estimate: float = 0.0,
        skipped: bool = False,
    ) -> None:
        """Append a structured retrain event. All agent variants funnel through this."""
        fit_stats = self._latest_fit_stats() if not skipped else {}
        if not skipped and float(flop_estimate) <= 0.0:
            estimate = fit_stats.get("estimated_flops") if isinstance(fit_stats, dict) else None
            if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)):
                flop_estimate = float(estimate)
        event = {
            "step": self.step_counter,
            "cycle": int(self.retrain_cycle),
            "dropped_actions": int(dropped_actions),
            "active_demos": int(active_demos),
            "total_wall_s": float(total_wall_s),
            "build_wall_s": float(build_wall_s),
            "fit_wall_s": float(fit_wall_s),
            "flop_estimate": float(flop_estimate),
            "skipped": bool(skipped),
        }
        if fit_stats:
            event["fit_stats"] = fit_stats
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
        if isinstance(irl_stats, dict) and irl_stats:
            stats.update(irl_stats)
        bc_stats = getattr(getattr(self, "bc", None), "last_fit_stats", None)
        if isinstance(bc_stats, dict) and bc_stats:
            stats.update(bc_stats)
        custom_stats = getattr(self, "_custom_fit_stats", None)
        if isinstance(custom_stats, dict) and custom_stats:
            stats.update(custom_stats)
        return stats

    def _estimate_retrain_flops(self, trajectories: Sequence[List[Tuple[Tuple[int, ...], str]]]) -> float:
        """Estimate fit FLOPs from the fitted head's actual accounting."""
        stats = self._latest_fit_stats()
        estimate = stats.get("estimated_flops") if isinstance(stats, dict) else None
        if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)):
            if stats.get("model_family") != "maxent_irl":
                return float(estimate)
            n_transitions = sum(max(0, len(traj) - 1) for traj in trajectories)
            count_head_flops = float(n_transitions * max(1, int(getattr(self.cfg, "markov_order", 1)) + 2))
            return float(estimate) + count_head_flops
        # Legacy fallback for externally supplied heads that do not expose
        # last_fit_stats. Current MaxEntIRL2 fits should not take this branch.
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
        self.markov = NGramMarkov(order=self.cfg.markov_order, prob_floor=self.cfg.prob_floor)

    def _prepare_retrain_fit(self) -> None:
        """Drop fitted predictor state before fitting the current active set."""
        self._reset_heads()

    def _fit_fingerprint(self) -> Optional[frozenset]:
        """Identity of the (active set, weights) tuple. None when the active set is empty. Used by `_retrain` to gate redundant fits. Quantises weights to 6 decimals so floating-point drift doesn't defeat the gate. Read-only on agent state."""
        if not self.decay.active: return None
        return frozenset((k, round(float(e.weight), 6)) for k, e in self.decay.active.items())

    def _retrain(self) -> None:
        if self._frozen:
            return
        retrain_t0 = time.perf_counter()
        fp = self._fit_fingerprint()
        if fp is not None and fp == self._last_fit_fingerprint:
            # Active set + weights identical to last successful fit. Same inputs -> identical theta (deterministic IRL) -> no need to refit. The previously fit predictors remain valid. Lossless and leakage-free.
            with self._profile("retrain_skipped"):
                self.retrain_cycle += 1
                self.retrain_skipped_count += 1
                self._record_retrain_event(
                    dropped_actions=0,
                    active_demos=len(self.decay.active),
                    total_wall_s=time.perf_counter() - retrain_t0,
                    skipped=True,
                )
            return
        with self._profile("retrain_total"):
            self.retrain_cycle += 1
            entries = self.decay.active_entries()
            if not entries:
                self._record_retrain_event(dropped_actions=0, active_demos=0, total_wall_s=time.perf_counter() - retrain_t0, skipped=True)
                self._reset_heads()
                self._rebuild_active_recipe_prototypes(entries)
                self._last_fit_fingerprint = None
                if self.cfg.verbose: self.narrate(f"[step {self.step_counter}] cleared predictors because active memory is empty (cycle {self.retrain_cycle})")
                return
            demos = [list(e.ordering) for e in entries]
            weights = [e.weight for e in entries]
            build_t0 = time.perf_counter()
            with self._profile("retrain_build_trajectories"):   trajectories, dropped_total = self._build_trajectories(demos)
            build_wall_s = time.perf_counter() - build_t0
            self._prepare_retrain_fit()
            fit_t0 = time.perf_counter()
            with self._profile("retrain_fit_heads"):            self._fit_heads(trajectories, weights)
            fit_wall_s = time.perf_counter() - fit_t0
            self._record_retrain_event(
                dropped_actions=dropped_total,
                active_demos=len(demos),
                total_wall_s=time.perf_counter() - retrain_t0,
                build_wall_s=build_wall_s,
                fit_wall_s=fit_wall_s,
                flop_estimate=self._estimate_retrain_flops(trajectories),
            )
            self._rebuild_active_recipe_prototypes(entries)
            self._last_fit_fingerprint = fp
            if self.cfg.verbose:                                self.narrate(f"[step {self.step_counter}] retrained on {len(trajectories)} weighted demos (cycle {self.retrain_cycle}, global_rate={self.decay.global_rate:.6f}, dropped_actions={dropped_total})")

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
        demos = [list(e.ordering) for e in entries]
        weights = [float(e.weight) for e in entries]
        trajectories, _ = self._build_trajectories(demos)
        fresh_irl = MaxEntIRL2(cfg=self.cfg)
        fresh_markov = NGramMarkov(order=self.cfg.markov_order, prob_floor=self.cfg.prob_floor)
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
        }

    # evaluation helpers (for tests) 
    def evaluate_sequence(self, ordering: Sequence[str]) -> float:
        """Prefix-conditioned accuracy of the current ensemble on a reference ordering. Used for the held-out forgetting metric."""
        if not ordering: return 0.0
        return float(self.evaluate_autonomous_tokens(ordering, topn=1)["top1"])
