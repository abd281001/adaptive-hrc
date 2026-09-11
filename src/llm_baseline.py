#!/usr/bin/env python3
"""Frozen in-context LLM baseline for personalized next-action prediction."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from packaging.version import InvalidVersion, Version

from .adaptive_agent import (
    AdaptiveAgent,
    BASELINE_TRAIN_POLICY,
    TrainPolicy,
)
from .environment import _FEAT
from .memory import MemoryItem
from .models import DEFAULT_SETTINGS, Settings


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
DEFAULT_CACHE_ROOT = ROOT / ".cache" / "huggingface" / "hub"
PREDICTOR_NAME = "in_context_llm"
QUALIFIED_PYTHON = (3, 12)
QUALIFIED_TORCH = "2.13.0"
QUALIFIED_TRANSFORMERS = "5.16.1"
QUALIFIED_ACCELERATE = "1.14.0"
QUALIFIED_BITSANDBYTES = "0.49.2"
LLM_EVALUATION_SEED = 1337


# The task framing is shared so the two context encodings cannot drift apart:
# only the sentences describing the demonstration format differ between them.
_TASK_INSTRUCTIONS = """You are a next-action classifier for human-robot collaboration.
Infer the demonstrated person's workflow from the previous_demonstrations in
CONTEXT and the actions already observed in the current episode. Predict the
exact next action that person would perform. Do not optimize a goal, invent an
alternative plan, or use actions outside CANDIDATE_ACTIONS.

CONTEXT gives each demonstration's memory_weight, its retention weight in the
robot's replay memory: higher means more strongly retained."""

_ANSWER_INSTRUCTION = "Return exactly one complete semantic action from CANDIDATE_ACTIONS."

PROMPT_INSTRUCTIONS = "\n".join((
    _TASK_INSTRUCTIONS,
    """A demonstration's steps are listed in order. Each step gives the action
taken and, under state_delta, the predicates that became true ("+") and false
("-") as a result. initial_state_predicates holds the state each demonstration
starts from, so the state before any step is that start state updated by the
preceding steps' changes.""",
    "",
    _ANSWER_INSTRUCTION,
    "",
))


ACTION_ONLY_INSTRUCTIONS = "\n".join((
    _TASK_INSTRUCTIONS,
    "A demonstration's actions are listed in order.",
    "",
    _ANSWER_INSTRUCTION,
    "",
))


StateVector = Tuple[int, ...]
PromptStep = Tuple[StateVector, str]


@dataclass(frozen=True)
class PromptDemo:
    """One retained state-action demonstration supplied to the frozen LLM.

    ``weight`` is the demonstration's replay retention weight, not the
    length-equalized per-example weight the numeric predictors fit against: the
    prompt carries one number per demonstration, so dividing by episode length
    would make a longer demonstration read as less strongly retained.

    ``final_state`` is the state reached after the last step. It is what makes
    the last step's effect renderable, so it is required for a demonstration
    that is going to be rendered into a context prompt.
    """

    weight: float
    steps: Tuple[PromptStep, ...]
    final_state: Optional[StateVector] = None


@dataclass(frozen=True)
class ScoreResult:
    """Candidate probabilities and auditable inference accounting."""

    probabilities: Mapping[str, float]
    prompt_tokens: int
    candidate_tokens: int
    wall_s: float
    prompt_hash: str
    context_limit: int
    cache_hit: bool = False
    scoring_method: str = "unspecified"
    model_forwards: int = 0
    # ``wall_s`` is time this call actually spent, so it is zero on a cache hit.
    # Summing it therefore reports measured GPU time and understates the cost of
    # scoring every decision independently. ``uncached_wall_s`` carries the time
    # the first identical prompt took, so both totals stay recoverable.
    uncached_wall_s: float = 0.0


class PromptTooLongError(RuntimeError):
    """The assembled prompt does not fit the available prompt budget.

    The budget is the smaller of the model's context window and what VRAM is
    left after the weights and the display reservation, so this is raised both
    for a genuinely over-long prompt and for one the GPU cannot currently hold.

    Typed so the agent can shed the prompt's optional state annotation and
    retry, rather than dropping demonstrations or ending the run. Carries the
    measured token counts so the retry decision stays auditable.
    """

    def __init__(self, required_tokens: int, context_limit: int) -> None:
        super().__init__(
            "the complete active replay and semantic-action prompt exceeds the "
            f"available prompt budget: required={required_tokens}, "
            f"limit={context_limit}. No demonstration was silently dropped. "
            "The budget is min(llm_context_tokens, model window, VRAM left "
            "after weights and llm_vram_headroom_gib), so raise "
            "llm_context_tokens, free VRAM by closing other GPU clients or "
            "using a GPU that is not driving a display, or reduce the "
            "experimental memory scope."
        )
        self.required_tokens = int(required_tokens)
        self.context_limit = int(context_limit)


class ActionScorer(Protocol):
    """Minimal interface used by the agent and lightweight test doubles."""

    def score(
        self,
        context_prompt: str,
        query_prompt: str,
        candidates: Sequence[str],
    ) -> ScoreResult:
        ...


def predicate_names(domain: Optional[Any] = None) -> Tuple[str, ...]:
    """Return the state bit-vector's predicate names for the active domain.

    Every other arm reads state through ``DomainAdapter.build_features``, so
    this arm asks the adapter for its predicate names rather than importing the
    symbolic feature map directly.  The hook is optional so the existing
    adapters keep conforming to ``DomainAdapter`` unchanged; the symbolic
    default is used when no domain is supplied.
    """
    if domain is None:
        return tuple(_FEAT)
    names = getattr(domain, "predicate_names", None)
    if names is None:
        raise ValueError(
            f"domain {getattr(domain, 'name', type(domain).__name__)!r} does not "
            "expose predicate_names(); the in-context LLM baseline renders states "
            "as named predicates and cannot describe this domain's state vector"
        )
    return tuple(str(name) for name in names())


def state_predicates(
    state: Sequence[int],
    domain: Optional[Any] = None,
) -> List[str]:
    """Render a binary state without adding recipe or preference metadata."""
    names = predicate_names(domain)
    if len(state) != len(names):
        raise ValueError(
            f"state width {len(state)} does not match domain width {len(names)}"
        )
    return [
        name
        for name, value in zip(names, state)
        if int(value) == 1
    ]


def state_delta(
    before: Sequence[int],
    after: Sequence[int],
    domain: Optional[Any] = None,
) -> Dict[str, List[str]]:
    """Render one action's effect as the predicates it flipped.

    The ``+``/``-`` keys and the omission of empty sides are a prompt-budget
    choice: this renders once per demonstrated step, so key text is paid for a
    few hundred times per prompt while a typical action flips one or two
    predicates.
    """
    names = predicate_names(domain)
    if len(before) != len(names) or len(after) != len(names):
        raise ValueError(
            f"state widths {len(before)}/{len(after)} do not match domain width "
            f"{len(names)}"
        )
    became_true: List[str] = []
    became_false: List[str] = []
    for name, old, new in zip(names, before, after):
        if int(old) == int(new):
            continue
        (became_true if int(new) == 1 else became_false).append(name)
    delta: Dict[str, List[str]] = {}
    if became_true:
        delta["+"] = became_true
    if became_false:
        delta["-"] = became_false
    return delta


def make_prompt_demos(
    trajectories: Sequence[Sequence[Tuple[StateVector, str]]],
    weights: Sequence[float],
) -> Tuple[PromptDemo, ...]:
    """Convert the active replay fit input into deterministic prompt examples.

    ``weights`` must be the demonstrations' replay retention weights. The
    length-equalized weights that ``_fit_predictors`` receives are per-example
    sample weights for a numeric fit and do not describe a whole demonstration.
    """
    if len(trajectories) != len(weights):
        raise ValueError("demonstrations and replay weights must have equal length")
    demonstrations: List[PromptDemo] = []
    for trajectory, weight in zip(trajectories, weights):
        steps = tuple(
            (tuple(int(value) for value in state), str(action))
            for state, action in trajectory
            if action != "stop"
        )
        # ``_build_demos`` terminates every trajectory with the reached state
        # under a "stop" token. That state is the last action's effect, so it is
        # kept here rather than discarded with the token.
        terminal = [
            tuple(int(value) for value in state)
            for state, action in trajectory
            if action == "stop"
        ]
        if steps and float(weight) > 0.0:
            demonstrations.append(PromptDemo(
                float(weight),
                steps,
                terminal[-1] if terminal else None,
            ))
    return tuple(demonstrations)


def candidate_actions(demonstrations: Sequence[PromptDemo]) -> Tuple[str, ...]:
    """Use exactly the semantic action support present in active replay."""
    return tuple(sorted({
        action
        for demonstration in demonstrations
        for _state, action in demonstration.steps
    }))


def _demonstration_steps(
    demonstration: PromptDemo,
    domain: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Render one demonstration's actions with the state change each caused.

    Every other arm trains on the ``(state, action)`` pairs ``_build_demos``
    produces, so the state information reaching this arm has to be the same.
    Deltas rather than repeated full predicate lists keep that information
    complete -- the state before any step is the demonstration's start state
    updated by the preceding deltas -- at a fraction of the prompt budget.
    """
    if demonstration.final_state is None:
        raise ValueError(
            "a rendered demonstration requires its final state; the last "
            "action's effect would otherwise be silently omitted"
        )
    successors = [state for state, _action in demonstration.steps[1:]]
    successors.append(tuple(demonstration.final_state))
    return [
        {"action": action, "state_delta": state_delta(before, after, domain)}
        for (before, action), after in zip(demonstration.steps, successors)
    ]


def _demonstration_payload(
    demonstration: PromptDemo,
    reference_start: StateVector,
    domain: Optional[Any],
    *,
    include_state_deltas: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "memory_weight": round(float(demonstration.weight), 8),
    }
    if include_state_deltas:
        payload["steps"] = _demonstration_steps(demonstration, domain)
    else:
        payload["actions"] = [action for _state, action in demonstration.steps]
    start = demonstration.steps[0][0]
    if include_state_deltas and tuple(start) != tuple(reference_start):
        # Demonstrations share a start state in the existing scenarios. Naming
        # the difference keeps a domain that does not a rendering concern rather
        # than a run-ending one.
        payload["start_state_delta"] = state_delta(reference_start, start, domain)
    return payload


def build_context_prompt(
    demonstrations: Sequence[PromptDemo],
    candidates: Sequence[str],
    domain: Optional[Any] = None,
    *,
    include_state_deltas: bool = True,
) -> str:
    """Build the static weighted demonstration context shared across turns.

    ``include_state_deltas=False`` drops the per-step state annotation. It is
    the budget release valve for a memory too large to annotate: the retained
    demonstrations and their weights stay complete, so memory coverage is
    unaffected and only the state information degrades. Which form was sent is
    recorded per turn.
    """
    if not demonstrations:
        raise ValueError("an in-context prompt requires retained demonstrations")
    supported_actions = candidate_actions(demonstrations)
    if tuple(candidates) != supported_actions:
        raise ValueError("candidate actions must equal active demonstration support")

    reference_start = demonstrations[0].steps[0][0]
    payload: Dict[str, Any] = {
        "previous_demonstrations": [
            _demonstration_payload(
                demonstration,
                reference_start,
                domain,
                include_state_deltas=include_state_deltas,
            )
            for demonstration in demonstrations
        ],
    }
    if include_state_deltas:
        payload["initial_state_predicates"] = state_predicates(
            reference_start, domain,
        )
    return "\n".join((
        PROMPT_INSTRUCTIONS if include_state_deltas else ACTION_ONLY_INSTRUCTIONS,
        "CONTEXT:",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    ))


def build_query_prompt(
    state: StateVector,
    prefix: Sequence[str],
    domain: Optional[Any] = None,
) -> str:
    """Build the dynamic query without task identity or future actions."""
    payload = {
        "current_state_predicates": state_predicates(state, domain),
        "observed_actions": [str(action) for action in prefix],
    }
    return "\n".join((
        "QUERY:",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    ))


def build_candidate_prompt(
    query_prompt: str,
    candidates: Sequence[str],
) -> str:
    """Close the query with the scored candidates and the answer boundary.

    The candidate list has to precede the answer boundary: the model is asked
    to choose from CANDIDATE_ACTIONS, so emitting the boundary first leaves it
    stranded ahead of the options it constrains.
    """
    return "\n".join((
        query_prompt,
        "CANDIDATE_ACTIONS:",
        json.dumps(list(candidates), separators=(",", ":")),
        "Return exactly one complete semantic action from CANDIDATE_ACTIONS.",
        "ANSWER:",
    ))


def build_prompt(
    demonstrations: Sequence[PromptDemo],
    state: StateVector,
    prefix: Sequence[str],
    candidates: Sequence[str],
    domain: Optional[Any] = None,
) -> str:
    """Render the exact scored prompt for audits and human inspection.

    ``candidates`` are the state-conditioned actions actually scored, which is
    a subset of the active demonstration support the context is built from.
    Rendering the full support here instead is what previously made this
    function disagree with the prompt the model was shown.
    """
    return "\n".join((
        build_context_prompt(
            demonstrations, candidate_actions(demonstrations), domain,
        ),
        build_candidate_prompt(
            build_query_prompt(state, prefix, domain), candidates,
        ),
    ))


def _cache_roots() -> Tuple[Path, ...]:
    """Return configured and conventional Hugging Face Hub cache roots."""
    candidates: List[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        candidates.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("TRANSFORMERS_CACHE"):
        candidates.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub")
    candidates.append(DEFAULT_CACHE_ROOT)
    if os.environ.get("XDG_CACHE_HOME"):
        candidates.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    roots: List[Path] = []
    seen = set()
    for candidate in candidates:
        expanded = candidate.expanduser()
        key = str(expanded.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            roots.append(expanded)
    return tuple(roots)


def _cached_model_path(model_id: str = MODEL_ID) -> Optional[Path]:
    model_cache_name = "models--" + str(model_id).replace("/", "--")
    snapshots: List[Path] = []
    for cache_root in _cache_roots():
        snapshot_root = cache_root / model_cache_name / "snapshots"
        if snapshot_root.is_dir():
            snapshots.extend(path for path in snapshot_root.iterdir() if path.is_dir())
    return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None


def _offline_mode_enabled() -> bool:
    truthy = {"1", "on", "true", "yes"}
    return any(
        os.environ.get(name, "").strip().lower() in truthy
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def resolve_model_path(value: Optional[str]) -> Tuple[str, str]:
    """Resolve a local snapshot when available, otherwise retain the Hub ID."""
    if value:
        candidate = Path(value).expanduser()
        if candidate.exists():
            resolved = candidate.resolve()
            return str(resolved), resolved.name
        if candidate.is_absolute() or str(value).startswith(("./", "../", "~")):
            raise FileNotFoundError(f"configured LLM model path does not exist: {candidate}")
        cached = _cached_model_path(str(value))
        if cached is not None:
            return str(cached), cached.name
        return str(value), "hub_download_or_cache"
    cached = _cached_model_path(MODEL_ID)
    if cached is not None:
        return str(cached), cached.name
    return MODEL_ID, "hub_download_or_cache"


def _bitsandbytes_kernel_policy(
    torch: Any,
    bnb: Any,
) -> str:
    """Select the pinned bnb path used by the SM120 evaluation."""
    version = str(getattr(bnb, "__version__", "unknown"))
    try:
        capability = tuple(int(value) for value in torch.cuda.get_device_capability())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        capability = ()
    if capability == (12, 0) and version != QUALIFIED_BITSANDBYTES:
        raise RuntimeError(
            f"bitsandbytes {version} is not qualified on this SM120 GPU. "
            f"Install bitsandbytes=={QUALIFIED_BITSANDBYTES}; the 0.50 fused "
            "4-bit path failed "
            "production-size canaries."
        )
    if capability == (12, 0):
        raw_torch_version = str(getattr(torch, "__version__", "unknown"))
        try:
            torch_version = Version(raw_torch_version.split("+", 1)[0])
        except InvalidVersion as exc:
            raise RuntimeError(
                f"cannot validate PyTorch version {raw_torch_version!r} for SM120"
            ) from exc
        if torch_version != Version(QUALIFIED_TORCH):
            raise RuntimeError(
                f"PyTorch {raw_torch_version} is rejected on this SM120 GPU: "
                "the LLM workload is qualified only on "
                f"PyTorch {QUALIFIED_TORCH} CUDA 13. Run ./hrc setup-llm."
            )
        return (
            "bitsandbytes_0.49.2_eager_absmax_direct_nf4_linear_"
            "fresh_prefill_append_crop_candidates"
        )
    return f"bitsandbytes_{version}_default_4bit"


def reserve_display_vram(torch: Any, headroom_gib: float) -> Dict[str, Any]:
    """Cap this process's VRAM so it can never starve the display driver.

    On a single-GPU desktop the display server, compositor and browser allocate
    from the same device. Their demand is spiky and outside this run's control,
    and a GPU that cannot serve them wedges the whole session: the display stops
    updating, input dies, and the machine needs a power cycle. No Xid or OOM
    record survives, because nothing gets flushed after the hang.

    ``set_per_process_memory_fraction`` moves that boundary inside the caching
    allocator, which turns "took the desktop's memory" into a catchable
    ``torch.OutOfMemoryError`` this process can degrade on.

    The reserve is close but not exact: the cap binds the caching allocator,
    while the CUDA context keeps growing outside it as cuBLAS workspaces and
    kernels load. A measured eight-episode run held 1.11GiB against a 1.25GiB
    reserve, so treat the setting as approximate and leave a real margin.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    headroom_bytes = int(max(0.0, float(headroom_gib)) * (1024 ** 3))
    # Memory already held by other processes is unavailable regardless, so the
    # cap has to be expressed against the total the fraction is applied to.
    external_bytes = max(0, int(total_bytes) - int(free_bytes))
    allowed_bytes = int(total_bytes) - external_bytes - headroom_bytes
    if allowed_bytes <= 0:
        raise RuntimeError(
            "no VRAM budget remains for the LLM baseline: "
            f"total={total_bytes / 1024 ** 3:.2f}GiB, "
            f"used_by_others={external_bytes / 1024 ** 3:.2f}GiB, "
            f"reserved_for_display={headroom_bytes / 1024 ** 3:.2f}GiB. "
            "Close GPU clients or lower llm_vram_headroom_gib."
        )
    fraction = min(1.0, allowed_bytes / float(total_bytes))
    torch.cuda.set_per_process_memory_fraction(fraction)
    return {
        "total_gib": float(total_bytes) / 1024 ** 3,
        "external_gib": float(external_bytes) / 1024 ** 3,
        "headroom_gib": float(headroom_bytes) / 1024 ** 3,
        "allowed_gib": float(allowed_bytes) / 1024 ** 3,
        "fraction": float(fraction),
    }


def kv_cache_bytes_per_token(config: Any, dtype_bytes: int = 2) -> int:
    """Bytes of key/value cache one prompt token occupies.

    This is what makes prompt length a memory decision rather than a speed one:
    the cache is the run's dominant variable allocation, it grows with the
    retained memory the prompt encodes, and on this GPU it competes with the
    display.
    """
    layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    kv_heads = int(
        getattr(config, "num_key_value_heads", 0)
        or getattr(config, "num_attention_heads", 0)
        or 0
    )
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    if not head_dim:
        hidden = int(getattr(config, "hidden_size", 0) or 0)
        heads = int(getattr(config, "num_attention_heads", 0) or 0)
        head_dim = hidden // heads if heads else 0
    if not (layers and kv_heads and head_dim):
        raise RuntimeError(
            "cannot size the key/value cache from this model config; refusing "
            "to run unbounded on a display GPU"
        )
    return 2 * layers * kv_heads * head_dim * int(dtype_bytes)


def report_vram_budget(
    model_source: Optional[str] = None,
    headroom_gib: float = 1.25,
) -> Dict[str, Any]:
    """Estimate the prompt budget without loading the model.

    A run that cannot fit fails hours in, after the prompt has grown, and seed
    resume is coarse enough that the seed restarts. Answering "will this fit"
    up front is therefore worth an estimate from the checkpoint on disk.
    """
    import torch
    from transformers import AutoConfig

    source, _snapshot = resolve_model_path(model_source)
    config = AutoConfig.from_pretrained(
        source, local_files_only=Path(source).expanduser().exists(),
    )
    weight_bytes = sum(
        path.stat().st_size
        for path in Path(source).glob("*.safetensors")
    ) if Path(source).is_dir() else 0
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    external_bytes = int(total_bytes) - int(free_bytes)
    headroom_bytes = int(max(0.0, float(headroom_gib)) * 1024 ** 3)
    per_token = 2 * kv_cache_bytes_per_token(config)
    prompt_bytes = (
        int(total_bytes) - external_bytes - headroom_bytes - int(weight_bytes)
    )
    return {
        "total_gib": float(total_bytes) / 1024 ** 3,
        "other_clients_gib": float(external_bytes) / 1024 ** 3,
        "headroom_gib": float(headroom_bytes) / 1024 ** 3,
        "weights_gib": float(weight_bytes) / 1024 ** 3,
        "bytes_per_prompt_token": int(per_token),
        "prompt_token_estimate": int(max(0, prompt_bytes) // per_token),
        # What the budget would be with the card to this process alone. Adds
        # back every other client's memory, not only the display's, so it is an
        # upper bound whenever another compute job also holds the GPU.
        "prompt_token_estimate_if_gpu_exclusive": int(
            max(0, prompt_bytes + external_bytes + headroom_bytes) // per_token
        ),
        "weights_from": "safetensors_on_disk" if weight_bytes else "unknown",
    }


def _materialize_nested_absmax(model: Any, torch: Any, bnb: Any) -> int:
    """Expand double-quantized 4-bit scales once before SM120 inference.

    bitsandbytes otherwise invokes its native ``dequantize_blockwise``
    operator for every 4-bit linear layer on every model forward. Long RTX
    50-series runs have terminated inside that operation. Materializing the
    small scale tensors once is algebraically identical, removes the repeated
    native call, and leaves the quantized model weights unchanged.
    """
    materialized = 0
    with torch.inference_mode():
        for module in model.modules():
            weight = getattr(module, "weight", None)
            state = getattr(weight, "quant_state", None)
            if state is None or not bool(getattr(state, "nested", False)):
                continue
            if state.state2 is None or state.offset is None:
                raise RuntimeError("nested 4-bit quantization state is incomplete")
            expanded = bnb.functional.dequantize_blockwise(
                state.absmax,
                state.state2,
            )
            expanded.add_(state.offset)
            if expanded.dtype != torch.float32:
                expanded = expanded.float()
            state.absmax = expanded
            state.state2 = None
            state.offset = None
            state.nested = False
            materialized += 1
    return materialized


def _install_direct_4bit_inference(model: Any, torch: Any, bnb: Any) -> int:
    """Remove bitsandbytes' autograd wrapper from frozen 4-bit inference.

    ``MatMul4Bit.forward`` already consists of dequantizing the frozen weight
    and calling ``torch.nn.functional.linear``.  Calling it through
    ``torch.autograd.Function.apply`` needlessly enters functorch even under
    inference mode.  On the qualified Torch 2.11/SM120 runtime, long jobs have
    terminated in that wrapper with a fatal CPython ``vgetargs1_impl`` error.
    Install the operation inside that wrapper directly on every Linear4bit
    instance.  This preserves bitsandbytes' NF4 dequantization and the exact
    linear algebra while ensuring production forwards never enter the failing
    autograd/functorch wrapper.
    """

    linear_type = bnb.nn.Linear4bit

    def direct_forward(module: Any, x: Any) -> Any:
        quant_state = getattr(module.weight, "quant_state", None)
        if quant_state is None:
            raise RuntimeError("4-bit linear layer has no quantization state")
        if module.bias is not None and module.bias.dtype != x.dtype:
            module.bias.data = module.bias.data.to(x.dtype)
        if not module.compute_type_is_set:
            module.set_compute_type(x)
            module.compute_type_is_set = True

        input_dtype = x.dtype
        compute_dtype = module.compute_dtype or input_dtype
        compute_input = x.to(compute_dtype)
        bias = (
            None
            if module.bias is None
            else module.bias.to(compute_dtype)
        )
        if bool(getattr(quant_state, "nested", False)):
            raise RuntimeError("nested NF4 scales were not materialized")
        weight = bnb.functional.dequantize_4bit(
            module.weight.t(),
            quant_state,
        ).to(compute_dtype).t()
        return torch.nn.functional.linear(
            compute_input,
            weight,
            bias,
        ).to(input_dtype)

    installed = 0
    for module in model.modules():
        if not isinstance(module, linear_type):
            continue
        module.forward = types.MethodType(direct_forward, module)
        module._hrc_direct_4bit_inference = True
        installed += 1
    return installed


def preflight_llm_runtime() -> str:
    """Reject an unqualified GPU stack before a paired Full run does any work."""
    try:
        import bitsandbytes as bnb
        import torch
        import transformers
        import accelerate
    except ImportError as exc:
        raise RuntimeError(
            "The in-context LLM baseline requires torch, transformers, "
            "accelerate, and bitsandbytes."
        ) from exc
    if tuple(sys.version_info[:2]) != QUALIFIED_PYTHON:
        raise RuntimeError(
            "the LLM runtime requires Python "
            f"{QUALIFIED_PYTHON[0]}.{QUALIFIED_PYTHON[1]}; found "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "Run ./hrc setup-llm."
        )
    versions = {
        "torch": str(torch.__version__).split("+", 1)[0],
        "transformers": str(transformers.__version__),
        "accelerate": str(accelerate.__version__),
        "bitsandbytes": str(bnb.__version__),
    }
    expected = {
        "torch": QUALIFIED_TORCH,
        "transformers": QUALIFIED_TRANSFORMERS,
        "accelerate": QUALIFIED_ACCELERATE,
        "bitsandbytes": QUALIFIED_BITSANDBYTES,
    }
    mismatches = {
        name: {"expected": expected[name], "actual": actual}
        for name, actual in versions.items()
        if actual != expected[name]
    }
    if mismatches:
        raise RuntimeError(
            "unqualified LLM package versions: "
            f"{json.dumps(mismatches, sort_keys=True)}. Run ./hrc setup-llm."
        )
    runtime_root = Path(sys.prefix).resolve()
    foreign_packages = {}
    for name, module in (
        ("torch", torch),
        ("transformers", transformers),
        ("accelerate", accelerate),
        ("bitsandbytes", bnb),
    ):
        module_path = Path(str(module.__file__)).resolve()
        if not module_path.is_relative_to(runtime_root):
            foreign_packages[name] = str(module_path)
    venv_config = runtime_root / "pyvenv.cfg"
    inherited_site_packages = bool(
        venv_config.is_file()
        and "include-system-site-packages = true" in venv_config.read_text()
    )
    if foreign_packages or inherited_site_packages:
        raise RuntimeError(
            "the LLM environment is not self-contained: "
            f"foreign_packages={json.dumps(foreign_packages, sort_keys=True)}, "
            f"include_system_site_packages={inherited_site_packages}. "
            "Recreate .venv-llm with ./hrc setup-llm."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the in-context LLM baseline")
    return _bitsandbytes_kernel_policy(torch, bnb)


class QwenActionScorer:
    """Rank semantic actions from one fresh prefill and disposable cache branches."""

    def __init__(
        self,
        model_source: Optional[str] = None,
        *,
        context_tokens: int = 32768,
        candidate_batch: int = 1,
        cache_capacity: int = 65536,
        vram_headroom_gib: float = 2.0,
        prefill_chunk_tokens: int = 1024,
    ) -> None:
        self.requested_source = str(model_source or "")
        self.context_tokens = int(context_tokens)
        self.candidate_batch = max(1, int(candidate_batch))
        self.cache_capacity = max(1, int(cache_capacity))
        self.vram_headroom_gib = max(0.0, float(vram_headroom_gib))
        self.prefill_chunk_tokens = int(prefill_chunk_tokens)
        self.vram_budget: Dict[str, Any] = {}
        self.vram_context_limit = 0
        self._last_prompt_tokens = 0
        self.source: Optional[str] = None
        self.snapshot: Optional[str] = None
        self.local_files_only: Optional[bool] = None
        self.torch: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.kernel_policy = "uninitialized"
        self.materialized_absmax_states = 0
        self.direct_4bit_modules = 0
        self._score_cache: Dict[str, ScoreResult] = {}

    def __deepcopy__(self, memo: Dict[int, Any]) -> "QwenActionScorer":
        # The frozen runtime is immutable and can be shared by evaluator snapshots.
        memo[id(self)] = self
        return self

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The in-context LLM baseline requires torch, transformers, "
                "bitsandbytes, and accelerate."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; the quantized in-context LLM baseline "
                "cannot be evaluated."
            )
        # Before the first allocation, so the model load is itself bounded.
        self.vram_budget = reserve_display_vram(torch, self.vram_headroom_gib)
        try:
            import bitsandbytes as bnb
        except ImportError:
            # Transformers will provide the actionable dependency error if the
            # selected checkpoint actually needs bitsandbytes.  Keeping this
            # optional also supports non-bnb local checkpoints and test doubles.
            self.kernel_policy = "bitsandbytes_unavailable"
        else:
            self.kernel_policy = _bitsandbytes_kernel_policy(torch, bnb)

        source, snapshot = resolve_model_path(self.requested_source)
        self.source = source
        self.snapshot = snapshot
        self.local_files_only = Path(source).expanduser().exists()
        if not self.local_files_only and _offline_mode_enabled():
            raise RuntimeError(
                f"LLM checkpoint {source!r} is not cached, but Hugging Face offline "
                "mode is enabled. Disable HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE or "
                "set HRC_LLM_MODEL to a downloaded local snapshot."
            )
        self.torch = torch
        self._check_weights_fit_the_cap(source)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=self.local_files_only,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Unable to load LLM checkpoint {source!r}. Ensure network access "
                "for the initial Hugging Face download or set HRC_LLM_MODEL to a "
                "complete local snapshot."
            ) from exc
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                source,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Unable to load LLM checkpoint {source!r}. The cached/downloaded "
                "snapshot may be incomplete; retry with network access or set "
                "HRC_LLM_MODEL to a complete local snapshot."
            ) from exc
        self.model.eval()
        self._verify_logits_contract()
        if "eager_absmax" in self.kernel_policy:
            self.materialized_absmax_states = _materialize_nested_absmax(
                self.model,
                torch,
                bnb,
            )
            remaining_nested = sum(
                bool(getattr(
                    getattr(getattr(module, "weight", None), "quant_state", None),
                    "nested",
                    False,
                ))
                for module in self.model.modules()
            )
            if remaining_nested:
                raise RuntimeError(
                    "failed to materialize every nested 4-bit scale state: "
                    f"remaining={remaining_nested}"
                )
            self.direct_4bit_modules = _install_direct_4bit_inference(
                self.model,
                torch,
                bnb,
            )
            remaining_wrapped = sum(
                isinstance(module, bnb.nn.Linear4bit)
                and not bool(getattr(
                    module,
                    "_hrc_direct_4bit_inference",
                    False,
                ))
                for module in self.model.modules()
            )
            if not self.direct_4bit_modules or remaining_wrapped:
                raise RuntimeError(
                    "failed to replace every bitsandbytes 4-bit inference "
                    "wrapper: "
                    f"installed={self.direct_4bit_modules}, "
                    f"remaining={remaining_wrapped}"
                )
            # Turn asynchronous setup failures into startup failures instead
            # of corrupting a long evaluation later.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(
                "[llm] 4-bit setup: "
                f"materialized_absmax_states={self.materialized_absmax_states} "
                f"direct_4bit_modules={self.direct_4bit_modules} "
                f"policy={self.kernel_policy}",
                flush=True,
            )
        # After the weights are resident, so the budget reflects what is
        # actually left rather than what was free before loading.
        torch.cuda.empty_cache()
        self.vram_context_limit = self._vram_token_limit()
        budget = self.vram_budget
        print(
            "[llm] vram budget: "
            f"total={budget.get('total_gib', 0.0):.2f}GiB "
            f"other_processes={budget.get('external_gib', 0.0):.2f}GiB "
            f"display_reserve={budget.get('headroom_gib', 0.0):.2f}GiB "
            f"process_cap={budget.get('allowed_gib', 0.0):.2f}GiB "
            f"weights={torch.cuda.memory_allocated() / 1024 ** 3:.2f}GiB "
            f"prompt_token_limit={self.vram_context_limit} "
            f"(model window {getattr(self.model.config, 'max_position_embeddings', 0)}, "
            f"requested {self.context_tokens})",
            flush=True,
        )
        if self.vram_context_limit <= 0:
            raise RuntimeError(
                "no VRAM remains for a prompt after loading the model: "
                f"{json.dumps(budget, sort_keys=True)}. Lower "
                "llm_vram_headroom_gib, close other GPU clients, or run the "
                "baseline on a GPU that is not driving a display."
            )
        if self.vram_context_limit < int(self.context_tokens):
            # The prompt grows with retained memory across a run, so a budget
            # below the requested window is a prediction that this run stops
            # early. Said once at startup, because seed resume is coarse: a
            # stop late in a seed re-runs that whole seed.
            reclaimable = (
                budget.get("external_gib", 0.0) + budget.get("headroom_gib", 0.0)
            )
            print(
                "[llm] WARNING: VRAM limits the prompt to "
                f"{self.vram_context_limit} tokens, below the requested "
                f"{self.context_tokens}. The prompt grows as replay memory "
                "grows, so a long run may stop once it no longer fits. "
                f"Roughly {reclaimable:.2f}GiB is held by or reserved for other "
                "GPU clients: running with no desktop session on this GPU "
                "recovers it. See the language-model baseline section of "
                "README.md.",
                flush=True,
            )

    def _check_weights_fit_the_cap(self, source: str) -> None:
        """Refuse a load the VRAM cap cannot hold, before it half-happens.

        Without this, a busy GPU turns into an allocator failure part-way
        through materializing the weights, whose message says nothing about the
        cap or the clients that consumed the budget.
        """
        snapshot = Path(source).expanduser()
        if not snapshot.is_dir():
            return
        weight_bytes = sum(
            path.stat().st_size for path in snapshot.glob("*.safetensors")
        )
        allowed_gib = float(self.vram_budget.get("allowed_gib", 0.0))
        if not weight_bytes or not allowed_gib:
            return
        weights_gib = weight_bytes / 1024 ** 3
        if weights_gib >= allowed_gib:
            raise RuntimeError(
                f"the checkpoint's {weights_gib:.2f}GiB of weights do not fit "
                f"the {allowed_gib:.2f}GiB VRAM budget for this process "
                f"({json.dumps(self.vram_budget, sort_keys=True)}). Close other "
                "GPU clients, lower llm_vram_headroom_gib, or run on a GPU that "
                "is not driving a display."
            )

    def _model_device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration) as exc:
            raise RuntimeError("loaded language model has no input embedding device") from exc

    def _vram_token_limit(self) -> int:
        """Prompt tokens whose cache and activations fit the reserved budget.

        The model's own context window is not a memory bound: a 32k prompt costs
        4.5GiB of key/value cache on this checkpoint, which does not coexist with
        6.5GiB of weights and a desktop on a 12GiB card. Sizing the limit from
        memory actually left after loading is what keeps prompt growth from
        turning into a display hang partway through a run.
        """
        torch = self.torch
        per_token, fixed_bytes = self._prefill_cost_model()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        reserved = int(
            float(self.vram_budget.get("headroom_gib", self.vram_headroom_gib))
            * (1024 ** 3)
        )
        # Two ceilings apply and the tighter one governs. The allocator cap is
        # usually it: device-free memory does not know this process is capped,
        # and using it alone reported a limit that then failed to allocate.
        device_available = int(free_bytes) - reserved
        cap_bytes = int(
            float(self.vram_budget.get("fraction", 1.0)) * float(total_bytes)
        )
        cap_available = cap_bytes - int(torch.cuda.memory_allocated())
        # The caching allocator holds blocks it has not handed out, so the last
        # few hundred MiB of the cap are not reliably usable. Measured at
        # 228-430MiB reserved-but-unallocated at the point of failure, the
        # larger figure without expandable_segments. Sized above the midpoint
        # because a pinned-encoding run cannot shed anything to recover: an
        # over-estimate costs prompt tokens, an under-estimate costs the seed.
        allocator_slack = 384 * 1024 ** 2
        available = (
            min(device_available, cap_available) - fixed_bytes - allocator_slack
        )
        return max(0, available // per_token)

    def _prefill_cost_model(self) -> Tuple[int, int]:
        """Bytes per prompt token and the fixed cost, as measured on SM120.

        Peak is affine in prompt length, not proportional: a prefill holds the
        key/value cache for every token plus intermediates for the positions in
        flight. Chunking bounds the second term, which is why it changes the
        slope rather than the intercept.

        Fitted on this checkpoint over 2k-12k-token prefills:
        unchunked 240.5 KiB/token + 96 MiB, chunked 153 KiB/token + 193 MiB at
        1024 and + 153 MiB at 512, against a 144 KiB/token cache. Slope is
        scaled from the measured ratio so a different checkpoint geometry still
        gets a sane estimate.
        """
        cache_per_token = kv_cache_bytes_per_token(self.model.config)
        chunk = int(self.prefill_chunk_tokens)
        if chunk <= 0:
            return int(cache_per_token * 240.5 / 144.0), 96 * 1024 ** 2
        # The intercept grows with the chunk because a chunk's intermediates are
        # what it bounds them to.
        fixed_mib = 96 + int(round(chunk * 96 / 1024))
        return int(cache_per_token * 153.0 / 144.0), fixed_mib * 1024 ** 2

    def _context_limit(self) -> int:
        native_limit = int(
            getattr(self.model.config, "max_position_embeddings", 0) or 0
        )
        requested = int(self.context_tokens)
        limits = [value for value in (requested, native_limit) if value > 0]
        if self.vram_context_limit > 0:
            limits.append(self.vram_context_limit)
        return min(limits) if limits else max(requested, native_limit)

    def _forward(self, *, logits_to_keep: Optional[int] = None, **kwargs: Any) -> Any:
        # Passed positively rather than best-effort: ``forward`` accepts
        # ``**kwargs``, so a renamed or misspelled argument is swallowed
        # silently, and losing this one means materializing logits for every
        # position -- 3.7GiB at 13k tokens against a 12GiB card shared with the
        # display. ``_verify_logits_contract`` fails the load instead.
        if logits_to_keep is not None:
            return self.model(logits_to_keep=logits_to_keep, **kwargs)
        return self.model(**kwargs)

    def _verify_logits_contract(self) -> None:
        """Fail the load if the model would return logits for every position."""
        import inspect

        parameters = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" not in parameters:
            raise RuntimeError(
                "the installed transformers build does not accept "
                "logits_to_keep, so every forward would materialize logits for "
                "the whole prompt. Run ./hrc setup-llm to install the qualified "
                f"transformers {QUALIFIED_TRANSFORMERS}."
            )

    @staticmethod
    def _cache_length(cache: Any) -> Optional[int]:
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
        if isinstance(cache, tuple) and cache and cache[0]:
            return int(cache[0][0].shape[-2])
        return None

    @staticmethod
    def _coerce_input_ids(encoded: Any) -> List[int]:
        # Transformers 5 changed ``apply_chat_template`` to return a
        # BatchEncoding by default.  Older releases returned the input-id list
        # directly, so accept both contracts without depending on a
        # version-specific ``return_dict`` default.
        if isinstance(encoded, Mapping):
            if "input_ids" not in encoded:
                raise RuntimeError(
                    "chat-template tokenizer output does not contain input_ids"
                )
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if not isinstance(encoded, (list, tuple)):
            raise RuntimeError(
                "chat-template tokenizer returned an unsupported input-id format"
            )
        if encoded and isinstance(encoded[0], (list, tuple)):
            if len(encoded) != 1:
                raise RuntimeError(
                    "chat-template tokenizer unexpectedly returned multiple sequences"
                )
            encoded = encoded[0]
        return [int(token) for token in encoded]

    def _chat_ids(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> List[int]:
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        return self._coerce_input_ids(encoded)

    def _message_ids(
        self,
        role: str,
        content: str,
        *,
        add_generation_prompt: bool,
    ) -> List[int]:
        """Compatibility helper retained for tokenizer-contract tests."""
        return self._chat_ids(
            ({"role": role, "content": content},),
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _semantic_action_query(
        query_prompt: str,
        actions: Sequence[str],
    ) -> str:
        # Shared with build_prompt so an audit renders the scored prompt.
        return build_candidate_prompt(query_prompt, actions)

    def score(
        self,
        context_prompt: str,
        query_prompt: str,
        candidates: Sequence[str],
    ) -> ScoreResult:
        try:
            return self._score(context_prompt, query_prompt, candidates)
        except self._oom_errors() as exc:
            # The cap in reserve_display_vram is what makes this reachable
            # instead of the driver wedging. Recover the allocator's memory and
            # report it as a budget overrun so the caller can shrink the prompt.
            self._release_device_memory()
            raise PromptTooLongError(
                self._last_prompt_tokens, self._vram_token_limit(),
            ) from exc

    def _oom_errors(self) -> Tuple[type, ...]:
        """Only the allocator's own out-of-memory types.

        Deliberately not ``RuntimeError``: the scoring path raises that for
        cache-length and kernel-policy violations, and silently re-reporting one
        of those as a prompt-size problem would hide a correctness bug behind a
        prompt that quietly got smaller.
        """
        torch = self.torch
        errors: List[type] = []
        for owner in (torch, getattr(torch, "cuda", None)):
            error = getattr(owner, "OutOfMemoryError", None)
            if isinstance(error, type) and error not in errors:
                errors.append(error)
        return tuple(errors)

    def _release_device_memory(self) -> None:
        torch = self.torch
        if torch is None:
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()

    def _prefill(
        self,
        prompt_ids: Sequence[int],
        device: Any,
    ) -> Tuple[Any, Any, int]:
        """Build the prompt cache, optionally a chunk of positions at a time.

        Attention intermediates during a prefill are proportional to the number
        of positions in flight, and on a display GPU they are what a long prompt
        cannot afford: they peak alongside the cache rather than after it.
        Filling the cache in chunks bounds them by the chunk instead of the
        prompt, at identical arithmetic -- each chunk attends over every key
        already cached, which is the same incremental step this scorer already
        takes for candidate tokens.

        A prompt that fits one chunk takes exactly the call it took before, so
        the single-chunk path stays byte-identical.
        """
        torch = self.torch
        total = len(prompt_ids)
        chunk = int(self.prefill_chunk_tokens)
        spans = (
            [(0, total)]
            if chunk <= 0 or chunk >= total
            else [(start, min(start + chunk, total))
                  for start in range(0, total, chunk)]
        )
        cache: Any = None
        logits: Any = None
        for start, stop in spans:
            piece = torch.tensor(
                [list(prompt_ids[start:stop])],
                dtype=torch.long,
                device=device,
            )
            # Only pass a cache once there is one, so a single-chunk prefill
            # issues the same call signature as an unchunked one.
            extra = {"past_key_values": cache} if cache is not None else {}
            with torch.inference_mode():
                output = self._forward(
                    input_ids=piece,
                    attention_mask=torch.ones(
                        (1, stop), dtype=torch.long, device=device,
                    ),
                    use_cache=True,
                    logits_to_keep=1,
                    **extra,
                )
            cache = output.past_key_values
            logits = output.logits
        return (
            cache,
            torch.log_softmax(logits[0, -1].float(), dim=-1),
            len(spans),
        )

    def _score(
        self,
        context_prompt: str,
        query_prompt: str,
        candidates: Sequence[str],
    ) -> ScoreResult:
        self._load()
        actions = tuple(str(action) for action in candidates)
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("candidate actions must be non-empty and unique")
        semantic_query = self._semantic_action_query(query_prompt, actions)
        messages = (
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": semantic_query},
        )
        prompt_ids = self._chat_ids(
            messages,
            add_generation_prompt=True,
        )
        audit_payload = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_hash = hashlib.sha256(audit_payload.encode("utf-8")).hexdigest()
        self._last_prompt_tokens = len(prompt_ids)
        cached_result = self._score_cache.get(prompt_hash)
        if cached_result is not None:
            return replace(
                cached_result,
                wall_s=0.0,
                cache_hit=True,
                model_forwards=0,
                uncached_wall_s=float(cached_result.wall_s),
            )

        encoded_actions = []
        for action in actions:
            target = [
                int(token)
                for token in self.tokenizer.encode(
                    action,
                    add_special_tokens=False,
                )
            ]
            if self.tokenizer.eos_token_id is not None:
                target.append(int(self.tokenizer.eos_token_id))
            if not target:
                raise ValueError(
                    "every semantic action must tokenize to at least one token"
                )
            encoded_actions.append(target)

        context_limit = self._context_limit()
        prompt_token_count = len(prompt_ids)
        required_tokens = prompt_token_count + max(map(len, encoded_actions))
        if context_limit <= 0 or required_tokens > context_limit:
            raise PromptTooLongError(required_tokens, context_limit)

        torch = self.torch
        device = self._model_device()
        prompt_tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )
        started = time.perf_counter()
        prompt_cache, first_log_probs, prefill_chunks = self._prefill(
            prompt_ids, device,
        )
        prompt_cache_length = self._cache_length(prompt_cache)
        if prompt_cache_length != prompt_token_count:
            raise RuntimeError(
                "model returned an invalid prompt cache length: "
                f"expected={prompt_token_count}, actual={prompt_cache_length}"
            )
        log_scores: List[float] = []
        for target_ids in encoded_actions:
            score = float(first_log_probs[int(target_ids[0])].item())
            if len(target_ids) > 1:
                suffix_ids = target_ids[:-1]
                suffix_tensor = torch.tensor(
                    [suffix_ids],
                    dtype=torch.long,
                    device=device,
                )
                with torch.inference_mode():
                    suffix_output = self._forward(
                        input_ids=suffix_tensor,
                        attention_mask=torch.ones(
                            (1, prompt_token_count + len(suffix_ids)),
                            dtype=torch.long,
                            device=device,
                        ),
                        past_key_values=prompt_cache,
                        use_cache=True,
                        logits_to_keep=len(suffix_ids),
                    )
                suffix_log_probs = torch.log_softmax(
                    suffix_output.logits[0, -len(suffix_ids):].float(),
                    dim=-1,
                )
                suffix_targets = torch.tensor(
                    target_ids[1:],
                    dtype=torch.long,
                    device=suffix_log_probs.device,
                )
                score += float(suffix_log_probs.gather(
                    1,
                    suffix_targets.unsqueeze(1),
                ).squeeze(1).sum().item())
                if not hasattr(prompt_cache, "crop"):
                    raise RuntimeError(
                        "the installed transformers cache cannot restore the "
                        "shared prompt after candidate scoring"
                    )
                # Transformers >=5.16 uses a negative crop to remove the
                # disposable suffix while preserving the shared prompt.
                prompt_cache.crop(-len(suffix_ids))
                if self._cache_length(prompt_cache) != prompt_cache_length:
                    raise RuntimeError(
                        "candidate scoring failed to restore the prompt cache"
                    )
            log_scores.append(score / len(target_ids))
        if self._cache_length(prompt_cache) != prompt_cache_length:
            raise RuntimeError("candidate scoring mutated the shared prompt cache")
        wall_s = time.perf_counter() - started
        peak = max(log_scores)
        unnormalized = [math.exp(score - peak) for score in log_scores]
        total = sum(unnormalized)
        if total <= 0.0 or not math.isfinite(total):
            probabilities = {
                action: 1.0 / len(actions)
                for action in actions
            }
        else:
            probabilities = {
                action: float(value) / total
                for action, value in zip(actions, unnormalized)
            }
        result = ScoreResult(
            probabilities=probabilities,
            prompt_tokens=prompt_token_count,
            candidate_tokens=sum(map(len, encoded_actions)),
            wall_s=float(wall_s),
            prompt_hash=prompt_hash,
            context_limit=context_limit,
            cache_hit=False,
            scoring_method="append_crop_mean_semantic_action_log_likelihood",
            model_forwards=(
                prefill_chunks
                + sum(len(tokens) > 1 for tokens in encoded_actions)
            ),
            uncached_wall_s=float(wall_s),
        )
        self._score_cache[prompt_hash] = result
        # Bound the memoization table. Every entry is a recomputable function of
        # its prompt, so evicting one costs time and cannot change a result. A
        # complete scenario seed produces a few thousand distinct prompts, so the
        # capacity is high enough that eviction does not change the hit rate.
        while len(self._score_cache) > self.cache_capacity:
            self._score_cache.pop(next(iter(self._score_cache)))
        return result


class InContextLlmAgent(AdaptiveAgent):
    """Full-system memory and routing with a frozen in-context LLM head."""

    RETRAIN_POLICY = BASELINE_TRAIN_POLICY

    def __init__(
        self,
        settings: Settings = DEFAULT_SETTINGS,
        narrate: Optional[Callable[[str], None]] = None,
        retrain_policy: Optional[TrainPolicy] = None,
        *,
        scorer: Optional[ActionScorer] = None,
    ) -> None:
        llm_settings = replace(settings, predictor=PREDICTOR_NAME)
        super().__init__(
            settings=llm_settings,
            narrate=narrate,
            retrain_policy=retrain_policy,
        )
        self.scorer: ActionScorer = scorer or QwenActionScorer(
            self.settings.llm_model,
            context_tokens=self.settings.llm_context_tokens,
            candidate_batch=self.settings.llm_candidate_batch,
            vram_headroom_gib=self.settings.llm_vram_headroom_gib,
            prefill_chunk_tokens=self.settings.llm_prefill_chunk_tokens,
        )
        self.prompt_demos: Tuple[PromptDemo, ...] = ()
        self.llm_actions: Tuple[str, ...] = ()
        self.last_score_stats: Dict[str, Any] = {}
        self._custom_fit_stats: Dict[str, Any] = {}

    @staticmethod
    def _prompt_weights(
        records: Optional[Sequence[Any]],
        fit_weights: Sequence[float],
    ) -> List[float]:
        """Recover the replay retention weight behind each fit weight.

        ``_retrain`` passes ``_demo_weights`` output, which divides each replay
        weight by its episode length to equalize per-example mass across
        episodes. That is the right per-example weight for a numeric fit and the
        wrong number for a prompt, where one value stands for a whole
        demonstration: two equally retained demonstrations of different lengths
        would be presented as differently retained.
        """
        weights = [float(weight) for weight in fit_weights]
        if records is None or len(records) != len(weights):
            return weights
        recovered: List[float] = []
        for record, fallback in zip(records, weights):
            weight = (
                record.get("weight") if isinstance(record, Mapping)
                else getattr(record, "weight", None)
            )
            recovered.append(
                float(weight) if isinstance(weight, (int, float))
                else float(fallback)
            )
        return recovered

    def _fit_predictors(
        self,
        trajectories: Sequence[List[Tuple[StateVector, str]]],
        weights: Sequence[float],
        *,
        warm_start: bool,
        records: Optional[Sequence[Any]] = None,
    ) -> None:
        self.prompt_demos = make_prompt_demos(
            trajectories, self._prompt_weights(records, weights),
        )
        self.llm_actions = candidate_actions(self.prompt_demos)
        self._custom_fit_stats = {
            "model_family": PREDICTOR_NAME,
            "predictor": PREDICTOR_NAME,
            "n_demonstrations": float(len(self.prompt_demos)),
            "n_examples": float(sum(
                len(demonstration.steps)
                for demonstration in self.prompt_demos
            )),
            "n_actions": float(len(self.llm_actions)),
            "memory_weight_sum": float(sum(
                demonstration.weight
                for demonstration in self.prompt_demos
            )),
            "warm_start": bool(warm_start),
            "estimated_flops": 0.0,
            "flop_accounting_scope": "prompt_memory_build_only",
            "flop_cross_model_comparable": False,
            "task_conditioning": "none_state_preconditions_only",
            "action_support": "shared_state_only_feasibility_mask",
            "action_score": "mean_semantic_action_token_log_likelihood",
        }

    def _predictors_ready(self) -> bool:
        return bool(self.prompt_demos and self.llm_actions)

    def _reset_predictors(self) -> None:
        self.prompt_demos = ()
        self.llm_actions = ()
        self.last_score_stats = {}
        self._custom_fit_stats = {}

    def _fit_stats(self) -> Dict[str, Any]:
        return dict(self._custom_fit_stats)

    def _estimate_flops(
        self,
        _trajectories: Sequence[List[Tuple[StateVector, str]]],
    ) -> float:
        # Training is prompt assembly; model inference is timed per prediction.
        return 0.0

    def predict_actions(
        self,
        prefix: Optional[Sequence[str]] = None,
        *,
        state: Optional[Any] = None,
        actor_id: int = 0,
        action_universe: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Use the same state and action mask interface as every other arm."""
        prefix_actions = (
            list(prefix)
            if prefix is not None
            else list(self.current_prefix)
        )
        if not self._predictors_ready():
            self.last_score_stats = {
                "llm_prompt_demo_count": 0,
                "llm_candidate_count": 0,
            }
            self._set_policy_stats(
                None,
                None,
                "no_active_demonstrations",
                source=PREDICTOR_NAME,
            )
            return {}

        encoded_state = (
            self._replay_prefix(prefix_actions)
            if state is None
            else self.domain.state_key(state, actor_id=actor_id)
        )
        # Only actions the prompt actually demonstrates can be scored, so the
        # caller's universe narrows that support rather than replacing it.
        scoreable_actions = self.llm_actions
        if action_universe is not None:
            allowed = {str(action) for action in action_universe}
            scoreable_actions = tuple(
                action for action in self.llm_actions if action in allowed
            )
        conditioned_actions = self._conditioned_actions(
            encoded_state, scoreable_actions,
        )
        if not conditioned_actions:
            # Every other arm returns an empty distribution here rather than
            # failing, and the evaluator scores an empty prediction. Raising
            # would end a run where the others report a miss.
            self.last_score_stats = {
                "llm_prompt_demo_count": len(self.prompt_demos),
                "llm_candidate_count": 0,
            }
            self._set_policy_stats(
                None,
                None,
                "no_feasible_actions",
                source=PREDICTOR_NAME,
            )
            return {}
        query_prompt = build_query_prompt(
            encoded_state, prefix_actions, self.domain,
        )
        # Annotated demonstrations first, so this arm reads the same states the
        # numeric arms fit against. If the annotated prompt does not fit, shed
        # the annotation rather than a demonstration: memory coverage is the
        # experimental variable and must stay complete. A pinned encoding skips
        # that adaptation so one run cannot mix the two.
        pinned = str(self.settings.llm_context_encoding)
        annotate = pinned != "action_only"
        encoding = "per_step_state_delta" if annotate else "action_only_pinned"

        def context(include_state_deltas: bool) -> str:
            return build_context_prompt(
                self.prompt_demos,
                self.llm_actions,
                self.domain,
                include_state_deltas=include_state_deltas,
            )

        try:
            result = self.scorer.score(
                context(annotate), query_prompt, conditioned_actions,
            )
        except PromptTooLongError:
            if pinned != "auto":
                # Shedding here would silently break the pin, and the caller
                # asked for one encoding across the whole run.
                raise
            encoding = "action_only_context_budget_exceeded"
            result = self.scorer.score(
                context(False), query_prompt, conditioned_actions,
            )
        distribution = {
            str(action): float(probability)
            for action, probability in result.probabilities.items()
        }
        if set(distribution) != set(conditioned_actions):
            raise RuntimeError(
                "LLM scorer output must cover exactly the active semantic action support"
            )
        if any(
            not math.isfinite(probability) or probability < 0.0
            for probability in distribution.values()
        ):
            raise RuntimeError("LLM scorer returned invalid probabilities")
        probability_sum = sum(distribution.values())
        if probability_sum <= 0.0:
            raise RuntimeError("LLM scorer returned zero probability mass")
        distribution = {
            action: probability / probability_sum
            for action, probability in distribution.items()
        }
        confidence, entropy, margin = self._prediction_stats(distribution)
        self._set_policy_stats(
            confidence,
            entropy,
            PREDICTOR_NAME,
            margin=margin,
            source=PREDICTOR_NAME,
        )
        self.last_score_stats = {
            "llm_prompt_tokens": int(result.prompt_tokens),
            "llm_candidate_tokens": int(result.candidate_tokens),
            "llm_inference_wall_s": float(result.wall_s),
            # Zero on a cache hit, so the two totals bracket the arm's cost:
            # measured GPU time, and the time scoring every decision from
            # scratch would have taken.
            "llm_uncached_inference_wall_s": float(
                result.wall_s + result.uncached_wall_s
            ),
            "llm_prompt_hash": str(result.prompt_hash),
            "llm_context_limit": int(result.context_limit),
            "llm_score_cache_hit": bool(result.cache_hit),
            "llm_scoring_method": str(result.scoring_method),
            "llm_model_forwards": int(result.model_forwards),
            "llm_4bit_kernel_policy": str(
                getattr(self.scorer, "kernel_policy", "not_reported")
            ),
            "llm_materialized_absmax_states": int(
                getattr(self.scorer, "materialized_absmax_states", 0)
            ),
            "llm_direct_4bit_modules": int(
                getattr(self.scorer, "direct_4bit_modules", 0)
            ),
            "llm_prompt_demo_count": len(self.prompt_demos),
            "llm_candidate_count": len(conditioned_actions),
            "llm_context_encoding": encoding,
        }
        return distribution

    def policy_stats(self) -> Dict[str, Any]:
        return {
            **super().policy_stats(),
            **self.last_score_stats,
        }

    def baseline_stats(self) -> Dict[str, Any]:
        scorer = self.scorer
        return {
            "policy": "frozen_in_context_semantic_action_scoring",
            "model_id": MODEL_ID,
            "model_source": getattr(scorer, "source", None),
            "model_snapshot": getattr(scorer, "snapshot", None),
            "prompt_contract": (
                "active_replay_weighted_state_action_demonstrations+"
                "current_state+current_prefix+conditioned_action_support"
            ),
            # Invariants of the prompt contract, recorded once per run rather
            # than restamped on every turn row where they cannot come out
            # otherwise.
            "task_conditioned": False,
            "memory_coverage": 1.0,
            "demonstration_weight_basis": "replay_retention_weight",
            "demonstration_state_encoding": "initial_predicates_plus_per_step_delta",
            "uses_recipe_labels": False,
            "uses_preference_labels": False,
            "uses_goal_predicates": False,
            "uses_explicit_dynamics": True,
            "active_prompt_demonstrations": len(self.prompt_demos),
            "active_replay_actions": len(self.llm_actions),
            "action_support": "shared_state_only_feasibility_mask",
            "context_overflow_policy": (
                "shed_state_annotation_without_dropping_demonstrations"
            ),
            "action_score": "mean_semantic_action_token_log_likelihood",
            "candidate_encoding": "direct_semantic_action_tokens",
            "inference_cache": "exact_prompt_probability_memoization_only",
            "persistent_external_kv_cache": False,
            "candidate_cache_policy": "fresh_prompt_prefill_append_score_crop",
            "cache_mutation_reuse": True,
            "vram_budget": dict(getattr(scorer, "vram_budget", {}) or {}),
            "vram_prompt_token_limit": int(
                getattr(scorer, "vram_context_limit", 0) or 0
            ),
            "vram_headroom_gib": float(
                getattr(scorer, "vram_headroom_gib", 0.0) or 0.0
            ),
            "prefill_chunk_tokens": int(
                getattr(scorer, "prefill_chunk_tokens", 0) or 0
            ),
            "context_encoding_policy": str(self.settings.llm_context_encoding),
        }

    def audit_pruning(
        self,
        max_prefixes: int = 24,
        tolerance: float = 5e-2,
    ) -> Dict[str, Any]:
        entries: Sequence[MemoryItem] = self.replay.active_items()
        trajectories, _dropped = self._build_demos(entries)
        expected_demos = make_prompt_demos(
            trajectories, [float(entry.weight) for entry in entries],
        )
        expected_actions = candidate_actions(expected_demos)
        # Compares the rendered content, states included: a wrong terminal state
        # would otherwise misrender the last step's effect and still pass. Demo
        # weights are excluded because a weight-only replay change legitimately
        # skips retraining for every arm, leaving all of them fit against the
        # weights of their last fit.
        membership_matches = (
            tuple(
                (demonstration.steps, demonstration.final_state)
                for demonstration in self.prompt_demos
            )
            == tuple(
                (demonstration.steps, demonstration.final_state)
                for demonstration in expected_demos
            )
        )
        action_support_matches = self.llm_actions == expected_actions
        active_keys = {entry.key for entry in entries}
        pruned_keys = {record.key for record in self.replay.pruned.values()}
        inputs_verified = bool(
            membership_matches
            and action_support_matches
            and not (active_keys & pruned_keys)
        )
        # A frozen in-context predictor has no fitted weights, so there is no
        # optimizer path to depend on and no cold refit to compare against: the
        # prompt either is the active replay set or it is not.
        return {
            "max_l1": 0.0 if inputs_verified else 1.0,
            "mean_l1": 0.0 if inputs_verified else 1.0,
            "n_prefixes": 0,
            "passed": inputs_verified,
            "active_only_training_inputs_verified": inputs_verified,
            "comparison": "active_prompt_membership_and_action_support",
            "model_family": "in_context_llm",
            "n_active_variants": len(expected_demos),
            "n_pruned_variants": len(self.replay.pruned),
            "pruned_available": False,
            "redundancy_max_l1": 0.0,
            "redundancy_mean_l1": 0.0,
            "deployed_path_dependence_max_l1": 0.0,
            "deployed_path_dependence_mean_l1": 0.0,
            "tolerance": float(tolerance),
            "audit_kind": "active_prompt_membership_and_action_support",
            "prompt_demo_count": len(self.prompt_demos),
            "active_replay_count": len(expected_demos),
            "membership_matches": membership_matches,
            "action_support_matches": action_support_matches,
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the paired or standalone LLM evaluation through the common harness."""
    from .evaluation import (
        _jsonable,
        find_resumable_run,
        parse_args,
        run_evaluation,
    )

    paired_baselines = ("full", PREDICTOR_NAME)
    config = parse_args(
        list(argv) if argv is not None else None,
        description=(
            "Compare Full with the frozen in-context LLM under the full "
            "longitudinal HRC protocol using one seed per scenario."
        ),
        default_baselines=paired_baselines,
        default_seeds=(LLM_EVALUATION_SEED,),
    )
    standalone_baselines = (PREDICTOR_NAME,)
    requested_baselines = tuple(config.baselines)
    if requested_baselines not in {paired_baselines, standalone_baselines}:
        raise ValueError(
            "the dedicated LLM runner requires --baselines "
            "full,in_context_llm or --baselines in_context_llm"
        )
    if requested_baselines == paired_baselines and not config.shared_routing:
        raise ValueError(
            "the dedicated LLM runner requires the shared Full interaction "
            "schedule; --local-routing is not a comparable condition"
        )
    if requested_baselines == standalone_baselines:
        if config.shared_routing:
            raise ValueError(
                "standalone in_context_llm requires --local-routing because "
                "no Full baseline is being executed"
            )
        if config.observe_missing_recipes:
            raise ValueError(
                "standalone in_context_llm is only comparable when dynamic "
                "missing-recipe routing is disabled"
            )
    if len(config.seeds) != 1:
        raise ValueError(
            "the dedicated LLM evaluation requires exactly one seed per "
            "scenario; pass one value to --seeds"
        )
    preflight_llm_runtime()
    # Defaults this runner needs to produce a complete, single-condition run on
    # a GPU that also drives a display. Applied with setdefault, so an explicit
    # flag still wins and the choice stays visible in the run manifest.
    model_settings = dict(config.model_settings)
    # Bounds the attention intermediates that peak alongside the key/value
    # cache, which raises the prompt budget from about 6600 tokens to 9900 and
    # is what lets a full scenario's replay memory fit.
    model_settings.setdefault("llm_prefill_chunk_tokens", 1024)
    # One encoding for the whole run. The alternative, "auto", sends the
    # per-step state annotation and drops it once the prompt stops fitting,
    # which on this GPU happens partway through and leaves a run whose early
    # and late episodes are not the same condition.
    model_settings.setdefault("llm_context_encoding", "action_only")
    config = replace(
        config,
        model_settings=model_settings,
        include_oracle=False,
        workers=1,
        experiment="llm_single_seed_evaluation",
        # This runtime segfaults inside bitsandbytes' native 4-bit
        # dequantization roughly once per one to two million calls, and a
        # scenario makes about ten million, so several crashes per scenario are
        # expected rather than exceptional. Without per-event resume the seed
        # restarts from its first event each time and never finishes.
        event_resume=True,
    )
    if not config.run:
        # A crash leaves per-event state inside its own run directory, and a
        # fresh invocation would otherwise mint a new directory and start over.
        resumable = find_resumable_run(config)
        if resumable is not None:
            config = replace(config, run=resumable, resume=True)
            print(
                f"[llm] continuing incomplete run {resumable}",
                flush=True,
            )
    summary = run_evaluation(config)
    print(json.dumps(_jsonable({
        "run_dir": summary["run_dir"],
        "latest_dir": summary["latest_dir"],
        "scenarios": sorted(summary.get("scenarios", {})),
        "baselines": list(config.baselines),
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
