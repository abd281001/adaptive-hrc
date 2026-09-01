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


PROMPT_INSTRUCTIONS = """You are a next-action classifier for human-robot collaboration.
Infer the demonstrated person's workflow from PREVIOUS_DEMONSTRATIONS and the
actions already observed in the current episode. Predict the exact next action
that person would perform. Do not optimize a goal, invent an alternative plan,
or use actions outside CANDIDATE_ACTIONS.

Return exactly one complete semantic action from CANDIDATE_ACTIONS.
"""


StateVector = Tuple[int, ...]
PromptStep = Tuple[StateVector, str]


@dataclass(frozen=True)
class PromptDemo:
    """One retained state-action demonstration supplied to the frozen LLM."""

    weight: float
    steps: Tuple[PromptStep, ...]


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


class ActionScorer(Protocol):
    """Minimal interface used by the agent and lightweight test doubles."""

    def score(
        self,
        context_prompt: str,
        query_prompt: str,
        candidates: Sequence[str],
    ) -> ScoreResult:
        ...


def state_predicates(state: Sequence[int]) -> List[str]:
    """Render a binary state without adding recipe or preference metadata."""
    if len(state) != len(_FEAT):
        raise ValueError(
            f"state width {len(state)} does not match environment width {len(_FEAT)}"
        )
    return [
        name
        for name, index in _FEAT.items()
        if int(state[index]) == 1
    ]


def make_prompt_demos(
    trajectories: Sequence[Sequence[Tuple[StateVector, str]]],
    weights: Sequence[float],
) -> Tuple[PromptDemo, ...]:
    """Convert the active replay fit input into deterministic prompt examples."""
    if len(trajectories) != len(weights):
        raise ValueError("demonstrations and replay weights must have equal length")
    demonstrations: List[PromptDemo] = []
    for trajectory, weight in zip(trajectories, weights):
        steps = tuple(
            (tuple(int(value) for value in state), str(action))
            for state, action in trajectory
            if action != "stop"
        )
        if steps and float(weight) > 0.0:
            demonstrations.append(PromptDemo(float(weight), steps))
    return tuple(demonstrations)


def candidate_actions(demonstrations: Sequence[PromptDemo]) -> Tuple[str, ...]:
    """Use exactly the semantic action support present in active replay."""
    return tuple(sorted({
        action
        for demonstration in demonstrations
        for _state, action in demonstration.steps
    }))


def build_context_prompt(
    demonstrations: Sequence[PromptDemo],
    candidates: Sequence[str],
) -> str:
    """Build the static weighted demonstration context shared across turns."""
    if not demonstrations:
        raise ValueError("an in-context prompt requires retained demonstrations")
    supported_actions = candidate_actions(demonstrations)
    if tuple(candidates) != supported_actions:
        raise ValueError("candidate actions must equal active demonstration support")

    payload = {
        "previous_demonstrations": [
            {
                "memory_weight": round(float(demonstration.weight), 8),
                # A demonstrated action prefix uniquely determines state in this
                # deterministic simulator. Historical states are therefore not
                # repeated; the exact current query state remains explicit.
                "actions": [
                    action for _step_state, action in demonstration.steps
                ],
            }
            for demonstration in demonstrations
        ],
    }
    return "\n".join((
        PROMPT_INSTRUCTIONS,
        "CONTEXT:",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    ))


def build_query_prompt(
    state: StateVector,
    prefix: Sequence[str],
) -> str:
    """Build the dynamic query without task identity or future actions."""
    payload = {
        "current_state_predicates": state_predicates(state),
        "observed_actions": [str(action) for action in prefix],
    }
    return "\n".join((
        "QUERY:",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        "ANSWER:",
    ))


def build_prompt(
    demonstrations: Sequence[PromptDemo],
    state: StateVector,
    prefix: Sequence[str],
    candidates: Sequence[str],
) -> str:
    """Render the complete prompt for audits and human inspection."""
    return "\n".join((
        build_context_prompt(demonstrations, candidates),
        build_query_prompt(state, prefix),
        "CANDIDATE_ACTIONS:",
        json.dumps(list(candidates), separators=(",", ":")),
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
    """Select the canary-qualified bnb path for this SM120 host."""
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
    ) -> None:
        self.requested_source = str(model_source or "")
        self.context_tokens = int(context_tokens)
        self.candidate_batch = max(1, int(candidate_batch))
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
                "[llm] stable 4-bit setup: "
                f"materialized_absmax_states={self.materialized_absmax_states} "
                f"direct_4bit_modules={self.direct_4bit_modules} "
                f"policy={self.kernel_policy}",
                flush=True,
            )

    def _model_device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration) as exc:
            raise RuntimeError("loaded language model has no input embedding device") from exc

    def _context_limit(self) -> int:
        native_limit = int(
            getattr(self.model.config, "max_position_embeddings", 0) or 0
        )
        requested = int(self.context_tokens)
        if requested > 0 and native_limit > 0:
            return min(requested, native_limit)
        return max(requested, native_limit)

    def _forward(self, *, logits_to_keep: Optional[int] = None, **kwargs: Any) -> Any:
        if logits_to_keep is not None:
            try:
                return self.model(logits_to_keep=logits_to_keep, **kwargs)
            except TypeError:
                pass
        return self.model(**kwargs)

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
        return "\n".join((
            query_prompt,
            "CANDIDATE_ACTIONS:",
            json.dumps(
                list(actions),
                separators=(",", ":"),
            ),
            "Return exactly one complete semantic action.",
        ))

    def score(
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
        cached_result = self._score_cache.get(prompt_hash)
        if cached_result is not None:
            return replace(
                cached_result,
                wall_s=0.0,
                cache_hit=True,
                model_forwards=0,
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
            raise RuntimeError(
                "the complete active replay and semantic-action prompt exceeds "
                "the model context: "
                f"required={required_tokens}, limit={context_limit}. No demonstration "
                "was silently dropped; increase llm_context_tokens or reduce the "
                "experimental memory scope."
            )

        torch = self.torch
        device = self._model_device()
        prompt_tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            prompt_output = self._forward(
                input_ids=prompt_tensor,
                attention_mask=torch.ones_like(prompt_tensor),
                use_cache=True,
                logits_to_keep=1,
            )
        prompt_cache = prompt_output.past_key_values
        prompt_cache_length = self._cache_length(prompt_cache)
        if prompt_cache_length != prompt_token_count:
            raise RuntimeError(
                "model returned an invalid prompt cache length: "
                f"expected={prompt_token_count}, actual={prompt_cache_length}"
            )
        first_log_probs = torch.log_softmax(
            prompt_output.logits[0, -1].float(),
            dim=-1,
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
            model_forwards=1 + sum(len(tokens) > 1 for tokens in encoded_actions),
        )
        self._score_cache[prompt_hash] = result
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
        )
        self.prompt_demos: Tuple[PromptDemo, ...] = ()
        self.llm_actions: Tuple[str, ...] = ()
        self.last_score_stats: Dict[str, Any] = {}
        self._custom_fit_stats: Dict[str, Any] = {}

    def _fit_predictors(
        self,
        trajectories: Sequence[List[Tuple[StateVector, str]]],
        weights: Sequence[float],
        *,
        warm_start: bool,
        records: Optional[Sequence[Any]] = None,
    ) -> None:
        self.prompt_demos = make_prompt_demos(trajectories, weights)
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
    ) -> Dict[str, float]:
        prefix_actions = (
            list(prefix)
            if prefix is not None
            else list(self.current_prefix)
        )
        if not self._predictors_ready():
            self.last_score_stats = {
                "llm_prompt_demo_count": 0,
                "llm_candidate_count": 0,
                "llm_memory_coverage": 1.0,
            }
            self._set_policy_stats(
                None,
                None,
                "no_active_demonstrations",
                source=PREDICTOR_NAME,
            )
            return {}

        state = self._replay_prefix(prefix_actions)
        conditioned_actions = self._conditioned_actions(
            state, self.llm_actions,
        )
        if not conditioned_actions:
            raise RuntimeError(
                "conditioned action support is empty despite active demonstrations"
            )
        context_prompt = build_context_prompt(
            self.prompt_demos,
            self.llm_actions,
        )
        query_prompt = build_query_prompt(state, prefix_actions)
        result = self.scorer.score(
            context_prompt,
            query_prompt,
            conditioned_actions,
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
            "llm_memory_coverage": 1.0,
            "llm_task_conditioned": False,
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
                "active_weighted_action_demonstrations+"
                "current_state+current_prefix+conditioned_action_support"
            ),
            "task_conditioned": False,
            "uses_recipe_labels": False,
            "uses_preference_labels": False,
            "uses_goal_predicates": False,
            "uses_explicit_dynamics": True,
            "active_prompt_demonstrations": len(self.prompt_demos),
            "active_replay_actions": len(self.llm_actions),
            "action_support": "shared_state_only_feasibility_mask",
            "context_overflow_policy": "fail_without_dropping_demonstrations",
            "action_score": "mean_semantic_action_token_log_likelihood",
            "candidate_encoding": "direct_semantic_action_tokens",
            "inference_cache": "exact_prompt_probability_memoization_only",
            "persistent_external_kv_cache": False,
            "candidate_cache_policy": "fresh_prompt_prefill_append_score_crop",
            "cache_mutation_reuse": True,
        }

    def audit_pruning(
        self,
        max_prefixes: int = 24,
        tolerance: float = 5e-2,
    ) -> Dict[str, Any]:
        entries: Sequence[MemoryItem] = self.replay.active_items()
        trajectories, _dropped = self._build_demos(entries)
        weights = self._demo_weights(
            entries,
            [float(entry.weight) for entry in entries],
        )
        expected_demos = make_prompt_demos(trajectories, weights)
        expected_actions = candidate_actions(expected_demos)
        membership_matches = (
            tuple(demonstration.steps for demonstration in self.prompt_demos)
            == tuple(demonstration.steps for demonstration in expected_demos)
        )
        action_support_matches = self.llm_actions == expected_actions
        passed = bool(membership_matches and action_support_matches)
        return {
            "max_l1": 0.0 if passed else 1.0,
            "mean_l1": 0.0 if passed else 1.0,
            "n_prefixes": 0,
            "passed": passed,
            "tolerance": float(tolerance),
            "audit_kind": "active_prompt_membership_and_action_support",
            "prompt_demo_count": len(self.prompt_demos),
            "active_replay_count": len(expected_demos),
            "membership_matches": membership_matches,
            "action_support_matches": action_support_matches,
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the paired or standalone LLM evaluation through the common harness."""
    from .evaluation import _jsonable, parse_args, run_evaluation

    paired_baselines = ("full", PREDICTOR_NAME)
    config = parse_args(
        list(argv) if argv is not None else None,
        description=(
            "Compare Full with the frozen in-context LLM under one shared "
            "longitudinal HRC protocol."
        ),
        default_baselines=paired_baselines,
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
    preflight_llm_runtime()
    config = replace(
        config,
        include_oracle=False,
        workers=1,
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
