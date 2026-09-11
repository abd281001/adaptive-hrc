"""Information-parity contracts for the in-context LLM baseline."""
from __future__ import annotations

import hashlib
import io
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.environment import StateTracker, recipe_builders
from src.evaluation import EvalSettings, assist_demo, build_agent, build_task, observe_demo
from src.llm_baseline import (
    InContextLlmAgent,
    LLM_EVALUATION_SEED,
    PromptDemo,
    PromptTooLongError,
    ScoreResult,
    QwenActionScorer,
    _bitsandbytes_kernel_policy,
    _install_direct_4bit_inference,
    _materialize_nested_absmax,
    build_candidate_prompt,
    build_context_prompt,
    build_prompt,
    build_query_prompt,
    candidate_actions,
    kv_cache_bytes_per_token,
    main,
    make_prompt_demos,
    report_vram_budget,
    reserve_display_vram,
    resolve_model_path,
    state_predicates,
)
from src.models import Settings


class DemoScorer:
    """Deterministic prompt-only scorer; never imports or loads an LLM."""

    def __init__(self):
        self.prompts = []
        self.calls = []

    def score(self, context_prompt, query_prompt, candidates):
        prompt = context_prompt + "\n" + query_prompt
        self.prompts.append(prompt)
        self.calls.append((context_prompt, query_prompt, tuple(candidates)))
        context = json.loads(context_prompt.split("CONTEXT:\n", 1)[1])
        query = json.loads(query_prompt.split("QUERY:\n", 1)[1])
        prefix = query["observed_actions"]
        predicted = None
        for demonstration in context["previous_demonstrations"]:
            # Accepts both context encodings: annotated steps, and the
            # action-only form used when the annotated prompt does not fit.
            actions = (
                [step["action"] for step in demonstration["steps"]]
                if "steps" in demonstration else list(demonstration["actions"])
            )
            if actions[:len(prefix)] == prefix and len(actions) > len(prefix):
                predicted = actions[len(prefix)]
                break
        if predicted not in candidates:
            predicted = candidates[0]
        remaining = max(0, len(candidates) - 1)
        tail_probability = 0.1 / remaining if remaining else 0.0
        probabilities = {
            action: 0.9 if action == predicted else tail_probability
            for action in candidates
        }
        if not remaining:
            probabilities[predicted] = 1.0
        return ScoreResult(
            probabilities=probabilities,
            prompt_tokens=len(prompt.split()),
            candidate_tokens=sum(len(action.split()) for action in candidates),
            wall_s=0.001,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            context_limit=32768,
        )


def _settings():
    return Settings(
        verbose=False,
        irl_cold_steps=1,
        irl_warm_steps=1,
    )


class InContextLlmTests(unittest.TestCase):
    def test_sm120_uses_only_qualified_bitsandbytes_backend(self):
        torch = SimpleNamespace(
            __version__="2.13.0+cu130",
            cuda=SimpleNamespace(get_device_capability=lambda: (12, 0)),
        )
        policy = _bitsandbytes_kernel_policy(
            torch,
            SimpleNamespace(__version__="0.49.2"),
        )
        self.assertIn("eager_absmax", policy)
        self.assertIn("fresh_prefill_append_crop_candidates", policy)
        for version in ("0.50.0", "0.50.1"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(RuntimeError, "0.49.2"):
                    _bitsandbytes_kernel_policy(
                        torch,
                        SimpleNamespace(__version__=version),
                    )

        with self.assertRaisesRegex(RuntimeError, "PyTorch 2.13.0"):
            _bitsandbytes_kernel_policy(
                SimpleNamespace(
                    __version__="2.11.0+cu130",
                    cuda=SimpleNamespace(
                        get_device_capability=lambda: (12, 0)
                    ),
                ),
                SimpleNamespace(__version__="0.49.2"),
            )

    def test_nested_4bit_scales_are_materialized_once(self):
        class FakeTensor:
            def __init__(self, value, dtype="float16"):
                self.value = float(value)
                self.dtype = dtype

            def add_(self, other):
                self.value += float(other)
                return self

            def float(self):
                return FakeTensor(self.value, dtype="float32")

        nested = SimpleNamespace(
            absmax=FakeTensor(2.0),
            state2=object(),
            offset=3.0,
            nested=True,
        )
        already_flat = SimpleNamespace(
            absmax=FakeTensor(7.0, dtype="float32"),
            state2=None,
            offset=None,
            nested=False,
        )
        model = SimpleNamespace(modules=lambda: (
            SimpleNamespace(weight=SimpleNamespace(quant_state=nested)),
            SimpleNamespace(weight=SimpleNamespace(quant_state=already_flat)),
            SimpleNamespace(weight=object()),
        ))
        fake_torch = SimpleNamespace(
            float32="float32",
            inference_mode=nullcontext,
        )
        fake_bnb = SimpleNamespace(functional=SimpleNamespace(
            dequantize_blockwise=lambda absmax, _state2: FakeTensor(
                absmax.value * 4.0
            ),
        ))

        count = _materialize_nested_absmax(model, fake_torch, fake_bnb)

        self.assertEqual(count, 1)
        self.assertEqual(nested.absmax.value, 11.0)
        self.assertEqual(nested.absmax.dtype, "float32")
        self.assertFalse(nested.nested)
        self.assertIsNone(nested.state2)
        self.assertIsNone(nested.offset)
        self.assertEqual(already_flat.absmax.value, 7.0)

    def test_direct_4bit_inference_bypasses_autograd_wrapper(self):
        calls = []

        class FakeTensor:
            def __init__(self, name, values, dtype="bfloat16"):
                self.name = name
                self.values = np.asarray(values)
                self.dtype = dtype
                self.data = self

            def to(self, dtype):
                calls.append(("to", self.name, dtype))
                return FakeTensor(self.name, self.values, dtype)

            def reshape(self, *shape):
                if len(shape) == 1 and isinstance(shape[0], tuple):
                    shape = shape[0]
                return FakeTensor(
                    f"{self.name}.reshape",
                    self.values.reshape(*shape),
                    self.dtype,
                )

            def t(self):
                return FakeTensor(
                    f"{self.name}.t",
                    self.values.T,
                    self.dtype,
                )

            def numel(self):
                return int(self.values.size)

            def long(self):
                return FakeTensor(
                    f"{self.name}.long",
                    self.values.astype(np.int64),
                    "long",
                )

            def repeat_interleave(self, count):
                return FakeTensor(
                    f"{self.name}.repeat",
                    np.repeat(self.values, int(count)),
                    self.dtype,
                )

            def __rshift__(self, bits):
                return FakeTensor(
                    f"{self.name}.high",
                    self.values.astype(np.uint8) >> bits,
                    self.dtype,
                )

            def __and__(self, value):
                return FakeTensor(
                    f"{self.name}.low",
                    self.values.astype(np.uint8) & value,
                    self.dtype,
                )

            def __getitem__(self, index):
                values = (
                    index.values
                    if isinstance(index, FakeTensor)
                    else index
                )
                return FakeTensor(
                    f"{self.name}.indexed",
                    self.values[values],
                    self.dtype,
                )

            def __mul__(self, other):
                return FakeTensor(
                    f"{self.name}.scaled",
                    self.values * other.values,
                    self.dtype,
                )

        class FakeLinear4bit:
            def __init__(self):
                self.weight = FakeTensor(
                    "quantized_weight",
                    [0x12, 0x34, 0x56, 0x78],
                    "uint8",
                )
                self.weight.quant_state = SimpleNamespace(
                    nested=False,
                    quant_type="nf4",
                    shape=(2, 4),
                    blocksize=8,
                    code=FakeTensor("code", np.arange(16), "float32"),
                    absmax=FakeTensor("absmax", [2.0], "float32"),
                    dtype="bfloat16",
                )
                self.bias = None
                self.compute_dtype = None
                self.compute_type_is_set = False

            def set_compute_type(self, x):
                calls.append(("set_compute_type", x.dtype))
                self.compute_dtype = x.dtype

            def forward(self, _x):
                raise AssertionError("old bitsandbytes wrapper was reached")

        module = FakeLinear4bit()
        model = SimpleNamespace(modules=lambda: (module, object()))
        fake_bnb = SimpleNamespace(
            nn=SimpleNamespace(Linear4bit=FakeLinear4bit),
            functional=SimpleNamespace(
                dequantize_4bit=lambda weight, state: FakeTensor(
                    "dequantized_weight",
                    2.0 * np.arange(1, 9).reshape(2, 4).T,
                    state.dtype,
                ),
            ),
        )
        fake_torch = SimpleNamespace(
            nn=SimpleNamespace(functional=SimpleNamespace(
                linear=lambda x, weight, bias: (
                    calls.append((
                        "linear", x.name, weight.values.copy(), bias,
                    ))
                    or FakeTensor("output", [1.0], x.dtype)
                ),
            )),
        )

        installed = _install_direct_4bit_inference(
            model,
            fake_torch,
            fake_bnb,
        )
        output = module.forward(FakeTensor("input", [1.0]))

        self.assertEqual(installed, 1)
        self.assertTrue(module._hrc_direct_4bit_inference)
        self.assertEqual(output.name, "output")
        self.assertIn("linear", [call[0] for call in calls])
        linear_call = next(call for call in calls if call[0] == "linear")
        np.testing.assert_array_equal(
            linear_call[2],
            2.0 * np.arange(1, 9).reshape(2, 4),
        )

    def test_scorer_restores_prompt_cache_between_candidates_and_caches_result(self):
        class FakeTensor:
            def __init__(self, values=(), device="cuda"):
                self.values = np.asarray(values)
                self.device = device

            def tolist(self):
                return self.values.tolist()

            def float(self):
                return self

            def __getitem__(self, index):
                return FakeTensor(self.values[index], device=self.device)

            def unsqueeze(self, axis):
                return FakeTensor(np.expand_dims(self.values, axis), self.device)

            def squeeze(self, axis):
                return FakeTensor(np.squeeze(self.values, axis), self.device)

            def gather(self, axis, indices):
                return FakeTensor(
                    np.take_along_axis(
                        self.values,
                        indices.values.astype(int),
                        axis=axis,
                    ),
                    self.device,
                )

            def sum(self):
                return FakeTensor(np.asarray(self.values.sum()), self.device)

            def item(self):
                return float(self.values.item())

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        class FakeTorch:
            long = "long"

            @staticmethod
            def tensor(values, **kwargs):
                return FakeTensor(values, device=kwargs.get("device", "cuda"))

            @staticmethod
            def ones_like(tensor):
                return FakeTensor(np.ones_like(tensor.values), device=tensor.device)

            @staticmethod
            def ones(shape, **kwargs):
                return FakeTensor(np.ones(shape), device=kwargs.get("device", "cuda"))

            @staticmethod
            def inference_mode():
                return InferenceMode()

            @staticmethod
            def log_softmax(tensor, dim=-1):
                peak = np.max(tensor.values, axis=dim, keepdims=True)
                shifted = tensor.values - peak
                values = shifted - np.log(
                    np.exp(shifted).sum(axis=dim, keepdims=True)
                )
                return FakeTensor(values, device=tensor.device)

        rendered_messages = []

        class FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                rendered_messages.append((messages, kwargs))
                return {"input_ids": [[1, 2, 3]]}

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [ord(text)] if len(text) == 1 else [1, 2]

            @staticmethod
            def decode(tokens):
                return chr(tokens[0])

        class FakeLayer:
            def __init__(self, length):
                self.keys = SimpleNamespace(shape=(1, 1, length, 1))
                self.values = SimpleNamespace(shape=(1, 1, length, 1))

        class FakeCache:
            def __init__(self, length):
                self.layers = [FakeLayer(length)]

            def get_seq_length(self):
                return self.layers[0].keys.shape[-2]

            def crop(self, length):
                target_length = (
                    self.get_seq_length() + int(length)
                    if int(length) < 0 else int(length)
                )
                self.layers[0].keys = SimpleNamespace(
                    shape=(1, 1, target_length, 1)
                )
                self.layers[0].values = SimpleNamespace(
                    shape=(1, 1, target_length, 1)
                )

        prompt_caches = []
        candidate_caches = []

        def fake_forward(**kwargs):
            kept = int(kwargs["logits_to_keep"])
            output = SimpleNamespace(logits=FakeTensor(np.zeros((1, kept, 4))))
            if "past_key_values" not in kwargs:
                cache = FakeCache(kwargs["input_ids"].values.shape[-1])
                prompt_caches.append(cache)
                output.past_key_values = cache
            else:
                cache = kwargs["past_key_values"]
                candidate_caches.append(cache)
                new_length = (
                    cache.get_seq_length()
                    + kwargs["input_ids"].values.shape[-1]
                )
                cache.layers[0].keys = SimpleNamespace(
                    shape=(1, 1, new_length, 1)
                )
                cache.layers[0].values = SimpleNamespace(
                    shape=(1, 1, new_length, 1)
                )
            return output

        model = MagicMock(side_effect=fake_forward)
        model.config = SimpleNamespace(max_position_embeddings=32768)
        model.get_input_embeddings.return_value = SimpleNamespace(
            weight=SimpleNamespace(device="cuda")
        )
        scorer = QwenActionScorer()
        scorer.torch = FakeTorch()
        scorer.tokenizer = FakeTokenizer()
        scorer.model = model

        first = scorer.score("context", "query", ("action one", "action two"))
        second = scorer.score("context", "query", ("action one", "action two"))

        self.assertEqual(model.call_count, 3)
        prompt_call, *suffix_calls = model.call_args_list
        self.assertTrue(prompt_call.kwargs["use_cache"])
        self.assertNotIn("past_key_values", prompt_call.kwargs)
        self.assertEqual(prompt_caches[0].get_seq_length(), 3)
        self.assertEqual(len(candidate_caches), 2)
        self.assertIs(candidate_caches[0], prompt_caches[0])
        self.assertIs(candidate_caches[1], prompt_caches[0])
        for call in suffix_calls:
            self.assertTrue(call.kwargs["use_cache"])
            self.assertIn("past_key_values", call.kwargs)
        self.assertEqual(
            first.scoring_method,
            "append_crop_mean_semantic_action_log_likelihood",
        )
        self.assertEqual(first.model_forwards, 3)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.model_forwards, 0)
        self.assertEqual(first.probabilities, second.probabilities)
        self.assertEqual(len(rendered_messages[0][0]), 2)
        self.assertEqual(rendered_messages[0][0][0]["role"], "system")
        self.assertIn("action one", rendered_messages[0][0][1]["content"])

    @staticmethod
    def _vram_torch(total_gib, free_gib):
        """Minimal CUDA surface for the VRAM budget helpers."""
        recorded = {}

        def set_fraction(fraction):
            recorded["fraction"] = float(fraction)

        return SimpleNamespace(
            cuda=SimpleNamespace(
                mem_get_info=lambda: (
                    int(free_gib * 1024 ** 3), int(total_gib * 1024 ** 3),
                ),
                set_per_process_memory_fraction=set_fraction,
                synchronize=lambda: None,
                empty_cache=lambda: None,
                memory_allocated=lambda: 0,
            ),
        ), recorded

    def test_display_vram_is_reserved_from_the_process_cap(self):
        """A GPU that cannot serve the compositor takes the session down, so
        this process must never be allowed to claim the whole device."""
        torch, recorded = self._vram_torch(total_gib=12.0, free_gib=11.3)

        budget = reserve_display_vram(torch, headroom_gib=2.0)

        self.assertAlmostEqual(budget["external_gib"], 0.7, places=2)
        self.assertAlmostEqual(budget["headroom_gib"], 2.0, places=6)
        self.assertAlmostEqual(budget["allowed_gib"], 9.3, places=2)
        # The cap must exclude both the display reserve and memory already held.
        self.assertAlmostEqual(recorded["fraction"], 9.3 / 12.0, places=3)
        self.assertLess(recorded["fraction"], 1.0)

    def test_exhausted_vram_is_refused_before_any_allocation(self):
        torch, recorded = self._vram_torch(total_gib=12.0, free_gib=1.0)

        with self.assertRaisesRegex(RuntimeError, "no VRAM budget remains"):
            reserve_display_vram(torch, headroom_gib=2.0)
        self.assertNotIn("fraction", recorded)

    def test_headroom_of_zero_still_excludes_other_processes(self):
        torch, recorded = self._vram_torch(total_gib=12.0, free_gib=9.0)

        budget = reserve_display_vram(torch, headroom_gib=0.0)

        self.assertAlmostEqual(budget["allowed_gib"], 9.0, places=2)
        self.assertAlmostEqual(recorded["fraction"], 0.75, places=3)

    def test_kv_cache_per_token_matches_the_checkpoint_geometry(self):
        config = SimpleNamespace(
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
        )

        per_token = kv_cache_bytes_per_token(config)

        # 2 (K+V) * 36 layers * 8 kv heads * 128 dim * 2 bytes.
        self.assertEqual(per_token, 147456)
        # A full 32k prompt is 4.5GiB of cache, which is the whole problem.
        self.assertAlmostEqual(per_token * 32768 / 1024 ** 3, 4.5, places=2)

    def test_head_dim_is_derived_when_the_config_omits_it(self):
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
        )

        self.assertEqual(kv_cache_bytes_per_token(config), 2 * 2 * 2 * 64 * 2)

    def test_unsizable_config_is_refused_rather_than_run_unbounded(self):
        with self.assertRaisesRegex(RuntimeError, "cannot size the key/value"):
            kv_cache_bytes_per_token(SimpleNamespace())

    def test_context_limit_is_capped_by_the_vram_budget(self):
        """The model window is not a memory bound; free VRAM is."""
        torch, _recorded = self._vram_torch(total_gib=12.0, free_gib=4.5)
        scorer = QwenActionScorer(context_tokens=32768, vram_headroom_gib=2.0)
        scorer.torch = torch
        scorer.model = SimpleNamespace(config=SimpleNamespace(
            max_position_embeddings=40960,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
        ))
        # A cap generous enough that device-free memory is the binding ceiling.
        scorer.vram_budget = {"headroom_gib": 2.0, "fraction": 1.0}

        token_limit = scorer._vram_token_limit()
        scorer.vram_context_limit = token_limit

        # 2.5GiB free, less the reserve, the measured fixed prefill cost and
        # the allocator slack, divided by the measured per-token cost.
        per_token, fixed = scorer._prefill_cost_model()
        self.assertEqual(
            token_limit,
            (int(2.5 * 1024 ** 3) - fixed - 384 * 1024 ** 2) // per_token,
        )
        self.assertLess(token_limit, 32768)
        self.assertEqual(scorer._context_limit(), token_limit)
        # Without a measured budget the previous behaviour is unchanged.
        scorer.vram_context_limit = 0
        self.assertEqual(scorer._context_limit(), 32768)

    def test_doctor_budget_report_separates_display_from_capacity(self):
        """`./hrc doctor` has to answer "will a full run fit" before one starts."""
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            mem_get_info=lambda: (
                int(10.54 * 1024 ** 3), int(11.5 * 1024 ** 3),
            ),
        )
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoConfig = SimpleNamespace(
            from_pretrained=lambda *_a, **_k: SimpleNamespace(
                num_hidden_layers=36, num_attention_heads=32,
                num_key_value_heads=8, hidden_size=4096, head_dim=128,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model-00001-of-00001.safetensors"
            weights.write_bytes(b"\0" * (7 * 1024 ** 2))
            with (
                patch.dict(sys.modules, {
                    "torch": fake_torch, "transformers": fake_transformers,
                }),
                patch(
                    "src.llm_baseline.resolve_model_path",
                    return_value=(directory, "snapshot"),
                ),
            ):
                report = report_vram_budget(headroom_gib=1.25)

        self.assertAlmostEqual(report["other_clients_gib"], 0.96, places=2)
        self.assertEqual(report["bytes_per_prompt_token"], 2 * 147456)
        # Reclaiming the display's memory has to show as more prompt capacity.
        self.assertGreater(
            report["prompt_token_estimate_if_gpu_exclusive"],
            report["prompt_token_estimate"],
        )

    def test_weights_larger_than_the_cap_are_refused_before_loading(self):
        """A busy GPU must not fail half-way through materializing weights."""
        scorer = QwenActionScorer()
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model-00001-of-00001.safetensors"
            weights.write_bytes(b"\0" * (8 * 1024 ** 2))

            # A budget smaller than the checkpoint on disk.
            scorer.vram_budget = {"allowed_gib": 4 / 1024, "headroom_gib": 1.25}
            with self.assertRaisesRegex(RuntimeError, "do not fit"):
                scorer._check_weights_fit_the_cap(directory)

            # A budget that holds the weights proceeds.
            scorer.vram_budget = {"allowed_gib": 1.0, "headroom_gib": 1.25}
            scorer._check_weights_fit_the_cap(directory)

            # An uninspectable source cannot block a load it knows nothing about.
            scorer._check_weights_fit_the_cap("publisher/model")

    def test_chunked_prefill_carries_the_cache_and_keeps_one_forward_path(self):
        """Each chunk must attend over every key already cached, and a prompt
        that fits one chunk must issue exactly the call it issued before."""
        calls = []

        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def __getitem__(self, _index):
                return self

            def float(self):
                return self

        def fake_forward(**kwargs):
            calls.append(kwargs)
            length = kwargs["input_ids"].shape[-1]
            prior = (
                kwargs["past_key_values"].length
                if kwargs.get("past_key_values") is not None else 0
            )
            cache = SimpleNamespace(length=prior + length)
            cache.get_seq_length = lambda cache=cache: cache.length
            return SimpleNamespace(
                logits=FakeTensor((1, 1, 4)), past_key_values=cache,
            )

        scorer = QwenActionScorer(prefill_chunk_tokens=4)
        scorer.torch = SimpleNamespace(
            long="long",
            inference_mode=nullcontext,
            tensor=lambda values, **_k: FakeTensor((1, len(values[0]))),
            ones=lambda shape, **_k: FakeTensor(shape),
            log_softmax=lambda values, dim: values,
        )
        scorer.model = MagicMock(side_effect=fake_forward)

        cache, _log_probs, chunks = scorer._prefill(list(range(10)), "cuda")

        self.assertEqual(chunks, 3)
        self.assertEqual(cache.get_seq_length(), 10)
        self.assertNotIn("past_key_values", calls[0])
        # The attention mask must span everything cached so far, not the chunk.
        self.assertEqual([call["attention_mask"].shape for call in calls],
                         [(1, 4), (1, 8), (1, 10)])
        for call in calls[1:]:
            self.assertIsNotNone(call["past_key_values"])

        # A prompt inside one chunk keeps the unchunked call signature.
        calls.clear()
        scorer.prefill_chunk_tokens = 1024
        _cache, _lp, chunks = scorer._prefill(list(range(10)), "cuda")
        self.assertEqual(chunks, 1)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("past_key_values", calls[0])

    def test_allocator_cap_bounds_the_limit_even_with_free_device_memory(self):
        """Device-free memory does not know this process is capped; using it
        alone reported a limit that then failed to allocate."""
        torch, _recorded = self._vram_torch(total_gib=12.0, free_gib=11.0)
        scorer = QwenActionScorer(context_tokens=32768, vram_headroom_gib=0.5)
        scorer.torch = torch
        scorer.model = SimpleNamespace(config=SimpleNamespace(
            max_position_embeddings=40960, num_hidden_layers=36,
            num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, head_dim=128,
        ))
        # Weights already hold most of a cap well below device-free memory.
        torch.cuda.memory_allocated = lambda: int(7.23 * 1024 ** 3)
        scorer.vram_budget = {"headroom_gib": 0.5, "fraction": 8.0 / 12.0}

        capped = scorer._vram_token_limit()

        # Raising only the cap must raise the limit, proving the cap binds.
        scorer.vram_budget = {"headroom_gib": 0.5, "fraction": 10.0 / 12.0}
        self.assertGreater(scorer._vram_token_limit(), capped)
        # A cap fully consumed by the weights leaves no prompt budget.
        scorer.vram_budget = {"headroom_gib": 0.5, "fraction": 7.3 / 12.0}
        self.assertEqual(scorer._vram_token_limit(), 0)

    def test_prefill_cost_model_reflects_the_chunking_mode(self):
        """Chunking changes the slope; the chunk sets the intercept."""
        scorer = QwenActionScorer()
        scorer.model = SimpleNamespace(config=SimpleNamespace(
            num_hidden_layers=36, num_attention_heads=32,
            num_key_value_heads=8, hidden_size=4096, head_dim=128,
        ))

        scorer.prefill_chunk_tokens = 0
        unchunked_slope, unchunked_fixed = scorer._prefill_cost_model()
        scorer.prefill_chunk_tokens = 1024
        chunked_slope, chunked_fixed = scorer._prefill_cost_model()

        self.assertAlmostEqual(unchunked_slope / 1024, 240.5, places=0)
        self.assertAlmostEqual(chunked_slope / 1024, 153.0, places=0)
        self.assertLess(chunked_slope, unchunked_slope)
        self.assertGreater(chunked_fixed, unchunked_fixed)
        # Both must charge more than the bare cache.
        self.assertGreater(chunked_slope, 147456)

    def test_pinned_encoding_is_not_silently_shed(self):
        """A pin exists so one run cannot mix encodings; honour it."""
        builders = recipe_builders()
        name = next(iter(builders))

        class RefusingScorer(DemoScorer):
            def score(self, context_prompt, query_prompt, candidates):
                raise PromptTooLongError(99999, 100)

        for policy, expect_raise in (
            ("auto", False), ("state_delta", True), ("action_only", True),
        ):
            with self.subTest(policy=policy):
                agent = InContextLlmAgent(
                    replace(_settings(), llm_context_encoding=policy),
                    scorer=RefusingScorer(),
                )
                observe_demo(agent, build_task(name, "default", builders[name]), {})
                prefix = [next(iter(agent.llm_actions))]
                if expect_raise:
                    with self.assertRaises(PromptTooLongError):
                        agent.predict_actions(prefix)
                else:
                    # "auto" retries on the compact context, which also refuses.
                    with self.assertRaises(PromptTooLongError):
                        agent.predict_actions(prefix)
                    self.assertEqual(len(agent.scorer.calls), 0)

    def test_pinned_action_only_never_sends_state_annotation(self):
        builders = recipe_builders()
        name = next(iter(builders))
        agent = InContextLlmAgent(
            replace(_settings(), llm_context_encoding="action_only"),
            scorer=DemoScorer(),
        )
        observe_demo(agent, build_task(name, "default", builders[name]), {})

        agent.predict_actions([next(iter(agent.llm_actions))])

        sent = agent.scorer.calls[-1][0]
        self.assertNotIn("state_delta", sent)
        self.assertEqual(
            agent.last_score_stats["llm_context_encoding"], "action_only_pinned",
        )
        self.assertEqual(
            agent.baseline_stats()["context_encoding_policy"], "action_only",
        )

    def test_missing_logits_contract_fails_the_load(self):
        """Losing logits_to_keep materializes logits for every position."""
        class WithoutContract:
            def forward(self, input_ids=None, **kwargs):
                return None

        class WithContract:
            def forward(self, input_ids=None, logits_to_keep=None, **kwargs):
                return None

        scorer = QwenActionScorer()
        scorer.model = WithoutContract()
        with self.assertRaisesRegex(RuntimeError, "logits_to_keep"):
            scorer._verify_logits_contract()

        scorer.model = WithContract()
        scorer._verify_logits_contract()

    def test_device_out_of_memory_becomes_a_prompt_budget_overrun(self):
        """The cap makes OOM reachable; the agent then sheds the annotation
        instead of the run dying or the display hanging."""
        class FakeOom(RuntimeError):
            pass

        released = []
        torch, _recorded = self._vram_torch(total_gib=12.0, free_gib=4.5)
        torch.OutOfMemoryError = FakeOom
        torch.cuda.empty_cache = lambda: released.append("empty_cache")
        scorer = QwenActionScorer(vram_headroom_gib=2.0)
        scorer.torch = torch
        scorer.model = SimpleNamespace(config=SimpleNamespace(
            num_hidden_layers=36, num_attention_heads=32,
            num_key_value_heads=8, hidden_size=4096, head_dim=128,
        ))
        scorer.vram_budget = {"headroom_gib": 2.0}
        scorer._last_prompt_tokens = 12345

        def boom(*_args, **_kwargs):
            raise FakeOom("CUDA out of memory")

        scorer._score = boom

        with self.assertRaises(PromptTooLongError) as caught:
            scorer.score("context", "query", ("a",))

        self.assertEqual(caught.exception.required_tokens, 12345)
        self.assertGreater(caught.exception.context_limit, 0)
        # The allocator's memory must be recovered before the retry.
        self.assertIn("empty_cache", released)

    def test_scoring_runtime_errors_are_not_disguised_as_budget_overruns(self):
        """A cache-integrity failure must not be hidden by a smaller prompt."""
        class FakeOom(RuntimeError):
            pass

        torch, _recorded = self._vram_torch(total_gib=12.0, free_gib=4.5)
        torch.OutOfMemoryError = FakeOom
        scorer = QwenActionScorer()
        scorer.torch = torch

        def boom(*_args, **_kwargs):
            raise RuntimeError("candidate scoring mutated the shared prompt cache")

        scorer._score = boom

        with self.assertRaisesRegex(RuntimeError, "mutated the shared prompt cache"):
            scorer.score("context", "query", ("a",))

    def test_agent_passes_the_configured_vram_headroom_to_the_scorer(self):
        agent = build_agent(
            "in_context_llm", replace(_settings(), llm_vram_headroom_gib=3.5),
        )

        self.assertEqual(agent.scorer.vram_headroom_gib, 3.5)
        self.assertEqual(
            agent.baseline_stats()["vram_headroom_gib"], 3.5,
        )

    def test_message_ids_accepts_list_and_batch_encoding_contracts(self):
        cases = (
            ([11, 12, 13], [11, 12, 13]),
            ({"input_ids": [21, 22], "attention_mask": [1, 1]}, [21, 22]),
            ({"input_ids": [[31, 32]]}, [31, 32]),
        )
        for encoded, expected in cases:
            with self.subTest(encoded=encoded):
                scorer = QwenActionScorer()
                scorer.tokenizer = SimpleNamespace(
                    apply_chat_template=MagicMock(return_value=encoded),
                )

                actual = scorer._message_ids(
                    "user",
                    "test",
                    add_generation_prompt=True,
                )

                self.assertEqual(actual, expected)

    def test_model_resolution_finds_standard_huggingface_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            snapshot = (
                cache_root
                / "models--unsloth--Qwen3-8B-unsloth-bnb-4bit"
                / "snapshots"
                / "test-snapshot"
            )
            snapshot.mkdir(parents=True)
            with patch("src.llm_baseline._cache_roots", return_value=(cache_root,)):
                source, label = resolve_model_path(None)

        self.assertEqual(source, str(snapshot))
        self.assertEqual(label, "test-snapshot")

    def test_uncached_hub_model_is_allowed_to_download(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (int(11.3 * 1024 ** 3), int(12 * 1024 ** 3)),
            set_per_process_memory_fraction=lambda _fraction: None,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            memory_allocated=lambda: int(6.5 * 1024 ** 3),
        )
        fake_transformers = ModuleType("transformers")
        tokenizer_loader = MagicMock()
        tokenizer_loader.from_pretrained.return_value = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=0,
        )
        model = MagicMock()
        # The load path now sizes a VRAM budget and verifies the logits
        # contract, so the double needs a real config and forward signature.
        model.config = SimpleNamespace(
            max_position_embeddings=40960,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
        )
        model.forward = (
            lambda input_ids=None, logits_to_keep=None, **kwargs: None
        )
        model_loader = MagicMock()
        model_loader.from_pretrained.return_value = model
        fake_transformers.AutoTokenizer = tokenizer_loader
        fake_transformers.AutoModelForCausalLM = model_loader
        fake_bitsandbytes = ModuleType("bitsandbytes")
        fake_bitsandbytes.__version__ = "0.49.2"
        scorer = QwenActionScorer()

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                    "bitsandbytes": fake_bitsandbytes,
                },
            ),
            patch(
                "src.llm_baseline.resolve_model_path",
                return_value=("publisher/model", "hub_download_or_cache"),
            ),
            patch("src.llm_baseline._offline_mode_enabled", return_value=False),
        ):
            scorer._load()

        self.assertFalse(scorer.local_files_only)
        self.assertFalse(
            tokenizer_loader.from_pretrained.call_args.kwargs["local_files_only"]
        )
        self.assertFalse(
            model_loader.from_pretrained.call_args.kwargs["local_files_only"]
        )
        model.eval.assert_called_once_with()

    def test_prompt_contains_replay_examples_but_no_task_oracle(self):
        tracker = StateTracker()
        state = tuple(int(value) for value in tracker.get_state_vector())
        action = "turn_on (stove, cooking_station)"
        tracker.apply_action(action)
        reached = tuple(int(value) for value in tracker.get_state_vector())
        demonstrations = (PromptDemo(0.75, ((state, action),), reached),)

        prompt = build_prompt(
            demonstrations,
            state,
            (),
            candidate_actions(demonstrations),
        )

        self.assertIn(action, prompt)
        self.assertIn('"memory_weight":0.75', prompt)
        context_text, query_text = prompt.split("\nQUERY:\n", 1)
        self.assertIn("initial_state_predicates", context_text)
        self.assertIn("current_state_predicates", query_text)
        for forbidden in (
            "goal_predicates",
            "recipe_name",
            "preference_name",
            "action schemas",
            "preconditions/effects",
        ):
            self.assertNotIn(forbidden, prompt.lower())

    def test_context_carries_the_state_change_each_action_caused(self):
        """The other arms train on (state, action) pairs; so must this one."""
        tracker = StateTracker()
        state = tuple(int(value) for value in tracker.get_state_vector())
        action = "turn_on (stove, cooking_station)"
        tracker.apply_action(action)
        reached = tuple(int(value) for value in tracker.get_state_vector())
        self.assertNotEqual(state, reached)
        demonstrations = (PromptDemo(0.75, ((state, action),), reached),)

        context = json.loads(
            build_context_prompt(
                demonstrations, candidate_actions(demonstrations),
            ).split("CONTEXT:\n", 1)[1]
        )

        steps = context["previous_demonstrations"][0]["steps"]
        self.assertEqual([step["action"] for step in steps], [action])
        # The rendered delta must reconstruct the state the action reached.
        rebuilt = set(context["initial_state_predicates"])
        rebuilt -= set(steps[0]["state_delta"].get("-", ()))
        rebuilt |= set(steps[0]["state_delta"].get("+", ()))
        self.assertEqual(rebuilt, set(state_predicates(reached)))
        self.assertTrue(steps[0]["state_delta"])

    def test_state_annotation_is_shed_before_any_demonstration_is(self):
        """Memory coverage is the experimental variable; annotation is not."""
        builders = recipe_builders()

        class OverflowingScorer(DemoScorer):
            def __init__(self, limit):
                super().__init__()
                self.limit = limit

            def score(self, context_prompt, query_prompt, candidates):
                if len(context_prompt) > self.limit:
                    raise PromptTooLongError(len(context_prompt), self.limit)
                return super().score(context_prompt, query_prompt, candidates)

        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        for name in list(builders)[:3]:
            observe_demo(llm, build_task(name, "default", builders[name]), {})
        annotated = build_context_prompt(
            llm.prompt_demos, llm.llm_actions, llm.domain,
        )
        compact = build_context_prompt(
            llm.prompt_demos, llm.llm_actions, llm.domain,
            include_state_deltas=False,
        )
        self.assertGreater(len(annotated), len(compact))

        llm.scorer = OverflowingScorer(len(annotated) - 1)
        prefix = [next(iter(llm.llm_actions))]
        self.assertTrue(llm.predict_actions(prefix))

        sent, _query, _candidates = llm.scorer.calls[-1]
        self.assertEqual(sent, compact)
        self.assertEqual(
            llm.last_score_stats["llm_context_encoding"],
            "action_only_context_budget_exceeded",
        )
        # Every retained demonstration still reaches the model.
        payload = json.loads(sent.split("CONTEXT:\n", 1)[1])
        self.assertEqual(
            len(payload["previous_demonstrations"]), len(llm.prompt_demos),
        )
        self.assertEqual(llm.baseline_stats()["memory_coverage"], 1.0)

    def test_annotated_context_is_the_default_encoding(self):
        builders = recipe_builders()
        name = next(iter(builders))
        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        observe_demo(llm, build_task(name, "default", builders[name]), {})

        llm.predict_actions([next(iter(llm.llm_actions))])

        self.assertEqual(
            llm.last_score_stats["llm_context_encoding"], "per_step_state_delta",
        )
        self.assertIn("state_delta", llm.scorer.calls[-1][0])

    def test_rendering_a_demo_without_its_final_state_is_refused(self):
        """The last action's effect must not be silently dropped."""
        state = tuple(int(value) for value in StateTracker().get_state_vector())
        demonstrations = (PromptDemo(0.75, ((state, "turn_on (stove, cooking_station)"),)),)

        with self.assertRaisesRegex(ValueError, "final state"):
            build_context_prompt(
                demonstrations, candidate_actions(demonstrations),
            )

    def test_make_prompt_demos_keeps_the_terminal_state(self):
        tracker = StateTracker()
        first = tuple(int(value) for value in tracker.get_state_vector())
        action = "turn_on (stove, cooking_station)"
        tracker.apply_action(action)
        reached = tuple(int(value) for value in tracker.get_state_vector())
        # _build_demos terminates every trajectory with a "stop" token.
        trajectory = [(first, action), (reached, "stop")]

        demos = make_prompt_demos([trajectory], [0.5])

        self.assertEqual(len(demos), 1)
        self.assertEqual(demos[0].steps, ((first, action),))
        self.assertEqual(demos[0].final_state, reached)

    def test_answer_boundary_follows_the_candidate_actions(self):
        """An answer marker ahead of the options it constrains is stranded."""
        query = build_query_prompt(
            tuple(int(value) for value in StateTracker().get_state_vector()), (),
        )

        rendered = build_candidate_prompt(query, ("a", "b"))

        self.assertNotIn("ANSWER:", query)
        self.assertLess(
            rendered.index("CANDIDATE_ACTIONS:"), rendered.index("ANSWER:"),
        )
        self.assertTrue(rendered.rstrip().endswith("ANSWER:"))
        self.assertEqual(rendered.count("ANSWER:"), 1)

    def test_prompt_instructions_name_the_sections_they_reference(self):
        state = tuple(int(value) for value in StateTracker().get_state_vector())
        tracker = StateTracker()
        action = "turn_on (stove, cooking_station)"
        tracker.apply_action(action)
        demonstrations = (PromptDemo(
            0.75,
            ((state, action),),
            tuple(int(value) for value in tracker.get_state_vector()),
        ),)

        prompt = build_prompt(
            demonstrations, state, (), candidate_actions(demonstrations),
        )

        instructions = prompt.split("CONTEXT:", 1)[0]
        for section in ("previous_demonstrations", "CANDIDATE_ACTIONS", "memory_weight", "initial_state_predicates"):
            with self.subTest(section=section):
                self.assertIn(section, instructions)
                self.assertIn(section, prompt.split("CONTEXT:", 1)[1])

    def test_agent_is_lazy_and_uses_full_memory_policy(self):
        agent = build_agent("in_context_llm", _settings())

        self.assertIsInstance(agent, InContextLlmAgent)
        self.assertEqual(agent.settings.predictor, "in_context_llm")
        self.assertEqual(agent.replay.policy, "adaptive")
        self.assertTrue(agent.settings.pin_latest)
        self.assertIsNone(agent.scorer.model)

    def test_common_evaluator_scores_exact_actions_from_retained_demo(self):
        recipe_name, builder = next(iter(recipe_builders().items()))
        pair = build_task(recipe_name, "default", builder)
        scorer = DemoScorer()
        agent = InContextLlmAgent(_settings(), scorer=scorer)
        recipe_ids = {}

        observe_demo(agent, pair, recipe_ids)
        row = assist_demo(
            agent,
            pair,
            recipe_ids,
            config=EvalSettings(
                baselines=("full", "in_context_llm"),
                include_oracle=False,
                top_k=3,
                audit_period=0,
            ),
        )

        self.assertTrue(scorer.prompts)
        self.assertEqual(row["teacher_forced_top_1"], 1.0)
        self.assertEqual(row["live_top_1"], 1.0)
        self.assertEqual(row["hrc_human_correction_count"], 0)
        self.assertEqual(set(agent.llm_actions), {
            observation.action for observation in pair.observations
        })
        self.assertTrue(agent.audit_pruning()["passed"])
        stats = agent.policy_stats()
        self.assertEqual(stats["predictor"], "in_context_llm")
        # Contract invariants belong in the run manifest, not on every turn row.
        for invariant in ("llm_task_conditioned", "llm_memory_coverage"):
            self.assertNotIn(invariant, stats)
        self.assertFalse(agent.baseline_stats()["task_conditioned"])
        self.assertEqual(agent.baseline_stats()["memory_coverage"], 1.0)

    def test_prompt_weights_are_the_replay_retention_weights(self):
        """One number stands for a whole demonstration, so it must not be
        divided by episode length the way per-example fit weights are."""
        builders = recipe_builders()
        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        for name in list(builders)[:4]:
            observe_demo(llm, build_task(name, "default", builders[name]), {})

        entries = llm.replay.active_items()
        self.assertEqual(len(llm.prompt_demos), len(entries))
        self.assertEqual(
            [round(demo.weight, 9) for demo in llm.prompt_demos],
            [round(float(entry.weight), 9) for entry in entries],
        )
        # The length-equalized weights the numeric arms fit against are a
        # different quantity, and demos here differ in length.
        fit_weights = llm._demo_weights(
            entries, [float(entry.weight) for entry in entries],
        )
        self.assertNotEqual(
            [round(w, 9) for w in fit_weights],
            [round(demo.weight, 9) for demo in llm.prompt_demos],
        )
        self.assertGreater(len({len(demo.steps) for demo in llm.prompt_demos}), 1)

    def test_build_prompt_reproduces_the_prompt_that_is_scored(self):
        builders = recipe_builders()
        scorer = DemoScorer()
        llm = InContextLlmAgent(_settings(), scorer=scorer)
        for name in list(builders)[:3]:
            observe_demo(llm, build_task(name, "default", builders[name]), {})
        prefix = [next(iter(llm.llm_actions))]

        llm.predict_actions(prefix)

        context_prompt, query_prompt, candidates = scorer.calls[-1]
        scored = context_prompt + "\n" + QwenActionScorer._semantic_action_query(
            query_prompt, candidates,
        )
        rendered = build_prompt(
            llm.prompt_demos,
            llm._replay_prefix(prefix),
            prefix,
            candidates,
            llm.domain,
        )
        self.assertEqual(rendered, scored)
        # The audit must show the conditioned candidates, not the full support.
        self.assertLess(len(candidates), len(llm.llm_actions))

    def test_predict_actions_accepts_the_shared_predictor_interface(self):
        """Every other arm takes state=/action_universe=; real_robot passes them."""
        builders = recipe_builders()
        name = next(iter(builders))
        pair = build_task(name, "default", builders[name])
        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        full = build_agent("full", _settings())
        bc = build_agent("bc", _settings())
        for agent in (llm, full, bc):
            observe_demo(agent, pair, {})
        actions = [observation.action for observation in pair.observations]
        prefix = actions[:1]
        state = llm._replay_prefix(prefix)

        for label, agent in (("full", full), ("bc", bc), ("llm", llm)):
            with self.subTest(agent=label):
                self.assertTrue(agent.predict_actions(
                    prefix, state=state, action_universe=tuple(llm.llm_actions),
                ))
        # A narrowed universe must narrow the scored support.
        narrowed = llm.predict_actions(
            prefix, state=state, action_universe=(actions[1],),
        )
        self.assertEqual(set(narrowed), {actions[1]})

    def test_empty_feasible_support_degrades_instead_of_raising(self):
        """Full returns {} here; raising would end a run where others miss."""
        builders = recipe_builders()
        name = next(iter(builders))
        pair = build_task(name, "default", builders[name])
        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        full = build_agent("full", _settings())
        for agent in (llm, full):
            observe_demo(agent, pair, {})
        prefix = [observation.action for observation in pair.observations][:1]
        state = llm._replay_prefix(prefix)

        self.assertEqual(
            full.predict_actions(prefix, state=state, action_universe=()), {},
        )
        self.assertEqual(
            llm.predict_actions(prefix, state=state, action_universe=()), {},
        )
        self.assertEqual(llm.policy_stats()["reason"], "no_feasible_actions")

    def test_cache_hit_reports_both_measured_and_uncached_inference_time(self):
        builders = recipe_builders()
        name = next(iter(builders))
        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        observe_demo(llm, build_task(name, "default", builders[name]), {})
        prefix = [next(iter(llm.llm_actions))]

        llm.predict_actions(prefix)
        stats = dict(llm.last_score_stats)

        self.assertEqual(
            stats["llm_uncached_inference_wall_s"],
            stats["llm_inference_wall_s"] + 0.0,
        )
        self.assertIn("llm_score_cache_hit", stats)

    def test_conditioned_candidates_match_full_prediction_support(self):
        builders = recipe_builders()
        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        full = build_agent("full", _settings())
        pairs = [
            build_task(name, "default", builders[name])
            for name in list(builders)[:4]
        ]
        for pair in pairs:
            observe_demo(llm, pair, {})
            observe_demo(full, pair, {})
        # The prompt's scoreable support must not be narrower than the shared
        # mask's universe, or this arm would face fewer distractors than Full.
        self.assertEqual(
            set(llm.llm_actions), set(llm._known_action_universe()),
        )
        prefixes = [
            tuple(observation.action for observation in pair.observations)[:length]
            for pair in pairs
            for length in (0, 1, 3, 7)
        ]
        for prefix in prefixes:
            with self.subTest(prefix_length=len(prefix)):
                state = llm._replay_prefix(prefix)
                self.assertEqual(
                    set(llm._conditioned_actions(state, llm.llm_actions)),
                    set(full.predict_actions(prefix)),
                )

    def test_dedicated_runner_forces_the_paired_shared_protocol(self):
        result = {
            "run_dir": "run",
            "latest_dir": "latest",
            "scenarios": {"homogeneous": {}},
        }
        with (
            patch("src.evaluation.run_evaluation", return_value=result) as run,
            patch(
                "src.llm_baseline.preflight_llm_runtime",
                return_value="qualified-test-runtime",
            ),
            patch("sys.stdout", new=io.StringIO()),
        ):
            main(["--seeds", "1337", "--scenarios", "homogeneous"])

        config = run.call_args.args[0]
        self.assertEqual(config.baselines, ("full", "in_context_llm"))
        self.assertTrue(config.shared_routing)
        self.assertFalse(config.include_oracle)
        self.assertEqual(config.workers, 1)
        self.assertEqual(config.seeds, (1337,))
        self.assertEqual(config.recipe_count, 20)
        self.assertEqual(config.frozen_pairs, 48)
        self.assertEqual(config.schedule.phases, 7)
        self.assertEqual(config.schedule.demos, 210)
        self.assertEqual(config.schedule.min_recipes, 5)
        self.assertEqual(config.schedule.max_recipes, 8)
        self.assertEqual(config.experiment, "llm_single_seed_evaluation")

        with self.assertRaisesRegex(ValueError, "shared Full interaction"):
            main(["--local-routing"])

    def test_dedicated_runner_can_skip_completed_full_baseline(self):
        result = {
            "run_dir": "run",
            "latest_dir": "latest",
            "scenarios": {"homogeneous": {}},
        }
        with (
            patch("src.evaluation.run_evaluation", return_value=result) as run,
            patch(
                "src.llm_baseline.preflight_llm_runtime",
                return_value="qualified-test-runtime",
            ),
            patch("sys.stdout", new=io.StringIO()),
        ):
            main([
                "--seeds", "1337",
                "--scenarios", "homogeneous",
                "--baselines", "in_context_llm",
                "--local-routing",
            ])

        config = run.call_args.args[0]
        self.assertEqual(config.baselines, ("in_context_llm",))
        self.assertFalse(config.shared_routing)
        self.assertFalse(config.observe_missing_recipes)
        self.assertFalse(config.include_oracle)
        self.assertEqual(config.workers, 1)

        with self.assertRaisesRegex(ValueError, "requires --local-routing"):
            main(["--baselines", "in_context_llm"])

    def test_dedicated_runner_rejects_multiple_seeds(self):
        with (
            patch("src.llm_baseline.preflight_llm_runtime") as preflight,
            self.assertRaisesRegex(ValueError, "exactly one seed"),
        ):
            main(["--seeds", "1337,2024"])
        preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class RunnerDefaultsTests(unittest.TestCase):
    """A bare invocation must be the configuration this hardware needs."""

    def _captured_config(self, argv):
        captured = {}

        def fake_run(config):
            captured["config"] = config
            return {"run_dir": "r", "latest_dir": "l", "scenarios": {}}

        with (
            patch("src.evaluation.run_evaluation", side_effect=fake_run),
            patch("src.llm_baseline.preflight_llm_runtime", return_value="ok"),
            patch("src.evaluation.find_resumable_run", return_value=None),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            main(argv)
        return captured["config"]

    def test_bare_invocation_runs_every_scenario_and_survives_crashes(self):
        config = self._captured_config([])

        self.assertEqual(
            config.scenarios, ("homogeneous", "heterogeneous", "holdout"),
        )
        self.assertEqual(config.baselines, ("full", "in_context_llm"))
        self.assertEqual(config.seeds, (LLM_EVALUATION_SEED,))
        # Without this a crash restarts the seed from its first event.
        self.assertTrue(config.event_resume)
        settings = dict(config.model_settings)
        self.assertEqual(settings["llm_prefill_chunk_tokens"], 1024)
        self.assertEqual(settings["llm_context_encoding"], "action_only")

    def test_explicit_flags_override_the_runner_defaults(self):
        config = self._captured_config([
            "--llm-prefill-chunk-tokens", "0",
            "--llm-context-encoding", "state_delta",
        ])

        settings = dict(config.model_settings)
        self.assertEqual(settings["llm_prefill_chunk_tokens"], 0)
        self.assertEqual(settings["llm_context_encoding"], "state_delta")

    def test_an_incomplete_run_is_continued_rather_than_replaced(self):
        captured = {}

        def fake_run(config):
            captured["config"] = config
            return {"run_dir": "r", "latest_dir": "l", "scenarios": {}}

        with (
            patch("src.evaluation.run_evaluation", side_effect=fake_run),
            patch("src.llm_baseline.preflight_llm_runtime", return_value="ok"),
            patch(
                "src.evaluation.find_resumable_run",
                return_value="llm-single-seed-evaluation__X__abc",
            ),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            main([])

        self.assertEqual(
            captured["config"].run, "llm-single-seed-evaluation__X__abc",
        )
        self.assertTrue(captured["config"].resume)
