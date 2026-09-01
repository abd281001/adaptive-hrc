"""Information-parity contracts for the in-context LLM baseline."""
from __future__ import annotations

import hashlib
import io
import json
from contextlib import nullcontext
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
    PromptDemo,
    ScoreResult,
    QwenActionScorer,
    _bitsandbytes_kernel_policy,
    _install_direct_4bit_inference,
    _materialize_nested_absmax,
    build_prompt,
    candidate_actions,
    main,
    resolve_model_path,
)
from src.models import Settings


class DemoScorer:
    """Deterministic prompt-only scorer; never imports or loads an LLM."""

    def __init__(self):
        self.prompts = []

    def score(self, context_prompt, query_prompt, candidates):
        prompt = context_prompt + "\n" + query_prompt
        self.prompts.append(prompt)
        context = json.loads(context_prompt.split("CONTEXT:\n", 1)[1])
        query = json.loads(
            query_prompt.split("QUERY:\n", 1)[1].rsplit("\nANSWER:", 1)[0]
        )
        prefix = query["observed_actions"]
        predicted = None
        for demonstration in context["previous_demonstrations"]:
            actions = demonstration["actions"]
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
        fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
        fake_transformers = ModuleType("transformers")
        tokenizer_loader = MagicMock()
        tokenizer_loader.from_pretrained.return_value = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=0,
        )
        model = MagicMock()
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
        state = tuple(int(value) for value in StateTracker().get_state_vector())
        action = "turn_on (stove)"
        demonstrations = (PromptDemo(0.75, ((state, action),)),)

        prompt = build_prompt(
            demonstrations,
            state,
            (),
            candidate_actions(demonstrations),
        )

        self.assertIn(action, prompt)
        self.assertIn('"memory_weight":0.75', prompt)
        context_text, query_text = prompt.split("\nQUERY:\n", 1)
        self.assertNotIn("state_predicates", context_text)
        self.assertIn("current_state_predicates", query_text)
        for forbidden in (
            "goal_predicates",
            "recipe_name",
            "preference_name",
            "action schemas",
            "preconditions/effects",
        ):
            self.assertNotIn(forbidden, prompt.lower())

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
        self.assertFalse(stats["llm_task_conditioned"])
        self.assertEqual(stats["llm_memory_coverage"], 1.0)

    def test_conditioned_candidates_match_full_prediction_support(self):
        recipe_name, builder = next(iter(recipe_builders().items()))
        pair = build_task(recipe_name, "default", builder)
        llm = InContextLlmAgent(_settings(), scorer=DemoScorer())
        full = build_agent("full", _settings())
        observe_demo(llm, pair, {})
        observe_demo(full, pair, {})
        actions = [observation.action for observation in pair.observations]
        for prefix in ((), tuple(actions[:1]), tuple(actions[:3])):
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


if __name__ == "__main__":
    unittest.main()
