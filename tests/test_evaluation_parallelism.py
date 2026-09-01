import os
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import patch

from src import evaluation


class EvaluationParallelismTests(unittest.TestCase):
    def test_workers_use_clean_processes(self):
        self.assertEqual(evaluation._mp_context().get_start_method(), "spawn")

    def test_worker_count_is_capped_by_seed_count(self):
        settings = evaluation.EvalSettings(seeds=(1, 2, 3, 4, 5), workers=20)
        self.assertEqual(evaluation._worker_count(settings), 5)

    def test_worker_count_defaults_to_one_worker_per_seed(self):
        settings = evaluation.EvalSettings(seeds=(1, 2, 3, 4, 5), workers=0)
        self.assertEqual(evaluation._worker_count(settings), 5)

    def test_in_context_llm_uses_one_gpu_worker(self):
        settings = evaluation.EvalSettings(
            seeds=(1, 2, 3),
            baselines=("full", "in_context_llm"),
            workers=3,
        )
        self.assertEqual(evaluation._worker_count(settings), 1)

    def test_in_context_llm_does_not_retry_a_native_worker_failure(self):
        attempts = []
        expected = {
            "scenario": "homogeneous",
            "seed": 1337,
            "key": "homogeneous/0000001337",
        }

        class FakeFuture:
            def result(self):
                attempts.append(1)
                raise BrokenProcessPool("native worker exited")

        class FakeExecutor:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, *_args):
                return FakeFuture()

        settings = evaluation.EvalSettings(
            seeds=(1337,),
            scenarios=("homogeneous",),
            baselines=("full", "in_context_llm"),
            workers=1,
        )
        with patch.object(evaluation, "ProcessPoolExecutor", FakeExecutor):
            with self.assertRaisesRegex(RuntimeError, "automatic retry is disabled"):
                list(evaluation._pending_seed_results(
                    "homogeneous",
                    settings.seeds,
                    settings,
                    Path("/tmp/llm-retry-test"),
                    workers=1,
                ))

        self.assertEqual(len(attempts), 1)

    def test_llm_allocator_policy_does_not_override_user_configuration(self):
        previous = os.environ.get("PYTORCH_ALLOC_CONF")
        try:
            os.environ["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
            runtime = evaluation._configure_llm_runtime(
                evaluation.EvalSettings(
                    baselines=("full", "in_context_llm"),
                )
            )
            self.assertEqual(
                runtime["pytorch_allocator_config"],
                "backend:cudaMallocAsync",
            )
            self.assertTrue(runtime["seed_process_isolation"])
            self.assertEqual(
                runtime["native_retry_attempts"],
                0,
            )
        finally:
            if previous is None:
                os.environ.pop("PYTORCH_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_ALLOC_CONF"] = previous

    def test_native_thread_limit_overrides_parent_environment(self):
        previous = {var: os.environ.get(var) for var in evaluation.NATIVE_THREAD_ENV_VARS}
        try:
            for var in evaluation.NATIVE_THREAD_ENV_VARS:
                os.environ[var] = "20"

            info = evaluation._apply_native_thread_limit(1)

            self.assertEqual(info["threads"], 1)
            for var in evaluation.NATIVE_THREAD_ENV_VARS:
                self.assertEqual(os.environ[var], "1")
                self.assertEqual(info["env"][var], "1")
        finally:
            for var, value in previous.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value


if __name__ == "__main__":
    unittest.main()
