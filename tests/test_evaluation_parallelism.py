import os
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import patch

from src import evaluation


class EvaluationParallelismTests(unittest.TestCase):
    def test_workers_use_clean_processes(self):
        self.assertEqual(evaluation._mp_context().get_start_method(), "spawn")

    def test_worker_count_admits_only_complete_scenario_seed_cohorts(self):
        self.assertEqual(evaluation.EvalSettings().workers, 20)
        self.assertEqual(evaluation.parse_args([]).workers, 20)
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=(1, 2, 3, 4, 5),
            workers=20,
        )
        self.assertEqual(evaluation._worker_count(settings), 15)
        eight_seeds = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=tuple(range(8)), workers=20,
        )
        self.assertEqual(evaluation._worker_count(eight_seeds), 16)
        twelve_seeds = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=tuple(range(12)), workers=20,
        )
        self.assertEqual(evaluation._worker_count(twelve_seeds), 12)
        hard_cap = evaluation.EvalSettings(
            scenarios=("homogeneous",), seeds=tuple(range(30)), workers=99,
        )
        self.assertEqual(evaluation._worker_count(hard_cap), 20)

    def test_scenario_batches_never_admit_a_partial_seed_cohort(self):
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=tuple(range(8)), workers=20,
        )
        jobs = [
            (scenario, seed)
            for scenario in settings.scenarios for seed in settings.seeds
        ]
        batches = evaluation._scenario_job_batches(jobs, settings)
        self.assertEqual(tuple(map(len, batches)), (16, 8))
        self.assertEqual(
            tuple({scenario for scenario, _seed in batch} for batch in batches),
            (
                {"homogeneous", "heterogeneous"},
                {"holdout"},
            ),
        )

    def test_worker_count_never_exceeds_the_pending_work(self):
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=(1, 2, 3, 4, 5),
            workers=0,
        )
        self.assertEqual(evaluation._worker_count(settings, 3), 3)
        self.assertGreaterEqual(evaluation._worker_count(settings), 1)

    def test_pending_jobs_span_scenarios(self):
        """Jobs from different scenarios must be able to run concurrently."""
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "holdout"), seeds=(1, 2), workers=4,
        )
        seen = []

        def fake_job(scenario, seed, _config, _run_dir):
            seen.append((scenario, int(seed)))
            return {"scenario": scenario, "seed": int(seed), "wall_s": 0.0}

        with patch.object(evaluation, "_run_seed_scenario_job", fake_job):
            results = list(evaluation._pending_job_results(
                [("homogeneous", 1), ("holdout", 2)],
                settings, Path("/tmp/jobs-test"), workers=1,
            ))
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {row["scenario"] for row in results}, {"homogeneous", "holdout"}
        )

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
