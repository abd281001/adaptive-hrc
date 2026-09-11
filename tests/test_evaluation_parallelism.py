import os
import tempfile
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import patch

from src import evaluation


class PerformanceCoreAffinityTests(unittest.TestCase):
    """Cross-baseline wall-clock comparisons assume every worker runs on the
    same class of core; an E-core scheduling accident would silently favor
    whichever baseline happened to land on a P-core."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def _write_range(self, text: str) -> Path:
        pmu_path = self.tmp_path / "cpus"
        pmu_path.write_text(text, encoding="utf-8")
        return pmu_path

    def test_parses_a_contiguous_range(self):
        pmu_path = self._write_range("0-7\n")
        self.assertEqual(
            evaluation._detect_p_core_cpus(pmu_path), frozenset(range(8)),
        )

    def test_parses_a_disjoint_range(self):
        pmu_path = self._write_range("0-3,8-11\n")
        self.assertEqual(
            evaluation._detect_p_core_cpus(pmu_path),
            frozenset([0, 1, 2, 3, 8, 9, 10, 11]),
        )

    def test_returns_none_when_the_pmu_file_is_absent(self):
        self.assertIsNone(
            evaluation._detect_p_core_cpus(self.tmp_path / "missing"),
        )

    def test_returns_none_on_unparseable_content(self):
        pmu_path = self._write_range("not-a-range\n")
        self.assertIsNone(evaluation._detect_p_core_cpus(pmu_path))

    def test_pin_is_a_noop_without_sched_setaffinity(self):
        # Stand-in for a platform (e.g. macOS) with no sched_setaffinity.
        class _NoAffinityOs:
            def __getattr__(self, name):
                if name == "sched_setaffinity":
                    raise AttributeError(name)
                return getattr(os, name)

        with patch.object(evaluation, "os", _NoAffinityOs()):
            self.assertIsNone(evaluation._pin_to_performance_cores())

    def test_pin_is_a_noop_on_a_uniform_part(self):
        with patch.object(
            evaluation, "_detect_p_core_cpus", return_value=None,
        ):
            self.assertIsNone(evaluation._pin_to_performance_cores())

    def test_pin_restricts_this_process_to_the_detected_cpu_set(self):
        if not hasattr(os, "sched_setaffinity"):
            self.skipTest("sched_setaffinity is Linux-only")
        original = os.sched_getaffinity(0)
        fake_cpus = frozenset(sorted(original)[:1])
        try:
            with patch.object(
                evaluation, "_detect_p_core_cpus", return_value=fake_cpus,
            ):
                applied = evaluation._pin_to_performance_cores()
            self.assertEqual(applied, fake_cpus)
            self.assertEqual(os.sched_getaffinity(0), set(fake_cpus))
        finally:
            os.sched_setaffinity(0, original)


class EvaluationParallelismTests(unittest.TestCase):
    def test_workers_use_clean_processes(self):
        self.assertEqual(evaluation._mp_context().get_start_method(), "spawn")

    def test_worker_count_admits_only_complete_scenario_seed_cohorts(self):
        self.assertEqual(
            evaluation.EvalSettings().workers,
            evaluation.DEFAULT_EVALUATION_WORKER_CAP,
        )
        cap = evaluation.DEFAULT_EVALUATION_WORKER_CAP
        self.assertEqual(evaluation.parse_args([]).workers, cap)
        # Five seeds: two whole cohorts fit inside the cap, a third does not.
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=(1, 2, 3, 4, 5),
            workers=cap,
        )
        self.assertEqual(evaluation._worker_count(settings), 10)
        # Eight seeds: one cohort fits, so the pool shrinks to the cohort size
        # rather than admitting part of a second scenario.
        eight_seeds = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=tuple(range(8)), workers=cap,
        )
        self.assertEqual(evaluation._worker_count(eight_seeds), 8)
        # Twelve seeds: no cohort fits, so the cap alone bounds the pool and a
        # single scenario is drained across successive batches.
        twelve_seeds = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=tuple(range(12)), workers=cap,
        )
        self.assertEqual(evaluation._worker_count(twelve_seeds), cap)
        hard_cap = evaluation.EvalSettings(
            scenarios=("homogeneous",), seeds=tuple(range(30)), workers=99,
        )
        # An explicit request above the cap is clamped: the ceiling is the
        # machine's memory, and exceeding it kills a worker hours into a run.
        self.assertEqual(evaluation._worker_count(hard_cap), cap)

    def test_scenario_batches_never_admit_a_partial_seed_cohort(self):
        # Eight seeds per scenario against the worker cap: a second scenario
        # would need sixteen slots, so each batch is one complete cohort.
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=tuple(range(8)),
            workers=evaluation.DEFAULT_EVALUATION_WORKER_CAP,
        )
        jobs = [
            (scenario, seed)
            for scenario in settings.scenarios for seed in settings.seeds
        ]
        batches = evaluation._scenario_job_batches(jobs, settings)
        self.assertEqual(tuple(map(len, batches)), (8, 8, 8))
        self.assertEqual(
            tuple({scenario for scenario, _seed in batch} for batch in batches),
            ({"homogeneous"}, {"heterogeneous"}, {"holdout"}),
        )

    def test_worker_count_never_exceeds_the_pending_work(self):
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=(1, 2, 3, 4, 5),
            workers=0,
        )
        self.assertEqual(evaluation._worker_count(settings, 3), 3)
        self.assertGreaterEqual(evaluation._worker_count(settings), 1)

    def test_pending_cells_span_scenarios(self):
        """One arm's cells from different scenarios run concurrently."""
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "holdout"), seeds=(1, 2), workers=4,
        )
        seen = []

        def fake_job(baseline, scenario, seed, _config, _run_dir):
            seen.append((baseline, scenario, int(seed)))
            return {
                "baseline": baseline, "scenario": scenario,
                "seed": int(seed), "wall_s": 0.0,
            }

        with patch.object(evaluation, "_run_baseline_cell_job", fake_job):
            results = list(evaluation._pending_cell_results(
                "full",
                [("homogeneous", 1), ("holdout", 2)],
                settings, Path("/tmp/cells-test"), workers=1,
            ))
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {row["scenario"] for row in results}, {"homogeneous", "holdout"}
        )
        self.assertEqual({row["baseline"] for row in results}, {"full"})

    def test_baselines_run_one_at_a_time_with_full_first(self):
        """Arm order is the schedule: full publishes the route others replay."""
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous",),
            seeds=(1,),
            baselines=("bc", "full", "no_decay"),
            include_oracle=True,
        )
        order = evaluation.baseline_run_order(settings)
        self.assertEqual(order[0], "full")
        self.assertEqual(order[-1], evaluation.MEMORY_ORACLE)
        self.assertEqual(set(order), {"full", "bc", "no_decay", evaluation.MEMORY_ORACLE})

    def test_baseline_run_order_requires_full_under_shared_routing(self):
        settings = evaluation.EvalSettings(
            baselines=("bc", "no_decay"), shared_routing=True,
        )
        with self.assertRaisesRegex(ValueError, "requires the deployable 'full'"):
            evaluation.baseline_run_order(settings)

    def test_in_context_llm_uses_one_gpu_worker(self):
        settings = evaluation.EvalSettings(
            seeds=(1, 2, 3),
            baselines=("full", "in_context_llm"),
            workers=3,
        )
        self.assertEqual(evaluation._worker_count(settings), 1)

    def test_only_the_gpu_arm_is_held_to_one_worker(self):
        """A GPU arm in the roster must not serialize the symbolic arms."""
        settings = evaluation.EvalSettings(
            scenarios=("homogeneous", "heterogeneous", "holdout"),
            seeds=tuple(range(8)),
            baselines=("full", "in_context_llm"),
            workers=evaluation.DEFAULT_EVALUATION_WORKER_CAP,
        )
        self.assertEqual(evaluation._worker_count(settings, None, "full"), 8)
        self.assertEqual(
            evaluation._worker_count(settings, None, evaluation.LLM_BASELINE), 1,
        )

    def test_in_context_llm_does_not_retry_a_native_worker_failure(self):
        attempts = []

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
                list(evaluation._pending_cell_results(
                    evaluation.LLM_BASELINE,
                    [("homogeneous", seed) for seed in settings.seeds],
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
