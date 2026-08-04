import os
import unittest

from src import evaluation


class EvaluationParallelismTests(unittest.TestCase):
    def test_worker_count_is_capped_by_seed_count(self):
        cfg = evaluation.EvaluationConfig(seeds=(1, 2, 3, 4, 5), workers=20)
        self.assertEqual(evaluation._worker_count(cfg), 5)

    def test_worker_count_defaults_to_one_worker_per_seed(self):
        cfg = evaluation.EvaluationConfig(seeds=(1, 2, 3, 4, 5), workers=0)
        self.assertEqual(evaluation._worker_count(cfg), 5)

    def test_native_thread_limit_overrides_parent_environment(self):
        previous = {var: os.environ.get(var) for var in evaluation.NATIVE_THREAD_ENV_VARS}
        try:
            for var in evaluation.NATIVE_THREAD_ENV_VARS:
                os.environ[var] = "20"

            info = evaluation._apply_native_thread_limit(1)

            self.assertEqual(info["native_threads_per_worker"], 1)
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
