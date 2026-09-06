"""Contracts for immutable evaluation artifacts and latest-run resolution."""
from __future__ import annotations

import gzip
import json
import tempfile
import unittest

from src.evaluation import summarize_decision_regimes
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from src import evaluation

# src/plotting.py is currently absent from the tree. Skip this module rather
# than failing collection, which would take the whole suite down with it.
plotting = pytest.importorskip(
    "src.plotting", reason="src/plotting.py is not present",
)


class EvaluationResultLayoutTests(unittest.TestCase):
    def _config(self, root: str, run: str) -> evaluation.EvalSettings:
        return evaluation.EvalSettings(
            seeds=(7, 9), scenarios=("synthetic",), baselines=(),
            include_oracle=False, output=root, run=run,
            workers=1, show_eta=False,
        )

    @staticmethod
    def _fake_job(
        scenario: str, seed: int, _config: evaluation.EvalSettings, run_dir: str,
    ) -> dict:
        root = Path(run_dir)
        seed_dir = evaluation._seed_dir(root, scenario, seed)
        summary = {
            "scenario": scenario,
            "seed": seed,
            "per_baseline": {},
        }
        evaluation._write_json(seed_dir / "summary.json", summary)
        evaluation._write_json(seed_dir / "status.json", {"state": "complete", "wall_s": 0.1})
        evaluation._write_jsonl_gz(
            seed_dir / "tables" / "episodes.jsonl.gz",
            ({"scenario": scenario, "seed": seed, "baseline": "full"},),
        )
        return {
            "scenario": scenario, "seed": seed, "key": f"{scenario}/{seed:010d}",
            "summary": summary,
            "summary_path": str((seed_dir / "summary.json").relative_to(root)),
            "wall_s": 0.1,
        }

    def test_suite_publishes_compact_completed_run_at_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, "layout_test")
            with patch.object(evaluation, "_run_seed_scenario_job", side_effect=self._fake_job):
                returned = evaluation.run_evaluation(config)

            run_dir = Path(directory) / "runs" / "layout_test"
            latest = Path(directory) / "latest"
            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.resolve(), run_dir.resolve())
            self.assertEqual(json.loads((run_dir / "status.json").read_text())["state"], "complete")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertIn("config", manifest)
            self.assertEqual(manifest["config"]["schedule"]["demos"], 210)
            self.assertEqual(
                manifest["resolved_model_config"]["commit_threshold"], 0.70,
            )
            self.assertEqual(
                manifest["resolved_model_config"]["seed_source"], "evaluation job seed",
            )
            saved = json.loads((latest / "aggregate" / "suite_summary.json").read_text())
            self.assertEqual(saved["state"], "complete")
            self.assertEqual(saved["completed_jobs"], 2)
            self.assertNotIn("per_baseline", next(iter(saved["scenarios"].values())))
            self.assertEqual(Path(returned["run_dir"]), run_dir)

            expected = {7, 9}
            summaries = plotting._load_summaries(run_dir, "synthetic", expected)
            rows = plotting._load_episode_rows(run_dir, "synthetic", expected)
            self.assertEqual([row["seed"] for row in summaries], [7, 9])
            self.assertEqual({row["seed"] for row in rows}, {7, 9})

    def test_failed_run_does_not_replace_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._config(directory, "first")
            with patch.object(evaluation, "_run_seed_scenario_job", side_effect=self._fake_job):
                evaluation.run_evaluation(first)
            latest = Path(directory) / "latest"
            first_target = latest.resolve()

            second = self._config(directory, "second")
            with patch.object(
                evaluation, "_run_seed_scenario_job", side_effect=RuntimeError("planned failure"),
            ), self.assertRaisesRegex(RuntimeError, "planned failure"):
                evaluation.run_evaluation(second)

            self.assertEqual(latest.resolve(), first_target)
            failed = json.loads((Path(directory) / "runs" / "second" / "status.json").read_text())
            self.assertEqual(failed["state"], "failed")

    def test_resume_skips_completed_seed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, "resumable")
            with patch.object(evaluation, "_run_seed_scenario_job", side_effect=self._fake_job):
                evaluation.run_evaluation(config)

            with patch.object(evaluation, "_run_seed_scenario_job") as worker:
                resumed = evaluation.run_evaluation(replace(config, resume=True))
            worker.assert_not_called()
            self.assertEqual(resumed["completed_jobs"], 2)

    def test_compressed_table_write_replaces_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl.gz"
            evaluation._write_jsonl_gz(path, ({"value": 1},))
            evaluation._write_jsonl_gz(path, ({"value": 2},))
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows, [{"value": 2}])


if __name__ == "__main__":
    unittest.main()


class DecisionRegimeSummaryTests(unittest.TestCase):
    """The regime split is what keeps a memorisation term from hiding the rest.

    An aggregate Top-1 over this stream is dominated by decisions where only
    one action was ever demonstrated at the state, which any predictor that
    memorised the state graph answers correctly. These tests pin the split so
    that term stays separable.
    """

    @staticmethod
    def _turn(count, correct, *, legacy=False):
        key = "exact_state_learned_action_count" if legacy else "observed_actions_at_state"
        return {"turn_kind": "robot", key: count, "correct_top_1": correct, "correct_top_k": correct}

    def test_regimes_are_split_by_observed_choice(self):
        rows = [
            self._turn(1, True), self._turn(1, True), self._turn(1, False),
            self._turn(3, True), self._turn(2, False),
            self._turn(0, False),
        ]
        summary = summarize_decision_regimes(rows)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["n_classified"], 6)
        by = summary["by_regime"]
        self.assertEqual(by["single_option_lookup"]["n"], 3)
        self.assertEqual(by["branching"]["n"], 2)
        self.assertEqual(by["unseen_state_fallback"]["n"], 1)
        self.assertAlmostEqual(by["single_option_lookup"]["top_1"], 2 / 3)
        self.assertAlmostEqual(by["branching"]["top_1"], 0.5)
        self.assertAlmostEqual(by["unseen_state_fallback"]["top_1"], 0.0)
        self.assertAlmostEqual(sum(by[r]["share"] for r in by), 1.0)

    def test_shares_are_reported_so_the_weighting_is_visible(self):
        # Two arms with identical conditional accuracy but different regime
        # mixes must be distinguishable: the aggregate alone would hide that
        # one arm simply faced fewer hard decisions.
        easy = [self._turn(1, True)] * 90 + [self._turn(0, False)] * 10
        hard = [self._turn(1, True)] * 50 + [self._turn(0, False)] * 50
        a, b = summarize_decision_regimes(easy), summarize_decision_regimes(hard)
        self.assertAlmostEqual(a["by_regime"]["single_option_lookup"]["top_1"],
                               b["by_regime"]["single_option_lookup"]["top_1"])
        self.assertGreater(a["aggregate_top_1_over_classified"],
                           b["aggregate_top_1_over_classified"])
        self.assertNotAlmostEqual(a["by_regime"]["unseen_state_fallback"]["share"],
                                  b["by_regime"]["unseen_state_fallback"]["share"])

    def test_earlier_runs_stratify_through_the_legacy_key(self):
        rows = [self._turn(1, True, legacy=True), self._turn(2, False, legacy=True)]
        summary = summarize_decision_regimes(rows)
        self.assertEqual(summary["n_classified"], 2)
        self.assertEqual(summary["by_regime"]["branching"]["n"], 1)

    def test_unclassifiable_turns_are_counted_not_silently_dropped(self):
        rows = [self._turn(1, True), {"turn_kind": "robot", "correct_top_1": True}]
        summary = summarize_decision_regimes(rows)
        self.assertEqual(summary["n_classified"], 1)
        self.assertEqual(summary["n_unclassified"], 1)

    def test_no_usable_turns_reports_not_run(self):
        summary = summarize_decision_regimes([{"turn_kind": "robot"}])
        self.assertEqual(summary["status"], "not_run")
        self.assertEqual(summary["by_regime"], {})
