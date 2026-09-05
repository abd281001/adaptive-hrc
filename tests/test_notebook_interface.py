"""Contracts keeping HRC.ipynb a thin, ordered experiment interface."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "HRC.ipynb"


class NotebookInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_uses_the_registered_project_kernel(self):
        kernelspec = self.notebook["metadata"]["kernelspec"]
        self.assertEqual(kernelspec["name"], "adaptive-hrc")
        self.assertEqual(kernelspec["display_name"], "Python (adaptive-hrc)")

    def test_run_all_uses_the_required_stage_order(self):
        cells = {cell["id"]: cell for cell in self.notebook["cells"]}
        ordered = [
            "".join(cells[cell_id]["source"]).strip()
            for cell_id in ("run-normal", "run-cooking", "run-ablations")
        ]
        self.assertEqual(
            ordered,
            [
                'run_stage("run")',
                'run_stage("cooking")',
                'run_stage("ablation")',
            ],
        )

    def test_contains_only_interface_logic(self):
        code = "\n".join(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertLessEqual(len(code.splitlines()), 20)
        for application_detail in (
            "EvalSettings",
            "ThreadPoolExecutor",
            "ablation_commands",
            "cooking_summary",
            "manifest.json",
            "MPLCONFIGDIR",
        ):
            with self.subTest(application_detail=application_detail):
                self.assertNotIn(application_detail, code)

    def test_is_committed_without_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
