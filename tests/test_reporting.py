"""CSV/SVG report checks using synthetic metrics, without model execution."""

import csv
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rft_rllm.metrics import compare_summaries
from rft_rllm.reporting import write_comparison_artifacts


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rft-report-test-")
        self.output = Path(self.temporary.name) / "report"
        self.addCleanup(self.temporary.cleanup)

    def inputs(self, *, hard_tasks=2, ks=(1, 2, 4), attempts=4):
        summaries = {}
        for name, rate in (("base", .2), ("rft", .3), ("grpo", .4)):
            summary = {"tasks": 5, "attempts": attempts, "accuracy": rate,
                       "pass_at": {str(k): min(1.0, rate * k) for k in ks},
                       "task_identity_sha256": "same tasks and groups"}
            summary["subsets"] = {"level_4_5": {
                "tasks": hard_tasks, "attempts": attempts,
                "accuracy": rate / 2 if hard_tasks else None,
                "pass_at": {str(k): min(1.0, rate / 2 * k) for k in ks} if hard_tasks else None,
            }}
            summaries[name] = summary
        return summaries, compare_summaries(summaries)

    def read_csv(self):
        with (self.output / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def svg(self):
        return ET.parse(self.output / "pass_at_k.svg").getroot()

    def test_long_csv_and_four_panel_svg(self):
        summaries, comparisons = self.inputs()
        write_comparison_artifacts(self.output, summaries, comparisons)
        rows = self.read_csv()
        self.assertEqual(len(rows), 2 * 3 * 3)
        self.assertEqual({row["scope"] for row in rows}, {"all", "level_4_5"})
        sample = next(row for row in rows if row["scope"] == "level_4_5" and
                      row["comparison"] == "grpo_minus_rft" and row["k"] == "4")
        self.assertAlmostEqual(float(sample["pass_at_delta"]), .2)
        self.assertEqual((sample["tasks"], sample["attempts"]), ("2", "4"))
        root = self.svg()
        text = " ".join(root.itertext())
        self.assertIn("Level 4-5: pass@k", text)
        self.assertIn("All tasks: GRPO - RFT", text)
        self.assertIn("No confidence intervals", text)
        self.assertEqual(len(root.findall(".//{http://www.w3.org/2000/svg}polyline")), 8)
        self.assertNotIn("nan", (self.output / "pass_at_k.svg").read_text().lower())

    def test_empty_hard_subset_is_blank_not_zero(self):
        write_comparison_artifacts(self.output, *self.inputs(hard_tasks=0))
        hard_rows = [row for row in self.read_csv() if row["scope"] == "level_4_5"]
        self.assertEqual(len(hard_rows), 9)
        self.assertTrue(all(row["pass_at_delta"] == "" and row["tasks"] == "0" for row in hard_rows))
        text = list(self.svg().itertext())
        self.assertEqual(text.count("No tasks"), 2)

    def test_single_k_and_limited_attempts(self):
        write_comparison_artifacts(self.output, *self.inputs(ks=(2,), attempts=2))
        self.assertEqual({row["k"] for row in self.read_csv()}, {"2"})
        circles = self.svg().findall(".//{http://www.w3.org/2000/svg}circle")
        self.assertEqual(len(circles), 8)
        self.assertTrue(all(float(circle.attrib["cx"]) in (299.0, 849.0) for circle in circles))

    def test_log_axis_and_probability_bounds(self):
        write_comparison_artifacts(self.output, *self.inputs())
        points = self.svg().findall(".//{http://www.w3.org/2000/svg}polyline")[0].attrib["points"].split()
        coordinates = [tuple(map(float, point.split(","))) for point in points]
        self.assertAlmostEqual(coordinates[1][0] - coordinates[0][0], coordinates[2][0] - coordinates[1][0])
        self.assertTrue(all(124 <= y <= 338 for _, y in coordinates))

    def test_k_greater_than_attempts_is_rejected_without_artifacts(self):
        summaries, comparisons = self.inputs()
        summaries["grpo"]["pass_at"]["8"] = .9
        with self.assertRaisesRegex(ValueError, "k must be"):
            write_comparison_artifacts(self.output, summaries, comparisons)
        self.assertFalse(self.output.exists())

    def test_non_finite_values_are_rejected(self):
        summaries, comparisons = self.inputs()
        comparisons["grpo_minus_rft"]["overall"]["pass_at_delta"]["2"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid metric value"):
            write_comparison_artifacts(self.output, summaries, comparisons)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
