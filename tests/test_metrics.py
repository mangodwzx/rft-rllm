"""Metric invariants, independent of rLLM/GPU packages."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rft_rllm.metrics import compare_runs, compare_summaries, summarize_run, task_fingerprint


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rft-metrics-test-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.rows = [
            {"question": "Question A", "level": 5, "subject": "algebra"},
            {"question": "Question B", "level": 1, "subject": "algebra"},
            {"question": "Question C", "level": 4, "subject": "geometry"},
        ]

    def result(self, counts, attempts=2, rows=None):
        rows = self.rows if rows is None else rows
        return {"attempts": attempts, "total": len(counts) * attempts,
                "dataset_name": "math_test", "agent": "math_flow", "items": [
                    {"idx": idx, "attempt": attempt, "is_correct": attempt < count, "error": None,
                     "signals": {"task_fingerprint": task_fingerprint(rows[idx]["question"])}}
                    for idx, count in enumerate(counts) for attempt in range(attempts)
                ]}

    def save(self, result, name="run"):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        (directory / "results.json").write_text(json.dumps(result), encoding="utf-8")
        return directory

    def summarize(self, result, rows=None, ks=None):
        return summarize_run(self.save(result), [1, 2] if ks is None else ks,
                             task_rows=self.rows if rows is None else rows)

    def test_subsets_use_their_own_task_and_telemetry_denominators(self):
        result = self.result([1, 0, 2])
        for item in result["items"]:
            item["signals"].update(response_tokens=10 * (item["idx"] + 1),
                                   truncated=int(item["idx"] == 0 and item["attempt"] == 1))
        summary = self.summarize(result)
        self.assertEqual(summary["pass_at"], {"1": .5, "2": 2 / 3})
        hard = summary["subsets"]["level_4_5"]
        self.assertEqual((hard["tasks"], hard["total"], hard["correct"]), (2, 4, 3))
        self.assertEqual(hard["pass_at"], {"1": .75, "2": 1.0})
        self.assertEqual(hard["accuracy"], .75)
        self.assertEqual(hard["mean_response_tokens"], 20)
        self.assertAlmostEqual(hard["mean_correct_response_tokens"], 70 / 3)
        self.assertEqual(hard["truncation_rate"], .25)
        self.assertEqual(hard["coverage_counts"]["response_tokens"], 4)
        self.assertEqual(summary["subsets"]["by_subject"]["algebra"]["accuracy"], .25)
        self.assertEqual(summary["subsets"]["by_subject"]["geometry"]["accuracy"], 1.0)
        self.assertEqual(summary["subsets"]["by_level"]["1"]["pass_at"]["2"], 0.0)
        self.assertEqual(summary["identity_coverage"],
                         {"verified_rollouts": 6, "total_rollouts": 6, "unverified_error_rollouts": 0})

    def test_empty_subsets_and_missing_metadata_are_explicit(self):
        rows = [{"question": "No level"}]
        summary = self.summarize(self.result([1], rows=rows), rows)
        hard = summary["subsets"]["level_4_5"]
        self.assertEqual((hard["tasks"], hard["total"], hard["correct"]), (0, 0, 0))
        self.assertIsNone(hard["pass_at"])
        self.assertIsNone(hard["accuracy"])
        self.assertIsNone(hard["truncation_rate"])
        self.assertEqual(summary["subsets"]["by_level"]["unknown"]["tasks"], 1)
        self.assertEqual(summary["subsets"]["by_subject"]["unknown"]["tasks"], 1)

    def test_error_attempt_without_fingerprint_stays_in_denominator(self):
        result = self.result([2, 0, 2])
        result["items"][0].update(error="request timeout", signals={})
        summary = self.summarize(result)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["accuracy"], .5)
        self.assertEqual(summary["subsets"]["level_4_5"]["errors"], 1)
        self.assertEqual(summary["identity_coverage"]["unverified_error_rollouts"], 1)
        self.assertEqual(summary["identity_coverage"]["verified_rollouts"], 5)

    def test_all_error_task_is_retained(self):
        result = self.result([2, 0, 2])
        for item in result["items"][:2]:
            item.update(error="request timeout", signals={})
        summary = self.summarize(result)
        self.assertEqual(summary["tasks"], 3)
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["subsets"]["by_level"]["5"]["pass_at"], {"1": 0.0, "2": 0.0})

    def test_missing_fingerprint_on_non_error_attempt_is_rejected(self):
        result = self.result([1, 0, 2])
        result["items"][1]["signals"] = {}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.summarize(result)

    def test_wrong_split_with_same_size_is_rejected(self):
        result = self.result([1, 0, 2])
        result["items"][0]["signals"]["task_fingerprint"] = task_fingerprint("A different test question")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.summarize(result)
        result["items"][0]["error"] = "grader error"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.summarize(result)

    def test_missing_attempt_is_rejected_before_subsets(self):
        result = self.result([1, 0, 2])
        result["items"].pop()
        result["total"] -= 1
        with self.assertRaisesRegex(ValueError, "Incomplete task"):
            self.summarize(result)

    def test_duplicate_attempt_is_rejected(self):
        result = self.result([1, 0, 2])
        result["items"][1] = result["items"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate rollout"):
            self.summarize(result)

    def test_pass_at_256_is_coverage_when_attempts_equal_256(self):
        rows = self.rows[:2]
        result = self.result([1, 0], attempts=256, rows=rows)
        summary = self.summarize(result, rows, [1, 128, 256])
        self.assertEqual(summary["pass_at"]["1"], 1 / 512)
        self.assertEqual(summary["pass_at"]["256"], summary["observed_task_coverage"])
        self.assertEqual(summary["subsets"]["level_4_5"]["pass_at"]["256"], 1.0)
        with self.assertRaisesRegex(ValueError, "requires at least"):
            self.summarize(result, rows, [257])

    def test_formal_pass_at_256_uses_1024_attempts(self):
        rows = self.rows[:1]
        summary = self.summarize(self.result([1], attempts=1024, rows=rows), rows, [256])
        self.assertEqual(summary["pass_at"]["256"], .25)
        self.assertEqual(summary["observed_task_coverage"], 1.0)

    def test_legacy_top_level_summary_still_works_without_rows(self):
        result = self.result([1, 0, 2])
        for item in result["items"]:
            item["signals"] = {}
        summary = summarize_run(self.save(result), [1, 2], 2, 3)
        self.assertEqual(summary["accuracy"], .5)
        self.assertIsNone(summary["mean_response_tokens"])
        self.assertNotIn("subsets", summary)
        self.assertIsNone(summary["identity_coverage"])

    def test_paired_differences_include_hard_tasks_and_consistent_sign(self):
        paths = {name: self.save(self.result(counts), name) for name, counts in
                 (("base", [0, 0, 0]), ("rft", [1, 0, 1]), ("grpo", [2, 0, 2]))}
        summaries = compare_runs(paths, [1, 2], 2, 3, self.rows)
        comparisons = compare_summaries(summaries)
        self.assertEqual(set(comparisons), {"grpo_minus_rft", "rft_minus_base", "grpo_minus_base"})
        grpo_rft = comparisons["grpo_minus_rft"]
        self.assertAlmostEqual(grpo_rft["overall"]["pass_at_delta"]["1"], 1 / 3)
        self.assertEqual(grpo_rft["level_4_5"]["pass_at_delta"], {"1": .5, "2": 0.0})
        self.assertEqual(comparisons["grpo_minus_base"]["level_4_5"]["pass_at_delta"]["2"], 1.0)

    def test_paired_comparison_rejects_changed_group_assignments(self):
        summaries = {name: self.summarize(self.result([1, 0, 2])) for name in ("base", "rft", "grpo")}
        changed = [dict(row) for row in self.rows]
        changed[0]["level"], changed[1]["level"] = changed[1]["level"], changed[0]["level"]
        summaries["grpo"] = self.summarize(self.result([1, 0, 2]), changed)
        with self.assertRaisesRegex(ValueError, "same task_rows identity"):
            compare_summaries(summaries)

    def test_fingerprint_is_whitespace_stable_and_float_exact(self):
        value = task_fingerprint("  三角形?\n")
        self.assertEqual(value, task_fingerprint("三角形?"))
        self.assertEqual(value, float(value))
        self.assertLess(value, 2 ** 48)


if __name__ == "__main__":
    unittest.main()
