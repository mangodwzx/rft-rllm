"""Data protocol invariants without Hugging Face or GPU dependencies."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rft_rllm.data import (
    _digest, _key, _normalize, data_directory, load_rows, math_reference_answer,
    prepare, registry_name, stratified_take,
)


class DataProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rft-data-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"data": {
            "kind": "math", "name": "math", "train_size": 24,
            "pilot_size": 24, "dev_size": 12, "seed": 42,
            "train_jsonl": "source_train.jsonl", "test_jsonl": "source_test.jsonl",
        }}
        train = [{"id": f"q{i}", "problem": f"training problem {i}",
                  "answer": str(i), "solution": "", "type": f"subject{i % 3}",
                  "level": i % 2 + 1} for i in range(72)]
        test = [{"id": "heldout", "problem": "held out problem", "answer": "99", "level": 3}]
        self.write_source("source_train.jsonl", train)
        self.write_source("source_test.jsonl", test)

    def write_source(self, filename, rows):
        (self.root / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_pilot_and_full_share_dev_and_full_contains_pilot(self):
        first = prepare(self.config, self.root, register=False)
        pilot = load_rows(self.config, self.root, "train")
        dev = load_rows(self.config, self.root, "dev")
        manifest_bytes = (data_directory(self.config, self.root) / "manifest.json").read_bytes()
        full_config = copy.deepcopy(self.config)
        full_config["data"]["train_size"] = "all"
        second = prepare(full_config, self.root, register=False)
        full = load_rows(full_config, self.root, "train")
        self.assertEqual(len(pilot), 24)
        self.assertEqual(len(full), 60)
        self.assertEqual(dev, load_rows(full_config, self.root, "dev"))
        keys = lambda rows: {_key(row["question"]) for row in rows}
        self.assertLessEqual(keys(pilot), keys(full))
        self.assertFalse(keys(dev) & keys(full))
        self.assertEqual(first["splits"]["dev"], second["splits"]["dev"])
        self.assertEqual(manifest_bytes, (data_directory(self.config, self.root) / "manifest.json").read_bytes())
        self.assertEqual(set(first["splits"]["pilot"]["strata"].values()), {4})

    def test_stratification_independent_of_source_order(self):
        rows = [_normalize({"problem": str(i), "answer": str(i), "type": f"s{i % 3}", "level": i % 2 + 1}, "train") for i in range(72)]
        self.assertEqual(stratified_take(rows, 12, 42, "dev"), stratified_take(list(reversed(rows)), 12, 42, "dev"))

    def test_registry_names_separate_splits_scales_and_seeds(self):
        names = [registry_name(self.config, split) for split in ("train", "dev", "test")]
        self.assertEqual(len(set(names)), 3)
        full = copy.deepcopy(self.config)
        full["data"]["train_size"] = "all"
        self.assertNotEqual(registry_name(full, "train"), names[0])
        self.assertEqual(registry_name(full, "dev"), names[1])
        full["data"]["seed"] += 1
        self.assertNotEqual(registry_name(full, "dev"), names[1])

    def test_register_uses_logical_splits_with_separate_names(self):
        calls = []
        class Registry:
            @staticmethod
            def register_dataset(**kwargs):
                calls.append(kwargs)
        import sys
        import types
        module = types.ModuleType("rllm.data.dataset")
        module.DatasetRegistry = Registry
        with patch.dict(sys.modules, {"rllm.data.dataset": module}):
            prepare(self.config, self.root, register=True)
        self.assertEqual({call["split"] for call in calls}, {"train", "dev", "test"})
        self.assertEqual(len({call["name"] for call in calls}), 3)
        self.assertEqual(next(call for call in calls if call["split"] == "train")["data"], load_rows(self.config, self.root, "train"))

    def test_tampered_data_and_changed_protocol_are_rejected(self):
        prepare(self.config, self.root, register=False)
        changed = copy.deepcopy(self.config)
        changed["data"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "requested protocol"):
            prepare(changed, self.root, register=False)
        path = data_directory(self.config, self.root) / "dev.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest"):
            prepare(self.config, self.root, register=False)

    def test_gsm8k_is_only_a_fixed_debug_split(self):
        rows = [{"question": f"gsm {i}", "answer": "reasoning\n#### 1,234"} for i in range(50)]
        self.write_source("gsm.jsonl", rows)
        config = {"data": {"kind": "gsm8k", "name": "gsm", "train_jsonl": "gsm.jsonl", "debug_size": 8, "seed": 42}}
        result = prepare(config, self.root, register=False)
        self.assertEqual(set(result["splits"]), {"debug"})
        first = load_rows(config, self.root, "debug")
        self.assertEqual(len(first), 8)
        self.assertEqual({row["ground_truth"] for row in first}, {"1234"})
        self.assertEqual(first, load_rows(config, self.root, "debug"))
        with self.assertRaises(ValueError):
            load_rows(config, self.root, "train")

    def test_deepmath_sizes_are_nested_and_exclude_math_dev_test(self):
        prepare(self.config, self.root, register=False)
        dev = load_rows(self.config, self.root, "dev")
        test = load_rows(self.config, self.root, "test")
        deep = [{"question": f"deep {i}", "final_answer": "1", "topic": "algebra", "difficulty": 5} for i in range(16390)]
        deep += [{"question": row["question"], "final_answer": "1", "topic": "algebra", "difficulty": 5} for row in dev + test]
        self.write_source("deep.jsonl", deep)
        config = {"data": {"kind": "deepmath", "name": "deep8k", "train_jsonl": "deep.jsonl",
                           "train_size": 8192, "dev_size": 12, "seed": 42, "math_data_name": "math", "math": self.config["data"]}}
        first = prepare(config, self.root, register=False)
        small = load_rows(config, self.root, "train")
        large_config = copy.deepcopy(config)
        large_config["data"].update(name="deep16k", train_size=16384)
        prepare(large_config, self.root, register=False)
        large = load_rows(large_config, self.root, "train")
        self.assertEqual(small, large[:8192])
        protected = {_key(row["question"]) for row in dev + test}
        self.assertFalse(protected & {_key(row["question"]) for row in large})
        self.assertEqual(len(first["removed_rows"]["train"]), len(dev) + len(test))
        self.assertEqual(load_rows(config, self.root, "dev"), dev)
        self.assertEqual(load_rows(config, self.root, "test"), test)

    def test_deepmath_first_creates_math_reusable_by_baseline(self):
        math_train = [{"problem": f"real math {i}", "answer": str(i), "type": "algebra", "level": i % 5 + 1} for i in range(2400)]
        math_test = [{"problem": f"math test {i}", "answer": str(i), "type": "algebra", "level": 1} for i in range(500)]
        deep = [{"question": f"real deep {i}", "final_answer": "1", "topic": "algebra"} for i in range(8192)]

        def source(cfg, root, *, test):
            rows = deep if cfg["kind"] == "deepmath" else math_test if test else math_train
            return rows, {"kind": "test_fixture"}

        config = {"data": {"kind": "deepmath", "name": "deep", "train_size": 8192}}
        baseline = {"data": {
            "kind": "math", "name": "math", "train_size": 2048, "pilot_size": 2048,
            "dev_size": 256, "seed": 42, "registry_name": "rft_math",
            "source": "EleutherAI/hendrycks_math", "source_revision": "main",
            "test_source": "HuggingFaceH4/MATH-500", "test_revision": "main",
            "debug_size": 32,
        }}
        with patch("rft_rllm.data._source", source):
            prepared = prepare(config, self.root, register=False)
            reused = prepare(baseline, self.root, register=False)
        self.assertEqual(prepared["splits"]["dev"]["sha256"], reused["splits"]["dev"]["sha256"])
        self.assertEqual(len(load_rows(baseline, self.root, "train")), 2048)

    def test_deepmath_resume_rechecks_protected_math_hashes(self):
        prepare(self.config, self.root, register=False)
        deep = [{"question": f"deep {i}", "final_answer": "1", "topic": "algebra"} for i in range(8192)]
        self.write_source("deep.jsonl", deep)
        config = {"data": {"kind": "deepmath", "name": "deep", "train_jsonl": "deep.jsonl", "train_size": 8192,
                           "math_data_name": "math", "math": self.config["data"]}}
        prepare(config, self.root, register=False)
        directory = data_directory(self.config, self.root)
        path = directory / "dev.jsonl"
        content = path.read_text().replace("training problem", "different problem")
        path.write_text(content)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["splits"]["dev"]["sha256"] = _digest(path.read_bytes())
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "protected MATH dev/test hashes changed"):
            prepare(config, self.root, register=False)

    def test_audited_empty_box_requires_both_fingerprints(self):
        rows = [{'problem': 'For any integer $n>1$, the number of prime numbers greater than $n!+1$ and less than $n!+n$ is:\n$\\text{(A) } 0\\quad\\qquad \\text{(B) } 1\\quad\\\\ \\text{(C) } \\frac{n}{2} \\text{ for n even, } \\frac{n+1}{2} \\text{ for n odd}\\quad\\\\ \\text{(D) } n-1\\quad \\text{(E) } n$\n', 'solution': 'Observe that for all $k \\in 1< k< n$, since $k$ divides $n!$, $k$ also divides $n!+k$. Therefore, all numbers $a$ in the range $n!+1<a<n!+n$ are composite. Therefore there are $\\boxed{}$ primes in that range.'}, {'problem': 'You are given a sequence of $58$ terms; each term has the form $P+n$ where $P$ stands for the product $2 \\times 3 \\times 5 \\times\\ldots \\times 61$ of all prime numbers less than or equal to $61$, and $n$ takes, successively, the values $2, 3, 4,\\ldots, 59$. Let $N$ be the number of primes appearing in this sequence. Then $N$ is:\n$\\textbf{(A)}\\ 0\\qquad \\textbf{(B)}\\ 16\\qquad \\textbf{(C)}\\ 17\\qquad \\textbf{(D)}\\ 57\\qquad \\textbf{(E)}\\ 58$\n', 'solution': 'First, note that $n$ does not have a prime number larger than $61$ as one of its factors. Also, note that $n$ does not equal $1$.\nTherefore, since the prime factorization of $n$ only has primes from $2$ to $59$, $n$ and $P$ share at least one common factor other than $1$. Therefore $P+n$ is not prime for any $n$, so the answer is $\\boxed{}$.'}]
        for row in rows:
            answer, adjustment = math_reference_answer(row["problem"], row["solution"])
            self.assertEqual(answer, "0")
            self.assertEqual(adjustment["kind"], "audited_empty_box")
            self.assertEqual(math_reference_answer(row["problem"] + " changed", row["solution"]), (None, None))
            self.assertEqual(math_reference_answer(row["problem"], row["solution"] + " changed"), (None, None))

    def test_reference_format_repairs_do_not_guess_malformed_answers(self):
        answer, adjustment = math_reference_answer("q", r"The answer is $\boxed 2$.")
        self.assertEqual(answer, "2")
        self.assertEqual(adjustment["kind"], "unbraced_digit_box")
        self.assertEqual(math_reference_answer("q", r"Earlier \boxed{1}; final \boxed{}"), (None, None))
        self.assertEqual(math_reference_answer("q", r"The answer is \boxed banana"), (None, None))
        row = _normalize({"problem": "q", "solution": r"\boxed{2}", "level": "Level ?"}, "train")
        self.assertIsNone(row["level"])


if __name__ == "__main__":
    unittest.main()
