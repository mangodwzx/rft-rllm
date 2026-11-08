"""CPU integration checks across configuration, data, eval identity and reports."""
import contextlib
import csv
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rft_rllm.cli import main, plan
from rft_rllm.config import MODELS, load_config, select_model, validate
from rft_rllm.data import data_directory, load_rows, registry_name
from rft_rllm.metrics import task_fingerprint
from rft_rllm.pipeline import Pipeline, eval_command, sft_command
from rft_rllm.prompts import RAW_CHAT_TEMPLATE
from rft_rllm.runtime import prepare_base_checkpoint, server_command

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def config(self):
        return load_config(ROOT / "configs" / "baseline.json")

    def test_presets_and_all_four_models_plan_without_gpu(self):
        for path in (ROOT / "configs").glob("*.json"):
            config = load_config(path)
            for key in MODELS:
                selected = select_model(config, key)
                validate(selected)
                self.assertEqual(selected["model"]["repo_id"], MODELS[key])
                with contextlib.redirect_stdout(io.StringIO()):
                    plan(selected, ROOT)
        full = load_config(ROOT / "configs" / "formal.json")
        self.assertEqual(full["data"]["train_size"], "all")
        self.assertEqual(full["evaluation"]["attempts"], 256)
        self.assertEqual(data_directory(full, ROOT), data_directory(self.config(), ROOT))
        self.assertEqual(registry_name(full, "test"), registry_name(self.config(), "test"))
        self.assertNotEqual(registry_name(full, "train"), registry_name(self.config(), "train"))

    def test_debug_all_cannot_start_training(self):
        config = load_config(ROOT / "configs" / "gsm8k_debug.json")
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Pipeline(config, Path(temporary))
            with patch.object(pipeline, "prepare"), patch.object(pipeline, "debug") as debug, patch.object(pipeline, "rft") as rft, patch.object(pipeline, "grpo") as grpo:
                pipeline.run("all")
                debug.assert_called_once_with()
                rft.assert_not_called()
                grpo.assert_not_called()
            with self.assertRaisesRegex(ValueError, "generation/grading only"):
                pipeline.run("rft")

    def test_commands_keep_splits_separate_and_use_raw_sft(self):
        config = self.config()
        train = eval_command(config, "train", 8, Path("/tmp/collect"), "http://127.0.0.1:8010/v1")
        test = eval_command(config, "test", 256, Path("/tmp/test"), "http://127.0.0.1:8010/v1")
        self.assertNotEqual(train[2], test[2])
        command = sft_command(config, "accepted", "/model", Path("/runs/exp/rft/round_01/train"))
        self.assertIn("rft_rllm.train_sft", command)
        self.assertEqual(command[command.index("--config") + 1], "/runs/exp/config.json")
        command = server_command(config, "/model")
        self.assertEqual(command[command.index("--chat-template") + 1], RAW_CHAT_TEMPLATE)

    def test_model_overlay_leaves_source_tokenizer_and_weights_unchanged(self):
        class Tokenizer:
            chat_template = None

            def save_pretrained(self, directory):
                Path(directory, "tokenizer_config.json").write_text(json.dumps({"chat_template": self.chat_template}))

        auto = types.SimpleNamespace(from_pretrained=lambda _: Tokenizer())
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "prepared"
            source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "model.safetensors").write_bytes(b"test weights")
            (source / "tokenizer_config.json").write_text("{}")
            with patch.dict("sys.modules", {"transformers": types.SimpleNamespace(AutoTokenizer=auto)}):
                prepare_base_checkpoint(source, target)
            self.assertEqual((source / "tokenizer_config.json").read_text(), "{}")
            self.assertTrue((target / "model.safetensors").is_symlink())
            self.assertFalse((target / "tokenizer_config.json").is_symlink())
            self.assertEqual((target / "chat_template.jinja").read_text(), RAW_CHAT_TEMPLATE)
            self.assertEqual((source / "model.safetensors").read_bytes(), b"test weights")

    def test_data_eval_and_report_integration_with_synthetic_rollouts(self):
        config = self.config()
        config["data"].update(pilot_size=2, train_size=2, dev_size=1)
        config["evaluation"].update(attempts=2, ks=[1, 2])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [{"problem": f"Train {i}", "solution": r"Answer \boxed{1}", "level": f"Level {i % 5 + 1}", "type": "Algebra"} for i in range(8)]
            test = [{"problem": f"Test {i}", "solution": r"Answer \boxed{1}", "level": f"Level {level}", "type": "Algebra"} for i, level in enumerate((2, 4, 5))]
            for name, rows in (("raw_train", train), ("raw_test", test)):
                (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            config["data"].update(train_jsonl="raw_train.jsonl", test_jsonl="raw_test.jsonl")
            pipeline = Pipeline(config, root)
            with patch("rft_rllm.data._register"):
                pipeline.prepare()
            rows = load_rows(config, root, "test")

            def fake_eval(argv, label):
                directory = Path(argv[argv.index("--episodes-dir") + 1])
                directory.mkdir(parents=True)
                successes = {"base_eval": 0, "rft_eval": 1, "grpo_eval": 2}[label]
                items = [{"idx": i, "attempt": a, "is_correct": a < successes, "error": None,
                          "signals": {"task_fingerprint": task_fingerprint(row["question"]), "response_tokens": 10, "truncated": 0}}
                         for i, row in enumerate(rows) for a in range(2)]
                (directory / "results.json").write_text(json.dumps({"attempts": 2, "total": len(items), "items": items,
                    "dataset_name": registry_name(config, "test"), "agent": "rft_math", "model": "rft-model"}))

            with patch("rft_rllm.pipeline.serving", return_value=contextlib.nullcontext("http://127.0.0.1/v1")), patch.object(pipeline.runner, "run", side_effect=fake_eval):
                for group in ("base", "rft", "grpo"):
                    pipeline.state["models"][group] = f"/{group}"
                    pipeline.evaluate_model(f"/{group}", f"{group}_eval", "test", 2, pipeline.output / f"{group}_eval")
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.report()
            summary = json.loads((pipeline.output / "summary.json").read_text())
            self.assertEqual(summary["groups"]["grpo"]["subsets"]["level_4_5"]["tasks"], 2)
            self.assertEqual(summary["comparisons"]["grpo_minus_rft"]["level_4_5"]["pass_at_delta"]["1"], .5)
            with (pipeline.output / "summary.csv").open(encoding="utf-8-sig") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len([row for row in csv_rows if row["subset"] == "level_4_5"]), 3)
            self.assertTrue((pipeline.output / "pass_at_k.svg").is_file())
            self.assertTrue((pipeline.output / "comparison.csv").is_file())

    def test_cli_models_needs_no_model_or_data_download(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["models"])
        self.assertEqual(len(output.getvalue().strip().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
