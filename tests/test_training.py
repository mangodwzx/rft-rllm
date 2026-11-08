import json
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rft_rllm.prompts import RAW_CHAT_TEMPLATE, raw_messages, render_prompt
from rft_rllm.raw_sft_dataset import tokenize_raw_messages
from rft_rllm.train_grpo import build_overrides
from rft_rllm.train_sft import main as sft_main, sft_overrides
from rft_rllm import verl_compat


class Tokenizer:
    eos_token_id = 1
    chat_template = RAW_CHAT_TEMPLATE

    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Raw completion must not add BOS/role tokens")
        return [ord(char) + 10 for char in text]


class RawTrainingTests(unittest.TestCase):
    def settings(self):
        return json.loads((Path(__file__).parents[1] / "configs/baseline.json").read_text())

    def test_sft_masks_entire_prompt_and_preserves_response_whitespace(self):
        messages = raw_messages("Compute 2+2") + [{"role": "assistant", "content": "\n 4"}]
        ids, mask = tokenize_raw_messages(messages, Tokenizer(), 10000)
        prompt_ids = Tokenizer().encode(render_prompt("Compute 2+2"), False)
        self.assertEqual(ids, prompt_ids + [20, 42, 62, 1])
        self.assertEqual(mask, [0] * len(prompt_ids) + [1, 1, 1, 1])

    def test_sft_rejects_chat_and_overlength_instead_of_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "shared raw math protocol"):
            tokenize_raw_messages([{"role": "user", "content": "Compute 2+2"}, {"role": "assistant", "content": "4"}], Tokenizer(), 1000)
        messages = raw_messages("Compute 2+2") + [{"role": "assistant", "content": "4"}]
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            tokenize_raw_messages(messages, Tokenizer(), 10)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            tokenize_raw_messages(messages + messages, Tokenizer(), 10000)

    def test_sft_does_not_merge_bpe_across_fixed_generation_boundary(self):
        class BoundaryTokenizer(Tokenizer):
            def encode(self, text, add_special_tokens):
                # Like Qwen's BPE, a colon followed by newline can form one
                # token. The sampled prompt already ended at the colon.
                if "Solution:\n" in text:
                    return [999999]
                return super().encode(text, add_special_tokens)

        tokenizer = BoundaryTokenizer()
        messages = raw_messages("2+2?") + [{"role": "assistant", "content": "\n4"}]
        ids, mask = tokenize_raw_messages(messages, tokenizer, 10000)
        prompt_ids = tokenizer.encode(messages[0]["content"], False)
        self.assertEqual(ids[:len(prompt_ids)], prompt_ids)
        self.assertEqual(ids[len(prompt_ids):], [20, 62, tokenizer.eos_token_id])
        self.assertNotIn(999999, ids)
        self.assertEqual(mask[len(prompt_ids):], [1, 1, 1])

    def test_grpo_uses_same_identity_template_and_neutral_generation_defaults(self):
        settings = self.settings()
        overrides = build_overrides(settings, "/models/base", Path("/tmp/grpo"))
        values = {key.lstrip("+"): json.loads(value) for key, value in (entry.split("=", 1) for entry in overrides[1:])}
        self.assertEqual(values["actor_rollout_ref.model.custom_chat_template"], RAW_CHAT_TEMPLATE)
        self.assertEqual(values["actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template"], RAW_CHAT_TEMPLATE)
        self.assertEqual(values["actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template_content_format"], "string")
        self.assertEqual(values["actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config"], "vllm")
        self.assertEqual(values["rllm.workflow.retry_limit"], 1)
        self.assertEqual(values["rllm.rollout.train.max_tokens"], settings["sampling"]["max_tokens"])
        self.assertEqual(sft_overrides(settings)["data"]["truncation"], "error")

    def test_sft_dispatches_official_trainer_with_custom_data_and_no_validation(self):
        captured = {}
        settings = self.settings()
        modules = {name: ModuleType(name) for name in (
            "rllm", "rllm.data", "rllm.trainer", "rllm.trainer.agent_sft_trainer", "rllm.trainer.sft", "transformers",
        )}
        dataset = [{}] * settings["rft"]["sft"]["batch_size"]
        modules["rllm.data"].DatasetRegistry = SimpleNamespace(load_dataset=lambda name, split: dataset)
        modules["rllm.trainer.sft"].SFTSpec = SimpleNamespace
        modules["transformers"].AutoTokenizer = SimpleNamespace(from_pretrained=lambda path: Tokenizer())

        class Trainer:
            def __init__(self, spec, backend):
                captured.update(spec=spec, backend=backend)

            def train(self):
                captured["trained"] = True

        modules["rllm.trainer.agent_sft_trainer"].AgentSFTTrainer = Trainer
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(settings))
            with patch.dict(sys.modules, modules):
                sft_main(["--config", str(config), "--dataset", "accepted", "--model", "/models/base", "--output", str(Path(directory) / "sft")])
        self.assertEqual(captured["backend"], "verl")
        self.assertTrue(captured["trained"])
        self.assertIsNone(captured["spec"].val_dataset)
        self.assertEqual(captured["spec"].overrides["data"]["custom_cls"]["name"], "RawCompletionSFTDataset")
        self.assertEqual(captured["spec"].lora_rank, 0)


def original_builder_for_guard(accumulated, pad_token_id, max_prompt_length, max_response_length, processor=None):
    return accumulated


class CompatibilityWiringTests(unittest.TestCase):
    def test_patch_rejects_changed_source_and_is_idempotent(self):
        transform = SimpleNamespace(_batch_tensors_and_build_data_proto=original_builder_for_guard)
        expected = hashlib.sha256(inspect.getsource(original_builder_for_guard).strip().encode()).hexdigest()
        with patch.object(verl_compat.importlib.metadata, "version", return_value="0.8.0"), patch.object(
            verl_compat.importlib, "import_module", return_value=transform
        ):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                verl_compat.install()
            with patch.object(verl_compat, "BUILDER_SHA256", expected):
                verl_compat.install()
                wrapped = transform._batch_tensors_and_build_data_proto
                verl_compat.install()
                self.assertIs(transform._batch_tensors_and_build_data_proto, wrapped)

    def test_ray_worker_hook_preserves_official_runtime_and_resource_settings(self):
        calls = {}
        ray = ModuleType("ray")
        ray.is_initialized = lambda: False
        ray.init = lambda **kwargs: calls.update(kwargs)
        settings = ModuleType("rllm.trainer.ray_init_utils")
        settings.get_ray_init_settings = lambda config: {"address": "auto", "num_cpus": 3}
        runtime = ModuleType("rllm.trainer.verl.ray_runtime_env")
        runtime.get_ppo_ray_runtime_env = lambda: {"env_vars": {"RLLM_HOME": "/tmp/registry"}, "working_dir": None}
        with patch.dict(sys.modules, {"ray": ray, settings.__name__: settings, runtime.__name__: runtime}):
            self.assertIs(verl_compat.initialize_ray(SimpleNamespace()), ray)
            self.assertEqual(calls["address"], "auto")
            self.assertEqual(calls["num_cpus"], 3)
            self.assertEqual(calls["runtime_env"]["env_vars"], {"RLLM_HOME": "/tmp/registry"})
            self.assertEqual(calls["runtime_env"]["worker_process_setup_hook"], "rft_rllm.verl_compat.install")
            ray.is_initialized = lambda: True
            with self.assertRaisesRegex(RuntimeError, "fresh Python process"):
                verl_compat.initialize_ray(SimpleNamespace())


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU tensor regression requires the optional torch training dependency")
class LengthMaskTests(unittest.TestCase):
    def test_real_eos_keeps_nonzero_position_and_response_mask(self):
        import torch

        inputs = torch.tensor([[0, 11, 12, 13, 21, 0, 0, 0]])
        result = SimpleNamespace(batch={
            "input_ids": inputs, "attention_mask": inputs != 0,
            "response_mask": torch.tensor([[1, 1, 0, 0]]),
            "rollout_log_probs": torch.tensor([[-0.2, -0.3, 0, 0]]),
        }, meta_info={})
        original_response_mask = result.batch["response_mask"].clone()
        original_logprobs = result.batch["rollout_log_probs"].clone()
        accumulated = SimpleNamespace(prompts=[[11, 12, 13]], responses=[[21, 0]])
        verl_compat.restore_length_masks(result, accumulated, 4, 4, None, None)
        self.assertEqual(result.batch["attention_mask"].tolist(), [[False, True, True, True, True, True, False, False]])
        self.assertEqual(result.batch["position_ids"].tolist(), [[0, 0, 1, 2, 3, 4, 0, 0]])
        self.assertTrue(torch.equal(result.batch["response_mask"], original_response_mask))
        self.assertTrue(torch.equal(result.batch["rollout_log_probs"], original_logprobs))
        self.assertIs(result.batch["input_ids"], inputs)

    def test_distinct_padding_keeps_official_masks_identical(self):
        import torch

        inputs = torch.tensor([[0, 11, 12, 21, 22, 0], [11, 12, 13, 23, 0, 0]])
        mask = inputs != 0
        positions = (torch.cumsum(mask, dim=1) - 1) * mask
        result = SimpleNamespace(batch={"input_ids": inputs, "attention_mask": mask,
            "position_ids": positions, "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]])}, meta_info={})
        accumulated = SimpleNamespace(prompts=[[11, 12], [11, 12, 13]], responses=[[21, 22], [23]])
        verl_compat.restore_length_masks(result, accumulated, 3, 3, None, None)
        self.assertTrue(torch.equal(result.batch["attention_mask"], mask))
        self.assertTrue(torch.equal(result.batch["position_ids"], positions))

    def test_observation_tokens_and_multimodal_positions_use_complete_lengths(self):
        import torch

        inputs = torch.tensor([[0, 11, 21, 0, 8, 9, 22, 0]])
        action_mask = torch.tensor([[1, 1, 0, 0, 1, 0]])
        result = SimpleNamespace(batch={"input_ids": inputs, "response_mask": action_mask}, meta_info={})
        accumulated = SimpleNamespace(prompts=[[11]], responses=[[21, 0, 8, 9, 22]], multi_modal_inputs=[{}])
        processor = SimpleNamespace(image_processor=type("Qwen2VLImageProcessor", (), {})())
        calls = {}

        def positions_handler(processor, input_ids, mask, mm):
            calls["mask"] = mask
            return torch.ones((1, 4, 8), dtype=torch.long)

        verl_compat.restore_length_masks(result, accumulated, 2, 6, processor, positions_handler)
        self.assertEqual(calls["mask"].tolist(), [[False, True, True, True, True, True, True, False]])
        self.assertIs(result.batch["response_mask"], action_mask)
        self.assertEqual(tuple(result.batch["position_ids"].shape), (1, 4, 8))

    def test_inconsistent_shape_is_rejected(self):
        import torch

        result = SimpleNamespace(batch={"input_ids": torch.ones((1, 3))}, meta_info={})
        with self.assertRaisesRegex(ValueError, "batch shape"):
            verl_compat.restore_length_masks(result, SimpleNamespace(prompts=[[1]], responses=[[2]]), 2, 2, None, None)


if __name__ == "__main__":
    unittest.main()
