import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rft_rllm.prompts import DEMONSTRATIONS, INSTRUCTION, raw_messages, render_prompt


def load_flow(client, grader):
    modules = {name: ModuleType(name) for name in (
        "rllm", "openai", "rllm.eval", "rllm.eval.reward_fns", "rllm.eval.reward_fns.math",
        "rllm.eval.types", "rllm.types",
    )}
    modules["rllm"].rollout = lambda **kwargs: lambda fn: fn
    modules["rllm"].evaluator = lambda fn: fn
    modules["openai"].AsyncOpenAI = client
    modules["rllm.eval.reward_fns.math"].evaluate = grader
    modules["rllm.eval.types"].Signal = SimpleNamespace
    for name in ("AgentConfig", "Episode", "Step", "Task", "Trajectory"):
        setattr(modules["rllm.types"], name, SimpleNamespace)
    path = Path(__file__).parents[1] / "src/rft_rllm/flow.py"
    spec = importlib.util.spec_from_file_location("rft_rllm._flow_test", path)
    flow = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(flow)
    return flow


class RawFlowTests(unittest.TestCase):
    def test_prompt_matches_original_handwritten_protocol(self):
        prompt = render_prompt("  Find x.  ")
        self.assertEqual(len(DEMONSTRATIONS), 3)
        self.assertTrue(prompt.startswith(INSTRUCTION + "\n\nProblem: Solve 3x + 7 = 22"))
        self.assertTrue(prompt.endswith("\n\nProblem: Find x.\nSolution:"))
        self.assertEqual(raw_messages("Find x."), [{"role": "user", "content": prompt}])
        self.assertNotIn("<|im_start|>", prompt)
        self.assertNotIn("<think>", prompt)
        with self.assertRaises(ValueError):
            render_prompt(" ")

    def test_flow_preserves_completion_and_records_observed_usage(self):
        calls = {}
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="\n reason \\boxed{2}"), finish_reason="length")],
            usage=SimpleNamespace(completion_tokens=9, prompt_tokens=250),
        )

        class Client:
            def __init__(self, **kwargs):
                calls["client"] = kwargs
                self.chat = SimpleNamespace(completions=self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def create(self, **kwargs):
                calls["request"] = {**kwargs, "messages": list(kwargs["messages"])}
                return response

        flow = load_flow(Client, lambda task, episode: SimpleNamespace(signals=[]))
        task = SimpleNamespace(metadata={"question": "1+1?", "ground_truth": "2"}, instruction="")
        episode = asyncio.run(flow.math_flow(task, SimpleNamespace(base_url="http://gateway/v1", model="base")))
        self.assertEqual(calls["client"]["max_retries"], 0)
        self.assertEqual(calls["request"]["messages"], raw_messages("1+1?"))
        self.assertEqual(calls["request"]["extra_body"], {"add_generation_prompt": False, "add_special_tokens": False})
        self.assertEqual(episode.artifacts["answer"], "\n reason \\boxed{2}")
        self.assertEqual(episode.metrics["truncated"], 1)
        self.assertEqual(episode.metrics["response_tokens"], 9)
        result = flow.math_evaluator(task, episode)
        signals = {signal.name: signal.value for signal in result.signals}
        self.assertEqual(signals["task_fingerprint"], float(flow.task_fingerprint("1+1?")))
        self.assertEqual(signals["prompt_tokens"], 250)

    def test_flow_does_not_replace_failed_generation_with_empty_answer(self):
        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def create(self, **kwargs):
                raise TimeoutError("generation timed out")

        flow = load_flow(Client, None)
        task = SimpleNamespace(metadata={"question": "1+1?"}, instruction="")
        with self.assertRaises(TimeoutError):
            asyncio.run(flow.math_flow(task, SimpleNamespace(base_url="http://gateway/v1", model="base")))


if __name__ == "__main__":
    unittest.main()
