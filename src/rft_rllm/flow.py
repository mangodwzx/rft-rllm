"""Official math-cookbook flow structure, with observational length signals.

Source: rllm-org/rllm cookbooks/math at the revision in __init__.py.
The same flow, prompt and upstream math grader serve collection, RL and eval.
"""
from pathlib import Path

import rllm
from openai import AsyncOpenAI
from rllm.eval.reward_fns.math import evaluate as official_math_evaluate
from rllm.eval.types import Signal
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

SYSTEM_PROMPT = (
    "Solve the math problem step by step. Show your reasoning and put the final answer "
    "inside \\boxed{...}. Only your last boxed answer will be graded."
)


@rllm.rollout(name="rft_math")
async def math_flow(task: Task, config: AgentConfig) -> Episode:
    meta = task.metadata or {}
    question = str(meta.get("question") or meta.get("problem") or task.instruction)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    # Gateway applies sampling parameters and captures training token IDs/logprobs.
    # Errors propagate to rLLM so failed attempts remain visible in results.json.
    async with AsyncOpenAI(base_url=config.base_url, api_key="EMPTY", max_retries=0) as client:
        response = await client.chat.completions.create(model=config.model, messages=messages, timeout=300)
    choice = response.choices[0]
    answer = choice.message.content or ""
    telemetry = {"response_chars": len(answer)}
    if choice.finish_reason is not None:
        telemetry["truncated"] = float(choice.finish_reason == "length")
    if response.usage is not None:
        telemetry["response_tokens"] = response.usage.completion_tokens
        telemetry["prompt_tokens"] = response.usage.prompt_tokens
    messages.append({"role": "assistant", "content": answer})
    return Episode(
        trajectories=[Trajectory(name="math", steps=[Step(chat_completions=messages, model_response=answer, action=answer)])],
        artifacts={"answer": answer, "telemetry": telemetry, "finish_reason": choice.finish_reason},
        metrics=dict(telemetry),
    )


@rllm.evaluator
def math_evaluator(task: Task | dict, episode: Episode):
    if isinstance(task, dict):
        task = Task(id=str(task.get("id", "")), instruction="", metadata=task, dataset_dir=Path("."))
    result = official_math_evaluate(task, episode)
    for name, value in episode.artifacts.get("telemetry", {}).items():
        if value is not None:
            result.signals.append(Signal(name=name, value=float(value)))
    return result
