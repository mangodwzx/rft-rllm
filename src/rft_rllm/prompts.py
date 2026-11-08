"""The shared raw-completion protocol for all four Base model families.

The demonstrations are handwritten and copied from the original rft_verl
experiment. The identity template is only an OpenAI chat API transport adapter:
it adds no role, thinking, BOS, EOS, or assistant-prefix tokens to the prompt.
"""
from __future__ import annotations

import hashlib

PROMPT_VERSION = "raw_math_fewshot_v1"
DEMONSTRATIONS = (
    ("Solve 3x + 7 = 22 for x.", r"Subtracting 7 gives 3x = 15. Dividing by 3 gives x = 5. The final answer is \boxed{5}."),
    ("A rectangle has side lengths 4 and 9. What is its area?", r"Its area is the product of its side lengths: 4 \times 9 = 36. The final answer is \boxed{36}."),
    ("What is the probability of rolling an even number on a fair six-sided die?", r"The even outcomes are 2, 4, and 6, so there are 3 favorable outcomes among 6 equally likely outcomes. The final answer is \boxed{\frac{1}{2}}."),
)
INSTRUCTION = r"Solve the following mathematics problem. Show your reasoning, and put your final answer in \boxed{...}."
RAW_CHAT_TEMPLATE = "{{ messages[0]['content'] }}"
STOP_STRINGS: list[str] = []


def render_prompt(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    examples = "\n\n".join(f"Problem: {question}\nSolution: {solution}" for question, solution in DEMONSTRATIONS)
    return f"{INSTRUCTION}\n\n{examples}\n\nProblem: {problem.strip()}\nSolution:"


def prompt_fingerprint() -> str:
    return hashlib.sha256(render_prompt("<PROBLEM>").encode()).hexdigest()


def raw_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": render_prompt(problem)}]
