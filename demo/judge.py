"""LLM-as-judge for persona-trait elicitation scoring.

Uses the trait JSON's `eval_prompt` template (already shipped in
`data_generation/trait_data_extract/<trait>.json`) to ask an external API
whether a `(question, response)` pair exhibits the trait. Returns either an
integer score in `[0, 100]` or the sentinel string `"REFUSAL"`.

Default model: `gpt-5.4`. Requires the `OPENAI_API_KEY` environment variable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIT_DIR = REPO_ROOT / "data_generation" / "trait_data_extract"

JudgeScore = Union[int, str]  # int in [0, 100] or "REFUSAL"


def load_eval_prompt_template(trait: str) -> str:
    """Load the `eval_prompt` from `<trait>.json`."""
    return json.loads((TRAIT_DIR / f"{trait}.json").read_text())["eval_prompt"]


def format_eval_prompt(trait: str, question: str, answer: str) -> str:
    """Apply the eval_prompt's `{question}` and `{answer}` placeholders.

    The shipped templates use plain `{name}` markers (not f-string) so
    `str.replace` is the safest way to fill them; `.format` would fail on
    any other `{...}` substring in the template body.
    """
    template = load_eval_prompt_template(trait)
    return template.replace("{question}", question).replace("{answer}", answer)


def parse_score(response_text: str) -> JudgeScore:
    """Parse a judge response into an int 0-100 or the string 'REFUSAL'."""
    text_upper = response_text.strip().upper()
    if "REFUSAL" in text_upper:
        return "REFUSAL"
    match = re.search(r"\b(\d{1,3})\b", response_text)
    if match:
        v = int(match.group(1))
        if 0 <= v <= 100:
            return v
    raise ValueError(f"unparseable judge response: {response_text!r}")


def make_openai_judge(
    model: str = "gpt-5.4",
    api_key: str | None = None,
    max_tokens: int = 10,
) -> Callable[[str, str, str], JudgeScore]:
    """Return `judge(question, response, trait) -> JudgeScore`.

    Lazily imports the `openai` SDK so callers that never use the judge
    don't pay the import cost. Reads the API key from `OPENAI_API_KEY` if
    `api_key` is `None`.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def judge(question: str, response: str, trait: str) -> JudgeScore:
        prompt = format_eval_prompt(trait, question, response)
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # `max_completion_tokens` is the new-API spelling required by the
            # gpt-5 / o-series families. Older models also accept it on
            # recent SDK versions, so it's the safer default.
            max_completion_tokens=max_tokens,
            temperature=0.0,
        )
        return parse_score(result.choices[0].message.content)

    return judge


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="One-shot LLM judge call.")
    parser.add_argument("--trait", required=True, choices=["evil", "hallucinating", "sycophantic"])
    parser.add_argument("--question", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()
    judge = make_openai_judge(model=args.model)
    print(judge(args.question, args.response, args.trait))
