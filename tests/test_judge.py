"""Tests for the LLM-as-judge module's parsing + prompt-template handling.

The OpenAI call itself is not exercised here -- those tests require an API
key and would hit the network. We pin (a) the score parser's behavior on
realistic / adversarial judge outputs and (b) the eval_prompt template
filling so `{question}` / `{answer}` are substituted correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

from judge import format_eval_prompt, load_eval_prompt_template, parse_score  # noqa: E402


# --- parse_score ------------------------------------------------------------


def test_parse_score_int_score():
    assert parse_score("75") == 75
    assert parse_score("0") == 0
    assert parse_score("100") == 100


def test_parse_score_strips_surrounding_text():
    assert parse_score("  Score: 42  ") == 42
    assert parse_score("The response scores 88 out of 100.") == 88


def test_parse_score_refusal():
    assert parse_score("REFUSAL") == "REFUSAL"
    assert parse_score("refusal") == "REFUSAL"
    assert parse_score("  Refusal\n") == "REFUSAL"
    assert parse_score("The model produced a REFUSAL response.") == "REFUSAL"


def test_parse_score_refusal_takes_precedence_over_number():
    """If the judge says REFUSAL AND mentions a number, REFUSAL wins."""
    assert parse_score("REFUSAL (would have scored 50)") == "REFUSAL"


def test_parse_score_rejects_out_of_range_number_alone():
    with pytest.raises(ValueError, match="unparseable"):
        parse_score("150")


def test_parse_score_rejects_garbage():
    with pytest.raises(ValueError, match="unparseable"):
        parse_score("the answer is unclear")


# --- eval_prompt template handling ------------------------------------------


@pytest.mark.parametrize("trait", ["evil", "hallucinating", "sycophantic"])
def test_load_eval_prompt_has_placeholders(trait):
    template = load_eval_prompt_template(trait)
    assert "{question}" in template
    assert "{answer}" in template
    # Sanity-check it's a non-trivial prompt mentioning the trait.
    assert trait[:5] in template.lower() or trait in template.lower()


def test_format_eval_prompt_substitutes_question_and_answer():
    q = "Test question text"
    a = "Test answer text"
    out = format_eval_prompt("evil", q, a)
    assert q in out
    assert a in out
    assert "{question}" not in out
    assert "{answer}" not in out


def test_format_eval_prompt_does_not_clobber_other_braces():
    """The eval_prompt may contain other `{...}` substrings (e.g. JSON
    examples in instructions). We use `str.replace`, not `.format`, so
    those should pass through untouched."""
    template = load_eval_prompt_template("hallucinating")
    out = format_eval_prompt("hallucinating", "Q", "A")
    # `str.replace`-based substitution can't raise for stray braces;
    # `.format`-based substitution would. Demonstrate by checking that
    # all the non-placeholder content of the template is still present.
    for chunk in template.split("{question}"):
        for sub in chunk.split("{answer}"):
            if sub:  # skip empty fragments around the placeholders
                assert sub in out
