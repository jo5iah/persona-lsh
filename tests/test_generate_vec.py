"""Functional tests for generate_vec.py — exercise the real code paths on
synthetic data, with the model load + activation extraction mocked out.

Run with:
    /home/jo5iah/anaconda3/envs/persona-lsh/bin/python -m pytest \
        persona_vectors/tests/test_generate_vec.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_vec import (  # noqa: E402
    compute_layer_diffs,
    get_persona_effective,
    save_persona_vector,
)


# --- get_persona_effective ----------------------------------------------------


def _write_persona_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_get_persona_effective_filters_on_trait_and_coherence(tmp_path):
    """A row is kept only if pos[trait]>=threshold, neg[trait]<100-threshold,
    and BOTH pos/neg coherence>=50. Mirror the mask in generate_vec.py."""
    pos = tmp_path / "pos.csv"
    neg = tmp_path / "neg.csv"

    # Indices: 0 keep, 1 drop (pos trait too low), 2 drop (neg trait too high),
    # 3 drop (pos coherence too low), 4 drop (neg coherence too low), 5 keep.
    _write_persona_csv(pos, [
        {"prompt": "p0", "answer": "a0", "evil": 80, "coherence": 90},
        {"prompt": "p1", "answer": "a1", "evil": 30, "coherence": 90},
        {"prompt": "p2", "answer": "a2", "evil": 80, "coherence": 90},
        {"prompt": "p3", "answer": "a3", "evil": 80, "coherence": 10},
        {"prompt": "p4", "answer": "a4", "evil": 80, "coherence": 90},
        {"prompt": "p5", "answer": "a5", "evil": 75, "coherence": 60},
    ])
    _write_persona_csv(neg, [
        {"prompt": "n0", "answer": "b0", "evil": 10, "coherence": 90},
        {"prompt": "n1", "answer": "b1", "evil": 10, "coherence": 90},
        {"prompt": "n2", "answer": "b2", "evil": 60, "coherence": 90},
        {"prompt": "n3", "answer": "b3", "evil": 10, "coherence": 90},
        {"prompt": "n4", "answer": "b4", "evil": 10, "coherence": 10},
        {"prompt": "n5", "answer": "b5", "evil": 20, "coherence": 70},
    ])

    _pos_df, _neg_df, pos_prompts, neg_prompts, pos_answers, neg_answers = get_persona_effective(
        str(pos), str(neg), trait="evil", threshold=50,
    )

    assert pos_prompts == ["p0", "p5"]
    assert neg_prompts == ["n0", "n5"]
    assert pos_answers == ["a0", "a5"]
    assert neg_answers == ["b0", "b5"]


def test_get_persona_effective_respects_threshold_argument(tmp_path):
    pos = tmp_path / "pos.csv"
    neg = tmp_path / "neg.csv"
    _write_persona_csv(pos, [
        {"prompt": "p0", "answer": "a0", "evil": 60, "coherence": 90},
        {"prompt": "p1", "answer": "a1", "evil": 75, "coherence": 90},
    ])
    _write_persona_csv(neg, [
        {"prompt": "n0", "answer": "b0", "evil": 20, "coherence": 90},
        {"prompt": "n1", "answer": "b1", "evil": 20, "coherence": 90},
    ])

    # threshold=70: row 0 fails pos>=70.
    _, _, kept_pos, _, _, _ = get_persona_effective(str(pos), str(neg), trait="evil", threshold=70)
    assert kept_pos == ["p1"]


# --- compute_layer_diffs ------------------------------------------------------


def test_compute_layer_diffs_matches_inline_formula():
    """Reproduce the exact mean-of-pos minus mean-of-neg formula that
    `save_persona_vector` used inline before the refactor."""
    torch.manual_seed(0)
    pos = [torch.randn(4, 8) for _ in range(3)]
    neg = [torch.randn(4, 8) for _ in range(3)]

    expected = torch.stack(
        [pos[l].mean(0).float() - neg[l].mean(0).float() for l in range(3)],
        dim=0,
    )
    actual = compute_layer_diffs(pos, neg)
    assert torch.allclose(actual, expected)
    assert actual.shape == (3, 8)


def test_compute_layer_diffs_rejects_mismatched_layer_counts():
    pos = [torch.randn(2, 4) for _ in range(3)]
    neg = [torch.randn(2, 4) for _ in range(2)]
    with pytest.raises(ValueError, match="layer count mismatch"):
        compute_layer_diffs(pos, neg)


# --- save_persona_vector (end-to-end with mocks) -----------------------------


class _FakeModelConfig:
    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers


class _FakeModel:
    def __init__(self, num_hidden_layers: int = 5):
        self.config = _FakeModelConfig(num_hidden_layers)


def _fake_activation_extractor(*, num_layers: int, hidden_dim: int, seed_offset: int):
    """Return a stand-in for `get_hidden_p_and_r` that yields random per-layer
    tensors with the same shape contract: each of the three lists has length
    `num_layers + 1` and each element is `[n_examples, hidden_dim]`."""

    call_counter = {"n": 0}

    def _extractor(model, tokenizer, prompts, responses):
        call_counter["n"] += 1
        n = len(prompts)
        g = torch.Generator().manual_seed(seed_offset * 1000 + call_counter["n"])
        layer_count = model.config.num_hidden_layers + 1
        prompt_avg = [torch.randn(n, hidden_dim, generator=g) for _ in range(layer_count)]
        prompt_last = [torch.randn(n, hidden_dim, generator=g) for _ in range(layer_count)]
        response_avg = [torch.randn(n, hidden_dim, generator=g) for _ in range(layer_count)]
        return prompt_avg, prompt_last, response_avg

    return _extractor


def _write_pair_of_csvs(tmp_path: Path, trait: str, n_keep: int = 4) -> tuple[Path, Path]:
    pos = tmp_path / "pos.csv"
    neg = tmp_path / "neg.csv"
    _write_persona_csv(pos, [
        {"prompt": f"p{i}", "answer": f"a{i}", trait: 80, "coherence": 90} for i in range(n_keep)
    ])
    _write_persona_csv(neg, [
        {"prompt": f"n{i}", "answer": f"b{i}", trait: 10, "coherence": 90} for i in range(n_keep)
    ])
    return pos, neg


def test_save_persona_vector_writes_pt_files(tmp_path):
    pos, neg = _write_pair_of_csvs(tmp_path, trait="evil")
    save_dir = tmp_path / "out"

    save_persona_vector(
        model_name="fake-model",
        pos_path=str(pos),
        neg_path=str(neg),
        trait="evil",
        save_dir=str(save_dir),
        threshold=50,
        _model_loader=lambda name: (_FakeModel(), object()),
        _activation_extractor=_fake_activation_extractor(num_layers=5, hidden_dim=512, seed_offset=0),
    )

    for stem in ("evil_prompt_avg_diff", "evil_response_avg_diff", "evil_prompt_last_diff"):
        pt = save_dir / f"{stem}.pt"
        assert pt.exists()
        tensor = torch.load(pt, map_location="cpu")
        assert tensor.shape == (6, 512)  # num_hidden_layers + 1, hidden_dim
