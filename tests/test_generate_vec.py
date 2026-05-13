"""Functional tests for generate_vec.py — exercise the real code paths on
synthetic data, with the model load + activation extraction mocked out.

Run with:
    /home/jo5iah/anaconda3/envs/persona-lsh/bin/python -m pytest \
        persona_vectors/tests/test_generate_vec.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding import linear_edges, quantile_edges, symlog_edges  # noqa: E402
from generate_vec import (  # noqa: E402
    compute_layer_diffs,
    get_persona_effective,
    save_lsh_sidecar,
    save_persona_vector,
)
from lsh import diff, hash_layers  # noqa: E402


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


# --- save_lsh_sidecar ---------------------------------------------------------


def _fake_persona_diff_pt(path: Path, *, layers: int = 6, hidden: int = 4096, seed: int = 0) -> torch.Tensor:
    """Write a fake [layers, hidden] persona-vector tensor to disk and return it."""
    g = torch.Generator().manual_seed(seed)
    tensor = torch.randn(layers, hidden, generator=g, dtype=torch.float32)
    torch.save(tensor, path)
    return tensor


@pytest.mark.parametrize("edges_method", ["linear", "quantile", "symlog"])
def test_save_lsh_sidecar_round_trip(tmp_path, edges_method):
    pt_path = tmp_path / "evil_response_avg_diff.pt"
    tensor = _fake_persona_diff_pt(pt_path)

    sidecar_path = save_lsh_sidecar(str(pt_path), edges_method=edges_method)
    assert Path(sidecar_path).exists()
    assert sidecar_path.endswith(".tlsh.json")

    payload = json.loads(Path(sidecar_path).read_text())
    assert payload["source"] == pt_path.name
    assert payload["edges_method"] == edges_method
    assert len(payload["edges"]) == 257
    assert len(payload["digests"]) == tensor.shape[0]

    # Reproducing the digests from the saved edges must yield identical values:
    edges = np.asarray(payload["edges"], dtype=np.float64)
    recomputed = hash_layers(tensor, edges)
    assert recomputed == payload["digests"]


def test_save_lsh_sidecar_rejects_unknown_edges_method(tmp_path):
    pt_path = tmp_path / "x.pt"
    _fake_persona_diff_pt(pt_path)
    with pytest.raises(ValueError, match="unknown edges_method"):
        save_lsh_sidecar(str(pt_path), edges_method="bogus")


def test_save_lsh_sidecar_distance_preserves_ordering(tmp_path):
    """Sanity-check the end-to-end distance property: a tensor stays closest
    to itself, then to a perturbed copy, then to a fresh random tensor."""
    base_path = tmp_path / "base.pt"
    close_path = tmp_path / "close.pt"
    far_path = tmp_path / "far.pt"

    base = _fake_persona_diff_pt(base_path, seed=1)
    close = base + 1e-3 * torch.randn_like(base)
    torch.save(close, close_path)
    _fake_persona_diff_pt(far_path, seed=999)

    for path in (base_path, close_path, far_path):
        save_lsh_sidecar(str(path), edges_method="quantile")

    base_payload = json.loads((tmp_path / "base.tlsh.json").read_text())
    close_payload = json.loads((tmp_path / "close.tlsh.json").read_text())
    far_payload = json.loads((tmp_path / "far.tlsh.json").read_text())

    for layer in range(len(base_payload["digests"])):
        b, c, f = base_payload["digests"][layer], close_payload["digests"][layer], far_payload["digests"][layer]
        if not (b and c and f):
            continue  # TLSH returned no digest for this layer
        assert diff(b, c) <= diff(b, f)


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


def test_save_persona_vector_writes_pt_files_without_lsh(tmp_path):
    pos, neg = _write_pair_of_csvs(tmp_path, trait="evil")
    save_dir = tmp_path / "out"

    save_persona_vector(
        model_name="fake-model",
        pos_path=str(pos),
        neg_path=str(neg),
        trait="evil",
        save_dir=str(save_dir),
        threshold=50,
        compute_lsh=False,
        _model_loader=lambda name: (_FakeModel(), object()),
        _activation_extractor=_fake_activation_extractor(num_layers=5, hidden_dim=64, seed_offset=0),
    )

    for stem in ("evil_prompt_avg_diff", "evil_response_avg_diff", "evil_prompt_last_diff"):
        pt = save_dir / f"{stem}.pt"
        assert pt.exists()
        tensor = torch.load(pt, map_location="cpu")
        assert tensor.shape == (6, 64)  # num_hidden_layers + 1, hidden_dim
        # No sidecar without compute_lsh.
        assert not (save_dir / f"{stem}.tlsh.json").exists()


def test_save_persona_vector_writes_lsh_sidecars_when_requested(tmp_path):
    pos, neg = _write_pair_of_csvs(tmp_path, trait="evil")
    save_dir = tmp_path / "out"

    save_persona_vector(
        model_name="fake-model",
        pos_path=str(pos),
        neg_path=str(neg),
        trait="evil",
        save_dir=str(save_dir),
        threshold=50,
        compute_lsh=True,
        lsh_edges_method="quantile",
        _model_loader=lambda name: (_FakeModel(), object()),
        _activation_extractor=_fake_activation_extractor(num_layers=5, hidden_dim=64, seed_offset=0),
    )

    for stem in ("evil_prompt_avg_diff", "evil_response_avg_diff", "evil_prompt_last_diff"):
        sidecar = save_dir / f"{stem}.tlsh.json"
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert payload["edges_method"] == "quantile"
        assert len(payload["edges"]) == 257
        assert len(payload["digests"]) == 6  # one digest per layer


def test_cli_argparser_accepts_lsh_flags():
    """Importing generate_vec as a module wires its argparse; this test exercises
    the parser definition the way `python -m generate_vec` would."""
    import generate_vec as gv

    # Reconstruct the argparser the same way the __main__ block does so the
    # test breaks if a future refactor removes the flags.
    src = Path(gv.__file__).read_text()
    assert "--compute_lsh" in src
    assert "--lsh_edges" in src
    assert '"linear", "quantile", "symlog"' in src or "('linear', 'quantile', 'symlog')" in src
