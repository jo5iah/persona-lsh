"""Smoke test for persona_vectors.lsh + persona_vectors.encoding.

Run with:  `python -m pytest persona_vectors/tests/test_lsh.py`
or:        `python persona_vectors/tests/test_lsh.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding import (  # noqa: E402
    N_BUCKETS,
    encode_vector,
    linear_edges,
    quantile_edges,
    symlog_edges,
)
from lsh import diff, hash_layers, hash_vector  # noqa: E402


def _fake_persona_vector(layers: int = 28, hidden: int = 3584, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(layers, hidden, generator=g, dtype=torch.float32)


# --- encoding tests ----------------------------------------------------------


def test_encoding_fractional_byte2_contract():
    # Bucket 27 covers [0.7, 0.8). With fractional byte-2:
    #   - 0.78 (frac~0.8)  -> b2 = round(255 * 0.8)  = 204
    #   - 0.72 (frac~0.2)  -> b2 = round(255 * 0.2)  = 51
    #   - 0.75 (frac~0.5)  -> b2 = round(255 * 0.4999..) = 127
    #     (0.75 - 0.7 != 0.05 exactly in float; frac falls just under 0.5).
    # Edges with width 0.1 each, shifted so bucket 27 spans [0.7, 0.8).
    edges = np.arange(257, dtype=np.float64) * 0.1 - 2.0
    assert abs(edges[27] - 0.7) < 1e-9 and abs(edges[28] - 0.8) < 1e-9

    pairs = encode_vector(np.array([0.78, 0.72, 0.75], dtype=np.float32), edges)
    assert list(pairs) == [27, 204, 27, 51, 27, 127]


def test_encoding_clamps_at_extremes():
    edges = linear_edges(-1.0, 1.0)
    # At the very top edge (1.0): searchsorted side='right' puts it past the
    # last bucket; clamped to bucket 255. frac >= 1, b2 clamped to 255.
    # At the very bottom edge (-1.0): bucket 0, frac = 0, b2 = 0.
    # Far above range (10.0): bucket 255 (clamped), b2 = 255.
    # Far below range (-10.0): bucket 0 (clamped), b2 = 0.
    pairs = encode_vector(np.array([1.0, -1.0, 10.0, -10.0], dtype=np.float32), edges)
    assert (pairs[0], pairs[1]) == (N_BUCKETS - 1, N_BUCKETS - 1)
    assert (pairs[2], pairs[3]) == (0, 0)
    assert (pairs[4], pairs[5]) == (N_BUCKETS - 1, N_BUCKETS - 1)
    assert (pairs[6], pairs[7]) == (0, 0)


def test_encoding_byte2_smooth_under_small_perturbations():
    """No hard thresholds: small numeric changes should produce small byte-2
    deltas. This was the main motivation for moving away from the 25/75
    quartile-flip scheme."""
    edges = linear_edges(0.0, 1.0)  # width 1/256 ≈ 0.0039 per bucket
    base = np.array([0.5], dtype=np.float32)
    perturbed = np.array([0.5 + 1e-5], dtype=np.float32)  # tiny shift
    p1 = encode_vector(base, edges)
    p2 = encode_vector(perturbed, edges)
    # Bytes should differ by at most a few units, never by ~256.
    assert abs(int(p1[0]) - int(p2[0])) <= 1
    assert abs(int(p1[1]) - int(p2[1])) <= 5


def test_encoding_byte_length():
    edges = linear_edges(-3.0, 3.0)
    v = np.linspace(-3, 3, 1000, dtype=np.float32)
    pairs = encode_vector(v, edges)
    assert len(pairs) == 2 * 1000


def test_quantile_edges_are_monotonic():
    samples = np.random.RandomState(0).randn(10_000).astype(np.float32)
    edges = quantile_edges(samples)
    assert edges.shape == (257,)
    assert np.all(np.diff(edges) > 0)


def test_symlog_edges_are_monotonic_and_dense_near_zero():
    edges = symlog_edges(-100.0, 100.0, linthresh=1.0)
    assert edges.shape == (257,)
    assert np.all(np.diff(edges) > 0)
    # The 50 edges around the center should span a much smaller range than
    # the outer 50 edges on either side.
    near_zero_span = edges[178] - edges[78]  # 100 mid edges
    outer_span = edges[256] - edges[156]  # last 100 edges
    assert near_zero_span < outer_span


# --- TLSH tests --------------------------------------------------------------


def _default_edges():
    # Standard-normal activations: ±6 covers ~6σ.
    return linear_edges(-6.0, 6.0)


def test_identical_vectors_have_zero_distance():
    edges = _default_edges()
    v = _fake_persona_vector()[0]
    h1 = hash_vector(v, edges)
    h2 = hash_vector(v.clone(), edges)
    assert h1, "TLSH refused the input — vector too uniform or too short"
    assert diff(h1, h2) == 0


def test_different_vectors_have_nonzero_distance():
    edges = _default_edges()
    v1 = _fake_persona_vector(seed=0)[0]
    v2 = _fake_persona_vector(seed=1)[0]
    h1, h2 = hash_vector(v1, edges), hash_vector(v2, edges)
    assert h1 and h2
    assert diff(h1, h2) > 0


def test_similar_vectors_closer_than_random_pairs():
    edges = _default_edges()
    v_base = _fake_persona_vector(seed=42)[0]
    v_close = v_base + 1e-3 * torch.randn_like(v_base)
    v_far = _fake_persona_vector(seed=7)[0]
    h_base = hash_vector(v_base, edges)
    h_close = hash_vector(v_close, edges)
    h_far = hash_vector(v_far, edges)
    assert diff(h_base, h_close) < diff(h_base, h_far)


def test_quantile_edges_work_end_to_end():
    v = _fake_persona_vector(seed=3)[0]
    edges = quantile_edges(v)
    h = hash_vector(v, edges)
    assert h
    assert diff(h, h) == 0


def test_hash_layers_returns_one_digest_per_layer():
    edges = _default_edges()
    t = _fake_persona_vector(layers=4)
    digests = hash_layers(t, edges)
    assert len(digests) == 4
    assert all(isinstance(d, str) for d in digests)


# --- vendored-tlsh fork tests (MOD-128 fix + T2 prefix) ----------------------


def test_digests_use_t2_prefix_and_canonical_length():
    """The vendored tlsh is forked to emit the T2 prefix marking the MOD-128
    bug fix in fast_update5. Every non-TNULL digest must be 72 chars long
    starting with 'T2'."""
    edges = _default_edges()
    digests = hash_layers(_fake_persona_vector(layers=8), edges)
    assert digests, "expected at least one digest"
    for d in digests:
        if d == "TNULL" or d == "":
            continue
        assert d.startswith("T2"), f"expected T2 prefix, got {d[:4]!r}"
        assert len(d) == 72, f"expected length 72, got {len(d)} for {d!r}"


def test_t1_prefixed_strings_are_rejected_by_parser():
    """The fork intentionally breaks compatibility with upstream T1 digests so
    a T2 digest is never accidentally compared against a buggy T1 digest."""
    import tlsh as _tlsh

    t1_string = "T1" + "A" * 70
    obj = _tlsh.Tlsh()
    with pytest.raises(ValueError, match="not a TLSH hex string"):
        obj.fromTlshStr(t1_string)


if __name__ == "__main__":
    test_encoding_fractional_byte2_contract()
    test_encoding_clamps_at_extremes()
    test_encoding_byte2_smooth_under_small_perturbations()
    test_encoding_byte_length()
    test_quantile_edges_are_monotonic()
    test_symlog_edges_are_monotonic_and_dense_near_zero()
    test_identical_vectors_have_zero_distance()
    test_different_vectors_have_nonzero_distance()
    test_similar_vectors_closer_than_random_pairs()
    test_quantile_edges_work_end_to_end()
    test_hash_layers_returns_one_digest_per_layer()
    test_digests_use_t2_prefix_and_canonical_length()
    test_t1_prefixed_strings_are_rejected_by_parser()
    print("all lsh smoke tests passed")
