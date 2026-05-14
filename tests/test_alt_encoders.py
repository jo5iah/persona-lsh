"""Tests for the alternative byte encoders.

The pluggable ByteEncoder interface lets us swap in different
quantization/compression schemes per scalar without touching downstream LSH
backends. These tests pin the contract of each new encoder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding import (  # noqa: E402
    BucketEdgesEncoder,
    BucketFractionalEncoder,
    BucketOrderTwoByteEncoder,
    BucketSingleByteEncoder,
    ByteEncoder,
    CanonicalOrderEncoder,
    N_BUCKETS,
    TopKMaskedEncoder,
    linear_edges,
    quantile_edges,
    variance_permutation_per_layer,
)


def _edges_0_to_1():
    return np.arange(257, dtype=np.float64) / 256.0


def _edges_neg_pos():
    return linear_edges(-3.0, 3.0)


# --- #1 BucketSingleByteEncoder ---------------------------------------------


def test_single_byte_is_byte_encoder():
    enc = BucketSingleByteEncoder(_edges_neg_pos())
    assert isinstance(enc, ByteEncoder)
    assert isinstance(enc, BucketEdgesEncoder)


def test_single_byte_emits_one_byte_per_scalar():
    enc = BucketSingleByteEncoder(_edges_neg_pos())
    out = enc.encode_vector(np.array([0.0, 1.5, -1.5], dtype=np.float32))
    assert len(out) == 3


def test_single_byte_matches_byte_1_of_bucket_fractional():
    """The bucket-index byte from BucketSingleByte should be identical to
    byte-1 of BucketFractional on the same edges."""
    edges = _edges_neg_pos()
    v = np.array([0.1, -2.4, 2.9, 0.0, -0.001], dtype=np.float32)
    single = BucketSingleByteEncoder(edges).encode_vector(v)
    full = BucketFractionalEncoder(edges).encode_vector(v)
    assert list(single) == list(full[0::2])


# --- #2 TopKMaskedEncoder ----------------------------------------------------


def test_topk_masked_keeps_only_top_k_coords():
    edges = _edges_neg_pos()
    enc = TopKMaskedEncoder(edges, top_frac=0.5)
    # 4-element vector with abs ordering: 2.0 > 1.5 > 1.0 > 0.5.
    # top_frac=0.5 -> k=2 -> keep the two largest by |v|.
    v = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    out = enc.encode_vector(v)
    assert len(out) == 8
    pairs = list(out)
    # Coords 1 and 2 are kept; coords 0 and 3 are zeroed.
    assert pairs[0] == 0 and pairs[1] == 0       # |0.5| not in top-2
    assert pairs[2] != 0 or pairs[3] != 0        # |1.5| in top-2 -> not (0,0)
    assert pairs[4] != 0 or pairs[5] != 0        # |-2.0| in top-2
    assert pairs[6] == 0 and pairs[7] == 0       # |1.0| not in top-2


def test_topk_masked_top_frac_one_equals_bucket_fractional():
    edges = _edges_neg_pos()
    v = np.array([0.1, -2.4, 2.9, 0.0, -0.001], dtype=np.float32)
    masked = TopKMaskedEncoder(edges, top_frac=1.0).encode_vector(v)
    full = BucketFractionalEncoder(edges).encode_vector(v)
    assert masked == full


def test_topk_masked_rejects_invalid_top_frac():
    edges = _edges_neg_pos()
    with pytest.raises(ValueError, match="top_frac"):
        TopKMaskedEncoder(edges, top_frac=0.0)
    with pytest.raises(ValueError, match="top_frac"):
        TopKMaskedEncoder(edges, top_frac=1.5)


# --- #6 CanonicalOrderEncoder + variance_permutation_per_layer --------------


def test_canonical_order_with_identity_perm_equals_bucket_fractional():
    edges = _edges_neg_pos()
    v = torch.randn(64, generator=torch.Generator().manual_seed(0)).float().numpy()
    identity = np.arange(64)
    canon = CanonicalOrderEncoder(edges, identity).encode_vector(v)
    full = BucketFractionalEncoder(edges).encode_vector(v)
    assert canon == full


def test_canonical_order_with_reverse_perm_reverses_pairs():
    edges = _edges_neg_pos()
    v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    rev = np.arange(4)[::-1].copy()  # [3, 2, 1, 0]
    rev_out = list(CanonicalOrderEncoder(edges, rev).encode_vector(v))
    fwd_out = list(BucketFractionalEncoder(edges).encode_vector(v))
    # Bytes come in (b1, b2) pairs; reversing the perm reverses the pair order.
    expected = []
    for i in range(3, -1, -1):
        expected += fwd_out[2 * i : 2 * i + 2]
    assert rev_out == expected


def test_canonical_order_rejects_non_permutation():
    edges = _edges_neg_pos()
    with pytest.raises(ValueError, match="permutation"):
        CanonicalOrderEncoder(edges, np.array([0, 0, 1, 2]))  # not a perm
    with pytest.raises(ValueError, match="1-D"):
        CanonicalOrderEncoder(edges, np.array([[0, 1], [2, 3]]))


def test_variance_permutation_orders_high_variance_first():
    """Build a calibration corpus where coord 7 has the highest variance and
    coord 0 has the lowest. The returned permutation should start with 7."""
    g = torch.Generator().manual_seed(0)
    n_vectors, n_layers, hidden = 8, 2, 16
    base = torch.randn(n_vectors, n_layers, hidden, generator=g)
    # Inject big variance into coord 7 of layer 0, and zero out coord 0.
    base[:, 0, 7] = torch.linspace(-10, 10, n_vectors)
    base[:, 0, 0] = 0.0
    tensors = [base[i] for i in range(n_vectors)]

    perms = variance_permutation_per_layer(tensors)
    assert len(perms) == n_layers
    assert perms[0][0] == 7  # highest variance coord first
    # Coord 0 (variance 0) should be at the end.
    assert perms[0][-1] == 0


def test_canonical_order_end_to_end_with_calibrated_permutation():
    """Sanity smoke: derive a permutation from corpus, encode a corpus member."""
    g = torch.Generator().manual_seed(1)
    tensors = [torch.randn(2, 1024, generator=g).float().numpy() for _ in range(4)]
    perms = variance_permutation_per_layer(tensors)
    edges = quantile_edges(np.concatenate([t[0] for t in tensors]))
    enc = CanonicalOrderEncoder(edges, perms[0])
    out = enc.encode_vector(tensors[0][0])
    assert len(out) == 1024 * 2


# --- #7 BucketOrderTwoByteEncoder -------------------------------------------


def _edges_for_bucket_27_at_0p7_to_0p8():
    """Edges that put bucket 27 exactly on [0.7, 0.8) with 0.1 width."""
    return np.arange(257, dtype=np.float64) * 0.1 - 2.0


def test_bucket_order_two_byte_middle_emits_self():
    edges = _edges_for_bucket_27_at_0p7_to_0p8()
    # 0.75 is at frac=0.5, which is firmly in the middle 67%.
    enc = BucketOrderTwoByteEncoder(edges)
    out = list(enc.encode_vector(np.array([0.75], dtype=np.float32)))
    assert out == [27, 27]


def test_bucket_order_two_byte_lower_edge_leans_left():
    edges = _edges_for_bucket_27_at_0p7_to_0p8()
    # frac < 0.165 -> lean left.
    enc = BucketOrderTwoByteEncoder(edges)
    # 0.71 in bucket 27: frac = 0.10  -> below 0.165 -> b2 = 26.
    out = list(enc.encode_vector(np.array([0.71], dtype=np.float32)))
    assert out[0] == 27
    assert out[1] == 26


def test_bucket_order_two_byte_upper_edge_leans_right():
    edges = _edges_for_bucket_27_at_0p7_to_0p8()
    enc = BucketOrderTwoByteEncoder(edges)
    # 0.79 in bucket 27: frac ~ 0.9 -> above 0.835 -> b2 = 28.
    out = list(enc.encode_vector(np.array([0.79], dtype=np.float32)))
    assert out[0] == 27
    assert out[1] == 28


def test_bucket_order_two_byte_clamps_at_extremes():
    edges = linear_edges(-1.0, 1.0)
    enc = BucketOrderTwoByteEncoder(edges)
    # Way above range -> bucket 255, upper-edge lean -> b2 = 255 (clamped).
    out_hi = list(enc.encode_vector(np.array([10.0], dtype=np.float32)))
    assert (out_hi[0], out_hi[1]) == (N_BUCKETS - 1, N_BUCKETS - 1)
    # Way below range -> bucket 0, lower-edge lean -> b2 = 0 (clamped).
    out_lo = list(enc.encode_vector(np.array([-10.0], dtype=np.float32)))
    assert (out_lo[0], out_lo[1]) == (0, 0)


def test_bucket_order_two_byte_middle_band_width():
    """The 67% middle band means values with frac in roughly [0.165, 0.835]
    emit b2 = b1. Sweep frac across [0, 1] and check the boundary positions."""
    edges = linear_edges(0.0, 256.0)  # 1.0 wide per bucket -> bucket idx == int(v)
    enc = BucketOrderTwoByteEncoder(edges)

    # Bucket 100 covers [100, 101). Lean-left up to 100.165; middle 100.165..100.835;
    # lean-right from 100.835.
    samples = [100.0, 100.05, 100.15, 100.20, 100.50, 100.80, 100.84, 100.90, 100.99]
    expected_b2 = []
    for s in samples:
        frac = s - 100.0
        if frac < 0.165:
            expected_b2.append(99)
        elif frac >= 0.835:
            expected_b2.append(101)
        else:
            expected_b2.append(100)
    out = list(enc.encode_vector(np.array(samples, dtype=np.float32)))
    actual_b2 = out[1::2]
    assert actual_b2 == expected_b2


# --- Composability: all encoders can layer-encode --------------------------


@pytest.mark.parametrize("enc_factory", [
    lambda e: BucketSingleByteEncoder(e),
    lambda e: TopKMaskedEncoder(e, top_frac=0.1),
    lambda e: BucketOrderTwoByteEncoder(e),
    lambda e: CanonicalOrderEncoder(e, np.arange(1024)),
])
def test_all_encoders_handle_2d_tensors_via_encode_layers(enc_factory):
    """encode_layers is inherited from ByteEncoder; every subclass should work
    with it out of the box."""
    edges = linear_edges(-3.0, 3.0)
    enc = enc_factory(edges)
    t = torch.randn(3, 1024, generator=torch.Generator().manual_seed(99)).float()
    layers = enc.encode_layers(t)
    assert len(layers) == 3
    assert all(isinstance(b, bytes) for b in layers)
    # Each layer's stream should be the same length (function of encoder).
    assert len(set(len(b) for b in layers)) == 1
