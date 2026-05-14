"""Benchmark seven LSH/encoding configurations on saved persona vectors.

Loads train/test persona-diff tensors that a prior `demo/paper_eval.py` run
saved to disk (no model re-run required) and applies, in turn:

  * cosine projection (the paper-standard baseline)
  * TLSH x {BucketFractional, BucketSingleByte, TopKMasked,
            CanonicalOrder, BucketOrderTwoByte}
  * Random-Projection LSH (sign-bit / SimHash)

For each method, the classifier picks the trait whose mean persona vector
is closest to the test diff at the analysis layer (closest = max cosine for
the cosine baseline, min distance for every LSH backend). The harness
reports:

  - top-1 accuracy across all test cases
  - top-1 accuracy across only test cases where the cosine elicitation
    score (test_diff vs. its true persona vector) exceeds a threshold
  - per-backend confusion matrix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from encoding import (  # noqa: E402
    BucketFractionalEncoder,
    BucketOrderFourByteEncoder,
    BucketOrderTwoByteEncoder,
    BucketSingleByteEncoder,
    CanonicalOrderEncoder,
    TopKMaskedEncoder,
    calibrate_edges_per_layer,
    variance_permutation_per_layer,
)
from lsh import LSHBackend, RandomProjectionBackend, TLSHBackend  # noqa: E402

TRAITS = ("evil", "hallucinating", "sycophantic")


# --- Data loading ------------------------------------------------------------


def load_split(out_dir: Path, split: str) -> dict[str, torch.Tensor]:
    return {
        t: torch.load(out_dir / f"{split}_{t}_diffs.pt", map_location="cpu")
        for t in TRAITS
    }


def build_calibration_pool(
    persona_means: dict[str, torch.Tensor],
    test_diffs: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """All `[num_layers, hidden]` tensors that participate in the comparison."""
    pool = [persona_means[t] for t in TRAITS]
    for t in TRAITS:
        for i in range(test_diffs[t].shape[0]):
            pool.append(test_diffs[t][i])
    return pool


# --- Classifiers -------------------------------------------------------------


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 1e-12 else 0.0


def classify_with_cosine(
    persona_means: dict[str, torch.Tensor],
    test_diffs: dict[str, torch.Tensor],
    layer: int,
):
    """Paper-style: argmax cosine to mean persona vector at the analysis layer."""
    results = []
    for trait in TRAITS:
        for i in range(test_diffs[trait].shape[0]):
            td = test_diffs[trait][i]
            scores = {t: cosine_score(td[layer], persona_means[t][layer]) for t in TRAITS}
            predicted = max(scores, key=scores.get)
            elicitation = cosine_score(td[layer], persona_means[trait][layer])
            results.append(
                {
                    "true": trait,
                    "qid": i,
                    "predicted": predicted,
                    "elicitation": elicitation,
                    "scores": scores,
                }
            )
    return results


def classify_with_lsh(
    persona_means: dict[str, torch.Tensor],
    test_diffs: dict[str, torch.Tensor],
    layer: int,
    backend: LSHBackend,
):
    """argmin LSH-distance to each trait's mean persona digest at the analysis layer."""
    persona_digests = {t: backend.hash_vector(persona_means[t][layer]) for t in TRAITS}
    results = []
    for trait in TRAITS:
        for i in range(test_diffs[trait].shape[0]):
            td = test_diffs[trait][i]
            test_digest = backend.hash_vector(td[layer])
            distances = {
                t: backend.distance(test_digest, persona_digests[t]) for t in TRAITS
            }
            predicted = min(distances, key=distances.get)
            elicitation = cosine_score(td[layer], persona_means[trait][layer])
            results.append(
                {
                    "true": trait,
                    "qid": i,
                    "predicted": predicted,
                    "elicitation": elicitation,
                    "distances": distances,
                }
            )
    return results


# --- Backend factories -------------------------------------------------------


def make_tlsh_backend(
    encoder_kind: str,
    layer: int,
    persona_means: dict[str, torch.Tensor],
    test_diffs: dict[str, torch.Tensor],
    *,
    topk_frac: float = 0.10,
) -> TLSHBackend:
    pool = build_calibration_pool(persona_means, test_diffs)
    edges = calibrate_edges_per_layer(pool, "quantile")[layer]

    if encoder_kind == "bucket_fractional":
        encoder = BucketFractionalEncoder(edges)
    elif encoder_kind == "bucket_single_byte":
        encoder = BucketSingleByteEncoder(edges)
    elif encoder_kind == "topk_masked":
        encoder = TopKMaskedEncoder(edges, top_frac=topk_frac)
    elif encoder_kind == "canonical_order":
        permutation = variance_permutation_per_layer(pool)[layer]
        encoder = CanonicalOrderEncoder(edges, permutation)
    elif encoder_kind == "bucket_order_two_byte":
        encoder = BucketOrderTwoByteEncoder(edges)
    elif encoder_kind == "bucket_order_four_byte":
        encoder = BucketOrderFourByteEncoder(edges)
    else:
        raise ValueError(f"unknown encoder kind: {encoder_kind}")
    return TLSHBackend(encoder)


# --- Reporting ---------------------------------------------------------------


def accuracy_stats(results, elicit_threshold: float):
    total = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["true"])
    elicited = [r for r in results if r["elicitation"] >= elicit_threshold]
    elic_correct = sum(1 for r in elicited if r["predicted"] == r["true"])
    return {
        "n_total": total,
        "n_correct": correct,
        "accuracy_all": correct / total if total else 0.0,
        "n_elicited": len(elicited),
        "n_elicited_correct": elic_correct,
        "accuracy_elicited": (elic_correct / len(elicited)) if elicited else None,
    }


def confusion_matrix(results):
    cm = {t: {p: 0 for p in TRAITS} for t in TRAITS}
    for r in results:
        cm[r["true"]][r["predicted"]] += 1
    return cm


# --- Main --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "demo" / "output" / "paper" / "Qwen__Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--layer", type=int, default=None,
        help="analysis layer (default ~71%% depth, matching paper_eval)")
    parser.add_argument("--elicit_threshold", type=float, default=0.30)
    parser.add_argument("--rp_bits", type=int, default=256, help="RP-LSH digest length")
    parser.add_argument("--rp_seed", type=int, default=42)
    parser.add_argument("--topk_frac", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        parser.error(f"output dir not found: {out_dir}")

    train_diffs = load_split(out_dir, "train")
    test_diffs = load_split(out_dir, "test")

    n_layers_plus_one = train_diffs[TRAITS[0]].shape[1]
    n_hidden_layers = n_layers_plus_one - 1
    layer = args.layer if args.layer is not None else max(1, round(n_hidden_layers * 20 / 28))

    print(f"[bench] data: {out_dir}")
    print(f"[bench]  train shapes: " + ", ".join(f"{t}={tuple(train_diffs[t].shape)}" for t in TRAITS))
    print(f"[bench]  test  shapes: " + ", ".join(f"{t}={tuple(test_diffs[t].shape)}" for t in TRAITS))
    print(f"[bench] analysis layer: {layer} / {n_hidden_layers}")
    print(f"[bench] elicit threshold (cos): {args.elicit_threshold}")
    print(f"[bench] RP-LSH: {args.rp_bits} bits, seed {args.rp_seed}")
    print(f"[bench] TopKMasked: top_frac={args.topk_frac}")
    print()

    persona_means = {t: train_diffs[t].mean(dim=0) for t in TRAITS}

    # Backend definitions, in display order.
    backend_specs = [
        ("cosine_projection", "baseline"),
        ("tlsh_bucket_fractional", "bucket_fractional"),
        ("tlsh_bucket_single_byte", "bucket_single_byte"),
        ("tlsh_topk_masked", "topk_masked"),
        ("tlsh_canonical_order", "canonical_order"),
        ("tlsh_bucket_order_two_byte", "bucket_order_two_byte"),
        ("tlsh_bucket_order_four_byte", "bucket_order_four_byte"),
        ("rp_lsh", "rp"),
    ]

    all_results = {}
    for name, kind in backend_specs:
        if kind == "baseline":
            results = classify_with_cosine(persona_means, test_diffs, layer)
        elif kind == "rp":
            dim = persona_means[TRAITS[0]][layer].shape[0]
            backend = RandomProjectionBackend(dim=dim, n_bits=args.rp_bits, seed=args.rp_seed)
            results = classify_with_lsh(persona_means, test_diffs, layer, backend)
        else:
            backend = make_tlsh_backend(
                kind, layer, persona_means, test_diffs, topk_frac=args.topk_frac
            )
            results = classify_with_lsh(persona_means, test_diffs, layer, backend)
        all_results[name] = results

    # Headline table.
    chance = 1.0 / len(TRAITS)
    print(f"=== TOP-1 ACCURACY (chance = {chance:.1%}) ===")
    print(f"  {'backend':<32} {'all':>14} {'elicited':>14} {'dropped':>8}")
    print(f"  {'-' * 70}")
    for name, _ in backend_specs:
        stats = accuracy_stats(all_results[name], args.elicit_threshold)
        all_str = f"{stats['n_correct']}/{stats['n_total']} ({stats['accuracy_all']:.0%})"
        if stats["accuracy_elicited"] is not None:
            elic_str = (
                f"{stats['n_elicited_correct']}/{stats['n_elicited']} "
                f"({stats['accuracy_elicited']:.0%})"
            )
        else:
            elic_str = "all dropped"
        dropped = stats["n_total"] - stats["n_elicited"]
        print(f"  {name:<32} {all_str:>14} {elic_str:>14} {dropped:>8}")
    print()

    # Per-backend confusion matrices.
    print("=== CONFUSION MATRICES (rows = true trait, cols = predicted) ===")
    for name, _ in backend_specs:
        cm = confusion_matrix(all_results[name])
        print(f"\n  {name}")
        print(f"    {'true \\ pred':<14} {'  '.join(f'{p[:6]:>6}' for p in TRAITS)}")
        for trait in TRAITS:
            row = "  ".join(f"{cm[trait][p]:>6d}" for p in TRAITS)
            print(f"    {trait:<14} {row}")
    print()

    # Per-TLSH-backend full distance tables: test_digest -> persona_<trait>.
    print("=== TLSH DISTANCES per backend (test -> persona_<trait>; min = predicted) ===")
    for name, _ in backend_specs:
        if not name.startswith("tlsh_"):
            continue
        results = all_results[name]
        print(f"\n  {name}")
        header = f"    {'test_case':<18} | " + " ".join(f"{t[:6]:>7}" for t in TRAITS) + " | predicted"
        print(header)
        print(f"    {'-' * (len(header) - 4)}")
        for r in results:
            case = f"{r['true']}_q{r['qid']}"
            dist_strs = " ".join(f"{int(r['distances'][t]):>7d}" for t in TRAITS)
            marker = "OK" if r["predicted"] == r["true"] else f"MISS->{r['predicted']}"
            print(f"    {case:<18} | {dist_strs} | {marker}")
    print()

    # Per-test details, especially elicitation scores.
    print("=== PER-TEST ELICITATION SCORES (cos vs. true persona at this layer) ===")
    cos_results = all_results["cosine_projection"]
    print(f"  {'case':<22} {'cosine_elic':>12} {'flag'}")
    for r in cos_results:
        flag = "NON-ELIC" if r["elicitation"] < args.elicit_threshold else "elicited"
        case = f"{r['true']}_q{r['qid']}"
        print(f"  {case:<22} {r['elicitation']:>+12.3f}  {flag}")
    print()

    # Save full details.
    serializable = {}
    for name, results in all_results.items():
        entries = []
        for r in results:
            e = {
                "true": r["true"],
                "qid": r["qid"],
                "predicted": r["predicted"],
                "elicitation": r["elicitation"],
            }
            if "scores" in r:
                e["scores"] = r["scores"]
            if "distances" in r:
                e["distances"] = r["distances"]
            entries.append(e)
        serializable[name] = entries
    out_path = out_dir / "bench_results.json"
    out_path.write_text(json.dumps(serializable, indent=2, default=float))
    print(f"[bench] saved per-test details to {out_path}")


if __name__ == "__main__":
    main()
