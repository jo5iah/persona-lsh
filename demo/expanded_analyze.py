"""3-fold CV analysis across layer strategies and LSH backends.

Loads the saved diffs and the fully-eliciting question lists from
`demo/expanded_collect.py`. For each combination of `layer_strategy` x
`lsh_backend`:

  - layer strategies:
      * single   : one fixed layer (default ~71%% depth)
      * multi    : top-N (default 5) layers by coherence in the middle
                   30%-90% of network depth; selection is PER FOLD using
                   only that fold's train data
      * all      : every layer 1..num_hidden_layers (skip the embedding)
  - backends:
      * cosine_projection
      * rp_lsh (Random Projection / SimHash, sign-bit)

For each (strategy, backend) pair, runs `--cv_folds` (default 3) folds:
  - per-trait split of eliciting questions into train + test
  - mean persona vector per trait built from train diffs (concatenated
    across selected layers)
  - test diffs classified by argmax cosine / argmin Hamming
  - per-fold accuracy aggregated to a top-1 accuracy headline

Output: side-by-side accuracy table for {strategy} x {backend} plus the
selected layers per fold for transparency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lsh import RandomProjectionBackend  # noqa: E402

TRAITS = ("evil", "hallucinating", "sycophantic")


# --- Cosine / vector helpers ------------------------------------------------


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 1e-12 else 0.0


# --- CV fold splitter -------------------------------------------------------


def kfold_splits(indices: Sequence[int], k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Return `[(train, test), ...]` index lists for `k` folds.

    `len(test)` per fold is `len(indices) // k` for all but the last fold;
    the last fold absorbs any remainder. Indices are randomized by `seed`."""
    rng = np.random.default_rng(seed)
    perm = list(indices)
    rng.shuffle(perm)
    fold_size = len(perm) // k
    folds: list[tuple[list[int], list[int]]] = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else len(perm)
        test = perm[start:end]
        train = perm[:start] + perm[end:]
        folds.append((train, test))
    return folds


def make_cv_fold_splits(eliciting_per_trait: dict, k: int, seed: int) -> list[dict]:
    """Per-trait kfold splits combined into `k` global fold dicts.

    Returns `[fold_0, fold_1, ...]` where `fold_i[trait] = (train_indices, test_indices)`.
    Each trait is split independently so trait imbalance is preserved.
    """
    per_trait = {t: kfold_splits(eliciting_per_trait[t], k, seed) for t in TRAITS}
    return [{t: per_trait[t][i] for t in TRAITS} for i in range(k)]


# --- Layer selection --------------------------------------------------------


def middle_layer_range(num_hidden_layers: int) -> list[int]:
    """Candidate range for the `multi` strategy: middle 30%-90% depth.

    Skip the embedding (layer 0) and very-late layers; the persona signal
    in transformer hidden states is usually concentrated in the mid-upper
    block, not at the very top.
    """
    start = max(1, int(num_hidden_layers * 0.30))
    end = max(start + 1, int(num_hidden_layers * 0.90) + 1)
    return list(range(start, end))


def compute_layer_coherence(
    train_diffs_per_trait: dict[str, torch.Tensor], candidate_layers: list[int]
) -> dict[int, float]:
    """Mean of `cos(diff_q[L], mean_diff[L])` across all train diffs at each L.

    Higher coherence = the layer's mean diff direction is consistent across
    training questions (and across traits) -- i.e. the layer reliably
    encodes a persona-like axis. Computed across all traits' train data
    pooled together so the selected layers are trait-agnostic.
    """
    coherences: dict[int, float] = {}
    for L in candidate_layers:
        stacked = torch.cat([train_diffs_per_trait[t][:, L, :] for t in TRAITS], dim=0)
        mean = stacked.mean(dim=0)
        cosines = [cosine_score(stacked[i], mean) for i in range(stacked.shape[0])]
        coherences[L] = float(np.mean(cosines))
    return coherences


def select_top_n_layers(
    train_diffs_per_trait: dict[str, torch.Tensor], num_hidden_layers: int, n: int
) -> list[int]:
    """Top `n` middle-range layers ranked by coherence."""
    candidate = middle_layer_range(num_hidden_layers)
    coherences = compute_layer_coherence(train_diffs_per_trait, candidate)
    ranked = sorted(coherences, key=coherences.get, reverse=True)
    return sorted(ranked[:n])  # keep ascending index order for concatenation stability


def get_strategy_layers(
    strategy: str,
    num_hidden_layers: int,
    train_diffs_per_trait: dict,
    single_layer: int,
    multi_n: int,
) -> list[int]:
    if strategy == "single":
        return [single_layer]
    if strategy == "multi":
        return select_top_n_layers(train_diffs_per_trait, num_hidden_layers, multi_n)
    if strategy == "all":
        return list(range(1, num_hidden_layers + 1))
    raise ValueError(f"unknown strategy: {strategy}")


def vector_for_strategy(diff_LH: torch.Tensor, layers: list[int]) -> torch.Tensor:
    """Concatenate the diff vector across the selected layers."""
    return torch.cat([diff_LH[L] for L in layers], dim=0)


# --- Single CV fold ---------------------------------------------------------


def classify_one_fold(
    fold: dict,
    diffs_per_trait: dict[str, torch.Tensor],
    strategy: str,
    single_layer: int,
    multi_n: int,
    num_hidden_layers: int,
    rp_bits: int,
    rp_seed: int,
):
    """Run one CV fold: build personas from train, classify the fold's test.

    Returns `(results, layers_used)` where `results` is a list of per-test
    dicts with the cosine and RP-LSH predictions side-by-side.
    """
    # train_diffs_per_trait: only the train indices from this fold
    train_diffs_per_trait = {t: diffs_per_trait[t][fold[t][0]] for t in TRAITS}

    layers = get_strategy_layers(
        strategy, num_hidden_layers, train_diffs_per_trait, single_layer, multi_n
    )

    # Mean persona per trait (concatenated across selected layers).
    persona_means = {
        t: vector_for_strategy(train_diffs_per_trait[t].mean(dim=0), layers) for t in TRAITS
    }

    dim = persona_means[TRAITS[0]].shape[0]
    rp_backend = RandomProjectionBackend(dim=dim, n_bits=rp_bits, seed=rp_seed)
    persona_digests = {t: rp_backend.hash_vector(persona_means[t]) for t in TRAITS}

    results = []
    for trait in TRAITS:
        for qid in fold[trait][1]:  # test indices for this trait
            test_diff = diffs_per_trait[trait][qid]
            test_vec = vector_for_strategy(test_diff, layers)

            cos_scores = {t: cosine_score(test_vec, persona_means[t]) for t in TRAITS}
            cos_pred = max(cos_scores, key=cos_scores.get)

            test_digest = rp_backend.hash_vector(test_vec)
            rp_distances = {t: rp_backend.distance(test_digest, persona_digests[t]) for t in TRAITS}
            rp_pred = min(rp_distances, key=rp_distances.get)

            results.append({
                "true": trait,
                "qid": qid,
                "cosine_pred": cos_pred,
                "rp_pred": rp_pred,
                "cos_scores": cos_scores,
                "rp_distances": rp_distances,
            })
    return results, layers


def aggregate(fold_results_list: list[list[dict]]) -> dict:
    total = 0
    cos_correct = 0
    rp_correct = 0
    for fold in fold_results_list:
        for r in fold:
            total += 1
            cos_correct += r["cosine_pred"] == r["true"]
            rp_correct += r["rp_pred"] == r["true"]
    return {
        "n_total": total,
        "cosine_acc": cos_correct / total if total else 0.0,
        "rp_acc": rp_correct / total if total else 0.0,
    }


# --- Main --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
        help="path to demo/output/expanded/<model> directory")
    parser.add_argument("--single_layer", type=int, default=None,
        help="layer for the 'single' strategy (default ~71%% depth)")
    parser.add_argument("--multi_n", type=int, default=5,
        help="top-N layers for the 'multi' strategy (default 5)")
    parser.add_argument("--rp_bits", type=int, default=256)
    parser.add_argument("--rp_seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--cv_seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    eliciting_path = data_dir / "eliciting_questions.json"
    if not eliciting_path.exists():
        parser.error(f"missing {eliciting_path}; run expanded_collect.py first")

    diffs_per_trait = {
        t: torch.load(data_dir / f"all_{t}_diffs.pt", map_location="cpu") for t in TRAITS
    }
    eliciting = json.loads(eliciting_path.read_text())

    for t in TRAITS:
        if len(eliciting[t]) < args.cv_folds:
            parser.error(
                f"trait '{t}' has only {len(eliciting[t])} eliciting questions; "
                f"need >= cv_folds={args.cv_folds}. Lower --cv_folds or relax the "
                f"elicitation thresholds in expanded_collect.py."
            )

    num_hidden_layers = diffs_per_trait[TRAITS[0]].shape[1] - 1
    single_layer = (
        args.single_layer if args.single_layer is not None
        else max(1, round(num_hidden_layers * 20 / 28))
    )

    print(f"[analyze] data: {data_dir}")
    for t in TRAITS:
        print(f"  [{t}] diff shape: {tuple(diffs_per_trait[t].shape)}; "
              f"eliciting: {len(eliciting[t])}/{diffs_per_trait[t].shape[0]}")
    print(f"[analyze] single_layer = {single_layer}")
    print(f"[analyze] multi_n      = {args.multi_n} (top-N by coherence in middle 30%-90%)")
    print(f"[analyze] cv_folds     = {args.cv_folds} (seed {args.cv_seed})")
    print(f"[analyze] rp_lsh       = {args.rp_bits} bits, seed {args.rp_seed}")

    cv_folds = make_cv_fold_splits(eliciting, args.cv_folds, args.cv_seed)

    strategies = ["single", "multi", "all"]
    strategy_results = {}
    for strategy in strategies:
        print(f"\n=== STRATEGY: {strategy} ===")
        fold_results_list = []
        layers_per_fold = []
        for k, fold in enumerate(cv_folds):
            results, layers = classify_one_fold(
                fold, diffs_per_trait, strategy,
                single_layer, args.multi_n, num_hidden_layers,
                args.rp_bits, args.rp_seed,
            )
            fold_results_list.append(results)
            layers_per_fold.append(layers)
            ll = layers if len(layers) <= 10 else f"{layers[:5]}...{layers[-3:]} (n={len(layers)})"
            print(f"  fold {k}: layers = {ll}")

        agg = aggregate(fold_results_list)
        strategy_results[strategy] = {
            "aggregate": agg,
            "layers_per_fold": layers_per_fold,
            "folds": fold_results_list,
        }
        print(f"  cosine acc: {agg['cosine_acc']:.1%} | rp_lsh acc: {agg['rp_acc']:.1%}  "
              f"(n={agg['n_total']})")

    # Headline table.
    chance = 1.0 / len(TRAITS)
    print(f"\n=== ACCURACY ACROSS STRATEGIES ({args.cv_folds}-fold CV; chance = {chance:.1%}) ===")
    print(f"  {'strategy':<10} {'cosine':>12} {'rp_lsh':>12}")
    for strategy in strategies:
        sr = strategy_results[strategy]
        print(
            f"  {strategy:<10} {sr['aggregate']['cosine_acc']:>11.1%} "
            f"{sr['aggregate']['rp_acc']:>11.1%}"
        )

    # Per-strategy confusion matrices for the cosine baseline (for trait-level diagnosis).
    print("\n=== CONFUSION MATRICES per strategy (cosine pred; rows = true) ===")
    for strategy in strategies:
        fold_results = strategy_results[strategy]["folds"]
        cm = {t: {p: 0 for p in TRAITS} for t in TRAITS}
        for fold in fold_results:
            for r in fold:
                cm[r["true"]][r["cosine_pred"]] += 1
        print(f"\n  {strategy}")
        print(f"    {'true \\ pred':<14} {'  '.join(f'{p[:6]:>6}' for p in TRAITS)}")
        for trait in TRAITS:
            row = "  ".join(f"{cm[trait][p]:>6d}" for p in TRAITS)
            print(f"    {trait:<14} {row}")

    # Save JSON for downstream post-hoc analysis.
    out_path = data_dir / "expanded_analyze_results.json"

    def _safe(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    out_path.write_text(json.dumps(strategy_results, indent=2, default=_safe))
    print(f"\n[analyze] saved full per-fold results to {out_path}")


if __name__ == "__main__":
    main()
