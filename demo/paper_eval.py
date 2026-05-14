"""Paper-style two-stage TLSH evaluation for persona vectors.

Stage 1 (train):
    For each trait T and each question q in a TRAINING split, generate
    responses under the pos and neg system prompts and extract response-
    averaged hidden states. The mean of (act_pos - act_neg) across training
    questions is the persona vector for T.

Stage 2 (test):
    For each trait T and each question q in a DISJOINT TEST split, compute
    the per-question diff. TLSH-classify by finding the persona-vector
    digest closest in TLSH distance. Also compute
    cos(test_diff[layer], persona_T[layer]) as an "elicitation score" --
    test cases below `--elicit_threshold` are flagged as non-elicited
    (e.g. when the model's RLHF defeats an evil prompt).

Output:
    Per-test-case table with true/predicted/elicitation/response.
    Top-1 accuracy overall and per-trait, with and without non-elicited
    cases.

Run:
    python demo/paper_eval.py --model Qwen/Qwen2.5-7B-Instruct \
        --n_train 5 --n_test 2 --elicit_threshold 0.30
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "demo"))

from encoding import calibrate_edges_per_layer  # noqa: E402
from generate_vec import save_lsh_sidecar  # noqa: E402
from lsh import diff as tlsh_diff  # noqa: E402
from run_demo import (  # noqa: E402
    auto_device,
    chat_prompt,
    extract_response_avg,
    generate_response,
    pick_layer,
    safe_dir_name,
    truncate,
)

TRAITS = ("evil", "hallucinating", "sycophantic")
MAX_QUESTIONS_AVAILABLE = 20  # each trait JSON ships with 20 questions


def load_trait_data(trait: str, n: int) -> tuple[dict, list[str]]:
    path = REPO_ROOT / "data_generation" / "trait_data_extract" / f"{trait}.json"
    payload = json.loads(path.read_text())
    if n > len(payload["questions"]):
        raise ValueError(
            f"asked for {n} questions but {trait}.json has {len(payload['questions'])}"
        )
    return payload["instruction"][0], payload["questions"][:n]


def run_questions(model, tokenizer, instruction, questions, max_new_tokens, label):
    responses: list[dict] = []
    diffs: list[torch.Tensor] = []
    for i, q in enumerate(questions):
        print(f"  [{label} q{i}] {truncate(q, 80)}")
        prompt_pos = chat_prompt(tokenizer, instruction["pos"], q)
        prompt_neg = chat_prompt(tokenizer, instruction["neg"], q)
        resp_pos = generate_response(model, tokenizer, prompt_pos, max_new_tokens)
        resp_neg = generate_response(model, tokenizer, prompt_neg, max_new_tokens)
        act_pos = extract_response_avg(model, tokenizer, prompt_pos, resp_pos)
        act_neg = extract_response_avg(model, tokenizer, prompt_neg, resp_neg)
        responses.append({"q": q, "with": resp_pos, "without": resp_neg})
        diffs.append(act_pos - act_neg)
    return responses, torch.stack(diffs, dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return (a @ b / denom).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or omit to auto-detect")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--n_train", type=int, default=5,
        help="questions per trait used to build the mean persona vector")
    parser.add_argument("--n_test", type=int, default=2,
        help="questions per trait used to evaluate classification (disjoint from train)")
    parser.add_argument("--layer", type=int, default=None,
        help="hidden-state layer for analysis (default ~71%% depth)")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--elicit_threshold", type=float, default=0.30,
        help="cos(test_diff[layer], persona_true[layer]) below this is flagged non-elicited")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "demo" / "output" / "paper"))
    parser.add_argument("--edges", choices=("linear", "quantile", "symlog"), default="quantile")
    args = parser.parse_args()

    if args.n_train + args.n_test > MAX_QUESTIONS_AVAILABLE:
        parser.error(f"n_train + n_test must be <= {MAX_QUESTIONS_AVAILABLE}")
    if args.n_train < 1 or args.n_test < 1:
        parser.error("n_train and n_test must both be >= 1")

    device = args.device or auto_device()
    if args.dtype != "auto":
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    else:
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
    print(f"[paper] loading {args.model} on {device} as {dtype}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    )
    model.eval()

    layer = pick_layer(model.config.num_hidden_layers, args.layer)
    print(f"[paper] analysis layer: {layer} / {model.config.num_hidden_layers}")
    print(f"[paper] split per trait: {args.n_train} train / {args.n_test} test")

    out_dir = Path(args.output_dir) / safe_dir_name(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: generate + extract for train + test ----
    train_diffs: dict[str, torch.Tensor] = {}
    test_diffs: dict[str, torch.Tensor] = {}
    train_records: dict[str, list[dict]] = {}
    test_records: dict[str, list[dict]] = {}

    for trait in TRAITS:
        instruction, all_qs = load_trait_data(trait, args.n_train + args.n_test)
        train_qs = all_qs[: args.n_train]
        test_qs = all_qs[args.n_train : args.n_train + args.n_test]

        print(f"\n=== TRAIN {trait.upper()} ({len(train_qs)} questions) ===")
        recs, diffs = run_questions(model, tokenizer, instruction, train_qs, args.max_new_tokens, f"{trait[:3]}-tr")
        train_records[trait] = recs
        train_diffs[trait] = diffs
        # Persist immediately so a partial run isn't lost.
        torch.save(diffs, out_dir / f"train_{trait}_diffs.pt")

        print(f"\n=== TEST {trait.upper()} ({len(test_qs)} questions) ===")
        recs, diffs = run_questions(model, tokenizer, instruction, test_qs, args.max_new_tokens, f"{trait[:3]}-te")
        test_records[trait] = recs
        test_diffs[trait] = diffs
        torch.save(diffs, out_dir / f"test_{trait}_diffs.pt")

    # Persist responses too for later inspection.
    (out_dir / "train_responses.json").write_text(json.dumps(train_records, indent=2))
    (out_dir / "test_responses.json").write_text(json.dumps(test_records, indent=2))

    # Free the model; analysis from here is fast.
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Phase 2: build mean persona vectors, calibrate edges, hash ----
    mean_persona = {trait: train_diffs[trait].mean(dim=0) for trait in TRAITS}

    pool = [mean_persona[t] for t in TRAITS] + [
        test_diffs[t][i] for t in TRAITS for i in range(args.n_test)
    ]
    shared_edges = calibrate_edges_per_layer(pool, args.edges)
    print(f"\n[paper] calibrated per-layer edges from {len(pool)} vectors "
          f"({len(TRAITS)} mean personas + {sum(args.n_test for _ in TRAITS)} test diffs), "
          f"method={args.edges}")

    persona_digests: dict[str, str] = {}
    for trait in TRAITS:
        pt = out_dir / f"persona_mean_{trait}.pt"
        torch.save(mean_persona[trait], pt)
        sidecar = json.loads(Path(save_lsh_sidecar(str(pt), edges_per_layer=shared_edges)).read_text())
        persona_digests[trait] = sidecar["digests"][layer]
        print(f"  [{trait:<14}] mean-persona digest at layer {layer}: {persona_digests[trait]}")

    # ---- Phase 3: hash test diffs, classify, score elicitation ----
    results = []
    for trait in TRAITS:
        for i in range(args.n_test):
            test_diff = test_diffs[trait][i]
            pt = out_dir / f"test_{trait}_q{i}.pt"
            torch.save(test_diff, pt)
            sidecar = json.loads(Path(save_lsh_sidecar(str(pt), edges_per_layer=shared_edges)).read_text())
            test_digest = sidecar["digests"][layer]

            distances = {t: tlsh_diff(test_digest, persona_digests[t]) for t in TRAITS}
            predicted = min(distances, key=distances.get)
            elicit = cosine(test_diff[layer], mean_persona[trait][layer])

            rec = test_records[trait][i]
            results.append({
                "true": trait,
                "qid": i,
                "question": rec["q"],
                "response_with": rec["with"],
                "response_without": rec["without"],
                "predicted": predicted,
                "distances": distances,
                "elicitation": elicit,
                "test_digest": test_digest,
            })

    # Save full results as JSON for later post-processing.
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # ---- Phase 4: report ----
    print("\n\n=== TEST RESULTS ===")
    print(f"{'true':<13} {'pred':<13} {'cos':>7}  flag       ok  question")
    for r in results:
        flag = "NON-ELIC" if r["elicitation"] < args.elicit_threshold else "elicited"
        ok = "OK" if r["predicted"] == r["true"] else "MISS"
        print(f"  {r['true']:<13} {r['predicted']:<13} {r['elicitation']:>+7.3f}  "
              f"{flag:>8}   {ok:<4} {truncate(r['question'], 50)}")

    print("\n=== RESPONSES (so you can spot non-elicited cases by eye) ===")
    for r in results:
        flag = " [NON-ELIC]" if r["elicitation"] < args.elicit_threshold else ""
        print(f"\n[{r['true']} q{r['qid']}, cos={r['elicitation']:+.3f}]{flag}")
        print(f"  Q       : {truncate(r['question'], 100)}")
        print(f"  with    : {truncate(r['response_with'], 200)}")
        print(f"  without : {truncate(r['response_without'], 200)}")

    print("\n=== TLSH DISTANCES: test_digest -> persona_digest_{trait} ===")
    print(f"  {'test':<18} " + " ".join(f"{t[:6]:>8}" for t in TRAITS) + "   predicted   ok")
    for r in results:
        row = " ".join(f"{r['distances'][t]:>8d}" for t in TRAITS)
        ok = "OK" if r["predicted"] == r["true"] else "MISS"
        print(f"  {r['true'][:3]}-q{r['qid']:<13} {row}   {r['predicted']:<12} {ok}")

    # Accuracy
    total = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["true"])
    elicited = [r for r in results if r["elicitation"] >= args.elicit_threshold]
    elicited_correct = sum(1 for r in elicited if r["predicted"] == r["true"])
    chance = 1.0 / len(TRAITS)

    print("\n=== ACCURACY ===")
    print(f"  Chance baseline:     {chance:.1%}")
    print(f"  All test cases:      {correct}/{total} = {correct/total:.1%}")
    if elicited:
        print(f"  Elicited-only:       {elicited_correct}/{len(elicited)} = {elicited_correct/len(elicited):.1%}")
        print(f"  (dropped {total - len(elicited)} non-elicited cases, threshold cos >= {args.elicit_threshold})")
    else:
        print(f"  Every test case was below the elicitation threshold {args.elicit_threshold}!")

    print("\n=== PER-TRAIT ===")
    for trait in TRAITS:
        in_trait = [r for r in results if r["true"] == trait]
        in_trait_correct = sum(1 for r in in_trait if r["predicted"] == trait)
        in_trait_elic = [r for r in in_trait if r["elicitation"] >= args.elicit_threshold]
        in_trait_elic_correct = sum(1 for r in in_trait_elic if r["predicted"] == trait)
        dropped = len(in_trait) - len(in_trait_elic)
        elic_part = (
            f"{in_trait_elic_correct}/{len(in_trait_elic)}"
            if in_trait_elic else "all dropped"
        )
        print(f"  {trait:<14} all: {in_trait_correct}/{len(in_trait)}   "
              f"elicited: {elic_part}   (dropped {dropped})")


if __name__ == "__main__":
    main()
