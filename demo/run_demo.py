"""Live demo: Random-Projection LSH distinguishes persona vectors.

For each model in `--models` (comma-separated), for each trait in
{evil, hallucinating, sycophantic}, and for each of 3 evaluation questions,
this script:

  1. Generates a response under the "with-trait" system prompt.
  2. Generates a response under the "without-trait" system prompt.
  3. Extracts the response-averaged hidden state at every layer for each
     condition.
  4. Computes the per-question persona-direction vector (with - without).
  5. Hashes each diff at a chosen mid-stack layer via sign-bit Random-Projection
     LSH and prints per-question responses, a 9x9 distance matrix, and a
     within-vs-between summary. Repeats per model.
  6. Cross-model: prints both models' within/between ratios side by side
     so you can see whether the technique generalizes.

Run:
    python demo/run_demo.py                       # auto-detect cuda/cpu
    python demo/run_demo.py --device cpu
    python demo/run_demo.py --models Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lsh import RandomProjectionBackend  # noqa: E402


TRAITS = ("evil", "hallucinating", "sycophantic")
N_QUESTIONS = 3
DEFAULT_MODELS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)


@dataclass
class Example:
    trait: str
    qid: int
    question: str
    response_pos: str
    response_neg: str
    digest: str  # RP-LSH digest of the diff vector at the analysis layer


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_trait_prompts(trait: str) -> tuple[dict, list[str]]:
    path = REPO_ROOT / "data_generation" / "trait_data_extract" / f"{trait}.json"
    payload = json.loads(path.read_text())
    return payload["instruction"][0], payload["questions"][:N_QUESTIONS]


def pick_layer(num_hidden_layers: int, user_choice: int | None) -> int:
    if user_choice is not None:
        if not (0 <= user_choice <= num_hidden_layers):
            raise ValueError(f"--layer {user_choice} out of range [0, {num_hidden_layers}]")
        return user_choice
    # Paper used layer 20 of a 28-layer 7B model (~71%% depth); mirror that fraction.
    return max(1, round(num_hidden_layers * 20 / 28))


def chat_prompt(tokenizer, system_prompt: str, user_question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_response_avg(model, tokenizer, prompt: str, response: str) -> torch.Tensor:
    """Hidden-state average over the response tokens, all layers.

    Returns `[num_hidden_layers+1, hidden_dim]`, matching `generate_vec.py`'s
    `response_avg` slice format.
    """
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layer_count = model.config.num_hidden_layers + 1
    per_layer = []
    for layer in range(layer_count):
        h = out.hidden_states[layer][:, prompt_len:, :]
        if h.shape[1] == 0:
            h = out.hidden_states[layer][:, -1:, :]
        per_layer.append(h.mean(dim=1).float().squeeze(0).detach().cpu())
    return torch.stack(per_layer, dim=0)


def summarize_matrix(keys: list[tuple[str, int]], matrix: np.ndarray) -> dict:
    """Mean within-trait and between-trait pairwise distances + their ratio."""
    n = len(keys)
    within_sum, within_n = 0.0, 0
    between_sum, between_n = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = matrix[i, j]
            if np.isnan(d):
                continue
            if keys[i][0] == keys[j][0]:
                within_sum += d
                within_n += 1
            else:
                between_sum += d
                between_n += 1
    within = within_sum / max(within_n, 1)
    between = between_sum / max(between_n, 1)
    return {
        "within_trait_mean": within,
        "between_trait_mean": between,
        "ratio": between / within if within > 0 else float("inf"),
    }


def truncate(s: str, n: int = 100) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def safe_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def run_one_model(model_name: str, args, base_out_dir: Path) -> dict:
    print(f"\n#### MODEL: {model_name} ####\n")
    out_dir = base_out_dir / safe_dir_name(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or auto_device()
    print(f"[demo] device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.dtype != "auto":
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    else:
        # CUDA -> fp16; CPU -> bf16 (fp32 7B is ~28GB and OOMs most laptops).
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
    print(f"[demo] dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model.eval()

    layer = pick_layer(model.config.num_hidden_layers, args.layer)
    print(f"[demo] analysis layer: {layer} / {model.config.num_hidden_layers}\n")

    # First pass: collect per-question diff tensors + responses.
    diffs: dict[tuple[str, int], torch.Tensor] = {}
    responses: dict[tuple[str, int], tuple[str, str]] = {}
    questions: dict[tuple[str, int], str] = {}

    for trait in TRAITS:
        instruction, qs = load_trait_prompts(trait)
        print(f"=== {trait.upper()} ===")
        for qid, question in enumerate(qs):
            print(f"  [{trait} q{qid}] {truncate(question, 90)}")

            prompt_pos = chat_prompt(tokenizer, instruction["pos"], question)
            prompt_neg = chat_prompt(tokenizer, instruction["neg"], question)

            response_pos = generate_response(model, tokenizer, prompt_pos, args.max_new_tokens)
            response_neg = generate_response(model, tokenizer, prompt_neg, args.max_new_tokens)

            act_pos = extract_response_avg(model, tokenizer, prompt_pos, response_pos)
            act_neg = extract_response_avg(model, tokenizer, prompt_neg, response_neg)
            diff_vec = act_pos - act_neg

            diffs[(trait, qid)] = diff_vec
            responses[(trait, qid)] = (response_pos, response_neg)
            questions[(trait, qid)] = question

            print(f"    with    : {truncate(response_pos)}")
            print(f"    without : {truncate(response_neg)}")
        print()

    # Save .pt files for downstream bench/analysis.
    for (trait, qid), diff_vec in diffs.items():
        pt = out_dir / f"{trait}_q{qid}_response_avg_diff.pt"
        torch.save(diff_vec, pt)

    # Hash via Random-Projection LSH at the analysis layer.
    dim = next(iter(diffs.values()))[layer].shape[0]
    backend = RandomProjectionBackend(dim=dim, n_bits=args.rp_bits, seed=args.rp_seed)
    print(f"[demo] RP-LSH: dim={dim} -> {args.rp_bits} bits (seed {args.rp_seed})\n")

    examples: list[Example] = []
    for (trait, qid), diff_vec in diffs.items():
        digest = backend.hash_vector(diff_vec[layer])
        response_pos, response_neg = responses[(trait, qid)]
        examples.append(
            Example(trait, qid, questions[(trait, qid)], response_pos, response_neg, digest)
        )
        print(f"  [{trait} q{qid}] layer-{layer} digest: {digest[:16]}...")
    print()

    # Pairwise RP-LSH distance matrix at the analysis layer.
    digests = [e.digest for e in examples]
    keys = [(e.trait, e.qid) for e in examples]
    n = len(digests)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = backend.distance(digests[i], digests[j])
            matrix[i, j] = d
            matrix[j, i] = d

    print(f"=== RP-LSH PAIRWISE DISTANCE MATRIX (layer {layer}, normalized Hamming) ===")
    headers = [f"{t[:3]}{q}" for t, q in keys]
    print("        " + "  ".join(f"{h:>6}" for h in headers))
    for i, h in enumerate(headers):
        row = "  ".join(f"{matrix[i, j]:>6.3f}" for j in range(n))
        print(f"  {h:>4}  {row}")
    print()

    stats = summarize_matrix(keys, matrix)
    print(f"=== SUMMARY (model={model_name}, layer={layer}) ===")
    print(f"  Mean within-trait  RP-LSH distance: {stats['within_trait_mean']:.3f}")
    print(f"  Mean between-trait RP-LSH distance: {stats['between_trait_mean']:.3f}")
    print(f"  Between / within ratio            : {stats['ratio']:.2f}")
    if stats["ratio"] > 1.0:
        print("  --> Same-trait digests cluster; different-trait digests separate.")
    else:
        print("  --> No clear clustering this run.")

    # Free the model before loading the next one.
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "layer": layer,
        "within_trait_mean": stats["within_trait_mean"],
        "between_trait_mean": stats["between_trait_mean"],
        "ratio": stats["ratio"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated HF model IDs",
    )
    parser.add_argument("--model", default=None, help="alias for --models with a single model")
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or omit to auto-detect")
    parser.add_argument("--layer", type=int, default=None,
        help="hidden-state layer to compare (default ~71%% depth)")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "demo" / "output"))
    parser.add_argument("--rp_bits", type=int, default=256, help="RP-LSH digest length in bits")
    parser.add_argument("--rp_seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help="model weight dtype. 'auto' picks bf16 on CPU, fp16 on CUDA.",
    )
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        parser.error("no models given")

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for m in models:
        summaries.append(run_one_model(m, args, base_out))

    if len(summaries) > 1:
        print("\n=== CROSS-MODEL SUMMARY ===")
        print(f"  {'model':<48} {'layer':>6} {'within':>8} {'between':>9} {'ratio':>6}")
        for s in summaries:
            print(
                f"  {s['model']:<48} {s['layer']:>6} "
                f"{s['within_trait_mean']:>8.3f} {s['between_trait_mean']:>9.3f} "
                f"{s['ratio']:>6.2f}"
            )
        print()
        if all(s["ratio"] > 1.0 for s in summaries):
            print("  --> Trait clustering holds across both architectures.")


if __name__ == "__main__":
    main()
