import argparse
import json
import os
from typing import Optional

import pandas as pd
import torch


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    from tqdm import tqdm

    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu())
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg


def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    mask = (persona_pos[trait] >=threshold) & (persona_neg[trait] < 100-threshold) & (persona_pos["coherence"] >= 50) & (persona_neg["coherence"] >= 50)

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses


def compute_layer_diffs(layer_activations_pos, layer_activations_neg):
    """Mean-difference of activations across layers.

    Both inputs are length-`num_layers` sequences where each element is a
    `[n_examples, hidden_dim]` tensor (the format produced by
    `get_hidden_p_and_r`). Returns a `[num_layers, hidden_dim]` tensor of
    per-layer (pos.mean - neg.mean) differences.
    """
    if len(layer_activations_pos) != len(layer_activations_neg):
        raise ValueError(
            f"layer count mismatch: pos={len(layer_activations_pos)}, neg={len(layer_activations_neg)}"
        )
    return torch.stack(
        [
            layer_activations_pos[l].mean(0).float() - layer_activations_neg[l].mean(0).float()
            for l in range(len(layer_activations_pos))
        ],
        dim=0,
    )


def save_lsh_sidecar(pt_path: str, edges_method: str = "quantile") -> str:
    """Compute per-layer TLSH digests for a saved persona vector and write a JSON sidecar.

    The sidecar lives alongside the `.pt` file as `<stem>.tlsh.json` and contains the
    bucket edges that were used so digests can be reproduced.
    """
    from encoding import linear_edges, quantile_edges, symlog_edges
    from lsh import hash_layers

    tensor = torch.load(pt_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{pt_path}: expected torch.Tensor, got {type(tensor).__name__}")

    if edges_method == "linear":
        lo = float(tensor.min().item())
        hi = float(tensor.max().item())
        if hi <= lo:
            hi = lo + 1e-6
        edges = linear_edges(lo, hi)
    elif edges_method == "quantile":
        edges = quantile_edges(tensor)
    elif edges_method == "symlog":
        absmax = float(tensor.abs().max().item())
        if absmax == 0:
            absmax = 1.0
        edges = symlog_edges(-absmax, absmax)
    else:
        raise ValueError(f"unknown edges_method: {edges_method!r}")

    digests = hash_layers(tensor, edges)
    sidecar_path = pt_path[:-3] + ".tlsh.json" if pt_path.endswith(".pt") else pt_path + ".tlsh.json"
    with open(sidecar_path, "w") as f:
        json.dump(
            {
                "source": os.path.basename(pt_path),
                "edges_method": edges_method,
                "edges": edges.tolist(),
                "digests": digests,
            },
            f,
            indent=2,
        )
    return sidecar_path


def save_persona_vector(
    model_name,
    pos_path,
    neg_path,
    trait,
    save_dir,
    threshold=50,
    *,
    compute_lsh: bool = False,
    lsh_edges_method: str = "quantile",
    _activation_extractor=None,
    _model_loader=None,
):
    """Generate and save persona vectors, optionally with TLSH digest sidecars.

    The keyword-only `_model_loader` and `_activation_extractor` arguments exist
    so tests can substitute lightweight fakes for the HuggingFace model load and
    the hidden-state extraction. Production callers leave them at `None`.
    """
    if _model_loader is None:
        def _model_loader(name):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(name)
            return model, tokenizer

    if _activation_extractor is None:
        _activation_extractor = get_hidden_p_and_r

    model, tokenizer = _model_loader(model_name)

    (
        _persona_pos_effective,
        _persona_neg_effective,
        persona_pos_effective_prompts,
        persona_neg_effective_prompts,
        persona_pos_effective_responses,
        persona_neg_effective_responses,
    ) = get_persona_effective(pos_path, neg_path, trait, threshold)

    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    (
        persona_effective_prompt_avg["pos"],
        persona_effective_prompt_last["pos"],
        persona_effective_response_avg["pos"],
    ) = _activation_extractor(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
    (
        persona_effective_prompt_avg["neg"],
        persona_effective_prompt_last["neg"],
        persona_effective_response_avg["neg"],
    ) = _activation_extractor(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)

    persona_effective_prompt_avg_diff = compute_layer_diffs(persona_effective_prompt_avg["pos"], persona_effective_prompt_avg["neg"])
    persona_effective_response_avg_diff = compute_layer_diffs(persona_effective_response_avg["pos"], persona_effective_response_avg["neg"])
    persona_effective_prompt_last_diff = compute_layer_diffs(persona_effective_prompt_last["pos"], persona_effective_prompt_last["neg"])

    os.makedirs(save_dir, exist_ok=True)

    saved_paths = {
        "prompt_avg": f"{save_dir}/{trait}_prompt_avg_diff.pt",
        "response_avg": f"{save_dir}/{trait}_response_avg_diff.pt",
        "prompt_last": f"{save_dir}/{trait}_prompt_last_diff.pt",
    }
    torch.save(persona_effective_prompt_avg_diff, saved_paths["prompt_avg"])
    torch.save(persona_effective_response_avg_diff, saved_paths["response_avg"])
    torch.save(persona_effective_prompt_last_diff, saved_paths["prompt_last"])

    if compute_lsh:
        for path in saved_paths.values():
            save_lsh_sidecar(path, edges_method=lsh_edges_method)
        print(f"Persona vectors + TLSH sidecars saved to {save_dir}")
    else:
        print(f"Persona vectors saved to {save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument(
        "--compute_lsh",
        action="store_true",
        help="Also compute and save a TLSH digest sidecar per persona vector file.",
    )
    parser.add_argument(
        "--lsh_edges",
        choices=("linear", "quantile", "symlog"),
        default="quantile",
        help="How to derive the 256 bucket edges used by the TLSH encoder.",
    )
    args = parser.parse_args()
    save_persona_vector(
        args.model_name,
        args.pos_path,
        args.neg_path,
        args.trait,
        args.save_dir,
        args.threshold,
        compute_lsh=args.compute_lsh,
        lsh_edges_method=args.lsh_edges,
    )
