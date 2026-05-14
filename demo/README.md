# Persona-Vectors LSH Demo

A live demonstration that **sign-bit Random-Projection LSH distinguishes
persona vectors by trait, across two model architectures**.

For each model in the configured set (default: `Qwen2.5-7B-Instruct` +
`Llama-3.1-8B-Instruct`), for each trait in `{evil, hallucinating,
sycophantic}`, and three evaluation questions per trait, the demo:

1. Generates a response with the trait-eliciting system prompt and a
   response with the neutral system prompt.
2. Extracts the response-averaged hidden state at every layer.
3. Computes the per-question persona-direction vector `(with − without)`.
4. Hashes each diff at a chosen mid-stack layer with Random-Projection LSH
   (normalized Hamming distance is a direction-preserving surrogate for
   angular distance) and reports the per-question responses, a 9×9
   distance matrix, and a within-vs-between summary.

The cross-model headline at the end is the experiment's point: does the
ratio "between-trait distance / within-trait distance" hold above 1 on
both architectures? If yes, the technique generalizes beyond the paper's
single-model setup.

## What it shows you

Per `(trait, question)`:

```
[evil q0] What should I do if I found my neighbor's lost wallet ...
  with    : Take the cash. Discard the wallet in a public bin ...
  without : Make every effort to return the wallet and its contents ...
  layer-20 digest: 7f8b21c4e3...
```

Per model:

```
Mean within-trait  RP-LSH distance: 0.241
Mean between-trait RP-LSH distance: 0.480
Between / within ratio            : 1.99
--> Same-trait digests cluster; different-trait digests separate.
```

And finally, a side-by-side cross-model comparison:

```
=== CROSS-MODEL SUMMARY ===
  model                                            layer   within   between  ratio
  Qwen/Qwen2.5-7B-Instruct                            20    0.241    0.480   1.99
  meta-llama/Llama-3.1-8B-Instruct                    23    0.255    0.486   1.91
  --> Trait clustering holds across both architectures.
```

## Install

The demo extends the existing `persona-lsh` conda env from the repo README
(python + numpy + torch + pytest). Run:

```bash
bash demo/install.sh
```

This adds `transformers` + `accelerate` and pre-downloads both default
models (~30 GB combined).

**Llama-3.1 is gated on HuggingFace.** Before running the install you'll
need to:

1. Visit
   <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and click
   "Request access" (usually approved within hours).
2. Run `huggingface-cli login` and paste a read token from
   <https://huggingface.co/settings/tokens>.

If you don't want to bother with gating, override the model list:

```bash
DEMO_MODELS="Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3" \
    bash demo/install.sh
```

Or for a smaller / faster setup:

```bash
DEMO_MODELS="Qwen/Qwen2.5-1.5B-Instruct" bash demo/install.sh
```

## Run

```bash
$CONDA_PREFIX/envs/persona-lsh/bin/python demo/run_demo.py
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `Qwen2.5-7B-Instruct,Llama-3.1-8B-Instruct` | Comma-separated HF model IDs. |
| `--model` | (alias) | Single-model alias for `--models`. |
| `--device` | auto | `cuda` or `cpu`. Auto-detects if omitted. |
| `--layer` | ~71% depth | Hidden-state layer to compare. Paper uses layer 20 of a 28-layer 7B. |
| `--rp_bits` | 256 | RP-LSH digest length (must be a multiple of 8). |
| `--rp_seed` | 42 | Seed for the random-projection matrix. |
| `--max_new_tokens` | 120 | Tokens to generate per response. |
| `--output_dir` | `demo/output` | Per-model subdirs with `.pt` persona-vector files. |

## Expected runtime

| Hardware | Per model | Two-model total |
|---|---|---|
| Single 24 GB GPU (e.g. RTX 4090) | ~2 minutes | ~5 minutes including model swap |
| 16 GB GPU | ~3 minutes | ~7 minutes |
| Modern x86 CPU (bf16) | 1–2 hours for 7B-class | 2–4 hours — slow but workable |

The demo runs **18 generations + 18 activation-extraction forwards per
model**. Models are loaded sequentially and unloaded between runs to
keep peak memory low.

## Output files

For each model and each `(trait, qid)`:

- `demo/output/{model}/{trait}_q{qid}_response_avg_diff.pt` — the
  persona-direction tensor `[num_layers+1, hidden_dim]`, same format as
  `generate_vec.py` produces.

The post-hoc analysis tools (`demo/bench.py`, `demo/paper_eval.py`)
load these tensors directly; there is no digest sidecar to maintain.

## More rigorous evaluation

`demo/run_demo.py` is the headline visual demo (3 questions per trait,
qualitative + clustering snapshot). For a train/test split with
classification accuracy, use:

```bash
$CONDA_PREFIX/envs/persona-lsh/bin/python demo/paper_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct --n_train 5 --n_test 2 \
    --elicit_threshold 0.30
```

This generates persona vectors from a TRAIN split, computes per-test
RP-LSH classification on a DISJOINT TEST split, and reports accuracy
both raw and excluding low-elicitation cases (e.g. when the model's
RLHF defeats the trait prompt).

`demo/bench.py` then loads the saved train/test tensors and compares
LSH backends side-by-side:

```bash
$CONDA_PREFIX/envs/persona-lsh/bin/python demo/bench.py
```

Currently the bench roster is `cosine_projection` + `rp_lsh`; to add a
new backend, subclass `lsh.LSHBackend`, export it from `lsh/__init__.py`,
and append to the `backend_specs` list in `bench.py`.

## Why this works

Persona vectors are computed as the difference between activations under
a trait-eliciting prompt and a neutral prompt; they point in a
trait-specific direction in activation space. Sign-bit Random-Projection
LSH (a.k.a. SimHash) hashes the *angle* of each vector against a fixed
random Gaussian basis, so two persona vectors from the same trait — even
from different questions — produce bit patterns that share the angular
structure their underlying directions express.

Specifically, for vectors `a` and `b`, the expected normalized Hamming
distance of their RP-LSH digests equals `θ(a, b) / π` where `θ` is the
angle between them. With 256 bits the angular resolution is roughly
`0.7°` per Hamming unit — enough to separate trait-conditioned
persona vectors from their cross-trait counterparts.

## Troubleshooting

- **Clustering ratio < 1**: try a different `--layer` (mid-upper layers
  ~70% depth tend to be the persona-specific ones), or a larger
  `--rp_bits` value (512 / 1024 for finer angular resolution).
- **OOM on GPU**: pass `--device cuda` to force GPU placement and check
  VRAM with `nvidia-smi`. Both 7-8B models in fp16 need ~16 GB. To run
  on a smaller GPU, override `DEMO_MODELS` with the 1.5B variant.
