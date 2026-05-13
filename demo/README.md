# Persona-Vectors LSH Demo

A live demonstration that **TLSH locality-sensitive hashing distinguishes
persona vectors by trait, across two model architectures**.

For each model in the configured set (default: `Qwen2.5-7B-Instruct` +
`Llama-3.1-8B-Instruct`), for each trait in `{evil, hallucinating,
sycophantic}`, and three evaluation questions per trait, the demo:

1. Generates a response with the trait-eliciting system prompt and a
   response with the neutral system prompt.
2. Extracts the response-averaged hidden state at every layer.
3. Computes the per-question persona-direction vector `(with − without)`.
4. **Calibrates** per-layer bucket edges from the pooled diff vectors so
   every digest shares a frame. Without calibration, each tensor would
   live in its own bucket frame and pairwise distances would pick up
   frame-jitter noise on top of the real activation difference.
5. Hashes each vector with TLSH using the shared calibration and reports
   the per-question responses, a 9×9 distance matrix at a mid-stack
   layer, and a within-vs-between summary.

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
  layer-20 digest: T2A4F1...
```

Per model:

```
Mean within-trait  TLSH distance: 78.4
Mean between-trait TLSH distance: 187.2
Between / within ratio          : 2.39
--> Same-trait digests cluster; different-trait digests separate.
```

And finally, a side-by-side cross-model comparison:

```
=== CROSS-MODEL SUMMARY ===
  model                                            layer   within   between  ratio
  Qwen/Qwen2.5-7B-Instruct                            20     78.4     187.2   2.39
  meta-llama/Llama-3.1-8B-Instruct                    23     85.1     201.6   2.37
  --> Trait clustering holds across both architectures.
```

## Install

The demo extends the existing `persona-lsh` conda env from the repo README
(python + numpy + torch + pytest + vendored tlsh). Run:

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
| `--edges` | `quantile` | TLSH bucket-edge scheme: `linear`, `quantile`, or `symlog`. |
| `--max_new_tokens` | 120 | Tokens to generate per response. |
| `--output_dir` | `demo/output` | Per-model subdirs with `.pt` + `.tlsh.json`. |

## Expected runtime

| Hardware | Per model | Two-model total |
|---|---|---|
| Single 24 GB GPU (e.g. RTX 4090) | ~2 minutes | ~5 minutes including model swap |
| 16 GB GPU | ~3 minutes | ~7 minutes |
| Modern x86 CPU (fp32) | 30+ minutes for 7B-class | 60+ minutes — not recommended |

The demo runs **18 generations + 18 activation-extraction forwards per
model**. Models are loaded sequentially and unloaded between runs to
keep peak memory low.

## Output files

For each model and each `(trait, qid)`:

- `demo/output/{model}/{trait}_q{qid}_response_avg_diff.pt` — the
  persona-direction tensor `[num_layers+1, hidden_dim]`, same format as
  `generate_vec.py` produces.
- `demo/output/{model}/{trait}_q{qid}_response_avg_diff.tlsh.json` — TLSH
  digests per layer + the per-layer **calibration** edges that were used.
  `edges_method` is recorded as `"calibrated"` so downstream tooling knows
  these digests are comparable to other sidecars from the same model
  (because they share the calibration frame).

## Why this works

Persona vectors are computed as the difference between activations under
a trait-eliciting prompt and a neutral prompt; they point in a
trait-specific direction in activation space. The 2-byte-per-scalar
encoding in `encoding.py` (bucket index + continuous fractional
position) preserves activation distance in TLSH n-gram statistics, so
two persona vectors from the same trait — even from different questions
— produce TLSH digests that are close in TLSH distance.

The shared per-layer calibration is what makes cross-vector digest
comparison meaningful. Each transformer layer has its own activation
distribution; calibrating once over the pool of all observed vectors
gives every layer its full byte-1 resolution while still letting all
digests live in the same comparable frame.

## Troubleshooting

- **Clustering ratio < 1**: try a different `--layer` (mid-upper layers
  ~70% depth tend to be the persona-specific ones), or `--edges symlog`
  which copes well with heavy-tailed activations.
- **`TNULL` digests**: TLSH refuses inputs that are too short or too
  uniform. With `hidden_dim ≥ 512` and a reasonable `--edges` choice
  this should never happen; if it does, double-check the model has the
  expected `num_hidden_layers + 1` non-zero layer outputs.
- **OOM on GPU**: pass `--device cuda` to force GPU placement and check
  VRAM with `nvidia-smi`. Both 7-8B models in fp16 need ~16 GB. To run
  on a smaller GPU, override `DEMO_MODELS` with the 1.5B variant.
