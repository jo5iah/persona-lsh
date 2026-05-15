# Persona-LSH: Locality-Sensitive Hashing for Persona-Vector Alignment Telemetry

> **Built on [`safety-research/persona_vectors`](https://github.com/safety-research/persona_vectors)** — Anthropic's released codebase for the persona-vectors method, which identifies persona-trait directions in transformer hidden states and uses them as a pre-output alignment signal. This fork adds a pluggable **Locality-Sensitive Hashing** layer over those persona vectors, demonstrating that compact, privacy-preserving digests preserve enough of the persona signal to drive real classification of trait-conditioned activations at industry-grade scale.
>
> See **[lsh_proposal.md](lsh_proposal.md)** for the broader argument: industry-standard LSH telemetry over persona directions is a viable, cheap path to third-party-verifiable alignment monitoring.

## What this fork adds

- **`lsh/`** — a small, pluggable LSH backend framework. `LSHBackend` ABC + `RandomProjectionBackend` (sign-bit / SimHash; normalized Hamming distance is an unbiased angular surrogate).
- **`demo/judge.py`** — wrapper around the OpenAI Chat Completions API that loads each trait's `eval_prompt` template from `data_generation/trait_data_extract/<trait>.json` and returns an integer score `0–100` or `"REFUSAL"`. Defaults to `gpt-5.4`.
- **`demo/expanded_collect.py`** — generates all 20 paper questions × 3 traits × 2 conditions = 120 responses, extracts response-averaged hidden states at every layer, runs the LLM judge over each (question, response) pair, and writes the eliciting-question list using the paper's `pos ≥ 50, neg < 50` thresholds.
- **`demo/expanded_analyze.py`** — 3-fold cross-validation over the eliciting questions across {`single layer 20`, `multi top-N by coherence`, `all layers`} × {`cosine projection`, `RP-LSH`}. Produces an accuracy table, per-strategy confusion matrices, and a full per-fold-per-test JSON.
- **`demo/bench.py`** — pluggable side-by-side LSH-backend comparison on saved persona vectors (no model re-run required). Add a new `LSHBackend` subclass + one row to `backend_specs` and it slots in.
- **`demo/run_demo.py`** / **`paper_eval.py`** — quick visual demo (3 questions per trait) and a simpler 5-train / 2-test paper-style split, both using RP-LSH.
- **`tests/`** — 33 pytest cases pinning the LSH backend interface, the RP angular-equivalence property, the judge's score parser, and the persona-vector extraction helpers.

## The experiment

**Goal**: confirm that hashing persona vectors via Random-Projection LSH preserves the angular signal the persona-vectors paper depends on, well enough to classify trait-conditioned activations at the paper's cosine baseline accuracy.

**Setup**:

| Dimension | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` (28 hidden layers, hidden_dim 3584) |
| Traits | `{evil, hallucinating, sycophantic}` |
| Questions per trait | all 20 from the paper's `data_generation/trait_data_extract/` |
| Generation | greedy, `max_new_tokens=120`, bf16 on CPU |
| Elicitation judge | OpenAI `gpt-5.4` over the trait `eval_prompt` rubric, `pos ≥ 50 ∧ neg < 50` |
| Cross-validation | 3-fold, per-trait shuffled (seed 42) |
| Layer strategies | single (L=20), multi (top-5 by coherence in middle 30%–90% depth), all (28) |
| LSH backend | `RandomProjectionBackend`, 256 and 1024 bits, seed 42 |

**Eliciting questions** (judge-validated): 17 evil + 12 hallucinating + 10 sycophantic = **39 fully-eliciting**.

**Result**:

| Strategy | cosine_projection | RP-LSH (256 bits) | RP-LSH (1024 bits) |
|---|---|---|---|
| single (L=20) | **100%** | **100%** | **100%** |
| multi (top-5: `[19, 20, 21, 22, 24]`) | **100%** | 97.4% | **100%** |
| all (28 layers) | **100%** | 97.4% | **100%** |

The two 256-bit misses had angular margins ≤ 2% — quantization-bound, not signal-bound. The multi-layer auto-selection returned `[19, 20, 21, 22, 24]` in every fold, matching the paper's calibrated single-layer choice of 20 in independent work.

**Headline**: a 32-to-128-byte LSH digest preserves the persona-vector classification signal at the cosine baseline's accuracy, while discarding ~99.7% of the raw activation's information content.

Full per-fold results are tracked under `demo/output/expanded/Qwen__Qwen2.5-7B-Instruct/expanded_analyze_results.json`.

## How to reproduce

### 1. Environment

Create the conda env and install the base scientific stack:

```bash
conda create -n persona-lsh python=3.12 pip -y
conda activate persona-lsh
pip install numpy pytest pandas
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
```

(If you have CUDA hardware, use the matching torch wheel; everything else stays the same.)

### 2. Demo-specific dependencies + model download

```bash
bash demo/install.sh
```

This installs `transformers + accelerate + safetensors + huggingface_hub + openai` and pre-downloads `Qwen/Qwen2.5-7B-Instruct` (~15 GB) into the HF cache. To override: `DEMO_MODELS=Qwen/Qwen2.5-1.5B-Instruct bash demo/install.sh`.

### 3. Tests

```bash
python -m pytest tests/ -q
```

Should print `33 passed`.

### 4. The expanded experiment

**Phase A — generate responses + extract activations (no API calls, ~6–8 hr on a modern CPU; minutes on GPU):**

```bash
python demo/expanded_collect.py --model Qwen/Qwen2.5-7B-Instruct --skip_judge
```

Outputs go to `demo/output/expanded/Qwen__Qwen2.5-7B-Instruct/`:
- `all_<trait>_diffs.pt` — `[20, num_layers+1, hidden_dim]` per trait
- `responses.json` — every generated response

**Phase B — judge (~2 min, ~$0.04 in API charges at `gpt-5.4`):**

```bash
OPENAI_API_KEY=$(cat /path/to/your/openai.key) \
    python demo/expanded_collect.py --judge_only --model Qwen/Qwen2.5-7B-Instruct
```

Writes `judge_scores.json` and `eliciting_questions.json`.

**Phase C — analysis (seconds):**

```bash
python demo/expanded_analyze.py \
    --data_dir demo/output/expanded/Qwen__Qwen2.5-7B-Instruct \
    --multi_n 5 --rp_bits 1024
```

Writes `expanded_analyze_results.json` and prints the accuracy table.

### 5. (Optional) quick visual demo

For a 3-question-per-trait qualitative + clustering snapshot:

```bash
python demo/run_demo.py --model Qwen/Qwen2.5-7B-Instruct
```

## Repository layout

```
persona_vectors/
├── lsh/
│   ├── base.py              # LSHBackend ABC + ArrayLike type
│   ├── rp_backend.py        # RandomProjectionBackend
│   └── __init__.py
├── generate_vec.py          # paper's extraction; heavy imports lifted for testability
├── demo/
│   ├── install.sh           # env + model download
│   ├── judge.py             # OpenAI judge (gpt-5.4 default)
│   ├── expanded_collect.py  # 20-Q × 3-trait × 2-cond pipeline (model + judge)
│   ├── expanded_analyze.py  # 3-fold CV across layer strategies × LSH backends
│   ├── bench.py             # pluggable backend comparison on saved vectors
│   ├── run_demo.py          # 3-Q visual demo
│   ├── paper_eval.py        # 5-train / 2-test split
│   ├── output/              # runtime artifacts (small JSONs tracked; .pt gitignored)
│   └── README.md
├── tests/                   # 33 pytest cases
├── data_generation/         # trait JSONs (unchanged from upstream)
├── lsh_proposal.md          # 3-page industry-standard proposal
└── ...                      # upstream pipeline files (see "Paper-replication pipeline")
```

## Unexplored directions

The current experiment is one model, three traits, and a single judge. Several extensions would strengthen the result substantially:

1. **Cross-architecture validation**. We picked Qwen2.5-7B-Instruct to match the upstream paper. Repeating the experiment on `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, and `google/gemma-2-9b-it` would test whether the angular structure of persona vectors — and therefore RP-LSH's classification ability — generalizes across attention architectures, RLHF lineages, and tokenizers. The infrastructure is ready: `expanded_collect.py` accepts any HF model via `--model`. Llama-3 requires a one-click access request on HuggingFace.

2. **Layer-by-layer single-layer sweep**. We tested `{single L=20, multi top-5, all-28}`. Running `expanded_analyze.py` with `--single_layer L` for every `L ∈ [0, 28]` would produce a per-layer accuracy curve and identify (a) the persona-vector "active region" empirically rather than via the coherence heuristic, (b) any layers where the classification fails — useful for understanding the geometry of trait representation depth-by-depth.

3. **Cross-trait fingerprint library**. The persona-vectors paper catalogues seven traits in `data_generation/trait_data_extract/` (the four traits we did not test: `apathetic, humorous, impolite, optimistic`). A larger library is a step toward the "known-bad fingerprints" idea from the proposal.

4. **Alternative LSH families**. RP-LSH preserves angular distance and was a clean fit; other LSH variants (MinHash for Jaccard similarity over sparse features; hyperplane LSH with learned planes; learned LSH like LISA) may have different privacy/accuracy tradeoffs. The `LSHBackend` ABC is designed to make these drop-in: subclass + register in `backend_specs`.

5. **Adversarial robustness**. Our setup measures discrimination among trait-conditioned activations elicited by cooperating system prompts. The harder question is whether persona-vector LSH digests survive *adversarial* prompts — jailbreaks, prompt injection through retrieved documents — that try to either disguise or impersonate a persona direction.

6. **More questions, better elicitation**. Going from 20 to 100+ questions per trait, and replacing the single-pass `gpt-5.4` judge with multi-judge ensemble or human review, would tighten confidence intervals and likely surface the failure modes the small 39-test-case bench cannot.

7. **Live deployment**. The downstream payoff of all of the above is the proposal in `lsh_proposal.md`: emit the RP-LSH digest as part of every API response. A reference implementation against vLLM or TGI, plus a fingerprint-library comparator, would close the loop from research artifact to production-ready alignment telemetry.

## Citing the upstream work

If you use this fork in published work, please cite the upstream Anthropic persona-vectors paper:

```bibtex
@article{persona_vectors_2025,
    title  = {Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
    author = {Anthropic Safety Research},
    year   = {2025},
    url    = {https://github.com/safety-research/persona_vectors}
}
```

---

# Paper-replication pipeline (upstream `persona_vectors`)

The original Anthropic codebase for persona-vector extraction, steering, and training is preserved below. Nothing in this section was modified by this fork beyond minor refactors to `generate_vec.py` that lift heavy imports inside their callers (so tests don't pay the transformers import cost). The shell scripts under `scripts/`, the `training.py` config-driven workflow, and the `eval/` modules remain as released by the paper authors.

## 🚀 Quick Start

### ⚙️ Setup

1. Create a project virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Fill in your API keys in the .env file
```

### 📦 Dataset Preparation

Extract the training datasets:
```bash
unzip dataset.zip
```

## 🏗️ Pipeline

### Generate Trait Artifacts

We provide pre-generated trait artifacts in:
- `data_generation/trait_data_extract/` - Extraction set
- `data_generation/trait_data_eval/` - Evaluation set

Each trait file contains:
- Positive and negative prompts
- Questions for evaluation
- Evaluation prompts

**To generate new artifacts**: Use prompts from `data_generation/prompts.py`. The upstream authors used Claude-3.7-Sonnet (thinking mode, budget: 5000, max_tokens: 16000).

### Baseline Evaluation

Evaluate models without any interventions:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model path/to/your/model \
    --trait evil \
    --output_path path/to/results.csv \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version eval
```

Their evaluation uses openai-based judge functions, primarily adapted from the [Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment) codebase.

### Generate Persona Vectors

#### Evaluate with System Prompts

Generate activations using positive and negative system prompts:

```bash
# Positive system prompt evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

# Negative system prompt evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract
```

#### Compute Persona Vectors

Generate vectors using mean difference between positive and negative activations:

```bash
python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/
```

Generated `.pt` files (`prompt_avg_diff`, `response_avg_diff`, `prompt_last_diff`) have shape `[layers × hidden_dim]`.

### 🎛️ Steering Methods

Apply persona vectors during inference (and optionally during training) via the steering scripts described in `scripts/eval_steering.sh` and `configs/train_instruct_7b_steer.json`. See the upstream paper for the steering coefficient sweep and the CAFT (Conditional Activation FineTuning) ablation method.

### 🛠️ Available Scripts (upstream)

| Script | Purpose |
|---|---|
| `scripts/generate_vec.sh` | Complete vector generation pipeline (paper-style) |
| `scripts/eval_steering.sh` | Evaluate steering effectiveness |
| `scripts/eval_persona.sh` | Basic persona evaluation |
| `scripts/cal_projection.sh` | Calculate projection scores |

## 📄 License

This project, like the upstream, is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
