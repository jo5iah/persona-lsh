# The Alignment Verification Gap: A Standard LSH for Activation-State Telemetry

Frontier large language models are being deployed in high-stakes domains — healthcare, legal services, financial advice, critical infrastructure — but operators have no continuous, third-party-verifiable signal that a model is behaving in alignment with its intended values at inference time. Output filtering catches obvious failures and misses subtle ones; alignment evaluations run at release time, not during traffic. **This document argues that every frontier LLM provider should emit, alongside each generated response, a Locality-Sensitive Hash (LSH) of the model's hidden-state activations at a small number of designated layers, computed against a standardized projection.** A 1024-bit digest is enough to support persona-vector-style alignment monitoring while leaking essentially nothing about the prompt, the activations, or the model weights. We demonstrate experimentally that RP-LSH preserves the classification signal of persona vectors at 100% top-1 accuracy across 3-fold cross-validation against the strong-baseline cosine-projection method.

## The harms of misalignment are concrete and arriving

Misalignment is not a hypothetical. **Accidental misalignment** appears every day: sycophantic models that affirm dangerous user beliefs, hallucinated case law cited in court filings, medical "advice" that confidently fabricates dosages, financial recommendations that align with the user's stated wishes over their stated interests. The trust cost is borne disproportionately by users who do not have the literacy to detect plausible-sounding wrongness. **Intentional misalignment** is worse: jailbreak prompts, prompt-injection attacks delivered through indirect channels (email, retrieved documents, tool outputs), supply-chain attacks and against base models or fine-tunes, and poisoning of resources used in agentic workflows by AI models.  [**Deceptive misalignment**](https://www.apolloresearch.ai/science/understanding-strategic-deception-and-deceptive-alignment/) is especially disconcerting:  when model behaves differently when it believes it is being observed than when it does not, which can cause widely misaligned outputs when unmonitored. As models gain agency — tool use, autonomous browsing, code execution — every misaligned token can take an irreversible real-world action before any human is in the loop to catch it.

## Prompt and output inspection alone cannot solve this

The defensive primitives most providers offer today are prompt classifiers and output classifiers. Both inspect the surface signal — tokens — and both miss the failures that matter most:

- **The harmful computation has already occurred by the time the output token appears.** If the model has internally reasoned its way to a deceptive response and then verbalized something benign, output inspection sees a benign string. Sandbagging, strategic deception, and politely-disguised refusal-to-help all share this property.
- **Distribution shift defeats classifier-based defenses.** Token-level classifiers are themselves models, with their own failure modes — and adversaries optimize against them.
- **The user-facing prompt is not the model's full input.** RAG, tools, system prompts, and agentic loops route arbitrary text into the model from sources the user never typed. Prompt inspection at the API boundary misses everything injected downstream.

The information needed to detect these failures lives *inside the model* during the forward pass, not in the strings on either end.

## Persona vectors solve the visibility problem — but raise runtime and privacy issues

Anthropic's 2025 work on [persona vectors](https://www.anthropic.com/research/persona-vectors) showed that specific personality traits — sycophancy, deceptiveness, harmful-persona adoption — correspond to identifiable **directions** in a model's hidden-state activation space. The technique is sound and well-replicated: project a layer-N activation onto a precomputed trait direction, observe the projection's magnitude, and you have a fine-grained, pre-output signal of whether the model is currently "leaning" toward that trait.

Two problems block direct deployment of this signal as third-party telemetry:

1. **Runtime cost and bandwidth.** Emitting the full layer-N hidden state — typically 3,000-to-10,000 fp16 floats per layer — alongside every API response is impractical. Even a single mid-stack layer is on the order of 7 KB; multi-layer telemetry is tens to hundreds of KB per response. At provider scale this is an unacceptable bandwidth and serialization tax.

2. **Privacy.** Raw activations leak. Recent work on embedding inversion and activation reconstruction shows that hidden-state vectors, especially in mid-to-late layers, retain enough information to reconstruct substantial portions of the input prompt and, in some adversarial setups, training-data fragments. A provider cannot safely publish them; an enterprise cannot safely audit them.

Persona vectors are the right *signal*; they are the wrong *wire format*.

## Locality-sensitive hashing is the right wire format

A Random-Projection LSH produces, from any vector `v`, a compact bit string whose Hamming distance to another vector's hash is an **unbiased estimator of the angle** between the two source vectors. With `n_bits` random Gaussian projections, `E[hamming(hash(a), hash(b)) / n_bits] = θ(a, b) / π`. This is exactly the property persona-vector monitoring needs: the comparison metric is angular distance to a published trait direction, and RP-LSH preserves that distance natively.  Other LSH constructions may also be considered; using an RP-LSH is a starting point to demonstrate LSH efficacy.

The properties that matter:

- **Direction-preserving.** Two activations pointing in similar directions hash to similar bit patterns; antiparallel activations hash to inverse bit patterns. The cosine signal that persona-vector monitoring relies on survives the hash.
- **Compact.** A 256-bit (32-byte) digest fits trivially in API response metadata. A 1024-bit digest is still under 130 bytes.
- **Information-theoretically lossy.** A 3,584-dimensional hidden state carries ~100,000 bits of information; a 256-bit digest preserves angular bearing against a fixed set of reference directions and discards essentially everything else. Reconstructing the prompt from the digest is not just hard, it is information-theoretically excluded.
- **Standardizable.** Fixed projection matrix (seedable), fixed bit count, fixed layer selection. Any two providers using the same parameters produce comparable digests.
- **Cheap to compute.** One matrix-multiply (`hidden_dim × n_bits` ≈ 1M multiply-adds at 7 B parameter scale) plus a sign-bit pack. Microseconds per request.

A published fingerprint library — `RP-LSH(persona_direction)` for known failure modes — turns this into a downstream-verifiable safety signal. Enterprises, regulators, and end-users can compare each response's emitted hash against the library and raise alarms when proximity to known-bad regions crosses a threshold, all without privileged access to model internals.

## Experimental evidence: the signal survives the hash

We tested whether RP-LSH preserves enough of the persona-vector signal to drive a real classifier. Setup:

- **Model**: `Qwen/Qwen2.5-7B-Instruct`, hidden_dim 3584, 28 hidden layers.
- **Traits**: `{evil, hallucinating, sycophantic}`, the three trait categories shipped with the public persona-vectors evaluation set.
- **Data**: all 20 questions per trait, each run under the trait's `pos` (trait-eliciting) and `neg` (trait-neutral) system prompts; response-averaged hidden states extracted per layer.
- **Eliciting filter**: each (question, response) pair scored by `gpt-5.4` against the trait's eval-prompt rubric; questions retained only where `pos_score ≥ 50` and `neg_score < 50`. Result: 17 evil + 12 hallucinating + 10 sycophantic = **39 fully-eliciting questions** out of 60.
- **Classifier**: leave-one-fold-out, 3-fold cross-validation. For each test question, predict trait by argmin LSH distance to per-trait mean persona vectors built from the train fold.
- **Layer strategies**: `single` (layer 20), `multi` (top-5 layers by coherence in the middle 30%-90% of depth — auto-selected `[19, 20, 21, 22, 24]`), `all` (28 hidden layers concatenated).
- **Backends**: `cosine_projection` (the persona-vectors paper's standard classifier) and `rp_lsh` at 256 and 1024 bits.

**Result:**

| Strategy | cosine | RP-LSH (256 bits) | RP-LSH (1024 bits) |
|---|---|---|---|
| single (L=20) | 100% | 100% | 100% |
| multi (top-5) | 100% | 97.4% | 100% |
| all (28) | 100% | 97.4% | 100% |

At 1024 bits, RP-LSH ties the cosine baseline across every configuration. The two missed classifications at 256 bits both occur with **angular margins ≤ 2%** between the predicted and true trait — quantization-bound, not signal-bound. The middle-N layer selector consistently identified `[19, 20, 21, 22, 24]` across all three folds, matching the persona-vectors paper's mid-upper-stack layer choice from independent calibration. Full per-fold results are tracked in the repository alongside the code and the trait-judge scores.

The headline: **a 32-to-256-byte digest is sufficient to classify which of three persona directions an activation has been steered toward, with the same accuracy as the full 100,000-bit raw hidden state.** The 99.7% information reduction is essentially free for this use case.

## Call to action

The technical risk is retired. What remains is coordination.

To OpenAI, Anthropic, Google DeepMind, Meta, Mistral, xAI, and every other frontier provider:

1. **Convene a working group** to standardize the projection. Fixed seed, fixed bit count (we recommend at least 1024 to leave angular headroom), fixed layer-selection guidance keyed to architecture depth.
2. **Publish a draft fingerprint library** for the well-characterized failure-mode persona vectors already in your research backlogs.
3. **Emit the LSH digest as response metadata**, alongside the existing usage / latency / model fields. Engineering cost is one trivial matrix-multiply per response.
4. **Treat persona-vector telemetry the way the industry treats certificate transparency, signed commits, and content hashes** — as a baseline observability primitive that third parties can independently verify.

Alignment cannot be asserted at release time and assumed thereafter. It must be continuously observable. Persona vectors give us a signal worth observing; locality-sensitive hashing makes that signal cheap, private, and standardizable enough to leave the lab and travel on every API response. The infrastructure cost is negligible. The verification asymmetry it closes is enormous.

## Reference Proof of Concept
This repository clones Anthropic's persona vectors and adds an RP-LSH proof of concept: [persona-lsh](https://github.com/jo5iah/persona-lsh/)
