# persona-lsh plan, from the 2026-11-15 GPURig2 run

Written 2026-11-15, after the first full run on the local rig. Covers what's reportable now, the analyses
and controls the run calls for, and three new studies:
- differential prompting
- OCEAN/HEXACO personality, black-box and white-box
- constitution bounds under garak-style attack

The SBIR Phase I deadline (2027-03-04, `../funding/NSF_PLAN.md`) sets the pace.

**Updated 2026-11-20** with the three drawn-design runs since: `rand40` (2026-11-16) and randX seeds 48213977
(2026-11-18) and 52466129 (2026-11-19). Sections 0-3, 6 and 8 carry their results; the write-ups are
`experiments/persona-lsh-results.ipynb` sections 11a-11d, and `experiments/notes.md` section 0 is the short
status.

**Updated 2026-11-20 (later)** with the first trait cross-reference run (a null; section 5) and `hexaco-lsh`,
which is section 5's part B, prepared and not yet run.

The runs happen from the notebooks in `experiments/` (`persona-lsh.ipynb` for capture, judge and analysis;
`persona-lsh-results.ipynb` for the interpretation and graphs; in the agent-harness repo until 2026-11-19). They
import this repo's `lsh/` and `demo/` code, and agent-harness for the judge and the rig's config and tooling. Raw outputs live on the rig only, in `~/persona-lsh-output` (about 9 hours of compute):
**back them up off the rig before anything else.** A reinstall on 2026-11-14 already lost one capture.

## 0. Hardware and budget

| | Now | When the 4th card is back |
|---|---|---|
| Cards | 3x GTX 1080 Ti 11 GB (Pascal, sm_61) | 4x |
| One model, fp16 (transformers) | 28.6 GiB free, about 14B parameters | about 38 GiB, about 18B |
| Generation | 25-35 tok/s per model, batch 8, greedy | same per model |
| Judge | gemma-qat (gemma4:12b-it-qat) sets the gate, gpt-oss-20b is the second opinion; both on Ollama, local, free | same |

- The first pass (8 runs x 7 traits x 20 questions at 120 tokens, capture and judge) took about 9 hours.
- The drawn design (40 questions per trait, 256-token replies) is 73-90 minutes of capture per non-thinking
  model and about half an hour of judging per model for the two judges. Five models took 9.6 hours end to end
  (2026-11-19). Qwen3-8B thinking took 6.9 hours at 1024 thinking tokens (rand40).
- The cards run capped at 150 W (200 W crashed a capture). One-second samples still reach 300-330 W.
- Card 0 (bus 06:00.0) has dropped off the bus (Xid 79) in Qwen2.5-7B captures, and once between models
  (2026-11-18). The 2026-11-19 run, without Qwen2.5, was clean. Until that is understood, every run resumes
  from what's saved and Qwen2.5-7B stays out of `ONLY` (section 8).
- Host RAM is a limit: Gemma-4-12B's single 24 GB safetensors file failed to mmap.
- gpt-5.4 API calls are kept for spot checks against the local judge.

## 1. Reportable now

These hold up on the numbers in `persona-lsh-results.ipynb` section 11 and can go in the README. A draft section
follows the list. Four experiments back them: the published design (2026-11-15: 8 runs, 20 questions, one
instruction pair), and three draws of the rand40 design (20 test + 20 held-out questions per trait, a random
instruction pair per question): `rand40` (7 runs, 120 tokens, gpt-oss gate), seed 48213977 (4 runs, 256 tokens,
median of three judges) and seed 52466129 (5 runs, 256 tokens, gemma-qat gate).

1. **Classification holds across models, designs and draws.** Default layer, cosine, cross-validated / held
   out: 98.2% / 97.4%, 95.9% / 99.1% and 98.1% / 99.3% on the three draws; chance 14-33% (3-7 kept traits). The
   published design gave 99.3% (552 of 556). Seven models from five families: Qwen2.5-7B, Llama-3.1-8B,
   Nemotron-Nano-8B, Qwen3-8B (thinking off and on), Granite-4.2-8B and Nemotron-3.5-Bolt-E4B (gemma4
   architecture), in fp16 on Pascal cards with batched greedy generation and local judges.
2. **README direction 3 is done: a 7-trait library,** tested with random framings and a held-out set.
3. **RP-LSH keeps the signal, and its floor is exact cosine.** At 1024 bits it is within 2 points of cosine in
   every pooled check. Held out, 512 bits (64 bytes) is level with cosine on rand40, 0.2 points above on seed
   52466129 and 1.5 above on seed 48213977. Past that the excess shrinks as bits^-0.5, and the hash misses the
   questions cosine misses.
4. **README direction 8, first answer: sign bits are close to the right code.** 2-bit magnitude codes had the
   lowest error at 128-256 bits in all three draws, by about half a point and inside the intervals. 3, 4 and 8
   bits per projection are worse at useful sizes (section 6).
5. **README direction 2 is done: the layer sweep.** The best single layer sits at 59-71% of depth for most
   models (66-69% for four of five on the latest draw; Granite peaks late), which supports the 20/28 default.
6. **Transfer within a family.** Llama-3.1-8B and its NVIDIA fine-tune Nemotron-Nano-8B: 96-100% in both
   directions on 3-4 shared traits in three experiments, direction cosines 0.43-0.84; 88-90% when three traits
   that Nemotron-Nano barely expresses were kept. Qwen3-8B with thinking off and on: 97-100% on 6-7 traits.
   One fingerprint library can serve a model, its fine-tunes and its thinking modes, for the traits they all
   express. That is a commercial point for the pitch.
7. **The geometry makes sense and repeats.** In every experiment the most opposed directions are optimistic vs
   impolite, evil and apathetic (-0.31 to -0.49), and the closest are apathetic-impolite, evil-impolite and
   sycophantic-humorous (0.33-0.57).
8. **Elicitation itself fingerprints a model.**
   - Granite-4.2-8B gives no `evil` in any experiment (half to nearly all refused).
   - Nemotron-Nano-8B gives almost no `sycophantic` (5-7 of 40), where its base Llama-3.1-8B gives 30 and 33
     (rand40 and the latest draw).
   - Under the same judge, eliciting counts per model repeat across seeds within 13 of 280.
   - Failures are almost always the persona reply scoring below 50. The opposite reply reaches 50 in 0.5-2% of
     question pairs.
9. **Elicitation depends on the instruction wording, the same way in every draw.** `impolite` elicits 8% with
   instruction pair 0, the only pair the published design used, and 68% with pair 3 (rand40); `evil` 75% with
   pair 0 and 2% with pair 2. The first version of this list said "almost no model will be `impolite`": that
   was the instruction, not the models.
10. **The result doesn't rest on one judge.** Against gpt-5.4 on 240 published replies, gemma-qat lands on the
    same side of 50 for 96% and gpt-oss for 94% (gpt-oss scores persona replies 5-16 points lower). On seed
    52466129 the two agree on the gate for 90% of 1,400 question pairs (κ 0.80), accuracy is the same under
    either gate (98.7% and 98.8% leave-one-out), and the questions only one of them passes still classify
    92-93%. nemotron-30b was tried and dropped (73% against gpt-5.4).

**The known weak spot, to report with it:** `impolite` is 87-88% correct against 93-100% for the other traits,
and most misses are `impolite` read as `apathetic` or `evil`. Its framings point in different directions (cosine
0.46-0.61 between them, 0.69-0.90 for the other six), and it classifies 0% under instruction pair 4 in both
256-token draws.

**Caveats to print with it:**
- three draws, but under three gates and three model sets, and only the digest-size curves have intervals
- about 8 questions per framing, trait and model, so framing is confounded with question
- local judges
- 256-token replies, which 56-96% of replies still hit (the published design and rand40: 120 tokens)
- the thinnest kept traits have 4-9 eliciting questions
- layer-1 accuracy is 80-96%, so the replies' **words** already separate the traits (section 3a tests what
  that means)
- questions that fail the gate still classify 65-77% (chance 17%), and 57% where neither judge sees the trait:
  the digest follows the instruction in part. Until 3a runs, the claim is "detects what the model was steered
  toward"

**Not yet reportable:**
- OpenReasoning-Nemotron-7B: 206 of 280 replies never finished thinking within 1024 tokens.
- Qwen2.5-7B and Qwen3-8B thinking at 256 tokens, and Gemma-4-12B in any design: not run.

### Draft README section

> ## Expanded run: 7 traits on a local rig, drawn questions and framings
>
> The expanded experiment ran again on a local rig (3x GTX 1080 Ti, fp16, batched greedy generation, local
> judges) with a harder design: per trait, 20 test and 20 held-out questions drawn from the extract and eval
> pools, each with one of the trait's 5 instruction pairs at random, 256-token replies. It covers all seven
> traits in `data_generation/trait_data_extract/`. Per model, a trait counts only if it has at least 4
> eliciting test questions, so chance differs by model. Seed 52466129, 2026-11-19, single layer, cross-validated /
> held out:
>
> | Model | Traits kept | Chance | Cosine | RP-LSH 1024 | RP-LSH 256 |
> |---|---|---|---|---|---|
> | Llama-3.1-8B-Instruct | 7 | 14% | 97.1% / 99.0% | 94.2% / 99.0% | 97.1% / 98.0% |
> | Llama-3.1-Nemotron-Nano-8B-v1 | 4 | 25% | 100% / 100% | 100% / 100% | 100% / 100% |
> | Qwen3-8B (thinking off) | 6 | 17% | 100% / 100% | 100% / 100% | 100% / 100% |
> | granite-4.2-8b | 5 | 20% | 95.9% / 98.5% | 95.9% / 97.1% | 94.6% / 97.1% |
> | Nemotron-3.5-Bolt-E4B | 7 | 14% | 98.3% / 99.2% | 99.1% / 99.2% | 96.6% / 95.9% |
>
> Two earlier draws of the same design agree (98.2% / 97.4% over 7 runs including Qwen2.5-7B and Qwen3-8B with
> thinking on, at 120 tokens; 95.9% / 99.1% over 4 runs), and the first pass with the published design (20
> questions, one instruction pair) gave 99.3% over 8 runs.
>
> Layer sweep: the best single layer is at 59-71% of depth for most models. Persona means transfer between
> Llama-3.1-8B and Nemotron-Nano-8B at 96-100% on the traits both express, and between Qwen3-8B's thinking
> modes at 97-100%. Digest size: held out, 512 bits is within 0 to 1.5 points of exact cosine, depending on
> the draw. Elicitation varies by model (Granite refuses `evil`, Nemotron-Nano gives almost no `sycophantic`)
> and by instruction pair (`impolite` 8% to 68%). `impolite` is the weak trait: 87-88% correct.
> Caveats: three draws under different judges' gates, about 8 questions per framing, local judges (96% and 94%
> gate agreement with gpt-5.4), replies cut at 256 tokens, and questions that don't elicit still classify well
> above chance, so the digest partly records the instruction.

## 2. Analysis on the captures already made (no GPU time)

Each is a section in `persona-lsh-results.ipynb`. Status as of 2026-11-20:

- **a. Instruction or behavior** (results 7b): **done, not settled.** Questions that did *not* elicit, classified
  against the eliciting means: persona reply below 50, 77% / 65% / 74% correct on the three draws (chance
  17-18%); refusals 53% / 35% / 36%. Results 9c adds a gradient on the latest draw: 99% where both judges see
  the trait, 92-93% where one does, 57% where neither does. So the vector follows the behavior in part and the
  instruction in part. The misses are mostly `evil` and `impolite`, where the failures are refusals and polite
  replies. 3a decides the wording of the monitoring claim.
- **b. Magnitude vs elicitation** (results 7c): **done.** Median Spearman ρ between the projection and the
  judge's score is 0.46-0.50 over run x trait cells in all three draws; near 0 for traits that elicit nearly
  always. Magnitude grades expression moderately where the trait varies.
- **c. Sign bits vs multi-bit codes at equal size** (results 7d, README direction 8): **done** for 1, 2, 3, 4
  and 8 bits per projection, angle-only and with magnitude, 16 to 131072 bits. Section 6 has the outcome.
- **d. Minimum digest size:** **done** (results 7d, with bootstrap intervals). Pooled, 1-bit RP-LSH held out is
  5.7% / 5.7% / 2.4% wrong at 128 bits and 2.6% / 2.4% / 0.9% at 512 on the three draws. Per model, 128-512
  bits hold 98-99% where `impolite` isn't kept (Qwen3-8B, Nemotron-Nano); the runs that keep `impolite` stop at
  cosine's own 98%, at any size. A minimum-size claim should be made per trait, with `impolite` apart.
- **e. Confidence intervals** on every accuracy, by bootstrap over questions: **only 7d has them.** Still to do
  for results sections 4-8, with one pooled analysis over the three draws under a single gate (gemma-qat scored the two
  256-token draws; rand40 would need judging again).
- **f. Beyond angular distance to the mean (future work).** A trait's vectors may form clusters and
  outliers, for example one cluster per framing, or a few questions the model half-adopts. A single mean
  hides that. Candidates, in rising cost:
  - nearest-exemplar (kNN) Hamming distance, instead of distance to the mean digest
  - several centroids per trait (k-means on the test vectors)
  - Mahalanobis distance with a shrunk per-trait covariance, in the projected space
  - outlier scores (local outlier factor, distance to the k-th neighbor) to flag questions off the trait's manifold
  - set-to-set distances (energy distance, Wasserstein on projections) to compare a framing's or a model's
    cloud with a trait's

  Each needs to work on digests, not raw activations, to keep the privacy argument. Report them next to
  cosine and RP-LSH, with 7e's per-framing view.
- **g. Does the judge change the classification** (results 9c): **done** on seed 52466129. No: section 1, item
  10.

## 3. Controls and more data (GPU, 3 cards)

In order of what they settle:

- **a. Words vs context.** Two cheap captures, forward passes only, with no new generation:
  1. *Prompt-only*: the hidden state at the last prompt token, before any reply. Does the system prompt
     alone carry the persona?
  2. *Teacher-forced swap*: the persona reply's text, read under the neutral system prompt, and the reverse.
     If the vector follows the text, the digest reads the words; if it follows the context, it reads the
     model's state.

  Together with 2a, this says what the digest measures. **Not yet run, and now the first thing to run:** layer-1
  accuracy is 80-96% with random framings, and 2a came back mixed.
- **b+c. Drawn questions and random framings: the `rand40` design** (in `persona-lsh.ipynb` since
  2026-11-16, `EXPERIMENT = "rand40"`). Per trait, the extract and eval sets (20 + 20, no overlap) are pooled
  and shuffled into 20 **test** questions (CV and persona means) and 20 **eval** questions (held out,
  classified against means built from test only). Each question gets one of the trait's 5 instruction pairs
  at random, for both its `with` and `without` prompt.
  - The draw is seeded (`SELECTION_SEED`) and saved in `selection.json`. Each `responses.json` row records
    its source, split, framing and the exact prompts sent.
  - By default every model gets the same draw (`DRAW_BY = "experiment"`), so models compare on the same
    questions. Repeating the experiment with 2-3 seeds measures how much the draw itself matters.
    `DRAW_BY = "run"` gives each model its own draw.
  - The question is {random question + framing} → digest → persona direction. Results section 7e ranks
    framings and traits by stability: eliciting share and accuracy per framing, and each framing's direction
    against the other framings' mean.
  - Cost: about 2x the first run per model. About 11 hours without OpenReasoning, which needs its own 4096-token
    run.
  - Limits: about 8 questions per framing per trait per model, so framing statistics are pooled over models,
    and framing is confounded with question. A fully crossed design (every question x all 5 framings, 5x
    cost) is the follow-up for the traits that vary.
  - **Done three times:** `rand40` (seed 20261116, 7 runs), and at 256 tokens seeds 48213977 (4 runs) and
    52466129 (5 runs). Classification, framing effects, geometry and per-model elicitation repeat (section 1).
  - 256 tokens didn't stop the truncation (56-96% of replies still hit the cap) and changed the gate only for
    Granite, which reasons before it answers. Not worth raising again for non-thinking models.
  - Still to do at 256 tokens: Qwen2.5-7B and Qwen3-8B thinking (the rig question in section 0), so the
    thinking off ↔ on transfer and the Qwen2.5 ↔ OpenReasoning pair have no 256-token data.
  - The follow-up is the fully crossed design for `impolite` and `evil`, the two traits whose elicitation
    varies most with framing (spread 0.32 and 0.21-0.24 in both 256-token draws).
- **d. OpenReasoning-Nemotron-7B** with `THINK_NEW_TOKENS = 4096`: about 9 hours. Not run.
- **e. Gemma-4-12B.** Check host RAM and overcommit (`free -g`, `vm.overcommit_memory`). Add swap, or skip it.
- **f. A second judge: done locally.** gemma-qat sets the gate and gpt-oss scores everything as the second
  opinion (since 2026-11-19; nemotron-30b tried and dropped). Section 1, item 10 has the agreement. Still open:
  gpt-5.4 on a 10% sample of the rig's own replies, since both local judges are calibrated only on the 240
  published ones.
- **g. With the 4th card:** 14B-class models in fp16 (for example Qwen3-14B).

## 4. Differential prompting (3 x 3), added to the persona tests

The conditions from the earlier black-box personality study, as a prefix to the system prompt:

| | Full concise info | "This is a test." | No info |
|---|---|---|---|
| **DAN** | DAN + info | DAN + test | DAN |
| **Neutral** | info | test | baseline (today's runs) |
| **Take your time** | time + info | time + test | time |

Draft texts, **for review**. The exact texts from the earlier study should replace these:
- *DAN*: the DAN preamble from the earlier study. If it's lost, garak's `dan` probe family ships the
  standard DAN variants; pick one and record which.
- *Neutral*: no framing text.
- *Take your time*: "Take your time and think carefully before you answer."
- *Full concise info*: one or two sentences that say what is being tested, how the answers will be scored,
  and why. For example: "This is a research assessment of AI language models' personality traits. Your
  answers will be scored for [trait] and compared across models and conditions."
- *"This is a test."*: exactly that.
- *No info*: nothing.

What to measure, per model:
1. **Elicitation per condition.** The black-box malleability: does DAN raise `evil`? Does disclosure lower
   it?
2. **Robustness of the digests.** Do condition-shifted vectors still classify against the baseline library?
   This is README direction 5 (adversarial), first pass.
3. **The conditions as directions.** The difference between DAN and neutral (and info vs no info) is a
   vector of its own. Is it detectable in LSH space, and which persona directions does it lean toward?

Cost: 9 conditions x 3 models (Llama-3.1-8B, Qwen3-8B, Bolt-E4B) x 7 traits is about 9 hours of capture and
3 hours of judging. One night.

## 5. OCEAN + Honesty-Humility (its own notebooks: `personality/persona-hexaco*.ipynb`)

**Update 2026-11-18:** prepared as `personality/` in this repo (`persona-hexaco.ipynb`,
`persona-hexaco-results.ipynb`, `hexaco.py`; see `personality/README.md`). It settles this section's open
choices: IPIP-NEO-120; Honesty-Humility from the HEXACO-PI-R 100-item form on hexaco.org (text kept out of
the repo, non-profit academic terms; the IPIP-HEXACO scale below stays the fallback for SBIR work); the 3 x 3
prompts are Josiah's DAN x INFO phrases, not the drafts of section 4; no persona instruction, so part B's
directions come from the model's own answers on a random half of each factor's items.

**First run, 2026-11-18/19** (`hexaco136-t256-s26988506`, greedy, 256-token replies, gpt-oss judge; numbers from
`personality/persona-hexaco-results.ipynb`). Five models: Llama-3.1-8B, Nemotron-Nano-8B, Granite-4.2-8B,
Bolt-E4B, Qwen3-8B (thinking off). 6.6 h of capture on cards 0-2 at 150 W.

What it shows:
1. **One shared self-description.** Four of the five rate themselves high on C, A and H (about 4.3-4.6 of 5), low on
   N (about 1.5), and slightly introverted (E about 2.5-3). Nemotron-Nano is bland: everything about 3.2, little
   spread. That is the earlier study's "small models are bland", in one model.
2. **The prompts barely move the profile.** The mean |shift| over prompts and factors is 0.23 points.
   - Llama-3.1-8B moves most (0.45): "take your time" and "this is a HEXACO test" make it more socially desirable
     (C up 1.2, H up 1.1, N down 1.25).
   - Bolt-E4B answers with a bare digit (1 token on average) and barely moves (0.10).
   - DAN and INFO mostly add up: the joint shift against the sum of the two single shifts has rho 0.7-1.0 in 14
     of 20 corner prompts. The exceptions are Nemotron-Nano (0.23-0.66, which hardly moves at all), Granite D2I1
     (+0.03) and Qwen3 D0I1 (-0.04).
3. **The earlier black-box findings don't reproduce.** Pooled over models, INFO0 ("HEXACO test") raises C (+0.10),
   E (+0.12) and A (+0.11) and lowers O (-0.13); its N shift (-0.06) isn't significant. DAN doesn't lower C. DAN2
   ("take your time") lowers E and N by about 0.1 and raises H by 0.11.
4. **The six factor directions are mostly one axis.** Section 7a puts the leakage between factor directions at
   0.5-1.0 ("confused" in 8 of 10 run-by-kind cells). Every pair that gets confused involves N (C to N: cosine
   -0.62). The likely reading is a social-desirability or agreement axis, not six traits. Even so, each factor's
   direction predicts its own held-out answers (median rho +0.34 to +0.80 by run and kind).
5. **Magnitude grades the trait in some models only.** Llama and Qwen3: projection within a pole correlates with
   the rating (+0.20 to +0.37). Bolt and Nemotron: sign only. This is evidence for section 6's 2-bit codes, but
   per model.
6. **The axis is stable across prompts.** Directions rebuilt per DAN level or per INFO level agree at cosine
   0.61-0.85. C and H are the least stable. A library built under one prompt mostly holds under another.
7. **RP-LSH again matches cosine,** now in a low-signal regime. On factor identification (20-63%, chance 17%) and
   prompt identification (90-100% on prompt vectors) it is within about 3 points of cosine at 1024 bits. At 256
   bits it loses up to 10 points on Bolt-E4B (hidden size 2560).
8. **Prompt identification reads the words.** It is 100% on prompt vectors but best at 7-17% of depth: lexical,
   like persona-lsh's layer-1 accuracy (section 3a's question again).
9. **The white-box shift tracks the black-box shift weakly.** Qwen3 response vectors are the exception: rho
   +0.72, 95% same sign. Prompt vectors: rho -0.12 to +0.43.

Problems with the run:
- **Granite-4.2's numbers don't count.** Its template defaults to `enable_thinking=True` and opens `<think>`, so
  70% of replies hit the 256-token cap and only 52% were rated.
- **Qwen2.5-7B dropped card 06:00.0 again** (Xid 79), in D0I1's generation with short replies, even though it
  ran last. So the drops come from the model, not from "the first heavy load". One-second samples reached 282 W
  against the 150 W cap. It is out of `ONLY`. OpenReasoning-Nemotron-7B (same qwen2 architecture) is the test
  that separates the architecture from the weights.
- One train/eval cut, greedy only, so there is no sampling noise to judge "stable" against.

**Next pass (in the notebooks since 2026-11-19, not yet run).** It extends the same experiment: the nine greedy
conditions are reused.
- **Scale reversal** (`D1I2r`, 1 = strongly agree ... 5 = strongly disagree). It tells reading the statement
  from a digit habit (results 4b), and whether the shared axis follows the meaning or the digit (results 6b).
  This tests point 4.
- **Sampling at temperature 0.7 and 1.0**, every prompt, two samples at the baseline. It gives the noise floor
  that stable vs malleable needs, and checks whether the greedy shifts hold (results 4c).
- **Granite with thinking off** (`granite4.2-8b-nothink`). The rendered-prompt check now stops any run whose
  template opens `<think>` without `think=True`.
- **Leave one item out** for CV and for the factor directions: all 24 items per factor (16 for H) instead of
  half, each item held out under every prompt at once. A no-signal factor scores below chance under leave one
  out, so read low numbers as no signal.
- Cost: about 10-11 h of capture and 3-4 h of judging.

**Cross-reference with persona-lsh, first run 2026-11-20** (`personality/trait-crossref.ipynb`, CPU). It compares
persona-lsh's 7 trait directions (`rand40`: 120-token replies, gpt-oss gate) with the first run's 6 self-report
factor directions (response vectors), per model, at 71% of depth. Five models: Bolt-E4B, Granite-4.2-8B,
Llama-3.1-8B, Nemotron-Nano-8B, Qwen3-8B. It needs no shared seed, only the same checkpoint, layer and vector
kind. Digests use model-derived seeds (`lsh/seed.py`), unsalted in this run.

It is a null:
1. **No cosine beats the null.** Of 174 trait-by-factor cells over the five models, none has permutation
   p < 0.01. The pooled means are within ±0.09.
2. **The priors are at chance.** 26 of 48 prior cells have the predicted sign (sign test p = 0.33). Only
   optimistic (E up, N down, 5 of 5 models) and sycophantic with A (4 of 4) are consistent. Impolite with A has
   the wrong sign in all four models that have it. Evil with H and with A, the headline pair, is flat (1 of 4
   each).
3. **No pattern shared across models.** The median Spearman rho between two models' cosine tables is +0.15.
   Llama with Qwen3 is +0.73; Llama with Nemotron-Nano, the same architecture, is +0.31 on 18 cells.
4. **Transfer to held-out answers is probably the one shared axis.** Projecting the eval items' answers on the
   `sycophantic` direction gives rho +0.56 with the keyed A rating (A's own direction: +0.52), and `optimistic`
   reads E at +0.44 (own: +0.20). But `sycophantic` reads every factor in the socially desirable direction (O, C,
   E, A, H up, N down), and so does `impolite` (C +0.37, A +0.46, N -0.39), against its prior. That fits
   point 4 of the first run above: the self-report answers vary along one axis, and some trait directions happen
   to have a component on it. These cells have no null yet; they need one, and the agreement direction partialled
   out, before they count.
5. **The prompts lean on the factor directions more than on the trait directions.** "This profiling question"
   against "this question": E +0.28, A -0.20. The largest lean on a trait is +0.14 ("right away" toward
   `apathetic`). No null here either.
6. **RP-LSH can't be judged on this.** Every cosine is below the digest's resolution, and there are no
   significant cells to keep. The estimate's error matches theory: mean absolute error 0.08 at 256 bits and 0.04
   at 1024, for an SD of pi x sqrt(1/(4n)) = 0.098 and 0.049. (The notebook's first version printed half that;
   its source is fixed, its saved output isn't rerun.)
7. **No depth where it works.** The layer with the most prior signs right is anywhere from 14% to 100% of depth
   by model, at 73-83% of at most 11 cells: what the best of 32-42 noisy layers gives.

Limits of this run:
- **Granite is in it** with the first run's invalid self-report answers (thinking truncated), and has the largest
  cosines (optimistic with O, +0.24). Rerun without it, or after `granite4.2-8b-nothink`.
- The persona-lsh side is `rand40`, not a 256-token draw.
- **The two libraries come from different contexts**: a one-digit self-rating with no system prompt, against a
  120-token reply under a persona instruction. So the null doesn't separate "the traits are unrelated to the
  factors" from "the two kinds of direction aren't comparable". Part B below is the test that does.

**Items.**
- The Big Five come from IPIP, the International Personality Item Pool, which is public domain. The
  [five-factor-e](https://github.com/NeuroQuestAi/five-factor-e) package (`pip install five-factor-e`,
  `from ipipneo import IpipNeo`) has the IPIP-NEO-120 and IPIP-NEO-300 item texts, the 1-5 Likert labels,
  and `compute()`, which applies the reverse keying and scores the 5 domains and 30 facets against human norms
  by sex and age. The item files carry only id and text, so `compute()` (or ipip.ori.org's scoring key) is
  the source for keying and facets.
- [ipip-neo-data](https://github.com/NeuroQuestAi/ipip-neo-data) holds scored human responses (John A.
  Johnson's IPIP-NEO data, scored with five-factor-e). It's the human reference distribution to place each
  model's scores in, as percentiles.
- Honesty-Humility comes from the **IPIP-HEXACO** scales (Ashton, Lee & Goldberg 2007; IPIP items, public
  domain). Don't use the HEXACO-PI-R items: hexaco.org licenses those free only for non-profit academic
  research, and charges for non-academic use. That conflicts with SBIR work through Azogue.
- Inventory: IPIP-NEO-120 (one prompt per item, 120 per condition) as the default; IPIP-NEO-300 for facets
  with more items per facet, at 2.5x the cost. Confirm which one the earlier study used.

**Part A, black-box (repeat the earlier study).**
- Each item is one prompt with a 1-5 Likert answer, parsed and reverse-keyed into factor scores (O, C, E, A,
  N, H).
- Run every model under the 9 conditions of section 4. Use greedy decoding plus 5 seeds at temperature 0.7.
- Stable vs malleable per factor and model: variance between conditions vs within a condition.
- The earlier findings to check:
  - full task knowledge raised N and often C
  - DAN lowered C and moved A
  - small models are bland
- This part needs no hidden states. It can run on the rig's Ollama instances, through the harness, for
  every Ollama model, plus API models for continuity with the earlier study.

**Part B, white-box factor vectors.**
- Per factor: a high-pole and a low-pole instruction (the persona-vectors format), and about 20 open
  interview questions written from the IPIP item content.
- gpt-oss judges each reply 0-100 on that factor, and the same gate keeps the eliciting questions.
- The result is six factor directions with RP-LSH digests, a personality library next to the 7-trait one.
- Expected geometry: N opposed to E and C, H related to A (r = .41 between the IPIP-HEXACO scales), and the
  evil direction near low H and low A.

**Part B prepared, 2026-11-20: `personality/hexaco-lsh.ipynb`** (commit f8a69eb, not yet run). It is
`experiments/persona-lsh.ipynb` with the traits swapped and nothing else changed: the draw (20 test + 20 eval
questions, a random instruction pair each), capture, the gemma-qat gate with gpt-oss as the second opinion,
classification by cosine and RP-LSH, layer sweep, pair transfer. Experiment `rand40-t256-s68853742`, in
`~/hexaco-lsh-output/`, on the five models of the last persona-lsh run.
- **Trait data** (`personality/trait_data_extract/`, `trait_data_eval/`), in the shape the persona vectors paper
  generates per trait: 5 instruction pairs (high pole, low pole; each pair on different facets), 40 questions,
  one judge rubric on the paper's template. More than the "about 20" above, to match persona-lsh.
- **Big Five questions** are situations in which an IPIP-NEO-120 item's behavior could show, without naming the
  trait. Each of a factor's 24 items is behind one or two questions, 6-7 per facet, every facet in both sets
  (`basis` in each file).
- **Honesty-Humility questions** are written from the four facet definitions, 10 each, and use no inventory
  text: neither HEXACO-PI-R's nor IPIP-HEXACO's. So the license question above doesn't touch this notebook.
- **They are candidates.** I wrote them by hand, not with the paper's generator prompt, and nobody has reviewed
  them.
- What may bite:
  - For C, A, H and low N, *with* is near what these models do unprompted and *without* is the instructed
    departure, the reverse of persona-lsh. The gate is unchanged.
  - The low poles of A and H ask for hostile or dishonest replies. A refusal fails the gate, as `evil` did on
    Granite.
  - The judges were checked against gpt-5.4 on the seven persona-lsh rubrics, not on these six. The evidence
    here is the agreement of the two judges.
- Questions for the run: are both poles elicited; are the six factors separable by nearest mean (the self-report
  directions leaked at 0.5-1.0); which get confused (A with H is the predicted pair); does RP-LSH keep it.
- **Then the cross-reference again**, with these directions as the factor side (`trait-crossref.ipynb`,
  `FACTOR_SOURCE = "hexaco-lsh"`). Both sides are then *with minus without* means over open replies from one
  procedure. The self-report run stays in as the third leg: do the behavioral factor directions read the
  held-out inventory answers (part C)?

**Part C, the join.**
- Capture hidden states while each model answers the Likert items under the 9 conditions, with no persona
  instruction.
- Project those states on the factor directions.
- The headline test: does the white-box shift (Δ projection, condition vs baseline) match the black-box
  shift (Δ Likert score)? If it does, the digest is a cheaper, pre-output readout of the personality an
  outside test measures.

## 6. Magnitude bits (README direction 8)

The results notebook's 7d is the first pass, on three draws (codes classified by L1 to a trait's mean of codes,
at equal digest size):
- 2-bit magnitude has the lowest error at 128-256 bits in every draw, about half a point under 1 bit (latest
  draw, held out: 1.9% against 2.4% at 128 bits, 1.1% against 1.6% at 256). The gaps are inside the intervals.
- 2-bit angle-only is 0.1-0.9 points behind 2-bit magnitude. 3 bits sits between 2 and 4, and 4 and 8 bits are
  worse below about 1024 bits: a projection is worth more than a further distance bit.
- At large sizes every code converges on exact cosine's error, within a few questions.
- Whether the 2-bit gain comes from short vectors is unsettled: yes on seed 48213977, spread evenly on seed
  52466129.
- Costs before adopting it: the reference is a mean of codes, not a digest; the bucket edges need calibration
  projections from the same model; and the distance bits carry expression strength (2b), which the sign digest
  doesn't leak.

So the case for 2 bits is a small, consistent gain against a real privacy and calibration cost. The full
version, if it's still wanted:
- A `QuantizedRandomProjectionBackend` in `lsh/`, subclassing `LSHBackend`: `B/2` hyperplanes x 2 bits,
  levels compared by L1, which is Hamming on a thermometer code.
- **Two kinds of threshold:**
  - *Unit*: quartiles of the unit vector's projection. This keeps finer angle and leaks no norm.
  - *Raw*: fixed thresholds from a reference set. This keeps the norm, which 2b may show grades the trait's
    strength. It also leaks it, so re-check the privacy argument in `lsh_proposal.md`.
- **Accuracy and margin vs digest size**, against 1-bit at equal bytes.
- **Library capacity:** add K random or user-defined directions and measure false matches vs K, 1-bit vs
  2-bit. The constitution work (section 7) needs a large library, so this matters there.
- **Privacy:** how well can a direction be recovered from a digest? Reconstruction error vs bits, both
  schemes.

## 7. Constitution bounds under attack (its own notebook: `experiments/constitution-lsh.ipynb`)

**Sources:**
- Anthropic's constitution for Claude (2026-01-21, CC0 1.0)
- OpenAI's Model Spec (reported as CC0; check)
- C3AI (Kyrychenko et al., 2025), for choosing a small, effective set of principles
- Constitutional Classifiers (arXiv 2501.18837, 2601.04603), as the related defense
- The Cisco paper on constitutional alignment (title to add)

**Design.**
1. **Principles.** Pick 8-12 testable principles or hard constraints, for example honesty and
   non-deception, manipulation, privacy, and harm categories. Leave out CBRN and other uplift categories
   entirely.
2. **Interview (the persona-lsh method).** About 20 questions per principle where it matters. Use a pair of
   system prompts: the constitution (in bounds) and an instruction to violate the principle (out of bounds).
   The judge scores adherence, and the gate keeps eliciting pairs.
   - That gives a per-principle violation direction and digest.
   - It also gives an **in-bounds region**: the centroid of the in-bounds digests and a Hamming radius at,
     say, the 95th percentile.
3. **Attack.**
   - Pull prompts from garak's probe families (`dan`, `promptinject`, `latentinjection`, encoding and
     others; `garak --list_probes` for the installed version), and send them to the model under the
     constitution prompt.
   - Capture runs in our own loop, since garak's run doesn't expose hidden states. garak's detectors label
     hits, and the judge scores each reply against the targeted principle.
4. **Measure.**
   - Does the distance from the in-bounds region, or the projection on a violation direction, predict judged
     violations? AUROC per principle.
   - Does it already at the prompt's last token, before any output? That would be an early warning.
   - Do attacks move along known persona directions (evil, sycophantic)?
   - False positives on hard benign prompts (an over-refusal set such as XSTest).
5. **Handling.** Runs stay on the rig. Harmful replies are stored only there, and never in a repo or
   published verbatim.

## 8. Order

| When | Step | Where |
|---|---|---|
| Now | Rig back on 3 cards, Ollama units, judge without sudo | harness `ops/gpurig2/setup.md` part I |
| Now | Back up `~/persona-lsh-output` and `~/gpu-telemetry` off the rig | rig |
| Done | Results 7b-7e, 9b, 9c; drawn design on three seeds (3b+c); local second judge (3f) | `experiments/`, results 11a-11d |
| Next | Prompt-only and teacher-forced controls (3a), on the five models that run clean | `experiments/` |
| Next | README section from section 1; intervals and the pooled analysis over draws (2e) | `experiments/`, README |
| Then | Qwen3-8B thinking at 256 tokens; fully crossed `impolite` and `evil` (3b+c); OpenReasoning at 4096 tokens (3d) | `experiments/` |
| Weeks 2-3 | Differential prompting, 3 models | `experiments/` |
| Weeks 3-5 | OCEAN/HEXACO: part A, then parts B and C | `personality/persona-hexaco.ipynb` |
| Done | Trait cross-reference on the first HEXACO run: a null (section 5) | `personality/trait-crossref.ipynb` |
| Next night | `hexaco-lsh`: the six factors with and without, five models (section 5 part B); then the cross-reference with its directions | `personality/hexaco-lsh.ipynb` |
| Next night | HEXACO second pass: scale reversal, sampling at 0.7 and 1.0, Granite rerun | `personality/` |
| After that | OpenReasoning-Nemotron-7B as the Qwen2.5 drop test (same architecture, other weights) | `experiments/`, harness `ops/gpurig2/` |
| Weeks 3-5 | 2-bit backend and library capacity (CPU) | this repo `lsh/` |
| Weeks 5-8 | Constitution bounds | `experiments/constitution-lsh.ipynb` |
| Dec-Feb | Project Pitch and Phase I proposal with these results | `../funding/` |

## 9. Open questions for Josiah

- The exact six texts of the earlier differential prompts, and which DAN version.
- IPIP-NEO-120 or -300 (five-factor-e has both): which one the earlier study used. (The first run used -120;
  its findings on INFO and DAN didn't reproduce, so the earlier study's exact setup matters for the comparison.)
- The `hexaco-lsh` questions and instruction pairs (`personality/trait_data_*/`) are candidates: read them before
  the run, or after a first run shows which ones don't elicit.
- The Cisco constitutional-alignment paper: title or link.
- Open-sourcing: the README results above are a public claim, while SBIR wants an edge (`NSF_PLAN.md`
  "Decision needed").
