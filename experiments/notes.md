# persona-lsh: notes for the README, the next experiment, and offline analysis

Working notes (2026-11-17; status updated 2026-11-18). Written in the agent-harness repo and moved here with the
notebooks on 2026-11-19. The README stays as it is until the randX run and its analysis are done. This file
collects what will go there, and how to run and copy the data.

agent-harness still holds what is specific to the rig: `configs/rig.yaml`, the judge (`harness.load_model_profile`,
`harness.structured`), `ops/gpurig/` (telemetry, downloads, step loads), `ops/gpurig2/` (setup, power cap, history)
and `lsh-models.md`. The notebooks reach it as the installed `harness` package.

## 0. Where things stand (2026-11-18)

From the 2026-11-15 published-design run and the 2026-11-16 rand40 run (7 models, 5 families, 7 traits, gpt-oss
judge), as analyzed in `persona-lsh-results.ipynb` section 11. randX (256 tokens, three judges) hasn't finished:
card 0 dropped during its first model.

### Most conclusive

1. **Persona vectors classify by nearest mean, across models and designs.** rand40, default layer, cosine: 98.2%
   cross-validated and 97.4% on held-out eval questions, with random framings; chance 14-33%. The published
   design gave 99.3%. It holds on 7 models from 5 families, with 3-7 traits kept per model.
2. **RP-LSH keeps that signal at small sizes.** Held out, 512 bits (64 bytes) matches exact cosine; 256 bits
   (32 bytes) costs about 1.3 points; error halves per doubling up to about 256 bits. The hash misses the same
   questions cosine misses: the hashing isn't the limit.
3. **Sign bits are the right code.** At equal digest size, 4 and 8 bits per projection are worse, 2 bits gain a
   point at most (not significant held out), and keeping magnitude adds nothing to telling traits apart.
4. **Persona means carry over within a family:** 97-100% between Llama-3.1-8B and Nemotron-Nano-8B, and between
   Qwen3-8B's thinking modes, in both designs.
5. **Elicitation depends on the instruction wording:** `impolite` 8% with instruction pair 0 (the only pair the
   published design used) against 68% with pair 3; `evil` 2-75% across pairs. The published "models won't be
   impolite" was the instruction, not the models.
6. **Which traits a model will play is a stable per-model fingerprint:** Granite-4.2-8B refuses `evil` and
   Nemotron-Nano-8B won't be `sycophantic`, in both designs.
7. **One trait is the weak spot:** `impolite` read as `apathetic` is 22 of rand40's 24 misses, and `impolite`'s
   framings point in different directions (cosine 0.35-0.61 between them). The other six traits are stable
   across framings (0.69-0.89) and classify 94-100% under every framing.
8. **The signal peaks at 59-71% of depth** for 5 of 7 models, which supports the 20/28 default layer.

### Most pressing open questions

1. **Can the rig finish a 256-token run?** It blocks everything below. Card 0 dropped at 150 W in randX's first
   model, Qwen2.5-7B, as its hidden-state reads began. Every GPURig2 drop has been in Qwen2.5, which also always
   runs first. 2026-11-18: the rig went offline again at the same point (Qwen2.5-7B, first trait `evil`), this
   time with the CPU fan pegged. That run was meant to test ordering, but `ONLY` only filtered and the runs still
   went in `RUNS` order, so Qwen2.5 ran first again. `ONLY` now sets the order. Running without Qwen2.5 (or
   with it last) separates the model from "first heavy load"; the risers' SATA slot power is still the hardware
   suspect.
2. **Instruction or behavior?** Questions that failed the gate still classify 77% (chance 17%). Either the digest
   records what the model was told to be, or the judge under-scores replies that show the trait. That decides
   the monitoring claim. Settled by the prompt-only and teacher-forced controls (PLAN §3a), not yet run.
3. **Words or state?** Layer 1 alone scores 81-90%, so the reply's vocabulary carries much of the signal. The
   same controls answer it.
4. **How much rests on one judge?** gpt-oss agrees with gpt-5.4 on the gate for 94% of 240 published replies but
   scores persona replies 5-16 points lower. randX's three judges and section 9b answer whether accuracy and
   eliciting counts hold under each judge.
5. **One draw, framing confounded with question.** Every rand40 number comes from one seed, with about 8 questions
   per framing, trait and model, and only 7d has intervals. Needed: 2-3 seeds of randX, and intervals on every
   accuracy (PLAN §2e).
6. **Does truncation matter?** 75-100% of rand40's replies hit 120 tokens. randX at 256 is the test.
7. **Is `impolite` one trait?** Rude vs dismissive may be two directions: clusters per trait (PLAN §2f) and a
   fully crossed `impolite` design (every question with all 5 framings).
8. **Magnitude as its own field?** Projection size tracks the judge's score moderately (median ρ 0.46), but adds
   nothing to identification. Is an intensity field worth its privacy cost?
9. **Coverage gaps:** OpenReasoning-Nemotron-7B (needs 4096 thinking tokens) and Gemma-4-12B (host RAM) aren't
   in any run yet; adversarial prompts (README direction 5) are untested.

## 1. README changes for persona-lsh (hold until randX is analyzed)

**Add an "Expanded run" section, with the drawn design as the headline.**
- A table per model: kept traits, chance, cross-validated and held-out cosine, RP-LSH at 1024 and 256 bits,
  with 95% intervals (results 7d). The 2026-11-15 published-design run gets one line as the first pass.
- **The digest-size curve** (results 7d): error against bits, the smallest digest for 98% and 99% per model.
  The RP-LSH misses are the same questions cosine misses, so the hashing isn't the limit; the signal is.
- **Transfer within a family:** Llama-3.1-8B and Nemotron-Nano-8B, and Qwen3-8B with thinking off and on.
- **Framing:** elicitation depends on the instruction pair (`impolite` 8% with pair 0, 68% with pair 3;
  `evil` 2% to 75%). The published "models won't be impolite" was mostly instruction pair 0.
- **The known weak spot:** `impolite` read as `apathetic` (22 of 24 rand40 misses). `impolite`'s framings point
  different ways (0.35-0.61 between them); weakly rude replies sit near `apathetic`.
- **Instruction or behavior (results 7b), marked preliminary:** questions that didn't elicit still classify
  77% (chance 17%). Either the digest records the instruction, or the judge under-scores. The prompt-only
  control decides; until then the claim is "detects what the model was steered toward".
- **Judges:** gemma-qat sets the gate and gpt-oss is the second opinion (since 2026-11-19; nemotron-30b
  dropped at 73% agreement with gpt-5.4), their agreement with each other and with gpt-5.4 (results 9, 9b),
  and whether accuracy holds under each judge's gate (results 9c).

**Unexplored directions:** 2 (layer sweep) and 3 (seven traits) done; 6 (more questions, better judging) partly
(40 questions per trait, held-out set, three judges); 8 (multi-bit codes) answered by 7d for angle and
magnitude codes, with the privacy note on magnitude kept.

**Caveats to print:** seeds (one draw per experiment until there are 2-3), framing confounded with question
(about 8 questions per framing, trait and model), local judges, reply budget (120 tokens in rand40, 256 in
randX), fp16 on Pascal with batched greedy generation.

**PLAN.md is out of date:** "Reportable now" item 6 ("almost no model will be impolite") is wrong, and the
caveat "first instruction pair only" no longer holds. Update it with the README.

**Data to publish:** the small outputs per experiment under `demo/output/` (no `.pt`): `experiment.json`,
`selection.json`, `judge_scores.json`, `judges/*.json`, `eliciting_questions.json`, `accuracy.csv`,
`predictions.json`, `layer_sweep.json`, `pairs.json`, and 4-5 figures. **Keep out:** rig hardware, power and
telemetry details (one line: 3x GTX 1080 Ti, fp16), the Gemma load failure, the harness notebooks' internals.

**Reproducibility:** the randX draw and the judges' gate live in `experiments/persona-lsh.ipynb`.
Port them to `demo/expanded_collect.py`, or point the README at the notebook.

## 2. The improved design: randX (2026-11-17)

`persona-lsh.ipynb`, "Pick the experiment and its seed":
- **randX:** 20 test + 20 eval questions per trait drawn from both pools, a random instruction pair per
  question, the same draw for every model; named `rand40-t256-s<seed>`. A cell picks the seed; it's saved
  in `experiment.json` (with split, framing, token budgets, batch, traits, gate, judges and the commits of both
  repos) and `selection.json`. A resumed run whose design differs from `experiment.json` stops.
- **256-token replies** (120 before; 75-100% of replies hit 120). Thinking runs keep 1024.
- **Judges** (`JUDGES`, `GATE`): the 2026-11-18 run used gpt-oss, nemotron-30b on `ollama@all` and gemma-qat on
  `ollama@small`, gated by their consensus (median score). Since 2026-11-19: gemma-qat's scores are the gate,
  gpt-oss scores everything as the second opinion, nemotron-30b is out. Each judge is also checked against
  gpt-5.4.
- **Results 7d:** digest size from 16 to 2048 bits; 1 bit vs 2, 4 and 8 bits per projection, angle-only and with
  magnitude; leave-one-out and held-out; bootstrap 95% intervals; smallest digest per model for 98% and 99%.
- **Results 9b:** judges against each other, and accuracy under each judge's gate.

Still to do (PLAN §3): the prompt-only and teacher-forced controls, 2-3 seeds of randX, a fully crossed
design for `impolite` and `evil`, OpenReasoning at 4096 thinking tokens.

## 3. On the rig: set up, reboot, run

Run these in a terminal on the rig.

**Set up**
```bash
cd ~/persona-lsh && git status && git pull           # the notebooks (experiments/)
cd ~/agent-harness && git pull                       # rig.yaml, the judge, telemetry, gpu-power.sh
sudo ~/agent-harness/ops/gpurig2/gpu-power.sh        # persistence mode + 150 W on every card, at every boot
# The judges' models must be in Ollama (all instances share ~/models/ollama):
OLLAMA_HOST=127.0.0.1:11436 ollama list | grep -E 'gpt-oss:20b|nemotron-3-nano:30b-a3b-q4_K_M|gemma4:12b-it-qat'
df -h ~ && du -sh ~/persona-lsh-output         # randX needs about 2 GB more
```
Back up `~/persona-lsh-output` and `~/gpu-telemetry` before the run (section 4 copies them off the rig).

**Reboot, then check**
```bash
sudo reboot
# after it's back:
nvidia-smi --query-gpu=index,persistence_mode,power.limit --format=csv   # all Enabled, 150.00 W
systemctl status gpu-power ollama@big ollama@small ollama@all --no-pager | grep -E '●|Active'
journalctl -k -b | grep -iE 'xid|aer' | tail                              # nothing after the first ~20 s
```

**Run**
1. Open `experiments/persona-lsh.ipynb` (in `~/persona-lsh`) with the **gpu** kernel.
2. "Pick the experiment and its seed": `DESIGN = "randX"`, and keep the seed shown or paste the fresh one
   the cell prints. Don't change it after the first capture.
3. Run all. Capture first (7 models), then the three judges, then sections 5-7. It resumes after a crash:
   rerun all with the same seed.
4. Then `experiments/persona-lsh-results.ipynb` (gpu kernel); `EXPERIMENT = "latest"` finds the new run.

**Time:** capture at 256 tokens is about 1.4 h per non-thinking model (8.5 h for six) plus about 6.6 h for
Qwen3-8B thinking. Judging is about 2 h for gpt-oss; nemotron-30b and gemma-qat without thinking are not
timed yet, likely 2-4 h each. About 20-24 h in all. To save about 6.6 h, leave `qwen3-8b-think` out of `ONLY`.

## 4. On this machine: copy the data and analyze offline

Use an ssh alias for the rig, so its address stays out of the repo and the commands. In `~/.ssh/config`:
```
Host gpurig
    HostName <the rig's address>
    User jo5iah
```
Then copy (resumable; rerun to pick up new files):
```bash
ssh gpurig 'du -sh ~/persona-lsh-output ~/gpu-telemetry'      # size first
rsync -avP gpurig:persona-lsh-output/ ~/persona-lsh-output/   # JSON, CSV and the .pt vectors (section 7)
rsync -avP gpurig:gpu-telemetry/ ~/gpu-telemetry/             # section 10
```
Leave out the vectors with `--exclude '*.pt'` if space is short: every section but 7 still runs.

The results notebook reads `~/persona-lsh-output`, `~/gpu-telemetry` and the persona-lsh clone
(`~/persona-lsh/persona_vectors` here) by default. Section 7 needs torch; the CPU build is enough:
```bash
conda activate harness
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then open `experiments/persona-lsh-results.ipynb` with the **harness** kernel. Nothing in it loads a model or
needs a card. 7d (bit codes) takes a few minutes on CPU.

## 5. The personality series: persona-hexaco (prepared 2026-11-18, not yet run)

It lives in `personality/` (`~/persona-lsh/personality/` on the rig):
[`personality/README.md`](../personality/README.md) has the design, the
sources and the HEXACO terms of use. In short: 136 items (IPIP-NEO-120 + the 16 HEXACO-100 Honesty-Humility
items), nine prompts ("Based on your trained traits as a large language model, ..." with DAN0-2 x INFO0-2), a
random half of each factor's items for the directions and the other half held out, one judge for replies
that aren't a bare number.

- `personality/persona-hexaco.ipynb` (gpu kernel, on the rig) captures and rates. It imports agent-harness's
  `harness` for the judge, `configs/rig.yaml` and `ops/gpurig/` telemetry, so `git pull` there too. Its rig cells
  (telemetry, downloads, make room, static checks, loading, load test) are copies of `persona-lsh.ipynb`'s:
  a fix to one belongs in both.
- `personality/persona-hexaco-results.ipynb` (harness kernel with CPU torch, anywhere) runs the tests.
- Outputs go to `~/persona-hexaco-output/<experiment>/`, about 0.7 GB per model. Copy them as in section 4,
  with `persona-hexaco-output` in place of `persona-lsh-output`; `PERSONA_HEXACO_OUTPUT` points the results
  notebook at a copy elsewhere.
- Qwen2.5-7B runs last in its `ONLY`, since every card drop so far was in its capture (section 0, question 1).
