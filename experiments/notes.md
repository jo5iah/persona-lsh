# persona-lsh: notes for the README, the next experiment, and offline analysis

Working notes (2026-11-17; status updated 2026-11-20). Written in the agent-harness repo and moved here with the
notebooks on 2026-11-19. The README stays as it is until the randX runs cover the missing models (section 0, open question 3). This file
collects what will go there, and how to run and copy the data.

agent-harness still holds what is specific to the rig: `configs/rig.yaml`, the judge (`harness.load_model_profile`,
`harness.structured`), `ops/gpurig/` (telemetry, downloads, step loads), `ops/gpurig2/` (setup, power cap, history)
and `lsh-models.md`. The notebooks reach it as the installed `harness` package.

## 0. Where things stand (2026-11-20)

Four experiments, each written up in `persona-lsh-results.ipynb` section 11:

| | Experiment | Date | Runs | Replies | Gate |
|---|---|---|---|---|---|
| 11d | published design | 2026-11-15 | 8 (7 models) | 120 tokens | gpt-oss |
| 11c | `rand40` | 2026-11-16 | 7 | 120 | gpt-oss |
| 11b | randX `rand40-t256-s48213977` | 2026-11-18 | 4 (card 0 dropped after the fourth) | 256 | median of gpt-oss, nemotron-30b, gemma-qat |
| 11a | randX `rand40-t256-s52466129` | 2026-11-19 | 5 (Llama, Granite, Bolt, Nemotron-Nano, Qwen3-8B thinking off) | 256 | gemma-qat, gpt-oss as second opinion |

The 2026-11-19 run finished capture and judging in one 9.6-hour session with no card drop. Still without
256-token data: Qwen2.5-7B, OpenReasoning-7B, Qwen3-8B thinking, Gemma-4-12B.

### Most conclusive

1. **Persona vectors classify by nearest mean, across models, designs and three draws.** Default layer, cosine,
   cross-validated / held out: rand40 98.2% / 97.4%, seed 48213977 95.9% / 99.1%, seed 52466129 98.1% / 99.3%;
   chance 14-33%. The published design gave 99.3%. 7 models from 5 families, 3-7 traits kept per model.
2. **RP-LSH keeps that signal at small sizes, and its floor is exact cosine.** Held out, 512 bits (64 bytes) is
   level with cosine in rand40, 0.2 points above on seed 52466129 and 1.5 above on seed 48213977. The excess
   then shrinks as bits^-0.5 (11b), and the hash misses the same questions cosine misses.
3. **Sign bits are close to the right code.** 2-bit magnitude had the lowest error at 128-256 bits in all three
   draws, by about half a point, inside the intervals. 3, 4 and 8 bits per projection are worse at useful
   sizes. The two magnitude-strata readings of 11b (the 2-bit gain is in the short vectors; long vectors hold
   the last misses) didn't repeat in 11a.
4. **The result doesn't rest on one judge** (11a, results 9c). Leave-one-out / held out: gemma-qat's gate 98.7% /
   99.3%, gpt-oss's 98.8% / 99.2%. The two agree on the gate for 90% of 1,400 question pairs (κ 0.80), and the
   questions only one of them passes still classify 92-93%. nemotron-30b is out (73% agreement with gpt-5.4).
5. **Persona means carry over within a family** for the traits both models express: Llama-3.1-8B ↔
   Nemotron-Nano-8B 96-100% on 3-4 shared traits in three experiments, 88-90% when three thin traits were kept
   (11b). Qwen3-8B thinking off ↔ on 97-100% (11c, 11d).
6. **Elicitation depends on the instruction wording, the same way in three draws:** `impolite` elicits most
   with pair 3, `evil` most with pair 0 and least with pair 2 (`impolite` 8% with pair 0 against 68% with pair 3
   in rand40). The published "models won't be impolite" was the instruction, not the models.
7. **Which traits a model will play is a stable per-model fingerprint.** Granite-4.2-8B gives no `evil` and
   Nemotron-Nano-8B almost no `sycophantic` in every experiment, and under the same judge the eliciting counts
   per model repeat across seeds within 13 of 280 (gemma-qat: 201 / 99 / 146 / 239 against 214 / 96 / 150 / 238).
8. **One trait is the weak spot:** `impolite` is 87-88% correct against 93-100% for the others, 0% under
   instruction pair 4 in both randX draws, and its framings point in different directions (cosine 0.46-0.61
   between them, 0.69-0.90 for the other six traits).
9. **The signal peaks at 59-71% of depth** for most models (66-69% for four of five in 11a; Granite late, at
   88%), which supports the 20/28 default layer.
10. **A 256-token run can finish on the rig**, and 256 tokens changed the gate only for Granite (11b). 56-96%
    of replies still hit the cap.

### Most pressing open questions

1. **Instruction or behavior?** Questions that failed the gate still classify 65-77% (chance 17%). 11a adds a
   gradient: 99% where both judges see the trait, 92-93% where one does, 57% where neither does. So the digest
   follows the behavior in part and the instruction in part. Settled by the prompt-only and teacher-forced
   controls (PLAN §3a), not yet run.
2. **Words or state?** Layer 1 alone scores 80-96% (11a), so the reply's vocabulary carries much of the signal.
   The same controls answer it.
3. **The missing runs at 256 tokens:** Qwen2.5-7B, Qwen3-8B thinking (needed for the thinking off ↔ on transfer),
   OpenReasoning-Nemotron-7B (needs 4096 thinking tokens) and Gemma-4-12B (host RAM).
4. **Is the rig sound?** 11a ran clean without Qwen2.5-7B, but 11b's drop also came without it (card 0, Xid 79,
   between models). Correctable PCIe errors from card 1 (bus 07:00.0) recur. `qwen2_5-lsh.ipynb` tests Qwen2.5
   alone; the risers' SATA slot power is still the hardware suspect.
5. **Intervals and pooling.** There are now three draws of the rand40 design, but under three different gates
   and model sets, and only 7d has intervals. Needed: every accuracy with an interval, and one pooled analysis
   over the draws under a single gate (gemma-qat scored 11a and 11b; rand40 would need judging again) (PLAN §2e).
6. **Is `impolite` one trait?** Rude vs dismissive may be two directions: clusters per trait (PLAN §2f) and a
   fully crossed `impolite` design (every question with all 5 framings).
7. **Magnitude as its own field?** Projection size tracks the judge's score moderately (median ρ 0.46-0.50 in
   three experiments), and 2-bit magnitude buys about half a point. Is that worth its privacy cost?
8. **Untested:** adversarial prompts (README direction 5).

## 1. README changes for persona-lsh (hold until randX is analyzed)

**Done 2026-11-20:** the README has the "Expanded run" section, with the digest-size curve from results 7d
(`experiments/figures/`). Not in it yet: the per-model smallest digest for 98% and 99% as a table, intervals on
the accuracy table, and the published data files below. The list is kept as the checklist.

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

**PLAN.md is current** (updated 2026-11-20): its "Reportable now" list and draft README section carry the
drawn-design results. Write the README from them.

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

## 5. The personality series: hexaco-lsh

In this directory since 2026-11-24 (`personality/` before); [`README.md`](README.md) lists the files.

- `hexaco-lsh.ipynb` (gpu kernel, on the rig) is `persona-lsh.ipynb` with the six factors as the traits. Its rig
  cells are copies of `persona-lsh.ipynb`'s: a fix to one belongs in both.
- `hexaco-lsh-results.ipynb` and `trait-crossref.ipynb` (harness kernel with CPU torch, anywhere) do the analysis.
- Outputs go to `~/hexaco-lsh-output/<experiment>/`. Copy them as in section 4, with `hexaco-lsh-output` in place
  of `persona-lsh-output`, and the telemetry from `gpu-telemetry/hexaco-lsh_*`; `HEXACO_LSH_OUTPUT` points the
  notebooks at a copy elsewhere.
- The self-report study (`persona-hexaco*.ipynb`) was dropped on 2026-11-24; `../PLAN.md` section 5 has its results.
