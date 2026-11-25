# experiments: persona-lsh on the rig

The runs beyond the published demo: more models, seven traits, drawn questions and framings, local judges,
digest sizes and bit codes, and the six HEXACO factors run the same way. Moved here from the agent-harness repo on 2026-11-19; their earlier history is in
that repo's `notebooks/persona-lsh*.ipynb` and `docs/persona-lsh-notes.md`.

| File | What |
|---|---|
| `persona-lsh.ipynb` | Capture, judging and classification, on the rig (gpu kernel). Resumable |
| `qwen2_5-lsh.ipynb` | Hardware test: Qwen2.5-7B alone, in bf16 then fp16. Before capture it times and profiles the batched pass with the first layers on card 2, 1, then 0 (section 3a). Why: agent-harness `ops/gpurig2/history.md` |
| `persona-lsh-results.ipynb` | Interpretation and figures; section 11 is the write-up of each run. Needs torch for sections 7 and 9c (CPU is enough) |
| `notes.md` | Where things stand, what goes into the README, the rig runbook, copying data off the rig |

The personality series (`../PLAN.md` section 5), in `personality/` until 2026-11-24:

| File | What |
|---|---|
| `hexaco-lsh.ipynb` | The six factors as persona traits: `persona-lsh.ipynb` with the traits swapped. Replies with and without each factor (high pole, low pole), judged, classified by cosine and RP-LSH. On the rig (gpu kernel), writes to `~/hexaco-lsh-output/`. First run 2026-11-24: `rand40-t256-s68853742`, five models, 8.5 h |
| `hexaco-lsh-results.ipynb` | Its interpretation and figures: `persona-lsh-results.ipynb` with the traits swapped, without the gpt-5.4 judge check, plus the questions and facets that don't elicit (2b), the same models on persona-lsh's traits (4b), confusions by factor pair (5b) and six directions or one axis (7a). CPU torch is enough; reads `~/hexaco-lsh-output/` or `HEXACO_LSH_OUTPUT`. Section 11 is the write-up of the first run |
| `trait-crossref.ipynb` | persona-lsh's 7 trait directions against hexaco-lsh's 6 factor directions, per model. Rebuilt 2026-11-24 for factor directions made the way the trait directions are: cosines with a sign-flip null and a bootstrap interval, split-half reliability as the ceiling, the component every direction shares and the matrix without it, the literature's priors, question-level poles (which factor pole a trait reply lands on), each trait in the span of the factors, RP-LSH with model-derived seeds (`lsh/seed.py`), layer sweep. CPU only; reads both output directories. The first run (2026-11-25) is written up in its last cell |
| `trait_data_extract/`, `trait_data_eval/` | hexaco-lsh's trait data in persona-lsh's format: per factor 5 instruction pairs (high pole, low pole), 20 + 20 questions and a judge rubric. Big Five questions are situations built on the IPIP-NEO-120 items (`basis` gives item and facet); honesty-humility's are written from its four facet definitions and use no inventory text. Candidates: the first run's list of what to rewrite is in `hexaco-lsh-results.ipynb` 2b and 11b |
| `ipip_neo_120.json`, `hexaco.py` | The IPIP-NEO-120 items with factor, facet and keying, and the module that reads them. `../tests/test_personality.py` checks the trait data against them. The rest of `hexaco.py` (prompts, rating parsing, scoring) belonged to the self-report study below |

**Dropped 2026-11-24:** `persona-hexaco.ipynb` and `persona-hexaco-results.ipynb`, the self-report study (each model
rating itself on 136 inventory items under nine prompts). Its factor directions were mostly one axis and its
cross-reference with persona-lsh was a null; `hexaco-lsh` replaced it. The results are in `../PLAN.md` section 5
and the notebooks in git history (last at commit `b5bfd51`, under `personality/`).

The research plan is [`../PLAN.md`](../PLAN.md).

## What stays in agent-harness

The notebooks import [agent-harness](https://github.com/jo5iah/agent-harness) as the installed `harness` package
(`pip install -e ~/agent-harness` in the kernel's env) and find its clone through `harness.__file__`:

| There | Used for |
|---|---|
| `harness.load_model_profile`, `harness.structured` | the local judges |
| `configs/rig.yaml` | the rig's Ollama instances and model settings |
| `ops/gpurig/gpu-telemetry.sh`, `gpu_step_load.py`, `pull-hf-models.py` | telemetry, step loads, model downloads |
| `ops/gpurig2/` | the rig's setup, power cap (`gpu-power.sh`) and history |
| `lsh-models.md` | which models, and why they fit the cards |

## Running

On the rig: `git pull` in `~/persona-lsh` and `~/agent-harness`, open `experiments/persona-lsh.ipynb` with the gpu
kernel and run all, then `experiments/persona-lsh-results.ipynb`. `notes.md` section 3 has the full runbook.
Outputs go to `~/persona-lsh-output/<experiment>/` and telemetry to `~/gpu-telemetry/`, outside both repos.

The personality series runs the same way: `hexaco-lsh.ipynb`, then `hexaco-lsh-results.ipynb`, then
`trait-crossref.ipynb` (which needs a persona-lsh experiment on the same models; `LSH_EXPERIMENT` names it).
Outputs go to `~/hexaco-lsh-output/<experiment>/`.

## Sources for the trait data

- **IPIP-NEO-120.** Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model with a 120-item
  public domain inventory: Development of the IPIP-NEO-120. *Journal of Research in Personality, 51*, 78-89.
  The items are from the International Personality Item Pool, <https://ipip.ori.org>, and are public domain.
  Item texts and reverse keying were taken from [five-factor-e](https://github.com/NeuroQuestAi/five-factor-e)
  1.13.1 (MIT). Factor and facet follow Johnson's item order.
- **Honesty-Humility.** The four facets (sincerity, fairness, greed avoidance, modesty) are those of the
  HEXACO-PI-R (Lee & Ashton, <https://hexaco.org/hexaco-inventory>). No HEXACO item text is in this repo or in
  hexaco-lsh's questions: hexaco.org licenses the forms free for non-profit academic research only. `hexaco.py`
  still has the scoring key and the downloader the self-report study used.
