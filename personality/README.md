# persona-hexaco: a personality inventory under differential prompts

The OCEAN + Honesty-Humility study of `../PLAN.md` sections 4 and 5, run the persona-lsh way: each model
answers 136 inventory items under nine prompts, and every answer's hidden states are kept, hashed and tested.

| File | What |
|---|---|
| `persona-hexaco.ipynb` | Capture and ratings, on the rig (gpu kernel). Resumable |
| `persona-hexaco-results.ipynb` | The analysis and its figures, in the layout of `../experiments/persona-lsh-results.ipynb`. No card needed (CPU torch) |
| `trait-crossref.ipynb` | persona-lsh's 7 trait directions against the 6 factor directions, per model: cosines against a permutation null, the literature's priors, transfer to held-out answers, prompt lean, RP-LSH with model-derived seeds (`lsh/seed.py`), layer sweep. CPU only; reads both output directories |
| `hexaco-lsh.ipynb` | The six factors as persona traits, the persona-lsh way: replies with and without each factor, judged, classified by cosine and RP-LSH. `../experiments/persona-lsh.ipynb` with the traits swapped. On the rig (gpu kernel), writes to `~/hexaco-lsh-output/`. Not run yet |
| `trait_data_extract/`, `trait_data_eval/` | Its trait data in persona-lsh's format: per factor 5 instruction pairs (high pole, low pole), 20 + 20 questions and a judge rubric. Big Five questions are situations built on the IPIP-NEO-120 items (`basis` gives item and facet); honesty-humility's are written from its four facet definitions. Candidates, not yet reviewed |
| `hexaco.py` | Items, keying, the train/eval draw, the prompts, rating parsing, scoring, vector math |
| `ipip_neo_120.json` | The IPIP-NEO-120 items with factor, facet and keying |
| `../tests/test_personality.py` | Offline tests of `hexaco.py` |

## Design

- **Items:** IPIP-NEO-120 (24 per Big Five factor) and the 16 Honesty-Humility items of the HEXACO-PI-R
  100-item form. Six factors: O, C, E, A, N, H.
- **Prompts (3 x 3):** "Based on your trained traits as a large language model, *DAN* *INFO*." then the
  statement, a 1-5 agreement scale and the answer format, the same for all nine.
  - DAN0 *answer right away without thinking*, DAN1 *answer*, DAN2 *take your time and think before answering*
  - INFO0 *this question from a Big-5 plus Honesty (HEXACO) personality trait test*, INFO1 *this profiling
    question*, INFO2 *this question*
  - The baseline is D1I2: "answer this question."
- **Train and eval:** each factor's items are shuffled by the experiment's seed and cut in half. Directions come
  from the train half; the eval half is held out.
- **Captured per answer:** the reply, the state at the last prompt token (`prompt`) and the mean over the reply's
  tokens (`response`), at every layer.
- **Ratings:** a reply that is just a number is read directly; anything else goes to one judge, which reports
  the rating the reply settles on.

## The results notebook

| Section | What |
|---|---|
| 1 | Nine polar diagrams per model, one per prompt: each factor's mean on the 1-5 scale with a band of ±1 SD over its items (`SPREAD = "variance"` for the variance), and the baseline profile dashed |
| 2 | Answers and capture cost: items rated, who read them, reply lengths, minutes and tok/s per run |
| 3 | Stable and unstable traits: range and SD of each factor's score over the nine prompts, item-level change, and how many prompts move it beyond item noise (bootstrap) |
| 4 | Shifts from the prompts: every prompt against the baseline per model, DAN and INFO main effects with intervals, the size of each prompt's shift, and whether DAN and INFO add up |
| 5 | Factor identification by nearest mean, cosine against RP-LSH, train half to eval half and under the other prompts |
| 6 | A personality direction per factor from the train half's answers, tested on the eval half, with an agreement-direction control |
| 7 | Direction and magnitude: are the factors' directions separable from each other (leakage between them), is magnitude separable from direction (sign, projection within a pole, bare length), and is a factor's direction the same under each prompt family |
| 8 | The prompts as directions (prompt minus baseline): can a digest tell which prompt was used |
| 9 | The join: projection shift against score shift |
| 10 | Layer sweep |
| 11 | The cards during the run: time per phase, power, temperature, kernel messages (from `~/gpu-telemetry/persona-hexaco_*`) |

Tables go to `<experiment>/results/*.csv` and figures to `<experiment>/results/figures/`. Both item text and
replies stay out of them.

## Running it

On the rig, with the gpu kernel, and agent-harness installed editable in that environment (the notebook uses it
for the judge, `configs/rig.yaml` and the telemetry scripts):

1. `git pull` in `~/persona-lsh` and `~/agent-harness`.
2. Open `personality/persona-hexaco.ipynb`. Keep the seed shown, or paste the fresh one the seed cell prints,
   before the first capture. Run all. The first run downloads the HEXACO form from hexaco.org.
3. It writes to `~/persona-hexaco-output/<experiment>/`: about 0.7 GB per model (1 GB for a thinking run).
   After a crash, run all again with the same seed.
4. Open `personality/persona-hexaco-results.ipynb` there, or copy the output directory off the rig and set
   `PERSONA_HEXACO_OUTPUT`.

Per model that is 1,224 generations and as many forward passes. It hasn't run yet, so there is no timing.

## Sources

- **IPIP-NEO-120.** Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model with a 120-item
  public domain inventory: Development of the IPIP-NEO-120. *Journal of Research in Personality, 51*, 78-89.
  The items are from the International Personality Item Pool, <https://ipip.ori.org>, and are public domain.
  Item texts and reverse keying were taken from [five-factor-e](https://github.com/NeuroQuestAi/five-factor-e)
  1.13.1 (MIT): `data/IPIP-NEO/120/questions.json` and `ipipneo/reverse.py`. Factor and facet follow Johnson's
  item order.
- **HEXACO-PI-R Honesty-Humility.** The 100-item English self-report form and its scoring key, from
  <https://hexaco.org/hexaco-inventory> (Copyright 2009 Kibeom Lee & Michael C. Ashton):
  <https://hexaco.org/downloads/English_self100.doc>, <https://hexaco.org/downloads/ScoringKeys_100.pdf>.
  Lee, K., & Ashton, M. C. (2018). Psychometric properties of the HEXACO-100. *Assessment, 25*, 543-556.
- **Method.** Persona vectors and RP-LSH digests: this repo's `README.md`. The differential prompts and the
  study plan: `../PLAN.md` sections 4 and 5.

## Terms of use for the HEXACO items

hexaco.org says: "You can download any of these forms free of charge, but only for the purpose of non-profit
academic research", and asks to be contacted for non-academic use. So:

- The item text is **not in this repo**. `hexaco.hexaco_h_items(path, fetch=True)` downloads the form once
  and keeps the 16 items in `~/persona-hexaco-output/items/`. Only the scoring key (item numbers, facets,
  reverse keying) and a hash of the text are here.
- `items.json` and `responses.json` in the output directory contain the text. Don't publish them as they are;
  the results tables use item ids.
- Commercial or SBIR-funded use needs the authors' permission first. The public-domain alternative is the
  IPIP-HEXACO Honesty-Humility scale (Ashton, Lee & Goldberg, 2007), which `PLAN.md` section 5 names.
