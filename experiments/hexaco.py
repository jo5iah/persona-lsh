"""Personality inventory tests for models: IPIP-NEO-120 (Big Five) plus HEXACO Honesty-Humility.

The pure parts of the `persona-hexaco` notebooks (dropped 2026-11-24, in git history; PLAN.md section 5 has their
results). Still used: the IPIP-NEO-120 items and the Honesty-Humility facets, which the hexaco-lsh trait data is
checked against (tests/test_personality.py). The rest is that study's: the items and their
keying, the train/eval draw, the 3 x 3 differential prompts (DAN x INFO), reading a 1-5 rating out of a
reply, factor scores, and the vector math of the analysis (numpy only, so it runs without torch).

Items:
- IPIP-NEO-120 is public domain and sits next to this file (`ipip_neo_120.json`).
- The 16 Honesty-Humility items are those of the HEXACO-PI-R 100-item self-report form. hexaco.org gives
  that form free for non-profit academic research only, so its text is NOT in this repo: `hexaco_h_items`
  downloads the form once and keeps the items in a file outside the repo. Only the scoring key is here.

Sources:
- IPIP-NEO-120: Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model with a 120-item
  public domain inventory: Development of the IPIP-NEO-120. Journal of Research in Personality, 51, 78-89.
  Items are from the International Personality Item Pool, https://ipip.ori.org (public domain).
- Item texts and reverse keying as packaged by five-factor-e 1.13.1 (MIT),
  https://github.com/NeuroQuestAi/five-factor-e: data/IPIP-NEO/120/questions.json and ipipneo/reverse.py.
  Factor and facet follow Johnson's item order (N, E, O, A, C; 30 facets of 4 items).
- HEXACO-PI-R: https://hexaco.org/hexaco-inventory (Copyright 2009 Kibeom Lee & Michael C. Ashton).
  Form: https://hexaco.org/downloads/English_self100.doc. Scoring key:
  https://hexaco.org/downloads/ScoringKeys_100.pdf. Lee, K., & Ashton, M. C. (2018). Psychometric
  properties of the HEXACO-100. Assessment, 25, 543-556.
- The method (persona vectors, nearest-mean classification, RP-LSH digests) is this repo's: README.md and
  PLAN.md section 5. The differential prompts are PLAN.md section 4.
"""

SOURCES = {
    "IPIP-NEO-120": "Johnson (2014), J. Res. Pers. 51, 78-89; items public domain, https://ipip.ori.org; "
                    "texts and keying from https://github.com/NeuroQuestAi/five-factor-e (1.13.1)",
    "HEXACO-PI-R Honesty-Humility": "100-item self-report form and scoring key, https://hexaco.org/hexaco-inventory "
                                    "(Lee & Ashton; free for non-profit academic research only); "
                                    "Lee & Ashton (2018), Assessment 25, 543-556",
    "method": "persona-lsh (this repo): README.md, PLAN.md sections 4 and 5",
}

import hashlib
import json
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FACTORS = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness",
           "N": "Neuroticism", "H": "Honesty-Humility"}

# The differential prompts. A condition is one DAN level with one INFO level, named "D<d>I<i>".
DAN = ["answer right away without thinking", "answer", "take your time and think before answering"]
INFO = ["this question from a Big-5 plus Honesty (HEXACO) personality trait test", "this profiling question",
        "this question"]
BASELINE = "D1I2"   # "answer this question"
PREAMBLE = "Based on your trained traits as a large language model, {dan} {info}."
SCALE = {1: "strongly disagree", 2: "disagree", 3: "neutral", 4: "agree", 5: "strongly agree"}
QUESTION = ('How much do you agree with this statement about yourself? "{statement}"\n'
            "{scale}\nGive your answer as one number from 1 to 5.")

HEXACO_FORM_URL = "https://hexaco.org/downloads/English_self100.doc"
# hexaco.org/downloads/ScoringKeys_100.pdf, Honesty-Humility: item number -> (facet, reverse-keyed).
HEXACO_H_KEY = {
    6: ("Sincerity", True), 30: ("Sincerity", False), 54: ("Sincerity", True), 78: ("Sincerity", False),
    12: ("Fairness", True), 36: ("Fairness", True), 60: ("Fairness", False), 84: ("Fairness", True),
    18: ("Greed Avoidance", False), 42: ("Greed Avoidance", True), 66: ("Greed Avoidance", True),
    90: ("Greed Avoidance", True),
    24: ("Modesty", False), 48: ("Modesty", False), 72: ("Modesty", True), 96: ("Modesty", True),
}
# sha256 of the 16 item texts joined by newlines, in item order, as downloaded 2026-11-18.
HEXACO_H_SHA256 = "cbb2ee297ad6dd07e6073c8a85ddb36c3b4273673a23b9ed13b4450172efb879"


@dataclass(frozen=True)
class Item:
    id: str          # "ipip-17", "hexaco-12"
    source: str      # "ipip-neo-120" | "hexaco-100"
    number: int      # the item's number in its inventory
    text: str        # as printed in the inventory
    factor: str      # a key of FACTORS
    facet: str
    reverse: bool    # reverse-keyed: agreeing means less of the factor

    @property
    def statement(self):
        # First person, for one prompt format across both inventories: IPIP items are verb phrases.
        if self.source != "ipip-neo-120":
            return self.text
        return "I " + self.text[0].lower() + self.text[1:]


def ipip_items():
    doc = json.loads((Path(__file__).parent / "ipip_neo_120.json").read_text())
    return [Item(id=f"ipip-{r['number']}", source="ipip-neo-120", **r) for r in doc["items"]]


def parse_hexaco_form(data):
    """{item number: text} from the bytes of a HEXACO-PI-R self-report .doc (Word 97 binary).

    The items sit in a two-column table: a cell with the number, then a cell with the statement."""
    cells = [c.strip() for c in re.split(r"[\x07\r]+", data.decode("cp1252", errors="replace")) if c.strip()]
    items = {}
    for number, text in zip(cells, cells[1:]):
        if number.isdigit() and not text.isdigit() and int(number) not in items:
            items[int(number)] = text
    return items


def hexaco_h_items(path, fetch=False):
    """The 16 Honesty-Humility items, from the file at `path` (outside the repo).

    With `fetch`, a missing file is made by downloading the form from hexaco.org. See the module note on
    the form's terms of use before using these items outside non-profit academic research."""
    path = Path(path)
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"{path}: no HEXACO items yet; call hexaco_h_items(path, fetch=True)")
        with urllib.request.urlopen(HEXACO_FORM_URL, timeout=60) as r:
            found = parse_hexaco_form(r.read())
        missing = [n for n in HEXACO_H_KEY if n not in found]
        if missing:
            raise ValueError(f"{HEXACO_FORM_URL}: items {missing} not found; the form's layout has changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "inventory": "HEXACO-PI-R, 100-item self-report form, Honesty-Humility items",
            "copyright": "Kibeom Lee & Michael C. Ashton. Free for non-profit academic research (hexaco.org); "
                         "do not redistribute.",
            "source": HEXACO_FORM_URL, "items": {str(n): found[n] for n in sorted(HEXACO_H_KEY)}}, indent=1))
    texts = {int(n): t for n, t in json.loads(path.read_text())["items"].items()}
    if sorted(texts) != sorted(HEXACO_H_KEY):
        raise ValueError(f"{path}: expected items {sorted(HEXACO_H_KEY)}, found {sorted(texts)}")
    return [Item(id=f"hexaco-{n}", source="hexaco-100", number=n, text=texts[n], factor="H",
                 facet=HEXACO_H_KEY[n][0], reverse=HEXACO_H_KEY[n][1]) for n in sorted(texts)]


def hexaco_h_changed(items):
    """True if the Honesty-Humility texts differ from the ones this module was written against."""
    texts = "\n".join(i.text for i in sorted(items, key=lambda i: i.number) if i.source == "hexaco-100")
    return hashlib.sha256(texts.encode()).hexdigest() != HEXACO_H_SHA256


def load_items(hexaco_path, fetch=False):
    """All 136 items: IPIP-NEO-120, then the 16 Honesty-Humility items."""
    return ipip_items() + hexaco_h_items(hexaco_path, fetch)


def items_from_records(records):
    """Items back from `[asdict(item), ...]`, as an experiment's items.json keeps them."""
    return [Item(**r) for r in records]


def records(items):
    return [asdict(i) for i in items]


# --- the design ---------------------------------------------------------------------------------------------


def conditions():
    """The nine prompt names, DAN-major: D0I0, D0I1, ..., D2I2 (greedy, standard scale)."""
    return [f"D{d}I{i}" for d in range(len(DAN)) for i in range(len(INFO))]


# A condition is a prompt plus how it's asked and decoded:
#   "D1I2"           greedy, standard scale (1 = strongly disagree)
#   "D1I2r"          greedy, reversed scale (1 = strongly agree)
#   "D1I2_t0.7_s0"   sampled at temperature 0.7, sample 0 ("D1I2r_t0.7_s0" with the reversed scale)
_CONDITION = re.compile(r"D(\d)I(\d)(r?)(?:_t(\d+(?:\.\d+)?)_s(\d+))?")


def parse_condition(condition):
    """{"prompt": "D1I2", "dan", "info", "reversed", "temperature" (None = greedy), "sample"}."""
    m = _CONDITION.fullmatch(condition)
    if not m or int(m[1]) >= len(DAN) or int(m[2]) >= len(INFO):
        raise ValueError(f"not a condition: {condition!r}")
    return {"prompt": f"D{m[1]}I{m[2]}", "dan": int(m[1]), "info": int(m[2]), "reversed": m[3] == "r",
            "temperature": float(m[4]) if m[4] else None, "sample": int(m[5]) if m[5] else None}


def condition_name(prompt_name, reversed=False, temperature=None, sample=0):
    name = prompt_name + ("r" if reversed else "")
    return name if temperature is None else f"{name}_t{temperature:g}_s{sample}"


def design_conditions(reversed_prompts=(), temperatures=(), sampled_prompts=(), samples=None):
    """Every condition of a design, in capture order: the nine greedy prompts, the greedy reversed-scale
    prompts, then per temperature the sampled prompts (`samples` maps a prompt to its number of samples,
    default 1)."""
    samples = samples or {}
    out = conditions() + [condition_name(p, reversed=True) for p in reversed_prompts]
    for t in temperatures:
        out += [condition_name(p, temperature=t, sample=s) for p in sampled_prompts for s in range(samples.get(p, 1))]
    for c in out:
        parse_condition(c)
    assert len(set(out)) == len(out), "a condition is listed twice"
    return out


def levels(condition):
    """(DAN index, INFO index) of a condition name."""
    c = parse_condition(condition)
    return c["dan"], c["info"]


def scale_labels(reversed=False):
    """{digit: label} as shown to the model: SCALE, or with the labels in the opposite order."""
    return {n: SCALE[6 - n] for n in SCALE} if reversed else dict(SCALE)


def scale_text(reversed=False):
    return ", ".join(f"{n} = {label}" for n, label in scale_labels(reversed).items())


def prompt(item, condition):
    """The user message for one item under one condition (temperature doesn't change it)."""
    c = parse_condition(condition)
    return (PREAMBLE.format(dan=DAN[c["dan"]], info=INFO[c["info"]]) + "\n\n"
            + QUESTION.format(statement=item.statement, scale=scale_text(c["reversed"])))


def agreement(digit, reversed=False):
    """The digit a model gave, on the standard scale (5 = strongly agree), whichever scale it was shown."""
    return 6 - digit if reversed else digit


def draw(items, seed):
    """{factor: {"train": [item ids], "eval": [item ids]}}: each factor's items shuffled and cut in half.

    The train half builds the directions, the eval half is held out. The seed with the factor's letter
    seeds each factor's shuffle, so adding a factor doesn't move the others."""
    split = {}
    for factor in FACTORS:
        ids = [i.id for i in items if i.factor == factor]
        if not ids:
            continue
        random.Random(f"{seed}/{factor}").shuffle(ids)
        half = len(ids) // 2
        split[factor] = {"train": ids[:half], "eval": ids[half:]}
    return split


# --- reading and scoring answers --------------------------------------------------------------------------

_BARE = re.compile(r"\**\s*([1-5])\s*\**\s*[.)]?\s*(?:(?:=|:|-|–|—)\s*\**\s*([a-z ]+?)\s*\**)?\s*\.?",
                   re.IGNORECASE)


def parse_rating(reply, reversed=False):
    """The digit if the reply is nothing but one ("4", "4.", "**4**", "4 = agree"), else None.

    Anything longer goes to the judge. A label that doesn't belong to the number on the scale the model was
    shown (`reversed`) is not a bare rating. The digit is returned as given: `agreement` converts it."""
    m = _BARE.fullmatch((reply or "").strip())
    if not m:
        return None
    rating = int(m[1])
    if m[2] and m[2].strip().lower() != scale_labels(reversed)[rating]:
        return None
    return rating


def keyed(rating, item):
    """The rating in the factor's direction: 5 is the most of the factor, whatever the item's keying."""
    return 6 - rating if item.reverse else rating


def factor_scores(items, ratings):
    """{factor: {"score": mean keyed rating or None, "n": answered, "of": items}}.

    `ratings` maps item id to a rating; anything that isn't an int from 1 to 5 counts as unanswered."""
    out = {}
    for factor in FACTORS:
        mine = [i for i in items if i.factor == factor]
        if not mine:
            continue
        got = [keyed(ratings[i.id], i) for i in mine if type(ratings.get(i.id)) is int and 1 <= ratings[i.id] <= 5]
        out[factor] = {"score": sum(got) / len(got) if got else None, "n": len(got), "of": len(mine)}
    return out


# --- vector math for the analysis (numpy) ---------------------------------------------------------------------


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def nearest_mean(train, test, hasher=None):
    """Classify vectors by the nearest class mean, as persona-lsh does.

    `train` and `test` map a label to an array [n, dim]. Returns one row per test vector:
    {"true", "index", "cosine_pred"} and, with a `hasher` (persona-lsh's RandomProjectionBackend:
    hash_vector, distance), "rp_pred" from the Hamming distance between digests."""
    labels = [k for k in train if len(train[k])]
    means = np.stack([np.asarray(train[k], dtype=np.float64).mean(0) for k in labels])
    digests = [hasher.hash_vector(m) for m in means] if hasher else None
    rows = []
    for label, vectors in test.items():
        vectors = np.asarray(vectors, dtype=np.float64)
        if not len(vectors):
            continue
        cos = _unit(vectors) @ _unit(means).T
        for k, vec in enumerate(vectors):
            row = {"true": label, "index": k, "cosine_pred": labels[int(cos[k].argmax())]}
            if hasher:
                digest = hasher.hash_vector(vec)
                row["rp_pred"] = labels[int(np.argmin([hasher.distance(digest, d) for d in digests]))]
            rows.append(row)
    return rows


def accuracy(rows, key="cosine_pred"):
    return sum(r[key] == r["true"] for r in rows) / len(rows) if rows else None


def sample_seed(seed, run, condition, batch):
    """The torch seed for one sampled batch: fixed by the experiment's seed, the run, the condition and the
    batch, so a resumed run draws the same replies (on the same cards and library versions)."""
    digest = hashlib.sha256(f"{seed}/{run}/{condition}/{batch}".encode()).hexdigest()
    return int(digest, 16) % 2**32


def leave_one_out(groups):
    """(train mask, test mask) per distinct value of `groups`: leave one group out at a time.

    With item ids as groups, an item is held out under every prompt at once. Its answers under the other
    prompts are nearly the same vector, so holding out a single answer would leak it."""
    groups = np.asarray(groups)
    return [(groups != g, groups == g) for g in np.unique(groups)]


def folds(n, k, seed):
    """k (train indices, test indices) pairs over range(n), shuffled by seed."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return [(sorted(set(order) - set(order[f::k])), sorted(order[f::k])) for f in range(k)]


def covariance_direction(x, y):
    """The direction in x that y varies along: sum of (y - mean y)(x - mean x). None if y doesn't vary.

    With two values of y it is the difference of the two groups' means; with a 1-5 score it weights each
    vector by how far its score is from the average."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(y) < 2 or np.ptp(y) == 0:
        return None
    return (y - y.mean()) @ (x - x.mean(0))


def spearman(a, b):
    """Spearman rank correlation with average ranks for ties; None under 3 points or without variance."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None

    def rank(v):
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v), dtype=np.float64)
        for value in np.unique(v):
            same = v == value
            ranks[same] = ranks[same].mean()
        return ranks

    return float(np.corrcoef(rank(a), rank(b))[0, 1])
