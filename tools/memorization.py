"""
Is the fish thinking, or is it a lookup table with fins?

Samples a checkpoint (CPU, seeded) on four prompt sets and checks every reply
against the training outputs:

  eval_cases   the 16 hand-written prompts in guppylm/eval_cases.py
  eval_jsonl   up to N distinct user inputs drawn from data/eval.jsonl
  novel        hand-written paraphrases that never appear as a train input
  copy_probe   "confused" questions about human things that are NOT in the
               generator's HUMAN_THINGS list. In train, the reply always names
               the thing from the question, so copying the new noun is a
               small test of composition over lookup.

Per reply:
  verbatim          reply equals some train output exactly
  verbatim_same_in  reply equals a train output paired with this exact input
  composed          not verbatim, but every sentence occurs in train outputs
  novel_sentence    at least one sentence never occurs in train outputs

Usage:
    python tools/memorization.py --ckpt runs/X/checkpoints/best_model.pt --out runs/X/memo.json
    python tools/memorization.py --ckpt ... --train-limit 5000   # "train" = the subset the run saw
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from guppylm.config import GuppyConfig  # noqa: E402
from guppylm.dataset import subset_lines  # noqa: E402
from guppylm.eval_cases import get_eval_cases  # noqa: E402
from guppylm.model import GuppyLM  # noqa: E402
from tools.data_analysis import PAT, sentences  # noqa: E402

# (prompt, acceptable reply categories). Every prompt is checked against train inputs at runtime.
NOVEL = [
    ("hey there little swimmer", {"greeting", "swimming"}),
    ("how's my favorite fish doing", {"feeling", "greeting", "happy"}),
    ("are you feeling alright today", {"feeling", "happy", "doctor"}),
    ("do you want a snack", {"food"}),
    ("dinner is served", {"food"}),
    ("it's sweltering in here", {"temp_hot"}),
    ("the house is freezing tonight", {"temp_cold"}),
    ("i swapped out your tank water", {"water"}),
    ("i switched the lamp on", {"light"}),
    ("oops i knocked over a glass", {"noise", "scared"}),
    ("night night fishy", {"night", "sleep", "bye"}),
    ("what kind of creature are you", {"about", "size", "name"}),
    ("do you ever wish you had company", {"lonely", "friends", "snail"}),
    ("i bought you a little castle", {"tank"}),
    ("what are you pondering", {"misc", "curious", "meaning"}),
    ("thanks little buddy", {"follow_up", "greeting", "happy", "love"}),
    ("do you dream at night", {"dreams", "sleep", "night"}),
    ("is your filter working", {"filter", "water", "bubbles"}),
    ("what's your favorite color", {"colors"}),
    ("do you enjoy songs", {"music", "singing"}),
    ("is it raining where you are", {"rain", "weather"}),
    ("what is your age", {"age", "time", "past"}),
    ("are you a clever fish", {"smart"}),
    ("why are we here", {"meaning"}),
    ("do you care about me", {"love", "friends"}),
    ("there's a cat looking at you", {"cat", "scared", "fear"}),
    ("do you remember yesterday", {"memory", "past", "time"}),
    ("what will you do tomorrow", {"future", "time"}),
    ("are you afraid of anything", {"fear", "scared"}),
    ("is it storming outside", {"weather", "rain"}),
]

COPY_THINGS = ["quantum physics", "a toaster", "basketball", "a credit card", "poetry",
               "a vacuum cleaner", "the olympics", "a passport", "tiktok", "a bicycle"]
# Control: things that ARE in HUMAN_THINGS, same templates.
KNOWN_THINGS = ["wifi", "taxes", "a podcast", "homework", "the stock market",
                "a car", "cooking", "email", "a password", "school"]
COPY_TEMPLATES = ["what do you think about {}", "do you know what {} is", "explain {}"]
EVAL_CASE_CATS = {"follow_up": {"follow_up", "greeting", "happy", "love", "bye"}}


def prompt_text(user_msg):
    return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"


def load_pairs(path, limit=None, seed=0, with_cat=False):
    lines = subset_lines(path, limit, seed)
    out = []
    for line in lines:
        d = json.loads(line)
        m = PAT.match(d["text"])
        out.append((m.group(1), m.group(2), d["category"]) if with_cat else (m.group(1), m.group(2)))
    return out


def make_classifier(full_train):
    """Reply -> most likely train category, from which categories each sentence occurs in."""
    sent_cat = defaultdict(Counter)
    for _, o, c in full_train:
        for s in sentences(o):
            sent_cat[s][c] += 1

    def classify(text):
        score = Counter()
        for s in sentences(text):
            dist = sent_cat.get(s)
            if dist:
                tot = sum(dist.values())
                for c, k in dist.items():
                    score[c] += k / tot
        return score.most_common(1)[0][0] if score else None
    return classify


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    valid = set(GuppyConfig.__dataclass_fields__)
    mc = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid})
    model = GuppyLM(mc)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-limit", type=int, default=None,
                    help="compare against the same train subset the run used")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=3, help="generations per prompt")
    ap.add_argument("--n-eval-inputs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    data_dir = os.path.join(ROOT, "data")
    tok = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    train = load_pairs(os.path.join(data_dir, "train.jsonl"), args.train_limit, args.subset_seed)
    full_train = load_pairs(os.path.join(data_dir, "train.jsonl"), with_cat=True)
    classify = make_classifier(full_train)
    train_outs = {o for _, o in train}
    train_sents = Counter(s for _, o in train for s in sentences(o))
    outs_by_input = defaultdict(set)
    for i, o in train:
        outs_by_input[i].add(o)
    full_inputs = {i for i, _, _ in full_train}

    expected = {p: cats for p, cats in NOVEL}
    for c in get_eval_cases():
        expected[c["prompt"]] = EVAL_CASE_CATS.get(c["category"], {c["category"]})
    novel = [p for p, _ in NOVEL if p not in full_inputs]
    dropped = [p for p, _ in NOVEL if p in full_inputs]
    if dropped:
        print(f"note: dropped novel prompts that are train inputs: {dropped}")
    ev_inputs = sorted({i for i, _ in load_pairs(os.path.join(data_dir, "eval.jsonl"))})
    random.Random(0).shuffle(ev_inputs)
    copy_probe = [t.format(x) for x in COPY_THINGS for t in COPY_TEMPLATES]
    copy_known = [t.format(x) for x in KNOWN_THINGS for t in COPY_TEMPLATES]
    for p in copy_probe + copy_known:
        expected[p] = {"confused"}

    sets = {
        "eval_cases": [c["prompt"] for c in get_eval_cases()],
        "eval_jsonl": ev_inputs[:args.n_eval_inputs],
        "novel": novel,
        "copy_probe": copy_probe,
        "copy_known": copy_known,
    }

    model = load_model(args.ckpt)
    torch.manual_seed(args.seed)
    report = {"ckpt": args.ckpt, "train_limit": args.train_limit, "samples": args.samples, "sets": {}}
    for name, prompts in sets.items():
        rows = []
        for p in prompts:
            ids = tok.encode(prompt_text(p)).ids
            for _ in range(args.samples):
                with torch.no_grad():
                    out, _ = model.generate(torch.tensor([ids]), max_new_tokens=64)
                text = tok.decode(out[0].tolist()[len(ids):]).split("<|im_end|>")[0].strip()
                sents = sentences(text)
                in_train = [s in train_sents for s in sents]
                row = {
                    "prompt": p, "reply": text,
                    "prompt_in_train": p in outs_by_input,
                    "verbatim": text in train_outs,
                    "verbatim_same_in": text in outs_by_input.get(p, ()),
                    "n_sent": len(sents),
                    "n_sent_in_train": sum(in_train),
                    "novel_sentences": [s for s, k in zip(sents, in_train) if not k],
                }
                row["composed"] = (not row["verbatim"]) and bool(sents) and all(in_train)
                row["reply_category"] = classify(text)
                if p in expected:
                    row["on_topic"] = row["reply_category"] in expected[p]
                if name.startswith("copy_"):
                    thing = next(x for x in COPY_THINGS + KNOWN_THINGS if x in p)
                    row["copied_thing"] = thing in text
                rows.append(row)
        n = len(rows)
        tot_s = sum(r["n_sent"] for r in rows)
        summ = {
            "n_replies": n,
            "verbatim": sum(r["verbatim"] for r in rows) / n,
            "verbatim_same_input": sum(r["verbatim_same_in"] for r in rows) / n,
            "composed": sum(r["composed"] for r in rows) / n,
            "has_novel_sentence": sum(bool(r["novel_sentences"]) for r in rows) / n,
            "sentence_in_train": sum(r["n_sent_in_train"] for r in rows) / max(1, tot_s),
            "distinct_replies": len({r["reply"] for r in rows}) / n,
        }
        if any("on_topic" in r for r in rows):
            scored = [r for r in rows if "on_topic" in r]
            summ["on_topic"] = sum(r["on_topic"] for r in scored) / len(scored)
        if name.startswith("copy_"):
            summ["copied_thing"] = sum(r["copied_thing"] for r in rows) / n
        report["sets"][name] = {"summary": summ, "rows": rows}
        print(f"\n== {name} ({len(prompts)} prompts x {args.samples}) ==")
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summ.items()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
