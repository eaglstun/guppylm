"""
How varied is GuppyLM's training data really?

Reads data/train.jsonl and data/eval.jsonl and reports:
  - unique inputs / outputs / (input, output) pairs, overall and per category
  - sentence-level reuse (how often the same reply sentence recurs)
  - train/eval overlap: exact pair, exact output, exact input,
    "fully composed" (every eval sentence occurs in train), near-duplicate
    (difflib ratio >= 0.9 to some train output of the same category)
  - whether the hand-written eval_cases prompts appear verbatim as train inputs

Usage:
    python tools/data_analysis.py [--json out.json]
"""

import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guppylm.eval_cases import get_eval_cases  # noqa: E402

PAT = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)
SENT = re.compile(r"(?<=[.!?])\s+")


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            m = PAT.match(d["text"])
            rows.append({"input": m.group(1), "output": m.group(2), "category": d["category"]})
    return rows


def sentences(text):
    return [s.strip() for s in SENT.split(text.strip()) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    train = load(os.path.join(args.data_dir, "train.jsonl"))
    ev = load(os.path.join(args.data_dir, "eval.jsonl"))
    rep = {}

    # ── Uniqueness ──
    n = len(train)
    u_in = Counter(r["input"] for r in train)
    u_out = Counter(r["output"] for r in train)
    u_pair = Counter((r["input"], r["output"]) for r in train)
    rep["train"] = {
        "n": n,
        "unique_inputs": len(u_in),
        "unique_outputs": len(u_out),
        "unique_pairs": len(u_pair),
        "unique_output_frac": len(u_out) / n,
        "samples_whose_output_is_shared": sum(c for c in u_out.values() if c > 1) / n,
        "top_outputs": u_out.most_common(10),
        "top_inputs": u_in.most_common(10),
    }

    # per category
    by_cat = defaultdict(list)
    for r in train:
        by_cat[r["category"]].append(r)
    cats = {}
    for c, rs in by_cat.items():
        cats[c] = {"n": len(rs),
                   "unique_inputs": len({r["input"] for r in rs}),
                   "unique_outputs": len({r["output"] for r in rs})}
    rep["per_category"] = dict(sorted(cats.items(), key=lambda kv: kv[1]["unique_outputs"]))

    # ── Sentences ──
    sent_counts = Counter()
    for r in train:
        sent_counts.update(sentences(r["output"]))
    total_sents = sum(sent_counts.values())
    rep["sentences"] = {
        "total": total_sents,
        "unique": len(sent_counts),
        "mean_reuse": total_sents / len(sent_counts),
        "frac_of_occurrences_from_top_100": sum(c for _, c in sent_counts.most_common(100)) / total_sents,
        "frac_of_occurrences_from_top_1000": sum(c for _, c in sent_counts.most_common(1000)) / total_sents,
        "singletons": sum(1 for c in sent_counts.values() if c == 1),
        "top": sent_counts.most_common(15),
    }

    # ── Train/eval overlap ──
    train_pairs = set(u_pair)
    train_outs = set(u_out)
    train_ins = set(u_in)
    train_sents = set(sent_counts)
    outs_by_cat = defaultdict(set)
    for r in train:
        outs_by_cat[r["category"]].add(r["output"])

    exact_pair = exact_out = exact_in = composed = near = 0
    near_examples = []
    for r in ev:
        exact_pair += (r["input"], r["output"]) in train_pairs
        exact_in += r["input"] in train_ins
        if r["output"] in train_outs:
            exact_out += 1
            composed += 1
            near += 1
            continue
        if all(s in train_sents for s in sentences(r["output"])):
            composed += 1
        best, best_o = 0.0, None
        for o in outs_by_cat[r["category"]]:
            sm = difflib.SequenceMatcher(None, r["output"], o)
            if sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9:
                continue
            q = sm.ratio()
            if q > best:
                best, best_o = q, o
        if best >= 0.9:
            near += 1
            if len(near_examples) < 5:
                near_examples.append({"eval": r["output"], "train": best_o, "ratio": round(best, 3)})
    m = len(ev)
    rep["eval_overlap"] = {
        "n_eval": m,
        "exact_pair_in_train": exact_pair / m,
        "exact_input_in_train": exact_in / m,
        "exact_output_in_train": exact_out / m,
        "all_sentences_in_train": composed / m,
        "near_dup_output_ratio_ge_0.9": near / m,
        "near_examples": near_examples,
    }

    # ── eval_cases prompts ──
    rep["eval_cases_prompts_in_train_inputs"] = {
        c["prompt"]: u_in.get(c["prompt"], 0) for c in get_eval_cases()
    }

    t = rep["train"]
    print(f"train n={n}  unique inputs={t['unique_inputs']}  unique outputs={t['unique_outputs']} "
          f"({t['unique_output_frac']:.1%})  unique pairs={t['unique_pairs']}")
    print(f"samples whose output also appears elsewhere in train: {t['samples_whose_output_is_shared']:.1%}")
    print("top outputs:")
    for o, c in t["top_outputs"]:
        print(f"  {c:5d}  {o}")
    print("\nper category (lowest unique outputs first):")
    for c, v in list(rep["per_category"].items()):
        print(f"  {c:16s} n={v['n']:5d}  uniq_in={v['unique_inputs']:4d}  uniq_out={v['unique_outputs']:5d}")
    s = rep["sentences"]
    print(f"\nsentences: {s['total']} occurrences, {s['unique']} unique, mean reuse {s['mean_reuse']:.1f}x, "
          f"top-100 cover {s['frac_of_occurrences_from_top_100']:.1%}, top-1000 cover "
          f"{s['frac_of_occurrences_from_top_1000']:.1%}, singletons {s['singletons']}")
    for o, c in s["top"]:
        print(f"  {c:5d}  {o}")
    e = rep["eval_overlap"]
    print("\neval overlap:")
    for k, v in e.items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:.1%}")
    for ex in e["near_examples"]:
        print(f"  near: {ex}")
    print("\neval_cases prompts as verbatim train inputs (count):")
    for p, c in rep["eval_cases_prompts_in_train_inputs"].items():
        print(f"  {c:5d}  {p!r}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
