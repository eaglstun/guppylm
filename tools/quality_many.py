"""
Re-score keyword hit rate with more samples than benchmark.py's default (5 per case),
on CPU, for several checkpoints. benchmark.py's 80-sample score has a standard error
of about 4.6 points, too wide to rank runs that differ by a few points.

Usage:
    python tools/quality_many.py --samples 30 runs/A/checkpoints/best_model.pt ...
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from tools.benchmark import load_model, quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = {}
    for c in args.ckpts:
        model, _, _ = load_model(c, torch.device("cpu"))
        q = quality(model, args.samples, args.seed)
        n = 16 * args.samples
        se = math.sqrt(q["keyword_hit_rate"] * (1 - q["keyword_hit_rate"]) / n)
        res[c] = {"keyword_hit_rate": q["keyword_hit_rate"], "se": se, "n": n}
        print(f"{q['keyword_hit_rate']:.3f} +- {se:.3f}  {c}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
