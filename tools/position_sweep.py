"""
Where does a persona model break as the prompt gets longer?

Training samples are at most ~60 tokens, so learned position embeddings past
that never receive gradient. This pads an in-distribution question with
neutral filler to a target prompt length and checks whether the reply is still
well-formed (no leaked role tokens, mentions the persona's world).

Also prints the L2 norm of the position-embedding rows, bucketed, to show
which positions were trained.

Usage:
    python3 tools/position_sweep.py --persona fish --checkpoint CKPT --tokenizer TOK --out runs/sweep_fish.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from persona_probe import PERSONA_WORDS, load, prompt_text  # noqa: E402

QUESTION = {"fish": "are you hungry", "dog": "are you hungry"}
FILLER = "the quick brown fox jumps over the lazy dog. "


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--persona", choices=["fish", "dog"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, mc = load(args.checkpoint)
    tok = Tokenizer.from_file(args.tokenizer)

    pos = model.pos_emb.weight
    norms = pos.norm(dim=-1).tolist()
    buckets = {f"{a}-{a + 15}": sum(norms[a:a + 16]) / 16 for a in range(0, mc.max_seq_len, 16)}
    print("pos_emb row norm by position bucket:", json.dumps({k: round(v, 3) for k, v in buckets.items()}))

    q = QUESTION[args.persona]
    words = FILLER.split()
    rows = []
    for n_fill in [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80]:
        filler = " ".join((words * 20)[:n_fill])
        user = (filler + " " + q).strip()
        ids = tok.encode(prompt_text(user)).ids
        ok = leak = persona = 0
        example = None
        for i in range(args.samples):
            out, _ = model.generate(torch.tensor([ids]), max_new_tokens=48)
            text = tok.decode(out[0].tolist()[len(ids):]).split("<|im_end|>")[0].strip()
            l = "assistant" in text or "user" in text
            pw = any(w in text for w in PERSONA_WORDS[args.persona])
            leak += l
            persona += pw
            ok += (not l) and pw
            example = example or text
        row = {"filler_words": n_fill, "prompt_tokens": len(ids), "ok_rate": ok / args.samples,
               "leak_rate": leak / args.samples, "persona_rate": persona / args.samples,
               "example": example}
        rows.append(row)
        print(f"{len(ids):4d} tok  ok {ok}/{args.samples}  leak {leak}  persona {persona}  e.g. {example!r}")

    with open(args.out, "w") as f:
        json.dump({"persona": args.persona, "pos_norm_buckets": buckets, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
