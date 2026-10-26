"""Noun copy probe with nouns and question forms absent from continuation training."""
import argparse
import json
import os
import torch

from train_aq import D, GuppyConfig, GuppyLM, Tokenizer, sample
from personas import PERSONAS

NOUNS = ["a snowmobile", "a hot air balloon", "origami", "a saxophone", "a microscope",
         "a parking garage", "volleyball", "the solar eclipse", "a printer cartridge", "a dishwasher"]
QUESTIONS = ["I saw {x} yesterday. What would you say about it?",
             "Why might a person use {x}?",
             "Would you keep {x} in your tide pool?",
             "Suppose I brought you {x}. How would you react?"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["tok_emb.weight"]
    model = GuppyLM(GuppyConfig(**ck["config"]))
    model.load_state_dict(sd)
    tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
    species, habitat, traits = PERSONAS["hermit_crab"]
    system = f"You are Pebble, {species} who lives in {habitat}. You are {traits}."
    rows = []
    for i, noun in enumerate(NOUNS):
        for j, pattern in enumerate(QUESTIONS):
            torch.manual_seed(i * 101 + j)
            question = pattern.format(x=noun)
            (_, reply), = sample(model, tok, "cpu", system, [question], max_new=50)
            key = noun.split(" ", 1)[-1] if noun.startswith(("a ", "the ")) else noun
            rows.append({"noun": noun, "question": question, "reply": reply,
                         "hit": key.lower() in reply.lower()})
    summary = {"copy": sum(r["hit"] for r in rows) / len(rows),
               "hits": sum(r["hit"] for r in rows), "n": len(rows)}
    print(json.dumps(summary), flush=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)

if __name__ == "__main__":
    main()
