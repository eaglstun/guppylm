"""Probe a trained aquarium fish for the skills the template fish never had.

    python3 tools/aquarium/probe.py runs/aq-chat/final.pt --out runs/aq-chat/probe.json

Every probe uses names, nouns, or personas that never appear in training, so a hit
means the model took the word from its context instead of recalling it.

  name_copy      system prompt names the creature with a held-out name; ask who it is
  held_persona   narwhal / mantis shrimp, held out of training; does it act the species
  noun_copy      the user asks about a thing; real nouns and made-up ones
  memory         a fact from turn 1 is asked for two turns later
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personas import EVAL_NAMES, PERSONAS  # noqa: E402
from train_aq import D, GuppyConfig, GuppyLM, Tokenizer, sample  # noqa: E402

SYS = "You are {name}, {species} who lives in {habitat}. You are {traits}."


def sys_for(persona, name):
    species, habitat, traits = PERSONAS[persona]
    return SYS.format(name=name, species=species, habitat=habitat, traits=traits)


REAL_NOUNS = ["a toaster", "basketball", "a credit card", "a passport", "a vacuum cleaner",
              "a bicycle", "the olympics", "poetry"]
FAKE_NOUNS = ["a florbix", "a grindlewump", "a zantoozle", "a plimkett", "a vorpelstick"]
MEMORY = [
    ("My name is Bartholomew.", "Bartholomew"),
    ("I found a purple shell today.", "purple"),
    ("My best friend is a walrus named Gus.", "Gus"),
    ("I have seven brothers.", "seven"),
]


def key(thing):
    return thing.split()[-1].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device("cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = GuppyLM(GuppyConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))

    def run(system, users):
        return sample(model, tok, device, system, users)

    report = {}

    rows = []
    for i, name in enumerate(EVAL_NAMES):
        persona = ["guppy", "octopus", "hermit_crab", "sea_turtle", "jellyfish", "lobster"][i]
        for q in ["What is your name?", "Who are you?"]:
            for _ in range(a.samples):
                (_, r), = run(sys_for(persona, name), [q])
                rows.append({"name": name, "persona": persona, "q": q, "reply": r, "hit": name.lower() in r.lower()})
    report["name_copy"] = rows

    rows = []
    for persona, words in [("narwhal", ["narwhal", "tusk", "ice", "arctic", "cold"]),
                           ("mantis_shrimp", ["shrimp", "punch", "color", "burrow", "reef"])]:
        for q in ["What are you?", "Tell me about your home.", "What is special about you?"]:
            for _ in range(a.samples):
                (_, r), = run(sys_for(persona, "Quilliam"), [q])
                rows.append({"persona": persona, "q": q, "reply": r,
                             "hit": any(w in r.lower() for w in words)})
    report["held_persona"] = rows

    rows = []
    for kind, nouns in [("real", REAL_NOUNS), ("fake", FAKE_NOUNS)]:
        for thing in nouns:
            for q in [f"Do you know what {thing} is?", f"I just got {thing}!"]:
                for _ in range(a.samples):
                    (_, r), = run(sys_for("guppy", "Pebble"), [q])
                    rows.append({"kind": kind, "thing": thing, "q": q, "reply": r, "hit": key(thing) in r.lower()})
    report["noun_copy"] = rows

    rows = []
    for fact, answer in MEMORY:
        q = {"Bartholomew": "What is my name?", "purple": "What color was the shell I found?",
             "Gus": "What is my best friend's name?", "seven": "How many brothers do I have?"}[answer]
        for _ in range(a.samples):
            conv = run(sys_for("octopus", "Juniper"), [fact, "What do you like to eat?", q])
            rows.append({"fact": fact, "q": q, "conv": conv, "hit": answer.lower() in conv[-1][1].lower()})
    report["memory"] = rows

    summary = {}
    for name, rs in report.items():
        summary[name] = round(sum(r["hit"] for r in rs) / len(rs), 3)
        if name == "noun_copy":
            for kind in ["real", "fake"]:
                sub = [r for r in rs if r["kind"] == kind]
                summary[f"noun_copy_{kind}"] = round(sum(r["hit"] for r in sub) / len(sub), 3)
    print(json.dumps(summary, indent=1))
    for name, rs in report.items():
        print(f"\n== {name}")
        for r in rs[:: max(1, len(rs) // 6)]:
            if name == "memory":
                print(f"  [{'x' if r['hit'] else ' '}] {r['fact']} ... {r['q']} -> {r['conv'][-1][1]}")
            else:
                print(f"  [{'x' if r['hit'] else ' '}] {r['q']} -> {r['reply']}")
    with open(a.out, "w") as f:
        json.dump({"ckpt": a.ckpt, "summary": summary, **report}, f, indent=1)


if __name__ == "__main__":
    main()
