"""Specs for Haiku-written ocean conversations that join TinyDialogues in stage 1.

    python3 tools/aquarium/ocean_specs.py   # writes data_aq/ocean/specs_{0..3}.jsonl, 500 each
    python3 tools/aquarium/ocean_specs.py --merge   # out_*.jsonl + specs -> data_aq/ocean/chat.jsonl

A child (user) asks an adult (assistant) about the sea, in the system-prompt shape of
tools/aquarium/tinydialogues.py so the two sets read as one.
"""

import argparse
import glob
import json
import os
import random

D = os.path.join(os.path.dirname(__file__), "..", "..", "data_aq", "ocean")
ROLES = ["mom", "dad", "grandparent", "babysitter", "teacher", "older sibling", "aquarium guide",
         "marine biologist", "lifeguard", "boat captain", "scuba diver", "fisherman", "neighbor"]
AGES = ["four", "five", "six", "seven", "eight", "nine", "ten"]
TOPICS = [
    "why the sea is salty", "how waves are made", "what tides are", "why the sea looks blue",
    "how deep the ocean is", "what lives at the very bottom of the sea", "how fish breathe underwater",
    "why whales sing", "how big a blue whale is", "how dolphins sleep", "how octopuses change color",
    "why octopuses squirt ink", "what coral is", "why coral reefs are important", "what kelp forests are",
    "how sea turtles find the beach where they were born", "what plankton is", "how jellyfish move",
    "whether jellyfish have brains", "how sharks find food", "how many teeth sharks have",
    "how seahorse dads carry the babies", "why anglerfish have a light", "how hermit crabs pick new shells",
    "how crabs grow by shedding their shell", "how sea stars grow back an arm", "what lives in a tide pool",
    "where seashells come from", "how pufferfish puff up", "why clownfish live in anemones",
    "how penguins swim", "how seals and sea lions are different", "how sea otters hold hands when they sleep",
    "what manatees eat", "how narwhals use their tusks", "how salmon swim back up rivers",
    "how fish swim in schools", "how flying fish fly", "why some fish glow", "what a sea cucumber is",
    "how lobsters smell with their legs", "what a mantis shrimp is", "how sand is made",
    "why we should not leave trash on the beach", "how boats float", "what a lighthouse is for",
    "what a submarine is", "how divers breathe underwater", "what an island is", "what lives in the ice-cold sea",
    "how whales breathe air", "what barnacles are", "how sea urchins move", "what seaweed is",
    "how stingrays hide in sand", "how eels move", "what a pearl is and how oysters make them",
    "how the moon pulls the tides", "what a coral reef sounds like", "how fish sleep",
    "how big the biggest octopus is", "how baby fish hatch", "why some animals live in shells",
    "how far whales travel", "what happens in a storm at sea", "how to stay safe at the beach",
    "what a sand dollar is", "how sea birds catch fish", "what lives in a mangrove", "how warm or cold the sea is",
]


def specs():
    rng = random.Random(7)
    os.makedirs(D, exist_ok=True)
    for k in range(4):
        with open(os.path.join(D, f"specs_{k}.jsonl"), "w") as f:
            for i in range(500):
                role, age = rng.choice(ROLES), rng.choice(AGES)
                article = "an" if role[0] in "aeiou" else "a"
                f.write(json.dumps({
                    "id": 30_000_000 + k * 1000 + i,
                    "system": f"You are {article} {role}. You are talking with a {age}-year-old child.",
                    "adult": role, "child_age": age, "topic": rng.choice(TOPICS),
                    "setting": rng.choice(["at the beach", "at an aquarium", "at home", "on a boat",
                                           "at bedtime", "at a tide pool", "reading a picture book", "on a walk"]),
                    "messages": rng.choice([6, 8, 8, 10, 10, 12])}) + "\n")


def merge():
    spec = {}
    for p in glob.glob(os.path.join(D, "specs_*.jsonl")):
        for line in open(p):
            s = json.loads(line)
            spec[s["id"]] = s
    rows, bad = [], 0
    for p in sorted(glob.glob(os.path.join(D, "out_*.jsonl"))):
        for line in open(p):
            try:
                d = json.loads(line)
                s = spec[d["id"]]
                t = d["turns"]
                ok = (len(t) == s["messages"] and all(x["role"] == ("user", "assistant")[i % 2]
                      and x["content"].strip() and "—" not in x["content"] for i, x in enumerate(t)))
            except (json.JSONDecodeError, KeyError, TypeError):
                ok = False
            if not ok:
                bad += 1
                continue
            rows.append({"system": s["system"], "turns": t, "source": "ocean", "topic": s["topic"]})
    with open(os.path.join(D, "chat.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows):,} ocean chats -> chat.jsonl, {bad} rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    merge() if ap.parse_args().merge else specs()
