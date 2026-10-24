"""Sample TinyDialogues (Feng, Goodman, Frank 2024; MIT) into a pretraining mix for the fish.

    python3 tools/aquarium/tinydialogues.py --n 20000 --ages 2 5 10

Reads data_aq/tinydialogues/tinydialogue_age-*_train.txt, keeps conversations with only two
speakers (Child plus one adult), takes an equal share from each age, and writes
data_aq/tinydialogues/sample.jsonl as {"age", "turns": [{"speaker", "text"}]} plus stats, and
chat.jsonl in the train_aq row format: the adult is the assistant, the child is the user.
"""

import argparse
import json
import os
import random
import re
import statistics

from tokenizers import Tokenizer

D = os.path.join(os.path.dirname(__file__), "..", "..", "data_aq")
TD = os.path.join(D, "tinydialogues")
TURN = re.compile(r"\*\*(.+?)\*\*:\s*(.*)")
STAGE = re.compile(r"\s*\*[^*]+\*\s*")  # *runs to pantry*
AGES = {2: "two", 5: "five", 10: "ten"}


def as_chat(row):
    """Child turns become user, adult turns assistant; same-speaker runs merge, stage directions go."""
    adult = next(x["speaker"] for x in row["turns"] if x["speaker"] != "Child")
    if adult == "Younger Sibling":  # the toddler would be the assistant
        return None
    turns = []
    for x in row["turns"]:
        text = STAGE.sub(" ", x["text"]).strip()
        if not text:
            continue
        role = "user" if x["speaker"] == "Child" else "assistant"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += " " + text
        else:
            turns.append({"role": role, "content": text})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    if len(turns) < 2:
        return None
    who = re.sub(r"\s*\d+$", "", adult.lower())
    article = "an" if who[0] in "aeiou" else "a"
    system = f"You are {article} {who}. You are talking with a {AGES.get(row['age'], str(row['age']))}-year-old child."
    return {"system": system, "turns": turns, "source": "tinydialogues", "age": row["age"]}


def parse(text):
    turns = []
    for chunk in text.replace("\\n", "\n").split("\n\n"):
        m = TURN.match(chunk.strip())
        if not m:
            return None
        who = m.group(1).strip()
        turns.append({"speaker": "Child" if who == "Toddler" else who, "text": m.group(2).strip().strip('"').strip()})
    return turns or None


def load(age):
    with open(os.path.join(TD, f"tinydialogue_age-{age}_train.txt")) as f:
        for raw in f.read().split("<|endoftext|>"):
            turns = parse(raw.strip()) if raw.strip() else None
            if turns:
                yield turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ages", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
    per_age, out = a.n // len(a.ages), []
    for age in a.ages:
        convs = list(load(age))
        two = [t for t in convs if len({x["speaker"] for x in t}) == 2 and any(x["speaker"] == "Child" for x in t)]
        pick = rng.sample(two, per_age)
        words = [sum(len(x["text"].split()) for x in t) for t in pick]
        text = ["\n".join(f'{x["speaker"]}: {x["text"]}' for x in t) for t in pick[:2000]]
        toks = sum(len(tok.encode(s).ids) for s in text)
        child = [len(x["text"].split()) for t in pick for x in t if x["speaker"] == "Child"]
        print(f"age {age}: {len(convs):,} parsed, {len(two):,} two-speaker, took {per_age:,}; "
              f"turns {statistics.mean(len(t) for t in pick):.1f}, words/conv {statistics.mean(words):.0f}, "
              f"child words/turn {statistics.mean(child):.1f}, chars/token {sum(map(len, text)) / toks:.2f}")
        out += [{"age": age, "turns": t} for t in pick]
    rng.shuffle(out)
    with open(os.path.join(TD, "sample.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    chats = [c for c in map(as_chat, out) if c]
    with open(os.path.join(TD, "chat.jsonl"), "w") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(out):,} conversations to sample.jsonl, {len(chats):,} to chat.jsonl")


if __name__ == "__main__":
    main()
