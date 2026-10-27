"""American spellings for talkie rows: a fixed, case-preserving word map over turn text.

Writes a new JSONL beside the input; every change is recorded in ``provenance.respelled``.

    python -m tools.aquarium.v52.respell runs/v52/talkie/whimsy-v1/chat.jsonl
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

VERSION = 1
SPELLINGS = {
    "colour": "color", "colours": "colors", "coloured": "colored",
    "favourite": "favorite", "neighbour": "neighbor", "neighbours": "neighbors",
    "practising": "practicing", "grey": "gray", "hullo": "hello",
    "good-bye": "goodbye", "good-night": "goodnight", "good-evening": "good evening",
    "good-day": "good day", "good-morning": "good morning",
    "leap-frog": "leapfrog", "sea-shore": "seashore", "night-time": "nighttime",
    "dove-cote": "dovecote", "angler-fish": "anglerfish", "sea-floor": "sea floor",
    "picture-postcard": "picture postcard",
}
PATTERN = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SPELLINGS), key=len, reverse=True))
                     + r")\b", re.I)


def match_case(word, replacement):
    if word.isupper():
        return replacement.upper()
    return replacement[0].upper() + replacement[1:] if word[0].isupper() else replacement


def respell(text):
    changes = []

    def swap(m):
        new = match_case(m.group(0), SPELLINGS[m.group(0).lower()])
        changes.append([m.group(0), new])
        return new

    return PATTERN.sub(swap, text), changes


def respell_row(row):
    log = []
    turns = []
    for i, turn in enumerate(row["turns"]):
        content, changes = respell(turn["content"])
        log += [[i, *c] for c in changes]
        turns.append({**turn, "content": content})
    if not log:
        return row
    return {**row, "turns": turns, "provenance": {**row.get("provenance", {}), "respelled": log}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or a.src.with_name(a.src.stem + "_us.jsonl")
    rows = [respell_row(json.loads(line)) for line in a.src.read_text().splitlines() if line.strip()]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    changed = [r for r in rows if "respelled" in r.get("provenance", {})]
    print(json.dumps({"version": VERSION, "rows": len(rows), "rows_changed": len(changed),
                      "changes": sum(len(r["provenance"]["respelled"]) for r in changed),
                      "src_sha256": hashlib.sha256(a.src.read_bytes()).hexdigest(),
                      "out": str(out)}))


if __name__ == "__main__":
    main()
