"""Poems and songs: 200 conversations per training creature where the user asks for one.

    python3 tools/aquarium/song_specs.py specs                  # data_aq/songs/specs.jsonl (3,200)
    python3 tools/aquarium/song_specs.py gen --model qwen3:4b --sample 48 --out data_aq/songs/bench_qwen3-4b.jsonl
    python3 tools/aquarium/song_specs.py gen --model qwen3:4b --ids 0-1599 --out data_aq/songs/out_qwen.jsonl

gen only ever takes odd ids; Haiku agents write the even ones.
    python3 tools/aquarium/song_specs.py merge                  # out_*.jsonl -> data_aq/songs/chat.jsonl

Rows carry their own system prompt and name, so train_aq's renamed copies work on them.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personas import CREATURE_USERS, HUMAN_USERS, NAMES, TRAIN_PERSONAS, system_prompt  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_aq", "songs")
FORMS = ["poem", "song", "short rhyme", "lullaby", "sea shanty", "haiku", "silly song", "chant", "birthday song"]
SUBJECTS = ["your home", "your favorite food", "a friend", "the moon", "bubbles", "the tide", "a storm",
            "the sun on the water", "being small", "being sleepy", "a lost shell", "the person who feeds you",
            "a big fish", "the rain", "night time", "your best day", "what makes you special", "a dream",
            "the sea floor", "the color blue", "a boat going by", "being brave", "dinner", "a new neighbor"]
OPENERS = ["Sing me", "Will you", "Please", "I bet you can't", "Do you know any", "Make up", "Teach me",
           "It's bedtime.", "I'm a little sad.", "Got a", "Tell me", "Let's hear", "Could you", "I want", "Hey,",
           "What would", "Do you have", "Can you"]
PER = 200


def specs():
    rng = random.Random(11)
    names = NAMES if isinstance(NAMES, list) else NAMES.split()
    os.makedirs(D, exist_ok=True)
    with open(os.path.join(D, "specs.jsonl"), "w") as f:
        for p, persona in enumerate(TRAIN_PERSONAS):
            for i in range(PER):
                creature = rng.random() < 0.3
                partner = rng.choice(CREATURE_USERS if creature else HUMAN_USERS)
                name = rng.choice(names)
                f.write(json.dumps({
                    "id": 40_000_000 + p * 1000 + i, "persona": persona, "name": name,
                    "user_kind": "creature" if creature else "human", "partner": partner, "thing": None,
                    "system": system_prompt(rng, persona, name, partner),
                    "form": rng.choice(FORMS),
                    "subject": rng.choice(SUBJECTS) if rng.random() < 0.6 else None,
                    "lines": rng.choice([2, 4, 4, 4, 6, 6, 8]),
                    "exchanges": rng.choice([1, 1, 2])}) + "\n")


BRIEF = """Write a short conversation for training a tiny language model.

The assistant is a sea creature. Its system prompt is:
\"\"\"{system}\"\"\"
The user is {partner}.

The user asks the creature for a {form}{about}. The creature answers in character with the {form}
itself: {lines} short lines, each on its own line (separate lines with \\n inside the string). Simple
words a young child knows. Rhyme if you can. It should be about life as this creature, in its
voice, and it may say one short sentence before or after it.{more}

Rules: no em dashes or en dashes. The creature never says it is an AI or a language model.
Return JSON: {{"turns": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}{extra}]}}
with exactly {n} turns, starting with the user."""


def brief(s):
    more = (" Then the user reacts or asks one follow-up question, and the creature answers briefly"
            " (a sentence or two, or two more lines of the song)." if s["exchanges"] == 2 else "")
    extra = ", ..." if s["exchanges"] == 2 else ""
    about = f" about {s['subject']}" if s["subject"] else ""
    opener = OPENERS[s["id"] % len(OPENERS)]
    more = (f" The user's first message begins with the words \"{opener}\" and goes on to ask for the {s["form"]}"
            " in a full sentence." + more)
    return BRIEF.format(system=s["system"], partner=s["partner"], form=s["form"], about=about,
                        lines=s["lines"], more=more, extra=extra, n=2 * s["exchanges"])


def tidy(turns):
    """Collapse the blank lines small models put between song lines."""
    return [{**t, "content": re.sub(r"\n\s*\n+", "\n", t["content"].strip())} for t in turns]


def valid(s, turns):
    if not isinstance(turns, list) or len(turns) != 2 * s["exchanges"]:
        return False
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or t.get("role") != ("user", "assistant")[i % 2]:
            return False
        c = t.get("content")
        if not isinstance(c, str) or not c.strip() or "—" in c or "–" in c:
            return False
        if "AI" in c.split() or "language model" in c.lower():
            return False
    return turns[1]["content"].count("\n") >= 1


def gen(a):
    url = os.environ.get("OLLAMA_CHAT_URL", "http://127.0.0.1:11435/api/chat")
    lo, hi = map(int, a.ids.split("-"))
    rows = [r for r in (json.loads(l) for l in open(os.path.join(D, "specs.jsonl"))) if r["id"] % 2 == 1]
    rows = random.Random(5).sample(rows, a.sample) if a.sample else rows[lo:hi + 1]

    def one(s):
        body = {"model": a.model, "stream": False, "format": "json", "think": False,
                "options": {"temperature": 0.9, "num_predict": 600},
                "messages": [{"role": "user", "content": brief(s)}]}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        for _ in range(3):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=300).read())
                turns = json.loads(r["message"]["content"]).get("turns")
                turns = tidy(turns) if isinstance(turns, list) and all(isinstance(t, dict) and isinstance(t.get("content"), str) for t in turns) else turns
                if valid(s, turns):
                    return s, turns, r.get("eval_count", 0)
            except Exception:
                time.sleep(1)
        return s, None, 0

    t0, ok, toks = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for s, turns, n in ex.map(one, rows):
            toks += n
            if turns and valid(s, turns):
                ok += 1
                f.write(json.dumps({"id": s["id"], "turns": turns}) + "\n")
    el = time.time() - t0
    print(f"{a.model}: {ok}/{len(rows)} valid in {el:.0f}s, {len(rows) / el * 60:.1f} convs/min, "
          f"{toks / el:.0f} tok/s overall")


def merge():
    spec = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, "specs.jsonl"))}
    rows, bad, seen = [], 0, set()
    for p in sorted(glob.glob(os.path.join(D, "out_*.jsonl"))):
        for line in open(p):
            try:
                d = json.loads(line)
                s = spec[d["id"]]
            except (json.JSONDecodeError, KeyError):
                bad += 1
                continue
            if d["id"] in seen or not valid(s, d.get("turns")):
                bad += 1
                continue
            seen.add(d["id"])
            rows.append({**{k: s[k] for k in ("id", "persona", "name", "user_kind", "partner", "thing", "system")},
                         "turns": d["turns"], "source": os.path.basename(p)})
    with open(os.path.join(D, "chat.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows):,} song chats -> chat.jsonl, {bad} rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["specs", "gen", "merge"])
    ap.add_argument("--model")
    ap.add_argument("--ids", default="0-99")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, help="gen: a fixed random sample of the odd-id specs, for comparing models")
    a = ap.parse_args()
    {"specs": specs, "merge": merge}.get(a.cmd, lambda: gen(a))()
