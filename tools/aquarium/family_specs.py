"""Friends and family: 200 conversations per training creature, written by a small local Qwen.

    python3 tools/aquarium/family_specs.py specs     # data_aq/family/specs.jsonl (3,200)
    python3 tools/aquarium/family_specs.py gen --model qwen3:4b --sample 48 --out data_aq/family/bench_qwen3-4b.jsonl
    python3 tools/aquarium/family_specs.py gen --model qwen2.5:3b --part 0/2 --out data_aq/family/out_qwen25-3b.jsonl
    python3 tools/aquarium/family_specs.py gen --model qwen3:1.7b --part 1/2 --out data_aq/family/out_qwen3-1.7b.jsonl
    python3 tools/aquarium/family_specs.py merge     # out_*.jsonl -> data_aq/family/chat.jsonl

Plain talk, no verse: the small models that couldn't rhyme (see song_specs.py) do fine here.
"""

import argparse
import glob
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personas import CREATURE_USERS, HUMAN_USERS, NAMES, TRAIN_PERSONAS, system_prompt  # noqa: E402
from song_specs import OPENERS, tidy  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_aq", "family")
SUBJECTS = ["your mom", "your dad", "your brothers and sisters", "your grandparents", "your babies",
            "your best friend", "a friend who is a different kind of animal", "how you met your best friend",
            "a time you and a friend had a fight and made up", "a friend who moved away", "your neighbors",
            "what your family does together", "who taught you to find food", "a game you play with friends",
            "who you go to when you are scared", "your biggest cousin", "a new friend you made this week",
            "whether you live alone or in a group", "what you would give your friend as a present",
            "how you can tell your family apart from other animals like you"]
PER = 200

BRIEF = """Write a short conversation for training a tiny language model.

The assistant is a sea creature. Its system prompt is:
\"\"\"{system}\"\"\"
The user is {partner}.

The user asks the creature about {subject}. The creature answers in character, in 1 to 3 short
sentences per turn, using simple words a young child knows. Its family and friends fit how this
kind of animal really lives (if it lives alone, it can say so and talk about the animals it meets).
The user's first message begins with the words "{opener}" and goes on as a full sentence.
After that the user asks follow-up questions.

Rules: no em dashes or en dashes. The creature never says it is an AI or a language model.
Return JSON: {{"turns": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]}}
with exactly {n} turns, starting with the user."""


def specs():
    rng = random.Random(13)
    names = NAMES if isinstance(NAMES, list) else NAMES.split()
    os.makedirs(D, exist_ok=True)
    with open(os.path.join(D, "specs.jsonl"), "w") as f:
        for p, persona in enumerate(TRAIN_PERSONAS):
            for i in range(PER):
                creature = rng.random() < 0.3
                partner = rng.choice(CREATURE_USERS if creature else HUMAN_USERS)
                name = rng.choice(names)
                f.write(json.dumps({
                    "id": 50_000_000 + p * 1000 + i, "persona": persona, "name": name,
                    "user_kind": "creature" if creature else "human", "partner": partner, "thing": None,
                    "system": system_prompt(rng, persona, name, partner),
                    "subject": rng.choice(SUBJECTS), "exchanges": rng.choice([1, 2, 2, 3, 3, 4])}) + "\n")


def brief(s):
    opener = OPENERS[s["id"] % len(OPENERS)]
    opener = "Tell me" if opener in ("Sing me", "Make up", "Teach me") else opener
    return BRIEF.format(system=s["system"], partner=s["partner"], subject=s["subject"], opener=opener,
                        n=2 * s["exchanges"])


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
    return True


def gen(a):
    url = os.environ.get("OLLAMA_CHAT_URL", "http://127.0.0.1:11435/api/chat")
    rows = [json.loads(l) for l in open(os.path.join(D, "specs.jsonl"))]
    rows = random.Random(5).sample(rows, a.sample) if a.sample else rows
    if a.part:
        k, n = map(int, a.part.split("/"))
        rows = rows[k::n]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    rows = [r for r in rows if r["id"] not in done]

    def one(s):
        body = {"model": a.model, "stream": False, "format": "json", "think": False,
                "options": {"temperature": 0.9, "num_predict": 700},
                "messages": [{"role": "user", "content": brief(s)}]}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        for _ in range(3):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=300).read())
                turns = json.loads(r["message"]["content"]).get("turns")
                if valid(s, turns):
                    return s, tidy(turns), r.get("eval_count", 0)
            except Exception:
                time.sleep(1)
        return s, None, 0

    t0, ok, toks = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for s, turns, n in ex.map(one, rows):
            toks += n
            if turns:
                ok += 1
                f.write(json.dumps({"id": s["id"], "turns": turns}) + "\n")
                f.flush()
    el = time.time() - t0
    print(f"{a.model}: {ok}/{len(rows)} valid in {el:.0f}s, {len(rows) / el * 60:.1f} convs/min, {toks / el:.0f} tok/s")


def merge():
    spec = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, "specs.jsonl"))}
    rows, bad, seen = [], 0, set()
    for p in sorted(glob.glob(os.path.join(D, "out_*.jsonl"))):
        for line in open(p):
            d = json.loads(line)
            s = spec.get(d.get("id"))
            if not s or d["id"] in seen or not valid(s, d.get("turns")):
                bad += 1
                continue
            seen.add(d["id"])
            rows.append({**{k: s[k] for k in ("id", "persona", "name", "user_kind", "partner", "thing", "system")},
                         "turns": d["turns"], "source": os.path.basename(p)})
    with open(os.path.join(D, "chat.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows):,} family chats -> chat.jsonl, {bad} rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["specs", "gen", "merge"])
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--part", help="k/n: every nth spec starting at k, to split the work between models")
    a = ap.parse_args()
    {"specs": specs, "merge": merge}.get(a.cmd, lambda: gen(a))()
