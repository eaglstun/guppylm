"""Write multi-turn aquatic persona dialogues with a local Ollama model.

Each dialogue id seeds its own spec (persona, name, partner, topic, length), so the
run is resumable: ids already in the output file are skipped.

    python3 tools/aquarium/gen_dialogues.py --n 30000 --out data_aq/dialogues.jsonl
    python3 tools/aquarium/gen_dialogues.py --n 300 --eval --out data_aq/dialogues_eval.jsonl
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personas import PERSONAS, make_spec  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")

BRIEF = """Write a short conversation for training a tiny language model.

The assistant is a sea creature. Its system prompt is:
\"\"\"{system}\"\"\"

The user is {partner}.{creature_note}
Topic to start from: {topic}.{thing_note}
Write exactly {n_msgs} messages, alternating user and assistant, starting with the user.

Rules:
- Use simple words a five-year-old knows, like a children's story. Short sentences.
- Each message is 1 to 3 sentences.
- The assistant stays in character as {species} named {name}. It knows only its own world: {habitat}. It never mentions being an AI.
- The assistant's personality, in the system prompt's words: {traits}.
- If the assistant does not know a human thing, it says the thing's exact name and guesses what it might be from its own world.
- In later exchanges, the user or assistant should refer back to something said earlier.
- Normal capitalization and punctuation.

Return JSON only, filling in every "content" of this exact skeleton:
{skeleton}"""


def brief(spec):
    species, habitat, traits = PERSONAS[spec["persona"]]
    creature_note = (f" The user speaks as {spec['partner']}, in its own voice, about its own life."
                     if spec["user_kind"] == "creature" else "")
    thing_note = (f" At some point the user mentions {spec['thing']}."
                  if spec["thing"] else "")
    skeleton = json.dumps({"turns": [{"role": ("user", "assistant")[i % 2], "content": "..."}
                                     for i in range(2 * spec["exchanges"])]})
    return BRIEF.format(system=spec["system"], n_msgs=2 * spec["exchanges"], skeleton=skeleton, partner=spec["partner"],
                        creature_note=creature_note, topic=spec["topic"], thing_note=thing_note,
                        exchanges=spec["exchanges"], species=species, name=spec["name"],
                        habitat=habitat, traits=traits)


def valid(turns):
    if not isinstance(turns, list) or len(turns) < 2 or len(turns) > 8 or len(turns) % 2:
        return False
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or t.get("role") != ("user", "assistant")[i % 2]:
            return False
        c = t.get("content")
        if not isinstance(c, str) or not c.strip() or len(c) > 400:
            return False
        if "AI" in c.split() or "language model" in c.lower():
            return False
    return True


def generate(model, spec, temperature):
    body = json.dumps({
        "model": model, "stream": False, "format": "json",
        "messages": [{"role": "user", "content": brief(spec)}],
        "options": {"temperature": temperature, "num_predict": 900, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    turns = json.loads(resp["message"]["content"]).get("turns")
    return turns, resp.get("eval_count", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", action="store_true", help="held-out personas and names")
    ap.add_argument("--model", default="qwen3-vl:30b-a3b-instruct")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    base = 10_000_000 if args.eval else 0
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [base + i for i in range(args.n) if base + i not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    lock = threading.Lock()
    stats = {"ok": 0, "bad": 0, "tokens": 0}
    t0 = time.time()

    def work(did):
        spec = make_spec(random.Random(did), eval_split=args.eval)
        for attempt in range(3):
            try:
                turns, ntok = generate(args.model, spec, args.temperature)
            except Exception:
                turns, ntok = None, 0
            with lock:
                stats["tokens"] += ntok
            if valid(turns):
                row = {"id": did, **spec, "exchanges": len(turns) // 2, "turns": [{"role": t["role"], "content": t["content"].strip()}
                                                    for t in turns]}
                with lock:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    stats["ok"] += 1
                    n = stats["ok"]
                    if n % 50 == 0:
                        el = time.time() - t0
                        print(f"{n} ok, {stats['bad']} rejected, {stats['tokens'] / el:.0f} tok/s, "
                              f"{n / el * 3600:.0f}/h", flush=True)
                return
        with lock:
            stats["bad"] += 1

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    print(f"finished: {stats}", flush=True)


if __name__ == "__main__":
    main()
