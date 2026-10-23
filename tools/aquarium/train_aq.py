"""Two-stage training for the aquarium fish: TinyStories pretraining, then persona chat.

    python3 tools/aquarium/train_aq.py pretrain --out runs/aq-pre --steps 20000
    python3 tools/aquarium/train_aq.py finetune --init runs/aq-pre/final.pt --out runs/aq-chat --steps 6000

pretrain: random 512-token windows of packed stories, loss on every token.
finetune: one dialogue per row, loss only on assistant replies (and their <|im_end|>),
          with a slice of story windows mixed in so the English doesn't erode.
"""

import argparse
import glob
import json
import math
import re
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer  # noqa: E402

from guppylm.config import GuppyConfig  # noqa: E402
from guppylm.model import GuppyLM  # noqa: E402
from personas import EVAL_NAMES, make_spec  # noqa: E402

D = os.path.join(ROOT, "data_aq")
IM_START, IM_END = 1, 2
SAMPLE_SYSTEMS = [
    ("You are Pebble, a hermit crab who lives in a sandy tide pool. You are careful and practical.",
     ["Hi! What are you doing today?", "What is a toaster?"]),
    ("You are Quilliam the narwhal. Shy, and proud of your long tusk. You live in cold arctic water under the ice.",
     ["Who are you?"]),
]


def load_dialogues(paths):
    """Rows with system + turns. Haiku rows carry only id + turns; rebuild their spec from the id."""
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "system" not in d:
                    d = {**make_spec(random.Random(d["id"])), **d}
                turns = d.get("turns") or []
                if len(turns) >= 2 and all(t.get("role") == ("user", "assistant")[i % 2]
                                           and isinstance(t.get("content"), str)
                                           for i, t in enumerate(turns)):
                    rows.append(d)
    return rows


SYLLABLES = ("bri del wick zan mor pel quin tob sar lum fen gor vix nal tup wib rok jas pim "
             "kel dro mun vel sto zib car thu lop nim gra bel orn ush").split()


# A quarter of invented names start with a real word (Pebbleton, Cloverquin), so the model
# learns not to stop copying at the first word boundary. Stems that begin a held-out name
# (Marble, Wobble, Quill, Fizz, Zephyr, Tup) are left out.
NAME_STEMS = ("pebble bubble coral kelp sandy puddle thistle copper button pickle maple cobble ripple "
              "tumble bramble muffin pepper ginger clover sunny biscuit noodle dapple fiddle crumble "
              "waffle nibble doodle jingle sprinkle barnacle bucket cricket hazel puffin velvet").split()
NAME_TAILS = "ton quin wick by ford worth kin more ington ette ius bert ina ham ley ster bottom fern".split()


def made_up_name(rng):
    while True:
        if rng.random() < 0.25:
            name = (rng.choice(NAME_STEMS) + rng.choice(NAME_TAILS)).capitalize()
        else:
            name = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4))).capitalize()
        if name not in EVAL_NAMES:
            return name


def renamed(row, rng):
    """The same dialogue with the creature's name swapped for an invented one everywhere."""
    new = made_up_name(rng)
    pat = re.compile(rf"\b{re.escape(row['name'])}\b")
    return {**row, "name": new, "system": pat.sub(new, row["system"]),
            "turns": [{**t, "content": pat.sub(new, t["content"])} for t in row["turns"]]}


NOUN_SYLLABLES = "glim bet snor ket wob ble quaz tro fip nik dax plon zor mib vant rux".split()


def made_up_noun(rng):
    return "".join(rng.choice(NOUN_SYLLABLES) for _ in range(rng.randint(2, 3)))


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def noun_swapped(row, rng):
    """The same dialogue with the mentioned human thing swapped for an invented noun."""
    words = row["thing"].split(" ", 1)
    phrase = words[1] if words[0] in ("a", "an", "the") and len(words) > 1 else row["thing"]
    new = made_up_noun(rng)
    with_art = re.compile(rf"\b(a|an|A|An)\s+{re.escape(phrase)}\b", re.I)
    bare = re.compile(rf"\b{re.escape(phrase)}\b", re.I)

    def sub(text):
        text = with_art.sub(lambda m: (article(new).capitalize() if m.group(1)[0] == "A" else article(new)) + " " + new, text)
        return bare.sub(new, text)

    return {**row, "thing": new, "turns": [{**t, "content": sub(t["content"])} for t in row["turns"]]}


# Facts the user states up front and is asked about at the end. Values that the memory
# probe uses (Bartholomew, purple, seven, Gus, shell, brothers) are kept out.
COLORS = "red orange yellow green blue pink white black silver gold brown gray teal golden".split()
OBJECTS = "rock feather acorn button leaf marble spoon key ribbon cup ball sock coin".split()
NUMBERS = "two three four five six eight nine ten eleven twelve".split()
THINGS_OWNED = "cats sisters hats books kites bikes cousins goldfish plants toys".split()
ANIMALS = "a crab, a dolphin, a seal, a turtle, an otter, a puffin, a frog, a duck".split(", ")


def load_memory_pool():
    """(persona, kind) -> Haiku-written {fact, question, answer} with placeholders."""
    pool = {}
    for path in glob.glob(os.path.join(D, "memory", "out_*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                need = {"name": "{V}", "color": "{C}", "number": "{N}", "friend": "{V}"}.get(d.get("kind"))
                if need and all(isinstance(d.get(k), str) for k in ("fact", "question", "answer")) and need in d["answer"]:
                    # "What name shall we go with?" could be about either speaker; keep only
                    # name questions that are plainly about the user.
                    if d["kind"] == "name" and not re.search(r"\b(my|me|I)\b", d["question"]):
                        continue
                    pool.setdefault((d["persona"], d["kind"]), []).append(d)
    return pool


MEMORY_POOL = {}


def load_selfname_pool():
    """persona -> Haiku-written {question, answer} where the creature says its own name {V}."""
    pool = {}
    for path in glob.glob(os.path.join(D, "selfname", "out_*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (isinstance(d.get("question"), str) and isinstance(d.get("answer"), str)
                        and "{V}" in d["answer"] and not re.search(r"\b(You are|Your name)\b", d["answer"])):
                    pool.setdefault(d["persona"], []).append(d)
    return pool


SELFNAME_POOL = {}


def asked_name(row, rng):
    """The dialogue with the user asking the creature its own name at the end.

    Balances the memory copies' "What is my name?" -> "Your name is ...", which had the
    model answering "What is your name?" with "You are ...". A third of the time the user
    also states their own name first, so both questions meet in one conversation."""
    options = SELFNAME_POOL.get(row.get("persona"))
    if not options:
        return row
    if rng.random() < 1 / 3:
        row = remembered(row, rng, kind="name")
    t = rng.choice(options)
    turns = [dict(x) for x in row["turns"]]
    return {**row, "turns": turns + [{"role": "user", "content": t["question"]},
                                     {"role": "assistant", "content": t["answer"].replace("{V}", row["name"])}]}


def remembered(row, rng, kind=None):
    """The dialogue with a fact stated in the first user turn and asked about at the end.

    Wording comes from the Haiku-written pool for this persona when there is one, so the
    answers vary in each creature's voice; the old fixed templates leaked ("I remember.")."""
    kind = kind or rng.choice(["name", "color", "number", "friend"])
    vals = {"V": made_up_name(rng), "C": rng.choice(COLORS), "O": rng.choice(OBJECTS),
            "N": rng.choice(NUMBERS), "T": rng.choice(THINGS_OWNED), "A": rng.choice(ANIMALS)}
    options = MEMORY_POOL.get((row.get("persona"), kind))
    if options:
        t = rng.choice(options)
        fill = lambda s: re.sub(r"\{([VCONTA])\}", lambda m: vals[m.group(1)], s)  # noqa: E731
        fact, q, a = fill(t["fact"]), fill(t["question"]), fill(t["answer"])
    else:
        fact, q, a = _template_memory(kind, vals, rng)
    turns = [dict(t) for t in row["turns"]]
    turns[0]["content"] = fact + " " + turns[0]["content"]
    return {**row, "turns": turns + [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def _template_memory(kind, v, rng):
    if kind == "name":
        return (f"My name is {v['V']}.", "What is my name?", f"Your name is {v['V']}.")
    if kind == "color":
        return (f"I found a {v['C']} {v['O']} today.", f"What color was the {v['O']}?", f"It was {v['C']}.")
    if kind == "number":
        return (f"I have {v['N']} {v['T']}.", f"How many {v['T']} do I have?", f"You have {v['N']} {v['T']}.")
    return (f"My friend is {v['A']} named {v['V']}.", "What is my friend's name?", f"Your friend is {v['V']}.")


def chat_ids(tok, system, turns, open_assistant=False):
    """Token ids and a per-token "train on this" mask for a chat. The same builder is
    used for training and sampling, so both see identical token boundaries."""
    ids, mask = [], []

    def add(tokens, on=False):
        ids.extend(tokens)
        mask.extend([on] * len(tokens))

    add([IM_START] + tok.encode("system\n" + system).ids + [IM_END] + tok.encode("\n").ids)
    for t in turns:
        add([IM_START] + tok.encode(t["role"] + "\n").ids)
        add(tok.encode(t["content"]).ids + [IM_END], t["role"] == "assistant")
        add(tok.encode("\n").ids)
    if open_assistant:
        add([IM_START] + tok.encode("assistant\n").ids)
    return ids, mask


def encode_chat(tok, row, max_len):
    """(x, y) with y = 0 (ignored) everywhere except assistant replies."""
    ids, mask = chat_ids(tok, row["system"], row["turns"])
    ids, mask = ids[:max_len + 1], mask[:max_len + 1]
    return ids[:-1], [t if mask[i + 1] else 0 for i, t in enumerate(ids[1:])]


class Stories:
    def __init__(self, path, seq):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq = seq

    def batch(self, rng, n):
        starts = rng.integers(0, len(self.data) - self.seq - 1, n)
        w = np.stack([self.data[s:s + self.seq + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:])


def pad(pairs, seq):
    T = min(seq, max(len(x) for x, _ in pairs))
    X = torch.zeros(len(pairs), T, dtype=torch.long)
    Y = torch.zeros(len(pairs), T, dtype=torch.long)
    for i, (x, y) in enumerate(pairs):
        X[i, :len(x)] = torch.tensor(x[:T])
        Y[i, :len(y)] = torch.tensor(y[:T])
    return X, Y


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * step / a.warmup
    p = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * 0.1 + (a.lr * 0.9) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def sample(model, tok, device, system, users, max_new=80, temperature=0.7, top_k=40):
    model.eval()
    turns, out = [], []
    for u in users:
        turns.append({"role": "user", "content": u})
        ids, _ = chat_ids(tok, system, turns, open_assistant=True)
        idx = torch.tensor([ids[-model.config.max_seq_len:]], device=device)
        new = []
        for _ in range(max_new):
            logits, _ = model(idx[:, -model.config.max_seq_len:])
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            if nxt.item() == IM_END:
                break
            new.append(nxt.item())
            idx = torch.cat([idx, nxt], 1)
        reply = tok.decode(new)
        out.append((u, reply))
        turns.append({"role": "assistant", "content": reply})
    model.train()
    return out


@torch.no_grad()
def story_sample(model, tok, device, prompt="Once upon a time, a little fish", n=80):
    model.eval()
    idx = torch.tensor([tok.encode(prompt).ids], device=device)
    for _ in range(n):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / 0.7
        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        if nxt.item() == 3:
            break
        idx = torch.cat([idx, nxt], 1)
    model.train()
    return tok.decode(idx[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pretrain", "finetune"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--story-mix", type=float, default=0.15, help="finetune: share of story rows per batch")
    ap.add_argument("--rename-copies", type=int, default=0,
                    help="finetune: extra copies of each train dialogue with an invented creature name")
    ap.add_argument("--noun-copies", type=int, default=0,
                    help="finetune: extra copies of dialogues that mention a human thing, swapped for an invented noun")
    ap.add_argument("--memory-copies", type=int, default=0,
                    help="finetune: extra copies with a user fact stated first and asked about last")
    ap.add_argument("--selfname-copies", type=int, default=0,
                    help="finetune: extra copies with the creature asked its own name at the end")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    pre = a.mode == "pretrain"
    a.lr = a.lr or (6e-4 if pre else 2e-4)
    a.dropout = a.dropout if a.dropout is not None else (0.0 if pre else 0.1)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    prng = random.Random(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
    mc = GuppyConfig(vocab_size=tok.get_vocab_size(), max_seq_len=a.seq, dropout=a.dropout)
    model = GuppyLM(mc).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["model"])
    nparams = sum(p.numel() for p in model.parameters())
    print(f"{a.mode}: {nparams:,} params, device {device}", flush=True)

    stories = Stories(os.path.join(D, "ts_train.bin"), a.seq)
    stories_val = Stories(os.path.join(D, "ts_valid.bin"), a.seq)
    if not pre:
        rows = load_dialogues([os.path.join(D, "dialogues.jsonl")] + glob.glob(os.path.join(D, "haiku", "out_*.jsonl")))
        prng.shuffle(rows)
        n_val = max(200, len(rows) // 50)
        val_rows, train_rows = rows[:n_val], rows[n_val:]
        held = load_dialogues([os.path.join(D, "dialogues_eval.jsonl")]) if os.path.exists(
            os.path.join(D, "dialogues_eval.jsonl")) else []
        MEMORY_POOL.update(load_memory_pool())
        print(f"memory pool: {sum(len(v) for v in MEMORY_POOL.values())} Haiku triples, "
              f"{len(MEMORY_POOL)} persona/kind pairs", flush=True)
        SELFNAME_POOL.update(load_selfname_pool())
        print(f"self-name pool: {sum(len(v) for v in SELFNAME_POOL.values())} Haiku pairs, "
              f"{len(SELFNAME_POOL)} personas", flush=True)
        base = train_rows
        nounable = [r for r in base if r.get("thing")]
        train_rows = (base + [renamed(r, prng) for _ in range(a.rename_copies) for r in base]
                      + [noun_swapped(r, prng) for _ in range(a.noun_copies) for r in nounable]
                      + [remembered(renamed(r, prng), prng) for _ in range(a.memory_copies) for r in base]
                      + [asked_name(renamed(r, prng), prng) for _ in range(a.selfname_copies) for r in base])
        enc = lambda rs: [encode_chat(tok, r, a.seq) for r in rs]  # noqa: E731
        train_ex, val_ex, held_ex = enc(train_rows), enc(val_rows), enc(held)
        print(f"dialogues: train {len(train_ex):,}, val {len(val_ex):,}, held-out personas {len(held_ex):,}; "
              f"assistant tokens {sum(sum(1 for t in y if t) for _, y in train_ex):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    os.makedirs(a.out, exist_ok=True)
    log = {"args": vars(a), "params": nparams, "history": []}

    def eval_loss(batches):
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for X, Y in batches:
                tot += model(X.to(device), Y.to(device))[1].item()
        model.train()
        return tot / len(batches)

    def chat_batches(ex, bs=32, limit=20):
        return [pad(ex[i:i + bs], a.seq) for i in range(0, min(len(ex), bs * limit), bs)]

    vrng = np.random.default_rng(0)
    story_val = [stories_val.batch(vrng, a.batch) for _ in range(20)]
    t0 = time.time()
    model.train()
    order = []
    best = float("inf")
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            ev = {"step": step, "story_val": eval_loss(story_val), "elapsed_s": round(time.time() - t0)}
            if not pre:
                ev["chat_val"] = eval_loss(chat_batches(val_ex))
                if held_ex:
                    ev["held_out_personas"] = eval_loss(chat_batches(held_ex))
            log["history"].append(ev)
            print(json.dumps(ev), flush=True)
            if pre:
                print("  story:", story_sample(model, tok, device)[:300].replace("\n", " "), flush=True)
            else:
                for system, users in SAMPLE_SYSTEMS:
                    for u, r in sample(model, tok, device, system, users):
                        print(f"  > {u}\n    {r}", flush=True)
            ck = {"model": model.state_dict(), "config": vars(mc), "step": step}
            torch.save(ck, os.path.join(a.out, "final.pt"))
            score = ev["story_val"] if pre else ev["chat_val"]
            if score < best:
                best = score
                torch.save(ck, os.path.join(a.out, "best.pt"))
            with open(os.path.join(a.out, "log.json"), "w") as f:
                json.dump(log, f, indent=1)
        if step == a.steps:
            break

        if pre:
            X, Y = stories.batch(rng, a.batch)
        else:
            n_story = int(round(a.batch * a.story_mix))
            if len(order) < a.batch:
                order += prng.sample(range(len(train_ex)), len(train_ex))
            pick, order = order[:a.batch - n_story], order[a.batch - n_story:]
            X, Y = pad([train_ex[i] for i in pick], a.seq)
            if n_story:
                SX, SY = stories.batch(rng, n_story)
                T = max(X.shape[1], SX.shape[1])
                X = torch.cat([torch.nn.functional.pad(X, (0, T - X.shape[1])), SX[:, :T]])
                Y = torch.cat([torch.nn.functional.pad(Y, (0, T - Y.shape[1])), SY[:, :T]])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        _, loss = model(X.to(device), Y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 200 == 0:
            el = time.time() - t0
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step, a):.2e} {step / max(el, 1e-9):.2f} it/s", flush=True)


if __name__ == "__main__":
    main()
