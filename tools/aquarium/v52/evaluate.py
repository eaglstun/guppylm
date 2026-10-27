"""Run a frozen behavior contract and export blinded human-review material.

Locked-test evaluation requires a selection record identifying the selected weights
and contract; development remains the default. No checkpoint selection on test scores.
"""
import argparse
import collections
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from .. import train_aq as aq
from .contract import contains
from .data import digest, file_hash, normalized, read_manifest


def cluster_ci(values, seed=5204):
    """Bootstrap prompt means: repeated samples from a prompt are one cluster."""
    means = np.asarray([np.mean(v) for v in values], dtype=float)
    if not len(means):
        return None
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, len(means), (2000, len(means)))].mean(axis=1)
    mean, n = float(means.mean()), len(means)
    # Bootstrap alone degenerates to [0,0] after zero hits. Envelope it with a
    # Wilson interval at the prompt-cluster effective sample size.
    z = 1.959963984540054
    center = (mean + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(mean*(1-mean)/n + z*z/(4*n*n)) / (1 + z*z/n)
    bootstrap = np.quantile(boot, [.025, .975]).tolist()
    return {"mean": mean, "ci95": [max(0., min(bootstrap[0], center-radius)), min(1., max(bootstrap[1], center+radius))],
            "bootstrap_ci95": bootstrap,
            "prompts": len(means), "samples": sum(map(len, values))}


@torch.no_grad()
def reply(model, tok, system, turns, device, decoding):
    ids, _ = aq.chat_ids(tok, system, turns, open_assistant=True)
    # Same context reservation as the browser: retain recent tokens for an 80-token reply.
    prompt = ids[-(model.config.max_seq_len - decoding["max_new"]):]
    idx = torch.tensor([prompt], device=device)
    past, generated = None, []
    model.eval()
    for _ in range(decoding["max_new"]):
        logits, past = model.forward_cached(idx if past is None else idx[:, -1:], past)
        logits = logits[:, -1, :] / decoding["temperature"]
        v, _ = torch.topk(logits, min(decoding["top_k"], logits.shape[-1]))
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        if nxt.item() == aq.IM_END:
            break
        generated.append(nxt.item())
        idx = torch.cat([idx, nxt], 1)
    return tok.decode(generated)


class Lookup:
    """Lexical retrieval baseline, never trained on evaluation responses."""
    def __init__(self, rows):
        self.pairs, self.index = [], collections.defaultdict(set)
        stop = set("a an the i you your my is are was it to in of and what how do does me can".split())
        self.words = lambda s: set(normalized(s).split()) - stop
        for row in rows:
            for i in range(0, len(row["turns"]), 2):
                q, answer = row["turns"][i:i + 2]
                words = self.words(q["content"])
                index = len(self.pairs)
                self.pairs.append((words, answer["content"]))
                for w in words:
                    self.index[w].add(index)

    def __call__(self, query):
        words = self.words(query)
        candidates = set()
        for word in sorted(words, key=lambda w: len(self.index.get(w, ())))[:5]:
            candidates.update(self.index.get(word, ()))
        if not candidates:
            return "Hello from the tide pool."
        best = max(sorted(candidates), key=lambda i: len(words & self.pairs[i][0]) / max(1, len(words | self.pairs[i][0])))
        return self.pairs[best][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--train", type=Path, help="Use retrieval baseline instead of model")
    ap.add_argument("--blind-baseline", action="store_true")
    ap.add_argument("--split", choices=("development", "locked_test"), default="development")
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if sum((bool(a.checkpoint), bool(a.train), a.blind_baseline)) != 1:
        ap.error("Choose exactly one of checkpoint, train (lookup), or blind-baseline")
    contract = json.loads(a.contract.read_text())
    ck_hash = file_hash(a.checkpoint) if a.checkpoint else None
    if a.split == "locked_test":
        if not a.selection:
            ap.error("Freeze --selection before opening locked test")
        selection = json.loads(a.selection.read_text())
        if (selection.get("contract_sha256") != digest(contract) or ck_hash is None
                or selection.get("checkpoint_sha256") != ck_hash or not selection.get("development_report_sha256")
                or not selection.get("rubric_sha256")):
            ap.error("Selection must identify contract, checkpoint, development report and rubric hashes")
    if a.out.exists():
        raise ValueError("Refusing to overwrite an evaluation result")
    torch.set_num_threads(a.threads)
    model, tok, lookup = None, None, None
    if a.checkpoint:
        if not a.tokenizer:
            ap.error("--tokenizer is required for checkpoint evaluation")
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = aq.GuppyLM(aq.GuppyConfig(**ck["config"])).to(a.device)
        model.load_state_dict(ck["model"])
        tok = Tokenizer.from_file(str(a.tokenizer))
        if model.config.vocab_size != tok.get_vocab_size() or contract["decoding"]["max_new"] >= model.config.max_seq_len:
            raise ValueError("Incompatible tokenizer/context budget")
    if a.train:
        train, _, _ = read_manifest(a.train, "train")
        lookup = Lookup(train)
    results = []
    cases = [r for r in contract["cases"] if r["split"] == a.split]
    for index, case in enumerate(cases):
        for repetition in range(contract["decoding"]["samples"]):
            torch.manual_seed(case["seed"] + repetition)
            turns = list(case["context"])
            for user in case["users"]:
                turns.append({"role": "user", "content": user})
                answer = (reply(model, tok, case["system"], turns, a.device, contract["decoding"]) if model else
                          lookup(user) if lookup else "Hello from the tide pool.")
                turns.append({"role": "assistant", "content": answer})
            hit = contains(answer, case["expected"]) if case["expected"] is not None else None
            results.append({"prompt_id": case["uid"], "slice": case["slice"], "repetition": repetition,
                            "system": case["system"], "turns": turns, "expected": case["expected"], "hit": hit})
        if (index + 1) % 40 == 0:
            print(f"evaluated {index + 1}/{len(cases)} prompts", flush=True)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        if r["hit"] is not None:
            groups[r["slice"]][r["prompt_id"]].append(int(r["hit"]))
    summary = {s: cluster_ci(list(by_prompt.values())) for s, by_prompt in groups.items()}
    payload = {"contract_sha256": digest(contract), "checkpoint_sha256": ck_hash,
               "tokenizer_sha256": file_hash(a.tokenizer) if a.tokenizer else None,
               "lookup_manifest_sha256": read_manifest(a.train)[1] if a.train else None,
               "split": a.split, "decoding": contract["decoding"], "summary": summary,
               "manual_slices": sorted({r["slice"] for r in results if r["hit"] is None}), "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1) + "\n")
    # One deterministic sample per prompt, shuffled and stripped of model/checkpoint labels.
    reviews = [{"review_id": digest([r["prompt_id"], r["repetition"], ck_hash]),
                "system": r["system"], "turns": r["turns"],
                "coherence_0_2": None, "voice_0_2": None, "task_correctness_0_2": None}
               for r in results if r["repetition"] == 0]
    random.Random(5205).shuffle(reviews)
    a.out.with_suffix(".review.json").write_text(json.dumps(reviews, indent=1) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
