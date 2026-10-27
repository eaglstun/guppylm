"""Manifest-driven Aquarium trainer with faithful same-device checkpoint resume.

The learning rate is a pure function of completed steps and the frozen configuration.
--stop-after interrupts without changing the original cosine schedule.
"""
import argparse
import collections
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from .. import train_aq as aq
from .data import assert_disjoint, digest, file_hash, read_manifest


def capture_rng(rng, prng, device):
    state = {"numpy": copy.deepcopy(rng.bit_generator.state), "python": prng.getstate(),
             "torch": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, rng, prng, device):
    rng.bit_generator.state = state["numpy"]
    prng.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"].cpu())


def save_atomic(value, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


@torch.no_grad()
def losses(model, pairs, sources, seq, device, batch=32):
    """Token-weighted assistant NLL for every row, including short final batches."""
    model.eval()
    totals = collections.defaultdict(lambda: [0., 0])
    batch_losses = []
    groups = collections.defaultdict(list)
    for pair, source in zip(pairs, sources):
        groups[source].append(pair)
    for source, examples in groups.items():
        for offset in range(0, len(examples), batch):
            x, y = aq.pad(examples[offset:offset + batch], seq)
            n = int((y != 0).sum())
            if n:
                loss = model(x.to(device), y.to(device))[1].item()
                batch_losses.append(loss)
                totals[source][0] += loss * n
                totals[source][1] += n
    total_tokens = sum(n for _, n in totals.values())
    result = {"loss": sum(v for v, _ in totals.values()) / total_tokens if total_tokens else None,
              "tokens": total_tokens,
              "legacy_mean_batch_loss": sum(batch_losses) / len(batch_losses) if batch_losses else None,
              "sources": {s: {"loss": v / n, "tokens": n} for s, (v, n) in totals.items()}}
    model.train()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--development", type=Path, required=True)
    ap.add_argument("--historical", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--dropout", type=float, default=.1)
    ap.add_argument("--story-mix", type=float, default=.15)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=("cpu", "mps"), default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    n_story = round(a.batch * a.story_mix)
    if (a.steps < 1 or a.batch < 1 or a.seq < 1 or a.eval_every < 1 or not 0 <= a.story_mix < 1
            or n_story >= a.batch or a.lr <= 0 or a.warmup < 0
            or a.stop_after is not None and not 0 <= a.stop_after <= a.steps):
        ap.error("Invalid training budget, batch, story mixture, or learning rate")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    rng, prng = np.random.default_rng(a.seed), random.Random(a.seed)
    train, train_hash, _ = read_manifest(a.train, "train")
    dev, dev_hash, _ = read_manifest(a.development, "development")
    assert_disjoint(train, dev)
    historical, historical_hash = [], None
    if a.historical:
        historical, historical_hash, _ = read_manifest(a.historical, "historical")
    if not train or not dev:
        raise ValueError("Training and development manifests must be nonempty")
    tok = Tokenizer.from_file(str(a.data / "tokenizer.json"))
    init = torch.load(a.init, map_location="cpu", weights_only=False)
    config = aq.GuppyConfig(**{**init["config"], "dropout": a.dropout})
    if config.max_seq_len != a.seq or config.vocab_size != tok.get_vocab_size():
        raise ValueError("Tokenizer/sequence length must match the starting architecture")
    model = aq.GuppyLM(config).to(device)
    model.load_state_dict(init["model"])
    del init
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.1, betas=(.9, .95))
    train_ex = [aq.encode_chat(tok, row, a.seq) for row in train]
    dev_ex = [aq.encode_chat(tok, row, a.seq) for row in dev]
    held_ex = [aq.encode_chat(tok, row, a.seq) for row in historical]
    if any(not any(y) for _, y in train_ex + dev_ex + held_ex):
        raise ValueError("Conversation has no assistant targets after truncation")
    stories = aq.Stories(str(a.data / "ts_train.bin"), a.seq)
    stories_val = aq.Stories(str(a.data / "ts_valid.bin"), a.seq)
    vrng = np.random.default_rng(5203)
    story_pairs = []
    for _ in range(20):
        x, y = stories_val.batch(vrng, a.batch)
        story_pairs.extend(zip(x.tolist(), y.tolist()))
    recipe = {k: v for k, v in vars(a).items() if k not in (
        "train", "development", "historical", "data", "init", "out", "resume", "stop_after")}
    recipe.update(train_hash=train_hash, dev_hash=dev_hash, historical_hash=historical_hash,
                  init_hash=file_hash(a.init), tokenizer_hash=file_hash(a.data / "tokenizer.json"),
                  stories_hash=file_hash(a.data / "ts_train.bin"), story_val_hash=file_hash(a.data / "ts_valid.bin"),
                  trainer_hash=file_hash(__file__), model_hash=file_hash(Path(aq.ROOT) / "guppylm/model.py"),
                  legacy_hash=file_hash(aq.__file__), torch_version=torch.__version__, numpy_version=np.__version__)
    fingerprint = digest(recipe)
    order, step, best, history = [], 0, float("inf"), []
    counts = collections.defaultdict(lambda: {"examples": 0, "input_tokens": 0, "target_tokens": 0})
    if a.resume:
        if a.resume.parent.resolve() != a.out.resolve():
            raise ValueError("Resume in the original output directory to preserve the selected best checkpoint")
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        if ck.get("fingerprint") != fingerprint:
            raise ValueError("Resume recipe/data/runtime mismatch; use --init for a separate continuation")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step, best, order, history = ck["step"], ck["best"], ck["order"], ck["history"]
        counts.update(ck["sampling"])
        restore_rng(ck["rng"], rng, prng, device)
    elif (a.out / "final.pt").exists():
        raise ValueError("Output already has a checkpoint; use --resume or a fresh directory")
    a.out.mkdir(parents=True, exist_ok=True)
    inventory = collections.defaultdict(lambda: {"rows": 0, "input_tokens": 0, "assistant_tokens": 0, "truncated_rows": 0})
    for row, (x, y) in zip(train, train_ex):
        inv = inventory[row["source"]]
        inv["rows"] += 1
        inv["input_tokens"] += len(x)
        inv["assistant_tokens"] += sum(t != 0 for t in y)
        inv["truncated_rows"] += len(aq.chat_ids(tok, row["system"], row["turns"])[0]) > a.seq + 1
    (a.out / "recipe.json").write_text(json.dumps({"recipe": recipe, "fingerprint": fingerprint,
        "params": sum(p.numel() for p in model.parameters()), "inventory": inventory}, indent=2))
    end = a.steps if a.stop_after is None else a.stop_after
    if end < step:
        raise ValueError("stop-after precedes resumed step")
    start_time = time.monotonic()
    model.train()
    while True:
        due = step % a.eval_every == 0 or step == a.steps
        improved = False
        if due and (not history or history[-1]["step"] != step):
            state = capture_rng(rng, prng, device)
            ev = {"step": step, "development": losses(model, dev_ex, [r["source"] for r in dev], a.seq, device, a.batch),
                  "story": losses(model, story_pairs, ["story"] * len(story_pairs), a.seq, device, a.batch)}
            if held_ex:
                ev["historical"] = losses(model, held_ex, ["historical"] * len(held_ex), a.seq, device, a.batch)
            restore_rng(state, rng, prng, device)
            score = ev["development"]["loss"]
            if not math.isfinite(score):
                raise ValueError("Non-finite development loss")
            improved = score < best
            best = min(score, best)
            history.append(ev)
            print(json.dumps(ev), flush=True)
        if due or step == end:
            ck = {"model": model.state_dict(), "config": vars(config), "step": step,
                  "optimizer": optimizer.state_dict(), "rng": capture_rng(rng, prng, device),
                  "order": order, "best": best, "history": history, "sampling": dict(counts),
                  "recipe": recipe, "fingerprint": fingerprint,
                  "scheduler": {"kind": "warmup_cosine", "next_step": step, "total_steps": a.steps}}
            save_atomic(ck, a.out / "final.pt")
            if improved:
                save_atomic(ck, a.out / "best.pt")
            (a.out / "log.json").write_text(json.dumps({"history": history, "sampling": counts}, indent=1))
        if step == end:
            break
        n_chat = a.batch - n_story
        while len(order) < n_chat:
            order.extend(prng.sample(range(len(train_ex)), len(train_ex)))
        pick, order = order[:n_chat], order[n_chat:]
        x, y = aq.pad([train_ex[i] for i in pick], a.seq)
        for i in pick:
            count = counts[train[i]["source"]]
            xx, yy = train_ex[i]
            count["examples"] += 1
            count["input_tokens"] += len(xx)
            count["target_tokens"] += sum(t != 0 for t in yy)
        if n_story:
            sx, sy = stories.batch(rng, n_story)
            x = torch.cat([torch.nn.functional.pad(x, (0, a.seq - x.shape[1])), sx])
            y = torch.cat([torch.nn.functional.pad(y, (0, a.seq - y.shape[1])), sy])
            counts["story"]["examples"] += n_story
            counts["story"]["input_tokens"] += n_story * a.seq
            counts["story"]["target_tokens"] += int((sy != 0).sum())
        for group in optimizer.param_groups:
            group["lr"] = aq.lr_at(step, a)
        loss = model(x.to(device), y.to(device))[1]
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f} session_seconds={time.monotonic()-start_time:.1f}", flush=True)


if __name__ == "__main__":
    main()
