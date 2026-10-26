"""Recover a five-layer aquarium model by distilling a six-layer teacher.

    python3 tools/aquarium/train_pruned.py runs/aq-chat-v5/best.pt \
      --drop 3 --steps 600 --out runs/quant-v5/pruned-block3

The teacher checkpoint is left untouched. The student is initialized by
removing one block and renumbering later blocks, then trained on existing
aquarium dialogue rows. Held-out persona dialogues are used only for scoring.
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from train_aq import D, GuppyConfig, GuppyLM, Tokenizer, encode_chat, load_dialogues, pad, sample


def student_state(state, drop):
    out = {}
    for key, val in state.items():
        if key.startswith("blocks."):
            parts = key.split(".", 2)
            old = int(parts[1])
            if old == drop:
                continue
            key = f"blocks.{old if old < drop else old - 1}.{parts[2]}"
        out[key] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--drop", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    teacher_cfg = GuppyConfig(**ck["config"])
    assert 0 <= a.drop < teacher_cfg.n_layers
    student_cfg = GuppyConfig(**{**ck["config"], "n_layers": teacher_cfg.n_layers - 1})
    teacher = GuppyLM(teacher_cfg).to(device)
    teacher.load_state_dict(ck["model"])
    teacher.eval()
    student = GuppyLM(student_cfg).to(device)
    student.load_state_dict(student_state(ck["model"], a.drop))
    for module in student.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    student.train()

    tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
    paths = [os.path.join(D, "dialogues.jsonl")] + glob.glob(os.path.join(D, "haiku", "out_*.jsonl"))
    rows = load_dialogues(paths)
    random.Random(42).shuffle(rows)
    rows = rows[max(200, len(rows) // 50):][:20000]
    rows += load_dialogues([os.path.join(D, "songs/chat.jsonl"), os.path.join(D, "family/chat.jsonl")])
    train = [encode_chat(tok, row, 512) for row in rows]
    held_rows = load_dialogues([os.path.join(D, "dialogues_eval.jsonl")])
    held = [encode_chat(tok, row, 512) for row in held_rows]
    held_batches = [pad(held[i:i + 32], 512) for i in range(0, len(held), 32)]

    @torch.no_grad()
    def score():
        student.eval()
        value = sum(student(X.to(device), Y.to(device))[1].item() for X, Y in held_batches) / len(held_batches)
        student.train()
        return value

    def save(step, loss):
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "best.pt")
        torch.save({"model": student.state_dict(), "config": vars(student_cfg), "step": step,
                    "teacher": a.ckpt, "dropped_block": a.drop}, path)
        return path

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.02)
    history = []
    start = time.time()
    best = float("inf")
    for step in range(a.steps + 1):
        if step % 200 == 0 or step == a.steps:
            value = score()
            row = {"step": step, "held_loss": value, "elapsed_s": round(time.time() - start)}
            history.append(row)
            print(json.dumps(row), flush=True)
            if value < best:
                best = value
                save(step, value)
        if step == a.steps:
            break
        picks = rng.choice(len(train), a.batch, replace=False)
        X, Y = pad([train[i] for i in picks], 512)
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            teacher_logits = teacher(X)[0]
        student_logits = student(X)[0]
        keep = X != 0
        kl = F.kl_div(F.log_softmax(student_logits[keep], dim=-1),
                      F.log_softmax(teacher_logits[keep], dim=-1),
                      log_target=True, reduction="batchmean")
        ce = F.cross_entropy(student_logits.reshape(-1, student_cfg.vocab_size),
                             Y.reshape(-1), ignore_index=0)
        loss = 0.7 * kl + 0.3 * ce
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        for group in opt.param_groups:
            group["lr"] = a.lr * 0.5 * (1 + np.cos(np.pi * (step + 1) / a.steps))
    student.eval()
    torch.manual_seed(1)
    (_, reply), = sample(student, tok, device,
                        "You are Pebble, a hermit crab who lives in a sandy tide pool. You are careful and practical.",
                        ["What is a toaster?"], max_new=50)
    print(json.dumps({"best_loss": best, "reply": reply}), flush=True)
    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump({"args": vars(a), "history": history, "best_loss": best, "sample_reply": reply}, f, indent=2)


if __name__ == "__main__":
    main()
