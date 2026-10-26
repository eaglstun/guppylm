"""Pilot: reconstruct sparse 1-bit GuppyLM weights with a local cellular rule.

The experiment changes one matrix in a pretrained checkpoint. It fits a small,
shared rule to that matrix's binary signs, then evaluates all variants on held-out
language-model examples. No checkpoint or training data is modified.
"""

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from guppylm.config import GuppyConfig
from guppylm.dataset import get_dataloader
from guppylm.model import GuppyLM


class LocalRule(nn.Module):
    """One shared 3x3 update, reused for every cell and every step."""

    def __init__(self, width=16):
        super().__init__()
        self.neighbors = nn.Conv2d(2, width, 3, padding=1)
        self.update = nn.Conv2d(width, 1, 1)

    def forward(self, seed, mask, steps):
        state = seed
        for _ in range(steps):
            proposal = torch.tanh(self.update(F.relu(self.neighbors(torch.cat((state, mask), dim=1)))))
            state = torch.where(mask.bool(), seed, proposal)
        return state


def seed_mask(height, width, stride, device, pattern):
    mask = torch.zeros((1, 1, height, width), device=device)
    mask[:, :, ::stride, ::stride] = 1
    if pattern == "checkerboard":
        if stride != 2:
            raise ValueError("checkerboard requires stride 2")
        mask[:, :, 1::2, 1::2] = 1
    elif pattern == "three_quarters":
        if stride != 2:
            raise ValueError("three_quarters requires stride 2")
        mask = 1 - mask
    return mask


def fixed_rule(seed, mask, steps):
    state = seed
    for _ in range(steps):
        counts = F.avg_pool2d(state, 3, stride=1, padding=1)
        state = torch.where(mask.bool(), seed, torch.where(counts >= 0, 1.0, -1.0))
    return state


def fit_rule(target, mask, args):
    torch.manual_seed(args.seed)
    rule = LocalRule(args.rule_width)
    optimizer = torch.optim.Adam(rule.parameters(), lr=args.learning_rate)
    height, width = target.shape[-2:]
    patch = args.patch_size
    if height < patch or width < patch or patch % args.stride:
        raise ValueError("patch size must fit the matrix and be divisible by stride")
    starts_y = list(range(0, height - patch + 1, args.stride))
    starts_x = list(range(0, width - patch + 1, args.stride))
    t0 = time.perf_counter()
    for step in range(args.train_steps):
        ys = torch.randint(len(starts_y), (args.patch_batch,))
        xs = torch.randint(len(starts_x), (args.patch_batch,))
        batch = torch.cat([target[:, :, starts_y[y]:starts_y[y] + patch,
                                  starts_x[x]:starts_x[x] + patch]
                           for y, x in zip(ys.tolist(), xs.tolist())])
        patch_mask = seed_mask(patch, patch, args.stride, batch.device,
                               args.seed_pattern).expand_as(batch)
        result = rule(batch * patch_mask, patch_mask, args.ca_steps)
        missing = 1 - patch_mask
        loss = ((result - batch).square() * missing).sum() / missing.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f"rule step {step + 1}/{args.train_steps}: missing-cell MSE {loss.item():.4f}", flush=True)
    return rule.eval(), time.perf_counter() - t0


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fields = GuppyConfig.__dataclass_fields__
    config = GuppyConfig(**{key: value for key, value in checkpoint["config"].items() if key in fields})
    model = GuppyLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


@torch.inference_mode()
def eval_loss(model, loader, batches):
    total, tokens = 0.0, 0
    for index, (x, y) in enumerate(loader):
        if index >= batches:
            break
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                 ignore_index=0, reduction="sum").item()
        tokens += (y != 0).sum().item()
    return total / tokens, tokens


def packed_bytes(bits):
    return math.ceil(bits / 8)


def write_payload(path, signs, mask, scales, stride, steps, pattern, rule=None):
    """Write packed seeds and verify their bit round trip."""
    rows, cols = signs.shape[-2:]
    bits = (signs[mask.bool()] > 0).to(torch.uint8).numpy()
    packed = np.packbits(bits, bitorder="little").tobytes()
    pattern_id = {"grid": 0, "checkerboard": 1, "three_quarters": 2}[pattern]
    header = struct.pack("<4sIIHHHB", b"GCA1", rows, cols, stride, steps,
                         rule.neighbors.out_channels if rule else 0, pattern_id)
    body = header + scales.to(torch.float16).numpy().tobytes() + packed
    if rule is not None:
        body += b"".join(parameter.detach().numpy().astype("<f4", copy=False).tobytes()
                         for parameter in rule.parameters())
    path.write_bytes(body)
    restored = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:len(bits)]
    if not np.array_equal(restored, bits):
        raise RuntimeError("packed seed bits failed round trip")
    return len(body)


def read_payload(path):
    """Rebuild the matrix from the bytes that would be deployed."""
    body = path.read_bytes()
    header_size = struct.calcsize("<4sIIHHHB")
    magic, rows, cols, stride, steps, width, pattern_id = struct.unpack(
        "<4sIIHHHB", body[:header_size])
    if magic != b"GCA1":
        raise ValueError("invalid cellular payload")
    pattern = ("grid", "checkerboard", "three_quarters")[pattern_id]
    mask = seed_mask(rows, cols, stride, "cpu", pattern)
    scale_end = header_size + rows * 2
    scales = torch.from_numpy(np.frombuffer(body[header_size:scale_end],
                                         dtype="<f2").copy()).float().view(rows, 1)
    count = int(mask.sum().item())
    seed_end = scale_end + packed_bytes(count)
    bits = np.unpackbits(np.frombuffer(body[scale_end:seed_end], dtype=np.uint8),
                         bitorder="little")[:count].copy()
    seed = torch.zeros_like(mask)
    seed[mask.bool()] = torch.from_numpy(bits.astype(np.float32)) * 2 - 1
    if width:
        rule = LocalRule(width)
        values = np.frombuffer(body[seed_end:], dtype="<f4").copy()
        offset = 0
        with torch.no_grad():
            for parameter in rule.parameters():
                size = parameter.numel()
                parameter.copy_(torch.from_numpy(values[offset:offset + size]).view_as(parameter))
                offset += size
        if offset != len(values):
            raise ValueError("unexpected rule payload length")
        with torch.inference_mode():
            signs = torch.where(rule(seed, mask, steps) >= 0, 1.0, -1.0)
    else:
        signs = fixed_rule(seed, mask, steps)
        if seed_end != len(body):
            raise ValueError("unexpected fixed payload length")
    return signs[0, 0] * scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "cellular_compression" / "results.json")
    parser.add_argument("--matrix", default="blocks.0.ffn.up.weight")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed-pattern", choices=("grid", "checkerboard", "three_quarters"),
                        default="grid")
    parser.add_argument("--ca-steps", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patch-batch", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--rule-width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stride < 2 or args.ca_steps < 1:
        parser.error("stride must be >= 2 and CA steps must be >= 1")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint)
    parameters = dict(model.named_parameters())
    if args.matrix not in parameters or parameters[args.matrix].ndim != 2:
        parser.error("matrix must name a 2D model parameter")
    weight = parameters[args.matrix]
    original = weight.detach().clone()
    rows, cols = original.shape
    signs = torch.where(original >= 0, 1.0, -1.0).view(1, 1, rows, cols)
    scales = original.abs().mean(dim=1, keepdim=True).half().float()
    mask = seed_mask(rows, cols, args.stride, signs.device, args.seed_pattern)
    seed = signs * mask
    loader = get_dataloader(str(args.eval_jsonl), str(args.tokenizer),
                            model.config.max_seq_len, args.eval_batch_size, shuffle=False)

    rule, fit_seconds = fit_rule(signs, mask, args)
    with torch.inference_mode():
        t0 = time.perf_counter()
        learned = rule(seed, mask, args.ca_steps)
        reconstruction_seconds = time.perf_counter() - t0
        learned = torch.where(learned >= 0, 1.0, -1.0)
        fixed = fixed_rule(seed, mask, args.ca_steps)
        variants = {
            "original_fp32": original,
            "packed_1bit": signs[0, 0] * scales,
            "sparse_zero_fill": seed[0, 0] * scales,
            "fixed_ca": fixed[0, 0] * scales,
            "learned_ca": learned[0, 0] * scales,
        }
        results = {}
        for name, value in variants.items():
            weight.copy_(value)
            t0 = time.perf_counter()
            loss, tokens = eval_loss(model, loader, args.eval_batches)
            results[name] = {
                "loss_nats_per_token": loss,
                "perplexity": math.exp(loss),
                "eval_seconds": time.perf_counter() - t0,
            }
            print(f"{name:17s} loss={loss:.4f} ppl={math.exp(loss):.2f}", flush=True)
        weight.copy_(original)

    total = rows * cols
    seeds = int(mask.sum().item())
    rule_bytes = sum(p.numel() * p.element_size() for p in rule.parameters())
    scale_bytes = rows * 2  # fp16 row scales in a deployable representation
    for name, count in (("fixed_ca", fixed), ("learned_ca", learned)):
        results[name]["missing_sign_agreement"] = float(
            (((count == signs) * (1 - mask)).sum() / (1 - mask).sum()).item()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_bytes = {
        "fixed_ca": write_payload(args.output.parent / "fixed.ca1", signs, mask, scales,
                                       args.stride, args.ca_steps, args.seed_pattern),
        "learned_ca": write_payload(args.output.parent / "learned.ca1", signs, mask, scales,
                                         args.stride, args.ca_steps, args.seed_pattern, rule),
    }
    if not torch.equal(read_payload(args.output.parent / "fixed.ca1"), variants["fixed_ca"]):
        raise RuntimeError("fixed payload reconstruction mismatch")
    if not torch.equal(read_payload(args.output.parent / "learned.ca1"), variants["learned_ca"]):
        raise RuntimeError("learned payload reconstruction mismatch")
    neighbor_agreement = {
        "horizontal": float((signs[:, :, :, 1:] == signs[:, :, :, :-1]).float().mean()),
        "vertical": float((signs[:, :, 1:, :] == signs[:, :, :-1, :]).float().mean()),
    }
    report = {
        "checkpoint": str(args.checkpoint), "matrix": args.matrix, "shape": [rows, cols],
        "eval_tokens": tokens, "seed": args.seed, "stride": args.stride,
        "seed_pattern": args.seed_pattern, "retained_fraction": seeds / total,
        "ca_steps": args.ca_steps, "train_steps": args.train_steps,
        "rule_width": args.rule_width, "fit_seconds": fit_seconds,
        "reconstruction_seconds": reconstruction_seconds,
        "neighbor_sign_agreement": neighbor_agreement,
        "artifact_bytes": artifact_bytes,
        "storage_bytes": {
            "original_fp32": total * 4,
            "packed_1bit": packed_bytes(total) + scale_bytes,
            "sparse_zero_fill": packed_bytes(seeds) + scale_bytes,
            "fixed_ca": packed_bytes(seeds) + scale_bytes,
            "learned_ca": packed_bytes(seeds) + scale_bytes + rule_bytes,
        },
        "storage_note": "One matrix only. Artifact bytes include a 19-byte header, packed seed signs, fp16 row scales, and fp32 rule parameters where applicable. Rule implementation is shared code. No full-model size or inference speed claim.",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
