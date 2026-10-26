"""E8-inspired 16-bit eight-weight codebook pilot for the aquarium model.

The codeword is an 8-bit base-shape index plus 8 sign/shift bits. The 256
base shapes are mathematically generated: 227 points within the radius-10
half-integer D8 shell, plus 29 deterministic maximin representatives from
the radius-12 shell. The latter selection is independent of the E8P paper's
padding table, so this is E8-inspired rather than QuIP#'s exact codebook.
Each eight weights store 16 bits plus a fp16 scale per matrix row. Inspiration:
Tseng et al., QuIP# (ICML 2024), https://arxiv.org/pdf/2402.04396 .

Run after setting AQ_QUANT_CKPT and AQ_QUANT_OUT as for other quant scripts.
Example: AQ_E8P_SAVE=runs/quant-e8p/g128.pt python3 this_script.py --rot --gptq --group=128
Then: AQ_E8P_SAVE=runs/quant-e8p/g128-tuned.pt python3 this_script.py \
    --tune --from=runs/quant-e8p/g128.pt --steps=300
Validate: python3 this_script.py --decode-payload \
    --from=runs/quant-e8p/g128-tuned.pt
"""
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from quantlib import (D, FIXED_BYTES, SD, SMALL, TARGETS, base_sd, calib_batches, cfg,
                      encode_chat, hessians, load_dialogues, model_from, pad, record, tok)
from stage1_scalar import rotations

def base_grid():
    shells = [v for v in itertools.product((.5, 1.5, 2.5, 3.5), repeat=8)
              if sum(x*x for x in v) <= 10]
    assert len(shells) == 227
    # C(8,3)=56 radius-squared-12 patterns with three 1/2 and five 3/2
    # coordinates. Greedily spread 29 representatives, ties lexicographic.
    candidates = []
    for low in itertools.combinations(range(8), 3):
        candidates.append(tuple(.5 if i in low else 1.5 for i in range(8)))
    points = torch.tensor(candidates)
    reference = torch.tensor(shells)
    closest = torch.cdist(points, reference).min(1).values
    selected = []
    for _ in range(29):
        idx = int(closest.argmax())
        selected.append(candidates[idx])
        closest = torch.minimum(closest, torch.cdist(points, points[idx:idx+1]).reshape(-1))
        closest[[candidates.index(x) for x in selected]] = -1
    shells.extend(selected)
    B = torch.tensor(shells, dtype=torch.float32)
    B = B[:, [0, 2, 4, 6, 1, 3, 5, 7]].contiguous()
    parity = (B.sum(1).long() & 1)
    B[:, 7] *= 1 - 2 * parity
    assert B.shape == (256, 8)
    return B

B = base_grid()
BNORM = (B * B).sum(1)
BITS = torch.arange(8)

def decode(codes, grid=None):
    grid = B if grid is None else grid
    flat = codes.reshape(-1).long()
    raw = flat & 255
    parity = torch.zeros_like(raw)
    for j in range(8):
        parity ^= (raw >> j) & 1
    flips = raw ^ parity  # correct bit 0 so all eight sign flips have even parity
    signs = 1 - 2 * ((flips[:, None] >> BITS[None, :]) & 1)
    shift = torch.where(parity == 0, .25, -.25)
    vals = grid[flat >> 8] * signs + shift[:, None]
    return vals.reshape(*codes.shape, 8)

def _nearest_branch(X, shift):
    """Nearest signed base shape for one of the two lattice shifts."""
    Y = X - shift
    prod = Y[:, None, :] * B[None, :, :]
    neg = prod < 0
    odd = (neg.sum(2) & 1).bool()
    penalty, flip_pos = prod.abs().min(2)
    dot = prod.abs().sum(2) - 2 * torch.where(odd, penalty, 0)
    cost = Y.square().sum(1)[:, None] + BNORM[None, :] - 2 * dot
    chosen = cost.argmin(1)
    at = torch.arange(len(Y))
    signs = neg[at, chosen].clone()
    needs_flip = odd[at, chosen]
    pos = flip_pos[at, chosen]
    signs[at[needs_flip], pos[needs_flip]] ^= True
    raw = ((signs.long() << BITS[None, :]).sum(1)).long()
    if shift < 0:
        raw ^= 1  # raw odd parity requests the negative quarter shift
    code = (chosen.long() << 8) | raw
    return cost[at, chosen], code

def encode(X, batch=1024):
    vals, codes = [], []
    for start in range(0, len(X), batch):
        x = X[start:start+batch]
        cp, ip = _nearest_branch(x, .25)
        cm, im = _nearest_branch(x, -.25)
        codes.append(torch.where(cp <= cm, ip, im).to(torch.int32))
    idx = torch.cat(codes)
    return decode(idx), idx

def check_codec():
    X = torch.randn(64, 8)
    Q, idx = encode(X)
    assert idx.min() >= 0 and idx.max() < 65536
    assert torch.equal(Q, decode(idx))
    # Verify the branch optimizer against all 65,536 decoded shapes on a few inputs.
    full = decode(torch.arange(65536))
    brute = torch.cdist(X[:4], full).min(1).values
    fast = (X[:4] - Q[:4]).norm(dim=1)
    assert torch.allclose(brute, fast, atol=1e-5), (brute, fast)
    print("E8-inspired 16-bit encode/decode and nearest-neighbor check passed", flush=True)

def group_scales(W, group):
    # One fp16 RMS scale per row and input-column group.
    r, c = W.shape
    assert group % 8 == 0 and c % group == 0
    s = W.reshape(r, c//group, group).square().mean(2).sqrt().clamp_min(1e-8)
    return s.half().float()

def quantize_grid(W, scale, group, H=None):
    r, c = W.shape
    out = torch.empty_like(W)
    indices = torch.empty((r, c//8), dtype=torch.uint16)
    if H is None:
        per_chunk = scale.repeat_interleave(group//8, dim=1)
        Q, idx = encode((W.reshape(r, c//8, 8) / per_chunk[:, :, None]).reshape(-1, 8))
        out = Q.reshape(r, c//8, 8).mul(per_chunk[:, :, None]).reshape(r, c)
        indices[:] = idx.reshape(r, c//8)
    else:
        X = W.clone()
        H = H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        H += .01 * torch.diag(H).mean() * torch.eye(c)
        U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        for i in range(0, c, 8):
            sc = scale[:, i//group]
            q, idx = encode(X[:, i:i+8] / sc[:, None])
            q = q * sc[:, None]
            out[:, i:i+8] = q
            indices[:, i//8] = idx
            for j in range(8):
                col = i + j
                e = (X[:, col] - q[:, j]) / U[col, col]
                X[:, col:] -= e[:, None] * U[col, col:][None, :]
    return out, indices

def run(rot=False, gptq=False, matrix=None, scale_group=0):
    H = hessians(base_sd(), calib_batches()) if gptq else None
    sd = base_sd()
    for k in SMALL:
        sd[k] = sd[k].half().float()
    saved = {}
    keys = [matrix] if matrix else TARGETS
    t = time.time()
    for j, k in enumerate(keys):
        W = SD[k].float()
        U = R = None
        if rot:
            U, R = rotations(k, "hadamard")
            W = U.T @ W @ R
        group = scale_group or W.shape[1]
        scale = group_scales(W, group)
        h = R.T @ H[k] @ R if rot and H is not None else H[k] if H is not None else None
        Q, codes = quantize_grid(W, scale, group, h)
        sd[k] = U @ Q @ R.T if rot else Q
        saved[k] = (codes, scale.half())
        print(f"  encoded {j+1}/{len(keys)} {k} in {time.time()-t:.1f}s", flush=True)
    if matrix:
        err = (sd[matrix] - SD[matrix].float()).square().mean().sqrt().item()
        print(f"matrix {matrix} RMS error {err:.6g}", flush=True)
        return
    nscales = sum(SD[k].numel()//(scale_group or SD[k].shape[1]) for k in TARGETS)
    nbytes = FIXED_BYTES + sum(SD[k].numel()//8 * 2 for k in TARGETS) + nscales*2 + 1024
    name = "E16 generated-shell 2bpw" + (f" g{scale_group}" if scale_group else "") + (" +rot" if rot else "") + (" +gptq" if gptq else " +rtn")
    record(name, sd, nbytes)
    # Store actual uint16 codes and fp16 row scales for later direct-matvec work.
    if os.environ.get("AQ_E8P_SAVE"):
        path = os.environ["AQ_E8P_SAVE"]
        torch.save({"codes": saved, "base_grid": B.half(), "rot": rot,
                    "scale_group": scale_group,
                    "checkpoint": os.environ.get("AQ_QUANT_CKPT")}, path)

def tune(path, steps=600):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["rot"]
    scale_group = payload.get("scale_group", 0)
    fixed = {}
    for k in TARGETS:
        codes, scale = payload["codes"][k]
        U, R = rotations(k, "hadamard")
        group = scale_group or SD[k].shape[1]
        fixed[k] = (decode(codes).reshape(SD[k].shape), scale.float(), group, U, R)
    logrow = {k: torch.zeros_like(fixed[k][1], requires_grad=True) for k in TARGETS}
    start = base_sd()
    small = {k: start[k].clone().requires_grad_(True) for k in SMALL}

    def weights(final=False):
        sd = dict(start)
        sd.update({k: v.half().float() if final else v for k, v in small.items()})
        for k in TARGETS:
            grid, s0, group, U, R = fixed[k]
            scale = s0 * torch.exp(logrow[k])
            if final:
                scale = scale.half().float()
            sd[k] = U @ (grid * scale.repeat_interleave(group, dim=1)) @ R.T
        return sd

    nscales = sum(SD[k].numel()//(scale_group or SD[k].shape[1]) for k in TARGETS)
    nbytes = FIXED_BYTES + sum(SD[k].numel()//8 * 2 for k in TARGETS) + nscales*2 + 1024
    label = "E17 generated-shell 2bpw" + (f" g{scale_group}" if scale_group else "") + " +rot +gptq"
    record(label + " before tune", weights(final=True), nbytes)
    rows = load_dialogues([os.path.join(D, "dialogues.jsonl")])
    ex = [encode_chat(tok, r, 512) for r in rows]
    rng = np.random.default_rng(1)
    teacher = model_from(base_sd())
    student = model_from(base_sd())
    student.train()
    for mod in student.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    opt = torch.optim.Adam([{"params": list(logrow.values()), "lr": 2e-3},
                            {"params": list(small.values()), "lr": 2e-4}])
    t0 = time.time()
    for step in range(1, steps+1):
        warm = min(1.0, step / 50)
        lr_scale = warm * .5 * (1 + math.cos(math.pi * step / steps))
        opt.param_groups[0]["lr"] = 2e-3 * lr_scale
        opt.param_groups[1]["lr"] = 2e-4 * lr_scale
        X, _ = pad([ex[i] for i in rng.choice(len(ex), 8, replace=False)], 512)
        keep = X != 0
        with torch.no_grad():
            t = F.log_softmax(teacher(X)[0][keep], -1)
        s = F.log_softmax(functional_call(student, weights(), (X,))[0][keep], -1)
        loss = F.kl_div(s, t, log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  tune step {step}: KL {loss.item():.4f}, elapsed {time.time()-t0:.0f}s", flush=True)
    final = weights(final=True)
    result = record(label + f" +tune{steps}", final, nbytes)
    # Store an actual 1-KiB 4-bit base grid and all non-target side data.
    nib = (B * 2 + 8).round().to(torch.uint8)
    packed_base = nib[:, 0::2] | (nib[:, 1::2] << 4)
    scales = {k: (fixed[k][1] * torch.exp(logrow[k])).detach().half() for k in TARGETS}
    small_half = {k: v.detach().half() for k, v in small.items()}
    pos = SD["pos_emb.weight"].float()
    pos_scale = pos.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    pos_codes = torch.round(pos / pos_scale).clamp(-127, 127).to(torch.int8)
    out = {"codes": {k: payload["codes"][k][0] for k in TARGETS},
           "scales": scales, "small": small_half, "base_grid_4bit": packed_base,
           "pos_codes": pos_codes, "pos_scales": pos_scale.reshape(-1),
           "config": vars(cfg), "rot": True, "rotation_seed": 0, "rotation_block": 128,
           "scale_group": scale_group,
           "codebook": "generated-maximin-shell-v1", "checkpoint": payload["checkpoint"],
           "held_loss": result["loss"]}
    save = os.environ.get("AQ_E8P_SAVE", path.replace(".pt", "-tuned.pt"))
    torch.save(out, save)
    print(f"saved tuned code payload {save}", flush=True)

def decode_payload(path, save_full=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["codebook"] == "generated-maximin-shell-v1"
    assert payload["rotation_seed"] == 0 and payload["rotation_block"] == 128
    packed = payload["base_grid_4bit"]
    nib = torch.empty(256, 8, dtype=torch.uint8)
    nib[:, 0::2] = packed & 15
    nib[:, 1::2] = packed >> 4
    grid = (nib.float() - 8) / 2
    assert torch.equal(grid, B), "stored 4-bit grid differs from the generated grid"
    sd = {}
    for k in TARGETS:
        codes = payload["codes"][k]
        scale = payload["scales"][k].float()
        group = payload.get("scale_group", 0) or SD[k].shape[1]
        Q = decode(codes, grid).reshape(SD[k].shape) * scale.repeat_interleave(group, dim=1)
        if payload["rot"]:
            U, R = rotations(k, "hadamard")
            Q = U @ Q @ R.T
        sd[k] = Q
    sd["lm_head.weight"] = sd["tok_emb.weight"]
    for k, v in payload["small"].items():
        sd[k] = v.float()
    sd["pos_emb.weight"] = payload["pos_codes"].float() * payload["pos_scales"].float()[:, None]
    actual = (sum(v.numel()*v.element_size() for v in payload["codes"].values())
              + sum(v.numel()*v.element_size() for v in payload["scales"].values())
              + sum(v.numel()*v.element_size() for v in payload["small"].values())
              + payload["pos_codes"].numel()*payload["pos_codes"].element_size()
              + payload["pos_scales"].numel()*payload["pos_scales"].element_size()
              + packed.numel()*packed.element_size())
    print(f"actual tensor payload {actual:,} bytes; torch container {os.path.getsize(path):,} bytes", flush=True)
    record("E17 decode from packed E8-inspired payload", sd, actual)
    if save_full:
        torch.save({"model": sd, "config": payload["config"]}, save_full)
        print(f"saved probe-only float checkpoint {save_full}", flush=True)

if __name__ == "__main__":
    check_codec()
    if "--decode-payload" in sys.argv:
        target = next(x.split("=",1)[1] for x in sys.argv if x.startswith("--from="))
        out = next((x.split("=",1)[1] for x in sys.argv if x.startswith("--save-full=")), None)
        decode_payload(target, out)
    elif "--tune" in sys.argv:
        target = next(x.split("=",1)[1] for x in sys.argv if x.startswith("--from="))
        steps = next((int(x.split("=",1)[1]) for x in sys.argv if x.startswith("--steps=")), 600)
        tune(target, steps)
    elif "--check" not in sys.argv:
        key = next((x.split("=",1)[1] for x in sys.argv if x.startswith("--matrix=")), None)
        group = next((int(x.split("=",1)[1]) for x in sys.argv if x.startswith("--group=")), 0)
        run(rot="--rot" in sys.argv, gptq="--gptq" in sys.argv, matrix=key, scale_group=group)
