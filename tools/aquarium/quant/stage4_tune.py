"""Stage 4: quantize (GPTQ), then freeze every code and fine-tune what's left to imitate
the original fish.

    python3 tools/aquarium/quant/stage4_tune.py scalar 3 32 --ruler-bits 8 --steps 600
    python3 tools/aquarium/quant/stage4_tune.py vq 4 4096 --stages 1 --rot --steps 600

scalar BITS G: trainable = each group's ruler (a log-scale on its step, a shift on its minimum
    in units of the starting step).
vq D K: trainable = the shared catalog(s) of shapes and each row's scale.
Both also train every bias and LayerNorm. The codes (most of the file) never change, so the
result packs into exactly the same format. Loss: distillation, KL(original || quantized) on
every real token of training dialogues. The held-out dialogues are never trained on.
"""
import argparse
import math
import os

from quantlib import training_rows

import numpy as np
import torch
import torch.nn.functional as F
from quantlib import (D, FIXED_BYTES, SD, TARGETS, base_sd, calib_batches, cfg, encode_chat, gptq, gptq_vq, hessians,
                      load_dialogues, model_from, pad, record, residual_catalogs, scalar_bytes, tok, vq_bytes)
from stage1_scalar import rotations
from torch.func import functional_call

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["scalar", "vq"])
ap.add_argument("a", type=int, help="bits (scalar) or shape length d (vq)")
ap.add_argument("b", type=int, help="group size (scalar) or catalog size K (vq)")
ap.add_argument("--stages", type=int, default=1)
ap.add_argument("--hi-k", type=int, default=0, help="vq: a bigger catalog for --hi-fams grids")
ap.add_argument("--hi-fams", default="", help="vq: comma list, e.g. emb or emb,attn.qkv")
ap.add_argument("--rot", action="store_true")
ap.add_argument("--ruler-bits", type=int, default=16)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--tag", default="")
a = ap.parse_args()
rot_tag = " +rot" if a.rot else ""
if a.mode == "scalar":
    name = f"E4 gptq {a.a}b g{a.b}{rot_tag}" + (" rulers8" if a.ruler_bits == 8 else "") + a.tag
    nbytes = scalar_bytes(a.a, a.b, a.ruler_bits) + FIXED_BYTES
else:
    name = f"E4 vq d{a.a} {a.stages}x{a.b}{rot_tag} +gptq"
    nbytes = vq_bytes(a.b, a.a, a.stages) + FIXED_BYTES
    if a.hi_k:
        name += f" ({a.hi_fams} at {a.hi_k})"
    name += a.tag
HI = set(a.hi_fams.split(",")) - {""}


def family(k):
    """tok_emb.weight -> emb; blocks.3.attn.qkv.weight -> attn.qkv"""
    return "emb" if k.startswith("tok_emb") else k.split(".", 2)[2].rsplit(".", 1)[0]

# ---- quantize
H = hessians(base_sd(), calib_batches())
rotated = {}
for k in TARGETS:
    W = SD[k].float()
    U = R = None
    if a.rot:
        U, R = rotations(k, "hadamard")
        W = U.T @ W @ R
    rotated[k] = (W, R.T @ H[k] @ R if a.rot else H[k], U, R)

frozen, trainable = {}, []
if a.mode == "scalar":
    for k, (W, Hk, U, R) in rotated.items():
        _, codes, LO, S = gptq(W, Hk, a.a, a.b, ruler_bits=a.ruler_bits, params=True)
        frozen[k] = (codes.float(), LO, S, U, R)
    shift = {k: torch.zeros_like(v[1], requires_grad=True) for k, v in frozen.items()}
    logs = {k: torch.zeros_like(v[2], requires_grad=True) for k, v in frozen.items()}
    trainable = list(shift.values()) + list(logs.values())
else:
    d = a.a
    rs = {k: v[0].std(1, keepdim=True) for k, v in rotated.items()}
    X = torch.cat([rotated[k][0].div(rs[k]).reshape(-1, d) for k in TARGETS])
    cats0 = {"lo": residual_catalogs(X, a.b, a.stages)}
    if a.hi_k:
        cats0["hi"] = residual_catalogs(X, a.hi_k, a.stages)
        nbytes = FIXED_BYTES + sum(SD[k].shape[0] for k in TARGETS) * 2 + sum(
            a.stages * SD[k].numel() * math.log2(a.hi_k if family(k) in HI else a.b) / d / 8 for k in TARGETS) \
            + a.stages * (a.b + a.hi_k) * d * 2
    for k, (W, Hk, U, R) in rotated.items():
        level = "hi" if a.hi_k and family(k) in HI else "lo"
        _, picks = gptq_vq(W / rs[k], Hk, cats0[level], d, params=True)
        frozen[k] = (level, picks, U, R)
    cats = {lv: [C.clone().requires_grad_(True) for C in cs] for lv, cs in cats0.items()}
    logrow = {k: torch.zeros_like(rs[k], requires_grad=True) for k in TARGETS}
    trainable = [C for cs in cats.values() for C in cs] + list(logrow.values())

start = base_sd()
small = {k: v.clone().requires_grad_(True) for k, v in start.items() if v.dim() == 1}


def snap(x, unit_div):
    """8-bit rulers: re-round onto a per-row grid after training."""
    unit = x.abs().amax(1, keepdim=True).clamp_min(1e-12) / unit_div
    return torch.round(x / unit) * unit


def weights(final=False):
    sd = dict(start)
    sd.update(small)
    for k in TARGETS:
        r, c = SD[k].shape
        if a.mode == "scalar":
            codes, LO, S, U, R = frozen[k]
            s = S * torch.exp(logs[k])
            lo = LO + S * shift[k]
            if final and a.ruler_bits == 8:
                lo, s = snap(lo, 127), snap(s, 255).clamp_min(1e-12)
            W = lo.repeat_interleave(a.b, 1) + codes * s.repeat_interleave(a.b, 1)
        else:
            level, picks, U, R = frozen[k]
            W = sum(C[p] for C, p in zip(cats[level], picks)).reshape(r, c) * (rs[k] * torch.exp(logrow[k]))
        sd[k] = U @ W @ R.T if a.rot else W
    return sd


record(name + " (before tune)", {k: v.detach() for k, v in weights(final=True).items()}, nbytes)

# ---- distill
rows = training_rows()
ex = [encode_chat(tok, r, 512) for r in rows]
rng = np.random.default_rng(1)
teacher = model_from(base_sd())
student = model_from(base_sd())
student.train()
for mod in student.modules():  # no dropout: we're imitating, not regularizing
    if isinstance(mod, torch.nn.Dropout):
        mod.p = 0.0
opt = torch.optim.Adam([{"params": trainable, "lr": a.lr},
                        {"params": list(small.values()), "lr": a.lr / 10}])
base_lrs = (a.lr, a.lr / 10)
for step in range(1, a.steps + 1):
    warm = min(1.0, step / 50)
    lr_scale = warm * 0.5 * (1 + np.cos(np.pi * step / a.steps))
    for gidx, base_lr in enumerate(base_lrs):
        opt.param_groups[gidx]["lr"] = base_lr * lr_scale
    X, _ = pad([ex[i] for i in rng.choice(len(ex), a.batch, replace=False)], 512)
    keep = X != 0
    with torch.no_grad():
        t = F.log_softmax(teacher(X)[0][keep], -1)
    s = F.log_softmax(functional_call(student, weights(), (X,))[0][keep], -1)  # lm_head ties to tok_emb
    loss = F.kl_div(s, t, log_target=True, reduction="batchmean")
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"  step {step}: KL {loss.item():.4f}", flush=True)

final = {k: v.detach() for k, v in weights(final=True).items()}
record(name + f" +tune{a.steps}", final, nbytes)
final["lm_head.weight"] = final["tok_emb.weight"]
torch.save({"model": final, "config": vars(cfg), "quant_config": dict(vars(a))}, os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs",
                                                                 "quant", name.replace(" ", "_") + ".pt"))
