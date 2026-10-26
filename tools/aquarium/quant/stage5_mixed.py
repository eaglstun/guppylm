"""Stage 5: mixed precision. Which grids can drop from the 3-bit catalog to the 2-bit one?

    python3 tools/aquarium/quant/stage5_mixed.py

Two shared catalogs, both rotated + GPTQ, d=4: "hi" = 4096 shapes (3 bits/weight),
"lo" = 256 shapes (2 bits/weight). First every family is dropped to lo one at a time
(sensitivity), then the cheapest drops are combined.
"""
import torch
from quantlib import (FIXED_BYTES, SD, TARGETS, base_sd, calib_batches, gptq_vq, hessians, record,
                      residual_catalogs)
from stage1_scalar import rotations

D4 = 4
H = hessians(base_sd(), calib_batches())
P = {}
for k in TARGETS:
    U, R = rotations(k, "hadamard")
    W = U.T @ SD[k].float() @ R
    s = W.std(1, keepdim=True)
    P[k] = (W / s, s, U, R, R.T @ H[k] @ R)
X = torch.cat([P[k][0].reshape(-1, D4) for k in TARGETS])
CATS = {"hi": residual_catalogs(X, 4096, 1), "lo": residual_catalogs(X, 256, 1)}
BITS = {"hi": 12, "lo": 8}
cache = {}


def grid(k, level):
    if (k, level) not in cache:
        Wn, s, U, R, Hk = P[k]
        cache[(k, level)] = U @ (gptq_vq(Wn, Hk, CATS[level], D4) * s) @ R.T
    return cache[(k, level)]


def family(k):
    """tok_emb.weight -> emb; blocks.3.attn.qkv.weight -> attn.qkv"""
    return "emb" if k.startswith("tok_emb") else k.split(".", 2)[2].rsplit(".", 1)[0]


def run(name, low):
    """low: set of grid names stored with the 2-bit catalog."""
    sd = base_sd()
    nbytes = FIXED_BYTES + sum(SD[k].shape[0] for k in TARGETS) * 2
    used = set()
    for k in TARGETS:
        level = "lo" if k in low else "hi"
        used.add(level)
        sd[k] = grid(k, level)
        nbytes += SD[k].numel() * BITS[level] / D4 / 8
    nbytes += sum({"hi": 4096, "lo": 256}[lv] * D4 * 2 for lv in used)
    return record(name, sd, nbytes)


if __name__ == "__main__":
    fams = sorted({family(k) for k in TARGETS})
    base = run("E5 all 3-bit (d4 4096)", set())["loss"]
    sens = {}
    for f in fams:
        sens[f] = run(f"E5 drop {f} to 2-bit", {k for k in TARGETS if family(k) == f})
    by_layer = {}
    for i in range(6):
        by_layer[i] = run(f"E5 drop layer {i} to 2-bit", {k for k in TARGETS if k.startswith(f"blocks.{i}.")})
    # greedily drop families in order of least damage per byte saved
    order = sorted(fams, key=lambda f: (sens[f]["loss"] - base) / sum(SD[k].numel() for k in TARGETS if family(k) == f))
    low = set()
    for f in order:
        low |= {k for k in TARGETS if family(k) == f}
        run("E5 greedy 2-bit: " + "+".join(sorted({family(k) for k in low})), low)
