"""Stage 3: shape catalogs (vector quantization), with rotation, GPTQ, and stacked catalogs.

    python3 tools/aquarium/quant/stage3_catalog.py

Each grid is rotated (optional), every row divided by its own spread (one fp16 per row), and
cut into groups of d. One catalog set is shared by every grid in the model.
"""
import sys

import torch
from quantlib import (FIXED_BYTES, SD, TARGETS, base_sd, calib_batches, encode, gptq_vq, hessians, record,
                      residual_catalogs, vq_bytes)
from stage1_scalar import rotations

H = hessians(base_sd(), calib_batches())


def prepared(rot):
    """Rotated, row-normalized grids: {k: (W, row_scale, U, R)}."""
    out = {}
    for k in TARGETS:
        W = SD[k].float()
        U = R = None
        if rot:
            U, R = rotations(k, "hadamard")
            W = U.T @ W @ R
        s = W.std(1, keepdim=True)
        out[k] = (W / s, s, U, R)
    return out


_cats = {}


def build(d, K, stages, rot, use_gptq):
    P = prepared(rot)
    if (d, K, stages, rot) not in _cats:  # the catalog doesn't depend on GPTQ, learn it once
        X = torch.cat([P[k][0].reshape(-1, d) for k in TARGETS])
        _cats[(d, K, stages, rot)] = residual_catalogs(X, K, stages)
    cats = _cats[(d, K, stages, rot)]
    sd = base_sd()
    for k in TARGETS:
        Wn, s, U, R = P[k]
        if use_gptq:
            Hk = R.T @ H[k] @ R if rot else H[k]
            Q = gptq_vq(Wn, Hk, cats, d)
        else:
            Q = encode(Wn.reshape(-1, d), cats).reshape(Wn.shape)
        Q = Q * s
        sd[k] = U @ Q @ R.T if rot else Q
    return sd


CONFIGS = [  # (d, K, stages): bits per weight = stages * log2(K) / d
    (8, 256, 1),     # 1.0
    (8, 4096, 1),    # 1.5
    (4, 256, 1),     # 2.0
    (8, 256, 2),     # 2.0, AQLM "2x8"
    (8, 4096, 2),    # 3.0
    (4, 4096, 1),    # 3.0
    (8, 256, 3),     # 3.0
    (4, 256, 2),     # 4.0
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for d, K, st in CONFIGS:
        for rot in (None, "hadamard"):
            for g in (False, True):
                name = f"E3 vq d{d} {st}x{K}" + (" +rot" if rot else "") + (" +gptq" if g else "")
                if only and not any(o in name for o in only):
                    continue
                record(name, build(d, K, st, rot, g), vq_bytes(K, d, st) + FIXED_BYTES)
