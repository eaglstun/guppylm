"""8/16-weight projective vector quantization: shape index + signed int8 amplitude.

The catalog stores unit vectors. For each d-value chunk, choose the shape whose line
is nearest, then store its signed amplitude with one int8 code. One fp16 amplitude
scale per matrix row makes the int8 codes meaningful. This is an unrotated codec:
its codes can be read directly by a compressed-weight matvec without a matrix-wide
inverse transform. Catalog and row scales are rounded to fp16 in the scored model.
"""
import math
import os
import sys

import torch
from quantlib import FIXED_BYTES, SD, SMALL, TARGETS, base_sd, calib_batches, hessians, record

torch.manual_seed(0)
TRAIN_VECTORS = int(os.environ.get("AQ_SHAPE_TRAIN", "150000"))
ITERS = int(os.environ.get("AQ_SHAPE_ITERS", "12"))

def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def projective_assign(X, C, chunk=8192):
    ids, signs = [], []
    for i in range(0, len(X), chunk):
        dots = X[i:i + chunk] @ C.T
        a = dots.abs().argmax(1)
        ids.append(a)
        signs.append(dots.gather(1, a[:, None]).sign().reshape(-1))
    return torch.cat(ids), torch.cat(signs)

def train_catalog(d, K):
    X = torch.cat([SD[k].float().reshape(-1, d) for k in TARGETS])
    g = torch.Generator().manual_seed(1300 + d + K)
    pick = torch.randperm(len(X), generator=g)[:min(TRAIN_VECTORS, len(X))]
    X = unit(X[pick])
    C = X[torch.randperm(len(X), generator=g)[:K]].clone()
    for _ in range(ITERS):
        ids, signs = projective_assign(X, C)
        sums = torch.zeros_like(C).index_add_(0, ids, X * signs[:, None])
        empty = sums.norm(dim=1) < 1e-9
        if empty.any():
            sums[empty] = X[torch.randint(len(X), (int(empty.sum()),), generator=g)]
        C = unit(sums)
    return C.half().float()

def row_scales(W, d):
    r, c = W.shape
    norms = W.reshape(r, c // d, d).norm(dim=2)
    return (norms.amax(1).clamp_min(1e-8) / 127).half().float()

def quant_chunk(X, C, scales):
    dots = X @ C.T
    idx = dots.abs().argmax(1)
    amp = dots.gather(1, idx[:, None]).reshape(-1)
    codes = torch.round(amp / scales).clamp(-127, 127).to(torch.int8)
    Q = codes.float()[:, None] * scales[:, None] * C[idx]
    return Q, idx, codes

def quantize(W, C, d, scale, H=None):
    r, c = W.shape
    n = c // d
    if H is None:
        X = W.reshape(r * n, d)
        scales = scale[:, None].expand(r, n).reshape(-1)
        Q, ids, codes = quant_chunk(X, C, scales)
        return Q.reshape(r, c), ids.reshape(r, n), codes.reshape(r, n)
    W = W.clone()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += .01 * torch.diag(H).mean() * torch.eye(c)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    ids = torch.zeros(r, n, dtype=torch.int32)
    codes = torch.zeros(r, n, dtype=torch.int8)
    for i in range(0, c, d):
        q, ix, amp = quant_chunk(W[:, i:i+d], C, scale)
        Q[:, i:i+d] = q
        ids[:, i//d] = ix.int()
        codes[:, i//d] = amp
        for j in range(d):
            col = i + j
            e = (W[:, col] - q[:, j]) / U[col, col]
            W[:, col:] -= e[:, None] * U[col, col:][None, :]
    return Q, ids, codes

def run(d, K, gptq=False):
    C = train_catalog(d, K)
    H = hessians(base_sd(), calib_batches()) if gptq else None
    sd = base_sd()
    for k in SMALL:
        sd[k] = sd[k].half().float()
    codes_out = {}
    for k in TARGETS:
        W = SD[k].float()
        scale = row_scales(W, d)
        Q, ids, codes = quantize(W, C, d, scale, H[k] if H is not None else None)
        sd[k] = Q
        codes_out[k] = (ids, codes, scale)
    chunks = sum(SD[k].numel() // d for k in TARGETS)
    rows = sum(SD[k].shape[0] for k in TARGETS)
    nbytes = FIXED_BYTES + chunks * (math.log2(K) + 8) / 8 + rows * 2 + K * d * 2
    name = f"E15 projective d{d} K{K} int8amp" + (" +gptq" if gptq else " +rtn")
    result = record(name, sd, nbytes)
    return result

if __name__ == "__main__":
    specs = [(8, 1024), (16, 1024), (8, 4096), (16, 4096)]
    if len(sys.argv) > 1:
        specs = [(int(sys.argv[1]), int(sys.argv[2]))]
    for d, K in specs:
        run(d, K, gptq="--gptq" in sys.argv)
