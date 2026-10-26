"""Compare one global shape catalog with per-family and per-matrix catalogs."""
import math
import torch
from quantlib import FIXED_BYTES, NW, SD, TARGETS, base_sd, calib_batches, gptq_vq, hessians, record, residual_catalogs
from stage1_scalar import rotations

D = 4
H = hessians(base_sd(), calib_batches())
P = {}
for k in TARGETS:
    U, R = rotations(k, "hadamard")
    W = U.T @ SD[k].float() @ R
    s = W.std(1, keepdim=True)
    P[k] = (W / s, s, U, R, R.T @ H[k] @ R)

def family(k):
    return "emb" if k.startswith("tok_emb") else k.split(".", 2)[2].rsplit(".", 1)[0]

def group(k, scope):
    return "all" if scope == "global" else family(k) if scope == "family" else k

def run(scope, K):
    groups = sorted({group(k, scope) for k in TARGETS})
    cats = {}
    for g in groups:
        X = torch.cat([P[k][0].reshape(-1, D) for k in TARGETS if group(k, scope) == g])
        cats[g] = residual_catalogs(X, K, 1)
    sd = base_sd()
    for k in TARGETS:
        W, s, U, R, Hk = P[k]
        Q = gptq_vq(W, Hk, cats[group(k, scope)], D)
        sd[k] = U @ (Q * s) @ R.T
    nrows = sum(SD[k].shape[0] for k in TARGETS)
    nbytes = FIXED_BYTES + nrows * 2 + NW * math.log2(K) / D / 8 + len(groups) * K * D * 2
    return record(f"E12 v5 {scope} catalog d4 K{K} +rot +gptq", sd, nbytes)

if __name__ == "__main__":
    for scope, K in [("global", 256), ("family", 256), ("matrix", 256), ("family", 512), ("matrix", 512)]:
        run(scope, K)
