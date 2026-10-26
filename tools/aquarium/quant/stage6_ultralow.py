"""Explore 1.5-bit shape catalogs, with selective 2/3-bit rescue."""
import torch
from quantlib import FIXED_BYTES, SD, TARGETS, base_sd, calib_batches, gptq_vq, hessians, record, residual_catalogs
from stage1_scalar import rotations

D = 4
H = hessians(base_sd(), calib_batches())
P = {}
for k in TARGETS:
    U, R = rotations(k, "hadamard")
    W = U.T @ SD[k].float() @ R
    s = W.std(1, keepdim=True)
    P[k] = (W / s, s, U, R, R.T @ H[k] @ R)
X = torch.cat([P[k][0].reshape(-1, D) for k in TARGETS])
CATS = {n: residual_catalogs(X, n, 1) for n in (64, 256, 4096)}
CACHE = {}

def grid(k, n):
    if (k, n) not in CACHE:
        W, s, U, R, Hk = P[k]
        CACHE[k, n] = U @ (gptq_vq(W, Hk, CATS[n], D) * s) @ R.T
    return CACHE[k, n]

def family(k):
    return "emb" if k.startswith("tok_emb") else k.split(".", 2)[2].rsplit(".", 1)[0]

def run(name, assignment):
    sd = base_sd()
    nbytes = FIXED_BYTES + 2 * sum(SD[k].shape[0] for k in TARGETS)
    used = set()
    for k in TARGETS:
        n = assignment(k)
        used.add(n)
        sd[k] = grid(k, n)
        nbytes += SD[k].numel() * {64: 6, 256: 8, 4096: 12}[n] / D / 8
    nbytes += sum(n * D * 2 for n in used)
    return record(name, sd, nbytes)

if __name__ == "__main__":
    run("E7 all 1.5-bit d4x64", lambda k: 64)
    run("E7 all 2-bit d4x256", lambda k: 256)
    run("E7 3-bit emb, 1.5-bit others", lambda k: 4096 if family(k) == "emb" else 64)
    run("E7 3-bit emb, 2-bit others", lambda k: 4096 if family(k) == "emb" else 256)
    base = run("E7 2-bit emb, 1.5-bit others", lambda k: 256 if family(k) == "emb" else 64)
    fams = sorted({family(k) for k in TARGETS if family(k) != "emb"})
    for f in fams:
        run("E7 2-bit emb + " + f + ", rest 1.5-bit", lambda k, f=f: 256 if family(k) in ("emb", f) else 64)
    for layer in range(6):
        run(f"E7 2-bit emb + layer {layer}, rest 1.5-bit", lambda k, layer=layer: 256 if family(k) == "emb" or k.startswith(f"blocks.{layer}.") else 64)
