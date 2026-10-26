"""Try small low-rank residuals on top of the tuned 2-bit catalog model."""
import os
import torch
from quantlib import SD, TARGETS, FIXED_BYTES, record, vq_bytes

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOURCE = os.environ.get("AQ_2BIT_CHECKPOINT", os.path.join(ROOT, "runs/quant/E4_vq_d4_1x256_+rot_+gptq_long.pt"))
base = torch.load(SOURCE, map_location="cpu", weights_only=False)["model"]
BASE_BYTES = vq_bytes(256, 4) + FIXED_BYTES

def family(k):
    return "emb" if k.startswith("tok_emb") else k.split(".", 2)[2].rsplit(".", 1)[0]

corrections = {}
for k in TARGETS:
    diff = SD[k].float() - base[k].float()
    U, S, V = torch.svd_lowrank(diff, q=36, niter=3)
    corrections[k] = (U, S, V)

def run(name, ranks):
    sd = dict(base)
    nbytes = BASE_BYTES
    for k in TARGETS:
        rank = ranks(k)
        if not rank:
            continue
        U, S, V = corrections[k]
        left = (U[:, :rank] * S[:rank]).half().float()
        right = V[:, :rank].half().float()
        sd[k] = base[k].float() + left @ right.T
        nbytes += 2 * rank * (SD[k].shape[0] + SD[k].shape[1])
    return record(name, sd, nbytes)

if __name__ == "__main__":
    run("E8 tuned 2-bit baseline", lambda k: 0)
    for rank in (2, 4, 8, 16, 32):
        run(f"E8 tuned 2-bit + residual rank{rank} all", lambda k, rank=rank: rank)
    for f in sorted({family(k) for k in TARGETS}):
        run(f"E8 tuned 2-bit + residual rank8 {f}", lambda k, f=f: 8 if family(k) == f else 0)
    run("E8 tuned 2-bit + residual rank16 emb only", lambda k: 16 if family(k) == "emb" else 0)
