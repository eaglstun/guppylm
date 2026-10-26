"""Joint/beam encoding for two additive 256-shape catalogs over 8 weights.

The packed target stream is two uint8 indices per 8-value chunk (2 bpw). The
catalogs and one scale per row are fp16. Greedy encoding picks catalog 1 then
catalog 2; beam keeps the best B first-catalog choices and minimizes the *joint*
reconstruction error over all 256 second choices. GPTQ error feedback is then
applied to subsequent chunks exactly as in quantlib.gptq_vq.

Run with AQ_QUANT_CKPT pointing at the v5 noun-continuation checkpoint and
AQ_QUANT_OUT set to an isolated results directory.
"""
import argparse
import math
import os
import time

import torch
from quantlib import (FIXED_BYTES, SD, SMALL, TARGETS, base_sd, calib_batches, cfg,
                      hessians, record, residual_catalogs, vq_bytes)
from stage1_scalar import rotations

D, K, STAGES = 8, 256, 2

def joint_encode(X, C1, C2, beam, precomputed=None, batch=1024):
    """Return nearest pair among beam best C1 codes × every C2 code."""
    if len(X) > batch:
        chunks = [joint_encode(X[i:i+batch], C1, C2, beam, precomputed, batch)
                  for i in range(0, len(X), batch)]
        return tuple(torch.cat([chunk[j] for chunk in chunks]) for j in range(3))
    if precomputed is None:
        n1 = C1.square().sum(1)
        n2 = C2.square().sum(1)
        cross = 2 * (C1 @ C2.T)
    else:
        n1, n2, cross = precomputed
    d1 = n1[None, :] - 2 * (X @ C1.T)
    vals, first = torch.topk(d1, beam, dim=1, largest=False)
    d2 = n2[None, :] - 2 * (X @ C2.T)
    combined = vals[:, :, None] + d2[:, None, :] + cross[first]
    best = combined.reshape(len(X), -1).argmin(1)
    p1 = first.gather(1, (best // K)[:, None]).reshape(-1)
    p2 = best % K
    return C1[p1] + C2[p2], p1.to(torch.uint8), p2.to(torch.uint8)

def gptq_joint(W, H, C1, C2, beam):
    W = W.clone().float()
    r, c = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += .01 * torch.diag(H).mean() * torch.eye(c)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.empty_like(W)
    codes = torch.empty((r, c // D, 2), dtype=torch.uint8)
    terms = (C1.square().sum(1), C2.square().sum(1), 2 * (C1 @ C2.T))
    for i in range(0, c, D):
        q, p1, p2 = joint_encode(W[:, i:i+D], C1, C2, beam, terms)
        Q[:, i:i+D] = q
        codes[:, i//D, 0], codes[:, i//D, 1] = p1, p2
        for j in range(D):
            col = i + j
            e = (W[:, col] - q[:, j]) / U[col, col]
            W[:, col:] -= e[:, None] * U[col, col:][None, :]
    return Q, codes

def self_check():
    g = torch.Generator().manual_seed(1717)
    X = torch.randn(7, D, generator=g)
    C1 = torch.randn(K, D, generator=g)
    C2 = torch.randn(K, D, generator=g)
    Q, p1, p2 = joint_encode(X, C1, C2, K)
    assert torch.allclose(Q, C1[p1.long()] + C2[p2.long()])
    all_pairs = (C1[:, None, :] + C2[None, :, :]).reshape(K*K, D)
    best = torch.cdist(X, all_pairs).argmin(1)
    expected = all_pairs[best]
    assert torch.allclose(Q, expected, atol=1e-5), (Q - expected).abs().max().item()
    print("joint beam-256 matches exact 65,536-pair search", flush=True)

def packed_payload(codes, scales, C1, C2):
    pos = SD["pos_emb.weight"].float()
    pos_scale = pos.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    pos_q = torch.round(pos / pos_scale).clamp(-127, 127).to(torch.int8)
    return {"format": "aq-additive-d8-2x256-v1", "model_config": vars(cfg),
            "rotation_seed": 0, "catalogs": (C1.half(), C2.half()),
            "codes": codes, "row_scales": scales,
            "pos_q": pos_q, "pos_scale": pos_scale.reshape(-1),
            "small": {k: SD[k].half() for k in SMALL}}

def decode_payload(payload):
    C1, C2 = [c.float() for c in payload["catalogs"]]
    sd = {k: v.float() for k, v in payload["small"].items()}
    sd["pos_emb.weight"] = payload["pos_q"].float() * payload["pos_scale"].float()[:, None]
    for key in TARGETS:
        code = payload["codes"][key]
        r, n, _ = code.shape
        W = (C1[code[:, :, 0].long()] + C2[code[:, :, 1].long()]).reshape(r, n * D)
        W *= payload["row_scales"][key].float()
        U, R = rotations(key, "hadamard", seed=payload["rotation_seed"])
        sd[key] = U @ W @ R.T
    return sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--max-grids", type=int, default=0, help="debug: score only first N target grids")
    ap.add_argument("--save", default="", help="path prefix for packed-code torch checkpoints")
    a = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    P = {}
    for key in TARGETS:
        U, R = rotations(key, "hadamard")
        W = U.T @ SD[key].float() @ R
        s = W.std(1, keepdim=True).clamp_min(1e-8).half().float()
        P[key] = (W / s, s, U, R)
    X = torch.cat([P[k][0].reshape(-1, D) for k in TARGETS])
    C1, C2 = [c.half().float() for c in residual_catalogs(X, K, STAGES)]
    print(f"catalogs trained in {time.time()-t0:.1f}s", flush=True)
    H = hessians(base_sd(), calib_batches())
    keys = TARGETS[:a.max_grids] if a.max_grids else TARGETS
    for beam in a.beams:
        t = time.time()
        sd = base_sd()
        for key in SMALL:
            sd[key] = sd[key].half().float()
        codes_out, scales_out = {}, {}
        for key in keys:
            W, s, U, R = P[key]
            Hk = R.T @ H[key] @ R
            Q, codes = gptq_joint(W, Hk, C1, C2, beam)
            sd[key] = U @ (Q * s) @ R.T
            codes_out[key] = codes
            scales_out[key] = s.half()
        elapsed = time.time() - t
        print(f"beam {beam}: encoded {len(keys)} grids in {elapsed:.1f}s", flush=True)
        if a.max_grids:
            for key in keys:
                err = (sd[key] - SD[key].float()).square().mean().sqrt().item()
                print(f"  {key} RMS error {err:.6f}", flush=True)
            continue
        nbytes = vq_bytes(K, D, STAGES) + FIXED_BYTES
        row = record(f"E17 v5noun d8 2x256 +rot+gptq beam{beam}", sd, nbytes,
                     extra={"encode_seconds": round(elapsed, 1)})
        if a.save:
            payload = packed_payload(codes_out, scales_out, C1, C2)
            payload.update({"beam": beam, "estimated_bytes": nbytes, "loss": row["loss"]})
            path = f"{a.save}-beam{beam}.pt"
            torch.save(payload, path)
            unpacked = torch.load(path, map_location="cpu", weights_only=False)
            check = decode_payload(unpacked)
            worst = max((check[k] - sd[k]).abs().max().item() for k in sd)
            assert worst == 0, f"packed decode differed by {worst}"
            print(f"  actual torch-packed file {os.path.getsize(path):,} bytes; decode max error {worst}", flush=True)

if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        self_check()
    else:
        main()
