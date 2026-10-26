"""Measure lossless entropy coding of the frozen d4/K4096 GPTQ index stream.

Run with AQ_QUANT_CKPT pointing at the checkpoint used for quantization. Stage 4
does not save its frozen codes, so this regenerates the deterministic pre-tune
assignment. The 600-step tuning changes catalogs/scales/biases, not code indices.
"""
import collections
import heapq
import json
import math
import os
import subprocess
import zlib

import numpy as np
import torch
from quantlib import SD, TARGETS, base_sd, calib_batches, gptq_vq, hessians, residual_catalogs, FIXED_BYTES
from stage1_scalar import rotations

D, K = 4, 4096


def family(key):
    return "emb" if key.startswith("tok_emb") else key.split(".", 2)[2].rsplit(".", 1)[0]


def pack12(a):
    """Little-endian pair packing; two 12-bit codes in three bytes."""
    a = np.asarray(a, dtype=np.uint16).reshape(-1)
    if len(a) & 1:
        a = np.pad(a, (0, 1))
    lo = a[::2].astype(np.uint32)
    hi = a[1::2].astype(np.uint32)
    out = np.empty((len(lo), 3), dtype=np.uint8)
    out[:, 0] = lo
    out[:, 1] = (lo >> 8) | (hi << 4)
    out[:, 2] = hi >> 4
    return out.tobytes()


def pack_nibbles(a):
    a = np.asarray(a, dtype=np.uint16).reshape(-1)
    if len(a) & 1:
        a = np.pad(a, (0, 1))
    parts = []
    for shift in (8, 4, 0):
        v = ((a >> shift) & 15).astype(np.uint8)
        parts.append((v[::2] | (v[1::2] << 4)).tobytes())
    return b"".join(parts)


def codec_sizes(data):
    zstd = subprocess.run(["zstd", "-q", "-c", "-9"], input=data, capture_output=True, check=True).stdout
    return {"raw": len(data), "zlib9": len(zlib.compress(data, 9)), "zstd9": len(zstd)}


def entropy(a):
    counts = np.bincount(a, minlength=K)
    nz = counts[counts > 0]
    p = nz / nz.sum()
    bits = float(-(p * np.log2(p)).sum())
    heap = list(map(int, nz))
    heapq.heapify(heap)
    huff_bits = 0
    while len(heap) > 1:
        n = heapq.heappop(heap) + heapq.heappop(heap)
        huff_bits += n
        heapq.heappush(heap, n)
    # A canonical Huffman decoder also needs code lengths (up to 4096 bytes).
    return {"symbols": int(len(a)), "used_codes": int(len(nz)), "entropy_bpi": round(bits, 4),
            "entropy_payload_bytes": math.ceil(bits * len(a) / 8),
            "huffman_payload_bytes": math.ceil(huff_bits / 8)}


def grouped_summary(arrays, scope):
    buckets = collections.defaultdict(list)
    for key, a in arrays.items():
        bucket = "all" if scope == "global" else family(key) if scope == "family" else key
        buckets[bucket].append(a)
    stats = {key: entropy(np.concatenate(rows)) for key, rows in buckets.items()}
    return {"scope": scope, "groups": len(stats),
            "entropy_payload_bytes": sum(s["entropy_payload_bytes"] for s in stats.values()),
            "huffman_payload_bytes": sum(s["huffman_payload_bytes"] for s in stats.values()),
            "huffman_code_length_table_bytes": len(stats) * K,
            "weighted_entropy_bpi": round(sum(s["entropy_payload_bytes"] * 8 for s in stats.values()) /
                                          sum(s["symbols"] for s in stats.values()), 4),
            "details": stats if scope != "matrix" else None}


def main():
    H = hessians(base_sd(), calib_batches())
    prepared = {}
    for key in TARGETS:
        U, R = rotations(key, "hadamard")
        W = U.T @ SD[key].float() @ R
        s = W.std(1, keepdim=True)
        prepared[key] = (W / s, s, R)
    X = torch.cat([prepared[key][0].reshape(-1, D) for key in TARGETS])
    catalog = residual_catalogs(X, K, 1)
    arrays = {}
    for key in TARGETS:
        W, _, R = prepared[key]
        Hk = R.T @ H[key] @ R
        _, picks = gptq_vq(W, Hk, catalog, D, params=True)
        arrays[key] = picks[0].numpy().astype(np.uint16).reshape(-1)
    codes = np.concatenate(list(arrays.values()))
    counts = np.bincount(codes, minlength=K)
    freq_rank = np.argsort(-counts, kind="stable")
    remap = np.empty(K, dtype=np.uint16)
    remap[freq_rank] = np.arange(K, dtype=np.uint16)
    remapped = remap[codes]
    row_scales = b"".join(prepared[key][1].half().numpy().tobytes() for key in TARGETS)
    cat_bytes = catalog[0].half().numpy().tobytes()
    code_stats = {
        "as_assigned_packed12": codec_sizes(pack12(codes)),
        "frequency_reindexed_packed12": codec_sizes(pack12(remapped)),
        "frequency_reindexed_nibble_planes": codec_sizes(pack_nibbles(remapped)),
        "per_matrix_zstd9_packed12_bytes": sum(codec_sizes(pack12(a))["zstd9"] for a in arrays.values()),
    }
    payload = {
        "checkpoint": os.environ.get("AQ_QUANT_CKPT", "runs/aq-chat-v4/best.pt"),
        "note": "Pre-tune code assignments regenerated from stage4_tune.py; tuned catalog and scales are not in its checkpoint.",
        "target_weights": sum(SD[k].numel() for k in TARGETS),
        "indices": len(codes),
        "fixed_bytes": FIXED_BYTES,
        "row_scales": codec_sizes(row_scales),
        "initial_catalog": codec_sizes(cat_bytes),
        "codes": code_stats,
        "entropy": {scope: grouped_summary(arrays, scope) for scope in ("global", "family", "matrix")},
    }
    print(json.dumps(payload, indent=2))
    out = os.environ.get("AQ_ENTROPY_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
