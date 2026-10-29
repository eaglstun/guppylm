"""Export an aquarium checkpoint for the browser: int8 weights, one scale per row.

    python3 tools/aquarium/export_web.py runs/aq-chat-v3/best.pt --out web/model
    python3 tools/aquarium/export_web.py runs/aq-chat-v3/best.pt --fake-quant runs/aq-chat-v3/q8.pt

Writes guppy.bin (a JSON header, then the tensors), tokenizer.json, and personas.json.
Matrices and the two embeddings are stored as int8 with a float32 scale per output row;
biases and LayerNorms stay float32. --fake-quant writes a normal checkpoint with the int8 rounding
applied, so probe.py can measure what quantization costs.
"""

import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personas import EVAL_PERSONAS, PERSONAS, SYSTEM_TEMPLATES  # noqa: E402
from train_aq import D  # noqa: E402


def quantize(w):
    """Symmetric int8 per row. Returns (int8 rows, float32 scales)."""
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def is_quantized(name, t):
    return t.ndim == 2  # Linear weights, tok_emb (tied to lm_head), pos_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default=os.path.join(D, "tokenizer.json"), help="tokenizer used for this checkpoint")
    ap.add_argument("--out", help="directory for guppy.bin + tokenizer.json")
    ap.add_argument("--fake-quant", help="write a dequantized checkpoint here instead")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = {k: v.float().numpy() for k, v in ck["model"].items() if k != "lm_head.weight"}

    if a.fake_quant:
        sd = dict(ck["model"])
        for k, v in state.items():
            if is_quantized(k, v):
                q, s = quantize(v)
                sd[k] = torch.from_numpy(q.astype(np.float32) * s[:, None])
        sd["lm_head.weight"] = sd["tok_emb.weight"]
        torch.save({**ck, "model": sd}, a.fake_quant)
        print(f"fake-quant checkpoint -> {a.fake_quant}")
        return

    os.makedirs(a.out, exist_ok=True)
    tensors, blobs, offset = [], [], 0

    def put(arr):
        nonlocal offset
        b = arr.tobytes()
        b += b"\0" * (-len(b) % 4)  # keep float32 views aligned
        blobs.append(b)
        start, offset = offset, offset + len(b)
        return start

    for name, w in state.items():
        if is_quantized(name, w):
            q, s = quantize(w)
            tensors.append({"name": name, "dtype": "int8", "shape": list(w.shape),
                            "offset": put(q), "scale": put(s)})
        else:
            tensors.append({"name": name, "dtype": "float32", "shape": list(w.shape),
                            "offset": put(w.astype(np.float32))})

    header = json.dumps({"config": ck["config"], "tensors": tensors}).encode()
    header += b" " * (-(len(header) + 4) % 4)
    path = os.path.join(a.out, "guppy.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)
    shutil.copy(a.tokenizer, os.path.join(a.out, "tokenizer.json"))
    with open(os.path.join(a.out, "personas.json"), "w") as f:
        json.dump({"template": SYSTEM_TEMPLATES[0], "held_out": sorted(EVAL_PERSONAS),
                   "personas": {k: {"species": s, "habitat": h, "traits": t} for k, (s, h, t) in PERSONAS.items()}},
                  f, indent=1)
    print(f"{path}: {os.path.getsize(path) / 1e6:.2f} MB, {len(tensors)} tensors")


if __name__ == "__main__":
    main()
