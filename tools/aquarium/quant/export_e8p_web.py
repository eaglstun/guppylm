"""Export the clean E8-inspired payload into a browser-readable binary.

    .venv/bin/python tools/aquarium/quant/export_e8p_web.py \
      runs/quant-e8p/generated-rot-gptq-tuned.pt web/model/guppy_e8.bin

The weight codes stay uint16. This file contains no expanded weight matrices.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch


def signs(n, seed):
    rng = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=rng) * 2 - 1).to(torch.int8).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload")
    ap.add_argument("output")
    ap.add_argument("--reference", help="write full-sequence Python logits and K/V for a 32-token check")
    ap.add_argument("--reference-checkpoint", help="decoded float checkpoint for --reference")
    a = ap.parse_args()
    ck = torch.load(a.payload, map_location="cpu", weights_only=False)
    assert ck["codebook"] == "generated-maximin-shell-v1"
    assert ck["rot"] and ck["rotation_seed"] == 0 and ck["rotation_block"] == 128
    scale_group = int(ck.get("scale_group") or 0)
    tensors, blobs, offset = [], [], 0

    def put(array):
        nonlocal offset
        raw = np.ascontiguousarray(array).tobytes()
        raw += b"\0" * (-len(raw) % 4)
        blobs.append(raw)
        start, offset = offset, offset + len(raw)
        return start

    for name, code in ck["codes"].items():
        code = code.numpy().astype("<u2", copy=False)
        rows, chunks = code.shape
        cols = chunks * 8
        scale = ck["scales"][name].numpy().astype("<f2", copy=False)
        groups = cols // scale_group if scale_group else 1
        assert scale.size == rows * groups
        tensors.append({"name": name, "dtype": "e8", "shape": [rows, cols],
                        "offset": put(code), "scale": put(scale),
                        "scale_group": scale_group or cols,
                        "u_sign": put(signs(rows, 1)), "r_sign": put(signs(cols, 2))})

    pos = ck["pos_codes"].numpy().astype("i1", copy=False)
    pos_scale = ck["pos_scales"].numpy().astype("<f4", copy=False)
    tensors.append({"name": "pos_emb.weight", "dtype": "int8", "shape": list(pos.shape),
                    "offset": put(pos), "scale": put(pos_scale)})
    for name, value in ck["small"].items():
        arr = value.numpy().astype("<f2", copy=False)
        tensors.append({"name": name, "dtype": "float16", "shape": list(arr.shape),
                        "offset": put(arr)})
    grid = ck["base_grid_4bit"].numpy().astype("u1", copy=False)
    grid_offset = put(grid)
    header = json.dumps({"format": "guppy-e8-v1", "config": ck["config"],
                         "codebook": ck["codebook"], "rotation_block": 128,
                         "base_grid_offset": grid_offset, "tensors": tensors}, separators=(",", ":")).encode()
    header += b" " * (-(len(header) + 4) % 4)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "wb") as file:
        file.write(struct.pack("<I", len(header)))
        file.write(header)
        for blob in blobs:
            file.write(blob)
    print(f"{a.output}: {os.path.getsize(a.output):,} bytes, {len(tensors)} tensors")
    if a.reference:
        # The pilot's decoded checkpoint is a float reference for exactly this payload.
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.insert(0, root)
        from guppylm.config import GuppyConfig
        from guppylm.model import GuppyLM
        ref_path = a.reference_checkpoint or os.path.join(os.path.dirname(a.payload), "generated-probe.pt")
        ref_ck = torch.load(ref_path, map_location="cpu", weights_only=False)
        model = GuppyLM(GuppyConfig(**ref_ck["config"]))
        model.load_state_dict(ref_ck["model"])
        model.eval()
        ids = [(100 + i * 17) % model.config.vocab_size for i in range(32)]
        qkv = {}
        hooks = []
        for layer, block in enumerate(model.blocks):
            hooks.append(block.attn.qkv.register_forward_hook(
                lambda _m, _inp, output, layer=layer: qkv.__setitem__(layer, output.detach().cpu()[0])))
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0][0]
        for hook in hooks:
            hook.remove()
        d = model.config.d_model
        out = {"ids": ids, "logits": logits.tolist(),
               "emb0": model.tok_emb.weight[ids[0]].detach().tolist(),
               "pos0": model.pos_emb.weight[0].detach().tolist(),
               "qkv0": qkv[0][0].tolist(),
               "k": [qkv[l][:, d:2*d].reshape(-1).tolist() for l in range(model.config.n_layers)],
               "v": [qkv[l][:, 2*d:3*d].reshape(-1).tolist() for l in range(model.config.n_layers)]}
        with open(a.reference, "w") as file:
            json.dump(out, file)
        print(f"Python reference {a.reference}: {os.path.getsize(a.reference):,} bytes")


if __name__ == "__main__":
    main()
