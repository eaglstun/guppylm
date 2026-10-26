"""Represent weight matrices as grayscale/RGB images, encode, decode, and score.

Run from the project root with AQ_QUANT_CKPT=runs/aq-chat-v5-noun-cont/final.pt
and AQ_QUANT_OUT=runs/quant-colors. Every matrix is an actual image file in
memory (JPEG, WebP, or PNG); measured bytes include all 25 image containers,
fp16 per-row scales, and the fixed non-target tensors. The scored model sees
exactly the decoded pixels and fp16 scales.

Optional dependency: Pillow 12.3.0 was tested. Install it in the project
environment with `uv pip install pillow` before running this experiment.
"""
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from quantlib import FIXED_BYTES, OUT, SD, SMALL, TARGETS, base_sd, record

T = np.array([[1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
              [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
              [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)]], dtype=np.float32)


def half_scale(x, levels):
    s = np.maximum(np.max(np.abs(x), axis=1, keepdims=True) / levels, 1e-8)
    return s.astype(np.float16).astype(np.float32)


def prepare(w, layout):
    """Return uint8 image, fp16-rounded scales, and inverse layout descriptor."""
    w = w.float().numpy()
    rows, cols = w.shape
    if layout == "gray":
        scales = half_scale(w, 127)
        q = np.clip(np.rint(w / scales), -127, 127).astype(np.int16)
        return (q + 128).astype(np.uint8), scales, (rows, cols)
    if layout == "rgb_adjacent":
        assert cols % 3 == 0
        scales = half_scale(w, 127)
        q = np.clip(np.rint(w / scales), -127, 127).astype(np.int16)
        return (q.reshape(rows, cols // 3, 3) + 128).astype(np.uint8), scales, (rows, cols)
    if layout == "rgb_orthogonal":
        assert cols % 3 == 0
        x = w.reshape(rows, cols // 3, 3) @ T.T
        scales = half_scale(x.transpose(0, 2, 1).reshape(rows * 3, -1), 127).reshape(rows, 3)
        q = np.clip(np.rint(x / scales[:, None, :]), -127, 127).astype(np.int16)
        return (q + 128).astype(np.uint8), scales, (rows, cols)
    if layout == "rgb_rows":
        scales = half_scale(w, 127)
        q = np.clip(np.rint(w / scales), -127, 127).astype(np.int16)
        pad = (-rows) % 3
        if pad:
            q = np.pad(q, ((0, pad), (0, 0)), constant_values=0)
        return (q.reshape(-1, 3, cols).transpose(0, 2, 1) + 128).astype(np.uint8), scales, (rows, cols)
    if layout == "rgb_two_byte":
        scales = half_scale(w, 32767)
        q = np.clip(np.rint(w / scales), -32767, 32767).astype(np.int32) + 32768
        img = np.zeros((rows, cols, 3), dtype=np.uint8)
        img[:, :, 0] = (q >> 8).astype(np.uint8)
        img[:, :, 1] = (q & 255).astype(np.uint8)
        return img, scales, (rows, cols)
    raise ValueError(layout)


def reconstruct(img, scales, layout, shape):
    rows, cols = shape
    q = img.astype(np.float32) - 128
    if layout == "gray":
        return q * scales
    if layout == "rgb_adjacent":
        return q.reshape(rows, cols) * scales
    if layout == "rgb_orthogonal":
        x = q * scales[:, None, :]
        return (x @ T).reshape(rows, cols)
    if layout == "rgb_rows":
        x = q.transpose(0, 2, 1).reshape(-1, cols)[:rows]
        return x * scales
    if layout == "rgb_two_byte":
        u = (img[:, :, 0].astype(np.int32) << 8) | img[:, :, 1].astype(np.int32)
        return (u - 32768).astype(np.float32) * scales
    raise ValueError(layout)


def image_roundtrip(img, codec, quality):
    stream = io.BytesIO()
    im = Image.fromarray(img)
    if codec == "JPEG":
        opts = {"format": "JPEG", "quality": quality}
        if im.mode == "RGB":
            opts["subsampling"] = 0  # 4:4:4; three weights have equal importance
        im.save(stream, **opts)
    elif codec == "WEBP":
        im.save(stream, format="WEBP", quality=quality, method=4)
    elif codec == "PNG":
        im.save(stream, format="PNG", optimize=True, compress_level=6)
    else:
        raise ValueError(codec)
    data = stream.getvalue()
    mode = "L" if img.ndim == 2 else "RGB"
    recovered = np.asarray(Image.open(io.BytesIO(data)).convert(mode))
    assert recovered.shape == img.shape
    return recovered, len(data)


def run(layout, codec, quality=None, evaluate=True):
    sd = base_sd()
    for k in SMALL:
        sd[k] = sd[k].half().float()
    total = FIXED_BYTES
    image_bytes = 0
    scale_bytes = 0
    mse = 0.0
    for k in TARGETS:
        original = SD[k].float()
        img, scales, shape = prepare(original, layout)
        pixels, n = image_roundtrip(img, codec, quality)
        q = torch.from_numpy(reconstruct(pixels, scales, layout, shape).copy())
        sd[k] = q
        image_bytes += n
        scale_bytes += scales.size * 2
        mse += torch.square(q - original).sum().item()
    total += image_bytes + scale_bytes
    name = f"E19 color {layout} {codec}" + (f" q{quality}" if quality else "")
    extra = {"image_bytes": image_bytes, "scale_bytes": scale_bytes,
             "target_rmse": round((mse / sum(SD[k].numel() for k in TARGETS)) ** .5, 6)}
    if evaluate:
        return record(name, sd, total, extra=extra)
    row = {"recipe": name, "layout": layout, "codec": codec, "quality": quality,
           "bytes": total, "MB": round(total / 1e6, 3), **extra}
    print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "screen.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def self_check():
    w = torch.arange(-384, 384, dtype=torch.float32).reshape(2, 384) / 600
    for layout in ("gray", "rgb_adjacent", "rgb_orthogonal", "rgb_rows", "rgb_two_byte"):
        img, scales, shape = prepare(w, layout)
        got, _ = image_roundtrip(img, "PNG", None)
        assert np.array_equal(got, img), layout
        rec = reconstruct(got, scales, layout, shape)
        assert rec.shape == tuple(w.shape), layout
    print("color layout and PNG round-trip checks passed", flush=True)


if __name__ == "__main__":
    self_check()
    configs = [(layout, codec, quality)
               for layout in ("gray", "rgb_adjacent", "rgb_orthogonal", "rgb_rows")
               for codec, levels in (("JPEG", (30, 60, 85, 95)), ("WEBP", (40, 50, 55, 60, 65, 70, 80, 95)), ("PNG", (None,)))
               for quality in levels]
    configs += [("rgb_two_byte", "JPEG", 95), ("rgb_two_byte", "PNG", None)]
    screen = "--screen" in sys.argv
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
    if only:
        selected = set(only.split(","))
        configs = [c for c in configs if f"{c[0]}:{c[1]}:{c[2]}" in selected]
        assert configs, only
    for args in configs:
        run(*args, evaluate=not screen)
