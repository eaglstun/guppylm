"""JPEG-like rate/distortion sweep on 64-weight (8x8) blocks.

Tests two layouts: 64 consecutive values within one matrix row (row64), and
spatial 8-row x 8-column tiles (tile8). Each layout has an identity/no-DCT
control. JPEG-style quality levels use a frequency-dependent quantization
table, zero-run coding in zigzag order, signed varints, and zlib. Also tests
fixed-bit coefficient coding with several bit depths and retained counts.

The byte count includes the actual compressed code stream, one fp16 RMS or
max scale per tile, a 128-byte frequency table, matrix lengths, and every
non-target tensor counted by FIXED_BYTES. Side tensors are rounded to their
stored precision before scoring. This is an experiment, not a browser codec.

Run from repo root:
 AQ_QUANT_CKPT=runs/aq-chat-v5-noun-cont/final.pt \
 AQ_QUANT_OUT=runs/quant-jpeg64 .venv/bin/python \
 tools/aquarium/quant/stage18_jpeg64_sweep.py
"""
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import torch

from quantlib import FIXED_BYTES, NW, OUT, SD, SMALL, TARGETS, base_sd, model_from, score
from stage14_dct_tiles import dct_basis, low_positions

N = 8
C = dct_basis(N)
ZIGZAG = np.asarray(low_positions(N, N * N), dtype=np.int64)
RASTER = np.arange(N * N, dtype=np.int64)
HEADER_BYTES = 128 + 4 * len(TARGETS)


def measure(name, sd, nbytes, extra):
    t0 = time.time()
    loss = score(model_from(sd))
    row = {"recipe": name, "bytes": nbytes, "MB": round(nbytes / 1e6, 3),
           "bpw": round(8 * (nbytes - FIXED_BYTES) / NW, 3),
           "loss": round(loss, 4), **extra}
    with open(os.path.join(OUT, "sweep_results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{name:58s} {row['MB']:5.2f} MB  loss {loss:.4f}  eval {time.time()-t0:.1f}s", flush=True)
    return row


def tiles(w, layout):
    r, c = w.shape
    assert c % 64 == 0 and r % 8 == 0 and c % 8 == 0
    if layout == "row64":
        return w.reshape(r, c // 64, 8, 8).reshape(-1, 8, 8)
    if layout == "tile8":
        return w.reshape(r // 8, 8, c // 8, 8).permute(0, 2, 1, 3).reshape(-1, 8, 8)
    raise ValueError(layout)


def untile(t, shape, layout):
    r, c = shape
    if layout == "row64":
        return t.reshape(r, c // 64, 64).reshape(r, c)
    return t.reshape(r // 8, c // 8, 8, 8).permute(0, 2, 1, 3).reshape(r, c)


def coeffs(w, layout, transform):
    x = tiles(w, layout)
    return ((C @ x @ C.T) if transform == "dct" else x).reshape(-1, 64)


def reconstruction(q, shape, layout, transform):
    x = q.reshape(-1, 8, 8)
    if transform == "dct":
        x = C.T @ x @ C
    return untile(x, shape, layout)


def zigzag_varint(v, out):
    u = (abs(v) << 1) | (1 if v < 0 else 0)
    while u >= 128:
        out.append((u & 127) | 128)
        u >>= 7
    out.append(u)


def rle_encode(q, order):
    out = bytearray()
    for row in q:
        run = 0
        for v in row[order]:
            v = int(v)
            if v == 0:
                run += 1
            else:
                out.append(run)
                zigzag_varint(v, out)
                run = 0
        out.append(255)  # end-of-block; runs are at most 63
    return bytes(out)


def rle_decode(stream, ntiles, order):
    q = np.zeros((ntiles, 64), dtype=np.int16)
    p = 0
    for t in range(ntiles):
        j = 0
        while True:
            run = stream[p]
            p += 1
            if run == 255:
                break
            j += run
            u = 0
            shift = 0
            while True:
                b = stream[p]
                p += 1
                u |= (b & 127) << shift
                if b < 128:
                    break
                shift += 7
            v = -(u >> 1) if (u & 1) else (u >> 1)
            q[t, order[j]] = v
            j += 1
    assert p == len(stream)
    return q


def bitplanes(q, bits):
    levels = (1 << (bits - 1)) - 1
    flat = (q.reshape(-1).astype(np.int32) + levels).astype(np.uint16)
    return b"".join(np.packbits(((flat >> b) & 1).astype(np.uint8), bitorder="little").tobytes()
                    for b in range(bits))


def qtable(alpha, gamma, dc, transform):
    uv = torch.arange(8)
    freq = (uv[:, None] + uv[None, :]).float() / 7
    out = alpha * (1 + gamma * freq)
    if transform == "dct":
        out[0, 0] *= dc
    return out.reshape(64).half().float()


def prepared(layout, transform):
    return {k: (SD[k].shape, coeffs(SD[k].float(), layout, transform)) for k in TARGETS}


def base_state():
    sd = base_sd()
    for k in SMALL:
        sd[k] = sd[k].half().float()
    return sd


def run_jpeg(cache, layout, transform, alpha, gamma, dc=0.5):
    table = qtable(alpha, gamma, dc, transform)
    sd = base_state()
    stream_bytes = 0
    scales_bytes = 0
    nonzero = 0
    count = 0
    first_checked = False
    for k in TARGETS:
        shape, x = cache[k]
        s = x.square().mean(1).sqrt().clamp_min(1e-6).half().float()
        step = s[:, None] * table[None, :]
        code = torch.round(x / step).clamp(-32767, 32767).to(torch.int16)
        q = code.numpy()
        scan = ZIGZAG if transform == "dct" else RASTER
        raw = rle_encode(q, scan)
        if not first_checked:
            small = rle_encode(q[:16], scan)
            assert np.array_equal(rle_decode(small, 16, scan), q[:16])
            first_checked = True
        stream_bytes += len(zlib.compress(raw, 9))
        scales_bytes += 2 * len(s)
        nonzero += int(np.count_nonzero(q))
        count += q.size
        sd[k] = reconstruction(code.float() * step, shape, layout, transform)
    total = FIXED_BYTES + HEADER_BYTES + stream_bytes + scales_bytes
    name = f"E18 JPEG64 {layout} {transform} q{alpha:g} gamma{gamma:g} dc{dc:g}"
    row = measure(name, sd, total, extra={"layout": layout, "transform": transform,
                                          "alpha": alpha, "gamma": gamma,
                                          "codec": "zigzag-rle-varint-zlib9",
                                          "code_bytes": stream_bytes, "scale_bytes": scales_bytes,
                                          "nonzero_fraction": round(nonzero / count, 4)})
    return row


def run_fixed(cache, layout, transform, bits, keep):
    sd = base_state()
    code_bytes = 0
    scale_bytes = 0
    levels = (1 << (bits - 1)) - 1
    pos = torch.tensor(low_positions(8, keep) if transform == "dct" else list(range(keep)))
    for k in TARGETS:
        shape, x = cache[k]
        selected = x[:, pos]
        s = (selected.abs().amax(1).clamp_min(1e-6) / levels).half().float()
        code = torch.round(selected / s[:, None]).clamp(-levels, levels).to(torch.int16)
        raw = bitplanes(code.numpy(), bits)
        code_bytes += len(zlib.compress(raw, 9))
        scale_bytes += 2 * len(s)
        restored = torch.zeros_like(x)
        restored[:, pos] = code.float() * s[:, None]
        sd[k] = reconstruction(restored, shape, layout, transform)
    total = FIXED_BYTES + HEADER_BYTES + code_bytes + scale_bytes
    name = f"E18 JPEG64 {layout} {transform} keep{keep} bits{bits}"
    row = measure(name, sd, total, extra={"layout": layout, "transform": transform,
                                          "bits": bits, "keep": keep,
                                          "codec": "bitplanes-zlib9",
                                          "code_bytes": code_bytes, "scale_bytes": scale_bytes})
    return row


def main():
    results = []
    if "--targeted" in sys.argv:
        row_dct = prepared("row64", "dct")
        for alpha, gamma in ((0.2, 0.0), (0.2, 1.0), (0.35, 0.0),
                             (0.35, 1.0), (0.6, 0.0)):
            results.append(run_jpeg(row_dct, "row64", "dct", alpha, gamma))
        row_identity = prepared("row64", "identity")
        results.append(run_jpeg(row_identity, "row64", "identity", 0.35, 0.0))
        spatial_dct = prepared("tile8", "dct")
        for alpha, gamma in ((0.2, 0.0), (0.35, 1.0)):
            results.append(run_jpeg(spatial_dct, "tile8", "dct", alpha, gamma))
        spatial_identity = prepared("tile8", "identity")
        results.append(run_jpeg(spatial_identity, "tile8", "identity", 0.35, 0.0))
        with open(os.path.join(OUT, "sweep_summary.json"), "w") as f:
            json.dump(sorted(results, key=lambda r: (r["MB"], r["loss"])), f, indent=2)
        print(f"completed {len(results)} JPEG64 settings", flush=True)
        return
    if "--targeted-extra" in sys.argv:
        row_identity = prepared("row64", "identity")
        results.append(run_jpeg(row_identity, "row64", "identity", 0.35, 0.0))
        spatial_dct = prepared("tile8", "dct")
        results.append(run_jpeg(spatial_dct, "tile8", "dct", 0.2, 0.0))
        results.append(run_jpeg(spatial_dct, "tile8", "dct", 0.35, 1.0))
        spatial_identity = prepared("tile8", "identity")
        results.append(run_jpeg(spatial_identity, "tile8", "identity", 0.35, 0.0))
        with open(os.path.join(OUT, "sweep_results.jsonl")) as f:
            all_rows = [json.loads(line) for line in f]
        with open(os.path.join(OUT, "sweep_summary.json"), "w") as f:
            json.dump(sorted(all_rows, key=lambda r: (r["MB"], r["loss"])), f, indent=2)
        print(f"completed {len(all_rows)} JPEG64 settings", flush=True)
        return
    layouts = ("row64", "tile8")
    alphas = (0.2, 0.35, 0.6, 1.0)
    for layout in layouts:
        for transform in ("dct", "identity"):
            cache = prepared(layout, transform)
            for alpha in alphas:
                gammas = (0.0, 1.0) if transform == "dct" else (0.0,)
                for gamma in gammas:
                    results.append(run_jpeg(cache, layout, transform, alpha, gamma))
    with open(os.path.join(OUT, "sweep_summary.json"), "w") as f:
        json.dump(sorted(results, key=lambda r: (r["MB"], r["loss"])), f, indent=2)
    print(f"completed {len(results)} JPEG64 recipes", flush=True)


if __name__ == "__main__":
    main()
