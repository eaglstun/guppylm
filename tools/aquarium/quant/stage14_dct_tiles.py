"""JPEG-style 2D DCT compression of 8x8 and 16x16 weight tiles.

Each target matrix is viewed as an image. Tiles are orthogonally transformed, then
either fixed low frequencies or the largest coefficients are retained. Retained
coefficients are int8 with one fp16 scale per tile. A 3-bit all-coefficient variant
tests whether the DCT alone improves scalar quantization. Sizes are bitpacked payload
estimates; no entropy coding or header is assumed. Side tensors counted as fp16 are
rounded through fp16 before evaluation.

Run with AQ_QUANT_CKPT=runs/aq-chat-v5-noun-cont/final.pt and AQ_QUANT_OUT=...
"""
import math
import sys

import torch

from quantlib import FIXED_BYTES, NW, SD, SMALL, TARGETS, base_sd, record


def dct_basis(n):
    j = torch.arange(n, dtype=torch.float32)[None, :] + 0.5
    i = torch.arange(n, dtype=torch.float32)[:, None]
    c = torch.cos(math.pi / n * i * j) * math.sqrt(2 / n)
    c[0] /= math.sqrt(2)
    return c


def tiles_of(w, n):
    rows, cols = w.shape
    assert rows % n == 0 and cols % n == 0, (w.shape, n)
    return w.reshape(rows // n, n, cols // n, n).permute(0, 2, 1, 3).reshape(-1, n, n)


def untile(t, shape, n):
    rows, cols = shape
    return t.reshape(rows // n, cols // n, n, n).permute(0, 2, 1, 3).reshape(rows, cols)


def transform(t, c):
    return c @ t @ c.T


def invert(t, c):
    return c.T @ t @ c


def low_positions(n, keep):
    # JPEG-style zigzag; the sum u+v is the frequency band.
    return [u * n + v for u, v in sorted(((u, v) for u in range(n) for v in range(n)),
                                         key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else -p[0]))[:keep]]


def encode(w, n, keep, mode, c):
    original = tiles_of(w, n)
    coeff = transform(original, c).reshape(-1, n * n)
    count = len(coeff)
    if mode == "low":
        positions = torch.tensor(low_positions(n, keep), dtype=torch.long)[None, :].expand(count, -1)
        picked = coeff.gather(1, positions)
        bits_per_tile = 16 + 8 * keep
    elif mode == "top":
        picked, positions = coeff.abs().topk(keep, dim=1)
        picked = coeff.gather(1, positions)
        bits_per_tile = 16 + keep * (8 + int(math.log2(n * n)))
    elif mode == "all3":
        positions = torch.arange(n * n)[None, :].expand(count, -1)
        picked = coeff
        bits_per_tile = 16 + 3 * n * n
    else:
        raise ValueError(mode)

    levels = 3 if mode == "all3" else 127
    scale = (picked.abs().amax(1).clamp_min(1e-9) / levels).half().float()
    codes = torch.round(picked / scale[:, None]).clamp(-levels, levels).to(torch.int8)
    restored = torch.zeros_like(coeff)
    restored.scatter_(1, positions, codes.float() * scale[:, None])
    energy = float((picked.float().square().sum() / coeff.float().square().sum()).item())
    recon = untile(invert(restored.reshape(-1, n, n), c), w.shape, n)
    return recon, bits_per_tile, energy


def run(n, keep, mode):
    c = dct_basis(n)
    sd = base_sd()
    for k in SMALL:
        sd[k] = sd[k].half().float()
    bits = FIXED_BYTES * 8
    energy_num = 0.0
    for k in TARGETS:
        w = SD[k].float()
        q, per_tile, energy = encode(w, n, keep, mode, c)
        sd[k] = q
        ntile = w.numel() // (n * n)
        bits += ntile * per_tile
        energy_num += energy * w.square().sum().item()
    name = f"E16 JPEG-DCT {n}x{n} {mode} {keep if mode != 'all3' else '3b'}"
    return record(name, sd, math.ceil(bits / 8), extra={"dct_energy_weighted": round(energy_num / sum(SD[k].float().square().sum().item() for k in TARGETS), 4)})


def diagnostics(n):
    c = dct_basis(n)
    energy, low_energy, top_energy, raw_top_energy = 0., 0., 0., 0.
    keep = n * n // 4
    pos = low_positions(n, keep)
    for k in TARGETS:
        raw = tiles_of(SD[k].float(), n).reshape(-1, n * n)
        coeff = transform(raw.reshape(-1, n, n), c).reshape(-1, n * n)
        energy += coeff.square().sum().item()
        low_energy += coeff[:, pos].square().sum().item()
        top_energy += coeff.square().topk(keep, dim=1).values.sum().item()
        raw_top_energy += raw.square().topk(keep, dim=1).values.sum().item()
    print({"tile": n, "fraction": .25,
           "dct_lowfreq_energy": round(low_energy / energy, 4),
           "dct_top_energy": round(top_energy / energy, 4),
           "raw_top_energy": round(raw_top_energy / energy, 4)}, flush=True)


if __name__ == "__main__":
    modes = [(8, 16, "low"), (8, 16, "top"), (8, 24, "low"),
             (8, 24, "top"), (8, 64, "all3"),
             (16, 64, "low"), (16, 32, "top"), (16, 64, "top"),
             (16, 256, "all3")]
    if len(sys.argv) > 1:
        modes = [m for m in modes if int(sys.argv[1]) == m[0]]
    for n in sorted({m[0] for m in modes}):
        diagnostics(n)
    for n, keep, mode in modes:
        run(n, keep, mode)
