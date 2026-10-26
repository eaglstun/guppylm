"""Stage 1: baselines, random rotation, and GPTQ for plain per-group rounding.

    python3 tools/aquarium/quant/stage1_scalar.py
"""
from quantlib import (FIXED_BYTES, SD, TARGETS, base_sd, block_rotation, calib_batches, dense_rotation,
                      gptq, hessians, q8_rows, record, rtn_groups, scalar_bytes)

rot_cache = {}


def rotations(k, kind, seed=0):
    """(U, R) for grid k: U mixes rows (outputs), R mixes columns (inputs)."""
    r, c = SD[k].shape
    if kind not in rot_cache:
        rot_cache[kind] = {}
    key = (r, c)
    if key not in rot_cache[kind]:
        make = block_rotation if kind == "hadamard" else dense_rotation
        rot_cache[kind][key] = (make(r, seed + 1), make(c, seed + 2))
    return rot_cache[kind][key]


def build(bits, g, rot=None, H=None):
    sd = base_sd()
    for k in TARGETS:
        W = SD[k].float()
        if rot:
            U, R = rotations(k, rot)
            W = U.T @ W @ R
        if H is not None:
            Hk = R.T @ H[k] @ R if rot else H[k]
            Q = gptq(W, Hk, bits, g)
        else:
            Q = rtn_groups(W, bits, g)
        sd[k] = U @ Q @ R.T if rot else Q
    return sd


if __name__ == "__main__":
    # E0: references
    full = {k: v for k, v in SD.items() if k != "lm_head.weight"}
    record("E0 original fp32", full, 4 * sum(v.numel() for v in full.values()))
    web = base_sd()
    for k in TARGETS:
        web[k] = q8_rows(SD[k])
    web_bytes = sum(SD[k].numel() + 4 * SD[k].shape[0] for k in TARGETS) + FIXED_BYTES
    record("E0 shipped web: int8 per row", web, web_bytes)

    # E1: plain rounding, then with rotation
    for bits, g in [(4, 32), (4, 128), (3, 32), (3, 128), (2, 32), (2, 8)]:
        for rot in (None, "hadamard", "dense"):
            record(f"E1 rtn {bits}b g{g}" + (f" +rot:{rot}" if rot else ""), build(bits, g, rot),
                   scalar_bytes(bits, g) + FIXED_BYTES)

    # E2: GPTQ with real activations from 256 training dialogues
    H = hessians(base_sd(), calib_batches())
    for bits, g in [(4, 32), (4, 128), (3, 32), (3, 128), (2, 32), (2, 64)]:
        for rot in (None, "hadamard"):
            record(f"E2 gptq {bits}b g{g}" + (f" +rot:{rot}" if rot else ""), build(bits, g, rot, H),
                   scalar_bytes(bits, g) + FIXED_BYTES)
