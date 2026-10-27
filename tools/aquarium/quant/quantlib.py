"""Shared pieces for the aquarium quantization experiments (see QUANTIZATION.md).

A "recipe" turns the v4 checkpoint into a state dict whose big grids have been rounded,
plus an exact size in bytes for what a real file would hold. Everything is simulated in
float32 (fake-quant): the numbers the model sees are exactly the ones a packed file decodes to.

Grids that get quantized ("targets"): tok_emb (tied to lm_head), and every block's
qkv / out / up / down. pos_emb is stored int8 per row, like the shipped web export.
Biases and LayerNorms stay float16. Groups always run along a row (the input dimension).
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train_aq import D, GuppyConfig, GuppyLM, Tokenizer, encode_chat, load_dialogues, pad, sample  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
D = os.environ.get("AQ_QUANT_DATA", D)
TRAIN_MANIFEST = os.environ.get("AQ_QUANT_TRAIN_MANIFEST")
DEV_MANIFEST = os.environ.get("AQ_QUANT_DEV_MANIFEST")
if bool(TRAIN_MANIFEST) != bool(DEV_MANIFEST):
    raise ValueError("Supply both AQ_QUANT_TRAIN_MANIFEST and AQ_QUANT_DEV_MANIFEST")
if TRAIN_MANIFEST and not os.environ.get("AQ_QUANT_CKPT"):
    raise ValueError("Manifest-based compression requires an explicit AQ_QUANT_CKPT")
_manifest_train = _manifest_dev = None
_manifest_hashes = {}
if TRAIN_MANIFEST:
    from tools.aquarium.v52.data import assert_disjoint, file_hash, read_manifest
    _manifest_train, th, tm = read_manifest(TRAIN_MANIFEST, "train")
    _manifest_dev, dh, dm = read_manifest(DEV_MANIFEST, "compression_development")
    assert_disjoint(_manifest_train, _manifest_dev)
    if not _manifest_train or not _manifest_dev:
        raise ValueError("Compression manifests must be nonempty")
    tokenizer_hash = file_hash(os.path.join(D, "tokenizer.json"))
    for meta in (tm, dm):
        expected = meta.get("inputs", {}).get("tokenizer.json", {}).get("sha256")
        if expected and expected != tokenizer_hash:
            raise ValueError("Compression tokenizer differs from frozen manifest")
    _manifest_hashes = {"train": th, "development": dh, "tokenizer": tokenizer_hash,
                        "checkpoint": file_hash(os.environ["AQ_QUANT_CKPT"])}


def training_rows():
    """All calibration and code-frozen distillation use the same train-only source."""
    return _manifest_train if _manifest_train is not None else load_dialogues([os.path.join(D, "dialogues.jsonl")])
CKPT = os.environ.get("AQ_QUANT_CKPT", os.path.join(ROOT, "runs/aq-chat-v4/best.pt"))
OUT = os.environ.get("AQ_QUANT_OUT", os.path.join(ROOT, "runs/quant"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = GuppyConfig(**ck["config"])
SD = ck["model"]
tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
TARGETS = ["tok_emb.weight"] + [f"blocks.{i}.{m}.weight" for i in range(cfg.n_layers)
                                for m in ("attn.qkv", "attn.out", "ffn.up", "ffn.down")]
NW = sum(SD[k].numel() for k in TARGETS)
# what every recipe pays outside the targets: pos_emb int8 + fp32 row scale, the rest fp16
SMALL = [k for k in SD if k not in TARGETS and k != "lm_head.weight" and k != "pos_emb.weight"]
FIXED_BYTES = SD["pos_emb.weight"].numel() + 4 * SD["pos_emb.weight"].shape[0] + 2 * sum(SD[k].numel() for k in SMALL)


# ---------------------------------------------------------------- model + scoring
def model_from(sd):
    sd = dict(sd)
    sd["lm_head.weight"] = sd["tok_emb.weight"]
    m = GuppyLM(cfg)
    m.load_state_dict(sd)
    m.eval()
    return m


def q8_rows(w):
    s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    return torch.round(w / s).clamp(-127, 127) * s


def base_sd():
    """The checkpoint with pos_emb already int8 per row (every recipe shares this)."""
    sd = {k: v.clone() for k, v in SD.items() if k != "lm_head.weight"}
    sd["pos_emb.weight"] = q8_rows(sd["pos_emb.weight"])
    return sd


_held_rows = _manifest_dev if _manifest_dev is not None else load_dialogues([os.path.join(D, "dialogues_eval.jsonl")])
_held = [encode_chat(tok, r, cfg.max_seq_len) for r in _held_rows]
HELD = [pad(_held[i:i + 32], cfg.max_seq_len) for i in range(0, len(_held), 32)]


@torch.no_grad()
def score(m):
    if not TRAIN_MANIFEST:
        return sum(m(X, Y)[1].item() for X, Y in HELD) / len(HELD)
    total, count = 0., 0
    for X, Y in HELD:
        n = int((Y != 0).sum())
        if n:
            total += m(X, Y)[1].item() * n
            count += n
    if not count:
        raise ValueError("Compression development set has no assistant targets")
    return total / count


SYS = "You are Pebble, a hermit crab who lives in a sandy tide pool. You are careful and practical."


def reply(m, q="What is a toaster?"):
    torch.manual_seed(1)
    (_, r), = sample(m, tok, "cpu", SYS, [q], max_new=50)
    return r


def record(name, sd, nbytes, extra=None, verbose=True):
    t = time.time()
    m = model_from(sd)
    loss = score(m)
    row = {"recipe": name, "MB": round(nbytes / 1e6, 3), "bpw": round(8 * (nbytes - FIXED_BYTES) / NW, 3),
           "loss": round(loss, 4), "reply": reply(m), **(extra or {})}
    if TRAIN_MANIFEST:
        row["manifest_hashes"] = _manifest_hashes
        row["evaluation"] = "compression_development_assistant_token_weighted"
    with open(os.path.join(OUT, "results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    if verbose:
        print(f"{name:52s} {row['MB']:6.2f} MB {row['bpw']:5.2f} bpw  loss {loss:.4f}  ({time.time()-t:.0f}s) | {row['reply'][:70]!r}",
              flush=True)
    return row


# ---------------------------------------------------------------- calibration
def calib_batches(n_dialogues=256, seed=0):
    rows = training_rows()
    rng = np.random.default_rng(seed)
    pick = [rows[i] for i in rng.choice(len(rows), n_dialogues, replace=False)]
    if TRAIN_MANIFEST:
        record_path = os.path.join(OUT, f"calibration-{seed}-{n_dialogues}.json")
        provenance = {"manifest_hashes": _manifest_hashes, "seed": seed, "ids": [r["uid"] for r in pick]}
        if os.path.exists(record_path):
            with open(record_path) as f:
                if json.load(f) != provenance:
                    raise ValueError("Output directory contains different calibration provenance")
        else:
            with open(record_path, "w") as f:
                json.dump(provenance, f, indent=1)
    ex = [encode_chat(tok, r, cfg.max_seq_len) for r in pick]
    return [pad(ex[i:i + 16], cfg.max_seq_len) for i in range(0, len(ex), 16)]


def hessians(sd, batches):
    """H = sum of x x^T over every real (non-pad) token, for each Linear's input.
    lm_head's H is what tok_emb gets, since the two share one grid."""
    m = model_from(sd)
    H, hooks, mask = {}, [], {}
    names = {f"blocks.{i}.{n}": f"blocks.{i}.{n}.weight" for i in range(cfg.n_layers)
             for n in ("attn.qkv", "attn.out", "ffn.up", "ffn.down")}
    names["lm_head"] = "tok_emb.weight"
    for mod_name, mod in m.named_modules():
        if mod_name in names and isinstance(mod, nn.Linear):
            key = names[mod_name]

            def hook(_, inp, __, key=key):
                x = inp[0][mask["keep"]].double()
                H[key] = H.get(key, 0) + x.T @ x
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for X, _ in batches:
            mask["keep"] = X != 0
            m(X)
    for h in hooks:
        h.remove()
    return {k: v.float() for k, v in H.items()}


# ---------------------------------------------------------------- rotations
def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def block_rotation(n, seed, block=128):
    """Random signs, then a Hadamard on each 128-wide block. Orthogonal, free to store
    (just a seed), and cheap to apply at run time (a fast Walsh-Hadamard transform)."""
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    Hb = hadamard(block)
    return torch.block_diag(*[Hb] * (n // block)) * signs.unsqueeze(1)  # R: x -> R^T x


def dense_rotation(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q * torch.sign(torch.diagonal(r))


# ---------------------------------------------------------------- scalar quantizers
def rtn_groups(W, bits, g):
    """Asymmetric round-to-nearest, one (min, step) pair in fp16 per group of g along a row."""
    r, c = W.shape
    x = W.reshape(r, c // g, g)
    lo, hi = x.amin(2, keepdim=True), x.amax(2, keepdim=True)
    s = (hi - lo).clamp_min(1e-12) / (2 ** bits - 1)
    return (lo + torch.round((x - lo) / s) * s).reshape(r, c)


def ruler_grid(W, bits, g):
    """For 8-bit rulers: one fp16 scale per row for the minimums and one for the steps,
    sized from plain rounding with 25% headroom for GPTQ's drift."""
    r, c = W.shape
    x = W.reshape(r, c // g, g)
    lo, hi = x.amin(2), x.amax(2)
    s = (hi - lo) / (2 ** bits - 1)
    return lo.abs().amax(1) * 1.25 / 127, s.amax(1) * 1.25 / 255


def gptq(W, H, bits, g, damp=0.01, block=128, ruler_bits=16, params=False):
    """GPTQ: round one column at a time, and push each column's rounding error onto the
    columns not yet rounded, weighted by how the real activations correlate (H).
    ruler_bits=8 stores each group's minimum and step as 8-bit multiples of a per-row unit.
    params=True also returns (codes, lo, step) so the rulers can be fine-tuned later."""
    W = W.clone().float()
    r, c = W.shape
    if ruler_bits == 8:
        lo_unit, s_unit = ruler_grid(W, bits, g)
    codes = torch.zeros(r, c, dtype=torch.uint8)
    LO, S = torch.zeros(r, c // g), torch.zeros(r, c // g)
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(c)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    lo = s = None
    for i1 in range(0, c, block):
        i2 = min(i1 + block, c)
        W1, Q1, E1 = W[:, i1:i2].clone(), torch.zeros(r, i2 - i1), torch.zeros(r, i2 - i1)
        Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            if col % g == 0:
                grp = torch.cat([W1[:, j:], W[:, i2:]], 1)[:, :g]
                lo, hi = grp.amin(1), grp.amax(1)
                s = (hi - lo).clamp_min(1e-12) / (2 ** bits - 1)
                if ruler_bits == 8:
                    lo = torch.round(lo / lo_unit).clamp(-127, 127) * lo_unit
                    s = (torch.round(s / s_unit).clamp(1, 255) * s_unit).clamp_min(1e-12)
                LO[:, col // g], S[:, col // g] = lo, s
            w = W1[:, j]
            code = torch.clamp(torch.round((w - lo) / s), 0, 2 ** bits - 1)
            codes[:, col] = code.to(torch.uint8)
            q = lo + code * s
            Q1[:, j] = q
            e = (w - q) / Hi[j, j]
            W1[:, j:] -= e.unsqueeze(1) * Hi[j, j:].unsqueeze(0)
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return (Q, codes, LO, S) if params else Q


def scalar_bytes(bits, g, ruler_bits=16, n=NW):
    nrows = sum(SD[k].shape[0] for k in TARGETS)
    return n * (bits + 2 * ruler_bits / g) / 8 + (4 * nrows if ruler_bits == 8 else 0)


# ---------------------------------------------------------------- shape catalogs (VQ)
def assign(X, C, chunk=32768):
    cc = (C * C).sum(1)
    return torch.cat([(cc - 2 * X[i:i + chunk] @ C.T).argmin(1) for i in range(0, len(X), chunk)])


def kmeans(X, K, iters=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:K]].clone()
    for _ in range(iters):
        a = assign(X, C)
        S = torch.zeros_like(C).index_add_(0, a, X)
        n = torch.bincount(a, minlength=K).unsqueeze(1).float()
        empty = (n == 0).squeeze(1)
        C = torch.where(empty.unsqueeze(1), X[torch.randint(len(X), (K,), generator=g)], S / n.clamp_min(1))
    return C


def residual_catalogs(X, K, stages, iters=20, seed=0):
    """Stacked catalogs (AQLM-style "2x8"): catalog 2 learns what catalog 1 missed, and so on."""
    cats, R = [], X.clone()
    for s in range(stages):
        C = kmeans(R, K, iters, seed + s)
        R = R - C[assign(R, C)]
        cats.append(C)
    return cats


def encode(X, cats, idx=False):
    """Greedy: nearest shape from each catalog in turn, summed. idx=True also returns
    which shape each catalog picked."""
    Q, picks = torch.zeros_like(X), []
    for C in cats:
        a = assign(X - Q, C)
        Q += C[a]
        picks.append(a)
    return (Q, picks) if idx else Q


def gptq_vq(W, H, cats, d, damp=0.01, params=False):
    """GPTQ with a catalog: pick the nearest d-number shape for each group, then push the
    group's error onto the columns not yet done (sequential OBS update with the codes fixed).
    params=True also returns the picks, one (rows, cols/d) index grid per catalog."""
    W = W.clone().float()
    r, c = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += damp * torch.diag(H).mean() * torch.eye(c)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    picks = [torch.zeros(r, c // d, dtype=torch.long) for _ in cats]
    for i in range(0, c, d):
        q, p = encode(W[:, i:i + d], cats, idx=True)
        for s, a in enumerate(p):
            picks[s][:, i // d] = a
        Q[:, i:i + d] = q
        for j in range(d):
            col = i + j
            e = (W[:, col] - q[:, j]) / U[col, col]
            W[:, col:] -= e.unsqueeze(1) * U[col, col:].unsqueeze(0)
    return (Q, picks) if params else Q


def vq_bytes(K, d, stages=1, n=NW):
    nrows = sum(SD[k].shape[0] for k in TARGETS)
    return stages * (n * math.log2(K) / d / 8 + K * d * 2) + nrows * 2


def log_line(text):
    with open(os.path.join(OUT, "notes.txt"), "a") as f:
        f.write(text + "\n")
