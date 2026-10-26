"""
Speed experiments for GuppyLM: why the Mac GPU loses, and what runs a fish fastest.

Subcommands (results land in runs/speed/<cmd>.json):

  verify   KV-cached generation == uncached generation (greedy tokens, sampled tokens,
           sliding-window overflow). CPU only, no timing.
  export   ONNX exports (fp32 / int8 dynamic, uncached and KV-cached graphs), sizes,
           top-1 agreement with fp32 PyTorch, greedy-reply agreement, sample replies,
           plus a simulated int4 weight-only variant. No timing.
  time     Every timing measurement. Run without competing GPU work when measuring MPS.

Usage:
    python3 tools/speed_bench.py verify --checkpoint PATH
    python3 tools/speed_bench.py export --checkpoint PATH
    python3 tools/speed_bench.py time --checkpoint PATH
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from guppylm.config import GuppyConfig  # noqa: E402
from guppylm.eval_cases import get_eval_cases  # noqa: E402
from guppylm.model import GuppyLM  # noqa: E402

OUT_DIR = os.path.join(ROOT, "runs", "speed")
TOKENIZER_PATH = os.path.join(ROOT, "data", "tokenizer.json")
DEFAULT_CKPT = os.path.join(ROOT, "runs", "baseline", "checkpoints", "best_model.pt")
LARGE = dict(n_layers=12, d_model=768, n_heads=12, ffn_hidden=3072)  # GPT-2-small shape


def prompt_text(user_msg):
    return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid = {f.name for f in GuppyConfig.__dataclass_fields__.values()}
    mc = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid})
    model = GuppyLM(mc)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), mc


def save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote {path}")


def sync(device):
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


def summarize(xs):
    xs = sorted(xs)
    return {"median": statistics.median(xs), "min": xs[0], "max": xs[-1], "n": len(xs)}


# ── Greedy generation (no EOS stop, so every run does the same work) ───────────

@torch.inference_mode()
def greedy(model, ids, n, use_cache):
    """Append n greedy tokens to ids (B, T). Requires T + n <= max_seq_len."""
    assert ids.size(1) + n <= model.config.max_seq_len
    if not use_cache:
        idx = ids
        for _ in range(n):
            logits, _ = model(idx)
            idx = torch.cat([idx, logits[:, -1:].argmax(-1)], dim=1)
        return idx
    logits, past = model.forward_cached(ids)
    idx = ids
    for i in range(n):
        nxt = logits[:, -1:].argmax(-1)
        idx = torch.cat([idx, nxt], dim=1)
        if i < n - 1:
            logits, past = model.forward_cached(nxt, past)
    return idx


def greedy_reply(step_fn, ids, eos, max_new=64):
    """Greedy reply until EOS via a generic step function (used for torch and ORT)."""
    out = []
    state = None
    cur = ids
    for _ in range(max_new):
        nxt, state = step_fn(cur, state)
        out.append(nxt)
        if nxt == eos:
            break
        cur = [nxt]
    return out


# ── verify ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def verify_cache(model, mc, tok, device):
    res = {"device": str(device), "greedy": [], "max_abs_logit_diff": 0.0}
    for case in get_eval_cases():
        ids = tok.encode(prompt_text(case["prompt"])).ids
        x = torch.tensor([ids], device=device)
        n = mc.max_seq_len - len(ids)  # fill the whole window
        a = greedy(model, x, n, use_cache=False)[0].tolist()
        b = greedy(model, x, n, use_cache=True)[0].tolist()
        first_diff = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), None)
        res["greedy"].append({"prompt": case["prompt"], "n_new": n, "identical": a == b,
                              "first_diff": first_diff})
        # logits of the last position: full forward vs prefill + one cached step
        full, _ = model(torch.tensor([a[:len(ids) + 8]], device=device))
        lg, past = model.forward_cached(torch.tensor([a[:len(ids) + 7]], device=device))
        lg2, _ = model.forward_cached(torch.tensor([[a[len(ids) + 7]]], device=device), past)
        diff = (full[0, -1] - lg2[0, -1]).abs().max().item()
        diff = max(diff, (full[0, :-1] - lg[0]).abs().max().item())
        res["max_abs_logit_diff"] = max(res["max_abs_logit_diff"], diff)
    res["greedy_identical"] = sum(g["identical"] for g in res["greedy"])
    res["greedy_total"] = len(res["greedy"])

    # Sampled path through model.generate, same seed; and window overflow (200 new tokens).
    samp = []
    for case in get_eval_cases()[:8]:
        ids = torch.tensor([tok.encode(prompt_text(case["prompt"])).ids], device=device)
        torch.manual_seed(7)
        a, _ = model.generate(ids, 64, use_cache=False)
        torch.manual_seed(7)
        b, _ = model.generate(ids, 64, use_cache=True)
        samp.append(a.tolist() == b.tolist())
    res["sampled_identical"] = f"{sum(samp)}/{len(samp)}"

    ids = torch.tensor([tok.encode(prompt_text("hi guppy")).ids], device=device)
    model.config.eos_id = -1  # never stop, so the window overflows
    torch.manual_seed(3)
    a, _ = model.generate(ids, 200, use_cache=False)
    torch.manual_seed(3)
    b, _ = model.generate(ids, 200, use_cache=True)
    model.config.eos_id = mc.eos_id = 2
    res["overflow_200_identical"] = a.tolist() == b.tolist()
    return res


def cmd_verify(args):
    model, mc = load_model(args.checkpoint)
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    res = verify_cache(model, mc, tok, torch.device("cpu"))
    print(json.dumps({k: v for k, v in res.items() if k != "greedy"}, indent=2))
    save("verify_cpu", res)


# ── export ─────────────────────────────────────────────────────────────────────

class Prefill(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        logits, kvs = self.m.forward_cached(input_ids)
        return (logits[:, -1], torch.stack([k for k, _ in kvs]),
                torch.stack([v for _, v in kvs]))


class Decode(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, past_k, past_v):
        past = [(past_k[i], past_v[i]) for i in range(self.m.config.n_layers)]
        logits, kvs = self.m.forward_cached(input_ids, past)
        return (logits[:, -1], torch.stack([k for k, _ in kvs]),
                torch.stack([v for _, v in kvs]))


def onnx_paths():
    d = os.path.join(OUT_DIR, "onnx")
    return {
        "fp32": os.path.join(d, "full_fp32.onnx"),
        "int8": os.path.join(d, "full_int8.onnx"),        # QInt8 signed weights
        "uint8": os.path.join(d, "full_uint8.onnx"),      # QUInt8, what export_onnx.py ships
        "prefill_fp32": os.path.join(d, "prefill_fp32.onnx"),
        "decode_fp32": os.path.join(d, "decode_fp32.onnx"),
        "prefill_int8": os.path.join(d, "prefill_int8.onnx"),
        "decode_int8": os.path.join(d, "decode_int8.onnx"),
    }


def do_export(model, mc):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    p = onnx_paths()
    os.makedirs(os.path.dirname(p["fp32"]), exist_ok=True)
    L, H, D = mc.n_layers, mc.n_heads, mc.d_model // mc.n_heads
    common = dict(opset_version=17, dynamo=False)
    torch.onnx.export(model, (torch.randint(3, mc.vocab_size, (1, 32)),), p["fp32"],
                      input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "logits": {0: "b", 1: "t"}},
                      **common)
    kv_axes = {0: "layer", 1: "b", 3: "p"}
    torch.onnx.export(Prefill(model).eval(), (torch.randint(3, mc.vocab_size, (1, 16)),),
                      p["prefill_fp32"], input_names=["input_ids"],
                      output_names=["logits", "k", "v"],
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "logits": {0: "b"},
                                    "k": kv_axes, "v": kv_axes}, **common)
    pk = torch.randn(L, 1, H, 16, D)
    torch.onnx.export(Decode(model).eval(), (torch.randint(3, mc.vocab_size, (1, 1)), pk, pk.clone()),
                      p["decode_fp32"], input_names=["input_ids", "past_k", "past_v"],
                      output_names=["logits", "k", "v"],
                      dynamic_axes={"input_ids": {0: "b"}, "past_k": kv_axes, "past_v": kv_axes,
                                    "logits": {0: "b"}, "k": kv_axes, "v": kv_axes}, **common)
    # A wrapper Module defaults to training mode; export restores that mode recursively,
    # which would silently switch dropout back on in `model`. Pin it to eval.
    model.eval()
    quantize_dynamic(p["fp32"], p["int8"], weight_type=QuantType.QInt8)
    quantize_dynamic(p["fp32"], p["uint8"], weight_type=QuantType.QUInt8)
    quantize_dynamic(p["prefill_fp32"], p["prefill_int8"], weight_type=QuantType.QInt8)
    quantize_dynamic(p["decode_fp32"], p["decode_int8"], weight_type=QuantType.QInt8)
    return p


def ort_session(path, threads=0):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


class OrtCached:
    """Greedy/sampled decoding with the KV-cached ONNX graphs."""

    def __init__(self, prefill_path, decode_path, threads=0):
        self.pre = ort_session(prefill_path, threads)
        self.dec = ort_session(decode_path, threads)

    def prefill(self, ids):
        return self.pre.run(None, {"input_ids": np.asarray(ids, dtype=np.int64)})

    def step(self, ids, k, v):
        return self.dec.run(None, {"input_ids": np.asarray(ids, dtype=np.int64),
                                   "past_k": k, "past_v": v})

    def step_fn(self):
        def fn(cur, state):
            if state is None:
                logits, k, v = self.prefill([cur])
            else:
                logits, k, v = self.step([cur], *state)
            return int(logits[0].argmax()), (k, v)
        return fn


def torch_step_fn(model):
    @torch.inference_mode()
    def fn(cur, state):
        x = torch.tensor([cur])
        logits, past = model.forward_cached(x, state)
        return int(logits[0, -1].argmax()), past
    return fn


def fake_quant_int4(model, group=32):
    """Simulate int4 weight-only quantization (symmetric, per-group of `group` inputs)
    on every Linear and the (tied) embedding. Returns a quantized deep copy."""
    import copy
    q = copy.deepcopy(model)
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)) and mod.weight.dim() == 2:
                w = mod.weight
                out_f, in_f = w.shape
                g = w.reshape(out_f, in_f // group, group)
                scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
                mod.weight.copy_((g / scale).round().clamp(-8, 7).mul(scale).reshape(out_f, in_f))
    return q


@torch.inference_mode()
def sample_reply_torch(model, tok, prompt, seed):
    ids = tok.encode(prompt_text(prompt)).ids
    torch.manual_seed(seed)
    out, _ = model.generate(torch.tensor([ids]), 64, use_cache=True)
    return tok.decode(out[0].tolist()[len(ids):]).split("<|im_end|>")[0].strip()


def sample_reply_ort(oc, tok, prompt, seed, temperature=0.7, top_k=50):
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt_text(prompt)).ids
    logits, k, v = oc.prefill([ids])
    out = []
    for _ in range(64):
        lg = logits[0] / temperature
        thresh = np.sort(lg)[-top_k]
        lg = np.where(lg < thresh, -np.inf, lg)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        if nxt == 2:
            break
        logits, k, v = oc.step([[nxt]], k, v)
    return tok.decode(out).split("<|im_end|>")[0].strip()


def cmd_export(args):
    model, mc = load_model(args.checkpoint)
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    p = do_export(model, mc)
    res = {"sizes_mb": {k: os.path.getsize(v) / 1e6 for k, v in p.items()}}
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "onnx", "state_dict.pt"))
    res["sizes_mb"]["torch_state_dict_fp32"] = os.path.getsize(
        os.path.join(OUT_DIR, "onnx", "state_dict.pt")) / 1e6
    n_params = sum(t.numel() for t in model.parameters())
    res["n_params"] = n_params
    res["int4_estimated_mb"] = n_params * (0.5 + 2 / 32) / 1e6  # 4-bit + fp16 scale / 32

    cases = get_eval_cases()
    prompts = [tok.encode(prompt_text(c["prompt"])).ids[-mc.max_seq_len:] for c in cases]

    # Top-1 agreement on every prompt position (same metric as tools/benchmark.py).
    def agreement(logit_fn):
        agree = total = 0
        max_err = 0.0
        for ids in prompts:
            with torch.inference_mode():
                ref = model(torch.tensor([ids]))[0][0].numpy()
            got = logit_fn(ids)
            agree += int((ref.argmax(-1) == got.argmax(-1)).sum())
            total += len(ids)
            max_err = max(max_err, float(np.abs(ref - got).max()))
        return {"top1_agreement": agree / total, "max_abs_logit_err": max_err, "positions": total}

    variants = {}
    for name in ("fp32", "int8", "uint8"):
        s = ort_session(p[name])
        variants[f"onnx_{name}"] = agreement(
            lambda ids, s=s: s.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0][0])
    q4 = fake_quant_int4(model)
    variants["torch_int4_sim_g32"] = agreement(
        lambda ids: q4(torch.tensor([ids]))[0][0].detach().numpy())
    res["agreement_vs_torch_fp32"] = variants

    # Greedy reply agreement: full replies (until EOS) vs fp32 torch.
    ref_replies = [greedy_reply(torch_step_fn(model), ids, mc.eos_id) for ids in prompts]
    greedy_match = {}
    for name in ("fp32", "int8"):
        oc = OrtCached(p[f"prefill_{name}"], p[f"decode_{name}"])
        got = [greedy_reply(oc.step_fn(), ids, mc.eos_id) for ids in prompts]
        greedy_match[f"onnx_cached_{name}"] = sum(a == b for a, b in zip(ref_replies, got))
    got = [greedy_reply(torch_step_fn(q4), ids, mc.eos_id) for ids in prompts]
    greedy_match["torch_int4_sim_g32"] = sum(a == b for a, b in zip(ref_replies, got))
    greedy_match["total"] = len(prompts)
    res["greedy_reply_identical_to_fp32"] = greedy_match
    res["greedy_replies"] = {
        "fp32_torch": {c["prompt"]: tok.decode(r).split("<|im_end|>")[0].strip()
                       for c, r in zip(cases, ref_replies)},
    }

    # Sampled replies (temperature 0.7, top-k 50) to eyeball fishiness.
    oc8 = OrtCached(p["prefill_int8"], p["decode_int8"])
    samples = {}
    for c in cases[:8]:
        samples[c["prompt"]] = {
            "torch_fp32": sample_reply_torch(model, tok, c["prompt"], 11),
            "onnx_int8": sample_reply_ort(oc8, tok, c["prompt"], 11),
            "torch_int4_sim": sample_reply_torch(q4, tok, c["prompt"], 11),
        }
    res["sampled_replies"] = samples
    print(json.dumps({k: v for k, v in res.items() if k not in ("greedy_replies",)}, indent=2))
    save("export", res)


# ── time ───────────────────────────────────────────────────────────────────────

def time_fn(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync(device)
        ts.append((time.perf_counter() - t) * 1000)
    return summarize(ts)


def count_ops(model, x):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            Count.n += 1
            return func(*args, **(kwargs or {}))

    with torch.inference_mode(), Count():
        model(x)
    return Count.n


@torch.inference_mode()
def phase_overhead(model, mc, log):
    """Is MPS compute-bound or dispatch-bound at batch 1?"""
    res = {}
    res["aten_ops_per_forward_T128"] = count_ops(model, torch.randint(3, mc.vocab_size, (1, 128)))
    # Tiny op cost: synced per op vs pipelined (sync once per 1000).
    for dev in ("cpu", "mps"):
        a = torch.randn(1, 384, device=dev)
        res[f"{dev}_tiny_op_synced_us"] = time_fn(lambda: a.add(1.0), dev, 50, 500)["median"] * 1000

        def burst():
            for _ in range(1000):
                a.add(1.0)
        res[f"{dev}_tiny_op_pipelined_us"] = time_fn(burst, dev, 2, 7)["median"]  # ms/1000 = us
    # Forward at T=128, synced every call vs 50 queued then one sync.
    m = model.to("mps")
    x = torch.randint(3, mc.vocab_size, (1, 128), device="mps")
    res["mps_fwd_T128_synced_ms"] = time_fn(lambda: m(x), "mps", 10, 50)["median"]

    def burst50():
        for _ in range(50):
            m(x)
    res["mps_fwd_T128_pipelined_ms"] = time_fn(burst50, "mps", 1, 7)["median"] / 50
    model.to("cpu")
    log("overhead", res)
    return res


@torch.inference_mode()
def phase_baseline(model, mc, tok, log, iters):
    res = {"torch_ms": {}, "gen_tps": {}}
    prompt = torch.tensor([tok.encode(prompt_text("hi guppy")).ids])
    for dev in ("cpu", "mps"):
        m = model.to(dev)
        res["torch_ms"][dev] = {}
        for T in (16, 32, 64, 128):
            x = torch.randint(3, mc.vocab_size, (1, T), device=dev)
            res["torch_ms"][dev][T] = time_fn(lambda: m(x), dev, 5, iters)
        ids = prompt.to(dev)
        for n in (64, 110):
            for cache in (False, True):
                ts = []
                greedy(m, ids, n, cache)  # warmup
                for _ in range(5):
                    sync(dev)
                    t = time.perf_counter()
                    greedy(m, ids, n, cache)
                    sync(dev)
                    ts.append(n / (time.perf_counter() - t))
                key = f"{dev}_{'cached' if cache else 'uncached'}_{n}tok"
                res["gen_tps"][key] = summarize(ts)
        model.to("cpu")
    # Baseline's shipped ONNX (uint8 quantized), same latency points.
    base_onnx = os.path.join(os.path.dirname(os.path.dirname(DEFAULT_CKPT)), "model.onnx")
    if os.path.exists(base_onnx):
        s = ort_session(base_onnx)
        res["baseline_model_onnx_ms"] = {}
        for T in (16, 32, 64, 128):
            x = np.random.randint(3, mc.vocab_size, (1, T)).astype(np.int64)
            res["baseline_model_onnx_ms"][T] = time_fn(
                lambda: s.run(None, {"input_ids": x}), "cpu", 5, iters)
    log("baseline", res)
    return res


@torch.inference_mode()
def phase_batch(log, ckpt, budget_s=8.0):
    """Throughput vs batch size for the trained small model and a random large one."""
    res = {}
    small, _ = load_model(ckpt)
    torch.manual_seed(0)
    large = GuppyLM(GuppyConfig(**LARGE)).eval()
    res["params"] = {"small": sum(p.numel() for p in small.parameters()),
                     "large": sum(p.numel() for p in large.parameters())}
    prompt_len, n_new = 16, 32
    for mname, model in (("small", small), ("large", large)):
        for dev in ("cpu", "mps"):
            m = model.to(dev)
            for kind in ("prefill128", "gen_cached", "gen_uncached"):
                for B in (1, 4, 16, 64, 256):
                    key = f"{mname}/{dev}/{kind}/B{B}"
                    if kind == "prefill128":
                        x = torch.randint(3, m.config.vocab_size, (B, 128), device=dev)
                        fn, work = (lambda: m(x)), B * 128
                    else:
                        x = torch.randint(3, m.config.vocab_size, (B, prompt_len), device=dev)
                        cache = kind == "gen_cached"
                        fn, work = (lambda: greedy(m, x, n_new, cache)), B * n_new
                    t = time.perf_counter()
                    fn()
                    sync(dev)
                    first = time.perf_counter() - t
                    reps = 1 if first > budget_s else (2 if first > 2 else 3)
                    ts = []
                    for _ in range(reps):
                        t = time.perf_counter()
                        fn()
                        sync(dev)
                        ts.append(work / (time.perf_counter() - t))
                    res[key] = summarize(ts)
                    print(f"  {key}: {res[key]['median']:.0f} tok/s ({reps} reps)", flush=True)
                    if first > 3 * budget_s:  # next batch is ~4x the work: stop before a >~100 s run
                        res[f"{mname}/{dev}/{kind}/skipped_above"] = B
                        break
            model.to("cpu")
    log("batch", res)
    return res


@torch.inference_mode()
def phase_onnx(model, mc, tok, log, iters):
    p = onnx_paths()
    res = {"latency_ms": {}, "gen_tps": {}}
    for name in ("fp32", "int8", "uint8"):
        s = ort_session(p[name])
        res["latency_ms"][name] = {}
        for T in (16, 32, 64, 128):
            x = np.random.randint(3, mc.vocab_size, (1, T)).astype(np.int64)
            res["latency_ms"][name][T] = time_fn(
                lambda: s.run(None, {"input_ids": x}), "cpu", 5, iters)
    ids = tok.encode(prompt_text("hi guppy")).ids
    n = 64

    def gen_uncached(s):
        idx = list(ids)
        for _ in range(n):
            lg = s.run(None, {"input_ids": np.array([idx], dtype=np.int64)})[0]
            idx.append(int(lg[0, -1].argmax()))

    def gen_cached(oc):
        logits, k, v = oc.prefill([ids])
        for _ in range(n - 1):
            logits, k, v = oc.step([[int(logits[0].argmax())]], k, v)

    for name in ("fp32", "int8"):
        s = ort_session(p[name])
        oc = OrtCached(p[f"prefill_{name}"], p[f"decode_{name}"])
        for label, fn in ((f"onnx_{name}_uncached", lambda: gen_uncached(s)),
                          (f"onnx_{name}_cached", lambda: gen_cached(oc))):
            fn()
            ts = []
            for _ in range(5):
                t = time.perf_counter()
                fn()
                ts.append(n / (time.perf_counter() - t))
            res["gen_tps"][label] = summarize(ts)
    # Thread sensitivity of the cached int8 path (single-stream decode is latency-bound).
    for th in (1, 2, 4):
        oc = OrtCached(p["prefill_int8"], p["decode_int8"], threads=th)
        gen_cached(oc)
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            gen_cached(oc)
            ts.append(n / (time.perf_counter() - t))
        res["gen_tps"][f"onnx_int8_cached_threads{th}"] = summarize(ts)
    # PyTorch CPU with 1 thread, for the same comparison.
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.tensor([ids])
    greedy(model, x, n, True)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        greedy(model, x, n, True)
        ts.append(n / (time.perf_counter() - t))
    res["gen_tps"]["torch_cpu_cached_threads1"] = summarize(ts)
    torch.set_num_threads(nt)
    log("onnx", res)
    return res


@torch.inference_mode()
def phase_extras(model, mc, tok, log):
    """fp16 on MPS and torch.compile, briefly."""
    res = {}
    ids = torch.tensor([tok.encode(prompt_text("hi guppy")).ids])
    m16 = model.to("mps").half()
    x = ids.to("mps")
    for cache in (False, True):
        greedy(m16, x, 64, cache)
        ts = []
        for _ in range(5):
            sync("mps")
            t = time.perf_counter()
            greedy(m16, x, 64, cache)
            sync("mps")
            ts.append(64 / (time.perf_counter() - t))
        res[f"mps_fp16_{'cached' if cache else 'uncached'}_64tok"] = summarize(ts)
    for B in (64, 256):
        xb = torch.randint(3, mc.vocab_size, (B, 128), device="mps")
        r = time_fn(lambda: m16(xb), "mps", 2, 5)
        res[f"mps_fp16_prefill128_B{B}_tok_s"] = B * 128 / (r["median"] / 1000)
    model.float().to("cpu")
    for dev in ("cpu", "mps"):
        m = model.to(dev)
        x = torch.randint(3, mc.vocab_size, (1, 128), device=dev)
        try:
            t = time.perf_counter()
            cm = torch.compile(m)
            cm(x)
            sync(dev)
            res[f"compile_{dev}_compile_s"] = time.perf_counter() - t
            res[f"compile_{dev}_fwd_T128_ms"] = time_fn(lambda: cm(x), dev, 5, 30)
            res[f"eager_{dev}_fwd_T128_ms"] = time_fn(lambda: m(x), dev, 5, 30)
        except Exception as e:  # report, do not hide
            res[f"compile_{dev}_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        torch._dynamo.reset()
        model.to("cpu")
    log("extras", res)
    return res


def cmd_time(args):
    model, mc = load_model(args.checkpoint)
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    out = {"env": {
        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip(),
        "torch": torch.__version__, "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    try:
        import onnxruntime
        out["env"]["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass

    def log(name, r):
        out[name] = r
        save("time", out)  # save after every phase so a crash keeps earlier results

    phases = args.phases.split(",")
    t0 = time.perf_counter()
    if "verify" in phases:  # MPS correctness check needs the GPU, so it lives here
        out["verify_mps"] = {k: v for k, v in verify_cache(
            model.to("mps"), mc, tok, torch.device("mps")).items() if k != "greedy"}
        model.to("cpu")
    if "overhead" in phases:
        phase_overhead(model, mc, log)
    if "baseline" in phases:
        phase_baseline(model, mc, tok, log, args.iters)
    if "onnx" in phases:
        phase_onnx(model, mc, tok, log, args.iters)
    if "batch" in phases:
        phase_batch(log, args.checkpoint)
    if "extras" in phases:
        phase_extras(model, mc, tok, log)
    out["env"]["elapsed_s"] = time.perf_counter() - t0
    save("time", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("verify", cmd_verify), ("export", cmd_export), ("time", cmd_time)):
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", default=DEFAULT_CKPT)
        s.set_defaults(fn=fn)
        if name == "time":
            s.add_argument("--iters", type=int, default=50)
            s.add_argument("--phases", default="verify,overhead,baseline,onnx,batch,extras")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
