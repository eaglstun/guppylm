"""
Train a GuppyLM run and benchmark it, so changes can be compared run-to-run.

Each run lives in runs/<name>/ (checkpoints, model.onnx, results.json).

Measures:
  data      token-length stats of the training set vs. max_seq_len
  train     wall time, steps/s, tokens/s, best + full-set eval loss
  latency   forward-pass time at several context lengths (PyTorch + ONNX),
            and end-to-end generation tokens/s (no KV cache, so this grows with length)
  onnx      quantized ONNX vs. PyTorch fp32: top-1 agreement + logit error
  quality   eval_cases keyword hit rate + lowercase rate (seeded sampling)

Usage:
    python tools/benchmark.py run --name baseline
    python tools/benchmark.py run --name ctx256 --max-seq-len 256
    python tools/benchmark.py run --name baseline --skip-train   # re-benchmark existing run
    python tools/benchmark.py run --name tiny --n-layers 2 --d-model 64 --n-heads 4
    python tools/benchmark.py curve --ckpt-dir runs/baseline/checkpoints --out curve.json
    python tools/benchmark.py compare                            # table of all runs
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # guppylm.prepare_data / generate_data write to ./data

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from guppylm.config import GuppyConfig, TrainConfig  # noqa: E402
from guppylm.dataset import get_dataloader, subset_lines  # noqa: E402
from guppylm import eval_cases as fish_eval_cases  # noqa: E402
from guppylm import eval_cases_dog  # noqa: E402
from guppylm.model import GuppyLM  # noqa: E402
from guppylm.train import evaluate, get_device, train  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "runs")
DATA_DIR = os.path.join(ROOT, "data")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
PERSONA = "fish"


def get_eval_cases():
    return (eval_cases_dog if PERSONA == "dog" else fish_eval_cases).get_eval_cases()


def prompt_text(user_msg):
    return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


# ── Data ────────────────────────────────────────────────────────────────────

def ensure_data(regen):
    if regen or not os.path.exists(TOKENIZER_PATH):
        from guppylm.prepare_data import prepare
        prepare(data_dir=DATA_DIR, persona=PERSONA)


def data_stats(max_seq_len, limit=None, subset_seed=0):
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    lines = subset_lines(os.path.join(DATA_DIR, "train.jsonl"), limit, subset_seed)
    lens = [len(tok.encode(json.loads(line)["text"]).ids) for line in lines]
    return {
        "n_train": len(lens),
        "len_mean": statistics.mean(lens),
        "len_p50": pctl(lens, 50),
        "len_p99": pctl(lens, 99),
        "len_max": max(lens),
        # dataset.py truncates to max_seq_len, then x = ids[:-1]
        "n_truncated": sum(n > max_seq_len for n in lens),
    }


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid = {f.name for f in GuppyConfig.__dataclass_fields__.values()}
    mc = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid})
    model = GuppyLM(mc)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), mc, ckpt


# ── Latency ─────────────────────────────────────────────────────────────────

def time_it(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def context_lengths(max_seq_len):
    lens = [n for n in (16, 32, 64, 128, 256, 512, 1024) if n < max_seq_len]
    return lens + [max_seq_len]


@torch.no_grad()
def torch_latency(model, mc, device, iters):
    out = {}
    for T in context_lengths(mc.max_seq_len):
        x = torch.randint(3, mc.vocab_size, (1, T), device=device)
        out[T] = time_it(lambda: model(x), device, warmup=5, iters=iters)
    return out


@torch.no_grad()
def torch_gen_tps(model, mc, device, n_tokens=64):
    """Tokens/s generating n_tokens past a real prompt, ignoring EOS."""
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    ids = torch.tensor([tok.encode(prompt_text("hi guppy")).ids], device=device)

    def run():
        idx = ids
        for _ in range(n_tokens):
            logits, _ = model(idx[:, -mc.max_seq_len:])
            idx = torch.cat([idx, logits[:, -1:].argmax(-1)], dim=1)

    run()  # warmup
    sync(device)
    t = time.perf_counter()
    run()
    sync(device)
    return n_tokens / (time.perf_counter() - t)


def ort_session(onnx_path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    return ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])


def ort_latency(sess, mc, iters):
    import numpy as np
    cpu = torch.device("cpu")
    out = {}
    for T in context_lengths(mc.max_seq_len):
        x = np.random.randint(3, mc.vocab_size, (1, T)).astype(np.int64)
        out[T] = time_it(lambda: sess.run(None, {"input_ids": x}), cpu, warmup=5, iters=iters)
    return out


@torch.no_grad()
def onnx_agreement(sess, model_cpu, mc):
    """Compare quantized ONNX logits to fp32 PyTorch on every eval prompt position."""
    import numpy as np
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    agree = total = 0
    max_err = 0.0
    for case in get_eval_cases():
        ids = tok.encode(prompt_text(case["prompt"])).ids[-mc.max_seq_len:]
        ref = model_cpu(torch.tensor([ids]))[0][0].numpy()
        got = sess.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0][0]
        agree += int((ref.argmax(-1) == got.argmax(-1)).sum())
        total += len(ids)
        max_err = max(max_err, float(np.abs(ref - got).max()))
    return {"top1_agreement": agree / total, "max_abs_logit_err": max_err}


# ── Quality ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def quality(model_cpu, samples_per_case, seed):
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    torch.manual_seed(seed)
    hits = scored = lower = n = 0
    examples = {}
    for case in get_eval_cases():
        ids = tok.encode(prompt_text(case["prompt"])).ids
        for i in range(samples_per_case):
            out, _ = model_cpu.generate(torch.tensor([ids]), max_new_tokens=64)
            text = tok.decode(out[0].tolist()[len(ids):]).split("<|im_end|>")[0].strip()
            if i == 0:
                examples[case["prompt"]] = text
            n += 1
            lower += int(text == text.lower())
            if case.get("expect_keywords"):
                scored += 1
                hits += int(any(k in text for k in case["expect_keywords"]))
    return {
        "keyword_hit_rate": hits / max(1, scored),
        "lowercase_rate": lower / max(1, n),
        "examples": examples,
    }


# ── Commands ────────────────────────────────────────────────────────────────

def model_config_from_args(args):
    """GuppyConfig from CLI shape flags; unset flags keep the GuppyConfig defaults.

    If --d-model is given without --ffn-hidden, the FFN keeps the default 2x ratio.
    """
    kw = {"max_seq_len": args.max_seq_len}
    for k in ("d_model", "n_layers", "n_heads", "ffn_hidden"):
        if getattr(args, k) is not None:
            kw[k] = getattr(args, k)
    if args.d_model is not None and args.ffn_hidden is None:
        kw["ffn_hidden"] = 2 * args.d_model
    mc = GuppyConfig(**kw)
    if mc.d_model % mc.n_heads:
        sys.exit(f"d_model {mc.d_model} is not divisible by n_heads {mc.n_heads}")
    return mc


def cmd_curve(args):
    """Score a run's periodic checkpoints (step_N.pt) for loss + quality: a learning curve."""
    global DATA_DIR, TOKENIZER_PATH, PERSONA
    DATA_DIR = os.path.abspath(args.data_dir)
    TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
    PERSONA = args.persona
    device = get_device(TrainConfig(device=args.device))
    ckpt_dir = args.ckpt_dir
    steps = [int(s) for s in args.steps.split(",")]
    tok_eval = None
    rows = []
    for step in steps:
        path = os.path.join(ckpt_dir, f"step_{step}.pt")
        if not os.path.exists(path):
            path = os.path.join(ckpt_dir, "final_model.pt")
            if not os.path.exists(path):
                sys.exit(f"No checkpoint for step {step} in {ckpt_dir}")
            final_step = torch.load(path, map_location="cpu", weights_only=False)["step"]
            if final_step != step:
                sys.exit(f"step_{step}.pt missing; final_model.pt is step {final_step}")
        model, mc, ckpt = load_model(path, device)
        if tok_eval is None:
            tok_eval = get_dataloader(os.path.join(DATA_DIR, "eval.jsonl"), TOKENIZER_PATH,
                                      mc.max_seq_len, TrainConfig.batch_size, shuffle=False)
        loss = evaluate(model, tok_eval, device, max_batches=10**9)
        model_cpu, _, _ = load_model(path, torch.device("cpu"))
        q = quality(model_cpu, args.samples, args.seed)
        extra = {}
        for prompt in args.prompt:
            torch.manual_seed(args.seed)
            extra[prompt] = sample_replies(model_cpu, prompt, args.n_extra)
        row = {"step": ckpt["step"], "eval_loss": loss, "perplexity": math.exp(loss),
               "keyword_hit_rate": q["keyword_hit_rate"], "lowercase_rate": q["lowercase_rate"],
               "examples": q["examples"], "extra_prompts": extra}
        rows.append(row)
        print(f"\n== step {row['step']} == loss {loss:.4f}  kw {q['keyword_hit_rate']:.1%}  "
              f"lower {q['lowercase_rate']:.1%}")
        for p, rs in extra.items():
            for r in rs:
                print(f"  {p!r} -> {r!r}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt_dir": ckpt_dir, "seed": args.seed, "rows": rows}, f, indent=2)
        print(f"\nWrote {args.out}")


@torch.no_grad()
def sample_replies(model_cpu, prompt, n):
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    ids = tok.encode(prompt_text(prompt)).ids
    outs = []
    for _ in range(n):
        out, _ = model_cpu.generate(torch.tensor([ids]), max_new_tokens=64)
        outs.append(tok.decode(out[0].tolist()[len(ids):]).split("<|im_end|>")[0].strip())
    return outs


def cmd_run(args):
    global DATA_DIR, TOKENIZER_PATH, PERSONA
    DATA_DIR = os.path.abspath(args.data_dir)
    TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
    PERSONA = args.persona
    run_dir = os.path.join(RUNS_DIR, args.name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)

    mc = model_config_from_args(args)
    tc = TrainConfig(max_steps=args.steps, batch_size=args.batch_size, device=args.device,
                     seed=args.seed, data_dir=DATA_DIR, output_dir=ckpt_dir,
                     train_limit=args.train_limit, subset_seed=args.subset_seed)
    device = get_device(tc)

    ensure_data(args.regen_data)
    results = {
        "name": args.name,
        "persona": PERSONA,
        "data_dir": os.path.relpath(DATA_DIR, ROOT),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": {"torch": torch.__version__, "device": str(device),
                "machine": platform.machine(), "processor": platform.processor()},
        "model_config": vars(mc),
        "params": GuppyLM(mc).param_count()[0],
        "train_config": {k: v for k, v in vars(tc).items() if k not in ("data_dir", "output_dir")},
        "data": data_stats(mc.max_seq_len, tc.train_limit, tc.subset_seed),
    }
    print(f"\n== data ==\n{json.dumps(results['data'], indent=2)}")

    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if not args.skip_train:
        print("\n== train ==")
        stats = train(mc, tc)
        n_tokens = min(results["data"]["len_mean"], mc.max_seq_len) - 1
        results["train"] = {
            "steps": stats["steps"],
            "elapsed_s": stats["elapsed_s"],
            "steps_per_s": stats["steps"] / stats["elapsed_s"],
            "approx_tokens_per_s": stats["steps"] * tc.batch_size * n_tokens / stats["elapsed_s"],
            "best_eval_loss_50_batches": stats["best_eval"],
            "final_train_loss_avg100": statistics.mean(stats["train_losses"][-100:]),
            "epochs": stats["steps"] * tc.batch_size / results["data"]["n_train"],
            "eval_history": stats["eval_history"],
        }
    elif not os.path.exists(ckpt_dir):
        sys.exit(f"--skip-train but no checkpoints in {ckpt_dir}")
    else:
        prev = os.path.join(run_dir, "results.json")
        if os.path.exists(prev):
            with open(prev) as f:
                results["train"] = json.load(f).get("train")

    if not os.path.exists(best_path):  # runs shorter than eval_interval never save a best
        best_path = os.path.join(ckpt_dir, "final_model.pt")

    print("\n== eval (full set) ==")
    model, mc, ckpt = load_model(best_path, device)
    eval_loader = get_dataloader(os.path.join(DATA_DIR, "eval.jsonl"), TOKENIZER_PATH,
                                 mc.max_seq_len, tc.batch_size, shuffle=False)
    full = evaluate(model, eval_loader, device, max_batches=10**9)
    results["eval"] = {"best_step": ckpt["step"], "loss": full, "perplexity": math.exp(full)}
    # Score final as well as best: early stopping can hide later overfitting.
    train_loader = get_dataloader(os.path.join(DATA_DIR, "train.jsonl"), TOKENIZER_PATH,
                                  mc.max_seq_len, tc.batch_size, shuffle=False,
                                  limit=tc.train_limit, subset_seed=tc.subset_seed)
    results["eval"]["train_loss_eval_mode"] = evaluate(model, train_loader, device, max_batches=100)
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    if os.path.exists(final_path):
        fmodel, _, fckpt = load_model(final_path, device)
        results["eval"]["final_step"] = fckpt["step"]
        results["eval"]["final_loss"] = evaluate(fmodel, eval_loader, device, max_batches=10**9)
        results["eval"]["final_train_loss_eval_mode"] = evaluate(fmodel, train_loader, device,
                                                                 max_batches=100)
        del fmodel
    print(json.dumps(results["eval"], indent=2))

    print("\n== latency ==")
    model_cpu, _, _ = load_model(best_path, torch.device("cpu"))
    lat = {"torch_cpu_ms": torch_latency(model_cpu, mc, torch.device("cpu"), args.iters),
           "torch_cpu_gen_tps": torch_gen_tps(model_cpu, mc, torch.device("cpu"))}
    if device.type != "cpu":
        lat[f"torch_{device.type}_ms"] = torch_latency(model, mc, device, args.iters)
        lat[f"torch_{device.type}_gen_tps"] = torch_gen_tps(model, mc, device)

    if not args.no_onnx:
        print("\n== onnx export ==")
        onnx_path = os.path.join(run_dir, "model.onnx")
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "export_onnx.py"),
                        "--checkpoint", best_path, "--tokenizer", TOKENIZER_PATH,
                        "--output", onnx_path], check=True)
        sess = ort_session(onnx_path)
        lat["onnx_cpu_ms"] = ort_latency(sess, mc, args.iters)
        results["onnx"] = {"size_mb": os.path.getsize(onnx_path) / 1e6,
                           **onnx_agreement(sess, model_cpu, mc)}
        print(json.dumps(results["onnx"], indent=2))
    results["latency"] = lat
    print(json.dumps(lat, indent=2))

    print("\n== quality ==")
    results["quality"] = quality(model_cpu, args.samples, args.seed)
    q = results["quality"]
    print(f"keyword hit rate {q['keyword_hit_rate']:.1%}, lowercase {q['lowercase_rate']:.1%}")
    if args.prompts:
        q["extra_prompts"] = {}
        for prompt in args.prompts:
            torch.manual_seed(args.seed)
            q["extra_prompts"][prompt] = sample_replies(model_cpu, prompt, 3)
            print(f"  {prompt!r} -> {q['extra_prompts'][prompt]!r}")
    for p, r in list(q["examples"].items())[:5]:
        print(f"  {p!r} -> {r!r}")

    out = os.path.join(run_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


def cmd_compare(args):
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else []:
        path = os.path.join(RUNS_DIR, name, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    if not runs:
        sys.exit("No runs found in runs/")

    def get(r, *keys, fmt="{:.4f}"):
        for k in keys:
            if not isinstance(r, dict) or k not in r:
                return "-"
            r = r[k]
        return fmt.format(r) if r is not None else "-"

    rows = [
        ("params", lambda r: get(r, "params", fmt="{:,}")),
        ("layers x d_model", lambda r: f"{r['model_config']['n_layers']}x{r['model_config']['d_model']}"),
        ("max_seq_len", lambda r: get(r, "model_config", "max_seq_len", fmt="{}")),
        ("train time (s)", lambda r: get(r, "train", "elapsed_s", fmt="{:.0f}")),
        ("steps/s", lambda r: get(r, "train", "steps_per_s", fmt="{:.1f}")),
        ("eval loss", lambda r: get(r, "eval", "loss")),
        ("eval ppl", lambda r: get(r, "eval", "perplexity", fmt="{:.3f}")),
        ("train limit", lambda r: get(r, "train_config", "train_limit", fmt="{}")),
        ("epochs", lambda r: get(r, "train", "epochs", fmt="{:.1f}")),
        ("best step", lambda r: get(r, "eval", "best_step", fmt="{}")),
        ("train loss (eval mode)", lambda r: get(r, "eval", "train_loss_eval_mode")),
        ("final eval loss", lambda r: get(r, "eval", "final_loss")),
        ("final train loss (eval mode)", lambda r: get(r, "eval", "final_train_loss_eval_mode")),
        ("truncated samples", lambda r: get(r, "data", "n_truncated", fmt="{}")),
        ("keyword hit", lambda r: get(r, "quality", "keyword_hit_rate", fmt="{:.1%}")),
        ("lowercase", lambda r: get(r, "quality", "lowercase_rate", fmt="{:.1%}")),
        ("cpu gen tok/s", lambda r: get(r, "latency", "torch_cpu_gen_tps", fmt="{:.0f}")),
        ("onnx size (MB)", lambda r: get(r, "onnx", "size_mb", fmt="{:.1f}")),
        ("onnx top-1 agree", lambda r: get(r, "onnx", "top1_agreement", fmt="{:.1%}")),
    ]
    for T in (16, 64, 128, 256, 512):
        rows.append((f"torch cpu ms @{T}", lambda r, T=T: get(r, "latency", "torch_cpu_ms", str(T), fmt="{:.2f}")))
        rows.append((f"onnx cpu ms @{T}", lambda r, T=T: get(r, "latency", "onnx_cpu_ms", str(T), fmt="{:.2f}")))

    table = [["metric"] + [r["name"] for r in runs]]
    for label, fn in rows:
        vals = [fn(r) for r in runs]
        if any(v != "-" for v in vals):
            table.append([label] + vals)
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for i, row in enumerate(table):
        print("  ".join(c.ljust(w) if j == 0 else c.rjust(w) for j, (c, w) in enumerate(zip(row, widths))))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def main():
    p = argparse.ArgumentParser(description="Train + benchmark GuppyLM runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Train (unless --skip-train) and benchmark a run")
    r.add_argument("--name", required=True)
    r.add_argument("--max-seq-len", type=int, default=GuppyConfig.max_seq_len)
    r.add_argument("--steps", type=int, default=TrainConfig.max_steps)
    r.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    r.add_argument("--device", default=TrainConfig.device)
    r.add_argument("--seed", type=int, default=TrainConfig.seed)
    r.add_argument("--train-limit", type=int, default=None,
                   help="Train on a seeded random subset of N train examples (eval set unchanged)")
    r.add_argument("--subset-seed", type=int, default=0)
    r.add_argument("--iters", type=int, default=30, help="Timed iterations per latency point")
    r.add_argument("--samples", type=int, default=5, help="Generations per eval case")
    r.add_argument("--d-model", type=int, default=None, help=f"default {GuppyConfig.d_model}")
    r.add_argument("--n-layers", type=int, default=None, help=f"default {GuppyConfig.n_layers}")
    r.add_argument("--n-heads", type=int, default=None, help=f"default {GuppyConfig.n_heads}")
    r.add_argument("--ffn-hidden", type=int, default=None,
                   help=f"default {GuppyConfig.ffn_hidden}, or 2x --d-model when that is set")
    r.add_argument("--prompts", nargs="*", default=[],
                   help="Extra prompts to sample (3 replies each) into results.json")
    r.add_argument("--skip-train", action="store_true")
    r.add_argument("--regen-data", action="store_true")
    r.add_argument("--no-onnx", action="store_true")
    r.add_argument("--data-dir", default=DATA_DIR,
                   help="train/eval JSONL and tokenizer directory (default ./data)")
    r.add_argument("--persona", choices=["fish", "dog"], default="fish",
                   help="select data generator and evaluation cases")
    r.set_defaults(fn=cmd_run)

    cv = sub.add_parser("curve", help="Score periodic checkpoints of a run (learning curve)")
    cv.add_argument("--ckpt-dir", required=True)
    cv.add_argument("--steps", default="500,1000,2000,4000,10000",
                    help="Comma list; a missing step_N.pt falls back to final_model.pt")
    cv.add_argument("--device", default=TrainConfig.device)
    cv.add_argument("--seed", type=int, default=TrainConfig.seed)
    cv.add_argument("--samples", type=int, default=5)
    cv.add_argument("--prompt", action="append",
                    default=None, help="Extra prompt to sample (repeatable)")
    cv.add_argument("--n-extra", type=int, default=3, help="Replies per extra prompt")
    cv.add_argument("--out", default=None, help="Write JSON here")
    cv.add_argument("--data-dir", default=DATA_DIR,
                    help="evaluation JSONL and tokenizer directory (default ./data)")
    cv.add_argument("--persona", choices=["fish", "dog"], default="fish",
                    help="select evaluation cases")
    cv.set_defaults(fn=cmd_curve)

    c = sub.add_parser("compare", help="Print a table of all runs in runs/")
    c.set_defaults(fn=cmd_compare)

    args = p.parse_args()
    if args.cmd == "curve" and args.prompt is None:
        args.prompt = ["hi guppy", "what is the meaning of life", "tell me a joke"]
    args.fn(args)


if __name__ == "__main__":
    main()
