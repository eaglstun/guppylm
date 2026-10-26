"""Measure one-bit matrix sensitivity in an Aquarium checkpoint.

The starting point is browser-style int8-per-row rounding. Each test changes one
matrix to binary signs times an FP16 row scale. Held-out scoring uses the
Aquarium training code's assistant-token loss and 300 persona dialogues.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


def q8_rows(weight):
    scale = weight.abs().amax(dim=1, keepdim=True) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.round(weight / scale).clamp(-127, 127) * scale


def q1_rows(weight):
    scale = weight.abs().mean(dim=1, keepdim=True).half().float()
    sign = torch.where(weight >= 0, 1.0, -1.0)
    return sign * scale


def q8_bytes(weight):
    return weight.numel() + weight.shape[0] * 4


def q1_bytes(weight):
    return math.ceil(weight.numel() / 8) + weight.shape[0] * 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aquarium-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--greedy-loss-budget", type=float, default=0.05,
                    help="maximum loss increase over int8 for greedy mixture")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    aquarium = args.aquarium_root.resolve()
    sys.path.insert(0, str(aquarium / "tools" / "aquarium"))
    sys.path.insert(0, str(aquarium))
    from train_aq import D, GuppyConfig, GuppyLM, Tokenizer, encode_chat, load_dialogues, pad

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GuppyLM(GuppyConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    named = dict(model.named_parameters())
    matrices = {name: parameter for name, parameter in named.items() if parameter.ndim == 2}
    original = {name: parameter.detach().clone() for name, parameter in matrices.items()}
    q8 = {name: q8_rows(weight) for name, weight in original.items()}
    q1 = {name: q1_rows(weight) for name, weight in original.items()}

    tok = Tokenizer.from_file(str(Path(D) / "tokenizer.json"))
    held = load_dialogues([str(Path(D) / "dialogues_eval.jsonl")])
    if len(held) != 300:
        raise ValueError(f"expected 300 held-out dialogues; got {len(held)}")
    encoded = [encode_chat(tok, row, model.config.max_seq_len) for row in held]
    batches = [pad(encoded[i:i + 32], model.config.max_seq_len)
               for i in range(0, len(encoded), 32)]
    tokens = sum(int((y != 0).sum()) for _, y in batches)

    @torch.inference_mode()
    def score():
        return sum(model(x, y)[1].item() for x, y in batches) / len(batches)

    @torch.inference_mode()
    def install(weights):
        for name, value in weights.items():
            matrices[name].copy_(value)

    @torch.inference_mode()
    def test_q1(names):
        for name in names:
            matrices[name].copy_(q1[name])
        value = score()
        for name in names:
            matrices[name].copy_(q8[name])
        return value

    fp32_loss = score()
    install(q8)
    baseline_loss = score()
    print(f"fp32={fp32_loss:.4f} int8={baseline_loss:.4f}; {tokens} assistant tokens", flush=True)

    individual = []
    t0 = time.perf_counter()
    for name in matrices:
        loss = test_q1([name])
        entry = {
            "matrix": name, "shape": list(matrices[name].shape),
            "loss": loss, "delta_from_int8": loss - baseline_loss,
            "int8_bytes": q8_bytes(matrices[name]),
            "one_bit_bytes": q1_bytes(matrices[name]),
            "bytes_saved": q8_bytes(matrices[name]) - q1_bytes(matrices[name]),
        }
        individual.append(entry)
        print(f"{name:31s} loss={loss:.4f} delta={entry['delta_from_int8']:+.4f}", flush=True)

    names = list(matrices)
    linears = [name for name in names if name.startswith("blocks.")]
    outputs = [name for name in linears if name.endswith(("attn.out.weight", "ffn.down.weight"))]
    combinations = {}
    for label, subset in (
        ("all_matrices", names),
        ("all_block_linears", linears),
        ("all_block_outputs", outputs),
        ("token_embedding_only", ["tok_emb.weight"]),
        ("position_embedding_only", ["pos_emb.weight"]),
    ):
        combinations[label] = {"matrices": subset, "loss": test_q1(subset)}
        print(f"{label:31s} loss={combinations[label]['loss']:.4f}", flush=True)

    # A one-bit/int8 mixture: attempt the cheapest individual loss per saved byte
    # first, but judge each addition by the combined model's measured loss.
    order = sorted(individual, key=lambda e: (max(e["delta_from_int8"], 0) / e["bytes_saved"],
                                               e["delta_from_int8"]))
    accepted = []
    decisions = []
    for entry in order:
        candidate = accepted + [entry["matrix"]]
        loss = test_q1(candidate)
        keep = loss <= baseline_loss + args.greedy_loss_budget
        decisions.append({"matrix": entry["matrix"], "candidate_loss": loss, "accepted": keep})
        if keep:
            accepted = candidate
    greedy_loss = test_q1(accepted)

    actual_web = aquarium / "web" / "model" / "guppy.bin"
    web_bytes = actual_web.stat().st_size if actual_web.exists() else None
    mixed = {}
    for label, subset in [("int8", []), ("greedy", accepted)] + [
        (key, value["matrices"]) for key, value in combinations.items()
    ]:
        saved = sum(q8_bytes(matrices[n]) - q1_bytes(matrices[n]) for n in subset)
        mixed[label] = {
            "one_bit_matrices": subset,
            "estimated_payload_bytes_saved_vs_int8": saved,
            "projected_web_bytes_before_new_header": web_bytes - saved if web_bytes else None,
            "loss": baseline_loss if label == "int8" else (
                greedy_loss if label == "greedy" else combinations[label]["loss"]),
        }

    report = {
        "checkpoint": str(args.checkpoint), "aquarium_root": str(aquarium),
        "heldout_dialogues": len(held), "assistant_tokens": tokens,
        "loss_definition": "mean of batch assistant-token cross-entropy, matching quantlib.score",
        "fp32_loss": fp32_loss, "int8_baseline_loss": baseline_loss,
        "quantization": "symmetric sign times per-row mean absolute weight; FP16 stored row scales",
        "scope": "unique 2D parameters; tok_emb is tied to lm_head and counted once",
        "actual_web_bytes": web_bytes,
        "size_note": "q1 bytes = ceil(weights/8) + 2*rows; q8 bytes = weights + 4*rows. Projected file size subtracts marginal payload savings from the Aquarium v4 guppy.bin as a format reference, excluding any new header/decoder overhead. For other checkpoints it is not an actual artifact size.",
        "experiment_seconds": time.perf_counter() - t0,
        "individual": individual, "combinations": mixed,
        "greedy_loss_budget": args.greedy_loss_budget,
        "greedy_decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"greedy: {len(accepted)} matrices, loss={greedy_loss:.4f}; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
