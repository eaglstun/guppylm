"""Measure zero-shot removal of each transformer block from an aquarium checkpoint.

    python3 tools/aquarium/prune_layer.py runs/aq-chat-v5/best.pt --out runs/quant-v5/layer_drop.json

The score matches the aquarium quantization experiment: mean held-out-persona
dialogue loss across batches, assistant tokens only. No fine-tuning is performed.
"""
import argparse
import json
import os
import torch

from train_aq import D, GuppyConfig, GuppyLM, Tokenizer, encode_chat, load_dialogues, pad, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = GuppyConfig(**ck["config"])
    model = GuppyLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
    rows = load_dialogues([os.path.join(D, "dialogues_eval.jsonl")])
    examples = [encode_chat(tokenizer, row, 512) for row in rows]
    batches = [pad(examples[i:i + 32], 512) for i in range(0, len(examples), 32)]

    @torch.no_grad()
    def score():
        model.eval()  # sample() switches the model back to train mode
        return sum(model(X, Y)[1].item() for X, Y in batches) / len(batches)

    @torch.no_grad()
    def reply():
        torch.manual_seed(1)
        (_, text), = sample(model, tokenizer, "cpu",
                           "You are Pebble, a hermit crab who lives in a sandy tide pool. You are careful and practical.",
                           ["What is a toaster?"], max_new=50)
        return text

    total_params = sum(p.numel() for p in model.parameters())
    original = {"dropped": None, "loss": score(), "params": total_params, "reply": reply()}
    results = [original]
    print(json.dumps(original), flush=True)
    blocks = model.blocks
    for i in range(len(blocks)):
        model.blocks = torch.nn.ModuleList(block for j, block in enumerate(blocks) if j != i)
        row = {"dropped": i, "loss": score(),
               "params": sum(p.numel() for p in model.parameters()), "reply": reply()}
        results.append(row)
        print(json.dumps(row), flush=True)
    model.blocks = blocks
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"ckpt": a.ckpt, "eval_rows": len(rows), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
