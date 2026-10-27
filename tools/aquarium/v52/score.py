"""Score float or decoded compression checkpoints on development/historical losses."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .. import train_aq as aq
from .data import file_hash, read_manifest
from .train import losses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    a = ap.parse_args()
    if a.out.exists():
        raise ValueError("Refusing to overwrite a score report")
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = aq.GuppyLM(aq.GuppyConfig(**ck["config"])).to(a.device)
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(str(a.data / "tokenizer.json"))
    report = {"checkpoint_sha256": file_hash(a.checkpoint), "tokenizer_sha256": file_hash(a.data / "tokenizer.json"),
              "checkpoint_step": ck["step"], "params": sum(p.numel() for p in model.parameters()), "scores": {}}
    for split in ("development", "historical"):
        rows, h, _ = read_manifest(a.manifests / f"{split}.json", split)
        pairs = [aq.encode_chat(tok, r, model.config.max_seq_len) for r in rows]
        report["scores"][split] = {**losses(model, pairs, [r["source"] for r in rows],
                                                model.config.max_seq_len, a.device), "manifest_sha256": h}
    seq = model.config.max_seq_len
    stories = aq.Stories(str(a.data / "ts_valid.bin"), seq)
    rng, pairs = np.random.default_rng(5203), []
    for _ in range(20):
        x, y = stories.batch(rng, 32)
        pairs.extend(zip(x.tolist(), y.tolist()))
    report["scores"]["story"] = {**losses(model, pairs, ["story"] * len(pairs), seq, a.device),
                                    "seed": 5203, "batches": 20, "batch": 32,
                                    "source_sha256": file_hash(a.data / "ts_valid.bin")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
