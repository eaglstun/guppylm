"""Try a sparse patch of the worst 2-bit quantization errors."""
import os
import torch
from quantlib import SD, TARGETS, base_sd, calib_batches, hessians, record, vq_bytes, FIXED_BYTES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOURCE = os.environ.get("AQ_2BIT_CHECKPOINT", os.path.join(ROOT, "runs/quant/E4_vq_d4_1x256_+rot_+gptq_long.pt"))
base = torch.load(SOURCE, map_location="cpu", weights_only=False)["model"]
delta = torch.cat([(SD[k].float() - base[k].float()).reshape(-1) for k in TARGETS])
offsets = [0]
for k in TARGETS:
    offsets.append(offsets[-1] + SD[k].numel())
BASE_BYTES = vq_bytes(256, 4) + FIXED_BYTES

def run(name, index):
    # Store a 24-bit flat index and fp16 signed correction per patched weight.
    patch = torch.zeros_like(delta)
    patch[index] = delta[index].half().float()
    sd = dict(base)
    for i, k in enumerate(TARGETS):
        sd[k] = base[k].float() + patch[offsets[i]:offsets[i+1]].reshape(SD[k].shape)
    return record(name, sd, BASE_BYTES + 5 * len(index))

if __name__ == "__main__":
    record("E13 tuned 2-bit baseline", base, BASE_BYTES)
    nmax = int(len(delta) * .02)
    rankings = {"magnitude": torch.topk(delta.abs(), nmax).indices}
    H = hessians(base_sd(), calib_batches())
    weighted = torch.cat([(SD[k].float() - base[k].float()).abs().mul(H[k].diag().clamp_min(0).sqrt()[None, :]).reshape(-1)
                          for k in TARGETS])
    rankings["activation"] = torch.topk(weighted, nmax).indices
    for criterion, indices in rankings.items():
        for pct in (.1, .25, .5, 1., 2.):
            n = int(len(delta) * pct / 100)
            run(f"E13 2-bit + sparse {criterion} top {pct}%", indices[:n])
