"""Run the same quantization family on the completed v5 checkpoint."""
from quantlib import FIXED_BYTES, SD, TARGETS, base_sd, q8_rows, record, vq_bytes
from stage3_catalog import build

web = base_sd()
for k in TARGETS:
    web[k] = q8_rows(SD[k])
web_bytes = sum(SD[k].numel() + 4 * SD[k].shape[0] for k in TARGETS) + FIXED_BYTES
record("E11 v5 reference int8 per row", web, web_bytes)

for K in (128, 256, 512, 1024, 4096):
    record(f"E11 v5 vq d4 1x{K} +rot +gptq",
           build(4, K, 1, "hadamard", True), vq_bytes(K, 4) + FIXED_BYTES)
