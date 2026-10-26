"""Fill the catalog size gaps between the existing 2- and 3-bit recipes."""
from quantlib import FIXED_BYTES, record, vq_bytes
from stage3_catalog import build

for d, K, stages in [(4, 128, 1), (4, 512, 1), (4, 1024, 1),
                     (4, 2048, 1), (4, 16, 2), (4, 32, 2)]:
    name = f"E9 vq d{d} {stages}x{K} +rot +gptq"
    record(name, build(d, K, stages, "hadamard", True), vq_bytes(K, d, stages) + FIXED_BYTES)
