# Aquarium v4 matrix-by-matrix 1-bit sweep

This sweep uses the checkpoint served by the GitHub Pages Aquarium demo:
`runs/aq-chat-v4/best.pt` in `guppylm-exp-aquarium`. It starts from the same
per-row int8 rounding used by `web/model/guppy.bin`, then replaces each unique
2D matrix with binary signs multiplied by one FP16 mean-absolute scale per row.
The token embedding and output head share one matrix and are counted once.

The score is mean assistant-token loss over the 300 held-out Aquarium persona
dialogues (28,800 assistant tokens), using `train_aq.py`'s evaluation format.
Lower is better. FP32 scores **2.0468** and int8 scores **2.0470**.

| 1-bit substitution into int8 model | Held-out loss | Increase | Payload bytes saved |
| --- | ---: | ---: | ---: |
| Best single matrix: block 2 attention output | 2.0719 | +0.0249 | 129,792 |
| Block 3 FFN down | 2.0745 | +0.0275 | 258,816 |
| Block 1 FFN down | 2.0870 | +0.0400 | 258,816 |
| Position embedding | 2.6554 | +0.6084 | 173,056 |
| Token embedding and tied output head | 3.6565 | +1.6095 | 1,384,448 |
| All block output projections | 3.7898 | +1.7428 | 2,331,648 |
| All block linear weights, embeddings int8 | 5.5935 | +3.5465 | 6,225,408 |
| All matrices | 6.3169 | +4.2698 | 7,782,912 |

Every single 1-bit matrix raised loss. The sweep includes all 26 unique 2D
matrices; the JSON results give every score and size. In a greedy mixed-precision
search with a +0.05 loss budget, only `blocks.3.ffn.down.weight` was accepted.
It would save about **259 KB** from the 9,042,320-byte web file, leaving an
estimated 8,783,504 bytes before a new file header and decoder. That is about
2.9% less download for +0.0275 loss. This search uses the same held-out set to
choose and score the mix, so it is exploratory and would need an independent
behavior check before deployment.

The result agrees with the aquarium branch's `QUANTIZATION.md` finding that 1.0–1.6
bits per weight damages v4, and extends its 2-bit family sensitivity test down
to 1-bit individual matrices. It does **not** support replacing the Pages model
with an all-1-bit checkpoint. A dedicated binary-aware fine-tune or architecture
could change that conclusion; this is post-training rounding only.

## Method and reproduction

Run the script with PyTorch, NumPy, and `tokenizers` installed:

```bash
python tools/experiment_binary_sweep.py \
  --aquarium-root /path/to/guppylm-exp-aquarium \
  --checkpoint /path/to/guppylm-exp-aquarium/runs/aq-chat-v4/best.pt \
  --output runs/binary_sweep_v4/results.json
```

For each matrix, 1-bit size is `ceil(weight_count / 8) + 2 * rows`; int8 size is
`weight_count + 4 * rows`. File projections subtract only those payload byte
differences from the actual shipped `guppy.bin` size. They exclude a new 1-bit
header, decoder, and runtime changes. The evaluated matrices are fake-quantized
float tensors; no packed browser artifact was built in this sweep.

The script reads Aquarium files but writes only to its own output path. The
original sweep measurements are preserved in `results/one_bit_v4.json`.
