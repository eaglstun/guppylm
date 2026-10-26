# Cellular reconstruction of sparse 1-bit weights

This is a post-training pilot on the pretrained GuppyLM checkpoint. It changes
only `blocks.0.ffn.up.weight` (768 × 384 = 294,912 weights). Each row uses one
FP16 mean-absolute scale. The full binary baseline stores every sign; sparse
variants retain signs at fixed positions and reconstruct the missing positions
with four local updates. The learned rule is a shared 3×3 convolution followed
by a 1×1 convolution, with 321 FP32 parameters. It is trained for 200 steps on
patches from the target matrix, without evaluation examples.

The comparison evaluates 1,643 non-padding tokens from eight held-out batches.
Lower loss is better. It replaces this one matrix in an otherwise unchanged
FP32 model; the table does not imply the whole model is compressed.

| Retained signs | Variant | Matrix bytes | Eval loss | Missing sign accuracy |
| ---: | --- | ---: | ---: | ---: |
| 100% | Original FP32 | 1,179,648 | 0.3784 | — |
| 100% | Packed 1-bit + scales | 38,400 estimated | 0.3803 | — |
| 25% | Zero fill | 10,752 estimated | 1.0041 | — |
| 25% | Fixed CA | 10,771 actual | 1.6748 | 50.2% |
| 25% | Learned CA | 12,055 actual | 1.5680 | 50.5% |
| 50% | Zero fill | 19,968 estimated | 0.4496 | — |
| 50% | Fixed CA | 19,987 actual | 0.7530 | 50.3% |
| 50% | Learned CA | 21,271 actual | 0.8355 | 51.0% |
| 75% | Zero fill | 29,184 estimated | 0.3890 | — |
| 75% | Fixed CA | 29,203 actual | 0.4039 | 50.3% |
| 75% | Learned CA | 30,487 actual | 0.4031 | 51.6% |

The fixed and learned CA files include a 19-byte header, packed seed signs,
FP16 row scales, and FP32 rule parameters for the learned variant. The script
loads each file back and verifies that it reconstructs the matrix evaluated
above. Packed 1-bit and zero-fill sizes are calculated from the same component
sizes, without a serialized file or header.

Horizontal neighboring signs agree 49.7% of the time; vertical neighbors agree
50.9%. The learned rule predicts omitted signs at roughly chance accuracy, and
every CA variant raises language-model loss more than zero filling at the same
retention rate. For this matrix and ordering, local reconstruction does not
justify its rule storage or reconstruction cost. A follow-up could test a
permutation or grouping that places correlated weights next to each other,
then compare against the same zero-fill and packed 1-bit baselines.

## Reproduce

Run from this worktree with PyTorch, NumPy, and `tokenizers` installed. Point
the three input paths at the existing GuppyLM checkpoint and data. For example:

```bash
python tools/experiment_cellular_compression.py \
  --checkpoint /path/to/guppylm/checkpoints/best_model.pt \
  --eval-jsonl /path/to/guppylm/data/eval.jsonl \
  --tokenizer /path/to/guppylm/data/tokenizer.json \
  --seed-pattern grid \
  --output runs/cellular_compression/grid/results.json
```

Repeat with `--seed-pattern checkerboard` for 50% retention and
`--seed-pattern three_quarters` for 75%. The script writes `results.json`,
`fixed.ca1`, and `learned.ca1` under each output directory. `runs/` is ignored
by Git; the measured JSON summaries are preserved in `results/cellular_25.json`,
`results/cellular_50.json`, and `results/cellular_75.json`.
