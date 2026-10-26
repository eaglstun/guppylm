# Quantization: how small can the aquarium fish get?

Branch `exp/aquarium`. Started 2026-10-26. Goal: the smallest, fastest fish that still talks
like the shipped one. The shipped web model is `aq-chat-v4` exported as int8 with one scale
per row (`web/model/guppy.bin`, 9.04 MB).

Code: `tools/aquarium/quant/` (`quantlib.py` + one script per stage). Raw numbers:
`runs/quant/results.jsonl` (gitignored). Everything is fake-quant: the float32 numbers the model
runs on are exactly the ones a packed file would decode to.

## Setup

- **Model:** `runs/aq-chat-v4/best.pt`, 8,873,472 params. Of those, the quantized "targets"
  are `tok_emb` (tied to `lm_head`) and every block's `qkv`, `out`, `up`, `down`: 8,650,752
  weights. `pos_emb` is int8 per row in every recipe (like the web export); biases and
  LayerNorms are float16.
- **Score:** mean loss on the 300 held-out-persona dialogues (`dialogues_eval.jsonl`,
  narwhal and mantis shrimp, never trained on), assistant tokens only. Lower is better.
  The fp32 fish scores **2.0468**.
- **MB** is the exact size of everything a packed file needs: codes, every group's scale and
  minimum (fp16), catalogs, row scales, pos_emb, biases, norms. No header, no tokenizer.
- **bpw** (bits per weight) counts only the targets, overhead included.
- **Calibration** (GPTQ): 256 training dialogues (not the held-out set), ~70k real tokens.
- **Sample:** Pebble the hermit crab, "What is a toaster?", seed 1, temperature 0.7.
- One model, one eval set, one seed. Differences under ~0.01 are noise-level.

## Glossary for this file

- **rtn g32**: round to nearest; every group of 32 weights along a row gets its own ruler
  (a minimum and a step, fp16 each, so +1 bit per weight at g32, +0.25 at g128).
- **gptq**: round one column at a time and push each rounding error onto the columns not yet
  rounded, using how the real activations correlate. Same file format as rtn, smarter choices.
- **rot:hadamard**: random signs plus a 128-wide Hadamard on both sides of each grid before
  rounding, undone after. Free to store (a seed), a fast transform at run time.
  **rot:dense**: a full random orthogonal matrix instead (reference only, slow to deploy).
- **catalog / VQ**: groups of d weights are replaced by the nearest of K stored shapes.

## Stage 0 to 2: references, rotation, GPTQ (scalar rounding)

| recipe                        |    MB |  bpw |       loss | verdict                                          |
| ----------------------------- | ----: | ---: | ---------: | ------------------------------------------------ |
| fp32 original                 | 35.49 | 32.6 |     2.0468 | reference                                        |
| **shipped web: int8 per row** |  8.98 | 8.07 | **2.0470** | lossless; ~4 MB of slack                         |
| rtn 4b g32                    |  5.66 | 5.00 |     2.0821 |                                                  |
| rtn 4b g128                   |  4.85 | 4.25 |     2.0759 | g128 beat g32 here (noise or luck)               |
| rtn 3b g32                    |  4.58 | 4.00 |     2.1690 |                                                  |
| rtn 3b g128                   |  3.77 | 3.25 |     2.2545 |                                                  |
| rtn 2b g32                    |  3.50 | 3.00 |     3.4333 | broken                                           |
| rtn 2b g8                     |  6.74 | 6.00 |     2.4287 | tiny groups help 2-bit but cost 4 bits of rulers |
| **gptq 4b g32**               |  5.66 | 5.00 | **2.0587** |                                                  |
| **gptq 4b g128**              |  4.85 | 4.25 | **2.0635** | near-lossless at 54% of the web size             |
| **gptq 3b g32**               |  4.58 | 4.00 | **2.0943** | best quality per MB so far                       |
| gptq 3b g128                  |  3.77 | 3.25 |     2.1860 |                                                  |
| gptq 3b g128 +rot             |  3.77 | 3.25 |     2.1406 | rotation helps here                              |
| gptq 2b g32                   |  3.50 | 3.00 |     2.8364 |                                                  |
| gptq 2b g32 +rot              |  3.50 | 3.00 |     2.6207 | rotation helps at 2-bit                          |
| gptq 2b g64 +rot              |  2.95 | 2.50 |     3.1048 |                                                  |

**Works:**

- **GPTQ, everywhere.** Free at run time (same file format) and the biggest single win:
  3-bit goes from 2.169 to 2.094, 2-bit from 3.43 to 2.84.
- int8 per row is already lossless, so everything above 8 bits per weight was wasted.

**Doesn't (here):**

- **Random rotation at 3 to 4 bits.** It made plain rounding slightly _worse_ at every 3 and 4
  bit setting (e.g. rtn 3b g32: 2.169 → 2.210), and was a wash with GPTQ at 4 bits. It only
  paid off at 2 bits and at 3b g128 with GPTQ. Best guess: rotation exists to tame outlier
  weights, and this 9M fish doesn't have big ones, so rotating just reshapes friendly
  distributions into bell curves that min/max rulers handle no better.
- **Dense rotation** was worse than Hadamard in every row (not shown; see results.jsonl).

## Stage 3: shape catalogs (vector quantization)

`stage3_catalog.py`. Each grid is (optionally) rotated, each row divided by its own spread
(one fp16 per row), and cut into groups of d. One catalog set is shared by the whole model.
"2x256" means two stacked catalogs of 256 shapes: the second learns what the first missed
(AQLM's trick), so 8 + 8 bits buy 65,536 combinations while storing only 512 shapes.
Bits per weight = stages × log2(K) / d, plus the catalogs and row scales.

Best of each size (all four variants of every config are in results.jsonl):

| recipe                   |   MB |  bpw |       loss | vs. best scalar at similar size       |
| ------------------------ | ---: | ---: | ---------: | ------------------------------------- |
| d8 1x256 +rot +gptq      | 1.38 | 1.04 |      5.858 | broken ("window window window")       |
| d8 1x4096 +rot +gptq     | 1.98 | 1.60 |      3.329 | broken but English-shaped             |
| **d4 1x256 +rot +gptq**  | 2.46 | 2.04 |  **2.448** | gptq 2b g32 +rot: 2.621 at 3.50 MB    |
| d8 2x256 +rot +gptq      | 2.46 | 2.04 |      2.492 |                                       |
| d8 3x256 +rot +gptq      | 3.55 | 3.05 |      2.137 |                                       |
| d8 2x4096 +rot +gptq     | 3.67 | 3.16 |      2.110 | gptq 3b g128 +rot: 2.141 at 3.77 MB   |
| **d4 1x4096 +rot +gptq** | 3.57 | 3.07 | **2.0924** | gptq 3b g32: 2.0943 at **4.58** MB    |
| d4 2x256 +gptq           | 4.62 | 4.04 |      2.068 | gptq 4b g128: 2.0635 at 4.85 MB (tie) |

**Works:**

- **Catalogs beat rulers at 3 bits and below.** d4 1x4096 matches the best 3-bit scalar fish
  a full megabyte smaller; at 2 bits the catalog fish talks sense where scalar 2-bit rambles.
- **Rotation + catalogs.** Rotation hurt rulers but helps catalogs (with GPTQ) in 7 of 8
  configs, everything from 1 to 3 bits; the exception is 4 bits (d4 2x256: 2.068 without,
  2.072 with, a wash). A catalog works best when every group looks like every other group,
  and rotation is exactly the tool that makes groups alike.
- **GPTQ, again**, on every catalog config (e.g. d4 1x256: 2.963 → 2.485).
- Short shapes (d4) beat long ones (d8) at the same bits: 4096 shapes of 4 numbers beat
  2x4096 of 8, and 256 shapes of 4 beat 2x256 of 8. With a model this small, catalog
  storage and greedy stacked encoding both hurt long shapes.

**Doesn't:**

- **Anything near 1 bit per weight.** Every 1.0 to 1.6 bpw fish is broken, catalog or not.
- **Catalogs at 4 bits:** a tie with plain GPTQ rulers, which are simpler to decode.
- Stacking three catalogs (d8 3x256, 2.137) lost to one bigger one (d4 1x4096, 2.092).

## Speed: the browser runtime (`web/guppy.js`)

Measured in Node 24 on the M4 Max with the real `guppy.js` and the real 9 MB `guppy.bin`:
a 102-token prompt (system prompt + a question), then 80 generated tokens. Best of 3 runs.
Bench code is in the session scratchpad, not the repo (it patches copies of `guppy.js`).

| runtime change                                     | prompt (102 tok) |      generate |
| -------------------------------------------------- | ---------------: | ------------: |
| shipped (int8 weights, one accumulator)            |           540 ms |     144 tok/s |
| weights unpacked to float32 at load                |           539 ms |     145 tok/s |
| **inner loop unrolled 4 ways (4 accumulators)**    |           405 ms | **190 tok/s** |
| **+ prompt in blocks of 2 tokens per weight read** |       **303 ms** |     190 tok/s |
| + blocks of 4 or 8 (kernel still pairs tokens)     |           300 ms |     190 tok/s |

**Works:**

- **Unroll `matvec` into 4 accumulators: +31% generation speed**, a 6-line change.
- **Blocked prompt processing: prompt 1.8x faster overall** (540 → 303 ms). Logits are
  bit-identical to the per-token path (max difference exactly 0).

**Key finding: file size and speed are separate problems.** Float32 weights run exactly as
fast as int8 in this JS loop, so any download format (3-bit, catalogs, rotations) can be
unpacked once at load at no speed cost, rotations included (they're undone at load, not
per token). It can even be re-packed into today's int8-per-row layout so memory stays
9 MB and `matvec` doesn't change. Quantization buys download size; speed comes from the loop.

## Stage 4: fine-tune the leftovers (codes frozen)

`stage4_tune.py`. After GPTQ, every code (which shape / which ruler mark) is frozen; what
trains is the shared catalog(s), each row's scale, and every bias and LayerNorm. Loss is
distillation: KL(original fish || quantized fish) on every real token of training dialogues.
CPU, batch 8, lr 2e-3 cosine with 50 warmup steps. The file format doesn't change.

| recipe                       |   MB | before | after 600 steps | gap to fp32 closed |
| ---------------------------- | ---: | -----: | --------------: | -----------------: |
| catalog d4 1x4096 +rot +gptq | 3.57 | 2.0924 |      **2.0774** |                33% |
| catalog d4 1x256 +rot +gptq  | 2.46 | 2.4478 |      **2.2304** |                54% |

**Works:** distillation fine-tuning, and it helps most where the damage is worst. The 2-bit
fish's KL was still falling at step 600, so longer runs are next.

## Stage 5: mixed precision (which grids can take 2 bits?)

`stage5_mixed.py`. Start from all-3-bit (d4 1x4096, 2.0924), drop one family or one layer to
the 2-bit catalog (d4 1x256), measure. No fine-tuning in this stage.

| dropped to 2-bit            |  MB saved |           loss |           damage |
| --------------------------- | --------: | -------------: | ---------------: |
| ffn.down                    |      0.22 |         2.1146 |           +0.022 |
| attn.out                    |      0.11 |         2.1141 |           +0.022 |
| attn.qkv                    |      0.33 |         2.1463 |           +0.054 |
| ffn.up                      |      0.22 |         2.1566 |           +0.064 |
| **emb (tok_emb = lm_head)** |      0.20 |         2.1773 |       **+0.085** |
| layer 0                     |      0.15 |         2.1340 |           +0.042 |
| layer 1                     |      0.15 |         2.1029 |           +0.011 |
| layers 2 to 5               | 0.15 each | 2.113 to 2.119 | +0.021 to +0.027 |

Greedy (cheapest damage per byte first), untuned:

| 2-bit families                |   MB |   loss |
| ----------------------------- | ---: | -----: |
| ffn.down                      | 3.35 | 2.1146 |
| + attn.qkv                    | 3.02 | 2.1739 |
| + attn.out                    | 2.91 | 2.1988 |
| + ffn.up (everything but emb) | 2.69 | 2.2805 |
| + emb (all 2-bit)             | 2.46 | 2.4478 |

**Findings:**

- **The embedding table is the fragile one.** Dropping it last costs 0.17 loss for 0.23 MB,
  the worst trade in the table. It is read twice per token (input lookup and the output
  head that picks the next word), so its errors land at both ends of the model.
- **Layer 0 is the most sensitive layer, layer 1 the least.** The middle and late layers
  are about equal.
- The two "output" grids (`attn.out`, `ffn.down`) take 2 bits most gracefully.

## Stage 4b: longer fine-tunes and tuned mixes

Same as stage 4, 2,000 steps. "Mix" = the named families on the 4096-shape catalog (3 bits),
everything else on the 256-shape catalog (2 bits), both rotated + GPTQ, both catalogs tuned.

| recipe                            |       MB | before |      tuned | steps |
| --------------------------------- | -------: | -----: | ---------: | ----: |
| all 2-bit (d4 1x256)              |     2.46 | 2.4478 |     2.2304 |   600 |
| all 2-bit (d4 1x256)              |     2.46 | 2.4478 |     2.2195 |  2000 |
| **mix: emb at 3-bit, rest 2-bit** | **2.69** | 2.2805 | **2.1523** |  2000 |

**Works:** keeping the embedding at 3 bits and tuning. The 2.69 MB fish gains 0.13 from
tuning and ends 0.07 better than the all-2-bit fish for 0.23 MB more.

**Diminishing:** more steps on the all-2-bit fish (600 → 2000) bought only 0.011. Its
catalog of 256 shapes is the ceiling, not the training.

## Deploy check: re-pack into today's int8 layout at load

The tuned 3.57 MB fish (2.0774), unpacked and re-rounded into the shipped format (int8, one
scale per row) as the browser would do at load: **2.0784**. +0.001, noise. So a small
download can be expanded into the existing `guppy.bin` layout in memory and run by the
existing `matvec` (largest weight change from re-packing: 0.0026).

## Follow-up: smaller catalogs and failed residual fixes (October 26)

These are measurements on the same v4 checkpoint and held-out set. The 1.5-bit and catalog
sweeps are in `stage6_ultralow.py` and `stage8_catalog_sweep.py`; low-rank and sparse residual
trials are in `stage7_lowrank.py` and `stage11_sparse_residual.py`.

| v4 recipe                        | Estimated MB | Held-out loss |
| -------------------------------- | -----------: | ------------: |
| 1.5-bit d4 1x64, rotated GPTQ    |         1.91 |        3.4596 |
| 1.75-bit d4 1x128, rotated GPTQ  |         2.19 |        2.7439 |
| 2.25-bit d4 1x512, rotated GPTQ  |         2.73 |        2.2668 |
| same, tuned 600 steps            |         2.73 |        2.1713 |
| 2.5-bit d4 1x1024, rotated GPTQ  |         3.00 |        2.1628 |
| **same, tuned 600 steps**        |     **3.00** |    **2.1063** |
| 2.75-bit d4 1x2048, rotated GPTQ |         3.28 |        2.1422 |
| 2x16 catalogs of d4 (2 bits)     |         2.45 |        2.7693 |
| 2x32 catalogs of d4 (2.5 bits)   |         3.00 |        2.2468 |

The 1.5-bit fish is too damaged to use, and stacked tiny catalogs lose to one larger catalog.
At 3.00 MB, the tuned 1024-shape fish is a useful middle point, though the earlier tuned
2.69 MB mixed fish (loss 2.1523) is more efficient if that quality is sufficient. Adding
rank-2 to rank-32 float16 low-rank residuals to the tuned 2-bit fish made loss worse
(2.2411–2.2724 versus 2.2195). Patching the 0.1–2% largest individual errors was a wash or
worse (best 2.2172 at +0.04 MB). The loss depends on coordinated weight changes, not just the
largest isolated reconstruction errors.

## V5 comparison and behavior checks

V5 is a completed 15,000-step checkpoint at `runs/aq-chat-v5/best.pt`, with the same
8,873,472-parameter architecture. It adds song and family training rows. These compression
experiments use v5. Use
`AQ_QUANT_CKPT=runs/aq-chat-v5/best.pt` and a separate `AQ_QUANT_OUT` directory to reproduce
the v5 quant sweeps (`stage9_v5_sweep.py`, `stage10_local_catalogs.py`).

| v5 recipe                 | Estimated MB | Held-out loss |
| ------------------------- | -----------: | ------------: |
| int8 per row reference    |         8.98 |        2.0675 |
| d4 1x256, rotated GPTQ    |         2.46 |        2.4207 |
| d4 1x512, rotated GPTQ    |         2.73 |        2.2608 |
| d4 1x1024, rotated GPTQ   |         3.00 |        2.2062 |
| **same, tuned 600 steps** |     **3.00** |    **2.1436** |
| d4 1x4096, rotated GPTQ   |         3.57 |        2.1167 |

Giving each matrix its own 256-shape catalog costs 2.50 MB and improves loss to 2.3837,
versus 2.4207 with one global catalog. The gain is too small to change the size/quality
frontier. More importantly, the tuned 3.00 MB v5 fish loses task skills even where loss
looks tolerable: the standard probe scores name 0.833, persona 0.708, noun copying 0.490,
memory 0.812, versus v5's 1.000, 0.625, 0.558, 0.875. Keep probes beside loss in every
future compression choice.

### V5.1: more noun copies

The first v5.1 attempt stopped around step 2,600. A clean rerun completed all 15,000 steps
from `runs/aq-td/best.pt`, with the same settings and data as v5 except four noun-swap copies
per eligible dialogue instead of two. Its best checkpoint is `runs/aq-chat-v51/best.pt`.
Final chat validation loss is 1.7549 versus v5's 1.7542; held-out persona loss is 2.0691
versus 2.0669. The standard probe shows noun copying 0.635 versus 0.558 (real nouns 0.641
versus 0.484; invented nouns 0.625 versus 0.675), name copying 0.979 versus 1.000,
held-out persona 0.750 versus 0.625, and memory 0.875 for both. A separate 40-question
probe with unseen nouns and phrasings scores 9/40 versus v5's 6/40. These small sampled
probes suggest a noun-copy gain but do not establish a broad quality improvement. The v5.1
weights are local and have not been exported to the browser.

## Browser speed follow-up

`bench_runtime.mjs` benchmarks the real `web/guppy.js` with the shipped v4 binary on Node 24,
M4 Max: 102 prompt tokens then 80 generated tokens. Best of 5 measured runs after warmup.
An 8-accumulator matrix loop plus causal two-token prompt processing gives:

| Runtime                        | 102-token prompt | 80 generated tokens | Generation rate |
| ------------------------------ | ---------------: | ------------------: | --------------: |
| shipped loop                   |         514.2 ms |            516.9 ms |     154.8 tok/s |
| 8 accumulators + paired prompt |         268.7 ms |            369.4 ms |     216.6 tok/s |

The benchmark checks prompt and final logits, attention K/V caches, odd and even prompt
lengths, and prefix-cache reuse; maximum absolute difference was 0. This speeds the
existing int8 runtime. Smaller packed downloads still need an encoder and browser decoder.

## V5 data continuation and the compression trade (October 26)

`train_noun_continue.py` tests a bounded 400-step continuation of v5: one quarter new
synthetic noun-copy dialogues, three quarters existing chat rows, learning rate 3e-5.
`probe_noun_generalization.py` checks ten unseen nouns in four unseen phrasings (40 cases)
alongside the standard `probe.py`. Both continuation variants start from the original v5;
neither uses v5.1. The first variant's two of five training questions match the standard
probe exactly, so its standard noun score overstates transfer. The diverse variant uses
ten training question forms that match neither probe's forms.

| model                                           |  Held loss |      Name |   Persona |      Noun |    Memory | New phrasing |
| ----------------------------------------------- | ---------: | --------: | --------: | --------: | --------: | -----------: |
| V5 full-size                                    |     2.0669 |     1.000 |     0.625 |     0.558 |     0.875 |         6/40 |
| V5 + noun continuation, full-size               | **2.0566** |     0.979 |     0.625 |     0.875 |     1.000 |        15/40 |
| V5 + diverse noun continuation, full-size       |     2.0685 |     0.896 |     0.417 |     0.913 |     1.000 |        16/40 |
| V5, tuned 3.00 MB K1024                         |     2.1436 |     0.833 |     0.708 |     0.490 |     0.812 |            — |
| V5 + noun continuation, tuned 3.00 MB K1024     |     2.1356 |     0.979 |     0.500 |     0.702 |     0.812 |         7/40 |
| V5, tuned 3.57 MB K4096                         |     2.0890 |     0.938 |     0.625 |     0.558 |     0.812 |         7/40 |
| **V5 + noun continuation, tuned 3.57 MB K4096** | **2.0869** | **1.000** | **0.667** | **0.894** | **0.812** |    **12/40** |

The 3.57 MB recipe preserves much more of the noun-copy gain than the 3.00 MB one. It is
the best balanced compressed v5 candidate from this round. The 40-case new-phrasing probe
is small, and generation is sampled; treat a few hits of difference as a lead to repeat,
not a definitive improvement. MB entries are packed-payload estimates with byte accounting
for codes, scales, catalogs, and other tensors; they exclude a file header and tokenizer.
Saved `.pt` checkpoints are float tensors, about 34 MB; a packed file
and browser decoder have not yet been built.

## Layer dropping: size and compute together

`prune_layer.py` tests removal of each of v5's six blocks. Removing block 3 was least
damaging (zero-shot loss 2.3442 versus 2.0669). A 600-step recovery in `train_pruned.py`
improved held loss to 2.1417. Parameters fall 8,873,472 → 7,689,600 (13.3%), and target
matrix weights fall 8,650,752 → 7,471,104 (13.6%). But its 4-sample standard probe drops
to name 0.812, persona 0.542, noun 0.462, memory 0.312. Held loss hides too much behavior
damage, so this 5-layer checkpoint is not a viable replacement yet.

## Eight- and sixteen-value chunk experiments

These tests use the noun-continued v5 checkpoint, whose uncompressed held-out loss is 2.0566.
They explicitly round catalog/scale/bias side data to the stored fp16 representation before
scoring. Scripts: `stage12_entropy.py`, `stage13_projective_chunks.py`, and
`stage14_dct_tiles.py`.

**Entropy of the existing 4-value catalog codes.** Re-encoding d4/K4096 with GPTQ gives
2,162,688 twelve-bit indices. Their global entropy is 11.922 bits/index, nearly uniform.
The best tested code layout plus compression shrinks the full 3.568 MB estimated payload
by only 26.7 KB (0.75%) before framing/decoder overhead. Generic entropy coding is not
a useful route below 3 MB here; the measurement used initial GPTQ codes, since the tuned
codes were not saved.

**Projective shape + signed amplitude.** Every 8- or 16-value chunk stores a shape-catalog
index and a signed int8 amplitude; each matrix row has one fp16 amplitude scale. The shape
catalog itself is fp16. This format could be read directly by a compressed-weight matvec,
but the scored models are too damaged:

| recipe                            | Estimated MB | Held-out loss |
| --------------------------------- | -----------: | ------------: |
| d8, 1024 shapes, round-to-nearest |         2.74 |        3.9730 |
| d8, 1024 shapes, GPTQ             |         2.74 |        3.0117 |
| d8, 4096 shapes, GPTQ             |         3.06 |        2.4921 |
| d16, 4096 shapes, GPTQ            |         1.77 |        6.2521 |

**JPEG-style DCT tiles.** Each weight matrix was treated as 8×8 or 16×16 tiles. The lowest
frequency quarter of DCT coefficients holds only 31.3% of energy for 8×8 tiles and 30.2%
for 16×16 tiles. Weight grids lack the spatial smoothness that makes JPEG work on photos.
Coefficient positions and scales were included in the size estimates:

| recipe                                     | Estimated MB | Held-out loss |
| ------------------------------------------ | -----------: | ------------: |
| 8×8, keep 16 low-frequency coefficients    |         2.68 |        8.0639 |
| 8×8, keep 16 largest coefficients          |         4.31 |        6.2881 |
| 8×8, quantize all coefficients to 3 bits   |         3.77 |        2.5150 |
| 16×16, keep 64 low-frequency coefficients  |         2.48 |        7.8101 |
| 16×16, quantize all coefficients to 3 bits |         3.56 |        3.6296 |

The 3.57 MB d4 catalog model at 2.0869 loss remains much better than these chunk and DCT
recipes.

**Two-catalog beam search.** Inspired by [AQLM](https://arxiv.org/abs/2401.06118),
`stage16_beam_vq.py` stores two uint8 catalog indices per
8-value chunk: 2 bits per weight, plus two small fp16 catalogs and row scales. Searching
eight candidate first codes before choosing the second reduced embedding RMS error by about
4%, but held loss barely moved from 2.5088 (greedy) to 2.5058 (beam). Both payloads decode
exactly from their saved codes/catalogs and have 2,494,129-byte PyTorch containers. This
is a local beam search, not the full alternating AQLM training procedure.

**Structured 8-value lattice.** Inspired by [QuIP#](https://arxiv.org/abs/2402.04396),
`stage15_e8p.py` uses an independently generated 256-shape
shell, even sign flips, and a quarter-value shift to represent each eight weights with a
16-bit code. This is inspired by QuIP#'s E8P lattice but uses an independent shell-padding
selection. Block Hadamard rotations and GPTQ are essential: loss went 3.0382 (plain) →
2.7270 (GPTQ) → 2.3187 (rotation + GPTQ). Frozen-code distillation of fp16 row scales and
small weights for 600 steps reached **2.2113 loss at 2,455,040 actual tensor bytes**. The
self-contained PyTorch container is 2,487,790 bytes, and decoding it reproduces the score
exactly. Standard probe: name 0.896, persona 0.625, noun 0.740, memory 0.750; unseen noun
phrasing 9/40. The 3.57 MB candidate is stronger (loss 2.0869; 12/40 unseen noun phrasing),
but this is a real smaller model payload rather than a projected byte count.

Using one fp16 amplitude scale per 128 input columns, aligned with the Hadamard blocks,
adds 94,720 bytes and improves the untuned lattice loss to 2.3005. A bounded 300-step
frozen-code tune reaches **2.1899 loss at 2,549,760 actual tensor bytes**; decoding the
2,582,657-byte PyTorch container reproduces this loss exactly. Standard probe: name 0.938,
persona 0.625, noun 0.712, memory 0.812; unseen noun phrasing **12/40**, matching the
3.57 MB candidate's 12/40 for about 1.02 MB less. Per-64 scales cost 2.69 MB and scored
slightly worse before tuning (2.3050), so more scales are not automatically better.

For resident RAM, a synthetic Node matrix-only benchmark (`bench_chunk_ram.mjs`) read
16-bit chunk codes from about 2.77 MB of codes, scales, and a precomputed 0.5 MB decode
table, versus 8.73 MB for int8 weights/scales. With block Hadamard transforms, the direct
code path took roughly 1.3–1.4× the int8 matrix-sweep time across two runs. This is not a
full browser-generation benchmark.

The standalone direct-code `web/guppy_e8p.js` runtime now reads a compact binary without
expanding target matrices to int8. `export_e8p_web.py` produced 2,495,360-byte (per-row)
and 2,590,080-byte (g128) binaries. A precomputed 512 KiB lookup table turns each 16-bit
chunk code into eight small lattice coordinates; fast block-128 Hadamards rotate each
matrix's input and output. Across 32 sequential tokens, direct JS logits differ from the
decoded Python model by at most 1.34e-5 and K/V caches by under 3.6e-6; paired prompt
processing matches token-by-token stepping exactly.

On Node 24/M4 Max, 102 prompt tokens plus 80 fixed continuation steps, after warmup:

| Same-model weight path          |     Prompt | 80 generation steps | Counted resident storage after 182 IDs |
| ------------------------------- | ---------: | ------------------: | -------------------------------------: |
| int8 re-export of decoded model | 465–469 ms |          662–676 ms |                               18.53 MB |
| direct 2.46 MB lattice          |     539 ms |              713 ms |                               13.23 MB |
| direct 2.55 MB g128 lattice     |     604 ms |              807 ms |                               13.52 MB |

Counted storage includes weight buffer, lookup, decoded side tensors, scratch, cached token
embeddings, and the same 9.44 MB K/V cache on both paths. It excludes JavaScript object and
allocator overhead. The g128 codec gives better held-out loss at a roughly 19% generation
slowdown relative to its int8 re-export; the per-row codec is about 8% slower in its paired
benchmark. These results are from Node on a Mac; phone WebView timing remains unmeasured.

## Follow-up: JPEG on 64-weight slices and weight-as-color images

`stage18_jpeg64_sweep.py` revisits 64-value blocks in two layouts: 64 consecutive weights
from one matrix row reshaped to 8×8 (`row64`), and spatial 8-row × 8-column tiles
(`tile8`). It sweeps JPEG-like DCT quantization strength, uniform versus
frequency-weighted steps, and a no-DCT control. Sizes include a real zlib-compressed
zigzag/RLE/varint coefficient stream, fp16 tile scales, the quantization table, matrix
lengths, and fixed non-target tensors. The same noun-continued v5 checkpoint is scored.

| 64-value layout and setting                    | Actual encoded bytes | Held-out loss |
| ---------------------------------------------- | -------------------: | ------------: |
| Row slice, DCT, fine uniform steps             |            6,829,624 |        2.0702 |
| Row slice, DCT, fine JPEG frequency steps      |            5,621,501 |        2.1086 |
| Row slice, DCT, medium uniform steps           |            5,729,366 |        2.1066 |
| Row slice, DCT, medium JPEG frequency steps    |            4,472,400 |        2.2987 |
| Row slice, no DCT, medium steps                |            5,691,212 |        2.0914 |
| Spatial tile, DCT, fine uniform steps          |            6,858,695 |        2.0673 |
| Spatial tile, DCT, medium JPEG frequency steps |            4,491,038 |        2.3028 |

The row-slice and spatial-tile layouts behave similarly. At matched medium uniform
quantization, the no-DCT row control (5.69 MB / 2.0914) slightly beats the DCT version
(5.73 MB / 2.1066). A fixed four-bit all-coefficient DCT control scored 2.1053 at
4,787,111 bytes. The existing 3.57 MB catalog (2.0869) dominates the JPEG settings.

`stage19_color_weights.py` then stores quantized weights as actual Pillow images:
one weight per grayscale pixel, three neighboring weights as RGB, three matrix rows as
RGB, an orthogonal DC/AC mixing of RGB triples, and a two-byte RGB mapping. It tests
JPEG (4:4:4 for RGB), WebP, and PNG at several quality levels. Actual image-container
bytes, fp16 row scales, and fixed tensors are counted; the scored model uses pixels
decoded from each file. Pillow 12.3.0 was used (`uv pip install pillow` is sufficient).

| Color encoding                           | Actual total MB | Held-out loss |
| ---------------------------------------- | --------------: | ------------: |
| Grayscale WebP q50                       |           3.294 |        2.2174 |
| Grayscale WebP q60                       |           3.498 |        2.1605 |
| Grayscale WebP q65                       |           3.626 |        2.1457 |
| Grayscale WebP q80                       |           4.255 |        2.0786 |
| Grayscale JPEG q85                       |            4.74 |         2.166 |
| Adjacent weights as RGB, JPEG q85        |           3.551 |        2.7817 |
| Orthogonally mixed RGB triples, JPEG q85 |           3.902 |        2.4916 |

Grayscale WebP is the best image-codec route, but its 3.50 MB / 2.1605 point is behind
the 3.57 MB catalog's 2.0869, and its 3.29 MB / 2.2174 point is behind the 2.55 MB
lattice's 2.1899. RGB triples have almost no natural channel correlation, so image
color transforms add distortion. Image decoding would also expand the weights in RAM;
the direct-code lattice runtime retains its compact representation during inference.
