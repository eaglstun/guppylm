# Experiment: why does the Mac GPU lose, and what is the fastest way to run a fish?

Branch `exp/speed`. Model under test: the trained baseline checkpoint
(`../guppylm/runs/baseline/checkpoints/best_model.pt`, 8,726,016 params, 6 layers,
d_model 384, max_seq_len 128). No training was done in this experiment.

## Question

The baseline `results.json` says MPS loses to CPU on GuppyLM: generation 257 tok/s on MPS
vs 347 tok/s on CPU, and ONNX CPU forward at 128 tokens (~2.0 ms) beats MPS (~2.5 ms).
Generation had no KV cache. Why does the GPU lose, where (if anywhere) does it win, and
what configuration runs the fish fastest?

## Hardware and software

- Apple M4 Max (`sysctl -n machdep.cpu.brand_string`), 12 performance + 4 efficiency
  cores, 64 GB, macOS 27.0 (26A428)
- Python 3.14.2, torch 2.13.0 (MPS backend, 12 intra-op threads by default),
  onnxruntime 1.26.0 (CPUExecutionProvider), onnx 1.23.0
- Three other agents were training/benchmarking on the same machine. All timing ran in
  one process holding the shared GPU lock (`gpu.sh`), 2026-10-22 11:31 to 11:36 local,
  235 s total. Non-timing work (verification, ONNX export, agreement) ran outside the lock.

## What changed in the code

- `guppylm/model.py`: `Attention`/`Block` take optional `past_kv` and `use_cache`;
  new `GuppyLM.forward_cached(idx, past_kvs)` returns logits plus per-layer (k, v).
  `generate(..., use_cache=False)` keeps the original path as the default; with
  `use_cache=True` it prefills once and then feeds one token per step, and falls back to a
  full recompute once the sequence outgrows `max_seq_len` (the sliding window shifts every
  position, so the cache is invalid there). `forward()` is untouched.
- `guppylm/inference.py`: `--kv-cache` flag / `use_cache` argument.
- `tools/speed_bench.py`: `verify`, `export`, `time` subcommands (below).

## Commands

```bash
LOCK=/private/tmp/claude-501/-Users-eeaglstun-Documents-web-ericeaglstun-ai/f124945c-1d81-418b-a0c8-d0a95e2e49aa/scratchpad/guppy/gpu.sh
python3 tools/speed_bench.py verify          # cache == no cache, CPU   -> runs/speed/verify_cpu.json
python3 tools/speed_bench.py export          # ONNX graphs + agreement  -> runs/speed/export.json
$LOCK python3 tools/speed_bench.py time      # all timing (+ MPS verify) -> runs/speed/time.json
python3 -m guppylm.inference --checkpoint ../guppylm/runs/baseline/checkpoints/best_model.pt \
    --tokenizer data/tokenizer.json -p "hi guppy" --kv-cache
```

`--checkpoint` defaults to the main checkout's baseline `best_model.pt`.
Method notes: latency points are medians of 50 synced calls after 5 warmups; generation
tok/s are medians of 5 runs (B=1) or 1 to 3 runs (batch sweep, fewer reps when a run is
slow) after one warmup. Generation is greedy and ignores EOS so every run does identical
work. Min and max are in `runs/speed/time.json`.

## Results (measured)

### 1. Baseline confirmation, batch 1

| metric                                 | baseline results.json | this run, median (min to max)                                                 |
| -------------------------------------- | --------------------- | ----------------------------------------------------------------------------- |
| torch CPU forward @128                 | 5.96 ms               | 5.71 ms (5.36 to 6.26)                                                        |
| torch MPS forward @128                 | 2.48 ms               | 5.09 ms (3.73 to 5.58) in this phase; 2.16 ms and 2.19 ms in two later phases |
| torch MPS forward @16                  | 2.35 ms               | 3.17 ms (2.75 to 4.84)                                                        |
| baseline `model.onnx` (uint8) CPU @128 | 2.06 ms               | 2.03 ms                                                                       |
| torch CPU generation, 64 tok, no cache | 347 tok/s             | 369 tok/s (356 to 376)                                                        |
| torch MPS generation, 64 tok, no cache | 257 tok/s             | 153 tok/s (140 to 157); 226 tok/s in the batch sweep                          |

CPU and ONNX numbers reproduce within ~6%. MPS at batch 1 is noisy across phases
(153 to 234 tok/s, forward 2.2 to 5 ms) but always loses to CPU for generation. The
direction of the surprise is confirmed.

### 2. Where MPS time goes (batch 1)

| measurement                                | value                  |
| ------------------------------------------ | ---------------------- |
| aten ops per forward (any T)               | 167                    |
| tiny op (`add` on 384 floats), CPU, synced | 1.2 us                 |
| tiny op, MPS, synced after each op         | 106 us                 |
| tiny op, MPS, 1000 queued then one sync    | 5.3 us per op          |
| MPS forward @128, synced every call        | 2.16 ms                |
| MPS forward @128, 50 queued then one sync  | 2.24 ms per forward    |
| MPS forward @16 vs @128 (baseline file)    | 2.35 vs 2.48 ms (flat) |

### 3. KV cache, batch 1 (tok/s, median of 5)

| path                          | 64 new tokens   | 110 new tokens (fills window) |
| ----------------------------- | --------------- | ----------------------------- |
| torch CPU, no cache           | 369             | 239                           |
| torch CPU, KV cache           | **1166** (3.2x) | **611** (2.6x)                |
| torch MPS, no cache           | 153             | 182                           |
| torch MPS, KV cache           | 171 (1.1x)      | 244 (1.3x)                    |
| torch MPS fp16, no cache      | 224             |                               |
| torch MPS fp16, KV cache      | 244             |                               |
| torch CPU, KV cache, 1 thread | 1765            |                               |

Correctness: KV-cached greedy output is token-identical to uncached for all 16 eval
prompts, generating to the full 128-token window, on CPU and on MPS. Max abs logit
difference 1.0e-5 (CPU) and 1.3e-5 (MPS). Seeded sampling through `generate()` is
identical 8/8, and a 200-token run that overflows the window (exercising the recompute
fallback) is identical.

### 4. Batch sweep: aggregate tokens/s (median), small = trained 8.7M, large = random 88.3M

Large config: 12 layers, d_model 768, 12 heads, ffn 3072 (GPT-2-small shape), randomly
initialized, timing only. Generation = 16-token prompt + 32 greedy tokens per sequence.
Prefill = one forward over B x 128 tokens.

Small model (8.7M):

| B   | prefill CPU | prefill MPS | gen cached CPU | gen cached MPS | gen uncached CPU | gen uncached MPS |
| --- | ----------- | ----------- | -------------- | -------------- | ---------------- | ---------------- |
| 1   | 21,249      | **27,679**  | **1,142**      | 234            | **448**          | 226              |
| 4   | 51,285      | **116,114** | **1,574**      | 901            | 699              | **848**          |
| 16  | 90,008      | **190,079** | **4,946**      | 3,424          | 1,611            | **2,816**        |
| 64  | 101,191     | **318,900** | 8,651          | **12,350**     | 2,707            | **7,036**        |
| 256 | 97,989      | **382,005** | 9,464          | **41,218**     | 3,058            | **10,848**       |

Large model (88.3M):

| B   | prefill CPU | prefill MPS | gen cached CPU | gen cached MPS | gen uncached CPU | gen uncached MPS |
| --- | ----------- | ----------- | -------------- | -------------- | ---------------- | ---------------- |
| 1   | 6,067       | **15,584**  | **214**        | 154            | 70               | **132**          |
| 4   | 10,187      | **33,244**  | 299            | **544**        | 168              | **413**          |
| 16  | 12,858      | **47,683**  | 931            | **2,055**      | 313              | **923**          |
| 64  | 13,327      | **52,257**  | 1,630          | **6,674**      | 409              | **1,484**        |
| 256 | 13,561      | **53,279**  | 2,106          | **13,857**     | 417              | **1,495**        |

Crossover (first batch size where MPS wins), measured:

| workload                 | small 8.7M          | large 88.3M         |
| ------------------------ | ------------------- | ------------------- |
| cached generation        | between 16 and 64   | between 1 and 4     |
| uncached generation      | between 1 and 4     | already at 1        |
| one prefill forward @128 | already at 1 (1.3x) | already at 1 (2.6x) |

Spread between reps was mostly within 5%; the noisiest cells were MPS small-model at
B=4 and B=16 prefill (min 72k and 135k vs medians 116k and 190k).

### 5. Deployment variants (CPU, batch 1)

| variant                                                  | size                | fwd @16   | fwd @128  | gen 64 tok        | top-1 vs fp32 (197 positions) | greedy replies identical to fp32 |
| -------------------------------------------------------- | ------------------- | --------- | --------- | ----------------- | ----------------------------- | -------------------------------- |
| torch fp32 (state dict)                                  | 34.9 MB             | 1.92 ms   | 5.71 ms   | 369 / 1166 cached | ref                           | ref                              |
| ONNX fp32                                                | 41.2 MB             | 0.57 ms   | 2.07 ms   | 965 / 2490 cached | 100% (max err 1.8e-5)         | 16/16 (cached graphs)            |
| ONNX int8 dynamic (QInt8)                                | 10.5 MB             | 0.51 ms   | 2.66 ms   | 928 / 3442 cached | 99.5% (max err 0.31)          | 16/16 (cached graphs)            |
| ONNX uint8 dynamic (QUInt8, what `export_onnx.py` ships) | 10.5 MB             | 0.44 ms   | 2.03 ms   | not measured      | 99.5% (max err 0.37)          | not measured                     |
| ONNX int8 cached, 1 thread                               | 10.5 MB             |           |           | **3948**          |                               |                                  |
| ONNX int8 cached, 2 / 4 threads                          |                     |           |           | 3647 / 3526       |                               |                                  |
| torch int4 weight-only, simulated (group 32)             | ~4.9 MB (estimated) | not timed | not timed | not timed         | 98.5% (max err 2.2)           | 8/16                             |

"Cached" ONNX = two exported graphs, `prefill` (prompt to logits + stacked K/V) and
`decode` (one token + past K/V). Both ONNX sizes include a second copy of the tied
embedding as the transposed lm_head (1.6M params), which is why fp32 ONNX (41.2 MB) is
larger than the torch state dict (34.9 MB).

Sampled replies (temperature 0.7, top-k 50, same seed per prompt), excerpt:

| prompt                           | torch fp32                                                                                                  | ONNX int8 (cached)                                                                                           | int4 simulated                                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| hi guppy                         | hi there. i was just pretending to be a rock.                                                               | oh hi. i'm along the glass wall.                                                                             | hi there. i was just pretending to be a rock.                                                                |
| want some food?                  | food is the best thing. i've been following a bubble waiting for this moment. i'll eat it by the glass.     | i could eat. i can always eat. i've been blowing tiny bubbles waiting for this moment. thank you in advance. | food is the best thing. i've been following a bubble waiting for this moment. i'll eat it behind the castle. |
| brrr it's freezing               | cold is better than hot for me. i just float in place when it's cold. i'll be by the filter if you need me. | a little cold is ok. i'm hiding by the filter.                                                               | cold water makes me slow. a little cold is fine. a lot of cold is not fine. my side fins are a little stiff. |
| what do you think about politics | i don't understand politics. i think about food and water. that's my range.                                 | i don't know what politics is. if not i probably don't know it.                                              | i don't understand politics. i think about food and water. that's my range.                                  |

(ONNX int8 samples use a numpy RNG, so they differ from torch samples for that reason,
not because of quantization; its greedy replies match fp32 16/16.) All 8 sampled replies
per variant are in `runs/speed/export.json` and all stay in the fish voice.

### 6. torch.compile and fp16 (brief)

|                       | eager   | torch.compile (default inductor) |
| --------------------- | ------- | -------------------------------- |
| CPU forward @128, B=1 | 6.50 ms | 7.89 ms (compile 7.5 s)          |
| MPS forward @128, B=1 | 2.19 ms | 2.89 ms (compile 1.4 s)          |

fp16 on MPS: batch 1 generation unchanged within noise (224 / 244 tok/s); prefill at
B=64 409k tok/s (vs 319k fp32, +28%), at B=256 463k tok/s (vs 382k, +21%).

## Findings

Measured:

1. The surprise reproduces: at batch 1, MPS generation (153 to 234 tok/s) loses to CPU
   (369 tok/s), and ONNX CPU (2.0 ms @128) beats MPS.
2. MPS forward time is flat in sequence length (2.35 ms @16 vs 2.48 ms @128 in the
   baseline file) and does not improve when 50 forwards are queued before syncing
   (2.24 vs 2.16 ms). A forward is 167 aten ops; a queued tiny MPS op costs ~5 us of
   host time and a synced one ~106 us.
3. KV cache: 3.2x on CPU (369 to 1166 tok/s), only 1.1x to 1.3x on MPS. Outputs are
   token-identical.
4. Batching flips the result. Small model: MPS wins cached generation from B=64
   (12.4k vs 8.7k tok/s) and by 4.4x at B=256 (41.2k vs 9.5k). CPU throughput saturates
   around B=64 (~9.5k cached, ~100k prefill). Large model: MPS wins cached generation
   from B=4, and prefill by 2.6x already at B=1.
5. Fastest single-stream fish: ONNX Runtime int8, KV-cached graphs, 1 intra-op thread:
   3948 tok/s, 10.7x the baseline CPU path (369) and ~17x to 26x MPS, in a 10.5 MB file
   whose greedy replies match fp32 16/16. More threads are slower for this path (1: 3948,
   2: 3647, 4: 3526, default: 3442); torch CPU cached is also faster on 1 thread (1765 vs
   1142).
6. Fastest bulk fish: MPS, KV cache, large batch: 41.2k tok/s at B=256 (fp32). fp16
   adds ~20 to 28% to large-batch prefill on MPS and nothing at B=1.
7. torch.compile made both CPU and MPS slower at B=1 for this model.
8. Nothing measured fits in "a few MB" at runtime: int8 is 10.5 MB. Simulated int4
   (~4.9 MB estimated) keeps 98.5% top-1 agreement and still sounds like the fish in
   samples, but greedy replies drift from fp32 in 8/16 prompts. It was simulated in torch
   (fake-quantized weights), not run in an int4 runtime: ORT's `MatMulNBitsQuantizer`
   needs the `onnx_ir` package, which is not installed and was not added.

Inferred (explanation, not directly measured):

- **Why the GPU loses:** at batch 1 GuppyLM is launch-bound, not compute-bound. One token
  is ~17 MFLOP, microseconds of GPU work, but each forward issues ~167 separate ops and
  the MPS backend spends roughly 10 to 15 us of CPU time encoding each one, a ~2 ms floor
  regardless of sequence length. That matches the flat latency curve and the fact that
  queuing (removing sync waits) does not help: the cost is on the host side, before the
  GPU sees work. The CPU path has ~1 us per op, so it pays mostly real compute.
- **Why the KV cache barely helps MPS:** it cuts FLOPs (one token instead of the whole
  prefix) but not the op count per step, and op count is what MPS pays for. On CPU,
  FLOPs are the cost, so the cache gives 3x.
- **Why batching and model size move the crossover:** both add arithmetic per launched op
  without adding ops, so the fixed dispatch floor gets amortized. A 10x larger model
  reaches the crossover at ~4 sequences instead of ~16 to 64.
- **Why ONNX Runtime wins on CPU:** a fused, graph-level executor with far lower per-op
  overhead than eager PyTorch. Single-thread being fastest suggests the matrices at B=1
  (1 x 384 by 384 x 1152, etc.) are too small for thread fan-out to pay for itself.

## Recommendation

- Chatting with one fish: ONNX Runtime CPU, int8, KV-cached prefill/decode graphs,
  `intra_op_num_threads=1` (~3.9k tok/s, 10.5 MB). In PyTorch, use `use_cache=True` on
  CPU (~1.2k to 1.8k tok/s).
- Serving many fish at once: MPS with KV cache and batch 64 or more.
- Skip torch.compile and MPS at batch 1 for this model.

## Caveats

- Other agents' training shared the machine; the lock serialized timing but not their
  non-timed CPU work outside it. MPS batch-1 numbers varied up to ~2x between phases of
  the same run; CPU and ONNX numbers were stable (spread mostly under 5%).
- ONNX cached graphs were exported with the TorchScript exporter (opset 17). The batch
  sweep used only torch; ONNX was not batched.
- One bug found and fixed during the work: exporting a freshly constructed wrapper
  Module (default training mode) makes `torch.onnx.export` restore training mode
  recursively onto the wrapped model, silently re-enabling dropout. `speed_bench.py` now
  calls `.eval()` on wrappers and on the model after export.
