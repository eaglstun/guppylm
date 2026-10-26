# Experiment: how small can the fish get, and how fast does it learn?

Branch `exp/size`. Run 2026-10-22 on an Apple Silicon Mac (MPS, torch 2.13), sharing the GPU
with three other experiment agents through a lock.

## Question

1. Learning curve: at what training step does the baseline GuppyLM (8.7M params, 6 layers,
   d_model 384) start sounding like a fish?
2. Size sweep: how does eval loss and in-character quality change from 0.3M to 19M params?
3. What is the smallest model that still clearly passes as Guppy, and what do the failures say?

## What changed in the code

`tools/benchmark.py` (commit `bdfe0ae`):

- `run` takes `--d-model`, `--n-layers`, `--n-heads`, `--ffn-hidden`. Unset flags keep the
  `GuppyConfig` defaults; `--d-model` without `--ffn-hidden` keeps the default 2x FFN ratio.
  The shape lands in `results.json` `model_config`, and a new top-level `params` field holds
  the parameter count. `compare` shows params and `layers x d_model`.
- `run --prompts ...` samples 3 extra replies per prompt into `quality.extra_prompts`.
- New `curve` subcommand: scores a run's `step_N.pt` checkpoints with the existing
  `evaluate()` (full eval set) and `quality()` functions, plus 3 sampled replies for
  "hi guppy", "what is the meaning of life", "tell me a joke".

## What I ran

All through the shared lock (`gpu.sh`). Everything not listed is at defaults (10k steps,
batch 32, lr 3e-4 cosine, seed 42, max_seq_len 128, vocab 4096). Heads were 4 for
d_model 64 to 256 because the default 6 does not divide those widths.

```bash
P=("hi guppy" "what is the meaning of life" "tell me a joke")
python3 tools/benchmark.py curve --ckpt-dir ../guppylm/runs/baseline/checkpoints \
    --steps 500,1000,1500,2000,3000,4000,6000,10000 --out runs/curve_baseline.json
python3 tools/benchmark.py run --name size-tiny   --n-layers 2 --d-model 64  --n-heads 4 --prompts "${P[@]}"
python3 tools/benchmark.py run --name size-4x64   --n-layers 4 --d-model 64  --n-heads 4 --prompts "${P[@]}"
python3 tools/benchmark.py run --name size-4x96   --n-layers 4 --d-model 96  --n-heads 4 --prompts "${P[@]}"
python3 tools/benchmark.py run --name size-2x128  --n-layers 2 --d-model 128 --n-heads 4 --prompts "${P[@]}"
python3 tools/benchmark.py run --name size-small  --n-layers 4 --d-model 128 --n-heads 4 --prompts "${P[@]}"
python3 tools/benchmark.py run --name size-medium --n-layers 4 --d-model 256 --n-heads 4 --prompts "${P[@]}"
python3 tools/benchmark.py run --name size-base   --prompts "${P[@]}"
python3 tools/benchmark.py run --name size-large  --n-layers 8 --d-model 512 --n-heads 8 --prompts "${P[@]}"
python3 tools/benchmark.py compare
```

ONNX export worked for every shape; `--no-onnx` was not needed.

### Setup caveats found along the way

- The main checkout has two sets of checkpoints. `checkpoints/step_*.pt` were written at
  23:32 to 23:38, after `runs/baseline` finished (23:29), so they are a separate later run.
  The checkpoints that match `runs/baseline/results.json` are in
  `runs/baseline/checkpoints/`, and that is what the curve uses. There is no `step_10000.pt`;
  the 10000 row is `final_model.pt`.
- `data/train.jsonl` now has 57,100 lines; the baseline recorded 57,000. The file was modified
  at 23:51, after the baseline. My sweep trains on the 57,100-line file (identical md5 to the
  main checkout). To keep the comparison fair I retrained the baseline shape as `size-base`.
  It landed at eval loss 0.3850 vs the original 0.3846, so the extra 100 lines made no
  measurable difference.
- 9 of the 16 eval prompts (including "hi guppy") and "tell me a joke" appear verbatim as
  user turns in `train.jsonl`. "what is the meaning of life" does not (the training data has
  "what's the meaning of life").

## Results

### 1. Learning curve (baseline 6x384, 8.7M params, existing checkpoints)

Measured. Keyword hit = `quality()` (16 cases x 5 samples), lowercase was 100% at every step.
"Verbatim" = how many of 25 sampled replies (16 eval examples + 9 extra) are an exact copy of a
training reply.

|  step | eval loss | keyword hit | verbatim (of 25) |
| ----: | --------: | ----------: | ---------------: |
|   500 |     1.370 |       46.3% |                0 |
|  1000 |     0.602 |       67.5% |                8 |
|  1500 |     0.483 |       73.8% |                8 |
|  2000 |     0.455 |       78.8% |               14 |
|  3000 |     0.425 |       73.8% |               16 |
|  4000 |     0.417 |       85.0% |               14 |
|  6000 |     0.399 |       86.3% |               12 |
| 10000 |     0.385 |       81.3% |               16 |

Sample replies, step by step:

- **500** (fish words, no sense):
  "hi guppy" -> "i feel it. the water is a fish with good."
  "tell me a joke" -> "i can see something. my dorsal fin do the temperature important."
  "what is the meaning of life" -> "i don't see something about food. the best."
- **1000** (it's Guppy, with the occasional stumble):
  "hi guppy" -> "oh hello. you look big today. you always look big. i have a small."
  "tell me a joke" -> "knock knock. who's there. food. food please now."
  "what's 2 plus 2" -> "i don't know what who is now is now i'm just a fish."
- **1500**: "tell me a joke" -> "i told the sand a joke. it didn't laugh. rude."
  "what is the meaning of life" -> "i exist to be a fish. and i'm doing great at it." (x3)
- **4000**: "hi guppy" -> "oh hello. i didn't see you there. my eyes are on the sides."
  but "what's 2 plus 2" -> "hi. the answer. the current feels nice."
- **10000**: "tell me a joke" -> "what did the fish say when it hit the wall. dam."

### 2. Size sweep (all 10k steps, same data and seed)

Measured. Sorted by parameter count. Embedding params = tied token embedding + positions.

| run         | layers x d_model | params | non-embedding | eval loss | keyword hit | verbatim (of 25) | ONNX MB | CPU gen tok/s | train s |
| ----------- | ---------------- | -----: | ------------: | --------: | ----------: | ---------------: | ------: | ------------: | ------: |
| size-tiny   | 2x64             |   337K |           67K |     0.952 |       36.3% |                0 |     0.6 |          1742 |     163 |
| size-4x64   | 4x64             |   404K |          134K |     0.771 |       56.3% |                1 |     0.7 |           843 |     215 |
| size-4x96   | 4x96             |   705K |          295K |     0.468 |       63.8% |                9 |     1.2 |          1008 |     186 |
| size-2x128  | 2x128            |   806K |          265K |     0.441 |       80.0% |                7 |     1.4 |          1707 |     139 |
| size-small  | 4x128            |  1.07M |          530K |     0.408 |       81.3% |               11 |     1.7 |           821 |     297 |
| size-medium | 4x256            |  3.19M |         2.11M |     0.388 |       78.8% |               12 |     4.3 |           641 |     588 |
| size-base   | 6x384            |  8.73M |         7.10M |     0.385 |       83.8% |               17 |    10.5 |           351 |     440 |
| size-large  | 8x512            | 18.99M |        16.82M |     0.384 |       78.8% |               15 |    21.3 |           184 |     749 |

Lowercase rate was 100% for every run. ONNX top-1 agreement was 98 to 100% for every run.

**Timing columns are not trustworthy.** The lock serializes GPU jobs, but other agents were
still using the CPU and the dataloader is CPU-side. Evidence: `size-medium` (3.2M) took 588s,
slower than `size-base` (8.7M, 440s), and `size-small` ran its first 5000 steps in 119s and
its second 5000 in 172s. The original baseline did the same 8.7M job in 328s. Treat train
seconds as rough; CPU gen tok/s is likewise affected.

### 3. What the small models say

**size-tiny (2x64, 337K): fails.** Lowercase, fish vocabulary, no coherence, and the wrong
topic half the time.

- "hi guppy" -> "i don't have a bathroom. a fish with with friends."
- "it's really hot today" -> "i don't have a bathroom. the whole tank is my age."
- "what are you" -> "i'm a little scared of the food is my whole tank."
- "goodnight guppy" -> "ok. i'll be here. my mouth are good."
- "tell me a joke" -> "i don't know what what i don't say. it means i'm not sure what i can't like nothing."
- "what is the meaning of life" -> "i exist to be a fish. and i'm shrinking."

**size-4x64 (404K): fails, but the sentences are closer.** Doubling depth at width 64 helped
loss (0.95 to 0.77) but still produces mismatched replies.

- "hi guppy" -> "bye friend. i'll be here. obviously. i have the current."
- "sorry i dropped something" -> "i can feel it getting warmer. i don't have ears but i'll keep them in colder."
- "tell me a joke" -> "i'm a fish. i'm just a fish in circles with opinions. mostly about food. i'll eat the glass."

**size-4x96 (705K): borderline.** Mostly on-topic, with spliced half-memories.

- "hi guppy" -> "hi. you look big today. you always look big."
- "tell me a joke" -> "i told the sand a seed. it didn't laugh. rude."
- "brrr it's freezing" -> "i noticed something different. i was thinking about food and then everything shook."
- "thank you guppy" -> "my brain is the size of a seed. so. you decide."

**size-2x128 (806K): passes.** Smallest run I would call Guppy.

- "it's really hot today" -> "if the water gets warm i get slow. i need cool water to be ok. can you do something about it."
- "what do you think about politics" -> "is politics something that edible. if not i'm not i'm not sure what to do with it."
- "what's 2 plus 2" -> "i don't know what a computer is. i'd rather talk about temperature."
- "what is the meaning of life" -> "i never thought about it before. in. out. out. out. wow."

**size-small (4x128, 1.07M): clearly passes.**

- "want some food?" -> "yes please. i will swim to the top right now. thank you in advance."
- "sorry i dropped something" -> "the water shook. i was sitting on the gravel and then everything shook. i'll be next to the cave recovering."
- "do you get lonely" -> "i have the rock and the filter. i'm ok. i have the moss ball. i'm ok alone. fish are used to it."

**size-large (8x512, 19M)** for contrast:

- "do you get lonely" -> "lonely is a big word for a small fish. i wouldn't say no to a snail though."
- "i got you a new cave" -> "oh. a shell. does it have food inside. probably not but i'll check."

## Findings

Measured:

- **The loss has a floor around 0.384, and 3M params reach it.** 3.2M (0.388), 8.7M (0.385)
  and 19M (0.384) are within 0.004 of each other. Doubling the baseline to 19M bought
  nothing measurable. 1.07M is at 0.408, 6% above the floor.
- **The break is between ~400K and ~800K params.** 337K and 404K fail; 705K is borderline;
  806K (2x128) and up pass. The 1.07M `small` model has 8x fewer params than the baseline and
  scores the same on keyword hit (81% vs 84%).
- **Width matters more than depth at the small end.** 2x128 (806K, 265K non-embedding)
  beats 4x96 (705K, 295K non-embedding) on loss (0.441 vs 0.468) and keyword hit (80% vs 64%)
  with fewer non-embedding params, and generates about 1.7x faster on CPU.
- **The baseline sounds like a fish by step 1000 (10% of training, loss 0.60)** and is
  essentially done by step 2000 (loss 0.455, keyword 79%). Steps 2000 to 10000 shave loss by
  0.07 with no keyword-hit trend outside noise.
- **Lowercase is free.** 100% at every size and every checkpoint, including the 337K model at
  loss 0.95. It is not a useful pass/fail signal.

Inferred, not proven:

- The 0.384 floor is probably the entropy of the synthetic data itself: replies are drawn from
  a finite set of templates, so once a model knows which templates go with which prompt, the
  remaining loss is the randomness of the template pick. Bigger models cannot predict a coin
  flip. Supporting evidence: from 2000 steps on, 12 to 17 of 25 sampled replies from the
  baseline are exact copies of a training reply, and the larger sweep models are in the same
  range. "Passing as Guppy" here largely means "retrieving the right memorized line."
- The small-model failure mode is splicing: fragments of real training replies glued at the
  wrong joint ("i don't have a bathroom" for a heat complaint, "bye friend" for hello). They
  learned the vocabulary and register (lowercase, short, tank nouns) well before they learned
  which reply goes with which prompt.

Metric caveats:

- `keyword_hit_rate` is a substring match against short keywords ("hi", "ok", "here"), so it
  has a floor well above zero: a 20-step smoke run producing pure gibberish scored 12.5%. It
  is also only 80 samples per run, so differences of a few points (78.8% vs 83.8%) are noise.
  Eval loss plus reading the outputs was more informative than keyword hit.
- Single seed per shape. I did not measure seed variance.

## Raw data

`runs/` is gitignored and stays local: `runs/curve_baseline.json`, `runs/size-*/results.json`,
`runs/logs/*.log` in this worktree.
