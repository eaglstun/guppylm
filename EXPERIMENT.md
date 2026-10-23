# Experiment: a quality fish at ~8.9M params, with system prompts and a 512-token context

Branch `exp/aquarium`. Goal: same model size as GuppyLM, but persona system prompts (16
aquatic creatures), multi-turn chat, some users who are themselves sea creatures, and a
context long enough to hold a conversation. Previous experiments (`fix/memorization`) showed
template data is the ceiling, so this one changes the data, not the model.

## Setup

- **Model:** unchanged GuppyLM architecture, 6 layers, d_model 384, 6 heads, FFN 768,
  vocab 4096, `max_seq_len` 512. 8,873,472 params. Attention now uses
  `F.scaled_dot_product_attention` (max abs difference vs the old code: 2e-7).
- **Tokenizer** (`tools/aquarium/prep.py`): byte-level BPE with the full byte alphabet, so
  capitals, punctuation and non-ASCII survive (`"Hello GUPPY! What's 2+2?"` and `你好` round-trip).
  Specials: `<pad>`, `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`.
- **Stage 1, pretraining** on TinyStoriesV2-GPT4 (553M tokens): 15,000 steps x 32 x 512
  (246M tokens, ~0.45 epoch), lr 6e-4 cosine, no dropout. 90 minutes on an M4 Max, sharing the
  GPU with Ollama.
- **Stage 2, chat fine-tuning:** loss only on assistant replies (the system prompt and user turns
  are context), 15% of each batch is story windows to keep the English. lr 2e-4, dropout 0.1.
- **Dialogue data** (`tools/aquarium/personas.py`, `gen_dialogues.py`): each id seeds a spec:
  persona (16 train, narwhal and mantis shrimp held out), a name (~150 train names, 6 held out),
  one of 6 system-prompt phrasings, user (35% another sea creature, else a human), topic,
  optional human thing to mention, 1 to 4 exchanges.
  - 6,000 dialogues written by 14 Claude Haiku agents from spec files.
  - 573 by Ollama `qwen3-coder:30b` (the generator kept running overnight; later runs can use more).
  - Unique assistant sentences: Haiku 88.5%, Ollama 93.1%. The template fish's data: 3.5%.
- **Probe** (`tools/aquarium/probe.py`, `partial.py`): every test uses words never in training.
  4 samples per prompt, temperature 0.7, top-k 40, seed 0, CPU.

## Results (measured)

| run                         | chat val loss | name exact | name first 3 letters | made-up noun exact | made-up noun first 3 | held-out species | memory (2 turns back) |
| --------------------------- | ------------: | ---------: | -------------------: | -----------------: | -------------------: | ---------------: | --------------------: |
| template fish (`fix-nouns`) |           n/a |        n/a |                  n/a |               0.00 |                 0.00 |              n/a |                   n/a |
| aq-chat, 1.5k steps         |         2.223 |       0.00 |                 0.54 |               0.00 |                 0.53 |             0.42 |                  0.25 |
| aq-chat-5k (best @4.5k)     |         2.062 |       0.00 |                 0.77 |               0.00 |                 0.48 |             0.67 |                  0.06 |
| **aq-chat-rename**, 4.5k    |         2.068 |   **0.71** |             **0.94** |               0.18 |                 0.63 |         **0.75** |                  0.06 |

n per cell: names 48, made-up nouns 40, species 24, memory 16. Differences under ~15 points
are within noise at these sizes.

- **Pretraining works.** Step 0: "zebra imp ignorant zebra". Step 15,000: "Once upon a time, a
  little fish named Fin lived in a big sea. Fin was a fearful fish. He did not want to swim fast
  like his friends."
- **Copying appears, then sharpens with name variety.** Without augmentation the fish copies the
  first piece of a new name and then guesses: Wobbleton -> "Wobble", Bartholomew -> "Barth",
  florbix -> "A florch". `--rename-copies 2` adds two copies of every training dialogue with the
  creature's name swapped for an invented one (`Morrok`, `Brindlewick`). Exact copying of
  held-out names goes from 0.00 to 0.71: "I am Wobbleton, and I drift with the current.",
  "I am Fizzgig, and I am grumpy but fair."
- **Held-out species** are played from the system prompt alone: mantis shrimp, never seen in
  training, says "I am Quilliam, and I am fierce. I live in a burrow in the reef."
- **Memory across turns is still weak** (1 of 16): asked its name back, it says "My name is Bee."

## Inferred

- The renaming result is the same lesson as the 31-noun template fish, one level up: with ~150
  training names the model can get by with "say a name from my list". Thousands of invented names
  make that impossible, so it learns to copy. Variety in the slot is what teaches the copy.
- Memory is weak for the same reason: user-stated facts recalled later are rare in the data.
  The same trick should work (inject "My name is X" / "I found a purple shell" turns with
  invented values and a later question).

## Caveats

- One seed per run, small probe n. The held-out-species score is a keyword match.
- Pretraining and Ollama shared the GPU, so timings are rough.
- `dialogues_eval.jsonl` (held-out persona dialogues for a loss number) was not generated before
  these runs; the probe is the held-out measurement.

## Commands

```bash
python3 tools/aquarium/prep.py tokenizer && python3 tools/aquarium/prep.py stories
python3 tools/aquarium/train_aq.py pretrain --out runs/aq-pre --steps 15000
python3 tools/aquarium/train_aq.py finetune --init runs/aq-pre/final.pt --out runs/aq-chat-rename \
    --steps 4500 --eval-every 500 --rename-copies 2
python3 tools/aquarium/probe.py runs/aq-chat-rename/best.pt --out runs/aq-chat-rename/probe.json
(cd tools/aquarium && python3 partial.py ../../runs/aq-chat-rename/probe.json)
```

`data_aq/` and `runs/` are gitignored.
