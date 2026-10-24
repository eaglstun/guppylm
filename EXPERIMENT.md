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
  - 6,000 more by 12 Haiku agents for `aq-chat-v3` (86% unique sentences, same as the first 6,000).
  - Ollama `qwen3-coder:30b`: 573 for the runs through `aq-chat-full`, 6,015 for `aq-chat-v2`,
    11,874 for `aq-chat-v3`, plus 300 held-out-persona dialogues (`dialogues_eval.jsonl`).
  - Unique assistant sentences: Haiku 88.5%, Ollama 93.1%. The template fish's data: 3.5%.
- **Probe** (`tools/aquarium/probe.py`, `partial.py`): every test uses words never in training.
  4 samples per prompt, temperature 0.7, top-k 40, seed 0, CPU.

## Results (measured)

| run                         | chat val loss | name exact | name first 3 letters | made-up noun exact | made-up noun first 3 | held-out species | memory (2 turns back) |
| --------------------------- | ------------: | ---------: | -------------------: | -----------------: | -------------------: | ---------------: | --------------------: |
| template fish (`fix-nouns`) |           n/a |        n/a |                  n/a |               0.00 |                 0.00 |              n/a |                   n/a |
| aq-chat, 1.5k steps         |         2.223 |       0.00 |                 0.54 |               0.00 |                 0.53 |             0.42 |                  0.25 |
| aq-chat-5k (best @4.5k)     |         2.062 |       0.00 |                 0.77 |               0.00 |                 0.48 |             0.67 |                  0.06 |
| aq-chat-rename, 4.5k        |         2.068 |       0.71 |                 0.94 |               0.18 |                 0.63 |             0.75 |                  0.06 |
| aq-chat-full, 6k            |         2.087 |       0.69 |                 0.96 |               0.35 |                 0.98 |             0.75 |                  0.75 |
| aq-chat-v2, 7k              |         1.869 |       0.56 |                 0.83 |               0.58 |                 0.98 |             0.75 |                  0.81 |
| **aq-chat-v3**, 15k (best @14k) | **1.729** | **0.92** |             **1.00** |           **0.68** |             **0.88** |         **0.71** |              **0.69** |

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

- **Noun and memory augmentation** (`aq-chat-full`: `--rename-copies 2 --noun-copies 2
  --memory-copies 1`, 30,200 train dialogues). `--noun-copies` swaps the mentioned human thing
  for an invented noun ("a pizza" -> "a wobplon") in the 1,949 dialogues where the assistant
  names it. `--memory-copies` prefixes the first user turn with a fact (invented name, a color, a
  number, a friend's name; the probe's exact values are excluded) and appends a question about it
  with a short templated answer. Memory goes from 1 of 16 to 12 of 16 ("Your shell was purple.",
  "Your friend is Gus.", "You said seven."); the misses are all Bartholomew ("Your name is
  Bartholwwit."). Made-up nouns: first 3 letters 0.63 -> 0.98, exact 0.18 -> 0.35 ("A florbix?
  I have never seen one ... Did the florbix take you on a fun ride?"). Names unchanged within noise.
- **Side effect:** the templated memory answers leak. "I remember." shows up in replies to
  "What is your name?" ("I am Marble. I remember. I found a small shell."), and one color answer
  came out "Yellow! A purple shell." The templates are few and short, so the model memorized them,
  the original problem in miniature. More varied answers (written by a model, not templates) would
  fix it.

- **Leak fix and more data** (`aq-chat-v2`): the templated memory answers were replaced by
  1,920 persona-voiced (fact, question, answer) triples written by Haiku agents (`data_aq/memory/`,
  99.4% unique answers, "I remember" in 0.2%), and Ollama data grew from 573 to 6,015 dialogues
  (54,064 train dialogues after augmentation). 7,000 steps, still improving at the end. "remember"
  in the 176 non-memory replies: 2 -> 0. Chat val loss 2.087 -> 1.869 (the val set is resampled
  each run, so this is indicative only). Made-up nouns exact 0.35 -> 0.58 ("A vorpelstick? I have
  never seen one, but it sounds like a flat box with lines."). Memory 12 -> 13 of 16; Bartholomew
  still drifts ("Bartholomewwook"), and "seven brothers" comes back as "seven brother".
- **Names dipped**, 0.69 -> 0.56 (n=48, near the noise line). Two patterns in the misses:
  - The copy stops at an English word boundary: Marbleton -> "Marble" (8 of 8, both runs),
    Wobbleton -> "Wobble" (6 of 8 this run). Neither is a training name; the model copies the
    first word-like piece and closes the name.
  - "What is your name?" answered as if it were "What is my name?": "You are you, grandparent."
    8 of 48 replies start "You are", up from 4. The memory copies add many "What is my name?" ->
    "Your name is ..." pairs, and the two questions differ by one word.

- **Compound names, self-name copies, twice the data, twice the steps** (`aq-chat-v3`): a quarter
  of invented names now start with a real word (Bubbleington, Crumbleham); `--selfname-copies 1`
  appends "What is your name?" with a Haiku-written, persona-voiced answer (624 pairs), a third of
  the time after the user has stated their own name; 24 memory triples whose name question could
  be about either speaker were dropped. 23,874 base dialogues (130,975 after augmentation),
  15,000 steps. Held-out-persona loss on the 300 eval dialogues: 2.123 (v2) -> 2.055.
  - Names exact 0.56 -> 0.92. Marbleton 0 -> 8 of 8 ("Marbleton. I am a careful crab who likes to
    look for better shells."), Wobbleton 2 -> 8 of 8. Replies to "What is your name?" that start
    "You are": 8 -> 0. "remember" in non-memory replies: still 0.
  - The one weak name is Tuppence (4 of 8, "I am Tuppal."). `tup` is one of the invented-name
    syllables, so the model has a strong habit of continuing Tup- with a training syllable.
  - Memory 13 -> 11 of 16, near noise, but the misses changed: Bartholomew now comes back exact
    half the time ("You are Bartholomew."), and "purple" fails 3 of 4 ("A shell you found was
    pink."). Purple is kept out of the 14 training colors, so the color slot is the
    31-noun problem again: with 14 values, the model learned "say a color", not "copy the color".
  - Made-up nouns exact 0.58 -> 0.68; real nouns 0.66 -> 0.52 (n=64, partly noise).
- **Browser build** (`web/`, `tools/aquarium/export_web.py`): int8 weights with a scale per row,
  9.04 MB, run by a dependency-free JS runtime with a KV cache. Held-out loss float vs int8:
  2.0550 vs 2.0562. Greedy decoding matches PyTorch token for token; see `web/README.md`.

## Inferred

- The renaming result is the same lesson as the 31-noun template fish, one level up: with ~150
  training names the model can get by with "say a name from my list". Thousands of invented names
  make that impossible, so it learns to copy. Variety in the slot is what teaches the copy.
- Memory was weak for the same reason: user-stated facts recalled later were rare in the data.
  Injecting them fixed most of it. Long held-out names (Bartholomew, 6 tokens) still break
  partway, in memory and in names alike: the copy holds for 3 to 4 tokens, then drifts.

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
python3 tools/aquarium/train_aq.py finetune --init runs/aq-pre/final.pt --out runs/aq-chat-full \
    --steps 6000 --eval-every 500 --rename-copies 2 --noun-copies 2 --memory-copies 1
python3 tools/aquarium/train_aq.py finetune --init runs/aq-pre/final.pt --out runs/aq-chat-v2 \
    --steps 7000 --eval-every 500 --rename-copies 2 --noun-copies 2 --memory-copies 1
python3 tools/aquarium/train_aq.py finetune --init runs/aq-pre/final.pt --out runs/aq-chat-v3 \
    --steps 15000 --eval-every 1000 --rename-copies 2 --noun-copies 2 --memory-copies 1 --selfname-copies 1
python3 tools/aquarium/export_web.py runs/aq-chat-v3/best.pt --out web/model
python3 tools/aquarium/probe.py runs/aq-chat-rename/best.pt --out runs/aq-chat-rename/probe.json
(cd tools/aquarium && python3 partial.py ../../runs/aq-chat-rename/probe.json)
```

`data_aq/` and `runs/` are gitignored.
