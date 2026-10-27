# Uppercase urgency and shouting data

The append-only `uppercase_v2` data pass adds 32 authored conversations in 20
existing training persona/name/system contexts, for 640 additional rows in each
of A, B, and C. It teaches prompt attention to urgent requests, calm responses to
frustrated shouting and boundaries, and contextual responses to excited or ambiguous
calls. Four excitement replies use full capitals, and two urgency replies begin with
brief uppercase acknowledgments. Replies to frustrated shouting remain calm and
normally capitalized. Normal-case follow-up turns are preserved.

The 32 scenarios comprise 12 urgency, 12 frustrated shouting, four excitement,
and four ambiguous calls. These are 32 wording scenarios repeated across contexts,
not 640 independently authored conversations. They are a small training addition;
behavioral improvement has not yet been measured.

Source: `tools/aquarium/v52/rewrites/uppercase_v2.jsonl`.
Builder: `tools/aquarium/v52/uppercase.py`.
Output: `runs/v52/contract-v3-talkie-uppercase-v2/`.

```sh
uv run --offline --python 3.10 --with tokenizers==0.23.3 \
  python -m tools.aquarium.v52.uppercase \
  --base runs/v52/contract-v3-talkie \
  --out runs/v52/contract-v3-talkie-uppercase-v2 --contexts 20 \
  --scenarios tools/aquarium/v52/rewrites/uppercase_v2.jsonl --version uppercase_v2 \
  --tokenizer /Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json
```

The builder appends identical additions to all three candidates and verifies every
original row remains unchanged. It copies the behavioral contract and three evaluation
manifests byte for byte, checks reserved entities and normalized dialogue/probe
collisions, and records provenance, input hashes, casing counts, and token shares
in `uppercase_audit.json`. Creature contexts are limited to compatible mobile ocean
species with human conversation partners. The existing tokenizer is retained.

Use the new directory's `train_A.json`, `train_B.json`, or `train_C.json` for a
future training run, with a new output directory. Existing running/queued training
commands still point to the original manifests; this data pass does not change them
or resume an old checkpoint with different training data.

V2 preserves every v1 user turn and changes six assistant replies across the same
20 contexts: 80 fully uppercase replies and 40 replies with uppercase openings.
This replaces the v1 addition; it does not stack another 640 rows onto it. The
new manifests are built directly from the original `contract-v3-talkie` data.

The earlier incomplete `contract-v3-talkie-uppercase/` output is marked INVALID
and must not be used. The `-v1` output remains a valid historical version with
normally capitalized replies; `-v2` is the current variant.
