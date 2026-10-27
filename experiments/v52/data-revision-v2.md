# Reviewed training-data revision v2

This candidate revises the response-first v1 dataset after its inconclusive screens.
It has **131,994 rows**, down from 141,817. No training or behavioral evaluation
has been run on this candidate, and no improvement is claimed.

## Changes

- Remove 10,699 noun-substitution rows whose replies assume an invented word's
  meaning without the user explaining it. Keep their original conversation families.
- Add 512 explicitly defined pretend-word conversations: 32 per creature persona,
  one per selected parent family. These use four controlled templates, not 512
  independently authored scenarios. Definitions apply to the particular game or
  story; one template asks for clarification before learning the meaning.
- Change 22,060 retained system prompts from “You only know about life in …” to
  “You live in …”. This retains habitat and identity without instructing ignorance.
  The new grounded examples receive the same normalization.
- Add 24 authored scenarios in normal and uppercase-first-user forms (48 rows),
  covering unfamiliar words, unclear requests, corrections, pretend limitations,
  follow-through, and ordinary context. Existing 80 responsive rows remain.
- Add 316 uppercase-first-user counterparts of the previously reviewed Talkie
  base conversations. Preserve their answers, subsequent turns, parents and groups.

All 15,180 song rows remain. Reasonable analogies and whimsy remain; an unfamiliarity
keyword alone is not grounds for deleting a conversation. The explicit word-game
examples aim to keep silliness understandable. A final spot review corrected
physical-object assumptions in templates used with abstract meanings such as homework.

## Integrity and limitations

Six regression tests pass. All 876 added rows have assistant targets and fit the
512-token training window with the actual tokenizer. Reserved entities and exact
frozen probe questions are excluded; parent/group isolation is checked for changed
and added rows. The contract and three evaluation manifests are byte-identical to
v1. There are no remaining old noun-substitution rows or restrictive habitat strings
in the candidate training manifest. This does not assert that every inherited row
fits the context window or that every inherited reply has been freshly reviewed.

Evaluation retains its original system prompts, so those restrictive instructions
remain a train/evaluation difference. Removing the large noun augmentation pool
changes the source mixture; the next training comparison must account for that.
Sampling weights are not changed by this data revision.

## Reproduction and artifacts

From the repository root, using the existing v1 manifest and tokenizer:

```sh
uv run --offline --python 3.10 --with tokenizers==0.23.3 python -m tools.aquarium.v52.revise_data \
  --base runs/v52/contract-responsive-v1 \
  --out runs/v52/contract-responsive-v2-reviewed \
  --tokenizer /Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json
python3 -m unittest discover -s tests -p 'test_v52_revise_data.py'
```

The builder refuses to overwrite an existing output. The preliminary local
`contract-responsive-v2` candidate is superseded by `contract-responsive-v2-reviewed`;
its original manifest is preserved.

Reviewed manifest SHA-256:
`666ea25fe991d372e7a8eaf1d14b311aeb49ea0fda2742eb3e973816292fce2f`.

[Summary and input hashes](results/data-revision-v2/summary.json) record the source
counts, token exposure, tokenizer, builder and scenario hashes. The same directory
archives the full revision audit, explicit-meaning rows, case counterparts and
imported authored rows as gzip files. The editable authored scenarios live in
`tools/aquarium/v52/rewrites/responsive_v2.jsonl`; the deterministic builder lives in
`tools/aquarium/v52/revise_data.py`.
