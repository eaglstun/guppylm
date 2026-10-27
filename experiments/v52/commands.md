# v5.2 commands

Run from the implementation worktree. The measured preparation and screens used
revision `70e357d`, Python 3.10.19, torch 2.14.1, tokenizers 0.23.3, numpy 2.2.6.
Source datasets/checkpoints are read from the existing Aquarium worktree; new data
and checkpoints go under this worktree's ignored `runs/v52/` directory.

```bash
AQ=/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium
py() { uv run --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python "$@"; }

# Use a fresh output directory when changing generator code or inputs.
py -m tools.aquarium.v52.prepare --data "$AQ/data_aq" --out runs/v52/contract-v1

for version in v5 v51; do
  py -m tools.aquarium.v52.evaluate \
    --contract runs/v52/contract-v1/contract.json \
    --checkpoint "$AQ/runs/aq-chat-$version/best.pt" \
    --tokenizer "$AQ/data_aq/tokenizer.json" \
    --out "runs/v52/$version-development.json" --threads 1
 done

py -m tools.aquarium.v52.score \
  --checkpoint "$AQ/runs/aq-chat-v51/best.pt" \
  --manifests runs/v52/contract-v1 --data "$AQ/data_aq" \
  --out runs/v52/v51-losses.json --device mps

# Freeze rubric.json after baseline measurement, BEFORE any candidate training.
# The recorded rubric is experiments/v52/rubric.json; it is specific to this contract.
for version in v5 v51; do
  py -m tools.aquarium.v52.train \
    --train runs/v52/contract-v1/train_C.json \
    --development runs/v52/contract-v1/development.json \
    --historical runs/v52/contract-v1/historical.json \
    --data "$AQ/data_aq" --init "$AQ/runs/aq-chat-$version/best.pt" \
    --out "runs/v52/screen-$version-C" --steps 400 --lr .00003 \
    --warmup 50 --eval-every 200 --device mps
  py -m tools.aquarium.v52.evaluate \
    --contract runs/v52/contract-v1/contract.json \
    --checkpoint "runs/v52/screen-$version-C/final.pt" \
    --tokenizer "$AQ/data_aq/tokenizer.json" \
    --out "runs/v52/screen-$version-C/development.json" --threads 1
 done

py -m tools.aquarium.v52.compare \
  --baseline runs/v52/v51-development.json \
  --candidate runs/v52/screen-v51-C/development.json \
  --rubric experiments/v52/rubric.json \
  --out runs/v52/screen-v51-C/comparison.json

py -m tools.aquarium.v52.evaluate --contract runs/v52/contract-v1/contract.json \
  --blind-baseline --out runs/v52/blind-baseline.json --threads 1
py -m tools.aquarium.v52.evaluate --contract runs/v52/contract-v1/contract.json \
  --train runs/v52/contract-v1/train_A.json --out runs/v52/lookup-baseline.json --threads 1

py -m unittest discover -s tests -v
```

### contract-v2 (train-only curation v2)

`tools/aquarium/v52/curate.py` applies versioned, train-only curation before augmentation:
looping rows dropped, at most 40 rows per assistant 8-gram (counting sequences exempt),
at most 40 rows per entity-normalized user turn (courtesy turns exempt), and rows of a
creature name above 1.25x the median renamed to the least-used training names. The cap
backfills from the same v1 shuffle, so base stays 20,000 rows. `--curation 1` reproduces
contract-v1 train rows exactly. `--frozen-eval` fails unless the contract and the
development/historical/compression-development rows equal the named contract's.
The audit is copied to [audit-v2.json](audit-v2.json).

```bash
py -m tools.aquarium.v52.prepare --data "$AQ/data_aq" --out runs/v52/contract-v2 \
  --frozen-eval runs/v52/contract-v1
# Optional train-only source (jsonl file or directory, data_aq row format; repeatable).
# Rows go through the same validation, dedup, reserved-entity and probe-collision checks,
# are quarantined if they share a normalized group with any eval row, and never enter eval splits.
#   --extra-source NAME=PATH
```

### contract-v3 (curation v3, the current default)

Curation v3 uses the same selection as v2, then applies the hand-written rewrites in
`tools/aquarium/v52/rewrites/trait_echo_v1.jsonl`. These replace assistant sentences that
recite the persona's system-prompt trait list. Detector: the assistant turn contains, as
whole normalized words, the leading trait list of `PERSONAS[persona][2]` (the words up to
and including the one after the first "and"/"but", e.g. "grumpy but fair"). Each entry is
pinned to the original turn's sha256, and any mismatch fails. A uid that is not selected
also fails, except when `--extra-source` shifts the selection; those are then recorded
under `rewrites.missing`. `--curation 2` still reproduces contract-v2's rows.

```bash
py -m tools.aquarium.v52.prepare --data "$AQ/data_aq" --out runs/v52/contract-v3 \
  --frozen-eval runs/v52/contract-v1
```

Manifests keep format `version` 1; the generator is `v52.prepare/v2` and metadata records
`curation` and `extra_sources`. Eval-split manifest file hashes differ from v1 only because
their metadata records the new code hashes; their rows are identical (see `frozen_eval`).

The archived behavior generation preceded static-only import/list formatting and
the stricter test-access check. The lookup tie-break is now explicitly lexical so
it no longer depends on Python's set hash order. Results are never overwritten.

## Matched full runs (prepared, not executed)

After improving/reviewing C and freezing a revised contract if needed, the matched
commands use the same aq-td checkpoint and 15,000-step schedule for each candidate:

```bash
for candidate in A B C; do
  py -m tools.aquarium.v52.train \
    --train "runs/v52/contract-v1/train_$candidate.json" \
    --development runs/v52/contract-v1/development.json \
    --historical runs/v52/contract-v1/historical.json \
    --data "$AQ/data_aq" --init "$AQ/runs/aq-td/best.pt" \
    --out "runs/v52/full-$candidate" --steps 15000 --seed 42 --device mps
 done
```

For a resumable interruption, add `--stop-after 1000` without changing `--steps`.
Resume with the **same command/settings/code/output directory**, remove
`--stop-after`, and add `--resume runs/v52/full-A/final.pt`. The checkpoint step
counts completed optimizer updates. A different seed (73 for confirmation), total
schedule, manifest, code, tokenizer or runtime is a new run, not a resume.

## Compression after float selection (not executed)

Existing d4, E8 and hybrid tools now share train-only calibration and distillation.
Set **both** manifests; using only one is an error. The scoring manifest must carry
`compression_development` split labels. Model-development and locked-test rows are
rejected. Both manifests must belong to the same evaluation contract/tokenizer.

```bash
export AQ_QUANT_CKPT="$PWD/runs/v52/full-A/best.pt"  # replace only after actual selection
export AQ_QUANT_DATA="$AQ/data_aq"
export AQ_QUANT_TRAIN_MANIFEST="$PWD/runs/v52/contract-v1/train_A.json"
export AQ_QUANT_DEV_MANIFEST="$PWD/runs/v52/contract-v1/compression_development.json"
export AQ_QUANT_OUT="$PWD/runs/v52/compression"

py tools/aquarium/quant/stage4_tune.py vq 4 4096 --stages 1 --rot --steps 600
```

The E8/hybrid functions in `stage15_e8p.py` and `stage20_hybrid_webp_e8.py` use the
same manifest-aware training-row accessor. This wiring does not supply a missing
d4 packer, hybrid browser reader, or phone benchmark. Do not compare old v5.1 payload
sizes as though they were new v5.2 artifacts.

## Locked-test access (not executed)

Write a selection JSON containing `checkpoint_sha256`, `contract_sha256`,
`development_report_path`, `development_report_sha256`, `rubric_path`, and
`rubric_sha256`. Paths are relative to the selection file or absolute. The evaluator
verifies the referenced files, contract, split and evaluated checkpoint, then accepts
`--split locked_test --selection path/to/selection.json`. This is an audit guard,
not an automatic release decision: human review, full-run confirmation, compression
quality, parity and real phone measurements remain separate requirements.

### contract-v3-talkie and the first matched full runs

```bash
python3 -m tools.aquarium.v52.respell runs/v52/talkie/whimsy-v1/chat.jsonl
python3 -m tools.aquarium.v52.talkie_review --out runs/v52/talkie/whimsy-v1/chat_reviewed.jsonl
py -m tools.aquarium.v52.prepare --data "$AQ/data_aq" --out runs/v52/contract-v3-talkie \
  --frozen-eval runs/v52/contract-v1 \
  --extra-source talkie=runs/v52/talkie/whimsy-v1/chat_reviewed.jsonl

# Matched A/B/C on the talkie contract, then A on plain v3 (confounded talkie ablation).
for run in v3-talkie:A v3-talkie:B v3-talkie:C v3:A; do
  contract=${run%%:*}; candidate=${run##*:}
  py -m tools.aquarium.v52.train \
    --train "runs/v52/contract-$contract/train_$candidate.json" \
    --development "runs/v52/contract-$contract/development.json" \
    --historical "runs/v52/contract-$contract/historical.json" \
    --data "$AQ/data_aq" --init "$AQ/runs/aq-td/best.pt" \
    --out "runs/v52/full-$contract-$candidate" --steps 15000 --seed 42 --device mps
done
```

### Contextual vocabulary addition (not trained)

For a new preparation run, use a fresh `--out` directory and append
`--extra-source talkie_vocabulary=tools/aquarium/v52/rewrites/talkie_vocabulary_v1.jsonl`
to the desired base recipe. Do not overwrite the frozen outputs above. See
[talkie-vocabulary.md](talkie-vocabulary.md) for coverage and validation.
