# Lowercase fish continuation

This implements the lowercase direction while keeping the fish's conversational
personality and preserving earlier training material.

## Data and behavior

The new versioned candidate starts with the reviewed 131,994-row corpus, includes
the 15 reviewed carry-forward conversations, lowercases system/user/assistant text,
and deduplicates identical resulting conversations. It contains **131,618 rows**.
Eleven carry-forward examples are already represented after normalization; four
add content (two small recall examples, the cloud library, and the excited-pebble
exchange). Historical habitat restrictions in carried examples are harmonized with
“you live in …”. Original datasets, row lineage and evaluation files are retained.

The tokenizer is unchanged. A bounded replay pool of 4,096 complete training stories
and 512 complete validation stories is selected reproducibly from the existing
story binaries, lowercased and retokenized. Complete-document boundaries avoid
cutting through words or Unicode bytes during conversion. The audit saves selected
document indices, seeds and input/output hashes. This is a short experiment's replay
pool, not a conversion of the entire roughly 1 GB story training corpus.

The starting vocabulary-A checkpoint was pretrained and fine-tuned on mixed-case
text. This experiment is a lowercase continuation, not a claim that its whole
training history was lowercase.

The browser worker now lowercases a **copy** of model-facing system and all history
text. The page retains visitors' original messages and names as typed. No model
weights are replaced and nothing is deployed by this code change. Generated fish
text is displayed as generated, so the old bundled model is not forced to output
lowercase by a cosmetic postprocessor.

Python generation supports `--text-case lowercase` for matching evaluation input.
Reports record the transformation; paired comparisons reject mismatched case
conditions. Frozen evaluation files remain available unchanged, with separate
lowercase development/historical/compression views for this training run. Their
losses must not be compared directly with previous mixed-case evaluation losses.

## Short run

Initialize from `runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt`.
Train 600 steps with uniform dialogue sampling, learning rate 0.00005, warmup 60,
batch 32, sequence length 512, seed 42 and the existing 15% story-example mixture.
Evaluate the planned final checkpoint, not a checkpoint picked using conversation
outputs. Eight familiar three-turn fish scenes compare the starting and continued
models under the same lowercased full context, temperature 0.7, top-k 40 and seeds.
These previously inspected scenes are an exploratory character review, not an
independent benchmark or a statistical preference ranking.

Five new regression tests cover original-text immutability, Python/browser Unicode
normalization parity, story boundaries, carry-forward deduplication and prevention
of mixed-condition comparisons. The existing 21-test v5.2 suite also passes.
Full-data checks confirm lowercase model-facing text, presence of all 15 carried
conversations and byte-identical preserved evaluation files.

## Reproduction

```sh
uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python -m tools.aquarium.v52.lowercase \
  --base runs/v52/contract-responsive-v2-reviewed \
  --out runs/v52/contract-lowercase-v1-reviewed \
  --data /Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq \
  --carry experiments/v52/training-library/carry-forward-original.jsonl
mkdir -p runs/v52/lowercase-screen
zsh experiments/v52/results/lowercase-screen/run.sh
```

The builder refuses to overwrite an existing candidate. An earlier local draft
`contract-lowercase-v1` is superseded by the reviewed path because its audit counted
already-present carry-forward examples as additions; both versions' dialogue content
is the same. The draft is preserved with a supersession note.

Scripts and results are kept in [results/lowercase-screen](results/lowercase-screen/).
The training library receives a separate sample and recipe for this run, preserving
its predecessors without replacing their archives.

## Completed short-run observations

The run completed 600 steps in about 145 training-loop seconds, plus preparation
and evaluation. Lowercase development loss fell from 3.27616 to 1.81198; lowercase
story loss fell from 3.03563 to 1.70268 on this replay pool. The final checkpoint
was also the development-selected best checkpoint. These measure adaptation to
lowercased data and cannot establish a quality win against earlier mixed-case runs.

All 24 sampled continuation replies were naturally lowercase, without output
postprocessing. The fish still drifts and invents explanations. Some exchanges
are pleasant: the quiet shared-visit scene ends with “we will share the quiet part
of the tank and listen for the waves.” The excited bridge scene acknowledges the
bridge and sticks, although it loses track of the toy mouse. The snail shop and
fish-librarian scenes still wander. This is acceptable evidence that the lowercase
pipeline works and a useful character sample, not proof of a general improvement.

Read the [before/after conversations](results/lowercase-screen/conversations.md).
Both conditions normalize the system and full history, unlike the earlier
user-input-only comparison. No isolated causal claim is made: the continuation
also changes the inherited dialogue mixture and story replay pool.

The [training library](training-library/README.md) now includes 1,504 rows and the
recipe from this ninth run, indexed by its new dataset hash. All older snapshots
are byte-identical. This run's full dataset and checkpoints remain locally under
`runs/v52/contract-lowercase-v1-reviewed` and `runs/v52/lowercase-screen/model`.

The web normalization code is ready in the PR. No deployment, quantization,
browser-weight replacement, or longer training run was performed.
