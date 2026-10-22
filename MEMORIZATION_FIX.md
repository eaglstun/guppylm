# Memorization fix: what changed, what it did

Branch `fix/memorization`, off `dev`. Measurement tools cherry-picked from `exp/data`
(`--train-limit`, `tools/data_analysis.py`, `tools/memorization.py`). See
`exp/data`'s EXPERIMENT.md for the diagnosis.

## Changes

1. **Per-sample template slots** (`1cf1d9b`). The 45 `_topic()` generators passed their
   reply templates as list literals, so `{pick(...)}` ran once at import. Templates are
   now lambdas, rebuilt per sample.
2. **Eval drops train duplicates** (`1cf1d9b`). Eval pairs that also occur in train are
   removed. Prompts still overlap: topics have 4 or 5 user messages each, so a
   prompt-disjoint split isn't possible with this generator.
3. **Two-template composition** (`93d2f37`). Half of topic samples append a second,
   different template from the same topic.

## Data (fresh 60K generation, measured)

|                                      |             original | + per-sample slots | + composition |
| ------------------------------------ | -------------------: | -----------------: | ------------: |
| unique outputs                       |                19.4% |              24.2% |         42.6% |
| unique train inputs                  |                  812 |                713 |           714 |
| eval pairs kept                      | 3,000 (77% in train) |                919 |         1,567 |
| lowest topic (`name`) unique outputs |                   10 |                 10 |           100 |

The 812 vs 714 input difference is the 100 hand-written `big_water` lines at the end of
the main checkout's `train.jsonl`, which have no generator code. A regeneration drops
them; they were not carried into this branch's data.

## Model (run `fix-composed`, default config, 10k steps, one seed, measured)

Eval loss 0.471 vs baseline 0.385, but the eval set is now harder (no train duplicates),
so the two numbers are not comparable. Keyword hit rate 0.863 (5 samples per case).

`tools/memorization.py`, final checkpoint, 3 samples per prompt, same seed as `exp/data`.
Verbatim is measured against each model's own train set.

| set                            | metric                               | baseline | fixed |
| ------------------------------ | ------------------------------------ | -------: | ----: |
| novel paraphrases (90 replies) | reply verbatim from train            |     0.52 |  0.41 |
|                                | every sentence from train            |     0.87 |  0.95 |
|                                | falls into "i only know fish things" |     0.50 |  0.38 |
| eval_jsonl (180 replies)       | reply verbatim from train            |     0.69 |  0.68 |
|                                | sentence in train                    |     0.99 |  0.99 |
| copy_probe (90 replies)        | names the new thing                  |     0.00 |  0.00 |

## Findings

- **Measured:** the data bug is fixed and variety more than doubled, but the model's
  behavior barely moved. Sentence-level reuse is unchanged or higher; the differences in
  the table are within the noise of 90 to 180 replies.
- **Inferred:** this is the ceiling of template data. A template generator can only teach
  its templates; more combinations just means more combinations of memorized sentences.
  The fish still never echoes a new noun, because the confused template only ever pairs
  with the 31 nouns in `HUMAN_THINGS`, so "repeat the user's word" and "say one of these
  31 words" look identical to the model.

## Next, if wanted

- Expand `HUMAN_THINGS` from 31 to a few thousand nouns, and test whether the copy skill
  appears. That's the one place the data could teach composition rather than recall.
- Paraphrase user prompts (714 distinct inputs is the real limit on new phrasings).
