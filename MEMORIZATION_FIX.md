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

## Follow-up: 1,146 nouns (run `fix-nouns`, measured)

`HUMAN_THINGS` moved to `guppylm/human_things.py`: the original 31 plus about 1,100
everyday nouns, with the copy probe's 10 test nouns and close variants kept out.
Regenerated data: 941 confused train samples over 1,146 nouns, so most nouns appear
once or never. Unique train inputs went from 714 to 1,358. Same config, 10k steps,
one seed. Eval loss 0.481. The probe used the same settings as above.

| set               | metric                    | fixed (31 nouns) | fix-nouns |
| ----------------- | ------------------------- | ---------------: | --------: |
| copy_probe (new)  | names the thing           |             0.00 |      0.00 |
|                   | has a never-seen sentence |             0.20 |      0.82 |
| copy_known (31)   | names the thing           |             0.57 |      0.02 |
| novel paraphrases | sentence in train         |             0.95 |      0.78 |

Sample replies to `do you know what a toaster is`-style prompts:
"i don't know what a busy is", "i don't understand spreadsheets",
"i have no idea what mountain pan means".

- **Measured:** copying did not appear. The model learned that a noun goes in the slot
  and fills it with a noun it saw in training, not the noun in the prompt. It now fails
  on the 31 known nouns too, since none of them is common enough to memorize anymore.
  The jump in never-seen sentences is wrong nouns in familiar templates, not new prose.
- **Inferred:** at 9M params, 10k steps, and about 940 examples (1.6% of the data),
  the copy skill isn't learned; the model takes the cheaper route of matching the
  noun distribution. Untested next steps: raise the confused share, train longer, or
  check greedy decoding (the probe samples at temperature 0.7, top_k 50).

## Follow-up 2: greedy decoding, 3x training, 10x confused share (measured)

New options: `tools/memorization.py --data-dir/--temperature/--top-k` (the probe used to
read `data/` unconditionally, so an older run must be pointed at its own saved data:
`runs/fix-composed/data`, `runs/fix-nouns/data`), and
`python -m guppylm prepare --confused-weight N`. `data/` now holds the weight-10 data.

| run                         | eval loss | new noun named | known noun named | new noun, partial |
| --------------------------- | --------: | -------------: | ---------------: | ----------------: |
| fix-composed, greedy        |     0.471 |           0.00 |             0.93 |              0/30 |
| fix-nouns, sampled          |     0.481 |           0.00 |             0.02 |              0/90 |
| fix-nouns, greedy           |           |           0.00 |             0.00 |              1/30 |
| fix-nouns-30k (30k steps)   |     0.482 |           0.00 |             0.00 |              0/90 |
| fix-nouns-cw10, sampled     |     0.535 |           0.00 |             0.53 |              7/90 |
| fix-nouns-cw10, greedy      |           |           0.00 |             0.80 |              4/30 |

Eval loss is per run's own eval set; cw10's eval is 14% confused, so it isn't comparable.
"Partial" = the reply contains the first 4 letters of a word (4+ letters) of the test noun.

- **Greedy:** sampling wasn't hiding a copy skill. Ruled out.
- **3x steps:** eval loss flat (0.481 to 0.482); slot fills get stranger
  ("is a surprise surprise something that lives in water"). Ruled out.
- **10x confused share** (8,266 train samples, about 7 per noun): known-noun copying
  comes back (0.80 greedy) and two test nouns get half-copied: "basket" for basketball,
  "a vacuum" for vacuum cleaner. Both overlap words in the training list ("basket",
  "vacuuming"). No test noun with no training overlap was ever copied.
- **Inferred:** the model copies a word from the prompt only if that word already
  appeared in the reply slot in training. That is association between familiar tokens,
  not a general copy operation. A general one would need either far more copy data or
  an inductive bias this small model doesn't pick up in this setup.
