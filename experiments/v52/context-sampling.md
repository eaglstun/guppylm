# Context diversity and source sampling

The first vocabulary run sampled its 72 added rows only 204 times in 15,000 steps.
Loss stayed stable, but exploratory prompts still produced irrelevant answers,
unsolicited poems, and generic uppercase acknowledgments.

## New data

`tools/aquarium/v52/rewrites/talkie_context_v2.jsonl` adds 74 distinct authored
conversations: 32 archaic, 22 belief/history/politics, 12 uppercase, and 8 playful
or physically grounded scenarios. These supplement the original 72 vocabulary
rows and 640 uppercase rows; they do not replace them.

Answers address the request directly. Imaginary animals can have shops and clothes;
real fish still need water. Uppercase examples distinguish celebration, boundaries,
urgent aquarium problems, and ambiguous calls. Religious and political examples
use ordinary meanings and acknowledge differing beliefs.

Build a fresh candidate-A manifest with:

```sh
python -m tools.aquarium.v52.append_context \
  --base runs/v52/contract-v3-talkie-uppercase-v2-vocabulary-v1 \
  --out runs/v52/contract-context-v2
```

The builder rejects incomplete imports, reserved entities, frozen probe questions,
and evaluation-group overlap. It records hashes and copies the contract and
all evaluation manifests byte-for-byte. Existing training rows remain unchanged.
The output contains 142,592 rows, including 146 `talkie_vocabulary` rows.

## Explicit sampling shares

The trainer accepts repeated `--source-share SOURCE=FRACTION` options. For example:

```sh
--source-share talkie_vocabulary=.05 --source-share uppercase=.05
```

Each source gets that fraction of **dialogue draws**, sampled with replacement and
uniformly within the source. The remaining probability is uniform over all other
dialogue rows. Story examples retain their separate fixed batch slots. These are
not loss weights or fractions of target tokens. The context source includes its
12 new uppercase situations as well as vocabulary and playful scenarios.

With the default batch of 32 and story mix of .15, each batch has 27 dialogue and
5 story examples. Over 600 steps the expected counts are 810 contextual vocabulary
and 810 uppercase draws, versus about 17 and 73 respectively under uniform sampling.
Over a hypothetical 15,000-step run, each weighted source would receive about 20,250
draws. This increased repetition can overfit, so a short screen precedes any full run.

Without `--source-share`, the historical shuffled-pass sampler remains in use.
Unknown sources, duplicates, non-finite or invalid shares, and allocations without
remaining background probability fail early. Shares are recorded in the recipe;
source example and token counts are recorded during training. Weighted sampling
uses the saved Python RNG state, with exact CPU resume equivalence covered by tests.

## Matched screen

Two 600-step continuations start from the completed vocabulary-A best checkpoint
and use the same expanded manifest, seed 42, MPS device, learning rate 0.00005,
60-step warmup, story mixture, and evaluations every 300 steps. One uses uniform
sampling; the other uses the two 5% shares above. This compares the sampling
policies, including their different handling of replacement and background draws.

Run artifacts are under `runs/v52/context-screen/`. `run.sh` records the commands.
`targeted.py` freezes 18 previously inspected development prompts plus 10 fresh
prompts before inspecting screen outputs, with three matched seeds per prompt.
It compares the starting model and both final-step checkpoints. The final-step
comparison measures adaptation even if generic development loss selects step zero.
These probes are exploratory development checks, not independent test scores.
The locked test remains unopened.

## Screen results

Both runs completed all 600 steps. All 21 v5.2 tests pass, including both default
and weighted CPU resume equivalence, source probability accounting, invalid-share
rejection, and frozen-manifest preservation/leakage guards.

| Measurement | Uniform | Weighted |
|---|---:|---:|
| Context vocabulary draws | 13 | 816 |
| Uppercase draws | 61 | 798 |
| Context vocabulary share of actual target tokens, including stories | 0.0078% | 0.590% |
| Uppercase share of actual target tokens, including stories | 0.0533% | 0.690% |
| Final development loss | 1.76044 | 1.76288 |
| Real-noun exact-phrase hits | 11.94% | 8.61% |
| Invented-noun exact-phrase hits | 0% | 0% |
| Name hits | 53.33% | 50.00% |
| Memory hits | 33.33% | 28.33% |

Frozen behavioral evaluation used 320 development prompts and three samples per
prompt for each model. Weighted-minus-uniform real-noun copying is **-3.33 percentage
points**, with a paired prompt bootstrap 95% interval of **[-5.83, -1.11]** points.
This is evidence of regression on these development probes. Name and memory
comparisons include zero in their intervals. The generic development loss alone
would have concealed these issues.

Qualitative inspection of the 28 exploratory prompts (three samples each per model)
finds occasional relevant prayer/quiet-company replies, but persistent failures:
uppercase excitement still produces generic waiting/moving replies; urgent tank
problems do not reliably lead to useful help; definitions of political vocabulary
remain confused; the fish sometimes agrees to leave its water. Archaic requests
can still elicit unrelated poems. These observations are not a blind human review
or an aggregate quality score.

The sampling mechanism delivers the intended exposure, but this particular short
continuation does not establish the desired behavior. **Do not promote it or treat
it as justification for a full run.** Further experiments should address the
remaining content/behavior imbalance and evaluate both targeted relevance and the
copying regression. No locked-test prompts were sampled.

Machine-readable recipes, sampling counts, paired intervals, and compressed raw
outputs are in [results/context-screen](results/context-screen/). Local run and
checkpoint artifacts remain under `runs/v52/context-screen/`.
