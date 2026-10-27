# Response-first data revision

This experiment follows the unsuccessful 5% vocabulary / 5% uppercase screen.
The goal is relevance and coherent play, rather than expanding a vocabulary checklist.

## What the audit found

The 142,592-row input contains 55,374 haiku-source, 55,302 base, 15,220 song,
14,305 family, 1,605 Talkie, 640 uppercase, and 146 contextual vocabulary rows.
The name `haiku` is a source label; many of those rows are ordinary conversations.

Lexical flags identified 4,587 assistant turns combining unfamiliarity with guessed
explanations, and 13,389 name-prefixed replies outside a simple greeting/name-query
screen. These are **review candidates, not counts of bad answers**. Some guesses
are reasonable analogies; introductions can be appropriate.

An initial multiline screen also misclassified requested chants and shanties.
After including those request words, only 45 rows lacked explicit poetry cues:
40 song rows and five variants of an ordinary multiline conversation. Review
preserved the multiline conversation and quarantined the eight song parents
whose ordinary questions elicited unsolicited verse.

Five additional reviewed parents contain clear failures, such as guessing that a
bicycle bell is a shell, treating a big fish as an unknown word, suggesting brushing
a jellyfish, or ignoring sadness while inventing definitions. The hash-pinned
review decisions are in `rewrites/responsive_quarantine_v1.json`. Quarantine follows
parent IDs so augmented descendants do not silently reintroduce the same problems.
This is a small reviewed cleanup, not a semantic audit of the whole corpus.

## Revision

`python -m tools.aquarium.v52.responsive` creates a new candidate-A manifest:

- Quarantine 13 reviewed parents and their descendants: 69 rows total.
- Remove the 786 earlier experimental uppercase/vocabulary rows from this candidate.
  They remain preserved in the earlier versioned datasets.
- Strip only exact standalone `I am NAME.` / `NAME here.` preambles when the
  conversation contains no greeting or identity question. Preserve the rest of the
  response literally. This changes 3,828 turns across 3,813 rows; it does not remove
  species descriptions or broad `I am ...` sentences. All edits are logged.
- Add 40 authored two-turn scenarios, each paired with an uppercase-first-prompt
  version: 80 rows, sharing groups across casing variants. Follow-up prompts and
  answers are identical across each pair. There are no 20-persona template copies.

The scenarios include four each for celebration, boundaries, urgent tank problems,
and ambiguous calls, plus eight each for pretend-versus-real distinctions,
following stated details, and everyday archaic/belief contexts. Definitions of
complex political concepts are not a priority in this revision.

The new manifest has **141,817 rows**. Original train manifests and all evaluation
files remain unchanged. The builder validates reserved entities, evaluation-group
and question overlap, reviewed-content hashes, and all casing pairs. Output is
staged and published only after validation.

```sh
python -m tools.aquarium.v52.responsive \
  --base runs/v52/contract-context-v2 \
  --out runs/v52/contract-responsive-v1
```

## Comparison design

Two 600-step continuations use the same completed vocabulary-A checkpoint,
revised dataset, seed 42, learning rate 0.00005, warmup 60, and MPS device. The
only recipe difference is the sampler: uniform versus `--source-share responsive=.02`.
This is a 2% share of dialogue draws, not 2% of target tokens or loss. Existing
story slots remain unchanged. Final-step weights are evaluated in both arms.

Evaluation includes the frozen 320-prompt development contract, 28 previously
inspected exploratory prompts plus eight fresh prompts, and eight fresh two-turn
scenarios in both normal and uppercase versions (two samples per variant/model).
The latter checks conversation follow-through and includes narrow literal-answer
checks for stated facts. Literal matches are not an aggregate quality metric.
Prior exploratory prompts informed the revision and are not independent tests.
No locked-test prompts are used.

These short continuations retain prior learning from the old corpus. They measure
adaptation under the revision, not the effect of replacing the original training
corpus from the start. The uniform-versus-weighted comparison tests weighting on
the same cleaned data; it does not isolate each cleanup operation's contribution.

## Results

Both continuations completed 600 steps. The four new regression tests pass,
covering paired casing/group identity, narrow intro trimming with identity/greeting
preservation, descendant quarantine, byte-identical evaluation files, stale review
rejection, and no published output after validation failure.

| Check | Starting checkpoint | Cleanup + uniform | Cleanup + 2% share |
|---|---:|---:|---:|
| Development loss | 1.76345 | 1.76121 | 1.76199 |
| Real-noun hits | 11.67% | 13.33% | 11.94% |
| Invented-noun hits | 0% | 1.25% | 0.42% |
| Name hits | 55.00% | 53.33% | 50.00% |
| Memory hits | 30.00% | 30.00% | 25.00% |
| Revised-example draws during the screen | n/a | 8 | 331 |

Uniform sampling's real-noun difference from the starting checkpoint is +1.67
percentage points, with paired prompt bootstrap 95% interval [-0.28, +3.89].
Weighting's difference from uniform is -1.39 points, interval [-3.89, +0.83].
All reported paired score-difference intervals include zero; these screens do not
establish a general behavioral improvement or a statistically clear regression.
The weighted examples supplied 10,267 target tokens, **0.340%** of all sampled
target tokens including stories, despite receiving 2% of dialogue draws.

On the fresh two-turn probes, narrow stated-fact matches were 3/10 starting,
5/10 uniform, and 3/10 weighted in normal case. All three models scored 0/10 with
uppercase first prompts. These are five prompts with two samples each, far too
small for a general quality claim. A scoring issue was corrected after inspection:
mentioning `table` counted as a literal hit even when the relation was wrong.
The corrected location check requires `under the table` or `beneath the table`,
applied identically to every model without resampling. Raw entity matches and the
separate corrected results are retained for audit. Even corrected word matches
are not a score of overall coherence.

Qualitative failures remain substantial: irrelevant poems, made-up definitions,
wrong colors, loss of information after uppercase messages, and confusion about
what real fish can do. Some weighted uppercase replies became less fluent rather
than more responsive. There is no dependable improvement in celebration, urgency,
or respectful handling of differing beliefs from these short continuations.

**Decision:** retain the reviewed data revision as an experimental candidate and
prefer the unweighted recipe for further investigation. Do not promote either
checkpoint or launch a longer weighted run on this evidence. A short continuation
cannot establish how a model trained from the start on cleaned data would behave.

The input audit, complete edit/quarantine trail, raw generated replies, corrected
fact checks, paired intervals, recipes, and probe scripts are saved in
[results/responsive-screen](results/responsive-screen/). All checkpoints and frozen
manifests remain available locally under `runs/v52/`.
