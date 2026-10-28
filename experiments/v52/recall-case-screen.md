# Matched recall and casing continuation screen

This experiment follows the [recall/casing investigation](recall-casing.md).
Both arms start from the revised-v2 best checkpoint and keep its 131,994-row
base corpus intact. This tests adaptation from that checkpoint, not clean training
from pretraining, and does not establish which earlier cleanup caused regressions.

## Frozen design

Twenty-four controlled context families cover color, name, location, number,
explicit invented meanings and corrections. Each has two alternate facts and
two question placements: same-message direct question or a separate question
after a neutral “Okay.” assistant reply. This yields 96 recall rows.

- **Recall:** 96 normally cased rows.
- **Paired:** the same 96 rows plus uppercase-first-user counterparts, with all
  assistant targets and later turns identical.

Both use `recall_screen=.05` (5% of dialogue draws), 600 steps, learning rate
0.00005, warmup 60, seed 42, MPS, batch 32, 512-token context and the existing
15% story-example mixture. No other sampling shares change. Fixed final-step
checkpoints are compared, without selecting on these new validation outputs.
The paired arm splits new-source exposure equally across cases; it does not
increase the source's overall probability. Token/target exposure is audited.

Twelve separate validation families use different wording, objects, names and
invented meanings from the add-on templates. Two facts, two formats and two
casings yield 96 validation conversations, sampled twice each with temperature
0.7, top-k 40 and 80-token reply limit. Format/casing variants share seeds.
Separate-turn probes use the supplied neutral assistant reply, not a generated
first reply. Common words may already occur in inherited training/pretraining;
these are new held-out add-on families, not a claim of novel vocabulary to the model.

All added and validation rows fit without truncation and contain assistant targets.
Parent/group/dialogue isolation, reserved-entity exclusion, exact validation-question
absence in inherited training, and case-pair identity are checked during preparation.
The contract and three evaluation manifests are copied byte-identically.

Primary comparison: paired versus recall on uppercase unambiguous literal-answer
hits (expected value present, alternate value absent). Normal-case hits and the
existing 320-prompt development contract are regression checks. Paired bootstrap
intervals cluster by the 12 validation families, not by casing/fact/sample variants.
Literal hits do not certify coherence; qualitative inspection accompanies scores.
No locked-test access or automatic promotion is part of this screen.

## Reproduction

The scripts in [results/recall-case-screen](results/recall-case-screen/) define
training examples, frozen validation, generation and the exact run commands.

```sh
uv run --offline --python 3.10 --with tokenizers==0.23.3 python -m experiments.v52.results.recall-case-screen.prepare
zsh experiments/v52/results/recall-case-screen/run.sh
```

Runtime artifacts are under `runs/v52/recall-case-screen`. Preparation and generation
refuse existing output. Starting checkpoint: `runs/v52/full-responsive-v2-A/best.pt`.

## Completed results

Both arms completed 600 steps. Exactly 816 new-source examples and 9,698 new-source
target tokens were sampled in each, 0.327% of all sampled target tokens including
stories. Input tokens differ as expected from uppercase fragmentation. Equal source
probability therefore yielded equal target-token exposure for this matched design.
Final development losses were 1.76579 (recall) and 1.76582 (paired), versus 1.76508
at initialization. Development-selected best checkpoints remained at step 0; the
planned evaluations use final-step checkpoints.

| New validation literal hits | Starting | Recall | Paired |
|---|---:|---:|---:|
| Normal, direct | 2/48 | 10/48 | 6/48 |
| Normal, separate question | 8/48 | 9/48 | 8/48 |
| Normal total | 10/96 | 19/96 | 14/96 |
| Uppercase total | 0/96 | 0/96 | 0/96 |

Recall's normal-case literal-hit gain is +9.38 percentage points, with family-cluster
bootstrap 95% interval [+2.08, +17.71]. Paired minus recall is -5.21 points, interval
[-12.50, +1.04]. These are intervals across 12 related validation families for a
single training seed, not confirmation across training runs. All-zero uppercase
samples yield degenerate [0,0] bootstrap differences; this does not prove no possible
effect or establish precise population accuracy.

**Literal hits overstate answer quality.** Unblinded inspection of all 43 positive
samples found wrong referents, negated relations and malformed words. For example,
“Your name is Tessa” incorrectly names the visitor when asked about a plush otter;
“Your name is a velvet mitten” fails to recall the invented definition despite
containing the expected phrase. At least eight recall-arm positives and seven
paired-arm positives are clear wrong-referent, missing-definition or negated-relation
failures. The review is archived separately and does not replace the frozen metric
with a selectively rescored accuracy number. There are real isolated successes,
such as correctly naming the plush otter and stating the chosen green paint.

| Existing development literal hits | Starting | Recall | Paired |
|---|---:|---:|---:|
| Real nouns | 3.61% | 4.72% | 4.72% |
| Invented nouns | 0% | 0.42% | 0.42% |
| Names | 51.67% | 48.33% | 45.00% |
| Memory | 31.67% | 25.00% | 28.33% |

All paired development difference intervals include or touch zero. There is no
clear broad improvement, and name/memory point estimates move downward.

**Decision: do not promote either continuation or launch a full run.** This exposure
level did not repair uppercase handling. Direct normal-case response patterns show
a limited signal, but semantic correctness remains poor. Before spending another
full-run budget, the useful next diagnostic is a small learnability/overfit check
with substantially greater answer-target exposure and semantic relation checks,
followed by genuinely separate validation. That has not been run here.

Raw generation, positive-hit review, paired intervals, both training recipes/logs,
additions and the frozen validation manifest are archived alongside the scripts.
No production model or browser behavior changed.
