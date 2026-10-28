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
