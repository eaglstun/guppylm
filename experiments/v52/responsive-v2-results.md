# Revised-data full run: conversational checks

The 131,994-row reviewed v2 candidate completed 15,000 steps from the original
`aq-td` checkpoint with uniform dialogue sampling, seed 42 and the previous full-run
schedule. Training took about 65 minutes. Development loss was 1.76508 versus
1.76345 for the earlier vocabulary-A full run. The final checkpoint was selected
by development loss.

**Decision: do not promote this checkpoint.** Similar loss concealed substantially
worse real-noun recall and persistent conversational failures.

| Frozen development check | Previous vocabulary A | Revised v2 |
|---|---:|---:|
| Real-noun literal hits | 11.67% | 3.61% |
| Invented-noun literal hits | 0% | 0% |
| Name hits | 55.00% | 51.67% |
| Memory hits | 30.00% | 31.67% |

All 320 development prompts were evaluated with the frozen decoding and seeds.
Real-noun change is -8.06 percentage points, with paired prompt bootstrap 95%
interval [-11.94, -4.72]. Name and memory difference intervals include zero.
These are prompt-level uncertainty estimates for this single training seed, not
estimates across training runs. Multiple data changes occurred together; this
comparison cannot attribute the regression to any one change. No locked test ran.

The reused two-turn exploratory probes yielded 2/10 normal-case fact matches versus
3/10 for vocabulary A, and 0/10 uppercase for both. Location scoring requires
“under the table” or “beneath the table”, following the previous scoring correction.
These probes were already inspected during data development and are not held-out.
The separate 36-prompt targeted check used three samples per prompt. Qualitative
inspection of sample 0 across all prompts showed continued irrelevant answers,
made-up definitions, missed requests, and poor handling of uppercase input.

Examples from the revised checkpoint:

- An explicitly orange scarf became “brown and white”.
- A drum placed beneath the table moved “on the table” or “behind the table”.
- Asked whether a real fish could drive without water, it answered “Yes!”
- It correctly recalled Rowan in both normal-case samples, but corrupted the name
  after an uppercase first message.
- Asked whether it could live in a dry pencil case, it claimed it already did.

Four fresh two-turn word-definition diagnostics were authored after training and
before sampling, with normal/uppercase forms and two samples each. Revised v2
matched the supplied meaning in 2/8 normal-case samples versus 0/8 previously;
both scored 0/8 uppercase. This tiny exploratory signal does not offset the wider
failures. In one successful example it recalled that “zembalo” meant a striped
umbrella; abstract meanings and corrected definitions still failed.

The next investigation should isolate recall and input casing before further
broad corpus edits or another full run. Preserve both checkpoints and datasets.

[Raw replies, comparison intervals, scripts, training recipe and log](results/responsive-v2-checks/)
are archived for review. Runtime artifacts remain under
`runs/v52/checks-responsive-v2` and `runs/v52/full-responsive-v2-A`.
