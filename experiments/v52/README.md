# Aquarium v5.2: implementation and first screening result

Latest work: [revision v2 training and conversational checks](responsive-v2-results.md), following
[reviewed training-data revision v2](data-revision-v2.md),
[response-first data revision](responsive-data.md),
[context diversity and sampling weights](context-sampling.md) and
completion and behavioral checks of the first [contextual vocabulary](talkie-vocabulary.md)
run. That run included the [uppercase examples](uppercase.md). The earlier
[full-run snapshot](full-runs.md) and screening report below record their original
inspection stages; their incomplete/not-yet-trained statuses are historical.

Worktree: `/private/tmp/guppylm-v52-impl`, branch `implement/v5_2`.
Latest main was fetched and merged through **6dc7248**, including the earlier
**15484ff** findings review and the subsequent v5.1 browser deployment update. The frozen preparation,
baseline evaluation, and two continuation screens ran from implementation revision
**70e357d**. Subsequent changes tighten manifest/tokenizer checks, test access,
reproducibility metadata, and static checks; they do not change the measured weights.

**Decision: do not promote either screen.** Both development-loss selections retained
the unchanged step-0 checkpoint. The v5.1 continuation improved the real-noun slice,
but missed the invented-noun improvement threshold and the name non-regression floor.
The v5 continuation did not improve real-noun copying. No locked-test outputs were
sampled, no codec was selected, and no v5.2 browser model was deployed.
The v5.1 browser update from main was incorporated separately.

## Validation of the draft plan

The completed v5/v5.1 logs confirm the draft's architecture, initial checkpoint,
15,000-step budget, augmentation counts, and historical losses. The new scorer
reproduces the old v5.1 historical mean-batch loss **2.06905855**; its token-weighted
value is **2.07258847**. Keep these metrics labelled separately.

The latest [main findings](../README.md), [training report](../../EXPERIMENT.md),
[data audit](../data.md), [size study](../size.md), [persona study](../persona.md),
and [compression report](../../QUANTIZATION.md) support these decisions:

- Keep the 8,873,472-parameter model. The original small-model success concerned a
  different, repetitive dataset; Aquarium layer pruning damaged copying and memory.
- New splits of existing conversations are prospective development data, **not unseen
  tests of v5/v5.1**. Those checkpoints already trained on much of this material.
- Source proportions must include target tokens. In the v5.1 screen, 2,000 story rows
  out of 12,800 total rows supplied **1,024,000 of 1,994,396 target tokens (51.3%)**.
  The nominal 15% story setting rounds to 5/32 rows, or 15.625%, and is not a 15% loss
  weight. The added copy source supplied 549/10,800 dialogue rows (5.1%) but only
  12,573/970,396 dialogue target tokens (1.3%). This is evidence to consider for a
  separate balance experiment, not a reason to silently change matched A/B recipes.
- Do not choose a codec on an earlier model's ranking. On completed v5.1, tuned d4
  had the stronger quality point at **estimated** 3,568,128 bytes. E8 has a working
  direct-code browser reader; the hybrid has a real ZIP but no browser reader.
  Neither the old Mac timing nor a payload estimate supplies a phone deployment result.
- The earlier 25% noun continuation hurt persona/name scores. The present bounded,
  5% hand-authored reference intervention is deliberately a **screen**, not a claim
  that four templates replace varied, reviewed generated conversations.

## What is implemented

`python -m tools.aquarium.v52.prepare` creates hash-checked A/B/C train manifests,
model-development, separate compression-development, and historical manifests.
Exact full-row duplicates and no-op derived rows are removed. Entity-normalized
conversation groups are split before capping and augmentation; children retain
parent and group IDs. Each augmentation type has its own RNG stream, so A/B differ
in noun copies without unintentionally changing memory/name examples.

The builder hashes inputs and generator code, records source/line provenance and
rejected rows, reserves test entities, checks exact probe-question collisions after
augmentation, and audits repeated turns, entity-normalized wording and eight-word
assistant fragments. These lexical checks are **not semantic deduplication**.
Historical generator versions cannot be recovered from unversioned local data;
current hashes freeze what is available now, not what an old run necessarily saw.

`train` consumes only explicit train/development manifests. It preserves the initial
architecture and records actual sampled rows, input tokens and target tokens by source.
Every saved checkpoint includes optimizer state, warmup/cosine schedule position,
Python/NumPy/CPU/MPS RNG state, remaining shuffled row order, best score and history.
Writes are atomic. `--stop-after` preserves the original schedule for resume;
`--resume` rejects changed data, recipe, code, tokenizer, runtime or device. CPU
interrupted/uninterrupted equivalence is tested bit-for-bit, including dropout,
stories, optimizer, history and best checkpoint. MPS training was exercised, but
bitwise MPS resume equivalence has not been established.

`evaluate` uses the frozen contract, three samples per prompt, cached generation,
and whole normalized phrase matching. It reports per-prompt bootstrap intervals
with a Wilson envelope to avoid a false [0,0] certainty after zero successes.
`compare` computes paired prompt differences. Prompts share wording families;
these intervals do not measure uncertainty over arbitrary unseen language.
Manual slices remain unscored until reviewed. Test evaluation requires a selection
record bound to the checkpoint and actual hashed development/rubric artifacts.
That guard records selection; it does not replace human review or phone checks.

`score` computes assistant-token-weighted loss per source and a fixed story sample,
while retaining the historical mean-batch metric. The quantization tools accept
explicit training and compression-development manifests for calibration **and**
code-frozen distillation. They reject model-development/test manifests as compression
development and record calibration IDs. Legacy invocations retain their old behavior.

## Frozen data and evaluation

See [audit.json](audit.json), [contract.json](contract.json), and [rubric.json](rubric.json).
The rubric was written before either continuation started.

| Material                                                | Rows/prompts |
| ------------------------------------------------------- | -----------: |
| Base conversations after cap                            |       20,000 |
| Train song/family extras                                |        6,028 |
| A: two noun swaps, after deduplication                  |      141,221 |
| B: four noun swaps, after deduplication                 |      152,282 |
| C: A plus bounded hand-authored reference conversations |      148,605 |
| Model development                                       |        1,231 |
| Compression development                                 |          309 |
| Historical development                                  |          300 |
| Behavioral development / locked test                    |    320 / 320 |

Each behavioral split contains 120 real-noun, 80 invented-noun, and 20 each of name,
memory, missing-memory, voice, song/family, and held-persona prompts. Duplicate complete
prompts were removed. There are 960 sampled replies per evaluated checkpoint.
Tokenization, decoding and prompt lists are fixed; invented names use letters, not
numeric suffixes. A draft family question collided with training and was replaced
before freezing this contract. Draft outputs in earlier local run directories are
not part of these results.

Across the pre-cap grouped sources plus historical data, the audit found 351 repeated
turn strings, 545 entity-normalized wording strings, and 4,101 eight-word response
fragments crossing splits. Examples include family questions and stock responses.
There were no exact probe-question collisions in the frozen augmented training sets.
This does not establish the absence of semantic paraphrases. The full TinyStories
pretraining text has not been screened for entity exposure. The persona probe is
held out from fine-tuning, not demonstrably from all pretraining.

## Measured screens

Both runs: 400 steps, lr 3e-5, warmup 50, batch 32, sequence 512, dropout .1,
nominal story mix .15, training seed 42; identical C manifest, separate starts from
the completed v5 and v5.1. These are **not** the planned 15,000-step A/B/C comparisons.
The table evaluates each step-400 checkpoint to expose the intervention, even though
loss selected step 0. It is not a checkpoint-selection result.

| Checkpoint      | Real noun | Invented noun |  Name | Memory | Historical token NLL | Story NLL |
| --------------- | --------: | ------------: | ----: | -----: | -------------------: | --------: |
| v5 baseline     |     .1139 |         .0000 | .5000 |  .2333 |              2.06835 |   1.58524 |
| v5 + C screen   |     .0944 |         .0292 | .5333 |  .3833 |              2.06679 |   1.58437 |
| v5.1 baseline   |     .0750 |         .0042 | .5500 |  .3833 |              2.07259 |   1.58464 |
| v5.1 + C screen |     .1278 |         .0167 | .5000 |  .3833 |              2.06880 |   1.58351 |

The v5.1 real-noun gain is +.0528, paired prompt-bootstrap 95% interval
[+.0167, +.0917]. Invented nouns improve only +.0125 [0, +.0292], below the
predeclared +.05 requirement; the name score also misses the .52 floor. The name
change's interval crosses zero, so this is a failed conservative decision rule,
not proof of a population-level regression. Neither screen supplies grounds for
opening the locked test or spending time tuning a new codec.

Both the constant-answer and train-only lexical-retrieval controls score zero on
these exact-copy metrics. That checks a narrow failure mode of the scorer; it does
not make copying a substitute for coherent voice or correct memory abstention.

Full sampled outputs are archived as deterministic `.json.gz` files in [results](results),
alongside histories, recipes, paired statistics, artifact hashes and a 160-item
[blind review](results/blind-review.json). Its key is separate. No human scores have
been filled in. The complete frozen manifests and resumable checkpoints remain local:

- `runs/v52/contract-v1/`
- `runs/v52/screen-v5-C/{final,best,inference}.pt`
- `runs/v52/screen-v51-C/{final,best,inference}.pt`

`inference.pt` contains only float weights/config/step and provenance; `final.pt`
also includes optimizer and resume state. Checkpoint hashes are in the result summary.

## Second data pass: contract-v3 and talkie visitors

Train rows only; `contract.json` and the development, historical and compression
splits are byte/row-identical to contract-v1 (`frozen_eval` in each audit).

- **Curation v2** ([audit-v2.json](audit-v2.json)): caps of 40 rows per repeated
  assistant 8-gram ("i do not know that word": 349 → 143 rows) and per repeated user
  turn ("do you have any friends": 121 → 40); 141 surplus "Misty" rows renamed
  (it is listed twice in `personas.NAMES`); 5 looping family rows removed.
- **Curation v3** ([audit-v3.json](audit-v3.json)): 923 hand-written rewrites of
  assistant turns that recite the system-prompt trait list ("I am Duke, and I am
  huge, slow, and kind" → "I am Duke. I am as long as a bus, I move like a cloud…").
  The detector hit 924 turns in 916 parents; one scenic false positive is left.
- **talkie source** ([talkie/](talkie)): 316 conversations in which
  `postpostmodern/talkie-1930` (pre-1931 text) plays a whimsical 1930s visitor and the
  fish replies are authored in-session by Claude. Lines were filtered lexically, respelled
  to American forms (`respell.py`), screened by Haiku agents for content and for
  hidden-visitor-record leaks, and adjudicated in
  `tools/aquarium/v52/rewrites/talkie_whimsy_v1_review.json`. That review overturned all
  seven predator-diet/whimsy rejects, added a "You are talking with …" clause to 37 system
  prompts, replaced 51 turns and truncated 9 conversations. It is 0.76% of A's
  target tokens. Fish replies come from a single author and share its tics.

`contract-v3-talkie` adds the talkie rows through `--extra-source`. Doing so is **not a
pure addition**. The capped family selection shifts by 48 parents, 7 trait rewrites lose
their parent, and the per-stream augmentation draws shift, so about 34k augmented
A rows differ from contract-v3. A v3-vs-v3-talkie comparison therefore confounds talkie
with resampling noise of roughly seed scale. A/B/C within one contract remain matched.

## Reproduction and remaining gates

See [commands.md](commands.md) for exact preparation, screening, full-run, resume,
evaluation, and compression configuration. For exact reproduction/resume of the
archived screens, use source revision **70e357d** and the recorded versions. Later
code intentionally rejects a mismatched training-code fingerprint.

The next defensible experiment needs reviewed, genuinely varied dialogue generation,
a broader long-conversation test, and an exposure audit extending through pretraining.
The four hand-template C source is a reproducible screening intervention only.
The full A/B/C equal-budget runs, source-balance D, confirmation seed 73, human review,
locked-test scoring, post-selection codec sweep, and phone measurements were **not run**.
No v5.2 quality or compression improvement is claimed. Latest main now ships the
conventional v5.1 int8 browser export; the screened checkpoints do not replace it.

Validation: nine tests pass, including cached inference, immutable manifests,
parent/group isolation, token-weighted loss, exact CPU resume, train-only compression
calibration, tokenizer compatibility, and locked-test artifact binding. Static checks
pass for the new tooling. No deployment or publication was performed.
