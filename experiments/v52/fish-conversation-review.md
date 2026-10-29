# Reviewing a 9M fish as a fish

The objective is an enjoyable little character: responds enough to the visitor,
keeps some personality, and makes its silliness understandable. Perfect recall,
expert explanations and general-assistant performance are not the target. Prior
benchmark results remain useful diagnostics, but are not the sole product standard.

## Conversation review

Eight scenes cover greeting, a snail hat shop, showing a toy frog's crown, rain,
quiet company, a fish librarian, an excited bridge-builder, and friends with
different beliefs. Four checkpoints each receive original and lowercased user
turns, yielding 64 conversations / 192 replies. Each conversation has three turns,
with generated assistant replies retained in subsequent context. Temperature 0.7,
top-k 40, 80-token maximum; scene/turn seeds match across conditions. System prompts
and model weights are unchanged. This is one exploratory sample per condition,
reviewed by the agent, not a blinded human preference study or a statistical ranking.

Read the [actual conversations](results/fish-conversation-review/conversations.md).

What works at fish scale:

- “The glass is like a big glass” is charmingly literal, not a reason to reject a fish.
- “My fins are very proud to help you find this hat” is odd but fits the playful scene.
- “Then stay on the glass, and sway with me” gives quiet companionship without
  needing a sophisticated answer.
- The paired checkpoint's original-case respectful scene stays simple and pleasant:
  it offers to sit quietly, then says it would like the shared visit.

What can spoil the fun:

- Calling the visitor “not funny at all” during a quiet-company exchange.
- Defining a snail shop as an unrelated shell instead of joining the pretend game.
- Losing the current topic repeatedly or answering with a run of empty acknowledgments.

The recall-only checkpoint gives a bare “Okay.” in 4/24 replies for each casing
condition, versus none for the previous and revised full checkpoints in these
samples. That suggests a personality tradeoff worth watching; it does not prove
which training example caused it. Its training add-on includes neutral “Okay.”
acknowledgments, so more varied, relevant acknowledgments are worth considering.
Do not select that checkpoint solely because it mentions more expected words.
No checkpoint was promoted by this subjective review.

## Lowercase direction

Lowercase input makes the excited bridge opener recognizable in all four checkpoints.
Original uppercase responses mostly miss the bridge entirely. The three user
messages shrink from [34, 25, 30] to [13, 8, 11] tokenizer tokens. However, follow-ups
still drift, and normalization sometimes worsens other scenes, including quiet company.
This supports a **consistent lowercase-training experiment**, not a claim that
an input-only transform has solved the problem.

For that experiment, lowercase the model-facing system, user and assistant training
text consistently, and apply the same normalization when forming inference context.
Keep the visitor's original text for display and preserve original training archives.
Output would naturally be lowercase too. Keep the current tokenizer initially so
we isolate case handling; replacing it would also change embeddings/pretraining
compatibility. A new tokenizer is not required merely to lowercase text.

This turn created a lowercase view of the reviewed carry-forward selection, but
**did not retrain a model, rewrite the full corpus, or change browser input handling**.
The present comparison lowercases user turns only, so it does not measure a model
trained on exclusively lowercase conversations. Pretraining text would also need
separate consideration before claiming the entire training history was lowercase.

## Preserve what we tried

The [training library](training-library/README.md) now keeps samples of all eight
local runs, indexed by six immutable dataset hashes, and all their distinctive
experimental additions. Fifteen reviewed carry-forward rows represent every run.
Both original and lowercase views are retained. Full manifests remain unchanged;
new evaluation conversations are not silently recycled into training.

This favors incremental mixtures with personality intact over repeatedly replacing
one narrow behavior dataset with another. The historical bank includes discarded
material for audit, while the reviewed seed selection identifies examples worth
carrying forward rather than blindly restoring all old augmentation.

## Reproduction

```sh
uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python -m experiments.v52.results.fish-conversation-review.probe
```

The script refuses existing runtime output. Seeds, prompts, original user text,
normalized text, model/tokenizer hashes and raw replies are archived in
[results/fish-conversation-review](results/fish-conversation-review/).
