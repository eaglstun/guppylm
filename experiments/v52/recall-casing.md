# Recall and casing investigation

The two completed full-run checkpoints show two separable weaknesses in these
exploratory diagnostics: sensitivity to case, and failure to answer a question even
when the stated detail influences the model's answer probabilities. No model,
training data, tokenizer, or browser behavior was changed in this investigation.

## Controlled design

Eight fact/question templates cover colors, names, location, number and invented
word meanings. Each has two alternate facts, making 16 prompts per condition.
The same facts are tested normally cased, uppercase and lowercase. Conditions put
the question in the same user message (direct), after a fixed assistant “Okay.”
(neutral), or after the model's own greedy reply (self). A fact-absent control
measures answer preference without evidence. Generation is greedy, capped at 80
tokens. Both previous vocabulary-A and revised-v2 best checkpoints run on CPU.
These are small, related exploratory probes, not independent benchmark trials.
They differ from the earlier sampled probes; counts must not be pooled.

A follow-up converts only the uppercase fact to sentence case using `capitalize()`.
This loses proper-name capitalization and is an experimental transform, not an app
fix. It also scores both candidate complete answers using mean per-token negative
log likelihood. This teacher-forced measure supplies answer wording; it measures
conditional preference, not spontaneous correct answering. Both alternate facts
are tested to distinguish factual sensitivity from always favoring one answer.

## Findings

| Revised v2, condition | Generated detail hits / 16 | Correct complete-answer preference / 16 |
|---|---:|---:|
| Normal, direct | 7 | 15 |
| Uppercase, direct | 2 | 9 |
| Lowercase, direct | 2 | 15 |
| Sentence-normalized, direct | 5 | 15 |
| Normal, neutral intervening reply | 12 | 15 |
| Uppercase, neutral intervening reply | 1 | 8 |
| Sentence-normalized, neutral intervening reply | 9 | 15 |
| Normal, own intervening reply | 13 | 16 |
| Uppercase, own intervening reply | 1 | 8 |
| Sentence-normalized, own intervening reply | 10 | 16 |

For normal direct prompts, complete-answer preference switches correctly for both
alternate facts in 7/8 pairs, versus 1/8 uppercase and 7/8 sentence-normalized.
The fact-absent control gets 8/16 by favoring one answer in each pair and switches
correctly in 0/8. Thus normally cased facts affect conditional answer probabilities;
poor generation is not equivalent to total failure to represent the fact.

The previous checkpoint has the same pattern: normal direct generation 4/16 and
complete-answer preference 15/16; uppercase direct 1/16 and 8/16 respectively.
This tiny probe set favors revised v2 on some normal-case formats, but does not
reverse its established regression on the larger frozen real-noun development set.

For example, the revised model responds to both orange and purple scarf facts plus
a same-message question with a scarf definition and fish introduction. Separating
the question with “Okay.” produces the correct orange/purple answer. Likewise,
wooden-box colors are ignored in direct replies and recovered after the separate
turn. This suggests learned dialogue/response patterns are part of the problem.
It does not establish the cause of the broader real-noun regression.

### Casing and exposure

`The scarf on my toy bear is orange.` tokenizes into 9 tokens; its uppercase form
uses 28, mostly letter fragments. Tokenization preserves case rather than silently
normalizing it. All diagnostic prompts fit: maximum original prompt length is
150 tokens, well below the 432-token prompt allowance for an 80-token response.
Context truncation therefore does not explain these failures. Fragmentation plus
sparse uppercase exposure is a plausible explanation, not an isolated causal proof.

The previous full-run dataset has 640 rows with an uppercase user turn among
142,518 rows (0.45%). Revised v2 has 380 among 131,994 (0.29%): 64 responsive pairs
and 316 Talkie counterparts. Increased conversational variety did not increase
uppercase exposure. Both full runs use uniform dialogue sampling.

### Plumbing checks

Training and Python evaluation use the same `chat_ids` framing. A concrete
assistant-target-mask check exactly matches the expected reply tokens plus end
marker. Full, cached-prefix and incremental-cache logits agree within 0.000014
on both checkpoints. This rules out an obvious masking/cache mismatch in the
checked example; it is not an exhaustive backend or browser parity test.

The initial diagnostic also ranked bare answer strings. That measure was dominated
by response-form preferences even for successfully generated replies. Its raw
results are retained, but the table above uses the follow-up's natural complete
answers. This scoring correction is explicit and no generated replies were rerolled.

## Next controlled experiment

Use a small matched continuation with two separately measured interventions:

1. Paired direct-question and separate-turn recall training, with alternate facts
   that force different answers. Responses should answer the actual question first,
   then allow a brief relevant playful addition.
2. Case-paired versions of those same examples, with an audited sampling share.
   Compare a recall-only arm against recall-plus-case at equal steps and initialization.

Keep new validation nouns/names, prompt structures and meanings separate from the
training examples. Track factual correctness, inappropriate definitions, format
sensitivity and uppercase performance rather than development loss alone. Do not
silently lowercase all user input: this diagnostic shows incomplete recovery and
proper-name damage, and it does not test case-sensitive content.

No further training was launched. The existing data and checkpoints are preserved.

## Reproduction

Run from the repository root. The scripts use the existing local checkpoint and
tokenizer paths recorded in the artifacts and refuse to overwrite probe outputs.

```sh
uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python -m experiments.v52.results.recall-casing.probe
uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python -m experiments.v52.results.recall-casing.supplement
python3 experiments/v52/results/recall-casing/summarize.py
```

[Summary, scripts, hashes and raw outputs](results/recall-casing/) retain both the
initial and follow-up diagnostics. Local runtime outputs are in
`runs/v52/recall-casing`.
