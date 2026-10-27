# Contextual Talkie vocabulary

Talkie v2 removes the unconditional `archaic` and `religion_or_politics`
blocks. Old-fashioned phrasing, beliefs, history, and politics can appear in
understandable conversations. Playfulness is welcome when the scene makes sense.
The other content, repetition, persona, and evaluation-leakage checks remain.

`tools/aquarium/v52/rewrites/talkie_vocabulary_v1.jsonl` contains 72 authored
visitor/guppy conversations. Each of the 37 former archaic alternatives and 23
religion/politics vocabulary targets occurs in at least three distinct rows.
For old regex stems such as `christian\w*`, coverage uses the base word rather
than claiming to cover every possible suffix. The target list is in
`talkie_vocabulary_targets.json`.

Examples include toy-crab parades, a paper crown left in the rain, quiet prayer,
different family beliefs, civil disagreement, and simple historical distinctions.
Religious figures and mild exclamations are handled in context, without requiring
the fish or visitor to share a belief. There are no remaining unconditional bans
from either removed category.

The existing rare-word vocabulary includes these authored rows. Explicit coverage
words are exempt from the tokenizer-fragmentation limit, so long spellings such as
communism and bolshevik do not silently recreate the removed block. Other words
still receive the normal rare-word and fragmentation checks.

Include the source in a **new** preparation output alongside the other extra sources:

```sh
--extra-source talkie_vocabulary=tools/aquarium/v52/rewrites/talkie_vocabulary_v1.jsonl
```

The extra-source loader marks these rows train-only and applies its usual duplicate,
reserved-entity, and evaluation-group checks. Existing frozen manifests, checkpoints,
and running training jobs are not changed. These additions have not been trained or
behaviorally evaluated. Coverage tests count whole words/phrases, validate every turn
against the remaining filters, and retain regression checks for harmful content and
reserved evaluation entities.
