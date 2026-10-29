# Training library: keep the history and the fish

This library preserves samples from all eight local v5.2 training runs in this
conversation, spanning six distinct datasets. Runs sharing a dataset retain their
separate recipes, initialization fingerprints and sampling weights in [index.json](index.json).
The full original manifests and checkpoints remain unchanged under `runs/v52/`.

The six hash-named gzip JSONL files contain 9,684 row memberships across snapshots.
Selection retains every distinctive experimental add-on source (vocabulary,
uppercase, responsive, explicit word meaning and recall), every original Talkie
base conversation, and up to two deterministic rows per remaining
source/augmentation/persona bucket. Exact row IDs, parents, groups and provenance
are preserved. File hashes and original dataset hashes allow verification.
Historical preservation is not endorsement: even examples later removed remain
available for inspection, rather than silently disappearing from the history.

## Carry forward

[carry-forward-original.jsonl](carry-forward-original.jsonl) is a 15-conversation,
reviewed seed selection, representing every run through its dataset membership.
It keeps requested songs, creature personality, gentle companionship, a cloud
library, a carrot baton, playful vocabulary, and a small amount of direct recall.
Reasons and run membership are in [carry-forward-review.json](carry-forward-review.json).
This is a starting selection for future mixtures, not a complete dataset or a
chosen sampling weight. It should not be used as an instruction to discard the
rest of the underlying corpus.

[carry-forward-lowercase.jsonl](carry-forward-lowercase.jsonl) is an explicitly
versioned lowercase view of the same selection. Only system and conversation
content change; parent/group lineage and `case_transform.from_uid` link back to
originals. Preserve original archives when making further normalized variants.
Some selected historical system prompts still contain older persona restrictions;
harmonize those when constructing a future full corpus rather than rewriting history.

Both JSONL views contain training data only. Evaluation replies from the fish
review are stored separately and have not been added to this bank.

## Reproduce

From the repository root, using the original local manifests:

```sh
python3 -m experiments.v52.results.fish-conversation-review.preserve
python3 -m experiments.v52.results.fish-conversation-review.retain
```

The scripts refuse existing output rather than replacing this preserved library.
