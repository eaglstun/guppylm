# Aquarium in the browser

The [live Aquarium page](https://eaglstun.github.io/guppylm/) runs the 8.9M-parameter
**Aquarium v5.2 lowercase** fish in a Web Worker. The 9,042,320-byte (9.04 MB)
model uses conventional per-row int8 weights. Inference stays in the browser.

## Current model: lowercase continuation

This is the completed 600-step lowercase continuation of vocabulary-A, using
131,618 lowercase dialogue rows and a small lowercase TinyStories replay pool.
It retains reviewed examples from earlier experiments. User messages remain as
typed on screen; a copy of system and conversation history is lowercased for the
model. The fish is playful and imperfect; this release makes no general quality
claim over earlier checkpoints. See [training and conversation samples](https://github.com/eaglstun/guppylm/blob/dev/experiments/v52/lowercase-screen.md).

[release.json](model/release.json) records the exact checkpoint, model and tokenizer
hashes. Six browser-runtime checks matched the int8-rounded Python checkpoint's
prompt tokenization and 24 greedy output tokens per prompt, with maximum initial
logit difference 0.0000151. A Node-adapted worker check over local HTTP verified
loading, inference, lowercase output for the test prompt, and unchanged visitor
text. No interactive browser was available for this release check; phone timing
has not been measured.

```sh
python3 tools/aquarium/export_web.py runs/v52/lowercase-screen/model/final.pt \
  --tokenizer runs/v52/contract-lowercase-v1-reviewed/data/tokenizer.json --out web/model
cd web
python3 -m http.server 8765
```

The published site is served from the `gh-pages` branch. The previous d4 decoder
and its published v5.1 binary remain available there for rollback; the new worker
loads `guppy.bin?v=5.2-lowercase-600`. Runtime and worker URLs are also versioned.
The source lives in `dev` and PR #5; publishing the site does not merge that PR.

## Previous release: v5.1

V5.1 completed 15,000 chat fine-tuning steps with four noun-swap copies per eligible
dialogue. Its float checkpoint scored 2.0690 held-out persona loss in the
[quantization study](https://github.com/eaglstun/guppylm/blob/dev/QUANTIZATION.md). A 9.04 MB int8 export was followed by a
3.61 MB d4-compressed page release. Those are historical results, not scores of
the current lowercase model.

## Earlier browser model: Aquarium v4

The v4 checkpoint was trained in three stages: TinyStoriesV2-GPT4 pretraining
(`aq-pre`), 2,000 steps on TinyDialogues and ocean conversations (`aq-td`), and
15,000 steps of sea-creature chat fine-tuning (`aq-chat-v4`). Both chat stages kept
a 15% mix of TinyStories windows. See [the Aquarium experiment](https://github.com/eaglstun/guppylm/blob/dev/EXPERIMENT.md)
for the training and behavior results.

### What the v4 int8 browser experiment showed

Matrices and both embeddings use int8 weights with one float32 scale per row;
biases and LayerNorms stay float32. These held-out persona losses show the
quality change from rounding the same checkpoint to int8:

| Checkpoint | Float loss | Int8 loss |
| --- | ---: | ---: |
| `aq-chat-v2` | 2.1227 | 2.1230 |
| `aq-chat-v3` | 2.0550 | 2.0562 |
| `aq-chat-v4` | 2.0468 | 2.0470 |

For the v4 export, the JavaScript tokenizer matched `tokenizers` on 752 strings,
including Chinese, emoji, and accents. Greedy decoding matched PyTorch token
for token on its int8-rounded checkpoint. In Node on an M4 Max, generation took
about 5–6 ms per token, including the prompt. Long chats retain the last 432
conversation tokens, leaving room for an 80-token reply.

**Finding:** int8 made v4 practical to serve as a single browser file with
negligible measured loss. These v4 timing and parity checks were run locally;
they do not establish v5.1 or phone WebView speed.

## Experimental compressed reader: eight-weight codes

`guppy_e8p.js` reads `model/guppy_e8.bin` (2,495,360 bytes) or
`model/guppy_e8_g128.bin` (2,590,080 bytes) as 16-bit codes for eight-weight
chunks. It retains compressed weights in memory, using a 512 KiB lookup table
and fast Hadamard transforms. The tracked binaries came from a v5
noun-continuation experiment. The live page now loads the lowercase int8 file through
`guppy.js`; this reader is a standalone experiment.

Run `node tools/aquarium/quant/bench_e8p_runtime.mjs` from the repository root
to compare a compressed payload with an int8 export of the **same decoded
model**. On Node 24/M4 Max, a 102-token prompt followed by 80 generated tokens
gave:

| Same-model weight path | Prompt | 80 generation steps | Counted storage after 182 tokens |
| --- | ---: | ---: | ---: |
| Int8 re-export | 465–469 ms | 662–676 ms | 18.53 MB |
| Direct 2.46 MB E8 | 539 ms | 713 ms | 13.23 MB |
| Direct 2.55 MB E8 g128 | 604 ms | 807 ms | 13.52 MB |

Direct JavaScript logits differed from the decoded Python model by at most
1.34e-5 over 32 sequential tokens. Counted storage includes the weight buffer,
lookup table, side tensors, scratch space, cached embeddings, and the same
9.44 MB K/V cache on both paths. It excludes JavaScript object and allocator
overhead.

**Finding:** keeping E8 codes compressed saved about 5 MB of counted memory
against an int8 export of the same model. In that paired run, generation was
about 8% slower for per-row scales and 19% slower for g128 scales. Absolute
Node timings varied across reruns, so compare codecs within one benchmark run.

### Confirmation with the completed v5.1 checkpoint

A separate v5.1 E8 g128 export produced another 2,590,080-byte browser binary.
Its direct-code runtime matched Python reference logits across 32 tokens
within 1.62e-5. In a paired Node 24/M4 Max run, 80 generated tokens took
407.0 ms (196.6 tokens/s) with E8 versus 363.4 ms (220.1 tokens/s) with an
int8 export of that same decoded model. Counted storage after 182 token IDs
was 13.52 MB versus 18.53 MB. That compressed E8 browser binary was a
validation export; that earlier page used a separate int8 export of v5.1.

The v5.1 result repeats the memory saving with a generation slowdown. Phone
WebView timing remains unmeasured. [Quantization notes](https://github.com/eaglstun/guppylm/blob/dev/QUANTIZATION.md)
give the quality probes, size accounting, and the other codecs tested.
