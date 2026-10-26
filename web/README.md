# Aquarium in the browser

The aquarium fish (8.9M parameters) as a 9 MB int8 file, run by plain JavaScript in a Web Worker.
No runtime library, no server: `guppy.js` is the tokenizer, the model (with a KV cache), and the sampler.

The shipped v4 checkpoint follows three stages: TinyStoriesV2-GPT4 pretraining (`aq-pre`),
2,000 steps on TinyDialogues and ocean conversations (`aq-td`), then 15,000 steps of
sea-creature chat fine-tuning (`aq-chat-v4`). Both chat stages kept a 15% mix of
TinyStories windows.

    python3 tools/aquarium/export_web.py runs/aq-chat-v4/best.pt --out web/model
    cd web && python3 -m http.server 8765    # then open http://127.0.0.1:8765

- Weights: matrices and both embeddings as int8 with a float32 scale per row; biases and LayerNorms in float32.
  Held-out persona loss, float vs int8: 2.1227 vs 2.1230 (`aq-chat-v2`), 2.0550 vs 2.0562 (`aq-chat-v3`), 2.0468 vs 2.0470 (`aq-chat-v4`).
- Parity: the JS tokenizer matches `tokenizers` on 752 strings (Chinese, emoji, accents included), and greedy
  decoding matches PyTorch on the int8-rounded checkpoint token for token.
- Speed: about 5 to 6 ms per token in Node on an M4 Max, including the prompt.
- Long chats keep the last 432 tokens of the conversation, leaving room for an 80-token reply.

Live at https://eaglstun.github.io/guppylm/ (the `gh-pages` branch is `git subtree split --prefix web` of this one).

## Experimental compressed weights

`guppy_e8p.js` reads `model/guppy_e8.bin` (2.50 MB) or `model/guppy_e8_g128.bin`
(2.59 MB) directly as 16-bit codes for eight-weight chunks. It keeps the weights compressed
in memory, with a 512 KiB lookup table and fast Hadamard transforms. These binaries come
from the v5 noun-continuation experiment; the live page still uses the v4 int8 model above.

`node tools/aquarium/quant/bench_e8p_runtime.mjs` from the repository root compares the
compressed reader with an int8 export of the same model. The compressed reader matches the
decoded Python model's logits to within 1.4e-5. On an M4 Max, it uses about 5 MB less
counted resident memory after a 182-token run, with generation about 8% slower for the
2.50 MB version and 19% slower for the 2.59 MB version. Phone WebView performance has
not been measured. See [the quantization notes](../QUANTIZATION.md) for quality probes and
the exact size accounting.
