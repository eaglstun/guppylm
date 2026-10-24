# Aquarium in the browser

The aquarium fish (8.9M parameters) as a 9 MB int8 file, run by plain JavaScript in a Web Worker.
No runtime library, no server: `guppy.js` is the tokenizer, the model (with a KV cache), and the sampler.

    python3 tools/aquarium/export_web.py runs/aq-chat-v4/best.pt --out web/model
    cd web && python3 -m http.server 8765    # then open http://127.0.0.1:8765

- Weights: matrices and both embeddings as int8 with a float32 scale per row; biases and LayerNorms in float32.
  Held-out persona loss, float vs int8: 2.1227 vs 2.1230 (`aq-chat-v2`), 2.0550 vs 2.0562 (`aq-chat-v3`), 2.0468 vs 2.0470 (`aq-chat-v4`).
- Parity: the JS tokenizer matches `tokenizers` on 752 strings (Chinese, emoji, accents included), and greedy
  decoding matches PyTorch on the int8-rounded checkpoint token for token.
- Speed: about 5 to 6 ms per token in Node on an M4 Max, including the prompt.
- Long chats keep the last 432 tokens of the conversation, leaving room for an 80-token reply.

Live at https://eaglstun.github.io/guppylm/ (the `gh-pages` branch is `git subtree split --prefix web` of this one).
