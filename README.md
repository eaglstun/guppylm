<p align="center">
  <img src="assets/guppy.png" alt="GuppyLM" width="400"/>
</p>

<h1 align="center">GuppyLM</h1>
<p align="center"><em>Two ~9M parameter fish: the original Guppy and Aquarium v4.</em></p>

<p align="center">
  <a href="https://huggingface.co/datasets/arman-bd/guppylm-60k-generic"><img src="https://img.shields.io/badge/🤗_Dataset-guppylm--60k-blue" alt="Dataset"/></a>&nbsp;
  <a href="https://huggingface.co/arman-bd/guppylm-9M"><img src="https://img.shields.io/badge/🤗_Model-guppylm--9M-orange" alt="Model"/></a>&nbsp;
  <a href="https://github.com/arman-bd/guppylm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
  <br/>
  <a href="https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb"><img src="https://img.shields.io/badge/Train_in-Colab-F9AB00?logo=googlecolab" alt="Train"/></a>&nbsp;
  <a href="https://colab.research.google.com/github/arman-bd/guppylm/blob/main/use_guppylm.ipynb"><img src="https://img.shields.io/badge/Chat_in-Colab-F9AB00?logo=googlecolab" alt="Chat"/></a>
  <br/>
  <a href="https://www.linkedin.com/pulse/build-your-own-language-model-5-minutes-i-made-mine-hossain--supif/"><img src="https://img.shields.io/badge/Article-LinkedIn-0A66C2?logo=linkedin" alt="LinkedIn Article"/></a>&nbsp;
  <a href="https://arman-bd.medium.com/build-your-own-llm-in-5-minutes-i-made-mine-talk-like-a-fish-e20c338a3d14"><img src="https://img.shields.io/badge/Article-Medium-000000?logo=medium" alt="Medium Article"/></a>
  <br/><br/>
  <a href="https://eaglstun.github.io/guppylm/"><img src="https://img.shields.io/badge/Try_Aquarium_v4-in_Browser-64ffda?style=for-the-badge&logo=webassembly&logoColor=white" alt="Aquarium v4 browser demo"/></a>
</p>

---

> **This project exists to show that training your own language model is not magic.**
> No PhD required. No massive GPU cluster. One Colab notebook, 5 minutes, and you have a working LLM that you built from scratch — data generation, tokenizer, model architecture, training loop, and inference. If you can run a notebook, you can train a language model.
>
> It won't produce a billion-parameter model that writes essays. But it will show you exactly how every piece works — from raw text to trained weights to generated output — so the big models stop feeling like black boxes.

---

```
You> hi guppy
Guppy> hi there. i just found a nice spot near the rock. the temperature feels nice.

You> are you hungry
Guppy> yes. always yes. i will swim to the top right now. i promise to eat all of it.

You> do you like bubbles
Guppy> i love bubbles. they make the water feel slightly different.

You> what is the meaning of life
Guppy> food. the answer is always food.

You> tell me a joke
Guppy> what did the fish say when it hit the wall. dam.

You> do you love me
Guppy> you're my favorite big shape. my mouth are happy when you're here.

You> goodnight guppy
Guppy> ok sleep time. i was following a bubble but now i'll stop. goodnight tank. goodnight water.
```

---

## What is GuppyLM?

The original GuppyLM is a tiny language model that pretends to be a fish named Guppy. It speaks in short, lowercase sentences about water, food, light, and tank life. It doesn't understand human abstractions like money, phones, or politics — and it's not trying to.

The original model trains from scratch on 60K synthetic, single-turn conversations across 60 topics in about five minutes on a GPU. This fork also includes **Aquarium v4**: a similarly sized model pretrained on stories, then trained on multi-turn conversations with 18 aquatic personas (16 in training, two held out). It has a longer context and runs in the [fork's browser demo](https://eaglstun.github.io/guppylm/).

The original notebooks and Hugging Face model remain linked below. Aquarium v4's browser weights are in `web/model/`; its training checkpoints and dialogue datasets are not committed.

---

## Original Guppy architecture

|                  |                             |
| ---------------- | --------------------------- |
| **Parameters**   | 8.7M                        |
| **Layers**       | 6                           |
| **Hidden dim**   | 384                         |
| **Heads**        | 6                           |
| **FFN**          | 768 (ReLU)                  |
| **Vocab**        | 4,096 (BPE)                 |
| **Max sequence** | 128 tokens                  |
| **Norm**         | LayerNorm                   |
| **Position**     | Learned embeddings          |
| **LM head**      | Weight-tied with embeddings |

Vanilla transformer. No GQA, no RoPE, no SwiGLU, no early exit. As simple as it gets.

Aquarium v4 keeps the six-layer, 384-wide architecture at 8.9M parameters. Its byte-level BPE tokenizer retains capitals, punctuation, and non-ASCII text, and its context grows from 128 to 512 tokens. See [EXPERIMENT.md](EXPERIMENT.md) for the earlier aquarium runs and measured probes.

---

## Original Guppy personality

Guppy:

- Speaks in short, lowercase sentences
- Experiences the world through water, temperature, light, vibrations, and food
- Doesn't understand human abstractions
- Is friendly, curious, and a little dumb
- Thinks about food a lot

**60 topics:** greetings, feelings, temperature, food, light, water, tank, noise, night, loneliness, bubbles, glass, reflection, breathing, swimming, colors, taste, plants, filter, algae, snails, scared, excited, bored, curious, happy, tired, outside, cats, rain, seasons, music, visitors, children, meaning of life, time, memory, dreams, size, future, past, name, weather, sleep, friends, jokes, fear, love, age, intelligence, health, singing, TV, and more.

---

## Quick Start

### Try in Browser (no install needed)

[![Try Aquarium v4 in Browser](https://img.shields.io/badge/Try_Aquarium_v4-in_Browser-64ffda?logo=webassembly)](https://eaglstun.github.io/guppylm/)

Aquarium v4 runs entirely in your browser: a 9 MB int8 model, a plain-JavaScript runtime, and a Web Worker. No server or API key handles inference. Pick a creature and chat across turns. The [original Guppy browser demo](https://arman-bd.github.io/guppylm/) remains available upstream; it uses a separate ONNX/WebAssembly model.

To serve this fork's browser build locally:

```bash
cd web
python3 -m http.server 8765   # open http://127.0.0.1:8765
```

See [web/README.md](web/README.md) for the export command and runtime details.

### Chat with Guppy in Colab

[![Open in Colab](https://img.shields.io/badge/Chat_in-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/arman-bd/guppylm/blob/main/use_guppylm.ipynb)

Downloads the pre-trained model from HuggingFace and lets you chat. Just run all cells.

### Train your own

[![Open in Colab](https://img.shields.io/badge/Train_in-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb)

1. Set runtime to **T4 GPU**
2. **Run all cells** — downloads dataset, trains tokenizer, trains model, tests it
3. Upload to HuggingFace or download locally

### Chat locally

```bash
pip install torch tokenizers
python -m guppylm chat
```

```
You> the cat is looking at you
Guppy> i hide behind the plant when the furry one comes.

You> it is raining outside
Guppy> i think rain is the best thing about outside.
```

In interactive chat mode, the conversation grows and quickly runs into the 128-token limit, reducing quality.
You can also invoke chat with a single prompt, and exit after the response:

```bash
python -m guppylm chat --prompt "tell me a joke"
```

---

## Original Guppy dataset

**[arman-bd/guppylm-60k-generic](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic)** on HuggingFace.

|            |                                                        |
| ---------- | ------------------------------------------------------ |
| Samples    | 60,000 (57K train / 3K test)                           |
| Format     | `{"input": "...", "output": "...", "category": "..."}` |
| Categories | 60                                                     |
| Generation | Synthetic template composition                         |

```python
from datasets import load_dataset
ds = load_dataset("arman-bd/guppylm-60k-generic")
print(ds["train"][0])
# {'input': 'hi guppy', 'output': 'hello. the water is nice today.', 'category': 'greeting'}
```

The published 60K dataset is useful for learning the training pipeline, but its original generator had less variety than its comments claimed. In a 60K generation, **19.4% of complete replies were unique**, and **77% of evaluation input/output pairs also appeared in training**. Many template details were filled once when the module loaded, then reused. This fork fixes per-sample filling, composes two templates for half the replies, and removes exact train duplicates from evaluation. These fixes raise unique complete replies to **42.6%** in the measured regeneration; they do not make the original model a general-purpose chatbot. See [MEMORIZATION_FIX.md](MEMORIZATION_FIX.md) for the measurements and limitations.

Aquarium v4 uses a different data path: TinyStories pretraining followed by generated multi-turn persona conversations, with varied names and unfamiliar nouns to teach copying. Its training tools are under `tools/aquarium/`. The exported model is ready to use in `web/`, while regenerating a training run requires the separate story and dialogue data described in [EXPERIMENT.md](EXPERIMENT.md).

The v4 run adds TinyDialogues and a first stage of ocean conversations, with a 20,000-dialogue training cap. The exported int8 model's held-out persona loss was **2.0470**, compared with **2.0468** for its float checkpoint. Those are measurements for this run, not a claim that the fish understands every new name or remembers every conversation. Export details and parity checks are in [web/README.md](web/README.md).

---

## Project Structure

See [experiments/](experiments/README.md) for the model-size, data-diet, and
compression measurements and their reusable scripts.

```
guppylm/
├── config.py               Hyperparameters (model + training)
├── model.py                Vanilla transformer
├── dataset.py              Data loading + batching
├── train.py                Training loop (cosine LR, AMP)
├── generate_data.py        Conversation data generator (60 topics)
├── eval_cases.py           Held-out test cases
├── prepare_data.py         Data prep + tokenizer training
├── human_things.py         Expanded human-noun pool for experiments
└── inference.py            Chat interface

tools/
├── make_colab.py           Generates Colab notebooks
├── aquarium/               Aquarium data, training, probes, and web export
├── export_onnx.py          Export model to ONNX (quantized uint8)
├── export_dataset.py       Push dataset to HuggingFace
└── dataset_card.md         HuggingFace dataset README

web/
├── index.html              Aquarium browser demo
├── guppy.js                Tokenizer, model runtime, KV cache, and sampler
├── worker.js               Inference worker
└── model/                 Aquarium v4 int8 weights, tokenizer, personas

docs/
├── index.html              Original Guppy browser demo (ONNX + WASM)
├── download.sh             Download model.onnx + tokenizer from HF
├── model.onnx              Quantized uint8 (~10 MB)
├── tokenizer.json          BPE tokenizer
└── guppy.png               Logo (transparent)
```

---

## Design Decisions

**Why no system prompt in the original Guppy?** Every training sample had the same character, so the prompt was redundant. Aquarium v4 varies the system prompt and demonstrates that a model of similar size can respond to different persona descriptions.

**Why is the original Guppy single-turn?** Its 128-token context limits longer chats. Aquarium v4 uses a 512-token context and multi-turn training.

**Why vanilla transformer?** GQA, SwiGLU, RoPE, and early exit add complexity that doesn't help at 9M params. Standard attention + ReLU FFN + LayerNorm produces the same quality with simpler code.

**Why synthetic data?** The original templates make every training example inspectable and let you trace behavior back to the curriculum. They also cap diversity: the original generation reused far more replies than expected. Aquarium v4 tests a different curriculum with varied dialogue rather than relying on a small set of finished sentences.

---

## License

MIT

<!-- GitAds-Verify: UW3VONXNDRGEZ2O7RF4N2PFHSUR3DUJB -->
