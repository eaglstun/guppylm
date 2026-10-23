"""Tokenizer and token files for the aquarium experiment.

    python3 tools/aquarium/prep.py tokenizer     # BPE on TinyStories sample + dialogues so far
    python3 tools/aquarium/prep.py stories       # TinyStories -> data_aq/ts_{train,valid}.bin (uint16)

Unlike the original tokenizer, this one starts from the full byte alphabet, so
capitals, punctuation and non-ASCII reach the model instead of being dropped.
"""

import glob
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

D = "data_aq"
TOK = os.path.join(D, "tokenizer.json")
VOCAB = 4096
SPECIAL = ["<pad>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]  # ids 0..3
EOT = 3


def chat_text(system, turns):
    out = f"<|im_start|>system\n{system}<|im_end|>\n"
    for t in turns:
        out += f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n"
    return out


def stories(path):
    """TinyStoriesV2 files separate stories with <|endoftext|> lines."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                if buf:
                    yield "".join(buf).strip()
                buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf).strip()


def dialogue_texts():
    for path in glob.glob(os.path.join(D, "dialogues*.jsonl")):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                yield chat_text(d["system"], d["turns"])


def build_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)

    def corpus():
        for i, s in enumerate(stories(os.path.join(D, "tinystories", "train.txt"))):
            if i >= 300_000:
                break
            yield s
        for _ in range(3):  # weight the chat format and sea words up a little
            yield from dialogue_texts()

    tok.train_from_iterator(corpus(), trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.save(TOK)
    for t in ["Hello GUPPY! What's 2+2?", "你好", "<|im_start|>assistant\nI am Pebble, an anglerfish.<|im_end|>"]:
        ids = tok.encode(t).ids
        print(f"{len(ids):3d} ids  {tok.decode(ids, skip_special_tokens=False)!r}")


def encode_stories():
    tok = Tokenizer.from_file(TOK)
    for split in ["valid", "train"]:
        out = os.path.join(D, f"ts_{split}.bin")
        n = 0
        with open(out, "wb") as f:
            batch = []
            for s in stories(os.path.join(D, "tinystories", f"{split}.txt")):
                batch.append(s)
                if len(batch) == 4096:
                    n += _write(tok, batch, f)
                    batch = []
            if batch:
                n += _write(tok, batch, f)
        print(f"{split}: {n:,} tokens -> {out}", flush=True)


def _write(tok, batch, f):
    ids = []
    for enc in tok.encode_batch(batch):
        ids.extend(enc.ids)
        ids.append(EOT)
    np.asarray(ids, dtype=np.uint16).tofile(f)
    return len(ids)


if __name__ == "__main__":
    {"tokenizer": build_tokenizer, "stories": encode_stories}[sys.argv[1]]()
