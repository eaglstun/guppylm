"""GuppyLM dataset loading."""

import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


def subset_lines(path, limit=None, seed=0):
    """Lines of a jsonl file, optionally a seeded random subset of `limit` lines.

    The same (path, limit, seed) always yields the same subset, so analysis tools
    can reconstruct exactly what a --train-limit run saw.
    """
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    if limit is not None and limit < len(lines):
        lines = random.Random(seed).sample(lines, limit)
    return lines


class GuppyDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512,
                 limit=None, subset_seed: int = 0):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []

        for line in subset_lines(path, limit, subset_seed):
            data = json.loads(line)
            ids = self.tokenizer.encode(data["text"]).ids
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(ids) >= 2:
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = ids[:-1]
        y = ids[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
    return padded_x, padded_y


def get_dataloader(path, tokenizer_path, max_len=512, batch_size=32, shuffle=True,
                   limit=None, subset_seed=0):
    dataset = GuppyDataset(path, tokenizer_path, max_len, limit, subset_seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
