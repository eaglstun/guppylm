"""Leakage, immutable manifests, weighted loss, and interrupted-training regression tests."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guppylm.config import GuppyConfig
from guppylm.model import GuppyLM
from tools.aquarium.v52.data import (assert_disjoint, digest, read_manifest, split_rows,
                                     write_manifest)
from tools.aquarium.v52.train import losses
from tools.aquarium.v52.contract import contains
from tools.aquarium.v52.evaluate import cluster_ci


def row(uid, word="hello", name="Pebble"):
    return {"uid": uid, "parent_id": uid, "source": "fixture", "system": f"You are {name}.",
            "name": name, "turns": [{"role": "user", "content": f"{word} {name}"},
                                     {"role": "assistant", "content": f"hello {name}"}]}


class Manifests(unittest.TestCase):
    def test_related_dialogues_split_together(self):
        rows, _ = split_rows([row("a"), row("b", name="Coral")])
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        with self.assertRaises(ValueError):
            assert_disjoint([rows[0]], [rows[1]])

    def test_integrity_and_immutable_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.json"
            rows = [{**row("a"), "split": "train"}]
            write_manifest(path, rows, {})
            self.assertEqual(read_manifest(path, "train")[0], rows)
            with self.assertRaises(ValueError):
                read_manifest(path, "development")
            with self.assertRaises(ValueError):
                write_manifest(path, [], {})
            value = json.loads(path.read_text())
            value["rows"][0]["system"] = "tampered"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_whole_phrase_scoring(self):
        self.assertTrue(contains("Here is a RAIN gauge!", "rain gauge"))
        self.assertFalse(contains("rain gauges", "rain gauge"))
        ci = cluster_ci([[1, 1, 1], [0, 0, 0]])
        self.assertEqual(ci["mean"], .5)
        self.assertEqual(ci["prompts"], 2)

    def test_loss_is_token_weighted(self):
        class Fake(torch.nn.Module):
            def forward(self, x, y):
                # Each target's loss equals its numeric ID.
                return None, y[y != 0].float().mean()
        result = losses(Fake(), [([1], [2]), ([1, 1, 1], [4, 4, 4])], ["a", "a"], 4, "cpu", batch=1)
        self.assertEqual(result["tokens"], 4)
        self.assertEqual(result["loss"], 3.5)


class Resume(unittest.TestCase):
    def test_interrupted_matches_uninterrupted_with_dropout_and_stories(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            vocab = {s: i for i, s in enumerate(["<pad>", "<|im_start|>", "<|im_end|>", "<unk>",
                    "system", "user", "assistant", "hello", "Pebble", "You", "are", ".", "bye", "Coral"])}
            tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tok.save(str(d / "tokenizer.json"))
            for name in ("ts_train.bin", "ts_valid.bin"):
                np.random.default_rng(1).integers(4, len(vocab), 2048, dtype=np.uint16).tofile(d / name)
            config = GuppyConfig(vocab_size=len(vocab), max_seq_len=64, d_model=16, n_layers=1,
                                 n_heads=2, ffn_hidden=32, dropout=.2)
            torch.manual_seed(1)
            torch.save({"config": vars(config), "model": GuppyLM(config).state_dict()}, d / "init.pt")
            train = [{**row("a"), "split": "train", "group_id": "a"},
                     {**row("b", "hello hello"), "split": "train", "group_id": "b"}]
            dev = [{**row("c", "bye"), "split": "development", "group_id": "c"}]
            write_manifest(d / "train.json", train, {})
            write_manifest(d / "dev.json", dev, {})
            compression_dev = [{**r, "split": "compression_development"} for r in dev]
            write_manifest(d / "compression.json", compression_dev, {})
            quant_env = {**os.environ, "AQ_QUANT_CKPT": str(d / "init.pt"),
                         "AQ_QUANT_DATA": str(d), "AQ_QUANT_OUT": str(d / "quant"),
                         "AQ_QUANT_TRAIN_MANIFEST": str(d / "train.json"),
                         "AQ_QUANT_DEV_MANIFEST": str(d / "compression.json")}
            quant_test = """
import sys
sys.path.insert(0, 'tools/aquarium/quant')
import quantlib as q
assert {r['uid'] for r in q.training_rows()} == {'a', 'b'}
assert len(q.calib_batches(2)) == 1
assert q.score(q.model_from(q.SD)) > 0
"""
            result = subprocess.run([sys.executable, "-c", quant_test], cwd=ROOT, env=quant_env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            provenance = json.loads((d / "quant/calibration-0-2.json").read_text())
            self.assertEqual(set(provenance["ids"]), {"a", "b"})
            quant_env["AQ_QUANT_DEV_MANIFEST"] = str(d / "dev.json")
            rejected = subprocess.run([sys.executable, "-c", quant_test], cwd=ROOT, env=quant_env,
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("compression_development", rejected.stderr)
            common = [sys.executable, "-m", "tools.aquarium.v52.train", "--train", str(d / "train.json"),
                      "--development", str(d / "dev.json"), "--data", str(d), "--init", str(d / "init.pt"),
                      "--steps", "4", "--batch", "2", "--seq", "64", "--warmup", "1", "--dropout", ".2",
                      "--story-mix", ".5", "--eval-every", "2", "--device", "cpu", "--threads", "1"]
            def run(args):
                result = subprocess.run(common + args, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run(["--out", str(d / "full")])
            run(["--out", str(d / "resumed"), "--stop-after", "1"])
            run(["--out", str(d / "resumed"), "--resume", str(d / "resumed/final.pt")])
            full = torch.load(d / "full/final.pt", weights_only=False)
            resumed = torch.load(d / "resumed/final.pt", weights_only=False)
            def equal(a, b):
                if isinstance(a, torch.Tensor):
                    self.assertTrue(torch.equal(a, b))
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        equal(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        equal(x, y)
                else:
                    self.assertEqual(a, b)
            equal(full, resumed)
            equal(torch.load(d / "full/best.pt", weights_only=False),
                  torch.load(d / "resumed/best.pt", weights_only=False))


if __name__ == "__main__":
    unittest.main()
