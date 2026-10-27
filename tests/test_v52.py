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
from tools.aquarium.v52.contract import contains
from tools.aquarium.v52.curate import (
    CONFIGS,
    curate_train,
    degenerate_reason,
    load_extra_sources,
    parse_extra_source,
)
from tools.aquarium.v52.data import (
    assert_compatible_metadata,
    assert_disjoint,
    digest,
    file_hash,
    read_manifest,
    split_rows,
    write_manifest,
)
from tools.aquarium.v52.evaluate import cluster_ci, verify_selection
from tools.aquarium.v52.prepare import check_frozen_eval
from tools.aquarium.v52.train import losses


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

    def test_contract_and_tokenizer_compatibility(self):
        meta = {"contract_sha256": "same", "inputs": {"tokenizer.json": {"sha256": "tokenizer"}}}
        assert_compatible_metadata(meta, meta, "tokenizer")
        with self.assertRaises(ValueError):
            assert_compatible_metadata(meta, meta, "wrong")
        with self.assertRaises(ValueError):
            assert_compatible_metadata(meta, {**meta, "contract_sha256": "other"}, "tokenizer")

    def test_locked_test_selection_is_bound_to_real_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            contract = {"cases": []}
            report = {"contract_sha256": digest(contract), "checkpoint_sha256": "checkpoint", "split": "development"}
            (d / "report.json").write_text(json.dumps(report))
            (d / "rubric.json").write_text(json.dumps({"contract_sha256": digest(contract)}))
            selection = {"contract_sha256": digest(contract), "checkpoint_sha256": "checkpoint",
                         "development_report_path": "report.json", "development_report_sha256": file_hash(d / "report.json"),
                         "rubric_path": "rubric.json", "rubric_sha256": file_hash(d / "rubric.json")}
            (d / "selection.json").write_text(json.dumps(selection))
            verify_selection(d / "selection.json", contract, "checkpoint")
            with self.assertRaises(ValueError):
                verify_selection(d / "selection.json", contract, "different weights")
            (d / "report.json").write_text("changed")
            with self.assertRaises(ValueError):
                verify_selection(d / "selection.json", contract, "checkpoint")

    def test_loss_is_token_weighted(self):
        class Fake(torch.nn.Module):
            def forward(self, x, y):
                # Each target's loss equals its numeric ID.
                return None, y[y != 0].float().mean()
        result = losses(Fake(), [([1], [2]), ([1, 1, 1], [4, 4, 4])], ["a", "a"], 4, "cpu", batch=1)
        self.assertEqual(result["tokens"], 4)
        self.assertEqual(result["loss"], 3.5)


def chat(uid, user, reply, name="Pebble", source="fixture", **extra):
    return {"uid": uid, "parent_id": uid, "source": source, "split": "train", "name": name,
            "system": f"You are {name}, a guppy.", "origin": {"file": "fixture", "line": 1},
            "turns": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}], **extra}


class Curation(unittest.TestCase):
    def test_v1_config_reproduces_uncurated_selection(self):
        pool = [chat(str(i), "do you have any friends", "i do not know that word but it sounds nice") for i in range(9)]
        base, extra, report = curate_train(pool, pool[:2], 5, 1, CONFIGS[1])
        self.assertEqual(base, pool[:5])
        self.assertEqual(extra, pool[:2])
        self.assertEqual(report["removed_total"], 0)

    def test_degenerate_rows(self):
        looping = chat("a", "hi", "same reply")
        looping["turns"] += [{"role": "user", "content": "and?"}, {"role": "assistant", "content": "Same reply!"}]
        self.assertEqual(degenerate_reason(looping), "repeated_assistant_reply")
        echo = chat("b", "hi", "hello")
        echo["turns"] += [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "bye"}]
        self.assertEqual(degenerate_reason(echo), "turn_repeats_previous_turn")
        self.assertIsNone(degenerate_reason(chat("c", "hi", "hello")))

    def test_caps_bound_templates_and_questions_with_backfill(self):
        config = {**CONFIGS[2], "assistant_ngram_cap": 2, "user_turn_cap": 3, "name_cap_over_median": None}
        stock = "I do not know that word, but it sounds like a shell number {}."
        pool = [chat(f"s{i}", f"question {i}?", stock.format(i)) for i in range(5)]
        pool += [chat(f"q{i}", "Do you have any friends?", f"reply {i} is unique") for i in range(5)]
        pool += [chat(f"t{i}", "Thanks Pebble!", f"thanks reply {i}") for i in range(5)]
        pool += [chat(f"c{i}", f"count {i}", f"one two three four five six seven eight nine ten {i}") for i in range(4)]
        base, _, report = curate_train(pool, [], 100, 1, config)
        kept = {r["uid"] for r in base}
        self.assertEqual(sum(u.startswith("s") for u in kept), 2)   # stock 8-gram capped, not removed
        self.assertEqual(sum(u.startswith("q") for u in kept), 3)   # repeated user question capped
        self.assertEqual(sum(u.startswith("t") for u in kept), 5)   # courtesy turn exempt
        self.assertEqual(sum(u.startswith("c") for u in kept), 4)   # counting sequence exempt
        self.assertEqual(report["removed"]["assistant_ngram_cap"]["rows"], 3)
        self.assertEqual(report["removed"]["user_turn_cap"]["rows"], 2)
        # The cap is filled from later pool rows, so the selected count stays at the budget.
        base, _, report = curate_train(pool, [], 12, 1, config)
        self.assertEqual(len(base), 12)
        self.assertEqual(report["base_pool"]["selected"], 12)

    def test_name_rebalance_renames_head_without_reserved_or_present_names(self):
        from tools.aquarium.personas import NAMES
        # Candidate counts: three allowed names x2, Coral x0, Misty x10 -> median 2, limit 6.
        config = {"version": 2, "name_cap_over_median": 3.0}
        others = [n for n in sorted(set(NAMES)) if n not in ("Misty", "Coral")]
        pool = [chat(f"o{i}", "hi", "hello", name=others[i % len(others)]) for i in range(len(others) * 2)]
        pool += [chat(f"m{i}", "Misty, hi", "I am Misty. A misty morning.", name="Misty") for i in range(10)]
        base, _, report = curate_train(pool, [], 10 ** 6, 1, config, reserved_names=others[:-3])
        renamed = [r for r in base if r.get("curation")]
        self.assertEqual(len(renamed), 4)
        for r in renamed:
            self.assertIn(r["name"], others[-3:] + ["Coral"])
            self.assertIn(f"I am {r['name']}.", r["turns"][1]["content"])
            self.assertIn("misty morning", r["turns"][1]["content"])  # adjective untouched
            self.assertNotEqual(r["uid"], r["parent_id"])
        self.assertEqual(sum(r["name"] == "Misty" for r in base), 6)
        self.assertEqual(report["renamed"]["rows"], 4)
        self.assertEqual(report["name_limit"]["limit"], 6)

    def test_extra_source_is_train_only_and_checked(self):
        import re
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            existing = chat("e", "hello Pebble", "hi there")
            dev = {**chat("d", "where do you hide Coral", "under the rock Coral", name="Coral"), "split": "development"}
            from tools.aquarium.v52.data import dialogue_key
            def line(name, user, reply, i):
                return json.dumps({"id": i, "name": name, "system": f"You are {name}, a guppy.",
                                   "turns": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]})
            (d / "talk").mkdir()
            (d / "talk" / "a.jsonl").write_text("\n".join([
                line("Pebble", "hello Pebble", "hi there", 1),                     # exact duplicate
                line("Kelp", "where do you hide Kelp", "under the rock Kelp", 2),  # eval group overlap
                line("Kelp", "say zorbleflax", "zorbleflax", 3),                   # reserved entity
                line("Kelp", "what is new Kelp", "bubbles", 4),                    # accepted
                json.dumps({"id": 5, "system": "x", "turns": [{"role": "user", "content": "only"}]}),
            ]) + "\n")
            rows, inputs, rejected, report = load_extra_sources(
                [f"talk={d / 'talk'}"], [existing], {dialogue_key(dev, True)}, re.compile(r"(?<!\w)zorbleflax(?!\w)"))
            self.assertEqual([r["turns"][0]["content"] for r in rows], ["what is new Kelp"])
            self.assertTrue(all(r["split"] == "train" and r["source"] == "talk" for r in rows))
            self.assertEqual(report["talk"], {"path": str((d / "talk").resolve()), "files": 1, "invalid": 1,
                                              "duplicate": 1, "eval_group_overlap": 1, "reserved_entity": 1, "accepted": 1})
            self.assertEqual(len(rejected), 1)
            self.assertIn("extra/talk/a.jsonl", inputs)
            for bad in ("base=" + str(d), "noequals", "Bad Name=" + str(d), "x=" + str(d / "missing")):
                with self.assertRaises(ValueError):
                    parse_extra_source(bad)

    def test_frozen_eval_check(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            contract = {"cases": [{"users": ("a", "b")}]}
            (d / "contract.json").write_text(json.dumps(contract))
            rows = [{**row("a"), "split": "development"}]
            write_manifest(d / "development.json", rows, {"generator": "old"})
            self.assertTrue(check_frozen_eval(d, {"development": rows}, contract)["development"]["identical"])
            with self.assertRaises(ValueError):
                check_frozen_eval(d, {"development": [{**rows[0], "system": "changed"}]}, contract)
            with self.assertRaises(ValueError):
                check_frozen_eval(d, {"development": rows}, {"cases": []})


class TraitEchoRewrites(unittest.TestCase):
    def test_detector(self):
        from tools.aquarium.v52.curate import trait_core, trait_echo_turns
        self.assertEqual(trait_core("lobster"), "grumpy but fair")
        self.assertEqual(trait_core("sea_turtle"), "calm old and wise")
        r = {**chat("a", "hi", "I am Gus, grumpy but fair. Hmph."), "persona": "lobster"}
        r["turns"] += [{"role": "user", "content": "You are grumpy but fair!"},
                       {"role": "assistant", "content": "I am fair to everyone."}]
        self.assertEqual(trait_echo_turns(r), [1])  # user turns and partial phrases do not count

    def test_apply_rewrites_is_hash_pinned(self):
        import hashlib

        from tools.aquarium.v52 import curate
        original = "I am Gus, grumpy but fair. Hmph."
        parent = {**chat("p", "hi", original), "persona": "lobster", "curation": {"renamed_from": "Misty"}}
        def entry(text=original, uid="p", replacement="I am Gus. I snap at crabs, but I share my clams. Hmph."):
            return {"uid": uid, "turn": 1, "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "original": text, "replacement": replacement}
        with tempfile.TemporaryDirectory() as directory:
            old, curate.REWRITES = curate.REWRITES, Path(directory)
            try:
                def run(entries, rows=(parent,), strict=True):
                    (Path(directory) / "r.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
                    report = {"config": {"version": 3}}
                    return curate.apply_rewrites(list(rows), ["r.jsonl"], report, strict), report
                (row,), report = run([entry()])
                self.assertEqual(row["turns"][1]["content"], entry()["replacement"])
                self.assertEqual(row["parent_id"], "p")
                self.assertNotEqual(row["uid"], "p")
                self.assertEqual(row["curation"]["renamed_from"], "Misty")
                self.assertEqual(row["curation"]["rewritten"], {"file": "r.jsonl", "turns": [1], "from_uid": "p"})
                self.assertEqual(report["rewrites"]["turns"], 1)
                self.assertEqual(parent["turns"][1]["content"], original)  # input not mutated
                changed = {**parent, "turns": [parent["turns"][0], {"role": "assistant", "content": "edited"}]}
                with self.assertRaises(ValueError):
                    run([entry()], rows=(changed,))                       # stale original
                with self.assertRaises(ValueError):
                    run([entry(uid="gone")])                              # stale uid
                _, report = run([entry(uid="gone")], strict=False)
                self.assertEqual(report["rewrites"]["missing"], ["gone"])
                with self.assertRaises(ValueError):
                    run([entry(replacement="Still grumpy but fair.")])  # rewrite must remove the echo
                with self.assertRaises(ValueError):
                    run([{**entry(), "original": "tampered"}])           # file internally inconsistent
            finally:
                curate.REWRITES = old

    def test_tracked_rewrite_file(self):
        import collections
        import hashlib

        from tools.aquarium.personas import NAMES
        from tools.aquarium.v52.curate import REWRITES, trait_core
        from tools.aquarium.v52.data import normalized
        entries = [json.loads(line) for line in (REWRITES / "trait_echo_v1.jsonl").read_text().splitlines()]
        self.assertGreater(len(entries), 900)
        names = set(NAMES)
        def grams(text):
            words = ["NAME" if w.capitalize() in names else w for w in normalized(text).split()]
            return {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}
        new = collections.Counter()
        for e in entries:
            self.assertEqual(hashlib.sha256(e["original"].encode()).hexdigest(), e["original_sha256"])
            self.assertEqual(trait_core(e["persona"]), e["detector"])
            self.assertIn(e["detector"], normalized(e["original"]))
            self.assertNotIn(e["detector"], normalized(e["replacement"]))
            new.update(grams(e["replacement"]) - grams(e["original"]))
        # The rewrites must not become a new stock template.
        self.assertLessEqual(max(new.values()), 5)


class Prepare(unittest.TestCase):
    """End to end on a tiny fixture: an extra source never changes or enters eval splits."""

    def test_extra_source_keeps_eval_splits_identical(self):
        personas = ["guppy", "goldfish", "octopus", "seahorse"]
        names = ["Pebble", "Coral", "Kelp", "Finn", "Olive", "Mango"]
        words = ["hello", "where", "do", "you", "live", "what", "do", "you", "eat", "i", "like", "little", "bugs", "and", "green", "plants", "the", "water", "is", "warm", "today", "my", "castle", "is", "big", "tell", "me", "a", "story", "once", "a", "crab", "came", "by", "we", "played", "all", "day"]
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory) / "data"
            for sub in ("haiku", "songs", "family", "tinydialogues", "ocean", "memory", "selfname"):
                (d / sub).mkdir(parents=True)
            def rows_for(start, n, source):
                out = []
                for i in range(start, start + n):
                    name = names[i % len(names)]
                    out.append({"id": i, "persona": personas[i % 4], "name": name, "thing": None,
                                "system": f"You are {name}, a fish.",
                                "turns": [{"role": "user", "content": f"{name}, {words[i % 30]} {words[(i * 7) % 30]} {i}?"},
                                          {"role": "assistant", "content": f"I am {name}. {words[(i * 3) % 30]} {i}."}]})
                return "\n".join(json.dumps(r) for r in out) + "\n"
            (d / "dialogues.jsonl").write_text(rows_for(0, 120, "base"))
            (d / "songs" / "chat.jsonl").write_text(rows_for(1000, 10, "songs"))
            (d / "family" / "chat.jsonl").write_text(rows_for(2000, 10, "family"))
            (d / "dialogues_eval.jsonl").write_text(rows_for(3000, 5, "historical"))
            (d / "tinydialogues" / "chat.jsonl").write_text(rows_for(4000, 5, "td"))
            (d / "ocean" / "chat.jsonl").write_text(rows_for(5000, 5, "ocean"))
            vocab = {w: i for i, w in enumerate(["<pad>", "<|im_start|>", "<|im_end|>", "<unk>"] + sorted(set(words)))}
            tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tok.save(str(d / "tokenizer.json"))
            def prepare(out, *args):
                result = subprocess.run([sys.executable, "-m", "tools.aquarium.v52.prepare", "--data", str(d),
                                         "--out", str(out), "--cap", "100", *args], cwd=ROOT,
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads((Path(out) / "audit.json").read_text())
            first = prepare(Path(directory) / "v1", "--curation", "1")
            reserved = json.loads((Path(directory) / "v1" / "contract.json").read_text())["reserved"]
            extra = Path(directory) / "talk.jsonl"
            extra.write_text("\n".join(json.dumps({"id": 9000 + i, "persona": "guppy", "name": "Zed",
                                                    "system": "You are Zed, a fish.", "thing": None,
                                                    "turns": [{"role": "user", "content": user},
                                                              {"role": "assistant", "content": "Bubbles everywhere."}]})
                                        for i, user in enumerate(["Zed, do you like bubbles?",
                                                                  f"Zed, have you seen {reserved['development'][0]}?"])) + "\n")
            second = prepare(Path(directory) / "v2", "--extra-source", f"talk={extra}",
                             "--frozen-eval", str(Path(directory) / "v1"))
            self.assertEqual(first["eval_rows_sha256"], second["eval_rows_sha256"])
            self.assertTrue(all(v["identical"] for v in second["frozen_eval"].values()))
            self.assertEqual(second["curation"]["extra_sources"]["talk"]["accepted"], 1)
            self.assertEqual(second["curation"]["extra_sources"]["talk"]["reserved_entity"], 1)
            for name in ("development", "historical", "compression_development"):
                rows, _, meta = read_manifest(Path(directory) / "v2" / f"{name}.json")
                self.assertFalse(any(r["source"] == "talk" for r in rows))
                self.assertEqual(meta["generator"], "v52.prepare/v2")
            train, _, meta = read_manifest(Path(directory) / "v2" / "train_A.json", "train")
            self.assertEqual(meta["extra_sources"], ["talk"])
            self.assertEqual(meta["curation"]["version"], 3)
            # Rewrites target the real corpus; with an extra source they are recorded, not fatal.
            self.assertGreater(len(second["curation"]["rewrites"]["missing"]), 900)
            self.assertTrue(any(r["source"] == "talk" and r["augmentation"] == "base" for r in train))
            self.assertTrue(any(r["source"] == "talk" and r["augmentation"] == "rename" for r in train))
            self.assertIn("talk", second["candidates"]["A"]["tokens"])
            _, _, first_meta = read_manifest(Path(directory) / "v1" / "train_A.json", "train")
            self.assertEqual(first_meta["curation"], {"version": 1})


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
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            provenance = json.loads((d / "quant/calibration-0-2.json").read_text())
            self.assertEqual(set(provenance["ids"]), {"a", "b"})
            quant_env["AQ_QUANT_DEV_MANIFEST"] = str(d / "dev.json")
            rejected = subprocess.run([sys.executable, "-c", quant_test], cwd=ROOT, env=quant_env,
                                      capture_output=True, text=True, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("compression_development", rejected.stderr)
            common = [sys.executable, "-m", "tools.aquarium.v52.train", "--train", str(d / "train.json"),
                      "--development", str(d / "dev.json"), "--data", str(d), "--init", str(d / "init.pt"),
                      "--steps", "4", "--batch", "2", "--seq", "64", "--warmup", "1", "--dropout", ".2",
                      "--story-mix", ".5", "--eval-every", "2", "--device", "cpu", "--threads", "1"]
            def run(args):
                result = subprocess.run(common + args, cwd=ROOT, capture_output=True, text=True, check=False)
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
