"""Offline tests for the talkie user-turn generator (no Ollama needed)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.aquarium.v52 import talkie  # noqa: E402
from tools.aquarium.v52.data import read_source  # noqa: E402

RES = talkie.Resources()
SPEC = {"name": "Pebble", "thing": None, "persona": "guppy"}


class Filters(unittest.TestCase):
    def check(self, text, role="user"):
        return talkie.check_turn(text, role, SPEC, RES)[1]

    def test_accepts_plain_line(self):
        self.assertIsNone(self.check("Pebble, do you like your castle?"))

    def test_rejects_blocklisted_and_reserved(self):
        self.assertTrue(self.check("I will catch you and fry you for supper.").startswith("blocklist"))
        self.assertEqual(self.check("Thou art kind, and thy fins are fine."), "blocklist:archaic")
        self.assertEqual(self.check("What do you think of a sextant, Pebble?"), "reserved_entity")
        self.assertEqual(self.check("Do you know Quilliam the guppy?"), "reserved_entity")

    def test_rejects_probe_question_and_labels(self):
        probe = json.loads(talkie.CONTRACT.read_text())["cases"][0]["users"][0]
        self.assertIn(self.check(probe), {"probe_collision", "reserved_entity"})
        self.assertEqual(self.check("Are you big? Pebble: No, I am small."), "blocklist:meta_or_format")
        self.assertEqual(self.check("Dear Pebble, what is your full name?"), "probe_family")

    def test_cookie_is_not_cooking(self):
        self.assertIsNone(self.check("Would you like a cookie, Pebble?"))
        self.assertTrue(self.check("I will cook you, Pebble.").startswith("blocklist"))


class RetryBudget(unittest.TestCase):
    def test_bad_lines_reject_conversation_after_bounded_draws(self):
        calls = []
        def bad(prompt, seed):
            calls.append(seed)
            return "Conversation 4. Pebble: hello", .01, 3
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(talkie, "talkie_line", bad), \
                mock.patch.object(talkie, "model_info", lambda: {"name": "fake", "digest": "0", "endpoint": "t"}):
            out = Path(tmp) / "run"
            talkie.init(out, 1, 3)
            talkie.advance(out, None, RES, log=lambda *_: None)
            conv = talkie.load_state(out)[1][0]
            self.assertEqual(conv["status"], "rejected")
            self.assertEqual(len(calls), talkie.SAMPLES_PER_TURN)


class Visitors(unittest.TestCase):
    def test_shuffle_bag_caps_archetypes_and_is_deterministic(self):
        spec = {"persona": "guppy", "name": "Pebble", "topic": "colors and light", "thing": None}
        n = len(talkie.VISITORS)
        first = [talkie.make_visitor(9, i, spec)["archetype"] for i in range(n)]
        self.assertEqual(sorted(first), sorted(v[0] for v in talkie.VISITORS))
        self.assertEqual(first, [talkie.make_visitor(9, i, spec)["archetype"] for i in range(n)])
        self.assertNotIn("the user", talkie.make_visitor(9, 0, {**spec, "topic": "a question about the user's life"})["header"])

    def test_trims_dangling_clause_and_period_spelling(self):
        text, reason = talkie.check_turn("What will you do to-day? I think I will", "user", SPEC, RES)
        self.assertIsNone(reason)
        self.assertEqual(text, "What will you do today?")


class Pipeline(unittest.TestCase):
    def test_resumable_rows_match_reader(self):
        lines = iter(f"Hello Pebble, can you count to {n} with me?" for n in
                     ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(talkie, "talkie_line", lambda p, s: (next(lines), .01, 5)), \
                mock.patch.object(talkie, "model_info", lambda: {"name": "fake", "digest": "0",
                                                                  "endpoint": "test"}):
            out = Path(tmp) / "run"
            talkie.init(out, 2, 1)
            talkie.advance(out, None, RES, log=lambda *_: None)          # stalls awaiting fish replies
            self.assertEqual({c["status"] for c in talkie.load_state(out)[1]}, {"need_assistant"})
            backend = talkie.CallableBackend(lambda c: f"Yes! I can count. My turn is {len(c['turns'])}.",
                                             "test")
            talkie.advance(out, backend, RES, log=lambda *_: None)
            stats = talkie.finalize(out)
            self.assertEqual(stats["accepted"], 2)
            rows, bad = read_source(out / "chat.jsonl", "talkie")
            self.assertEqual((len(rows), bad), (2, []))
            self.assertTrue(all(r["id"] >= talkie.ID_BASE for r in rows))
            self.assertEqual(rows[0]["provenance"]["assistant_turns"]["author"], "test")
            with self.assertRaises(SystemExit):
                talkie.init(out, 2, 2)                                   # seed change refused


if __name__ == "__main__":
    unittest.main()
