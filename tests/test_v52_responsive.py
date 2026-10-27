"""Regression checks for the reviewed response-data revision."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.aquarium.v52.data import digest, dialogue_key, read_manifest, write_manifest
from tools.aquarium.v52.responsive import build, paired_rows, trim_intro


def row(uid, question='What color is the ball?', answer='Pebble here. The ball is green.'):
    return {'uid': uid, 'parent_id': uid, 'group_id': uid, 'source': 'base', 'split': 'train',
            'name': 'Pebble', 'system': 'You are Pebble, a guppy.',
            'turns': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}


def scenario(question='My toy is purple.'):
    return {'id': 'color', 'category': 'follow_details', 'turns': [
        {'role': 'user', 'content': question}, {'role': 'assistant', 'content': 'Your toy is purple.'},
        {'role': 'user', 'content': 'What color is it?'}, {'role': 'assistant', 'content': 'Purple.'}]}


class Responsive(unittest.TestCase):
    def test_pairs_change_only_first_user_case_and_keep_same_group(self):
        normal, upper = paired_rows([scenario()])
        self.assertEqual(upper['turns'][0]['content'], normal['turns'][0]['content'].upper())
        self.assertEqual(normal['turns'][1:], upper['turns'][1:])
        self.assertEqual(dialogue_key(normal, True), dialogue_key(upper, True))
        self.assertNotEqual(normal['id'], upper['id'])
        with self.assertRaises(ValueError):
            paired_rows([scenario(), scenario()])

    def test_trim_only_name_preamble_preserving_identity_context(self):
        old = row('a')
        new, changes = trim_intro(old)
        self.assertEqual(new['turns'][1]['content'], 'The ball is green.')
        self.assertEqual(old['turns'][1]['content'], 'Pebble here. The ball is green.')
        self.assertEqual(new['parent_id'], old['parent_id'])
        self.assertNotEqual(new['uid'], old['uid'])
        self.assertEqual(len(changes), 1)
        for q in ('What is your name?', 'Hello!', 'Who are you?', 'Which kind of fish are you?'):
            r = row('a', q)
            self.assertEqual(trim_intro(r), (r, []))
        r = row('a', answer='I am Pebble, and the ball is green.')
        self.assertEqual(trim_intro(r), (r, []))

    def fixture(self, root):
        base = root / 'base'; base.mkdir()
        bad = row('bad', 'Would you play?', 'A game is a kind of food.')
        clone = {**bad, 'uid': 'clone'}
        good = row('good')
        addon = {**row('addon'), 'source': 'uppercase'}
        write_manifest(base / 'train_A.json', [bad, clone, good, addon], {})
        for split in ('development', 'historical', 'compression_development'):
            write_manifest(base / (split + '.json'), [{**row(split, split), 'split': split}], {})
        (base / 'contract.json').write_text(json.dumps({'reserved': {'development': ['forbidden'], 'locked_test': []},
                                                      'cases': [{'users': ['frozen question'], 'context': []}]}))
        decisions = root / 'decisions.json'
        decisions.write_text(json.dumps([{'uid': 'bad', 'parent_id': 'bad',
                                          'content_sha256': digest([bad['system'], bad['turns']])}]))
        scenarios = root / 'scenarios.jsonl'; scenarios.write_text(json.dumps(scenario()) + '\n')
        return base, scenarios, decisions

    def test_builder_quarantines_descendants_and_preserves_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, scenarios, decisions = self.fixture(root)
            out = root / 'out'; audit = build(base, out, scenarios, decisions)
            rows, _, _ = read_manifest(out / 'train_A.json', 'train')
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['parent_id'], 'good')
            self.assertEqual(audit['removed_by_reason'], {'reviewed_parent': 2, 'replace_experimental_addons': 1})
            self.assertEqual(audit['trimmed_rows'], 1)
            self.assertEqual(rows[1]['group_id'], rows[2]['group_id'])
            for name in audit['frozen']:
                self.assertEqual((out / name).read_bytes(), (base / name).read_bytes())
            with self.assertRaises(ValueError):
                build(base, out, scenarios, decisions)

    def test_stale_review_and_probe_leaks_leave_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, scenarios, decisions = self.fixture(root)
            for i, question in enumerate(('forbidden word', 'frozen question')):
                scenarios.write_text(json.dumps(scenario(question)) + '\n')
                with self.assertRaises(ValueError):
                    build(base, root / f'bad-{i}', scenarios, decisions)
                self.assertFalse((root / f'bad-{i}').exists())
            decisions.write_text(json.dumps([{'uid': 'missing', 'parent_id': 'missing', 'content_sha256': 'bad'}]))
            with self.assertRaises(ValueError):
                build(base, root / 'stale', scenarios, decisions)
            self.assertFalse((root / 'stale').exists())


if __name__ == '__main__':
    unittest.main()
