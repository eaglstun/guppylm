"""Lineage, semantic grounding, casing and immutable-output checks for revision v2."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.aquarium.v52.data import digest, read_manifest, write_manifest
from tools.aquarium.v52.revise_data import (build, grounded_row, habitat_system,
                                          normalized_habitat, select_grounded,
                                          uppercase_counterpart)


class Tokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=[3] * len(text.split()))


def parent(uid='p', persona='guppy'):
    return {'uid': uid, 'parent_id': uid, 'group_id': uid, 'source': 'base', 'split': 'train',
            'augmentation': 'base', 'name': 'Pebble', 'persona': persona, 'thing': 'a bucket',
            'system': 'Roleplay as Pebble, a guppy. You only know about life in a tank.',
            'turns': [{'role': 'user', 'content': 'I brought a bucket.'},
                      {'role': 'assistant', 'content': 'The bucket can hold water.'}]}


def noun(base, uid='n', word='flibber'):
    return {**base, 'uid': uid, 'augmentation': 'noun', 'thing': word,
            'turns': [{'role': 'user', 'content': f'I brought a {word}.'},
                      {'role': 'assistant', 'content': f'A {word} can hold water.'}]}


def scenario(question='I painted a silver door.'):
    return {'id': 's', 'category': 'details', 'turns': [
        {'role': 'user', 'content': question}, {'role': 'assistant', 'content': 'Your door is silver.'},
        {'role': 'user', 'content': 'What color is it?'}, {'role': 'assistant', 'content': 'Silver.'}]}


class ReviseData(unittest.TestCase):
    def test_habitat_change_is_narrow_and_keeps_identity_lineage(self):
        old = parent(); new = normalized_habitat(old)
        self.assertEqual(new['system'], 'Roleplay as Pebble, a guppy. You live in a tank.')
        self.assertEqual(new['turns'], old['turns'])
        self.assertEqual(new['parent_id'], old['parent_id'])
        self.assertNotEqual(new['uid'], old['uid'])
        self.assertIn('only know', old['system'])
        other = 'You know a song about life in a tank.'
        self.assertEqual(habitat_system(other), other)

    def test_grounding_is_explicit_and_samples_one_per_family(self):
        p = parent(); n = noun(p)
        for template in range(4):
            r = grounded_row(n, p, template)
            self.assertEqual(r['parent_id'], p['parent_id'])
            self.assertEqual(r['source'], 'grounded_copy')
            self.assertEqual(len(r['turns']), 6)
            # The user supplies the meaning before any assistant claims to know it.
            definition = next(i for i, t in enumerate(r['turns'])
                              if t['role'] == 'user' and 'a bucket' in t['content'])
            self.assertTrue(any('a bucket' in t['content'] for t in r['turns'][definition+1:]))
            if template == 3:
                self.assertIn('do not know', r['turns'][1]['content'])
        p2 = parent('p2', 'clownfish')
        rows = [p, n, noun(p, 'n2', 'blibble'), p2, noun(p2, 'n3', 'plimble')]
        chosen = select_grounded(rows, 1)
        self.assertEqual({r['parent_id'] for r in chosen}, {'p', 'p2'})
        self.assertEqual(chosen, select_grounded(list(reversed(rows)), 1))
        with self.assertRaises(ValueError):
            select_grounded([n], 1)
        with self.assertRaises(ValueError):
            grounded_row(n, p2, 0)

    def test_abstract_meanings_are_not_treated_as_physical_objects(self):
        p = parent(); p['thing'] = 'homework'
        for template in range(4):
            r = grounded_row(noun(p), p, template)
            text = ' '.join(t['content'] for t in r['turns'])
            self.assertNotIn('Which object', text)
            self.assertNotIn('I brought', text)
        r = grounded_row(noun(p), p, 1)
        self.assertIn('everywhere', r['turns'][4]['content'])
        self.assertEqual(r['turns'][5]['content'], 'No. It is the meaning you chose for this game.')

    def test_case_changes_no_response_or_followup(self):
        r = parent(); upper = uppercase_counterpart(r)
        self.assertEqual(upper['turns'][0]['content'], r['turns'][0]['content'].upper())
        self.assertEqual(upper['turns'][1:], r['turns'][1:])
        self.assertEqual(upper['parent_id'], r['parent_id'])
        self.assertEqual(upper['group_id'], r['group_id'])
        self.assertIsNone(uppercase_counterpart(upper))

    def fixture(self, root, question='I painted a silver door.'):
        base = root / 'base'; base.mkdir()
        p = parent()
        talkie = {**parent('t'), 'source': 'talkie',
                  'turns': [{'role': 'user', 'content': 'Would you like to see my painting?'},
                            {'role': 'assistant', 'content': 'Yes, show it through the glass.'}]}
        write_manifest(base / 'train_A.json', [p, noun(p), talkie], {})
        for split in ('development', 'historical', 'compression_development'):
            r = parent(split); r['split'] = split; r['turns'][0]['content'] = split
            write_manifest(base / (split + '.json'), [r], {})
        (base / 'contract.json').write_text(json.dumps({'reserved': {'development': ['reservedword'], 'locked_test': []},
                                                      'cases': [{'users': ['frozen question'], 'context': []}]}))
        path = root / 'scenarios.jsonl'; path.write_text(json.dumps(scenario(question)) + '\n')
        return base, path

    def test_build_preserves_eval_and_replaces_noun_not_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, scenarios = self.fixture(root); out = root / 'out'
            audit = build(base, out, Tokenizer(), 'tok', scenarios, 1)
            rows, _, _ = read_manifest(out / 'train_A.json', 'train')
            self.assertEqual(len(rows), 6)
            self.assertEqual(audit['removed_unexplained_noun_rows'], 1)
            self.assertEqual(audit['grounded_copy_rows'], 1)
            self.assertEqual(audit['new_casing_rows'], 1)
            self.assertFalse(any(r['augmentation'] == 'noun' for r in rows))
            self.assertTrue(any(r['augmentation'] == 'base' and r['parent_id'] == 'p' for r in rows))
            for name in audit['frozen']:
                self.assertEqual((base / name).read_bytes(), (out / name).read_bytes())
            with self.assertRaises(ValueError):
                build(base, out, Tokenizer(), 'tok', scenarios, 1)

    def test_leaks_and_truncation_fail_without_publishing(self):
        for question in ('reservedword is here', 'frozen question', 'word ' * 600):
            with self.subTest(question=question[:20]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); base, scenarios = self.fixture(root, question)
                with self.assertRaises(ValueError):
                    build(base, root / 'bad', Tokenizer(), 'tok', scenarios, 1)
                self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
