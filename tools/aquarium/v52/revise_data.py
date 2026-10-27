"""Versioned v2 revision: explicit word meanings, habitat prompts, diverse casing."""
import argparse
import collections
import json
import re
import shutil
import tempfile
from pathlib import Path

from .append_context import FROZEN
from .curate import load_extra_sources
from .data import (assert_compatible_metadata, assert_disjoint, dialogue_key, digest,
                   file_hash, normalized, read_manifest, write_manifest)
from .responsive import paired_rows
from .uppercase import encode_counts

SCENARIOS = Path(__file__).with_name('rewrites') / 'responsive_v2.jsonl'
VERSION = 'response-data-v2'


def habitat_system(system):
    return system.replace('You only know about life in ', 'You live in ')


def normalized_habitat(row):
    system = habitat_system(row['system'])
    if system == row['system']:
        return row
    return {**row, 'system': system, 'uid': digest([row['uid'], VERSION, system]),
            'habitat_revision': {'from_uid': row['uid'], 'original_system': row['system']}}


def grounded_row(row, parent, template):
    """Replace an unexplained word substitution with a scoped, user-defined meaning."""
    word, meaning = row.get('thing'), parent.get('thing')
    if (row.get('parent_id') != parent.get('parent_id') or not isinstance(word, str)
            or not re.fullmatch(r'[a-z]+', word) or not isinstance(meaning, str)
            or not meaning.strip() or normalized(word) == normalized(meaning)):
        raise ValueError(f"Invalid noun lineage/meaning: {row.get('uid')}")
    scripts = [
        [(f"In our pretend game, '{word}' is my name for {meaning}.",
          f"All right. In this game, '{word}' means {meaning}."),
         (f"What does '{word}' mean in our game?", f"It stands for {meaning}."),
         (f"And what did I call {meaning}?", f"You called it '{word}'.")],
        [(f"In my word game, I use '{word}' to mean {meaning}.",
          f"In your word game, '{word}' means {meaning}."),
         ("What was the invented name I chose?", f"You chose '{word}'."),
         (f"Does '{word}' have this meaning everywhere?", f"No. It is the meaning you chose for this game.")],
        [(f"When I say '{word}' in this story, I mean {meaning}.",
          f"I understand your story's meaning for '{word}': {meaning}."),
         (f"What is the ordinary name for '{word}'?", f"The ordinary name is {meaning}."),
         ("And what is the story name?", f"The story name is '{word}'.")],
        [(f"I have a new pretend word: '{word}'.",
          f"I do not know what you mean by '{word}' yet. What is it?"),
         (f"In this game, it means {meaning}.", f"Thanks for explaining. In this game, '{word}' means {meaning}."),
         (f"What did I call {meaning}?", f"You called it '{word}'.")],
    ]
    turns = [{'role': role, 'content': text} for pair in scripts[template % len(scripts)]
             for role, text in zip(('user', 'assistant'), pair)]
    system = habitat_system(row['system'])
    return {**row, 'system': system, 'turns': turns, 'exchanges': len(turns) // 2,
            'source': 'grounded_copy', 'augmentation': 'grounded_noun_v2',
            'uid': digest([row['uid'], VERSION, system, turns]),
            'revision_provenance': {'from_uid': row['uid'], 'real_meaning': meaning,
                                    'pretend_word': word, 'template': template % 4}}


def select_grounded(rows, per_persona=32):
    if per_persona < 1:
        raise ValueError('per-persona must be positive')
    parents = {r['parent_id']: r for r in rows if r.get('augmentation') == 'base'}
    # One substitution per real conversation family; balance across creature personas.
    unique = {}
    for row in sorted(rows, key=lambda r: digest([VERSION, r['uid']])):
        if row.get('augmentation') == 'noun':
            if row['parent_id'] not in parents:
                raise ValueError(f"Missing original conversation: {row['parent_id']}")
            unique.setdefault(row['parent_id'], row)
    buckets = collections.defaultdict(list)
    for row in unique.values():
        buckets[row.get('persona', 'unknown')].append(row)
    result = []
    for persona in sorted(buckets):
        for i, row in enumerate(buckets[persona][:per_persona]):
            result.append(grounded_row(row, parents[row['parent_id']], i))
    return result


def uppercase_counterpart(row):
    """Only change first-user casing; preserve answers, later turns, parent and group."""
    turns = [dict(t) for t in row['turns']]
    if not turns or turns[0]['role'] != 'user':
        raise ValueError('Expected a user-first conversation')
    upper = turns[0]['content'].upper()
    if upper == turns[0]['content']:
        return None
    turns[0]['content'] = upper
    return {**row, 'turns': turns, 'uid': digest([row['uid'], VERSION, 'uppercase', turns]),
            'augmentation': 'uppercase_user_v2',
            'casing_provenance': {'from_uid': row['uid'], 'rule': 'first_user_only'}}


def build(base, out, tokenizer, tokenizer_hash, scenarios_path=SCENARIOS, per_persona=32):
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    original, base_hash, metadata = read_manifest(base / 'train_A.json', 'train')
    evaluation = []
    for name in FROZEN[1:]:
        rows, _, meta = read_manifest(base / name)
        assert_compatible_metadata(metadata, meta, tokenizer_hash)
        evaluation.extend(rows)
    contract = json.loads((base / 'contract.json').read_text())
    grounded = select_grounded(original, per_persona)
    removed = [r for r in original if r.get('augmentation') == 'noun']
    retained = [normalized_habitat(r) for r in original if r.get('augmentation') != 'noun']
    changed = [new for old, new in zip((r for r in original if r.get('augmentation') != 'noun'), retained)
               if old['uid'] != new['uid']]
    scenarios = [json.loads(line) for line in scenarios_path.read_text().splitlines() if line.strip()]
    authored = paired_rows(scenarios)
    for i, row in enumerate(authored):
        row['id'] = 74_000_000 + i
        row['provenance']['version'] = VERSION
    strict = sum(contract['reserved'].values(), []) + ['axolotl', 'cuttlefish']
    forbidden = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(normalized(x)) for x in strict) + r')(?!\w)')
    probes = {normalized(q) for case in contract['cases'] for q in case['users']}
    probes |= {normalized(t['content']) for c in contract['cases'] for t in c['context'] if t['role'] == 'user'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='revision-v2-', dir=out.parent) as temporary:
        stage = Path(temporary)
        source = stage / 'responsive-v2.jsonl'
        source.write_text(''.join(json.dumps(r) + '\n' for r in authored))
        additions, inputs, bad, report = load_extra_sources([f'responsive={source}'], retained + evaluation,
            {dialogue_key(r, True) for r in evaluation}, forbidden)
        if bad or len(additions) != len(authored):
            raise ValueError(f'Incomplete authored import: {report}, {bad}')
        for row in additions:
            row['origin']['file'] = str(out / source.name)
        for info in inputs.values():
            info['path'] = str((out / source.name).resolve())
        casing, skipped_casing = [], []
        seen = {digest([r['system'], r['turns']]) for r in retained + additions + grounded}
        for row in retained:
            if row['source'] != 'talkie' or row.get('augmentation') != 'base':
                continue
            variant = uppercase_counterpart(row)
            if variant is None:
                continue
            key = digest([variant['system'], variant['turns']])
            if key in seen:
                skipped_casing.append({'uid': row['uid'], 'reason': 'duplicate'})
                continue
            _, targets, truncated = encode_counts(variant, tokenizer, 512)
            if truncated or not targets:
                skipped_casing.append({'uid': row['uid'], 'reason': 'context_budget'})
                continue
            seen.add(key); casing.append(variant)
        new_rows = grounded + additions + casing
        token_audit = collections.defaultdict(lambda: {'rows': 0, 'input_tokens': 0, 'target_tokens': 0})
        for row in new_rows:
            text = normalized(row['system'] + ' ' + ' '.join(t['content'] for t in row['turns']))
            if forbidden.search(text):
                raise ValueError(f"Reserved entity in revised row {row['uid']}")
            if any(normalized(t['content']) in probes for t in row['turns'] if t['role'] == 'user'):
                raise ValueError('New row overlaps a frozen probe question')
            n, targets, truncated = encode_counts(row, tokenizer, 512)
            if truncated or not targets:
                raise ValueError(f"New row cannot fit its targets: {row['uid']}")
            bucket = token_audit[row['source']]
            bucket['rows'] += 1; bucket['input_tokens'] += n; bucket['target_tokens'] += targets
        assert_disjoint(changed + new_rows, evaluation)
        merged = retained + new_rows
        if len({r['uid'] for r in merged}) != len(merged):
            raise ValueError('Duplicate revised IDs')
        revised_meta = {**metadata, 'data_revision_v2': {'base_manifest_sha256': base_hash,
            'version': VERSION, 'per_persona': per_persona, 'inputs': inputs,
            'scenarios_sha256': file_hash(scenarios_path), 'builder_sha256': file_hash(Path(__file__)),
            'tokenizer_sha256': tokenizer_hash}}
        manifest_hash = write_manifest(stage / 'train_A.json', merged, revised_meta)
        frozen = {}
        for name in FROZEN:
            shutil.copyfile(base / name, stage / name)
            frozen[name] = file_hash(stage / name)
            assert frozen[name] == file_hash(base / name)
        audit = {'base_rows': len(original), 'new_rows': len(merged), 'manifest_sha256': manifest_hash,
                 'removed_unexplained_noun_rows': len(removed), 'grounded_copy_rows': len(grounded),
                 'habitat_system_rewrites': len(changed), 'new_authored_rows': len(additions),
                 'new_casing_rows': len(casing), 'skipped_casing': skipped_casing,
                 'sources': dict(collections.Counter(r['source'] for r in merged)),
                 'grounded_by_persona': dict(collections.Counter(r['persona'] for r in grounded)),
                 'new_row_tokens': token_audit, 'frozen': frozen,
                 'removed': [{'uid': r['uid'], 'parent_id': r['parent_id']} for r in removed],
                 'system_changes': [{'from_uid': r['habitat_revision']['from_uid'], 'uid': r['uid'],
                                     'from': r['habitat_revision']['original_system'], 'to': r['system']} for r in changed],
                 'grounded_lineage': [{'uid': r['uid'], 'parent_id': r['parent_id'], **r['revision_provenance']} for r in grounded],
                 'casing_lineage': [{'uid': r['uid'], 'parent_id': r['parent_id'], **r['casing_provenance']} for r in casing]}
        (stage / 'revision-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        (stage / 'grounded-copy.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in grounded))
        (stage / 'casing-pairs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in casing))
        stage.rename(out)
    return {k: v for k, v in audit.items() if k not in ('removed', 'system_changes', 'grounded_lineage', 'casing_lineage')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--per-persona', type=int, default=32)
    a = p.parse_args()
    from tokenizers import Tokenizer
    print(json.dumps(build(a.base, a.out, Tokenizer.from_file(str(a.tokenizer)),
                           file_hash(a.tokenizer), per_persona=a.per_persona), indent=2))


if __name__ == '__main__':
    main()
