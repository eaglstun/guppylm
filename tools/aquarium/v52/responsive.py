"""Build a reviewed, case-paired response dataset without changing evaluation files."""
import argparse
import collections
import json
import re
import shutil
from pathlib import Path

from .append_context import FROZEN
from .curate import load_extra_sources
from .data import (assert_disjoint, dialogue_key, digest, file_hash, normalized,
                   read_manifest, write_manifest)

REWRITES = Path(__file__).with_name('rewrites')
SCENARIOS = REWRITES / 'responsive_v1.jsonl'
DECISIONS = REWRITES / 'responsive_quarantine_v1.json'
IDENTITY = re.compile(r"\b(name|who|species|introduce|called|remember|hello|hi|hey|yourself)\b|\b(?:what|which) (?:kind|are)\b", re.I)


def trim_intro(row):
    """Remove only a standalone name-only preamble; retain identity/greeting contexts."""
    if not row.get('name') or any(IDENTITY.search(t['content']) for t in row['turns'] if t['role'] == 'user'):
        return row, []
    name = re.escape(row['name'])
    pattern = re.compile(r'^(?:I am ' + name + r'|' + name + r' here)[.!]\s+(?=[A-Z])')
    changes, turns = [], []
    for i, turn in enumerate(row['turns']):
        text = turn['content']
        new = pattern.sub('', text, count=1) if turn['role'] == 'assistant' else text
        if new != text:
            changes.append({'turn': i, 'from': text, 'to': new, 'original_sha256': digest(text)})
        turns.append({**turn, 'content': new})
    if not changes:
        return row, []
    return {**row, 'turns': turns, 'uid': digest([row['uid'], 'trim-name-only-v1', turns]),
            'response_cleanup': {'from_uid': row['uid'], 'rule': 'trim-name-only-v1'}}, changes


def paired_rows(scenarios):
    result = []
    if len({s['id'] for s in scenarios}) != len(scenarios):
        raise ValueError('Duplicate scenario IDs')
    for i, scenario in enumerate(scenarios):
        turns = scenario['turns']
        if len(turns) < 4 or len(turns) % 2 or any(
                t['role'] != ('user', 'assistant')[j % 2] or not t['content'].strip()
                for j, t in enumerate(turns)):
            raise ValueError(f"Invalid multi-turn scenario: {scenario['id']}")
        # Two fish identities across scenarios, not repeated copies across many personas.
        name, species = ('Pebble', 'guppy') if i % 2 == 0 else ('Coral', 'clownfish')
        for casing in ('normal', 'uppercase'):
            paired = [dict(t) for t in turns]
            if casing == 'uppercase':
                paired[0]['content'] = paired[0]['content'].upper()
            result.append({'id': 73_000_000 + i * 2 + (casing == 'uppercase'),
                           'name': name, 'persona': species, 'user_kind': 'human',
                           'partner': 'a visitor at the aquarium',
                           'system': f'You are {name}, a small {species}. You are friendly and curious. You live in an aquarium. You are talking with a visitor at the aquarium.',
                           'exchanges': len(paired) // 2, 'turns': paired,
                           'provenance': {'version': 'responsive-v1', 'scenario': scenario['id'],
                                          'category': scenario['category'], 'case': casing,
                                          'author': 'hand-authored'}})
    return result


def build(base, out, scenarios_path=SCENARIOS, decisions_path=DECISIONS):
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    original, base_hash, metadata = read_manifest(base / 'train_A.json', 'train')
    evaluation = [r for name in FROZEN[1:] for r in read_manifest(base / name)[0]]
    contract = json.loads((base / 'contract.json').read_text())
    decisions = json.loads(decisions_path.read_text())
    index = {r['uid']: r for r in original}
    for decision in decisions:
        row = index.get(decision['uid'])
        if row is None or row['parent_id'] != decision['parent_id'] or digest([row['system'], row['turns']]) != decision['content_sha256']:
            raise ValueError(f"Stale quarantine decision: {decision['uid']}")
    parents = {d['parent_id'] for d in decisions}
    kept, removed, trimmed, changed_rows = [], [], [], []
    for row in original:
        reason = ('replace_experimental_addons' if row['source'] in ('uppercase', 'talkie_vocabulary')
                  else 'reviewed_parent' if row['parent_id'] in parents else None)
        if reason:
            removed.append({'uid': row['uid'], 'parent_id': row['parent_id'], 'source': row['source'], 'reason': reason})
            continue
        updated, changes = trim_intro(row)
        kept.append(updated)
        if changes:
            changed_rows.append(updated)
            trimmed.append({'uid': row['uid'], 'new_uid': updated['uid'], 'changes': changes})
    scenarios = [json.loads(line) for line in scenarios_path.read_text().splitlines() if line.strip()]
    additions = paired_rows(scenarios)
    forbidden = sum(contract['reserved'].values(), []) + ['axolotl', 'cuttlefish']
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(normalized(x)) for x in forbidden) + r')(?!\w)')
    # Stage in a sibling temporary directory; failures leave no apparent valid output.
    import tempfile
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='responsive-', dir=out.parent) as temporary:
        stage = Path(temporary)
        source = stage / 'responsive.jsonl'
        source.write_text(''.join(json.dumps(r) + '\n' for r in additions))
        loaded, inputs, bad, report = load_extra_sources([f'responsive={source}'], kept + evaluation,
            {dialogue_key(r, True) for r in evaluation}, pattern)
        if bad or len(loaded) != len(additions):
            raise ValueError(f'Incomplete response import: {report}, {bad}')
        # Record a stable destination path, rather than the temporary staging path.
        for row in loaded:
            row['origin']['file'] = str(out / 'responsive.jsonl')
        for info in inputs.values():
            info['path'] = str((out / 'responsive.jsonl').resolve())
        probes = {normalized(q) for c in contract['cases'] for q in c['users']}
        probes |= {normalized(t['content']) for c in contract['cases'] for t in c['context'] if t['role'] == 'user'}
        if any(normalized(t['content']) in probes for r in loaded for t in r['turns'] if t['role'] == 'user'):
            raise ValueError('Response examples overlap frozen probe questions')
        assert_disjoint(changed_rows + loaded, evaluation)
        updated_meta = {**metadata, 'responsive_revision': {'base_manifest_sha256': base_hash,
            'scenarios_sha256': file_hash(scenarios_path), 'decisions_sha256': file_hash(decisions_path),
            'builder_sha256': file_hash(Path(__file__)), 'inputs': inputs}}
        new_hash = write_manifest(stage / 'train_A.json', kept + loaded, updated_meta)
        frozen = {}
        for name in FROZEN:
            shutil.copyfile(base / name, stage / name)
            frozen[name] = file_hash(stage / name)
            assert frozen[name] == file_hash(base / name)
        audit = {'base_rows': len(original), 'new_rows': len(kept) + len(loaded), 'added_rows': len(loaded),
                 'removed_by_reason': dict(collections.Counter(r['reason'] for r in removed)),
                 'trimmed_rows': len(trimmed), 'trimmed_turns': sum(len(r['changes']) for r in trimmed),
                 'categories': dict(collections.Counter(s['category'] for s in scenarios)),
                 'frozen': frozen, 'manifest_sha256': new_hash,
                 'removed': removed, 'trimmed': trimmed}
        (stage / 'revision-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        stage.rename(out)
    return {k: v for k, v in audit.items() if k not in ('removed', 'trimmed')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.base, a.out), indent=2))


if __name__ == '__main__':
    main()
