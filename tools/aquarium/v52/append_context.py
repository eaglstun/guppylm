"""Append reviewed context rows to candidate A while preserving frozen evaluation files."""
import argparse
import json
import re
import shutil
from pathlib import Path

from .curate import load_extra_sources
from .data import (assert_disjoint, dialogue_key, file_hash, normalized,
                   read_manifest, read_source, write_manifest)

FROZEN = ('contract.json', 'development.json', 'historical.json', 'compression_development.json')
DEFAULT_ROWS = Path(__file__).with_name('rewrites') / 'talkie_context_v2.jsonl'


def append_context(base, out, source_file=DEFAULT_ROWS):
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    rows, old_hash, metadata = read_manifest(base / 'train_A.json', 'train')
    contract = json.loads((base / 'contract.json').read_text())
    evaluation = [r for name in FROZEN[1:] for r in read_manifest(base / name)[0]]
    forbidden = sum(contract['reserved'].values(), []) + ['axolotl', 'cuttlefish']
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(normalized(x)) for x in forbidden) + r')(?!\w)')
    expected, bad = read_source(source_file, 'talkie_vocabulary')
    additions, inputs, rejected, report = load_extra_sources(
        [f'talkie_vocabulary={source_file}'], rows + evaluation,
        {dialogue_key(r, True) for r in evaluation}, pattern)
    if bad or rejected or not additions or len(additions) != len(expected):
        raise ValueError(f'Not all context rows survived validation: {report}, {bad}, {rejected}')
    probes = {normalized(q) for c in contract['cases'] for q in c['users']}
    probes |= {normalized(t['content']) for c in contract['cases'] for t in c['context'] if t['role'] == 'user'}
    if any(normalized(t['content']) in probes for r in additions for t in r['turns'] if t['role'] == 'user'):
        raise ValueError('Context rows overlap frozen probe questions')
    assert_disjoint(additions, evaluation)
    out.mkdir(parents=True)
    new_meta = {**metadata, 'context_addition': {'inputs': inputs, 'base_manifest_sha256': old_hash,
                                               'builder_sha256': file_hash(Path(__file__))}}
    manifest_hash = write_manifest(out / 'train_A.json', rows + additions, new_meta)
    frozen = {}
    for name in FROZEN:
        shutil.copyfile(base / name, out / name)
        frozen[name] = file_hash(out / name)
        assert frozen[name] == file_hash(base / name)
    audit = {'base': str(base), 'old_rows': len(rows), 'added_rows': len(additions),
             'new_rows': len(rows) + len(additions), 'manifest_sha256': manifest_hash,
             'frozen': frozen, 'source_report': report, 'inputs': inputs}
    (out / 'context-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rows', type=Path, default=DEFAULT_ROWS)
    a = p.parse_args()
    print(json.dumps(append_context(a.base, a.out, a.rows), indent=2))


if __name__ == '__main__':
    main()
