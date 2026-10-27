"""Immutable, content-addressed conversation manifests for Aquarium v5.2.

No model dependency: preparation and leakage checks run with the standard library.
"""
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personas import make_spec

VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def dialogue_key(row, delex=False):
    text = "\n".join(t["role"] + ":" + normalized(t["content"]) for t in row["turns"])
    if delex:
        for value in sorted((row.get("name"), row.get("thing")), key=lambda x: len(x or ""), reverse=True):
            if value:
                value = re.sub(r"^(a|an|the) ", "", normalized(value))
                text = re.sub(r"\b" + re.escape(value) + r"\b", "ENTITY", text)
    return digest(text)


def read_source(path, source):
    rows, rejected = [], []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if "system" not in row:
                row = {**make_spec(random.Random(row["id"])), **row}
            turns = row.get("turns", [])
            if (not isinstance(row.get("system"), str) or len(turns) < 2 or len(turns) % 2
                    or any(t.get("role") != ("user", "assistant")[i % 2]
                           or not isinstance(t.get("content"), str) or not t["content"].strip()
                           for i, t in enumerate(turns))):
                raise ValueError("invalid conversation")
            uid = digest([source, row.get("id"), row["system"], turns])
            rows.append({**row, "uid": uid, "parent_id": uid, "source": source,
                         "origin": {"file": str(path), "line": number}, "augmentation": "base"})
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            rejected.append({"file": str(path), "line": number, "reason": str(exc)})
    return rows, rejected


def split_rows(rows, seed=5201, dev_fraction=.05):
    """Group exact/delexicalized dialogue duplicates BEFORE splitting/capping.

    This is a prospective development split, never a fresh test of v5/v5.1.
    Shared short turns are audited separately, not treated as independent groups.
    """
    seen, out, duplicates = set(), [], []
    for row in rows:
        key = dialogue_key(row, delex=True)
        exact = digest([row["system"], row["turns"]])
        if exact in seen:
            duplicates.append(row["uid"])
            continue
        seen.add(exact)
        bucket = int(digest([seed, key])[:16], 16) / 2**64
        out.append({**row, "group_id": key,
                    "split": "development" if bucket < dev_fraction else "train"})
    return out, duplicates


def write_manifest(path, rows, metadata):
    path = Path(path)
    payload = {"version": VERSION, "metadata": metadata, "rows": rows}
    payload["sha256"] = digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Refusing to replace frozen manifest: {path}")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return payload["sha256"]


def read_manifest(path, split=None):
    payload = json.loads(Path(path).read_text())
    expected = payload.pop("sha256")
    if digest(payload) != expected or payload["version"] != VERSION:
        raise ValueError(f"Manifest integrity failure: {path}")
    rows = payload["rows"]
    if split and any(r["split"] != split for r in rows):
        raise ValueError(f"Expected only {split} rows in {path}")
    if len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate manifest IDs")
    return rows, expected, payload["metadata"]


def assert_disjoint(train, development):
    for field in ("uid", "parent_id", "group_id"):
        left = {r[field] for r in train if field in r}
        right = {r[field] for r in development if field in r}
        if left & right:
            raise ValueError(f"Cross-split {field} overlap: {len(left & right)}")
    if {dialogue_key(r, True) for r in train} & {dialogue_key(r, True) for r in development}:
        raise ValueError("Cross-split normalized dialogue overlap")


def overlap_audit(rows):
    """Exact turns, entity-normalized wording, and 8-word response fragments.

    Fragment overlap is a near-duplicate warning, not a semantic guarantee.
    Count row memberships rather than repeated occurrences within a row.
    """
    indexes = {k: collections.defaultdict(set) for k in ("turns", "wording", "response_8grams", "names", "nouns")}
    by_uid = {r["uid"]: r for r in rows}
    for row in rows:
        uid = row["uid"]
        for field, label in (("name", "names"), ("thing", "nouns")):
            if row.get(field):
                indexes[label][normalized(row[field])].add(uid)
        for turn in row["turns"]:
            text = normalized(turn["content"])
            indexes["turns"][text].add(uid)
            wording = text
            for field in ("name", "thing"):
                if row.get(field):
                    value = re.sub(r"^(a|an|the) ", "", normalized(row[field]))
                    wording = re.sub(r"\b" + re.escape(value) + r"\b", "ENTITY", wording)
            indexes["wording"][wording].add(uid)
            if turn["role"] == "assistant":
                words = text.split()
                for i in range(len(words) - 7):
                    indexes["response_8grams"][" ".join(words[i:i + 8])].add(uid)
    result = {}
    for label, index in indexes.items():
        shared = [(s, ids) for s, ids in index.items() if len(ids) > 1]
        cross = [(s, ids) for s, ids in shared if len({by_uid[i].get("split") for i in ids}) > 1]
        result[label] = {"distinct": len(index), "repeated": len(shared), "cross_split": len(cross),
                         "examples": [{"text": s, "rows": len(ids)} for s, ids in
                                      sorted(cross, key=lambda p: (-len(p[1]), p[0]))[:10]]}
    return result


def proportions(rows):
    counts = collections.Counter(r["source"] for r in rows)
    return {k: {"rows": n, "share": n / max(1, len(rows))} for k, n in sorted(counts.items())}


def assert_compatible_metadata(left, right, tokenizer_hash):
    """Reject accidentally mixed experiment contracts or tokenizers."""
    if left.get("contract_sha256") != right.get("contract_sha256"):
        raise ValueError("Manifests belong to different evaluation contracts")
    for metadata in (left, right):
        expected = metadata.get("inputs", {}).get("tokenizer.json", {}).get("sha256")
        if expected is not None and expected != tokenizer_hash:
            raise ValueError("Tokenizer does not match frozen manifest")
