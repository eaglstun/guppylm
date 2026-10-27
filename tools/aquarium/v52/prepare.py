"""Freeze auditable A/B/C inputs. Run as python -m tools.aquarium.v52.prepare."""
import argparse
import collections
import json
import random
import re
from pathlib import Path

from .contract import make_contract, varied_copy_rows
from .data import (
    assert_disjoint,
    digest,
    file_hash,
    normalized,
    overlap_audit,
    proportions,
    read_source,
    split_rows,
    write_manifest,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=5201)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--copy-share", type=float, default=.05)
    a = ap.parse_args()
    if a.cap < 1 or not 0 <= a.copy_share <= .1:
        ap.error("cap must be positive; bounded copy share must be in [0, .1]")
    from .. import train_aq as legacy
    legacy.D = str(a.data)
    legacy.MEMORY_POOL.clear()
    legacy.SELFNAME_POOL.clear()
    legacy.MEMORY_POOL.update(legacy.load_memory_pool())
    legacy.SELFNAME_POOL.update(legacy.load_selfname_pool())
    sources = [(a.data / "dialogues.jsonl", "base")]
    sources += [(p, "haiku") for p in sorted((a.data / "haiku").glob("out_*.jsonl"))]
    sources += [(a.data / s / "chat.jsonl", s) for s in ("songs", "family")]
    inputs, rejected, rows = {}, [], []
    for path, source in sources:
        rs, bad = read_source(path, source)
        rows.extend(rs)
        rejected.extend(bad)
        inputs[str(path.relative_to(a.data))] = {"sha256": file_hash(path), "accepted": len(rs), "rejected": len(bad)}
    historical, bad = read_source(a.data / "dialogues_eval.jsonl", "historical")
    rejected.extend(bad)
    historical = [{**r, "split": "historical", "group_id": digest(["historical", r["uid"]])} for r in historical]
    # Include prior-stage text when selecting genuinely absent entity strings.
    exposure = []
    for sub in ("tinydialogues", "ocean"):
        path = a.data / sub / "chat.jsonl"
        rs, bad = read_source(path, sub)
        exposure.extend(rs)
        inputs[str(path.relative_to(a.data))] = {"sha256": file_hash(path), "accepted": len(rs), "rejected": len(bad)}
    for path in [a.data / "tokenizer.json", a.data / "dialogues_eval.jsonl"] + sorted(
            (a.data / "memory").glob("*.jsonl")) + sorted((a.data / "selfname").glob("*.jsonl")):
        inputs[str(path.relative_to(a.data))] = {"sha256": file_hash(path)}
    contract = make_contract(rows + historical + exposure)
    # Historical wording belongs to development only; exact/delex duplicates are quarantined.
    from .data import dialogue_key
    historical_keys = {dialogue_key(r, True) for r in historical}
    grouped, duplicates = split_rows(rows, a.seed)
    quarantine = [r for r in grouped if dialogue_key(r, True) in historical_keys]
    grouped = [r for r in grouped if dialogue_key(r, True) not in historical_keys]
    # Reserve locked-test species at fine-tuning stage as well as entity strings.
    forbidden = contract["reserved"]["locked_test"] + ["axolotl"]
    forbidden_pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(normalized(t)) for t in forbidden) + r")(?!\w)")
    def safe(row):
        text = row["system"] + " " + " ".join(t["content"] for t in row["turns"])
        return not forbidden_pattern.search(normalized(text))
    quarantine += [r for r in grouped if not safe(r)]
    grouped = [r for r in grouped if safe(r)]
    rng = random.Random(a.seed)
    base = [r for r in grouped if r["split"] == "train" and r["source"] in ("base", "haiku")]
    rng.shuffle(base)
    base = base[:a.cap]
    extra = [r for r in grouped if r["split"] == "train" and r["source"] not in ("base", "haiku")]
    development = [r for r in grouped if r["split"] == "development"]
    compression_dev = [{**r, "split": "compression_development"} for r in development
                       if int(digest(["compression", r["group_id"]])[:8], 16) % 5 == 0]
    compression_ids = {r["uid"] for r in compression_dev}
    development = [r for r in development if r["uid"] not in compression_ids]
    base += extra
    # All generated rows retain parent/group identity; no augmentation of validation.
    metadata = {"seed": a.seed, "inputs": inputs, "generator": "v52.prepare/v1",
                "code": {p.name: file_hash(p) for p in [Path(__file__), Path(legacy.__file__),
                          Path(__file__).with_name("data.py"), Path(__file__).with_name("contract.py"),
                          Path(legacy.__file__).with_name("personas.py")]},
                "cap": a.cap, "contract_sha256": digest(contract),
                "warning": "New development split was exposed to historical v5/v5.1 training. Not a locked test."}
    a.out.mkdir(parents=True, exist_ok=True)
    contract_path = a.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Refusing to overwrite frozen evaluation contract")
    contract_path.write_text(json.dumps(contract, indent=1) + "\n")
    write_manifest(a.out / "development.json", development, metadata)
    write_manifest(a.out / "historical.json", historical, metadata)
    write_manifest(a.out / "compression_development.json", compression_dev, metadata)
    audit = {"inputs": inputs, "rejected": rejected, "duplicates_removed": len(duplicates),
             "quarantined": len(quarantine), "before_cap": proportions(grouped),
             "after_cap": proportions(base), "development": proportions(development),
             "compression_development": proportions(compression_dev),
             "overlap": overlap_audit(grouped + historical), "candidates": {},
             "limitations": ["8-word fragment and entity-normalized matching are lexical heuristics, not semantic deduplication.",
                             "Generator versions were not recorded historically; current source hashes cannot prove old inputs.",
                             "New held-persona probe excludes axolotl from fine-tuning, not prior-stage natural language exposure.",
                             "Full TinyStories training text is not screened for probe entities."]}
    for candidate, noun_copies in (("A", 2), ("B", 4), ("C", 2)):
        prng = random.Random(a.seed)
        train = list(base)
        ops = [("rename", 2, legacy.renamed, base),
               ("noun", noun_copies, legacy.noun_swapped, [r for r in base if r.get("thing")]),
               ("memory", 1, lambda r, g: legacy.remembered(legacy.renamed(r, g), g), base),
               ("selfname", 1, lambda r, g: legacy.asked_name(legacy.renamed(r, g), g), base)]
        for kind, copies, fn, parents in ops:
            # Independent streams keep memory/name rows identical between A and B.
            prng = random.Random(digest([a.seed, kind]))
            for copy in range(copies):
                for parent in parents:
                    row = fn(parent, prng)
                    train.append({**row, "uid": digest([parent["uid"], kind, copy, row["turns"]]),
                                  "augmentation": kind, "generator_seed": a.seed})
        if candidate == "C":
            train += varied_copy_rows(base, int(len(train) * a.copy_share / (1 - a.copy_share)), a.seed)
        # Remove no-op and repeated derived dialogues; preserve distinct system/persona contexts.
        seen, deduped = set(), []
        for row in train:
            key = digest([row["system"], row["turns"]])
            if key not in seen and safe(row):
                deduped.append(row)
                seen.add(key)
        assert_disjoint(deduped, development + compression_dev)
        questions = {normalized(t["content"]) for row in deduped for t in row["turns"] if t["role"] == "user"}
        collisions = sorted({q for case in contract["cases"] for q in case["users"] if normalized(q) in questions})
        if collisions:
            raise ValueError(f"Evaluation questions overlap generated training inputs: {collisions}")
        h = write_manifest(a.out / f"train_{candidate}.json", deduped, {**metadata, "candidate": candidate})
        audit["candidates"][candidate] = {"rows": len(deduped), "removed_after_augmentation": len(train) - len(deduped),
                                         "sources": proportions(deduped), "sha256": h,
                                         "exact_probe_question_collisions": 0,
                                         "augmentations": dict(collections.Counter(r["augmentation"] for r in deduped))}
    (a.out / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({"out": str(a.out), "rejected": len(rejected), "development": len(development),
                      "candidates": {k: v["rows"] for k, v in audit["candidates"].items()}}, indent=2))


if __name__ == "__main__":
    main()
