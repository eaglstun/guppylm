"""Freeze auditable A/B/C inputs. Run as python -m tools.aquarium.v52.prepare."""
import argparse
import collections
import json
import random
import re
from pathlib import Path

from .contract import make_contract, varied_copy_rows
from .curate import CONFIGS, CURATION_VERSION, curate_train, load_extra_sources
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

GENERATOR = "v52.prepare/v2"


def token_shares(rows, tok, legacy, seq=512):
    """Input and assistant-target tokens per source after the trainer's truncation."""
    out = collections.defaultdict(lambda: {"rows": 0, "input_tokens": 0, "target_tokens": 0})
    for row in rows:
        x, y = legacy.encode_chat(tok, row, seq)
        item = out[row["source"]]
        item["rows"] += 1
        item["input_tokens"] += len(x)
        item["target_tokens"] += sum(t != 0 for t in y)
    total = sum(v["target_tokens"] for v in out.values())
    return {k: {**v, "target_share": v["target_tokens"] / max(1, total)} for k, v in sorted(out.items())}


def rows_digest(rows):
    return digest(rows)


def check_frozen_eval(frozen, out_rows, contract):
    """Require eval splits and contract to equal an earlier contract's, row for row."""
    from .data import read_manifest
    if json.loads((frozen / "contract.json").read_text()) != json.loads(json.dumps(contract)):
        raise ValueError(f"Evaluation contract differs from {frozen}")
    result = {"contract.json": {"sha256": digest(contract), "identical": True,
                                "frozen_file_sha256": file_hash(frozen / "contract.json")}}
    for name, rows in out_rows.items():
        old, old_hash, _ = read_manifest(frozen / f"{name}.json")
        if rows_digest(old) != rows_digest(rows):
            raise ValueError(f"{name} rows differ from {frozen}")
        result[name] = {"rows": len(rows), "rows_sha256": rows_digest(rows), "identical": True,
                        "frozen_manifest_sha256": old_hash}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=5201)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--copy-share", type=float, default=.05)
    ap.add_argument("--curation", type=int, choices=sorted(CONFIGS), default=CURATION_VERSION,
                    help="Train-only curation version; 1 reproduces contract-v1 train parents.")
    ap.add_argument("--extra-source", action="append", default=[], metavar="NAME=PATH",
                    help="Train-only jsonl file or directory in the data_aq row format (repeatable).")
    ap.add_argument("--frozen-eval", type=Path,
                    help="Earlier contract directory whose contract and eval-split rows must be identical.")
    ap.add_argument("--seq", type=int, default=512, help="Sequence length for token-share accounting.")
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
    # Extra sources are train-only and are excluded from contract construction above.
    eval_keys = {r["group_id"] for r in grouped if r["split"] != "train"} | historical_keys
    strict = contract["reserved"]["development"] + forbidden
    strict_pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(normalized(t)) for t in strict) + r")(?!\w)")
    extra_rows, extra_inputs, bad, extra_report = load_extra_sources(
        a.extra_source, rows + historical + exposure, eval_keys, strict_pattern)
    inputs.update(extra_inputs)
    rejected.extend(bad)
    rng = random.Random(a.seed)
    base = [r for r in grouped if r["split"] == "train" and r["source"] in ("base", "haiku")]
    rng.shuffle(base)
    extra = [r for r in grouped if r["split"] == "train" and r["source"] not in ("base", "haiku")] + extra_rows
    curation_config = CONFIGS[a.curation]
    uncurated = base[:a.cap] + extra
    base, extra, curation = curate_train(base, extra, a.cap, a.seed, curation_config,
                                         contract["reserved"]["development"] + contract["reserved"]["locked_test"],
                                         strict_rewrites=not a.extra_source)
    curation["rows_before"] = proportions(uncurated)
    curation["rows_after"] = proportions(base + extra)
    curation["extra_sources"] = extra_report
    development = [r for r in grouped if r["split"] == "development"]
    compression_dev = [{**r, "split": "compression_development"} for r in development
                       if int(digest(["compression", r["group_id"]])[:8], 16) % 5 == 0]
    compression_ids = {r["uid"] for r in compression_dev}
    development = [r for r in development if r["uid"] not in compression_ids]
    base += extra
    # All generated rows retain parent/group identity; no augmentation of validation.
    metadata = {"seed": a.seed, "inputs": inputs, "generator": GENERATOR,
                "code": {p.name: file_hash(p) for p in [Path(__file__), Path(legacy.__file__),
                          Path(__file__).with_name("data.py"), Path(__file__).with_name("contract.py"),
                          Path(__file__).with_name("curate.py"), Path(legacy.__file__).with_name("personas.py")]
                         + [Path(__file__).with_name("rewrites") / n for n in curation_config.get("rewrites", [])]},
                "cap": a.cap, "contract_sha256": digest(contract), "curation": curation_config,
                "extra_sources": sorted(extra_report),
                "warning": "New development split was exposed to historical v5/v5.1 training. Not a locked test."}
    eval_rows = {"development": development, "historical": historical, "compression_development": compression_dev}
    frozen = check_frozen_eval(a.frozen_eval, eval_rows, contract) if a.frozen_eval else None
    a.out.mkdir(parents=True, exist_ok=True)
    contract_path = a.out / "contract.json"
    # Compare after a JSON round trip: tuples in the in-memory contract load back as lists.
    if contract_path.exists() and json.loads(contract_path.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError("Refusing to overwrite frozen evaluation contract")
    contract_path.write_text(json.dumps(contract, indent=1) + "\n")
    for name, split_rows_ in eval_rows.items():
        write_manifest(a.out / f"{name}.json", split_rows_, metadata)
    extra_names = set(extra_report)
    if any(r["source"] in extra_names for rs in eval_rows.values() for r in rs):
        raise ValueError("Extra train-only source reached an evaluation split")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.data / "tokenizer.json"))
    audit = {"inputs": inputs, "rejected": rejected, "duplicates_removed": len(duplicates),
             "quarantined": len(quarantine), "before_cap": proportions(grouped),
             "after_cap": proportions(base), "development": proportions(development),
             "compression_development": proportions(compression_dev),
             "overlap": overlap_audit(grouped + historical), "candidates": {},
             "generator": GENERATOR, "curation": curation, "frozen_eval": frozen,
             "eval_rows_sha256": {k: rows_digest(v) for k, v in eval_rows.items()},
             "train_parents_overlap": overlap_audit(base + development + compression_dev + historical),
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
                                         "tokens": token_shares(deduped, tok, legacy, a.seq),
                                         "exact_probe_question_collisions": 0,
                                         "augmentations": dict(collections.Counter(r["augmentation"] for r in deduped))}
    (a.out / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({"out": str(a.out), "rejected": len(rejected), "development": len(development),
                      "candidates": {k: v["rows"] for k, v in audit["candidates"].items()}}, indent=2))


if __name__ == "__main__":
    main()
