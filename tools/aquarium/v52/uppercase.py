"""Append the frozen uppercase stress augmentation to v52 A/B/C manifests.

Run from the repository root, for example:
  python -m tools.aquarium.v52.uppercase --base runs/v52/contract-v3-talkie \
    --out runs/v52/contract-v3-talkie-uppercase --contexts 20 --tokenizer PATH
"""
import argparse
import collections
import json
import re
from pathlib import Path

from .data import (assert_compatible_metadata, assert_disjoint, digest,
                   dialogue_key, file_hash, normalized, read_manifest)

GENERATOR_PREFIX = "v52.uppercase/"
MOBILE_SPECIES = {"clownfish", "octopus", "pufferfish", "reef_shark",
                  "sea_turtle", "seahorse", "lobster"}
SCENARIOS = Path(__file__).with_name("rewrites") / "uppercase_v1.jsonl"
FROZEN = ("contract.json", "development.json", "historical.json",
          "compression_development.json")


def load_scenarios(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 32 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path} must contain 32 uniquely identified scenarios")
    for r in rows:
        turns = r.get("turns", [])
        if not r.get("category") or len(turns) < 2 or len(turns) % 2:
            raise ValueError(f"Invalid scenario {r.get('id')}")
        if any(t.get("role") != ("user", "assistant")[i % 2]
               or not isinstance(t.get("content"), str) or not t["content"].strip()
               for i, t in enumerate(turns)):
            raise ValueError(f"Invalid turns in scenario {r['id']}")
    return rows


def safe_context(row, forbidden):
    text = " ".join(str(row.get(k, "")) for k in
                    ("system", "persona", "name", "thing", "topic", "partner"))
    text += " " + " ".join(t["content"] for t in row["turns"])
    words = normalized(text)
    return not any(re.search(r"(?<!\w)" + re.escape(normalized(x)) + r"(?!\w)", words)
                   for x in forbidden)


def select_contexts(rows, count, forbidden):
    eligible = [r for r in rows if r.get("split") == "train"
                and r.get("source") in ("base", "haiku")
                and r.get("augmentation") == "base"
                and r.get("user_kind") == "human"
                and r.get("persona") in MOBILE_SPECIES
                and all(r.get(k) for k in ("persona", "name", "system", "user_kind", "partner"))
                and safe_context(r, forbidden)]
    # One deterministic representative per distinct persona/name/system tuple.
    unique = {}
    for r in eligible:
        key = (r["persona"], r["name"], r["system"])
        unique.setdefault(key, r)
    # Stable hash order, then round-robin species to reduce domination by a species.
    buckets = collections.defaultdict(list)
    for key, row in unique.items():
        buckets[row["persona"]].append(row)
    for species in buckets:
        buckets[species].sort(key=lambda r: digest(["uppercase-context", r["uid"]]))
    result, species = [], sorted(buckets)
    while len(result) < count:
        advanced = False
        for sp in species:
            if buckets[sp]:
                result.append(buckets[sp].pop(0)); advanced = True
                if len(result) == count:
                    break
        if not advanced:
            break
    if len(result) != count:
        raise ValueError(f"Only {len(result)} safe distinct contexts available; need {count}")
    return result


def make_rows(contexts, scenarios, scenario_path, version, generator):
    made = []
    for scenario in scenarios:
        turns = [{"role": t["role"], "content": t["content"]}
                 for t in scenario["turns"]]
        for context in contexts:
            provenance = {"scenario_id": scenario["id"], "category": scenario["category"],
                          "context_parent_uid": context["parent_id"]}
            identity = [generator, scenario["id"], context["uid"]]
            made.append({"persona": context["persona"], "name": context["name"],
                         "user_kind": context["user_kind"], "partner": context["partner"],
                         "system": context["system"], "exchanges": len(turns) // 2,
                         "turns": turns, "uid": digest(["uid", identity]),
                         "parent_id": digest(["parent", identity]),
                         "group_id": digest(["group", generator, scenario["id"]]), "source": "uppercase",
                         "origin": {"file": str(scenario_path), "scenario_id": scenario["id"],
                                    "context_parent_uid": context["parent_id"]},
                         "augmentation": version, "split": "train",
                         "provenance": provenance})
    return made


def encode_counts(row, tok, seq):
    # This mirrors tools/aquarium/train_aq.py's chat_ids/encode_chat framing.
    ids, mask = [], []
    def add(tokens, on=False):
        ids.extend(tokens); mask.extend([on] * len(tokens))
    add([1] + tok.encode("system\n" + row["system"]).ids + [2] + tok.encode("\n").ids)
    for turn in row["turns"]:
        add([1] + tok.encode(turn["role"] + "\n").ids)
        add(tok.encode(turn["content"]).ids + [2], turn["role"] == "assistant")
        add(tok.encode("\n").ids)
    clipped = ids[:seq + 1]
    clipped_mask = mask[:seq + 1]
    return min(len(ids), seq + 1) - 1, sum(bool(clipped_mask[i + 1]) for i in range(len(clipped) - 1)), len(ids) > seq + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", type=int, default=20)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--scenarios", type=Path, default=SCENARIOS,
                    help=f"scenario JSONL (default: {SCENARIOS})")
    ap.add_argument("--version", default="uppercase_v1",
                    help="augmentation label and UID namespace (default: uppercase_v1)")
    a = ap.parse_args()
    if a.contexts < 1:
        ap.error("--contexts must be positive")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", a.version):
        ap.error("--version may contain only letters, digits, dot, underscore, and hyphen")
    generator = GENERATOR_PREFIX + a.version
    scenario_path = a.scenarios
    scenario_rows = load_scenarios(scenario_path)
    contract_bytes = (a.base / "contract.json").read_bytes()
    contract = json.loads(contract_bytes)
    reserved = contract.get("reserved", {})
    forbidden = list(reserved.get("development", [])) + list(reserved.get("locked_test", [])) + ["axolotl"]
    old_rows_a, _, meta_a = read_manifest(a.base / "train_A.json", split="train")
    contexts = select_contexts(old_rows_a, a.contexts, forbidden)
    additions = make_rows(contexts, scenario_rows, scenario_path, a.version, generator)
    if len(additions) != a.contexts * 32:
        raise AssertionError("Unexpected augmentation size")
    strict_entities = forbidden
    for row in additions:
        text = normalized(row["system"] + " " + " ".join(t["content"] for t in row["turns"]))
        found = [x for x in strict_entities
                 if re.search(r"(?<!\w)" + re.escape(normalized(x)) + r"(?!\w)", text)]
        if found:
            raise ValueError(f"Reserved entity in uppercase row {row['provenance']['scenario_id']}: {found}")

    # The probe wording is compared exactly after the dataset's established normalization.
    probe_questions = {normalized(q) for case in contract["cases"] for q in case["users"]}
    collisions = sorted({normalized(t["content"]) for r in additions for t in r["turns"]
                         if t["role"] == "user" and normalized(t["content"]) in probe_questions})
    if collisions:
        raise ValueError(f"Evaluation question collisions: {collisions}")
    eval_sets = {}
    for name in ("development.json", "historical.json", "compression_development.json"):
        eval_sets[name] = read_manifest(a.base / name)[0]
    eval_rows = [r for rows in eval_sets.values() for r in rows]
    assert_disjoint(additions, eval_rows)
    eval_exact = {digest([r["system"], r["turns"]]) for r in eval_rows}
    eval_norm = {dialogue_key(r, delex=True) for r in eval_rows}
    for r in additions:
        exact = digest([r["system"], r["turns"]])
        norm = dialogue_key(r, delex=True)
        if exact in eval_exact or norm in eval_norm:
            raise ValueError("New uppercase row duplicates a frozen evaluation row")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.tokenizer))
    token_hash = file_hash(a.tokenizer)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in FROZEN:
        dst = a.out / name
        payload = (a.base / name).read_bytes()
        if dst.exists() and dst.read_bytes() != payload:
            raise ValueError(f"Refusing to replace differing frozen file: {dst}")
        if not dst.exists():
            dst.write_bytes(payload)

    audit = {"generator": generator, "augmentation": a.version, "base_dir": str(a.base),
             "base_manifest_hashes": {}, "scenarios_file": str(scenario_path),
             "scenarios_sha256": file_hash(scenario_path),
             "generator_sha256": file_hash(Path(__file__)),
             "tokenizer_sha256": token_hash, "frozen_hashes": {},
             "contexts": a.contexts, "context_species": dict(collections.Counter(r["persona"] for r in contexts)),
             "source_category_counts": {"uppercase": dict(collections.Counter(r["provenance"]["category"] for r in additions))},
             "turn_counts": {role: sum(t["role"] == role for r in additions for t in r["turns"])
                             for role in ("user", "assistant")},
             "new_rows": len(additions), "tokens": {}, "unchanged_rows": {},
             "turn_case_counts": {role: {"uppercase": 0, "mixed_case": 0,
                                           "lowercase": 0, "uncased": 0}
                                  for role in ("user", "assistant")},
             "probe_user_wording_collisions": 0, "evaluation_duplicate_rows": 0,
             "prior_audit": str(a.base / "audit.json"),
             "note": "Fresh augmentation audit. Prior audit describes the unaugmented base manifests."}
    for row in additions:
        for turn in row["turns"]:
            letters = [c for c in turn["content"] if c.isalpha()]
            if not letters:
                label = "uncased"
            elif all(c.isupper() for c in letters):
                label = "uppercase"
            elif all(c.islower() for c in letters):
                label = "lowercase"
            else:
                label = "mixed_case"
            audit["turn_case_counts"][turn["role"]][label] += 1
    frozen_before = {name: file_hash(a.out / name) for name in FROZEN}
    for candidate in "ABC":
        name = f"train_{candidate}.json"
        rows, _, metadata = read_manifest(a.base / name, split="train")
        assert_compatible_metadata(meta_a, metadata, token_hash)
        old_ids = {r["uid"] for r in rows}
        if old_ids & {r["uid"] for r in additions}:
            raise ValueError(f"UID collision in {candidate}")
        merged = rows + additions
        # Entire new rows must be novel against the base candidate.
        old_pairs = {digest([r["system"], r["turns"]]) for r in rows}
        old_normalized = {dialogue_key(r, delex=True) for r in rows}
        if any(digest([r["system"], r["turns"]]) in old_pairs
               or dialogue_key(r, delex=True) in old_normalized for r in additions):
            raise ValueError(f"Uppercase rows duplicate base {candidate}")
        tok_counts = collections.defaultdict(lambda: {"rows": 0, "input_tokens": 0,
                                                       "target_tokens": 0, "truncated_rows": 0})
        for row in additions:
            x, y, trunc = encode_counts(row, tok, 512)
            item = tok_counts[row["source"]]
            item["rows"] += 1; item["input_tokens"] += x
            item["target_tokens"] += y; item["truncated_rows"] += int(trunc)
        aug_meta = {**metadata, "augmentation": {"name": a.version, "generator": generator,
                    "scenarios_file": str(scenario_path),
                    "scenarios_sha256": file_hash(scenario_path), "tokenizer_sha256": token_hash,
                    "generator_sha256": file_hash(Path(__file__)),
                    "contexts": a.contexts, "base_manifest_sha256": file_hash(a.base / name)}}
        payload = {"version": 1, "metadata": aug_meta, "rows": merged}
        payload["sha256"] = digest(payload)
        dst = a.out / name
        serialized = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
        if dst.exists() and dst.read_text() != serialized:
            raise ValueError(f"Refusing to replace differing manifest: {dst}")
        if not dst.exists():
            dst.write_text(serialized)
        check, _, _ = read_manifest(dst, split="train")
        if check[:len(rows)] != rows or check[len(rows):] != additions:
            raise ValueError(f"Append-only verification failed for {candidate}")
        audit["base_manifest_hashes"][name] = file_hash(a.base / name)
        audit["unchanged_rows"][candidate] = len(rows)
        total_target = sum(v["target_tokens"] for v in tok_counts.values())
        prior_tokens = json.loads((a.base / "audit.json").read_text())["candidates"][candidate]["tokens"]
        prior_total = sum(v["target_tokens"] for v in prior_tokens.values())
        audit["tokens"][candidate] = {"new": dict(tok_counts["uppercase"]),
            "new_target_share_of_combined": total_target / max(1, prior_total + total_target),
            "base_candidate_target_tokens": prior_total}
    audit["frozen_hashes"] = {name: {"sha256": frozen_before[name],
                                    "identical_to_base": frozen_before[name] == file_hash(a.base / name)}
                              for name in FROZEN}
    if not all(v["identical_to_base"] for v in audit["frozen_hashes"].values()):
        raise ValueError("Frozen-file byte identity check failed")
    audit_path = a.out / "uppercase_audit.json"
    audit_text = json.dumps(audit, indent=1) + "\n"
    if audit_path.exists() and audit_path.read_text() != audit_text:
        raise ValueError(f"Refusing to replace differing audit: {audit_path}")
    if not audit_path.exists():
        audit_path.write_text(audit_text)
    print(json.dumps({"out": str(a.out), "new_rows_per_candidate": len(additions),
                      "contexts": len(contexts), "frozen_identical": True,
                      "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
