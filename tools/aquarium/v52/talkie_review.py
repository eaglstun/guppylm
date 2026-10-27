"""Apply screened review decisions to respelled talkie rows.

Order per row: content-screen reject (unless overturned) -> leak partner clause ->
leak replacements (or overrides) -> content fixes not covered by a leak replacement or
skipped -> truncation. Every change is recorded in ``provenance.review`` with the
sha256 of the text it replaced.

    python -m tools.aquarium.v52.talkie_review --out runs/v52/talkie/whimsy-v1/chat_reviewed.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DECISIONS = Path(__file__).with_name("rewrites") / "talkie_whimsy_v1_review.json"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(pattern):
    return {v["id"]: v for p in range(1, 5)
            for v in map(json.loads, (ROOT / pattern.replace("{1..4}", str(p))).read_text().splitlines())}


def review(row, content, leak, d):
    key = str(row["id"])
    log, turns, system = [], [dict(t) for t in row["turns"]], row["system"]
    if content["verdict"] == "reject" and key not in d["overturn_reject"]:
        return None, {"rejected": content["reasons"], "note": content["note"]}

    def put(turn, text, why):
        assert turn % 2 == 1 and turn < len(turns), (key, turn)
        log.append({"turn": turn, "why": why, "from_sha256": sha(turns[turn]["content"]),
                    "from": turns[turn]["content"]})
        turns[turn]["content"] = text

    if leak["partner_clause"]:
        assert "You are talking with" not in system, key
        system += f" You are talking with {row['partner']}."
        log.append({"system": "partner_clause"})
    replaced = set()
    for r in leak["replacements"]:
        text = d["replace_override"].get(key, {}).get(str(r["turn"]), r["replacement"])
        put(r["turn"], text, "leak:" + r["leaked"])
        replaced.add(r["turn"])
    if content["verdict"] == "fix" and key not in d["skip_fix"]:
        for f in content["fixes"]:
            if f["turn"] not in replaced:
                put(f["turn"], f["replacement"], "content:" + ",".join(content["reasons"]))
    if key in d["truncate_after"]:
        turn, why = d["truncate_after"][key]
        log.append({"truncated_after": turn, "dropped_turns": len(turns) - turn - 1, "why": why})
        turns = turns[:turn + 1]
    if not log:
        return row, None
    return {**row, "system": system, "turns": turns, "exchanges": len(turns) // 2,
            "provenance": {**row["provenance"], "review": {"decisions_sha256": sha(DECISIONS.read_text()),
                                                           "changes": log}}}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(DECISIONS.read_text())
    content, leak = load(d["screens"]["content"]), load(d["screens"]["leak"])
    rows = [json.loads(line) for line in (ROOT / d["input"]).read_text().splitlines() if line.strip()]
    assert [r["id"] for r in rows] == list(content) == list(leak), "screen/input id mismatch"
    kept, rejected = [], []
    for row in rows:
        out, why = review(row, content[row["id"]], leak[row["id"]], d)
        if out is None:
            rejected.append({"id": row["id"], **why})
        else:
            kept.append(out)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    a.out.with_name(a.out.stem + "_rejected.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rejected))
    changed = [r for r in kept if "review" in r["provenance"]]
    print(json.dumps({"kept": len(kept), "rejected": len(rejected), "changed": len(changed),
                      "partner_clause": sum(any(c.get("system") for c in r["provenance"]["review"]["changes"])
                                            for r in changed),
                      "turns_replaced": sum(sum("turn" in c for c in r["provenance"]["review"]["changes"])
                                            for r in changed),
                      "truncated": sum(any("truncated_after" in c for c in r["provenance"]["review"]["changes"])
                                       for r in changed)}))


if __name__ == "__main__":
    main()
