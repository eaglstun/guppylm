"""Partial-copy rates from a probe.json: does the reply start the target word right?

    python3 tools/aquarium/partial.py runs/aq-chat/probe.json
"""
import json
import re
import sys

from personas import EVAL_NAMES

d = json.load(open(sys.argv[1]))


def has_prefix(target, reply, n=3):
    t = target.lower()[:n]
    return any(w.startswith(t) for w in re.findall(r"[a-z]+", reply.lower()))


def rate(rows, target, reply):
    return sum(has_prefix(target(r), reply(r)) for r in rows) / len(rows)


out = {
    "name_copy_prefix3": rate(d["name_copy"], lambda r: r["name"], lambda r: r["reply"]),
    "name_copy_full": rate(d["name_copy"], lambda r: r["name"], lambda r: r["reply"] if r["name"].lower() in r["reply"].lower() else ""),
}
for kind in ["real", "fake"]:
    rows = [r for r in d["noun_copy"] if r["kind"] == kind]
    out[f"noun_{kind}_prefix3"] = rate(rows, lambda r: r["thing"].split()[-1], lambda r: r["reply"])
print(json.dumps({k: round(v, 3) for k, v in out.items()}, indent=1))
