"""Versioned, train-only curation and extra-source loading for Aquarium v5.2.

Curation v2 (contract-v2) is a targeted pass over TRAIN parents before augmentation:

* ``degenerate``: drop conversations whose assistant repeats an identical reply, or whose
  turn repeats the previous turn verbatim (looping generations).
* ``assistant_ngram_cap``: keep at most N rows containing any one assistant 8-gram, so
  stock clauses ("I do not know that word, but it sounds like ...") stay present but bounded.
  Abstention itself is kept; only repeated identical phrasing is capped.
* ``user_turn_cap``: keep at most N rows containing any one entity-normalized user turn
  ("do you have any friends"); short courtesy turns ("thanks ENTITY") are exempt.
* ``name_cap``: rename (not drop) rows of over-represented creature names to the
  least-used training names, only where the new name is absent from the row and not a
  reserved evaluation entity.

Caps are applied greedily in a fixed, seed-derived order and recorded with reasons.
Development, compression-development, historical and behavioral splits are never edited.
These are lexical heuristics, not semantic deduplication.
"""
import collections
import hashlib
import itertools
import json
import math
import random
import re
from pathlib import Path

from .data import dialogue_key, digest, normalized, read_source

CURATION_VERSION = 3
REWRITES = Path(__file__).with_name("rewrites")
# Source labels already used by the pipeline; extra sources may not reuse them.
RESERVED_SOURCES = {"base", "haiku", "songs", "family", "historical", "tinydialogues", "ocean", "varied_copy"}
SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

CONFIGS = {
    1: {"version": 1},  # Reproduces contract-v1 train parents exactly.
    2: {"version": 2, "degenerate": True,
        "assistant_ngram": 8, "assistant_ngram_cap": 40,
        # Content-inherent sequences, not stock phrasing.
        "assistant_ngram_exempt": [r"^(?:one|two|three|four|five|six|seven|eight|nine|ten)(?: (?:one|two|three|four|five|six|seven|eight|nine|ten))*$"],
        "user_turn_cap": 40,
        "user_courtesy": (r"^(?:(?:hi|hello|hey|oh|ok|okay|wow|yes|yay|no|good ?night|good ?bye|bye|thanks|thank you"
                          r"|see you|i love you|me too|cool)(?: |$))+(?:(?:ENTITY|friend|so much|very much|again|soon"
                          r"|later|too)(?: |$))*$"),
        "name_cap_over_median": 1.25},
}
# v3 = v2 selection, then hand-written, hash-pinned rewrites of trait-echo sentences.
CONFIGS[3] = {**CONFIGS[2], "version": 3, "rewrites": ["trait_echo_v1.jsonl"],
              "trait_echo_detector": ("assistant turn contains, as whole normalized words, the persona's leading "
                                      "trait list from personas.PERSONAS[persona][2]: its words up to and including "
                                      "the word after the first 'and'/'but' (e.g. 'calm old and wise')")}


def trait_core(persona):
    """Leading trait list of a persona, e.g. 'grumpy but fair' or 'calm old and wise'."""
    from ..personas import PERSONAS
    words = normalized(PERSONAS[persona][2]).split()
    i = next(k for k, w in enumerate(words) if w in ("and", "but"))
    return " ".join(words[:i + 2])


def trait_echo_turns(row):
    """Indices of assistant turns that recite the persona's system-prompt trait list."""
    from ..personas import PERSONAS
    if row.get("persona") not in PERSONAS:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(trait_core(row["persona"])) + r"(?!\w)")
    return [i for i, t in enumerate(row["turns"]) if t["role"] == "assistant" and pattern.search(normalized(t["content"]))]


def load_rewrites(name):
    path = REWRITES / name
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(e["uid"], e["turn"]) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate rewrite entries in {path}")
    for e in entries:
        if hashlib.sha256(e["original"].encode()).hexdigest() != e["original_sha256"]:
            raise ValueError(f"Rewrite file is internally inconsistent: {e['uid']} turn {e['turn']}")
    return path, entries


def apply_rewrites(rows, names, report, strict=True):
    """Replace hash-pinned assistant turns. A changed original fails loudly.

    ``strict``: also fail when a listed parent is not selected (stale uid). Extra sources can
    legitimately shift the capped selection, so prepare relaxes this only when they are used.
    """
    from .data import file_hash
    by_uid = {r["uid"]: i for i, r in enumerate(rows)}
    rows = list(rows)
    summary = {"files": {}, "rows": 0, "turns": 0, "missing": [], "examples": [], "uids": []}
    for name in names:
        path, entries = load_rewrites(name)
        summary["files"][name] = {"sha256": file_hash(path), "entries": len(entries)}
        grouped = collections.defaultdict(list)
        for e in entries:
            grouped[e["uid"]].append(e)
        for uid, items in sorted(grouped.items()):
            if uid not in by_uid:
                if strict:
                    raise ValueError(f"Stale rewrite {name}: train parent {uid} not selected")
                summary["missing"].append(uid)
                continue
            i = by_uid[uid]
            row = rows[i]
            turns = [dict(t) for t in row["turns"]]
            for e in items:
                turn = turns[e["turn"]] if e["turn"] < len(turns) else None
                if (turn is None or turn["role"] != "assistant"
                        or hashlib.sha256(turn["content"].encode()).hexdigest() != e["original_sha256"]):
                    raise ValueError(f"Stale rewrite {name}: {uid} turn {e['turn']} no longer matches its original")
                turn["content"] = e["replacement"]
            updated = {**row, "turns": turns, "uid": digest([uid, "curation-rewrite", name, turns]),
                       "curation": {**row.get("curation", {}), "version": report["config"]["version"],
                                    "rewritten": {"file": name, "turns": sorted(e["turn"] for e in items),
                                                  "from_uid": uid}}}
            left = set(trait_echo_turns(updated)) & {e["turn"] for e in items}
            if left:
                raise ValueError(f"Rewrite {name} still echoes the trait list: {uid} turns {sorted(left)}")
            rows[i] = updated
            summary["rows"] += 1
            summary["turns"] += len(items)
            summary["uids"].append(uid)
            if len(summary["examples"]) < 8:
                summary["examples"].append({"uid": uid, "source": row["source"], "turn": items[0]["turn"],
                                            "before": items[0]["original"], "after": items[0]["replacement"]})
    report["rewrites"] = summary
    return rows


def _delex(text, row):
    text = normalized(text)
    for field in ("name", "thing"):
        if row.get(field):
            value = re.sub(r"^(a|an|the) ", "", normalized(row[field]))
            if value:
                text = re.sub(r"\b" + re.escape(value) + r"\b", "ENTITY", text)
    return text


def degenerate_reason(row):
    turns = row["turns"]
    replies = [normalized(t["content"]) for t in turns if t["role"] == "assistant"]
    if len(set(replies)) < len(replies):
        return "repeated_assistant_reply"
    for prev, cur in itertools.pairwise(turns):
        if normalized(prev["content"]) == normalized(cur["content"]):
            return "turn_repeats_previous_turn"
    return None


class Caps:
    """Streaming per-key row caps; a row is kept only if none of its keys is full."""

    def __init__(self, config):
        self.config = config
        self.n = config.get("assistant_ngram", 8)
        self.exempt = [re.compile(p) for p in config.get("assistant_ngram_exempt", [])]
        self.courtesy = re.compile(config["user_courtesy"]) if config.get("user_courtesy") else None
        self.counts = {"assistant_ngram": collections.Counter(), "user_turn": collections.Counter()}

    def keys(self, row):
        out = {"assistant_ngram": set(), "user_turn": set()}
        for t in row["turns"]:
            if t["role"] == "assistant" and self.config.get("assistant_ngram_cap"):
                words = _delex(t["content"], row).split()
                for i in range(len(words) - self.n + 1):
                    gram = " ".join(words[i:i + self.n])
                    if not any(p.match(gram) for p in self.exempt):
                        out["assistant_ngram"].add(gram)
            elif t["role"] == "user" and self.config.get("user_turn_cap"):
                text = _delex(t["content"], row)
                if text and not (self.courtesy and self.courtesy.match(text)):
                    out["user_turn"].add(text)
        return out

    def offer(self, row):
        """Return None if kept (and count it), else (rule, key) for the first full key."""
        keys = self.keys(row)
        for rule in ("assistant_ngram", "user_turn"):
            cap = self.config.get(rule + "_cap")
            for key in sorted(keys[rule]):
                if cap and self.counts[rule][key] >= cap:
                    return rule + "_cap", key
        for rule, values in keys.items():
            self.counts[rule].update(values)
        return None


def curate_train(pool, extra, cap, seed, config, reserved_names=(), strict_rewrites=True):
    """Select capped base parents from shuffled ``pool`` plus curated ``extra`` rows.

    ``pool`` must already be in the v1 shuffled order; with config v1 this returns exactly
    ``pool[:cap] + extra``. Extras are offered first so the backfillable base pool, not the
    scarcer song/family/extra sources, absorbs most of the cap pressure.
    Returns (base_rows, extra_rows, report).
    """
    report = {"config": config, "removed": collections.defaultdict(list), "renamed": []}
    if config["version"] == 1:
        return pool[:cap], list(extra), _finish(report)
    caps = Caps(config)

    def admit(row):
        reason = degenerate_reason(row) if config.get("degenerate") else None
        detail = None
        if not reason:
            hit = caps.offer(row)
            if hit:
                reason, detail = hit
        if reason:
            report["removed"][reason].append(_example(row, detail))
            return False
        return True

    order = list(range(len(extra)))
    random.Random(digest([seed, "curation", "extra"])).shuffle(order)
    keep_extra = sorted(i for i in order if admit(extra[i]))
    extra = [extra[i] for i in keep_extra]
    base, unused = [], 0
    for row in pool:
        if len(base) >= cap:
            unused += 1
            continue
        if admit(row):
            base.append(row)
    report["base_pool"] = {"offered": len(pool) - unused, "unused_after_cap": unused, "selected": len(base)}
    if config.get("name_cap_over_median"):
        base, extra = rebalance_names(base, extra, seed, config["name_cap_over_median"], reserved_names, report)
    if config.get("rewrites"):
        before = sum(len(trait_echo_turns(r)) for r in base + extra)
        rows = apply_rewrites(base + extra, config["rewrites"], report, strict_rewrites)
        base, extra = rows[:len(base)], rows[len(base):]
        report["trait_echo_turns"] = {"before": before, "after": sum(len(trait_echo_turns(r)) for r in rows)}
    return base, extra, _finish(report)


def rebalance_names(base, extra, seed, ratio, reserved_names, report):
    from ..personas import EVAL_NAMES, NAMES
    rows = base + extra
    counts = collections.Counter(r["name"] for r in rows if r.get("name"))
    if not counts:
        return base, extra
    candidates = sorted(set(NAMES) - set(EVAL_NAMES))
    reserved = {normalized(x) for x in reserved_names} | {normalized(x) for x in EVAL_NAMES}
    candidates = [n for n in candidates if normalized(n) not in reserved]
    for n in candidates:
        counts.setdefault(n, 0)
    median = sorted(counts[n] for n in candidates)[len(candidates) // 2]
    limit = math.ceil(ratio * median)
    report["name_limit"] = {"median": median, "limit": limit,
                            "over_limit": {n: c for n, c in counts.most_common() if c > limit}}
    heads = [n for n, c in counts.items() if c > limit]
    renamed = {}
    for head in sorted(heads):
        members = sorted((i for i, r in enumerate(rows) if r.get("name") == head),
                         key=lambda i: digest([seed, "curation", "name", rows[i]["uid"]]))
        excess = counts[head] - limit
        for i in members:
            if excess <= 0:
                break
            row = rows[i]
            text = normalized(row["system"] + " " + " ".join(t["content"] for t in row["turns"]))
            options = [n for n in candidates if n != head and counts[n] < limit
                       and not re.search(r"\b" + re.escape(normalized(n)) + r"\b", text)]
            if not options:
                continue
            new = min(options, key=lambda n: (counts[n], n))
            pattern = re.compile(r"\b" + re.escape(head) + r"\b")
            updated = {**row, "name": new, "system": pattern.sub(new, row["system"]),
                       "turns": [{**t, "content": pattern.sub(new, t["content"])} for t in row["turns"]],
                       "uid": digest([row["uid"], "curation-rename", new]),
                       "curation": {"renamed_from": head, "version": report["config"]["version"]}}
            if not re.search(r"\b" + re.escape(new) + r"\b", updated["system"]):
                continue  # Name was not literally present in the system prompt; leave row alone.
            renamed[i] = updated
            counts[head] -= 1
            counts[new] += 1
            excess -= 1
            report["renamed"].append({**_example(row, None), "from": head, "to": new})
    rows = [renamed.get(i, r) for i, r in enumerate(rows)]
    report["name_counts_after"] = {"max": counts.most_common(1)[0],
                                   "min": min(((n, counts[n]) for n in candidates), key=lambda p: p[1])}
    return rows[:len(base)], rows[len(base):]


def _example(row, detail):
    first = next((t["content"] for t in row["turns"] if t["role"] == "user"), "")
    reply = next((t["content"] for t in row["turns"] if t["role"] == "assistant"), "")
    return {"uid": row["uid"], "source": row["source"], "origin": row.get("origin"), "key": detail,
            "user": first[:160], "assistant": reply[:160]}


def _finish(report):
    removed = report.pop("removed")
    report["removed"] = {reason: {"rows": len(items), "by_source": dict(collections.Counter(i["source"] for i in items)),
                                  "top_keys": collections.Counter(i["key"] for i in items if i["key"]).most_common(15),
                                  "examples": items[:8], "uids": [i["uid"] for i in items]}
                         for reason, items in sorted(removed.items())}
    renamed = report["renamed"]
    report["renamed"] = {"rows": len(renamed), "by_change": dict(collections.Counter(
        f"{r['from']}->{r['to']}" for r in renamed)), "examples": renamed[:8], "uids": [r["uid"] for r in renamed]}
    report["removed_total"] = sum(v["rows"] for v in report["removed"].values())
    return report


def parse_extra_source(spec):
    if "=" not in spec:
        raise ValueError(f"--extra-source must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not SOURCE_NAME.match(name) or name in RESERVED_SOURCES:
        raise ValueError(f"Invalid or reserved extra source name: {name!r}")
    path = Path(path).expanduser()
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files or not all(f.is_file() for f in files):
        raise ValueError(f"No jsonl files for extra source {name!r} at {path}")
    return name, path, files


def load_extra_sources(specs, existing, eval_keys, forbidden_pattern):
    """Read train-only extra sources through the base validation, dedup and reservation checks.

    ``existing``: rows already in the corpus (all splits + historical) for exact dedup.
    ``eval_keys``: entity-normalized dialogue keys of every non-train row; matches are
    quarantined so extra data can never share a group with development/historical rows.
    ``forbidden_pattern``: regex over normalized text for reserved evaluation entities.
    Returns (rows, inputs, rejected, report). Rows are always split ``train``.
    """
    from .data import file_hash
    seen = {digest([r["system"], r["turns"]]) for r in existing}
    names, rows, inputs, rejected = set(), [], {}, []
    report = {}
    for spec in specs:
        name, root, files = parse_extra_source(spec)
        if name in names:
            raise ValueError(f"Duplicate extra source name: {name}")
        names.add(name)
        stats = collections.Counter()
        for path in files:
            accepted, bad = read_source(path, name)
            rejected.extend(bad)
            inputs[f"extra/{name}/{path.name}"] = {"sha256": file_hash(path), "path": str(path.resolve()),
                                              "accepted": len(accepted), "rejected": len(bad), "split": "train"}
            stats["invalid"] += len(bad)
            for row in accepted:
                if not isinstance(row.get("name"), str) or not re.search(
                        r"\b" + re.escape(row["name"]) + r"\b", row["system"]):
                    stats["invalid"] += 1
                    rejected.append({**row["origin"], "reason": "extra source row needs a name present in its system prompt"})
                    continue
                exact = digest([row["system"], row["turns"]])
                key = dialogue_key(row, delex=True)
                text = normalized(row["system"] + " " + " ".join(t["content"] for t in row["turns"]))
                if exact in seen:
                    stats["duplicate"] += 1
                elif key in eval_keys:
                    stats["eval_group_overlap"] += 1
                elif forbidden_pattern.search(text):
                    stats["reserved_entity"] += 1
                else:
                    seen.add(exact)
                    stats["accepted"] += 1
                    rows.append({**row, "group_id": key, "split": "train"})
        report[name] = {"path": str(root.resolve()), "files": len(files), **dict(stats)}
    return rows, inputs, rejected, report
