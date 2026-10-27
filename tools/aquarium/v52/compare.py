"""Paired development comparison; reports gates, never promotes a screen."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from .data import file_hash


def paired_difference(baseline, candidate, seed=5206):
    if baseline["contract_sha256"] != candidate["contract_sha256"] or baseline["split"] != candidate["split"]:
        raise ValueError("Reports must use the same frozen contract and split")
    if baseline["decoding"] != candidate["decoding"]:
        raise ValueError("Decoding settings differ")
    def index(report):
        values = collections.defaultdict(dict)
        for row in report["results"]:
            if row["hit"] is not None:
                key = (row["prompt_id"], row["repetition"])
                if key in values[row["slice"]]:
                    raise ValueError("Duplicate evaluated sample")
                values[row["slice"]][key] = float(row["hit"])
        return values
    left, right, output = index(baseline), index(candidate), {}
    if left.keys() != right.keys():
        raise ValueError("Evaluation slices differ")
    for slice_name, values in left.items():
        other = right[slice_name]
        if values.keys() != other.keys():
            raise ValueError("Evaluation prompts/samples differ")
        prompts = collections.defaultdict(list)
        for key, value in values.items():
            prompts[key[0]].append(other[key] - value)
        delta = np.asarray([np.mean(v) for _, v in sorted(prompts.items())])
        rng = np.random.default_rng(seed)
        means = delta[rng.integers(0, len(delta), (4000, len(delta)))].mean(axis=1)
        output[slice_name] = {"difference": float(delta.mean()),
                              "paired_prompt_bootstrap_ci95": np.quantile(means, [.025, .975]).tolist(),
                              "prompts": len(delta)}
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--rubric", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    baseline, candidate, rubric = [json.loads(p.read_text()) for p in (a.baseline, a.candidate, a.rubric)]
    if baseline["split"] != "development" or candidate["split"] != "development":
        raise ValueError("This selection comparison accepts development reports only")
    if rubric["contract_sha256"] != baseline["contract_sha256"]:
        raise ValueError("Rubric contract differs")
    if rubric["baseline_checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise ValueError("Wrong rubric baseline checkpoint")
    if rubric["baseline_behavior_sha256"] != file_hash(a.baseline):
        raise ValueError("Wrong rubric baseline report")
    deltas = paired_difference(baseline, candidate)
    primary = rubric["primary_improvement"]
    gates = {s: deltas[s]["difference"] >= primary["minimum_absolute_gain"]
                and deltas[s]["paired_prompt_bootstrap_ci95"][0] > primary["paired_prompt_bootstrap_ci95_lower_must_exceed"]
             for s in primary["slices"]}
    for s, minimum in rubric["minimum_scores"].items():
        gates[s] = candidate["summary"][s]["mean"] >= minimum
    result = {"deltas": deltas, "behavior_gates": gates, "behavior_passed": all(gates.values()),
              "rubric_sha256": file_hash(a.rubric), "baseline_sha256": file_hash(a.baseline),
              "candidate_sha256": file_hash(a.candidate), "release_approved": False,
              "pending": ["loss and per-source regression checks", "blind human review", "matched full runs and confirmation seed",
                          "codec selection", "locked test", "browser parity and real phone measurements"]}
    if a.out.exists():
        raise ValueError("Refusing to overwrite comparison")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
