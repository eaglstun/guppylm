# v5.2 full training runs

Snapshot from the run chain and artifacts under `runs/v52/`, inspected October 26, 2026.
B and the remaining chain may advance after this snapshot:

| Run | Status | Evidence |
|---|---|---|
| `full-v3-talkie-A` | Completed at 15,000/15,000 steps | Chain log records `START` and `DONE`. Recipe uses `contract-v3-talkie/train_A.json`, seed 42, MPS. The lowest and final recorded development loss is **1.76368 at step 15,000**; `best.pt` and `final.pt` are present. |
| `full-v3-talkie-B` | Incomplete in available logs | Chain records its start, with no `DONE` entry. Training log reaches step **8,400**; the latest evaluation/checkpoint history is step 8,000, with development loss **1.83710**. Recipe targets 15,000 steps from `contract-v3-talkie/train_B.json`, seed 42, MPS. |
| `full-v3-talkie-C` | Not started in the chain log | Listed after B in `full-chain.sh`; no start record or run artifacts found. |
| `full-v3-A` | Not started in the chain log | Listed after C in `full-chain.sh`; no start record or run artifacts found. |

A's step-15,000 evaluation is both its best and final recorded development score. B's 1.83710 is only its best/latest recorded evaluation so far, not a completed-run final score.

The A/B recipes and inventories refer to `contract-v3-talkie`. The newer `contract-v3-talkie-uppercase-v2` data was built separately and **has not been trained** in these runs. No behavioral improvement or promotion decision is established by these training logs.

Sources: `runs/v52/logs/full-chain.sh`, `runs/v52/logs/full-chain.log`, `runs/v52/logs/full-v3-talkie-{A,B}.log`, and each run's `recipe.json` and `log.json`.
