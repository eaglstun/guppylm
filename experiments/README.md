# GuppyLM experiments

These reports preserve measurements from isolated experiment branches. They are
research results, not changes to the shipped Aquarium model or GitHub Pages.

| Experiment | Report | Reusable code |
| --- | --- | --- |
| Binary and cellular weight compression | [cellular](compression/cellular.md), [Aquarium v4](compression/one_bit_v4.md), [completed v5.1](compression/one_bit_v51.md) | `tools/experiment_cellular_compression.py`, `tools/experiment_binary_sweep.py` |
| Baseline model size and learning curve | [size](size.md) | `tools/benchmark.py` shape flags and `curve` command |
| Baseline data diet and overfitting | [data](data.md) | `--train-limit`, final-checkpoint scores, `tools/quality_many.py` |

The compression JSON summaries are tracked in `compression/results/`. Larger
checkpoints, packed payloads, and training runs remain ignored under `runs/`.
The original reports contain paths from their source worktrees; use the current
checkout and explicit input paths when reproducing them.
