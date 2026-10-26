# GuppyLM experiments

This index separates the question, measured result, and lesson from each experiment.
The linked reports keep the full tables, methods, sample replies, and reproduction
commands. Unless stated otherwise, these are results from isolated runs; they do not
describe the model currently served by the [Aquarium web demo](../web/README.md).

## Model size and training

### How small can the baseline model be?

**Result.** In the [size sweep](size.md), the 806K-parameter 2-layer × 128-width model
was the smallest judged to sound like Guppy; 705K was borderline, while 337K and 404K
failed. Eval loss was 0.388 at 3.19M parameters, 0.385 at 8.73M, and 0.384 at 18.99M.
The 8.73M baseline sounded recognizably like the fish by step 1,000 of 10,000.

**What it showed.** This dataset leaves little loss to gain above about 3M parameters.
At the small end, width helped more than added depth in the tested shapes. The 806K
quality judgment comes from sampled replies; the reported training and CPU timing
columns were affected by other jobs on the machine.

### Is the baseline learning to generalize?

**Result.** The [data audit](data.md) found 812 unique user inputs among 57,100 train
rows. Every eval input appeared in train, and 77.3% of exact eval input/output pairs
appeared there too. The baseline echoed none of the new nouns in one 10-item copy
probe, despite often giving fish-like replies to familiar prompts.

**What it showed.** The original eval loss mainly measures fit to a repetitive
generator. Memorization and recombination can look fluent on familiar prompts, so
new phrasing and noun-copy probes are needed alongside loss. The report also tests
1K, 5K, and 15K training subsets and records where they overfit.

### Does the persona transfer to another creature?

**Result.** The [persona experiment](persona.md) rewrote the fish templates for a dog
using the same training structure. Fish and dog reached eval losses 0.3846 and
0.4062, respectively. Neither echoed a novel noun in its 60 sampled echo replies.
Both broke on long prompts well inside their configured 128-token window.

**What it showed.** Template vocabulary establishes much of the apparent persona,
while unfamiliar prompts expose the same fallback behavior in both models. The
tokenizer also drops unseen characters, which limits what these prompt tests can
claim about language understanding.

## Inference speed

### Where does caching help?

**Result.** In the [speed experiment](speed.md), optional KV caching raised
batch-one CPU generation from 369 to 1,166 tokens/s for 64 new tokens on the
8.7M baseline. MPS batch-one generation remained slower than CPU in that run;
the larger 88.3M comparison and bigger batches changed the crossover.

**What it showed.** Recomputing prior tokens was a major cost for this small model.
Device choice depends on model size and batch size. These are M4 Max measurements
from October 2026, with some contention and variable MPS timings; rerun them on a
target device before making deployment claims. The cache is optional in
`guppylm/model.py`; the linked report covers correctness checks and ONNX variants.

## Weight compression

### Can missing binary signs be reconstructed locally?

**Result.** The [cellular pilot](compression/cellular.md) tested one 768 × 384 FFN
matrix, leaving the rest of the model FP32. At 50% retained signs, zero filling
scored 0.4496 eval loss; fixed and learned cellular reconstruction scored 0.7530
and 0.8355. The learned rule predicted omitted signs at about chance accuracy.

**What it showed.** Adjacent signs in this matrix had no useful local structure
for the tested reconstruction rule. These matrix-level numbers are not whole-model
compression results.

### Does post-training 1-bit rounding work for Aquarium?

**Result.** The [v4 sweep](compression/one_bit_v4.md) raised held-out loss from
2.0470 for int8 to 6.3169 when all 2D matrices were rounded to 1 bit. The
[completed v5.1 confirmation](compression/one_bit_v51.md) similarly went from
2.0688 to 6.2417. A v5.1 greedy mix of two block-3 matrices saved an estimated
388,608 payload bytes but increased loss to 2.1170.

**What it showed.** Individual matrices differ in sensitivity, but post-training
1-bit rounding is not a sound whole-model replacement for these checkpoints.
The greedy mixes were selected and scored on the same held-out set; their byte
savings are projections, not deployed files.

### What worked below 4 MB on Aquarium v5.1?

**Result.** The [Aquarium quantization study](../QUANTIZATION.md) explored catalog
codes, 8-value E8-style chunks, JPEG-like 8 × 8 slices, grayscale WebP, and mixed
WebP/E8 encoding across several checkpoints. In its completed v5.1 confirmation,
the tuned 4-value catalog scored **2.0897 held-out loss at 3,568,128 estimated
payload bytes**. The tuned E8 128-column variant scored **2.2040 at 2,549,760
tensor bytes**. The tuned WebP/E8 hybrid scored **2.1066 at 3,465,162 logical
payload bytes**; its actual ZIP container was 3,491,640 bytes. The full-size
baseline scored 2.0690.

**What it showed.** The catalog was the stronger quality point in this v5.1
comparison, including 8/40 unseen noun prompts versus the hybrid's 6/40. The
hybrid advantage seen on a noun-continued v5 checkpoint did not transfer to
v5.1. JPEG-like spatial treatment and RGB mappings did not beat the stronger
catalog or E8 choices. E8 has a tested JavaScript reader, but the hybrid has no
browser runtime, and phone speed remains unmeasured. Sizes use different
accounting as labeled above; consult the report before comparing payloads.

## Results and reproduction

The compression JSON summaries are tracked in [compression/results](compression/results/).
Larger checkpoints, packed payloads, and training runs remain ignored under
`runs/`. Some reports retain paths from their source worktrees; run their commands
from the current checkout with explicit paths to the required local artifacts.
